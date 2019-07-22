#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <stdbool.h>
#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>
#include <signal.h>
#include <dogleg.h>

#include <suitesparse/cholmod.h>

#include "mrcal.h"


#if PY_MAJOR_VERSION == 3
#define PyString_FromString PyUnicode_FromString
#define PyString_FromFormat PyUnicode_FromFormat
#define PyInt_FromLong      PyLong_FromLong
#define PyString_AsString   PyUnicode_AsUTF8
#define PyInt_Check         PyLong_Check
#define PyInt_AsLong        PyLong_AsLong
#define STRING_OBJECT       "U"
#else
#define STRING_OBJECT       "S"
#endif


// Python is silly. There's some nuance about signal handling where it sets a
// SIGINT (ctrl-c) handler to just set a flag, and the python layer then reads
// this flag and does the thing. Here I'm running C code, so SIGINT would set a
// flag, but not quit, so I can't interrupt the solver. Thus I reset the SIGINT
// handler to the default, and put it back to the python-specific version when
// I'm done
#define SET_SIGINT() struct sigaction sigaction_old;                    \
do {                                                                    \
    if( 0 != sigaction(SIGINT,                                          \
                       &(struct sigaction){ .sa_handler = SIG_DFL },    \
                       &sigaction_old) )                                \
    {                                                                   \
        PyErr_SetString(PyExc_RuntimeError, "sigaction() failed");      \
        goto done;                                                      \
    }                                                                   \
} while(0)
#define RESET_SIGINT() do {                                             \
    if( 0 != sigaction(SIGINT,                                          \
                       &sigaction_old, NULL ))                          \
        PyErr_SetString(PyExc_RuntimeError, "sigaction-restore failed"); \
} while(0)

#define QUOTED_LIST_WITH_COMMA(s,n) "'" #s "',"

#define CHECK_CONTIGUOUS(x) do {                                        \
    if( !PyArray_IS_C_CONTIGUOUS(x) )                                   \
    {                                                                   \
        PyErr_SetString(PyExc_RuntimeError, "All inputs must be c-style contiguous arrays (" #x ")"); \
        return false;                                                   \
    } } while(0)


#define COMMA ,
#define ARG_DEFINE(     name, pytype, initialvalue, parsecode, parseprearg, name_pyarrayobj, npy_type, dims_ref) pytype name = initialvalue;
#define ARG_LIST_DEFINE(name, pytype, initialvalue, parsecode, parseprearg, name_pyarrayobj, npy_type, dims_ref) pytype name,
#define ARG_LIST_CALL(  name, pytype, initialvalue, parsecode, parseprearg, name_pyarrayobj, npy_type, dims_ref) name,
#define NAMELIST(       name, pytype, initialvalue, parsecode, parseprearg, name_pyarrayobj, npy_type, dims_ref) #name ,
#define PARSECODE(      name, pytype, initialvalue, parsecode, parseprearg, name_pyarrayobj, npy_type, dims_ref) parsecode
#define PARSEARG(       name, pytype, initialvalue, parsecode, parseprearg, name_pyarrayobj, npy_type, dims_ref) parseprearg &name,
#define FREE_PYARRAY(   name, pytype, initialvalue, parsecode, parseprearg, name_pyarrayobj, npy_type, dims_ref) Py_XDECREF(name_pyarrayobj);
#define CHECK_LAYOUT(   name, pytype, initialvalue, parsecode, parseprearg, name_pyarrayobj, npy_type, dims_ref) \
    if(name_pyarrayobj != NULL && (PyObject*)name_pyarrayobj != (PyObject*)Py_None) {                  \
        int dims[] = dims_ref;                                          \
        int ndims = (int)sizeof(dims)/(int)sizeof(dims[0]);             \
                                                                        \
        if( ndims > 0 )                                                 \
        {                                                               \
            if( PyArray_NDIM((PyArrayObject*)name_pyarrayobj) != ndims )          \
            {                                                           \
                PyErr_Format(PyExc_RuntimeError, "'" #name "' must have exactly %d dims; got %d", ndims, PyArray_NDIM((PyArrayObject*)name_pyarrayobj)); \
                return false;                                           \
            }                                                           \
            for(int i=0; i<ndims; i++)                                  \
                if(dims[i] >= 0 && dims[i] != PyArray_DIMS((PyArrayObject*)name_pyarrayobj)[i]) \
                {                                                       \
                    PyErr_Format(PyExc_RuntimeError, "'" #name "'must have dimensions '" #dims_ref "' where <0 means 'any'. Dims %d got %ld instead", i, PyArray_DIMS((PyArrayObject*)name_pyarrayobj)[i]); \
                    return false;                                       \
                }                                                       \
        }                                                               \
        if( (int)npy_type >= 0 )                                        \
        {                                                               \
            if( PyArray_TYPE((PyArrayObject*)name_pyarrayobj) != npy_type )       \
            {                                                           \
                PyErr_SetString(PyExc_RuntimeError, "'" #name "' must have type: " #npy_type); \
                return false;                                           \
            }                                                           \
            if( !PyArray_IS_C_CONTIGUOUS((PyArrayObject*)name_pyarrayobj) )       \
            {                                                           \
                PyErr_SetString(PyExc_RuntimeError, "'" #name "'must be c-style contiguous"); \
                return false;                                           \
            }                                                           \
        }                                                               \
    }
#define PYMETHODDEF_ENTRY(function_prefix, name, args) {#name,          \
                                                        (PyCFunction)function_prefix ## name, \
                                                        args,           \
                                                        function_prefix ## name ## _docstring}


// A wrapper around a solver context and various solver metadata. I need the
// optimization to be able to keep this, and I need Python to free it as
// necessary when the refcount drops to 0
typedef struct {
    PyObject_HEAD
    dogleg_solverContext_t* ctx;

    distortion_model_t distortion_model;
    mrcal_problem_details_t problem_details;

    int Ncameras, Nframes, Npoints;
    int NobservationsBoard;
    int calibration_object_width_n;

} SolverContext;
static void SolverContext_dealloc(SolverContext* self)
{
    mrcal_free_context((void**)&self->ctx);
    Py_TYPE(self)->tp_free((PyObject*)self);
}
static PyObject* SolverContext_str(SolverContext* self)
{
    if(self->ctx == NULL)
        return PyString_FromString("Empty context");
    return PyString_FromFormat("Non-empty context made with        %s\n"
                               "Ncameras:                          %d\n"
                               "Nframes:                           %d\n"
                               "Npoints:                           %d\n"
                               "NobservationsBoard:                %d\n"
                               "calibration_object_width_n:        %d\n"
                               "do_optimize_intrinsic_core:        %d\n"
                               "do_optimize_intrinsic_distortions: %d\n"
                               "do_optimize_cahvor_optical_axis:   %d\n",
                               mrcal_distortion_model_name(self->distortion_model),
                               self->Ncameras, self->Nframes, self->Npoints,
                               self->NobservationsBoard,
                               self->calibration_object_width_n,
                               self->problem_details.do_optimize_intrinsic_core,
                               self->problem_details.do_optimize_intrinsic_distortions,
                               self->problem_details.do_optimize_cahvor_optical_axis);
}

static PyObject* SolverContext_J(SolverContext* self)
{
    if( self->ctx == NULL )
    {
        PyErr_SetString(PyExc_RuntimeError, "I need a non-empty context");
        return NULL;
    }

    // I do the Python equivalent of this;
    // scipy.sparse.csr_matrix((data, indices, indptr))
    PyObject* module = NULL;
    PyObject* method = NULL;
    PyObject* result = NULL;
    PyObject* args   = NULL;
    if(NULL == (module = PyImport_ImportModule("scipy.sparse")))
    {
        PyErr_SetString(PyExc_RuntimeError, "Couldn't import scipy.sparse. I need that to represent J");
        goto done;
    }
    if(NULL == (method = PyObject_GetAttrString(module, "csr_matrix")))
    {
        PyErr_SetString(PyExc_RuntimeError, "Couldn't find 'csr_matrix' in scipy.sparse");
        goto done;
    }

    cholmod_sparse* Jt = self->ctx->beforeStep->Jt;
    // Here I'm assuming specific types in my cholmod arrays. I tried to
    // static_assert it, but internally cholmod uses void*, so I can't do that
    PyObject* P         = PyArray_SimpleNewFromData(1, ((npy_intp[]){Jt->ncol + 1}), NPY_INT32,  Jt->p);
    PyObject* I         = PyArray_SimpleNewFromData(1, ((npy_intp[]){Jt->nzmax   }), NPY_INT32,  Jt->i);
    PyObject* X         = PyArray_SimpleNewFromData(1, ((npy_intp[]){Jt->nzmax   }), NPY_DOUBLE, Jt->x);
    PyObject* MatrixDef = PyTuple_Pack(3, X, I, P);
    args                = PyTuple_Pack(1, MatrixDef);
    Py_DECREF(P);
    Py_DECREF(I);
    Py_DECREF(X);
    Py_DECREF(MatrixDef);

    if(NULL == (result = PyObject_CallObject(method, args)))
        goto done; // reuse already-set error


    // Testing code to dump out a dense representation of this matrix to a file.
    // Can compare that file to what this function returns like this:
    //   Jf = np.fromfile("/tmp/J_17014_444.dat").reshape(17014,444)
    //   np.linalg.norm( Jf - J.toarray() )
    // {
    // #define P(A, index) ((unsigned int*)((A)->p))[index]
    // #define I(A, index) ((unsigned int*)((A)->i))[index]
    // #define X(A, index) ((double*      )((A)->x))[index]
    //         char logfilename[128];
    //         sprintf(logfilename, "/tmp/J_%d_%d.dat",(int)Jt->ncol,(int)Jt->nrow);
    //         FILE* fp = fopen(logfilename, "w");
    //         double* Jrow;
    //         Jrow = malloc(Jt->nrow*sizeof(double));
    //         for(unsigned int icol=0; icol<Jt->ncol; icol++)
    //         {
    //             memset(Jrow, 0, Jt->nrow*sizeof(double));
    //             for(unsigned int i=P(Jt, icol); i<P(Jt, icol+1); i++)
    //             {
    //                 int irow = I(Jt,i);
    //                 double x = X(Jt,i);
    //                 Jrow[irow] = x;
    //             }
    //             fwrite(Jrow,sizeof(double),Jt->nrow,fp);
    //         }
    //         fclose(fp);
    //         free(Jrow);
    // #undef P
    // #undef I
    // #undef X
    // }


 done:
    Py_XDECREF(module);
    Py_XDECREF(method);
    Py_XDECREF(args);

    return result;
}

static PyObject* SolverContext_state_index_intrinsic_core(SolverContext* self,
                                                          PyObject* args)
{
    if( self->ctx == NULL )
    {
        PyErr_SetString(PyExc_RuntimeError, "I need a non-empty context");
        return NULL;
    }
    PyObject* result = NULL;
    int i_camera = -1;
    if(!PyArg_ParseTuple( args, "i", &i_camera )) goto done;
    if( i_camera < 0 || i_camera >= self->Ncameras )
    {
        PyErr_Format(PyExc_RuntimeError,
                     "i_camera must refer to a valid camera, i.e. be in the range [0,%d] inclusive. Instead I got %d",
                     self->Ncameras,i_camera);
        goto done;
    }
    result = Py_BuildValue("i",
                           mrcal_state_index_intrinsic_core(i_camera,
                                                            self->problem_details,
                                                            self->distortion_model));
 done:
    return result;
}
static PyObject* SolverContext_state_index_intrinsic_distortions(SolverContext* self,
                                                                 PyObject* args)
{
    if( self->ctx == NULL )
    {
        PyErr_SetString(PyExc_RuntimeError, "I need a non-empty context");
        return NULL;
    }
    PyObject* result = NULL;
    int i_camera = -1;
    if(!PyArg_ParseTuple( args, "i", &i_camera )) goto done;
    if( i_camera < 0 || i_camera >= self->Ncameras )
    {
        PyErr_Format(PyExc_RuntimeError,
                     "i_camera must refer to a valid camera, i.e. be in the range [0,%d] inclusive. Instead I got %d",
                     self->Ncameras,i_camera);
        goto done;
    }
    result = Py_BuildValue("i",
                           mrcal_state_index_intrinsic_distortions(i_camera,
                                                                   self->problem_details,
                                                                   self->distortion_model));
 done:
    return result;
}
static PyObject* SolverContext_state_index_camera_rt(SolverContext* self,
                                                     PyObject* args)
{
    if( self->ctx == NULL )
    {
        PyErr_SetString(PyExc_RuntimeError, "I need a non-empty context");
        return NULL;
    }
    PyObject* result = NULL;
    int i_camera = -1;
    if(!PyArg_ParseTuple( args, "i", &i_camera )) goto done;
    if( i_camera < 1 || i_camera >= self->Ncameras )
    {
        PyErr_Format(PyExc_RuntimeError,
                     "i_camera must refer to a valid camera that's NOT the first camera i.e. be in the range [1,%d] inclusive. Instead I got %d. The first camera defines the reference coordinate system, so it has no state",
                     self->Ncameras,i_camera);
        goto done;
    }
    result = Py_BuildValue("i",
                           mrcal_state_index_camera_rt(i_camera,
                                                       self->Ncameras,
                                                       self->problem_details,
                                                       self->distortion_model));
 done:
    return result;
}
static PyObject* SolverContext_state_index_frame_rt(SolverContext* self,
                                                    PyObject* args)
{
    if( self->ctx == NULL )
    {
        PyErr_SetString(PyExc_RuntimeError, "I need a non-empty context");
        return NULL;
    }
    PyObject* result = NULL;
    int i_frame = -1;
    if(!PyArg_ParseTuple( args, "i", &i_frame )) goto done;
    if( i_frame < 0 || i_frame >= self->Nframes )
    {
        PyErr_Format(PyExc_RuntimeError,
                     "i_frame must refer to a valid frame i.e. be in the range [0,%d] inclusive. Instead I got %d",
                     self->Nframes,i_frame);
        goto done;
    }
    result = Py_BuildValue("i",
                           mrcal_state_index_frame_rt(i_frame,
                                                      self->Ncameras,
                                                      self->problem_details,
                                                      self->distortion_model));
 done:
    return result;
}
static PyObject* SolverContext_state_index_point(SolverContext* self,
                                                 PyObject* args)
{
    if( self->ctx == NULL )
    {
        PyErr_SetString(PyExc_RuntimeError, "I need a non-empty context");
        return NULL;
    }
    PyObject* result = NULL;
    int i_point = -1;
    if(!PyArg_ParseTuple( args, "i", &i_point )) goto done;
    if( i_point < 0 || i_point >= self->Nframes )
    {
        PyErr_Format(PyExc_RuntimeError,
                     "i_point must refer to a valid point i.e. be in the range [0,%d] inclusive. Instead I got %d",
                     self->Npoints,i_point);
        goto done;
    }
    result = Py_BuildValue("i",
                           mrcal_state_index_point(i_point,
                                                   self->Nframes, self->Ncameras,
                                                   self->problem_details,
                                                   self->distortion_model));
 done:
    return result;
}
static PyObject* SolverContext_state_index_calobject_warp(SolverContext* self,
                                                          PyObject* args)
{
    if( self->ctx == NULL )
    {
        PyErr_SetString(PyExc_RuntimeError, "I need a non-empty context");
        return NULL;
    }

    return Py_BuildValue("i",
                         mrcal_state_index_calobject_warp(self->Npoints,
                                                          self->Nframes, self->Ncameras,
                                                          self->problem_details,
                                                          self->distortion_model));
}

static PyObject* SolverContext_num_measurements(SolverContext* self)
{
    if( self->ctx == NULL )
    {
        PyErr_SetString(PyExc_RuntimeError, "I need a non-empty context");
        return NULL;
    }

    int Nmeasurements_all = self->ctx->beforeStep->Jt->ncol;
    int Nmeasurements_regularization =
        mrcal_getNmeasurements_regularization(self->Ncameras,
                                              self->problem_details,
                                              self->distortion_model);
    int Nmeasurements_boards =
        mrcal_getNmeasurements_boards( self->NobservationsBoard,
                                       self->calibration_object_width_n);
    int Nmeasurements_points =
        Nmeasurements_all - Nmeasurements_regularization - Nmeasurements_boards;

    PyObject* result = PyDict_New();
    PyObject* x;

    x = PyInt_FromLong(Nmeasurements_regularization);
    PyDict_SetItemString(result, "regularization", x);
    Py_DECREF(x);

    x = PyInt_FromLong(Nmeasurements_boards);
    PyDict_SetItemString(result, "boards", x);
    Py_DECREF(x);

    x = PyInt_FromLong(Nmeasurements_points);
    PyDict_SetItemString(result, "points", x);
    Py_DECREF(x);

    x = PyInt_FromLong(Nmeasurements_all);
    PyDict_SetItemString(result, "all", x);
    Py_DECREF(x);

    return result;
}

static PyObject* SolverContext_pack_unpack(SolverContext* self,
                                           PyObject* args,
                                           bool pack)
{
    if( self->ctx == NULL )
    {
        PyErr_SetString(PyExc_RuntimeError, "I need a non-empty context");
        return NULL;
    }

    PyObject*      result = NULL;
    PyArrayObject* p      = NULL;
    if(!PyArg_ParseTuple( args, "O&", PyArray_Converter, &p )) goto done;

    if( PyArray_TYPE(p) != NPY_DOUBLE )
    {
        PyErr_SetString(PyExc_RuntimeError, "The input array MUST have values of type 'float'");
        goto done;
    }

    if( !PyArray_IS_C_CONTIGUOUS(p) )
    {
        PyErr_SetString(PyExc_RuntimeError, "The input array MUST be a C-style contiguous array");
        goto done;
    }

    int       ndim       = PyArray_NDIM(p);
    npy_intp* dims       = PyArray_DIMS(p);
    if( ndim <= 0 || dims[ndim-1] <= 0 )
    {
        PyErr_SetString(PyExc_RuntimeError, "The input array MUST have non-degenerate data in it");
        goto done;
    }

    int Nstate = self->ctx->beforeStep->Jt->nrow;
    if( dims[ndim-1] != Nstate )
    {
        PyErr_Format(PyExc_RuntimeError, "The input array MUST have last dimension of size Nstate=%d; instead got %ld",
                     Nstate, dims[ndim-1]);
        goto done;
    }

    double* x = (double*)PyArray_DATA(p);
    if(pack)
        for(int i=0; i<PyArray_SIZE(p)/Nstate; i++)
        {
            mrcal_pack_solver_state_vector( x,
                                            self->distortion_model, self->problem_details,
                                            self->Ncameras, self->Nframes, self->Npoints );
            x = &x[Nstate];
        }
    else
        for(int i=0; i<PyArray_SIZE(p)/Nstate; i++)
        {
            mrcal_unpack_solver_state_vector( x,
                                              self->distortion_model, self->problem_details,
                                              self->Ncameras, self->Nframes, self->Npoints );
            x = &x[Nstate];
        }


    Py_INCREF(Py_None);
    result = Py_None;

 done:
    Py_XDECREF(p);
    return result;
}
static PyObject* SolverContext_pack(SolverContext* self, PyObject* args)
{
    return SolverContext_pack_unpack(self, args, true);
}
static PyObject* SolverContext_unpack(SolverContext* self, PyObject* args)
{
    return SolverContext_pack_unpack(self, args, false);
}




static const char SolverContext_J_docstring[] =
#include "SolverContext_J.docstring.h"
    ;
static const char SolverContext_state_index_intrinsic_core_docstring[] =
#include "SolverContext_state_index_intrinsic_core.docstring.h"
    ;
static const char SolverContext_state_index_intrinsic_distortions_docstring[] =
#include "SolverContext_state_index_intrinsic_distortions.docstring.h"
    ;
static const char SolverContext_state_index_camera_rt_docstring[] =
#include "SolverContext_state_index_camera_rt.docstring.h"
    ;
static const char SolverContext_state_index_frame_rt_docstring[] =
#include "SolverContext_state_index_frame_rt.docstring.h"
    ;
static const char SolverContext_state_index_point_docstring[] =
#include "SolverContext_state_index_point.docstring.h"
    ;
static const char SolverContext_state_index_calobject_warp_docstring[] =
#include "SolverContext_state_index_calobject_warp.docstring.h"
    ;
static const char SolverContext_num_measurements_docstring[] =
#include "SolverContext_num_measurements.docstring.h"
    ;
static const char SolverContext_pack_docstring[] =
#include "SolverContext_pack.docstring.h"
    ;
static const char SolverContext_unpack_docstring[] =
#include "SolverContext_unpack.docstring.h"
    ;

static PyMethodDef SolverContext_methods[] =
    { PYMETHODDEF_ENTRY(SolverContext_, J,                                 METH_NOARGS),
      PYMETHODDEF_ENTRY(SolverContext_, state_index_intrinsic_core,        METH_VARARGS),
      PYMETHODDEF_ENTRY(SolverContext_, state_index_intrinsic_distortions, METH_VARARGS),
      PYMETHODDEF_ENTRY(SolverContext_, state_index_camera_rt,             METH_VARARGS),
      PYMETHODDEF_ENTRY(SolverContext_, state_index_frame_rt,              METH_VARARGS),
      PYMETHODDEF_ENTRY(SolverContext_, state_index_point,                 METH_VARARGS),
      PYMETHODDEF_ENTRY(SolverContext_, state_index_calobject_warp,        METH_NOARGS),
      PYMETHODDEF_ENTRY(SolverContext_, num_measurements,                  METH_NOARGS),
      PYMETHODDEF_ENTRY(SolverContext_, pack,                              METH_VARARGS),
      PYMETHODDEF_ENTRY(SolverContext_, unpack,                            METH_VARARGS),
      {}
    };

static PyTypeObject SolverContextType =
{
     PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "mrcal.SolverContext",
    .tp_basicsize = sizeof(SolverContext),
    .tp_new       = PyType_GenericNew,
    .tp_dealloc   = (destructor)SolverContext_dealloc,
    .tp_methods   = SolverContext_methods,
    .tp_str       = (reprfunc)SolverContext_str,
    .tp_repr      = (reprfunc)SolverContext_str,
    .tp_flags     = Py_TPFLAGS_DEFAULT,
    .tp_doc       = "Opaque solver context used by mrcal",
};


static PyObject* getNdistortionParams(PyObject* NPY_UNUSED(self),
                                      PyObject* args)
{
    PyObject* result = NULL;
    SET_SIGINT();

    PyObject* distortion_model_string = NULL;
    if(!PyArg_ParseTuple( args, STRING_OBJECT, &distortion_model_string ))
        goto done;

    const char* distortion_model_cstring =
        PyString_AsString(distortion_model_string);
    if( distortion_model_cstring == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Distortion model was not passed in. Must be a string, one of ("
                        DISTORTION_LIST( QUOTED_LIST_WITH_COMMA )
                        ")");
        goto done;
    }

    distortion_model_t distortion_model = mrcal_distortion_model_from_name(distortion_model_cstring);
    if( distortion_model == DISTORTION_INVALID )
    {
        PyErr_Format(PyExc_RuntimeError, "Invalid distortion model was passed in: '%s'. Must be a string, one of ("
                     DISTORTION_LIST( QUOTED_LIST_WITH_COMMA )
                     ")",
                     distortion_model_cstring);
        goto done;
    }

    int Ndistortions = mrcal_getNdistortionParams(distortion_model);

    result = Py_BuildValue("i", Ndistortions);

 done:
    RESET_SIGINT();
    return result;
}

static PyObject* getSupportedDistortionModels(PyObject* NPY_UNUSED(self),
                                              PyObject* NPY_UNUSED(args))
{
    PyObject* result = NULL;
    SET_SIGINT();
    const char* const* names = mrcal_getSupportedDistortionModels();

    // I now have a NULL-terminated list of NULL-terminated strings. Get N
    int N=0;
    while(names[N] != NULL)
        N++;

    result = PyTuple_New(N);
    if(result == NULL)
    {
        PyErr_Format(PyExc_RuntimeError, "Failed PyTuple_New(%d)", N);
        goto done;
    }

    for(int i=0; i<N; i++)
    {
        PyObject* name = Py_BuildValue("s", names[i]);
        if( name == NULL )
        {
            PyErr_Format(PyExc_RuntimeError, "Failed Py_BuildValue...");
            Py_DECREF(result);
            result = NULL;
            goto done;
        }
        PyTuple_SET_ITEM(result, i, name);
    }

 done:
    RESET_SIGINT();
    return result;
}

static PyObject* getNextDistortionModel(PyObject* NPY_UNUSED(self),
                                        PyObject* args)
{
    PyObject* result = NULL;
    SET_SIGINT();

    PyObject* distortion_model_now_string   = NULL;
    PyObject* distortion_model_final_string = NULL;
    if(!PyArg_ParseTuple( args, STRING_OBJECT STRING_OBJECT,
                          &distortion_model_now_string,
                          &distortion_model_final_string))
        goto done;

    const char* distortion_model_now_cstring = PyString_AsString(distortion_model_now_string);
    if( distortion_model_now_cstring == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "distortion_model_now was not passed in. Must be a string, one of ("
                        DISTORTION_LIST( QUOTED_LIST_WITH_COMMA )
                        ")");
        goto done;
    }
    const char* distortion_model_final_cstring = PyString_AsString(distortion_model_final_string);
    if( distortion_model_final_cstring == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "distortion_model_final was not passed in. Must be a string, one of ("
                        DISTORTION_LIST( QUOTED_LIST_WITH_COMMA )
                        ")");
        goto done;
    }

    distortion_model_t distortion_model_now = mrcal_distortion_model_from_name(distortion_model_now_cstring);
    if( distortion_model_now == DISTORTION_INVALID )
    {
        PyErr_Format(PyExc_RuntimeError, "Invalid distortion_model_now was passed in: '%s'. Must be a string, one of ("
                     DISTORTION_LIST( QUOTED_LIST_WITH_COMMA )
                     ")",
                     distortion_model_now_cstring);
        goto done;
    }
    distortion_model_t distortion_model_final = mrcal_distortion_model_from_name(distortion_model_final_cstring);
    if( distortion_model_final == DISTORTION_INVALID )
    {
        PyErr_Format(PyExc_RuntimeError, "Invalid distortion_model_final was passed in: '%s'. Must be a string, one of ("
                     DISTORTION_LIST( QUOTED_LIST_WITH_COMMA )
                     ")",
                     distortion_model_final_cstring);
        goto done;
    }

    distortion_model_t distortion_model =
        mrcal_getNextDistortionModel(distortion_model_now, distortion_model_final);
    if(distortion_model == DISTORTION_INVALID)
    {
        PyErr_Format(PyExc_RuntimeError, "Couldn't figure out the 'next' distortion model from '%s' to '%s'",
                     distortion_model_now_cstring, distortion_model_final_cstring);
        goto done;
    }

    result = Py_BuildValue("s", mrcal_distortion_model_name(distortion_model));

 done:
    RESET_SIGINT();
    return result;
}

// just like PyArray_Converter(), but leave None as None
static
int PyArray_Converter_leaveNone(PyObject* obj, PyObject** address)
{
    if(obj == Py_None)
    {
        *address = Py_None;
        Py_INCREF(Py_None);
        return 1;
    }
    return PyArray_Converter(obj,address);
}


// project(), unproject() and unproject_z1() have very similar arguments and
// operation, so the logic is consolidated as much as possible in these
// functions. The first arg is called "points" in both cases, but is 2d in one
// case, and 3d in the other

#define PROJECT_ARGUMENTS_REQUIRED(_)                                   \
    _(points,           PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, points,     NPY_DOUBLE, {} ) \
    _(distortion_model, PyObject*,      NULL,    STRING_OBJECT,                         , NULL,       -1,         {} ) \
    _(intrinsics,       PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, intrinsics, NPY_DOUBLE, {} ) \

#define PROJECT_ARGUMENTS_OPTIONAL(_) \
    _(get_gradients,    PyObject*,  Py_False,    "O",                                   , NULL,      -1, {})

static bool _un_project_validate_args( // out
                                      distortion_model_t* distortion_model_type,

                                      // in
                                      int dim_points, // 3 for project(), 2 for unproject()
                                      PROJECT_ARGUMENTS_REQUIRED(ARG_LIST_DEFINE)
                                      PROJECT_ARGUMENTS_OPTIONAL(ARG_LIST_DEFINE)
                                      void* dummy __attribute__((unused)))
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-to-pointer-cast"
    PROJECT_ARGUMENTS_REQUIRED(CHECK_LAYOUT);
    PROJECT_ARGUMENTS_OPTIONAL(CHECK_LAYOUT);
#pragma GCC diagnostic pop

    if( PyArray_NDIM(intrinsics) != 1 )
    {
        PyErr_SetString(PyExc_RuntimeError, "'intrinsics' must have exactly 1 dim");
        return false;
    }

    if( PyArray_NDIM(points) < 1 )
    {
        PyErr_SetString(PyExc_RuntimeError, "'points' must have ndims >= 1");
        return false;
    }
    if( dim_points != PyArray_DIMS(points)[ PyArray_NDIM(points)-1 ] )
    {
        PyErr_Format(PyExc_RuntimeError, "points.shape[-1] MUST be %d. Instead got %ld",
                     dim_points,
                     PyArray_DIMS(points)[PyArray_NDIM(points)-1] );
        return false;
    }

    const char* distortion_model_cstring = PyString_AsString(distortion_model);
    if( distortion_model_cstring == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Distortion model was not passed in. Must be a string, one of ("
                        DISTORTION_LIST( QUOTED_LIST_WITH_COMMA )
                        ")");
        return false;
    }

    *distortion_model_type = mrcal_distortion_model_from_name(distortion_model_cstring);
    if( *distortion_model_type == DISTORTION_INVALID )
    {
        PyErr_Format(PyExc_RuntimeError, "Invalid distortion model was passed in: '%s'. Must be a string, one of ("
                     DISTORTION_LIST( QUOTED_LIST_WITH_COMMA )
                     ")",
                     distortion_model_cstring);
        return false;
    }

    int NdistortionParams = mrcal_getNdistortionParams(*distortion_model_type);
    if( N_INTRINSICS_CORE + NdistortionParams != PyArray_DIMS(intrinsics)[0] )
    {
        PyErr_Format(PyExc_RuntimeError, "intrinsics.shape[0] MUST be %d. Instead got %ld",
                     N_INTRINSICS_CORE + NdistortionParams,
                     PyArray_DIMS(intrinsics)[0] );
        return false;
    }

    return true;
}

#define _UN_PROJECT_PREAMBLE(ARGUMENTS_REQUIRED,ARGUMENTS_OPTIONAL,ARGUMENTS_OPTIONAL_VALIDATE,dim_points_in,dim_points_out) \
    PyObject*      result          = NULL;                              \
                                                                        \
    SET_SIGINT();                                                       \
    PyArrayObject* out             = NULL;                              \
    __attribute__((unused)) PyArrayObject* dxy_dintrinsics = NULL;      \
    __attribute__((unused)) PyArrayObject* dxy_dp          = NULL;      \
                                                                        \
    ARGUMENTS_REQUIRED(ARG_DEFINE);                                     \
    ARGUMENTS_OPTIONAL(ARG_DEFINE);                                     \
                                                                        \
    char* keywords[] = { ARGUMENTS_REQUIRED(NAMELIST)                   \
                         ARGUMENTS_OPTIONAL(NAMELIST)                   \
                         NULL};                                         \
    if(!PyArg_ParseTupleAndKeywords( args, kwargs,                      \
                                     ARGUMENTS_REQUIRED(PARSECODE) "|"  \
                                     ARGUMENTS_OPTIONAL(PARSECODE),     \
                                                                        \
                                     keywords,                          \
                                                                        \
                                     ARGUMENTS_REQUIRED(PARSEARG)       \
                                     ARGUMENTS_OPTIONAL(PARSEARG) NULL)) \
        goto done;                                                      \
                                                                        \
    /* if the input points array is degenerate, return a degenerate thing */ \
    if( points == NULL  || (PyObject*)points == Py_None)                \
    {                                                                   \
        result = Py_None;                                               \
        Py_INCREF(result);                                              \
        goto done;                                                      \
    }                                                                   \
                                                                        \
    distortion_model_t distortion_model_type;                           \
    if(!_un_project_validate_args( &distortion_model_type,              \
                                   dim_points_in,                       \
                                   ARGUMENTS_REQUIRED(ARG_LIST_CALL)    \
                                   ARGUMENTS_OPTIONAL_VALIDATE(ARG_LIST_CALL) \
                                   NULL))                               \
        goto done;                                                      \
                                                                        \
    int Nintrinsics = PyArray_DIMS(intrinsics)[0];                      \
                                                                        \
    /* poor man's broadcasting of the inputs. I compute the total number of */ \
    /* points by multiplying the extra broadcasted dimensions. And I set up the */ \
    /* outputs to have the appropriate broadcasted dimensions        */ \
    const npy_intp* leading_dims  = PyArray_DIMS(points);               \
    int             Nleading_dims = PyArray_NDIM(points)-1;             \
    int Npoints = PyArray_SIZE(points) / leading_dims[Nleading_dims];   \
    bool get_gradients_bool = get_gradients && PyObject_IsTrue(get_gradients); \
                                                                        \
    {                                                                   \
        npy_intp dims[Nleading_dims+2]; /* one extra for the gradients */ \
        memcpy(dims, leading_dims, Nleading_dims*sizeof(dims[0]));      \
                                                                        \
        dims[Nleading_dims + 0] = dim_points_out;                       \
        out = (PyArrayObject*)PyArray_SimpleNew(Nleading_dims+1,        \
                                                dims,                   \
                                                NPY_DOUBLE);            \
        if( get_gradients_bool )                                        \
        {                                                               \
            dims[Nleading_dims + 0] = 2;                                \
            dims[Nleading_dims + 1] = Nintrinsics;                      \
            dxy_dintrinsics = (PyArrayObject*)PyArray_SimpleNew(Nleading_dims+2, \
                                                                dims,   \
                                                                NPY_DOUBLE); \
                                                                        \
            dims[Nleading_dims + 0] = 2;                                \
            dims[Nleading_dims + 1] = 3;                                \
            dxy_dp          = (PyArrayObject*)PyArray_SimpleNew(Nleading_dims+2, \
                                                                dims,   \
                                                                NPY_DOUBLE); \
        }                                                               \
    }

#define _UN_PROJECT_POSTAMBLE(ARGUMENTS_REQUIRED,       \
                              ARGUMENTS_OPTIONAL)       \
                                                        \
 done:                                                  \
    ARGUMENTS_REQUIRED(FREE_PYARRAY) ;                  \
    ARGUMENTS_OPTIONAL(FREE_PYARRAY) ;                  \
    RESET_SIGINT();                                     \
    return result



static PyObject* project(PyObject* NPY_UNUSED(self),
                         PyObject* args,
                         PyObject* kwargs)
{
    _UN_PROJECT_PREAMBLE(PROJECT_ARGUMENTS_REQUIRED,
                         PROJECT_ARGUMENTS_OPTIONAL,
                         PROJECT_ARGUMENTS_OPTIONAL,
                         3, 2);

    if(! mrcal_project((point2_t*)PyArray_DATA(out),
                       get_gradients_bool ? (double*)PyArray_DATA(dxy_dintrinsics) : NULL,
                       get_gradients_bool ? (point3_t*)PyArray_DATA(dxy_dp)  : NULL,

                       (const point3_t*)PyArray_DATA(points),
                       Npoints,
                       distortion_model_type,
                       // core, distortions concatenated
                       (const double*)PyArray_DATA(intrinsics)))
    {
        PyErr_SetString(PyExc_RuntimeError, "mrcal_project() failed!");
        Py_DECREF((PyObject*)out);
        goto done;
    }

    if( get_gradients_bool )
    {
        result = PyTuple_Pack(3, out, dxy_dp, dxy_dintrinsics);
        Py_DECREF(out);
        Py_DECREF(dxy_dp);
        Py_DECREF(dxy_dintrinsics);
    }
    else
        result = (PyObject*)out;

    _UN_PROJECT_POSTAMBLE(PROJECT_ARGUMENTS_REQUIRED,
                          PROJECT_ARGUMENTS_OPTIONAL);
}

#define UNPROJECT_ARGUMENTS_REQUIRED(_) PROJECT_ARGUMENTS_REQUIRED(_)
#define UNPROJECT_ARGUMENTS_OPTIONAL(_)

static PyObject* unproject(PyObject* NPY_UNUSED(self),
                           PyObject* args,
                           PyObject* kwargs)
{
    // unproject, unproject_z1() have the same arguments as project(), except no
    // optional ones (no gradient reporting). The first arg is called "points"
    // in both cases, but is 2d in one case, and 3d in the other
    PyObject* get_gradients = NULL;

#define UNPROJECT_ARGUMENTS_OPTIONAL(_)
    _UN_PROJECT_PREAMBLE(UNPROJECT_ARGUMENTS_REQUIRED,
                         UNPROJECT_ARGUMENTS_OPTIONAL,
                         PROJECT_ARGUMENTS_OPTIONAL,
                         2, 3);

    if(! mrcal_unproject((point3_t*)PyArray_DATA(out),

                         (const point2_t*)PyArray_DATA(points),
                         Npoints,
                         distortion_model_type,
                         /* core, distortions concatenated */
                         (const double*)PyArray_DATA(intrinsics)))
    {
        PyErr_SetString(PyExc_RuntimeError, "mrcal_unproject() failed!");
        Py_DECREF((PyObject*)out);
        goto done;
    }

    result = (PyObject*)out;

    _UN_PROJECT_POSTAMBLE(UNPROJECT_ARGUMENTS_REQUIRED,
                          UNPROJECT_ARGUMENTS_OPTIONAL);
}

static PyObject* unproject_z1(PyObject* NPY_UNUSED(self),
                              PyObject* args,
                              PyObject* kwargs)
{
    // unproject, unproject_z1() have the same arguments as project(), except no
    // optional ones (no gradient reporting). The first arg is called "points"
    // in both cases, but is 2d in one case, and 3d in the other
    PyObject* get_gradients = NULL;

    _UN_PROJECT_PREAMBLE(UNPROJECT_ARGUMENTS_REQUIRED,
                         UNPROJECT_ARGUMENTS_OPTIONAL,
                         PROJECT_ARGUMENTS_OPTIONAL,
                         2, 2);

    if(! mrcal_unproject_z1((point2_t*)PyArray_DATA(out),

                            (const point2_t*)PyArray_DATA(points),
                            Npoints,
                            distortion_model_type,
                            /* core, distortions concatenated */
                            (const double*)PyArray_DATA(intrinsics)))
    {
        PyErr_SetString(PyExc_RuntimeError, "mrcal_unproject_z1() failed!");
        Py_DECREF((PyObject*)out);
        goto done;
    }

    result = (PyObject*)out;

    _UN_PROJECT_POSTAMBLE(UNPROJECT_ARGUMENTS_REQUIRED,
                          UNPROJECT_ARGUMENTS_OPTIONAL);
}

#define OPTIMIZE_ARGUMENTS_REQUIRED(_)                                  \
    _(intrinsics,                         PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, intrinsics,                  NPY_DOUBLE, {-1 COMMA -1       } ) \
    _(extrinsics,                         PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, extrinsics,                  NPY_DOUBLE, {-1 COMMA  6       } ) \
    _(frames,                             PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, frames,                      NPY_DOUBLE, {-1 COMMA  6       } ) \
    _(points,                             PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, points,                      NPY_DOUBLE, {-1 COMMA  3       } ) \
    _(observations_board,                 PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, observations_board,          NPY_DOUBLE, {-1 COMMA -1 COMMA -1 COMMA 3 } ) \
    _(indices_frame_camera_board,         PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, indices_frame_camera_board,  NPY_INT,    {-1 COMMA  2       } ) \
    _(observations_point,                 PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, observations_point,          NPY_DOUBLE, {-1 COMMA  4       } ) \
    _(indices_point_camera_points,        PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, indices_point_camera_points, NPY_INT,    {-1 COMMA  2       } ) \
    _(distortion_model,                   PyObject*,      NULL,    STRING_OBJECT,  ,                        NULL,                        -1,         {}                   ) \
    _(imagersizes,                        PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, imagersizes,                 NPY_INT,    {-1 COMMA 2        } )

#define OPTIMIZE_ARGUMENTS_OPTIONAL(_) \
    _(calobject_warp,                     PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, calobject_warp,              NPY_DOUBLE, {2}                  ) \
    _(do_optimize_intrinsic_core,         PyObject*,      Py_True, "O",  ,                                  NULL,           -1,         {})  \
    _(do_optimize_intrinsic_distortions,  PyObject*,      Py_True, "O",  ,                                  NULL,           -1,         {})  \
    _(do_optimize_extrinsics,             PyObject*,      Py_True, "O",  ,                                  NULL,           -1,         {})  \
    _(do_optimize_frames,                 PyObject*,      Py_True, "O",  ,                                  NULL,           -1,         {})  \
    _(do_optimize_cahvor_optical_axis,    PyObject*,      Py_True, "O",  ,                                  NULL,           -1,         {})  \
    _(do_optimize_calobject_warp,         PyObject*,      Py_False,"O",  ,                                  NULL,           -1,         {})  \
    _(skipped_observations_board,         PyObject*,      NULL,    "O",  ,                                  NULL,           -1,         {})  \
    _(skipped_observations_point,         PyObject*,      NULL,    "O",  ,                                  NULL,           -1,         {})  \
    _(calibration_object_spacing,         double,         -1.0,    "d",  ,                                  NULL,           -1,         {})  \
    _(calibration_object_width_n,         int,            -1,      "i",  ,                                  NULL,           -1,         {})  \
    _(outlier_indices,                    PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, outlier_indices,NPY_INT,    {-1} ) \
    _(roi,                                PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, roi,            NPY_DOUBLE, {-1 COMMA 4} ) \
    _(VERBOSE,                            PyObject*,      NULL,    "O",  ,                                  NULL,           -1,         {})  \
    _(get_invJtJ_intrinsics,              PyObject*,      NULL,    "O",  ,                                  NULL,           -1,         {})  \
    _(skip_outlier_rejection,             PyObject*,      NULL,    "O",  ,                                  NULL,           -1,         {})  \
    _(skip_regularization,                PyObject*,      NULL,    "O",  ,                                  NULL,           -1,         {})  \
    _(observed_pixel_uncertainty,         double,         -1.0,    "d",  ,                                  NULL,           -1,         {})  \
    _(solver_context,                     SolverContext*, NULL,    "O",  (PyObject*),                       NULL,           -1,         {})

#define OPTIMIZE_ARGUMENTS_ALL(_) \
    OPTIMIZE_ARGUMENTS_REQUIRED(_) \
    OPTIMIZE_ARGUMENTS_OPTIONAL(_)

static bool optimize_validate_args( // out
                                    distortion_model_t* distortion_model_type,

                                    // in
                                    OPTIMIZE_ARGUMENTS_REQUIRED(ARG_LIST_DEFINE)
                                    OPTIMIZE_ARGUMENTS_OPTIONAL(ARG_LIST_DEFINE)

                                    void* dummy __attribute__((unused)))
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-to-pointer-cast"
    OPTIMIZE_ARGUMENTS_ALL(CHECK_LAYOUT) ;
#pragma GCC diagnostic pop

    if(PyObject_IsTrue(do_optimize_calobject_warp) &&
       (calobject_warp == NULL || (PyObject*)calobject_warp == Py_None) )
    {
        PyErr_SetString(PyExc_RuntimeError, "if(do_optimize_calobject_warp) then calobject_warp MUST be given as an array to seed the optimization and to receive the results");
        return false;
    }

    int Ncameras = PyArray_DIMS(intrinsics)[0];
    if( Ncameras-1 !=
        PyArray_DIMS(extrinsics)[0] )
    {
        PyErr_Format(PyExc_RuntimeError, "Inconsistent Ncameras: 'extrinsics' says %ld, 'intrinsics' says %ld",
                     PyArray_DIMS(extrinsics)[0] + 1,
                     PyArray_DIMS(intrinsics)[0] );
        return false;
    }
    if( PyArray_DIMS(imagersizes)[0] != Ncameras )
    {
        PyErr_Format(PyExc_RuntimeError, "Inconsistent Ncameras: 'extrinsics' says %ld, 'imagersizes' says %ld",
                     PyArray_DIMS(extrinsics)[0] + 1,
                     PyArray_DIMS(imagersizes)[0]);
        return false;
    }
    if( roi != NULL && (PyObject*)roi != Py_None && PyArray_DIMS(roi)[0] != Ncameras )
    {
        PyErr_Format(PyExc_RuntimeError, "Inconsistent Ncameras: 'extrinsics' says %ld, 'roi' says %ld",
                     PyArray_DIMS(extrinsics)[0] + 1,
                     PyArray_DIMS(roi)[0]);
        return false;
    }

    static_assert( sizeof(pose_t)/sizeof(double) == 6, "pose_t is assumed to contain 6 elements");

    long int NobservationsBoard = PyArray_DIMS(observations_board)[0];
    if( PyArray_DIMS(indices_frame_camera_board)[0] != NobservationsBoard )
    {
        PyErr_Format(PyExc_RuntimeError, "Inconsistent NobservationsBoard: 'observations_board' says %ld, 'indices_frame_camera_board' says %ld",
                     NobservationsBoard,
                     PyArray_DIMS(indices_frame_camera_board)[0]);
        return false;
    }

    // calibration_object_spacing and calibration_object_width_n must be > 0 OR
    // we have to not be using a calibration board
    if( NobservationsBoard > 0 )
    {
        if( calibration_object_spacing <= 0.0 )
        {
            PyErr_Format(PyExc_RuntimeError, "We have board observations, so calibration_object_spacing MUST be a valid float > 0");
            return false;
        }

        if( calibration_object_width_n <= 0 )
        {
            PyErr_Format(PyExc_RuntimeError, "We have board observations, so calibration_object_width_n MUST be a valid int > 0");
            return false;
        }


        if( calibration_object_width_n != PyArray_DIMS(observations_board)[1] ||
            calibration_object_width_n != PyArray_DIMS(observations_board)[2] )
        {
            PyErr_Format(PyExc_RuntimeError, "observations_board.shape[1:] MUST be (%d,%d,3). Instead got (%ld,%ld,%ld)",
                         calibration_object_width_n, calibration_object_width_n,
                         PyArray_DIMS(observations_board)[1],
                         PyArray_DIMS(observations_board)[2],
                         PyArray_DIMS(observations_board)[3]);
            return false;
        }
    }

    long int NobservationsPoint = PyArray_DIMS(observations_point)[0];
    if( PyArray_DIMS(indices_point_camera_points)[0] != NobservationsPoint )
    {
        PyErr_Format(PyExc_RuntimeError, "Inconsistent NobservationsPoint: 'observations_point' says %ld, 'indices_point_camera_points' says %ld",
                     NobservationsPoint,
                     PyArray_DIMS(indices_point_camera_points)[0]);
        return false;
    }

    const char* distortion_model_cstring =
        PyString_AsString(distortion_model);
    if( distortion_model_cstring == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Distortion model was not passed in. Must be a string, one of ("
                        DISTORTION_LIST( QUOTED_LIST_WITH_COMMA )
                        ")");
        return false;
    }

    *distortion_model_type = mrcal_distortion_model_from_name(distortion_model_cstring);
    if( *distortion_model_type == DISTORTION_INVALID )
    {
        PyErr_Format(PyExc_RuntimeError, "Invalid distortion model was passed in: '%s'. Must be a string, one of ("
                     DISTORTION_LIST( QUOTED_LIST_WITH_COMMA )
                     ")",
                     distortion_model_cstring);
        return false;
    }


    int NdistortionParams = mrcal_getNdistortionParams(*distortion_model_type);
    if( N_INTRINSICS_CORE + NdistortionParams != PyArray_DIMS(intrinsics)[1] )
    {
        PyErr_Format(PyExc_RuntimeError, "intrinsics.shape[1] MUST be %d. Instead got %ld",
                     N_INTRINSICS_CORE + NdistortionParams,
                     PyArray_DIMS(intrinsics)[1] );
        return false;
    }

    if( skipped_observations_board != NULL &&
        skipped_observations_board != Py_None)
    {
        if( !PySequence_Check(skipped_observations_board) )
        {
            PyErr_Format(PyExc_RuntimeError, "skipped_observations_board MUST be None or an iterable of monotonically-increasing integers >= 0");
            return false;
        }

        int Nskipped_observations = (int)PySequence_Size(skipped_observations_board);
        long iskip_last = -1;
        for(int i=0; i<Nskipped_observations; i++)
        {
            PyObject* nextskip = PySequence_GetItem(skipped_observations_board, i);
            if(!PyInt_Check(nextskip))
            {
                PyErr_Format(PyExc_RuntimeError, "skipped_observations_board MUST be None or an iterable of monotonically-increasing integers >= 0");
                return false;
            }
            long iskip = PyInt_AsLong(nextskip);
            if(iskip <= iskip_last)
            {
                PyErr_Format(PyExc_RuntimeError, "skipped_observations_board MUST be None or an iterable of monotonically-increasing integers >= 0");
                return false;
            }
            iskip_last = iskip;
        }
    }

    if( skipped_observations_point != NULL &&
        skipped_observations_point != Py_None)
    {
        if( !PySequence_Check(skipped_observations_point) )
        {
            PyErr_Format(PyExc_RuntimeError, "skipped_observations_point MUST be None or an iterable of monotonically-increasing integers >= 0");
            return false;
        }

        int Nskipped_observations = (int)PySequence_Size(skipped_observations_point);
        long iskip_last = -1;
        for(int i=0; i<Nskipped_observations; i++)
        {
            PyObject* nextskip = PySequence_GetItem(skipped_observations_point, i);
            if(!PyInt_Check(nextskip))
            {
                PyErr_Format(PyExc_RuntimeError, "skipped_observations_point MUST be None or an iterable of monotonically-increasing integers >= 0");
                return false;
            }
            long iskip = PyInt_AsLong(nextskip);
            if(iskip <= iskip_last)
            {
                PyErr_Format(PyExc_RuntimeError, "skipped_observations_point MUST be None or an iterable of monotonically-increasing integers >= 0");
                return false;
            }
            iskip_last = iskip;
        }
    }

    // make sure the indices arrays are valid: the data is monotonic and
    // in-range
    int Nframes = 0;
    if( frames != NULL && Py_None != (PyObject*)frames )
        Nframes = PyArray_DIMS(frames)[0];
    int i_frame_last  = -1;
    int i_camera_last = -1;
    for(int i_observation=0; i_observation<NobservationsBoard; i_observation++)
    {
        // check for monotonicity and in-rangeness
        int i_frame  = ((int*)PyArray_DATA(indices_frame_camera_board))[i_observation*2 + 0];
        int i_camera = ((int*)PyArray_DATA(indices_frame_camera_board))[i_observation*2 + 1];

        // First I make sure everything is in-range
        if(i_frame < 0 || i_frame >= Nframes)
        {
            PyErr_Format(PyExc_RuntimeError, "i_frame MUST be in [0,%d], instead got %d in row %d of indices_frame_camera_board",
                         Nframes-1, i_frame, i_observation);
            return false;
        }
        if(i_camera < 0 || i_camera >= Ncameras)
        {
            PyErr_Format(PyExc_RuntimeError, "i_camera MUST be in [0,%d], instead got %d in row %d of indices_frame_camera_board",
                         Ncameras-1, i_camera, i_observation);
            return false;
        }

        // And then I check monotonicity
        if(i_frame == i_frame_last)
        {
            if( i_camera <= i_camera_last )
            {
                PyErr_Format(PyExc_RuntimeError, "i_camera MUST be monotonically increasing in indices_frame_camera_board. Instead row %d (frame %d) of indices_frame_camera_board has i_camera=%d after seeing i_camera=%d",
                             i_observation, i_frame, i_camera, i_camera_last);
                return false;
            }
        }
        else if( i_frame < i_frame_last )
        {
            PyErr_Format(PyExc_RuntimeError, "i_frame MUST be monotonically increasing in indices_frame_camera_board. Instead row %d of indices_frame_camera_board has i_frame=%d after seeing i_frame=%d",
                         i_observation, i_frame, i_frame_last);
            return false;
        }

        i_frame_last  = i_frame;
        i_camera_last = i_camera;
    }
    int Npoints = 0;
    if( points != NULL && Py_None != (PyObject*)points )
        Npoints = PyArray_DIMS(points)[0];
    int i_point_last = -1;
    i_camera_last = -1;
    for(int i_observation=0; i_observation<NobservationsPoint; i_observation++)
    {
        int i_point  = ((int*)PyArray_DATA(indices_point_camera_points))[i_observation*2 + 0];
        int i_camera = ((int*)PyArray_DATA(indices_point_camera_points))[i_observation*2 + 1];

        // First I make sure everything is in-range
        if(i_point < 0 || i_point >= Npoints)
        {
            PyErr_Format(PyExc_RuntimeError, "i_point MUST be in [0,%d], instead got %d in row %d of indices_point_camera_points",
                         Npoints-1, i_point, i_observation);
            return false;
        }
        if(i_camera < 0 || i_camera >= Ncameras)
        {
            PyErr_Format(PyExc_RuntimeError, "i_camera MUST be in [0,%d], instead got %d in row %d of indices_point_camera_points",
                         Ncameras-1, i_camera, i_observation);
            return false;
        }

        // And then I check monotonicity
        if(i_point == i_point_last)
        {
            if( i_camera <= i_camera_last )
            {
                PyErr_Format(PyExc_RuntimeError, "i_camera MUST be monotonically increasing in indices_point_camera_points. Instead row %d (point %d) of indices_point_camera_points has i_camera=%d after seeing i_camera=%d",
                             i_observation, i_point, i_camera, i_camera_last);
                return false;
            }
        }
        else if( i_point < i_point_last )
        {
            PyErr_Format(PyExc_RuntimeError, "i_point MUST be monotonically increasing in indices_point_camera_points. Instead row %d of indices_point_camera_points has i_point=%d after seeing i_point=%d",
                         i_observation, i_point, i_point_last);
            return false;
        }

        i_point_last  = i_point;
        i_camera_last = i_camera;
    }




    if( PyObject_IsTrue(skip_outlier_rejection) )
    {
        // skipping outlier rejection. The pixel uncertainty isn't used and
        // doesn't matter
    }
    else
    {
        // not skipping outlier rejection. The pixel uncertainty is used and
        // must be valid
        if( observed_pixel_uncertainty <= 0.0 )
        {
            PyErr_Format(PyExc_RuntimeError, "Observed_pixel_uncertainty MUST be a valid float > 0");
            return false;
        }
    }

    if( !(solver_context == NULL ||
          (PyObject*)solver_context == Py_None ||
          Py_TYPE(solver_context) == &SolverContextType) )
    {
        PyErr_Format(PyExc_RuntimeError, "solver_context must be None or of type mrcal.SolverContext");
        return false;
    }

    return true;
}
static PyObject* optimize(PyObject* NPY_UNUSED(self),
                          PyObject* args,
                          PyObject* kwargs)
{
    // are -log2 of the accuracy of the observation.
    //   I.e. 0 = full-res, 1 = 1/2 scale, 2 = 1/4 scale, ...


    PyObject* result = NULL;

    PyArrayObject* x_final                             = NULL;
    PyArrayObject* invJtJ_intrinsics_full              = NULL;
    PyArrayObject* invJtJ_intrinsics_observations_only = NULL;
    PyArrayObject* outlier_indices_final               = NULL;
    PyArrayObject* outside_ROI_indices_final           = NULL;
    PyObject*      pystats                             = NULL;

    SET_SIGINT();

    OPTIMIZE_ARGUMENTS_ALL(ARG_DEFINE) ;
    char* keywords[] = { OPTIMIZE_ARGUMENTS_REQUIRED(NAMELIST)
                         OPTIMIZE_ARGUMENTS_OPTIONAL(NAMELIST)
                         NULL};

    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     OPTIMIZE_ARGUMENTS_REQUIRED(PARSECODE) "|"
                                     OPTIMIZE_ARGUMENTS_OPTIONAL(PARSECODE),

                                     keywords,

                                     OPTIMIZE_ARGUMENTS_REQUIRED(PARSEARG)
                                     OPTIMIZE_ARGUMENTS_OPTIONAL(PARSEARG) NULL))
        goto done;


    // Some of my input arguments can be empty (None). The code all assumes that
    // everything is a properly-dimensions numpy array, with "empty" meaning
    // some dimension is 0. Here I make this conversion. The user can pass None,
    // and we still do the right thing.
    //
    // There's a silly implementation detail here: if you have a preprocessor
    // macro M(x), and you pass it M({1,2,3}), the preprocessor see 3 separate
    // args, not 1. That's why I have a __VA_ARGS__ here and why I instantiate a
    // separate dims[] (PyArray_SimpleNew is a macro too)
#define SET_SIZE0_IF_NONE(x, type, ...)                                 \
    ({                                                                  \
        if( x == NULL || Py_None == (PyObject*)x )                      \
        {                                                               \
            if( x != NULL ) Py_DECREF(x);                               \
            npy_intp dims[] = {__VA_ARGS__};                            \
            x = (PyArrayObject*)PyArray_SimpleNew(sizeof(dims)/sizeof(dims[0]), \
                                                  dims, type);          \
        }                                                               \
    })

    SET_SIZE0_IF_NONE(extrinsics,                 NPY_DOUBLE, 0,6);

    SET_SIZE0_IF_NONE(frames,                     NPY_DOUBLE, 0,6);
    SET_SIZE0_IF_NONE(observations_board,         NPY_DOUBLE, 0,179,171,3); // arbitrary numbers; shouldn't matter
    SET_SIZE0_IF_NONE(indices_frame_camera_board, NPY_INT,    0,2);

    SET_SIZE0_IF_NONE(points,                     NPY_DOUBLE, 0,3);
    SET_SIZE0_IF_NONE(observations_point,         NPY_DOUBLE, 0,4);
    SET_SIZE0_IF_NONE(indices_point_camera_points,NPY_INT,    0,2);
    SET_SIZE0_IF_NONE(imagersizes,                NPY_INT,    0,2);
#undef SET_NULL_IF_NONE




    distortion_model_t distortion_model_type;
    if( !optimize_validate_args(&distortion_model_type,
                                OPTIMIZE_ARGUMENTS_REQUIRED(ARG_LIST_CALL)
                                OPTIMIZE_ARGUMENTS_OPTIONAL(ARG_LIST_CALL)
                                NULL))
        goto done;

    {
        int Ncameras           = PyArray_DIMS(intrinsics)[0];
        int Nframes            = PyArray_DIMS(frames)[0];
        int Npoints            = PyArray_DIMS(points)[0];
        int NobservationsBoard = PyArray_DIMS(observations_board)[0];
        int NobservationsPoint = PyArray_DIMS(observations_point)[0];

        // The checks in optimize_validate_args() make sure these casts are kosher
        double*       c_intrinsics     = (double*)  PyArray_DATA(intrinsics);
        pose_t*       c_extrinsics     = (pose_t*)  PyArray_DATA(extrinsics);
        pose_t*       c_frames         = (pose_t*)  PyArray_DATA(frames);
        point3_t*     c_points         = (point3_t*)PyArray_DATA(points);
        point2_t*     c_calobject_warp =
            calobject_warp == NULL || (PyObject*)calobject_warp == Py_None ?
            NULL : (point2_t*)PyArray_DATA(calobject_warp);


        observation_board_t c_observations_board[NobservationsBoard];
        int Nskipped_observations_board =
            ( skipped_observations_board == NULL ||
              skipped_observations_board == Py_None ) ?
            0 :
            (int)PySequence_Size(skipped_observations_board);
        int i_skipped_observation_board = 0;
        int i_observation_board_next_skip = -1;
        if( i_skipped_observation_board < Nskipped_observations_board )
        {
            PyObject* nextskip = PySequence_GetItem(skipped_observations_board, i_skipped_observation_board);
            i_observation_board_next_skip = (int)PyInt_AsLong(nextskip);
        }

        int i_frame_current_skipped = -1;
        int i_frame_last            = -1;
        for(int i_observation=0; i_observation<NobservationsBoard; i_observation++)
        {
            int i_frame  = ((int*)PyArray_DATA(indices_frame_camera_board))[i_observation*2 + 0];
            int i_camera = ((int*)PyArray_DATA(indices_frame_camera_board))[i_observation*2 + 1];

            c_observations_board[i_observation].i_camera         = i_camera;
            c_observations_board[i_observation].i_frame          = i_frame;
            c_observations_board[i_observation].px               = &((point3_t*)PyArray_DATA(observations_board))[calibration_object_width_n*calibration_object_width_n*i_observation];

            // I skip this frame if I skip ALL observations of this frame
            if( i_frame_current_skipped >= 0 &&
                i_frame_current_skipped != i_frame )
            {
                // Ooh! We moved past the frame where we skipped all
                // observations. So I need to go back, and mark all of those as
                // skipping that frame
                for(int i_observation_skip_frame = i_observation-1;
                    i_observation_skip_frame >= 0 && c_observations_board[i_observation_skip_frame].i_frame == i_frame_current_skipped;
                    i_observation_skip_frame--)
                {
                    c_observations_board[i_observation_skip_frame].skip_frame = true;
                }
            }
            else
                c_observations_board[i_observation].skip_frame = false;

            if( i_observation == i_observation_board_next_skip )
            {
                if( i_frame_last != i_frame )
                    i_frame_current_skipped = i_frame;

                c_observations_board[i_observation].skip_observation = true;

                i_skipped_observation_board++;
                if( i_skipped_observation_board < Nskipped_observations_board )
                {
                    PyObject* nextskip = PySequence_GetItem(skipped_observations_board, i_skipped_observation_board);
                    i_observation_board_next_skip = (int)PyInt_AsLong(nextskip);
                }
                else
                    i_observation_board_next_skip = -1;
            }
            else
            {
                c_observations_board[i_observation].skip_observation = false;
                i_frame_current_skipped = -1;
            }

            i_frame_last = i_frame;
        }
        // check for frame-skips on the last observation
        if( i_frame_current_skipped >= 0 )
        {
            for(int i_observation_skip_frame = NobservationsBoard - 1;
                i_observation_skip_frame >= 0 && c_observations_board[i_observation_skip_frame].i_frame == i_frame_current_skipped;
                i_observation_skip_frame--)
            {
                c_observations_board[i_observation_skip_frame].skip_frame = true;
            }
        }





        observation_point_t c_observations_point[NobservationsPoint];
        int Nskipped_observations_point =
            ( skipped_observations_point == NULL ||
              skipped_observations_point == Py_None ) ?
            0 :
            (int)PySequence_Size(skipped_observations_point);
        int i_skipped_observation_point = 0;
        int i_observation_point_next_skip = -1;
        if( i_skipped_observation_point < Nskipped_observations_point )
        {
            PyObject* nextskip = PySequence_GetItem(skipped_observations_point, i_skipped_observation_point);
            i_observation_point_next_skip = (int)PyInt_AsLong(nextskip);
        }

        int i_point_current_skipped = -1;
        int i_point_last            = -1;
        for(int i_observation=0; i_observation<NobservationsPoint; i_observation++)
        {
            int i_point  = ((int*)PyArray_DATA(indices_point_camera_points))[i_observation*2 + 0];
            int i_camera = ((int*)PyArray_DATA(indices_point_camera_points))[i_observation*2 + 1];

            c_observations_point[i_observation].i_camera         = i_camera;
            c_observations_point[i_observation].i_point          = i_point;
            c_observations_point[i_observation].px               = *(point3_t*)(&((double*)PyArray_DATA(observations_point))[i_observation*4]);
            c_observations_point[i_observation].dist             = ((double*)PyArray_DATA(observations_point))[i_observation*4 + 3];

            // I skip this point if I skip ALL observations of this point
            if( i_point_current_skipped >= 0 &&
                i_point_current_skipped != i_point )
            {
                // Ooh! We moved past the point where we skipped all
                // observations. So I need to go back, and mark all of those as
                // skipping that point
                for(int i_observation_skip_point = i_observation-1;
                    i_observation_skip_point >= 0 && c_observations_point[i_observation_skip_point].i_point == i_point_current_skipped;
                    i_observation_skip_point--)
                {
                    c_observations_point[i_observation_skip_point].skip_point = true;
                }
            }
            else
                c_observations_point[i_observation].skip_point = false;

            if( i_observation == i_observation_point_next_skip )
            {
                if( i_point_last != i_point )
                    i_point_current_skipped = i_point;

                c_observations_point[i_observation].skip_observation = true;

                i_skipped_observation_point++;
                if( i_skipped_observation_point < Nskipped_observations_point )
                {
                    PyObject* nextskip = PySequence_GetItem(skipped_observations_point, i_skipped_observation_point);
                    i_observation_point_next_skip = (int)PyInt_AsLong(nextskip);
                }
                else
                    i_observation_point_next_skip = -1;
            }
            else
            {
                c_observations_point[i_observation].skip_observation = false;
                i_point_current_skipped = -1;
            }

            i_point_last = i_point;
        }
        // check for point-skips on the last observation
        if( i_point_current_skipped >= 0 )
        {
            for(int i_observation_skip_point = NobservationsPoint - 1;
                i_observation_skip_point >= 0 && c_observations_point[i_observation_skip_point].i_point == i_point_current_skipped;
                i_observation_skip_point--)
            {
                c_observations_point[i_observation_skip_point].skip_point = true;
            }
        }





        mrcal_problem_details_t problem_details =
            { .do_optimize_intrinsic_core        = PyObject_IsTrue(do_optimize_intrinsic_core),
              .do_optimize_intrinsic_distortions = PyObject_IsTrue(do_optimize_intrinsic_distortions),
              .do_optimize_extrinsics            = PyObject_IsTrue(do_optimize_extrinsics),
              .do_optimize_frames                = PyObject_IsTrue(do_optimize_frames),
              .do_optimize_cahvor_optical_axis   = PyObject_IsTrue(do_optimize_cahvor_optical_axis),
              .do_optimize_calobject_warp        = PyObject_IsTrue(do_optimize_calobject_warp),
              .do_skip_regularization            = skip_regularization && PyObject_IsTrue(skip_regularization)
            };

        int Nmeasurements = mrcal_getNmeasurements_all(Ncameras, NobservationsBoard,
                                                       c_observations_point, NobservationsPoint,
                                                       calibration_object_width_n,
                                                       problem_details,
                                                       distortion_model_type);

        x_final = (PyArrayObject*)PyArray_SimpleNew(1, ((npy_intp[]){Nmeasurements}), NPY_DOUBLE);
        double* c_x_final = PyArray_DATA(x_final);

        int Nintrinsics_all = mrcal_getNintrinsicParams(distortion_model_type);

        double* c_invJtJ_intrinsics_full              = NULL;
        double* c_invJtJ_intrinsics_observations_only = NULL;
        if(Nintrinsics_all != 0 &&
           get_invJtJ_intrinsics && PyObject_IsTrue(get_invJtJ_intrinsics))
        {
            invJtJ_intrinsics_full =
                (PyArrayObject*)PyArray_SimpleNew(3,
                                                  ((npy_intp[]){Ncameras,Nintrinsics_all,Nintrinsics_all}), NPY_DOUBLE);
            c_invJtJ_intrinsics_full = PyArray_DATA(invJtJ_intrinsics_full);
            invJtJ_intrinsics_observations_only =
                (PyArrayObject*)PyArray_SimpleNew(3,
                                                  ((npy_intp[]){Ncameras,Nintrinsics_all,Nintrinsics_all}), NPY_DOUBLE);
            c_invJtJ_intrinsics_observations_only = PyArray_DATA(invJtJ_intrinsics_observations_only);
        }

        const int Npoints_fromBoards =
            NobservationsBoard *
            calibration_object_width_n*calibration_object_width_n;
        outlier_indices_final     = (PyArrayObject*)PyArray_SimpleNew(1, ((npy_intp[]){Npoints_fromBoards}), NPY_INT);
        outside_ROI_indices_final = (PyArrayObject*)PyArray_SimpleNew(1, ((npy_intp[]){Npoints_fromBoards}), NPY_INT);

        // output
        int* c_outlier_indices_final     = PyArray_DATA(outlier_indices_final);
        int* c_outside_ROI_indices_final = PyArray_DATA(outside_ROI_indices_final);
        // input
        int* c_outlier_indices;
        int Noutlier_indices;
        if(outlier_indices == NULL || (PyObject*)outlier_indices == Py_None)
        {
            c_outlier_indices = NULL;
            Noutlier_indices  = 0;
        }
        else
        {
            c_outlier_indices = PyArray_DATA(outlier_indices);
            Noutlier_indices = PyArray_DIMS(outlier_indices)[0];
        }

        double* c_roi;
        if(roi == NULL || (PyObject*)roi == Py_None)
            c_roi = NULL;
        else
            c_roi = PyArray_DATA(roi);

        int* c_imagersizes = PyArray_DATA(imagersizes);

        dogleg_solverContext_t** solver_context_optimizer = NULL;
        if(solver_context != NULL && (PyObject*)solver_context != Py_None)
        {
            solver_context_optimizer                   = &solver_context->ctx;
            solver_context->distortion_model           = distortion_model_type;
            solver_context->problem_details            = problem_details;
            solver_context->Ncameras                   = Ncameras;
            solver_context->Nframes                    = Nframes;
            solver_context->Npoints                    = Npoints;
            solver_context->NobservationsBoard         = NobservationsBoard;
            solver_context->calibration_object_width_n = calibration_object_width_n;

        }

        mrcal_stats_t stats =
        mrcal_optimize( c_x_final,
                        c_invJtJ_intrinsics_full,
                        c_invJtJ_intrinsics_observations_only,
                        c_outlier_indices_final,
                        c_outside_ROI_indices_final,
                        (void**)solver_context_optimizer,
                        c_intrinsics,
                        c_extrinsics,
                        c_frames,
                        c_points,
                        c_calobject_warp,

                        Ncameras, Nframes, Npoints,

                        c_observations_board,
                        NobservationsBoard,
                        c_observations_point,
                        NobservationsPoint,

                        false,
                        Noutlier_indices,
                        c_outlier_indices,
                        c_roi,
                        VERBOSE &&                PyObject_IsTrue(VERBOSE),
                        skip_outlier_rejection && PyObject_IsTrue(skip_outlier_rejection),
                        distortion_model_type,
                        observed_pixel_uncertainty,
                        c_imagersizes,
                        problem_details,

                        calibration_object_spacing,
                        calibration_object_width_n);

        if(stats.rms_reproj_error__pixels < 0.0)
        {
            // Error! I throw an exception
            PyErr_SetString(PyExc_RuntimeError, "mrcal.optimize() failed!");
            goto done;
        }

        pystats = PyDict_New();
        if(pystats == NULL)
        {
            PyErr_SetString(PyExc_RuntimeError, "PyDict_New() failed!");
            goto done;
        }
    #define MRCAL_STATS_ITEM_POPULATE_DICT(type, name, pyconverter)     \
        {                                                               \
            PyObject* obj = pyconverter( (type)stats.name);             \
            if( obj == NULL)                                            \
            {                                                           \
                PyErr_SetString(PyExc_RuntimeError, "Couldn't make PyObject for '" #name "'"); \
                goto done;                                              \
            }                                                           \
                                                                        \
            if( 0 != PyDict_SetItemString(pystats, #name, obj) )        \
            {                                                           \
                PyErr_SetString(PyExc_RuntimeError, "Couldn't add to stats dict '" #name "'"); \
                Py_DECREF(obj);                                         \
                goto done;                                              \
            }                                                           \
        }
        MRCAL_STATS_ITEM(MRCAL_STATS_ITEM_POPULATE_DICT);

        if( 0 != PyDict_SetItemString(pystats, "x",
                                      (PyObject*)x_final) )
        {
            PyErr_SetString(PyExc_RuntimeError, "Couldn't add to stats dict 'x'");
            goto done;
        }
        if( invJtJ_intrinsics_full &&
            0 != PyDict_SetItemString(pystats, "invJtJ_intrinsics_full",
                                      (PyObject*)invJtJ_intrinsics_full) )
        {
            PyErr_SetString(PyExc_RuntimeError, "Couldn't add to stats dict 'invJtJ_intrinsics_full'");
            goto done;
        }
        if( invJtJ_intrinsics_observations_only &&
            0 != PyDict_SetItemString(pystats, "invJtJ_intrinsics_observations_only",
                                      (PyObject*)invJtJ_intrinsics_observations_only) )
        {
            PyErr_SetString(PyExc_RuntimeError, "Couldn't add to stats dict 'invJtJ_intrinsics_observations_only'");
            goto done;
        }
        // The outlier_indices_final numpy array has Nfeatures elements,
        // but I want to return only the first Noutliers elements
        if( NULL == PyArray_Resize(outlier_indices_final,
                                   &(PyArray_Dims){ .ptr = ((npy_intp[]){stats.Noutliers}),
                                                    .len = 1 },
                                   true,
                                   NPY_ANYORDER))
        {
            PyErr_Format(PyExc_RuntimeError, "Couldn't resize outlier_indices_final to %d elements",
                         stats.Noutliers);
            goto done;
        }
        if( 0 != PyDict_SetItemString(pystats, "outlier_indices",
                                      (PyObject*)outlier_indices_final) )
        {
            PyErr_SetString(PyExc_RuntimeError, "Couldn't add to stats dict 'outlier_indices'");
            goto done;
        }
        // The outside_ROI_indices_final numpy array has Nfeatures elements,
        // but I want to return only the first NoutsideROI elements
        if( NULL == PyArray_Resize(outside_ROI_indices_final,
                                   &(PyArray_Dims){ .ptr = ((npy_intp[]){stats.NoutsideROI}),
                                                    .len = 1 },
                                   true,
                                   NPY_ANYORDER))
        {
            PyErr_Format(PyExc_RuntimeError, "Couldn't resize outside_ROI_indices_final to %d elements",
                         stats.NoutsideROI);
            goto done;
        }
        if( 0 != PyDict_SetItemString(pystats, "outside_ROI_indices",
                                      (PyObject*)outside_ROI_indices_final) )
        {
            PyErr_SetString(PyExc_RuntimeError, "Couldn't add to stats dict 'outside_ROI_indices'");
            goto done;
        }

        result = pystats;
        Py_INCREF(result);
    }

 done:
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-to-pointer-cast"
    OPTIMIZE_ARGUMENTS_REQUIRED(FREE_PYARRAY) ;
    OPTIMIZE_ARGUMENTS_OPTIONAL(FREE_PYARRAY) ;
#pragma GCC diagnostic pop

    if(x_final)               Py_DECREF(x_final);
    if(invJtJ_intrinsics_full)
        Py_DECREF(invJtJ_intrinsics_full);
    if(invJtJ_intrinsics_observations_only)
        Py_DECREF(invJtJ_intrinsics_observations_only);
    if(outlier_indices_final) Py_DECREF(outlier_indices_final);
    if(pystats)               Py_DECREF(pystats);

    RESET_SIGINT();
    return result;
}


static const char optimize_docstring[] =
#include "optimize.docstring.h"
    ;
static const char getNdistortionParams_docstring[] =
#include "getNdistortionParams.docstring.h"
    ;
static const char getSupportedDistortionModels_docstring[] =
#include "getSupportedDistortionModels.docstring.h"
    ;
static const char getNextDistortionModel_docstring[] =
#include "getNextDistortionModel.docstring.h"
    ;
static const char project_docstring[] =
#include "project.docstring.h"
    ;
static const char unproject_docstring[] =
#include "unproject.docstring.h"
    ;
static const char unproject_z1_docstring[] =
#include "unproject_z1.docstring.h"
    ;
static PyMethodDef methods[] =
    { PYMETHODDEF_ENTRY(,optimize,                     METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(,getNdistortionParams,         METH_VARARGS),
      PYMETHODDEF_ENTRY(,getSupportedDistortionModels, METH_NOARGS),
      PYMETHODDEF_ENTRY(,getNextDistortionModel,       METH_VARARGS),
      PYMETHODDEF_ENTRY(,project,                      METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(,unproject,                    METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(,unproject_z1,                 METH_VARARGS | METH_KEYWORDS),
      {}
    };


#if PY_MAJOR_VERSION == 2

PyMODINIT_FUNC init_mrcal(void)
{
    if (PyType_Ready(&SolverContextType) < 0)
        return;

    PyObject* module =
        Py_InitModule3("_mrcal", methods,
                       "Calibration and SFM routines");
    Py_INCREF(&SolverContextType);
    PyModule_AddObject(module, "SolverContext", (PyObject *)&SolverContextType);

    import_array();
}

#else

static struct PyModuleDef module_def =
    {
     PyModuleDef_HEAD_INIT,
     "_mrcal",
     "Calibration and SFM routines",
     -1,
     methods
    };

PyMODINIT_FUNC PyInit__mrcal(void)
{
    if (PyType_Ready(&SolverContextType) < 0)
        return NULL;

    PyObject* module =
        PyModule_Create(&module_def);

    Py_INCREF(&SolverContextType);
    PyModule_AddObject(module, "SolverContext", (PyObject *)&SolverContextType);

    import_array();

    return module;
}

#endif

