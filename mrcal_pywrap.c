#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <stdbool.h>
#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>
#include <signal.h>

#include "mrcal.h"


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


// Silly wrapper around a solver context and various solver metadata. I need the
// optimization to be able to keep this, and I need Python to free it as
// necessary when the refcount drops to 0
typedef struct {
    PyObject_HEAD
    void* ctx;
    enum distortion_model_t distortion_model;
    bool do_optimize_intrinsic_core;
    bool do_optimize_intrinsic_distortions;
} SolverContext;
static void SolverContext_free(SolverContext* self)
{
    mrcal_free_context(&self->ctx);
    Py_TYPE(self)->tp_free((PyObject*)self);
}
static PyObject* SolverContext_str(SolverContext* self)
{
    if(self->ctx == NULL)
        return PyString_FromString("Empty context");
    return PyString_FromFormat("Non-empty context made with        %s\n"
                               "do_optimize_intrinsic_core:        %d\n"
                               "do_optimize_intrinsic_distortions: %d\n",
                               mrcal_distortion_model_name(self->distortion_model),
                               self->do_optimize_intrinsic_core,
                               self->do_optimize_intrinsic_distortions);
}
static PyTypeObject SolverContextType =
{
     PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "mrcal.SolverContext",
    .tp_basicsize = sizeof(SolverContext),
    .tp_new       = PyType_GenericNew,
    .tp_dealloc   = (destructor)SolverContext_free,
    .tp_str       = (reprfunc)SolverContext_str,
    .tp_repr      = (reprfunc)SolverContext_str,
    .tp_flags     = Py_TPFLAGS_DEFAULT,
    .tp_doc       = "Opaque solver context used by mrcal",
};


static bool optimize_validate_args( // out
                                    enum distortion_model_t* distortion_model,

                                    // in
                                    PyArrayObject* intrinsics,
                                    PyArrayObject* extrinsics,
                                    PyArrayObject* frames,
                                    PyArrayObject* points,
                                    PyArrayObject* observations_board,
                                    PyArrayObject* indices_frame_camera_board,
                                    PyArrayObject* observations_point,
                                    PyArrayObject* indices_point_camera_points,
                                    PyObject*      skipped_observations_board,
                                    PyObject*      skipped_observations_point,
                                    PyObject*      testing_cull_points_left_of,
                                    PyObject*      testing_cull_points_rad_off_center,
                                    PyObject*      calibration_object_spacing,
                                    PyObject*      calibration_object_width_n,
                                    PyObject*      distortion_model_string,
                                    PyArrayObject* imagersizes,
                                    SolverContext* solver_context)
{
    if( PyArray_NDIM(intrinsics) != 2 )
    {
        PyErr_SetString(PyExc_RuntimeError, "'intrinsics' must have exactly 2 dims");
        return false;
    }
    if( PyArray_NDIM(extrinsics) != 2 )
    {
        PyErr_SetString(PyExc_RuntimeError, "'extrinsics' must have exactly 2 dims");
        return false;
    }
    if( PyArray_NDIM(frames) != 2 )
    {
        PyErr_SetString(PyExc_RuntimeError, "'frames' must have exactly 2 dims");
        return false;
    }
    if( PyArray_NDIM(points) != 2 )
    {
        PyErr_SetString(PyExc_RuntimeError, "'points' must have exactly 2 dims");
        return false;
    }
    if( PyArray_NDIM(observations_board) != 4 )
    {
        PyErr_SetString(PyExc_RuntimeError, "'observations_board' must have exactly 4 dims");
        return false;
    }
    if( PyArray_NDIM(indices_frame_camera_board) != 2 )
    {
        PyErr_SetString(PyExc_RuntimeError, "'indices_frame_camera_board' must have exactly 2 dims");
        return false;
    }
    if( PyArray_NDIM(observations_point) != 2 )
    {
        PyErr_SetString(PyExc_RuntimeError, "'observations_point' must have exactly 2 dims");
        return false;
    }
    if( PyArray_NDIM(indices_point_camera_points) != 2 )
    {
        PyErr_SetString(PyExc_RuntimeError, "'indices_point_camera_points' must have exactly 2 dims");
        return false;
    }
    if( PyArray_NDIM(imagersizes) != 2 )
    {
        PyErr_SetString(PyExc_RuntimeError, "'imagersizes' must have exactly 2 dims");
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
    if( PyArray_DIMS(imagersizes)[1] != 2 )
    {
        PyErr_Format(PyExc_RuntimeError, "imagersizes must have shape (Ncameras,2); instead shape[-1] is %ld",
                     PyArray_DIMS(imagersizes)[1]);
        return false;
    }


    static_assert( sizeof(struct pose_t)/sizeof(double) == 6, "pose_t is assumed to contain 6 elements");

    if( 6 != PyArray_DIMS(extrinsics)[1] )
    {
        PyErr_Format(PyExc_RuntimeError, "extrinsics.shape[1] MUST be 6. Instead got %ld",
                     PyArray_DIMS(extrinsics)[1] );
        return false;
    }
    if( 6 != PyArray_DIMS(frames)[1] )
    {
        PyErr_Format(PyExc_RuntimeError, "frames.shape[1] MUST be 6. Instead got %ld",
                     PyArray_DIMS(frames)[1] );
        return false;
    }
    if( 3 != PyArray_DIMS(points)[1] )
    {
        PyErr_Format(PyExc_RuntimeError, "points.shape[1] MUST be 3. Instead got %ld",
                     PyArray_DIMS(points)[1] );
        return false;
    }

    long int NobservationsBoard = PyArray_DIMS(observations_board)[0];
    if( PyArray_DIMS(indices_frame_camera_board)[0] != NobservationsBoard )
    {
        PyErr_Format(PyExc_RuntimeError, "Inconsistent NobservationsBoard: 'observations_board' says %ld, 'indices_frame_camera_board' says %ld",
                     NobservationsBoard,
                     PyArray_DIMS(indices_frame_camera_board)[0]);
        return false;
    }
    if( PyArray_DIMS(indices_frame_camera_board)[1] != 2 )
    {
        PyErr_Format(PyExc_RuntimeError, "indices_frame_camera_board must have shape (NobservationsBoard,2); instead shape[-1] is %ld",
                     PyArray_DIMS(indices_frame_camera_board)[1]);
        return false;
    }

    // calibration_object_spacing and calibration_object_width_n must be > 0 OR
    // we have to not be using a calibration board
    int c_calibration_object_width_n = 0;
    if( NobservationsBoard > 0 )
    {
        if(testing_cull_points_left_of != NULL    &&
           testing_cull_points_left_of != Py_None &&
           !PyFloat_Check(testing_cull_points_left_of))
        {
            PyErr_Format(PyExc_RuntimeError, "We have board observations, so testing_cull_points_left_of MUST be a valid float");
            return false;
        }
        if(testing_cull_points_rad_off_center != NULL    &&
           testing_cull_points_rad_off_center != Py_None &&
           !PyFloat_Check(testing_cull_points_rad_off_center))
        {
            PyErr_Format(PyExc_RuntimeError, "We have board observations, so testing_cull_points_rad_off_center MUST be a valid float");
            return false;
        }


        if(!PyFloat_Check(calibration_object_spacing))
        {
            PyErr_Format(PyExc_RuntimeError, "We have board observations, so calibration_object_spacing MUST be a valid float > 0");
            return false;
        }

        double c_calibration_object_spacing =
            PyFloat_AS_DOUBLE(calibration_object_spacing);
        if( c_calibration_object_spacing <= 0.0 )
        {
            PyErr_Format(PyExc_RuntimeError, "We have board observations, so calibration_object_spacing MUST be a valid float > 0");
            return false;
        }

        if(!PyInt_Check(calibration_object_width_n))
        {
            PyErr_Format(PyExc_RuntimeError, "We have board observations, so calibration_object_width_n MUST be a valid int > 0");
            return false;
        }

        c_calibration_object_width_n = (int)PyInt_AS_LONG(calibration_object_width_n);
        if( c_calibration_object_width_n <= 0 )
        {
            PyErr_Format(PyExc_RuntimeError, "We have board observations, so calibration_object_width_n MUST be a valid int > 0");
            return false;
        }


        if( c_calibration_object_width_n != PyArray_DIMS(observations_board)[1] ||
            c_calibration_object_width_n != PyArray_DIMS(observations_board)[2] ||
            2  != PyArray_DIMS(observations_board)[3] )
        {
            PyErr_Format(PyExc_RuntimeError, "observations_board.shape[1:] MUST be (%d,%d,2). Instead got (%ld,%ld,%ld)",
                         c_calibration_object_width_n, c_calibration_object_width_n,
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
    if( PyArray_DIMS(indices_point_camera_points)[1] != 2 )
    {
        PyErr_Format(PyExc_RuntimeError, "indices_point_camera_points must have shape (NobservationsPoint,2); instead shape[-1] is %ld",
                     PyArray_DIMS(indices_point_camera_points)[1]);
        return false;
    }
    if( 3  != PyArray_DIMS(observations_point)[1] )
    {
        PyErr_Format(PyExc_RuntimeError, "observations_point.shape[1] MUST be (3). Instead got (%ld)",
                     PyArray_DIMS(observations_point)[1]);
        return false;
    }


    if( PyArray_TYPE(intrinsics)         != NPY_DOUBLE ||
        PyArray_TYPE(extrinsics)         != NPY_DOUBLE ||
        PyArray_TYPE(frames)             != NPY_DOUBLE ||
        PyArray_TYPE(points)             != NPY_DOUBLE ||
        PyArray_TYPE(observations_board) != NPY_DOUBLE ||
        PyArray_TYPE(observations_point) != NPY_DOUBLE )
    {
        PyErr_SetString(PyExc_RuntimeError, "All inputs must contain double-precision floating-point data");
        return false;
    }

    CHECK_CONTIGUOUS(intrinsics);
    CHECK_CONTIGUOUS(extrinsics);
    CHECK_CONTIGUOUS(frames);
    CHECK_CONTIGUOUS(points);
    CHECK_CONTIGUOUS(observations_board);
    CHECK_CONTIGUOUS(observations_point);


    if( PyArray_TYPE(indices_frame_camera_board)   != NPY_INT )
    {
        PyErr_SetString(PyExc_RuntimeError, "indices_frame_camera_board must contain int data");
        return false;
    }
    CHECK_CONTIGUOUS(indices_frame_camera_board);

    if( PyArray_TYPE(indices_point_camera_points)   != NPY_INT )
    {
        PyErr_SetString(PyExc_RuntimeError, "indices_point_camera_points must contain int data");
        return false;
    }
    CHECK_CONTIGUOUS(indices_point_camera_points);

    if( PyArray_TYPE(imagersizes) != NPY_INT )
    {
        PyErr_SetString(PyExc_RuntimeError, "imagersizes must contain int data");
        return false;
    }
    CHECK_CONTIGUOUS(imagersizes);


    const char* distortion_model_cstring =
        PyString_AsString(distortion_model_string);
    if( distortion_model_cstring == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Distortion model was not passed in. Must be a string, one of ("
                        DISTORTION_LIST( QUOTED_LIST_WITH_COMMA )
                        ")");
        return false;
    }

    *distortion_model = mrcal_distortion_model_from_name(distortion_model_cstring);
    if( *distortion_model == DISTORTION_INVALID )
    {
        PyErr_Format(PyExc_RuntimeError, "Invalid distortion model was passed in: '%s'. Must be a string, one of ("
                     DISTORTION_LIST( QUOTED_LIST_WITH_COMMA )
                     ")",
                     distortion_model_cstring);
        return false;
    }


    int NdistortionParams = mrcal_getNdistortionParams(*distortion_model);
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

    if( !(solver_context == NULL ||
          (PyObject*)solver_context == Py_None ||
          Py_TYPE(solver_context) == &SolverContextType) )
    {
        PyErr_Format(PyExc_RuntimeError, "solver_context must be None or of type mrcal.SolverContext");
        return false;
    }

    return true;
}

static PyObject* getNdistortionParams(PyObject* NPY_UNUSED(self),
                                      PyObject* args)
{
    PyObject* result = NULL;
    SET_SIGINT();

    PyObject* distortion_model_string = NULL;
    if(!PyArg_ParseTuple( args, "S", &distortion_model_string ))
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

    enum distortion_model_t distortion_model = mrcal_distortion_model_from_name(distortion_model_cstring);
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

static bool project_validate_args( // out
                                  enum distortion_model_t* distortion_model,

                                  // in
                                  PyArrayObject* points,
                                  PyArrayObject* intrinsics,
                                  PyObject*      distortion_model_string)
{
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
    if( 3 != PyArray_DIMS(points)[ PyArray_NDIM(points)-1 ] )
    {
        PyErr_Format(PyExc_RuntimeError, "points.shape[-1] MUST be 3. Instead got %ld",
                     PyArray_DIMS(points)[PyArray_NDIM(points)-1] );
        return false;
    }

    if( PyArray_TYPE(intrinsics) != NPY_DOUBLE ||
        PyArray_TYPE(points)     != NPY_DOUBLE )
    {
        PyErr_SetString(PyExc_RuntimeError, "All inputs must contain double-precision floating-point data");
        return false;
    }

    CHECK_CONTIGUOUS(intrinsics);
    CHECK_CONTIGUOUS(points);

    const char* distortion_model_cstring =
        PyString_AsString(distortion_model_string);
    if( distortion_model_cstring == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Distortion model was not passed in. Must be a string, one of ("
                        DISTORTION_LIST( QUOTED_LIST_WITH_COMMA )
                        ")");
        return false;
    }

    *distortion_model = mrcal_distortion_model_from_name(distortion_model_cstring);
    if( *distortion_model == DISTORTION_INVALID )
    {
        PyErr_Format(PyExc_RuntimeError, "Invalid distortion model was passed in: '%s'. Must be a string, one of ("
                     DISTORTION_LIST( QUOTED_LIST_WITH_COMMA )
                     ")",
                     distortion_model_cstring);
        return false;
    }

    int NdistortionParams = mrcal_getNdistortionParams(*distortion_model);
    if( N_INTRINSICS_CORE + NdistortionParams != PyArray_DIMS(intrinsics)[0] )
    {
        PyErr_Format(PyExc_RuntimeError, "intrinsics.shape[1] MUST be %d. Instead got %ld",
                     N_INTRINSICS_CORE + NdistortionParams,
                     PyArray_DIMS(intrinsics)[1] );
        return false;
    }

    return true;
}

static PyObject* project(PyObject* NPY_UNUSED(self),
                         PyObject* args,
                         PyObject* kwargs)
{
    PyObject*      result          = NULL;
    SET_SIGINT();

    PyArrayObject* out             = NULL;
    PyArrayObject* dxy_dintrinsics = NULL;
    PyArrayObject* dxy_dp          = NULL;

    PyArrayObject* points                  = NULL;
    PyArrayObject* intrinsics              = NULL;
    PyObject*      distortion_model_string = NULL;
    PyObject*      get_gradients           = Py_False;

    char* keywords[] = {"points",
                        "distortion_model",
                        "intrinsics",

                        // optional kwargs
                        "get_gradients",

                        NULL};

    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     "O&SO&|O", keywords,
                                     PyArray_Converter, &points,
                                     &distortion_model_string,
                                     PyArray_Converter, &intrinsics,
                                     &get_gradients))
        goto done;

    enum distortion_model_t distortion_model;
    if(!project_validate_args(&distortion_model,
                              points, intrinsics,
                              distortion_model_string))
        goto done;

    int Nintrinsics = PyArray_DIMS(intrinsics)[0];

    // poor man's broadcasting of the inputs. I compute the total number of
    // points by multiplying the extra broadcasted dimensions. And I set up the
    // outputs to have the appropriate broadcasted dimensions
    const npy_intp* leading_dims  = PyArray_DIMS(points);
    int             Nleading_dims = PyArray_NDIM(points)-1;
    int Npoints = 1;
    for(int i=0; i<Nleading_dims; i++)
        Npoints *= leading_dims[i];
    {
        npy_intp dims[Nleading_dims+2];
        memcpy(dims, leading_dims, Nleading_dims*sizeof(dims[0]));

        dims[Nleading_dims + 0] = 2;
        out = (PyArrayObject*)PyArray_SimpleNew(Nleading_dims+1,
                                                dims,
                                                NPY_DOUBLE);
        if( get_gradients )
        {
            dims[Nleading_dims + 0] = 2;
            dims[Nleading_dims + 1] = Nintrinsics;
            dxy_dintrinsics = (PyArrayObject*)PyArray_SimpleNew(Nleading_dims+2,
                                                                dims,
                                                                NPY_DOUBLE);

            dims[Nleading_dims + 0] = 2;
            dims[Nleading_dims + 1] = 3;
            dxy_dp          = (PyArrayObject*)PyArray_SimpleNew(Nleading_dims+2,
                                                                dims,
                                                                NPY_DOUBLE);
        }
    }

    mrcal_project((union point2_t*)PyArray_DATA(out),
                  get_gradients ? (double*)PyArray_DATA(dxy_dintrinsics) : NULL,
                  get_gradients ? (union point3_t*)PyArray_DATA(dxy_dp)  : NULL,

                  (const union point3_t*)PyArray_DATA(points),
                  Npoints,
                  distortion_model,
                  // core, distortions concatenated
                  (const double*)PyArray_DATA(intrinsics));

    if( PyObject_IsTrue(get_gradients) )
    {
        result = PyTuple_New(3);
        PyTuple_SET_ITEM(result, 0, (PyObject*)out);
        PyTuple_SET_ITEM(result, 1, (PyObject*)dxy_dintrinsics);
        PyTuple_SET_ITEM(result, 2, (PyObject*)dxy_dp);
    }
    else
        result = (PyObject*)out;

 done:
    RESET_SIGINT();
    return result;
}

static
bool queryIntrinsicOutliernessAt_validate_args(PyArrayObject* v,
                                               int i_camera,
                                               SolverContext* solver_context)
{
    if( PyArray_NDIM(v) < 1 )
    {
        PyErr_SetString(PyExc_RuntimeError, "'v' must have ndims >= 1");
        return false;
    }
    if( 3 != PyArray_DIMS(v)[ PyArray_NDIM(v)-1 ] )
    {
        PyErr_Format(PyExc_RuntimeError, "v.shape[-1] MUST be 3. Instead got %ld",
                     PyArray_DIMS(v)[PyArray_NDIM(v)-1] );
        return false;
    }

    CHECK_CONTIGUOUS(v);

    if(i_camera < 0)
    {
        PyErr_Format(PyExc_RuntimeError, "i_camera>=0 should be true");
        return false;
    }

    if( solver_context == NULL ||
        (PyObject*)solver_context == Py_None ||
        Py_TYPE(solver_context) != &SolverContextType)
    {
        PyErr_Format(PyExc_RuntimeError, "solver_context must be of type mrcal.SolverContext");
        return false;
    }
    if(((SolverContext*)solver_context)->ctx == NULL)
    {
        PyErr_Format(PyExc_RuntimeError, "solver_context must contain a non-empty context");
        return false;
    }

    return true;
}

static PyObject* queryIntrinsicOutliernessAt(PyObject* NPY_UNUSED(self),
                                             PyObject* args,
                                             PyObject* kwargs)
{
    PyObject* result = NULL;
    SET_SIGINT();

    PyArrayObject* v                       = NULL;
    SolverContext* solver_context          = NULL;
    int            i_camera = -1;
    int            Noutliers = 0;

    char* keywords[] = {"v",
                        "i_camera",
                        "solver_context",
                        "Noutliers",
                        NULL};

    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     "O&iO|i", keywords,
                                     PyArray_Converter, &v,
                                     &i_camera,
                                     &solver_context,
                                     &Noutliers))
        goto done;

    if(!queryIntrinsicOutliernessAt_validate_args(v,
                                                  i_camera,
                                                  solver_context))
        goto done;

    int N = PyArray_SIZE(v) / 3;
    PyArrayObject* traces = (PyArrayObject*)PyArray_SimpleNew(PyArray_NDIM(v)-1, PyArray_DIMS(v), NPY_DOUBLE);
    void* ctx = solver_context->ctx;
    if(!mrcal_queryIntrinsicOutliernessAt((double*)PyArray_DATA(traces),
                                          solver_context->distortion_model,
                                          solver_context->do_optimize_intrinsic_core,
                                          solver_context->do_optimize_intrinsic_distortions,
                                          i_camera,
                                          (const union point3_t*)PyArray_DATA(v),
                                          N, Noutliers,
                                          ctx))
    {
        Py_DECREF(traces);
        goto done;
    }

    result = (PyObject*)traces;

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

static PyObject* optimize(PyObject* NPY_UNUSED(self),
                          PyObject* args,
                          PyObject* kwargs)
{
    SET_SIGINT();
    PyObject* result = NULL;

    PyArrayObject* intrinsics                  = NULL;
    PyArrayObject* extrinsics                  = NULL;
    PyArrayObject* frames                      = NULL;
    PyArrayObject* points                      = NULL;
    PyArrayObject* observations_board          = NULL;
    PyArrayObject* indices_frame_camera_board  = NULL;
    PyArrayObject* observations_point          = NULL;
    PyArrayObject* indices_point_camera_points = NULL;
    PyArrayObject* imagersizes                 = NULL;
    PyObject*      pystats                     = NULL;
    PyObject*      VERBOSE                     = NULL;
    PyObject*      skip_outlier_rejection      = NULL;
    PyObject*      skip_regularization         = NULL;
    SolverContext* solver_context              = NULL;

    PyArrayObject* x_final                     = NULL;
    PyArrayObject* intrinsic_covariances       = NULL;
    PyArrayObject* outlier_indices_final       = NULL;

    PyObject* distortion_model_string           = NULL;
    PyObject* do_optimize_intrinsic_core        = Py_True;
    PyObject* do_optimize_intrinsic_distortions = Py_True;
    PyObject* do_optimize_extrinsics            = Py_True;
    PyObject* do_optimize_frames                = Py_True;
    PyObject* skipped_observations_board        = NULL;
    PyObject* skipped_observations_point        = NULL;
    PyObject* testing_cull_points_left_of       = NULL;
    PyObject* testing_cull_points_rad_off_center= NULL;
    PyObject* calibration_object_spacing        = NULL;
    PyObject* calibration_object_width_n        = NULL;

    char* keywords[] = {"intrinsics",
                        "extrinsics",
                        "frames",
                        "points",
                        "observations_board",
                        "indices_frame_camera_board",
                        "observations_point",
                        "indices_point_camera_points",

                        "distortion_model",
                        "imagersizes",

                        // optional kwargs
                        "do_optimize_intrinsic_core",
                        "do_optimize_intrinsic_distortions",
                        "do_optimize_extrinsics",
                        "do_optimize_frames",
                        "skipped_observations_board",
                        "skipped_observations_point",
                        "testing_cull_points_left_of",
                        "testing_cull_points_rad_off_center",
                        "calibration_object_spacing",
                        "calibration_object_width_n",
                        "VERBOSE",
                        "skip_outlier_rejection",
                        "skip_regularization",
                        "solver_context",

                        NULL};

    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     "O&O&O&O&O&O&O&O&SO&|OOOOOOOOOOOOOO",
                                     keywords,
                                     PyArray_Converter_leaveNone, &intrinsics,
                                     PyArray_Converter_leaveNone, &extrinsics,
                                     PyArray_Converter_leaveNone, &frames,
                                     PyArray_Converter_leaveNone, &points,
                                     PyArray_Converter_leaveNone, &observations_board,
                                     PyArray_Converter_leaveNone, &indices_frame_camera_board,
                                     PyArray_Converter_leaveNone, &observations_point,
                                     PyArray_Converter_leaveNone, &indices_point_camera_points,

                                     &distortion_model_string,
                                     PyArray_Converter_leaveNone, &imagersizes,

                                     &do_optimize_intrinsic_core,
                                     &do_optimize_intrinsic_distortions,
                                     &do_optimize_extrinsics,
                                     &do_optimize_frames,
                                     &skipped_observations_board,
                                     &skipped_observations_point,
                                     &testing_cull_points_left_of,
                                     &testing_cull_points_rad_off_center,
                                     &calibration_object_spacing,
                                     &calibration_object_width_n,
                                     &VERBOSE,
                                     &skip_outlier_rejection,
                                     &skip_regularization,
                                     (PyObject*)&solver_context))
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
    SET_SIZE0_IF_NONE(observations_board,         NPY_DOUBLE, 0,179,171,2);
    SET_SIZE0_IF_NONE(indices_frame_camera_board, NPY_INT,    0,2);

    SET_SIZE0_IF_NONE(points,                     NPY_DOUBLE, 0,3);
    SET_SIZE0_IF_NONE(observations_point,         NPY_DOUBLE, 0,3);
    SET_SIZE0_IF_NONE(indices_point_camera_points,NPY_INT,    0,2);
    SET_SIZE0_IF_NONE(imagersizes,                NPY_INT,    0,2);
#undef SET_NULL_IF_NONE




    enum distortion_model_t distortion_model;
    if( !optimize_validate_args(&distortion_model,

                                intrinsics,
                                extrinsics,
                                frames,
                                points,
                                observations_board,
                                indices_frame_camera_board,
                                observations_point,
                                indices_point_camera_points,
                                skipped_observations_board,
                                skipped_observations_point,
                                testing_cull_points_left_of,
                                testing_cull_points_rad_off_center,
                                calibration_object_spacing,
                                calibration_object_width_n,
                                distortion_model_string,
                                imagersizes,
                                solver_context) )
        goto done;

    {
        int Ncameras           = PyArray_DIMS(intrinsics)[0];
        int Nframes            = PyArray_DIMS(frames)[0];
        int Npoints            = PyArray_DIMS(points)[0];
        int NobservationsBoard = PyArray_DIMS(observations_board)[0];
        int NobservationsPoint = PyArray_DIMS(observations_point)[0];


        double c_testing_cull_points_left_of = -1.0;
        double c_testing_cull_points_rad_off_center= -1.0;
        double c_calibration_object_spacing  = 0.0;
        int    c_calibration_object_width_n  = 0;

        if( NobservationsBoard )
        {
            if(testing_cull_points_left_of != NULL   &&
               testing_cull_points_left_of != Py_None)
                c_testing_cull_points_left_of = PyFloat_AS_DOUBLE(testing_cull_points_left_of);
            if(testing_cull_points_rad_off_center != NULL   &&
               testing_cull_points_rad_off_center != Py_None)
                c_testing_cull_points_rad_off_center = PyFloat_AS_DOUBLE(testing_cull_points_rad_off_center);

            if(PyFloat_Check(calibration_object_spacing))
                c_calibration_object_spacing = PyFloat_AS_DOUBLE(calibration_object_spacing);
            if(PyInt_Check(calibration_object_width_n))
                c_calibration_object_width_n = (int)PyInt_AS_LONG(calibration_object_width_n);
        }


        // The checks in optimize_validate_args() make sure these casts are kosher
        double*              c_intrinsics = (double*)         PyArray_DATA(intrinsics);
        struct pose_t*       c_extrinsics = (struct pose_t*)  PyArray_DATA(extrinsics);
        struct pose_t*       c_frames     = (struct pose_t*)  PyArray_DATA(frames);
        union  point3_t*     c_points     = (union  point3_t*)PyArray_DATA(points);



        struct observation_board_t c_observations_board[NobservationsBoard];
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
            c_observations_board[i_observation].px               = &((union point2_t*)PyArray_DATA(observations_board))[c_calibration_object_width_n*c_calibration_object_width_n*i_observation];

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





        struct observation_point_t c_observations_point[NobservationsPoint];
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
            c_observations_point[i_observation].px               = *(union point2_t*)(&((double*)PyArray_DATA(observations_point))[i_observation*3]);
            c_observations_point[i_observation].dist             = ((double*)PyArray_DATA(observations_point))[i_observation*3 + 2];

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





        struct mrcal_variable_select optimization_variable_choice      = {};
        optimization_variable_choice.do_optimize_intrinsic_core        = PyObject_IsTrue(do_optimize_intrinsic_core);
        optimization_variable_choice.do_optimize_intrinsic_distortions = PyObject_IsTrue(do_optimize_intrinsic_distortions);
        optimization_variable_choice.do_optimize_extrinsics            = PyObject_IsTrue(do_optimize_extrinsics);
        optimization_variable_choice.do_optimize_frames                = PyObject_IsTrue(do_optimize_frames);
        optimization_variable_choice.do_skip_regularization            = skip_regularization && PyObject_IsTrue(skip_regularization);

        int Nmeasurements = mrcal_getNmeasurements(Ncameras, NobservationsBoard,
                                                   c_observations_point, NobservationsPoint,
                                                   c_calibration_object_width_n,
                                                   optimization_variable_choice,
                                                   distortion_model);

        x_final = (PyArrayObject*)PyArray_SimpleNew(1, ((npy_intp[]){Nmeasurements}), NPY_DOUBLE);
        double* c_x_final = PyArray_DATA(x_final);

        int Nintrinsics_all = mrcal_getNintrinsicParams(distortion_model);
        double* c_intrinsic_covariances = NULL;
        if(Nintrinsics_all != 0)
        {
            intrinsic_covariances =
                (PyArrayObject*)PyArray_SimpleNew(3,
                                                  ((npy_intp[]){Ncameras,Nintrinsics_all,Nintrinsics_all}), NPY_DOUBLE);
            c_intrinsic_covariances = PyArray_DATA(intrinsic_covariances);
        }

        const int Npoints_fromBoards =
            NobservationsBoard *
            c_calibration_object_width_n*c_calibration_object_width_n;
        outlier_indices_final = (PyArrayObject*)PyArray_SimpleNew(1, ((npy_intp[]){Npoints_fromBoards}), NPY_INT);
        int* c_outlier_indices_final = PyArray_DATA(outlier_indices_final);

        int* c_imagersizes = PyArray_DATA(imagersizes);

        void** solver_context_optimizer = NULL;
        if(solver_context != NULL && (PyObject*)solver_context != Py_None)
        {
            solver_context_optimizer = &solver_context->ctx;
            solver_context->distortion_model = distortion_model;
            solver_context->do_optimize_intrinsic_core =
                optimization_variable_choice.do_optimize_intrinsic_core;
            solver_context->do_optimize_intrinsic_distortions =
                optimization_variable_choice.do_optimize_intrinsic_distortions;
        }


        struct mrcal_stats_t stats =
        mrcal_optimize( c_x_final,
                        c_intrinsic_covariances,
                        c_outlier_indices_final,
                        solver_context_optimizer,
                        c_intrinsics,
                        c_extrinsics,
                        c_frames,
                        c_points,
                        Ncameras, Nframes, Npoints,

                        c_observations_board,
                        NobservationsBoard,
                        c_observations_point,
                        NobservationsPoint,

                        false,
                        VERBOSE &&                PyObject_IsTrue(VERBOSE),
                        skip_outlier_rejection && PyObject_IsTrue(skip_outlier_rejection),
                        distortion_model,
                        c_imagersizes,
                        optimization_variable_choice,

                        c_testing_cull_points_left_of,
                        c_testing_cull_points_rad_off_center,
                        c_calibration_object_spacing,
                        c_calibration_object_width_n);
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
        if( intrinsic_covariances &&
            0 != PyDict_SetItemString(pystats, "intrinsic_covariances",
                                      (PyObject*)intrinsic_covariances) )
        {
            PyErr_SetString(PyExc_RuntimeError, "Couldn't add to stats dict 'intrinsic_covariances'");
            goto done;
        }
        // The outlier_indices_final numpy array has Nmeasurements elements,
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

        result = pystats;
        Py_INCREF(result);
    }

 done:
    if(intrinsics)                  Py_DECREF(intrinsics);
    if(extrinsics)                  Py_DECREF(extrinsics);
    if(frames)                      Py_DECREF(frames);
    if(points)                      Py_DECREF(points);
    if(observations_board)          Py_DECREF(observations_board);
    if(indices_frame_camera_board)  Py_DECREF(indices_frame_camera_board);
    if(observations_point)          Py_DECREF(observations_point);
    if(indices_point_camera_points) Py_DECREF(indices_point_camera_points);
    if(x_final)                     Py_DECREF(x_final);
    if(intrinsic_covariances)       Py_DECREF(intrinsic_covariances);
    if(outlier_indices_final)       Py_DECREF(outlier_indices_final);
    if(pystats)                     Py_DECREF(pystats);
    if(imagersizes)                 Py_DECREF(imagersizes);

    RESET_SIGINT();
    return result;
}

PyMODINIT_FUNC init_mrcal(void)
{
    static const char optimize_docstring[] =
#include "optimize.docstring.h"
        ;
    static const char getNdistortionParams_docstring[] =
#include "getNdistortionParams.docstring.h"
        ;
    static const char getSupportedDistortionModels_docstring[] =
#include "getSupportedDistortionModels.docstring.h"
        ;
    static const char project_docstring[] =
#include "project.docstring.h"
        ;
    static const char queryIntrinsicOutliernessAt_docstring[] =
#include "queryIntrinsicOutliernessAt.docstring.h"
        ;

#define PYMETHODDEF_ENTRY(x, args) {#x, (PyCFunction)x, args, x ## _docstring}
    static PyMethodDef methods[] =
        { PYMETHODDEF_ENTRY(optimize,                     METH_VARARGS | METH_KEYWORDS),
          PYMETHODDEF_ENTRY(getNdistortionParams,         METH_VARARGS),
          PYMETHODDEF_ENTRY(getSupportedDistortionModels, METH_NOARGS),
          PYMETHODDEF_ENTRY(project,                      METH_VARARGS | METH_KEYWORDS),
          PYMETHODDEF_ENTRY(queryIntrinsicOutliernessAt,  METH_VARARGS | METH_KEYWORDS),
          {}
        };

    if (PyType_Ready(&SolverContextType) < 0)
        return;

    PyImport_AddModule("_mrcal");
    PyObject* module = Py_InitModule3("_mrcal", methods,
                                      "Calibration and SFM routines");

    Py_INCREF(&SolverContextType);
    PyModule_AddObject(module, "SolverContext", (PyObject *)&SolverContextType);

    import_array();
}
