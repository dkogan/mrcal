#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <stdbool.h>
#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>
#include <signal.h>
#include <dogleg.h>

#if (CHOLMOD_VERSION > (CHOLMOD_VER_CODE(2,2)))
#include <suitesparse/cholmod_function.h>
#endif

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

#define IS_NULL(x) ((x) == NULL || (PyObject*)(x) == Py_None)

#define BARF(fmt, ...) PyErr_Format(PyExc_RuntimeError, "%s:%d %s(): "fmt, __FILE__, __LINE__, __func__, ## __VA_ARGS__)

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
        BARF("sigaction() failed");      \
        goto done;                                                      \
    }                                                                   \
} while(0)
#define RESET_SIGINT() do {                                             \
    if( 0 != sigaction(SIGINT,                                          \
                       &sigaction_old, NULL ))                          \
        BARF("sigaction-restore failed"); \
} while(0)

#define PERCENT_S_COMMA(s,n) "'%s',"
#define COMMA_LENSMODEL_NAME(s,n) , mrcal_lensmodel_name_unconfigured( &(mrcal_lensmodel_t){.type = MRCAL_##s} )
#define VALID_LENSMODELS_FORMAT  "(" MRCAL_LENSMODEL_LIST(PERCENT_S_COMMA) ")"
#define VALID_LENSMODELS_ARGLIST MRCAL_LENSMODEL_LIST(COMMA_LENSMODEL_NAME)

#define CHECK_CONTIGUOUS(x) do {                                        \
    if( !PyArray_IS_C_CONTIGUOUS(x) )                                   \
    {                                                                   \
        BARF("All inputs must be c-style contiguous arrays (" #x ")"); \
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
    if(!IS_NULL(name_pyarrayobj)) {                                     \
        int dims[] = dims_ref;                                          \
        int ndims = (int)sizeof(dims)/(int)sizeof(dims[0]);             \
                                                                        \
        if( ndims > 0 )                                                 \
        {                                                               \
            if( PyArray_NDIM((PyArrayObject*)name_pyarrayobj) != ndims )          \
            {                                                           \
                BARF("'" #name "' must have exactly %d dims; got %d", ndims, PyArray_NDIM((PyArrayObject*)name_pyarrayobj)); \
                return false;                                           \
            }                                                           \
            for(int i=0; i<ndims; i++)                                  \
                if(dims[i] >= 0 && dims[i] != PyArray_DIMS((PyArrayObject*)name_pyarrayobj)[i]) \
                {                                                       \
                    BARF("'" #name "' must have dimensions '" #dims_ref "' where <0 means 'any'. Dims %d got %ld instead", i, PyArray_DIMS((PyArrayObject*)name_pyarrayobj)[i]); \
                    return false;                                       \
                }                                                       \
        }                                                               \
        if( (int)npy_type >= 0 )                                        \
        {                                                               \
            if( PyArray_TYPE((PyArrayObject*)name_pyarrayobj) != npy_type )       \
            {                                                           \
                BARF("'" #name "' must have type: " #npy_type); \
                return false;                                           \
            }                                                           \
            if( !PyArray_IS_C_CONTIGUOUS((PyArrayObject*)name_pyarrayobj) )       \
            {                                                           \
                BARF("'" #name "' must be c-style contiguous"); \
                return false;                                           \
            }                                                           \
        }                                                               \
    }
#define PYMETHODDEF_ENTRY(function_prefix, name, args) {#name,          \
                                                        (PyCFunction)function_prefix ## name, \
                                                        args,           \
                                                        function_prefix ## name ## _docstring}


// adds a reference to P,I,X, unless an error is reported
static PyObject* csr_from_cholmod_sparse( PyObject* P,
                                          PyObject* I,
                                          PyObject* X )
{
    // I do the Python equivalent of this;
    // scipy.sparse.csr_matrix((data, indices, indptr))


    PyObject* result = NULL;

    PyObject* module = NULL;
    PyObject* method = NULL;
    PyObject* args   = NULL;
    if(NULL == (module = PyImport_ImportModule("scipy.sparse")))
    {
        BARF("Couldn't import scipy.sparse. I need that to represent J");
        goto done;
    }
    if(NULL == (method = PyObject_GetAttrString(module, "csr_matrix")))
    {
        BARF("Couldn't find 'csr_matrix' in scipy.sparse");
        goto done;
    }

    // Here I'm assuming specific types in my cholmod arrays. I tried to
    // static_assert it, but internally cholmod uses void*, so I can't do that
    PyObject* MatrixDef = PyTuple_Pack(3, X, I, P);
    args                = PyTuple_Pack(1, MatrixDef);
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

// A container for a CHOLMOD factorization
typedef struct {
    PyObject_HEAD

    // if(inited_common), the "common" has been initialized
    // if(factorization), the factorization has been initialized
    //
    // So to use the object we need inited_common && factorization
    bool            inited_common;
    cholmod_common  common;
    cholmod_factor* factorization;

    // optimizer_callback should return it
    // and I should have two solve methods:
} CHOLMOD_factorization;


// stolen from libdogleg
static int cholmod_error_callback(const char* s, ...)
{
  va_list ap;
  va_start(ap, s);
  int ret = vfprintf(stderr, s, ap);
  va_end(ap);
  fprintf(stderr, "\n");
  return ret;
}

// for my internal C usage
static void _CHOLMOD_factorization_release_internal(CHOLMOD_factorization* self)
{
    if( self->factorization )
    {
        cholmod_free_factor(&self->factorization, &self->common);
        self->factorization = NULL;
    }
    if( self->inited_common )
        cholmod_finish(&self->common);
    self->inited_common = false;
}

// for my internal C usage
static bool
_CHOLMOD_factorization_init_from_cholmod_sparse(CHOLMOD_factorization* self, cholmod_sparse* Jt)
{
    if( !self->inited_common )
    {
        if( !cholmod_start(&self->common) )
        {
            BARF("Error trying to cholmod_start");
            return false;
        }
        self->inited_common = true;

        // stolen from libdogleg

        // I want to use LGPL parts of CHOLMOD only, so I turn off the supernodal routines. This gave me a
        // 25% performance hit in the solver for a particular set of optical calibration data.
        self->common.supernodal = 0;

        // I want all output to go to STDERR, not STDOUT
#if (CHOLMOD_VERSION <= (CHOLMOD_VER_CODE(2,2)))
        self->common.print_function = cholmod_error_callback;
#else
        CHOLMOD_FUNCTION_DEFAULTS ;
        CHOLMOD_FUNCTION_PRINTF(&self->common) = cholmod_error_callback;
#endif
    }

    self->factorization = cholmod_analyze(Jt, &self->common);

    if(self->factorization == NULL)
    {
        BARF("cholmod_analyze() failed");
        return false;
    }
    if( !cholmod_factorize(Jt, self->factorization, &self->common) )
    {
        BARF("cholmod_factorize() failed");
        return false;
    }
    if(self->factorization->minor != self->factorization->n)
    {
        BARF("Got singular JtJ!");
        return false;
    }
    return true;
}


static int
CHOLMOD_factorization_init(CHOLMOD_factorization* self, PyObject* args, PyObject* kwargs)
{
    // Any existing factorization goes away. If this function fails, we lose the
    // existing factorization, which is fine. I'm placing this on top so that
    // __init__() will get rid of the old state
    _CHOLMOD_factorization_release_internal(self);


    // error by default
    int result = -1;

    char* keywords[] = {"J", NULL};
    PyObject* Py_J                  = NULL;
    PyObject* module                = NULL;
    PyObject* csr_matrix_type       = NULL;

    PyObject* Py_shape              = NULL;
    PyObject* Py_nnz                = NULL;
    PyObject* Py_data               = NULL;
    PyObject* Py_indices            = NULL;
    PyObject* Py_indptr             = NULL;
    PyObject* Py_has_sorted_indices = NULL;

    PyObject* Py_h = NULL;
    PyObject* Py_w = NULL;

    if( !PyArg_ParseTupleAndKeywords(args, kwargs,
                                     "|O:CHOLMOD_factorization.__init__",
                                     keywords, &Py_J))
        goto done;

    if( Py_J == NULL )
    {
        // Success. Nothing to do
        result = 0;
        goto done;
    }

    if(NULL == (module = PyImport_ImportModule("scipy.sparse")))
    {
        BARF("Couldn't import scipy.sparse. I need that to represent J");
        goto done;
    }
    if(NULL == (csr_matrix_type = PyObject_GetAttrString(module, "csr_matrix")))
    {
        BARF("Couldn't find 'csr_matrix' in scipy.sparse");
        goto done;
    }
    if(!PyObject_IsInstance(Py_J, csr_matrix_type))
    {
        BARF("Argument J is must have type scipy.sparse.csr_matrix");
        goto done;
    }

#define GETATTR(x)                                              \
    if(NULL == (Py_ ## x = PyObject_GetAttrString(Py_J, #x)))   \
    {                                                           \
        BARF("Couldn't get J." # x);                            \
        goto done;                                              \
    }

    GETATTR(shape);
    GETATTR(nnz);
    GETATTR(data);
    GETATTR(indices);
    GETATTR(indptr);
    GETATTR(has_sorted_indices);

    if(!PySequence_Check(Py_shape))
    {
        BARF("J.shape should be an iterable");
        goto done;
    }
    int lenshape = PySequence_Length(Py_shape);
    if( lenshape != 2 )
    {
        if(lenshape < 0)
            BARF("Failed to get len(J.shape)");
        else
            BARF("len(J.shape) should be exactly 2, but instead got %d", lenshape);
        goto done;
    }

    Py_h = PySequence_GetItem(Py_shape, 0);
    if(Py_h == NULL)
    {
        BARF("Error getting J.shape[0]");
        goto done;
    }
    Py_w = PySequence_GetItem(Py_shape, 1);
    if(Py_w == NULL)
    {
        BARF("Error getting J.shape[1]");
        goto done;
    }

    long nnz;
    if(PyLong_Check(Py_nnz)) nnz = PyLong_AsLong(Py_nnz);
    else
    {
        BARF("Error interpreting nnz as an integer");
        goto done;
    }

    long h;
    if(PyLong_Check(Py_h)) h = PyLong_AsLong(Py_h);
    else
    {
        BARF("Error interpreting J.shape[0] as an integer");
        goto done;
    }
    long w;
    if(PyLong_Check(Py_w)) w = PyLong_AsLong(Py_w);
    else
    {
        BARF("Error interpreting J.shape[1] as an integer");
        goto done;
    }

#define CHECK_NUMPY_ARRAY(x, dtype)                                     \
    if( !PyArray_Check((PyArrayObject*)Py_ ## x) )                      \
    {                                                                   \
        BARF("J."#x " must be a numpy array");                          \
        goto done;                                                      \
    }                                                                   \
    if( 1 != PyArray_NDIM((PyArrayObject*)Py_ ## x) )                   \
    {                                                                   \
        BARF("J."#x " must be a 1-dimensional numpy array. Instead got %d dimensions", \
             PyArray_NDIM((PyArrayObject*)Py_ ## x));                   \
        goto done;                                                      \
    }                                                                   \
    if( PyArray_TYPE((PyArrayObject*)Py_ ## x) != dtype )               \
    {                                                                   \
        BARF("J."#x " must have dtype: " #dtype);                       \
        goto done;                                                      \
    }                                                                   \
    if( !PyArray_IS_C_CONTIGUOUS((PyArrayObject*)Py_ ## x) )            \
    {                                                                   \
        BARF("J."#x " must live in contiguous memory");                 \
        goto done;                                                      \
    }

    CHECK_NUMPY_ARRAY(data,    NPY_FLOAT64);
    CHECK_NUMPY_ARRAY(indices, NPY_INT32);
    CHECK_NUMPY_ARRAY(indptr,  NPY_INT32);

    // OK, the input looks good. I guess I can tell CHOLMOD about it

    // My convention is to store row-major matrices, but CHOLMOD stores
    // col-major matrices. So I keep the same data, but tell CHOLMOD that I'm
    // storing Jt and not J
    cholmod_sparse Jt = {
        .nrow   = w,
        .ncol   = h,
        .nzmax  = nnz,
        .p      = PyArray_DATA((PyArrayObject*)Py_indptr),
        .i      = PyArray_DATA((PyArrayObject*)Py_indices),
        .x      = PyArray_DATA((PyArrayObject*)Py_data),
        .stype  = 0,            // not symmetric
        .itype  = CHOLMOD_INT,
        .xtype  = CHOLMOD_REAL,
        .dtype  = CHOLMOD_DOUBLE,
        .sorted = PyObject_IsTrue(Py_has_sorted_indices),
        .packed = 1
    };

    if(!_CHOLMOD_factorization_init_from_cholmod_sparse(self, &Jt))
        goto done;

    result = 0;

 done:
    if(result != 0)
        _CHOLMOD_factorization_release_internal(self);

    Py_XDECREF(module);
    Py_XDECREF(csr_matrix_type);
    Py_XDECREF(Py_shape);
    Py_XDECREF(Py_nnz);
    Py_XDECREF(Py_data);
    Py_XDECREF(Py_indices);
    Py_XDECREF(Py_indptr);
    Py_XDECREF(Py_has_sorted_indices);
    Py_XDECREF(Py_h);
    Py_XDECREF(Py_w);

    return result;

#undef GETATTR
#undef CHECK_NUMPY_ARRAY
}

static void CHOLMOD_factorization_dealloc(CHOLMOD_factorization* self)
{
    _CHOLMOD_factorization_release_internal(self);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* CHOLMOD_factorization_str(CHOLMOD_factorization* self)
{
    if(!(self->inited_common && self->factorization))
        return PyString_FromString("No factorization given");

    return PyString_FromFormat("Initialized with a valid factorization. N=%d",
                               self->factorization->n);
}

static PyObject*
CHOLMOD_factorization_solve_xt_JtJ_bt(CHOLMOD_factorization* self, PyObject* args, PyObject* kwargs)
{
    // error by default
    PyObject* result = NULL;
    PyObject* Py_out = NULL;

    char* keywords[] = {"bt", NULL};
    PyObject* Py_bt   = NULL;

    if(!(self->inited_common && self->factorization))
    {
        BARF("No factorization has been computed");
        goto done;
    }

    if( !PyArg_ParseTupleAndKeywords(args, kwargs,
                                     "O:CHOLMOD_factorization.solve_xt_JtJ_bt",
                                     keywords, &Py_bt))
        goto done;

    if( Py_bt == NULL || !PyArray_Check((PyArrayObject*)Py_bt) )
    {
        BARF("bt must be a numpy array");
        goto done;
    }

    int ndim = PyArray_NDIM((PyArrayObject*)Py_bt);
    if( ndim < 1 )
    {
        BARF("bt must be at least a 1-dimensional numpy array. Instead got %d dimensions",
             ndim);
        goto done;
    }

    int Nstate = (int)PyArray_DIMS((PyArrayObject*)Py_bt)[ndim-1];
    int Nrhs   = (int)PyArray_SIZE((PyArrayObject*)Py_bt) / Nstate;

    if( self->factorization->n != (unsigned)Nstate )
    {
        BARF("bt must be a 2-dimensional numpy array with %d cols (that's what the factorization has). Instead got %d cols",
             self->factorization->n,
             Nstate);
        goto done;
    }
    if( PyArray_TYPE((PyArrayObject*)Py_bt) != NPY_FLOAT64 )
    {
        BARF("bt must have dtype=float");
        goto done;
    }
    if( !PyArray_IS_C_CONTIGUOUS((PyArrayObject*)Py_bt) )
    {
        BARF("bt must live in contiguous memory");
        goto done;
    }

    // Alright. b looks good-enough to use
    if( 0 == Nrhs )
    {
        // Degenerate input (0 columns). Just return it, and I'm done
        result = Py_bt;
        Py_INCREF(result);
        goto done;
    }

    cholmod_dense b = {
        .nrow  = Nstate,
        .ncol  = Nrhs,
        .nzmax = Nrhs * Nstate,
        .d     = Nstate,
        .x     = PyArray_DATA((PyArrayObject*)Py_bt),
        .xtype = CHOLMOD_REAL,
        .dtype = CHOLMOD_DOUBLE };

    Py_out = PyArray_SimpleNew(ndim,
                               PyArray_DIMS((PyArrayObject*)Py_bt),
                               NPY_DOUBLE);
    if(Py_out == NULL)
    {
        BARF("Couldn't allocate Py_out");
        goto done;
    }

    cholmod_dense out = {
        .nrow  = Nstate,
        .ncol  = Nrhs,
        .nzmax = Nrhs * Nstate,
        .d     = Nstate,
        .x     = PyArray_DATA((PyArrayObject*)Py_out),
        .xtype = CHOLMOD_REAL,
        .dtype = CHOLMOD_DOUBLE };

    cholmod_dense* M = &out;
    cholmod_dense* Y = NULL;
    cholmod_dense* E = NULL;

    if(!cholmod_solve2( CHOLMOD_A, self->factorization,
                        &b, NULL,
                        &M, NULL, &Y, &E,
                        &self->common))
    {
        BARF("cholmod_solve2() failed");
        goto done;
    }
    if( M != &out )
    {
        BARF("cholmod_solve2() reallocated out! We leaked memory");
        goto done;
    }

    cholmod_free_dense (&E, &self->common);
    cholmod_free_dense (&Y, &self->common);

    Py_INCREF(Py_out);
    result = Py_out;

 done:
    Py_XDECREF(Py_out);

    return result;
}

static const char CHOLMOD_factorization_docstring[] =
#include "CHOLMOD_factorization.docstring.h"
    ;
static const char CHOLMOD_factorization_solve_xt_JtJ_bt_docstring[] =
#include "CHOLMOD_factorization_solve_xt_JtJ_bt.docstring.h"
    ;

static PyMethodDef CHOLMOD_factorization_methods[] =
    {
        PYMETHODDEF_ENTRY(CHOLMOD_factorization_, solve_xt_JtJ_bt, METH_VARARGS | METH_KEYWORDS),
        {}
    };


#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-braces"
// PyObject_HEAD_INIT throws
//   warning: missing braces around initializer []
// This isn't mine to fix, so I'm ignoring it
static PyTypeObject CHOLMOD_factorization_type =
{
     PyObject_HEAD_INIT(NULL)
    .tp_name      = "mrcal.CHOLMOD_factorization",
    .tp_basicsize = sizeof(CHOLMOD_factorization),
    .tp_new       = PyType_GenericNew,
    .tp_init      = (initproc)CHOLMOD_factorization_init,
    .tp_dealloc   = (destructor)CHOLMOD_factorization_dealloc,
    .tp_methods   = CHOLMOD_factorization_methods,
    .tp_str       = (reprfunc)CHOLMOD_factorization_str,
    .tp_flags     = Py_TPFLAGS_DEFAULT,
    .tp_doc       = CHOLMOD_factorization_docstring,
};
#pragma GCC diagnostic pop


// For the C code. Create a new Python CHOLMOD_factorization object from a C
// cholmod_sparse structure
static PyObject*
CHOLMOD_factorization_from_cholmod_sparse(cholmod_sparse* Jt)
{
    PyObject* self = PyObject_CallObject((PyObject*)&CHOLMOD_factorization_type, NULL);
    if(NULL == self)
        return NULL;

    if(!_CHOLMOD_factorization_init_from_cholmod_sparse((CHOLMOD_factorization*)self, Jt))
    {
        Py_DECREF(self);
        return NULL;
    }

    return self;
}



static bool parse_lensmodel_from_arg(// output
                                     mrcal_lensmodel_t* lensmodel,
                                     // input
                                     PyObject* lensmodel_string)
{
    const char* lensmodel_cstring = PyString_AsString(lensmodel_string);
    if( lensmodel_cstring == NULL)
    {
        BARF("The lens model must be given as a string");
        return false;
    }

    mrcal_lensmodel_from_name(lensmodel, lensmodel_cstring);
    if( !mrcal_lensmodel_type_is_valid(lensmodel->type) )
    {
        if(lensmodel->type == MRCAL_LENSMODEL_INVALID_BADCONFIG)
        {
            BARF("Couldn't parse the configuration of the given lens model '%s'",
                         lensmodel_cstring);
            return false;
        }
        BARF("Invalid lens model was passed in: '%s'. Must be one of " VALID_LENSMODELS_FORMAT,
                     lensmodel_cstring
                     VALID_LENSMODELS_ARGLIST);
        return false;
    }
    return true;
}

static PyObject* lensmodel_metadata_and_config(PyObject* NPY_UNUSED(self),
                                               PyObject* args)
{
    PyObject* result = NULL;
    SET_SIGINT();

    PyObject* lensmodel_string = NULL;
    if(!PyArg_ParseTuple( args, STRING_OBJECT, &lensmodel_string ))
        goto done;
    mrcal_lensmodel_t lensmodel;
    if(!parse_lensmodel_from_arg(&lensmodel, lensmodel_string))
        goto done;

    mrcal_lensmodel_metadata_t meta = mrcal_lensmodel_metadata(&lensmodel);

#define MRCAL_ITEM_BUILDVALUE_DEF(  name, type, pybuildvaluecode, PRIcode,SCNcode, bitfield, cookie) " s "pybuildvaluecode
#define MRCAL_ITEM_BUILDVALUE_VALUE(name, type, pybuildvaluecode, PRIcode,SCNcode, bitfield, cookie) , #name, cookie name

    if(lensmodel.type == MRCAL_LENSMODEL_CAHVORE )
        result = Py_BuildValue("{"
                               MRCAL_LENSMODEL_META_LIST(MRCAL_ITEM_BUILDVALUE_DEF, )
                               MRCAL_LENSMODEL_CAHVORE_CONFIG_LIST(MRCAL_ITEM_BUILDVALUE_DEF, )
                               "}"
                               MRCAL_LENSMODEL_META_LIST(MRCAL_ITEM_BUILDVALUE_VALUE, meta.)
                               MRCAL_LENSMODEL_CAHVORE_CONFIG_LIST(MRCAL_ITEM_BUILDVALUE_VALUE, lensmodel.LENSMODEL_CAHVORE__config.));
    else if(lensmodel.type == MRCAL_LENSMODEL_SPLINED_STEREOGRAPHIC )
        result = Py_BuildValue("{"
                               MRCAL_LENSMODEL_META_LIST(MRCAL_ITEM_BUILDVALUE_DEF, )
                               MRCAL_LENSMODEL_SPLINED_STEREOGRAPHIC_CONFIG_LIST(MRCAL_ITEM_BUILDVALUE_DEF, )
                               "}"
                               MRCAL_LENSMODEL_META_LIST(MRCAL_ITEM_BUILDVALUE_VALUE, meta.)
                               MRCAL_LENSMODEL_SPLINED_STEREOGRAPHIC_CONFIG_LIST(MRCAL_ITEM_BUILDVALUE_VALUE, lensmodel.LENSMODEL_SPLINED_STEREOGRAPHIC__config.));
    else
        result = Py_BuildValue("{"
                               MRCAL_LENSMODEL_META_LIST(MRCAL_ITEM_BUILDVALUE_DEF, )
                               "}"
                               MRCAL_LENSMODEL_META_LIST(MRCAL_ITEM_BUILDVALUE_VALUE, meta.));

    Py_INCREF(result);

 done:
    RESET_SIGINT();
    return result;
}

static PyObject* knots_for_splined_models(PyObject* NPY_UNUSED(self),
                                          PyObject* args)
{
    PyObject*      result = NULL;
    PyArrayObject* py_ux  = NULL;
    PyArrayObject* py_uy  = NULL;
    SET_SIGINT();

    PyObject* lensmodel_string = NULL;
    if(!PyArg_ParseTuple( args, STRING_OBJECT, &lensmodel_string ))
        goto done;
    mrcal_lensmodel_t lensmodel;
    if(!parse_lensmodel_from_arg(&lensmodel, lensmodel_string))
        goto done;

    if(lensmodel.type != MRCAL_LENSMODEL_SPLINED_STEREOGRAPHIC)
    {
        BARF( "This function works only with the MRCAL_LENSMODEL_SPLINED_STEREOGRAPHIC model. %S passed in",
              lensmodel_string);
        goto done;
    }

    {
        double ux[lensmodel.LENSMODEL_SPLINED_STEREOGRAPHIC__config.Nx];
        double uy[lensmodel.LENSMODEL_SPLINED_STEREOGRAPHIC__config.Ny];
        if(!mrcal_knots_for_splined_models(ux,uy, &lensmodel))
        {
            BARF( "mrcal_knots_for_splined_models() failed");
            goto done;
        }


        npy_intp dims[1];

        dims[0] = lensmodel.LENSMODEL_SPLINED_STEREOGRAPHIC__config.Nx;
        py_ux = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        if(py_ux == NULL)
        {
            BARF("Couldn't allocate ux");
            goto done;
        }

        dims[0] = lensmodel.LENSMODEL_SPLINED_STEREOGRAPHIC__config.Ny;
        py_uy = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        if(py_uy == NULL)
        {
            BARF("Couldn't allocate uy");
            goto done;
        }

        memcpy(PyArray_DATA(py_ux), ux, lensmodel.LENSMODEL_SPLINED_STEREOGRAPHIC__config.Nx*sizeof(double));
        memcpy(PyArray_DATA(py_uy), uy, lensmodel.LENSMODEL_SPLINED_STEREOGRAPHIC__config.Ny*sizeof(double));
    }

    result = Py_BuildValue("OO", py_ux, py_uy);

 done:
    Py_XDECREF(py_ux);
    Py_XDECREF(py_uy);
    RESET_SIGINT();
    return result;
}

static PyObject* lensmodel_num_params(PyObject* NPY_UNUSED(self),
                                 PyObject* args)
{
    PyObject* result = NULL;
    SET_SIGINT();

    PyObject* lensmodel_string = NULL;
    if(!PyArg_ParseTuple( args, STRING_OBJECT, &lensmodel_string ))
        goto done;
    mrcal_lensmodel_t lensmodel;
    if(!parse_lensmodel_from_arg(&lensmodel, lensmodel_string))
        goto done;

    int Nparams = mrcal_lensmodel_num_params(&lensmodel);

    result = Py_BuildValue("i", Nparams);

 done:
    RESET_SIGINT();
    return result;
}

static PyObject* supported_lensmodels(PyObject* NPY_UNUSED(self),
                                      PyObject* NPY_UNUSED(args))
{
    PyObject* result = NULL;
    SET_SIGINT();
    const char* const* names = mrcal_supported_lensmodel_names();

    // I now have a NULL-terminated list of NULL-terminated strings. Get N
    int N=0;
    while(names[N] != NULL)
        N++;

    result = PyTuple_New(N);
    if(result == NULL)
    {
        BARF("Failed PyTuple_New(%d)", N);
        goto done;
    }

    for(int i=0; i<N; i++)
    {
        PyObject* name = Py_BuildValue("s", names[i]);
        if( name == NULL )
        {
            BARF("Failed Py_BuildValue...");
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

#define ARGDEF_observations_point_triangulated(_)                       \
 _(observations_point_triangulated,    PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, observations_point_triangulated,          NPY_DOUBLE, {-1 COMMA  3       } )
#define ARGDEF_indices_point_triangulated_camintrinsics_camextrinsics(_) \
 _(indices_point_triangulated_camintrinsics_camextrinsics,PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, indices_point_triangulated_camintrinsics_camextrinsics, NPY_INT32,    {-1 COMMA  3       } )

#define OPTIMIZE_ARGUMENTS_REQUIRED(_)                                  \
    _(intrinsics,                         PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, intrinsics,                  NPY_DOUBLE, {-1 COMMA -1       } ) \
    _(extrinsics_rt_fromref,              PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, extrinsics_rt_fromref,       NPY_DOUBLE, {-1 COMMA  6       } ) \
    _(frames_rt_toref,                    PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, frames_rt_toref,             NPY_DOUBLE, {-1 COMMA  6       } ) \
    _(points,                             PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, points,                      NPY_DOUBLE, {-1 COMMA  3       } ) \
    _(observations_board,                 PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, observations_board,          NPY_DOUBLE, {-1 COMMA -1 COMMA -1 COMMA 3 } ) \
    _(indices_frame_camintrinsics_camextrinsics,PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, indices_frame_camintrinsics_camextrinsics,  NPY_INT32,    {-1 COMMA  3       } ) \
    _(observations_point,                 PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, observations_point,          NPY_DOUBLE, {-1 COMMA  3       } ) \
    _(indices_point_camintrinsics_camextrinsics,PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, indices_point_camintrinsics_camextrinsics, NPY_INT32,    {-1 COMMA  3       } ) \
    ARGDEF_observations_point_triangulated(_)                        \
    ARGDEF_indices_point_triangulated_camintrinsics_camextrinsics(_) \
    _(lensmodel,                          PyObject*,      NULL,    STRING_OBJECT,  ,                        NULL,                        -1,         {}                   ) \
    _(imagersizes,                        PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, imagersizes,                 NPY_INT32,    {-1 COMMA 2        } )

// Defaults for do_optimize... MUST match those in ingest_packed_state()
//
// Accepting observed_pixel_uncertainty for backwards compatibility. It doesn't
// do anything anymore
#define OPTIMIZE_ARGUMENTS_OPTIONAL(_) \
    _(observed_pixel_uncertainty,         double,         -1.0,    "d",  ,                                  NULL,           -1,         {})  \
    _(calobject_warp,                     PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, calobject_warp, NPY_DOUBLE, {2}) \
    _(Npoints_fixed,                      int,            0,       "i",  ,                                  NULL,           -1,         {})  \
    _(do_optimize_intrinsics_core,        int,            1,       "p",  ,                                  NULL,           -1,         {})  \
    _(do_optimize_intrinsics_distortions, int,            1,       "p",  ,                                  NULL,           -1,         {})  \
    _(do_optimize_extrinsics,             int,            1,       "p",  ,                                  NULL,           -1,         {})  \
    _(do_optimize_frames,                 int,            1,       "p",  ,                                  NULL,           -1,         {})  \
    _(do_optimize_calobject_warp,         int,            1,       "p",  ,                                  NULL,           -1,         {})  \
    _(calibration_object_spacing,         double,         -1.0,    "d",  ,                                  NULL,           -1,         {})  \
    _(point_min_range,                    double,         -1.0,    "d",  ,                                  NULL,           -1,         {})  \
    _(point_max_range,                    double,         -1.0,    "d",  ,                                  NULL,           -1,         {})  \
    _(verbose,                            int,            0,       "p",  ,                                  NULL,           -1,         {})  \
    _(do_apply_regularization,            int,            1,       "p",  ,                                  NULL,           -1,         {})  \
    _(do_apply_outlier_rejection,         int,            1,       "p",  ,                                  NULL,           -1,         {})  \
    _(imagepaths,                         PyObject*,      NULL,    "O",  ,                                  NULL,           -1,         {})
/* imagepaths is in the argument list purely to make the
   mrcal-show-residuals-board-observation tool work. The python code doesn't
   actually touch it */

#define OPTIMIZER_CALLBACK_ARGUMENTS_OPTIONAL_EXTRA(_) \
    _(no_jacobian,                        int,               0,    "p",  ,                                  NULL,           -1,         {}) \
    _(no_factorization,                   int,               0,    "p",  ,                                  NULL,           -1,         {})


// Using this for both optimize() and optimizer_callback()
static bool optimize_validate_args( // out
                                    mrcal_lensmodel_t* mrcal_lensmodel,

                                    // in
                                    bool is_optimize, // or optimizer_callback
                                    OPTIMIZE_ARGUMENTS_REQUIRED(ARG_LIST_DEFINE)
                                    OPTIMIZE_ARGUMENTS_OPTIONAL(ARG_LIST_DEFINE)
                                    OPTIMIZER_CALLBACK_ARGUMENTS_OPTIONAL_EXTRA(ARG_LIST_DEFINE)

                                    void* dummy __attribute__((unused)))
{
    static_assert( sizeof(mrcal_pose_t)/sizeof(double) == 6, "mrcal_pose_t is assumed to contain 6 elements");

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-to-pointer-cast"
    OPTIMIZE_ARGUMENTS_REQUIRED(CHECK_LAYOUT);
    OPTIMIZE_ARGUMENTS_OPTIONAL(CHECK_LAYOUT);
    OPTIMIZER_CALLBACK_ARGUMENTS_OPTIONAL_EXTRA(CHECK_LAYOUT);
#pragma GCC diagnostic pop

    int Ncameras_intrinsics = PyArray_DIMS(intrinsics)[0];
    int Ncameras_extrinsics = PyArray_DIMS(extrinsics_rt_fromref)[0];
    if( PyArray_DIMS(imagersizes)[0] != Ncameras_intrinsics )
    {
        BARF("Inconsistent Ncameras: 'intrinsics' says %ld, 'imagersizes' says %ld",
                     Ncameras_intrinsics,
                     PyArray_DIMS(imagersizes)[0]);
        return false;
    }

    long int Nobservations_board = PyArray_DIMS(observations_board)[0];
    if( PyArray_DIMS(indices_frame_camintrinsics_camextrinsics)[0] != Nobservations_board )
    {
        BARF("Inconsistent Nobservations_board: 'observations_board' says %ld, 'indices_frame_camintrinsics_camextrinsics' says %ld",
                     Nobservations_board,
                     PyArray_DIMS(indices_frame_camintrinsics_camextrinsics)[0]);
        return false;
    }

    if( Nobservations_board > 0)
    {
        if( calibration_object_spacing <= 0.0 )
        {
            BARF("We have board observations, so calibration_object_spacing MUST be a valid float > 0");
            return false;
        }

        if(do_optimize_calobject_warp &&
           IS_NULL(calobject_warp))
        {
            BARF("do_optimize_calobject_warp is True, so calobject_warp MUST be given as an array to seed the optimization and to receive the results");
            return false;
        }
    }

    int Nobservations_point = PyArray_DIMS(observations_point)[0];
    if( PyArray_DIMS(indices_point_camintrinsics_camextrinsics)[0] != Nobservations_point )
    {
        BARF("Inconsistent Nobservations_point: 'observations_point...' says %ld, 'indices_point_camintrinsics_camextrinsics' says %ld",
                     Nobservations_point,
                     PyArray_DIMS(indices_point_camintrinsics_camextrinsics)[0]);
        return false;
    }

    int Nobservations_point_triangulated = PyArray_DIMS(observations_point_triangulated)[0];
    if( PyArray_DIMS(indices_point_triangulated_camintrinsics_camextrinsics)[0] != Nobservations_point_triangulated )
    {
        BARF("Inconsistent Nobservations_point_triangulated: 'observations_point_triangulated...' says %ld, 'indices_triangulated_point_camintrinsics_camextrinsics' says %ld",
                     Nobservations_point_triangulated,
                     PyArray_DIMS(indices_point_triangulated_camintrinsics_camextrinsics)[0]);
        return false;
    }

    if(!parse_lensmodel_from_arg(mrcal_lensmodel, lensmodel))
        return false;

    int NlensParams = mrcal_lensmodel_num_params(mrcal_lensmodel);
    if( NlensParams != PyArray_DIMS(intrinsics)[1] )
    {
        BARF("intrinsics.shape[1] MUST be %d. Instead got %ld",
                     NlensParams,
                     PyArray_DIMS(intrinsics)[1] );
        return false;
    }

    // make sure the indices arrays are valid: the data is monotonic and
    // in-range
    int Nframes = PyArray_DIMS(frames_rt_toref)[0];
    int iframe_last  = -1;
    int icam_intrinsics_last = -1;
    int icam_extrinsics_last = -1;
    for(int i_observation=0; i_observation<Nobservations_board; i_observation++)
    {
        // check for monotonicity and in-rangeness
        int32_t iframe          = ((int32_t*)PyArray_DATA(indices_frame_camintrinsics_camextrinsics))[i_observation*3 + 0];
        int32_t icam_intrinsics = ((int32_t*)PyArray_DATA(indices_frame_camintrinsics_camextrinsics))[i_observation*3 + 1];
        int32_t icam_extrinsics = ((int32_t*)PyArray_DATA(indices_frame_camintrinsics_camextrinsics))[i_observation*3 + 2];

        // First I make sure everything is in-range
        if(iframe < 0 || iframe >= Nframes)
        {
            BARF("iframe MUST be in [0,%d], instead got %d in row %d of indices_frame_camintrinsics_camextrinsics",
                         Nframes-1, iframe, i_observation);
            return false;
        }
        if(icam_intrinsics < 0 || icam_intrinsics >= Ncameras_intrinsics)
        {
            BARF("icam_intrinsics MUST be in [0,%d], instead got %d in row %d of indices_frame_camintrinsics_camextrinsics",
                         Ncameras_intrinsics-1, icam_intrinsics, i_observation);
            return false;
        }
        if(icam_extrinsics < -1 || icam_extrinsics >= Ncameras_extrinsics)
        {
            BARF("icam_extrinsics MUST be in [-1,%d], instead got %d in row %d of indices_frame_camintrinsics_camextrinsics",
                         Ncameras_extrinsics-1, icam_extrinsics, i_observation);
            return false;
        }
        // And then I check monotonicity
        if(iframe == iframe_last)
        {
            if( icam_intrinsics < icam_intrinsics_last )
            {
                BARF("icam_intrinsics MUST be monotonically increasing in indices_frame_camintrinsics_camextrinsics. Instead row %d (frame %d) of indices_frame_camintrinsics_camextrinsics has icam_intrinsics=%d after previously seeing icam_intrinsics=%d",
                             i_observation, iframe, icam_intrinsics, icam_intrinsics_last);
                return false;
            }
            if( icam_extrinsics < icam_extrinsics_last )
            {
                BARF("icam_extrinsics MUST be monotonically increasing in indices_frame_camintrinsics_camextrinsics. Instead row %d (frame %d) of indices_frame_camintrinsics_camextrinsics has icam_extrinsics=%d after previously seeing icam_extrinsics=%d",
                             i_observation, iframe, icam_extrinsics, icam_extrinsics_last);
                return false;
            }
        }
        else if( iframe < iframe_last )
        {
            BARF("iframe MUST be monotonically increasing in indices_frame_camintrinsics_camextrinsics. Instead row %d of indices_frame_camintrinsics_camextrinsics has iframe=%d after previously seeing iframe=%d",
                         i_observation, iframe, iframe_last);
            return false;
        }

        iframe_last          = iframe;
        icam_intrinsics_last = icam_intrinsics;
        icam_extrinsics_last = icam_extrinsics;
    }
    int Npoints = PyArray_DIMS(points)[0];
    if( Npoints > 0 )
    {
        if(Npoints_fixed > Npoints)
        {
            BARF("I have Npoints=len(points)=%d, but Npoints_fixed=%d. Npoints_fixed > Npoints makes no sense",
                 Npoints, Npoints_fixed);
            return false;
        }
        if(point_min_range <= 0.0 ||
           point_max_range <= 0.0 ||
           point_min_range >= point_max_range)
        {
            BARF("Point observations were given, so point_min_range and point_max_range MUST have been given usable values > 0 and max>min");
            return false;
        }
    }
    else
    {
        if(Npoints_fixed)
        {
            BARF("No 'points' were given, so it's 'Npoints_fixed' doesn't do anything, and shouldn't be given");
            return false;
        }
    }

    for(int i_observation=0; i_observation<Nobservations_point; i_observation++)
    {
        int32_t i_point         = ((int32_t*)PyArray_DATA(indices_point_camintrinsics_camextrinsics))[i_observation*3 + 0];
        int32_t icam_intrinsics = ((int32_t*)PyArray_DATA(indices_point_camintrinsics_camextrinsics))[i_observation*3 + 1];
        int32_t icam_extrinsics = ((int32_t*)PyArray_DATA(indices_point_camintrinsics_camextrinsics))[i_observation*3 + 2];

        // First I make sure everything is in-range
        if(i_point < 0 || i_point >= Npoints)
        {
            BARF("i_point MUST be in [0,%d], instead got %d in row %d of indices_point_camintrinsics_camextrinsics",
                         Npoints-1, i_point, i_observation);
            return false;
        }
        if(icam_intrinsics < 0 || icam_intrinsics >= Ncameras_intrinsics)
        {
            BARF("icam_intrinsics MUST be in [0,%d], instead got %d in row %d of indices_point_camintrinsics_camextrinsics",
                         Ncameras_intrinsics-1, icam_intrinsics, i_observation);
            return false;
        }
        if(icam_extrinsics < -1 || icam_extrinsics >= Ncameras_extrinsics)
        {
            BARF("icam_extrinsics MUST be in [-1,%d], instead got %d in row %d of indices_point_camintrinsics_camextrinsics",
                         Ncameras_extrinsics-1, icam_extrinsics, i_observation);
            return false;
        }
    }

    for(int i_observation=0; i_observation<Nobservations_point_triangulated; i_observation++)
    {
        int32_t i_point         = ((int32_t*)PyArray_DATA(indices_point_triangulated_camintrinsics_camextrinsics))[i_observation*3 + 0];
        int32_t icam_intrinsics = ((int32_t*)PyArray_DATA(indices_point_triangulated_camintrinsics_camextrinsics))[i_observation*3 + 1];
        int32_t icam_extrinsics = ((int32_t*)PyArray_DATA(indices_point_triangulated_camintrinsics_camextrinsics))[i_observation*3 + 2];

        if(icam_intrinsics < 0 || icam_intrinsics >= Ncameras_intrinsics)
        {
            BARF("icam_intrinsics MUST be in [0,%d], instead got %d in row %d of indices_point_triangulated_camintrinsics_camextrinsics",
                         Ncameras_intrinsics-1, icam_intrinsics, i_observation);
            return false;
        }
        if(icam_extrinsics < -1 || icam_extrinsics >= Ncameras_extrinsics)
        {
            BARF("icam_extrinsics MUST be in [-1,%d], instead got %d in row %d of indices_point_triangulated_camintrinsics_camextrinsics",
                         Ncameras_extrinsics-1, icam_extrinsics, i_observation);
            return false;
        }
    }
    // There are more checks for triangulated points, but I run them later, in
    // fill_c_observations_point_triangulated()

    return true;
}

static void fill_c_observations_board(// out
                                      mrcal_observation_board_t* c_observations_board,

                                      // in
                                      int Nobservations_board,
                                      PyArrayObject* indices_frame_camintrinsics_camextrinsics)
{
    for(int i_observation=0; i_observation<Nobservations_board; i_observation++)
    {
        int32_t iframe          = ((int32_t*)PyArray_DATA(indices_frame_camintrinsics_camextrinsics))[i_observation*3 + 0];
        int32_t icam_intrinsics = ((int32_t*)PyArray_DATA(indices_frame_camintrinsics_camextrinsics))[i_observation*3 + 1];
        int32_t icam_extrinsics = ((int32_t*)PyArray_DATA(indices_frame_camintrinsics_camextrinsics))[i_observation*3 + 2];

        c_observations_board[i_observation].icam.intrinsics = icam_intrinsics;
        c_observations_board[i_observation].icam.extrinsics = icam_extrinsics;
        c_observations_board[i_observation].iframe          = iframe;
    }
}

static void fill_c_observations_point(// out
                                      mrcal_observation_point_t* c_observations_point,

                                      // in
                                      int Nobservations_point,
                                      PyArrayObject* indices_point_camintrinsics_camextrinsics,
                                      const mrcal_point3_t* observations_point_pool)
{
    for(int i_observation=0; i_observation<Nobservations_point; i_observation++)
    {
        int32_t i_point         = ((int32_t*)PyArray_DATA(indices_point_camintrinsics_camextrinsics))[i_observation*3 + 0];
        int32_t icam_intrinsics = ((int32_t*)PyArray_DATA(indices_point_camintrinsics_camextrinsics))[i_observation*3 + 1];
        int32_t icam_extrinsics = ((int32_t*)PyArray_DATA(indices_point_camintrinsics_camextrinsics))[i_observation*3 + 2];

        c_observations_point[i_observation].icam.intrinsics = icam_intrinsics;
        c_observations_point[i_observation].icam.extrinsics = icam_extrinsics;
        c_observations_point[i_observation].i_point         = i_point;

        c_observations_point[i_observation].px = observations_point_pool[i_observation];
    }
}

// return the number of points, or <0 on error
static
int fill_c_observations_point_triangulated(// output. I fill in the given arrays
                                           mrcal_observation_point_triangulated_t* c_observations_point_triangulated,

                                           // input
                                           const PyArrayObject* observations_point_triangulated, // may be NULL
                                           const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics)
{
    if(indices_point_triangulated_camintrinsics_camextrinsics == NULL)
        return 0;

    bool validate_arguments(void)
    {
        if(observations_point_triangulated != NULL)
        {
            ARGDEF_observations_point_triangulated(CHECK_LAYOUT);
        }
        ARGDEF_indices_point_triangulated_camintrinsics_camextrinsics(CHECK_LAYOUT);

        return true;
    }

    if(!validate_arguments())
        return -1;

    int N = (int)PyArray_DIM(indices_point_triangulated_camintrinsics_camextrinsics, 0);
    if(observations_point_triangulated != NULL)
    {
        if(N != (int)PyArray_DIM(observations_point_triangulated, 0))
        {
            BARF("Inconsistent point counts. observations_point_triangulated.shape[0] = %d, but indices_point_triangulated_camintrinsics_camextrinsics.shape[0] = %d",
                (int)PyArray_DIM(observations_point_triangulated, 0),N);
            return -1;
        }
    }

    const double* observations_point_triangulated__data =
        (observations_point_triangulated != NULL) ?
          (const double*)PyArray_DATA((PyArrayObject*)observations_point_triangulated) :
          NULL;
    const int32_t* indices_point_triangulated_camintrinsics_camextrinsics__data =
        (const int32_t*)PyArray_DATA((PyArrayObject*)indices_point_triangulated_camintrinsics_camextrinsics);

    int ipoint_current = -1;
    int Npoints_in_this_set = 0;

    bool finish_set(int ipoint_last_in_set)
    {
        if(ipoint_last_in_set < 0)
            // No "last" set exists. Nothing to do.
            return true;

        c_observations_point_triangulated[ipoint_last_in_set].last_in_set = true;
        if(Npoints_in_this_set < 2)
        {
            BARF("Error in indices_point_triangulated_camintrinsics_camextrinsics[%d]. Each point must be obseved at least 2 times",
                ipoint_last_in_set);
            return false;
        }
        return true;
    }

    for(int i=0; i<N; i++)
    {
        const int32_t* row = &indices_point_triangulated_camintrinsics_camextrinsics__data[3*i];

        const int32_t ipoint          = row[0];
        const int32_t icam_intrinsics = row[1];
        const int32_t icam_extrinsics = row[2];

        c_observations_point_triangulated[i].last_in_set = false;
        c_observations_point_triangulated[i].icam = (mrcal_camera_index_t){.intrinsics = icam_intrinsics,
                                                                         .extrinsics = icam_extrinsics};
        if(observations_point_triangulated__data != NULL)
        {
            mrcal_point3_t* px = (mrcal_point3_t*)(&observations_point_triangulated__data[3*i]);
            c_observations_point_triangulated[i].px = *px;
        }
        else
            c_observations_point_triangulated[i].px = (mrcal_point3_t){};


        if(ipoint < 0)
        {
            BARF("Error in indices_point_triangulated_camintrinsics_camextrinsics[%d]. Saw ipoint=%d. Each one must be >=0",
                i, ipoint);
            return -1;
        }
        else if(ipoint == ipoint_current)
        {
            Npoints_in_this_set++;
        }
        else if(ipoint == ipoint_current+1)
        {
            // The previous point was the last in the set
            if(!finish_set(i-1))
                return -1;

            ipoint_current = ipoint;
            Npoints_in_this_set = 1;
        }
        else
        {
            BARF("Error in indices_point_triangulated_camintrinsics_camextrinsics[%d]. All ipoint must be consecutive and monotonic",
                i);
            return -1;
        }
    }

    if(!finish_set(N-1))
        return -1;

    return N;
}

static
PyObject* _optimize(bool is_optimize, // or optimizer_callback
                    PyObject* args,
                    PyObject* kwargs)
{
    PyObject* result = NULL;

    PyArrayObject* p_packed_final = NULL;
    PyArrayObject* x_final        = NULL;
    PyObject*      pystats        = NULL;

    PyArrayObject* P             = NULL;
    PyArrayObject* I             = NULL;
    PyArrayObject* X             = NULL;
    PyObject*      factorization = NULL;
    PyObject*      jacobian      = NULL;

    SET_SIGINT();

    OPTIMIZE_ARGUMENTS_REQUIRED(ARG_DEFINE);
    OPTIMIZE_ARGUMENTS_OPTIONAL(ARG_DEFINE);
    OPTIMIZER_CALLBACK_ARGUMENTS_OPTIONAL_EXTRA(ARG_DEFINE);

    int calibration_object_height_n = -1;
    int calibration_object_width_n  = -1;

    if(is_optimize)
    {
        char* keywords[] = { OPTIMIZE_ARGUMENTS_REQUIRED(NAMELIST)
                             OPTIMIZE_ARGUMENTS_OPTIONAL(NAMELIST)
                             NULL};
        if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                         OPTIMIZE_ARGUMENTS_REQUIRED(PARSECODE) "|"
                                         OPTIMIZE_ARGUMENTS_OPTIONAL(PARSECODE)
                                         ":mrcal.optimize",

                                         keywords,

                                         OPTIMIZE_ARGUMENTS_REQUIRED(PARSEARG)
                                         OPTIMIZE_ARGUMENTS_OPTIONAL(PARSEARG) NULL))
            goto done;
    }
    else
    {
        // optimizer_callback
        char* keywords[] = { OPTIMIZE_ARGUMENTS_REQUIRED(NAMELIST)
                             OPTIMIZE_ARGUMENTS_OPTIONAL(NAMELIST)
                             OPTIMIZER_CALLBACK_ARGUMENTS_OPTIONAL_EXTRA(NAMELIST)
                             NULL};
        if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                         OPTIMIZE_ARGUMENTS_REQUIRED(PARSECODE) "|"
                                         OPTIMIZE_ARGUMENTS_OPTIONAL(PARSECODE)
                                         OPTIMIZER_CALLBACK_ARGUMENTS_OPTIONAL_EXTRA(PARSECODE)
                                         ":mrcal.optimizer_callback",

                                         keywords,

                                         OPTIMIZE_ARGUMENTS_REQUIRED(PARSEARG)
                                         OPTIMIZE_ARGUMENTS_OPTIONAL(PARSEARG)
                                         OPTIMIZER_CALLBACK_ARGUMENTS_OPTIONAL_EXTRA(PARSEARG) NULL))
            goto done;
    }

    // Some of my input arguments can be empty (None). The code all assumes that
    // everything is a properly-dimensioned numpy array, with "empty" meaning
    // some dimension is 0. Here I make this conversion. The user can pass None,
    // and we still do the right thing.
    //
    // There's a silly implementation detail here: if you have a preprocessor
    // macro M(x), and you pass it M({1,2,3}), the preprocessor see 3 separate
    // args, not 1. That's why I have a __VA_ARGS__ here and why I instantiate a
    // separate dims[] (PyArray_SimpleNew is a macro too)
#define SET_SIZE0_IF_NONE(x, type, ...)                                 \
    ({                                                                  \
        if( IS_NULL(x) )                                                \
        {                                                               \
            if( x != NULL ) Py_DECREF(x);                               \
            npy_intp dims[] = {__VA_ARGS__};                            \
            x = (PyArrayObject*)PyArray_SimpleNew(sizeof(dims)/sizeof(dims[0]), \
                                                  dims, type);          \
        }                                                               \
    })

    SET_SIZE0_IF_NONE(extrinsics_rt_fromref,      NPY_DOUBLE, 0,6);

    SET_SIZE0_IF_NONE(frames_rt_toref,            NPY_DOUBLE, 0,6);
    SET_SIZE0_IF_NONE(observations_board,         NPY_DOUBLE, 0,179,171,3); // arbitrary numbers; shouldn't matter
    SET_SIZE0_IF_NONE(indices_frame_camintrinsics_camextrinsics, NPY_INT32,    0,3);

    SET_SIZE0_IF_NONE(points,                                                 NPY_DOUBLE, 0, 3);
    SET_SIZE0_IF_NONE(observations_point,                                     NPY_DOUBLE, 0, 3);
    SET_SIZE0_IF_NONE(indices_point_camintrinsics_camextrinsics,              NPY_INT32,  0, 3);
    SET_SIZE0_IF_NONE(observations_point_triangulated,                        NPY_DOUBLE, 0, 3);
    SET_SIZE0_IF_NONE(indices_point_triangulated_camintrinsics_camextrinsics, NPY_INT32,  0, 3);
    SET_SIZE0_IF_NONE(imagersizes,                                            NPY_INT32,  0, 2);
#undef SET_NULL_IF_NONE


    mrcal_lensmodel_t mrcal_lensmodel;
    // Check the arguments for optimize(). If optimizer_callback, then the other
    // stuff is defined, but it all has valid, default values
    if( !optimize_validate_args(&mrcal_lensmodel,
                                is_optimize,
                                OPTIMIZE_ARGUMENTS_REQUIRED(ARG_LIST_CALL)
                                OPTIMIZE_ARGUMENTS_OPTIONAL(ARG_LIST_CALL)
                                OPTIMIZER_CALLBACK_ARGUMENTS_OPTIONAL_EXTRA(ARG_LIST_CALL)
                                NULL))
        goto done;

    // Can't compute a factorization without a jacobian. That's what we're factoring
    if(!no_factorization) no_jacobian = false;

    {
        int Ncameras_intrinsics = PyArray_DIMS(intrinsics)[0];
        int Ncameras_extrinsics = PyArray_DIMS(extrinsics_rt_fromref)[0];
        int Nframes             = PyArray_DIMS(frames_rt_toref)[0];
        int Npoints             = PyArray_DIMS(points)[0];
        int Nobservations_board = PyArray_DIMS(observations_board)[0];

        if( Nobservations_board > 0 )
        {
            calibration_object_height_n = PyArray_DIMS(observations_board)[1];
            calibration_object_width_n  = PyArray_DIMS(observations_board)[2];
        }

        // The checks in optimize_validate_args() make sure these casts are kosher
        double*             c_intrinsics     = (double*)  PyArray_DATA(intrinsics);
        mrcal_pose_t*       c_extrinsics     = (mrcal_pose_t*)  PyArray_DATA(extrinsics_rt_fromref);
        mrcal_pose_t*       c_frames         = (mrcal_pose_t*)  PyArray_DATA(frames_rt_toref);
        mrcal_point3_t*     c_points         = (mrcal_point3_t*)PyArray_DATA(points);
        mrcal_calobject_warp_t*     c_calobject_warp =
            IS_NULL(calobject_warp) ?
            NULL : (mrcal_calobject_warp_t*)PyArray_DATA(calobject_warp);


        mrcal_point3_t* c_observations_board_pool = (mrcal_point3_t*)PyArray_DATA(observations_board); // must be contiguous; made sure above
        mrcal_observation_board_t c_observations_board[Nobservations_board];
        fill_c_observations_board(c_observations_board,
                                  Nobservations_board,
                                  indices_frame_camintrinsics_camextrinsics);

        int Nobservations_point = PyArray_DIMS(observations_point)[0];
        mrcal_observation_point_t c_observations_point[Nobservations_point];
        fill_c_observations_point(c_observations_point,
                                  Nobservations_point,
                                  indices_point_camintrinsics_camextrinsics,
                                  (mrcal_point3_t*)PyArray_DATA(observations_point));

        int Nobservations_point_triangulated = PyArray_DIMS(observations_point_triangulated)[0];
        mrcal_observation_point_triangulated_t c_observations_point_triangulated[Nobservations_point_triangulated];
        if( fill_c_observations_point_triangulated(c_observations_point_triangulated,
                                                   observations_point_triangulated,
                                                   indices_point_triangulated_camintrinsics_camextrinsics)
            < 0 )
        {
            goto done;
        }


        mrcal_problem_selections_t problem_selections =
            { .do_optimize_intrinsics_core       = do_optimize_intrinsics_core,
              .do_optimize_intrinsics_distortions= do_optimize_intrinsics_distortions,
              .do_optimize_extrinsics            = do_optimize_extrinsics,
              .do_optimize_frames                = do_optimize_frames,
              .do_optimize_calobject_warp        = do_optimize_calobject_warp,
              .do_apply_regularization           = do_apply_regularization,
              .do_apply_outlier_rejection        = do_apply_outlier_rejection
            };

        mrcal_problem_constants_t problem_constants =
            {.point_min_range = point_min_range,
             .point_max_range = point_max_range};

        int Nmeasurements = mrcal_num_measurements(Nobservations_board,
                                                   Nobservations_point,
                                                   c_observations_point_triangulated,
                                                   Nobservations_point_triangulated,
                                                   calibration_object_width_n,
                                                   calibration_object_height_n,
                                                   Ncameras_intrinsics, Ncameras_extrinsics,
                                                   Nframes,
                                                   Npoints, Npoints_fixed,
                                                   problem_selections,
                                                   &mrcal_lensmodel);

        int Nintrinsics_state = mrcal_num_intrinsics_optimization_params(problem_selections, &mrcal_lensmodel);

        // input
        int* c_imagersizes = PyArray_DATA(imagersizes);

        int Nstate = mrcal_num_states(Ncameras_intrinsics, Ncameras_extrinsics,
                                      Nframes, Npoints, Npoints_fixed, Nobservations_board,
                                      problem_selections, &mrcal_lensmodel);

        // both optimize() and optimizer_callback() use this
        p_packed_final = (PyArrayObject*)PyArray_SimpleNew(1, ((npy_intp[]){Nstate}), NPY_DOUBLE);
        double* c_p_packed_final = PyArray_DATA(p_packed_final);

        x_final = (PyArrayObject*)PyArray_SimpleNew(1, ((npy_intp[]){Nmeasurements}), NPY_DOUBLE);
        double* c_x_final = PyArray_DATA(x_final);

        if( is_optimize )
        {
            // we're wrapping mrcal_optimize()
            const int Npoints_fromBoards =
                Nobservations_board *
                calibration_object_width_n*calibration_object_height_n;

            mrcal_stats_t stats =
                mrcal_optimize( c_p_packed_final,
                                Nstate*sizeof(double),
                                c_x_final,
                                Nmeasurements*sizeof(double),
                                c_intrinsics,
                                c_extrinsics,
                                c_frames,
                                c_points,
                                c_calobject_warp,

                                Ncameras_intrinsics, Ncameras_extrinsics,
                                Nframes, Npoints, Npoints_fixed,

                                c_observations_board,
                                c_observations_point,
                                Nobservations_board,
                                Nobservations_point,

                                c_observations_point_triangulated,
                                Nobservations_point_triangulated,

                                c_observations_board_pool,

                                &mrcal_lensmodel,
                                c_imagersizes,
                                problem_selections, &problem_constants,

                                calibration_object_spacing,
                                calibration_object_width_n,
                                calibration_object_height_n,
                                verbose,

                                false);

            if(stats.rms_reproj_error__pixels < 0.0)
            {
                // Error! I throw an exception
                BARF("mrcal.optimize() failed!");
                goto done;
            }

            pystats = PyDict_New();
            if(pystats == NULL)
            {
                BARF("PyDict_New() failed!");
                goto done;
            }
#define MRCAL_STATS_ITEM_POPULATE_DICT(type, name, pyconverter)         \
            {                                                           \
                PyObject* obj = pyconverter( (type)stats.name);         \
                if( obj == NULL)                                        \
                {                                                       \
                    BARF("Couldn't make PyObject for '" #name "'"); \
                    goto done;                                          \
                }                                                       \
                                                                        \
                if( 0 != PyDict_SetItemString(pystats, #name, obj) )    \
                {                                                       \
                    BARF("Couldn't add to stats dict '" #name "'"); \
                    Py_DECREF(obj);                                     \
                    goto done;                                          \
                }                                                       \
            }
            MRCAL_STATS_ITEM(MRCAL_STATS_ITEM_POPULATE_DICT);

            if( 0 != PyDict_SetItemString(pystats, "p_packed",
                                          (PyObject*)p_packed_final) )
            {
                BARF("Couldn't add to stats dict 'p_packed'");
                goto done;
            }
            if( 0 != PyDict_SetItemString(pystats, "x",
                                          (PyObject*)x_final) )
            {
                BARF("Couldn't add to stats dict 'x'");
                goto done;
            }

            result = pystats;
            Py_INCREF(result);
        }
        else
        {
            // we're wrapping mrcal_optimizer_callback()

            int N_j_nonzero = _mrcal_num_j_nonzero(Nobservations_board,
                                                   Nobservations_point,
                                                   c_observations_point_triangulated,
                                                   Nobservations_point_triangulated,
                                                   calibration_object_width_n,
                                                   calibration_object_height_n,
                                                   Ncameras_intrinsics, Ncameras_extrinsics,
                                                   Nframes,
                                                   Npoints, Npoints_fixed,
                                                   c_observations_board,
                                                   c_observations_point,
                                                   problem_selections,
                                                   &mrcal_lensmodel);
            cholmod_sparse Jt = {
                .nrow   = Nstate,
                .ncol   = Nmeasurements,
                .nzmax  = N_j_nonzero,
                .stype  = 0,
                .itype  = CHOLMOD_INT,
                .xtype  = CHOLMOD_REAL,
                .dtype  = CHOLMOD_DOUBLE,
                .sorted = 1,
                .packed = 1 };

            if(!no_jacobian)
            {
                // above I made sure that no_jacobian was false if !no_factorization
                P = (PyArrayObject*)PyArray_SimpleNew(1, ((npy_intp[]){Nmeasurements + 1}), NPY_INT32);
                I = (PyArrayObject*)PyArray_SimpleNew(1, ((npy_intp[]){N_j_nonzero      }), NPY_INT32);
                X = (PyArrayObject*)PyArray_SimpleNew(1, ((npy_intp[]){N_j_nonzero      }), NPY_DOUBLE);
                Jt.p = PyArray_DATA(P);
                Jt.i = PyArray_DATA(I);
                Jt.x = PyArray_DATA(X);
            }

            if(!mrcal_optimizer_callback( // out
                                         c_p_packed_final,
                                         Nstate*sizeof(double),
                                         c_x_final,
                                         Nmeasurements*sizeof(double),
                                         no_jacobian ? NULL : &Jt,

                                         // in
                                         c_intrinsics,
                                         c_extrinsics,
                                         c_frames,
                                         c_points,
                                         c_calobject_warp,

                                         Ncameras_intrinsics, Ncameras_extrinsics,
                                         Nframes, Npoints, Npoints_fixed,

                                         c_observations_board,
                                         c_observations_point,
                                         Nobservations_board,
                                         Nobservations_point,

                                         c_observations_point_triangulated,
                                         Nobservations_point_triangulated,

                                         c_observations_board_pool,

                                         &mrcal_lensmodel,
                                         c_imagersizes,
                                         problem_selections, &problem_constants,

                                         calibration_object_spacing,
                                         calibration_object_width_n,
                                         calibration_object_height_n,
                                         verbose) )
            {
                BARF("mrcal_optimizer_callback() failed!'");
                goto done;
            }

            if(no_factorization)
            {
                factorization = Py_None;
                Py_INCREF(factorization);
            }
            else
            {
                // above I made sure that no_jacobian was false if !no_factorization
                factorization = CHOLMOD_factorization_from_cholmod_sparse(&Jt);
                if(factorization == NULL)
                {
                    // Couldn't compute factorization. I don't barf, but set the
                    // factorization to None
                    factorization = Py_None;
                    Py_INCREF(factorization);
                    PyErr_Clear();
                }
            }

            if(no_jacobian)
            {
                jacobian = Py_None;
                Py_INCREF(jacobian);
            }
            else
            {
                jacobian = csr_from_cholmod_sparse((PyObject*)P,
                                                   (PyObject*)I,
                                                   (PyObject*)X);
                if(jacobian == NULL)
                {
                    // reuse the existing error
                    goto done;
                }
            }

            result = PyTuple_Pack(4,
                                  (PyObject*)p_packed_final,
                                  (PyObject*)x_final,
                                  jacobian,
                                  factorization);
        }
    }

 done:
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-to-pointer-cast"
    OPTIMIZE_ARGUMENTS_REQUIRED(FREE_PYARRAY);
    OPTIMIZE_ARGUMENTS_OPTIONAL(FREE_PYARRAY);
    OPTIMIZER_CALLBACK_ARGUMENTS_OPTIONAL_EXTRA(FREE_PYARRAY);
#pragma GCC diagnostic pop

    Py_XDECREF(p_packed_final);
    Py_XDECREF(x_final);
    Py_XDECREF(pystats);
    Py_XDECREF(P);
    Py_XDECREF(I);
    Py_XDECREF(X);
    Py_XDECREF(factorization);
    Py_XDECREF(jacobian);

    RESET_SIGINT();
    return result;
}

static PyObject* optimizer_callback(PyObject* NPY_UNUSED(self),
                                   PyObject* args,
                                   PyObject* kwargs)
{
    return _optimize(false, args, kwargs);
}
static PyObject* optimize(PyObject* NPY_UNUSED(self),
                          PyObject* args,
                          PyObject* kwargs)
{
    return _optimize(true, args, kwargs);
}


// The state_index_... python functions don't need the full data but many of
// them do need to know the dimensionality of the data. Thus these can take the
// same arguments as optimizer_callback(). OR in lieu of that, the dimensions can
// be passed-in explicitly with arguments
//
// If both are given, the explicit arguments take precedence. If neither are
// given, I assume 0.
//
// This means that the arguments that are required in optimizer_callback() are
// only optional here
typedef int (callback_state_index_t)(int i,
                                     int Ncameras_intrinsics,
                                     int Ncameras_extrinsics,
                                     int Nframes,
                                     int Npoints,
                                     int Npoints_fixed,
                                     int Nobservations_board,
                                     int Nobservations_point,
                                     int calibration_object_width_n,
                                     int calibration_object_height_n,
                                     const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                     const mrcal_lensmodel_t* lensmodel,
                                     mrcal_problem_selections_t problem_selections);
#define STATE_INDEX_GENERIC(f, ...) state_index_generic(callback_ ## f, \
                                                        #f,             \
                                                        __VA_ARGS__ )
static PyObject* state_index_generic(callback_state_index_t cb,
                                     const char* called_function,
                                     PyObject* self, PyObject* args, PyObject* kwargs,
                                     const char* argname)
{
    // This is VERY similar to _pack_unpack_state(). Please consolidate
    // Also somewhat similar to _optimize()

    PyObject* result = NULL;

    OPTIMIZE_ARGUMENTS_REQUIRED(ARG_DEFINE);
    OPTIMIZE_ARGUMENTS_OPTIONAL(ARG_DEFINE);

    int i = -1;

    int Ncameras_intrinsics = -1;
    int Ncameras_extrinsics = -1;
    int Nframes             = -1;
    int Npoints             = -1;
    int Nobservations_board = -1;
    int Nobservations_point = -1;

    char* keywords[] = { (char*)argname,
                         OPTIMIZE_ARGUMENTS_REQUIRED(NAMELIST)

                         "Ncameras_intrinsics",
                         "Ncameras_extrinsics",
                         "Nframes",
                         "Npoints",
                         "Nobservations_board",
                         "Nobservations_point",
                         OPTIMIZE_ARGUMENTS_OPTIONAL(NAMELIST)
                         NULL};
    char** keywords_noargname = &keywords[1];

    // needs to be big-enough to store the largest-possible called_function
#define CALLED_FUNCTION_BUFFER "123456789012345678901234567890123456789012345678901234567890"

    if(argname != NULL)
    {
        char arg_string[] =
            "i"
            "|" // everything is optional. I apply
                // logic down the line to get what
                // I need
            OPTIMIZE_ARGUMENTS_REQUIRED(PARSECODE)
            "iiiiii"
            OPTIMIZE_ARGUMENTS_OPTIONAL(PARSECODE)
            ":mrcal." CALLED_FUNCTION_BUFFER;
        if(strlen(CALLED_FUNCTION_BUFFER) < strlen(called_function))
        {
            BARF("CALLED_FUNCTION_BUFFER too small for '%s'. This is a a bug", called_function);
            goto done;
        }
        arg_string[strlen(arg_string) - strlen(CALLED_FUNCTION_BUFFER)] = '\0';
        strcat(arg_string, called_function);

        if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                         arg_string,
                                         keywords,

                                         &i,
                                         OPTIMIZE_ARGUMENTS_REQUIRED(PARSEARG)
                                         &Ncameras_intrinsics,
                                         &Ncameras_extrinsics,
                                         &Nframes,
                                         &Npoints,
                                         &Nobservations_board,
                                         &Nobservations_point,
                                         OPTIMIZE_ARGUMENTS_OPTIONAL(PARSEARG) NULL))
            goto done;
    }
    else
    {
        char arg_string[] =
            OPTIMIZE_ARGUMENTS_REQUIRED(PARSECODE) "|"
            "iiiiii"
            OPTIMIZE_ARGUMENTS_OPTIONAL(PARSECODE)
            ":mrcal." CALLED_FUNCTION_BUFFER;
        if(strlen(CALLED_FUNCTION_BUFFER) < strlen(called_function))
        {
            BARF("CALLED_FUNCTION_BUFFER too small for '%s'. This is a a bug", called_function);
            goto done;
        }
        arg_string[strlen(arg_string) - strlen(CALLED_FUNCTION_BUFFER)] = '\0';
        strcat(arg_string, called_function);

        if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                         arg_string,

                                         keywords_noargname,

                                         OPTIMIZE_ARGUMENTS_REQUIRED(PARSEARG)
                                         &Ncameras_intrinsics,
                                         &Ncameras_extrinsics,
                                         &Nframes,
                                         &Npoints,
                                         &Nobservations_board,
                                         &Nobservations_point,
                                         OPTIMIZE_ARGUMENTS_OPTIONAL(PARSEARG) NULL))
            goto done;
    }
#undef CALLED_FUNCTION_BUFFER


    if(lensmodel == NULL)
    {
        BARF("The 'lensmodel' argument is required");
        goto done;
    }

    const mrcal_problem_selections_t problem_selections =
        { .do_optimize_intrinsics_core       = do_optimize_intrinsics_core,
          .do_optimize_intrinsics_distortions= do_optimize_intrinsics_distortions,
          .do_optimize_extrinsics            = do_optimize_extrinsics,
          .do_optimize_frames                = do_optimize_frames,
          .do_optimize_calobject_warp        = do_optimize_calobject_warp,
          .do_apply_regularization           = do_apply_regularization
        };

    mrcal_lensmodel_t mrcal_lensmodel;
    if(!parse_lensmodel_from_arg(&mrcal_lensmodel, lensmodel))
        goto done;

    // checks dimensionality of array !IS_NULL. So if any array isn't passed-in,
    // that's OK! After I do this and if !IS_NULL, then I can ask for array
    // dimensions safely
    bool check(void)
    {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-to-pointer-cast"
        OPTIMIZE_ARGUMENTS_REQUIRED(CHECK_LAYOUT);
        OPTIMIZE_ARGUMENTS_OPTIONAL(CHECK_LAYOUT);
#pragma GCC diagnostic pop
        return true;
    }
    if(!check()) goto done;

    // If explicit dimensions are given, use them. If they're not given, but we
    // have an array, use those dimensions. If an array isn't given either, use
    // 0
    if(Ncameras_intrinsics < 0) Ncameras_intrinsics = IS_NULL(intrinsics)            ? 0 : PyArray_DIMS(intrinsics)            [0];
    if(Ncameras_extrinsics < 0) Ncameras_extrinsics = IS_NULL(extrinsics_rt_fromref) ? 0 : PyArray_DIMS(extrinsics_rt_fromref) [0];
    if(Nframes < 0)             Nframes             = IS_NULL(frames_rt_toref)       ? 0 : PyArray_DIMS(frames_rt_toref)       [0];
    if(Npoints < 0)             Npoints             = IS_NULL(points)                ? 0 : PyArray_DIMS(points)                [0];
    if(Nobservations_board < 0) Nobservations_board = IS_NULL(observations_board)    ? 0 : PyArray_DIMS(observations_board)    [0];
    if(Nobservations_point < 0) Nobservations_point = IS_NULL(observations_point)    ? 0 : PyArray_DIMS(observations_point)    [0];

    int calibration_object_height_n = -1;
    int calibration_object_width_n  = -1;
    if( Nobservations_board > 0 )
    {
        calibration_object_height_n = PyArray_DIMS(observations_board)[1];
        calibration_object_width_n  = PyArray_DIMS(observations_board)[2];
    }


    int index = cb(i,
                   Ncameras_intrinsics,
                   Ncameras_extrinsics,
                   Nframes,
                   Npoints,
                   Npoints_fixed,
                   Nobservations_board,
                   Nobservations_point,
                   calibration_object_width_n,
                   calibration_object_height_n,
                   indices_point_triangulated_camintrinsics_camextrinsics,
                   &mrcal_lensmodel,
                   problem_selections);

    if(PyErr_Occurred())
        goto done;

    if(index >= 0)
        result = PyLong_FromLong(index);
    else
    {
        result = Py_None;
        Py_INCREF(result);
    }

 done:
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-to-pointer-cast"
    OPTIMIZE_ARGUMENTS_REQUIRED(FREE_PYARRAY) ;
    OPTIMIZE_ARGUMENTS_OPTIONAL(FREE_PYARRAY) ;
#pragma GCC diagnostic pop

    return result;
}

static int callback_state_index_intrinsics(int i,
                                           int Ncameras_intrinsics,
                                           int Ncameras_extrinsics,
                                           int Nframes,
                                           int Npoints,
                                           int Npoints_fixed,
                                           int Nobservations_board,
                                           int Nobservations_point,
                                           int calibration_object_width_n,
                                           int calibration_object_height_n,
                                           const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                           const mrcal_lensmodel_t* lensmodel,
                                           mrcal_problem_selections_t problem_selections)
{
    return mrcal_state_index_intrinsics(i,
                                        Ncameras_intrinsics, Ncameras_extrinsics,
                                        Nframes,
                                        Npoints, Npoints_fixed, Nobservations_board,
                                        problem_selections,
                                        lensmodel);
}
static PyObject* state_index_intrinsics(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(state_index_intrinsics,
                               self, args, kwargs,
                               "icam_intrinsics");
}

static int callback_num_states_intrinsics(int i,
                                          int Ncameras_intrinsics,
                                          int Ncameras_extrinsics,
                                          int Nframes,
                                          int Npoints,
                                          int Npoints_fixed,
                                          int Nobservations_board,
                                          int Nobservations_point,
                                          int calibration_object_width_n,
                                          int calibration_object_height_n,
                                          const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                          const mrcal_lensmodel_t* lensmodel,
                                          mrcal_problem_selections_t problem_selections)
{
    return mrcal_num_states_intrinsics(Ncameras_intrinsics,
                                       problem_selections, lensmodel);
}
static PyObject* num_states_intrinsics(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(num_states_intrinsics,
                               self, args, kwargs,
                               NULL);
}

static int callback_state_index_extrinsics(int i,
                                           int Ncameras_intrinsics,
                                           int Ncameras_extrinsics,
                                           int Nframes,
                                           int Npoints,
                                           int Npoints_fixed,
                                           int Nobservations_board,
                                           int Nobservations_point,
                                           int calibration_object_width_n,
                                           int calibration_object_height_n,
                                           const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                           const mrcal_lensmodel_t* lensmodel,
                                           mrcal_problem_selections_t problem_selections)
{
    return
        mrcal_state_index_extrinsics(i,
                                     Ncameras_intrinsics, Ncameras_extrinsics,
                                     Nframes,
                                     Npoints, Npoints_fixed, Nobservations_board,
                                     problem_selections,
                                     lensmodel);
}
static PyObject* state_index_extrinsics(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(state_index_extrinsics,
                               self, args, kwargs,
                               "icam_extrinsics");
}

static int callback_num_states_extrinsics(int i,
                                          int Ncameras_intrinsics,
                                          int Ncameras_extrinsics,
                                          int Nframes,
                                          int Npoints,
                                          int Npoints_fixed,
                                          int Nobservations_board,
                                          int Nobservations_point,
                                          int calibration_object_width_n,
                                          int calibration_object_height_n,
                                          const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                          const mrcal_lensmodel_t* lensmodel,
                                          mrcal_problem_selections_t problem_selections)
{
    return
        mrcal_num_states_extrinsics(Ncameras_extrinsics, problem_selections);
}
static PyObject* num_states_extrinsics(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(num_states_extrinsics,
                               self, args, kwargs,
                               NULL);
}

static int callback_state_index_frames(int i,
                                       int Ncameras_intrinsics,
                                       int Ncameras_extrinsics,
                                       int Nframes,
                                       int Npoints,
                                       int Npoints_fixed,
                                       int Nobservations_board,
                                       int Nobservations_point,
                                       int calibration_object_width_n,
                                       int calibration_object_height_n,
                                       const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                       const mrcal_lensmodel_t* lensmodel,
                                       mrcal_problem_selections_t problem_selections)
{
    return
        mrcal_state_index_frames(i,
                                 Ncameras_intrinsics, Ncameras_extrinsics,
                                 Nframes,
                                 Npoints, Npoints_fixed, Nobservations_board,
                                 problem_selections,
                                 lensmodel);
}
static PyObject* state_index_frames(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(state_index_frames,
                               self, args, kwargs,
                               "iframe");
}

static int callback_num_states_frames(int i,
                                      int Ncameras_intrinsics,
                                      int Ncameras_extrinsics,
                                      int Nframes,
                                      int Npoints,
                                      int Npoints_fixed,
                                      int Nobservations_board,
                                      int Nobservations_point,
                                      int calibration_object_width_n,
                                      int calibration_object_height_n,
                                      const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                      const mrcal_lensmodel_t* lensmodel,
                                      mrcal_problem_selections_t problem_selections)
{
    return
        mrcal_num_states_frames(Nframes, problem_selections);
}
static PyObject* num_states_frames(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(num_states_frames,
                               self, args, kwargs,
                               NULL);
}

static int callback_state_index_points(int i,
                                       int Ncameras_intrinsics,
                                       int Ncameras_extrinsics,
                                       int Nframes,
                                       int Npoints,
                                       int Npoints_fixed,
                                       int Nobservations_board,
                                       int Nobservations_point,
                                       int calibration_object_width_n,
                                       int calibration_object_height_n,
                                       const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                       const mrcal_lensmodel_t* lensmodel,
                                       mrcal_problem_selections_t problem_selections)
{
    return
        mrcal_state_index_points(i,
                                 Ncameras_intrinsics, Ncameras_extrinsics,
                                 Nframes,
                                 Npoints, Npoints_fixed, Nobservations_board,
                                 problem_selections,
                                 lensmodel);
}
static PyObject* state_index_points(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(state_index_points,
                               self, args, kwargs,
                               "i_point");
}

static int callback_num_states_points(int i,
                                      int Ncameras_intrinsics,
                                      int Ncameras_extrinsics,
                                      int Nframes,
                                      int Npoints,
                                      int Npoints_fixed,
                                      int Nobservations_board,
                                      int Nobservations_point,
                                      int calibration_object_width_n,
                                      int calibration_object_height_n,
                                      const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                      const mrcal_lensmodel_t* lensmodel,
                                      mrcal_problem_selections_t problem_selections)
{
    return
        mrcal_num_states_points(Npoints, Npoints_fixed, problem_selections);
}
static PyObject* num_states_points(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(num_states_points,
                               self, args, kwargs,
                               NULL);
}

static int callback_state_index_calobject_warp(int i,
                                               int Ncameras_intrinsics,
                                               int Ncameras_extrinsics,
                                               int Nframes,
                                               int Npoints,
                                               int Npoints_fixed,
                                               int Nobservations_board,
                                               int Nobservations_point,
                                               int calibration_object_width_n,
                                               int calibration_object_height_n,
                                               const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                               const mrcal_lensmodel_t* lensmodel,
                                               mrcal_problem_selections_t problem_selections)
{
    return
        mrcal_state_index_calobject_warp( Ncameras_intrinsics, Ncameras_extrinsics,
                                          Nframes,
                                          Npoints, Npoints_fixed, Nobservations_board,
                                          problem_selections,
                                          lensmodel);
}
static PyObject* state_index_calobject_warp(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(state_index_calobject_warp,
                               self, args, kwargs,
                               NULL);
}

static int callback_num_states_calobject_warp(int i,
                                              int Ncameras_intrinsics,
                                              int Ncameras_extrinsics,
                                              int Nframes,
                                              int Npoints,
                                              int Npoints_fixed,
                                              int Nobservations_board,
                                              int Nobservations_point,
                                              int calibration_object_width_n,
                                              int calibration_object_height_n,
                                              const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                              const mrcal_lensmodel_t* lensmodel,
                                              mrcal_problem_selections_t problem_selections)
{
    return
        mrcal_num_states_calobject_warp(problem_selections, Nobservations_board);
}
static PyObject* num_states_calobject_warp(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(num_states_calobject_warp,
                               self, args, kwargs,
                               NULL);
}

static int callback_num_states(int i,
                               int Ncameras_intrinsics,
                               int Ncameras_extrinsics,
                               int Nframes,
                               int Npoints,
                               int Npoints_fixed,
                               int Nobservations_board,
                               int Nobservations_point,
                               int calibration_object_width_n,
                               int calibration_object_height_n,
                               const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                               const mrcal_lensmodel_t* lensmodel,
                               mrcal_problem_selections_t problem_selections)
{
    return
        mrcal_num_states(Ncameras_intrinsics, Ncameras_extrinsics,
                         Nframes, Npoints, Npoints_fixed, Nobservations_board,
                         problem_selections,
                         lensmodel);
}
static PyObject* num_states(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(num_states,
                               self, args, kwargs,
                               NULL);
}

static int callback_num_intrinsics_optimization_params(int i,
                                                       int Ncameras_intrinsics,
                                                       int Ncameras_extrinsics,
                                                       int Nframes,
                                                       int Npoints,
                                                       int Npoints_fixed,
                                                       int Nobservations_board,
                                                       int Nobservations_point,
                                                       int calibration_object_width_n,
                                                       int calibration_object_height_n,
                                                       const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                                       const mrcal_lensmodel_t* lensmodel,
                                                       mrcal_problem_selections_t problem_selections)
{
    return
        mrcal_num_intrinsics_optimization_params(problem_selections,
                                                 lensmodel);
}
static PyObject* num_intrinsics_optimization_params(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(num_intrinsics_optimization_params,
                               self, args, kwargs,
                               NULL);
}

static int callback_measurement_index_boards(int i,
                                             int Ncameras_intrinsics,
                                             int Ncameras_extrinsics,
                                             int Nframes,
                                             int Npoints,
                                             int Npoints_fixed,
                                             int Nobservations_board,
                                             int Nobservations_point,
                                             int calibration_object_width_n,
                                             int calibration_object_height_n,
                                             const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                             const mrcal_lensmodel_t* lensmodel,
                                             mrcal_problem_selections_t problem_selections)
{
    return
        mrcal_measurement_index_boards(i,
                                       Nobservations_board,
                                       Nobservations_point,
                                       calibration_object_width_n,
                                       calibration_object_height_n);
}
static PyObject* measurement_index_boards(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(measurement_index_boards,
                               self, args, kwargs,
                               "i_observation_board");
}

static int callback_num_measurements_boards(int i,
                                            int Ncameras_intrinsics,
                                            int Ncameras_extrinsics,
                                            int Nframes,
                                            int Npoints,
                                            int Npoints_fixed,
                                            int Nobservations_board,
                                            int Nobservations_point,
                                            int calibration_object_width_n,
                                            int calibration_object_height_n,
                                            const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                            const mrcal_lensmodel_t* lensmodel,
                                            mrcal_problem_selections_t problem_selections)
{
    return
        mrcal_num_measurements_boards(Nobservations_board,
                                      calibration_object_width_n,
                                      calibration_object_height_n);
}
static PyObject* num_measurements_boards(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(num_measurements_boards,
                               self, args, kwargs,
                               NULL);
}

static int callback_measurement_index_points(int i,
                                             int Ncameras_intrinsics,
                                             int Ncameras_extrinsics,
                                             int Nframes,
                                             int Npoints,
                                             int Npoints_fixed,
                                             int Nobservations_board,
                                             int Nobservations_point,
                                             int calibration_object_width_n,
                                             int calibration_object_height_n,
                                             const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                             const mrcal_lensmodel_t* lensmodel,
                                             mrcal_problem_selections_t problem_selections)
{
    return
        mrcal_measurement_index_points(i,
                                       Nobservations_board,
                                       Nobservations_point,
                                       calibration_object_width_n,
                                       calibration_object_height_n);
}
static PyObject* measurement_index_points(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(measurement_index_points,
                               self, args, kwargs,
                               "i_observation_point");
}

static int callback_measurement_index_points_triangulated(int i,
                                             int Ncameras_intrinsics,
                                             int Ncameras_extrinsics,
                                             int Nframes,
                                             int Npoints,
                                             int Npoints_fixed,
                                             int Nobservations_board,
                                             int Nobservations_point,
                                             int calibration_object_width_n,
                                             int calibration_object_height_n,
                                             const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                             const mrcal_lensmodel_t* lensmodel,
                                             mrcal_problem_selections_t problem_selections)
{
#warning "triangulated-solve: finish this thing. Add mrcal_measurement_index_points_triangulated() and wrap it here"
    return -1;
        // mrcal_measurement_index_points_triangulated(i,
        //                                Nobservations_board,
        //                                Nobservations_point,
        //                                calibration_object_width_n,
        //                                calibration_object_height_n);
}
static PyObject* measurement_index_points_triangulated(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(measurement_index_points_triangulated,
                               self, args, kwargs,
                               "i_observation_point");
}

static int callback_num_measurements_points(int i,
                                            int Ncameras_intrinsics,
                                            int Ncameras_extrinsics,
                                            int Nframes,
                                            int Npoints,
                                            int Npoints_fixed,
                                            int Nobservations_board,
                                            int Nobservations_point,
                                            int calibration_object_width_n,
                                            int calibration_object_height_n,
                                            const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                            const mrcal_lensmodel_t* lensmodel,
                                            mrcal_problem_selections_t problem_selections)
{
    return
        mrcal_num_measurements_points(Nobservations_point);
}
static PyObject* num_measurements_points(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(num_measurements_points,
                               self, args, kwargs,
                               NULL);
}

static int callback_num_measurements_points_triangulated(int i,
                                                         int Ncameras_intrinsics,
                                                         int Ncameras_extrinsics,
                                                         int Nframes,
                                                         int Npoints,
                                                         int Npoints_fixed,
                                                         int Nobservations_board,
                                                         int Nobservations_point,
                                                         int calibration_object_width_n,
                                                         int calibration_object_height_n,
                                                         const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                                         const mrcal_lensmodel_t* lensmodel,
                                                         mrcal_problem_selections_t problem_selections)
{
    if(indices_point_triangulated_camintrinsics_camextrinsics == NULL)
        return -1;
    int N = PyArray_DIM(indices_point_triangulated_camintrinsics_camextrinsics, 0);
    if(N < 0)
        return -1;

    mrcal_observation_point_triangulated_t c_observations_point_triangulated[N];

    int Nobservations_point_triangulated =
        fill_c_observations_point_triangulated(c_observations_point_triangulated,
                                               NULL,
                                               indices_point_triangulated_camintrinsics_camextrinsics);
    if(Nobservations_point_triangulated < 0)
        return -1;

    return
        mrcal_num_measurements_points_triangulated(c_observations_point_triangulated,
                                                   Nobservations_point_triangulated);
}
static PyObject* num_measurements_points_triangulated(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(num_measurements_points_triangulated,
                               self, args, kwargs,
                               NULL);
}

static int callback_measurement_index_regularization(int i,
                                                     int Ncameras_intrinsics,
                                                     int Ncameras_extrinsics,
                                                     int Nframes,
                                                     int Npoints,
                                                     int Npoints_fixed,
                                                     int Nobservations_board,
                                                     int Nobservations_point,
                                                     int calibration_object_width_n,
                                                     int calibration_object_height_n,
                                                     const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                                     const mrcal_lensmodel_t* lensmodel,
                                                     mrcal_problem_selections_t problem_selections)
{
    return
        mrcal_measurement_index_regularization(Nobservations_board,
                                               Nobservations_point,
                                               calibration_object_width_n,
                                               calibration_object_height_n);
}
static PyObject* measurement_index_regularization(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(measurement_index_regularization,
                               self, args, kwargs,
                               NULL);
}

static int callback_num_measurements_regularization(int i,
                                                    int Ncameras_intrinsics,
                                                    int Ncameras_extrinsics,
                                                    int Nframes,
                                                    int Npoints,
                                                    int Npoints_fixed,
                                                    int Nobservations_board,
                                                    int Nobservations_point,
                                                    int calibration_object_width_n,
                                                    int calibration_object_height_n,
                                                    const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                                    const mrcal_lensmodel_t* lensmodel,
                                                    mrcal_problem_selections_t problem_selections)
{
    return
        mrcal_num_measurements_regularization(Ncameras_intrinsics, Ncameras_extrinsics,
                                              Nframes,
                                              Npoints, Npoints_fixed, Nobservations_board,
                                              problem_selections,
                                              lensmodel);
}
static PyObject* num_measurements_regularization(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(num_measurements_regularization,
                               self, args, kwargs,
                               NULL);
}


static int callback_num_measurements(int i,
                                     int Ncameras_intrinsics,
                                     int Ncameras_extrinsics,
                                     int Nframes,
                                     int Npoints,
                                     int Npoints_fixed,
                                     int Nobservations_board,
                                     int Nobservations_point,
                                     int calibration_object_width_n,
                                     int calibration_object_height_n,
                                     const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                     const mrcal_lensmodel_t* lensmodel,
                                     mrcal_problem_selections_t problem_selections)
{
#warning "triangulated-solve: add tests to the num_measurements_..., state_index_... ..."

    if(indices_point_triangulated_camintrinsics_camextrinsics == NULL)
        return -1;
    int N = PyArray_DIM(indices_point_triangulated_camintrinsics_camextrinsics, 0);
    if(N < 0)
        return -1;

    mrcal_observation_point_triangulated_t c_observations_point_triangulated[N];

    int Nobservations_point_triangulated =
        fill_c_observations_point_triangulated(c_observations_point_triangulated,
                                               NULL,
                                               indices_point_triangulated_camintrinsics_camextrinsics);
    if(Nobservations_point_triangulated < 0)
        return -1;

    return
        mrcal_num_measurements(Nobservations_board,
                               Nobservations_point,
                               c_observations_point_triangulated,
                               Nobservations_point_triangulated,
                               calibration_object_width_n,
                               calibration_object_height_n,
                               Ncameras_intrinsics, Ncameras_extrinsics,
                               Nframes,
                               Npoints, Npoints_fixed,
                               problem_selections,
                               lensmodel);
}
static PyObject* num_measurements(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(num_measurements,
                               self, args, kwargs,
                               NULL);
}



static PyObject* _pack_unpack_state(PyObject* self, PyObject* args, PyObject* kwargs,
                                    bool pack)
{
    // This is VERY similar to INDEX_GENERIC(). Please consolidate
    PyObject*      result = NULL;
    PyArrayObject* p      = NULL;

    OPTIMIZE_ARGUMENTS_REQUIRED(ARG_DEFINE);
    OPTIMIZE_ARGUMENTS_OPTIONAL(ARG_DEFINE);

    int Ncameras_intrinsics = -1;
    int Ncameras_extrinsics = -1;
    int Nframes             = -1;
    int Npoints             = -1;
    int Nobservations_board  = -1;
    int Nobservations_point  = -1;

    char* keywords[] = { "p",
                         OPTIMIZE_ARGUMENTS_REQUIRED(NAMELIST)

                         "Ncameras_intrinsics",
                         "Ncameras_extrinsics",
                         "Nframes",
                         "Npoints",
                         "Nobservations_board",
                         "Nobservations_point",
                         OPTIMIZE_ARGUMENTS_OPTIONAL(NAMELIST)
                         NULL};

#define UNPACK_STATE "unpack_state"
    char arg_string[] =
        "O&"
        "|" // everything is optional. I apply
            // logic down the line to get what
            // I need
        OPTIMIZE_ARGUMENTS_REQUIRED(PARSECODE)
        "iiiiii"
        OPTIMIZE_ARGUMENTS_OPTIONAL(PARSECODE)
        ":mrcal." UNPACK_STATE;
    if(pack)
    {
        arg_string[strlen(arg_string) - strlen(UNPACK_STATE)] = '\0';
        strcat(arg_string, "pack_state");
    }
#undef UNPACK_STATE

    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     arg_string,

                                     keywords,

                                     PyArray_Converter, &p,
                                     OPTIMIZE_ARGUMENTS_REQUIRED(PARSEARG)
                                     &Ncameras_intrinsics,
                                     &Ncameras_extrinsics,
                                     &Nframes,
                                     &Npoints,
                                     &Nobservations_board,
                                     &Nobservations_point,
                                     OPTIMIZE_ARGUMENTS_OPTIONAL(PARSEARG) NULL))
        goto done;

    if(lensmodel == NULL)
    {
        BARF("The 'lensmodel' argument is required");
        goto done;
    }

    const mrcal_problem_selections_t problem_selections =
        { .do_optimize_intrinsics_core       = do_optimize_intrinsics_core,
          .do_optimize_intrinsics_distortions= do_optimize_intrinsics_distortions,
          .do_optimize_extrinsics            = do_optimize_extrinsics,
          .do_optimize_frames                = do_optimize_frames,
          .do_optimize_calobject_warp        = do_optimize_calobject_warp,
          .do_apply_regularization           = do_apply_regularization
        };

    mrcal_lensmodel_t mrcal_lensmodel;
    if(!parse_lensmodel_from_arg(&mrcal_lensmodel, lensmodel))
        goto done;

    // checks dimensionality of array !IS_NULL. So if any array isn't passed-in,
    // that's OK! After I do this and if !IS_NULL, then I can ask for array
    // dimensions safely
    bool check(void)
    {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-to-pointer-cast"
        OPTIMIZE_ARGUMENTS_REQUIRED(CHECK_LAYOUT);
        OPTIMIZE_ARGUMENTS_OPTIONAL(CHECK_LAYOUT);
#pragma GCC diagnostic pop
        return true;
    }
    if(!check()) goto done;

    // If explicit dimensions are given, use them. If they're not given, but we
    // have an array, use those dimensions. If an array isn't given either, use
    // 0
    if(Ncameras_intrinsics < 0) Ncameras_intrinsics = IS_NULL(intrinsics)            ? 0 : PyArray_DIMS(intrinsics)            [0];
    if(Ncameras_extrinsics < 0) Ncameras_extrinsics = IS_NULL(extrinsics_rt_fromref) ? 0 : PyArray_DIMS(extrinsics_rt_fromref) [0];
    if(Nframes < 0)             Nframes             = IS_NULL(frames_rt_toref)       ? 0 : PyArray_DIMS(frames_rt_toref)       [0];
    if(Npoints < 0)             Npoints             = IS_NULL(points)                ? 0 : PyArray_DIMS(points)                [0];
    if(Nobservations_board < 0) Nobservations_board = IS_NULL(observations_board)    ? 0 : PyArray_DIMS(observations_board)    [0];
    if(Nobservations_point < 0) Nobservations_point = IS_NULL(observations_point)    ? 0 : PyArray_DIMS(observations_point)    [0];


    if( PyArray_TYPE(p) != NPY_DOUBLE )
    {
        BARF("The given array MUST have values of type 'float'");
        goto done;
    }

    if( !PyArray_IS_C_CONTIGUOUS(p) )
    {
        BARF("The given array MUST be a C-style contiguous array");
        goto done;
    }

    int       ndim = PyArray_NDIM(p);
    npy_intp* dims = PyArray_DIMS(p);
    if( ndim < 1 )
    {
        BARF("The given array MUST have at least one dimension");
        goto done;
    }

    int Nstate =
        mrcal_num_states(Ncameras_intrinsics, Ncameras_extrinsics,
                         Nframes, Npoints, Npoints_fixed, Nobservations_board,
                         problem_selections,
                         &mrcal_lensmodel);

    if( dims[ndim-1] != Nstate )
    {
        BARF("The given array MUST have last dimension of size Nstate=%d; instead got %ld",
             Nstate, dims[ndim-1]);
        goto done;
    }

    double* x = (double*)PyArray_DATA(p);
    if(pack)
        for(int i=0; i<PyArray_SIZE(p)/Nstate; i++)
        {
            mrcal_pack_solver_state_vector( x,
                                            Ncameras_intrinsics, Ncameras_extrinsics,
                                            Nframes, Npoints, Npoints_fixed,
                                            problem_selections, &mrcal_lensmodel);
            x = &x[Nstate];
        }
    else
        for(int i=0; i<PyArray_SIZE(p)/Nstate; i++)
        {
            mrcal_unpack_solver_state_vector( x,
                                              Ncameras_intrinsics, Ncameras_extrinsics,
                                              Nframes, Npoints, Npoints_fixed,
                                              problem_selections, &mrcal_lensmodel);
            x = &x[Nstate];
        }

    Py_INCREF(Py_None);
    result = Py_None;

 done:
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-to-pointer-cast"
    OPTIMIZE_ARGUMENTS_REQUIRED(FREE_PYARRAY) ;
    OPTIMIZE_ARGUMENTS_OPTIONAL(FREE_PYARRAY) ;
#pragma GCC diagnostic pop

    Py_XDECREF(p);
    return result;
}
static PyObject* pack_state(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return _pack_unpack_state(self, args, kwargs, true);
}
static PyObject* unpack_state(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return _pack_unpack_state(self, args, kwargs, false);
}

static PyObject* corresponding_icam_extrinsics(PyObject* self, PyObject* args, PyObject* kwargs)
{
    // This is VERY similar to INDEX_GENERIC(). Please consolidate
    PyObject* result          = NULL;
    int       icam_intrinsics = -1;

    OPTIMIZE_ARGUMENTS_REQUIRED(ARG_DEFINE);
    OPTIMIZE_ARGUMENTS_OPTIONAL(ARG_DEFINE);

    int Ncameras_intrinsics = -1;
    int Ncameras_extrinsics = -1;
    int Nobservations_board  = -1;
    int Nobservations_point  = -1;

    char* keywords[] = { "icam_intrinsics",
                         OPTIMIZE_ARGUMENTS_REQUIRED(NAMELIST)

                         "Ncameras_intrinsics",
                         "Ncameras_extrinsics",
                         "Nobservations_board",
                         "Nobservations_point",
                         OPTIMIZE_ARGUMENTS_OPTIONAL(NAMELIST)
                         NULL};

    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     "i"
                                     "|" // everything is optional. I apply
                                     // logic down the line to get what
                                     // I need
                                     OPTIMIZE_ARGUMENTS_REQUIRED(PARSECODE)
                                     "iiii"
                                     OPTIMIZE_ARGUMENTS_OPTIONAL(PARSECODE)
                                     ":mrcal.corresponding_icam_extrinsics",

                                     keywords,

                                     &icam_intrinsics,
                                     OPTIMIZE_ARGUMENTS_REQUIRED(PARSEARG)
                                     &Ncameras_intrinsics,
                                     &Ncameras_extrinsics,
                                     &Nobservations_board,
                                     &Nobservations_point,
                                     OPTIMIZE_ARGUMENTS_OPTIONAL(PARSEARG) NULL))
        goto done;

    // checks dimensionality of array !IS_NULL. So if any array isn't passed-in,
    // that's OK! After I do this and if !IS_NULL, then I can ask for array
    // dimensions safely
    bool check(void)
    {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-to-pointer-cast"
        OPTIMIZE_ARGUMENTS_REQUIRED(CHECK_LAYOUT);
        OPTIMIZE_ARGUMENTS_OPTIONAL(CHECK_LAYOUT);
#pragma GCC diagnostic pop
        return true;
    }
    if(!check()) goto done;

    // If explicit dimensions are given, use them. If they're not given, but we
    // have an array, use those dimensions. If an array isn't given either, use
    // 0
    if(Ncameras_intrinsics < 0) Ncameras_intrinsics = IS_NULL(intrinsics)            ? 0 : PyArray_DIMS(intrinsics)            [0];
    if(Ncameras_extrinsics < 0) Ncameras_extrinsics = IS_NULL(extrinsics_rt_fromref) ? 0 : PyArray_DIMS(extrinsics_rt_fromref) [0];
    if(Nobservations_board < 0) Nobservations_board = IS_NULL(observations_board)    ? 0 : PyArray_DIMS(observations_board)    [0];
    if(Nobservations_point < 0) Nobservations_point = IS_NULL(observations_point)    ? 0 : PyArray_DIMS(observations_point)    [0];


    if( icam_intrinsics < 0 || icam_intrinsics >= Ncameras_intrinsics )
    {
        BARF("The given icam_intrinsics=%d is out of bounds. Must be >= 0 and < %d",
             icam_intrinsics, Ncameras_intrinsics);
        goto done;
    }



    int icam_extrinsics;
    {
        mrcal_observation_board_t c_observations_board[Nobservations_board];
        fill_c_observations_board(c_observations_board,
                                  Nobservations_board,
                                  indices_frame_camintrinsics_camextrinsics);

        mrcal_observation_point_t c_observations_point[Nobservations_point];
        fill_c_observations_point(c_observations_point,
                                  Nobservations_point,
                                  indices_point_camintrinsics_camextrinsics,
                                  (mrcal_point3_t*)PyArray_DATA(observations_point));

        if(!mrcal_corresponding_icam_extrinsics(&icam_extrinsics,

                                                icam_intrinsics,
                                                Ncameras_intrinsics,
                                                Ncameras_extrinsics,
                                                Nobservations_board,
                                                c_observations_board,
                                                Nobservations_point,
                                                c_observations_point))
        {
            BARF("Error calling mrcal_corresponding_icam_extrinsics()");
            goto done;
        }
    }

    result = PyLong_FromLong(icam_extrinsics);

 done:
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-to-pointer-cast"
    OPTIMIZE_ARGUMENTS_REQUIRED(FREE_PYARRAY) ;
    OPTIMIZE_ARGUMENTS_OPTIONAL(FREE_PYARRAY) ;
#pragma GCC diagnostic pop

    return result;
}

static const char state_index_intrinsics_docstring[] =
#include "state_index_intrinsics.docstring.h"
    ;
static const char state_index_extrinsics_docstring[] =
#include "state_index_extrinsics.docstring.h"
    ;
static const char state_index_frames_docstring[] =
#include "state_index_frames.docstring.h"
    ;
static const char state_index_points_docstring[] =
#include "state_index_points.docstring.h"
    ;
static const char state_index_calobject_warp_docstring[] =
#include "state_index_calobject_warp.docstring.h"
    ;
static const char num_states_intrinsics_docstring[] =
#include "num_states_intrinsics.docstring.h"
    ;
static const char num_states_extrinsics_docstring[] =
#include "num_states_extrinsics.docstring.h"
    ;
static const char num_states_frames_docstring[] =
#include "num_states_frames.docstring.h"
    ;
static const char num_states_points_docstring[] =
#include "num_states_points.docstring.h"
    ;
static const char num_states_calobject_warp_docstring[] =
#include "num_states_calobject_warp.docstring.h"
    ;
static const char pack_state_docstring[] =
#include "pack_state.docstring.h"
    ;
static const char unpack_state_docstring[] =
#include "unpack_state.docstring.h"
    ;
static const char measurement_index_boards_docstring[] =
#include "measurement_index_boards.docstring.h"
    ;
static const char measurement_index_points_docstring[] =
#include "measurement_index_points.docstring.h"
    ;
static const char measurement_index_points_triangulated_docstring[] =
#include "measurement_index_points_triangulated.docstring.h"
    ;
static const char measurement_index_regularization_docstring[] =
#include "measurement_index_regularization.docstring.h"
    ;
static const char num_measurements_boards_docstring[] =
#include "num_measurements_boards.docstring.h"
    ;
static const char num_measurements_points_docstring[] =
#include "num_measurements_points.docstring.h"
    ;
static const char num_measurements_points_triangulated_docstring[] =
#include "num_measurements_points_triangulated.docstring.h"
    ;
static const char num_measurements_regularization_docstring[] =
#include "num_measurements_regularization.docstring.h"
    ;
static const char num_measurements_docstring[] =
#include "num_measurements.docstring.h"
    ;
static const char corresponding_icam_extrinsics_docstring[] =
#include "corresponding_icam_extrinsics.docstring.h"
    ;

static const char num_states_docstring[] =
#include "num_states.docstring.h"
    ;
static const char num_intrinsics_optimization_params_docstring[] =
#include "num_intrinsics_optimization_params.docstring.h"
    ;






static const char optimize_docstring[] =
#include "optimize.docstring.h"
    ;
static const char optimizer_callback_docstring[] =
#include "optimizer_callback.docstring.h"
    ;
static const char lensmodel_metadata_and_config_docstring[] =
#include "lensmodel_metadata_and_config.docstring.h"
    ;
static const char lensmodel_num_params_docstring[] =
#include "lensmodel_num_params.docstring.h"
    ;
static const char supported_lensmodels_docstring[] =
#include "supported_lensmodels.docstring.h"
    ;
static const char knots_for_splined_models_docstring[] =
#include "knots_for_splined_models.docstring.h"
    ;
static PyMethodDef methods[] =
    { PYMETHODDEF_ENTRY(,optimize,                         METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(,optimizer_callback,               METH_VARARGS | METH_KEYWORDS),

      PYMETHODDEF_ENTRY(, state_index_intrinsics,          METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(, state_index_extrinsics,          METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(, state_index_frames,              METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(, state_index_points,              METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(, state_index_calobject_warp,      METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(, num_states_intrinsics,           METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(, num_states_extrinsics,           METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(, num_states_frames,               METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(, num_states_points,               METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(, num_states_calobject_warp,       METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(, num_states,                      METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(, num_intrinsics_optimization_params,METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(, pack_state,                      METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(, unpack_state,                    METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(, measurement_index_boards,        METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(, measurement_index_points,        METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(, measurement_index_points_triangulated,METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(, measurement_index_regularization,METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(, num_measurements_boards,         METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(, num_measurements_points,         METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(, num_measurements_points_triangulated,METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(, num_measurements_regularization, METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(, num_measurements,                METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(, corresponding_icam_extrinsics,METH_VARARGS | METH_KEYWORDS),

      PYMETHODDEF_ENTRY(,lensmodel_metadata_and_config,METH_VARARGS),
      PYMETHODDEF_ENTRY(,lensmodel_num_params,         METH_VARARGS),
      PYMETHODDEF_ENTRY(,supported_lensmodels,         METH_NOARGS),
      PYMETHODDEF_ENTRY(,knots_for_splined_models,     METH_VARARGS),
      {}
    };


static void _init_mrcal_common(PyObject* module)
{
    Py_INCREF(&CHOLMOD_factorization_type);
    PyModule_AddObject(module, "CHOLMOD_factorization", (PyObject *)&CHOLMOD_factorization_type);

}


#define MODULE_DOCSTRING                                                \
    "Low-level routines for core mrcal operations\n"                    \
    "\n"                                                                \
    "This is the written-in-C Python extension module that underlies the routines in\n" \
    "mrcal.h. Most of the functions in this module (those prefixed with \"_\") are\n" \
    "not meant to be called directly, but have Python wrappers that should be used\n" \
    "instead.\n"                                                        \
    "\n"                                                                \
    "All functions are exported into the mrcal module. So you can call these via\n" \
    "mrcal._mrcal.fff() or mrcal.fff(). The latter is preferred.\n"

#if PY_MAJOR_VERSION == 2

PyMODINIT_FUNC init_mrcal(void)
{
    if (PyType_Ready(&CHOLMOD_factorization_type) < 0)
        return;

    PyObject* module =
        Py_InitModule3("_mrcal", methods,
                       MODULE_DOCSTRING);
    _init_mrcal_common(module);
    import_array();
}

#else

static struct PyModuleDef module_def =
    {
     PyModuleDef_HEAD_INIT,
     "_mrcal",
     MODULE_DOCSTRING,
     -1,
     methods
    };

PyMODINIT_FUNC PyInit__mrcal(void)
{
    if (PyType_Ready(&CHOLMOD_factorization_type) < 0)
        return NULL;

    PyObject* module =
        PyModule_Create(&module_def);

    _init_mrcal_common(module);
    import_array();

    return module;
}

#endif
