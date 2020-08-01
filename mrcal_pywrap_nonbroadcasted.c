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
#define IS_TRUE(x) ((x) != NULL && PyObject_IsTrue(x))

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
#define COMMA_LENSMODEL_NAME(s,n) , mrcal_lensmodel_name( (lensmodel_t){.type = s} )
#define VALID_LENSMODELS_FORMAT  "(" LENSMODEL_LIST(PERCENT_S_COMMA) ")"
#define VALID_LENSMODELS_ARGLIST LENSMODEL_LIST(COMMA_LENSMODEL_NAME)

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


static PyObject* csr_from_cholmod_sparse( cholmod_sparse* Jt,

                                          // These are allowed to be NULL; If
                                          // so, I'll use the data in Jt. THE Jt
                                          // STILL OWNS THE DATA
                                          PyObject* P,
                                          PyObject* I,
                                          PyObject* X)
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
    if(P == NULL) P = PyArray_SimpleNewFromData(1, ((npy_intp[]){Jt->ncol + 1}), NPY_INT32,  Jt->p);
    else Py_INCREF(P);

    if(I == NULL) I = PyArray_SimpleNewFromData(1, ((npy_intp[]){Jt->nzmax   }), NPY_INT32,  Jt->i);
    else Py_INCREF(I);

    if(X == NULL) X = PyArray_SimpleNewFromData(1, ((npy_intp[]){Jt->nzmax   }), NPY_DOUBLE, Jt->x);
    else Py_INCREF(X);

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

    // optimizerCallback should return it
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
                                     "|O", keywords, &Py_J))
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
CHOLMOD_factorization_solve_JtJ_x_b(CHOLMOD_factorization* self, PyObject* args, PyObject* kwargs)
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
                                     "O", keywords, &Py_bt))
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
static const char CHOLMOD_factorization_solve_JtJ_x_b_docstring[] =
#include "CHOLMOD_factorization_solve_JtJ_x_b.docstring.h"
    ;

static PyMethodDef CHOLMOD_factorization_methods[] =
    {
        PYMETHODDEF_ENTRY(CHOLMOD_factorization_, solve_JtJ_x_b, METH_VARARGS | METH_KEYWORDS),
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
                                     lensmodel_t* lensmodel,
                                     // input
                                     PyObject* lensmodel_string)
{
    const char* lensmodel_cstring = PyString_AsString(lensmodel_string);
    if( lensmodel_cstring == NULL)
    {
        BARF("The lens model must be given as a string");
        return false;
    }

    *lensmodel = mrcal_lensmodel_from_name(lensmodel_cstring);
    if( !mrcal_lensmodel_type_is_valid(lensmodel->type) )
    {
        if(lensmodel->type == LENSMODEL_INVALID_BADCONFIG)
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

static PyObject* getLensModelMeta(PyObject* NPY_UNUSED(self),
                                  PyObject* args)
{
    PyObject* result = NULL;
    SET_SIGINT();

    PyObject* lensmodel_string = NULL;
    if(!PyArg_ParseTuple( args, STRING_OBJECT, &lensmodel_string ))
        goto done;
    lensmodel_t lensmodel;
    if(!parse_lensmodel_from_arg(&lensmodel, lensmodel_string))
        goto done;

    mrcal_lensmodel_meta_t meta = mrcal_lensmodel_meta(lensmodel);

#define MRCAL_ITEM_BUILDVALUE_DEF(  name, type, pybuildvaluecode, PRIcode,SCNcode, bitfield, cookie) " s "pybuildvaluecode
#define MRCAL_ITEM_BUILDVALUE_VALUE(name, type, pybuildvaluecode, PRIcode,SCNcode, bitfield, cookie) , #name, cookie name

    if(lensmodel.type == LENSMODEL_SPLINED_STEREOGRAPHIC )
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

static PyObject* getKnotsForSplinedModels(PyObject* NPY_UNUSED(self),
                                          PyObject* args)
{
    PyObject*      result = NULL;
    PyArrayObject* py_ux  = NULL;
    PyArrayObject* py_uy  = NULL;
    SET_SIGINT();

    PyObject* lensmodel_string = NULL;
    if(!PyArg_ParseTuple( args, STRING_OBJECT, &lensmodel_string ))
        goto done;
    lensmodel_t lensmodel;
    if(!parse_lensmodel_from_arg(&lensmodel, lensmodel_string))
        goto done;

    if(lensmodel.type != LENSMODEL_SPLINED_STEREOGRAPHIC)
    {
        BARF( "This function works only with the LENSMODEL_SPLINED_STEREOGRAPHIC model. %S passed in",
              lensmodel_string);
        goto done;
    }

    {
        double ux[lensmodel.LENSMODEL_SPLINED_STEREOGRAPHIC__config.Nx];
        double uy[lensmodel.LENSMODEL_SPLINED_STEREOGRAPHIC__config.Ny];
        if(!mrcal_get_knots_for_splined_models(ux,uy, lensmodel))
        {
            BARF( "mrcal_get_knots_for_splined_models() failed");
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

static PyObject* getNlensParams(PyObject* NPY_UNUSED(self),
                                PyObject* args)
{
    PyObject* result = NULL;
    SET_SIGINT();

    PyObject* lensmodel_string = NULL;
    if(!PyArg_ParseTuple( args, STRING_OBJECT, &lensmodel_string ))
        goto done;
    lensmodel_t lensmodel;
    if(!parse_lensmodel_from_arg(&lensmodel, lensmodel_string))
        goto done;

    int Nparams = mrcal_getNlensParams(lensmodel);

    result = Py_BuildValue("i", Nparams);

 done:
    RESET_SIGINT();
    return result;
}

static PyObject* getSupportedLensModels(PyObject* NPY_UNUSED(self),
                                        PyObject* NPY_UNUSED(args))
{
    PyObject* result = NULL;
    SET_SIGINT();
    const char* const* names = mrcal_getSupportedLensModels();

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


// project_stereographic(), and unproject_stereographic() have very similar
// arguments and operation, so the logic is consolidated as much as possible in
// these functions. The first arg is called "points" in both cases, but is 2d in
// one case, and 3d in the other
#define UN_PROJECT_STEREOGRAPHIC_ARGUMENTS_REQUIRED(_)                                   \
    _(points,     PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, points,     NPY_DOUBLE, {} )
#define UN_PROJECT_STEREOGRAPHIC_ARGUMENTS_OPTIONAL(_)                                   \
    _(fx,                  double,       1.0,    "d",  , NULL, -1, {} ) \
    _(fy,                  double,       1.0,    "d",  , NULL, -1, {} ) \
    _(cx,                  double,       0.0,    "d",  , NULL, -1, {} ) \
    _(cy,                  double,       0.0,    "d",  , NULL, -1, {} ) \
    _(get_gradients,    PyObject*,  Py_False,    "O",  , NULL, -1, {} )

static bool _un_project_stereographic_validate_args(// in
                                                    int dim_points_in, // 3 for project(), 2 for unproject()
                                                    UN_PROJECT_STEREOGRAPHIC_ARGUMENTS_REQUIRED(ARG_LIST_DEFINE)
                                                    UN_PROJECT_STEREOGRAPHIC_ARGUMENTS_OPTIONAL(ARG_LIST_DEFINE)
                                                    void* dummy __attribute__((unused)))
{
    if( PyArray_NDIM(points) < 1 )
    {
        BARF("'points' must have ndims >= 1");
        return false;
    }
    if( dim_points_in != PyArray_DIMS(points)[ PyArray_NDIM(points)-1 ] )
    {
        BARF("points.shape[-1] MUST be %d. Instead got %ld",
                     dim_points_in,
                     PyArray_DIMS(points)[PyArray_NDIM(points)-1] );
        return false;
    }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-to-pointer-cast"
    UN_PROJECT_STEREOGRAPHIC_ARGUMENTS_REQUIRED(CHECK_LAYOUT);
    UN_PROJECT_STEREOGRAPHIC_ARGUMENTS_OPTIONAL(CHECK_LAYOUT);
#pragma GCC diagnostic pop

    return true;
}

static PyObject* _un_project_stereographic(PyObject* NPY_UNUSED(self),
                                           PyObject* args,
                                           PyObject* kwargs,
                                           bool projecting)
{
    int dim_points_in, dim_points_out;
    if(projecting)
    {
        dim_points_in  = 3;
        dim_points_out = 2;
    }
    else
    {
        dim_points_in  = 2;
        dim_points_out = 3;
    }

    PyObject* result = NULL;

    SET_SIGINT();
    PyArrayObject* out  = NULL;
    PyArrayObject* grad = NULL;

    UN_PROJECT_STEREOGRAPHIC_ARGUMENTS_REQUIRED(ARG_DEFINE);
    UN_PROJECT_STEREOGRAPHIC_ARGUMENTS_OPTIONAL(ARG_DEFINE);

    char* keywords[] = { UN_PROJECT_STEREOGRAPHIC_ARGUMENTS_REQUIRED(NAMELIST)
                         UN_PROJECT_STEREOGRAPHIC_ARGUMENTS_OPTIONAL(NAMELIST)
                         NULL};
    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     UN_PROJECT_STEREOGRAPHIC_ARGUMENTS_REQUIRED(PARSECODE) "|"
                                     UN_PROJECT_STEREOGRAPHIC_ARGUMENTS_OPTIONAL(PARSECODE),

                                     keywords,

                                     UN_PROJECT_STEREOGRAPHIC_ARGUMENTS_REQUIRED(PARSEARG)
                                     UN_PROJECT_STEREOGRAPHIC_ARGUMENTS_OPTIONAL(PARSEARG) NULL))
        goto done;

    /* if the input points array is degenerate, return a degenerate thing */
    if( IS_NULL(points) )
    {
        result = Py_None;
        Py_INCREF(result);
        goto done;
    }

    if(!_un_project_stereographic_validate_args( dim_points_in,
                                                 UN_PROJECT_STEREOGRAPHIC_ARGUMENTS_REQUIRED(ARG_LIST_CALL)
                                                 UN_PROJECT_STEREOGRAPHIC_ARGUMENTS_OPTIONAL(ARG_LIST_CALL)
                                                 NULL))
        goto done;

    /* poor man's broadcasting of the inputs. I compute the total number of */
    /* points by multiplying the extra broadcasted dimensions. And I set up the */
    /* outputs to have the appropriate broadcasted dimensions        */
    const npy_intp* leading_dims  = PyArray_DIMS(points);
    int             Nleading_dims = PyArray_NDIM(points)-1;
    int Npoints = PyArray_SIZE(points) / leading_dims[Nleading_dims];
    bool get_gradients_bool = IS_TRUE(get_gradients);

    {
        npy_intp dims[Nleading_dims+2]; /* one extra for the gradients */
        memcpy(dims, leading_dims, Nleading_dims*sizeof(dims[0]));

        dims[Nleading_dims + 0] = dim_points_out;
        out = (PyArrayObject*)PyArray_SimpleNew(Nleading_dims+1,
                                                dims,
                                                NPY_DOUBLE);
        if( get_gradients_bool )
        {
            dims[Nleading_dims + 0] = dim_points_out;
            dims[Nleading_dims + 1] = dim_points_in;
            grad = (PyArrayObject*)PyArray_SimpleNew(Nleading_dims+2,
                                                     dims,
                                                     NPY_DOUBLE);
        }
    }

    if(projecting)
        mrcal_project_stereographic((mrcal_point2_t*)PyArray_DATA(out),
                                    get_gradients_bool ? (mrcal_point3_t*)PyArray_DATA(grad)  : NULL,
                                    (const mrcal_point3_t*)PyArray_DATA(points),
                                    Npoints,
                                    fx,fy,cx,cy);
    else
        mrcal_unproject_stereographic((mrcal_point3_t*)PyArray_DATA(out),
                                      get_gradients_bool ? (mrcal_point2_t*)PyArray_DATA(grad) : NULL,
                                      (const mrcal_point2_t*)PyArray_DATA(points),
                                      Npoints,
                                      fx,fy,cx,cy);

    if( get_gradients_bool )
    {
        result = PyTuple_Pack(2, out, grad);
        Py_DECREF(out);
        Py_DECREF(grad);
    }
    else
        result = (PyObject*)out;

 done:
    UN_PROJECT_STEREOGRAPHIC_ARGUMENTS_REQUIRED(FREE_PYARRAY) ;
    UN_PROJECT_STEREOGRAPHIC_ARGUMENTS_OPTIONAL(FREE_PYARRAY) ;
    RESET_SIGINT();
    return result;
}

static PyObject* project_stereographic(PyObject* self,
                                       PyObject* args,
                                       PyObject* kwargs)
{
    return _un_project_stereographic(self, args, kwargs, true);
}
static PyObject* unproject_stereographic(PyObject* self,
                                         PyObject* args,
                                         PyObject* kwargs)
{
    return _un_project_stereographic(self, args, kwargs, false);
}



#define OPTIMIZERCALLBACK_ARGUMENTS_REQUIRED(_)                                  \
    _(intrinsics,                         PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, intrinsics,                  NPY_DOUBLE, {-1 COMMA -1       } ) \
    _(extrinsics_rt_fromref,              PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, extrinsics_rt_fromref,       NPY_DOUBLE, {-1 COMMA  6       } ) \
    _(frames_rt_toref,                    PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, frames_rt_toref,             NPY_DOUBLE, {-1 COMMA  6       } ) \
    _(points,                             PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, points,                      NPY_DOUBLE, {-1 COMMA  3       } ) \
    _(observations_board,                 PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, observations_board,          NPY_DOUBLE, {-1 COMMA -1 COMMA -1 COMMA 3 } ) \
    _(indices_frame_camintrinsics_camextrinsics,PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, indices_frame_camintrinsics_camextrinsics,  NPY_INT32,    {-1 COMMA  3       } ) \
    _(observations_point,                 PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, observations_point,          NPY_DOUBLE, {-1 COMMA  3       } ) \
    _(indices_point_camintrinsics_camextrinsics,PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, indices_point_camintrinsics_camextrinsics, NPY_INT32,    {-1 COMMA  3       } ) \
    _(lensmodel,                          PyObject*,      NULL,    STRING_OBJECT,  ,                        NULL,                        -1,         {}                   ) \
    _(imagersizes,                        PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, imagersizes,                 NPY_INT32,    {-1 COMMA 2        } )

#define OPTIMIZERCALLBACK_ARGUMENTS_OPTIONAL(_) \
    _(observed_pixel_uncertainty,         double,         -1.0,    "d",  ,                                  NULL,           -1,         {})  \
    _(calobject_warp,                     PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, calobject_warp, NPY_DOUBLE, {2}) \
    _(Npoints_fixed,                      int,            0,       "i",  ,                                  NULL,           -1,         {})  \
    _(do_optimize_intrinsic_core,         PyObject*,      Py_True, "O",  ,                                  NULL,           -1,         {})  \
    _(do_optimize_intrinsic_distortions,  PyObject*,      Py_True, "O",  ,                                  NULL,           -1,         {})  \
    _(do_optimize_extrinsics,             PyObject*,      Py_True, "O",  ,                                  NULL,           -1,         {})  \
    _(do_optimize_frames,                 PyObject*,      Py_True, "O",  ,                                  NULL,           -1,         {})  \
    _(do_optimize_calobject_warp,         PyObject*,      Py_False,"O",  ,                                  NULL,           -1,         {})  \
    _(calibration_object_spacing,         double,         -1.0,    "d",  ,                                  NULL,           -1,         {})  \
    _(calibration_object_width_n,         int,            -1,      "i",  ,                                  NULL,           -1,         {})  \
    _(calibration_object_height_n,        int,            -1,      "i",  ,                                  NULL,           -1,         {})  \
    _(point_min_range,                    double,         -1.0,    "d",  ,                                  NULL,           -1,         {})  \
    _(point_max_range,                    double,         -1.0,    "d",  ,                                  NULL,           -1,         {})  \
    _(verbose,                            PyObject*,      NULL,    "O",  ,                                  NULL,           -1,         {})  \
    _(skip_regularization,                PyObject*,      NULL,    "O",  ,                                  NULL,           -1,         {})

#define OPTIMIZERCALLBACK_ARGUMENTS_ALL(_) \
    OPTIMIZERCALLBACK_ARGUMENTS_REQUIRED(_) \
    OPTIMIZERCALLBACK_ARGUMENTS_OPTIONAL(_)

#define OPTIMIZE_ARGUMENTS_REQUIRED(_) OPTIMIZERCALLBACK_ARGUMENTS_REQUIRED(_)
#define OPTIMIZE_ARGUMENTS_OPTIONAL(_) OPTIMIZERCALLBACK_ARGUMENTS_OPTIONAL(_) \
    _(skip_outlier_rejection,             PyObject*,      NULL,    "O",  ,                                  NULL,           -1,         {})

#define OPTIMIZE_ARGUMENTS_ALL(_) \
    OPTIMIZE_ARGUMENTS_REQUIRED(_) \
    OPTIMIZE_ARGUMENTS_OPTIONAL(_)

// Using this for both optimize() and optimizerCallback()
static bool optimize_validate_args( // out
                                    lensmodel_t* lensmodel_type,

                                    // in
                                    OPTIMIZE_ARGUMENTS_REQUIRED(ARG_LIST_DEFINE)
                                    OPTIMIZE_ARGUMENTS_OPTIONAL(ARG_LIST_DEFINE)

                                    void* dummy __attribute__((unused)))
{
    if(PyObject_IsTrue(do_optimize_calobject_warp) &&
       IS_NULL(calobject_warp))
    {
        BARF("if(do_optimize_calobject_warp) then calobject_warp MUST be given as an array to seed the optimization and to receive the results");
        return false;
    }

    static_assert( sizeof(mrcal_pose_t)/sizeof(double) == 6, "mrcal_pose_t is assumed to contain 6 elements");

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-to-pointer-cast"
    OPTIMIZERCALLBACK_ARGUMENTS_ALL(CHECK_LAYOUT) ;
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

    long int NobservationsBoard = PyArray_DIMS(observations_board)[0];
    if( PyArray_DIMS(indices_frame_camintrinsics_camextrinsics)[0] != NobservationsBoard )
    {
        BARF("Inconsistent NobservationsBoard: 'observations_board' says %ld, 'indices_frame_camintrinsics_camextrinsics' says %ld",
                     NobservationsBoard,
                     PyArray_DIMS(indices_frame_camintrinsics_camextrinsics)[0]);
        return false;
    }

    // calibration_object_spacing and calibration_object_width_n and
    // calibration_object_height_n must be > 0 OR we have to not be using a
    // calibration board
    if( NobservationsBoard > 0 )
    {
        if( calibration_object_spacing <= 0.0 )
        {
            BARF("We have board observations, so calibration_object_spacing MUST be a valid float > 0");
            return false;
        }

        if( calibration_object_width_n <= 0 || calibration_object_height_n <= 0)
        {
            BARF("We have board observations, so calibration_object_width_n and calibration_object_height_n MUST both be a valid int > 0");
            return false;
        }

        if( calibration_object_height_n != PyArray_DIMS(observations_board)[1] ||
            calibration_object_width_n  != PyArray_DIMS(observations_board)[2] )
        {
            BARF("observations_board.shape MUST be (...,%d,%d,3). Instead got (%ld,%ld,%ld)",
                         calibration_object_height_n, calibration_object_width_n,
                         PyArray_DIMS(observations_board)[1],
                         PyArray_DIMS(observations_board)[2],
                         PyArray_DIMS(observations_board)[3]);
            return false;
        }
    }

    int NobservationsPoint = PyArray_DIMS(observations_point)[0];
    if( PyArray_DIMS(indices_point_camintrinsics_camextrinsics)[0] != NobservationsPoint )
    {
        BARF("Inconsistent NobservationsPoint: 'observations_point...' says %ld, 'indices_point_camintrinsics_camextrinsics' says %ld",
                     NobservationsPoint,
                     PyArray_DIMS(indices_point_camintrinsics_camextrinsics)[0]);
        return false;
    }

    if(!parse_lensmodel_from_arg(lensmodel_type, lensmodel))
        return false;

    int NlensParams = mrcal_getNlensParams(*lensmodel_type);
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
    int i_frame_last  = -1;
    int i_cam_intrinsics_last = -1;
    int i_cam_extrinsics_last = -1;
    for(int i_observation=0; i_observation<NobservationsBoard; i_observation++)
    {
        // check for monotonicity and in-rangeness
        int i_frame          = ((int*)PyArray_DATA(indices_frame_camintrinsics_camextrinsics))[i_observation*3 + 0];
        int i_cam_intrinsics = ((int*)PyArray_DATA(indices_frame_camintrinsics_camextrinsics))[i_observation*3 + 1];
        int i_cam_extrinsics = ((int*)PyArray_DATA(indices_frame_camintrinsics_camextrinsics))[i_observation*3 + 2];

        // First I make sure everything is in-range
        if(i_frame < 0 || i_frame >= Nframes)
        {
            BARF("i_frame MUST be in [0,%d], instead got %d in row %d of indices_frame_camintrinsics_camextrinsics",
                         Nframes-1, i_frame, i_observation);
            return false;
        }
        if(i_cam_intrinsics < 0 || i_cam_intrinsics >= Ncameras_intrinsics)
        {
            BARF("i_cam_intrinsics MUST be in [0,%d], instead got %d in row %d of indices_frame_camintrinsics_camextrinsics",
                         Ncameras_intrinsics-1, i_cam_intrinsics, i_observation);
            return false;
        }
        if(i_cam_extrinsics < -1 || i_cam_extrinsics >= Ncameras_extrinsics)
        {
            BARF("i_cam_extrinsics MUST be in [-1,%d], instead got %d in row %d of indices_frame_camintrinsics_camextrinsics",
                         Ncameras_extrinsics-1, i_cam_extrinsics, i_observation);
            return false;
        }
        // And then I check monotonicity
        if(i_frame == i_frame_last)
        {
            if( i_cam_intrinsics < i_cam_intrinsics_last )
            {
                BARF("i_cam_intrinsics MUST be monotonically increasing in indices_frame_camintrinsics_camextrinsics. Instead row %d (frame %d) of indices_frame_camintrinsics_camextrinsics has i_cam_intrinsics=%d after previously seeing i_cam_intrinsics=%d",
                             i_observation, i_frame, i_cam_intrinsics, i_cam_intrinsics_last);
                return false;
            }
            if( i_cam_extrinsics < i_cam_extrinsics_last )
            {
                BARF("i_cam_extrinsics MUST be monotonically increasing in indices_frame_camintrinsics_camextrinsics. Instead row %d (frame %d) of indices_frame_camintrinsics_camextrinsics has i_cam_extrinsics=%d after previously seeing i_cam_extrinsics=%d",
                             i_observation, i_frame, i_cam_extrinsics, i_cam_extrinsics_last);
                return false;
            }
        }
        else if( i_frame < i_frame_last )
        {
            BARF("i_frame MUST be monotonically increasing in indices_frame_camintrinsics_camextrinsics. Instead row %d of indices_frame_camintrinsics_camextrinsics has i_frame=%d after previously seeing i_frame=%d",
                         i_observation, i_frame, i_frame_last);
            return false;
        }

        i_frame_last          = i_frame;
        i_cam_intrinsics_last = i_cam_intrinsics;
        i_cam_extrinsics_last = i_cam_extrinsics;
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
    int i_point_last = -1;
    i_cam_intrinsics_last = -1;
    i_cam_extrinsics_last = -1;
    for(int i_observation=0; i_observation<NobservationsPoint; i_observation++)
    {
        int i_point          = ((int*)PyArray_DATA(indices_point_camintrinsics_camextrinsics))[i_observation*3 + 0];
        int i_cam_intrinsics = ((int*)PyArray_DATA(indices_point_camintrinsics_camextrinsics))[i_observation*3 + 1];
        int i_cam_extrinsics = ((int*)PyArray_DATA(indices_point_camintrinsics_camextrinsics))[i_observation*3 + 2];

        // First I make sure everything is in-range
        if(i_point < 0 || i_point >= Npoints)
        {
            BARF("i_point MUST be in [0,%d], instead got %d in row %d of indices_point_camintrinsics_camextrinsics",
                         Npoints-1, i_point, i_observation);
            return false;
        }
        if(i_cam_intrinsics < 0 || i_cam_intrinsics >= Ncameras_intrinsics)
        {
            BARF("i_cam_intrinsics MUST be in [0,%d], instead got %d in row %d of indices_point_camintrinsics_camextrinsics",
                         Ncameras_intrinsics-1, i_cam_intrinsics, i_observation);
            return false;
        }
        if(i_cam_extrinsics < -1 || i_cam_extrinsics >= Ncameras_extrinsics)
        {
            BARF("i_cam_extrinsics MUST be in [-1,%d], instead got %d in row %d of indices_point_camintrinsics_camextrinsics",
                         Ncameras_extrinsics-1, i_cam_extrinsics, i_observation);
            return false;
        }
        // And then I check monotonicity
        if( i_point < i_point_last )
        {
            BARF("i_point MUST be monotonically increasing in indices_point_camintrinsics_camextrinsics. Instead row %d of indices_point_camintrinsics_camextrinsics has i_point=%d after previously seeing i_point=%d",
                         i_observation, i_point, i_point_last);
            return false;
        }

        i_point_last          = i_point;
        i_cam_intrinsics_last = i_cam_intrinsics;
        i_cam_extrinsics_last = i_cam_extrinsics;
    }

    if( !(skip_outlier_rejection && PyObject_IsTrue(skip_outlier_rejection)))
    {
        // The pixel uncertainty is used and must be valid
        if( observed_pixel_uncertainty <= 0.0 )
        {
            BARF("observed_pixel_uncertainty MUST be a valid float > 0");
            return false;
        }
    }

    return true;
}

static void fill_c_observations_board(// out
                                      observation_board_t* c_observations_board,

                                      // in
                                      int NobservationsBoard,
                                      PyArrayObject* indices_frame_camintrinsics_camextrinsics)
{
    for(int i_observation=0; i_observation<NobservationsBoard; i_observation++)
    {
        int i_frame          = ((int*)PyArray_DATA(indices_frame_camintrinsics_camextrinsics))[i_observation*3 + 0];
        int i_cam_intrinsics = ((int*)PyArray_DATA(indices_frame_camintrinsics_camextrinsics))[i_observation*3 + 1];
        int i_cam_extrinsics = ((int*)PyArray_DATA(indices_frame_camintrinsics_camextrinsics))[i_observation*3 + 2];

        c_observations_board[i_observation].i_cam_intrinsics = i_cam_intrinsics;
        c_observations_board[i_observation].i_cam_extrinsics = i_cam_extrinsics;
        c_observations_board[i_observation].i_frame          = i_frame;
    }
}

static
PyObject* _optimize(bool is_optimize, // or optimizerCallback
                    PyObject* args,
                    PyObject* kwargs)
{
    PyObject* result = NULL;

    PyArrayObject* p_packed_final = NULL;
    PyArrayObject* x_final        = NULL;
    PyObject*      pystats        = NULL;

    SET_SIGINT();

    // define a superset of the variables: the ones used in optimize()
    OPTIMIZE_ARGUMENTS_ALL(ARG_DEFINE) ;

    if( is_optimize )
    {
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
    }
    else
    {
        char* keywords[] = { OPTIMIZERCALLBACK_ARGUMENTS_REQUIRED(NAMELIST)
                             OPTIMIZERCALLBACK_ARGUMENTS_OPTIONAL(NAMELIST)
                             NULL};
        if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                         OPTIMIZERCALLBACK_ARGUMENTS_REQUIRED(PARSECODE) "|"
                                         OPTIMIZERCALLBACK_ARGUMENTS_OPTIONAL(PARSECODE),

                                         keywords,

                                         OPTIMIZERCALLBACK_ARGUMENTS_REQUIRED(PARSEARG)
                                         OPTIMIZERCALLBACK_ARGUMENTS_OPTIONAL(PARSEARG) NULL))
            goto done;

        skip_outlier_rejection = Py_True;
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

    SET_SIZE0_IF_NONE(points,                     NPY_DOUBLE, 0,3);
    SET_SIZE0_IF_NONE(observations_point,         NPY_DOUBLE, 0,3);
    SET_SIZE0_IF_NONE(indices_point_camintrinsics_camextrinsics,NPY_INT32, 0,3);
    SET_SIZE0_IF_NONE(imagersizes,                NPY_INT32,    0,2);
#undef SET_NULL_IF_NONE




    lensmodel_t lensmodel_type;
    // Check the arguments for optimize(). If optimizerCallback, then the other
    // stuff is defined, but it all has valid, default values
    if( !optimize_validate_args(&lensmodel_type,
                                OPTIMIZE_ARGUMENTS_REQUIRED(ARG_LIST_CALL)
                                OPTIMIZE_ARGUMENTS_OPTIONAL(ARG_LIST_CALL)
                                NULL))
        goto done;

    {
        int Ncameras_intrinsics= PyArray_DIMS(intrinsics)[0];
        int Ncameras_extrinsics= PyArray_DIMS(extrinsics_rt_fromref)[0];
        int Nframes            = PyArray_DIMS(frames_rt_toref)[0];
        int Npoints            = PyArray_DIMS(points)[0];
        int NobservationsBoard = PyArray_DIMS(observations_board)[0];

        // The checks in optimize_validate_args() make sure these casts are kosher
        double*       c_intrinsics     = (double*)  PyArray_DATA(intrinsics);
        mrcal_pose_t*       c_extrinsics     = (mrcal_pose_t*)  PyArray_DATA(extrinsics_rt_fromref);
        mrcal_pose_t*       c_frames         = (mrcal_pose_t*)  PyArray_DATA(frames_rt_toref);
        mrcal_point3_t*     c_points         = (mrcal_point3_t*)PyArray_DATA(points);
        mrcal_point2_t*     c_calobject_warp =
            IS_NULL(calobject_warp) ?
            NULL : (mrcal_point2_t*)PyArray_DATA(calobject_warp);


        mrcal_point3_t* c_observations_board_pool = (mrcal_point3_t*)PyArray_DATA(observations_board); // must be contiguous; made sure above

        observation_board_t c_observations_board[NobservationsBoard];
        fill_c_observations_board(c_observations_board,
                                  NobservationsBoard,
                                  indices_frame_camintrinsics_camextrinsics);

        int NobservationsPoint = PyArray_DIMS(observations_point)[0];

        observation_point_t c_observations_point[NobservationsPoint];
        for(int i_observation=0; i_observation<NobservationsPoint; i_observation++)
        {
            int i_point          = ((int*)PyArray_DATA(indices_point_camintrinsics_camextrinsics))[i_observation*3 + 0];
            int i_cam_intrinsics = ((int*)PyArray_DATA(indices_point_camintrinsics_camextrinsics))[i_observation*3 + 1];
            int i_cam_extrinsics = ((int*)PyArray_DATA(indices_point_camintrinsics_camextrinsics))[i_observation*3 + 2];

            c_observations_point[i_observation].i_cam_intrinsics = i_cam_intrinsics;
            c_observations_point[i_observation].i_cam_extrinsics = i_cam_extrinsics;
            c_observations_point[i_observation].i_point          = i_point;

            c_observations_point[i_observation].px = ((mrcal_point3_t*)PyArray_DATA(observations_point))[i_observation];
        }



        mrcal_problem_details_t problem_details =
            { .do_optimize_intrinsic_core        = PyObject_IsTrue(do_optimize_intrinsic_core),
              .do_optimize_intrinsic_distortions = PyObject_IsTrue(do_optimize_intrinsic_distortions),
              .do_optimize_extrinsics            = PyObject_IsTrue(do_optimize_extrinsics),
              .do_optimize_frames                = PyObject_IsTrue(do_optimize_frames),
              .do_optimize_calobject_warp        = PyObject_IsTrue(do_optimize_calobject_warp),
              .do_skip_regularization            = skip_regularization && PyObject_IsTrue(skip_regularization)
            };

        mrcal_problem_constants_t problem_constants =
            {.point_min_range = point_min_range,
             .point_max_range = point_max_range};

        int Nmeasurements = mrcal_getNmeasurements_all(Ncameras_intrinsics, NobservationsBoard, NobservationsPoint,
                                                       calibration_object_width_n,
                                                       calibration_object_height_n,
                                                       problem_details,
                                                       lensmodel_type);

        int Nintrinsics_state = mrcal_getNintrinsicOptimizationParams(problem_details, lensmodel_type);

        // input
        int* c_imagersizes = PyArray_DATA(imagersizes);

        int Nstate = mrcal_getNstate(Ncameras_intrinsics, Ncameras_extrinsics,
                                     Nframes, Npoints-Npoints_fixed,
                                     problem_details, lensmodel_type);

        // both optimize() and optimizerCallback() use this
        p_packed_final = (PyArrayObject*)PyArray_SimpleNew(1, ((npy_intp[]){Nstate}), NPY_DOUBLE);
        double* c_p_packed_final = PyArray_DATA(p_packed_final);

        x_final = (PyArrayObject*)PyArray_SimpleNew(1, ((npy_intp[]){Nmeasurements}), NPY_DOUBLE);
        double* c_x_final = PyArray_DATA(x_final);

        if( is_optimize )
        {
            // we're wrapping mrcal_optimize()
            const int Npoints_fromBoards =
                NobservationsBoard *
                calibration_object_width_n*calibration_object_height_n;

            mrcal_stats_t stats =
                mrcal_optimize( c_p_packed_final,
                                Nstate*sizeof(double),
                                c_x_final,
                                Nmeasurements*sizeof(double),
                                NULL,
                                c_intrinsics,
                                c_extrinsics,
                                c_frames,
                                c_points,
                                c_calobject_warp,

                                c_observations_board_pool,
                                NobservationsBoard,

                                Ncameras_intrinsics, Ncameras_extrinsics,
                                Nframes, Npoints, Npoints_fixed,

                                c_observations_board,
                                c_observations_point,
                                NobservationsPoint,

                                false,
                                verbose &&                PyObject_IsTrue(verbose),
                                skip_outlier_rejection && PyObject_IsTrue(skip_outlier_rejection),
                                lensmodel_type,
                                observed_pixel_uncertainty,
                                c_imagersizes,
                                problem_details, &problem_constants,

                                calibration_object_spacing,
                                calibration_object_width_n,
                                calibration_object_height_n);

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
            // we're wrapping mrcal_optimizerCallback()

            int N_j_nonzero = mrcal_getN_j_nonzero(Ncameras_intrinsics, Ncameras_extrinsics,
                                                   c_observations_board, NobservationsBoard,
                                                   c_observations_point, NobservationsPoint,
                                                   Npoints, Npoints_fixed,
                                                   problem_details,
                                                   lensmodel_type,
                                                   calibration_object_width_n,
                                                   calibration_object_height_n);

            PyArrayObject* P = (PyArrayObject*)PyArray_SimpleNew(1, ((npy_intp[]){Nmeasurements + 1}), NPY_INT32);
            PyArrayObject* I = (PyArrayObject*)PyArray_SimpleNew(1, ((npy_intp[]){N_j_nonzero      }), NPY_INT32);
            PyArrayObject* X = (PyArrayObject*)PyArray_SimpleNew(1, ((npy_intp[]){N_j_nonzero      }), NPY_DOUBLE);

            cholmod_sparse Jt = {
                .nrow   = Nstate,
                .ncol   = Nmeasurements,
                .nzmax  = N_j_nonzero,
                .stype  = 0,
                .itype  = CHOLMOD_INT,
                .xtype  = CHOLMOD_REAL,
                .dtype  = CHOLMOD_DOUBLE,
                .sorted = 1,
                .packed = 1,
                .p = PyArray_DATA(P),
                .i = PyArray_DATA(I),
                .x = PyArray_DATA(X) };

            if(!mrcal_optimizerCallback( // out
                                         c_p_packed_final,
                                         Nstate*sizeof(double),
                                         c_x_final,
                                         Nmeasurements*sizeof(double),
                                         &Jt,

                                         // in
                                         c_intrinsics,
                                         c_extrinsics,
                                         c_frames,
                                         c_points,
                                         c_calobject_warp,

                                         Ncameras_intrinsics, Ncameras_extrinsics,
                                         Nframes, Npoints, Npoints_fixed,

                                         c_observations_board,
                                         c_observations_board_pool,
                                         NobservationsBoard,
                                         c_observations_point,
                                         NobservationsPoint,

                                         verbose && PyObject_IsTrue(verbose),
                                         lensmodel_type,
                                         observed_pixel_uncertainty,
                                         c_imagersizes,
                                         problem_details, &problem_constants,

                                         calibration_object_spacing,
                                         calibration_object_width_n,
                                         calibration_object_height_n) )
            {
                BARF("mrcal_optimizerCallback() failed!'");
                goto done;
            }


            PyObject* factorization = CHOLMOD_factorization_from_cholmod_sparse(&Jt);
            if(factorization == NULL)
                goto done;

            result = PyTuple_New(4);
            PyTuple_SET_ITEM(result, 0, (PyObject*)p_packed_final);
            PyTuple_SET_ITEM(result, 1, (PyObject*)x_final);
            PyTuple_SET_ITEM(result, 2,
                             csr_from_cholmod_sparse(&Jt,
                                                     (PyObject*)P,
                                                     (PyObject*)I,
                                                     (PyObject*)X));
            PyTuple_SET_ITEM(result, 3, factorization);

            for(int i=0; i<PyTuple_Size(result); i++)
                Py_INCREF(PyTuple_GET_ITEM(result,i));
            Py_DECREF(P);
            Py_DECREF(I);
            Py_DECREF(X);
        }
    }

 done:
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-to-pointer-cast"
    OPTIMIZERCALLBACK_ARGUMENTS_REQUIRED(FREE_PYARRAY) ;
    OPTIMIZERCALLBACK_ARGUMENTS_OPTIONAL(FREE_PYARRAY) ;
#pragma GCC diagnostic pop

    if(x_final) Py_DECREF(x_final);
    if(pystats) Py_DECREF(pystats);

    RESET_SIGINT();
    return result;
}

static PyObject* optimizerCallback(PyObject* NPY_UNUSED(self),
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
// same arguments as optimizerCallback(). OR in lieu of that, the dimensions can
// be passed-in explicitly with arguments
//
// If both are given, the explicit arguments take precedence. If neither are
// given, I assume 0.
//
// This means that the arguments that are required in optimizerCallback() are
// only optional here
typedef int (callback_state_index_t)(int i,
                                     int Ncameras_intrinsics,
                                     int Ncameras_extrinsics,
                                     int Nframes,
                                     int Npoints,
                                     int Npoints_fixed,
                                     int NobservationsBoard,
                                     int NobservationsPoint,
                                     int calibration_object_width_n,
                                     int calibration_object_height_n,
                                     lensmodel_t lensmodel,
                                     mrcal_problem_details_t problem_details);

static PyObject* state_index_generic(PyObject* self, PyObject* args, PyObject* kwargs,
                                     const char* argname,
                                     callback_state_index_t cb)
{
    // This is VERY similar to _pack_unpack_state(). Please consolidate
    // Also somewhat similar to _optimize()

    PyObject* result = NULL;

    OPTIMIZERCALLBACK_ARGUMENTS_ALL(ARG_DEFINE) ;
    int i = -1;

    int Ncameras_intrinsics = -1;
    int Ncameras_extrinsics = -1;
    int Nframes             = -1;
    int Npoints             = -1;
    int NobservationsBoard  = -1;
    int NobservationsPoint  = -1;

    char* keywords[] = { (char*)argname,
                         OPTIMIZERCALLBACK_ARGUMENTS_REQUIRED(NAMELIST)

                         "Ncameras_intrinsics",
                         "Ncameras_extrinsics",
                         "Nframes",
                         "Npoints",
                         "NobservationsBoard",
                         "NobservationsPoint",
                         OPTIMIZERCALLBACK_ARGUMENTS_OPTIONAL(NAMELIST)
                         NULL};
    char** keywords_noargname = &keywords[1];

    if(argname != NULL)
    {
        if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                         "i"
                                         "|" // everything is optional. I apply
                                             // logic down the line to get what
                                             // I need
                                         OPTIMIZERCALLBACK_ARGUMENTS_REQUIRED(PARSECODE)
                                         "iiiiii"
                                         OPTIMIZERCALLBACK_ARGUMENTS_OPTIONAL(PARSECODE),

                                         keywords,

                                         &i,
                                         OPTIMIZERCALLBACK_ARGUMENTS_REQUIRED(PARSEARG)
                                         &Ncameras_intrinsics,
                                         &Ncameras_extrinsics,
                                         &Nframes,
                                         &Npoints,
                                         &NobservationsBoard,
                                         &NobservationsPoint,
                                         OPTIMIZERCALLBACK_ARGUMENTS_OPTIONAL(PARSEARG) NULL))
            goto done;
    }
    else
    {
        if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                         OPTIMIZERCALLBACK_ARGUMENTS_REQUIRED(PARSECODE) "|"
                                         "iiiiii"
                                         OPTIMIZERCALLBACK_ARGUMENTS_OPTIONAL(PARSECODE),

                                         keywords_noargname,

                                         OPTIMIZERCALLBACK_ARGUMENTS_REQUIRED(PARSEARG)
                                         &Ncameras_intrinsics,
                                         &Ncameras_extrinsics,
                                         &Nframes,
                                         &Npoints,
                                         &NobservationsBoard,
                                         &NobservationsPoint,
                                         OPTIMIZERCALLBACK_ARGUMENTS_OPTIONAL(PARSEARG) NULL))
            goto done;
    }

    if(lensmodel == NULL)
    {
        BARF("The 'lensmodel' argument is required");
        goto done;
    }

    mrcal_problem_details_t problem_details =
        { .do_optimize_intrinsic_core        = PyObject_IsTrue(do_optimize_intrinsic_core),
          .do_optimize_intrinsic_distortions = PyObject_IsTrue(do_optimize_intrinsic_distortions),
          .do_optimize_extrinsics            = PyObject_IsTrue(do_optimize_extrinsics),
          .do_optimize_frames                = PyObject_IsTrue(do_optimize_frames),
          .do_optimize_calobject_warp        = PyObject_IsTrue(do_optimize_calobject_warp),
          .do_skip_regularization            = skip_regularization && PyObject_IsTrue(skip_regularization)
        };

    lensmodel_t lensmodel_type;
    if(!parse_lensmodel_from_arg(&lensmodel_type, lensmodel))
        goto done;

    // checks dimensionality of array !IS_NULL. So if any array isn't passed-in,
    // that's OK! After I do this and if !IS_NULL, then I can ask for array
    // dimensions safely
    bool check(void)
    {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-to-pointer-cast"
        OPTIMIZERCALLBACK_ARGUMENTS_ALL(CHECK_LAYOUT) ;
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
    if(NobservationsBoard < 0)  NobservationsBoard  = IS_NULL(observations_board)    ? 0 : PyArray_DIMS(observations_board)    [0];
    if(NobservationsPoint < 0)  NobservationsPoint  = IS_NULL(observations_point)    ? 0 : PyArray_DIMS(observations_point)    [0];

    int index = cb(i,
                   Ncameras_intrinsics,
                   Ncameras_extrinsics,
                   Nframes,
                   Npoints,
                   Npoints_fixed,
                   NobservationsBoard,
                   NobservationsPoint,
                   calibration_object_width_n,
                   calibration_object_height_n,
                   lensmodel_type,
                   problem_details);
    if(index < 0)
        goto done;

    result = Py_BuildValue("i", index);

 done:
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-to-pointer-cast"
    OPTIMIZERCALLBACK_ARGUMENTS_REQUIRED(FREE_PYARRAY) ;
    OPTIMIZERCALLBACK_ARGUMENTS_OPTIONAL(FREE_PYARRAY) ;
#pragma GCC diagnostic pop

    return result;
}

static int callback_state_index_intrinsics(int i,
                                           int Ncameras_intrinsics,
                                           int Ncameras_extrinsics,
                                           int Nframes,
                                           int Npoints,
                                           int Npoints_fixed,
                                           int NobservationsBoard,
                                           int NobservationsPoint,
                                           int calibration_object_width_n,
                                           int calibration_object_height_n,
                                           lensmodel_t lensmodel,
                                           mrcal_problem_details_t problem_details)
{
    return mrcal_state_index_intrinsics(i,
                                        problem_details,
                                        lensmodel);
}
static PyObject* state_index_intrinsics(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return state_index_generic(self, args, kwargs,
                               "i_cam_intrinsics",
                               callback_state_index_intrinsics);
}

static int callback_state_index_camera_rt(int i,
                                          int Ncameras_intrinsics,
                                          int Ncameras_extrinsics,
                                          int Nframes,
                                          int Npoints,
                                          int Npoints_fixed,
                                          int NobservationsBoard,
                                          int NobservationsPoint,
                                          int calibration_object_width_n,
                                          int calibration_object_height_n,
                                          lensmodel_t lensmodel,
                                          mrcal_problem_details_t problem_details)
{
    if( i < 0 || i >= Ncameras_extrinsics)
    {
        BARF( "i_cam_extrinsics must refer to a valid camera, i.e. be in the range [0,%d] inclusive. Instead I got %d",
              Ncameras_extrinsics-1,i);
        return -1;
    }
    return
        mrcal_state_index_camera_rt(i,
                                    Ncameras_intrinsics,
                                    problem_details,
                                    lensmodel);
}
static PyObject* state_index_camera_rt(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return state_index_generic(self, args, kwargs,
                               "i_cam_extrinsics",
                               callback_state_index_camera_rt);
}

static int callback_state_index_frame_rt(int i,
                                         int Ncameras_intrinsics,
                                         int Ncameras_extrinsics,
                                         int Nframes,
                                         int Npoints,
                                         int Npoints_fixed,
                                         int NobservationsBoard,
                                         int NobservationsPoint,
                                         int calibration_object_width_n,
                                         int calibration_object_height_n,
                                         lensmodel_t lensmodel,
                                         mrcal_problem_details_t problem_details)
{
    if( i < 0 || i >= Nframes)
    {
        BARF( "i_frames must refer to a valid frame, i.e. be in the range [0,%d] inclusive. Instead I got %d",
              Nframes-1,i);
        return -1;
    }
    return
        mrcal_state_index_frame_rt(i,
                                   Ncameras_intrinsics,
                                   Ncameras_extrinsics,
                                   problem_details,
                                   lensmodel);
}
static PyObject* state_index_frame_rt(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return state_index_generic(self, args, kwargs,
                               "i_frame",
                               callback_state_index_frame_rt);
}

static int callback_state_index_point(int i,
                                      int Ncameras_intrinsics,
                                      int Ncameras_extrinsics,
                                      int Nframes,
                                      int Npoints,
                                      int Npoints_fixed,
                                      int NobservationsBoard,
                                      int NobservationsPoint,
                                      int calibration_object_width_n,
                                      int calibration_object_height_n,
                                      lensmodel_t lensmodel,
                                      mrcal_problem_details_t problem_details)
{
    if( i < 0 || i >= Npoints-Npoints_fixed)
    {
        BARF( "i_frames must refer to a valid frame, i.e. be in the range [0,%d] inclusive. Instead I got %d",
              Nframes-1,i);
        return -1;
    }
    return
        mrcal_state_index_point(i,
                                Nframes,
                                Ncameras_intrinsics,
                                Ncameras_extrinsics,
                                problem_details,
                                lensmodel);
}
static PyObject* state_index_point(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return state_index_generic(self, args, kwargs,
                               "i_point",
                               callback_state_index_point);
}

static int callback_state_index_calobject_warp(int i,
                                               int Ncameras_intrinsics,
                                               int Ncameras_extrinsics,
                                               int Nframes,
                                               int Npoints,
                                               int Npoints_fixed,
                                               int NobservationsBoard,
                                               int NobservationsPoint,
                                               int calibration_object_width_n,
                                               int calibration_object_height_n,
                                               lensmodel_t lensmodel,
                                               mrcal_problem_details_t problem_details)
{
    return
        mrcal_state_index_calobject_warp(Npoints - Npoints_fixed,
                                         Nframes,
                                         Ncameras_intrinsics,
                                         Ncameras_extrinsics,
                                         problem_details,
                                         lensmodel);
}
static PyObject* state_index_calobject_warp(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return state_index_generic(self, args, kwargs,
                               NULL,
                               callback_state_index_calobject_warp);
}

static int callback_getNmeasurements_all(int i,
                                         int Ncameras_intrinsics,
                                         int Ncameras_extrinsics,
                                         int Nframes,
                                         int Npoints,
                                         int Npoints_fixed,
                                         int NobservationsBoard,
                                         int NobservationsPoint,
                                         int calibration_object_width_n,
                                         int calibration_object_height_n,
                                         lensmodel_t lensmodel,
                                         mrcal_problem_details_t problem_details)
{
    return
        mrcal_getNmeasurements_all(Ncameras_intrinsics,
                                   NobservationsBoard,
                                   NobservationsPoint,
                                   calibration_object_width_n,
                                   calibration_object_height_n,
                                   problem_details,
                                   lensmodel);
}
static PyObject* getNmeasurements_all(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return state_index_generic(self, args, kwargs,
                               NULL,
                               callback_getNmeasurements_all);
}

static int callback_getNmeasurements_boards(int i,
                                            int Ncameras_intrinsics,
                                            int Ncameras_extrinsics,
                                            int Nframes,
                                            int Npoints,
                                            int Npoints_fixed,
                                            int NobservationsBoard,
                                            int NobservationsPoint,
                                            int calibration_object_width_n,
                                            int calibration_object_height_n,
                                            lensmodel_t lensmodel,
                                            mrcal_problem_details_t problem_details)
{
    return
        mrcal_getNmeasurements_boards(NobservationsBoard,
                                      calibration_object_width_n,
                                      calibration_object_height_n);
}
static PyObject* getNmeasurements_boards(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return state_index_generic(self, args, kwargs,
                               NULL,
                               callback_getNmeasurements_boards);
}

static int callback_getNmeasurements_points(int i,
                                            int Ncameras_intrinsics,
                                            int Ncameras_extrinsics,
                                            int Nframes,
                                            int Npoints,
                                            int Npoints_fixed,
                                            int NobservationsBoard,
                                            int NobservationsPoint,
                                            int calibration_object_width_n,
                                            int calibration_object_height_n,
                                            lensmodel_t lensmodel,
                                            mrcal_problem_details_t problem_details)
{
    return
        mrcal_getNmeasurements_points(NobservationsPoint);
}
static PyObject* getNmeasurements_points(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return state_index_generic(self, args, kwargs,
                               NULL,
                               callback_getNmeasurements_points);
}

static int callback_getNmeasurements_regularization(int i,
                                                    int Ncameras_intrinsics,
                                                    int Ncameras_extrinsics,
                                                    int Nframes,
                                                    int Npoints,
                                                    int Npoints_fixed,
                                                    int NobservationsBoard,
                                                    int NobservationsPoint,
                                                    int calibration_object_width_n,
                                                    int calibration_object_height_n,
                                                    lensmodel_t lensmodel,
                                                    mrcal_problem_details_t problem_details)
{
    return
        mrcal_getNmeasurements_regularization(Ncameras_intrinsics,
                                              problem_details,
                                              lensmodel);
}
static PyObject* getNmeasurements_regularization(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return state_index_generic(self, args, kwargs,
                               NULL,
                               callback_getNmeasurements_regularization);
}



static PyObject* _pack_unpack_state(PyObject* self, PyObject* args, PyObject* kwargs,
                                    bool pack)
{
    // This is VERY similar to state_index_generic(). Please consolidate
    PyObject*      result = NULL;
    PyArrayObject* p      = NULL;

    OPTIMIZERCALLBACK_ARGUMENTS_ALL(ARG_DEFINE) ;

    int Ncameras_intrinsics = -1;
    int Ncameras_extrinsics = -1;
    int Nframes             = -1;
    int Npoints             = -1;
    int NobservationsBoard  = -1;
    int NobservationsPoint  = -1;

    char* keywords[] = { "p",
                         OPTIMIZERCALLBACK_ARGUMENTS_REQUIRED(NAMELIST)

                         "Ncameras_intrinsics",
                         "Ncameras_extrinsics",
                         "Nframes",
                         "Npoints",
                         "NobservationsBoard",
                         "NobservationsPoint",
                         OPTIMIZERCALLBACK_ARGUMENTS_OPTIONAL(NAMELIST)
                         NULL};

    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     "O&"
                                     "|" // everything is optional. I apply
                                     // logic down the line to get what
                                     // I need
                                     OPTIMIZERCALLBACK_ARGUMENTS_REQUIRED(PARSECODE)
                                     "iiiiii"
                                     OPTIMIZERCALLBACK_ARGUMENTS_OPTIONAL(PARSECODE),

                                     keywords,

                                     PyArray_Converter, &p,
                                     OPTIMIZERCALLBACK_ARGUMENTS_REQUIRED(PARSEARG)
                                     &Ncameras_intrinsics,
                                     &Ncameras_extrinsics,
                                     &Nframes,
                                     &Npoints,
                                     &NobservationsBoard,
                                     &NobservationsPoint,
                                     OPTIMIZERCALLBACK_ARGUMENTS_OPTIONAL(PARSEARG) NULL))
        goto done;

    if(lensmodel == NULL)
    {
        BARF("The 'lensmodel' argument is required");
        goto done;
    }

    mrcal_problem_details_t problem_details =
        { .do_optimize_intrinsic_core        = PyObject_IsTrue(do_optimize_intrinsic_core),
          .do_optimize_intrinsic_distortions = PyObject_IsTrue(do_optimize_intrinsic_distortions),
          .do_optimize_extrinsics            = PyObject_IsTrue(do_optimize_extrinsics),
          .do_optimize_frames                = PyObject_IsTrue(do_optimize_frames),
          .do_optimize_calobject_warp        = PyObject_IsTrue(do_optimize_calobject_warp),
          .do_skip_regularization            = skip_regularization && PyObject_IsTrue(skip_regularization)
        };

    lensmodel_t lensmodel_type;
    if(!parse_lensmodel_from_arg(&lensmodel_type, lensmodel))
        goto done;

    // checks dimensionality of array !IS_NULL. So if any array isn't passed-in,
    // that's OK! After I do this and if !IS_NULL, then I can ask for array
    // dimensions safely
    bool check(void)
    {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-to-pointer-cast"
        OPTIMIZERCALLBACK_ARGUMENTS_ALL(CHECK_LAYOUT) ;
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
    if(NobservationsBoard < 0)  NobservationsBoard  = IS_NULL(observations_board)    ? 0 : PyArray_DIMS(observations_board)    [0];
    if(NobservationsPoint < 0)  NobservationsPoint  = IS_NULL(observations_point)    ? 0 : PyArray_DIMS(observations_point)    [0];


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
        mrcal_getNstate(Ncameras_intrinsics, Ncameras_extrinsics,
                        Nframes, Npoints - Npoints_fixed,
                        problem_details,
                        lensmodel_type);

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
                                            lensmodel_type, problem_details,
                                            Ncameras_intrinsics, Ncameras_extrinsics,
                                            Nframes, Npoints-Npoints_fixed );
            x = &x[Nstate];
        }
    else
        for(int i=0; i<PyArray_SIZE(p)/Nstate; i++)
        {
            mrcal_unpack_solver_state_vector( x,
                                              lensmodel_type, problem_details,
                                              Ncameras_intrinsics, Ncameras_extrinsics,
                                              Nframes, Npoints-Npoints_fixed );
            x = &x[Nstate];
        }

    Py_INCREF(Py_None);
    result = Py_None;

 done:
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-to-pointer-cast"
    OPTIMIZERCALLBACK_ARGUMENTS_REQUIRED(FREE_PYARRAY) ;
    OPTIMIZERCALLBACK_ARGUMENTS_OPTIONAL(FREE_PYARRAY) ;
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

static PyObject* get_corresponding_icam_extrinsics(PyObject* self, PyObject* args, PyObject* kwargs)
{
    // This is VERY similar to state_index_generic(). Please consolidate
    PyObject* result          = NULL;
    int       icam_intrinsics = -1;

    OPTIMIZERCALLBACK_ARGUMENTS_ALL(ARG_DEFINE) ;

    int Ncameras_intrinsics = -1;
    int Ncameras_extrinsics = -1;
    int NobservationsBoard  = -1;
    int NobservationsPoint  = -1;

    char* keywords[] = { "icam_intrinsics",
                         OPTIMIZERCALLBACK_ARGUMENTS_REQUIRED(NAMELIST)

                         "Ncameras_intrinsics",
                         "Ncameras_extrinsics",
                         "NobservationsBoard",
                         "NobservationsPoint",
                         OPTIMIZERCALLBACK_ARGUMENTS_OPTIONAL(NAMELIST)
                         NULL};

    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     "i"
                                     "|" // everything is optional. I apply
                                     // logic down the line to get what
                                     // I need
                                     OPTIMIZERCALLBACK_ARGUMENTS_REQUIRED(PARSECODE)
                                     "iiii"
                                     OPTIMIZERCALLBACK_ARGUMENTS_OPTIONAL(PARSECODE),

                                     keywords,

                                     &icam_intrinsics,
                                     OPTIMIZERCALLBACK_ARGUMENTS_REQUIRED(PARSEARG)
                                     &Ncameras_intrinsics,
                                     &Ncameras_extrinsics,
                                     &NobservationsBoard,
                                     &NobservationsPoint,
                                     OPTIMIZERCALLBACK_ARGUMENTS_OPTIONAL(PARSEARG) NULL))
        goto done;

    // checks dimensionality of array !IS_NULL. So if any array isn't passed-in,
    // that's OK! After I do this and if !IS_NULL, then I can ask for array
    // dimensions safely
    bool check(void)
    {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-to-pointer-cast"
        OPTIMIZERCALLBACK_ARGUMENTS_ALL(CHECK_LAYOUT) ;
#pragma GCC diagnostic pop
        return true;
    }
    if(!check()) goto done;

    // If explicit dimensions are given, use them. If they're not given, but we
    // have an array, use those dimensions. If an array isn't given either, use
    // 0
    if(Ncameras_intrinsics < 0) Ncameras_intrinsics = IS_NULL(intrinsics)            ? 0 : PyArray_DIMS(intrinsics)            [0];
    if(Ncameras_extrinsics < 0) Ncameras_extrinsics = IS_NULL(extrinsics_rt_fromref) ? 0 : PyArray_DIMS(extrinsics_rt_fromref) [0];
    if(NobservationsBoard < 0)  NobservationsBoard  = IS_NULL(observations_board)    ? 0 : PyArray_DIMS(observations_board)    [0];
    if(NobservationsPoint < 0)  NobservationsPoint  = IS_NULL(observations_point)    ? 0 : PyArray_DIMS(observations_point)    [0];


    if( icam_intrinsics < 0 || icam_intrinsics >= Ncameras_intrinsics )
    {
        BARF("The given icam_intrinsics=%d is out of bounds. Must be >= 0 and < %d",
             icam_intrinsics, Ncameras_intrinsics);
        goto done;
    }



    int icam_extrinsics;
    {
        observation_board_t c_observations_board[NobservationsBoard];
        fill_c_observations_board(c_observations_board,
                                  NobservationsBoard,
                                  indices_frame_camintrinsics_camextrinsics);
        if(!mrcal_get_corresponding_icam_extrinsics(&icam_extrinsics,

                                                    icam_intrinsics,
                                                    Ncameras_intrinsics,
                                                    Ncameras_extrinsics,
                                                    NobservationsBoard,
                                                    c_observations_board))
        {
            BARF("Error calling mrcal_get_corresponding_icam_extrinsics()");
            goto done;
        }
    }

    result = PyLong_FromLong(icam_extrinsics);

 done:
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-to-pointer-cast"
    OPTIMIZERCALLBACK_ARGUMENTS_REQUIRED(FREE_PYARRAY) ;
    OPTIMIZERCALLBACK_ARGUMENTS_OPTIONAL(FREE_PYARRAY) ;
#pragma GCC diagnostic pop

    return result;
}

static const char state_index_intrinsics_docstring[] =
#include "state_index_intrinsics.docstring.h"
    ;
static const char state_index_camera_rt_docstring[] =
#include "state_index_camera_rt.docstring.h"
    ;
static const char state_index_frame_rt_docstring[] =
#include "state_index_frame_rt.docstring.h"
    ;
static const char state_index_point_docstring[] =
#include "state_index_point.docstring.h"
    ;
static const char state_index_calobject_warp_docstring[] =
#include "state_index_calobject_warp.docstring.h"
    ;
static const char pack_state_docstring[] =
#include "pack_state.docstring.h"
    ;
static const char unpack_state_docstring[] =
#include "unpack_state.docstring.h"
    ;
static const char getNmeasurements_all_docstring[] =
#include "getNmeasurements_all.docstring.h"
    ;
static const char getNmeasurements_boards_docstring[] =
#include "getNmeasurements_boards.docstring.h"
    ;
static const char getNmeasurements_points_docstring[] =
#include "getNmeasurements_points.docstring.h"
    ;
static const char getNmeasurements_regularization_docstring[] =
#include "getNmeasurements_regularization.docstring.h"
    ;
static const char get_corresponding_icam_extrinsics_docstring[] =
#include "get_corresponding_icam_extrinsics.docstring.h"
    ;






static const char optimize_docstring[] =
#include "optimize.docstring.h"
    ;
static const char optimizerCallback_docstring[] =
#include "optimizerCallback.docstring.h"
    ;
static const char getLensModelMeta_docstring[] =
#include "getLensModelMeta.docstring.h"
    ;
static const char getNlensParams_docstring[] =
#include "getNlensParams.docstring.h"
    ;
static const char getSupportedLensModels_docstring[] =
#include "getSupportedLensModels.docstring.h"
    ;
static const char getKnotsForSplinedModels_docstring[] =
#include "getKnotsForSplinedModels.docstring.h"
    ;
static const char project_stereographic_docstring[] =
#include "project_stereographic.docstring.h"
    ;
static const char unproject_stereographic_docstring[] =
#include "unproject_stereographic.docstring.h"
    ;
static PyMethodDef methods[] =
    { PYMETHODDEF_ENTRY(,optimize,                 METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(,optimizerCallback,        METH_VARARGS | METH_KEYWORDS),

      PYMETHODDEF_ENTRY(, state_index_intrinsics,          METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(, state_index_camera_rt,           METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(, state_index_frame_rt,            METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(, state_index_point,               METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(, state_index_calobject_warp,      METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(, pack_state,                      METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(, unpack_state,                    METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(, getNmeasurements_all,            METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(, getNmeasurements_boards,         METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(, getNmeasurements_points,         METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(, getNmeasurements_regularization, METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(, get_corresponding_icam_extrinsics,METH_VARARGS | METH_KEYWORDS),

      PYMETHODDEF_ENTRY(,getLensModelMeta,         METH_VARARGS),
      PYMETHODDEF_ENTRY(,getNlensParams,           METH_VARARGS),
      PYMETHODDEF_ENTRY(,getSupportedLensModels,   METH_NOARGS),
      PYMETHODDEF_ENTRY(,getKnotsForSplinedModels, METH_VARARGS),
      PYMETHODDEF_ENTRY(,project_stereographic,    METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(,unproject_stereographic,  METH_VARARGS | METH_KEYWORDS),
      {}
    };


static void _init_mrcal_common(PyObject* module)
{
    Py_INCREF(&CHOLMOD_factorization_type);
    PyModule_AddObject(module, "CHOLMOD_factorization", (PyObject *)&CHOLMOD_factorization_type);

}

#if PY_MAJOR_VERSION == 2

PyMODINIT_FUNC init_mrcal_nonbroadcasted(void)
{
    if (PyType_Ready(&CHOLMOD_factorization_type) < 0)
        return;

    PyObject* module =
        Py_InitModule3("_mrcal_nonbroadcasted", methods,
                       "Internal python wrappers for non-broadcasting functions");
    _init_mrcal_common(module);
    import_array();
}

#else

static struct PyModuleDef module_def =
    {
     PyModuleDef_HEAD_INIT,
     "_mrcal_nonbroadcasted",
     "Internal python wrappers for non-broadcasting functions",
     -1,
     methods
    };

PyMODINIT_FUNC PyInit__mrcal_nonbroadcasted(void)
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
