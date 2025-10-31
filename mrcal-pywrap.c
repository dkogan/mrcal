// Copyright (c) 2017-2023 California Institute of Technology ("Caltech"). U.S.
// Government sponsorship acknowledged. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <stdbool.h>
#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>
// Required for numpy 2. They now #include complex.h, so I is #defined to be the
// complex I, which conflicts with my usage here
#undef I

#include <signal.h>
#include <dogleg.h>

#if (CHOLMOD_VERSION > (CHOLMOD_VER_CODE(2,2))) && (CHOLMOD_VERSION < (CHOLMOD_VER_CODE(4,0)))
#include <cholmod_function.h>
#endif

#include "mrcal.h"
#include "mrcal-image.h"
#include "stereo.h"

#include "python-wrapping-utilities.h"



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
    // _Static_assert it, but internally cholmod uses void*, so I can't do that
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

    // initialized the first time cholmod_solve2() is called
    cholmod_dense* Y;
    cholmod_dense* E;

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
    if(self->E != NULL)
    {
        cholmod_free_dense (&self->E, &self->common);
        self->E = NULL;
    }
    if(self->Y != NULL)
    {
        cholmod_free_dense (&self->Y, &self->common);
        self->Y = NULL;
    }

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
#elif (CHOLMOD_VERSION < (CHOLMOD_VER_CODE(4,0)))
        CHOLMOD_FUNCTION_DEFAULTS ;
        CHOLMOD_FUNCTION_PRINTF(&self->common) = cholmod_error_callback;
#else
        SuiteSparse_config_printf_func_set(cholmod_error_callback);
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
        return PyUnicode_FromString("No factorization given");

    return PyUnicode_FromFormat("Initialized with a valid factorization. N=%d",
                               self->factorization->n);
}

static PyObject*
CHOLMOD_factorization_solve_xt_JtJ_bt(CHOLMOD_factorization* self, PyObject* args, PyObject* kwargs)
{
    cholmod_dense* M = NULL;


    // error by default
    PyObject* result = NULL;
    PyObject* Py_out = NULL;

    char* keywords[] = {"bt",
                        "sys",
                        NULL};
    PyObject* Py_bt = NULL;
    char*     sys   = "A";


    if(!(self->inited_common && self->factorization))
    {
        BARF("No factorization has been computed");
        goto done;
    }

    if( !PyArg_ParseTupleAndKeywords(args, kwargs,
                                     "O|s:CHOLMOD_factorization.solve_xt_JtJ_bt",
                                     keywords, &Py_bt, &sys))
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

#define LIST_SYS(_)                             \
  _(A)                                          \
  _(LDLt)                                       \
  _(LD)                                         \
  _(DLt)                                        \
  _(L)                                          \
  _(Lt)                                         \
  _(D)                                          \
  _(P)                                          \
  _(Pt)

#define SYS_CHECK(s) \
    else if(0 == strcmp(sys,           #s) || \
            0 == strcmp(sys,"CHOLMOD_" #s)) \
        CHOLMOD_system = CHOLMOD_ ## s;

#define SYS_NAME(s) \
    #s ","

    int CHOLMOD_system;
    if(0) ; LIST_SYS(SYS_CHECK)
    else
    {
        BARF("Unknown sys '%s' given. Known values of sys: (" LIST_SYS(SYS_NAME) ")",
             sys);
        goto done;
    }
#undef LIST_SYS
#undef SYS_CHECK
#undef SYS_NAME

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

    M = &out;

    if(!cholmod_solve2( CHOLMOD_system, self->factorization,
                        &b, NULL,
                        &M, NULL, &self->Y, &self->E,
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

    Py_INCREF(Py_out);
    result = Py_out;

 done:
    Py_XDECREF(Py_out);

    return result;
}

static PyObject*
CHOLMOD_factorization_rcond(CHOLMOD_factorization* self,
                            PyObject* NPY_UNUSED(args))
{
    if(!(self->inited_common && self->factorization))
    {
        BARF("No factorization has been computed");
        return NULL;
    }

    return PyFloat_FromDouble(cholmod_rcond( self->factorization,
                                             &self->common));
}

static const char CHOLMOD_factorization_docstring[] =
#include "CHOLMOD_factorization.docstring.h"
    ;
static const char CHOLMOD_factorization_solve_xt_JtJ_bt_docstring[] =
#include "CHOLMOD_factorization_solve_xt_JtJ_bt.docstring.h"
    ;
static const char CHOLMOD_factorization_rcond_docstring[] =
#include "CHOLMOD_factorization_rcond.docstring.h"
    ;

static PyMethodDef CHOLMOD_factorization_methods[] =
    {
        PYMETHODDEF_ENTRY(CHOLMOD_factorization_, solve_xt_JtJ_bt, METH_VARARGS | METH_KEYWORDS),
        PYMETHODDEF_ENTRY(CHOLMOD_factorization_, rcond,           METH_NOARGS),
        {}
    };


#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-braces"
// PyObject_HEAD_INIT throws
//   warning: missing braces around initializer []
// This isn't mine to fix, so I'm ignoring it
static PyTypeObject CHOLMOD_factorization_type =
{
    .ob_base      = PyVarObject_HEAD_INIT(NULL, 0)
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
                                     const char* lensmodel_cstring)
{
    mrcal_lensmodel_from_name(lensmodel, lensmodel_cstring);
    if( !mrcal_lensmodel_type_is_valid(lensmodel->type) )
    {
        switch(lensmodel->type)
        {
        case MRCAL_LENSMODEL_INVALID:
            // this should never (rarely?) happen
            BARF("Lens model '%s': error parsing",
                         lensmodel_cstring);
            return false;
        case MRCAL_LENSMODEL_INVALID_BADCONFIG:
            BARF("Lens model '%s': error parsing the required configuration",
                         lensmodel_cstring);
            return false;
        case MRCAL_LENSMODEL_INVALID_MISSINGCONFIG:
            BARF("Lens model '%s': missing the required configuration",
                         lensmodel_cstring);
            return false;
        case MRCAL_LENSMODEL_INVALID_TYPE:
            BARF("Invalid lens model type was passed in: '%s'. Must be one of " VALID_LENSMODELS_FORMAT,
                 lensmodel_cstring
                 VALID_LENSMODELS_ARGLIST);
            return false;
        default:
            BARF("Lens model '%s' produced an unexpected error: lensmodel->type=%d. This should never happen",
                 lensmodel_cstring,
                 (int)lensmodel->type);
            return false;
        }
        return false;
    }
    return true;
}

static PyObject* lensmodel_metadata_and_config(PyObject* NPY_UNUSED(self),
                                               PyObject* args)
{
    PyObject* result = NULL;

    char* lensmodel_string = NULL;
    if(!PyArg_ParseTuple( args, "s", &lensmodel_string ))
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
    return result;
}

static PyObject* knots_for_splined_models(PyObject* NPY_UNUSED(self),
                                          PyObject* args)
{
    PyObject*      result = NULL;
    PyArrayObject* py_ux  = NULL;
    PyArrayObject* py_uy  = NULL;

    char* lensmodel_string = NULL;
    if(!PyArg_ParseTuple( args, "s", &lensmodel_string ))
        goto done;
    mrcal_lensmodel_t lensmodel;
    if(!parse_lensmodel_from_arg(&lensmodel, lensmodel_string))
        goto done;

    if(lensmodel.type != MRCAL_LENSMODEL_SPLINED_STEREOGRAPHIC)
    {
        BARF( "This function works only with the MRCAL_LENSMODEL_SPLINED_STEREOGRAPHIC model. %s passed in",
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
    return result;
}

static PyObject* lensmodel_num_params(PyObject* NPY_UNUSED(self),
                                 PyObject* args)
{
    PyObject* result = NULL;

    char* lensmodel_string = NULL;
    if(!PyArg_ParseTuple( args, "s", &lensmodel_string ))
        goto done;
    mrcal_lensmodel_t lensmodel;
    if(!parse_lensmodel_from_arg(&lensmodel, lensmodel_string))
        goto done;

    int Nparams = mrcal_lensmodel_num_params(&lensmodel);

    result = Py_BuildValue("i", Nparams);

 done:
    return result;
}

static PyObject* supported_lensmodels(PyObject* NPY_UNUSED(self),
                                      PyObject* NPY_UNUSED(args))
{
    PyObject* result = NULL;
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
static
int PyArray_Converter_checkrenamed_leaveNone(PyObject* obj, PyObject** address)
{
    // If we see the ERROR poison string from renamed() in
    // _deserialize_optimization_inputs() in cameramodel.py, we produce None.
    // This string isn't supposed to actually be used
    if(PyUnicode_Check(obj) &&
       0 == strncmp("ERROR:", PyUnicode_AsUTF8(obj), 6))
    {
        *address = Py_None;
        Py_INCREF(Py_None);
        return 1;
    }        
    
    return PyArray_Converter_leaveNone(obj,address);
}

// For various utility functions. Accepts ONE lens model, not N of them like the optimization function
#define LENSMODEL_ONE_ARGUMENTS(_, suffix)                                     \
    _(lensmodel  ## suffix,                         char*,          NULL,    "s",  ,                                  NULL,                        -1,         {}          ) \
    _(intrinsics ## suffix,                         PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, intrinsics ## suffix,        NPY_DOUBLE, {-1       } )

#define ARGDEF_observations_point_triangulated(_)                       \
 _(observations_point_triangulated,    PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, observations_point_triangulated,          NPY_DOUBLE, {-1 COMMA  3       } )
#define ARGDEF_indices_point_triangulated_camintrinsics_camextrinsics(_) \
 _(indices_point_triangulated_camintrinsics_camextrinsics,PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, indices_point_triangulated_camintrinsics_camextrinsics, NPY_INT32,    {-1 COMMA  3       } )

#define OPTIMIZE_ARGUMENTS_REQUIRED(_)                                  \
    _(intrinsics,                         PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, intrinsics,                  NPY_DOUBLE, {-1 COMMA -1       } ) \
    _(lensmodel,                          char*,          NULL,    "s",  ,                        NULL,                        -1,         {}                   ) \
    _(imagersizes,                        PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, imagersizes,                 NPY_INT32,    {-1 COMMA 2        } )

// Defaults for do_optimize... MUST match those in ingest_packed_state()
//
// Accepting observed_pixel_uncertainty for backwards compatibility. It doesn't
// do anything anymore
#define OPTIMIZE_ARGUMENTS_OPTIONAL(_) \
    /* old flavors (new ones work too) */ \
    _(extrinsics_rt_fromref,              PyArrayObject*, NULL,    "O&", PyArray_Converter_checkrenamed_leaveNone COMMA, extrinsics_rt_fromref,       NPY_DOUBLE, {-1 COMMA  6       } ) \
    _(frames_rt_toref,                    PyArrayObject*, NULL,    "O&", PyArray_Converter_checkrenamed_leaveNone COMMA, frames_rt_toref,             NPY_DOUBLE, {-1 COMMA  6       } ) \
    /* new flavors (old ones work too) */ \
    _(rt_cam_ref,              PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, rt_cam_ref,       NPY_DOUBLE, {-1 COMMA  6       } ) \
    _(rt_ref_frame,                    PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, rt_ref_frame,             NPY_DOUBLE, {-1 COMMA  6       } ) \
    _(points,                             PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, points,                      NPY_DOUBLE, {-1 COMMA  3       } ) \
    _(observations_board,                 PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, observations_board,          NPY_DOUBLE, {-1 COMMA -1 COMMA -1 COMMA 3 } ) \
    _(indices_frame_camintrinsics_camextrinsics,PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, indices_frame_camintrinsics_camextrinsics,  NPY_INT32,    {-1 COMMA  3       } ) \
    _(observations_point,                 PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, observations_point,          NPY_DOUBLE, {-1 COMMA  3       } ) \
    _(indices_point_camintrinsics_camextrinsics,PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, indices_point_camintrinsics_camextrinsics, NPY_INT32,    {-1 COMMA  3       } ) \
    ARGDEF_observations_point_triangulated(_)                        \
    ARGDEF_indices_point_triangulated_camintrinsics_camextrinsics(_) \
    _(observed_pixel_uncertainty,         double,         -1.0,    "d",  ,                                  NULL,           -1,         {})  \
    _(calobject_warp,                     PyArrayObject*, NULL,    "O&", PyArray_Converter_leaveNone COMMA, calobject_warp, NPY_DOUBLE, {2}) \
    _(Npoints_fixed,                      int,            0,       "i",  ,                                  NULL,           -1,         {})  \
    _(do_optimize_intrinsics_core,        int,           -1,       "p",  ,                                  NULL,           -1,         {})  \
    _(do_optimize_intrinsics_distortions, int,           -1,       "p",  ,                                  NULL,           -1,         {})  \
    _(do_optimize_extrinsics,             int,           -1,       "p",  ,                                  NULL,           -1,         {})  \
    _(do_optimize_frames,                 int,           -1,       "p",  ,                                  NULL,           -1,         {})  \
    _(do_optimize_calobject_warp,         int,           -1,       "p",  ,                                  NULL,           -1,         {})  \
    _(calibration_object_spacing,         double,         -1.0,    "d",  ,                                  NULL,           -1,         {})  \
    _(point_min_range,                    double,         -1.0,    "d",  ,                                  NULL,           -1,         {})  \
    _(point_max_range,                    double,         -1.0,    "d",  ,                                  NULL,           -1,         {})  \
    _(verbose,                            int,            0,       "p",  ,                                  NULL,           -1,         {})  \
    _(do_apply_regularization,            int,            1,       "p",  ,                                  NULL,           -1,         {})  \
    /* The default for this MUST be 0. See mrcal.cameramodel._serialize_optimization_inputs()*/ \
    _(do_apply_regularization_unity_cam01,int,            0,       "p",  ,                                  NULL,           -1,         {})  \
    _(do_apply_outlier_rejection,         int,            1,       "p",  ,                                  NULL,           -1,         {})  \
    _(imagepaths,                         PyObject*,      NULL,    "O",  ,                                  NULL,           -1,         {})
/* imagepaths is in the argument list purely to make the
   mrcal-show-residuals-board-observation tool work. The python code doesn't
   actually touch it */

#define OPTIMIZER_CALLBACK_ARGUMENTS_OPTIONAL_EXTRA(_) \
    _(no_jacobian,                        int,               0,    "p",  ,                                  NULL,           -1,         {}) \
    _(no_factorization,                   int,               0,    "p",  ,                                  NULL,           -1,         {})


typedef enum {
    OPTIMIZEMODE_OPTIMIZE,
    OPTIMIZEMODE_CALLBACK,
    OPTIMIZEMODE_DRTRRP_DB
} optimizemode_t;

static bool lensmodel_one_validate_args( // out
                                         mrcal_lensmodel_t* mrcal_lensmodel,

                                         // in
                                         LENSMODEL_ONE_ARGUMENTS(ARG_LIST_DEFINE, )
                                         bool do_check_layout)
{
    if(do_check_layout)
    {
        LENSMODEL_ONE_ARGUMENTS(CHECK_LAYOUT, );
    }

    if(!parse_lensmodel_from_arg(mrcal_lensmodel, lensmodel))
        return false;
    int NlensParams      = mrcal_lensmodel_num_params(mrcal_lensmodel);
    int NlensParams_have = PyArray_DIMS(intrinsics)[PyArray_NDIM(intrinsics)-1];
    if( NlensParams != NlensParams_have )
    {
        BARF("intrinsics.shape[-1] MUST be %d. Instead got %ld",
             NlensParams,
             NlensParams_have );
        return false;
    }

    return true;
 done:
    return false;
}

// Using this for both optimize() and optimizer_callback()
static bool optimize_validate_args( // out
                                    mrcal_lensmodel_t* mrcal_lensmodel,

                                    // in
                                    optimizemode_t optimizemode,
                                    OPTIMIZE_ARGUMENTS_REQUIRED(ARG_LIST_DEFINE)
                                    OPTIMIZE_ARGUMENTS_OPTIONAL(ARG_LIST_DEFINE)
                                    OPTIMIZER_CALLBACK_ARGUMENTS_OPTIONAL_EXTRA(ARG_LIST_DEFINE)

                                    void* dummy __attribute__((unused)))
{
    _Static_assert( sizeof(mrcal_pose_t)/sizeof(double) == 6, "mrcal_pose_t is assumed to contain 6 elements");

    OPTIMIZE_ARGUMENTS_REQUIRED(CHECK_LAYOUT);
    OPTIMIZE_ARGUMENTS_OPTIONAL(CHECK_LAYOUT);
    OPTIMIZER_CALLBACK_ARGUMENTS_OPTIONAL_EXTRA(CHECK_LAYOUT);

    int Ncameras_intrinsics = PyArray_DIMS(intrinsics)[0];
    int Ncameras_extrinsics = PyArray_DIMS(rt_cam_ref)[0];
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

    // I reuse the single-lensmodel validation function. That function expects
    // ONE set of intrinsics instead of N intrinsics, like this function does.
    // But I already did the CHECK_LAYOUT() at the start of this function, and
    // I'm not going to do that again here: passing do_check_layout=false. So
    // that difference doesn't matter
    if(!lensmodel_one_validate_args(mrcal_lensmodel, lensmodel, intrinsics,
                                    false))
        return false;

    // make sure the indices arrays are valid: the data is monotonic and
    // in-range
    int Nframes = PyArray_DIMS(rt_ref_frame)[0];
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
        else if( iframe-iframe_last != 1 )
        {
            BARF("iframe MUST be increasing sequentially in indices_frame_camintrinsics_camextrinsics. Instead row %d of indices_frame_camintrinsics_camextrinsics has iframe=%d after previously seeing iframe=%d",
                         i_observation, iframe, iframe_last);
            return false;
        }

        iframe_last          = iframe;
        icam_intrinsics_last = icam_intrinsics;
        icam_extrinsics_last = icam_extrinsics;
    }
    if(Nobservations_board>0)
    {
        int i_observation_lastrow = Nobservations_board-1;
        int32_t iframe_lastrow = ((int32_t*)PyArray_DATA(indices_frame_camintrinsics_camextrinsics))[i_observation_lastrow*3 + 0];
        if(iframe_lastrow != Nframes-1)
        {
            BARF("iframe in indices_frame_camintrinsics_camextrinsics must cover ALL frames. Instead the last row of indices_frame_camintrinsics_camextrinsics has iframe=%d, but Nframes=%d",
                 iframe_lastrow, Nframes);
            return false;
        }
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

    // I allow i_point to be non-monotonic, but I do make sure that it covers
    // all Npoints of my array.
    int32_t i_point_biggest = -1;
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

        if(i_point > i_point_biggest)
        {
            if(i_point > i_point_biggest+1)
            {
                BARF("indices_point_camintrinsics_camextrinsics should contain i_point that extend the existing set by one point at a time at most. However row %d has i_point=%d while the biggest-seen-so-far i_point=%d",
                     i_observation, i_point, i_point_biggest);
                return false;
            }
            i_point_biggest = i_point;
        }
    }
    if(i_point_biggest != Npoints-1)
    {
        BARF("indices_point_camintrinsics_camextrinsics should cover all point indices in [0,%d], but there are gaps. The biggest i_point=%d",
             Npoints-1, i_point_biggest);
        return false;
    }

    i_point_biggest = -1;
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

        if(i_point > i_point_biggest)
        {
            if(i_point > i_point_biggest+1)
            {
                BARF("indices_point_triangulated_camintrinsics_camextrinsics should contain i_point that extend the existing set by one point at a time at most. However row %d has i_point=%d while the biggest-seen-so-far i_point=%d",
                     i_observation, i_point, i_point_biggest);
                return false;
            }
            i_point_biggest = i_point;
        }
    }

    // There are more checks for triangulated points, but I run them later, in
    // fill_c_observations_point_triangulated()

    return true;
 done:
    return false;
}

static void fill_c_observations_board(// out
                                      mrcal_observation_board_t* c_observations_board,

                                      // in
                                      int Nobservations_board,
                                      const PyArrayObject* indices_frame_camintrinsics_camextrinsics)
{
    for(int i_observation=0; i_observation<Nobservations_board; i_observation++)
    {
        int32_t iframe          = ((const int32_t*)PyArray_DATA((PyArrayObject*)indices_frame_camintrinsics_camextrinsics))[i_observation*3 + 0];
        int32_t icam_intrinsics = ((const int32_t*)PyArray_DATA((PyArrayObject*)indices_frame_camintrinsics_camextrinsics))[i_observation*3 + 1];
        int32_t icam_extrinsics = ((const int32_t*)PyArray_DATA((PyArrayObject*)indices_frame_camintrinsics_camextrinsics))[i_observation*3 + 2];

        c_observations_board[i_observation].icam.intrinsics = icam_intrinsics;
        c_observations_board[i_observation].icam.extrinsics = icam_extrinsics;
        c_observations_board[i_observation].iframe          = iframe;
    }
}

static void fill_c_observations_point(// out
                                      mrcal_observation_point_t* c_observations_point,

                                      // in
                                      int Nobservations_point,
                                      const PyArrayObject* indices_point_camintrinsics_camextrinsics)
{
    for(int i_observation=0; i_observation<Nobservations_point; i_observation++)
    {
        int32_t i_point         = ((const int32_t*)PyArray_DATA((PyArrayObject*)indices_point_camintrinsics_camextrinsics))[i_observation*3 + 0];
        int32_t icam_intrinsics = ((const int32_t*)PyArray_DATA((PyArrayObject*)indices_point_camintrinsics_camextrinsics))[i_observation*3 + 1];
        int32_t icam_extrinsics = ((const int32_t*)PyArray_DATA((PyArrayObject*)indices_point_camintrinsics_camextrinsics))[i_observation*3 + 2];

        c_observations_point[i_observation].icam.intrinsics = icam_intrinsics;
        c_observations_point[i_observation].icam.extrinsics = icam_extrinsics;
        c_observations_point[i_observation].i_point         = i_point;
    }
}

#if defined ENABLE_TRIANGULATED_WARNINGS && ENABLE_TRIANGULATED_WARNINGS
#warning "triangulated-solve: document observation order. All the points must be grouped together"
#endif
static bool _finish_triangulated_set(// out
                                     mrcal_observation_point_triangulated_t* c_observations_point_triangulated,
                                     // in
                                     const int Npoints_in_this_set,
                                     int ipoint_last_in_set)
{
    if(ipoint_last_in_set < 0)
        // No "last" set exists. Nothing to do.
        return true;
#if defined ENABLE_TRIANGULATED_WARNINGS && ENABLE_TRIANGULATED_WARNINGS
#warning "triangulated-solve: indices_point_triangulated_camintrinsics_camextrinsics are silly: ipoint is used ONLY to figure out where the set ends"
#endif
    c_observations_point_triangulated[ipoint_last_in_set].last_in_set = true;
    if(Npoints_in_this_set < 2)
    {
        BARF("Error in indices_point_triangulated_camintrinsics_camextrinsics[%d]. Each point must be observed at least 2 times",
             ipoint_last_in_set);
        return false;
    }
    return true;
}

// return the number of points, or <0 on error
static
int fill_c_observations_point_triangulated(// output. I fill in the given arrays
                                           mrcal_observation_point_triangulated_t* c_observations_point_triangulated,

                                           // input
                                           const PyArrayObject* observations_point_triangulated, // may be NULL
                                           // used only if observations_point_triangulated != NULL
                                           const mrcal_lensmodel_t* lensmodel,
                                           const double* intrinsics,

                                           const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics)
{
    if(indices_point_triangulated_camintrinsics_camextrinsics == NULL)
        return 0;

    int result = -1;

    // CHECK_LAYOUT will goto done on error
    if(observations_point_triangulated != NULL)
    {
        ARGDEF_observations_point_triangulated(CHECK_LAYOUT);
    }
    ARGDEF_indices_point_triangulated_camintrinsics_camextrinsics(CHECK_LAYOUT);

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

    // Needed for the unproject() below
    int                            Nintrinsics_state = 0;
    mrcal_projection_precomputed_t precomputed;
    if(lensmodel != NULL)
    {
        Nintrinsics_state = mrcal_lensmodel_num_params(lensmodel);
        mrcal_lensmodel_metadata_t meta = mrcal_lensmodel_metadata(lensmodel);
        if(!meta.has_gradients)
        {
            BARF("mrcal_unproject(lensmodel='%s') is not yet implemented: we need gradients",
                 mrcal_lensmodel_name_unconfigured(lensmodel));
            return -1;
        }
        _mrcal_precompute_lensmodel_data(&precomputed, lensmodel);
    }

    for(int i=0; i<N; i++)
    {
        const int32_t* row = &indices_point_triangulated_camintrinsics_camextrinsics__data[3*i];

        const int32_t ipoint          = row[0];
        const int32_t icam_intrinsics = row[1];
        const int32_t icam_extrinsics = row[2];

        c_observations_point_triangulated[i].last_in_set = false;
        c_observations_point_triangulated[i].outlier     = false;
        c_observations_point_triangulated[i].icam = (mrcal_camera_index_t){.intrinsics = icam_intrinsics,
                                                                           .extrinsics = icam_extrinsics};
        if(observations_point_triangulated__data != NULL)
        {
            const mrcal_point3_t* px_weight = (const mrcal_point3_t*)(&observations_point_triangulated__data[3*i]);

            // For now the triangulated observations are local observation vectors
            if(!_mrcal_unproject_internal( // out
                                           &c_observations_point_triangulated[i].px,

                                           // in
                                           (const mrcal_point2_t*)(px_weight->xyz), 1,
                                           lensmodel,
                                           &intrinsics[icam_intrinsics*Nintrinsics_state],
                                           &precomputed))
            {
                BARF("mrcal_unproject() failed");
                return -1;
            }

            c_observations_point_triangulated[i].outlier = (px_weight->z <= 0.0);
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
            if(!_finish_triangulated_set(c_observations_point_triangulated,
                                         Npoints_in_this_set,
                                         i-1))
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

    if(!_finish_triangulated_set(c_observations_point_triangulated,
                                 Npoints_in_this_set,
                                 N-1))
        return -1;

    return N;

    // Used for error checking; CHECK_LAYOUT above does "goto done" on error
 done: return -1;
}

#define PROBLEM_SELECTIONS_SET_BIT(x) .x = x,
#define CONSTRUCT_PROBLEM_SELECTIONS() ({                               \
    /* By default we optimize everything we can; these are default at <0 */                         \
    if(do_optimize_intrinsics_core        < 0) do_optimize_intrinsics_core = Ncameras_intrinsics>0; \
    if(do_optimize_intrinsics_distortions < 0) do_optimize_intrinsics_core = Ncameras_intrinsics>0; \
    if(do_optimize_extrinsics             < 0) do_optimize_extrinsics      = Ncameras_extrinsics>0; \
    if(do_optimize_frames                 < 0) do_optimize_frames          = Nframes            >0; \
    if(do_optimize_calobject_warp         < 0) do_optimize_calobject_warp  = Nobservations_board>0; \
    /* stuff not in the above if doesn't have a <0 default; those are all 0 or 1 already */          \
    (mrcal_problem_selections_t) { MRCAL_PROBLEM_SELECTIONS_LIST(PROBLEM_SELECTIONS_SET_BIT) }; \
})

// Handle legacy aliases. If they're not both defined and the new name isn't
// defined, use the old name
#define handle_renamed(name_old, name_new) \
    _handle_renamed(&name_old, #name_old,   \
                    &name_new, #name_new)
static bool _handle_renamed(PyArrayObject** old, const char* name_old,
                            PyArrayObject** new, const char* name_new)
{
    if(!IS_NULL(*new) && PyArray_SIZE(*new))
    {
        if(!IS_NULL(*old) && PyArray_SIZE(*old))
        {
            BARF("Both %s and %s are given: the former is a legacy alias for the latter",
                 name_old, name_new);
            return false;
        }
    }

    if(IS_NULL(*new))
    {
        Py_XDECREF(*new);
        *new = *old;
        *old = NULL;
    }
    return true;
}

// This is mrcal._optimization_inputs_known_keys in Python. But I'm being lazy
// and just save it as a global here
static PyObject* _optimization_inputs_known_keys_frozenset;
//  Replace the kwargs to keep only those that are known. The unknown ones that
//  are not null trigger an error. This is intended for forwards compatibility
static bool optimization_inputs_kwargs_delete_unknown(PyObject** kwargs,
                                                      bool* need_decref_kwargs)
{
    *need_decref_kwargs = false;

    if(!PyDict_Check(*kwargs))
        return true;

    PyObject* kwargs_new = PyDict_New();
    if(kwargs_new == NULL)
    {
        BARF("Couldn't make a new kwargs");
        return false;
    }

    PyObject* key;
    PyObject* value;
    Py_ssize_t i = 0;
    while (PyDict_Next(*kwargs, &i, &key, &value))
    {
        int contains = PySet_Contains(_optimization_inputs_known_keys_frozenset,
                                      key);
        if(contains < 0)
        {
            BARF("Couldn't check contents of _optimization_inputs_known_keys_frozenset");
            Py_DECREF(kwargs_new);
            return false;
        }
        if(contains)
        {
            if(0 != PyDict_SetItem(kwargs_new, key, value))
            {
                BARF("Couldn't add '%S' to kwargs_new", key);
                Py_DECREF(kwargs_new);
                return false;
            }
            continue;
        }

        // This element of kwargs is NOT known

        // If it's None or an empty array, I simply ignore it (do NOT add to
        // kwargs_new): it does nothing and is safe to ignore. Otherwise I throw
        // an error
        if( (!PyArray_Check(value) && PyObject_IsTrue(value)) ||
            ( PyArray_Check(value) && PyArray_SIZE((PyArrayObject*)value) != 0) )
        {
            BARF("optimization_inputs key '%S' has a non-null value. Unsupported in this version of mrcal", key);
            Py_DECREF(kwargs_new);
            return false;
        }

        // NULL unknown key. I simply ignore it
    }

    // Not decreading the kwargs reference. This was causing breakage. I guess
    // this is part of the function-calling machinery and I'm not supposed to
    // mess with it
    //Py_DECREF(*kwargs);

    *kwargs = kwargs_new;
    *need_decref_kwargs = true;

    return true;
}

static
PyObject* _optimize(optimizemode_t optimizemode,
                    PyObject* args,
                    PyObject* kwargs)
{
    PyObject* result = NULL;

    PyArrayObject* b_packed_final = NULL;
    PyArrayObject* x_final        = NULL;
    PyObject*      pystats        = NULL;

    PyArrayObject* P             = NULL;
    PyArrayObject* I             = NULL;
    PyArrayObject* X             = NULL;
    PyObject*      factorization = NULL;
    PyObject*      jacobian      = NULL;

    OPTIMIZE_ARGUMENTS_REQUIRED(ARG_DEFINE);
    OPTIMIZE_ARGUMENTS_OPTIONAL(ARG_DEFINE);
    OPTIMIZER_CALLBACK_ARGUMENTS_OPTIONAL_EXTRA(ARG_DEFINE);

    bool need_decref_kwargs = false;
    if(!optimization_inputs_kwargs_delete_unknown(&kwargs, &need_decref_kwargs))
        goto done;

    int calibration_object_height_n = -1;
    int calibration_object_width_n  = -1;

    SET_SIGINT();

    if(optimizemode == OPTIMIZEMODE_OPTIMIZE   ||
       optimizemode == OPTIMIZEMODE_DRTRRP_DB)
    {
        char* keywords[] = { OPTIMIZE_ARGUMENTS_REQUIRED(NAMELIST)
                             OPTIMIZE_ARGUMENTS_OPTIONAL(NAMELIST)
                             NULL};
        if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                         OPTIMIZE_ARGUMENTS_REQUIRED(PARSECODE) "|$"
                                         OPTIMIZE_ARGUMENTS_OPTIONAL(PARSECODE)
                                         ":mrcal.optimize",

                                         keywords,

                                         OPTIMIZE_ARGUMENTS_REQUIRED(PARSEARG)
                                         OPTIMIZE_ARGUMENTS_OPTIONAL(PARSEARG) NULL))
            goto done;
    }
    else if(optimizemode == OPTIMIZEMODE_CALLBACK)
    {
        char* keywords[] = { OPTIMIZE_ARGUMENTS_REQUIRED(NAMELIST)
                             OPTIMIZE_ARGUMENTS_OPTIONAL(NAMELIST)
                             OPTIMIZER_CALLBACK_ARGUMENTS_OPTIONAL_EXTRA(NAMELIST)
                             NULL};
        if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                         OPTIMIZE_ARGUMENTS_REQUIRED(PARSECODE) "|$"
                                         OPTIMIZE_ARGUMENTS_OPTIONAL(PARSECODE)
                                         OPTIMIZER_CALLBACK_ARGUMENTS_OPTIONAL_EXTRA(PARSECODE)
                                         ":mrcal.optimizer_callback",

                                         keywords,

                                         OPTIMIZE_ARGUMENTS_REQUIRED(PARSEARG)
                                         OPTIMIZE_ARGUMENTS_OPTIONAL(PARSEARG)
                                         OPTIMIZER_CALLBACK_ARGUMENTS_OPTIONAL_EXTRA(PARSEARG) NULL))
            goto done;
    }
    else
    {
        BARF("ERROR: Unknown optimizemode=%d. Giving up",
             (int)optimizemode);
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

    SET_SIZE0_IF_NONE(rt_cam_ref,      NPY_DOUBLE, 0,6);

    SET_SIZE0_IF_NONE(rt_ref_frame,                                        NPY_DOUBLE, 0,6);
    SET_SIZE0_IF_NONE(observations_board,                                     NPY_DOUBLE, 0,179,171,3); // arbitrary numbers; shouldn't matter
    SET_SIZE0_IF_NONE(indices_frame_camintrinsics_camextrinsics,              NPY_INT32,    0,3);

    SET_SIZE0_IF_NONE(points,                                                 NPY_DOUBLE, 0, 3);
    SET_SIZE0_IF_NONE(observations_point,                                     NPY_DOUBLE, 0, 3);
    SET_SIZE0_IF_NONE(indices_point_camintrinsics_camextrinsics,              NPY_INT32,  0, 3);
    SET_SIZE0_IF_NONE(observations_point_triangulated,                        NPY_DOUBLE, 0, 3);
    SET_SIZE0_IF_NONE(indices_point_triangulated_camintrinsics_camextrinsics, NPY_INT32,  0, 3);
    SET_SIZE0_IF_NONE(imagersizes,                                            NPY_INT32,  0, 2);
#undef SET_NULL_IF_NONE

    if(!handle_renamed(extrinsics_rt_fromref, rt_cam_ref))
        return false;
    if(!handle_renamed(frames_rt_toref, rt_ref_frame))
        return false;

    mrcal_lensmodel_t mrcal_lensmodel;
    // Check the arguments for optimize(). If optimizer_callback, then the other
    // stuff is defined, but it all has valid, default values
    if( !optimize_validate_args(&mrcal_lensmodel,
                                optimizemode,
                                OPTIMIZE_ARGUMENTS_REQUIRED(ARG_LIST_CALL)
                                OPTIMIZE_ARGUMENTS_OPTIONAL(ARG_LIST_CALL)
                                OPTIMIZER_CALLBACK_ARGUMENTS_OPTIONAL_EXTRA(ARG_LIST_CALL)
                                NULL))
        goto done;

    // Can't compute a factorization without a jacobian. That's what we're factoring
    if(!no_factorization)
        no_jacobian = false;



    {
        int Ncameras_intrinsics = PyArray_DIMS(intrinsics)[0];
        int Ncameras_extrinsics = PyArray_DIMS(rt_cam_ref)[0];
        int Nframes             = PyArray_DIMS(rt_ref_frame)[0];
        int Npoints             = PyArray_DIMS(points)[0];
        int Nobservations_board = PyArray_DIMS(observations_board)[0];
        int Nobservations_point = PyArray_DIMS(observations_point)[0];

        if( Nobservations_board > 0 )
        {
            calibration_object_height_n = PyArray_DIMS(observations_board)[1];
            calibration_object_width_n  = PyArray_DIMS(observations_board)[2];
        }

        // The checks in optimize_validate_args() make sure these casts are kosher
        double*             c_intrinsics     = (double*)  PyArray_DATA(intrinsics);
        mrcal_pose_t*       c_extrinsics     = (mrcal_pose_t*)  PyArray_DATA(rt_cam_ref);
        mrcal_pose_t*       c_frames         = (mrcal_pose_t*)  PyArray_DATA(rt_ref_frame);
        mrcal_point3_t*     c_points         = (mrcal_point3_t*)PyArray_DATA(points);
        mrcal_calobject_warp_t*     c_calobject_warp =
            IS_NULL(calobject_warp) ?
            NULL : (mrcal_calobject_warp_t*)PyArray_DATA(calobject_warp);



        // Is contiguous; I made sure above
        mrcal_point3_t* c_observations_board_pool = (mrcal_point3_t*)PyArray_DATA(observations_board);
        mrcal_observation_board_t c_observations_board[Nobservations_board];
        fill_c_observations_board(// output
                                  c_observations_board,
                                  // input
                                  Nobservations_board,
                                  indices_frame_camintrinsics_camextrinsics);

        // Is contiguous; I made sure above
        mrcal_point3_t* c_observations_point_pool = (mrcal_point3_t*)PyArray_DATA(observations_point);
        mrcal_observation_point_t c_observations_point[Nobservations_point];
        fill_c_observations_point(// output
                                  c_observations_point,
                                  // input
                                  Nobservations_point,
                                  indices_point_camintrinsics_camextrinsics);

        int Nobservations_point_triangulated = PyArray_DIMS(observations_point_triangulated)[0];
        mrcal_observation_point_triangulated_t c_observations_point_triangulated[Nobservations_point_triangulated];
        if( fill_c_observations_point_triangulated(c_observations_point_triangulated,
                                                   observations_point_triangulated,
                                                   &mrcal_lensmodel, c_intrinsics,
                                                   indices_point_triangulated_camintrinsics_camextrinsics)
            < 0 )
        {
            goto done;
        }


        mrcal_problem_selections_t problem_selections = CONSTRUCT_PROBLEM_SELECTIONS();


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
        b_packed_final = (PyArrayObject*)PyArray_SimpleNew(1, ((npy_intp[]){Nstate}), NPY_DOUBLE);
        double* c_b_packed_final = PyArray_DATA(b_packed_final);

        x_final = (PyArrayObject*)PyArray_SimpleNew(1, ((npy_intp[]){Nmeasurements}), NPY_DOUBLE);
        double* c_x_final = PyArray_DATA(x_final);

        if(optimizemode == OPTIMIZEMODE_OPTIMIZE)
        {
            // we're wrapping mrcal_optimize()
            const int Npoints_fromBoards =
                Nobservations_board *
                calibration_object_width_n*calibration_object_height_n;

            mrcal_stats_t stats =
                mrcal_optimize( c_b_packed_final,
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
                                c_observations_point_pool,

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

            if( 0 != PyDict_SetItemString(pystats, "b_packed",
                                          (PyObject*)b_packed_final) )
            {
                BARF("Couldn't add to stats dict 'b_packed'");
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
        else if(optimizemode == OPTIMIZEMODE_CALLBACK ||
                optimizemode == OPTIMIZEMODE_DRTRRP_DB)
        {
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
                                         c_b_packed_final,
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
                                         c_observations_point_pool,

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

            if(optimizemode == OPTIMIZEMODE_CALLBACK)
            {
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
                                      (PyObject*)b_packed_final,
                                      (PyObject*)x_final,
                                      jacobian,
                                      factorization);
            }
            else
            {
                // OPTIMIZEMODE_DRTRRP_DB
                const int state_index_frame0 =
                    mrcal_state_index_frames(0,
                                             Ncameras_intrinsics, Ncameras_extrinsics,
                                             Nframes,
                                             Npoints, Npoints_fixed, Nobservations_board,
                                             problem_selections,
                                             &mrcal_lensmodel);
                const int state_index_point0 =
                    mrcal_state_index_points(0,
                                             Ncameras_intrinsics, Ncameras_extrinsics,
                                             Nframes,
                                             Npoints, Npoints_fixed, Nobservations_board,
                                             problem_selections,
                                             &mrcal_lensmodel);
                const int state_index_calobject_warp0 =
                    mrcal_state_index_calobject_warp(Ncameras_intrinsics, Ncameras_extrinsics,
                                                     Nframes,
                                                     Npoints, Npoints_fixed, Nobservations_board,
                                                     problem_selections,
                                                     &mrcal_lensmodel);

                // _mrcal_drt_ref_refperturbed__dbpacked() returns an array of
                // shape (6,Nstate_noi_noe). I eventually want to use each of
                // its rows to solve a linear system using the big cholesky
                // factorization: factorization.solve_xt_JtJ_bt(K). This uses
                // CHOLMOD internally. CHOLMOD has no good API interface to use
                // a subset of the state vector for its RHS (Nstate_noi_noe
                // instead of Nstate). I can pass in a sparsity pattern, but
                // that feels like it wouldn't win me anything. So I construct
                // and use a full K, filling the unused entries with 0
                PyObject* K =
                    PyArray_ZEROS(2, ((npy_intp[]){6,Nstate}), NPY_DOUBLE, 0);
                if(K == NULL)
                {
                    BARF("Couldn't allocate K");
                    goto done;
                }
                if(!PyArray_IS_C_CONTIGUOUS((PyArrayObject*)K))
                {
                    BARF("New array K should be contiguous");
                    Py_DECREF(K);
                    goto done;
                }

                const npy_intp* strides = PyArray_STRIDES((PyArrayObject*)K);

                if(!_mrcal_drt_ref_refperturbed__dbpacked(// output
                                                         state_index_frame0 >= 0 ?
                                                         &((double*)(PyArray_DATA((PyArrayObject*)K)))[state_index_frame0] : NULL,
                                                         (int)strides[0],
                                                         (int)strides[1],

                                                         state_index_point0 >= 0 ?
                                                         &((double*)(PyArray_DATA((PyArrayObject*)K)))[state_index_point0] : NULL,
                                                         (int)strides[0],
                                                         (int)strides[1],

                                                         state_index_calobject_warp0 >= 0 ?
                                                         &((double*)(PyArray_DATA((PyArrayObject*)K)))[state_index_calobject_warp0] : NULL,
                                                         (int)strides[0],
                                                         (int)strides[1],

                                                         c_b_packed_final, Nstate*sizeof(double),
                                                         &Jt,

                                                         Ncameras_intrinsics, Ncameras_extrinsics, Nframes,
                                                         Npoints, Npoints_fixed,
                                                         Nobservations_board,
                                                         Nobservations_point,
                                                         &mrcal_lensmodel,
                                                         problem_selections,

                                                         calibration_object_width_n,
                                                         calibration_object_height_n))
                {
                    BARF("_mrcal_drt_ref_refperturbed__dbpacked() failed");
                    Py_DECREF(K);
                    goto done;
                }

                result = K;
            }
        }
        else
        {
            BARF("ERROR: Unknown optimizemode=%d. Giving up",
                 (int)optimizemode);
            goto done;
        }
    }

 done:
    if(need_decref_kwargs)
        Py_DECREF(kwargs);
    OPTIMIZE_ARGUMENTS_REQUIRED(FREE_PYARRAY);
    OPTIMIZE_ARGUMENTS_OPTIONAL(FREE_PYARRAY);
    OPTIMIZER_CALLBACK_ARGUMENTS_OPTIONAL_EXTRA(FREE_PYARRAY);

    Py_XDECREF(b_packed_final);
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
    return _optimize(OPTIMIZEMODE_CALLBACK, args, kwargs);
}
static PyObject* optimize(PyObject* NPY_UNUSED(self),
                          PyObject* args,
                          PyObject* kwargs)
{
    return _optimize(OPTIMIZEMODE_OPTIMIZE,
                     args, kwargs);
}
static PyObject* drt_ref_refperturbed__dbpacked(PyObject* NPY_UNUSED(self),
                                                PyObject* args,
                                                PyObject* kwargs)
{
    return _optimize(OPTIMIZEMODE_DRTRRP_DB, args, kwargs);
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
//
// The callbacks return the Python object that will be returned. A callback
// should indicate an error by calling PyErr_...() as usual. If a callback
// returns NULL without setting an error, we return None from Python
typedef PyObject* (callback_state_index_t)(int i,
                                           int Ncameras_intrinsics,
                                           int Ncameras_extrinsics,
                                           int Nframes,
                                           int Npoints,
                                           int Npoints_fixed,
                                           int Nobservations_board,
                                           int Nobservations_point,
                                           int calibration_object_width_n,
                                           int calibration_object_height_n,
                                           const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
                                           const PyArrayObject* indices_point_camintrinsics_camextrinsics,
                                           const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                           const PyArrayObject* observations_point,
                                           const mrcal_lensmodel_t* lensmodel,
                                           mrcal_problem_selections_t problem_selections);
#define STATE_INDEX_GENERIC(f, ...) state_index_generic(callback_ ## f, \
                                                        #f,             \
                                                        __VA_ARGS__ )
#if defined ENABLE_TRIANGULATED_WARNINGS && ENABLE_TRIANGULATED_WARNINGS
#warning "triangulated-solve: more kwargs for the triangulated-solve measurements? Look in state_index_generic()"
#endif
static PyObject* state_index_generic(callback_state_index_t cb,
                                     const char* called_function,
                                     PyObject* self, PyObject* args, PyObject* kwargs,
                                     bool need_lensmodel,
                                     const char* argname)
{
    // This is VERY similar to _pack_unpack_state(). Please consolidate
    // Also somewhat similar to _optimize()

    PyObject* result = NULL;

    OPTIMIZE_ARGUMENTS_REQUIRED(ARG_DEFINE);
    OPTIMIZE_ARGUMENTS_OPTIONAL(ARG_DEFINE);

    bool need_decref_kwargs = false;
    if(!optimization_inputs_kwargs_delete_unknown(&kwargs, &need_decref_kwargs))
        goto done;

    int i = -1;

    int Ncameras_intrinsics = -1;
    int Ncameras_extrinsics = -1;
    int Nframes             = -1;
    int Npoints             = -1;
    int Nobservations_board = -1;
    int Nobservations_point = -1;

    char* keywords[] = { (char*)argname,
                         "Ncameras_intrinsics",
                         "Ncameras_extrinsics",
                         "Nframes",
                         "Npoints",
                         "Nobservations_board",
                         "Nobservations_point",
                         OPTIMIZE_ARGUMENTS_REQUIRED(NAMELIST)
                         OPTIMIZE_ARGUMENTS_OPTIONAL(NAMELIST)
                         NULL};

    // needs to be big-enough to store the largest-possible called_function
#define CALLED_FUNCTION_BUFFER "123456789012345678901234567890123456789012345678901234567890"
    char arg_string[] =
        "i"
        "|$" // everything is kwarg-only and optional. I apply logic down the
             // line to get what I need
        "iiiiii"
        OPTIMIZE_ARGUMENTS_REQUIRED(PARSECODE)
        OPTIMIZE_ARGUMENTS_OPTIONAL(PARSECODE)
        ":mrcal." CALLED_FUNCTION_BUFFER;
    if(strlen(CALLED_FUNCTION_BUFFER) < strlen(called_function))
    {
        BARF("CALLED_FUNCTION_BUFFER too small for '%s'. This is a a bug", called_function);
        goto done;
    }
    arg_string[strlen(arg_string) - strlen(CALLED_FUNCTION_BUFFER)] = '\0';
    strcat(arg_string, called_function);


    if(argname != NULL)
    {
        if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                         arg_string,
                                         keywords,

                                         &i,
                                         &Ncameras_intrinsics,
                                         &Ncameras_extrinsics,
                                         &Nframes,
                                         &Npoints,
                                         &Nobservations_board,
                                         &Nobservations_point,
                                         OPTIMIZE_ARGUMENTS_REQUIRED(PARSEARG)
                                         OPTIMIZE_ARGUMENTS_OPTIONAL(PARSEARG) NULL))
            goto done;
    }
    else
    {
        if(!PyArg_ParseTupleAndKeywords( args, kwargs,

                                         // skip the initial "i". There is no "argname" here
                                         &arg_string[1],
                                         &keywords  [1],

                                         &Ncameras_intrinsics,
                                         &Ncameras_extrinsics,
                                         &Nframes,
                                         &Npoints,
                                         &Nobservations_board,
                                         &Nobservations_point,
                                         OPTIMIZE_ARGUMENTS_REQUIRED(PARSEARG)
                                         OPTIMIZE_ARGUMENTS_OPTIONAL(PARSEARG) NULL))
            goto done;
    }
#undef CALLED_FUNCTION_BUFFER

    if(!handle_renamed(extrinsics_rt_fromref, rt_cam_ref))
        return false;
    if(!handle_renamed(frames_rt_toref, rt_ref_frame))
        return false;

    mrcal_lensmodel_t mrcal_lensmodel = {};

    if(need_lensmodel)
    {
        if(lensmodel == NULL)
        {
            BARF("The 'lensmodel' argument is required");
            goto done;
        }
        if(!parse_lensmodel_from_arg(&mrcal_lensmodel, lensmodel))
            goto done;
    }

    // checks dimensionality of array !IS_NULL. So if any array isn't passed-in,
    // that's OK! After I do this and if !IS_NULL, then I can ask for array
    // dimensions safely
    OPTIMIZE_ARGUMENTS_REQUIRED(CHECK_LAYOUT);
    OPTIMIZE_ARGUMENTS_OPTIONAL(CHECK_LAYOUT);

    // If explicit dimensions are given, use them. If they're not given, but we
    // have an array, use those dimensions. If an array isn't given either, use
    // 0
    if(Ncameras_intrinsics < 0) Ncameras_intrinsics = IS_NULL(intrinsics)            ? 0 : PyArray_DIMS(intrinsics)            [0];
    if(Ncameras_extrinsics < 0) Ncameras_extrinsics = IS_NULL(rt_cam_ref)            ? 0 : PyArray_DIMS(rt_cam_ref)            [0];
    if(Nframes < 0)             Nframes             = IS_NULL(rt_ref_frame)          ? 0 : PyArray_DIMS(rt_ref_frame)          [0];
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


    mrcal_problem_selections_t problem_selections = CONSTRUCT_PROBLEM_SELECTIONS();

    result = cb(i,
                Ncameras_intrinsics,
                Ncameras_extrinsics,
                Nframes,
                Npoints,
                Npoints_fixed,
                Nobservations_board,
                Nobservations_point,
                calibration_object_width_n,
                calibration_object_height_n,
                indices_frame_camintrinsics_camextrinsics,
                indices_point_camintrinsics_camextrinsics,
                indices_point_triangulated_camintrinsics_camextrinsics,
                observations_point,
                &mrcal_lensmodel,
                problem_selections);

    // If an error is set I return it. result SHOULD be NULL, but if it isn't, I
    // release it.
    if(result != NULL && PyErr_Occurred())
    {
        Py_DECREF(result);
        result = NULL;
    }
    // A callback returning NULL without setting an error indicates that we
    // should return None
    else if(result == NULL && !PyErr_Occurred())
    {
        result = Py_None;
        Py_INCREF(result);
    }

    if(result == NULL)
        // error
        goto done;

 done:
    if(need_decref_kwargs)
        Py_DECREF(kwargs);
    OPTIMIZE_ARGUMENTS_REQUIRED(FREE_PYARRAY) ;
    OPTIMIZE_ARGUMENTS_OPTIONAL(FREE_PYARRAY) ;

    return result;
}

static PyObject* callback_state_index_intrinsics(int i,
                                           int Ncameras_intrinsics,
                                           int Ncameras_extrinsics,
                                           int Nframes,
                                           int Npoints,
                                           int Npoints_fixed,
                                           int Nobservations_board,
                                           int Nobservations_point,
                                           int calibration_object_width_n,
                                           int calibration_object_height_n,
                                           const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
                                           const PyArrayObject* indices_point_camintrinsics_camextrinsics,
                                           const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                           const PyArrayObject* observations_point,
                                           const mrcal_lensmodel_t* lensmodel,
                                           mrcal_problem_selections_t problem_selections)
{
    int index =
        mrcal_state_index_intrinsics(i,
                                     Ncameras_intrinsics, Ncameras_extrinsics,
                                     Nframes,
                                     Npoints, Npoints_fixed, Nobservations_board,
                                     problem_selections,
                                     lensmodel);

    if(index >= 0)
        return PyLong_FromLong(index);

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* state_index_intrinsics(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(state_index_intrinsics,
                               self, args, kwargs,
                               true,
                               "icam_intrinsics");
}

static PyObject* callback_num_states_intrinsics(int i,
                                          int Ncameras_intrinsics,
                                          int Ncameras_extrinsics,
                                          int Nframes,
                                          int Npoints,
                                          int Npoints_fixed,
                                          int Nobservations_board,
                                          int Nobservations_point,
                                          int calibration_object_width_n,
                                          int calibration_object_height_n,
                                          const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
                                          const PyArrayObject* indices_point_camintrinsics_camextrinsics,
                                          const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                          const PyArrayObject* observations_point,
                                          const mrcal_lensmodel_t* lensmodel,
                                          mrcal_problem_selections_t problem_selections)
{
    int index =
        mrcal_num_states_intrinsics(Ncameras_intrinsics,
                                    problem_selections, lensmodel);
    if(index >= 0)
        return PyLong_FromLong(index);

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* num_states_intrinsics(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(num_states_intrinsics,
                               self, args, kwargs,
                               true,
                               NULL);
}

static PyObject* callback_state_index_extrinsics(int i,
                                           int Ncameras_intrinsics,
                                           int Ncameras_extrinsics,
                                           int Nframes,
                                           int Npoints,
                                           int Npoints_fixed,
                                           int Nobservations_board,
                                           int Nobservations_point,
                                           int calibration_object_width_n,
                                           int calibration_object_height_n,
                                           const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
                                           const PyArrayObject* indices_point_camintrinsics_camextrinsics,
                                           const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                           const PyArrayObject* observations_point,
                                           const mrcal_lensmodel_t* lensmodel,
                                           mrcal_problem_selections_t problem_selections)
{
    int index =
        mrcal_state_index_extrinsics(i,
                                     Ncameras_intrinsics, Ncameras_extrinsics,
                                     Nframes,
                                     Npoints, Npoints_fixed, Nobservations_board,
                                     problem_selections,
                                     lensmodel);
    if(index >= 0)
        return PyLong_FromLong(index);

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* state_index_extrinsics(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(state_index_extrinsics,
                               self, args, kwargs,
                               true,
                               "icam_extrinsics");
}

static PyObject* callback_num_states_extrinsics(int i,
                                          int Ncameras_intrinsics,
                                          int Ncameras_extrinsics,
                                          int Nframes,
                                          int Npoints,
                                          int Npoints_fixed,
                                          int Nobservations_board,
                                          int Nobservations_point,
                                          int calibration_object_width_n,
                                          int calibration_object_height_n,
                                          const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
                                          const PyArrayObject* indices_point_camintrinsics_camextrinsics,
                                          const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                          const PyArrayObject* observations_point,
                                          const mrcal_lensmodel_t* lensmodel,
                                          mrcal_problem_selections_t problem_selections)
{
    int index =
        mrcal_num_states_extrinsics(Ncameras_extrinsics, problem_selections);
    if(index >= 0)
        return PyLong_FromLong(index);

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* num_states_extrinsics(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(num_states_extrinsics,
                               self, args, kwargs,
                               false,
                               NULL);
}

static PyObject* callback_state_index_frames(int i,
                                       int Ncameras_intrinsics,
                                       int Ncameras_extrinsics,
                                       int Nframes,
                                       int Npoints,
                                       int Npoints_fixed,
                                       int Nobservations_board,
                                       int Nobservations_point,
                                       int calibration_object_width_n,
                                       int calibration_object_height_n,
                                       const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
                                       const PyArrayObject* indices_point_camintrinsics_camextrinsics,
                                       const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                       const PyArrayObject* observations_point,
                                       const mrcal_lensmodel_t* lensmodel,
                                       mrcal_problem_selections_t problem_selections)
{
    int index =
        mrcal_state_index_frames(i,
                                 Ncameras_intrinsics, Ncameras_extrinsics,
                                 Nframes,
                                 Npoints, Npoints_fixed, Nobservations_board,
                                 problem_selections,
                                 lensmodel);
    if(index >= 0)
        return PyLong_FromLong(index);

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* state_index_frames(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(state_index_frames,
                               self, args, kwargs,
                               true,
                               "iframe");
}

static PyObject* callback_num_states_frames(int i,
                                      int Ncameras_intrinsics,
                                      int Ncameras_extrinsics,
                                      int Nframes,
                                      int Npoints,
                                      int Npoints_fixed,
                                      int Nobservations_board,
                                      int Nobservations_point,
                                      int calibration_object_width_n,
                                      int calibration_object_height_n,
                                      const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
                                      const PyArrayObject* indices_point_camintrinsics_camextrinsics,
                                      const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                      const PyArrayObject* observations_point,
                                      const mrcal_lensmodel_t* lensmodel,
                                      mrcal_problem_selections_t problem_selections)
{
    int index =
        mrcal_num_states_frames(Nframes, problem_selections);
    if(index >= 0)
        return PyLong_FromLong(index);

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* num_states_frames(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(num_states_frames,
                               self, args, kwargs,
                               false,
                               NULL);
}

static PyObject* callback_state_index_points(int i,
                                       int Ncameras_intrinsics,
                                       int Ncameras_extrinsics,
                                       int Nframes,
                                       int Npoints,
                                       int Npoints_fixed,
                                       int Nobservations_board,
                                       int Nobservations_point,
                                       int calibration_object_width_n,
                                       int calibration_object_height_n,
                                       const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
                                       const PyArrayObject* indices_point_camintrinsics_camextrinsics,
                                       const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                       const PyArrayObject* observations_point,
                                       const mrcal_lensmodel_t* lensmodel,
                                       mrcal_problem_selections_t problem_selections)
{
    int index =
        mrcal_state_index_points(i,
                                 Ncameras_intrinsics, Ncameras_extrinsics,
                                 Nframes,
                                 Npoints, Npoints_fixed, Nobservations_board,
                                 problem_selections,
                                 lensmodel);
    if(index >= 0)
        return PyLong_FromLong(index);

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* state_index_points(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(state_index_points,
                               self, args, kwargs,
                               true,
                               "i_point");
}

static PyObject* callback_num_states_points(int i,
                                       int Ncameras_intrinsics,
                                       int Ncameras_extrinsics,
                                       int Nframes,
                                       int Npoints,
                                       int Npoints_fixed,
                                       int Nobservations_board,
                                       int Nobservations_point,
                                       int calibration_object_width_n,
                                       int calibration_object_height_n,
                                       const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
                                       const PyArrayObject* indices_point_camintrinsics_camextrinsics,
                                       const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                       const PyArrayObject* observations_point,
                                       const mrcal_lensmodel_t* lensmodel,
                                       mrcal_problem_selections_t problem_selections)
{
    int index =
        mrcal_num_states_points(Npoints, Npoints_fixed, problem_selections);
    if(index >= 0)
        return PyLong_FromLong(index);

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* num_states_points(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(num_states_points,
                               self, args, kwargs,
                               false,
                               NULL);
}

static PyObject* callback_state_index_calobject_warp(int i,
                                               int Ncameras_intrinsics,
                                               int Ncameras_extrinsics,
                                               int Nframes,
                                               int Npoints,
                                               int Npoints_fixed,
                                               int Nobservations_board,
                                               int Nobservations_point,
                                               int calibration_object_width_n,
                                               int calibration_object_height_n,
                                               const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
                                               const PyArrayObject* indices_point_camintrinsics_camextrinsics,
                                               const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                               const PyArrayObject* observations_point,
                                               const mrcal_lensmodel_t* lensmodel,
                                               mrcal_problem_selections_t problem_selections)
{
    int index =
        mrcal_state_index_calobject_warp( Ncameras_intrinsics, Ncameras_extrinsics,
                                          Nframes,
                                          Npoints, Npoints_fixed, Nobservations_board,
                                          problem_selections,
                                          lensmodel);
    if(index >= 0)
        return PyLong_FromLong(index);

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* state_index_calobject_warp(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(state_index_calobject_warp,
                               self, args, kwargs,
                               true,
                               NULL);
}

static PyObject* callback_num_states_calobject_warp(int i,
                                              int Ncameras_intrinsics,
                                              int Ncameras_extrinsics,
                                              int Nframes,
                                              int Npoints,
                                              int Npoints_fixed,
                                              int Nobservations_board,
                                              int Nobservations_point,
                                              int calibration_object_width_n,
                                              int calibration_object_height_n,
                                              const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
                                              const PyArrayObject* indices_point_camintrinsics_camextrinsics,
                                              const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                              const PyArrayObject* observations_point,
                                              const mrcal_lensmodel_t* lensmodel,
                                              mrcal_problem_selections_t problem_selections)
{
    int index =
        mrcal_num_states_calobject_warp(problem_selections, Nobservations_board);
    if(index >= 0)
        return PyLong_FromLong(index);

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* num_states_calobject_warp(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(num_states_calobject_warp,
                               self, args, kwargs,
                               false,
                               NULL);
}

static PyObject* callback_num_states(int i,
                               int Ncameras_intrinsics,
                               int Ncameras_extrinsics,
                               int Nframes,
                               int Npoints,
                               int Npoints_fixed,
                               int Nobservations_board,
                               int Nobservations_point,
                               int calibration_object_width_n,
                               int calibration_object_height_n,
                               const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
                               const PyArrayObject* indices_point_camintrinsics_camextrinsics,
                               const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                               const PyArrayObject* observations_point,
                               const mrcal_lensmodel_t* lensmodel,
                               mrcal_problem_selections_t problem_selections)
{
    int index =
        mrcal_num_states(Ncameras_intrinsics, Ncameras_extrinsics,
                         Nframes, Npoints, Npoints_fixed, Nobservations_board,
                         problem_selections,
                         lensmodel);
    if(index >= 0)
        return PyLong_FromLong(index);

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* num_states(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(num_states,
                               self, args, kwargs,
                               true,
                               NULL);
}

static PyObject* callback_num_intrinsics_optimization_params(int i,
                               int Ncameras_intrinsics,
                               int Ncameras_extrinsics,
                               int Nframes,
                               int Npoints,
                               int Npoints_fixed,
                               int Nobservations_board,
                               int Nobservations_point,
                               int calibration_object_width_n,
                               int calibration_object_height_n,
                               const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
                               const PyArrayObject* indices_point_camintrinsics_camextrinsics,
                               const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                               const PyArrayObject* observations_point,
                               const mrcal_lensmodel_t* lensmodel,
                               mrcal_problem_selections_t problem_selections)
{
    int index =
        mrcal_num_intrinsics_optimization_params(problem_selections,
                                                 lensmodel);
    if(index >= 0)
        return PyLong_FromLong(index);

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* num_intrinsics_optimization_params(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(num_intrinsics_optimization_params,
                               self, args, kwargs,
                               true,
                               NULL);
}

static PyObject* callback_measurement_index_boards(int i,
                                             int Ncameras_intrinsics,
                                             int Ncameras_extrinsics,
                                             int Nframes,
                                             int Npoints,
                                             int Npoints_fixed,
                                             int Nobservations_board,
                                             int Nobservations_point,
                                             int calibration_object_width_n,
                                             int calibration_object_height_n,
                                             const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
                                             const PyArrayObject* indices_point_camintrinsics_camextrinsics,
                                             const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                             const PyArrayObject* observations_point,
                                             const mrcal_lensmodel_t* lensmodel,
                                             mrcal_problem_selections_t problem_selections)
{
    int index = -1;

    if(calibration_object_width_n  > 0 &&
       calibration_object_height_n > 0)
        index =
            mrcal_measurement_index_boards(i,
                                           Nobservations_board,
                                           Nobservations_point,
                                           calibration_object_width_n,
                                           calibration_object_height_n);
    if(index >= 0)
        return PyLong_FromLong(index);

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* measurement_index_boards(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(measurement_index_boards,
                               self, args, kwargs,
                               false,
                               "i_observation_board");
}

static PyObject* callback_num_measurements_boards(int i,
                                            int Ncameras_intrinsics,
                                            int Ncameras_extrinsics,
                                            int Nframes,
                                            int Npoints,
                                            int Npoints_fixed,
                                            int Nobservations_board,
                                            int Nobservations_point,
                                            int calibration_object_width_n,
                                            int calibration_object_height_n,
                                            const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
                                            const PyArrayObject* indices_point_camintrinsics_camextrinsics,
                                            const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                            const PyArrayObject* observations_point,
                                            const mrcal_lensmodel_t* lensmodel,
                                            mrcal_problem_selections_t problem_selections)
{
    int index = 0;

    if(calibration_object_width_n  > 0 &&
       calibration_object_height_n > 0)
        index =
            mrcal_num_measurements_boards(Nobservations_board,
                                          calibration_object_width_n,
                                          calibration_object_height_n);
    if(index >= 0)
        return PyLong_FromLong(index);

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* num_measurements_boards(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(num_measurements_boards,
                               self, args, kwargs,
                               false,
                               NULL);
}

static PyObject* callback_measurement_index_points(int i,
                                             int Ncameras_intrinsics,
                                             int Ncameras_extrinsics,
                                             int Nframes,
                                             int Npoints,
                                             int Npoints_fixed,
                                             int Nobservations_board,
                                             int Nobservations_point,
                                             int calibration_object_width_n,
                                             int calibration_object_height_n,
                                             const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
                                             const PyArrayObject* indices_point_camintrinsics_camextrinsics,
                                             const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                             const PyArrayObject* observations_point,
                                             const mrcal_lensmodel_t* lensmodel,
                                             mrcal_problem_selections_t problem_selections)
{
    int index =
        mrcal_measurement_index_points(i,
                                       Nobservations_board,
                                       Nobservations_point,
                                       calibration_object_width_n,
                                       calibration_object_height_n);
    if(index >= 0)
        return PyLong_FromLong(index);

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* measurement_index_points(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(measurement_index_points,
                               self, args, kwargs,
                               false,
                               "i_observation_point");
}

static PyObject* callback_measurement_index_points_triangulated(int i,
                                                          int Ncameras_intrinsics,
                                                          int Ncameras_extrinsics,
                                                          int Nframes,
                                                          int Npoints,
                                                          int Npoints_fixed,
                                                          int Nobservations_board,
                                                          int Nobservations_point,
                                                          int calibration_object_width_n,
                                                          int calibration_object_height_n,
                                                          const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
                                                          const PyArrayObject* indices_point_camintrinsics_camextrinsics,
                                                          const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                                          const PyArrayObject* observations_point,
                                                          const mrcal_lensmodel_t* lensmodel,
                                                          mrcal_problem_selections_t problem_selections)
{
    // VERY similar to callback_num_measurements_points_triangulated() and
    // callback_measurement_index_regularization() and maybe others. Please
    // consolidate
    int N = 0;
    if(!IS_NULL(indices_point_triangulated_camintrinsics_camextrinsics))
        N = PyArray_DIM(indices_point_triangulated_camintrinsics_camextrinsics, 0);

    mrcal_observation_point_triangulated_t c_observations_point_triangulated[N];

    int Nobservations_point_triangulated =
        N <= 0 ? 0 :
        fill_c_observations_point_triangulated(c_observations_point_triangulated,
                                               NULL, NULL, NULL,
                                               indices_point_triangulated_camintrinsics_camextrinsics);
    if(Nobservations_point_triangulated < 0)
    {
        BARF("Error parsing triangulated points");
        return NULL;
    }

    int index =
        mrcal_measurement_index_points_triangulated(i,
                                                    Nobservations_board,
                                                    Nobservations_point,
                                                    c_observations_point_triangulated,
                                                    Nobservations_point_triangulated,
                                                    calibration_object_width_n,
                                                    calibration_object_height_n);
    if(index >= 0)
        return PyLong_FromLong(index);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject* measurement_index_points_triangulated(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(measurement_index_points_triangulated,
                               self, args, kwargs,
                               false,
                               "i_point_triangulated");
}

static PyObject* callback_num_measurements_points(int i,
                                            int Ncameras_intrinsics,
                                            int Ncameras_extrinsics,
                                            int Nframes,
                                            int Npoints,
                                            int Npoints_fixed,
                                            int Nobservations_board,
                                            int Nobservations_point,
                                            int calibration_object_width_n,
                                            int calibration_object_height_n,
                                            const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
                                            const PyArrayObject* indices_point_camintrinsics_camextrinsics,
                                            const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                            const PyArrayObject* observations_point,
                                            const mrcal_lensmodel_t* lensmodel,
                                            mrcal_problem_selections_t problem_selections)
{
    int index =
        mrcal_num_measurements_points(Nobservations_point);
    if(index >= 0)
        return PyLong_FromLong(index);

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* num_measurements_points(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(num_measurements_points,
                               self, args, kwargs,
                               false,
                               NULL);
}

static PyObject* callback_num_measurements_points_triangulated(int i,
                                                         int Ncameras_intrinsics,
                                                         int Ncameras_extrinsics,
                                                         int Nframes,
                                                         int Npoints,
                                                         int Npoints_fixed,
                                                         int Nobservations_board,
                                                         int Nobservations_point,
                                                         int calibration_object_width_n,
                                                         int calibration_object_height_n,
                                                         const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
                                                         const PyArrayObject* indices_point_camintrinsics_camextrinsics,
                                                         const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                                         const PyArrayObject* observations_point,
                                                         const mrcal_lensmodel_t* lensmodel,
                                                         mrcal_problem_selections_t problem_selections)
{
    // VERY similar to callback_measurement_index_regularization(). Please
    // consolidate
    int N = 0;
    if(!IS_NULL(indices_point_triangulated_camintrinsics_camextrinsics))
        N = PyArray_DIM(indices_point_triangulated_camintrinsics_camextrinsics, 0);

    if(N == 0)
        return PyLong_FromLong(0);

    mrcal_observation_point_triangulated_t c_observations_point_triangulated[N];

    int Nobservations_point_triangulated =
        fill_c_observations_point_triangulated(c_observations_point_triangulated,
                                               NULL, NULL, NULL,
                                               indices_point_triangulated_camintrinsics_camextrinsics);
    if(Nobservations_point_triangulated < 0)
    {
        BARF("Error parsing triangulated points");
        return NULL;
    }
    int index =
        mrcal_num_measurements_points_triangulated(c_observations_point_triangulated,
                                                   Nobservations_point_triangulated);
    if(index >= 0)
        return PyLong_FromLong(index);

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* num_measurements_points_triangulated(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(num_measurements_points_triangulated,
                               self, args, kwargs,
                               false,
                               NULL);
}

static PyObject* callback_measurement_index_regularization(int i,
                                                     int Ncameras_intrinsics,
                                                     int Ncameras_extrinsics,
                                                     int Nframes,
                                                     int Npoints,
                                                     int Npoints_fixed,
                                                     int Nobservations_board,
                                                     int Nobservations_point,
                                                     int calibration_object_width_n,
                                                     int calibration_object_height_n,
                                                     const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
                                                     const PyArrayObject* indices_point_camintrinsics_camextrinsics,
                                                     const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                                     const PyArrayObject* observations_point,
                                                     const mrcal_lensmodel_t* lensmodel,
                                                     mrcal_problem_selections_t problem_selections)
{
    // VERY similar to callback_num_measurements_points_triangulated(). Please
    // consolidate
    int N = 0;
    if(!IS_NULL(indices_point_triangulated_camintrinsics_camextrinsics))
        N = PyArray_DIM(indices_point_triangulated_camintrinsics_camextrinsics, 0);

    mrcal_observation_point_triangulated_t c_observations_point_triangulated[N];

    int Nobservations_point_triangulated =
        N <= 0 ? 0 :
        fill_c_observations_point_triangulated(c_observations_point_triangulated,
                                               NULL, NULL, NULL,
                                               indices_point_triangulated_camintrinsics_camextrinsics);
    if(Nobservations_point_triangulated < 0)
    {
        BARF("Error parsing triangulated points");
        return NULL;
    }
    int index =
        mrcal_measurement_index_regularization(c_observations_point_triangulated,
                                               Nobservations_point_triangulated,
                                               calibration_object_width_n,
                                               calibration_object_height_n,
                                               Ncameras_intrinsics, Ncameras_extrinsics,
                                               Nframes,
                                               Npoints, Npoints_fixed, Nobservations_board, Nobservations_point,
                                               problem_selections,
                                               lensmodel);
    if(index >= 0)
        return PyLong_FromLong(index);

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* measurement_index_regularization(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(measurement_index_regularization,
                               self, args, kwargs,
                               true,
                               NULL);
}

static PyObject* callback_num_measurements_regularization(int i,
                                                    int Ncameras_intrinsics,
                                                    int Ncameras_extrinsics,
                                                    int Nframes,
                                                    int Npoints,
                                                    int Npoints_fixed,
                                                    int Nobservations_board,
                                                    int Nobservations_point,
                                                    int calibration_object_width_n,
                                                    int calibration_object_height_n,
                                                    const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
                                                    const PyArrayObject* indices_point_camintrinsics_camextrinsics,
                                                    const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                                    const PyArrayObject* observations_point,
                                                    const mrcal_lensmodel_t* lensmodel,
                                                    mrcal_problem_selections_t problem_selections)
{
    int index =
        mrcal_num_measurements_regularization(Ncameras_intrinsics, Ncameras_extrinsics,
                                              Nframes,
                                              Npoints, Npoints_fixed, Nobservations_board,
                                              problem_selections,
                                              lensmodel);
    if(index >= 0)
        return PyLong_FromLong(index);

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* num_measurements_regularization(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(num_measurements_regularization,
                               self, args, kwargs,
                               true,
                               NULL);
}


static PyObject* callback_num_measurements(int i,
                                     int Ncameras_intrinsics,
                                     int Ncameras_extrinsics,
                                     int Nframes,
                                     int Npoints,
                                     int Npoints_fixed,
                                     int Nobservations_board,
                                     int Nobservations_point,
                                     int calibration_object_width_n,
                                     int calibration_object_height_n,
                                     const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
                                     const PyArrayObject* indices_point_camintrinsics_camextrinsics,
                                     const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                     const PyArrayObject* observations_point,
                                     const mrcal_lensmodel_t* lensmodel,
                                     mrcal_problem_selections_t problem_selections)
{

#if defined ENABLE_TRIANGULATED_WARNINGS && ENABLE_TRIANGULATED_WARNINGS
#warning "triangulated-solve: add tests to the num_measurements_..., state_index_... ..."
#endif


    mrcal_observation_point_triangulated_t* observations_point_triangulated  = NULL;
    int                                     Nobservations_point_triangulated = 0;


    int N = 0;

    if(indices_point_triangulated_camintrinsics_camextrinsics != NULL)
        N = PyArray_DIM(indices_point_triangulated_camintrinsics_camextrinsics, 0);
    else
    {
        // No triangulated points. No error. I have N = 0 in this path
    }


    mrcal_observation_point_triangulated_t c_observations_point_triangulated[N];

    if(N > 0)
    {
        Nobservations_point_triangulated =
            fill_c_observations_point_triangulated(c_observations_point_triangulated,
                                                   NULL, NULL, NULL,
                                                   indices_point_triangulated_camintrinsics_camextrinsics);
        if(Nobservations_point_triangulated < 0)
        {
            BARF("Error parsing triangulated points");
            return NULL;
        }
        observations_point_triangulated = c_observations_point_triangulated;
    }
    int index =
        mrcal_num_measurements(Nobservations_board,
                               Nobservations_point,
                               observations_point_triangulated,
                               Nobservations_point_triangulated,
                               calibration_object_width_n,
                               calibration_object_height_n,
                               Ncameras_intrinsics, Ncameras_extrinsics,
                               Nframes,
                               Npoints, Npoints_fixed,
                               problem_selections,
                               lensmodel);
    if(index >= 0)
        return PyLong_FromLong(index);

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* num_measurements(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(num_measurements,
                               self, args, kwargs,
                               true,
                               NULL);
}

static PyObject* callback_corresponding_icam_extrinsics(int icam_intrinsics,
                                                  int Ncameras_intrinsics,
                                                  int Ncameras_extrinsics,
                                                  int Nframes,
                                                  int Npoints,
                                                  int Npoints_fixed,
                                                  int Nobservations_board,
                                                  int Nobservations_point,
                                                  int calibration_object_width_n,
                                                  int calibration_object_height_n,
                                                  const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
                                                  const PyArrayObject* indices_point_camintrinsics_camextrinsics,
                                                  const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                                  const PyArrayObject* observations_point,
                                                  const mrcal_lensmodel_t* lensmodel,
                                                  mrcal_problem_selections_t problem_selections)
{

#if defined ENABLE_TRIANGULATED_WARNINGS && ENABLE_TRIANGULATED_WARNINGS
#warning "triangulated-solve: barf if we have any triangulated points"
#endif

    if( icam_intrinsics < 0 || icam_intrinsics >= Ncameras_intrinsics )
    {
        BARF("The given icam_intrinsics=%d is out of bounds. Must be >= 0 and < %d",
             icam_intrinsics, Ncameras_intrinsics);
        return NULL;
    }

    int icam_extrinsics;

    if(Nobservations_board > 0 && indices_frame_camintrinsics_camextrinsics == NULL)
    {
        BARF("Have Nobservations_board > 0, but indices_frame_camintrinsics_camextrinsics == NULL. Some required arguments missing?");
        return NULL;
    }
    mrcal_observation_board_t c_observations_board[Nobservations_board];
    fill_c_observations_board(// output
                              c_observations_board,
                              // input
                              Nobservations_board,
                              indices_frame_camintrinsics_camextrinsics);

    if(Nobservations_point > 0)
    {
        if(indices_point_camintrinsics_camextrinsics == NULL)
        {
            BARF("Have Nobservations_point > 0, but indices_point_camintrinsics_camextrinsics == NULL. Some required arguments missing?");
            return NULL;
        }
        if(observations_point == NULL)
        {
            BARF("Have Nobservations_point > 0, but observations_point == NULL. Some required arguments missing?");
            return NULL;
        }
    }
    mrcal_observation_point_t c_observations_point[Nobservations_point];
    fill_c_observations_point(// output
                              c_observations_point,
                              // input
                              Nobservations_point,
                              indices_point_camintrinsics_camextrinsics);

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
        return NULL;
    }

    return PyLong_FromLong(icam_extrinsics);
}
static PyObject* corresponding_icam_extrinsics(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(corresponding_icam_extrinsics,
                               self, args, kwargs,
                               false,
                               "icam_intrinsics");
}

static PyObject* callback_decode_observation_indices_points_triangulated(int imeasurement,
                                     int Ncameras_intrinsics,
                                     int Ncameras_extrinsics,
                                     int Nframes,
                                     int Npoints,
                                     int Npoints_fixed,
                                     int Nobservations_board,
                                     int Nobservations_point,
                                     int calibration_object_width_n,
                                     int calibration_object_height_n,
                                     const PyArrayObject* indices_frame_camintrinsics_camextrinsics,
                                     const PyArrayObject* indices_point_camintrinsics_camextrinsics,
                                     const PyArrayObject* indices_point_triangulated_camintrinsics_camextrinsics,
                                     const PyArrayObject* observations_point,
                                     const mrcal_lensmodel_t* lensmodel,
                                     mrcal_problem_selections_t problem_selections)
{


    if(indices_point_triangulated_camintrinsics_camextrinsics == NULL)
    {
        BARF("No triangulated points in this solve. Nothing to decode");
        return NULL;
    }

    int N = PyArray_DIM(indices_point_triangulated_camintrinsics_camextrinsics, 0);
    if(N <= 0)
    {
        BARF("No triangulated points in this solve. Nothing to decode");
        return NULL;
    }

    mrcal_observation_point_triangulated_t c_observations_point_triangulated[N];

    int Nobservations_point_triangulated =
        fill_c_observations_point_triangulated(c_observations_point_triangulated,
                                               NULL, NULL, NULL,
                                               indices_point_triangulated_camintrinsics_camextrinsics);
    if(Nobservations_point_triangulated < 0)
    {
        BARF("Error parsing triangulated points");
        return NULL;
    }

    mrcal_observation_point_triangulated_t* observations_point_triangulated =
        c_observations_point_triangulated;

    int iobservation0;
    int iobservation1;
    int iobservation_point0;
    int Nobservations_this_point;
    int Nmeasurements_this_point;
    int ipoint;

    bool result =
        mrcal_decode_observation_indices_points_triangulated(&iobservation0,
                                                             &iobservation1,
                                                             &iobservation_point0,
                                                             &Nobservations_this_point,
                                                             &Nmeasurements_this_point,
                                                             &ipoint,

                                                             imeasurement,
                                                             observations_point_triangulated,
                                                             Nobservations_point_triangulated);
    if(!result)
    {
        BARF("Error decoding indices");
        return NULL;
    }

    return Py_BuildValue( "{sisisisisisi}",
                          "iobservation0",            iobservation0,
                          "iobservation1",            iobservation1,
                          "iobservation_point0",      iobservation_point0,
                          "Nobservations_this_point", Nobservations_this_point,
                          "Nmeasurements_this_point", Nmeasurements_this_point,
                          "ipoint",                   ipoint );
}
static PyObject* decode_observation_indices_points_triangulated(PyObject* self, PyObject* args, PyObject* kwargs)
{
    return STATE_INDEX_GENERIC(decode_observation_indices_points_triangulated,
                               self, args, kwargs,
                               false,
                               "imeasurement");
}

static PyObject* _pack_unpack_state(PyObject* self, PyObject* args, PyObject* kwargs,
                                    bool pack)
{
    // This is VERY similar to state_index_generic(). Please consolidate
    PyObject* result = NULL;

    OPTIMIZE_ARGUMENTS_REQUIRED(ARG_DEFINE);
    OPTIMIZE_ARGUMENTS_OPTIONAL(ARG_DEFINE);

    PyArrayObject* b = NULL;

    bool need_decref_kwargs = false;
    if(!optimization_inputs_kwargs_delete_unknown(&kwargs, &need_decref_kwargs))
        goto done;

    int Ncameras_intrinsics = -1;
    int Ncameras_extrinsics = -1;
    int Nframes             = -1;
    int Npoints             = -1;
    int Nobservations_board = -1;
    int Nobservations_point = -1;

    char* keywords[] = { "b",
                         "Ncameras_intrinsics",
                         "Ncameras_extrinsics",
                         "Nframes",
                         "Npoints",
                         "Nobservations_board",
                         "Nobservations_point",
                         OPTIMIZE_ARGUMENTS_REQUIRED(NAMELIST)
                         OPTIMIZE_ARGUMENTS_OPTIONAL(NAMELIST)
                         NULL};

#define UNPACK_STATE "unpack_state"
    char arg_string[] =
        "O&"
        "|$" // everything is kwarg-only and optional. I apply logic down the
             // line to get what I need
        "iiiiii"
        OPTIMIZE_ARGUMENTS_REQUIRED(PARSECODE)
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

                                     PyArray_Converter, &b,
                                     &Ncameras_intrinsics,
                                     &Ncameras_extrinsics,
                                     &Nframes,
                                     &Npoints,
                                     &Nobservations_board,
                                     &Nobservations_point,
                                     OPTIMIZE_ARGUMENTS_REQUIRED(PARSEARG)
                                     OPTIMIZE_ARGUMENTS_OPTIONAL(PARSEARG) NULL))
        goto done;

    if(lensmodel == NULL)
    {
        BARF("The 'lensmodel' argument is required");
        goto done;
    }

    mrcal_lensmodel_t mrcal_lensmodel;
    if(!parse_lensmodel_from_arg(&mrcal_lensmodel, lensmodel))
        goto done;

    if(!handle_renamed(extrinsics_rt_fromref, rt_cam_ref))
        return false;
    if(!handle_renamed(frames_rt_toref, rt_ref_frame))
        return false;


    // checks dimensionality of array !IS_NULL. So if any array isn't passed-in,
    // that's OK! After I do this and if !IS_NULL, then I can ask for array
    // dimensions safely
    OPTIMIZE_ARGUMENTS_REQUIRED(CHECK_LAYOUT);
    OPTIMIZE_ARGUMENTS_OPTIONAL(CHECK_LAYOUT);

    // If explicit dimensions are given, use them. If they're not given, but we
    // have an array, use those dimensions. If an array isn't given either, use
    // 0
    if(Ncameras_intrinsics < 0) Ncameras_intrinsics = IS_NULL(intrinsics)            ? 0 : PyArray_DIMS(intrinsics)            [0];
    if(Ncameras_extrinsics < 0) Ncameras_extrinsics = IS_NULL(rt_cam_ref)            ? 0 : PyArray_DIMS(rt_cam_ref)            [0];
    if(Nframes < 0)             Nframes             = IS_NULL(rt_ref_frame)          ? 0 : PyArray_DIMS(rt_ref_frame)          [0];
    if(Npoints < 0)             Npoints             = IS_NULL(points)                ? 0 : PyArray_DIMS(points)                [0];
    if(Nobservations_board < 0) Nobservations_board = IS_NULL(observations_board)    ? 0 : PyArray_DIMS(observations_board)    [0];
    if(Nobservations_point < 0) Nobservations_point = IS_NULL(observations_point)    ? 0 : PyArray_DIMS(observations_point)    [0];


    mrcal_problem_selections_t problem_selections = CONSTRUCT_PROBLEM_SELECTIONS();

    if( PyArray_TYPE(b) != NPY_DOUBLE )
    {
        BARF("The given array MUST have values of type 'float'");
        goto done;
    }

    if( !PyArray_IS_C_CONTIGUOUS(b) )
    {
        BARF("The given array MUST be a C-style contiguous array");
        goto done;
    }

    int       ndim = PyArray_NDIM(b);
    npy_intp* dims = PyArray_DIMS(b);
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

    double* x = (double*)PyArray_DATA(b);
    if(pack)
        for(int i=0; i<PyArray_SIZE(b)/Nstate; i++)
        {
            mrcal_pack_solver_state_vector( x,
                                            Ncameras_intrinsics, Ncameras_extrinsics,
                                            Nframes, Npoints, Npoints_fixed,Nobservations_board,
                                            problem_selections, &mrcal_lensmodel);
            x = &x[Nstate];
        }
    else
        for(int i=0; i<PyArray_SIZE(b)/Nstate; i++)
        {
            mrcal_unpack_solver_state_vector( x,
                                              Ncameras_intrinsics, Ncameras_extrinsics,
                                              Nframes, Npoints, Npoints_fixed,Nobservations_board,
                                              problem_selections, &mrcal_lensmodel);
            x = &x[Nstate];
        }

    Py_INCREF(Py_None);
    result = Py_None;

 done:
    if(need_decref_kwargs)
        Py_DECREF(kwargs);
    OPTIMIZE_ARGUMENTS_REQUIRED(FREE_PYARRAY) ;
    OPTIMIZE_ARGUMENTS_OPTIONAL(FREE_PYARRAY) ;

    Py_XDECREF(b);
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


static
PyObject* load_image(PyObject* NPY_UNUSED(self),
                     PyObject* args,
                     PyObject* kwargs)
{
    // THIS IS IMPLEMENTED IN A NOT-GREAT WAY
    //
    // mrcal_image_TYPE_load() allocates a new array, and we then allocate a
    // numpy array to copy the data into it. I should be doing the allocation
    // once, and numpy should reuse the data. Doing THAT is easy, but hooking
    // the free() to happen when the numpy array is released takes a LOT of
    // typing.

    PyObject* result = NULL;

    const char* filename       = NULL;
    int         bits_per_pixel = -1;
    int         channels       = -1;

    // could be any type; not just uint8
    mrcal_image_uint8_t image = {};

    PyObject* image_array = NULL;

    char* keywords[] = { "filename",
                         "bits_per_pixel",
                         "channels",
                         NULL};
    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     "s|ii:mrcal.load_image",
                                     keywords,
                                     &filename, &bits_per_pixel, &channels ))
        goto done;

    if((bits_per_pixel <= 0 && channels >  0) ||
       (bits_per_pixel  > 0 && channels <= 0))
    {
        BARF("Both bits_per_pixel and channels should be given valid values >0, or neither should be");
        goto done;
    }

    if(bits_per_pixel <= 0)
    {
        if(!mrcal_image_anytype_load(&image,
                                     &bits_per_pixel, &channels,
                                     filename))
        {
            BARF("Error loading image '%s' with autodetected bits_per_pixel,channels",
                 filename);
            goto done;
        }
    }

    // I support a small number of combinations:
    // - bits_per_pixel = 8,  channels = 1
    // - bits_per_pixel = 16, channels = 1
    // - bits_per_pixel = 24, channels = 3
    if(bits_per_pixel == 8 && channels == 1)
    {
        if(image.data == NULL &&
           !mrcal_image_uint8_load((mrcal_image_uint8_t*)&image,
                                   filename))
        {
            BARF("Error loading image '%s'", filename);
            goto done;
        }
        image_array = PyArray_SimpleNew(2,
                                        ((npy_intp[]){image.h, image.w}),
                                        NPY_UINT8);
    }
    else if(bits_per_pixel == 16 && channels == 1)
    {
        if(image.data == NULL &&
           !mrcal_image_uint16_load((mrcal_image_uint16_t*)&image,
                                    filename))
        {
            BARF("Error loading image '%s'", filename);
            goto done;
        }
        image_array = PyArray_SimpleNew(2,
                                        ((npy_intp[]){image.h, image.w}),
                                        NPY_UINT16);
    }
    else if(bits_per_pixel == 24 && channels == 3)
    {
        if(image.data == NULL &&
           !mrcal_image_bgr_load((mrcal_image_bgr_t*)&image,
                                 filename))
        {
            BARF("Error loading image '%s' with bits_per_pixel=%d and channels=%d",
                 filename,
                 bits_per_pixel,
                 channels);
            goto done;
        }
        image_array = PyArray_SimpleNew(3,
                                        ((npy_intp[]){image.h, image.w, 3}),
                                        NPY_UINT8);
    }
    else
    {
        BARF("Unsupported format requested. I only support (bits_per_pixel,channels) = (8,1) and (16,1) and (24,3)");
        goto done;
    }

    if(image_array == NULL)
        goto done;

    // The numpy array is dense, but the image array may not be. Copy one line
    // at a time
    for(int i=0; i<image.h; i++)
        memcpy(&((uint8_t*)PyArray_DATA((PyArrayObject*)image_array))[image.w*bits_per_pixel/8*i],
               &((uint8_t*)image.data)[image.stride*i],
               image.w*bits_per_pixel/8);
    result = image_array;

 done:

    free(image.data);

    if(result == NULL)
        Py_XDECREF(image_array);

    return result;
}

static
PyObject* save_image(PyObject* NPY_UNUSED(self),
                     PyObject* args,
                     PyObject* kwargs)
{
    PyObject* result = NULL;

    const char*    filename    = NULL;
    PyArrayObject* image_array = NULL;

    char* keywords[] = { "filename",
                         "array",
                         NULL};
    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     "sO:mrcal.save_image",
                                     keywords,
                                     &filename, &image_array ))
        goto done;

    // I support a small number of combinations:
    // - bits_per_pixel = 8,  channels = 1
    // - bits_per_pixel = 16, channels = 1
    // - bits_per_pixel = 24, channels = 3
    if(!PyArray_Check(image_array))
    {
        BARF("I only know how to save numpy arrays");
        goto done;
    }

    int             ndim    = PyArray_NDIM(image_array);
    const npy_intp* dims    = PyArray_DIMS(image_array);
    int             dtype   = PyArray_TYPE(image_array);
    const npy_intp* strides = PyArray_STRIDES(image_array);

    if(ndim == 2 && dtype == NPY_UINT8)
    {
        if(strides[ndim-1] != 1)
        {
            BARF("Saving 8-bit grayscale array. I only know how to handle stride[-1] == 1");
            goto done;
        }
        mrcal_image_uint8_t image = {.w      = dims[1],
                                     .h      = dims[0],
                                     .stride = strides[0],
                                     .data   = PyArray_DATA(image_array) };
        if(!mrcal_image_uint8_save(filename, &image))
        {
            BARF("Error saving image '%s'", filename);
            goto done;
        }
    }
    else if(ndim == 2 && dtype == NPY_UINT16)
    {
        if(strides[ndim-1] != 2)
        {
            BARF("Saving 16-bit grayscale array. I only know how to handle stride[-1] == 2");
            goto done;
        }
        mrcal_image_uint16_t image = {.w      = dims[1],
                                      .h      = dims[0],
                                      .stride = strides[0],
                                      .data   = PyArray_DATA(image_array) };
        if(!mrcal_image_uint16_save(filename, &image))
        {
            BARF("Error saving image '%s'", filename);
            goto done;
        }
    }
    else if(ndim == 3 && dtype == NPY_UINT8)
    {
        if(dims[2] != 3)
        {
            BARF("Saving 3-dimensional array. shape[-1] != 3, so not BGR. Don't know what to do");
            goto done;
        }

        if(strides[ndim-1] != 1 ||
           strides[ndim-2] != 3)
        {
            BARF("Saving 8-bit BGR array. I only know how to handle stride[-1] == 1 && stride[-2] == 3");
            goto done;
        }
        mrcal_image_bgr_t image = {.w      = dims[1],
                                   .h      = dims[0],
                                   .stride = strides[0],
                                   .data   = PyArray_DATA(image_array) };
        if(!mrcal_image_bgr_save(filename, &image))
        {
            BARF("Error saving image '%s'", filename);
            goto done;
        }
    }
    else
    {
        BARF("Unsupported array. I only support (bits_per_pixel,channels) = (8,1) and (16,1) and (24,3)");
        goto done;
    }

    Py_INCREF(Py_None);
    result = Py_None;

 done:

    return result;
}

// LENSMODEL_ONE_ARGUMENTS followed by these
#define RECTIFIED_RESOLUTION_ARGUMENTS(_)                               \
    _(R_cam0_rect0, PyArrayObject*, NULL, "O&", PyArray_Converter COMMA, R_cam0_rect0, NPY_DOUBLE, {3 COMMA 3 } )
static bool
rectified_resolution_validate_args(RECTIFIED_RESOLUTION_ARGUMENTS(ARG_LIST_DEFINE)
                                   void* dummy __attribute__((unused)))
{
    RECTIFIED_RESOLUTION_ARGUMENTS(CHECK_LAYOUT);
    return true;
 done:
    return false;
}

static
PyObject* _rectified_resolution(PyObject* NPY_UNUSED(self),
                                PyObject* args,
                                PyObject* kwargs)
{
    PyObject* result = NULL;

    LENSMODEL_ONE_ARGUMENTS(ARG_DEFINE, );
    RECTIFIED_RESOLUTION_ARGUMENTS(ARG_DEFINE);

    // input and output
    double pixels_per_deg_az;
    double pixels_per_deg_el;

    // input
    mrcal_lensmodel_t mrcal_lensmodel;
    mrcal_point2_t    azel_fov_deg;
    mrcal_point2_t    azel0_deg;
    char*             rectification_model_string;
    mrcal_lensmodel_t rectification_model;


    char* keywords[] = { LENSMODEL_ONE_ARGUMENTS(NAMELIST, )
                         RECTIFIED_RESOLUTION_ARGUMENTS(NAMELIST)
                         "az_fov_deg",
                         "el_fov_deg",
                         "az0_deg",
                         "el0_deg",
                         "pixels_per_deg_az",
                         "pixels_per_deg_el",
                         "rectification_model",
                         NULL};
    // This function is internal, so EVERYTHING is required
    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     LENSMODEL_ONE_ARGUMENTS(PARSECODE, )
                                     RECTIFIED_RESOLUTION_ARGUMENTS(PARSECODE)
                                     "dddddds:mrcal.rectified_resolution",

                                     keywords,

                                     LENSMODEL_ONE_ARGUMENTS(PARSEARG, )
                                     RECTIFIED_RESOLUTION_ARGUMENTS(PARSEARG)
                                     &azel_fov_deg.x,
                                     &azel_fov_deg.y,
                                     &azel0_deg.x,
                                     &azel0_deg.y,
                                     &pixels_per_deg_az,
                                     &pixels_per_deg_el,
                                     &rectification_model_string ))
        goto done;


    if( !lensmodel_one_validate_args(&mrcal_lensmodel,
                                     LENSMODEL_ONE_ARGUMENTS(ARG_LIST_CALL, )
                                     true /* DO check the layout */ ))
        goto done;

    if(!parse_lensmodel_from_arg(&rectification_model, rectification_model_string))
        goto done;

    if(!rectified_resolution_validate_args(RECTIFIED_RESOLUTION_ARGUMENTS(ARG_LIST_CALL)
                                           NULL))
        goto done;

    if(!mrcal_rectified_resolution( &pixels_per_deg_az,
                                    &pixels_per_deg_el,

                                    // input
                                    &mrcal_lensmodel,
                                    PyArray_DATA(intrinsics),
                                    &azel_fov_deg,
                                    &azel0_deg,
                                    PyArray_DATA(R_cam0_rect0),
                                    rectification_model.type))
    {
        BARF("mrcal_rectified_resolution() failed!");
        goto done;
    }

    result = Py_BuildValue("(dd)",
                           pixels_per_deg_az,
                           pixels_per_deg_el);

 done:

    LENSMODEL_ONE_ARGUMENTS(FREE_PYARRAY, );
    RECTIFIED_RESOLUTION_ARGUMENTS(FREE_PYARRAY);

    return result;
}

// LENSMODEL_ONE_ARGUMENTS followed by these
#define RECTIFIED_SYSTEM_ARGUMENTS(_)                               \
    _(rt_cam0_ref, PyArrayObject*, NULL, "O&", PyArray_Converter COMMA, rt_cam0_ref, NPY_DOUBLE, {6 } ) \
    _(rt_cam1_ref, PyArrayObject*, NULL, "O&", PyArray_Converter COMMA, rt_cam1_ref, NPY_DOUBLE, {6 } )
static bool
rectified_system_validate_args(RECTIFIED_SYSTEM_ARGUMENTS(ARG_LIST_DEFINE)
                               void* dummy __attribute__((unused)))
{
    RECTIFIED_SYSTEM_ARGUMENTS(CHECK_LAYOUT);
    return true;
 done:
    return false;
}

static
PyObject* _rectified_system(PyObject* NPY_UNUSED(self),
                            PyObject* args,
                            PyObject* kwargs)
{
    PyObject* result = NULL;

    LENSMODEL_ONE_ARGUMENTS(ARG_DEFINE, 0);
    RECTIFIED_SYSTEM_ARGUMENTS(ARG_DEFINE);

    // output
    unsigned int   imagersize_rectified[2];
    PyArrayObject* fxycxy_rectified = NULL;
    PyArrayObject* rt_rect0_ref     = NULL;
    double         baseline;

    // input and output
    double         pixels_per_deg_az;
    double         pixels_per_deg_el;
    mrcal_point2_t azel_fov_deg;
    mrcal_point2_t azel0_deg;

    // input
    mrcal_lensmodel_t mrcal_lensmodel0;
    char*             rectification_model_string;
    mrcal_lensmodel_t rectification_model;

    bool az0_deg_autodetect    = false;
    bool el0_deg_autodetect    = false;
    bool az_fov_deg_autodetect = false;
    bool el_fov_deg_autodetect = false;

    fxycxy_rectified =
        (PyArrayObject*)PyArray_SimpleNew(1,
                                          ((npy_intp[]){4}),
                                          NPY_DOUBLE);
    if(NULL == fxycxy_rectified)
    {
        BARF("Couldn't allocate fxycxy_rectified");
        goto done;
    }
    rt_rect0_ref =
        (PyArrayObject*)PyArray_SimpleNew(1,
                                          ((npy_intp[]){6}),
                                          NPY_DOUBLE);
    if(NULL == rt_rect0_ref)
    {
        BARF("Couldn't allocate rt_rect0_ref");
        goto done;
    }

    char* keywords[] = { LENSMODEL_ONE_ARGUMENTS(NAMELIST, 0)
                         RECTIFIED_SYSTEM_ARGUMENTS(NAMELIST)
                         "az_fov_deg",
                         "el_fov_deg",
                         "az0_deg",
                         "el0_deg",
                         "pixels_per_deg_az",
                         "pixels_per_deg_el",
                         "rectification_model",
                         NULL};
    // This function is internal, so EVERYTHING is required
    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     LENSMODEL_ONE_ARGUMENTS(PARSECODE, 0)
                                     RECTIFIED_SYSTEM_ARGUMENTS(PARSECODE)
                                     "dddddds:mrcal.rectified_system",

                                     keywords,

                                     LENSMODEL_ONE_ARGUMENTS(PARSEARG, 0)
                                     RECTIFIED_SYSTEM_ARGUMENTS(PARSEARG)
                                     &azel_fov_deg.x,
                                     &azel_fov_deg.y,
                                     &azel0_deg.x,
                                     &azel0_deg.y,
                                     &pixels_per_deg_az,
                                     &pixels_per_deg_el,
                                     &rectification_model_string ))
        goto done;

    if(azel0_deg.x > 1e6)
        az0_deg_autodetect = true;

    if( !lensmodel_one_validate_args(&mrcal_lensmodel0,
                                     LENSMODEL_ONE_ARGUMENTS(ARG_LIST_CALL, 0)
                                     true /* DO check the layout */ ))
        goto done;

    if(!parse_lensmodel_from_arg(&rectification_model, rectification_model_string))
        goto done;

    if(!rectified_system_validate_args(RECTIFIED_SYSTEM_ARGUMENTS(ARG_LIST_CALL)
                                       NULL))
        goto done;

    if(!mrcal_rectified_system( // output
                                imagersize_rectified,
                                PyArray_DATA(fxycxy_rectified),
                                PyArray_DATA(rt_rect0_ref),
                                &baseline,

                                // input, output
                                &pixels_per_deg_az,
                                &pixels_per_deg_el,

                                // input, output
                                &azel_fov_deg,
                                &azel0_deg,

                                // input
                                &mrcal_lensmodel0,
                                PyArray_DATA(intrinsics0),
                                PyArray_DATA(rt_cam0_ref),
                                PyArray_DATA(rt_cam1_ref),
                                rectification_model.type,
                                az0_deg_autodetect,
                                el0_deg_autodetect,
                                az_fov_deg_autodetect,
                                el_fov_deg_autodetect))
    {
        BARF("mrcal_rectified_system() failed!");
        goto done;
    }

    result = Py_BuildValue("(ddiiOOddddd)",
                           pixels_per_deg_az,
                           pixels_per_deg_el,
                           imagersize_rectified[0], imagersize_rectified[1],
                           fxycxy_rectified,
                           rt_rect0_ref,
                           baseline,
                           azel_fov_deg.x, azel_fov_deg.y,
                           azel0_deg.x, azel0_deg.y);

 done:

    LENSMODEL_ONE_ARGUMENTS(FREE_PYARRAY, 0);
    RECTIFIED_SYSTEM_ARGUMENTS(FREE_PYARRAY);

    Py_XDECREF(fxycxy_rectified);
    Py_XDECREF(rt_rect0_ref);

    return result;
}

// LENSMODEL_ONE_ARGUMENTS followed by these
#define RECTIFICATION_MAPS_ARGUMENTS(_)                               \
    _(r_cam0_ref,           PyArrayObject*, NULL, "O&", PyArray_Converter COMMA, r_cam0_ref,           NPY_DOUBLE, {3} ) \
    _(r_cam1_ref,           PyArrayObject*, NULL, "O&", PyArray_Converter COMMA, r_cam1_ref,           NPY_DOUBLE, {3} ) \
    _(r_rect0_ref,          PyArrayObject*, NULL, "O&", PyArray_Converter COMMA, r_rect0_ref,          NPY_DOUBLE, {3} ) \
    _(rectification_maps,   PyArrayObject*, NULL, "O&", PyArray_Converter COMMA, rectification_maps,   NPY_FLOAT,  {2 COMMA -1 COMMA -1 COMMA 2} )

static bool
rectification_maps_validate_args(RECTIFICATION_MAPS_ARGUMENTS(ARG_LIST_DEFINE)
                                 void* dummy __attribute__((unused)))
{
    RECTIFICATION_MAPS_ARGUMENTS(CHECK_LAYOUT);
    return true;
 done:
    return false;
}

static
PyObject* _rectification_maps(PyObject* NPY_UNUSED(self),
                              PyObject* args,
                              PyObject* kwargs)
{
    PyObject* result = NULL;

    unsigned int imagersize_rectified[2];

    LENSMODEL_ONE_ARGUMENTS(ARG_DEFINE, 0);
    LENSMODEL_ONE_ARGUMENTS(ARG_DEFINE, 1);
    LENSMODEL_ONE_ARGUMENTS(ARG_DEFINE, _rectified);
    RECTIFICATION_MAPS_ARGUMENTS(ARG_DEFINE);

    // input
    mrcal_lensmodel_t mrcal_lensmodel0;
    mrcal_lensmodel_t mrcal_lensmodel1;
    mrcal_lensmodel_t mrcal_lensmodel_rectified;

    char* keywords[] = { LENSMODEL_ONE_ARGUMENTS(NAMELIST, 0)
                         LENSMODEL_ONE_ARGUMENTS(NAMELIST, 1)
                         LENSMODEL_ONE_ARGUMENTS(NAMELIST, _rectified)
                         RECTIFICATION_MAPS_ARGUMENTS(NAMELIST)
                         NULL};
    // This function is internal, so EVERYTHING is required
    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     LENSMODEL_ONE_ARGUMENTS(PARSECODE, 0)
                                     LENSMODEL_ONE_ARGUMENTS(PARSECODE, 1)
                                     LENSMODEL_ONE_ARGUMENTS(PARSECODE, _rectified)
                                     RECTIFICATION_MAPS_ARGUMENTS(PARSECODE)
                                     ":mrcal.rectification_maps",

                                     keywords,

                                     LENSMODEL_ONE_ARGUMENTS(PARSEARG, 0)
                                     LENSMODEL_ONE_ARGUMENTS(PARSEARG, 1)
                                     LENSMODEL_ONE_ARGUMENTS(PARSEARG, _rectified)
                                     RECTIFICATION_MAPS_ARGUMENTS(PARSEARG)
                                     NULL ))
        goto done;

    if( !lensmodel_one_validate_args(&mrcal_lensmodel0,
                                     LENSMODEL_ONE_ARGUMENTS(ARG_LIST_CALL, 0)
                                     true /* DO check the layout */ ))
        goto done;
    if( !lensmodel_one_validate_args(&mrcal_lensmodel1,
                                     LENSMODEL_ONE_ARGUMENTS(ARG_LIST_CALL, 1)
                                     true /* DO check the layout */ ))
        goto done;
    if( !lensmodel_one_validate_args(&mrcal_lensmodel_rectified,
                                     LENSMODEL_ONE_ARGUMENTS(ARG_LIST_CALL, _rectified)
                                     true /* DO check the layout */ ))
        goto done;

    if(!rectification_maps_validate_args(RECTIFICATION_MAPS_ARGUMENTS(ARG_LIST_CALL)
                                         NULL))
        goto done;

    // rectification_maps has shape (Ncameras=2, Nel, Naz, Nxy=2)
    imagersize_rectified[0] = PyArray_DIMS(rectification_maps)[2];
    imagersize_rectified[1] = PyArray_DIMS(rectification_maps)[1];

    if(!mrcal_rectification_maps( // output
                                  PyArray_DATA(rectification_maps),

                                  // input
                                  &mrcal_lensmodel0,
                                  PyArray_DATA(intrinsics0),
                                  PyArray_DATA(r_cam0_ref),

                                  &mrcal_lensmodel1,
                                  PyArray_DATA(intrinsics1),
                                  PyArray_DATA(r_cam1_ref),

                                  mrcal_lensmodel_rectified.type,
                                  PyArray_DATA(intrinsics_rectified),
                                  imagersize_rectified,
                                  PyArray_DATA(r_rect0_ref)))
    {
        BARF("mrcal_rectification_maps() failed!");
        goto done;
    }

    Py_INCREF(Py_None);
    result = Py_None;

 done:

    LENSMODEL_ONE_ARGUMENTS(FREE_PYARRAY, 0);
    LENSMODEL_ONE_ARGUMENTS(FREE_PYARRAY, 1);
    RECTIFICATION_MAPS_ARGUMENTS(FREE_PYARRAY);

    return result;
}


static bool
callback_sensor_link_C(const uint16_t idx_to, const uint16_t idx_from, void* cookie)
{
    PyObject* callback_sensor_link = (PyObject*)cookie;

    PyObject* py_idx_to   = NULL;
    PyObject* py_idx_from = NULL;
    PyObject* result      = NULL;

    py_idx_to   = PyLong_FromLong(idx_to);
    if(py_idx_to == NULL) goto done;

    py_idx_from = PyLong_FromLong(idx_from);
    if(py_idx_from == NULL) goto done;

    result = PyObject_CallFunctionObjArgs(callback_sensor_link,
                                          py_idx_to,
                                          py_idx_from,
                                          NULL);

 done:
    Py_XDECREF(py_idx_to);
    Py_XDECREF(py_idx_from);

    if(result == NULL)
        return false;

    Py_DECREF(result);
    return true;
}
static
PyObject* traverse_sensor_links(PyObject* NPY_UNUSED(self),
                                      PyObject* args,
                                      PyObject* kwargs)
{
    PyObject* result = NULL;

    int            Nsensors             = 0;
    PyArrayObject* connectivity_matrix  = NULL;
    PyObject*      callback_sensor_link = NULL;

    char* keywords[] = { "connectivity_matrix",
                         "callback_sensor_link",
                         NULL};
    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     "$O&O:mrcal.traverse_sensor_links",
                                     keywords,
                                     PyArray_Converter, &connectivity_matrix,
                                     &callback_sensor_link))
        goto done;

    if(PyArray_NDIM(connectivity_matrix) != 2)
    {
        BARF("The connectivity_matrix must have 2 dimensions");
        goto done;
    }
    Nsensors = PyArray_DIMS(connectivity_matrix)[1];

    if(!_check_layout("connectivity_matrix",
                      connectivity_matrix,
                      NPY_UINT16, "NPY_UINT16",
                      (int[]){Nsensors,Nsensors}, 2, "{Nsensors,Nsensors}",
                      false))
        goto done;

    if(!PyCallable_Check(callback_sensor_link))
    {
        BARF("callback_sensor_link is not callable");
        goto done;
    }

    if(Nsensors > UINT16_MAX)
    {
        BARF("Nsensors=%d doesn't fit into a uint16_t", Nsensors);
        goto done;
    }

    // Arguments are good. Let's massage them to do the right thing


    {
        // We reconstruct just the upper triangle of the connectivity_matrix
        uint16_t connectivity_matrix_upper[Nsensors*(Nsensors-1)/2];
        int k = 0;
        for(int i=0; i<Nsensors; i++)
            for(int j=i+1; j<Nsensors; j++)
                connectivity_matrix_upper[k++] =
                    *(uint16_t*)PyArray_GETPTR2(connectivity_matrix,i,j);

        if(!mrcal_traverse_sensor_links( Nsensors,
                                         connectivity_matrix_upper,
                                         &callback_sensor_link_C,
                                         callback_sensor_link))
        {
            if(!PyErr_Occurred())
                BARF("mrcal_traverse_sensor_links() failed");
            goto done;
        }
    }

    Py_INCREF(Py_None);
    result = Py_None;

 done:

    Py_XDECREF(connectivity_matrix);
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
static const char decode_observation_indices_points_triangulated_docstring[] =
#include "decode_observation_indices_points_triangulated.docstring.h"
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
static const char drt_ref_refperturbed__dbpacked_docstring[] =
#include "drt_ref_refperturbed__dbpacked.docstring.h"
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
static const char load_image_docstring[] =
#include "load_image.docstring.h"
    ;
static const char save_image_docstring[] =
#include "save_image.docstring.h"
    ;
static const char _rectified_resolution_docstring[] =
#include "_rectified_resolution.docstring.h"
    ;
static const char _rectified_system_docstring[] =
#include "_rectified_system.docstring.h"
    ;
static const char _rectification_maps_docstring[] =
#include "_rectification_maps.docstring.h"
    ;
static const char traverse_sensor_links_docstring[] =
#include "traverse_sensor_links.docstring.h"
    ;
static PyMethodDef methods[] =
    { PYMETHODDEF_ENTRY(,optimize,                         METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(,optimizer_callback,               METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(,drt_ref_refperturbed__dbpacked,   METH_VARARGS | METH_KEYWORDS),

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
      PYMETHODDEF_ENTRY(, corresponding_icam_extrinsics,   METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(, decode_observation_indices_points_triangulated,METH_VARARGS | METH_KEYWORDS),

      PYMETHODDEF_ENTRY(,lensmodel_metadata_and_config,METH_VARARGS),
      PYMETHODDEF_ENTRY(,lensmodel_num_params,         METH_VARARGS),
      PYMETHODDEF_ENTRY(,supported_lensmodels,         METH_NOARGS),
      PYMETHODDEF_ENTRY(,knots_for_splined_models,     METH_VARARGS),

      PYMETHODDEF_ENTRY(, load_image,                  METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(, save_image,                  METH_VARARGS | METH_KEYWORDS),

      PYMETHODDEF_ENTRY(,_rectified_resolution,        METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(,_rectified_system,            METH_VARARGS | METH_KEYWORDS),
      PYMETHODDEF_ENTRY(,_rectification_maps,          METH_VARARGS | METH_KEYWORDS),

      PYMETHODDEF_ENTRY(, traverse_sensor_links, METH_VARARGS | METH_KEYWORDS),
      {}
    };
#if defined ENABLE_TRIANGULATED_WARNINGS && ENABLE_TRIANGULATED_WARNINGS
#warning "triangulated-solve: fill in the new xxxx.docstring"
#endif



static bool _init_mrcal_common(PyObject* module)
{
    Py_INCREF(&CHOLMOD_factorization_type);
    if(0 != PyModule_Add(module,
                         "CHOLMOD_factorization",
                         (PyObject *)&CHOLMOD_factorization_type))
    {
        BARF("Could not add mrcal.CHOLMOD_factorization");
        return false;
    }

#define COUNT(name, pytype, initialvalue, parsecode, parseprearg, name_pyarrayobj, npy_type, dims_ref) \
    +1
    const int Nkeys = 0
        OPTIMIZE_ARGUMENTS_REQUIRED(COUNT)
        OPTIMIZE_ARGUMENTS_OPTIONAL(COUNT)
        OPTIMIZER_CALLBACK_ARGUMENTS_OPTIONAL_EXTRA(COUNT);
#undef COUNT

    PyObject* optimization_inputs_known_keys_tuple;
    if(NULL == (optimization_inputs_known_keys_tuple = PyTuple_New(Nkeys)))
    {
        BARF("Could not create optimization_inputs_known_keys_tuple");
        return false;
    }

    PyObject* value;
    int i=0;

#define ADD_TO_TUPLE(name, pytype, initialvalue, parsecode, parseprearg, name_pyarrayobj, npy_type, dims_ref) \
    if( NULL == (value = PyUnicode_FromString(#name)))                  \
    {                                                                   \
        BARF("Couldn't create '" #name "' string");                     \
        Py_DECREF(optimization_inputs_known_keys_tuple);                \
        return false;                                                   \
    }                                                                   \
    PyTuple_SET_ITEM(optimization_inputs_known_keys_tuple, i++, value);

    OPTIMIZE_ARGUMENTS_REQUIRED(ADD_TO_TUPLE);
    OPTIMIZE_ARGUMENTS_OPTIONAL(ADD_TO_TUPLE);
    OPTIMIZER_CALLBACK_ARGUMENTS_OPTIONAL_EXTRA(ADD_TO_TUPLE);
#undef ADD_TO_TUPLE

    _optimization_inputs_known_keys_frozenset =
        PyFrozenSet_New(optimization_inputs_known_keys_tuple);
    if(_optimization_inputs_known_keys_frozenset == NULL)
    {
        BARF("Couldn't create optimization_inputs_known_keys_frozenset");
        Py_DECREF(optimization_inputs_known_keys_tuple);
        return false;
    }

    Py_DECREF(optimization_inputs_known_keys_tuple);
#if PY_VERSION_HEX >= 0x030A0000
    // >= Python 3.10
    // New path. The PyModule_AddObject() call in the legacy path is deprecated
    if(0 != PyModule_Add(module,
                         "_optimization_inputs_known_keys",
                         (PyObject *)_optimization_inputs_known_keys_frozenset))
    {
        BARF("Could not add mrcal._optimization_inputs_known_keys_frozenset");
        return false;
    }
#else
    // Legacy path for ancient Python. No PyModule_Add()
    if(0 != PyModule_AddObject(module,
                               "_optimization_inputs_known_keys",
                               (PyObject *)_optimization_inputs_known_keys_frozenset))
    {
        Py_XDECREF((PyObject *)_optimization_inputs_known_keys_frozenset);
        BARF("Could not add mrcal._optimization_inputs_known_keys_frozenset");
        return false;
    }
#endif
    return true;
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

    if(!_init_mrcal_common(module))
        return NULL;

    import_array();

    return module;
}
