#pragma once

#ifdef __cplusplus
extern "C" {
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
#define PYMETHODDEF_ENTRY(function_prefix, name, args) {#name,          \
                                                        (PyCFunction)function_prefix ## name, \
                                                        args,           \
                                                        function_prefix ## name ## _docstring}
#define CHECK_LAYOUT(   name, pytype, initialvalue, parsecode, parseprearg, name_pyarrayobj, npy_type, dims_ref) \
    {                                                                   \
        const int dims[] = dims_ref;                                    \
        int       ndims  = (int)sizeof(dims)/(int)sizeof(dims[0]);      \
        if(!_check_layout( #name, (PyArrayObject*)name_pyarrayobj, (int)npy_type, #npy_type, dims, ndims, #dims_ref, true )) \
            goto done;                                               \
    }

static bool _check_layout(const char*    name,
                          PyArrayObject* pyarrayobj,
                          int            npy_type,
                          const char*    npy_type_string,
                          const int*     dims_ref,
                          int            Ndims_ref,
                          const char*    dims_ref_string,
                          bool           do_check_for_contiguity)
{
    if(!IS_NULL(pyarrayobj))
    {
        if( Ndims_ref > 0 )
        {
            if( PyArray_NDIM(pyarrayobj) != Ndims_ref )
            {
                BARF("'%s' must have exactly %d dims; got %d",
                     name,
                     Ndims_ref, PyArray_NDIM(pyarrayobj));
                return false;
            }
            for(int i=0; i<Ndims_ref; i++)
                if(dims_ref[i] >= 0 && dims_ref[i] != PyArray_DIMS(pyarrayobj)[i])
                {
                    BARF("'%s' must have dimensions '%s' where <0 means 'any'. Dims %d got %ld instead",
                         name, dims_ref_string,
                         i, PyArray_DIMS(pyarrayobj)[i]);
                    return false;
                }
        }
        if( npy_type >= 0 )
        {
            if( PyArray_TYPE(pyarrayobj) != npy_type )
            {
                BARF("'%s' must have type: %s",
                     name, npy_type_string);
                return false;
            }
            if( do_check_for_contiguity &&
                !PyArray_IS_C_CONTIGUOUS(pyarrayobj) )
            {
                BARF("'%s' must be c-style contiguous",
                     name);
                return false;
            }
        }
    }
    return true;
}

#ifdef __cplusplus
}
#endif
