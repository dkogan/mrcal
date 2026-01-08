// mrcal_cameramodel_converter is a "converter" function that can be used with
// "O&" conversions in PyArg_ParseTupleAndKeywords() calls. Can interpret either
// path strings or mrcal.cameramodel objects as mrcal_cameramodel_VOID_t structures
//
// This isn't a part of the mrcal Python wrapping, but helps other python
// wrapping programs work with mrcal_cameramodel_VOID_t. I link this into
// libmrcal.so, but libmrcal.so does NOT link with libpython. 99% of the usage
// of libmrcal.so will not use this, so it should work without libpython. People
// using this function will be doing so as part of
// PyArg_ParseTupleAndKeywords(), so they will be linking to libpython anyway.
// Thus I weaken all the references to libpython AFTER this is compiled. This is
// done in the Makefile, see the comment there for all sorts of gory details

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#ifdef WEAKEN_PY_REFS
  #include "python-cameramodel-converter-py-symbol-refs.h"
  #define WEAKEN(f) extern __typeof__(f) f __attribute__((weak));
  PY_REFS(WEAKEN)
#endif


#include "mrcal.h"

#include "python-wrapping-utilities.h"

#define CHECK_LAYOUT3(name, npy_type, dims_ref) \
    CHECK_LAYOUT(name,xxx,xxx,xxx,xxx,name,npy_type,dims_ref)



#define BARF_AND_GOTO_DONE(fmt, ...) do { BARF(fmt, ##__VA_ARGS__); goto done; } while(0)

int mrcal_cameramodel_converter(PyObject*             py_model,
                                mrcal_cameramodel_VOID_t** model)
{
    // Define the PyArray_API. See here:
    //   https://numpy.org/doc/stable/reference/c-api/array.html#c.import_array
    // PyArray_ImportNumPyAPI() has this if() logic, but that's only available
    // in numpy >= 2.0. Without this, the PyArray_API function table will be
    // NULL. It is static in each compile unit.
    if(PyArray_API == NULL)
        import_array();

    int result = 0; // failure by default

    PyObject* call_result = NULL;


    if(py_model == Py_None)
        *model = NULL;
    else if(PyUnicode_Check(py_model))
    {
        // This is a string. Assume it's a filename.
        const char* filename = PyUnicode_AsUTF8AndSize(py_model, NULL);
        if(filename == NULL)
            BARF_AND_GOTO_DONE("The model argument claims to be a string, but I could not get this string out of it");
        *model = mrcal_read_cameramodel_file(filename);
        if(*model == NULL)
            BARF_AND_GOTO_DONE("Couldn't read mrcal_cameramodel_VOID_t from '%s'", filename);
    }
    else
    {
        call_result = PyObject_CallMethod(py_model, "intrinsics", NULL);
        if(call_result == NULL)
            BARF_AND_GOTO_DONE("Couldn't call cameramodel.intrinsics()");
        if(!PyTuple_Check(call_result))
            BARF_AND_GOTO_DONE("cameramodel.intrinsics() result isn't a tuple");
        if(2 != PyTuple_Size(call_result))
            BARF_AND_GOTO_DONE("cameramodel.intrinsics() result must be a tuple of length 2");
        PyObject* py_lensmodel  = PyTuple_GetItem(call_result,0);
        PyObject* intrinsics    = PyTuple_GetItem(call_result,1);
        if(!(PyUnicode_Check(py_lensmodel) && PyArray_Check(intrinsics)))
            BARF_AND_GOTO_DONE("cameramodel.intrinsics() result must contain (string,array)");
        const char* lensmodel = PyUnicode_AsUTF8AndSize(py_lensmodel, NULL);
        if(lensmodel == NULL)
            BARF_AND_GOTO_DONE("The lensmodel claims to be a string, but I could not get this string out of it");
        CHECK_LAYOUT3(intrinsics, NPY_DOUBLE, {-1});
        int Nparams = PyArray_SIZE((PyArrayObject*)intrinsics);
        *model = malloc(sizeof(mrcal_cameramodel_VOID_t) +
                        Nparams*sizeof(double));
        if(NULL == *model)
            BARF_AND_GOTO_DONE("Couldn't allocate cameramodel with %d intrinsics", Nparams);
        if(!mrcal_lensmodel_from_name(&((*model)->lensmodel),
                                      lensmodel))
            BARF_AND_GOTO_DONE("Couldn't parsse lensmodel from '%s'", lensmodel);
        memcpy(&(*model)->intrinsics[0],
               PyArray_DATA((PyArrayObject*)intrinsics),
               Nparams*sizeof(double));
        Py_DECREF(call_result);
        call_result = NULL;

        call_result = PyObject_CallMethod(py_model, "imagersize", NULL);
        if(call_result == NULL)
            BARF_AND_GOTO_DONE("Couldn't call cameramodel.imagersize()");
        if(!PyArray_Check((PyArrayObject*)call_result))
            BARF_AND_GOTO_DONE("cameramodel.imagersize() result must be a numpy array");
        CHECK_LAYOUT3(call_result, NPY_INT32, {2});
        (*model)->imagersize[0] = ((int32_t*)PyArray_DATA((PyArrayObject*)call_result))[0];
        (*model)->imagersize[1] = ((int32_t*)PyArray_DATA((PyArrayObject*)call_result))[1];
        Py_DECREF(call_result);
        call_result = NULL;

        call_result = PyObject_CallMethod(py_model, "rt_cam_ref", NULL);
        if(call_result == NULL)
            BARF_AND_GOTO_DONE("Couldn't call cameramodel.rt_cam_ref()");
        if(!PyArray_Check((PyArrayObject*)call_result))
            BARF_AND_GOTO_DONE("cameramodel.rt_cam_ref() result must be a numpy array");
        CHECK_LAYOUT3(call_result, NPY_DOUBLE, {6});
        memcpy(&(*model)->rt_cam_ref[0],
               PyArray_DATA((PyArrayObject*)call_result),
               6*sizeof(double));
        Py_DECREF(call_result);
        call_result = NULL;
    }

    // success!
    result = 1;

 done:
    Py_XDECREF(call_result);
    if(result == 0)
    {
        free(*model);
        *model = NULL;
    }
    return result;
}
