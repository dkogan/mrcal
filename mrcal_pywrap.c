#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <stdbool.h>
#include <Python.h>
#include <numpy/arrayobject.h>

const char optimize_docstring[] =
#include "optimize.docstring.h"
    ;

static bool optimize_validate_args( PyArrayObject* camera_intrinsics,
                                    PyArrayObject* camera_extrinsics,
                                    PyArrayObject* frames,
                                    PyArrayObject* observations)
{
    if( PyArray_NDIM(camera_intrinsics) != 2 )
    {
        PyErr_SetString(PyExc_RuntimeError, "'camera_intrinsics' must have exactly 2 dims");
        return false;
    }
    if( PyArray_NDIM(camera_extrinsics) != 2 )
    {
        PyErr_SetString(PyExc_RuntimeError, "'camera_extrinsics' must have exactly 2 dims");
        return false;
    }
    if( PyArray_NDIM(frames) != 2 )
    {
        PyErr_SetString(PyExc_RuntimeError, "'frames' must have exactly 2 dims");
        return false;
    }
    if( PyArray_NDIM(observations) != 4 )
    {
        PyErr_SetString(PyExc_RuntimeError, "'observations' must have exactly 4 dims");
        return false;
    }

    int Ncameras = PyArray_DIMS(camera_intrinsics)[0];
    if( Ncameras-1 !=
        PyArray_DIMS(camera_extrinsics)[0] )
    {
        PyErr_Format(PyExc_RuntimeError, "Inconsistent Ncameras: 'extrinsics' says %ld, intrinsics says %ld",
                     PyArray_DIMS(camera_extrinsics)[0] + 1,
                     PyArray_DIMS(camera_intrinsics)[0] );
        return false;
    }
    if( 6 != PyArray_DIMS(camera_extrinsics)[1] )
    {
        PyErr_Format(PyExc_RuntimeError, "extrinsics.shape[1] MUST be 6. Instead got %ld",
                     PyArray_DIMS(camera_extrinsics)[1] );
        return false;
    }
    if( 6 != PyArray_DIMS(frames)[1] )
    {
        PyErr_Format(PyExc_RuntimeError, "frames.shape[1] MUST be 6. Instead got %ld",
                     PyArray_DIMS(frames)[1] );
        return false;
    }

    int Nframes = PyArray_DIMS(frames)[0];
    if( Nframes != PyArray_DIMS(observations)[0] )
    {
        PyErr_Format(PyExc_RuntimeError, "Inconsistent Nframes: 'frames' says %ld, 'observations' says %ld",
                     PyArray_DIMS(frames)[0],
                     PyArray_DIMS(observations)[0]);
        return false;
    }
    if( Ncameras != PyArray_DIMS(observations)[1] )
    {
        PyErr_Format(PyExc_RuntimeError, "Inconsistent Ncameras: 'camera_intrinsics' says %ld, 'observations' says %ld",
                     PyArray_DIMS(camera_intrinsics)[0],
                     PyArray_DIMS(observations)[1]);
        return false;
    }
    if( 10 != PyArray_DIMS(observations)[2] ||
        10 != PyArray_DIMS(observations)[3] )
    {
        PyErr_Format(PyExc_RuntimeError, "observations.shape[2:] MUST be (10,10). Instead got (%ld,%ld)",
                     PyArray_DIMS(observations)[2],
                     PyArray_DIMS(observations)[3]);
        return false;
    }


    if( PyArray_TYPE(camera_intrinsics) != NPY_DOUBLE ||
        PyArray_TYPE(camera_extrinsics) != NPY_DOUBLE ||
        PyArray_TYPE(frames)            != NPY_DOUBLE ||
        PyArray_TYPE(observations)      != NPY_DOUBLE )
    {
        PyErr_SetString(PyExc_RuntimeError, "All inputs must contain double-precision floating-point data");
        return false;
    }

    if( !PyArray_IS_C_CONTIGUOUS(camera_intrinsics) ||
        !PyArray_IS_C_CONTIGUOUS(camera_extrinsics) ||
        !PyArray_IS_C_CONTIGUOUS(frames)            ||
        !PyArray_IS_C_CONTIGUOUS(observations)      )
    {
        PyErr_SetString(PyExc_RuntimeError, "All inputs must be c-style contiguous arrays");
        return false;
    }

    return true;
}
static PyObject* optimize(PyObject* NPY_UNUSED(self),
                          PyObject* args,
                          PyObject* NPY_UNUSED(kwargs))
{
    PyObject* result = NULL;

    PyArrayObject* camera_intrinsics = NULL;
    PyArrayObject* camera_extrinsics = NULL;
    PyArrayObject* frames            = NULL;
    PyArrayObject* observations      = NULL;

    if(!PyArg_ParseTuple( args,
                          "O&O&O&O&",
                          PyArray_Converter, &camera_intrinsics,
                          PyArray_Converter, &camera_extrinsics,
                          PyArray_Converter, &frames,
                          PyArray_Converter, &observations))
        return NULL;

    if( !optimize_validate_args(camera_intrinsics,
                                camera_extrinsics,
                                frames,
                                observations ))
        goto done;





    // int        ndim     = PyArray_NDIM(arr);
    // npy_intp*  dims     = PyArray_DIMS(arr);
    // int        typenum  = PyArray_TYPE(arr);


    // // Useful metadata about this matrix
    // __attribute__((unused)) char*      data0    = PyArray_DATA    (arr);
    // __attribute__((unused)) char*      data1    = PyArray_BYTES   (arr);
    // __attribute__((unused)) npy_intp  *strides  = PyArray_STRIDES (arr);
    // __attribute__((unused)) int        ndim     = PyArray_NDIM    (arr);
    // __attribute__((unused)) npy_intp*  dims     = PyArray_DIMS    (arr);
    // __attribute__((unused)) npy_intp   itemsize = PyArray_ITEMSIZE(arr);
    // __attribute__((unused)) int        typenum  = PyArray_TYPE    (arr);


    // // Two ways to grab the data out of the matrix:
    // //
    // // 1. Call a function. Clear what this does, but if we need to access a lot
    // // of data in a loop, this has overhead: this function is pre-compiled, so
    // // the overhead will not be inlined
    // double d0 = *(double*)PyArray_GetPtr( arr, (npy_intp[]){i, j} );

    // // 2. inline the function ourselves. It's pretty simple
    // double d1 = *(double*)&data0[ i*strides[0] + j*strides[1] ];

    // // The two methods should be identical. If not, this example is wrong, and I
    // // barf
    // if( d0 != d1 )
    // {
    //     PyErr_Format(PyExc_RuntimeError, "PyArray_GetPtr() inlining didn't work: %f != %f", d0, d1);
    //     goto done;
    // }


    result = Py_BuildValue( "d", 5.0 );

 done:
    Py_DECREF(camera_intrinsics);
    Py_DECREF(camera_extrinsics);
    Py_DECREF(frames);
    Py_DECREF(observations);
    return result;
}

PyMODINIT_FUNC initmrcal(void)
{
    static PyMethodDef methods[] =
        { {"optimize", (PyCFunction)optimize, METH_VARARGS | METH_KEYWORDS, optimize_docstring},
         {}
        };


    PyImport_AddModule("mrcal");
    Py_InitModule3("mrcal", methods,
                   "Calibration and SFM routines");

    import_array();
}
