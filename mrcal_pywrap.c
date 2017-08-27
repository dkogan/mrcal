#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <stdbool.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <signal.h>

#include "mrcal.h"

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
    if( PyArray_NDIM(observations) != 5 )
    {
        PyErr_SetString(PyExc_RuntimeError, "'observations' must have exactly 5 dims");
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
    if( NUM_INTRINSIC_PARAMS != PyArray_DIMS(camera_intrinsics)[1] )
    {
        PyErr_Format(PyExc_RuntimeError, "intrinsics.shape[1] MUST be %d. Instead got %ld",
                     NUM_INTRINSIC_PARAMS,
                     PyArray_DIMS(camera_intrinsics)[1] );
        return false;
    }

    static_assert( sizeof(struct pose_t)/sizeof(double) == 6, "pose_t is assumed to contain 6 elements");

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
        10 != PyArray_DIMS(observations)[3] ||
        2  != PyArray_DIMS(observations)[4] )
    {
        PyErr_Format(PyExc_RuntimeError, "observations.shape[2:] MUST be (10,10,2). Instead got (%ld,%ld,%ld)",
                     PyArray_DIMS(observations)[2],
                     PyArray_DIMS(observations)[3],
                     PyArray_DIMS(observations)[4]);
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

    // Python is silly. There's some nuance about signal handling where it sets
    // a SIGINT (ctrl-c) handler to just set a flag, and the python layer then
    // reads this flag and does the thing. Here I'm running C code, so SIGINT
    // would set a flag, but not quit, so I can't interrupt the solver. Thus I
    // reset the SIGINT handler to the default, and put it back to the
    // python-specific version when I'm done
    struct sigaction sigaction_old;
    if( 0 != sigaction(SIGINT,
                       &(struct sigaction){ .sa_handler = SIG_DFL },
                       &sigaction_old) )
    {
        PyErr_SetString(PyExc_RuntimeError, "sigaction() failed");
        goto done;
    }

    if(!PyArg_ParseTuple( args,
                          "O&O&O&O&",
                          PyArray_Converter, &camera_intrinsics,
                          PyArray_Converter, &camera_extrinsics,
                          PyArray_Converter, &frames,
                          PyArray_Converter, &observations))
        goto done;

    if( !optimize_validate_args(camera_intrinsics,
                                camera_extrinsics,
                                frames,
                                observations ))
        goto done;


    {
        int Ncameras      = PyArray_DIMS(camera_intrinsics)[0];
        int Nframes       = PyArray_DIMS(frames)[0];
        int Nobservations = Ncameras * Nframes;

        // The checks in optimize_validate_args() make sure these casts are kosher
        struct intrinsics_t* c_camera_intrinsics = (struct intrinsics_t*)PyArray_DATA(camera_intrinsics);
        struct pose_t*       c_camera_extrinsics = (struct pose_t*)      PyArray_DATA(camera_extrinsics);
        struct pose_t*       c_frames            = (struct pose_t*)      PyArray_DATA(frames);

        struct observation_t c_observations[Nobservations];
        int i_observation = 0;
        for( int i_frame=0; i_frame<Nframes; i_frame++ )
            for( int i_camera=0; i_camera<Ncameras; i_camera++, i_observation++ )
            {
                c_observations[i_observation].i_camera = i_camera;
                c_observations[i_observation].i_frame  = i_frame;
                c_observations[i_observation].px       = &((union point2_t*)PyArray_DATA(observations))[10*10*i_observation];
            }

        mrcal_optimize( c_camera_intrinsics,
                        c_camera_extrinsics,
                        c_frames,
                        Ncameras, Nframes,

                        c_observations,
                        Nobservations,
                        false );
    }

    Py_INCREF(Py_None);
    result = Py_None;

 done:
    if(camera_intrinsics) Py_DECREF(camera_intrinsics);
    if(camera_extrinsics) Py_DECREF(camera_extrinsics);
    if(frames)            Py_DECREF(frames);
    if(observations)      Py_DECREF(observations);

    if( 0 != sigaction(SIGINT,
                       &sigaction_old, NULL ))
        PyErr_SetString(PyExc_RuntimeError, "sigaction-restore failed");

    return result;
}

PyMODINIT_FUNC initmrcal(void)
{
    static const char optimize_docstring[] =
#include "optimize.docstring.h"
        ;
    static PyMethodDef methods[] =
        { {"optimize", (PyCFunction)optimize, METH_VARARGS | METH_KEYWORDS, optimize_docstring},
         {}
        };


    PyImport_AddModule("mrcal");
    Py_InitModule3("mrcal", methods,
                   "Calibration and SFM routines");

    import_array();
}
