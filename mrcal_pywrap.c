#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <stdbool.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <signal.h>

#include "mrcal.h"

static bool optimize_validate_args( // out
                                    enum distortion_model_t* distortion_model,

                                    // in
                                    PyArrayObject* intrinsics,
                                    PyArrayObject* extrinsics,
                                    PyArrayObject* frames,
                                    PyArrayObject* observations,
                                    PyObject*      distortion_model_string)
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
    if( PyArray_NDIM(observations) != 5 )
    {
        PyErr_SetString(PyExc_RuntimeError, "'observations' must have exactly 5 dims");
        return false;
    }

    int Ncameras = PyArray_DIMS(intrinsics)[0];
    if( Ncameras-1 !=
        PyArray_DIMS(extrinsics)[0] )
    {
        PyErr_Format(PyExc_RuntimeError, "Inconsistent Ncameras: 'extrinsics' says %ld, intrinsics says %ld",
                     PyArray_DIMS(extrinsics)[0] + 1,
                     PyArray_DIMS(intrinsics)[0] );
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
        PyErr_Format(PyExc_RuntimeError, "Inconsistent Ncameras: 'intrinsics' says %ld, 'observations' says %ld",
                     PyArray_DIMS(intrinsics)[0],
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


    if( PyArray_TYPE(intrinsics)   != NPY_DOUBLE ||
        PyArray_TYPE(extrinsics)   != NPY_DOUBLE ||
        PyArray_TYPE(frames)       != NPY_DOUBLE ||
        PyArray_TYPE(observations) != NPY_DOUBLE )
    {
        PyErr_SetString(PyExc_RuntimeError, "All inputs must contain double-precision floating-point data");
        return false;
    }

    if( !PyArray_IS_C_CONTIGUOUS(intrinsics) ||
        !PyArray_IS_C_CONTIGUOUS(extrinsics) ||
        !PyArray_IS_C_CONTIGUOUS(frames)     ||
        !PyArray_IS_C_CONTIGUOUS(observations) )
    {
        PyErr_SetString(PyExc_RuntimeError, "All inputs must be c-style contiguous arrays");
        return false;
    }

    const char* distortion_model_cstring =
        PyString_AsString(distortion_model_string);
    if( distortion_model_cstring == NULL)
    {
#define QUOTED_LIST_WITH_COMMA(s,n) "'" #s "',"
        PyErr_SetString(PyExc_RuntimeError, "Distortion model was not passed in. Must be a string, one of ("
                        DISTORTION_LIST( QUOTED_LIST_WITH_COMMA )
                        ")");
        return false;
    }

    *distortion_model = DISTORTION_INVALID;
    do
    {
#define CHECK_AND_SET(s,n)                                      \
        if( 0 == strcmp( distortion_model_cstring, #s) )        \
        {                                                       \
            *distortion_model = s;                              \
            break;                                              \
        }

        DISTORTION_LIST( CHECK_AND_SET );
    } while(0);

    if( *distortion_model == DISTORTION_INVALID )
    {
        PyErr_Format(PyExc_RuntimeError, "Invalid distortion model was passed in: '%s'. Must be a string, one of ("
                     DISTORTION_LIST( QUOTED_LIST_WITH_COMMA )
                     ")",
                     distortion_model_cstring);
        return false;
    }


    int NdistortionParams = getNdistortionParams(*distortion_model);
    if( N_INTRINSICS_CORE + NdistortionParams != PyArray_DIMS(intrinsics)[1] )
    {
        PyErr_Format(PyExc_RuntimeError, "intrinsics.shape[1] MUST be %d. Instead got %ld",
                     N_INTRINSICS_CORE + NdistortionParams,
                     PyArray_DIMS(intrinsics)[1] );
        return false;
    }

    return true;
}

static PyObject* optimize(PyObject* NPY_UNUSED(self),
                          PyObject* args,
                          PyObject* kwargs)
{
    PyObject* result = NULL;

    PyArrayObject* intrinsics   = NULL;
    PyArrayObject* extrinsics   = NULL;
    PyArrayObject* frames       = NULL;
    PyArrayObject* observations = NULL;

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


    char* keywords[] = {"intrinsics",
                        "extrinsics",
                        "frames",
                        "observations",
                        "distortion_model",

                        // optional kwargs
                        "do_optimize_intrinsics",
                        NULL};

    PyObject* distortion_model_string = NULL;
    PyObject* do_optimize_intrinsics = Py_True;
    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     "O&O&O&O&S|O",
                                     keywords,
                                     PyArray_Converter, &intrinsics,
                                     PyArray_Converter, &extrinsics,
                                     PyArray_Converter, &frames,
                                     PyArray_Converter, &observations,

                                     &distortion_model_string,
                                     &do_optimize_intrinsics))
        goto done;

    enum distortion_model_t distortion_model;
    if( !optimize_validate_args(&distortion_model,

                                intrinsics,
                                extrinsics,
                                frames,
                                observations,
                                distortion_model_string))
        goto done;

    {
        int Ncameras      = PyArray_DIMS(intrinsics)[0];
        int Nframes       = PyArray_DIMS(frames)[0];
        int Nobservations = Ncameras * Nframes;

        // The checks in optimize_validate_args() make sure these casts are kosher
        struct intrinsics_t* c_intrinsics = (struct intrinsics_t*)PyArray_DATA(intrinsics);
        struct pose_t*       c_extrinsics = (struct pose_t*)      PyArray_DATA(extrinsics);
        struct pose_t*       c_frames     = (struct pose_t*)      PyArray_DATA(frames);

        struct observation_t c_observations[Nobservations];
        int i_observation = 0;
        for( int i_frame=0; i_frame<Nframes; i_frame++ )
            for( int i_camera=0; i_camera<Ncameras; i_camera++, i_observation++ )
            {
                c_observations[i_observation].i_camera = i_camera;
                c_observations[i_observation].i_frame  = i_frame;
                c_observations[i_observation].px       = &((union point2_t*)PyArray_DATA(observations))[10*10*i_observation];
            }

        mrcal_optimize( c_intrinsics,
                        c_extrinsics,
                        c_frames,
                        Ncameras, Nframes,

                        c_observations,
                        Nobservations,
                        false,
                        distortion_model,
                        PyObject_IsTrue(do_optimize_intrinsics));
    }

    Py_INCREF(Py_None);
    result = Py_None;

 done:
    if(intrinsics)   Py_DECREF(intrinsics);
    if(extrinsics)   Py_DECREF(extrinsics);
    if(frames)       Py_DECREF(frames);
    if(observations) Py_DECREF(observations);

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
