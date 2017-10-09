#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <stdbool.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <signal.h>

#include "mrcal.h"


#define Nwant CALOBJECT_W


static bool optimize_validate_args( // out
                                    enum distortion_model_t* distortion_model,

                                    // in
                                    PyArrayObject* intrinsics,
                                    PyArrayObject* extrinsics,
                                    PyArrayObject* frames,
                                    PyArrayObject* observations,
                                    PyArrayObject* indices_frame_camera,
                                    PyObject*      skipped_observations,
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
    if( PyArray_NDIM(observations) != 4 )
    {
        PyErr_SetString(PyExc_RuntimeError, "'observations' must have exactly 4 dims");
        return false;
    }
    if( PyArray_NDIM(indices_frame_camera) != 2 )
    {
        PyErr_SetString(PyExc_RuntimeError, "'indices_frame_camera' must have exactly 2 dims");
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

    long int Nobservations = PyArray_DIMS(observations)[0];
    if( PyArray_DIMS(indices_frame_camera)[0] != Nobservations )
    {
        PyErr_Format(PyExc_RuntimeError, "Inconsistent Nobservations: 'observations' says %ld, 'indices_frame_camera' says %ld",
                     Nobservations,
                     PyArray_DIMS(indices_frame_camera)[0]);
        return false;
    }
    if( Nwant != PyArray_DIMS(observations)[1] ||
        Nwant != PyArray_DIMS(observations)[2] ||
        2  != PyArray_DIMS(observations)[3] )
    {
        PyErr_Format(PyExc_RuntimeError, "observations.shape[1:] MUST be (Nwant,Nwant,2). Instead got (%ld,%ld,%ld)",
                     PyArray_DIMS(observations)[1],
                     PyArray_DIMS(observations)[2],
                     PyArray_DIMS(observations)[3]);
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

    if( PyArray_TYPE(indices_frame_camera)   != NPY_INT )
    {
        PyErr_SetString(PyExc_RuntimeError, "indices_frame_camera must contain int data");
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

    if( skipped_observations != NULL &&
        skipped_observations != Py_None)
    {
        if( !PySequence_Check(skipped_observations) )
        {
            PyErr_Format(PyExc_RuntimeError, "skipped_observations MUST be None or an iterable of monotonically-increasing integers >= 0");
            return false;
        }

        int Nskipped_observations = (int)PySequence_Size(skipped_observations);
        long iskip_last = -1;
        for(int i=0; i<Nskipped_observations; i++)
        {
            PyObject* nextskip = PySequence_GetItem(skipped_observations, i);
            if(!PyInt_Check(nextskip))
            {
                PyErr_Format(PyExc_RuntimeError, "skipped_observations MUST be None or an iterable of monotonically-increasing integers >= 0");
                return false;
            }
            long iskip = PyInt_AsLong(nextskip);
            if(iskip <= iskip_last)
            {
                PyErr_Format(PyExc_RuntimeError, "skipped_observations MUST be None or an iterable of monotonically-increasing integers >= 0");
                return false;
            }
            iskip_last = iskip;
        }
    }
    return true;
}

static PyObject* getNdistortionParams(PyObject* NPY_UNUSED(self),
                                      PyObject* args)
{
    PyObject* result = NULL;

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
    return result;
}

static PyObject* optimize(PyObject* NPY_UNUSED(self),
                          PyObject* args,
                          PyObject* kwargs)
{
    PyObject* result = NULL;

    PyArrayObject* intrinsics           = NULL;
    PyArrayObject* extrinsics           = NULL;
    PyArrayObject* frames               = NULL;
    PyArrayObject* observations         = NULL;
    PyArrayObject* indices_frame_camera = NULL;

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
                        "indices_frame_camera",
                        "distortion_model",

                        // optional kwargs
                        "do_optimize_intrinsic_core",
                        "do_optimize_intrinsic_distortions",
                        "skipped_observations",
                        NULL};

    PyObject* distortion_model_string = NULL;
    PyObject* do_optimize_intrinsic_core        = Py_True;
    PyObject* do_optimize_intrinsic_distortions = Py_True;
    PyObject* skipped_observations              = NULL;
    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     "O&O&O&O&O&S|OOO",
                                     keywords,
                                     PyArray_Converter, &intrinsics,
                                     PyArray_Converter, &extrinsics,
                                     PyArray_Converter, &frames,
                                     PyArray_Converter, &observations,
                                     PyArray_Converter, &indices_frame_camera,

                                     &distortion_model_string,

                                     &do_optimize_intrinsic_core,
                                     &do_optimize_intrinsic_distortions,
                                     &skipped_observations))
        goto done;

    enum distortion_model_t distortion_model;
    if( !optimize_validate_args(&distortion_model,

                                intrinsics,
                                extrinsics,
                                frames,
                                observations,
                                indices_frame_camera,
                                skipped_observations,
                                distortion_model_string))
        goto done;

    {
        int Ncameras      = PyArray_DIMS(intrinsics)[0];
        int Nframes       = PyArray_DIMS(frames)[0];
        int Nobservations = PyArray_DIMS(observations)[0];

        // The checks in optimize_validate_args() make sure these casts are kosher
        struct intrinsics_t* c_intrinsics = (struct intrinsics_t*)PyArray_DATA(intrinsics);
        struct pose_t*       c_extrinsics = (struct pose_t*)      PyArray_DATA(extrinsics);
        struct pose_t*       c_frames     = (struct pose_t*)      PyArray_DATA(frames);

        struct observation_board_t c_observations[Nobservations];

        int Nskipped_observations =
            ( skipped_observations == NULL ||
              skipped_observations == Py_None ) ?
            0 :
            (int)PySequence_Size(skipped_observations);
        int i_skipped_observation = 0;
        int i_observation_next_skip = -1;
        if( i_skipped_observation < Nskipped_observations )
        {
            PyObject* nextskip = PySequence_GetItem(skipped_observations, i_skipped_observation);
            i_observation_next_skip = (int)PyInt_AsLong(nextskip);
        }

        int i_frame_current_skipped = -1;
        int i_frame_last            = -1;
        for(int i_observation=0; i_observation<Nobservations; i_observation++)
        {
            int i_frame  = ((int*)PyArray_DATA(indices_frame_camera))[i_observation*2 + 0];
            int i_camera = ((int*)PyArray_DATA(indices_frame_camera))[i_observation*2 + 1];

            c_observations[i_observation].i_camera         = i_camera;
            c_observations[i_observation].i_frame          = i_frame;
            c_observations[i_observation].px               = &((union point2_t*)PyArray_DATA(observations))[Nwant*Nwant*i_observation];

            // I skip this frame if I skip ALL observations of this frame
            if( i_frame_current_skipped >= 0 &&
                i_frame_current_skipped != i_frame )
            {
                // Ooh! We moved past the frame where we skipped all
                // observations. So I need to go back, and mark all of those as
                // skipping that frame
                for(int i_observation_skip_frame = i_observation-1;
                    i_observation_skip_frame >= 0 && c_observations[i_observation_skip_frame].i_frame == i_frame_current_skipped;
                    i_observation_skip_frame--)
                {
                    c_observations[i_observation_skip_frame].skip_frame = true;
                }
            }
            else
                c_observations[i_observation].skip_frame = false;

            if( i_observation == i_observation_next_skip )
            {
                if( i_frame_last != i_frame )
                    i_frame_current_skipped = i_frame;

                c_observations[i_observation].skip_observation = true;

                i_skipped_observation++;
                if( i_skipped_observation < Nskipped_observations )
                {
                    PyObject* nextskip = PySequence_GetItem(skipped_observations, i_skipped_observation);
                    i_observation_next_skip = (int)PyInt_AsLong(nextskip);
                }
                else
                    i_observation_next_skip = -1;
            }
            else
            {
                c_observations[i_observation].skip_observation = false;
                i_frame_current_skipped = -1;
            }

            i_frame_last = i_frame;
        }
        // check for frame-skips on the last observation
        if( i_frame_current_skipped >= 0 )
        {
            // Ooh! We moved past the frame where we skipped all
            // observations. So I need to go back, and mark all of those as
            // skipping that frame
            for(int i_observation_skip_frame = Nobservations - 1;
                i_observation_skip_frame >= 0 && c_observations[i_observation_skip_frame].i_frame == i_frame_current_skipped;
                i_observation_skip_frame--)
            {
                c_observations[i_observation_skip_frame].skip_frame = true;
            }
        }

        struct mrcal_variable_select optimization_variable_choice;
        optimization_variable_choice.do_optimize_intrinsic_core =
            PyObject_IsTrue(do_optimize_intrinsic_core);
        optimization_variable_choice.do_optimize_intrinsic_distortions =
            PyObject_IsTrue(do_optimize_intrinsic_distortions);
        mrcal_optimize( c_intrinsics,
                        c_extrinsics,
                        c_frames,
                        Ncameras, Nframes,

                        c_observations,
                        Nobservations,
                        false,
                        distortion_model,
                        optimization_variable_choice);
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
    static const char getNdistortionParams_docstring[] =
#include "getNdistortionParams.docstring.h"
        ;
    static PyMethodDef methods[] =
        { {"optimize",             (PyCFunction)optimize,             METH_VARARGS | METH_KEYWORDS, optimize_docstring},
          {"getNdistortionParams", (PyCFunction)getNdistortionParams, METH_VARARGS,                 getNdistortionParams_docstring},
         {}
        };


    PyImport_AddModule("mrcal");
    Py_InitModule3("mrcal", methods,
                   "Calibration and SFM routines");

    import_array();
}
