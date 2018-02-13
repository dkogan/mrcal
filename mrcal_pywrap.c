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
                                    PyArrayObject* points,
                                    PyArrayObject* observations_board,
                                    PyArrayObject* indices_frame_camera_board,
                                    PyArrayObject* observations_point,
                                    PyArrayObject* indices_point_camera_points,
                                    PyObject*      skipped_observations_board,
                                    PyObject*      skipped_observations_point,
                                    PyObject*      calibration_object_spacing,
                                    PyObject*      calibration_object_width_n,
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

    // calibration_object_spacing and calibration_object_width_n must be > 0 OR
    // we have to not be using a calibration board
    int c_calibration_object_width_n = 0;
    if( NobservationsBoard > 0 )
    {
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

    long int NobservationsPoint = PyArray_DIMS(observations_point)[0];
    if( PyArray_DIMS(indices_point_camera_points)[0] != NobservationsPoint )
    {
        PyErr_Format(PyExc_RuntimeError, "Inconsistent NobservationsPoint: 'observations_point' says %ld, 'indices_point_camera_points' says %ld",
                     NobservationsPoint,
                     PyArray_DIMS(indices_point_camera_points)[0]);
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

    if( !PyArray_IS_C_CONTIGUOUS(intrinsics)         ||
        !PyArray_IS_C_CONTIGUOUS(extrinsics)         ||
        !PyArray_IS_C_CONTIGUOUS(frames)             ||
        !PyArray_IS_C_CONTIGUOUS(points)             ||
        !PyArray_IS_C_CONTIGUOUS(observations_board) ||
        !PyArray_IS_C_CONTIGUOUS(observations_point) )
    {
        PyErr_SetString(PyExc_RuntimeError, "All inputs must be c-style contiguous arrays");
        return false;
    }

    if( PyArray_TYPE(indices_frame_camera_board)   != NPY_INT )
    {
        PyErr_SetString(PyExc_RuntimeError, "indices_frame_camera_board must contain int data");
        return false;
    }
    if( !PyArray_IS_C_CONTIGUOUS(indices_frame_camera_board) )
    {
        PyErr_SetString(PyExc_RuntimeError, "All inputs must be c-style contiguous arrays");
        return false;
    }

    if( PyArray_TYPE(indices_point_camera_points)   != NPY_INT )
    {
        PyErr_SetString(PyExc_RuntimeError, "indices_point_camera_points must contain int data");
        return false;
    }
    if( !PyArray_IS_C_CONTIGUOUS(indices_point_camera_points) )
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

static PyObject* getSupportedDistortionModels(PyObject* NPY_UNUSED(self),
                                              PyObject* NPY_UNUSED(args))
{
    PyObject* result = NULL;
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
    return result;
}

static PyObject* optimize(PyObject* NPY_UNUSED(self),
                          PyObject* args,
                          PyObject* kwargs)
{
    PyObject* result = NULL;

    PyArrayObject* intrinsics                  = NULL;
    PyArrayObject* extrinsics                  = NULL;
    PyArrayObject* frames                      = NULL;
    PyArrayObject* points                      = NULL;
    PyArrayObject* observations_board          = NULL;
    PyArrayObject* indices_frame_camera_board  = NULL;
    PyArrayObject* observations_point          = NULL;
    PyArrayObject* indices_point_camera_points = NULL;

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
                        "points",
                        "observations_board",
                        "indices_frame_camera_board",
                        "observation_point",
                        "indices_point_camera_points",

                        "distortion_model",

                        // optional kwargs
                        "do_optimize_intrinsic_core",
                        "do_optimize_intrinsic_distortions",
                        "do_optimize_extrinsics",
                        "do_optimize_frames",
                        "skipped_observations_board",
                        "skipped_observations_point",
                        "calibration_object_spacing",
                        "calibration_object_width_n",

                        NULL};

    PyObject* distortion_model_string           = NULL;
    PyObject* do_optimize_intrinsic_core        = Py_True;
    PyObject* do_optimize_intrinsic_distortions = Py_True;
    PyObject* do_optimize_extrinsics            = Py_True;
    PyObject* do_optimize_frames                = Py_True;
    PyObject* skipped_observations_board        = NULL;
    PyObject* skipped_observations_point        = NULL;
    PyObject* calibration_object_spacing        = NULL;
    PyObject* calibration_object_width_n        = NULL;
    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     "O&O&O&O&O&O&O&O&S|OOOOOOOO",
                                     keywords,
                                     PyArray_Converter, &intrinsics,
                                     PyArray_Converter, &extrinsics,
                                     PyArray_Converter, &frames,
                                     PyArray_Converter, &points,
                                     PyArray_Converter, &observations_board,
                                     PyArray_Converter, &indices_frame_camera_board,
                                     PyArray_Converter, &observations_point,
                                     PyArray_Converter, &indices_point_camera_points,

                                     &distortion_model_string,

                                     &do_optimize_intrinsic_core,
                                     &do_optimize_intrinsic_distortions,
                                     &do_optimize_extrinsics,
                                     &do_optimize_frames,
                                     &skipped_observations_board,
                                     &skipped_observations_point,
                                     &calibration_object_spacing,
                                     &calibration_object_width_n))
        goto done;

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
                                calibration_object_spacing,
                                calibration_object_width_n,
                                distortion_model_string))
        goto done;

    {
        int Ncameras           = PyArray_DIMS(intrinsics)[0];
        int Nframes            = PyArray_DIMS(frames)[0];
        int Npoints            = PyArray_DIMS(points)[0];
        int NobservationsBoard = PyArray_DIMS(observations_board)[0];
        int NobservationsPoint = PyArray_DIMS(observations_point)[0];


        double c_calibration_object_spacing = 0.0;
        if(PyFloat_Check(calibration_object_spacing))
            c_calibration_object_spacing = PyFloat_AS_DOUBLE(calibration_object_spacing);
        int c_calibration_object_width_n = 0;
        if(PyInt_Check(calibration_object_width_n))
            c_calibration_object_width_n = (int)PyInt_AS_LONG(calibration_object_width_n);


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





        struct mrcal_variable_select optimization_variable_choice = {};
        optimization_variable_choice.do_optimize_intrinsic_core = PyObject_IsTrue(do_optimize_intrinsic_core);
        optimization_variable_choice.do_optimize_intrinsic_distortions = PyObject_IsTrue(do_optimize_intrinsic_distortions);
        optimization_variable_choice.do_optimize_extrinsics = PyObject_IsTrue(do_optimize_extrinsics);
        optimization_variable_choice.do_optimize_frames = PyObject_IsTrue(do_optimize_frames);

        mrcal_optimize( c_intrinsics,
                        c_extrinsics,
                        c_frames,
                        c_points,
                        Ncameras, Nframes, Npoints,

                        c_observations_board,
                        NobservationsBoard,
                        c_observations_point,
                        NobservationsPoint,

                        false,
                        distortion_model,
                        optimization_variable_choice,

                        c_calibration_object_spacing,
                        c_calibration_object_width_n);
    }

    Py_INCREF(Py_None);
    result = Py_None;

 done:
    if(intrinsics)         Py_DECREF(intrinsics);
    if(extrinsics)         Py_DECREF(extrinsics);
    if(frames)             Py_DECREF(frames);
    if(points)             Py_DECREF(points);
    if(observations_board) Py_DECREF(observations_board);
    if(observations_point) Py_DECREF(observations_point);

    if( 0 != sigaction(SIGINT,
                       &sigaction_old, NULL ))
        PyErr_SetString(PyExc_RuntimeError, "sigaction-restore failed");

    return result;
}

PyMODINIT_FUNC initoptimizer(void)
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

#define PYMETHODDEF_ENTRY(x, args) {#x, (PyCFunction)x, args, x ## _docstring}

    static PyMethodDef methods[] =
        { PYMETHODDEF_ENTRY(optimize,                     METH_VARARGS | METH_KEYWORDS),
          PYMETHODDEF_ENTRY(getNdistortionParams,         METH_VARARGS),
          PYMETHODDEF_ENTRY(getSupportedDistortionModels, METH_NOARGS),
          {}
        };

    PyImport_AddModule("optimizer");
    Py_InitModule3("optimizer", methods,
                   "Calibration and SFM routines");

    import_array();
}
