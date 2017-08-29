#pragma once

#include <stdbool.h>


// this is copied from the stereo-server project. Please consolidate
union point2_t
{
    struct
    {
        double x,y;
    };
    double xy[2];
};

union point3_t
{
    struct
    {
        double x,y,z;
    };
    double xyz[3];
};



#warning generalize to other calibration objects and lone points
#define CALOBJECT_W                    10
#define NUM_POINTS_IN_CALOBJECT        (CALOBJECT_W*CALOBJECT_W)
#define CALIBRATION_OBJECT_DOT_SPACING (4.0 * 2.54 / 100.0) /* 4 inches */


// unconstrained 6DOF pose containing a rodrigues rotation and a translation
struct pose_t
{
    union point3_t r,t;
};

struct observation_t
{
#warning I need i_camera, but maybe i_frame should live in a separate frame_start[] ?
    int i_camera, i_frame;

    union point2_t* px; // NUM_POINTS_IN_CALOBJECT of these
};



#define INTRINSICS_CORE                         \
    double focal_xy [2];                        \
    double center_xy[2]

struct intrinsics_t
{
    INTRINSICS_CORE;
    double distortions[];
};
#define N_INTRINSICS_CORE 4


// names of distortion models, number of distortion parameters
#define DISTORTION_LIST(_)                      \
    _(DISTORTION_NONE,    0)                    \
    _(DISTORTION_OPENCV4, 4)                    \
    _(DISTORTION_OPENCV5, 5)                    \
    _(DISTORTION_OPENCV8, 8)                    \
    _(DISTORTION_CAHVOR,  5)                    \
    _(DISTORTION_CAHVORE, 8)

#define LIST_WITH_COMMA(s,n) s,
enum distortion_model_t
    { DISTORTION_LIST( LIST_WITH_COMMA ) DISTORTION_INVALID };

#define DECLARE_CUSTOM_INTRINSICS(s,n) \
    struct intrinsics_ ## s ## _t { INTRINSICS_CORE; double distortions[n]; };
DISTORTION_LIST( DECLARE_CUSTOM_INTRINSICS )


const char* mrcal_distortion_model_name( enum distortion_model_t model );
int mrcal_getNdistortionParams(const enum distortion_model_t m);

double mrcal_optimize( // out, in (seed on input)

                      // These are the state. I don't have a state_t because Ncameras
                      // and Nframes aren't known at compile time.
                      //
                      // camera_intrinsics is struct intrinsics_t: a
                      // concatenation of the intrinsics core and the distortion
                      // params. The specific distortion parameters may vary,
                      // depending on distortion_model, so this is a
                      // variable-length structure
                      struct intrinsics_t* camera_intrinsics,  // Ncameras of these
                      struct pose_t*       camera_extrinsics,  // Ncameras-1 of these. Transform FROM camera0 frame
                      struct pose_t*       frames,             // Nframes of these.    Transform TO   camera0 frame

                      // in
                      int Ncameras, int Nframes,

                      const struct observation_t* observations,
                      int Nobservations,

                      bool check_gradient,
                      enum distortion_model_t distortion_model,
                      bool do_optimize_intrinsics);
