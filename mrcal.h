#pragma once

#include <stdbool.h>

#include "basic_points.h"







// unconstrained 6DOF pose containing a rodrigues rotation and a translation
struct pose_t
{
    union point3_t r,t;
};

// An observation of a calibration board. Each "observation" is ONE camera
// observing a board
struct observation_board_t
{
    int  i_camera         : 31;
    bool skip_frame       : 1;
    int  i_frame          : 31;
    bool skip_observation : 1;

    union point2_t* px; // NUM_POINTS_IN_CALOBJECT of these
};

// An observation of a point in space. Here each "observation" is two cameras
// observing the point. The 3d position of the point is NOT given in the
// parameter vector, but the reprojection error is computed directly from the
// camera geometry. Optionally, the distance to from each camera to the point is
// given also. This is optional, and used only if the given distance is > 0.
struct observation_point_t
{
    int  i_camera         : 31;
    bool skip_point       : 1;
    int  i_point          : 31;
    bool skip_observation : 1;

    // Observed pixel coordinates
    union point2_t px;

    // Reference distance. This is optional; skipped if <= 0
    double dist;
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
    _(DISTORTION_CAHVORE, 9) /* CAHVORE is CAHVOR + E + linearity */

#define LIST_WITH_COMMA(s,n) s,
enum distortion_model_t
    { DISTORTION_LIST( LIST_WITH_COMMA ) DISTORTION_INVALID };

#define DECLARE_CUSTOM_INTRINSICS(s,n) \
    struct intrinsics_ ## s ## _t { INTRINSICS_CORE; double distortions[n]; };
DISTORTION_LIST( DECLARE_CUSTOM_INTRINSICS )


const char*             mrcal_distortion_model_name       ( enum distortion_model_t model );
enum distortion_model_t mrcal_distortion_model_from_name  ( const char* name );
int                     mrcal_getNdistortionParams        ( const enum distortion_model_t m );
const char* const*      mrcal_getSupportedDistortionModels( void ); // NULL-terminated array of char* strings

struct mrcal_variable_select
{
    bool do_optimize_intrinsic_core        : 1;
    bool do_optimize_intrinsic_distortions : 1;
};

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
                      union  point3_t*     points,             // Npoints of these.    In the camera0 frame

                      // in
                      int Ncameras, int Nframes, int Npoints,

                      const struct observation_board_t* observations_board,
                      int NobservationsBoard,

                      const struct observation_point_t* observations_point,
                      int NobservationsPoint,

                      bool check_gradient,
                      enum distortion_model_t distortion_model,
                      struct mrcal_variable_select optimization_variable_choice,

                      double calibration_object_spacing,
                      int calibration_object_width_n);
