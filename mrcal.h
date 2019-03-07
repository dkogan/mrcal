#pragma once

#include <stdbool.h>

#include "basic_points.h"







// unconstrained 6DOF pose containing a rodrigues rotation and a translation
typedef struct
{
    point3_t r,t;
} pose_t;

// An observation of a calibration board. Each "observation" is ONE camera
// observing a board
typedef struct
{
    int  i_camera         : 31;
    bool skip_frame       : 1;
    int  i_frame          : 31;
    bool skip_observation : 1;

    point2_t* px; // NUM_POINTS_IN_CALOBJECT of these
} observation_board_t;

typedef struct
{
    int  i_camera         : 31;
    bool skip_point       : 1;
    int  i_point          : 31;
    bool skip_observation : 1;

    // Observed pixel coordinates
    point2_t px;

    // Reference distance. This is optional; skipped if <= 0
    double dist;
} observation_point_t;



typedef struct
{
    double focal_xy [2];
    double center_xy[2];
} intrinsics_core_t;
#define N_INTRINSICS_CORE ((int)(sizeof(intrinsics_core_t)/sizeof(double)))


// names of distortion models, number of distortion parameters
#define DISTORTION_LIST(_)                      \
    _(DISTORTION_NONE,    0)                    \
    _(DISTORTION_OPENCV4, 4)                    \
    _(DISTORTION_OPENCV5, 5)                    \
    _(DISTORTION_OPENCV8, 8)                    \
    _(DISTORTION_OPENCV12,12) /* available in OpenCV >= 3.0.0) */ \
    _(DISTORTION_OPENCV14,14) /* available in OpenCV >= 3.1.0) */ \
    _(DISTORTION_CAHVOR,  5)                    \
    _(DISTORTION_CAHVORE, 9) /* CAHVORE is CAHVOR + E + linearity */
#define DISTORTION_OPENCV_FIRST DISTORTION_OPENCV4
#define DISTORTION_OPENCV_LAST  DISTORTION_OPENCV14
#define DISTORTION_CAHVOR_FIRST DISTORTION_CAHVOR
#define DISTORTION_CAHVOR_LAST  DISTORTION_CAHVORE
#define DISTORTION_IS_OPENCV(d) (DISTORTION_OPENCV_FIRST <= (d) && (d) <= DISTORTION_OPENCV_LAST)
#define DISTORTION_IS_CAHVOR(d) (DISTORTION_CAHVOR_FIRST <= (d) && (d) <= DISTORTION_CAHVOR_LAST)

#define LIST_WITH_COMMA(s,n) ,s
typedef enum
    { DISTORTION_INVALID DISTORTION_LIST( LIST_WITH_COMMA ) } distortion_model_t;


typedef struct
{
    bool do_optimize_intrinsic_core        : 1;
    bool do_optimize_intrinsic_distortions : 1;
    bool do_optimize_extrinsics            : 1;
    bool do_optimize_frames                : 1;
    bool do_skip_regularization            : 1;
    bool do_optimize_cahvor_optical_axis   : 1;
} mrcal_problem_details_t;
#define DO_OPTIMIZE_ALL ((mrcal_problem_details_t) { .do_optimize_intrinsic_core        = true, \
                                                     .do_optimize_intrinsic_distortions = true, \
                                                     .do_optimize_extrinsics            = true, \
                                                     .do_optimize_frames                = true, \
                                                     .do_optimize_cahvor_optical_axis   = true, \
                                                     .do_skip_regularization            = false})
#define IS_OPTIMIZE_NONE(x)                     \
    (!(x).do_optimize_intrinsic_core &&         \
     !(x).do_optimize_intrinsic_distortions &&  \
     !(x).do_optimize_extrinsics &&             \
     !(x).do_optimize_frames)


const char*             mrcal_distortion_model_name       ( distortion_model_t model );
distortion_model_t      mrcal_distortion_model_from_name  ( const char* name );
int                     mrcal_getNdistortionParams        ( const distortion_model_t m );
int                     mrcal_getNintrinsicParams         ( const distortion_model_t m );
int                     mrcal_getNintrinsicOptimizationParams( mrcal_problem_details_t problem_details,
                                                               distortion_model_t m );
const char* const*      mrcal_getSupportedDistortionModels( void ); // NULL-terminated array of char* strings

// Returns the 'next' distortion model in a family
//
// In a family of distortion models we have a sequence of models with increasing
// complexity. Subsequent models add distortion parameters to the end of the
// vector. Ealier models are identical, but with the extra paramaters set to 0.
// This function returns the next model in a sequence.
//
// If this is the last model in the sequence, returns the current model. This
// function takes in both the current model, and the last model we're aiming
// for. The second part is required because all familie begin at
// DISTORTION_NONE, so the next model from DISTORTION_NONE is not well-defined
// without more information
distortion_model_t mrcal_getNextDistortionModel( distortion_model_t distortion_model_now,
                                                      distortion_model_t distortion_model_final);

void mrcal_project( // out
                   point2_t* out,

                   // core, distortions concatenated. Stored as a row-first
                   // array of shape (N,2,Nintrinsics)
                   double*         dxy_dintrinsics,
                   // Stored as a row-first array of shape (N,2). Each element
                   // of this array is a point3_t
                   point3_t* dxy_dp,

                   // in
                   const point3_t* p,
                   int N,
                   distortion_model_t distortion_model,
                   // core, distortions concatenated
                   const double* intrinsics);

// Maps a set of undistorted 2D imager points q to a set of imager points that
// would result from observing the same vectors with a distorted model. Here the
// undistorted model is a pinhole camera with
//
// - the same center pixel coord as the distorted camera
// - the distorted-camera focal length scaled by a factor of scale_f_pinhole
bool mrcal_distort( // out
                   point2_t* out,

                   // in
                   const point2_t* q,
                   int N,
                   distortion_model_t distortion_model,
                   // core, distortions concatenated
                   const double* intrinsics,
                   double scale_f_pinhole);



#define MRCAL_STATS_ITEM(_)                                           \
    _(double,         rms_reproj_error__pixels,   PyFloat_FromDouble) \
    _(int,            NoutsideROI,                PyInt_FromLong)     \
    _(int,            Noutliers,                  PyInt_FromLong)

#define MRCAL_STATS_ITEM_DEFINE(type, name, pyconverter) type name;

typedef struct
{
    MRCAL_STATS_ITEM(MRCAL_STATS_ITEM_DEFINE)
} mrcal_stats_t;

mrcal_stats_t
mrcal_optimize( // out
                // These may be NULL. They're for diagnostic reporting to the
                // caller
                double* x_final,
                double* invJtJ_intrinsics_full,
                double* invJtJ_intrinsics_observations_only,
                // Buffer should be at least Nfeatures long. stats->Noutliers
                // elements will be filled in
                int*    outlier_indices_final,
                // Buffer should be at least Nfeatures long. stats->NoutsideROI
                // elements will be filled in
                int*    outside_ROI_indices_final,

                // out, in
                //
                // This is a dogleg_solverContext_t. I don't want to #include
                // <dogleg.h> here, so this is void
                //
                // if(_solver_context != NULL) then this is a persistent solver
                // context. The context is NOT freed on exit.
                // mrcal_free_context() should be called to release it
                //
                // if(*_solver_context != NULL), the given context is reused
                // if(*_solver_context == NULL), a context is created, and
                // returned here on exit
                void** _solver_context,

                // These are a seed on input, solution on output
                // These are the state. I don't have a state_t because Ncameras
                // and Nframes aren't known at compile time.
                //
                // camera_intrinsics is a concatenation of the intrinsics
                // core and the distortion params. The specific distortion
                // parameters may vary, depending on distortion_model, so
                // this is a variable-length structure
                double*       camera_intrinsics,  // Ncameras * (N_INTRINSICS_CORE + Ndistortions)
                pose_t*       camera_extrinsics,  // Ncameras-1 of these. Transform FROM camera0 frame
                pose_t*       frames,             // Nframes of these.    Transform TO   camera0 frame
                point3_t*     points,             // Npoints of these.    In the camera0 frame

                // in
                int Ncameras, int Nframes, int Npoints,

                const observation_board_t* observations_board,
                int NobservationsBoard,

                const observation_point_t* observations_point,
                int NobservationsPoint,

                bool check_gradient,
                int Noutlier_indices_input,
                int* outlier_indices_input,

                // region-of-interest. If not NULL, errors for observations
                // outside this region are strongly attenuated. The region is
                // specified separately for each camera. Each region is an
                // ellipse, represented as a 4-double slice with values
                // (x_center, y_center, x_width, y_width)
                const double* roi,

                bool VERBOSE,
                const bool skip_outlier_rejection,

                distortion_model_t distortion_model,
                double observed_pixel_uncertainty,
                const int* imagersizes, // Ncameras*2 of these
                mrcal_problem_details_t problem_details,

                double calibration_object_spacing,
                int calibration_object_width_n);

int mrcal_getNmeasurements_all(int Ncameras, int NobservationsBoard,
                               const observation_point_t* observations_point,
                               int NobservationsPoint,
                               int calibration_object_width_n,
                               mrcal_problem_details_t problem_details,
                               distortion_model_t distortion_model);
int mrcal_getNmeasurements_boards(int NobservationsBoard,
                                  int calibration_object_width_n);
int mrcal_getNmeasurements_points(const observation_point_t* observations_point,
                                  int NobservationsPoint);
int mrcal_getNmeasurements_regularization(int Ncameras,
                                          mrcal_problem_details_t problem_details,
                                          distortion_model_t distortion_model);

// frees a dogleg_solverContext_t. I don't want to #include <dogleg.h> here, so
// this is void
void mrcal_free_context(void** ctx);


int mrcal_state_index_intrinsic_core(int i_camera,
                                     mrcal_problem_details_t problem_details,
                                     distortion_model_t distortion_model);
int mrcal_state_index_intrinsic_distortions(int i_camera,
                                            mrcal_problem_details_t problem_details,
                                            distortion_model_t distortion_model);
int mrcal_state_index_camera_rt(int i_camera, int Ncameras,
                                mrcal_problem_details_t problem_details,
                                distortion_model_t distortion_model);
int mrcal_state_index_frame_rt(int i_frame, int Ncameras,
                               mrcal_problem_details_t problem_details,
                               distortion_model_t distortion_model);
int mrcal_state_index_point(int i_point, int Nframes, int Ncameras,
                            mrcal_problem_details_t problem_details,
                            distortion_model_t distortion_model);

// packs/unpacks a vector
void mrcal_pack_solver_state_vector( // out, in
                                     double* p, // unitless, FULL state on
                                                // input, scaled, decimated
                                                // (subject to problem_details),
                                                // meaningful state on output

                                     // in
                                     const distortion_model_t distortion_model,
                                     mrcal_problem_details_t problem_details,
                                     int Ncameras, int Nframes, int Npoints);

void mrcal_unpack_solver_state_vector( // out, in
                                       double* p, // unitless state on input,
                                                  // scaled, meaningful state on
                                                  // output

                                       // in
                                       const distortion_model_t distortion_model,
                                       mrcal_problem_details_t problem_details,
                                       int Ncameras, int Nframes, int Npoints);
