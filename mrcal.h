#pragma once

#include <stdbool.h>
#include <stdint.h>

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

    // NUM_POINTS_IN_CALOBJECT of these.
    // .x, .y are the pixel observations
    // .z is the weight of the observation. Most of the weights are expected to
    // be 1.0, which implies that the noise on the observation is gaussian,
    // independent on x,y, and has standard deviation of
    // observed_pixel_uncertainty. observed_pixel_uncertainty scales inversely
    // with the weight.
    point3_t* px;
} observation_board_t;

typedef struct
{
    int  i_camera         : 31;
    bool skip_point       : 1;
    int  i_point          : 31;
    bool skip_observation : 1;

    // Observed pixel coordinates
    // .x, .y are the pixel observations
    // .z is the weight of the observation. Most of the weights are expected to
    // be 1.0, which implies that the noise on the observation is gaussian,
    // independent on x,y, and has standard deviation of
    // observed_pixel_uncertainty. observed_pixel_uncertainty scales inversely
    // with the weight.
    point3_t px;

    // Reference distance. This is optional; skipped if <= 0
    double dist;
} observation_point_t;



typedef struct
{
    double focal_xy [2];
    double center_xy[2];
} intrinsics_core_t;

// names of the lens models, intrinsic parameter counts
#define LENSMODEL_LIST(_)                                             \
    _(LENSMODEL_PINHOLE, 4)                                           \
    _(LENSMODEL_OPENCV4, 8)                                           \
    _(LENSMODEL_OPENCV5, 9)                                           \
    _(LENSMODEL_OPENCV8, 12)                                          \
    _(LENSMODEL_OPENCV12,16)   /* available in OpenCV >= 3.0.0) */    \
    _(LENSMODEL_OPENCV14,18)   /* available in OpenCV >= 3.1.0) */    \
    _(LENSMODEL_CAHVOR,  9)                                           \
    _(LENSMODEL_CAHVORE, 13)   /* CAHVORE is CAHVOR + E + linearity */\
    _(LENSMODEL_UV,      0)


// parametric models have no extra configuration
typedef struct {} LENSMODEL_PINHOLE__config_t;
typedef struct {} LENSMODEL_OPENCV4__config_t;
typedef struct {} LENSMODEL_OPENCV5__config_t;
typedef struct {} LENSMODEL_OPENCV8__config_t;
typedef struct {} LENSMODEL_OPENCV12__config_t;
typedef struct {} LENSMODEL_OPENCV14__config_t;
typedef struct {} LENSMODEL_CAHVOR__config_t;
typedef struct {} LENSMODEL_CAHVORE__config_t;

typedef struct
{
    uint16_t a,b;
} LENSMODEL_UV__config_t;

#define LENSMODEL_OPENCV_FIRST LENSMODEL_OPENCV4
#define LENSMODEL_OPENCV_LAST  LENSMODEL_OPENCV14
#define LENSMODEL_CAHVOR_FIRST LENSMODEL_CAHVOR
#define LENSMODEL_CAHVOR_LAST  LENSMODEL_CAHVORE
#define LENSMODEL_IS_OPENCV(d) (LENSMODEL_OPENCV_FIRST <= (d) && (d) <= LENSMODEL_OPENCV_LAST)
#define LENSMODEL_IS_CAHVOR(d) (LENSMODEL_CAHVOR_FIRST <= (d) && (d) <= LENSMODEL_CAHVOR_LAST)

#define LIST_WITH_COMMA(s,n) ,s
typedef enum
    { LENSMODEL_INVALID LENSMODEL_LIST( LIST_WITH_COMMA ) } lensmodel_type_t;
typedef struct
{
    lensmodel_type_t type;
    union
    {
#define CONFIG_STRUCT(s,n) s##__config_t s##__config;
        LENSMODEL_LIST(CONFIG_STRUCT);
#undef CONFIG_STRUCT
    };
} lensmodel_t;


typedef struct
{
    // Applies only to those models that HAVE a core (fx,fy,cx,cy). For those
    // mrcal_modelHasCore_fxfycxcy(model) returns true
    bool do_optimize_intrinsic_core        : 1;

    // For models that have a core, these are all the non-core parameters. For
    // models that do not, these are ALL the parameters
    bool do_optimize_intrinsic_distortions : 1;
    bool do_optimize_extrinsics            : 1;
    bool do_optimize_frames                : 1;
    bool do_skip_regularization            : 1;
    bool do_optimize_cahvor_optical_axis   : 1;
    bool do_optimize_calobject_warp        : 1;
} mrcal_problem_details_t;
#define DO_OPTIMIZE_ALL ((mrcal_problem_details_t) { .do_optimize_intrinsic_core        = true, \
                                                     .do_optimize_intrinsic_distortions = true, \
                                                     .do_optimize_extrinsics            = true, \
                                                     .do_optimize_frames                = true, \
                                                     .do_optimize_cahvor_optical_axis   = true, \
                                                     .do_optimize_calobject_warp        = true, \
                                                     .do_skip_regularization            = false})

const char*        mrcal_lensmodel_name                 ( lensmodel_t model );
lensmodel_t       mrcal_lensmodel_from_name            ( const char* name );
bool               mrcal_modelHasCore_fxfycxcy           ( const lensmodel_t m );
int                mrcal_getNlensParams                  ( const lensmodel_t m );
int                mrcal_getNintrinsicOptimizationParams ( mrcal_problem_details_t problem_details,
                                                           lensmodel_t m );
const char* const* mrcal_getSupportedLensModels          ( void ); // NULL-terminated array of char* strings

// Returns the 'next' lens model in a family
//
// In a family of lens models we have a sequence of models with increasing
// complexity. Subsequent models add distortion parameters to the end of the
// vector. Ealier models are identical, but with the extra paramaters set to 0.
// This function returns the next model in a sequence.
//
// If this is the last model in the sequence, returns the current model. This
// function takes in both the current model, and the last model we're aiming
// for. The second part is required because all familie begin at
// LENSMODEL_PINHOLE, so the next model from LENSMODEL_PINHOLE is not well-defined
// without more information
lensmodel_t mrcal_getNextLensModel( lensmodel_t lensmodel_now,
                                     lensmodel_t lensmodel_final);

// Wrapper around the internal project() function: the function used in the
// inner optimization loop. These map world points to their observed pixel
// coordinates, and to optionally provide gradients. dxy_dintrinsics and/or
// dxy_dp are allowed to be NULL if we're not interested those gradients.
//
// This function supports CAHVORE distortions if we don't ask for gradients
bool mrcal_project( // out
                   point2_t* q,

                   // core, distortions concatenated. Stored as a row-first
                   // array of shape (N,2,Nintrinsics)
                   double*   dq_dintrinsics,
                   // Stored as a row-first array of shape (N,2,3). Each
                   // trailing ,3 dimension element is a point3_t
                   point3_t* dq_dp,

                   // in
                   const point3_t* p,
                   int N,
                   lensmodel_t lensmodel,
                   // core, distortions concatenated
                   const double* intrinsics);

// Similar to mrcal_project(), but instead of projecting 3D points world
// coordinates, takes in 2D vectors Vxy, which represent a 3D point (Vx,Vy,1).
// For each point this produces a 2x2 gradient dxy_dVxy instead of a 2x3
// gradient dxy_dp
bool mrcal_project_z1( // out
                       point2_t* q,

                       // core, distortions concatenated. Stored as a row-first
                       // array of shape (N,2,Nintrinsics)
                       double*   dq_dintrinsics,
                       // Stored as a row-first array of shape (N,2,2). Each
                       // trailing ,2 dimension element is a point2_t
                       point2_t* dq_dVxy,

                       // in
                       const point2_t* Vxy,
                       int N,
                       lensmodel_t lensmodel,
                       // core, distortions concatenated
                       const double* intrinsics);

// Maps a set of distorted 2D imager points q to a 3d vector in camera
// coordinates that produced these pixel observations. The 3d vector is defined
// up-to-length, so the vectors reported here will all have z = 1.
//
// This is the "reverse" direction, so an iterative nonlinear optimization is
// performed internally to compute this result. This is much slower than
// mrcal_project. For OpenCV models specifically, OpenCV has cvUndistortPoints()
// (and cv2.undistortPoints()), but these are inaccurate:
// https://github.com/opencv/opencv/issues/8811
//
// This function does NOT support CAHVORE
bool mrcal_unproject( // out
                     point3_t* out,

                     // in
                     const point2_t* q,
                     int N,
                     lensmodel_t lensmodel,
                     // core, distortions concatenated
                     const double* intrinsics);
// Exactly the same as mrcal_unproject(), but reports 2d points, omitting the
// redundant z=1
bool mrcal_unproject_z1( // out
                        point2_t* out,

                        // in
                        const point2_t* q,
                        int N,
                        lensmodel_t lensmodel,
                        // core, distortions concatenated
                        const double* intrinsics);



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
                double* covariance_intrinsics_full,
                double* covariance_intrinsics,
                double* covariance_extrinsics,

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
                // parameters may vary, depending on lensmodel, so
                // this is a variable-length structure
                double*       camera_intrinsics,  // Ncameras * NlensParams
                pose_t*       camera_extrinsics,  // Ncameras-1 of these. Transform FROM camera0 frame
                pose_t*       frames,             // Nframes of these.    Transform TO   camera0 frame
                point3_t*     points,             // Npoints of these.    In the camera0 frame
                point2_t*     calobject_warp,     // 1 of these. May be NULL if !problem_details.do_optimize_calobject_warp

                // in
                int Ncameras, int Nframes, int Npoints,

                const observation_board_t* observations_board,
                int NobservationsBoard,

                const observation_point_t* observations_point,
                int NobservationsPoint,

                bool check_gradient,
                // input outliers. These are respected regardless of
                // skip_outlier_rejection.
                int Noutlier_indices_input,
                int* outlier_indices_input,

                // region-of-interest. If not NULL, errors for observations
                // outside this region are strongly attenuated. The region is
                // specified separately for each camera. Each region is an
                // ellipse, represented as a 4-double slice with values
                // (x_center, y_center, x_width, y_width)
                const double* roi,

                bool verbose,
                // Whether to try to find NEW outliers. These would be added to
                // the outlier_indices_input, which are respected regardless
                const bool skip_outlier_rejection,

                lensmodel_t lensmodel,
                double observed_pixel_uncertainty,
                const int* imagersizes, // Ncameras*2 of these
                mrcal_problem_details_t problem_details,

                double calibration_object_spacing,
                int calibration_object_width_n);

struct cholmod_sparse_struct;
// callback function. This is primarily for debugging
void mrcal_optimizerCallback(// output measurements
                             double*         x,

                             // output Jacobian. May be NULL if we don't need it.
                             struct cholmod_sparse_struct* Jt,

                             // in
                             // intrinsics is a concatenation of the intrinsics core
                             // and the distortion params. The specific distortion
                             // parameters may vary, depending on lensmodel, so
                             // this is a variable-length structure
                             const double*       intrinsics, // Ncameras * NlensParams
                             const pose_t*       extrinsics, // Ncameras-1 of these. Transform FROM camera0 frame
                             const pose_t*       frames,     // Nframes of these.    Transform TO   camera0 frame
                             const point3_t*     points,     // Npoints of these.    In the camera0 frame
                             const point2_t*     calobject_warp, // 1 of these. May be NULL if !problem_details.do_optimize_calobject_warp

                             int Ncameras, int Nframes, int Npoints,

                             const observation_board_t* observations_board,
                             int NobservationsBoard,

                             const observation_point_t* observations_point,
                             int NobservationsPoint,

                             int Noutlier_indices_input,
                             const int* outlier_indices_input,
                             const double* roi,
                             bool verbose,

                             lensmodel_t lensmodel,
                             const int* imagersizes, // Ncameras*2 of these

                             mrcal_problem_details_t problem_details,

                             double calibration_object_spacing,
                             int calibration_object_width_n,
                             int Nintrinsics, int Nmeasurements, int N_j_nonzero);


int mrcal_getNmeasurements_all(int Ncameras, int NobservationsBoard,
                               const observation_point_t* observations_point,
                               int NobservationsPoint,
                               int calibration_object_width_n,
                               mrcal_problem_details_t problem_details,
                               lensmodel_t lensmodel);
int mrcal_getNmeasurements_boards(int NobservationsBoard,
                                  int calibration_object_width_n);
int mrcal_getNmeasurements_points(const observation_point_t* observations_point,
                                  int NobservationsPoint);
int mrcal_getNmeasurements_regularization(int Ncameras,
                                          mrcal_problem_details_t problem_details,
                                          lensmodel_t lensmodel);
int mrcal_getNstate(int Ncameras, int Nframes, int Npoints,
                    mrcal_problem_details_t problem_details,
                    lensmodel_t lensmodel);
int mrcal_getN_j_nonzero( int Ncameras,
                          const observation_board_t* observations_board,
                          int NobservationsBoard,
                          const observation_point_t* observations_point,
                          int NobservationsPoint,
                          mrcal_problem_details_t problem_details,
                          lensmodel_t lensmodel,
                          int calibration_object_width_n);

// frees a dogleg_solverContext_t. I don't want to #include <dogleg.h> here, so
// this is void
void mrcal_free_context(void** ctx);


int mrcal_state_index_intrinsic_core(int i_camera,
                                     mrcal_problem_details_t problem_details,
                                     lensmodel_t lensmodel);
int mrcal_state_index_intrinsic_distortions(int i_camera,
                                            mrcal_problem_details_t problem_details,
                                            lensmodel_t lensmodel);
int mrcal_state_index_camera_rt(int i_camera, int Ncameras,
                                mrcal_problem_details_t problem_details,
                                lensmodel_t lensmodel);
int mrcal_state_index_frame_rt(int i_frame, int Ncameras,
                               mrcal_problem_details_t problem_details,
                               lensmodel_t lensmodel);
int mrcal_state_index_point(int i_point, int Nframes, int Ncameras,
                            mrcal_problem_details_t problem_details,
                            lensmodel_t lensmodel);
int mrcal_state_index_calobject_warp(int Npoints,
                                     int Nframes, int Ncameras,
                                     mrcal_problem_details_t problem_details,
                                     lensmodel_t lensmodel);

// packs/unpacks a vector
void mrcal_pack_solver_state_vector( // out, in
                                     double* p, // unitless, FULL state on
                                                // input, scaled, decimated
                                                // (subject to problem_details),
                                                // meaningful state on output

                                     // in
                                     const lensmodel_t lensmodel,
                                     mrcal_problem_details_t problem_details,
                                     int Ncameras, int Nframes, int Npoints);

void mrcal_unpack_solver_state_vector( // out, in
                                       double* p, // unitless state on input,
                                                  // scaled, meaningful state on
                                                  // output

                                       // in
                                       const lensmodel_t lensmodel,
                                       mrcal_problem_details_t problem_details,
                                       int Ncameras, int Nframes, int Npoints);
