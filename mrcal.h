#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "mrcal-types.h"
#include "poseutils.h"
#include "triangulation.h"

////////////////////////////////////////////////////////////////////////////////
//////////////////// Lens models
////////////////////////////////////////////////////////////////////////////////

// Return an array of strings listing all the available lens models
//
// These are all "unconfigured" strings that use "..." placeholders for any
// configuration values. Each returned string is a \0-terminated const char*. The
// end of the list is signified by a NULL pointer
const char* const* mrcal_supported_lensmodel_names( void ); // NULL-terminated array of char* strings


// Return true if the given mrcal_lensmodel_type_t specifies a valid lens model
__attribute__((unused))
static bool mrcal_lensmodel_type_is_valid(mrcal_lensmodel_type_t t)
{
    return t >= 0;
}

// Evaluates to true if the given lens model is one of the supported OpenCV
// types
#define MRCAL_LENSMODEL_IS_OPENCV(d) (MRCAL_LENSMODEL_OPENCV4 <= (d) && (d) <= MRCAL_LENSMODEL_OPENCV12)


// Return a string describing a lens model.
//
// This function returns a static string. For models with no configuration, this
// is the FULL string for that model. For models with a configuration, the
// configuration values have "..." placeholders. These placeholders mean that
// the resulting strings do not define a lens model fully, and cannot be
// converted to a mrcal_lensmodel_t with mrcal_lensmodel_from_name()
//
// This is the inverse of mrcal_lensmodel_type_from_name()
const char* mrcal_lensmodel_name_unconfigured( const mrcal_lensmodel_t* lensmodel );


// Return a CONFIGURED string describing a lens model.
//
// This function generates a fully-configured string describing the given lens
// model. For models with no configuration, this is just the static string
// returned by mrcal_lensmodel_name_unconfigured(). For models that have a
// configuration, however, the configuration values are filled-in. The resulting
// string may be converted back into a mrcal_lensmodel_t by calling
// mrcal_lensmodel_from_name().
//
// This function writes the string into the given buffer "out". The size of the
// buffer is passed in the "size" argument. The meaning of "size" is as with
// snprintf(), which is used internally. Returns true on success
//
// This is the inverse of mrcal_lensmodel_from_name()
bool mrcal_lensmodel_name( char* out, int size,
                           const mrcal_lensmodel_t* lensmodel );


// Parse the lens model type from a lens model name string
//
// The configuration is ignored. Thus this function works even if the
// configuration is missing or unparseable. Unknown model names return
// MRCAL_LENSMODEL_INVALID
//
// This is the inverse of mrcal_lensmodel_name_unconfigured()
mrcal_lensmodel_type_t mrcal_lensmodel_type_from_name( const char* name );


// Parse the full configured lens model from a lens model name string
//
// The lens mode type AND the configuration are read into a mrcal_lensmodel_t
// structure, which this function returns. Strings with valid model names but
// missing or unparseable configuration return
//
//   {.type = MRCAL_LENSMODEL_INVALID_BADCONFIG}.
//
// Any other errors result in some other invalid lensmodel.type values, which
// can be checked with mrcal_lensmodel_type_is_valid(lensmodel->type)
//
// This is the inverse of mrcal_lensmodel_name()
bool mrcal_lensmodel_from_name( // output
                                mrcal_lensmodel_t* lensmodel,

                                // input
                                const char* name );

// Return a structure containing a model's metadata
//
// The available metadata is described in the definition of the
// MRCAL_LENSMODEL_META_LIST() macro
mrcal_lensmodel_metadata_t mrcal_lensmodel_metadata( const mrcal_lensmodel_t* lensmodel );


// Return the number of parameters required to specify a given lens model
//
// For models that have a configuration, the parameter count value generally
// depends on the configuration. For instance, splined models use the model
// parameters as the spline control points, so the spline density (specified in
// the configuration) directly affects how many parameters such a model requires
int mrcal_lensmodel_num_params( const mrcal_lensmodel_t* lensmodel );


// Return the locations of x and y spline knots

// Splined models are defined by the locations of their control points. These
// are arranged in a grid, the size and density of which is set by the model
// configuration. We fill-in the x knot locations into ux[] and the y locations
// into uy[]. ux[] and uy[] must be large-enough to hold configuration->Nx and
// configuration->Ny values respectively.
//
// This function applies to splined models only. Returns true on success
bool mrcal_knots_for_splined_models( double* ux, double* uy,
                                     const mrcal_lensmodel_t* lensmodel);



////////////////////////////////////////////////////////////////////////////////
//////////////////// Projections
////////////////////////////////////////////////////////////////////////////////

// Project the given camera-coordinate-system points
//
// Compute a "projection", a mapping of points defined in the camera coordinate
// system to their observed pixel coordinates. If requested, gradients are
// computed as well.
//
// We project N 3D points p to N 2D pixel coordinates q using the given
// lensmodel and intrinsics parameter values.
//
// if (dq_dp != NULL) we report the gradient dq/dp in a dense (N,2,3) array
// ((N,2) mrcal_point3_t objects).
//
// if (dq_dintrinsics != NULL) we report the gradient dq/dintrinsics in a dense
// (N,2,Nintrinsics) array. Note that splined models have very high Nintrinsics
// and very sparse gradients. THIS function reports the gradients densely,
// however, so it is inefficient for splined models.
//
// This function supports CAHVORE distortions only if we don't ask for any
// gradients
//
// Projecting out-of-bounds points (beyond the field of view) returns undefined
// values. Generally things remain continuous even as we move off the imager
// domain. Pinhole-like projections will work normally if projecting a point
// behind the camera. Splined projections clamp to the nearest spline segment:
// the projection will fly off to infinity quickly since we're extrapolating a
// polynomial, but the function will remain continuous.
bool mrcal_project( // out
                   mrcal_point2_t* q,
                   mrcal_point3_t* dq_dp,
                   double*         dq_dintrinsics,

                   // in
                   const mrcal_point3_t* p,
                   int N,
                   const mrcal_lensmodel_t* lensmodel,
                   // core, distortions concatenated
                   const double* intrinsics);


// Unproject the given pixel coordinates
//
// Compute an "unprojection", a mapping of pixel coordinates to the camera
// coordinate system.
//
// We unproject N 2D pixel coordinates q to N 3D direction vectors v using the
// given lensmodel and intrinsics parameter values. The returned vectors v are
// not normalized, and may have any length.

// This is the "reverse" direction, so an iterative nonlinear optimization is
// performed internally to compute this result. This is much slower than
// mrcal_project(). For OpenCV models specifically, OpenCV has
// cvUndistortPoints() (and cv2.undistortPoints()), but these are unreliable:
// https://github.com/opencv/opencv/issues/8811
//
// This function does NOT support CAHVORE
bool mrcal_unproject( // out
                     mrcal_point3_t* v,

                     // in
                     const mrcal_point2_t* q,
                     int N,
                     const mrcal_lensmodel_t* lensmodel,
                     // core, distortions concatenated
                     const double* intrinsics);


// Project the given camera-coordinate-system points using a pinhole
// model. See the docs for projection details:
// http://mrcal.secretsauce.net/lensmodels.html#lensmodel-pinhole
//
// This is a simplified special case of mrcal_project(). We project N
// camera-coordinate-system points p to N pixel coordinates q
//
// if (dq_dp != NULL) we report the gradient dq/dp in a dense (N,2,3) array
// ((N,2) mrcal_point3_t objects).
void mrcal_project_pinhole( // output
                            mrcal_point2_t* q,
                            mrcal_point3_t* dq_dp,

                             // input
                            const mrcal_point3_t* p,
                            int N,
                            const double* fxycxy);


// Unproject the given pixel coordinates using a pinhole model.
// See the docs for projection details:
// http://mrcal.secretsauce.net/lensmodels.html#lensmodel-pinhole
//
// This is a simplified special case of mrcal_unproject(). We unproject N 2D
// pixel coordinates q to N camera-coordinate-system vectors v. The returned
// vectors v are not normalized, and may have any length.
//
// if (dv_dq != NULL) we report the gradient dv/dq in a dense (N,3,2) array
// ((N,3) mrcal_point2_t objects).
void mrcal_unproject_pinhole( // output
                              mrcal_point3_t* v,
                              mrcal_point2_t* dv_dq,

                              // input
                              const mrcal_point2_t* q,
                              int N,
                              const double* fxycxy);

// Project the given camera-coordinate-system points using a stereographic
// model. See the docs for projection details:
// http://mrcal.secretsauce.net/lensmodels.html#lensmodel-stereographic
//
// This is a simplified special case of mrcal_project(). We project N
// camera-coordinate-system points p to N pixel coordinates q
//
// if (dq_dp != NULL) we report the gradient dq/dp in a dense (N,2,3) array
// ((N,2) mrcal_point3_t objects).
void mrcal_project_stereographic( // output
                                 mrcal_point2_t* q,
                                 mrcal_point3_t* dq_dp,

                                  // input
                                 const mrcal_point3_t* p,
                                 int N,
                                 const double* fxycxy);


// Unproject the given pixel coordinates using a stereographic model.
// See the docs for projection details:
// http://mrcal.secretsauce.net/lensmodels.html#lensmodel-stereographic
//
// This is a simplified special case of mrcal_unproject(). We unproject N 2D
// pixel coordinates q to N camera-coordinate-system vectors v. The returned
// vectors v are not normalized, and may have any length.
//
// if (dv_dq != NULL) we report the gradient dv/dq in a dense (N,3,2) array
// ((N,3) mrcal_point2_t objects).
void mrcal_unproject_stereographic( // output
                                   mrcal_point3_t* v,
                                   mrcal_point2_t* dv_dq,

                                   // input
                                   const mrcal_point2_t* q,
                                   int N,
                                   const double* fxycxy);


// Project the given camera-coordinate-system points using an equirectangular
// projection. See the docs for projection details:
// http://mrcal.secretsauce.net/lensmodels.html#lensmodel-lonlat
//
// This is a simplified special case of mrcal_project(). We project N
// camera-coordinate-system points p to N pixel coordinates q
//
// if (dq_dp != NULL) we report the gradient dq/dp in a dense (N,2,3) array
// ((N,2) mrcal_point3_t objects).
void mrcal_project_lonlat( // output
                           mrcal_point2_t* q,
                           mrcal_point3_t* dq_dv, // May be NULL. Each point
                                                  // gets a block of 2 mrcal_point3_t
                                                  // objects

                           // input
                           const mrcal_point3_t* v,
                           int N,
                           const double* fxycxy);

// Unproject the given pixel coordinates using an equirectangular projection.
// See the docs for projection details:
// http://mrcal.secretsauce.net/lensmodels.html#lensmodel-lonlat
//
// This is a simplified special case of mrcal_unproject(). We unproject N 2D
// pixel coordinates q to N camera-coordinate-system vectors v. The returned
// vectors v are normalized.
//
// if (dv_dq != NULL) we report the gradient dv/dq in a dense (N,3,2) array
// ((N,3) mrcal_point2_t objects).
void mrcal_unproject_lonlat( // output
                            mrcal_point3_t* v,
                            mrcal_point2_t* dv_dq, // May be NULL. Each point
                                                   // gets a block of 3 mrcal_point2_t
                                                   // objects

                            // input
                            const mrcal_point2_t* q,
                            int N,
                            const double* fxycxy);


// Project the given camera-coordinate-system points using a transverse
// equirectangular projection. See the docs for projection details:
// http://mrcal.secretsauce.net/lensmodels.html#lensmodel-latlon
//
// This is a simplified special case of mrcal_project(). We project N
// camera-coordinate-system points p to N pixel coordinates q
//
// if (dq_dp != NULL) we report the gradient dq/dp in a dense (N,2,3) array
// ((N,2) mrcal_point3_t objects).
void mrcal_project_latlon( // output
                           mrcal_point2_t* q,
                           mrcal_point3_t* dq_dv, // May be NULL. Each point
                                                  // gets a block of 2 mrcal_point3_t
                                                  // objects

                           // input
                           const mrcal_point3_t* v,
                           int N,
                           const double* fxycxy);

// Unproject the given pixel coordinates using a transverse equirectangular
// projection. See the docs for projection details:
// http://mrcal.secretsauce.net/lensmodels.html#lensmodel-latlon
//
// This is a simplified special case of mrcal_unproject(). We unproject N 2D
// pixel coordinates q to N camera-coordinate-system vectors v. The returned
// vectors v are normalized.
//
// if (dv_dq != NULL) we report the gradient dv/dq in a dense (N,3,2) array
// ((N,3) mrcal_point2_t objects).
void mrcal_unproject_latlon( // output
                            mrcal_point3_t* v,
                            mrcal_point2_t* dv_dq, // May be NULL. Each point
                                                   // gets a block of 3 mrcal_point2_t
                                                   // objects

                            // input
                            const mrcal_point2_t* q,
                            int N,
                            const double* fxycxy);



////////////////////////////////////////////////////////////////////////////////
//////////////////// Optimization
////////////////////////////////////////////////////////////////////////////////



// Return the number of parameters needed in optimizing the given lens model
//
// This is identical to mrcal_lensmodel_num_params(), but takes into account the
// problem selections. Any intrinsics parameters locked down in the
// mrcal_problem_selections_t do NOT count towards the optimization parameters
int mrcal_num_intrinsics_optimization_params( mrcal_problem_selections_t problem_selections,
                                              const mrcal_lensmodel_t* lensmodel );


// Scales a state vector to the packed, unitless form used by the optimizer
//
// In order to make the optimization well-behaved, we scale all the variables in
// the state and the gradients before passing them to the optimizer. The internal
// optimization library thus works only with unitless (or "packed") data.
//
// This function takes an (Nstate,) array of full-units values b[], and scales
// it to produce packed data. This function applies the scaling directly to the
// input array; the input is modified, and nothing is returned.
//
// This is the inverse of mrcal_unpack_solver_state_vector()
void mrcal_pack_solver_state_vector( // out, in
                                     double* b,

                                     // in
                                     int Ncameras_intrinsics, int Ncameras_extrinsics,
                                     int Nframes,
                                     int Npoints, int Npoints_fixed,
                                     mrcal_problem_selections_t problem_selections,
                                     const mrcal_lensmodel_t* lensmodel);


// Scales a state vector from the packed, unitless form used by the optimizer
//
// In order to make the optimization well-behaved, we scale all the variables in
// the state and the gradients before passing them to the optimizer. The internal
// optimization library thus works only with unitless (or "packed") data.
//
// This function takes an (Nstate,) array of unitless values b[], and scales it
// to produce full-units data. This function applies the scaling directly to the
// input array; the input is modified, and nothing is returned.
//
// This is the inverse of mrcal_pack_solver_state_vector()
void mrcal_unpack_solver_state_vector( // out, in
                                       double* b, // unitless state on input,
                                                  // scaled, meaningful state on
                                                  // output

                                       // in
                                       int Ncameras_intrinsics, int Ncameras_extrinsics,
                                       int Nframes,
                                       int Npoints, int Npoints_fixed,
                                       mrcal_problem_selections_t problem_selections,
                                       const mrcal_lensmodel_t* lensmodel);


// Reports the icam_extrinsics corresponding to a given icam_intrinsics.
//
// If we're solving a vanilla calibration problem (stationary cameras observing
// a moving calibration object), each camera has a unique intrinsics index and a
// unique extrinsics index. This function reports the latter, given the former.
//
// On success, the result is written to *icam_extrinsics, and we return true. If
// the given camera is at the reference coordinate system, it has no extrinsics,
// and we report -1.
//
// If we have moving cameras (NOT a vanilla calibration problem), there isn't a
// single icam_extrinsics for a given icam_intrinsics, and we report an error by
// returning false
bool mrcal_corresponding_icam_extrinsics(// out
                                         int* icam_extrinsics,

                                         // in
                                         int icam_intrinsics,
                                         int Ncameras_intrinsics,
                                         int Ncameras_extrinsics,
                                         int Nobservations_board,
                                         const mrcal_observation_board_t* observations_board,
                                         int Nobservations_point,
                                         const mrcal_observation_point_t* observations_point);

// Solve the given optimization problem
//
// This is the entry point to the mrcal optimization routine. The argument list
// is commented.
mrcal_stats_t
mrcal_optimize( // out
                // Each one of these output pointers may be NULL
                // Shape (Nstate,)
                double* b_packed,
                // used only to confirm that the user passed-in the buffer they
                // should have passed-in. The size must match exactly
                int buffer_size_b_packed,

                // Shape (Nmeasurements,)
                double* x,
                // used only to confirm that the user passed-in the buffer they
                // should have passed-in. The size must match exactly
                int buffer_size_x,

                // out, in

                // These are a seed on input, solution on output

                // intrinsics is a concatenation of the intrinsics core and the
                // distortion params. The specific distortion parameters may
                // vary, depending on lensmodel, so this is a variable-length
                // structure
                double*             intrinsics,         // Ncameras_intrinsics * NlensParams
                mrcal_pose_t*       extrinsics_fromref, // Ncameras_extrinsics of these. Transform FROM the reference frame
                mrcal_pose_t*       frames_toref,       // Nframes of these.    Transform TO the reference frame
                mrcal_point3_t*     points,             // Npoints of these.    In the reference frame
                mrcal_calobject_warp_t* calobject_warp,     // 1 of these. May be NULL if !problem_selections.do_optimize_calobject_warp

                // in
                int Ncameras_intrinsics, int Ncameras_extrinsics, int Nframes,
                int Npoints, int Npoints_fixed, // at the end of points[]

                const mrcal_observation_board_t* observations_board,
                const mrcal_observation_point_t* observations_point,
                int Nobservations_board,
                int Nobservations_point,

                // All the board pixel observations, in an array of shape
                //
                // ( Nobservations_board,
                //   calibration_object_height_n,
                //   calibration_object_width_n )
                //
                // .x, .y are the
                // pixel observations .z is the weight of the observation. Most
                // of the weights are expected to be 1.0. Less precise
                // observations have lower weights.
                //
                // .z<0 indicates that this is an outlier. This is respected on
                // input (even if !do_apply_outlier_rejection). New outliers are
                // marked with .z<0 on output, so this isn't const
                mrcal_point3_t* observations_board_pool,

                const mrcal_lensmodel_t* lensmodel,
                const int* imagersizes, // Ncameras_intrinsics*2 of these
                mrcal_problem_selections_t       problem_selections,
                const mrcal_problem_constants_t* problem_constants,
                double calibration_object_spacing,
                int calibration_object_width_n,
                int calibration_object_height_n,
                bool verbose,

                bool check_gradient);


// This is cholmod_sparse. I don't want to include the full header that defines
// it in mrcal.h, and I don't need to: mrcal.h just needs to know that it's a
// structure
struct cholmod_sparse_struct;

// Evaluate the value of the callback function at the given operating point
//
// The main optimization routine in mrcal_optimize() searches for optimal
// parameters by repeatedly calling a function to evaluate each hypothethical
// parameter set. This evaluation function is available by itself here,
// separated from the optimization loop. The arguments are largely the same as
// those to mrcal_optimize(), but the inputs are all read-only It is expected
// that this will be called from Python only.
bool mrcal_optimizer_callback(// out

                             // These output pointers may NOT be NULL, unlike
                             // their analogues in mrcal_optimize()

                             // Shape (Nstate,)
                             double* b_packed,
                             // used only to confirm that the user passed-in the buffer they
                             // should have passed-in. The size must match exactly
                             int buffer_size_b_packed,

                             // Shape (Nmeasurements,)
                             double* x,
                             // used only to confirm that the user passed-in the buffer they
                             // should have passed-in. The size must match exactly
                             int buffer_size_x,

                             // output Jacobian. May be NULL if we don't need
                             // it. This is the unitless Jacobian, used by the
                             // internal optimization routines
                             struct cholmod_sparse_struct* Jt,


                             // in

                             // intrinsics is a concatenation of the intrinsics core
                             // and the distortion params. The specific distortion
                             // parameters may vary, depending on lensmodel, so
                             // this is a variable-length structure
                             const double*             intrinsics,         // Ncameras_intrinsics * NlensParams
                             const mrcal_pose_t*       extrinsics_fromref, // Ncameras_extrinsics of these. Transform FROM the reference frame
                             const mrcal_pose_t*       frames_toref,       // Nframes of these.    Transform TO the reference frame
                             const mrcal_point3_t*     points,             // Npoints of these.    In the reference frame
                             const mrcal_calobject_warp_t* calobject_warp, // 1 of these. May be NULL if !problem_selections.do_optimize_calobject_warp

                             int Ncameras_intrinsics, int Ncameras_extrinsics, int Nframes,
                             int Npoints, int Npoints_fixed, // at the end of points[]

                             const mrcal_observation_board_t* observations_board,
                             const mrcal_observation_point_t* observations_point,
                             int Nobservations_board,
                             int Nobservations_point,

                             // All the board pixel observations, in an array of shape
                             //
                             // ( Nobservations_board,
                             //   calibration_object_height_n,
                             //   calibration_object_width_n )
                             //
                             // .x, .y are the pixel observations .z is the
                             // weight of the observation. Most of the weights
                             // are expected to be 1.0. Less precise
                             // observations have lower weights.
                             //
                             // .z<0 indicates that this is an outlier
                             const mrcal_point3_t* observations_board_pool,

                             const mrcal_lensmodel_t* lensmodel,
                             const int* imagersizes, // Ncameras_intrinsics*2 of these
                             mrcal_problem_selections_t       problem_selections,
                             const mrcal_problem_constants_t* problem_constants,
                             double calibration_object_spacing,
                             int calibration_object_width_n,
                             int calibration_object_height_n,
                             bool verbose);


////////////////////////////////////////////////////////////////////////////////
//////////////////// Layout of the measurement and state vectors
////////////////////////////////////////////////////////////////////////////////

// The optimization routine tries to minimize the length of the measurement
// vector x by moving around the state vector b.
//
// Depending on the specific optimization problem being solved and the
// mrcal_problem_selections_t, the state vector may contain any of
// - The lens parameters
// - The geometry of the cameras
// - The geometry of the observed chessboards and discrete points
// - The chessboard shape
//
// The measurement vector may contain
// - The errors in observations of the chessboards
// - The errors in observations of discrete points
// - The penalties in the solved point positions
// - The regularization terms
//
// Given the problem selections and a vector b or x it is often useful to know
// where specific quantities lie in those vectors. We have 4 sets of functions
// to answer such questions:
//
// int mrcal_measurement_index_THING()
//   Returns the index in the measurement vector x where the contiguous block of
//   values describing the THING begins. THING is any of
//   - boards
//   - points
//   - regularization
//
// int mrcal_num_measurements_THING()
//   Returns the number of values in the contiguous block in the measurement
//   vector x that describe the given THING. THING is any of
//   - boards
//   - points
//   - regularization
//
// int mrcal_state_index_THING()
//   Returns the index in the state vector b where the contiguous block of
//   values describing the THING begins. THING is any of
//   - intrinsics
//   - extrinsics
//   - frames
//   - points
//   - calobject_warp
//   If we're not optimizing the THING, return <0
//
// int mrcal_num_states_THING()
//   Returns the number of values in the contiguous block in the state
//   vector b that describe the given THING. THING is any of
//   - intrinsics
//   - extrinsics
//   - frames
//   - points
//   - calobject_warp
//   If we're not optimizing the THING, return 0
int mrcal_measurement_index_boards(int i_observation_board,
                                   int Nobservations_board,
                                   int Nobservations_point,
                                   int calibration_object_width_n,
                                   int calibration_object_height_n);
int mrcal_num_measurements_boards(int Nobservations_board,
                                  int calibration_object_width_n,
                                  int calibration_object_height_n);
int mrcal_measurement_index_points(int i_observation_point,
                                   int Nobservations_board,
                                   int Nobservations_point,
                                   int calibration_object_width_n,
                                   int calibration_object_height_n);
int mrcal_num_measurements_points(int Nobservations_point);
int mrcal_measurement_index_regularization(int calibration_object_width_n,
                                           int calibration_object_height_n,
                                           int Ncameras_intrinsics, int Ncameras_extrinsics,
                                           int Nframes,
                                           int Npoints, int Npoints_fixed, int Nobservations_board, int Nobservations_point,
                                           mrcal_problem_selections_t problem_selections,
                                           const mrcal_lensmodel_t* lensmodel);
int mrcal_num_measurements_regularization(int Ncameras_intrinsics, int Ncameras_extrinsics,
                                          int Nframes,
                                          int Npoints, int Npoints_fixed, int Nobservations_board,
                                          mrcal_problem_selections_t problem_selections,
                                          const mrcal_lensmodel_t* lensmodel);

int mrcal_num_measurements(int Nobservations_board,
                           int Nobservations_point,
                           int calibration_object_width_n,
                           int calibration_object_height_n,
                           int Ncameras_intrinsics, int Ncameras_extrinsics,
                           int Nframes,
                           int Npoints, int Npoints_fixed,
                           mrcal_problem_selections_t problem_selections,
                           const mrcal_lensmodel_t* lensmodel);

int mrcal_num_states(int Ncameras_intrinsics, int Ncameras_extrinsics,
                     int Nframes,
                     int Npoints, int Npoints_fixed, int Nobservations_board,
                     mrcal_problem_selections_t problem_selections,
                     const mrcal_lensmodel_t* lensmodel);
int mrcal_state_index_intrinsics(int icam_intrinsics,
                                 int Ncameras_intrinsics, int Ncameras_extrinsics,
                                 int Nframes,
                                 int Npoints, int Npoints_fixed, int Nobservations_board,
                                 mrcal_problem_selections_t problem_selections,
                                 const mrcal_lensmodel_t* lensmodel);
int mrcal_num_states_intrinsics(int Ncameras_intrinsics,
                                mrcal_problem_selections_t problem_selections,
                                const mrcal_lensmodel_t* lensmodel);
int mrcal_state_index_extrinsics(int icam_extrinsics,
                                 int Ncameras_intrinsics, int Ncameras_extrinsics,
                                 int Nframes,
                                 int Npoints, int Npoints_fixed, int Nobservations_board,
                                 mrcal_problem_selections_t problem_selections,
                                 const mrcal_lensmodel_t* lensmodel);
int mrcal_num_states_extrinsics(int Ncameras_extrinsics,
                                mrcal_problem_selections_t problem_selections);
int mrcal_state_index_frames(int iframe,
                             int Ncameras_intrinsics, int Ncameras_extrinsics,
                             int Nframes,
                             int Npoints, int Npoints_fixed, int Nobservations_board,
                             mrcal_problem_selections_t problem_selections,
                             const mrcal_lensmodel_t* lensmodel);
int mrcal_num_states_frames(int Nframes,
                            mrcal_problem_selections_t problem_selections);
int mrcal_state_index_points(int i_point,
                             int Ncameras_intrinsics, int Ncameras_extrinsics,
                             int Nframes,
                             int Npoints, int Npoints_fixed, int Nobservations_board,
                             mrcal_problem_selections_t problem_selections,
                             const mrcal_lensmodel_t* lensmodel);
int mrcal_num_states_points(int Npoints, int Npoints_fixed,
                            mrcal_problem_selections_t problem_selections);
int mrcal_state_index_calobject_warp(int Ncameras_intrinsics, int Ncameras_extrinsics,
                                     int Nframes,
                                     int Npoints, int Npoints_fixed, int Nobservations_board,
                                     mrcal_problem_selections_t problem_selections,
                                     const mrcal_lensmodel_t* lensmodel);
int mrcal_num_states_calobject_warp(mrcal_problem_selections_t problem_selections,
                                    int Nobservations_board);


// if len>0, the string doesn't need to be 0-terminated. If len<=0, the end of
// the buffer IS indicated by a '\0' byte
mrcal_cameramodel_t* mrcal_read_cameramodel_string(const char* string, int len);
mrcal_cameramodel_t* mrcal_read_cameramodel_file  (const char* filename);
void                 mrcal_free_cameramodel(mrcal_cameramodel_t** cameramodel);

bool mrcal_write_cameramodel_file(const char* filename,
                                  const mrcal_cameramodel_t* cameramodel);

// Public ABI stuff, that's not for end-user consumption
#include "mrcal-internal.h"

