#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "basic_geometry.h"
#include "poseutils.h"
#include "triangulation.h"


////////////////////////////////////////////////////////////////////////////////
//////////////////// Lens models
////////////////////////////////////////////////////////////////////////////////

// These are an "X macro": https://en.wikipedia.org/wiki/X_Macro
//
// The supported lens models and their parameter counts. Models with a
// configuration report their parameter counts in the
// LENSMODEL_XXX__lensmodel_num_params() function, called by
// mrcal_lensmodel_num_params(). So their parameter counts here are ignored.
#define MRCAL_LENSMODEL_NOCONFIG_LIST(_)                                         \
    _(LENSMODEL_PINHOLE,               4)                                        \
    _(LENSMODEL_STEREOGRAPHIC,         4)  /* Simple stereographic-only model */ \
    _(LENSMODEL_LONLAT,                4)                                        \
    _(LENSMODEL_LATLON,                4)                                        \
    _(LENSMODEL_OPENCV4,               8)                                        \
    _(LENSMODEL_OPENCV5,               9)                                        \
    _(LENSMODEL_OPENCV8,               12)                                       \
    _(LENSMODEL_OPENCV12,              16) /* available in OpenCV >= 3.0.0) */   \
    _(LENSMODEL_CAHVOR,                9)
#define MRCAL_LENSMODEL_WITHCONFIG_LIST(_)                                       \
    _(LENSMODEL_CAHVORE,               0)                                        \
    _(LENSMODEL_SPLINED_STEREOGRAPHIC, 0)
#define MRCAL_LENSMODEL_LIST(_)                                                  \
    MRCAL_LENSMODEL_NOCONFIG_LIST(_)                                             \
    MRCAL_LENSMODEL_WITHCONFIG_LIST(_)


// parametric models have no extra configuration
typedef struct {} mrcal_LENSMODEL_PINHOLE__config_t;
typedef struct {} mrcal_LENSMODEL_STEREOGRAPHIC__config_t;
typedef struct {} mrcal_LENSMODEL_LONLAT__config_t;
typedef struct {} mrcal_LENSMODEL_LATLON__config_t;
typedef struct {} mrcal_LENSMODEL_OPENCV4__config_t;
typedef struct {} mrcal_LENSMODEL_OPENCV5__config_t;
typedef struct {} mrcal_LENSMODEL_OPENCV8__config_t;
typedef struct {} mrcal_LENSMODEL_OPENCV12__config_t;
typedef struct {} mrcal_LENSMODEL_CAHVOR__config_t;

#define _MRCAL_ITEM_DEFINE_ELEMENT(name, type, pybuildvaluecode, PRIcode,SCNcode, bitfield, cookie) type name bitfield;

#ifndef __cplusplus
// This barfs with g++ 4.8, so I disable it
_Static_assert(sizeof(uint16_t) == sizeof(unsigned short int), "I need a short to be 16-bit. Py_BuildValue doesn't let me just specify that. H means 'unsigned short'");
#endif

// Configuration for CAHVORE. These are given as an an
// "X macro": https://en.wikipedia.org/wiki/X_Macro
#define MRCAL_LENSMODEL_CAHVORE_CONFIG_LIST(_, cookie) \
    _(linearity,    double, "d", ".2f", "lf", , cookie)
typedef struct
{
    MRCAL_LENSMODEL_CAHVORE_CONFIG_LIST(_MRCAL_ITEM_DEFINE_ELEMENT, )
} mrcal_LENSMODEL_CAHVORE__config_t;

// Configuration for the splined stereographic models. These are given as an an
// "X macro": https://en.wikipedia.org/wiki/X_Macro
#define MRCAL_LENSMODEL_SPLINED_STEREOGRAPHIC_CONFIG_LIST(_, cookie)                \
    /* Maximum degree of each 1D polynomial. This is almost certainly 2 */          \
    /* (quadratic splines, C1 continuous) or 3 (cubic splines, C2 continuous) */    \
    _(order,        uint16_t, "H", PRIu16,SCNu16, , cookie)                         \
    /* We have a Nx by Ny grid of control points */                                 \
    _(Nx,           uint16_t, "H", PRIu16,SCNu16, , cookie)                         \
    _(Ny,           uint16_t, "H", PRIu16,SCNu16, , cookie)                         \
    /* The horizontal field of view. Not including fov_y. It's proportional with */ \
    /* Ny and Nx */                                                                 \
    _(fov_x_deg,    uint16_t, "H", PRIu16,SCNu16, , cookie)
typedef struct
{
    MRCAL_LENSMODEL_SPLINED_STEREOGRAPHIC_CONFIG_LIST(_MRCAL_ITEM_DEFINE_ELEMENT, )
} mrcal_LENSMODEL_SPLINED_STEREOGRAPHIC__config_t;


// An X-macro-generated enum mrcal_lensmodel_type_t. This has an element for
// each entry in MRCAL_LENSMODEL_LIST (with "MRCAL_" prepended). This lensmodel
// type selects the lens model, but does NOT provide the configuration.
// mrcal_lensmodel_t does that.
#define _LIST_WITH_COMMA(s,n) ,MRCAL_ ## s
typedef enum
    { MRCAL_LENSMODEL_INVALID           = -2,
      MRCAL_LENSMODEL_INVALID_BADCONFIG = -1
      // The rest, starting with 0
      MRCAL_LENSMODEL_LIST( _LIST_WITH_COMMA ) } mrcal_lensmodel_type_t;
#undef _LIST_WITH_COMMA


// Defines a lens model: the type AND the configuration values
typedef struct
{
    // The type of lensmodel. This is an enum, selecting elements of
    // MRCAL_LENSMODEL_LIST (with "MRCAL_" prepended)
    mrcal_lensmodel_type_t type;

    // A union of all the possible configuration structures. We pick the
    // structure type based on the value of "type
    union
    {
#define CONFIG_STRUCT(s,n) mrcal_ ##s##__config_t s##__config;
        MRCAL_LENSMODEL_LIST(CONFIG_STRUCT);
#undef CONFIG_STRUCT
    };
} mrcal_lensmodel_t;


typedef union
{
    struct
    {
        double x2, y2;
    };
    double values[2];
} mrcal_calobject_warp_t;

#define MRCAL_NSTATE_CALOBJECT_WARP ((int)((sizeof(mrcal_calobject_warp_t)/sizeof(double))))


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


// An X-macro-generated mrcal_lensmodel_metadata_t. Each lens model type has
// some metadata that describes its inherent properties. These properties can be
// queried by calling mrcal_lensmodel_metadata() in C and
// mrcal.lensmodel_metadata_and_config() in Python. The available properties and
// their meaning are listed in MRCAL_LENSMODEL_META_LIST()
#define MRCAL_LENSMODEL_META_LIST(_, cookie)                            \
    /* If true, this model contains an "intrinsics core". This is described */ \
    /* in mrcal_intrinsics_core_t. If present, the 4 core parameters ALWAYS */ \
    /* appear at the start of a model's parameter vector                    */ \
    _(has_core,                  bool, "i",,, :1, cookie)               \
                                                                        \
    /* Whether a model is able to project points behind the camera          */ \
    /* (z<0 in the camera coordinate system). Models based on a pinhole     */ \
    /* projection (pinhole, OpenCV, CAHVOR(E)) cannot do this. models based */ \
    /* on a stereographic projection (stereographic, splined stereographic) */ \
    /* can                                                                  */ \
    _(can_project_behind_camera, bool, "i",,, :1, cookie)               \
                                                                        \
    /* Whether gradients are available for this model. Currently only */ \
    /* CAHVORE does not have gradients                                */ \
    _(has_gradients,             bool, "i",,, :1, cookie)

typedef struct
{
    MRCAL_LENSMODEL_META_LIST(_MRCAL_ITEM_DEFINE_ELEMENT, )
} mrcal_lensmodel_metadata_t;


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

// Used to specify which camera is making an observation. The "intrinsics" index
// is used to identify a specific camera, while the "extrinsics" index is used
// to locate a camera in space. If I have a camera that is moving over time, the
// intrinsics index will remain the same, while the extrinsics index will change
typedef struct
{
    // indexes the intrinsics array
    int  intrinsics;
    // indexes the extrinsics array. -1 means "at coordinate system reference"
    int  extrinsics;
} mrcal_camera_index_t;


// An observation of a calibration board. Each "observation" is ONE camera
// observing a board
typedef struct
{
    // which camera is making this observation
    mrcal_camera_index_t icam;

    // indexes the "frames" array to select the pose of the calibration object
    // being observed
    int                  iframe;
} mrcal_observation_board_t;

// An observation of a discrete point. Each "observation" is ONE camera
// observing a single point in space
typedef struct
{
    // which camera is making this observation
    mrcal_camera_index_t icam;

    // indexes the "points" array to select the position of the point being
    // observed
    int                  i_point;

    // Observed pixel coordinates. This works just like elements of
    // observations_board_pool:
    //
    // .x, .y are the pixel observations
    // .z is the weight of the observation. Most of the weights are expected to
    // be 1.0. Less precise observations have lower weights.
    // .z<0 indicates that this is an outlier. This is respected on
    // input
    //
    // Unlike observations_board_pool, outlier rejection is NOT YET IMPLEMENTED
    // for points, so outlier points will NOT be found and reported on output in
    // .z<0
    mrcal_point3_t px;
} mrcal_observation_point_t;


// Bits indicating which parts of the optimization problem being solved. We can
// ask mrcal to solve for ALL the lens parameters and ALL the geometry and
// everything else. OR we can ask mrcal to lock down some part of the
// optimization problem, and to solve for the rest. If any variables are locked
// down, we use their initial values passed-in to mrcal_optimize()
typedef struct
{
    // If true, we solve for the intrinsics core. Applies only to those models
    // that HAVE a core (fx,fy,cx,cy)
    bool do_optimize_intrinsics_core        : 1;

    // If true, solve for the non-core lens parameters
    bool do_optimize_intrinsics_distortions : 1;

    // If true, solve for the geometry of the cameras
    bool do_optimize_extrinsics             : 1;

    // If true, solve for the poses of the calibration object
    bool do_optimize_frames                 : 1;

    // If true, optimize the shape of the calibration object
    bool do_optimize_calobject_warp         : 1;

    // If true, apply the regularization terms in the solver
    bool do_apply_regularization            : 1;

    // Whether to try to find NEW outliers. The outliers given on
    // input are respected regardless
    bool do_apply_outlier_rejection         : 1;

} mrcal_problem_selections_t;

// Constants used in a mrcal optimization. This is similar to
// mrcal_problem_selections_t, but contains numerical values rather than just
// bits
typedef struct
{
    // The minimum distance of an observed discrete point from its observing
    // camera. Any observation of a point below this range will be penalized to
    // encourage the optimizer to move the point further away from the camera
    double  point_min_range;


    // The maximum distance of an observed discrete point from its observing
    // camera. Any observation of a point abive this range will be penalized to
    // encourage the optimizer to move the point closer to the camera
    double  point_max_range;
} mrcal_problem_constants_t;



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

// An X-macro-generated mrcal_stats_t. This structure is returned by the
// optimizer, and contains some statistics about the optimization
#define MRCAL_STATS_ITEM(_)                                             \
    /* The RMS error of the optimized fit at the optimum. Generally the residual */ \
    /* vector x contains error values for each element of q, so N observed pixels */ \
    /* produce 2N measurements: len(x) = 2*N. And the RMS error is */   \
    /*   sqrt( norm2(x) / N ) */                                        \
    _(double,         rms_reproj_error__pixels,   PyFloat_FromDouble)   \
                                                                        \
    /* How many pixel observations were thrown out as outliers. Each pixel */ \
    /* observation produces two measurements. Note that this INCLUDES any */ \
    /* outliers that were passed-in at the start */                     \
    _(int,            Noutliers,                  PyLong_FromLong)
#define MRCAL_STATS_ITEM_DEFINE(type, name, pyconverter) type name;
typedef struct
{
    MRCAL_STATS_ITEM(MRCAL_STATS_ITEM_DEFINE)
} mrcal_stats_t;


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
                int buffer_size_p_packed,

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
                             int buffer_size_p_packed,

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

// The "intrinsics core" of a camera. This defines the final step of a
// projection operation. For instance with a pinhole model we have
//
//   q[0] = focal_xy[0] * x/z + center_xy[0]
//   q[1] = focal_xy[1] * y/z + center_xy[1]
typedef struct
{
    double focal_xy [2];
    double center_xy[2];
} mrcal_intrinsics_core_t;

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
int mrcal_measurement_index_regularization(int Nobservations_board,
                                           int Nobservations_point,
                                           int calibration_object_width_n,
                                           int calibration_object_height_n);
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



// structure containing a camera pose + lens model. Used for .cameramodel
// input/output
typedef struct
{
    double            rt_cam_ref[6];
    unsigned int      imagersize[2];
    mrcal_lensmodel_t lensmodel;
    double            intrinsics[];
} mrcal_cameramodel_t;

// if len>0, the string doesn't need to be 0-terminated. If len<=0, the end of
// the buffer IS indicated by a 0 bytes
mrcal_cameramodel_t* mrcal_read_cameramodel_string(const char* string, int len);
mrcal_cameramodel_t* mrcal_read_cameramodel_file  (const char* filename);
void                 mrcal_free_cameramodel(mrcal_cameramodel_t** cameramodel);

bool mrcal_write_cameramodel_file(const char* filename,
                                  const mrcal_cameramodel_t* cameramodel);

// Public ABI stuff, that's not for end-user consumption
#include "mrcal_internal.h"
