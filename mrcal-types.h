#pragma once

#include <stdint.h>
#include <stdbool.h>

#include "basic-geometry.h"

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
// This barfs with g++ 4.8, so I disable it for C++ in general. Checking it for
// C code is sufficient
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
    _(has_gradients,             bool, "i",,, :1, cookie)               \
                                                                        \
    /* Whether this is a noncentral model.Currently the only noncentral  */ \
    /* model we have is CAHVORE. There will be more later.               */ \
    _(noncentral,                bool, "i",,, :1, cookie)

typedef struct
{
    MRCAL_LENSMODEL_META_LIST(_MRCAL_ITEM_DEFINE_ELEMENT, )
} mrcal_lensmodel_metadata_t;


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


// structure containing a camera pose + lens model. Used for .cameramodel
// input/output
typedef struct
{
    double            rt_cam_ref[6];
    unsigned int      imagersize[2];
    mrcal_lensmodel_t lensmodel;
    double            intrinsics[];
} mrcal_cameramodel_t;
