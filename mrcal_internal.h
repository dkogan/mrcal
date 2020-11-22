#pragma once

// THESE ARE NOT A PART OF THE EXTERNAL API. Exported for the mrcal python
// wrapper only

// These models have no precomputed data
typedef struct {} mrcal_LENSMODEL_PINHOLE__precomputed_t;
typedef struct {} mrcal_LENSMODEL_STEREOGRAPHIC__precomputed_t;
typedef struct {} mrcal_LENSMODEL_OPENCV4__precomputed_t;
typedef struct {} mrcal_LENSMODEL_OPENCV5__precomputed_t;
typedef struct {} mrcal_LENSMODEL_OPENCV8__precomputed_t;
typedef struct {} mrcal_LENSMODEL_OPENCV12__precomputed_t;
typedef struct {} mrcal_LENSMODEL_CAHVOR__precomputed_t;
typedef struct {} mrcal_LENSMODEL_CAHVORE__precomputed_t;

// The splined stereographic models configuration parameters can be used to
// compute the segment size. I cache this computation
typedef struct
{
    // The distance between adjacent knots (1 segment) is u_per_segment =
    // 1/segments_per_u
    double segments_per_u;
} mrcal_LENSMODEL_SPLINED_STEREOGRAPHIC__precomputed_t;

typedef struct
{
    bool ready;
    union
    {
#define PRECOMPUTED_STRUCT(s,n) mrcal_ ##s##__precomputed_t s##__precomputed;
        MRCAL_LENSMODEL_LIST(PRECOMPUTED_STRUCT);
#undef PRECOMPUTED_STRUCT
    };
} mrcal_projection_precomputed_t;


void _mrcal_project_internal_opencv( // outputs
                                    mrcal_point2_t* q,
                                    mrcal_point3_t* dq_dp,               // may be NULL
                                    double* dq_dintrinsics_nocore, // may be NULL

                                    // inputs
                                    const mrcal_point3_t* p,
                                    int N,
                                    const double* intrinsics,
                                    int Nintrinsics);
bool _mrcal_project_internal_cahvore( // out
                                     mrcal_point2_t* out,

                                     // in
                                     const mrcal_point3_t* v,
                                     int N,

                                     // core, distortions concatenated
                                     const double* intrinsics);
bool _mrcal_project_internal( // out
                             mrcal_point2_t* q,

                             // Stored as a row-first array of shape (N,2,3). Each
                             // trailing ,3 dimension element is a mrcal_point3_t
                             mrcal_point3_t* dq_dp,
                             // core, distortions concatenated. Stored as a row-first
                             // array of shape (N,2,Nintrinsics). This is a DENSE array.
                             // High-parameter-count lens models have very sparse
                             // gradients here, and the internal project() function
                             // returns those sparsely. For now THIS function densifies
                             // all of these
                             double*   dq_dintrinsics,

                             // in
                             const mrcal_point3_t* p,
                             int N,
                             mrcal_lensmodel_t lensmodel,
                             // core, distortions concatenated
                             const double* intrinsics,

                             int Nintrinsics,
                             const mrcal_projection_precomputed_t* precomputed);
void _mrcal_precompute_lensmodel_data(mrcal_projection_precomputed_t* precomputed,
                                      mrcal_lensmodel_t lensmodel);
bool _mrcal_unproject_internal( // out
                               mrcal_point3_t* out,

                               // in
                               const mrcal_point2_t* q,
                               int N,
                               mrcal_lensmodel_t lensmodel,
                               // core, distortions concatenated
                               const double* intrinsics,
                               const mrcal_projection_precomputed_t* precomputed);

// Report the number of non-zero entries in the optimization jacobian
int _mrcal_num_j_nonzero(int Nobservations_board,
                         int Nobservations_point,
                         int calibration_object_width_n,
                         int calibration_object_height_n,
                         int Ncameras_intrinsics, int Ncameras_extrinsics,
                         int Nframes,
                         int Npoints, int Npoints_fixed,
                         const mrcal_observation_board_t* observations_board,
                         const mrcal_observation_point_t* observations_point,
                         mrcal_problem_details_t problem_details,
                         mrcal_lensmodel_t lensmodel);



// Constants used in a mrcal optimization
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
    _(int,            Noutliers,                  PyInt_FromLong)
#define MRCAL_STATS_ITEM_DEFINE(type, name, pyconverter) type name;
typedef struct
{
    MRCAL_STATS_ITEM(MRCAL_STATS_ITEM_DEFINE)
} mrcal_stats_t;


// Solve the given optimization problem
//
// This is the entry point to the mrcal optimization routine. The argument list
// is commented. It is expected that this will be called from Python only.
mrcal_stats_t
mrcal_optimize( // out
                // Each one of these output pointers may be NULL
                // Shape (Nstate,)
                double* p_packed_final,
                // used only to confirm that the user passed-in the buffer they
                // should have passed-in. The size must match exactly
                int buffer_size_p_packed_final,

                // Shape (Nmeasurements,)
                double* x_final,
                // used only to confirm that the user passed-in the buffer they
                // should have passed-in. The size must match exactly
                int buffer_size_x_final,

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

                // intrinsics is a concatenation of the intrinsics core and the
                // distortion params. The specific distortion parameters may
                // vary, depending on lensmodel, so this is a variable-length
                // structure
                double*             intrinsics,         // Ncameras_intrinsics * NlensParams
                mrcal_pose_t*       extrinsics_fromref, // Ncameras_extrinsics of these. Transform FROM the reference frame
                mrcal_pose_t*       frames_toref,       // Nframes of these.    Transform TO the reference frame
                mrcal_point3_t*     points,             // Npoints of these.    In the reference frame
                mrcal_point2_t*     calobject_warp,     // 1 of these. May be NULL if !problem_details.do_optimize_calobject_warp

                // All the board pixel observations, in order. .x, .y are the
                // pixel observations .z is the weight of the observation. Most
                // of the weights are expected to be 1.0. Less precise
                // observations have lower weights.
                //
                // z<0 indicates that this is an outlier. This is respected on
                // input (even if !do_apply_outlier_rejection). New outliers are
                // marked with z<0 on output, so this isn't const
                mrcal_point3_t* observations_board_pool,
                int Nobservations_board,

                // in
                int Ncameras_intrinsics, int Ncameras_extrinsics, int Nframes,
                int Npoints, int Npoints_fixed, // at the end of points[]

                const mrcal_observation_board_t* observations_board,
                const mrcal_observation_point_t* observations_point,
                int Nobservations_point,

                bool check_gradient,
                bool verbose,
                // Whether to try to find NEW outliers. The outliers given on
                // input are respected regardless
                const bool do_apply_outlier_rejection,

                mrcal_lensmodel_t lensmodel,
                double observed_pixel_uncertainty,
                const int* imagersizes, // Ncameras_intrinsics*2 of these
                mrcal_problem_details_t          problem_details,
                const mrcal_problem_constants_t* problem_constants,

                double calibration_object_spacing,
                int calibration_object_width_n,
                int calibration_object_height_n);


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
                             double* p_packed,
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
                             const mrcal_pose_t*       extrinsics_fromref, // Ncameras_extrinsics of these. Transform FROM reference frame
                             const mrcal_pose_t*       frames_toref,       // Nframes of these.    Transform TO reference frame
                             const mrcal_point3_t*     points,             // Npoints of these.    In the reference frame
                             const mrcal_point2_t*     calobject_warp,     // 1 of these. May be NULL if !problem_details.do_optimize_calobject_warp

                             int Ncameras_intrinsics, int Ncameras_extrinsics, int Nframes,
                             int Npoints, int Npoints_fixed, // at the end of points[]

                             const mrcal_observation_board_t* observations_board,

                             // All the board pixel observations, in order. .x,
                             // .y are the pixel observations .z is the weight
                             // of the observation. Most of the weights are
                             // expected to be 1.0. Less precise observations
                             // have lower weights.
                             //
                             // z<0 indicates that this is an outlier
                             const mrcal_point3_t* observations_board_pool,
                             int Nobservations_board,

                             const mrcal_observation_point_t* observations_point,
                             int Nobservations_point,
                             bool verbose,

                             mrcal_lensmodel_t lensmodel,
                             double observed_pixel_uncertainty,
                             const int* imagersizes, // Ncameras_intrinsics*2 of these

                             mrcal_problem_details_t          problem_details,
                             const mrcal_problem_constants_t* problem_constants,

                             double calibration_object_spacing,
                             int calibration_object_width_n,
                             int calibration_object_height_n);


// frees a dogleg_solverContext_t. I don't want to #include <dogleg.h> here, so
// this is void
void mrcal_free_context(void** ctx);
