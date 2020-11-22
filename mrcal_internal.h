#pragma once

// THESE ARE NOT A PART OF THE EXTERNAL API. Exported for the mrcal python
// wrapper only


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
