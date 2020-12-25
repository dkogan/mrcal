#pragma once

#include "basic_geometry.h"

enum mrcal_intersection_result_t
    {
        MRCAL_OK = 0,
        MRCAL_RAYS_PARALLEL,
        MRCAL_BEHIND_CAMERA
    };

// Basic closest-approach-in-3D routine
enum mrcal_intersection_result_t
triangulate_geometric( // outputs
                      mrcal_point3_t* m,

                      // These all may be NULL
                      mrcal_point3_t* dm_dv0,
                      mrcal_point3_t* dm_dv1,
                      mrcal_point3_t* dm_dt01,

                      // inputs

                      // not-necessarily normalized vectors in the camera-0
                      // coord system
                      const mrcal_point3_t* v0,
                      const mrcal_point3_t* v1,
                      const mrcal_point3_t* t01);

// Minimize L2 pinhole reprojection error. Described in "Triangulation Made
// Easy", Peter Lindstrom, IEEE Conference on Computer Vision and Pattern
// Recognition, 2010.
enum mrcal_intersection_result_t
triangulate_lindstrom( // outputs
                      mrcal_point3_t* m,

                      // These all may be NULL
                      mrcal_point3_t* dm_dv0,
                      mrcal_point3_t* dm_dv1,
                      mrcal_point3_t* dm_dRt01,

                      // inputs

                      // not-necessarily normalized vectors in the LOCAL
                      // coordinate system. This is different from the other
                      // triangulation routines
                      const mrcal_point3_t* v0_local,
                      const mrcal_point3_t* v1_local,
                      const mrcal_point3_t* Rt01);

// Minimize L1 angle error. Described in "Closed-Form Optimal Two-View
// Triangulation Based on Angular Errors", Seong Hun Lee and Javier Civera. ICCV
// 2019.
enum mrcal_intersection_result_t
triangulate_leecivera_l1( // outputs
                         mrcal_point3_t* m,

                         // These all may be NULL
                         mrcal_point3_t* dm_dv0,
                         mrcal_point3_t* dm_dv1,
                         mrcal_point3_t* dm_dt01,

                         // inputs

                         // not-necessarily normalized vectors in the camera-0
                         // coord system
                         const mrcal_point3_t* v0,
                         const mrcal_point3_t* v1,
                         const mrcal_point3_t* t01);

// Minimize L-infinity angle error. Described in "Closed-Form Optimal Two-View
// Triangulation Based on Angular Errors", Seong Hun Lee and Javier Civera. ICCV
// 2019.
enum mrcal_intersection_result_t
triangulate_leecivera_linf( // outputs
                           mrcal_point3_t* m,

                           // These all may be NULL
                           mrcal_point3_t* dm_dv0,
                           mrcal_point3_t* dm_dv1,
                           mrcal_point3_t* dm_dt01,

                           // inputs

                           // not-necessarily normalized vectors in the camera-0
                           // coord system
                           const mrcal_point3_t* v0,
                           const mrcal_point3_t* v1,
                           const mrcal_point3_t* t01);
