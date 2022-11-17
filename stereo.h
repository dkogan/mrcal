#pragma once

#include "mrcal-types.h"

// The equivalent function in Python is _rectified_resolution_python() in
// stereo.py
//
// Documentation is in rectified_resolution.docstring
bool mrcal_rectified_resolution( // output and input
                                 // > 0: use given value
                                 // < 0: autodetect and scale
                                 double* pixels_per_deg_az,
                                 double* pixels_per_deg_el,

                                 // input
                                 const mrcal_lensmodel_t*     lensmodel,
                                 const double*                intrinsics,
                                 const mrcal_point2_t*        azel_fov_deg,
                                 const mrcal_point2_t*        azel0_deg,
                                 const double*                R_cam0_rect0,
                                 const mrcal_lensmodel_type_t rectification_model_type);
