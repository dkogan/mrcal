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

// The equivalent function in Python is _rectified_system_python() in stereo.py
//
// Documentation is in the docstring of mrcal.rectified_system()
bool mrcal_rectified_system(// output
                            unsigned int*     imagersize_rectified,
                            double*           fxycxy_rectified,
                            double*           rt_rect0_ref,
                            double*           baseline,

                            // input, output
                            // > 0: use given value
                            // < 0: autodetect and scale
                            double* pixels_per_deg_az,
                            double* pixels_per_deg_el,

                            // input, output
                            // if(..._autodetect) { the results are returned here }
                            mrcal_point2_t* azel_fov_deg,
                            mrcal_point2_t* azel0_deg,

                            // input
                            const mrcal_lensmodel_t* lensmodel0,
                            const double*            intrinsics0,

                            const double*            rt_cam0_ref,
                            const double*            rt_cam1_ref,

                            const mrcal_lensmodel_type_t rectification_model_type,

                            bool   az0_deg_autodetect,
                            bool   el0_deg_autodetect,
                            bool   az_fov_deg_autodetect,
                            bool   el_fov_deg_autodetect);