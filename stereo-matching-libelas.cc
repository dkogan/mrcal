// Copyright (c) 2017-2023 California Institute of Technology ("Caltech"). U.S.
// Government sponsorship acknowledged. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include <elas.h>

extern "C"
{
#include "stereo-matching-libelas.h"
};

void
mrcal_stereo_matching_libelas( // output
                               float* disparity0,
                               float* disparity1,

                               // input
                               const uint8_t* image0,
                               const uint8_t* image1,

                               // image dimensions. The stride applies to the
                               // input only. The output is stored densely
                               uint32_t width, uint32_t height, uint32_t stride,

                               // parameters. These are the fields in
                               // Elas::Elas::parameters. This function provides
                               // no defaults: eerything must be set
                               int32_t disp_min,
                               int32_t disp_max,
                               float   support_threshold,
                               int32_t support_texture,
                               int32_t candidate_stepsize,
                               int32_t incon_window_size,
                               int32_t incon_threshold,
                               int32_t incon_min_support,
                               bool    add_corners,
                               int32_t grid_size,
                               float   beta,
                               float   gamma,
                               float   sigma,
                               float   sradius,
                               int32_t match_texture,
                               int32_t lr_threshold,
                               float   speckle_sim_threshold,
                               int32_t speckle_size,
                               int32_t ipol_gap_width,
                               bool    filter_median,
                               bool    filter_adaptive_mean,
                               bool    postprocess_only_left,
                               bool    subsampling )
{
    Elas::Elas::parameters param;

    param.disp_min              = disp_min;
    param.disp_max              = disp_max;
    param.support_threshold     = support_threshold;
    param.support_texture       = support_texture;
    param.candidate_stepsize    = candidate_stepsize;
    param.incon_window_size     = incon_window_size;
    param.incon_threshold       = incon_threshold;
    param.incon_min_support     = incon_min_support;
    param.add_corners           = add_corners;
    param.grid_size             = grid_size;
    param.beta                  = beta;
    param.gamma                 = gamma;
    param.sigma                 = sigma;
    param.sradius               = sradius;
    param.match_texture         = match_texture;
    param.lr_threshold          = lr_threshold;
    param.speckle_sim_threshold = speckle_sim_threshold;
    param.speckle_size          = speckle_size;
    param.ipol_gap_width        = ipol_gap_width;
    param.filter_median         = filter_median;
    param.filter_adaptive_mean  = filter_adaptive_mean;
    param.postprocess_only_left = postprocess_only_left;
    param.subsampling           = subsampling;

    Elas::Elas elas(param);
    const int32_t dimensions[] = { (int32_t)width,
                                   (int32_t)height,
                                   (int32_t)stride };

    elas.process((uint8_t*)image0, // these really are const
                 (uint8_t*)image1,
                 disparity0, disparity1,
                 dimensions);
}
