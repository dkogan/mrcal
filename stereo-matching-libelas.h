#pragma once

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
                               bool    subsampling );
