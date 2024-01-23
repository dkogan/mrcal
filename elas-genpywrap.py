#!/usr/bin/python3

# Copyright (c) 2017-2023 California Institute of Technology ("Caltech"). U.S.
# Government sponsorship acknowledged. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

r'''Python-wrap the libelas stereo matching

'''

import sys
import os

import numpy as np
import numpysane as nps

import numpysane_pywrap as npsp


docstring_module = '''libelas stereo matching

This is the written-in-C Python extension module that wraps the stereo matching
routine in libelas.

All functions are exported into the mrcal module. So you can call these via
mrcal._elas_npsp.fff() or mrcal.fff(). The latter is preferred.

'''

m = npsp.module( name      = "_elas_npsp",
                 header    = '#include "stereo-matching-libelas.h"',
                 docstring = docstring_module )

m.function( "stereo_matching_libelas",
            r'''Compute a stereo disparity map using libelas

SYNOPSIS

    image0 = mrcal.load_image('left.png',  bpp = 8, channels = 1)
    image1 = mrcal.load_image('right.png', bpp = 8, channels = 1)

    disparity0, disparity1 = \
      mrcal.stereo_matching_libelas(image0, image1)

libelas is an external library able to compute disparity maps. This function
provides a Python interface to this library. This function supports broadcasting
fully.

The inputs are given as np.uint8 grayscale images. They don't need to be stored
densely, but the strides must match between the two images.

The output disparities are returned as np.float32 images. These are stored
densely. Both left and right disparities are currently returned.

The matcher options are given in keyword arguments. Anything omitted will be
taken from the "ROBOTICS" setting, as defined in libelas/src/elas.h.

ARGUMENTS

- image0: an array of shape (H,W) and dtype np.uint8. This is the left rectified
  image

- image1: an array of shape (H,W) and dtype np.uint8. This is the right rectified
  image

The rest are parameters, with the description coming directly from the comment
in libelas/src/elas.h

- disparity_min: min disparity

- disparity_max: max disparity

- support_threshold: max. uniqueness ratio (best vs. second best support match)

- support_texture: min texture for support points

- candidate_stepsize: step size of regular grid on which support points are
  matched

- incon_window_size: window size of inconsistent support point check

- incon_threshold: disparity similarity threshold for support point to be
  considered consistent

- incon_min_support: minimum number of consistent support points

- add_corners: add support points at image corners with nearest neighbor
  disparities

- grid_size: size of neighborhood for additional support point extrapolation

- beta: image likelihood parameter

- gamma: prior constant

- sigma: prior sigma

- sradius: prior sigma radius

- match_texture: min texture for dense matching

- lr_threshold: disparity threshold for left/right consistency check

- speckle_sim_threshold: similarity threshold for speckle segmentation

- speckle_size: maximal size of a speckle (small speckles get removed)

- ipol_gap_width: interpolate small gaps (left<->right, top<->bottom)

- filter_median: optional median filter (approximated)

- filter_adaptive_mean: optional adaptive mean filter (approximated)

- postprocess_only_left: saves time by not postprocessing the right image

- subsampling: saves time by only computing disparities for each 2nd pixel. For
  this option D1 and D2 must be passed with size width/2 x height/2 (rounded
  towards zero)

RETURNED VALUE

A length-2 tuple containing the left and right disparity images. Each one is a
numpy array with the same shape as the input images, but with dtype.np=float32

    ''',
            args_input       = ('image0', 'image1'),
            prototype_input  = (('H','W'), ('H','W')),
            prototype_output = (('H','W'), ('H','W')),

            # default values are the "ROBOTICS" setting, as defined in elas.h
            extra_args = (("int",   "disparity_min",         "0",    "i"),
                          ("int",   "disparity_max",         "255",  "i"),
                          ("float", "support_threshold",     "0.85", "f"),
                          ("int",   "support_texture",       "10",   "i"),
                          ("int",   "candidate_stepsize",    "5",    "i"),
                          ("int",   "incon_window_size",     "5",    "i"),
                          ("int",   "incon_threshold",       "5",    "i"),
                          ("int",   "incon_min_support",     "5",    "i"),
                          ("bool",  "add_corners",           "0",    "p"),
                          ("int",   "grid_size",             "20",   "i"),
                          ("float", "beta",                  "0.02", "f"),
                          ("float", "gamma",                 "3",    "f"),
                          ("float", "sigma",                 "1",    "f"),
                          ("float", "sradius",               "2",    "f"),
                          ("int",   "match_texture",         "1",    "i"),
                          ("int",   "lr_threshold",          "2",    "i"),
                          ("float", "speckle_sim_threshold", "1",    "f"),
                          ("int",   "speckle_size",          "200",  "i"),
                          ("int",   "ipol_gap_width",        "3",    "i"),
                          ("bool",  "filter_median",         "0",    "p"),
                          ("bool",  "filter_adaptive_mean",  "1",    "p"),
                          ("bool",  "postprocess_only_left", "1",    "p"),
                          ("bool",  "subsampling",           "0",    "p" )),

            Ccode_validate = r'''return CHECK_CONTIGUOUS_AND_SETERROR__output0() &&
                                        CHECK_CONTIGUOUS_AND_SETERROR__output1() &&
                                        strides_slice__image0[1] == 1            &&
                                        strides_slice__image1[1] == 1            &&
                                        strides_slice__image0[0] == strides_slice__image1[0];''',

            Ccode_slice_eval = \
                { (np.uint8,np.uint8,  np.float32, np.float32):
                 r'''
                 mrcal_stereo_matching_libelas( (float*)data_slice__output0,
                                                (float*)data_slice__output1,
                                                (const uint8_t*)data_slice__image0,
                                                (const uint8_t*)data_slice__image1,
                                                dims_slice__image0[1], dims_slice__image0[0], strides_slice__image0[0],
                                                *disparity_min,
                                                *disparity_max,
                                                *support_threshold,
                                                *support_texture,
                                                *candidate_stepsize,
                                                *incon_window_size,
                                                *incon_threshold,
                                                *incon_min_support,
                                                *add_corners,
                                                *grid_size,
                                                *beta,
                                                *gamma,
                                                *sigma,
                                                *sradius,
                                                *match_texture,
                                                *lr_threshold,
                                                *speckle_sim_threshold,
                                                *speckle_size,
                                                *ipol_gap_width,
                                                *filter_median,
                                                *filter_adaptive_mean,
                                                *postprocess_only_left,
                                                *subsampling );
                 return true;
                 '''},
)

m.write()
