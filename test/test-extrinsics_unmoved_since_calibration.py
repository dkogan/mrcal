#!/usr/bin/python3

r'''Checks mrcal._extrinsics_unmoved_since_calibration()
'''

import sys
import argparse
import re
import os


testdir = os.path.dirname(os.path.realpath(__file__))

# I import the LOCAL mrcal since that's what I'm testing
sys.path[:0] = f"{testdir}/..",
import mrcal
import testutils
import numpy as np
import numpysane as nps

from test_calibration_helpers import calibration_baseline

############# Set up my world, and compute all the perfect positions, pixel
############# observations of everything
pixel_uncertainty_stdev = 1.5
object_spacing          = 0.1
object_width_n          = 10
object_height_n         = 9
calobject_warp_true     = np.array((0.002, -0.005))

model       = 'opencv4'
Ncameras    = 4
Nframes     = 10
fixedframes = False

extrinsics_rt_fromref_true = \
    np.array(((0,    0,    0,      0,   0,   0),
              (0.08, 0.2,  0.02,   1.,  0.9, 0.1),
              (0.01, 0.07, 0.2,    2.1, 0.4, 0.2),
              (-0.1, 0.08, 0.08,   3.4, 0.2, 0.1), ))

optimization_inputs_baseline,                          \
models_true, models_baseline,                          \
indices_frame_camintrinsics_camextrinsics,             \
lensmodel, Nintrinsics, imagersizes,                   \
intrinsics_true, extrinsics_true_mounted, frames_true, \
observations_true,                                     \
Nframes =                                         \
    calibration_baseline(model,
                         Ncameras,
                         Nframes,
                         None,
                         pixel_uncertainty_stdev,
                         object_width_n,
                         object_height_n,
                         object_spacing,
                         extrinsics_rt_fromref_true,
                         calobject_warp_true,
                         fixedframes,
                         testdir)

for i in range(len(models_baseline)):
    testutils.confirm(models_baseline[i]._extrinsics_unmoved_since_calibration(),
                      msg = f"Camera {i} unmoved")

extrinsics_rt_fromref = extrinsics_rt_fromref_true.copy()
extrinsics_rt_fromref[:,-1] += 1e-3

for i in range(len(models_baseline)):
    models_baseline[i].extrinsics_rt_fromref(extrinsics_rt_fromref[i])

for i in range(len(models_baseline)):
    testutils.confirm(not models_baseline[i]._extrinsics_unmoved_since_calibration(),
                      msg = f"Camera {i} moved")

testutils.finish()
