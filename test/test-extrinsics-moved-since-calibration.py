#!/usr/bin/env python3

r'''Checks mrcal._extrinsics_moved_since_calibration()
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
object_spacing          = 0.1
object_width_n          = 10
object_height_n         = 9
calobject_warp_true     = np.array((0.002, -0.005))

model       = 'opencv4'
Ncameras    = 4
Nframes     = 10
fixedframes = False

rt_cam_ref_true = \
    np.array(((0,    0,    0,      0,   0,   0),
              (0.08, 0.2,  0.02,   1.,  0.9, 0.1),
              (0.01, 0.07, 0.2,    2.1, 0.4, 0.2),
              (-0.1, 0.08, 0.08,   3.4, 0.2, 0.1), ))

optimization_inputs_baseline, \
models_true,                  \
frames_true =                 \
    calibration_baseline(model,
                         Ncameras,
                         Nframes,
                         None,
                         object_width_n,
                         object_height_n,
                         object_spacing,
                         rt_cam_ref_true,
                         calobject_warp_true,
                         fixedframes,
                         testdir)

models_baseline = \
    [ mrcal.cameramodel( optimization_inputs = optimization_inputs_baseline,
                         icam_intrinsics     = i) \
      for i in range(Ncameras) ]


for i in range(Ncameras):
    testutils.confirm(not models_baseline[i]._extrinsics_moved_since_calibration(),
                      msg = f"Camera {i} unmoved")

rt_cam_ref = rt_cam_ref_true.copy()
rt_cam_ref[:,-1] += 1e-3

for i in range(Ncameras):
    models_baseline[i].rt_cam_ref(rt_cam_ref[i])

for i in range(Ncameras):
    testutils.confirm(models_baseline[i]._extrinsics_moved_since_calibration(),
                      msg = f"Camera {i} moved")

testutils.finish()
