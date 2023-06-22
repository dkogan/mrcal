#!/usr/bin/env python3

r'''Validates estimate_monocular_calobject_poses_Rt_tocam()

estimate_monocular_calobject_poses_Rt_tocam() is part of the calibration seeding
functions. This uses solvePnP(), which is sensitive to the user-supplied
focal-length estimates. Here I load some data that has been a problem in the
past, and make sure that estimate_monocular_calobject_poses_Rt_tocam() can
handle it

'''

import sys
import numpy as np
import numpysane as nps
import os

testdir = os.path.dirname(os.path.realpath(__file__))

# I import the LOCAL mrcal since that's what I'm testing
sys.path[:0] = f"{testdir}/..",
import mrcal
import testutils

import pickle


# Load the data. These are written by the disabled-by-default code at the end of
# estimate_monocular_calobject_poses_Rt_tocam()
test_cases = (
              # ./mrcal-calibrate-cameras                                                       \
              #   --corners-cache newfish-from-mike-email-extreme-fisheys/VuzeXR/calibration1/fishcorners.vnl \
              #   --lensmodel LENSMODEL_OPENCV8 \
              #   --focal 500 \
              #   --object-spacing 0.048                                                        \
              #   --object-width-n 10                                                           \
              #   --observed-pixel-uncertainty 2                                                \
              #   'HET_0315_L*.jpg' 'HET_0315_R*.jpg'
              'solvepnp-ultrawide-focal-too-long',

              # ./mrcal-calibrate-cameras                               \
              # --corners-cache doc/out/external/2022-11-05--dtla-overpass--samyang--alpha7/2-f22-infinity/corners.vnl \
              # --lensmodel LENSMODEL_OPENCV8                           \
              # --focal 1900                                            \
              # --object-spacing 0.0588                                 \
              # --object-width-n 14                                     \
              # --observed-pixel-uncertainty 2                          \
              # '*.JPG'
              'solvepnp-wide-focal-too-wide')

for t in test_cases:
    filename = f'{testdir}/data/{t}.pickle'
    with open(filename,'rb') as f:
        args = pickle.load(f)
        testutils.confirm_does_not_raise( \
            lambda: mrcal.estimate_monocular_calobject_poses_Rt_tocam(*args),
            msg = t)

testutils.finish()
