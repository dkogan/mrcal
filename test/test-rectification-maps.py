#!/usr/bin/python3

r'''Tests mrcal_rectification_maps() and mrcal.rectification_maps()
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


# I want to generate a stereo pair. I tweak the intrinsics a bit
model0 = mrcal.cameramodel(f"{testdir}/data/cam0.opencv8.cameramodel")
model1 = mrcal.cameramodel(model0)
model1.intrinsics()[1][ :2] *= np.array((1.01, -0.98))
model1.intrinsics()[1][2:4] += np.array((50, 80.))

# Left-right stereo, with sizeable rotation and position fuzz.
# I especially make sure there's a forward/back shift
rt01 = np.array((0.1, 0.2, 0.05,  3.0, 0.2, 1.0))
model1.extrinsics_rt_toref( mrcal.compose_rt(model0.extrinsics_rt_toref(),
                                             rt01))

for rectification in ('LENSMODEL_LATLON', 'LENSMODEL_PINHOLE'):

    # I use the canonical rectified-system function here to make sure that this
    # test checks only the rectification_maps function
    az_fov_deg = 90
    el_fov_deg = 50
    models_rectified = \
        mrcal.stereo._rectified_system_python( (model0, model1),
                                               az_fov_deg = az_fov_deg,
                                               el_fov_deg = el_fov_deg,
                                               pixels_per_deg_az = -1./8.,
                                               pixels_per_deg_el = -1./4.,
                                               rectification_model = rectification)

    rectification_maps_ref = \
        mrcal.stereo._rectification_maps_python((model0,model1),
                                                models_rectified)

    rectification_maps = \
        mrcal.rectification_maps((model0,model1),
                                 models_rectified)

    testutils.confirm_equal(rectification_maps,
                            rectification_maps_ref,
                            msg=f'Pixel error with ({rectification})',
                            worstcase = True,
                            eps = 1e-5)

testutils.finish()
