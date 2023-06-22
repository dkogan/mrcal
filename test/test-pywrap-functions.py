#!/usr/bin/env python3

r'''Tests the python-wrapped C API
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


model_splined = mrcal.cameramodel(f"{testdir}/data/cam0.splined.cameramodel")
ux,uy = mrcal.knots_for_splined_models(model_splined.intrinsics()[0])
testutils.confirm_equal(ux,
                        np.array([-1.33234678,-1.15470054,-0.9770543,-0.79940807,-0.62176183,-0.44411559,-0.26646936,-0.08882312,0.08882312,0.26646936,0.44411559,0.62176183,0.79940807,0.9770543,1.15470054,1.33234678]),
                        msg=f"knots_for_splined_models ux")
testutils.confirm_equal(uy,
                        np.array([-0.88823118,-0.71058495,-0.53293871,-0.35529247,-0.17764624,0.,0.17764624,0.35529247,0.53293871,0.71058495,0.88823118]),
                        msg=f"knots_for_splined_models uy")



meta = mrcal.lensmodel_metadata_and_config(model_splined.intrinsics()[0])
meta_ref = {'has_core': 1,
            'has_gradients': 1,
            'noncentral': 0,
            'can_project_behind_camera': 1,
            'order': 3,
            'Nx': 16,
            'Ny': 11,
            'fov_x_deg': 120}
testutils.confirm_equal(meta, meta_ref,
                        msg="lensmodel_metadata_and_config() keys")

testutils.finish()
