#!/usr/bin/python3

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


model_splined = mrcal.cameramodel(f"{testdir}/data/cam.splined.cameramodel")
ux,uy = mrcal.getKnotsForSplinedModels(model_splined.intrinsics()[0])
testutils.confirm_equal(ux,
                        np.array([-2.7502006 , -2.38350719, -2.01681377, -1.65012036, -1.28342695, -0.91673353, -0.55004012, -0.18334671,  0.18334671,  0.55004012,  0.91673353,  1.28342695,  1.65012036,  2.01681377,  2.38350719,  2.7502006 ]),
                        msg=f"getKnotsForSplinedModels ux")
testutils.confirm_equal(uy,
                        np.array([-2.38350719, -2.01681377, -1.65012036, -1.28342695, -0.91673353, -0.55004012, -0.18334671,  0.18334671,  0.55004012,  0.91673353,  1.28342695,  1.65012036,  2.01681377,  2.38350719]),
                        msg=f"getKnotsForSplinedModels uy")



meta = mrcal.getLensModelMeta(model_splined.intrinsics()[0])
meta_ref = {'has_core': 1,
            'can_project_behind_camera': 1,
            'spline_order': 3,
            'Nx': 16,
            'Ny': 14,
            'fov_x_deg': 200}
testutils.confirm_equal(meta, meta_ref,
                        msg="getLensModelMeta() keys")

testutils.finish()
