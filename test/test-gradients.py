#!/usr/bin/python3

r'''Tests gradients reported by the C code

Here I make sure that the gradients reported by the projection function do
describe the gradient of the values reported by that function. This test does
NOT make sure that the values are RIGHT, just that the values are consistent
with the gradients. test-projections.py looks at the values only, so together
these two tests validate the projection functionality

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
import subprocess
from io import StringIO


tests = ( # mostly all
          "LENSMODEL_PINHOLE extrinsics frames intrinsic-core intrinsic-distortions calobject-warp",
          "LENSMODEL_PINHOLE extrinsics frames                intrinsic-distortions",
          "LENSMODEL_PINHOLE extrinsics frames intrinsic-core",
          "LENSMODEL_PINHOLE extrinsics frames",
          "LENSMODEL_CAHVOR  extrinsics frames intrinsic-core intrinsic-distortions calobject-warp",
          "LENSMODEL_CAHVOR  extrinsics frames                intrinsic-distortions",
          "LENSMODEL_CAHVOR  extrinsics frames intrinsic-core",
          "LENSMODEL_CAHVOR  extrinsics frames",
          "LENSMODEL_OPENCV4 extrinsics frames intrinsic-core intrinsic-distortions calobject-warp",
          "LENSMODEL_OPENCV4 extrinsics frames                intrinsic-distortions",
          "LENSMODEL_OPENCV4 extrinsics frames intrinsic-core",
          "LENSMODEL_OPENCV4 extrinsics frames",

          # partials; cahvor
          "LENSMODEL_CAHVOR  frames intrinsic-core intrinsic-distortions",
          "LENSMODEL_CAHVOR  frames                intrinsic-distortions",
          "LENSMODEL_CAHVOR  frames intrinsic-core",
          "LENSMODEL_CAHVOR  frames",

          "LENSMODEL_CAHVOR  extrinsics intrinsic-core intrinsic-distortions",
          "LENSMODEL_CAHVOR  extrinsics                intrinsic-distortions",
          "LENSMODEL_CAHVOR  extrinsics intrinsic-core",
          "LENSMODEL_CAHVOR  extrinsics",

          "LENSMODEL_CAHVOR  intrinsic-core intrinsic-distortions",
          "LENSMODEL_CAHVOR                 intrinsic-distortions",
          "LENSMODEL_CAHVOR  intrinsic-core",

          # partials; opencv4
          "LENSMODEL_OPENCV4 frames intrinsic-core intrinsic-distortions",
          "LENSMODEL_OPENCV4 frames                intrinsic-distortions",
          "LENSMODEL_OPENCV4 frames intrinsic-core",
          "LENSMODEL_OPENCV4 frames",

          "LENSMODEL_OPENCV4 extrinsics intrinsic-core intrinsic-distortions",
          "LENSMODEL_OPENCV4 extrinsics                intrinsic-distortions",
          "LENSMODEL_OPENCV4 extrinsics intrinsic-core",
          "LENSMODEL_OPENCV4 extrinsics",

          "LENSMODEL_OPENCV4 intrinsic-core intrinsic-distortions",
          "LENSMODEL_OPENCV4                intrinsic-distortions",
          "LENSMODEL_OPENCV4 intrinsic-core"
         )

for test in tests:

    try:
        out = \
            subprocess.check_output(f"{testdir}/../test-gradients {test} 2>/dev/null | " + \
                                    "vnl-filter -p error_relative",
                                    shell = True,
                                    encoding = 'ascii')
    except Exception as e:
        testutils.confirm(False, msg=f"failed to check gradients for '{test}")
        continue

    with StringIO(out) as f:
        err_relative = nps.transpose(np.loadtxt(f))

    err_relative_99percentile = np.percentile(err_relative, 99, interpolation='lower')
    testutils.confirm(err_relative_99percentile < 1e-4, f"99%-percentile relative error={err_relative_99percentile} for {test}")

testutils.finish()
