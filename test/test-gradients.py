#!/usr/bin/python3

r'''Tests gradients reported by the C code'''

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


tests = ( "LENSMODEL_PINHOLE extrinsics frames intrinsic-core intrinsic-distortions",
          "LENSMODEL_PINHOLE extrinsics frames                intrinsic-distortions",
          "LENSMODEL_PINHOLE extrinsics frames intrinsic-core",
          "LENSMODEL_PINHOLE extrinsics frames",
          "LENSMODEL_CAHVOR  extrinsics frames intrinsic-core intrinsic-distortions",
          "LENSMODEL_CAHVOR  extrinsics frames                intrinsic-distortions",
          "LENSMODEL_CAHVOR  extrinsics frames intrinsic-core",
          "LENSMODEL_CAHVOR  extrinsics frames",
          "LENSMODEL_OPENCV4 extrinsics frames intrinsic-core intrinsic-distortions",
          "LENSMODEL_OPENCV4 extrinsics frames                intrinsic-distortions",
          "LENSMODEL_OPENCV4 extrinsics frames intrinsic-core",
          "LENSMODEL_OPENCV4 extrinsics frames",

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
          "LENSMODEL_CAHVOR  intrinsic-core"
         )

for test in tests:

    try:
        out = \
            subprocess.check_output(f"{testdir}/../test-gradients {test} 2>/dev/null | " + \
                                    "vnl-filter --eval '{print ivar,imeasurement,error,error_relative}'",
                                    shell = True,
                                    encoding = 'ascii')
    except Exception as e:
        testutils.confirm(False, msg="failed to check gradients for '{test}")
        continue

    err_relative_max           = -1.0
    err_absolute_corresponding = -1.0

    with StringIO(out) as lines:
        for line in lines:
            (ivar,imeas,err_absolute,err_relative) = [float(x) for x in line.split()]

            if abs(err_absolute) > 1e-8 and err_relative > err_relative_max:
                err_relative_max           = err_relative
                err_absolute_corresponding = err_absolute
                ivar_corresponding         = ivar
                imeas_corresponding        = imeas

    print(f"var/meas: {ivar_corresponding}/{imeas_corresponding} err_relative_max: {err_relative_max}: err_absolute: {err_absolute_corresponding}")
