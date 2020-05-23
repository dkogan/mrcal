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
import re


tests = ( # mostly all
          "LENSMODEL_PINHOLE extrinsics frames intrinsic-core intrinsic-distortions calobject-warp",
          "LENSMODEL_PINHOLE extrinsics frames                intrinsic-distortions",
          "LENSMODEL_PINHOLE extrinsics frames intrinsic-core",
          "LENSMODEL_PINHOLE extrinsics frames",

          "LENSMODEL_STEREOGRAPHIC extrinsics frames intrinsic-core intrinsic-distortions calobject-warp",
          "LENSMODEL_STEREOGRAPHIC extrinsics                       intrinsic-distortions",
          "LENSMODEL_STEREOGRAPHIC extrinsics frames intrinsic-core",
          "LENSMODEL_STEREOGRAPHIC frames",

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
          "LENSMODEL_OPENCV4 intrinsic-core",

          # splined
          "LENSMODEL_SPLINED_STEREOGRAPHIC_3 extrinsics intrinsic-distortions",
          "LENSMODEL_SPLINED_STEREOGRAPHIC_3 frames extrinsics intrinsic-distortions calobject-warp",
          "LENSMODEL_SPLINED_STEREOGRAPHIC_2 extrinsics intrinsic-distortions",
          "LENSMODEL_SPLINED_STEREOGRAPHIC_2 frames extrinsics intrinsic-distortions calobject-warp",
         )


def get_variable_map(s):
    r'''Generates a perl snippet to classify a variable

    The output classes are

    0 intrinsics-core
    1 intrinsics-distortions
    2 extrinsics
    3 frames
    4 discrete points
    5 calobject_warp'''

    m = re.search("^## Intrinsics: (\d+) variables per camera \((\d+) for the core, (\d+) for the rest; (\d+) total\). Starts at variable (\d+)", s, re.M)
    NintrinsicsPerCamera            = int(m.group(1))
    NintrinsicsCorePerCamera        = int(m.group(2))
    NintrinsicsDistortionsPerCamera = int(m.group(3))
    Nvar_intrinsics                 = int(m.group(4))
    ivar0_intrinsics                = int(m.group(5))

    m = re.search("^## Extrinsics: \d+ variables per camera for all cameras except camera 0 \((\d+) total\). Starts at variable (\d+)", s, re.M)
    Nvar_extrinsics  = int(m.group(1))
    ivar0_extrinsics = int(m.group(2))

    m = re.search("^## Frames: \d+ variables per frame \((\d+) total\). Starts at variable (\d+)", s, re.M)
    Nvar_frames  = int(m.group(1))
    ivar0_frames = int(m.group(2))

    m = re.search("^## Discrete points: \d+ variables per point \((\d+) total\). Starts at variable (\d+)", s, re.M)
    Nvar_points  = int(m.group(1))
    ivar0_points = int(m.group(2))

    m = re.search("^## calobject_warp: (\d+) variables. Starts at variable (\d+)", s, re.M)
    Nvar_calobject_warp  = int(m.group(1))
    ivar0_calobject_warp = int(m.group(2))


    intrinsics_any = f"ivar >= {ivar0_intrinsics}     && ivar < {ivar0_intrinsics}    +{Nvar_intrinsics}"
    extrinsics     = f"ivar >= {ivar0_extrinsics}     && ivar < {ivar0_extrinsics}    +{Nvar_extrinsics}"
    frames         = f"ivar >= {ivar0_frames}         && ivar < {ivar0_frames}        +{Nvar_frames}"
    points         = f"ivar >= {ivar0_points}         && ivar < {ivar0_points}        +{Nvar_points}"
    calobject_warp = f"ivar >= {ivar0_calobject_warp} && ivar < {ivar0_calobject_warp}+{Nvar_calobject_warp}"

    if NintrinsicsCorePerCamera == 0:
        intrinsics_classify = 1
    else:
        intrinsics_classify = f"(ivar - {ivar0_intrinsics}) % {NintrinsicsPerCamera} < {NintrinsicsCorePerCamera} ? 0 : 1"
    err = 'die("Could not classify variable ivar. Giving up")'
    return f"({calobject_warp}) ? 5 : (({points}) ? 4 : (({frames})? 3 : (({extrinsics}) ? 2 : (({intrinsics_any}) ? ({intrinsics_classify}) : {err}))))"

def vartype_name(i):
    d = { 0: "intrinsics-core",
          1: "intrinsics-distortions",
          2: "extrinsics",
          3: "frames",
          4: "discrete points",
          5: "calobject_warp" }
    return d[i]

def get_measurement_map(s):
    r'''Generates a perl snippet to classify a measurement

    The output classes are

    0 boards
    1 points
    2 regularization'''


    m = re.search("^## Measurement calobjects: (\d+) measurements. Starts at measurement (\d+)", s, re.M)
    Nmeas_boards  = m.group(1)
    imeas0_boards = m.group(2)

    m = re.search("^## Measurement points: (\d+) measurements. Starts at measurement (\d+)", s, re.M)
    Nmeas_points  = m.group(1)
    imeas0_points = m.group(2)

    m = re.search("^## Measurement regularization: (\d+) measurements. Starts at measurement (\d+)", s, re.M)
    Nmeas_regularization  = m.group(1)
    imeas0_regularization = m.group(2)

    boards         = f"imeasurement >= {imeas0_boards}         && imeas < {imeas0_boards}         + {Nmeas_boards}"
    points         = f"imeasurement >= {imeas0_points}         && imeas < {imeas0_points}         + {Nmeas_points}"
    regularization = f"imeasurement >= {imeas0_regularization} && imeas < {imeas0_regularization} + {Nmeas_regularization}"

    err = 'die("Could not classify measurement imeasurement. Giving up")'
    return f"({regularization}) ? 2 : (({points}) ? 1 : (({boards})? 0 : {err}))"

def meastype_name(i):
    d = { 0: "boards",
          1: "points",
          2: "regularization" }
    return d[i]


for test in tests:

    try:
        full = \
            subprocess.check_output( [f"{testdir}/../test-gradients"] + test.split(),
                                     shell = False,
                                     encoding = 'ascii')
    except Exception as e:
        testutils.confirm(False, msg=f"failed to check gradients for '{test}'")
        continue

    varmap  = get_variable_map(full)
    measmap = get_measurement_map(full)

    cut = subprocess.check_output( ("vnl-filter", "--perl", "-p", f"error_relative,type={varmap},meastype={measmap}"),
                                   input=full,
                                   encoding='ascii')
    with StringIO(cut) as f:
        err_class = np.loadtxt(f)

    # I now have all the relative errors and all the variable, measurement
    # classifications. I check each class for mismatches separately. Each class
    # allows a small number of outliers. The reason I check each class
    # separately is because different classses have very different point.
    # Looking at everything together could result in ALL of one class being
    # considered an outlier, and the test would pass
    i_thisvar_all  = [err_class[:,1] == vartype  for vartype  in range(6)]
    i_thismeas_all = [err_class[:,2] == meastype for meastype in range(3)]

    for vartype in range(6):
        for meastype in range(3):
            i = i_thisvar_all[vartype] * i_thismeas_all[meastype]
            err = err_class[ i, 0 ]
            if len(err) <= 0: continue

            err_relative_99percentile = np.percentile(err, 99, interpolation='higher')
            testutils.confirm(err_relative_99percentile < 1e-3, f"99%-percentile relative error={err_relative_99percentile} for vars {vartype_name(vartype)}, meas {meastype_name(meastype)} in {test}")

testutils.finish()
