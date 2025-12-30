#!/usr/bin/env python3
import sys
import numpy as np
import numpysane as nps
import os

testdir = os.path.dirname(os.path.realpath(__file__))

# I import the LOCAL mrcal since that's what I'm testing
sys.path[:0] = f"{testdir}/..",
import mrcal
import testutils

model_filename = f"{testdir}/data/cam0.opencv8.cameramodel"
model          = mrcal.cameramodel(model_filename)

error = None
try:
    model_loaded = mrcal._mrcal._test_python_cameramodel_converter(model_filename)
except Exception as e:
    model_loaded = None
    error = e
if model_loaded is None:
    testutils.confirm(False,
                      msg = f"Failed to load '{model_filename}': '{error}'")
else:
    testutils.confirm_equal(model_loaded['lensmodel'], model.intrinsics()[0],
                            msg = f"Loaded (from file) the correct lensmodel type")
    testutils.confirm_equal(model_loaded['intrinsics'], model.intrinsics()[1],
                            msg = f"Loaded (from file) the correct intrinsics")
    testutils.confirm_equal(model_loaded['imagersize'], model.imagersize(),
                            msg = f"Loaded (from file) the imagersize")
    testutils.confirm_equal(model_loaded['rt_cam_ref'], model.rt_cam_ref(),
                            msg = f"Loaded (from file) the correct rt_cam_ref")


error = None
try:
    model_loaded = mrcal._mrcal._test_python_cameramodel_converter(model)
except Exception as e:
    model_loaded = None
    error = e
if model_loaded is None:
    testutils.confirm(False,
                      msg = f"Failed to load cameramodel('{model_filename}'): '{error}'")
else:
    testutils.confirm_equal(model_loaded['lensmodel'], model.intrinsics()[0],
                            msg = f"Loaded (from object) the correct lensmodel type")
    testutils.confirm_equal(model_loaded['intrinsics'], model.intrinsics()[1],
                            msg = f"Loaded (from object) the correct intrinsics")
    testutils.confirm_equal(model_loaded['imagersize'], model.imagersize(),
                            msg = f"Loaded (from object) the imagersize")
    testutils.confirm_equal(model_loaded['rt_cam_ref'], model.rt_cam_ref(),
                            msg = f"Loaded (from object) the correct rt_cam_ref")


testutils.finish()
