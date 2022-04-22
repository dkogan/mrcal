#!/usr/bin/python3

r'''Broadcasting test for various functions
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

def test_ref_calibration_object():
    obj = mrcal.ref_calibration_object(10,9,5)
    testutils.confirm_equal(obj.shape,
                            (9,10,3),
                            msg = "ref_calibration_object() baseline case: shape")
    testutils.confirm_equal(obj[0,1,0] - obj[0,0,0],
                            5,
                            msg = "ref_calibration_object() baseline case: dx")
    testutils.confirm_equal(obj[1,0,1] - obj[0,0,1],
                            5,
                            msg = "ref_calibration_object() baseline case: dy")


    obj = mrcal.ref_calibration_object(10,9, (5,6))
    testutils.confirm_equal(obj.shape,
                            (9,10,3),
                            msg = "ref_calibration_object() different x,y spacing: shape")
    testutils.confirm_equal(obj[0,1,0] - obj[0,0,0],
                            5,
                            msg = "ref_calibration_object() different x,y spacing: dx")
    testutils.confirm_equal(obj[1,0,1] - obj[0,0,1],
                            6,
                            msg = "ref_calibration_object() different x,y spacing: dy")


    obj = mrcal.ref_calibration_object(10,9, np.array(((5,6), (2,3))))
    testutils.confirm_equal(obj.shape,
                            (2, 9,10,3),
                            msg = "ref_calibration_object() different x,y spacing, broadcasted: shape")
    testutils.confirm_equal(obj[0,0,1,0] - obj[0,0,0,0],
                            5,
                            msg = "ref_calibration_object() different x,y spacing, broadcasted: dx[0]")
    testutils.confirm_equal(obj[0,1,0,1] - obj[0,0,0,1],
                            6,
                            msg = "ref_calibration_object() different x,y spacing, broadcasted: dy[0]")
    testutils.confirm_equal(obj[1,0,1,0] - obj[1,0,0,0],
                            2,
                            msg = "ref_calibration_object() different x,y spacing, broadcasted: dx[1]")
    testutils.confirm_equal(obj[1,1,0,1] - obj[1,0,0,1],
                            3,
                            msg = "ref_calibration_object() different x,y spacing, broadcasted: dy[1]")


    obj = mrcal.ref_calibration_object(10,9,5, calobject_warp = np.array((3,4)))
    testutils.confirm_equal(obj.shape,
                            (9,10,3),
                            msg = "ref_calibration_object() one calobject_warp: shape")


    obj = mrcal.ref_calibration_object(10,9,5, calobject_warp = np.array(((3,4),(2,5))))
    testutils.confirm_equal(obj.shape,
                            (2,9,10,3),
                            msg = "ref_calibration_object() multiple calobject_warp: shape")

    obj = mrcal.ref_calibration_object(10,9,
                                       nps.dummy(np.array(((5,6), (2,3))), -2), # shape (2,1,2)
                                       calobject_warp = np.array(((3,4),(2,5),(0.1,0.2))))
    testutils.confirm_equal(obj.shape,
                            (2,3,9,10,3),
                            msg = "ref_calibration_object() multiple calobject_warp, x,y spacing: shape")


test_ref_calibration_object()

testutils.finish()
