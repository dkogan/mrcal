#!/usr/bin/python3

r'''diff test

Make sure the projection-diff function produces correct results
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


model       = mrcal.cameramodel(f"{testdir}/data/cam.splined.cameramodel")
gridn_width = 50

# Compare the model to itself. I should get 0 diff and identity transform
difflen, diff, q0, implied_Rt10 = \
    mrcal.projection_diff( (model,model),
                           gridn_width       = gridn_width,
                           distance          = None,
                           use_uncertainties = False )

testutils.confirm_equal( difflen.shape[1], gridn_width,
                         msg = "Expected number of columns" )
testutils.confirm_equal( difflen.shape[0], int(round( model.imagersize()[1] / model.imagersize()[0] * gridn_width)),
                         msg = "Expected number of rows" )

testutils.confirm_equal( difflen*0, difflen,
                         eps = 0.02,
                         worstcase = True,
                         relative  = False,
                         msg = "diff(model,model) at infinity should be 0")

testutils.confirm_equal( 0, np.arccos((np.trace(implied_Rt10[:3,:]) - 1) / 2.) * 180./np.pi,
                         eps = 0.01,
                         msg = "diff(model,model) at infinity should produce a rotation of 0 deg")

testutils.confirm_equal( 0, nps.mag(implied_Rt10[3,:]),
                         eps = 0.01,
                         msg = "diff(model,model) at infinity should produce a rotation of 0 m")

difflen, diff, q0, implied_Rt10 = \
    mrcal.projection_diff( (model,model),
                           gridn_width       = 50,
                           distance          = 3.,
                           use_uncertainties = False )

testutils.confirm_equal( difflen*0, difflen,
                         eps = 0.02,
                         worstcase = True,
                         relative  = False,
                         msg = "diff(model,model) at 3m should be 0")

testutils.confirm_equal( 0, np.arccos((np.trace(implied_Rt10[:3,:]) - 1) / 2.) * 180./np.pi,
                         eps = 0.01,
                         msg = "diff(model,model) at 3m should produce a rotation of 0 deg")

testutils.confirm_equal( 0, nps.mag(implied_Rt10[3,:]),
                         eps = 0.01,
                         msg = "diff(model,model) at 3m should produce a rotation of 0 m")

testutils.finish()
