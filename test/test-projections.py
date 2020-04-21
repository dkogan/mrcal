#!/usr/bin/python3

r'''Regression tests project()

Here I make sure the projection functions return the correct values. This is a
regression test, so the "right" values were recorded at some point, and any
deviation is flagged.

This test confirms the correct values, and test-gradients.py confirms that these
values are consistent with the reported gradients. So together these two tests
validate the projection functionality

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


# a few points, some wide, some not
p = np.array(((1.0, 2.0, 10.0),
              (-1.1, 0.3, 1.0),
              (-0.9, -1.5, 1.0)))



testutils.confirm_equal(mrcal.project(p, 'LENSMODEL_PINHOLE',
                                      np.array((1512., 1112, 500., 333.))),
                        np.array([[  651.2,   555.4],
                                  [-1163.2,   666.6],
                                  [ -860.8, -1335. ]]),
                        msg = "Projecting LENSMODEL_PINHOLE",
                        eps = 1e-2)

testutils.confirm_equal(mrcal.project(p, 'LENSMODEL_OPENCV4',
                                      np.array((1512., 1112, 500., 333.,
                                                -0.012, 0.035, -0.001, 0.002))),
                        np.array([[  651.27371  ,   555.23042  ],
                                  [-1223.38516  ,   678.01468  ],
                                  [-1246.7310448, -1822.799928 ]]),
                        msg = "Projecting LENSMODEL_OPENCV4",
                        eps = 1e-2)

testutils.confirm_equal(mrcal.project(p, 'LENSMODEL_OPENCV5',
                                      np.array((1512., 1112, 500., 333.,
                                                -0.012, 0.035, -0.001, 0.002, 0.019))),
                        np.array([[  651.2740691 ,   555.2309482 ],
                                  [-1292.8121176 ,   691.9401448 ],
                                  [-1987.550162  , -2730.85863427]]),
                        msg = "Projecting LENSMODEL_OPENCV5",
                        eps = 1e-2)

testutils.confirm_equal(mrcal.project(p, 'LENSMODEL_OPENCV8',
                                      np.array((1512., 1112, 500., 333.,
                                                -0.012, 0.035, -0.001, 0.002, 0.019, 0.014, -0.056, 0.050))),
                        np.array([[  651.1885442 ,   555.10514968],
                                  [-1234.45480366,   680.23499814],
                                  [ -770.03274263, -1238.4871943 ]]),
                        msg = "Projecting LENSMODEL_OPENCV8",
                        eps = 1e-2)

testutils.confirm_equal(mrcal.project(p, 'LENSMODEL_CAHVOR',
                                      np.array((4842.918, 4842.771, 1970.528, 1085.302,
                                                -0.001, 0.002, -0.637, -0.002, 0.016))),
                        np.array([[ 2143.17840406,  1442.93419919],
                                  [  -92.63813066,  1653.09646897],
                                  [ -249.83199315, -2606.46477164]]),
                        msg = "Projecting LENSMODEL_CAHVOR",
                        eps = 1e-2)

testutils.confirm_equal(mrcal.project(p, 'LENSMODEL_CAHVORE',
                                      np.array((4842.918, 4842.771, 1970.528, 1085.302,
                                                -0.001, 0.002, -0.637, -0.002, 0.016, 1e-8, 2e-8, 3e-8, 0.0))),
                        np.array([[2172.38542773, 1500.18979351],
                                  [ 496.63466183, 1493.31670568],
                                  [ 970.11788845, -568.301136  ]]),
                        msg = "Projecting LENSMODEL_CAHVORE",
                        eps = 1e-2)

testutils.confirm_equal(mrcal.project(p, 'LENSMODEL_CAHVORE',
                                      np.array((4842.918, 4842.771, 1970.528, 1085.302,
                                                -0.001, 0.002, -0.637, -0.002, 0.016, 1e-2, 2e-2, 3e-2, 0.0))),
                        np.array([[2173.83776184, 1503.03685706],
                                  [ 491.68297523, 1494.65934244],
                                  [ 962.73055224, -580.64311844]]),
                        msg = "Projecting LENSMODEL_CAHVORE",
                        eps = 1e-2)

testutils.confirm_equal(mrcal.project(p, 'LENSMODEL_CAHVORE',
                                      np.array((4842.918, 4842.771, 1970.528, 1085.302,
                                                -0.001, 0.002, -0.637, -0.002, 0.016, 1e-2, 2e-2, 3e-2, 0.4))),
                        np.array([[2204.24898633, 1562.65309019],
                                  [ 426.27859312, 1512.39356812],
                                  [ 882.92624231, -713.97174524]]),
                        msg = "Projecting LENSMODEL_CAHVORE",
                        eps = 1e-2)

testutils.finish()
