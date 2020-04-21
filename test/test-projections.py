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



def check(intrinsics, p_ref, q_ref):
    q_projected = mrcal.project(p_ref, *intrinsics)
    testutils.confirm_equal(q_projected,
                            q_ref,
                            msg = f"Projecting {intrinsics[0]}",
                            eps = 1e-2)
    p_unprojected = mrcal.unproject(q_projected, *intrinsics)

    cos = nps.inner(p_unprojected, p_ref) / (nps.mag(p_ref)*nps.mag(p_unprojected))
    cos = np.clip(cos, -1, 1)
    testutils.confirm_equal( np.arccos(cos),
                             np.zeros((3,), dtype=float),
                             msg = f"Unprojecting {intrinsics[0]}",
                             eps = 1e-6)



# a few points, some wide, some not
p = np.array(((1.0, 2.0, 10.0),
              (-1.1, 0.3, 1.0),
              (-0.9, -1.5, 1.0)))



check( ('LENSMODEL_PINHOLE', np.array((1512., 1112, 500., 333.))),
       p,
       np.array([[  651.2,   555.4],
                 [-1163.2,   666.6],
                 [ -860.8, -1335. ]]))

check( ('LENSMODEL_OPENCV4', np.array((1512., 1112, 500., 333.,
                                       -0.012, 0.035, -0.001, 0.002))),
       p,
       np.array([[  651.27371  ,   555.23042  ],
                 [-1223.38516  ,   678.01468  ],
                 [-1246.7310448, -1822.799928 ]]))

check( ('LENSMODEL_OPENCV5', np.array((1512., 1112, 500., 333.,
                                       -0.012, 0.035, -0.001, 0.002, 0.019))),
       p,
       np.array([[  651.2740691 ,   555.2309482 ],
                 [-1292.8121176 ,   691.9401448 ],
                 [-1987.550162  , -2730.85863427]]))

check( ('LENSMODEL_OPENCV8', np.array((1512., 1112, 500., 333.,
                                       -0.012, 0.035, -0.001, 0.002, 0.019, 0.014, -0.056, 0.050))),
       p,
       np.array([[  651.1885442 ,   555.10514968],
                 [-1234.45480366,   680.23499814],
                 [ -770.03274263, -1238.4871943 ]]))

check( ('LENSMODEL_CAHVOR', np.array((4842.918, 4842.771, 1970.528, 1085.302,
                                      -0.001, 0.002, -0.637, -0.002, 0.016))),
       p,
       np.array([[ 2143.17840406,  1442.93419919],
                 [  -92.63813066,  1653.09646897],
                 [ -249.83199315, -2606.46477164]]))

check( ('LENSMODEL_CAHVORE', np.array((4842.918, 4842.771, 1970.528, 1085.302,
                                       -0.001, 0.002, -0.637, -0.002, 0.016, 1e-8, 2e-8, 3e-8, 0.0))),
       p,
       np.array([[2172.38542773, 1500.18979351],
                 [ 496.63466183, 1493.31670568],
                 [ 970.11788845, -568.301136  ]]))

check( ('LENSMODEL_CAHVORE', np.array((4842.918, 4842.771, 1970.528, 1085.302,
                                       -0.001, 0.002, -0.637, -0.002, 0.016, 1e-2, 2e-2, 3e-2, 0.0))),
       p,
       np.array([[2173.83776184, 1503.03685706],
                 [ 491.68297523, 1494.65934244],
                 [ 962.73055224, -580.64311844]]))

check( ('LENSMODEL_CAHVORE', np.array((4842.918, 4842.771, 1970.528, 1085.302,
                                       -0.001, 0.002, -0.637, -0.002, 0.016, 1e-2, 2e-2, 3e-2, 0.4))),
       p,
       np.array([[2204.24898633, 1562.65309019],
                 [ 426.27859312, 1512.39356812],
                 [ 882.92624231, -713.97174524]]))

testutils.finish()
