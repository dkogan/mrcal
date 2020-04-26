#!/usr/bin/python3

r'''Tests gradients reported by the python code

This is conceptually similar to test-gradients.py, but here I validate the
python code path

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


intrinsics = \
    ( ('LENSMODEL_PINHOLE',
       np.array((1512., 1112, 500., 333.))),
      ('LENSMODEL_OPENCV4',
       np.array((1512., 1112, 500., 333.,
                 -0.012, 0.035, -0.001, 0.002))),
      ('LENSMODEL_CAHVOR',
       np.array((4842.918,4842.771,1970.528,1085.302,
                 -0.001, 0.002, -0.637, -0.002, 0.016))))

# a few points, some wide, some not. None behind the camera
p = np.array(((1.0, 2.0, 10.0),
              (-1.1, 0.3, 1.0),
              (-0.9, -1.5, 1.0)))

delta = 1e-6

for i in intrinsics:

    q,dq_dp,dq_di = mrcal.project(p, *i, get_gradients=True)

    Nintrinsics = mrcal.getNlensParams(i[0])
    testutils.confirm_equal(dq_di.shape[-1], Nintrinsics,
                            msg=f"{i[0]}: Nintrinsics match for {i[0]}")
    if Nintrinsics != dq_di.shape[-1]:
        continue

    for ivar in range(dq_dp.shape[-1]):

        # center differences
        p1 = p.copy()
        p1[..., ivar] = p[..., ivar] - delta/2
        q1 = mrcal.project(p1, *i, get_gradients=False)
        p1[..., ivar] += delta
        q2 = mrcal.project(p1, *i, get_gradients=False)

        dq_dp_observed = (q2 - q1) / delta
        dq_dp_reported = dq_dp[..., ivar]

        testutils.confirm_equal(dq_dp_reported, dq_dp_observed,
                                eps=1e-5,
                                msg=f"{i[0]}: dq_dp matches for var {ivar}")

    for ivar in range(dq_di.shape[-1]):

        # center differences
        i1 = i[1].copy()
        i1[..., ivar] = i[1][..., ivar] - delta/2
        q1 = mrcal.project(p, i[0], i1, get_gradients=False)
        i1[..., ivar] += delta
        q2 = mrcal.project(p, i[0], i1, get_gradients=False)

        dq_di_observed = (q2 - q1) / delta
        dq_di_reported = dq_di[..., ivar]

        testutils.confirm_equal(dq_di_reported, dq_di_observed,
                                eps=1e-5,
                                msg=f"{i[0]}: dq_di matches for var {ivar}")


testutils.finish()
