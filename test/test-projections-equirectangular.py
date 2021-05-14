#!/usr/bin/python3

r'''Tests project_equirectangular()

I do 3 things:

Here I make sure the equirectangular projection functions return the correct
values. This is a regression test, so the "right" values were recorded at some
point, and any deviation is flagged.

I make sure that project(unproject(x)) == x

I run a simple gradient check

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


# pixels/rad
fx,fy = 3000., 2000.

# pixel where latlon = (0,0) projects to. May be negative
cx,cy = (-10000., 4000.)

# a few points, some wide, some not. Some behind the camera
p = np.array(((1.0, 2.0, 10.0),
              (-1.1, 0.3, -1.0),
              (-0.9, -1.5, -1.0)))

q_projected_ref = np.array([[ -9700.99404253,   4392.88198287],
                            [-16925.83416075,   4398.25498944],
                            [-17226.33265541,   2320.61601685]])

q_projected = mrcal.project_equirectangular(p, fx,fy,cx,cy)
testutils.confirm_equal(q_projected,
                        q_projected_ref,
                        msg = f"Projecting",
                        eps = 1e-3)

p_unprojected = mrcal.unproject_equirectangular(q_projected, fx,fy,cx,cy)
cos = nps.inner(p_unprojected, p) / (nps.mag(p)*nps.mag(p_unprojected))
cos = np.clip(cos, -1, 1)
testutils.confirm_equal( np.arccos(cos),
                         np.zeros((p.shape[0],), dtype=float),
                         msg = "Unprojecting",
                         eps = 1e-6)


# Now gradients for project()
delta = 1e-6
q_projected,dq_dp_reported = mrcal.project_equirectangular(p, fx,fy,cx,cy, get_gradients=True)
testutils.confirm_equal(q_projected,
                        q_projected_ref,
                        msg = f"Projecting",
                        eps = 1e-2)
for ivar in range(3):
    p0 = p.copy()
    p1 = p.copy()
    p0[...,ivar] -= delta/2
    p1[...,ivar] += delta/2
    dq_dpivar_observed = \
        (mrcal.project_equirectangular(p1, fx,fy,cx,cy) - mrcal.project_equirectangular(p0, fx,fy,cx,cy)) / delta
    testutils.confirm_equal(dq_dp_reported[..., ivar],
                            dq_dpivar_observed,
                            msg = f"project() gradient var {ivar}",
                            eps = 1e-3)

# Now gradients for unproject()
p_unprojected,dp_dq_reported = mrcal.unproject_equirectangular(q_projected, fx,fy,cx,cy, get_gradients=True)
cos = nps.inner(p_unprojected, p) / (nps.mag(p)*nps.mag(p_unprojected))
cos = np.clip(cos, -1, 1)
testutils.confirm_equal( np.arccos(cos),
                         np.zeros((p.shape[0],), dtype=float),
                         msg = "Unprojecting",
                         eps = 1e-6)
for ivar in range(2):
    q0 = q_projected.copy()
    q1 = q_projected.copy()
    q0[...,ivar] -= delta/2
    q1[...,ivar] += delta/2
    dp_dqivar_observed = \
        (mrcal.unproject_equirectangular(q1, fx,fy,cx,cy) - mrcal.unproject_equirectangular(q0, fx,fy,cx,cy)) / delta

    testutils.confirm_equal(dp_dq_reported[..., ivar],
                            dp_dqivar_observed,
                            msg = f"unproject() gradient var {ivar}",
                            eps = 1e-3)

testutils.finish()
