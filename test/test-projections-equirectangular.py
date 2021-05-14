#!/usr/bin/python3

r'''Tests project_equirectangular()

I do 3 things:

Here I make sure the equirectangular projection functions return the correct
values. This is a regression test, so the "right" values were recorded at some
point, and any deviation is flagged.

I make sure that project(unproject(x)) == x

I run a gradient check. I do these for the simple project_equirectangular()
function AND the generic project() function.

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
from test_calibration_helpers import grad


# pixels/rad
fx,fy = 3000., 2000.

# pixel where latlon = (0,0) projects to. May be negative
cx,cy = (-10000., 4000.)

intrinsics = ('LENSMODEL_EQUIRECTANGULAR', np.array((fx,fy,cx,cy)))

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
                        msg = f"project_equirectangular()",
                        worstcase = True,
                        relative  = True)

testutils.confirm_equal(mrcal.project(p, *intrinsics),
                        q_projected,
                        msg = "project() returns the same as unproject_equirectangular()",
                        worstcase = True,
                        relative  = True)

v_unprojected = mrcal.unproject_equirectangular(q_projected, fx,fy,cx,cy)
testutils.confirm_equal( nps.mag(v_unprojected),
                         1.,
                         msg = "unproject_equirectangular() returns normalized vectors",
                         worstcase = True,
                         relative  = True)
testutils.confirm_equal( v_unprojected,
                         p / nps.dummy(nps.mag(p), axis=-1),
                         msg = "unproject_equirectangular()",
                         worstcase = True,
                         relative  = True)

testutils.confirm_equal( mrcal.unproject(q_projected, *intrinsics),
                         v_unprojected,
                         msg = "unproject() returns the same as unproject_equirectangular()",
                         worstcase = True,
                         relative  = True)


# Now gradients for project()
ipt = 1
_,dq_dp_reported = mrcal.project_equirectangular(p[ipt], fx,fy,cx,cy, get_gradients=True)
dq_dp_observed = grad(lambda p: mrcal.project_equirectangular(p, fx,fy,cx,cy),
                      p[ipt])
testutils.confirm_equal(dq_dp_reported,
                        dq_dp_observed,
                        msg = f"project_equirectangular() dq/dp",
                        worstcase = True,
                        relative  = True)
_,dq_dp_reported,dq_di_reported = mrcal.project(p[ipt], *intrinsics, get_gradients=True)
dq_dp_observed = grad(lambda p: mrcal.project(p, *intrinsics),
                      p[ipt])
dq_di_observed = grad(lambda intrinsics_data: mrcal.project(p[ipt], intrinsics[0],intrinsics_data),
                      intrinsics[1])
testutils.confirm_equal(dq_dp_reported,
                        dq_dp_observed,
                        msg = f"project() dq/dp",
                        worstcase = True,
                        relative  = True)
testutils.confirm_equal(dq_di_reported,
                        dq_di_observed,
                        msg = f"project() dq/di",
                        worstcase = True,
                        relative  = True,
                        eps = 1e-5)

# Now gradients for unproject()
ipt = 1
_,dv_dq_reported = mrcal.unproject_equirectangular(q_projected[ipt], fx,fy,cx,cy, get_gradients=True)
dv_dq_observed = grad(lambda q: mrcal.unproject_equirectangular(q, fx,fy,cx,cy),
                      q_projected[ipt])
testutils.confirm_equal(dv_dq_reported,
                        dv_dq_observed,
                        msg = f"unproject_equirectangular() dv/dq",
                        worstcase = True,
                        relative  = True)
v_unprojected,dv_dq_reported,dv_di_reported = mrcal.unproject(q_projected[ipt], *intrinsics, get_gradients=True)
dv_dq_observed = grad(lambda q: mrcal.unproject(q, *intrinsics),
                      q_projected[ipt])
dv_di_observed = grad(lambda intrinsics_data: mrcal.unproject(q_projected[ipt], intrinsics[0],intrinsics_data),
                      intrinsics[1])
testutils.confirm_equal(dv_dq_reported,
                        dv_dq_observed,
                        msg = f"unproject() dv/dq",
                        worstcase = True,
                        relative  = True)
testutils.confirm_equal(dv_di_reported,
                        dv_di_observed,
                        msg = f"unproject() dv/di",
                        worstcase = True,
                        relative  = True,
                        eps = 1e-5)

v_unprojected_inplace  = v_unprojected.copy() *0
dv_dq_reported_inplace = dv_dq_reported.copy()*0
dv_di_reported_inplace = dv_di_reported.copy()*0

mrcal.unproject(q_projected[ipt], *intrinsics, get_gradients=True,
                out = (v_unprojected_inplace,dv_dq_reported_inplace,dv_di_reported_inplace))
testutils.confirm_equal(v_unprojected_inplace,
                        v_unprojected,
                        msg = f"unproject() works in-place: v_unprojected",
                        worstcase = True,
                        relative  = True)
testutils.confirm_equal(dv_dq_reported_inplace,
                        dv_dq_reported,
                        msg = f"unproject() works in-place: dv_dq",
                        worstcase = True,
                        relative  = True)
testutils.confirm_equal(dv_di_reported_inplace,
                        dv_di_reported,
                        msg = f"unproject() works in-place: dv_di",
                        worstcase = True,
                        relative  = True)

testutils.finish()
