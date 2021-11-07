#!/usr/bin/python3

r'''Tests special-case projection functions

Simple things like project_lonlat(), project_stereographic(), etc.

I do 3 things:

Here I make sure the projection functions return the correct values. This is a
regression test, so the "right" values were recorded at some point, and any
deviation is flagged.

I make sure that project(unproject(x)) == x

I run a gradient check. I do these for the simple project_...()
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

if len(sys.argv) != 2:
    raise Exception("Need one argument on the commandline: the projection type. Currently I support 'pinhole','latlon','lonlat','stereographic'")
if   sys.argv[1] == 'pinhole' or \
     sys.argv[1] == 'latlon'  or \
     sys.argv[1] == 'lonlat':

    # pixels/rad
    fx,fy = 3000., 2000.

    # pixel where latlon = (0,0) projects to. May be negative
    cx,cy = (-10000., 4000.)

    # a few points, some wide, some not. Some behind the camera
    p = np.array(((1.0, 2.0, 10.0),
                  (-1.1, 0.3, -1.0),
                  (-0.9, -1.5, -1.0)))

    if sys.argv[1] == 'pinhole': unproject_is_normalized = False
    else:                        unproject_is_normalized = True

    if sys.argv[1] == 'pinhole':

        # pinhole projects ahead only
        p[:,2] = abs(p[:,2])

    if sys.argv[1] == 'pinhole':
        lensmodel      = 'LENSMODEL_PINHOLE'
        func_project   = mrcal.project_pinhole
        func_unproject = mrcal.unproject_pinhole
        name           = 'pinhole'

        q_projected_ref = np.array([[ -9700.,  4400.],
                                    [ -13300., 4600.],
                                    [ -12700., 1000.]])

    elif sys.argv[1] == 'lonlat':
        lensmodel      = 'LENSMODEL_LONLAT'
        func_project   = mrcal.project_lonlat
        func_unproject = mrcal.unproject_lonlat
        name           = 'lonlat'

        q_projected_ref = np.array([[ -9700.99404253,   4392.88198287],
                                    [-16925.83416075,   4398.25498944],
                                    [-17226.33265541,   2320.61601685]])

    elif sys.argv[1] == 'latlon':

        lensmodel      = 'LENSMODEL_LATLON'
        func_project   = mrcal.project_latlon
        func_unproject = mrcal.unproject_latlon
        name           = 'latlon'

        q_projected_ref = np.array([[ -9706.7632608 ,   4394.7911197 ],
                                    [-12434.4909092 ,   9700.27171822],
                                    [-11389.09468198,   -317.59786068]])

elif sys.argv[1] == 'stereographic':
    lensmodel      = 'LENSMODEL_STEREOGRAPHIC'
    func_project   = mrcal.project_stereographic
    func_unproject = mrcal.unproject_stereographic
    name           = 'stereographic'

    fx,fy,cx,cy = 1512., 1112, 500., 333.

    # a few points, some wide, some not. Some behind the camera
    p = np.array(((1.0, 2.0, 10.0),
                  (-1.1, 0.3, -1.0),
                  (-0.9, -1.5, -1.0)))

    q_projected_ref = np.array([[  649.35582325,   552.6874014 ],
                                [-5939.33490417,  1624.58376866],
                                [-2181.52681292, -2953.8803086 ]])

    unproject_is_normalized = False

else:
    raise Exception("Unknown projection type. Currently I support 'lonlat','stereographic'")





intrinsics = (lensmodel, np.array((fx,fy,cx,cy)))

q_projected = func_project(p, intrinsics[1])
testutils.confirm_equal(q_projected,
                        q_projected_ref,
                        msg = f"project_{name}()",
                        worstcase = True,
                        relative  = True)

testutils.confirm_equal(mrcal.project(p, *intrinsics),
                        q_projected,
                        msg = f"project({name}) returns the same as project_{name}()",
                        worstcase = True,
                        relative  = True)

v_unprojected = func_unproject(q_projected, intrinsics[1])
if unproject_is_normalized:
    testutils.confirm_equal( nps.mag(v_unprojected),
                             1.,
                             msg = f"unproject_{name}() returns normalized vectors",
                             worstcase = True,
                             relative  = True)
    testutils.confirm_equal( v_unprojected,
                             p / nps.dummy(nps.mag(p), axis=-1),
                             msg = f"unproject_{name}()",
                             worstcase = True,
                             relative  = True)
else:
    cos = nps.inner(v_unprojected, p) / (nps.mag(p)*nps.mag(v_unprojected))
    cos = np.clip(cos, -1, 1)
    testutils.confirm_equal( np.arccos(cos),
                             np.zeros((p.shape[0],), dtype=float),
                             msg = f"unproject_{name}()",
                             worstcase = True)

    # Not normalized by default. Make sure that if I ask for it to be
    # normalized, that it is
    testutils.confirm_equal( nps.mag( mrcal.unproject(q_projected, *intrinsics, normalize = True) ),
                             1.,
                             msg = f"unproject({name},normalize = True) returns normalized vectors",
                             worstcase = True,
                             relative  = True)
    testutils.confirm_equal( nps.mag( mrcal.unproject(q_projected, *intrinsics, normalize = True, get_gradients = True)[0] ),
                             1.,
                             msg = f"unproject({name},normalize = True, get_gradients=True) returns normalized vectors",
                             worstcase = True,
                             relative  = True)



testutils.confirm_equal( mrcal.unproject(q_projected, *intrinsics),
                         v_unprojected,
                         msg = f"unproject({name}) returns the same as unproject_{name}()",
                         worstcase = True,
                         relative  = True)

testutils.confirm_equal( mrcal.project(mrcal.unproject(q_projected, *intrinsics),*intrinsics),
                         q_projected,
                         msg = f"project(unproject()) is an identity",
                         worstcase = True,
                         relative  = True)

testutils.confirm_equal( func_project(func_unproject(q_projected,intrinsics[1]),intrinsics[1]),
                         q_projected,
                         msg = f"project_{name}(unproject_{name}()) is an identity",
                         worstcase = True,
                         relative  = True)


# Now gradients for project()
ipt = 1
_,dq_dp_reported = func_project(p[ipt], intrinsics[1], get_gradients=True)
dq_dp_observed = grad(lambda p: func_project(p, intrinsics[1]),
                      p[ipt])
testutils.confirm_equal(dq_dp_reported,
                        dq_dp_observed,
                        msg = f"project_{name}() dq/dp",
                        worstcase = True,
                        relative  = True)
_,dq_dp_reported,dq_di_reported = mrcal.project(p[ipt], *intrinsics, get_gradients=True)
dq_dp_observed = grad(lambda p: mrcal.project(p, *intrinsics),
                      p[ipt])
dq_di_observed = grad(lambda intrinsics_data: mrcal.project(p[ipt], intrinsics[0],intrinsics_data),
                      intrinsics[1])
testutils.confirm_equal(dq_dp_reported,
                        dq_dp_observed,
                        msg = f"project({name}) dq/dp",
                        worstcase = True,
                        relative  = True)
testutils.confirm_equal(dq_di_reported,
                        dq_di_observed,
                        msg = f"project({name}) dq/di",
                        worstcase = True,
                        relative  = True,
                        eps = 1e-5)

# Now gradients for unproject()
ipt = 1
_,dv_dq_reported = func_unproject(q_projected[ipt], intrinsics[1], get_gradients=True)
dv_dq_observed = grad(lambda q: func_unproject(q, intrinsics[1]),
                      q_projected[ipt])
testutils.confirm_equal(dv_dq_reported,
                        dv_dq_observed,
                        msg = f"unproject_{name}() dv/dq",
                        worstcase = True,
                        relative  = True,
                        eps       = 2e-6)


for normalize in (False, True):

    v_unprojected,dv_dq_reported,dv_di_reported = \
        mrcal.unproject(q_projected[ipt], *intrinsics,
                        get_gradients = True,
                        normalize = normalize)
    dv_dq_observed = grad(lambda q: mrcal.unproject(q, *intrinsics, normalize=normalize),
                          q_projected[ipt])
    dv_di_observed = grad(lambda intrinsics_data: mrcal.unproject(q_projected[ipt], intrinsics[0],intrinsics_data, normalize=normalize),
                          intrinsics[1])
    testutils.confirm_equal(dv_dq_reported,
                            dv_dq_observed,
                            msg = f"unproject({name}, normalize={normalize}) dv/dq",
                            worstcase = True,
                            relative  = True,
                            eps = 1e-5)
    testutils.confirm_equal(dv_di_reported,
                            dv_di_observed,
                            msg = f"unproject({name}, normalize={normalize}) dv/di",
                            worstcase = True,
                            relative  = True,
                            eps = 1e-5)

    v_unprojected_inplace  = v_unprojected.copy() *0
    dv_dq_reported_inplace = dv_dq_reported.copy()*0
    dv_di_reported_inplace = dv_di_reported.copy()*0

    mrcal.unproject(q_projected[ipt], *intrinsics, get_gradients=True, normalize=normalize,
                    out = [v_unprojected_inplace,dv_dq_reported_inplace,dv_di_reported_inplace])
    testutils.confirm_equal(v_unprojected_inplace,
                            v_unprojected,
                            msg = f"unproject({name}, normalize={normalize}) works in-place: v_unprojected",
                            worstcase = True,
                            relative  = True)
    testutils.confirm_equal(dv_dq_reported_inplace,
                            dv_dq_reported,
                            msg = f"unproject({name}, normalize={normalize}) works in-place: dv_dq",
                            worstcase = True,
                            relative  = True)
    testutils.confirm_equal(dv_di_reported_inplace,
                            dv_di_reported,
                            msg = f"unproject({name}, normalize={normalize}) works in-place: dv_di",
                            worstcase = True,
                            relative  = True)

testutils.finish()
