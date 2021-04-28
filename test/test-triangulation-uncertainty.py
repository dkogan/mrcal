#!/usr/bin/python3

import sys
import numpy as np
import numpysane as nps
import gnuplotlib as gp

import os
directory = '/home/dima/projects/mrcal'

sys.path[:0] = directory,
import mrcal

models = ( f'{directory}/doc/data/figueroa-overpass-looking-S/splined-{i}.cameramodel' for i in (0,1) )

# 1km straight ahead

# shape (Npoints,3)
pref = np.array(((-100., 0., 10000.),
                 ( 100., 0., 10000.)))

observed_pixel_uncertainty = 0.2

binwidth = 0.4*1000

models = [ mrcal.cameramodel(m) for m in models ]

Rt10 = mrcal.compose_Rt( models[1].extrinsics_Rt_fromref(),
                         models[0].extrinsics_Rt_toref() )
Rt01 = mrcal.invert_Rt(Rt10)

# shape (Ncameras,Npoints,3)
pref_local = nps.cat( pref,
                      mrcal.transform_point_Rt(Rt10, pref) )

# Pixel coords at the perfect intersection
# shape (Npoints,Ncameras,2)
qref = nps.xchg( np.array([ mrcal.project(pref_local[i], *models[i].intrinsics()) \
                            for i in range(len(models))]),
                 0,1)

# I triangulate the perfect pixel coords, keeping track of the uncertainties
# vlocal0, vlocal1 have shape (Npoints,3)
vlocal0, dvlocal0_dq0, dvlocal0_dintrinsics0 = \
    mrcal.unproject(qref[:,0], *models[0].intrinsics(), get_gradients = True)
vlocal1, dvlocal1_dq1, dvlocal1_dintrinsics1 = \
    mrcal.unproject(qref[:,1], *models[1].intrinsics(), get_gradients = True)

v0 = vlocal0
v1, dv1_dR01, dv1_dvlocal1 = \
    mrcal.rotate_point_R(Rt01[:3,:], vlocal1, get_gradients = True)

# p has shape (Npoints,3)
p, dp_dv0, dp_dv1, dp_dt01 = \
    mrcal.triangulate_leecivera_mid2(v0, v1, Rt01[3,:],
                                     get_gradients = True)


if np.max(nps.mag(p-pref)) > 1e-3:
    raise Exception(f"Triangulation didn't return the perfect point. Returned {p}. Should have returned {pref}")

diffp          = p[0] - p[1]
ddiffp_dp01 = nps.glue(  np.eye(3, dtype=float),
                        -np.eye(3, dtype=float),
                         axis = -1)
distancep      = nps.mag(diffp)
ddistancep_dp0 =  diffp / distancep
ddistancep_dp1 = -diffp / distancep

# I now have all the gradients and all the internal variances, so I can
# propagate everything. The big vector I want to propagate is
#
# - q: pixel noise
# - intrinsics01
# - extrinsics


# For now let's just do q

# I have 4 pixel observations. Let's say qij is the pixel observation of
# point i in camera j



def compute_var_p01(var_x):
    # shape (Npoints, 3, Npoints,Ncameras,2)
    # The trailing (Npoints,Ncameras,2) indexes the x
    dp_dx = np.zeros((2,3,2,2,2), dtype=float)

    # each has shape (3,2)
    dp0_dq00 = dp_dx[0,:,0,0,:]
    dp0_dq01 = dp_dx[0,:,0,1,:]
    dp1_dq10 = dp_dx[1,:,1,0,:]
    dp1_dq11 = dp_dx[1,:,1,1,:]

    # each has shape (Npoints,3,2)
    dv0_dq0 = dvlocal0_dq0
    dv1_dq1 = nps.matmult2(dv1_dvlocal1, dvlocal1_dq1)

    nps.matmult2(dp_dv0[0], dv0_dq0[0], out=dp0_dq00)
    nps.matmult2(dp_dv1[0], dv1_dq1[0], out=dp0_dq01)
    nps.matmult2(dp_dv0[1], dv0_dq0[1], out=dp1_dq10)
    nps.matmult2(dp_dv1[1], dv1_dq1[1], out=dp1_dq11)

    # shape (6,8)
    dpflat_dx = nps.clump(nps.clump(dp_dx, n=-3), n=2)
    return nps.matmult( dpflat_dx, var_x, nps.transpose(dpflat_dx))


def compute_var_range(p, var_p):
    # r = np.mag(p)
    # dr_dp = p/r
    # var(r) = dr_dp var_p dr_dpT
    #        = p var_p pT / norm2(p)
    return nps.matmult(p,var_p,nps.transpose(p))[0] / nps.norm2(p)


def compute_var_distancep_direct(var_x):
    # Distance between 2 3D points

    # I have 4 pixel observations. Let's say qij is the pixel observation of
    # point i in camera j

    ddistancep_dx = np.zeros((1,8), dtype=float)
    ddistancep_dq00 = ddistancep_dx[:,0:2]
    ddistancep_dq01 = ddistancep_dx[:,2:4]
    ddistancep_dq10 = ddistancep_dx[:,4:6]
    ddistancep_dq11 = ddistancep_dx[:,6:8]

    dv00_dq00,dv10_dq10 = dvlocal0_dq0
    dv01_dq01,dv11_dq11 = nps.matmult2(dv1_dvlocal1, dvlocal1_dq1)

    dp0_dv00,dp1_dv10 = dp_dv0
    dp0_dv01,dp1_dv11 = dp_dv1

    ddistancep_dv00 = nps.matmult( ddistancep_dp0, dp0_dv00 )
    ddistancep_dv01 = nps.matmult( ddistancep_dp0, dp0_dv01 )
    ddistancep_dv10 = nps.matmult( ddistancep_dp1, dp1_dv10 )
    ddistancep_dv11 = nps.matmult( ddistancep_dp1, dp1_dv11 )

    nps.matmult2(ddistancep_dv00, dv00_dq00, out=ddistancep_dq00)
    nps.matmult2(ddistancep_dv01, dv01_dq01, out=ddistancep_dq01)
    nps.matmult2(ddistancep_dv10, dv10_dq10, out=ddistancep_dq10)
    nps.matmult2(ddistancep_dv11, dv11_dq11, out=ddistancep_dq11)

    return nps.matmult( ddistancep_dx, var_x, nps.transpose(ddistancep_dx)).ravel()[0]


def compute_var_distancep_from_var_p(var_p):

    ddistancep_dp = nps.glue( ddistancep_dp0, ddistancep_dp1, axis=-1)
    return nps.matmult( ddistancep_dp, var_p, nps.transpose(ddistancep_dp)).ravel()[0]



# Let's say the pixel observations are all independent; this is not
# obviously true. Order in x: q00 q01 q10 q11
Nx = 8
var_x_independent = np.diagflat( (observed_pixel_uncertainty*observed_pixel_uncertainty,) * Nx )

var_x = var_x_independent.copy()
var_x[0,2] = var_x[2,0] = observed_pixel_uncertainty*observed_pixel_uncertainty*0.9
var_x[1,3] = var_x[3,1] = observed_pixel_uncertainty*observed_pixel_uncertainty*0.9
var_x[4,6] = var_x[6,4] = observed_pixel_uncertainty*observed_pixel_uncertainty*0.9
var_x[5,7] = var_x[7,5] = observed_pixel_uncertainty*observed_pixel_uncertainty*0.9

var_distancep             = compute_var_distancep_direct(var_x)
var_distancep_independent = compute_var_distancep_direct(var_x_independent)
var_p                     = compute_var_p01(var_x)
var_p_independent         = compute_var_p01(var_x_independent)

var_r0             = compute_var_range(pref[0], var_p[:3,:3])
var_r0_independent = compute_var_range(pref[0], var_p_independent[:3,:3])

var_diffp = nps.matmult(ddiffp_dp01, var_p, nps.transpose(ddiffp_dp01))



if np.abs(compute_var_distancep_from_var_p(var_p) - var_distancep) > 1e-6:
    raise Exception("Var(distancep) should identical whether you compute it from Var(p) or not")



# Let's actually apply the noise to compute var(distancep) empirically to compare
# against the var(distancep) prediction I just computed
Nsamples = 10000
# shape (Nsamples,Npoints,Ncameras,2)
dq = \
    np.random.multivariate_normal( mean = np.zeros((Nx,),),
                                   cov  = var_x,
                                   size = Nsamples ).reshape(Nsamples,2,2,2)


vlocal0 = mrcal.unproject(qref[:,0] + dq[:,:,0,:], *models[0].intrinsics())
vlocal1 = mrcal.unproject(qref[:,1] + dq[:,:,1,:], *models[1].intrinsics())
v0      = vlocal0
v1      = mrcal.rotate_point_R(Rt01[:3,:], vlocal1)
p       = mrcal.triangulate_leecivera_mid2(v0, v1, Rt01[3,:])

distancep     = nps.mag(p[:,0,:] - p[:,1,:])
distancep_ref = nps.mag(pref[0] - pref[1])

r0            = nps.mag(p[:,0,:])
r0_ref        = nps.mag(pref[0])

equation_distancep_observed_gaussian = \
    mrcal.fitted_gaussian_equation(x        = distancep,
                                   binwidth = binwidth,
                                   legend   = "Idealized gaussian fit to data")
equation_distancep_predicted_gaussian = \
    mrcal.fitted_gaussian_equation(mean     = distancep_ref,
                                   sigma    = np.sqrt(var_distancep),
                                   N        = len(distancep),
                                   binwidth = binwidth,
                                   legend   = "Predicted")
equation_distancep_predicted_independent_gaussian = \
    mrcal.fitted_gaussian_equation(mean     = distancep_ref,
                                   sigma    = np.sqrt(var_distancep_independent),
                                   N        = len(distancep),
                                   binwidth = binwidth,
                                   legend   = "Predicted, assuming independent noise")

equation_r0_observed_gaussian = \
    mrcal.fitted_gaussian_equation(x        = r0,
                                   binwidth = binwidth,
                                   legend   = "Idealized gaussian fit to data")
equation_r0_predicted_gaussian = \
    mrcal.fitted_gaussian_equation(mean     = r0_ref,
                                   sigma    = np.sqrt(var_r0),
                                   N        = len(r0),
                                   binwidth = binwidth,
                                   legend   = "Predicted")
equation_r0_predicted_independent_gaussian = \
    mrcal.fitted_gaussian_equation(mean     = r0_ref,
                                   sigma    = np.sqrt(var_r0_independent),
                                   N        = len(r0),
                                   binwidth = binwidth,
                                   legend   = "Predicted, assuming independent noise")

gp.plot(distancep,
        histogram       = True,
        binwidth        = binwidth,
        equation_above  = (equation_distancep_predicted_independent_gaussian,
                           equation_distancep_predicted_gaussian,
                           equation_distancep_observed_gaussian),
        xlabel          = "Distance between points",
        ylabel          = "Frequency",
        title           = f"Triangulated distance between points: sensitivity to pixel noise. Predicted stdev: {np.sqrt(var_distancep):.0f}m",
        _set            = 'samples 1000',
        hardcopy = '/tmp/distance-between.pdf',
        wait=1)



sys.path[:0] = f'{directory}/test',
import test_calibration_helpers

# I look at the xz uncertainty because y is very low. I just look at p[0]
V = var_p[:3,:3]
if np.min((V[0,0],V[2,2]))/V[1,1] < 10:
    raise Exception("Assumption that var(y) << var(xz) is false. I want the worst-case ratio to be >10")

V = V[(0,2),:][:,(0,2)]

ellipse = \
    test_calibration_helpers.plot_arg_covariance_ellipse(pref[0,(0,2)], V,
                                                         'Observed point. Var(p_xz)')

gp.plot( ( nps.glue( np.zeros((2,),),
                     Rt01[3,(0,2)],
                     axis=-2),
           dict(legend    = 'cameras',
                _with     = 'points pt 8 ps 1',
                tuplesize = -2)),
         ellipse,
         square = True,
         xlabel= 'x (m)',
         ylabel= 'y (m)',
         title = 'Top-down view of the world',
         hardcopy = '/tmp/world.pdf')

gp.plot( ellipse,
         square = True,
         xlabel= 'x (m)',
         ylabel= 'y (m)',
         title = 'Top-down view of the world; ',
         hardcopy = '/tmp/uncertainty.pdf',
        )

gp.plot(r0,
        histogram       = True,
        binwidth        = binwidth,
        equation_above  = (equation_r0_predicted_independent_gaussian,
                           equation_r0_predicted_gaussian,
                           equation_r0_observed_gaussian),
        xlabel          = "Range to triangulated point",
        ylabel          = "Frequency",
        title           = f"Triangulated distance to the observation at 1600m: sensitivity to pixel noise. Predicted stdev: {np.sqrt(var_r0):.0f}m",
        hardcopy = '/tmp/range0.pdf',
        _set            = 'samples 1000',
        wait = True,
        )



