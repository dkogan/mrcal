#!/usr/bin/python3

r'''Basic camera-calibration test

I observe, with noise, a number of chessboards from various angles with several
cameras. And I make sure that I can more or less compute the camera intrinsics
and extrinsics

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

# I want the RNG to be deterministic
np.random.seed(0)

############# Set up my world, and compute all the perfect positions, pixel
############# observations of everything
models = ( mrcal.cameramodel(f"{testdir}/data/cam0.opencv8.cameramodel"),
           mrcal.cameramodel(f"{testdir}/data/cam0.opencv8.cameramodel"),
           mrcal.cameramodel(f"{testdir}/data/cam1.opencv8.cameramodel"),
           mrcal.cameramodel(f"{testdir}/data/cam1.opencv8.cameramodel") )

imagersizes = nps.cat( *[m.imagersize() for m in models] )
lensmodel   = models[0].intrinsics()[0]
# I have opencv8 models, but let me truncate to opencv4 models to keep this
# simple and fast
lensmodel = 'LENSMODEL_OPENCV4'
for m in models:
    m.intrinsics( imagersize=imagersizes[0],
                  intrinsics = (lensmodel, m.intrinsics()[1][:8]))

Ncameras = len(models)
Nframes  = 50

models[0].extrinsics_rt_fromref(np.zeros((6,), dtype=float))
models[1].extrinsics_rt_fromref(np.array((0.08,0.2,0.02, 1., 0.9,0.1)))
models[2].extrinsics_rt_fromref(np.array((0.01,0.07,0.2, 2.1,0.4,0.2)))
models[3].extrinsics_rt_fromref(np.array((-0.1,0.08,0.08, 4.4,0.2,0.1)))


pixel_uncertainty  = 1.5
object_spacing     = 0.1
object_width_n     = 10
object_height_n    = 9
calobject_warp_ref = np.array((0.002, -0.005))

# shapes (Nframes, Ncameras, Nh, Nw, 2),
#        (Nframes, 4,3)
p,Rt_cam0_boardref = \
    mrcal.make_synthetic_board_observations(models,
                                            object_width_n, object_height_n, object_spacing,
                                            calobject_warp_ref,
                                            np.array((0,   0,   5.0,  0.,  0.,  0.)),
                                            np.array((1.5, 1.5, 1.0,  40., 30., 30.)),
                                            Nframes)

p_noise = np.random.randn(*p.shape) * pixel_uncertainty
p_noisy = p + p_noise

# I want observations of shape (Nframes*Ncameras, Nh, Nw, 3) where each row is
# (x,y,weight)
observations = nps.clump(p_noisy, n=2)
observations = nps.glue(observations,
                        np.ones(observations[...,(0,)].shape, dtype=float),
                        axis=-1)

# Dense observations. All the cameras see all the boards
indices_frame_camera = np.zeros( (Nframes*Ncameras, 2), dtype=np.int32)
indices_frame = indices_frame_camera[:,0].reshape(Nframes,Ncameras)
indices_frame.setfield(nps.outer(np.arange(Nframes, dtype=np.int32),
                                 np.ones((Ncameras,), dtype=np.int32)),
                       dtype = np.int32)
indices_camera = indices_frame_camera[:,1].reshape(Nframes,Ncameras)
indices_camera.setfield(nps.outer(np.ones((Nframes,), dtype=np.int32),
                                 np.arange(Ncameras, dtype=np.int32)),
                       dtype = np.int32)

indices_frame_camintrinsics_camextrinsics = \
    nps.glue(indices_frame_camera,
             indices_frame_camera[:,(1,)]-1,
             axis=-1)

intrinsics_data,extrinsics,frames = \
    mrcal.make_seed_no_distortion(imagersizes          = imagersizes,
                                  focal_estimate       = 1500,
                                  Ncameras             = Ncameras,
                                  indices_frame_camera = indices_frame_camera,
                                  observations         = observations,
                                  object_spacing       = object_spacing,
                                  object_width_n       = object_width_n,
                                  object_height_n      = object_height_n)

# I have a pinhole intrinsics estimate. Mount it into a full distortiony model,
# seeded with random numbers
intrinsics = nps.cat(*[m.intrinsics()[1] for m in models])
intrinsics[:,:4] = intrinsics_data
intrinsics[:,4:] = np.random.random( (Ncameras, intrinsics.shape[1]-4) ) * 1e-6

args = ( intrinsics,
         extrinsics,
         frames, None,
         observations, indices_frame_camintrinsics_camextrinsics,
         None, None,
         lensmodel)

kwargs_callback = \
    dict( imagersizes                       = imagersizes,
          calibration_object_spacing        = object_spacing,
          calibration_object_width_n        = object_width_n,
          calibration_object_height_n       = object_height_n,
          skip_regularization               = False,
          verbose                           = False )

kwargs_optimize = dict(kwargs_callback,
                       observed_pixel_uncertainty = pixel_uncertainty,
                       skip_outlier_rejection     = True)

# Simpler pre-solves
stats = mrcal.optimize( *args, **kwargs_optimize,
                        do_optimize_frames                = True,
                        do_optimize_intrinsic_core        = False,
                        do_optimize_intrinsic_distortions = False,
                        do_optimize_extrinsics            = True)
stats = mrcal.optimize( *args, **kwargs_optimize,
                        do_optimize_frames                = True,
                        do_optimize_intrinsic_core        = True,
                        do_optimize_intrinsic_distortions = False,
                        do_optimize_extrinsics            = True)

# Complete final solve
calobject_warp = np.array((0.001, 0.001))
solver_context = mrcal.SolverContext()
stats = mrcal.optimize( *args, **kwargs_optimize,
                        do_optimize_frames                = True,
                        do_optimize_intrinsic_core        = True,
                        do_optimize_intrinsic_distortions = True,
                        do_optimize_extrinsics            = True,

                        do_optimize_calobject_warp        = True,
                        calobject_warp                    = calobject_warp,
                        get_covariances                   = True,
                        solver_context                    = solver_context)

covariance_intrinsics = stats.get('covariance_intrinsics')
covariance_extrinsics = stats.get('covariance_extrinsics')

models_solved = \
    [ mrcal.cameramodel( imagersize            = imagersizes[0],
                         intrinsics            = (lensmodel, intrinsics[i,:]),
                         covariance_intrinsics = covariance_intrinsics[i],
                         observed_pixel_uncertainty = pixel_uncertainty) \
      for i in range(Ncameras)]
for i in range(1,Ncameras):
    models_solved[i].extrinsics_rt_fromref( extrinsics[i-1,:] )

testutils.confirm_equal(stats['rms_reproj_error__pixels'], 0,
                        eps = 2.5,
                        msg = "Converged to a low RMS error")

testutils.confirm_equal( calobject_warp,
                         calobject_warp_ref,
                         eps = 2e-3,
                         msg = "Recovered the calibration object shape" )

testutils.confirm_equal( np.std(stats['x']),
                         pixel_uncertainty,
                         eps = pixel_uncertainty*0.1,
                         msg = "Residual have the expected distribution" )

# Checking the extrinsics. These aren't defined absolutely: each solve is free
# to put the observed frames anywhere it likes. The intrinsics-diff code
# computes a compensating rotation to address this. Here I simply look at the
# relative transformations between cameras, which would cancel out any extra
# rotations. AND since camera0 is fixed at the identity transformation, I can
# simply look at each extrinsics transformation.
for icam in range(1,len(models)):

    Rt_extrinsics_err = \
        mrcal.compose_Rt( models_solved[icam].extrinsics_Rt_fromref(),
                          models       [icam].extrinsics_Rt_toref() )

    testutils.confirm_equal( nps.mag(Rt_extrinsics_err[3,:]),
                             0.0,
                             eps = 0.05,
                             msg = f"Recovered extrinsic translation for camera {icam}")

    testutils.confirm_equal( (np.trace(Rt_extrinsics_err[:3,:]) - 1) / 2.,
                             1.0,
                             eps = np.cos(1. * np.pi/180.0), # 1 deg
                             msg = f"Recovered extrinsic rotation for camera {icam}")

Rt_frame_err = \
    mrcal.compose_Rt( mrcal.Rt_from_rt(frames),
                      mrcal.invert_Rt(Rt_cam0_boardref) )

testutils.confirm_equal( np.max(nps.mag(Rt_frame_err[..., 3,:])),
                         0.0,
                         eps = 0.08,
                         msg = "Recovered frame translation")
testutils.confirm_equal( np.min( (nps.trace(Rt_frame_err[..., :3,:]) - 1)/2. ),
                         1.0,
                         eps = np.cos(1. * np.pi/180.0), # 1 deg
                         msg = "Recovered frame rotation")






import gnuplotlib as gp


r'''Test computeUncertaintyMatrices() and compute_intrinsics_uncertainty()

The uncertainty computation is documented in the docstring for
compute_intrinsics_uncertainty(). The math and the implementation are
tricky, so this function exists to empirically confirm that the thing being
computed is correct, both in implementation and intent.

Call this after we just solved a full optimization problem

This function checks some linearization assumptions. It uses a densified
jacobian to do the math, so this is very inefficient, and only viable for
small problems.

'''


def callback(intrinsics_data):
    x,Joptimizer = \
        mrcal.optimizerCallback(intrinsics_data,extrinsics,frames, None,
                                observations, indices_frame_camintrinsics_camextrinsics,
                                None, None,
                                lensmodel,
                                **kwargs_callback,
                                do_optimize_calobject_warp        = True,
                                calobject_warp                    = calobject_warp,
                                do_optimize_intrinsic_core        = True,
                                do_optimize_intrinsic_distortions = True,
                                do_optimize_extrinsics            = True)

    Joptimizer = Joptimizer.toarray()
    J = Joptimizer.copy()
    solver_context.pack(J)
    return x,J,Joptimizer


# State and measurements at the optimal operating point
p0 = solver_context.p().copy()
x0,J0,Joptimizer0 = callback(intrinsics)

###########################################################################
# First a very basic gradient check. Looking at anarbitrary state variable. The
# test-gradients tool does this much more thoroughly
icam        = 1
Nintrinsics = mrcal.getNlensParams(lensmodel)
delta       = np.random.randn(Nintrinsics) * 1e-6
ivar        = solver_context.state_index_intrinsics(icam)
J0_slice    = J0[:,ivar:ivar+Nintrinsics]
intrinsics_perturbed = intrinsics.copy()
intrinsics_perturbed[icam] += delta
x1                   = callback(intrinsics_perturbed)[0]
dx                   = x1 - x0
dx_predicted         = nps.inner(J0_slice, delta)
testutils.confirm_equal( dx_predicted, dx,
                         eps = 1e-3,
                         worstcase = True,
                         relative = True,
                         msg = "Trivial, sanity-checking gradient check")

###########################################################################
# We're supposed to be at the optimum. E = norm2(x) ~ norm2(x0 + J dp) =
# norm2(x0) + 2 dpt Jt x0 + norm2(J dp). At the optimum Jt x0 = 0 -> E =
# norm2(x0) + norm2(J dp). dE = norm2(J dp) = norm2(dx_predicted)
x_predicted  = x0 + dx_predicted
dE           = nps.norm2(x1) - nps.norm2(x0)
dE_predicted = nps.norm2(dx_predicted)
testutils.confirm_equal( dE_predicted, dE,
                         eps = 1e-3,
                         relative = True,
                         msg = "diff(E) predicted")

###########################################################################
# Fine. Let's make sure the noise propagation works as it should. First off, is
# the implementation correct? Derivation in compute_intrinsics_uncertainty()
# says intrinsics covariance:
#
#   Var(intrinsics) = (inv(JtJ)[intrinsicsrows] Jobservationst)
#                     (inv(JtJ)[intrinsicsrows] Jobservationst)t
#                     s^2
#
# Let's make sure the computation was done correctly
Nobservations_board = indices_frame_camera.shape[0]
Nmeasurements_board = Nobservations_board * object_height_n * object_width_n * 2

invJtJ = np.linalg.inv(nps.matmult(nps.transpose(J0), J0))
J0observations = J0[:Nmeasurements_board,:]

for icam in range(Ncameras):

    covariance_intrinsics_predicted = \
        (pixel_uncertainty*pixel_uncertainty) * \
        nps.matmult( invJtJ[Nintrinsics*icam:Nintrinsics*(icam+1),:],
                     nps.transpose(J0observations),
                     J0observations,
                     invJtJ[:,Nintrinsics*icam:Nintrinsics*(icam+1)])

    testutils.confirm_equal( covariance_intrinsics_predicted,
                             covariance_intrinsics[icam, ...],
                             relative = True,
                             eps = 1e-3,
                             msg = f"covariance_intrinsics predicted for camera {icam}")

testutils.finish()

err_inv_jtj_extrinsics = \
    np.linalg.norm(covariance_extrinsics/(pixel_uncertainty*pixel_uncertainty) -
                   nps.matmult( invJtJ[   Nintrinsics*Ncameras:Nintrinsics*Ncameras + 6*(Ncameras-1), :],
                                nps.transpose(J0obs),
                                J0obs,
                                invJtJ[:, Nintrinsics*Ncameras:Nintrinsics*Ncameras + 6*(Ncameras-1)   ]))

print("invJtJ_extrinsics error (should be 0): {}".format(err_inv_jtj_extrinsics))

###########################################################################
# I confirmed that I'm computing what I thought I'm computing
# So if I perturb my input observation vector qref by dqref, the resulting
# effect on the parameters is dp = M dqref
#
#   where M = inv(JtJ) Jobservationst W
def make_perturbation(stdev):
    return np.random.randn(Nmeasurements_board) * stdev

def optimize_perturbed_observations(dqref):

    intrinsics_data_resolved = intrinsics.copy()
    extrinsics_resolved      = extrinsics.copy()
    frames_resolved          = frames.copy()
    calobject_warp_resolved  = calobject_warp.copy() if calobject_warp is not None else calobject_warp

    observations_perturbed_noweight = (observations[...,:2].ravel() + dqref).reshape(observations.shape[:-1] + (2,))
    observations_perturbed = nps.glue( observations_perturbed_noweight,
                                       observations[..., (2,)],
                                       axis = -1 )

    stats1 = mrcal.optimize(intrinsics_data_resolved,
                            extrinsics_resolved,
                            frames_resolved,
                            None,
                            observations_perturbed, indices_frame_camintrinsics_camextrinsics,
                            None, None,
                            lensmodel,
                            do_optimize_calobject_warp        = True,
                            calobject_warp                    = calobject_warp_resolved,
                            imagersizes                       = imagersizes,
                            do_optimize_intrinsic_core        = True,
                            do_optimize_intrinsic_distortions = True,
                            do_optimize_extrinsics            = True,
                            calibration_object_spacing        = object_spacing,
                            calibration_object_width_n        = object_width_n,
                            calibration_object_height_n       = object_height_n,
                            skip_outlier_rejection            = True, # use outliers in outlier_indices
                            skip_regularization               = False,
                            outlier_indices                   = None,
                            get_covariances                   = False,
                            verbose                           = False,
                            solver_context                    = solver_context)
    return solver_context.p().copy()

dqref = make_perturbation(1e-4)

p1 = optimize_perturbed_observations(dqref)
dp = p1-p0

# Slow! Inefficient!
w = observations[..., np.array((2,2))].ravel()
M = np.linalg.solve( nps.matmult(nps.transpose(Joptimizer0),Joptimizer0),
                     nps.transpose(Joptimizer0[:Nmeasurements_board, :]) ) * w
dp_predicted = nps.matmult( dqref, nps.transpose(M)).ravel()

print("Popping up a plot of expected and observed dp. This should match well. Intrinsics may be a bit off")
plot_dp = gp.gnuplotlib(title='Parameter shift due to input observations shift')
plot_dp.plot(nps.cat(dp, dp_predicted), legend=np.array(('dp_reoptimized', 'dp_reoptimized_predicted')))


###########################################################################
# Now I do a bigger, statistical thing. I compute many random perturbations
# of the input, reoptimize for each, and look at how that affects a
# particular projection point. I have a prediction on the variance that I
# can check

# where I'm evaluating the projection. I look at 1/3 (w,h). I'd like to be
# in the center-ish, but not AT the center
v0 = mrcal.unproject(imagersizes[0] / 3,
                     lensmodel, intrinsics[0])

stdev = 1e-4
q_sampled = np.zeros((100,Ncameras,2), dtype=float)
Nintrinsics = intrinsics.shape[-1]
for i in range(len(q_sampled)):
    p = optimize_perturbed_observations( make_perturbation(stdev) )
    solver_context.unpack(p)

    for icam in range(Ncameras):
        q_sampled[i,icam] = \
            mrcal.project(v0, lensmodel,
                          p[Nintrinsics*icam:Nintrinsics*(icam+1)])

plot_distribution = [None] * Ncameras
for icam in range(Ncameras):
    dq = q_sampled[:, icam, ...] - mrcal.project(v0, lensmodel, intrinsics[icam])

    dq_mean0 = dq - np.mean(dq,axis=-2)
    C = nps.matmult(nps.transpose(dq_mean0), dq_mean0)
    l,V = np.linalg.eig(C)
    l0 = l[0]
    l1 = l[1]
    V0 = V[:,0]
    V1 = V[:,1]
    if l0 < l1:
        l1,l0 = l0,l1
        V1,V0 = V0,V1

    Var = np.mean( nps.outer(dq_mean0,dq_mean0), axis=-3 )
    err_1sigma_observed_major       = np.sqrt(l0/q_sampled.shape[0])
    err_1sigma_observed_minor       = np.sqrt(l1/q_sampled.shape[0])
    err_1sigma_observed_anisotropic = np.diag(np.sqrt( Var ))
    err_1sigma_observed_isotropic   = np.sqrt( np.trace(Var) / 2. )

    err_1sigma_predicted_isotropic = \
        mrcal.compute_intrinsics_uncertainty(models_solved[icam], v0,
                                             focus_radius = 0) \
        / models_solved[icam].observed_pixel_uncertainty() * stdev


    print("Noisy input, recalibration produced RMS reprojection error {:.2g} pixels. Predicted {:.2g} pixels". \
          format(err_1sigma_observed_isotropic,
                 err_1sigma_predicted_isotropic))
    plot_distribution[icam] = gp.gnuplotlib(square=1, title='Uncertainty reprojection distribution for camera {}'.format(icam))
    plot_distribution[icam]. \
        plot( (dq[:,0], dq[:,1], dict(_with='points pt 7 ps 2')),
              (0,0, 2*err_1sigma_observed_major, 2*err_1sigma_observed_minor, 180./np.pi*np.arctan2(V0[1],V0[0]),
               dict(_with='ellipses lw 2', tuplesize=5, legend='Observed 1-sigma, full covariance')),
              (0,0, 2*err_1sigma_observed_anisotropic[0], 2*err_1sigma_observed_anisotropic[1],
               dict(_with='ellipses lw 2', tuplesize=4, legend='Observed 1-sigma, independent x,y')),
              (0,0, err_1sigma_observed_isotropic,
               dict(_with='circles lw 2', tuplesize=3, legend='Observed 1-sigma; isotropic')),
              (0,0, err_1sigma_predicted_isotropic,
               dict(_with='circles lw 2', tuplesize=3, legend='Predicted 1-sigma; isotropic')))

import IPython
IPython.embed()
sys.exit()
