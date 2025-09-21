#!/usr/bin/env python3

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

from test_calibration_helpers import sample_dqref
import copy

# I want the RNG to be deterministic
np.random.seed(0)

############# Set up my world, and compute all the perfect positions, pixel
############# observations of everything
models_ref = ( mrcal.cameramodel(f"{testdir}/data/cam0.opencv8.cameramodel"),
               mrcal.cameramodel(f"{testdir}/data/cam0.opencv8.cameramodel"),
               mrcal.cameramodel(f"{testdir}/data/cam1.opencv8.cameramodel"),
               mrcal.cameramodel(f"{testdir}/data/cam1.opencv8.cameramodel") )

imagersizes = nps.cat( *[m.imagersize() for m in models_ref] )
lensmodel   = models_ref[0].intrinsics()[0]
# I have opencv8 models_ref, but let me truncate to opencv4 models_ref to keep this
# simple and fast
lensmodel = 'LENSMODEL_OPENCV4'
for m in models_ref:
    m.intrinsics( intrinsics = (lensmodel, m.intrinsics()[1][:8]))
Nintrinsics = mrcal.lensmodel_num_params(lensmodel)

Ncameras = len(models_ref)
Nframes  = 50

models_ref[0].rt_cam_ref(np.zeros((6,), dtype=float))
models_ref[1].rt_cam_ref(np.array((0.08,0.2,0.02, 1., 0.9,0.1)))
models_ref[2].rt_cam_ref(np.array((0.01,0.07,0.2, 2.1,0.4,0.2)))
models_ref[3].rt_cam_ref(np.array((-0.1,0.08,0.08, 4.4,0.2,0.1)))


pixel_uncertainty_stdev = 1.5
object_spacing          = 0.1
object_width_n          = 10
object_height_n         = 9
calobject_warp_ref      = np.array((0.002, -0.005))

# shapes (Nframes, Ncameras, Nh, Nw, 2),
#        (Nframes, 4,3)
q_ref,Rt_ref_board_ref = \
    mrcal.synthesize_board_observations(models_ref,
                                        object_width_n                  = object_width_n,
                                        object_height_n                 = object_height_n,
                                        object_spacing                  = object_spacing,
                                        calobject_warp                  = calobject_warp_ref,
                                        rt_ref_boardcenter              = np.array((0.,  0.,  0., -2,   0,  4.0)),
                                        rt_ref_boardcenter__noiseradius = np.array((np.pi/180.*30., np.pi/180.*30., np.pi/180.*20., 2.5, 2.5, 2.0)),
                                        Nframes                         = Nframes)
frames_ref = mrcal.rt_from_Rt(Rt_ref_board_ref)

############# I have perfect observations in q_ref. I corrupt them by noise
# weight has shape (Nframes, Ncameras, Nh, Nw),
weight01 = (np.random.rand(*q_ref.shape[:-1]) + 1.) / 2. # in [0,1]
weight0 = 0.2
weight1 = 1.0
weight = weight0 + (weight1-weight0)*weight01

# I want observations of shape (Nframes*Ncameras, Nh, Nw, 3) where each row is
# (x,y,weight)
observations_ref = nps.clump( nps.glue(q_ref,
                                       nps.dummy(weight,-1),
                                       axis=-1),
                              n=2)

q_noise,observations = sample_dqref(observations_ref,
                                    pixel_uncertainty_stdev,
                                    make_outliers = True)

# Now I make some of the observations bogus, and mark them as input outliers.
# The solve should be robust to that, but any code that uses the bogus data
# DESPITE it being marked as bogus will generate a test failure
#
# Let's pretend the center of the chessboard has an apriltag, so all those
# observations are bogus. I block out a 5x5 chunk in the center
i0 = object_height_n//2
j0 = object_width_n//2
observations[..., i0-2:i0+3,j0-2:j0+3, 2] = -1.    # weight<=0: outlier
observations[..., i0-2:i0+3,j0-2:j0+3,:2] = -100.0 # all the values are bogus

############# Now I pretend that the noisy observations are all I got, and I run
############# a calibration from those

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


intrinsics_data,rt_cam_ref,rt_ref_frame = \
    mrcal.seed_stereographic(imagersizes          = imagersizes,
                             focal_estimate       = 1500,
                             indices_frame_camera = indices_frame_camera,
                             observations         = observations,
                             object_spacing       = object_spacing)

# I have a stereographic intrinsics estimate. Mount it into a full distortiony
# model, seeded with random numbers
intrinsics = np.zeros((Ncameras,Nintrinsics), dtype=float)
intrinsics[:,:4] = intrinsics_data
intrinsics[:,4:] = np.random.random( (Ncameras, intrinsics.shape[1]-4) ) * 1e-6

optimization_inputs = \
    dict( intrinsics                                = intrinsics,
          rt_cam_ref                                = rt_cam_ref,
          rt_ref_frame                              = rt_ref_frame,
          points                                    = None,
          observations_board                        = observations,
          indices_frame_camintrinsics_camextrinsics = indices_frame_camintrinsics_camextrinsics,
          observations_point                        = None,
          indices_point_camintrinsics_camextrinsics = None,
          lensmodel                                 = lensmodel,
          calobject_warp                            = None,
          imagersizes                               = imagersizes,
          calibration_object_spacing                = object_spacing,
          verbose                                   = False,
          do_apply_regularization                   = True)

# Solve this thing incrementally
optimization_inputs['do_optimize_intrinsics_core']        = False
optimization_inputs['do_optimize_intrinsics_distortions'] = False
optimization_inputs['do_optimize_extrinsics']             = True
optimization_inputs['do_optimize_frames']                 = True
optimization_inputs['do_optimize_calobject_warp']         = False
mrcal.optimize(**optimization_inputs,
               do_apply_outlier_rejection = True)

optimization_inputs['do_optimize_intrinsics_core']        = True
optimization_inputs['do_optimize_intrinsics_distortions'] = False
optimization_inputs['do_optimize_extrinsics']             = True
optimization_inputs['do_optimize_frames']                 = True
optimization_inputs['do_optimize_calobject_warp']         = False
mrcal.optimize(**optimization_inputs,
               do_apply_outlier_rejection = True)

testutils.confirm_equal( mrcal.num_states(**optimization_inputs),
                         4*Ncameras + 6*(Ncameras-1) + 6*Nframes,
                         msg="num_states()")
testutils.confirm_equal( mrcal.num_states_intrinsics(**optimization_inputs),
                         4*Ncameras,
                         msg="num_states_intrinsics()")
testutils.confirm_equal( mrcal.num_intrinsics_optimization_params(**optimization_inputs),
                         4,
                         msg="num_intrinsics_optimization_params()")
testutils.confirm_equal( mrcal.num_states_extrinsics(**optimization_inputs),
                         6*(Ncameras-1),
                         msg="num_states_extrinsics()")
testutils.confirm_equal( mrcal.num_states_frames(**optimization_inputs),
                         6*Nframes,
                         msg="num_states_frames()")
testutils.confirm_equal( mrcal.num_states_points(**optimization_inputs),
                         0,
                         msg="num_states_points()")
testutils.confirm_equal( mrcal.num_states_calobject_warp(**optimization_inputs),
                         0,
                         msg="num_states_calobject_warp()")

testutils.confirm_equal( mrcal.num_measurements_boards(**optimization_inputs),
                         object_width_n*object_height_n*2*Nframes*Ncameras,
                         msg="num_measurements_boards()")
testutils.confirm_equal( mrcal.num_measurements_points(**optimization_inputs),
                         0,
                         msg="num_measurements_points()")
testutils.confirm_equal( mrcal.num_measurements_regularization(**optimization_inputs),
                         Ncameras * 2,
                         msg="num_measurements_regularization()")


optimization_inputs['do_optimize_intrinsics_core']        = True
optimization_inputs['do_optimize_intrinsics_distortions'] = True
optimization_inputs['do_optimize_extrinsics']             = True
optimization_inputs['do_optimize_frames']                 = True
optimization_inputs['do_optimize_calobject_warp']         = True

optimization_inputs['calobject_warp'] = np.array((0.001, 0.001))
stats = mrcal.optimize(**optimization_inputs,
                       do_apply_outlier_rejection = True)

rmserr = stats['rms_reproj_error__pixels']


testutils.confirm_equal( mrcal.state_index_intrinsics(2, **optimization_inputs),
                         8*2,
                         msg="state_index_intrinsics()")
testutils.confirm_equal( mrcal.state_index_extrinsics(2, **optimization_inputs),
                         8*Ncameras + 6*2,
                         msg="state_index_extrinsics()")
testutils.confirm_equal( mrcal.state_index_frames(2, **optimization_inputs),
                         8*Ncameras + 6*(Ncameras-1) + 6*2,
                         msg="state_index_frames()")
testutils.confirm_equal( mrcal.state_index_calobject_warp(**optimization_inputs),
                         8*Ncameras + 6*(Ncameras-1) + 6*Nframes,
                         msg="state_index_calobject_warp()")

testutils.confirm_equal( mrcal.measurement_index_boards(2, **optimization_inputs),
                         object_width_n*object_height_n*2* 2,
                         msg="measurement_index_boards()")
testutils.confirm_equal( mrcal.measurement_index_regularization(**optimization_inputs),
                         object_width_n*object_height_n*2*Nframes*Ncameras,
                         msg="measurement_index_regularization()")


############# Calibration computed. Now I see how well I did
models_solved = \
    [ mrcal.cameramodel( optimization_inputs = optimization_inputs,
                         icam_intrinsics     = i )
      for i in range(Ncameras)]

# if 0:
#     for i in range(1,Ncameras):
#         models_solved[i].write(f'/tmp/tst{i}.cameramodel')


testutils.confirm_equal(rmserr, 0,
                        eps = 2.5,
                        msg = "Converged to a low RMS error")

testutils.confirm_equal( optimization_inputs['calobject_warp'],
                         calobject_warp_ref,
                         eps = 2e-3,
                         msg = "Recovered the calibration object shape" )

testutils.confirm_equal( np.std( mrcal.measurements_board(optimization_inputs,
                                                          x = stats['x'])),
                         pixel_uncertainty_stdev,
                         eps = pixel_uncertainty_stdev*0.1,
                         msg = "Residual have the expected distribution" )

# Checking the extrinsics. These aren't defined absolutely: each solve is free
# to put the observed frames anywhere it likes. The projection-diff code
# computes a transformation to address this. Here I simply look at the relative
# transformations between cameras, which would cancel out any extra
# transformations, AND since camera0 is fixed at the identity transformation, I
# can simply look at each extrinsics transformation.
for icam in range(1,len(models_ref)):

    Rt_extrinsics_err = \
        mrcal.compose_Rt( models_solved[icam].Rt_cam_ref(),
                          models_ref   [icam].Rt_ref_cam() )

    testutils.confirm_equal( nps.mag(Rt_extrinsics_err[3,:]),
                             0.0,
                             eps = 0.05,
                             msg = f"Recovered extrinsic translation for camera {icam}")

    testutils.confirm_equal( (np.trace(Rt_extrinsics_err[:3,:]) - 1) / 2.,
                             1.0,
                             eps = np.cos(1. * np.pi/180.0), # 1 deg
                             msg = f"Recovered extrinsic rotation for camera {icam}")

Rt_frame_err = \
    mrcal.compose_Rt( mrcal.Rt_from_rt(optimization_inputs['rt_ref_frame']),
                      mrcal.invert_Rt(Rt_ref_board_ref) )

testutils.confirm_equal( np.max(nps.mag(Rt_frame_err[..., 3,:])),
                         0.0,
                         eps = 0.08,
                         msg = "Recovered frame translation")
testutils.confirm_equal( np.min( (nps.trace(Rt_frame_err[..., :3,:]) - 1)/2. ),
                         1.0,
                         eps = np.cos(1. * np.pi/180.0), # 1 deg
                         msg = "Recovered frame rotation")


# Checking the intrinsics. Each intrinsics vector encodes an implicit
# transformation. I compute and apply this transformation when making my
# intrinsics comparisons. I make sure that within some distance of the pixel
# center, the projections match up to within some number of pixels
Nw = 60
def projection_diff(models_ref, max_dist_from_center):
    lensmodels      = [model.intrinsics()[0] for model in models_ref]
    intrinsics_data = [model.intrinsics()[1] for model in models_ref]

    # v  shape (...,Ncameras,Nheight,Nwidth,...)
    # q0 shape (...,         Nheight,Nwidth,...)
    v,q0 = \
        mrcal.sample_imager_unproject(Nw,None,
                                      *imagersizes[0],
                                      lensmodels, intrinsics_data,
                                      normalize = True)

    W,H = imagersizes[0]
    focus_center = None
    focus_radius = -1
    if focus_center is None: focus_center = ((W-1.)/2., (H-1.)/2.)
    if focus_radius < 0:     focus_radius = min(W,H)/6.


    implied_Rt10 = \
        mrcal.implied_Rt10__from_unprojections(q0,
                                               v[0,...], v[1,...],
                                               focus_center = focus_center,
                                               focus_radius = focus_radius)

    q1 = mrcal.project( mrcal.transform_point_Rt(implied_Rt10,
                                                 v[0,...]),
                       lensmodels[1], intrinsics_data[1])
    diff = nps.mag(q1 - q0)

    # zero-out everything too far from the center
    center = (imagersizes[0] - 1.) / 2.
    diff[ nps.norm2(q0 - center) > max_dist_from_center*max_dist_from_center ] = 0
    # gp.plot(diff,
    #         ascii = True,
    #         using = mrcal.imagergrid_using(imagersizes[0], Nw),
    #         square=1, _with='image', tuplesize=3, hardcopy='/tmp/yes.gp', cbmax=3)

    return diff


for icam in range(len(models_ref)):
    diff = projection_diff( (models_ref[icam], models_solved[icam]), 800)

    testutils.confirm_equal(diff, 0,
                            worstcase = True,
                            eps = 6.,
                            msg = f"Recovered intrinsics for camera {icam}")


# It would be nice to check the outlier detections, but this is iffy. Here I'm
# generating 1% outliers (hard-coded in sample_dqref()), but the outlier
# rejection is overly aggressive. I'm currently seeing 4.4%:
#
#   np.count_nonzero(observations[...,2]<=0) / observations[...,0].ravel().size
#
# The outlier rejection scheme just cuts out 3sigma residuals and above, so it's
# not great. I'm not entirely sure why it's over-reporting the outliers here,
# but I should investigate that at the same time as I overhaul the outlier
# rejection scheme (presumably to use one of my flavors of Cook's D factor)

# I test make_perfect_observations(). Doing it here is easy; doing it elsewhere
# it much more work
if True:
    optimization_inputs_perfect = copy.deepcopy(optimization_inputs)

    mrcal.make_perfect_observations(optimization_inputs_perfect,
                                    observed_pixel_uncertainty=0)
    x = mrcal.optimizer_callback(**optimization_inputs_perfect,
                                 no_jacobian      = True,
                                 no_factorization = True)[1]

    Nmeas = mrcal.num_measurements_boards(**optimization_inputs_perfect)
    if Nmeas > 0:
        i_meas0 = mrcal.measurement_index_boards(0, **optimization_inputs_perfect)
        testutils.confirm_equal( x[i_meas0:i_meas0+Nmeas],
                                 0,
                                 worstcase = True,
                                 eps = 1e-8,
                                 msg = f"make_perfect_observations() works for boards")
    else:
        testutils.confirm( False,
                           msg = f"Nmeasurements_boards <= 0")
testutils.finish()
