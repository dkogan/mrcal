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

kwargs = dict( imagersizes                       = imagersizes,
               observed_pixel_uncertainty        = pixel_uncertainty,
               calibration_object_spacing        = object_spacing,
               calibration_object_width_n        = object_width_n,
               calibration_object_height_n       = object_height_n,
               skip_outlier_rejection            = True,
               skip_regularization               = False,
               verbose                           = False )

stats = mrcal.optimize( *args, **kwargs,
                        do_optimize_frames                = True,
                        do_optimize_intrinsic_core        = False,
                        do_optimize_intrinsic_distortions = False,
                        do_optimize_extrinsics            = True)
stats = mrcal.optimize( *args, **kwargs,
                        do_optimize_frames                = True,
                        do_optimize_intrinsic_core        = True,
                        do_optimize_intrinsic_distortions = False,
                        do_optimize_extrinsics            = True)

calobject_warp = np.array((0.001, 0.001))
stats = mrcal.optimize( *args, **kwargs,
                        do_optimize_frames                = True,
                        do_optimize_intrinsic_core        = True,
                        do_optimize_intrinsic_distortions = True,
                        do_optimize_extrinsics            = True,

                        do_optimize_calobject_warp        = True,
                        calobject_warp                    = calobject_warp)

models_solved = \
    [ mrcal.cameramodel( imagersize = imagersizes[0],
                         intrinsics = (lensmodel, intrinsics[i,:])) \
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

testutils.finish()
