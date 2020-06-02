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

Ncameras = len(models)
Nframes  = 150

models[0].extrinsics_rt_fromref(np.zeros((6,), dtype=float))
models[1].extrinsics_rt_fromref(np.array((0.08,0.02,0.02, 1., 0.9,0.1)))
models[2].extrinsics_rt_fromref(np.array((0.01,0.07,0.02, 2.1,0.4,0.2)))
models[3].extrinsics_rt_fromref(np.array((0.06,0.08,0.08, 4.4,0.2,0.1)))


object_spacing     = 0.1
object_width_n     = 10
object_height_n    = 9
calobject_warp_ref = np.array((0.002, -0.005))

print("making synth")

# shape (Nframes, Ncameras, Nh, Nw, 2)
p = mrcal.make_synthetic_board_observations(models,
                                            object_width_n, object_height_n, object_spacing,
                                            calobject_warp_ref,
                                            np.array((0,   0,   5.0,  0.,  0.,  0.)),
                                            np.array((1.5, 1.5, 1.0,  40., 30., 30.)),
                                            Nframes)
print("made synth")
p_noise = np.random.randn(*p.shape) * 1.0
p_noisy = p + p_noise

imagersizes = nps.cat( *[m.imagersize() for m in models] )


# I want observations of shape (Nframes*Ncameras, Nw, Nh, 3) where each row is
# (x,y,weight)
observations = nps.xchg( nps.clump(p_noisy, n=2),
                         -2,-3 )
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

print("optimizing")
stats = mrcal.optimize( intrinsics,
                        extrinsics,
                        frames, None,
                        observations, indices_frame_camintrinsics_camextrinsics,
                        None, None,

                        models[0].intrinsics()[0],
                        imagersizes                       = imagersizes,
                        observed_pixel_uncertainty        = 1.0,
                        do_optimize_intrinsic_core        = False,
                        do_optimize_intrinsic_distortions = False,
                        do_optimize_extrinsics            = True,
                        do_optimize_frames                = True,
                        calibration_object_spacing        = object_spacing,
                        calibration_object_width_n        = object_width_n,
                        skip_outlier_rejection            = True,
                        skip_regularization               = False,
                        verbose                           = False)
print("optimizing")
stats = mrcal.optimize( intrinsics,
                        extrinsics,
                        frames, None,
                        observations, indices_frame_camintrinsics_camextrinsics,
                        None, None,

                        models[0].intrinsics()[0],
                        imagersizes                       = imagersizes,
                        observed_pixel_uncertainty        = 1.0,
                        do_optimize_intrinsic_core        = True,
                        do_optimize_intrinsic_distortions = False,
                        do_optimize_extrinsics            = True,
                        do_optimize_frames                = True,
                        calibration_object_spacing        = object_spacing,
                        calibration_object_width_n        = object_width_n,
                        skip_outlier_rejection            = True,
                        skip_regularization               = False,
                        verbose                           = False)
print("optimizing")
calobject_warp = np.array((0.001, 0.001))
stats = mrcal.optimize( intrinsics,
                        extrinsics,
                        frames, None,
                        observations, indices_frame_camintrinsics_camextrinsics,
                        None, None,

                        models[0].intrinsics()[0],
                        calobject_warp                    = calobject_warp,
                        imagersizes                       = imagersizes,
                        observed_pixel_uncertainty        = 1.0,
                        do_optimize_calobject_warp        = True,
                        do_optimize_intrinsic_core        = True,
                        do_optimize_intrinsic_distortions = True,
                        do_optimize_extrinsics            = True,
                        do_optimize_frames                = True,
                        calibration_object_spacing        = object_spacing,
                        calibration_object_width_n        = object_width_n,
                        skip_outlier_rejection            = True,
                        skip_regularization               = False,
                        verbose                           = False)

print(f"solved calobject_warp: {calobject_warp}; this is backwards")
models_solved = \
    [ mrcal.cameramodel( imagersize = models[0].imagersize(),
                         intrinsics = (models[0].intrinsics()[0], intrinsics[i,:])) \
      for i in range(Ncameras)]
for i in range(1,Ncameras):
    models_solved[i].extrinsics_rt_fromref( extrinsics[i-1,:] )

import IPython
IPython.embed()
sys.exit()

mrcal.show_intrinsics_diff((models_solved[0], models[0]),)

testutils.confirm_equal(fit_rms, 0,
                        msg = f"Solved at ref coords with known-position points",
                        eps = 1.0)

testutils.finish()
