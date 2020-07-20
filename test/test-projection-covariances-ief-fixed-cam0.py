#!/usr/bin/python3

r'''Make sure the covariances_ief matrices are being computed correctly

This is a subset of test-calibration-uncertainty-fixed-cam0.py. In fact that
test already makes sure the covariances_ief are correct. But that test uses
simple LENSMODEL_OPENCV4 distortions while here I use more complex
LENSMODEL_OPENCV8 distortions. Using opencv8 makes the other test way too slow,
but it would have found the bug I fixed in 9ed0ed3.

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

from test_calibration_helpers import optimize,sample_dqref,get_var_ief

import re
if   re.search("fixed-cam0",   sys.argv[0]): fixedframes = False
elif re.search("fixed-frames", sys.argv[0]): fixedframes = True
else:
    raise Exception("This script should contain either 'fixed-cam0' or 'fixed-frames' in the filename")

# I want the RNG to be deterministic
np.random.seed(0)

############# Set up my world, and compute all the perfect positions, pixel
############# observations of everything
models_ref = ( mrcal.cameramodel(f"{testdir}/data/cam0.opencv8.cameramodel"),
               mrcal.cameramodel(f"{testdir}/data/cam1.opencv8.cameramodel") )

imagersizes = nps.cat( *[m.imagersize() for m in models_ref] )
lensmodel   = models_ref[0].intrinsics()[0]
Nintrinsics = mrcal.getNlensParams(lensmodel)

Ncameras = len(models_ref)
Nframes  = 50

models_ref[0].extrinsics_rt_fromref(np.zeros((6,), dtype=float))
models_ref[1].extrinsics_rt_fromref(np.array((-0.1,0.08,0.08, 4.4,0.2,0.1)))

pixel_uncertainty_stdev = 1.5
object_spacing          = 0.1
object_width_n          = 10
object_height_n         = 9
calobject_warp_ref      = np.array((0.002, -0.005))

# shapes (Nframes, Ncameras, Nh, Nw, 2),
#        (Nframes, 4,3)
#
# I'm placing the boards very close to the cameras: z = 2.0 +- 1.5m. If I place
# them further out, I'll get few observations at the edges, the higher-order
# distortion terms won't have strong effects ( abs(dx/dp) very small for those
# p), Jt) will be nearly-singular, and inv(JtJ) will show large numberical
# errors. I don't yet know if this is a real problem that needs to be dealt with
# in the general case, but for the THIS test I make sure we have board
# observations at the edges, and things work
q_ref,Rt_cam0_board_ref = \
    mrcal.make_synthetic_board_observations(models_ref,
                                            object_width_n, object_height_n, object_spacing,
                                            calobject_warp_ref,
                                            np.array((-2,   0,  2.0,  0.,  0.,  0.)),
                                            np.array((2.5, 2.5, 1.5,  40., 30., 30.)),
                                            Nframes)

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

# These are perfect
intrinsics_ref = nps.cat( *[m.intrinsics()[1]         for m in models_ref] )
if fixedframes:
    extrinsics_ref = nps.cat( *[m.extrinsics_rt_fromref() for m in models_ref] )
else:
    extrinsics_ref = nps.cat( *[m.extrinsics_rt_fromref() for m in models_ref[1:]] )
if extrinsics_ref.size == 0:
    extrinsics_ref = np.zeros((0,6), dtype=float)
frames_ref     = mrcal.rt_from_Rt(Rt_cam0_board_ref)


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
             indices_frame_camera[:,(1,)],
             axis=-1)
if not fixedframes:
    indices_frame_camintrinsics_camextrinsics[:,2] -= 1



def sample_reoptimized_parameters(do_optimize_frames, apply_noise=True, **kwargs):
    global solver_context
    if apply_noise:
        _, observations_perturbed = sample_dqref(observations_ref,
                                                 pixel_uncertainty_stdev)
    else:
        observations_perturbed = observations_ref.copy()
    intrinsics_solved,extrinsics_solved,frames_solved,_, \
    idx_outliers, \
    _,  _, _,                  \
    covariance_intrinsics,_,covariances_ief, covariances_ief_rotationonly, \
    solver_context =           \
        optimize(intrinsics_ref, extrinsics_ref, frames_ref, observations_perturbed,
                 indices_frame_camintrinsics_camextrinsics,
                 lensmodel,
                 imagersizes,
                 object_spacing, object_width_n, object_height_n,
                 pixel_uncertainty_stdev,
                 calobject_warp                    = calobject_warp_ref,
                 do_optimize_intrinsic_core        = True,
                 do_optimize_intrinsic_distortions = True,
                 do_optimize_extrinsics            = True,
                 do_optimize_frames                = do_optimize_frames,
                 do_optimize_calobject_warp        = True,
                 skip_outlier_rejection            = True,
                 skip_regularization               = True,
                 **kwargs)

    return intrinsics_solved,extrinsics_solved,frames_solved, \
        covariance_intrinsics, covariances_ief, covariances_ief_rotationonly


covariances_ief,covariances_ief_rotationonly = \
    sample_reoptimized_parameters(do_optimize_frames = not fixedframes,
                                  apply_noise        = False,
                                  get_covariances    = True)[-2:]

optimize_kwargs = \
    dict( intrinsics                                = intrinsics_ref,
          extrinsics_rt_fromref                     = extrinsics_ref,
          frames_rt_toref                           = frames_ref,
          points                                    = None,
          observations_board                        = observations_ref,
          indices_frame_camintrinsics_camextrinsics = indices_frame_camintrinsics_camextrinsics,
          observations_point                        = None,
          indices_point_camintrinsics_camextrinsics = None,
          lensmodel                                 = lensmodel,
          imagersizes                               = imagersizes,
          calobject_warp                            = calobject_warp_ref,
          do_optimize_intrinsic_core                = True,
          do_optimize_intrinsic_distortions         = True,
          do_optimize_extrinsics                    = True,
          do_optimize_frames                        = not fixedframes,
          do_optimize_calobject_warp                = True,
          calibration_object_spacing                = object_spacing,
          calibration_object_width_n                = object_width_n,
          calibration_object_height_n               = object_height_n,
          skip_regularization                       = True,
          observed_pixel_uncertainty                = pixel_uncertainty_stdev)

# I want to treat the extrinsics arrays as if all the camera transformations are
# stored there
if fixedframes:
    extrinsics_ref_mounted = extrinsics_ref
else:
    extrinsics_ref_mounted = \
        nps.glue( np.zeros((6,), dtype=float),
                  extrinsics_ref,
                  axis = -2)

cache_invJtJ = [None]

covariances_ief_ref_rt = \
    [ get_var_ief(icam_intrinsics          = icam_intrinsics,
                  icam_extrinsics          = icam_intrinsics - (0 if fixedframes else 1),
                  did_optimize_extrinsics  = True,
                  did_optimize_frames      = not fixedframes,
                  Nstate_intrinsics_onecam = Nintrinsics,
                  Nframes                  = Nframes,
                  pixel_uncertainty_stdev  = pixel_uncertainty_stdev,
                  rotation_only            = False,
                  solver_context           = solver_context,
                  cache_invJtJ             = cache_invJtJ) for icam_intrinsics in range(Ncameras) ]

covariances_ief_ref_r = \
    [ get_var_ief(icam_intrinsics          = icam_intrinsics,
                  icam_extrinsics          = icam_intrinsics - (0 if fixedframes else 1),
                  did_optimize_extrinsics  = True,
                  did_optimize_frames      = not fixedframes,
                  Nstate_intrinsics_onecam = Nintrinsics,
                  Nframes                  = Nframes,
                  pixel_uncertainty_stdev  = pixel_uncertainty_stdev,
                  rotation_only            = True,
                  solver_context           = solver_context,
                  cache_invJtJ             = cache_invJtJ) for icam_intrinsics in range(Ncameras) ]

for i in range(Ncameras):
    testutils.confirm_equal(covariances_ief[i], covariances_ief_ref_rt[i],
                            eps = 0.1,
                            percentile= 99.9,
                            relative  = True,
                            msg = f"covariances_ief with full rt matches for camera {i}")

for i in range(Ncameras):
    testutils.confirm_equal(covariances_ief_rotationonly[i], covariances_ief_ref_r[i],
                            eps = 0.1,
                            percentile= 99.9,
                            relative  = True,
                            msg = f"covariances_ief with rotation-only matches for camera {i}")

testutils.finish()
