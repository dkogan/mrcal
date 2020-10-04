#!/usr/bin/python3

r'''Tests the mrcal optimization callback function

There's some hairy logic involving locking down some subset of the variables,
and that needs testing

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

# deterministic "random". Not random at all.
from functools import reduce
def linspace_shaped(*shape):
    product = reduce( lambda x,y: x*y, shape)
    return np.linspace(0,1,product).reshape(*shape)

# I read the synthetic-data observations. These have 3 frames with 2 cameras
# each. I want to make things uneven, so I make the first two frames have only 1
# camera each
observations, indices_frame_camera, paths = \
    mrcal.chessboard_observations(10, 10,
                                      ('frame*-cam0.xxx','frame*-cam1.xxx'),
                                      f"{testdir}/data/synthetic-board-observations.vnl")
indices_frame_camintrinsics_camextrinsics = np.zeros((len(indices_frame_camera), 3), dtype=indices_frame_camera.dtype)
indices_frame_camintrinsics_camextrinsics[:, :2] = indices_frame_camera
indices_frame_camintrinsics_camextrinsics[:,  2] = indices_frame_camintrinsics_camextrinsics[:, 1]-1

i = (1,2,4,5)
observations                              = observations        [i, ...]
indices_frame_camintrinsics_camextrinsics = indices_frame_camintrinsics_camextrinsics[i, ...]
paths                                     = [paths[_] for _ in i]

# reference models
models = [ mrcal.cameramodel(m) for m in ( f"{testdir}/data/cam0.opencv8.cameramodel",
                                           f"{testdir}/data/cam1.opencv8.cameramodel",) ]

lensmodel       = models[0].intrinsics()[0]
intrinsics_data = nps.cat(models[0].intrinsics()[1],
                          models[1].intrinsics()[1])
extrinsics_rt_fromref = mrcal.compose_rt( models[1].extrinsics_rt_fromref(),
                                          models[0].extrinsics_rt_toref() )

imagersizes = nps.cat(models[0].imagersize(),
                      models[1].imagersize())
# I now have the "right" camera parameters. I don't have the frames or points,
# but it's fine to just make them up. This is a regression test.
frames_rt_toref = linspace_shaped(3,6)
frames_rt_toref[:,5] += 5 # push them back

indices_point_camintrinsics_camextrinsics = \
    np.array(((0,1,-1),
              (1,0,-1),
              (1,1, 0),
              (2,0,-1),
              (2,1, 0)),
             dtype = np.int32)

points                      = 10. + 2.*linspace_shaped(3,3)
observations_point_xy       = 1000. + 500. * linspace_shaped(5,2)
observations_point_weights  = np.array((0.9, 0.8, 0.9, 1.3, 1.8))

observations_point = \
    nps.glue(observations_point_xy,
             nps.transpose(observations_point_weights),
             axis = -1)

all_test_kwargs = ( dict(do_optimize_intrinsics_core       = False,
                         do_optimize_intrinsics_distortions= True,
                         do_optimize_extrinsics            = False,
                         do_optimize_frames                = False,
                         do_optimize_calobject_warp        = False,
                         do_apply_regularization           = True),

                    dict(do_optimize_intrinsics_core       = True,
                         do_optimize_intrinsics_distortions= False,
                         do_optimize_extrinsics            = False,
                         do_optimize_frames                = False,
                         do_optimize_calobject_warp        = False,
                         do_apply_regularization           = True),

                    dict(do_optimize_intrinsics_core       = False,
                         do_optimize_intrinsics_distortions= False,
                         do_optimize_extrinsics            = False,
                         do_optimize_frames                = True,
                         do_optimize_calobject_warp        = False,
                         do_apply_regularization           = True),

                    dict(do_optimize_intrinsics_core       = True,
                         do_optimize_intrinsics_distortions= True,
                         do_optimize_extrinsics            = False,
                         do_optimize_frames                = True,
                         do_optimize_calobject_warp        = False,
                         do_apply_regularization           = True),

                    dict(do_optimize_intrinsics_core       = True,
                         do_optimize_intrinsics_distortions= True,
                         do_optimize_extrinsics            = True,
                         do_optimize_frames                = True,
                         do_optimize_calobject_warp        = True,
                         do_apply_regularization           = False),

                    dict(do_optimize_intrinsics_core       = True,
                         do_optimize_intrinsics_distortions= True,
                         do_optimize_extrinsics            = True,
                         do_optimize_frames                = True,
                         do_optimize_calobject_warp        = True,
                         do_apply_regularization           = False,
                         outlier_indices = np.array((1,2), dtype=np.int32)))

itest = 0
for kwargs in all_test_kwargs:

    observations_copy = observations.copy()

    if 'outlier_indices' in kwargs:
        # mark the requested outliers, and delete the old way of specifying
        # these
        for i in kwargs['outlier_indices']:
            nps.clump(observations_copy, n=3)[i,2] = -1.
        del kwargs['outlier_indices']

    optimization_inputs = \
        dict( intrinsics                                = intrinsics_data,
              extrinsics_rt_fromref                     = nps.atleast_dims(extrinsics_rt_fromref, -2),
              frames_rt_toref                           = frames_rt_toref,
              points                                    = points,
              observations_board                        = observations_copy,
              indices_frame_camintrinsics_camextrinsics = indices_frame_camintrinsics_camextrinsics,
              observations_point                        = observations_point,
              indices_point_camintrinsics_camextrinsics = indices_point_camintrinsics_camextrinsics,
              lensmodel                                 = lensmodel,
              calobject_warp                            = np.array((1e-3, 2e-3)),
              imagersizes                               = imagersizes,
              calibration_object_spacing                = 0.1,
              point_min_range                           = 1.0,
              point_max_range                           = 1000.0,
              verbose                                   = False,
              **kwargs )

    x,J = mrcal.optimizer_callback( **optimization_inputs )[1:3]
    J = J.toarray()

    # I compare full-state J so that I can change SCALE_... without breaking the
    # test
    mrcal.pack_state(J, **optimization_inputs)

    # Set this to True to store the current values as the "true" values
    if False:
        np.save(f"{testdir}/data/test-optimizer-callback-ref-x-{itest}.npy", x)
        np.save(f"{testdir}/data/test-optimizer-callback-ref-J-{itest}.npy", J)
    else:
        x_ref = np.load(f"{testdir}/data/test-optimizer-callback-ref-x-{itest}.npy")
        J_ref = np.load(f"{testdir}/data/test-optimizer-callback-ref-J-{itest}.npy")

        testutils.confirm_equal(x, x_ref, msg = f"comparing x for case {itest}")
        testutils.confirm_equal(J, J_ref, msg = f"comparing J for case {itest}")

    itest += 1

testutils.finish()
