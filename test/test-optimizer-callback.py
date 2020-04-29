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
    mrcal.get_chessboard_observations(10, 10,
                                      ('frame*-cam0.xxx','frame*-cam1.xxx'),
                                      f"{testdir}/data/synthetic-board-observations.vnl")
i = (1,2,4,5)
observations         = observations        [i, ...]
indices_frame_camera = indices_frame_camera[i, ...]
paths                = [paths[_] for _ in i]

# reference models
models = [ mrcal.cameramodel(m) for m in ( f"{testdir}/data/cam0.opencv8.cameramodel",
                                           f"{testdir}/data/cam1.opencv8.cameramodel",) ]

lensmodel       = models[0].intrinsics()[0]
intrinsics_data = nps.cat(models[0].intrinsics()[1],
                          models[1].intrinsics()[1])
extrinsics_rt10 = mrcal.compose_rt( models[1].extrinsics_rt_fromref(),
                                    models[0].extrinsics_rt_toref() )

imagersizes = nps.cat(models[0].imagersize(),
                      models[1].imagersize())
# I now have the "right" camera parameters. I don't have the frames or points,
# but it's fine to just make them up. This is a regression test.
frames = linspace_shaped(3,6)
frames[:,5] += 5 # push them back

points                      = 10. + 2.*linspace_shaped(2,3)
observations_point_xy       = 1000. + 500. * linspace_shaped(2,2)
observations_point_weights  = np.array(((0.9,), (0.8,)))
observations_point_distance = np.array(((10.,), (-1,)))
observations_point          = nps.glue(observations_point_xy,
                                       observations_point_weights,
                                       observations_point_distance,
                                       axis = -1)
indices_point_camera_points = np.array(((0,1),
                                        (1,0)),
                                       dtype = np.int32)

all_test_kwargs = ( dict(do_optimize_intrinsic_core        = False,
                         do_optimize_intrinsic_distortions = True,
                         do_optimize_extrinsics            = False,
                         do_optimize_frames                = False,
                         do_optimize_calobject_warp        = False,
                         skip_regularization               = False),

                    dict(do_optimize_intrinsic_core        = True,
                         do_optimize_intrinsic_distortions = False,
                         do_optimize_extrinsics            = False,
                         do_optimize_frames                = False,
                         do_optimize_calobject_warp        = False,
                         skip_regularization               = False),

                    dict(do_optimize_intrinsic_core        = False,
                         do_optimize_intrinsic_distortions = False,
                         do_optimize_extrinsics            = False,
                         do_optimize_frames                = True,
                         do_optimize_calobject_warp        = False,
                         skip_regularization               = False),

                    dict(do_optimize_intrinsic_core        = True,
                         do_optimize_intrinsic_distortions = True,
                         do_optimize_extrinsics            = False,
                         do_optimize_frames                = True,
                         do_optimize_calobject_warp        = False,
                         skip_regularization               = False),

                    dict(do_optimize_intrinsic_core        = True,
                         do_optimize_intrinsic_distortions = True,
                         do_optimize_extrinsics            = True,
                         do_optimize_frames                = True,
                         do_optimize_calobject_warp        = True,
                         skip_regularization               = True), )

itest = 0
for kwargs in all_test_kwargs:
    x,Jt = mrcal.optimizerCallback( intrinsics_data,
                                    nps.atleast_dims(extrinsics_rt10, -2),
                                    frames, points,
                                    observations,       indices_frame_camera,
                                    observations_point, indices_point_camera_points,

                                    lensmodel,
                                    imagersizes                       = imagersizes,
                                    calibration_object_spacing        = 0.1,
                                    calibration_object_width_n        = 10,
                                    verbose                           = False,
                                    calobject_warp                    = np.array((1e-3, 2e-3)),

                                    **kwargs)
    Jt = Jt.toarray()

    if False:
        np.save(f"{testdir}/data/test-optimizer-callback-ref-x-{itest}.npy",  x)
        np.save(f"{testdir}/data/test-optimizer-callback-ref-Jt-{itest}.npy", Jt)
    else:
        x_ref  = np.load(f"{testdir}/data/test-optimizer-callback-ref-x-{itest}.npy")
        Jt_ref = np.load(f"{testdir}/data/test-optimizer-callback-ref-Jt-{itest}.npy")

        testutils.confirm_equal(x,  x_ref,  msg = f"comparing x for case {itest}")
        testutils.confirm_equal(Jt, Jt_ref, msg = f"comparing Jt for case {itest}")

    itest += 1

testutils.finish()
