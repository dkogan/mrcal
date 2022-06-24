#!/usr/bin/python3

r'''Basic structure-from-motion test

I observe, with noise, a number of points from various angles with a single
camera, and I make sure that I can accurately compute the locations of the
camera.

I simulate a horizontal plane with many features on it, with the camera moving
around just above this horizontal plane (like a car driving around a flat-ish
area)

#warning "triangulated-solve: comment"

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

import numpy.random
np.random.seed(0)


############# Set up my world, and compute all the perfect positions, pixel
############# observations of everything
(W,H) = (4000,2200)
m = mrcal.cameramodel( intrinsics = ('LENSMODEL_PINHOLE',
                                     np.array((1000., 1000., (W-1)/2, (H-1)/2))),
                       imagersize = (W,H) )

# Camera0 is fixed: it's the reference coordinate system in the solve. The
# points I'm observing lie on a perfect horizontal plane (y = constant) sitting
# a bit below the cameras. The points span a forward arc at different ranges
ranges         = np.array((3, 4, 5, 8, 20, 50, 200, 1000))
points_fov_deg = 120.

# The camera starts out at the center, and moves steadily to the right (along
# the x axis). Orientation is always straight ahead (pointed along the z axis)
# and there's no front/back of up/down motion
Ncameras                    = 10
step                        = 1.0
camera_height_above_surface = 1.5


th = np.linspace(-points_fov_deg/2.*np.pi/180.,
                  points_fov_deg/2.*np.pi/180.)
cth = np.cos(th)
sth = np.sin(th)

# In the ref coordinate system (locked to camera0)
v = nps.glue( nps.transpose(sth),
              np.zeros(cth.shape + (1,)),
              nps.transpose(cth),
              axis = -1)

# In the ref coordinate system (locked to camera0)
# shape (Npoints, 3)
points_true = \
    nps.clump( v * nps.mv(ranges,  -1,-3),
               n = 2 ) + \
    np.array((0, camera_height_above_surface, 0))

Npoints = points_true.shape[0]

# import gnuplotlib as gp
# gp.plot(points_true,
#         _3d=1,
#         xlabel='x',
#         ylabel='y',
#         zlabel='z',
#         _with='points',
#         square=1,
#         tuplesize=-3,
#         wait=1)
# sys.exit()

x_ref_cam_true  = np.arange(Ncameras) * step

# shape (Ncameras,6)
rt_ref_cam_true = nps.glue( nps.transpose(x_ref_cam_true),
                            np.zeros((Ncameras,5)),
                            axis = -1 )
rt_cam_ref_true = mrcal.invert_rt(rt_ref_cam_true)

# I project all the points into all the cameras. Anything that's in view, I keep

# shape (Npoints, Ncameras, 3)
pcam_true = mrcal.transform_point_rt(rt_cam_ref_true,
                                     nps.mv(points_true, -2, -3))

# shape (Npoints, Ncameras, 2)
qcam_true = mrcal.project(pcam_true, *m.intrinsics())

# ALL the indices. I'm about to cut these down to the visible ones
# shape (Npoints, Ncameras)
indices_cam, indices_point = \
    np.meshgrid(np.arange(Ncameras),
                np.arange(Npoints))

valid_observation_index = \
    (qcam_true[...,0] >= 0  ) * \
    (qcam_true[...,0] <= W-1) * \
    (qcam_true[...,1] >= 0  ) * \
    (qcam_true[...,1] <= H-1) * \
    (pcam_true[...,2] >  0)

indices_point_camintrinsics_camextrinsics = \
    nps.glue( nps.transpose(indices_point[valid_observation_index]),
              np.zeros((np.count_nonzero(valid_observation_index),1)),
              nps.transpose(indices_cam  [valid_observation_index]) - 1,
              axis = -1 ).astype(np.int32)

observations_true = qcam_true[valid_observation_index]


# To sufficiently constrain the geometry of the problem I lock

# - The pose of the first camera (using it as the reference coordinate system)
#
# - One observed point. I only need to lock the scale of the problem, so this is
#   overkill, but it's what I have available for now
Npoints_fixed = 1



rt_cam_ref_noise_ratio = 0.2 # 20%
rt_cam_ref_noise = \
    ((np.random.random_sample(rt_cam_ref_true.shape) * 2) - 1) * rt_cam_ref_noise_ratio
rt_cam_ref_noisy = rt_cam_ref_true * (1.0 + rt_cam_ref_noise)


points_noise_ratio = 0.2 # 20%
points_noise = \
    ((np.random.random_sample(points_true.shape) * 2) - 1) * points_noise_ratio
# The fixed points are perfect
points_noise[-Npoints_fixed:, :] = 0
points_noisy = points_true * (1. + points_noise)

observations_noise_pixels = 2
observations_noise = \
    ((np.random.random_sample(observations_true.shape) * 2) - 1) * observations_noise_pixels
observations_noisy = observations_true + observations_noise



points       = points_noisy
observations = observations_noisy
rt_cam_ref   = rt_cam_ref_noisy


# The TRAILING Npoints_fixed points are fixed. The leading ones are triangulatd
idx_points_fixed = indices_point_camintrinsics_camextrinsics[:,0] >= Npoints-Npoints_fixed

if np.count_nonzero(idx_points_fixed) == 0:
    print("No fixed point observations. Change the problem definition",
          file=sys.stderr)
    sys.exit(1)

indices_point_camintrinsics_camextrinsics_fixed = \
    indices_point_camintrinsics_camextrinsics[idx_points_fixed].copy()
indices_point_camintrinsics_camextrinsics_fixed[:,0] -= (Npoints-Npoints_fixed)
observations_fixed = observations[idx_points_fixed].copy()

# add "weight" column
observations_fixed = nps.glue(observations_fixed,
                              np.ones((len(observations_fixed),1)),
                              axis = -1)

points_fixed = points[-Npoints_fixed:]

indices_point_camintrinsics_camextrinsics_triangulated = \
    indices_point_camintrinsics_camextrinsics[~idx_points_fixed].copy()
observations_triangulated = observations[~idx_points_fixed].copy()

# For now "observations_triangulated" are local observation vectors
observations_triangulated = mrcal.unproject(observations_triangulated[:,:2], *m.intrinsics())

optimization_inputs = \
    dict( intrinsics            = nps.atleast_dims(m.intrinsics()[1], -2),
          extrinsics_rt_fromref = rt_cam_ref,
          points                = points_fixed,

          # Explicit points. Fixed only
          observations_point                                     = observations_fixed,
          indices_point_camintrinsics_camextrinsics              = indices_point_camintrinsics_camextrinsics_fixed,

          observations_point_triangulated                        = observations_triangulated,
          indices_point_triangulated_camintrinsics_camextrinsics = indices_point_camintrinsics_camextrinsics_triangulated,

          lensmodel                         = m.intrinsics()[0],
          imagersizes                       = nps.atleast_dims(m.imagersize(), -2),
          Npoints_fixed                     = Npoints_fixed,
          point_min_range                   = 1.0,
          point_max_range                   = 1000.0,
          do_optimize_intrinsics_core       = False,
          do_optimize_intrinsics_distortions= False,
          do_optimize_extrinsics            = True,
          do_optimize_frames                = True,
          do_apply_outlier_rejection        = False,
          do_apply_regularization           = True,
          verbose                           = False)


import IPython
IPython.embed()
sys.exit()

#optimization_inputs['verbose'] = True
stats = mrcal.optimizer_callback(**optimization_inputs)

# Got a solution. How well do they fit?
fit_rms = np.sqrt(np.mean(nps.norm2(points_fixed - points_true)))

testutils.confirm_equal(fit_rms, 0,
                        msg = f"Solved at ref coords with known-position points",
                        eps = 1.0)

testutils.finish()
