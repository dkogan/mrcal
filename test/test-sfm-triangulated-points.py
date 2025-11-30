#!/usr/bin/env python3

r'''Basic structure-from-motion test

I observe, with noise, a number of points from various angles with a single
camera, and I make sure that I can accurately compute the locations of the
camera.

I simulate a horizontal plane with many features on it, with the forward-looking
camera moving around just above this horizontal plane (like a car driving around
a flat-ish area)

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
                                     np.array((600., 600., (W-1)/2, (H-1)/2))),
                       imagersize = (W,H) )

# Camera0 is fixed: it's the reference coordinate system in the solve. The
# points I'm observing lie on a perfect horizontal plane (y = constant) sitting
# a bit below the cameras. The points span a forward arc at different ranges
ranges = np.array(( 1000,
                    500,
                    200,
                    100,
                    50,
                    30,
                    20,
                    15,
                    8,
                    5,
                    4,
                    3,
                   ))

points_fov_deg    = 160.
Npoints_per_range = 50

# The camera starts out at the center, and moves steadily to the right along the
# x axis, with some smaller nonlinear y motion as well. I want to avoid
# collinear camera locations (this produces singular solves). Orientation is
# always straight ahead (pointed along the z axis) and there's no front/back motion
Ncamera_poses = 10
step          = 1.0



th = np.linspace(-points_fov_deg/2.*np.pi/180.,
                  points_fov_deg/2.*np.pi/180.,
                 Npoints_per_range)
cth = np.cos(th)
sth = np.sin(th)

# In the ref coordinate system (locked to camera0)
v = nps.glue( nps.transpose(sth),
              np.zeros(cth.shape + (1,)),
              nps.transpose(cth),
              axis = -1)

# In the ref coordinate system (locked to camera0)
# Baseline is flat, below the cameras. Curved left/right. Rising as we move away
# from the cameras
# shape (Npoints, 3)
points_true = \
    nps.clump( v * nps.mv(ranges,  -1,-3),
               n = 2 ) + \
    np.array((0, 1.5, 0))
points_true[:,1] += (points_true[:,0]/ranges[0]*15.)**2.
points_true[:,1] -= (points_true[:,2]/ranges[0]*200.)

Npoints = points_true.shape[0]

if False:
    import gnuplotlib as gp
    gp.plot(points_true,
            _3d=1,
            xlabel='x',
            ylabel='y',
            zlabel='z',
            _with='points',
            square=1,
            tuplesize=-3,
            wait=1)
    sys.exit()

x_ref_cam_true  = np.arange(Ncamera_poses) * step
y_ref_cam_true  = np.zeros((Ncamera_poses,),)
y_ref_cam_true  = (np.arange(Ncamera_poses) - (Ncamera_poses-1.)/2.) ** 2. / 100.
y_ref_cam_true -= y_ref_cam_true[0]

# shape (Ncamera_poses,6)
rt_ref_cam_true = nps.glue( np.zeros((Ncamera_poses,3)), # r
                            nps.transpose(x_ref_cam_true),
                            nps.transpose(y_ref_cam_true),
                            np.zeros((Ncamera_poses,1)), # z
                            axis = -1 )
rt_cam_ref_true = mrcal.invert_rt(rt_ref_cam_true)

if False:
    mrcal.show_geometry(rt_cam_ref_true,
                        #points = points_true
                        )

# I project all the points into all the cameras. Anything that's in view, I keep

# shape (Npoints, Ncamera_poses, 3)
pcam_true = mrcal.transform_point_rt(rt_cam_ref_true,
                                     nps.mv(points_true, -2, -3))

# shape (Npoints, Ncamera_poses, 2)
qcam_true = mrcal.project(pcam_true, *m.intrinsics())

# ALL the indices. I'm about to cut these down to the visible ones
# shape (Npoints, Ncamera_poses)
indices_cam, indices_point = \
    np.meshgrid(np.arange(Ncamera_poses),
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

# Any point observed by a single camera is thrown out
#
# I have a sequence of observed-point indices. Most point are observed by
# multiple cameras, so the indices will appear more than once. But some indices
# will appear just once, and I want to throw those away. How do I find these?
#
# 1. I compute the diff: nonzero entries signify transitions between different
#    points. Single points will have consecutive transitions. So...
# 2. diff(diff(ipoint)) == 0 AND both sides are a transition signifies single
#    points
ipoint = indices_point_camintrinsics_camextrinsics[:,0]
d = nps.glue(True,
             np.diff(ipoint).astype(bool),
             True,
             axis=-1)
ipoint_not_single_mask = np.diff(d) + np.logical_not(d[:-1])

indices_point_camintrinsics_camextrinsics = \
    indices_point_camintrinsics_camextrinsics[ipoint_not_single_mask]
observations_true = \
    observations_true[ipoint_not_single_mask]

# I just threw away points with a single observation, which created some gaps in
# the point arrays. I reorder the point array and the array referencing it so
# that the remaining points are indexed with sequential integers, starting at 0

Npoints_old = Npoints

ipoint = indices_point_camintrinsics_camextrinsics[:,0]
ipoint_unique = np.unique(ipoint)
new_point_delta = np.zeros((Npoints_old,), dtype=int)
new_point_delta[ipoint_unique] = 1

# shape (Npoints_old,)
# For each i=ipoint_old I have ipoint_new = index_ipoint_new[i]
index_ipoint_new = np.cumsum(new_point_delta) - 1
# shape (Npoints_new,)
# For each i=ipoint_new I have ipoint_old = index_ipoint_old[i]
index_ipoint_old = ipoint_unique

indices_point_camintrinsics_camextrinsics[:,0] = \
    index_ipoint_new[indices_point_camintrinsics_camextrinsics[:,0]]
points_true = points_true[index_ipoint_old]

Npoints = len(ipoint_unique)

r_cam_ref_noise_rad = 0.1
t_cam_ref_noise_m   = 0.5
r_cam_ref_noise = \
    ((np.random.random_sample(rt_cam_ref_true[:,:3].shape) * 2) - 1) * r_cam_ref_noise_rad
t_cam_ref_noise = \
    ((np.random.random_sample(rt_cam_ref_true[:,3:].shape) * 2) - 1) * t_cam_ref_noise_m
# first camera is at the origin
r_cam_ref_noise[0] *= 0
t_cam_ref_noise[0] *= 0

rt_cam_ref_noisy = rt_cam_ref_true.copy()
rt_cam_ref_noisy[:,:3] += r_cam_ref_noise
rt_cam_ref_noisy[:,3:] += t_cam_ref_noise

observations_noise_pixels = 2
observations_noise = \
    ((np.random.random_sample(observations_true.shape) * 2) - 1) * observations_noise_pixels
observations_noisy = observations_true + observations_noise


observations = np.array(observations_noisy)
rt_cam_ref   = np.array(rt_cam_ref_noisy)


if nps.norm2(rt_cam_ref[0]) != 0:
    print("First camera is assumed to sit at the origin, but it isn't there",
          file=sys.stderr)
    sys.exit()

# Add weight column. All weights are 1.0
observations_triangulated = nps.glue(observations,
                                     np.ones(observations.shape[:-1] + (1,),
                                             dtype = np.float32),
                                     axis = -1)

optimization_inputs = \
    dict( intrinsics            = nps.atleast_dims(m.intrinsics()[1], -2),
          rt_cam_ref            = rt_cam_ref[1:], # I made sure camera0 is at the origin

          observations_point_triangulated                        = observations_triangulated,
          indices_point_triangulated_camintrinsics_camextrinsics = indices_point_camintrinsics_camextrinsics,

          lensmodel                           = m.intrinsics()[0],
          imagersizes                         = nps.atleast_dims(m.imagersize(), -2),
          do_optimize_intrinsics_core         = False,
          do_optimize_intrinsics_distortions  = False,
          do_optimize_extrinsics              = True,
          do_optimize_frames                  = True,
          do_apply_outlier_rejection          = False,
          do_apply_regularization             = True,
          do_apply_regularization_unity_cam01 = True,
          verbose                             = False)

stats = mrcal.optimize(**optimization_inputs)

### We have a 5DOF solve. We rescale it to match the original
rt_cam_ref[:,3:] *= np.sqrt( nps.norm2(rt_cam_ref_true[-1,3:])/nps.norm2(rt_cam_ref[-1,3:]) )

if False:
    mrcal.show_geometry( nps.glue(rt_cam_ref, rt_cam_ref_true, axis=-2),
                         cameranames=[f"cam{i}-{what}" for what in ("solved","true") for i in range(Ncamera_poses)])

rt_err = mrcal.compose_rt( rt_cam_ref, rt_cam_ref_true, inverted1=True)

err_r_deg = nps.mag(rt_err[:,:3]) * 180./np.pi
err_t     = nps.mag(rt_err[:,3:])

testutils.confirm_equal(err_r_deg, 0,
                        worstcase = True,
                        eps = 2.,
                        msg = f"Recovered rotation")
testutils.confirm_equal(err_t, 0,
                        worstcase = True,
                        eps = 0.1,
                        msg = f"Recovered translation")


testutils.finish()
