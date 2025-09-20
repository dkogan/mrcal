#!/usr/bin/env python3

r'''Basic structure-from-motion test

I observe, with noise, a number of points from various angles with a single
camera, and I make sure that I can accurately compute the locations of the
points.

To sufficiently constrain the problem this test

- fixes the pose of the first camera
- uses the unity_cam01 regularization to set the scale of the problem

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
import test_sfm_helpers

add_outlier = False

############# Set up my world, and compute all the perfect positions, pixel
############# observations of everything
model,                \
imagersize,           \
lensmodel,            \
intrinsics_data,      \
indices_point_camera, \
                      \
pref_true,            \
rt_cam_ref_true,      \
qcam_true,            \
observations_true,    \
                      \
pref_noisy,           \
rt_cam_ref_noisy,     \
qcam_noisy,           \
observations_noisy =  \
    test_sfm_helpers.generate_world()

# The reference coordinate system is at the first camera. And all distances are
# normalized
indices_point_camintrinsics_camextrinsics = \
    nps.glue( indices_point_camera[:,(0,)],
              indices_point_camera[:,(0,)] * 0,
              indices_point_camera[:,(1,)] - 1,
              axis = -1 )

rt_ref_0_true = mrcal.invert_rt(rt_cam_ref_true[0])

rt10_true = mrcal.compose_rt( rt_cam_ref_true[1],
                              rt_ref_0_true )
cam01_distance_true = nps.mag(rt10_true[3:])

rt_cam_0_noisy = \
    mrcal.compose_rt( rt_cam_ref_noisy,
                      rt_ref_0_true )

p0_noisy = mrcal.transform_point_rt( mrcal.invert_rt(rt_ref_0_true),
                                     pref_noisy )

# normalize distances
rt_cam_0_noisy[:,3:] /= cam01_distance_true
p0_noisy             /= cam01_distance_true



if add_outlier:
    # Make an outlier point: the first observation x coord is way off
    observations_noisy[0,0] += 100.

optimization_inputs = \
    dict( intrinsics                                = nps.atleast_dims(intrinsics_data, -2),
          rt_cam_ref                                = rt_cam_0_noisy[1:],
          points                                    = p0_noisy,
          observations_point                        = observations_noisy,
          indices_point_camintrinsics_camextrinsics = indices_point_camintrinsics_camextrinsics,

          lensmodel                           = lensmodel,
          imagersizes                         = nps.atleast_dims(imagersize, -2),
          point_min_range                     = 0.1,
          point_max_range                     = 1000.0,
          do_optimize_intrinsics_core         = False,
          do_optimize_intrinsics_distortions  = False,
          do_optimize_extrinsics              = True,
          do_optimize_frames                  = True,
          do_apply_outlier_rejection          = False,
          do_apply_regularization_unity_cam01 = True,
          verbose                             = False)


stats = mrcal.optimize(**optimization_inputs)

pref_noisy_solved = mrcal.transform_point_rt( rt_ref_0_true,
                                              p0_noisy * cam01_distance_true )


# If we have outliers, I need to be able to detect it. If I can detect it,
# markOutliers() should do that, mark it, and re-optimize without that
# observation. This code (disabled currently) shows that we can't detect it for
# discrete points, so markOutliers() ignores discrete points. If you want to
# figure out how to do this right, start by re-enabling this code
if add_outlier:
    p,x,J,f = mrcal.optimizer_callback(**optimization_inputs)

    if 0: # print measurements x

        import gnuplotlib as gp
        gp.plot(x, _with='points pt 7', wait=1)
        sys.exit()

    elif 1: # print outlierness. Not 100% sure this code is right
        J = np.array(J.todense())

        outlierness = np.zeros((len(observations_noisy),), dtype=float)
        for iobservation in range(len(observations_noisy)):

            Ji = J[iobservation*3:iobservation*3+3, :]
            xi = x[iobservation*3:iobservation*3+3   ]

            # A = J* inv(JtJ) J*t
            # B = inv(A - I)
            # Dima's self+others:        x*t (-B      ) x*

            A = \
                nps.matmult( f.solve_xt_JtJ_bt(Ji),
                             nps.transpose(Ji) )
            B = np.linalg.inv(A - np.eye(3))

            outlierness[iobservation] = -nps.inner(xi, nps.inner(B,xi))


        import gnuplotlib as gp
        gp.plot(outlierness, _with='points pt 7', wait=1)
        sys.exit()




testutils.confirm_equal(pref_noisy_solved,
                        pref_true,
                        msg = f"Solved at ref coords with known-position points",
                        eps = 0.2)

testutils.finish()
