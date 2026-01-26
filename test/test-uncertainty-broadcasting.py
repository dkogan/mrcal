#!/usr/bin/env python3

r'''Triangulation and projection uncertainty broadcasting test

The uncertainty routines support broadcasting, which we evaluate here

'''

import sys
import argparse
import re
import os

def parse_args():

    parser = \
        argparse.ArgumentParser(description = __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--fixed',
                        type=str,
                        choices=('cam0','frames'),
                        default = 'cam0',
                        help='''Are we putting the origin at camera0, or are all the frames at a fixed (and
                        non-optimizeable) pose? One or the other is required.''')
    parser.add_argument('--model',
                        type=str,
                        choices=('opencv4','opencv8','splined'),
                        default = 'opencv4',
                        help='''Which lens model we're using. Must be one of
                        ('opencv4','opencv8','splined')''')
    parser.add_argument('--Nframes',
                        type=int,
                        default=50,
                        help='''How many chessboard poses to simulate. These are dense observations: every
                        camera sees every corner of every chessboard pose''')
    parser.add_argument('--stabilize-coords',
                        action = 'store_true',
                        help='''Whether we report the triangulation in the camera-0 coordinate system (which
                        is moving due to noise) or in a stabilized coordinate
                        system based on the frame poses''')
    parser.add_argument('--cull-left-of-center',
                        action = 'store_true',
                        help='''If given, the calibration data in the left half of the imager is thrown
                        out''')
    parser.add_argument('--q-calibration-stdev',
                        type    = float,
                        default = 0.0,
                        help='''The observed pixel uncertainty of the chessboard
                        observations at calibration time. Defaults to 0.0. At
                        least one of --q-calibration-stdev and
                        --q-observation-stdev MUST be given as > 0''')
    parser.add_argument('--q-observation-stdev',
                        type    = float,
                        default = 0.0,
                        help='''The observed pixel uncertainty of the point
                        observations at triangulation time. Defaults to 0.0. At
                        least one of --q-calibration-stdev and
                        --q-observation-stdev MUST be given as > 0''')
    parser.add_argument('--q-observation-stdev-correlation',
                        type    = float,
                        default = 0.0,
                        help='''By default, the noise in the observation-time
                        pixel observations is assumed independent. This isn't
                        entirely realistic: observations of the same feature in
                        multiple cameras originate from an imager correlation
                        operation, so they will have some amount of correlation.
                        If given, this argument specifies how much correlation.
                        This is a value in [0,1] scaling the stdev. 0 means
                        "independent" (the default). 1.0 means "100%%
                        correlated".''')
    parser.add_argument('--baseline',
                        type    = float,
                        default = 2.,
                        help='''The baseline of the camera pair. This is the
                        horizontal distance between each pair of adjacent
                        cameras''')

    args = parser.parse_args()

    if args.q_calibration_stdev <= 0.0 and \
       args.q_observation_stdev <= 0.0:
        raise Exception('At least one of --q-calibration-stdev and --q-observation-stdev MUST be given as > 0')
    return args


args = parse_args()

testdir = os.path.dirname(os.path.realpath(__file__))

# I import the LOCAL mrcal since that's what I'm testing
sys.path[:0] = f"{testdir}/..",
import mrcal
import testutils
import copy
import numpy as np
import numpysane as nps

from test_calibration_helpers import calibration_baseline



Ncameras = 3
Npoints  = 5

# shape (Npoints,3)
p_triangulated_true0 = np.zeros((Npoints,3), dtype=float)
p_triangulated_true0[:,0] = np.arange(Npoints)       - 2.77
p_triangulated_true0[:,1] = np.arange(Npoints) / 10. - 0.344
p_triangulated_true0[:,2] = 100.

############# Set up my world, and compute all the perfect positions, pixel
############# observations of everything
fixedframes = (args.fixed == 'frames')
object_spacing          = 0.1
object_width_n          = 10
object_height_n         = 9
calobject_warp_true     = np.array((0.002, -0.005))

# I want the RNG to be deterministic
np.random.seed(0)


rt_cam_ref_true = np.zeros((Ncameras,6), dtype=float)
rt_cam_ref_true[:,:3] = np.random.randn(Ncameras,3) * 0.1
rt_cam_ref_true[:, 3] = args.baseline * np.arange(Ncameras)
rt_cam_ref_true[:,4:] = np.random.randn(Ncameras,2) * 0.1

# cam0 is at the identity. This makes my life easy: I can assume that the
# optimization_inputs returned by calibration_baseline() use the same ref
# coordinate system as these transformations.
rt_cam_ref_true[0] *= 0

optimization_inputs_baseline, \
models_true,                  \
frames_true =                 \
    calibration_baseline(args.model,
                         Ncameras,
                         args.Nframes,
                         None,
                         object_width_n,
                         object_height_n,
                         object_spacing,
                         rt_cam_ref_true,
                         calobject_warp_true,
                         fixedframes,
                         testdir,
                         cull_left_of_center = args.cull_left_of_center)

models_baseline = \
    [ mrcal.cameramodel( optimization_inputs = optimization_inputs_baseline,
                         icam_intrinsics     = i) \
      for i in range(Ncameras) ]



####################################################################
# I have the calibrated models, and I can compute the triangulations
#
# I triangulate my 5 points while observed by cameras (2,1) and cameras (1,0)
icameras    = ( (2,1), (1,0) )
Nmodelpairs = len(icameras)

# shape (Nmodelpairs,2)
M = [ [models_baseline[i] for i in icameras[0]],
      [models_baseline[i] for i in icameras[1]] ]

# shape (Npoints,Nmodelpairs, 2,2)
q = np.zeros((Npoints,Nmodelpairs, 2,2), dtype=float)
for ipt in range(Npoints):
    for imp in range(Nmodelpairs):
        q[ipt,imp,0,:] = mrcal.project(p_triangulated_true0[ipt],
                                       *M[imp][0].intrinsics())
        q[ipt,imp,1,:] = mrcal.project(mrcal.transform_point_Rt( mrcal.compose_Rt( M[imp][1].Rt_cam_ref(),
                                                                                   M[imp][0].Rt_ref_cam() ),
                                                                 p_triangulated_true0[ipt]),
                                       *M[imp][1].intrinsics())

p, \
Var_p0p1_calibration_big, \
Var_p0p1_observation_big, \
Var_p0p1_joint_big = \
    mrcal.triangulate( q, M,
                       q_calibration_stdev             = args.q_calibration_stdev,
                       q_observation_stdev             = args.q_observation_stdev,
                       q_observation_stdev_correlation = args.q_observation_stdev_correlation,
                       stabilize_coords                = args.stabilize_coords )

testutils.confirm_equal(p.shape,
                        (Npoints,Nmodelpairs,3),
                        msg = "point array has the right shape")

testutils.confirm_equal(Var_p0p1_calibration_big.shape,
                        (Npoints,Nmodelpairs,3, Npoints,Nmodelpairs,3),
                        msg = "Big covariance (calibration) matrix has the right shape")
testutils.confirm_equal(Var_p0p1_observation_big.shape,
                        (Npoints,Nmodelpairs,3,3),
                        msg = "Big covariance (observation) matrix has the right shape")
testutils.confirm_equal(Var_p0p1_joint_big.shape,
                        (Npoints,Nmodelpairs,3, Npoints,Nmodelpairs,3),
                        msg = "Big covariance (joint) matrix has the right shape")

# Now I check each block in the diagonal individually
for ipt in range(Npoints):
    for imp in range(Nmodelpairs):
        p, \
        _, _, \
        Var_p0p1 = \
            mrcal.triangulate( q[ipt,imp], M[imp],
                               q_calibration_stdev             = args.q_calibration_stdev,
                               q_observation_stdev             = args.q_observation_stdev,
                               q_observation_stdev_correlation = args.q_observation_stdev_correlation,
                               stabilize_coords                = args.stabilize_coords )

        testutils.confirm_equal(Var_p0p1_joint_big[ipt,imp,:,ipt,imp,:],
                                Var_p0p1,
                                msg = f"Covariance (joint) sub-matrix ipt={ipt} imp={imp}")
        testutils.confirm_equal(Var_p0p1_calibration_big[ipt,imp,:,ipt,imp,:] + Var_p0p1_observation_big[ipt,imp,:,:],
                                Var_p0p1,
                                msg = f"Covariance (cal + obs) sub-matrix ipt={ipt} imp={imp}")

testutils.finish()
