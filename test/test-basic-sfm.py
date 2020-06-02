#!/usr/bin/python3

r'''Basic structure-from-motion test

I observe, with noise, a number of points from various angles with a single
camera, and I make sure that I can accurately compute the locations of the
points

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
m = mrcal.cameramodel(f"{testdir}/data/cam0.opencv8.cameramodel")
imagersize = m.imagersize()
lensmodel,intrinsics_data = m.intrinsics()

ref_p = np.array((( 10.,  20., 100.),
                  ( 25.,  30.,  90.),
                  (  5.,  10.,  94.),
                  (-45., -20.,  95.),
                  (-35.,  14.,  77.),
                  (  5.,  -0., 110.),
                  (  1.,  50.,  50.)))

# The points are all somewhere at +z. So the Camera poses are all ~ identity
ref_extrinsics_rt_fromref = np.array(((-0.1, -0.07, 0.01,  10.0, 4.0, -7.0),
                                      (-0.01, 0.05,-0.02,  30.0,-8.0, -8.0),
                                      (-0.1,  0.03,-0.03,  10.0,-9.0, 20.0),
                                      ( 0.04,-0.04, 0.03, -20.0, 2.0,-11.0),
                                      ( 0.01, 0.05,-0.05, -10.0, 3.0,  9.0)))

# shape (Ncamposes, Npoints, 3)
ref_p_cam = mrcal.transform_point_rt(nps.mv(ref_extrinsics_rt_fromref, -2,-3),
                                     ref_p)

# shape (Ncamposes, Npoints, 2)
ref_q_cam = mrcal.project(ref_p_cam, lensmodel, intrinsics_data)

# Observations are incomplete. Not all points are observed from everywhere
indices_point_camintrinsics_camextrinsics = \
    np.array(((0, 0, 1),
              (0, 0, 2),
              (0, 0, 4),
              (1, 0, 0),
              (1, 0, 1),
              (1, 0, 4),
              (2, 0, 0),
              (2, 0, 1),
              (2, 0, 2),
              (3, 0, 1),
              (3, 0, 2),
              (3, 0, 3),
              (3, 0, 4),
              (4, 0, 0),
              (4, 0, 3),
              (4, 0, 4),
              (5, 0, 0),
              (5, 0, 1),
              (5, 0, 2),
              (5, 0, 3),
              (5, 0, 4),
              (6, 0, 2),
              (6, 0, 3),
              (6, 0, 4)),
             dtype = np.int32)


def make_noisy_inputs():
    r'''Construct incomplete, noisy observations to feed to the solver'''
    # The seed points array is the true array, but corrupted by noise. All the
    # points are observed at some point
    #print(repr((np.random.random(points.shape)-0.5)/3))
    points_noise = np.array([[-0.16415198,  0.10697666,  0.07137079],
                             [-0.02353459,  0.07269802,  0.05804911],
                             [-0.05218085, -0.09302461, -0.16626839],
                             [ 0.03649283, -0.04345566, -0.1589429 ],
                             [-0.05530528,  0.03942736, -0.02755858],
                             [-0.16252387,  0.07792151, -0.12200266],
                             [-0.02611094, -0.13695699,  0.06799326]])
    points_noisy = ref_p * (1. + points_noise)

    Ncamposes,Npoints = ref_p_cam.shape[:2]
    ipoints   = indices_point_camintrinsics_camextrinsics[:,0]
    icamposes = indices_point_camintrinsics_camextrinsics[:,2]
    ref_q_cam_indexed = nps.clump(ref_q_cam, n=2)[icamposes*Npoints+ipoints,:]

    #print(repr(np.random.randn(*ref_q_cam_indexed.shape) * 1.0))
    q_cam_noise = np.array([[-0.40162837, -0.60884836],
                            [-0.65186956, -2.23240529],
                            [ 0.40217293, -0.40160168],
                            [ 2.05376895, -1.47389235],
                            [-0.01090807,  0.35468639],
                            [-0.37916168, -1.06052742],
                            [-0.08546853, -2.69946391],
                            [ 0.76133345, -1.38759769],
                            [-1.05998307, -0.27779779],
                            [-2.22203688,  1.47809028],
                            [ 1.68526798,  0.83635394],
                            [ 1.26203342,  2.58905488],
                            [ 1.18282463, -0.41362789],
                            [ 0.41615768,  2.06621809],
                            [ 0.27271605,  1.19721072],
                            [-1.48421641,  3.20841776],
                            [ 1.10563011,  0.38313526],
                            [ 0.25591618, -0.97987565],
                            [-0.2431585 , -1.34797656],
                            [ 1.57805536, -0.26467537],
                            [ 1.23762306,  0.94616712],
                            [ 0.29441229, -0.78921128],
                            [-1.33799634, -1.65173241],
                            [-0.24854348, -0.14145806]])
    q_cam_indexed_noisy = ref_q_cam_indexed + q_cam_noise

    observations = nps.glue(q_cam_indexed_noisy,
                            nps.transpose(np.ones((q_cam_indexed_noisy.shape[0],))),
                            axis = -1)

    #print(repr((np.random.random(ref_extrinsics_rt_fromref.shape)-0.5)/10))
    extrinsics_rt_fromref_noise = \
        np.array([[-0.00781127, -0.04067386, -0.01039731,  0.02057068, -0.0461704 ,  0.02112582],
                  [-0.02466267, -0.01445134, -0.01290107, -0.01956848,  0.04604318,  0.0439563 ],
                  [-0.02335697,  0.03171099, -0.00900416, -0.0346394 , -0.0392821 ,  0.03892269],
                  [ 0.00229462, -0.01716853,  0.01336239, -0.0228473 , -0.03919978,  0.02671576],
                  [ 0.03782446, -0.016981  ,  0.03949906, -0.03256744,  0.02496247,  0.02924358]])
    extrinsics_rt_fromref_noisy = ref_extrinsics_rt_fromref * (1.0 + extrinsics_rt_fromref_noise)

    return extrinsics_rt_fromref_noisy, points_noisy, observations



############### Do everything in the ref coord system, with a few fixed-position
############### points to set the coords
extrinsics_rt_fromref, points, observations = make_noisy_inputs()

# De-noise the fixed points. We know where they are exactly. And correctly
Npoints_fixed = 3
points[-Npoints_fixed:, ...] = ref_p[-Npoints_fixed:, ...]

stats = mrcal.optimize( nps.atleast_dims(intrinsics_data, -2),
                        extrinsics_rt_fromref,
                        None, points,
                        None, None,
                        observations,
                        indices_point_camintrinsics_camextrinsics,
                        lensmodel,
                        imagersizes                       = nps.atleast_dims(imagersize, -2),
                        Npoints_fixed                     = Npoints_fixed,
                        point_min_range                   = 1.0,
                        point_max_range                   = 1000.0,
                        observed_pixel_uncertainty        = 1.0,
                        do_optimize_intrinsic_core        = False,
                        do_optimize_intrinsic_distortions = False,
                        do_optimize_extrinsics            = True,
                        do_optimize_frames                = True,
                        skip_outlier_rejection            = True,
                        skip_regularization               = False,
                        verbose                           = False)

# Got a solution. How well do they fit?
fit_rms = np.sqrt(np.mean(nps.norm2(points - ref_p)))

testutils.confirm_equal(fit_rms, 0,
                        msg = f"Solved at ref coords with known-position points",
                        eps = 1.0)

testutils.finish()
