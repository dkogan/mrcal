#!/usr/bin/python3

r'''Basic structure-from-motion test

I observe, with noise, a number of points from various angles with a single
camera, and I make sure that I can accurately compute the locations of the
points

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


############# Set up my world, and compute all the perfect positions, pixel
############# observations of everything
m = mrcal.cameramodel(f"{testdir}/data/cam0.opencv8.cameramodel")
imagersize = m.imagersize()
lensmodel,intrinsics_data = m.intrinsics()

pref_true = np.array((( 10.,  20., 100.),
                      ( 25.,  30.,  90.),
                      (  5.,  10.,  94.),
                      (-45., -20.,  95.),
                      (-35.,  14.,  77.),
                      (  5.,  -0., 110.),
                      (  1.,  50.,  50.)))

# The points are all somewhere at +z. So the Camera rotations are all ~ identity
rt_cam_ref_true = np.array(((-0.1, -0.07, 0.01,  10.0, 4.0, -7.0),
                            (-0.01, 0.05,-0.02,  30.0,-8.0, -8.0),
                            (-0.1,  0.03,-0.03,  10.0,-9.0, 20.0),
                            ( 0.04,-0.04, 0.03, -20.0, 2.0,-11.0),
                            ( 0.01, 0.05,-0.05, -10.0, 3.0,  9.0)))

# shape (Ncamposes, Npoints, 3)
pcam_true = mrcal.transform_point_rt(nps.mv(rt_cam_ref_true, -2,-3),
                                pref_true)

# shape (Ncamposes, Npoints, 2)
qcam_true = mrcal.project(pcam_true, lensmodel, intrinsics_data)

# This many points are not optimized, and we know exactly where they are. We
# need this to set the scale of the problem. Otherwise the solve is ambiguous.
# These points appear at the END of the points list
Npoints_fixed = 3

# Observations are incomplete. Not all points are observed from everywhere. In
# particular, the last Npoints_fixed points are NOT observed by camera 0. The
# triangulated points alone determine its pose
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
              (4, 0, 3),
              (4, 0, 4),
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

    # The fixed points are perfect
    points_noise[-Npoints_fixed:, :] = 0

    pref_noisy = pref_true * (1. + points_noise)

    Ncamposes,Npoints = pcam_true.shape[:2]
    ipoints   = indices_point_camintrinsics_camextrinsics[:,0]
    icamposes = indices_point_camintrinsics_camextrinsics[:,2]
    qcam_indexed = nps.clump(qcam_true, n=2)[icamposes*Npoints+ipoints,:]

    #print(repr(np.random.randn(*qcam_indexed.shape) * 1.0))
    qcam_noise = np.array([[-0.40162837, -0.60884836],
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
                           [-0.2431585 , -1.34797656],
                           [ 1.57805536, -0.26467537],
                           [ 1.23762306,  0.94616712],
                           [ 0.29441229, -0.78921128],
                           [-1.33799634, -1.65173241],
                           [-0.24854348, -0.14145806]])
    qcam_indexed_noisy = qcam_indexed + qcam_noise

    observations = nps.glue(qcam_indexed_noisy,
                            nps.transpose(np.ones((qcam_indexed_noisy.shape[0],))),
                            axis = -1)

    #print(repr((np.random.random(rt_cam_ref_true.shape)-0.5)/10))
    rt_cam_ref_noise = \
        np.array([[-0.00781127, -0.04067386, -0.01039731,  0.02057068, -0.0461704 ,  0.02112582],
                  [-0.02466267, -0.01445134, -0.01290107, -0.01956848,  0.04604318,  0.0439563 ],
                  [-0.02335697,  0.03171099, -0.00900416, -0.0346394 , -0.0392821 ,  0.03892269],
                  [ 0.00229462, -0.01716853,  0.01336239, -0.0228473 , -0.03919978,  0.02671576],
                  [ 0.03782446, -0.016981  ,  0.03949906, -0.03256744,  0.02496247,  0.02924358]])
    rt_cam_ref_noisy = rt_cam_ref_true * (1.0 + rt_cam_ref_noise)

    return rt_cam_ref_noisy, pref_noisy, observations



############### Do everything in the ref coord system, with a few fixed-position
############### points to set the scale
rt_cam_ref_noisy, pref_noisy, observations = make_noisy_inputs()


# The TRAILING Npoints_fixed points are fixed. The leading ones are triangulatd

Npoints = len(pref_true)
idx_points_fixed = indices_point_camintrinsics_camextrinsics[:,0] >= Npoints-Npoints_fixed

indices_point_camintrinsics_camextrinsics_fixed = \
    indices_point_camintrinsics_camextrinsics[idx_points_fixed].copy()
indices_point_camintrinsics_camextrinsics_fixed[:,0] -= (Npoints-Npoints_fixed)
observations_fixed = observations[idx_points_fixed].copy()

pref_noisy = pref_noisy[-Npoints_fixed:]

indices_point_camintrinsics_camextrinsics_triangulated = \
    indices_point_camintrinsics_camextrinsics[~idx_points_fixed].copy()
observations_triangulated = observations[~idx_points_fixed].copy()


######### warning "triangulated-solve: mrcal.optimize() should accept kwargs only"


optimization_inputs = \
    dict( intrinsics            = nps.atleast_dims(intrinsics_data, -2),
          extrinsics_rt_fromref = rt_cam_ref_noisy,
          points                = pref_noisy,

          # Explicit points. Fixed only
          observations_point                                     = observations_fixed,
          indices_point_camintrinsics_camextrinsics              = indices_point_camintrinsics_camextrinsics_fixed,

          observations_point_triangulated                        = observations_triangulated,
          indices_point_triangulated_camintrinsics_camextrinsics = indices_point_camintrinsics_camextrinsics_triangulated,

          lensmodel                         = lensmodel,
          imagersizes                       = nps.atleast_dims(imagersize, -2),
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


#optimization_inputs['verbose'] = True
pref_noisy_orig = pref_noisy.copy()
stats = mrcal.optimize(**optimization_inputs)

import IPython
IPython.embed()
sys.exit()

# Got a solution. How well do they fit?
fit_rms = np.sqrt(np.mean(nps.norm2(pref_noisy - pref_true)))

testutils.confirm_equal(fit_rms, 0,
                        msg = f"Solved at ref coords with known-position points",
                        eps = 1.0)

testutils.finish()
