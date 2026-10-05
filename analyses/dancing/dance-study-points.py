#!/usr/bin/env python3


import sys
import argparse
import re
import os

import numpy as np
import numpysane as nps
import gnuplotlib as gp

# I import the LOCAL mrcal
scriptdir = os.path.dirname(os.path.realpath(__file__))
testdir  = f"{scriptdir}/../../test"
sys.path[:0] = f"{scriptdir}/../..",
import mrcal



# model = mrcal.cameramodel('2.uncertainty.1186/tst.cameramodel')
# mrcal.show_projection_uncertainty(model,
#                                   gridn_width = 10,
#                                   cbmax = 30,
#                                   wait=True)
# sys.exit()


np.random.seed(0)

model = mrcal.cameramodel(f"{testdir}/data/cam0.splined.cameramodel")
W,H = model.imagersize()


observed_pixel_uncertainty = 0.3






# The world frame has the ground at z=0. It is xyz ~ North,East,down

if False:
    Nobservations_total     = 4000
    Npoint_observations_min = 6
    Ncam_observing_min      = 4

    # camera is upright, facing DOWN, 5m above the ground
    Rt_NED_cam0 = np.array(((1, 0, 0),
                            (0, 1, 0),
                            (0, 0, 1),
                            (0, 0, -5.)))
    R_cam_camnext           = mrcal.R_from_r( np.array((0.01, 0.02, 0.25),) )
    t_cam_camnext__world    = np.array((1.5, 0.1, -0.4))
    rt_ref_cam__callback    = None

    track_length            = 20
    gridn                   = 4
    Nobservations_image     = 10
    p_NED_static            = None

else:
    Nobservations_total     = 1000
    Npoint_observations_min = 6
    Ncam_observing_min      = 4

    Rt_NED_cam0             = None
    R_cam_camnext           = None
    t_cam_camnext__world    = None

    def rt_ref_cam__new(iframe):
        r'''I want to yaw and oscillate up/down and pitch back/forth at
        different rates'''

        # camera starts out upright, facing DOWN, 5m above the ground
        r_NED_cam0 = mrcal.identity_r()
        t_NED_cam0 = np.array((0, 0, -5.))

        yawrate_frames_rotation = 40
        updownrate_frames_cycle = 37
        updown_amplitude        = 2.
        pitchrate_frames_cycle  = 43
        pitch_amplitude         = 20. * np.pi/180.

        yaw    = 2.*np.pi * iframe / yawrate_frames_rotation
        dz     = updown_amplitude * np.sin(2.*np.pi * iframe / updownrate_frames_cycle)
        dpitch = pitch_amplitude  * np.sin(2.*np.pi * iframe / pitchrate_frames_cycle)

        ryaw   = np.array((0,      0, yaw))
        rpitch = np.array((dpitch, 0, 0))

        r_NED_cam = mrcal.compose_r(ryaw, r_NED_cam0, rpitch)
        t_NED_cam = t_NED_cam0 + np.array((0,0,dz))

        return nps.glue( r_NED_cam, t_NED_cam, axis=-1)

    rt_ref_cam__callback    = rt_ref_cam__new

    track_length            = None
    gridn                   = None
    Nobservations_image     = None

    # The points are in a regular grid on the ground
    N = 5
    x = np.linspace(-3, 3, N)
    y = np.linspace(-3, 3, N)
    # shape (Ny,Nx,2); each slice is (x,y)
    pxy = \
        nps.mv( np.array(np.meshgrid(x,y)),
                0, -1 )
    # shape (Ny,Nx,3); each slice is (x,y,0)
    p_NED_static = nps.glue(pxy, np.zeros(pxy.shape[:-1] + (1,)),
                            axis = -1)
    # shape (Ny*Nx,3); each slice is (x,y,0)
    p_NED_static = nps.clump(p_NED_static, n=2)







(indices_point_camintrinsics_camextrinsics,
 points,
 rt_cam_ref,
 observations_point) = \
    mrcal.make_tracks(model,
                      Nobservations_total     = Nobservations_total, # I aim for this
                      Npoint_observations_min = Npoint_observations_min,
                      Ncam_observing_min      = Ncam_observing_min,

                      rt_ref_cam__callback    = rt_ref_cam__callback,
                      Rt_ref_cam0             = Rt_NED_cam0,
                      R_cam_camnext           = R_cam_camnext,
                      t_cam_camnext__world    = t_cam_camnext__world,

                      track_length            = track_length,
                      gridn                   = gridn,
                      Nobservations_image     = Nobservations_image, # desired feature density
                      p_static                = p_NED_static,)

_,point_counts = np.unique(indices_point_camintrinsics_camextrinsics[:,0], return_counts=True)
plot_point_counts = gp.gnuplotlib(xlabel = "Point count",
                                  ylabel = "How many points have been seen this many times",
                                  title  = "Track lengths")
plot_point_counts.plot(point_counts,
                       histogram = True)


print(f"{len(rt_cam_ref)=}")
# Re-reference to cam0
rt_cam_cam0 = \
    mrcal.compose_rt(rt_cam_ref[1:],
                     rt_cam_ref[0],
                     inverted1=True)
points_cam0 = mrcal.transform_point_rt(rt_cam_ref[0], points)
rt_cam_ref = rt_cam_cam0
points     = points_cam0
indices_point_camintrinsics_camextrinsics[:,2] -= 1

# make unity_cam01 fit perfectly
d = nps.mag(rt_cam_ref[0,3:])
rt_cam_ref[:,3:] /= d
points           /= d

optimization_inputs = \
    dict(
        lensmodel                                 = model.intrinsics()[0],
        intrinsics                                = nps.atleast_dims(model.intrinsics()[1], -2),
        imagersizes                               = nps.atleast_dims(model.imagersize(),    -2),
        indices_point_camintrinsics_camextrinsics = indices_point_camintrinsics_camextrinsics,
        points                                    = points,
        rt_cam_ref                                = rt_cam_ref,
        observations_point                        = observations_point,

        do_apply_regularization_unity_cam01 = True,
        do_apply_regularization             = True,
        do_optimize_intrinsics_core         = False,
        do_optimize_intrinsics_distortions  = True,
        do_optimize_extrinsics              = True,
        do_optimize_frames                  = True,
    )
imeas_regularization = mrcal.measurement_index_regularization(**optimization_inputs)
b,x = mrcal.optimizer_callback(**optimization_inputs,
                               no_factorization = True,
                               no_jacobian      = True)[:2]
if np.max(np.abs(x[:imeas_regularization])) > 2e-8:
    raise Exception("Simulated data isn't perfect")

# from make_perfect_observations()
for what in ('observations_board','observations_point'):

    if what in optimization_inputs and \
       optimization_inputs[what] is not None and \
       optimization_inputs[what].size:

        noise_nominal = \
            observed_pixel_uncertainty * \
            np.random.randn(*optimization_inputs[what][...,:2].shape)

        weight = nps.dummy( optimization_inputs[what][...,2],
                            axis = -1 )
        weight[ weight<=0 ] = 1. # to avoid dividing by 0

        optimization_inputs[what][...,:2] += \
            noise_nominal / weight



b,x = mrcal.optimizer_callback(**optimization_inputs,
                               no_factorization = True,
                               no_jacobian      = True)[:2]
if np.max(np.abs(x[:imeas_regularization])) < 2e-8:
    raise Exception("Simulated data is perfect after adding noise")


plot_features = gp.gnuplotlib(_xrange = [0,W-1],
                              _yrange = [H-1,0],
                              unset = 'key',
                              square=True)
plot_features.plot(*[ ( observations_point[indices_point_camintrinsics_camextrinsics[:,0] == ipoint,:2],
                        dict(legend = ipoint) ) \
                      for ipoint in range(len(points))],
                   tuplesize = -2)


mrcal.optimize(**optimization_inputs,
               verbose=True)

b,x = mrcal.optimizer_callback(**optimization_inputs,
                               no_factorization = True,
                               no_jacobian      = True)[:2]
gp.plot(x,
        _set = mrcal.plotoptions_measurement_boundaries(**optimization_inputs))


model = mrcal.cameramodel(optimization_inputs = optimization_inputs,
                          icam_intrinsics = 0,
                          icam_extrinsics = -1)
model.write("/tmp/tst.cameramodel")
mrcal.show_projection_uncertainty(model,
                                  gridn_width = 30,
                                  cbmax = 30,
                                  wait=True)



r'''
add some randomness so that all of the features don't all disappear at the
same time

The simple model has odd-looking tracks. Mostly circular (makes sense), but some linear around the edges (does not make sense)

'''
