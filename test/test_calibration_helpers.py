#!/usr/bin/python3

import sys
import numpy as np
import numpysane as nps
import copy
import os
import re

# I import the LOCAL mrcal since that's what I'm testing
testdir = os.path.dirname(os.path.realpath(__file__))
sys.path[:0] = f"{testdir}/..",
import mrcal
import mrcal.utils

def sample_dqref(observations,
                 pixel_uncertainty_stdev,
                 make_outliers = False):

    # Outliers have weight < 0. The code will adjust the outlier observations
    # also. But that shouldn't matter: they're outliers so those observations
    # should be ignored
    weight  = observations[...,-1]
    q_noise = np.random.randn(*observations.shape[:-1], 2) * pixel_uncertainty_stdev / nps.mv(nps.cat(weight,weight),0,-1)

    if make_outliers:
        if not hasattr(sample_dqref, 'idx_outliers_ref_flat'):
            NobservedPoints = observations.size // 3
            sample_dqref.idx_outliers_ref_flat = \
                np.random.choice( NobservedPoints,
                                  (NobservedPoints//100,), # 1% outliers
                                  replace = False )
        nps.clump(q_noise, n=3)[sample_dqref.idx_outliers_ref_flat, :] *= 20

    observations_perturbed = observations.copy()
    observations_perturbed[...,:2] += q_noise
    return q_noise, observations_perturbed


def grad(f, x, step=1e-6):
    r'''Computes df/dx at x

    f is a function of one argument. If the input has shape Si and the output
    has shape So, the returned gradient has shape So+Si. This applies central
    differences

    '''

    d     = np.zeros(x.shape,dtype=float)
    dflat = d.ravel()

    def df_dxi(i, d,dflat):

        dflat[i] = step
        fplus  = f(x+d)
        fminus = f(x-d)
        j = (fplus-fminus)/(2.*step)
        dflat[i] = 0
        return j

    # grad variable is in first dim
    Jflat = nps.cat(*[df_dxi(i, d,dflat) for i in range(len(dflat))])
    # grad variable is in last dim
    Jflat = nps.mv(Jflat, 0, -1)
    return Jflat.reshape( Jflat.shape[:-1] + d.shape )


def plot_arg_covariance_ellipse(q_mean, Var, what):

    # if the variance is 0, the ellipse is infinitely small, and I don't even
    # try to plot it. Gnuplot has the arguably-buggy behavior where drawing an
    # ellipse with major_diam = minor_diam = 0 plots a nominally-sized ellipse.
    if np.max(np.abs(Var)) == 0:
        return None

    l,v   = mrcal.utils._sorted_eig(Var)
    l0,l1 = l
    v0,v1 = nps.transpose(v)

    major = np.sqrt(l0)
    minor = np.sqrt(l1)

    return \
      (q_mean[0], q_mean[1], 2*major, 2*minor, 180./np.pi*np.arctan2(v0[1],v0[0]),
       dict(_with='ellipses', tuplesize=5, legend=what))


def plot_args_points_and_covariance_ellipse(q, what):
    q_mean  = np.mean(q,axis=-2)
    q_mean0 = q - q_mean
    Var     = np.mean( nps.outer(q_mean0,q_mean0), axis=0 )
    return ( plot_arg_covariance_ellipse(q_mean,Var, what),
             ( q, dict(_with = 'points pt 6 ps 0.5',
                         tuplesize = -2)) )


def calibration_baseline(model, Ncameras, Nframes, extra_observation_at,
                         pixel_uncertainty_stdev,
                         object_width_n,
                         object_height_n,
                         object_spacing,
                         extrinsics_rt_fromref_true,
                         calobject_warp_true,
                         fixedframes,
                         testdir,
                         cull_left_of_center = False,
                         allow_nonidentity_cam0_transform = False):
    r'''Compute a calibration baseline as a starting point for experiments

This is a perfect, noiseless solve. Regularization IS enabled, and the returned
model is at the optimization optimum. So the returned models will not sit
exactly at the ground-truth.

NOTE: if not fixedframes: the ref frame in the returned
optimization_inputs_baseline is NOT the ref frame used by the returned
extrinsics and frames arrays. The arrays in optimization_inputs_baseline had to
be transformed to reference off camera 0. If the extrinsics of camera 0 are the
identity, then the two ref coord systems are the same. To avoid accidental bugs,
we have a kwarg allow_nonidentity_cam0_transform, which defaults to False. if
not allow_nonidentity_cam0_transform and norm(extrinsics_rt_fromref_true[0]) >
0: raise

This logic is here purely for safety. A caller that handles non-identity cam0
transforms has to explicitly say that

ARGUMENTS

- model: string. 'opencv4' or 'opencv8' or 'splined'

- ...

    '''

    if re.match('opencv',model):
        models_true = ( mrcal.cameramodel(f"{testdir}/data/cam0.opencv8.cameramodel"),
                        mrcal.cameramodel(f"{testdir}/data/cam0.opencv8.cameramodel"),
                        mrcal.cameramodel(f"{testdir}/data/cam1.opencv8.cameramodel"),
                        mrcal.cameramodel(f"{testdir}/data/cam1.opencv8.cameramodel") )

        if model == 'opencv4':
            # I have opencv8 models_true, but I truncate to opencv4 models_true
            for m in models_true:
                m.intrinsics( intrinsics = ('LENSMODEL_OPENCV4', m.intrinsics()[1][:8]))
    elif model == 'splined':
        models_true = ( mrcal.cameramodel(f"{testdir}/data/cam0.splined.cameramodel"),
                        mrcal.cameramodel(f"{testdir}/data/cam0.splined.cameramodel"),
                        mrcal.cameramodel(f"{testdir}/data/cam1.splined.cameramodel"),
                        mrcal.cameramodel(f"{testdir}/data/cam1.splined.cameramodel") )
    else:
        raise Exception("Unknown lens being tested")

    models_true = models_true[:Ncameras]
    lensmodel   = models_true[0].intrinsics()[0]
    Nintrinsics = mrcal.lensmodel_num_params(lensmodel)

    for i in range(Ncameras):
        models_true[i].extrinsics_rt_fromref(extrinsics_rt_fromref_true[i])

    if not allow_nonidentity_cam0_transform and \
       nps.norm2(extrinsics_rt_fromref_true[0]) > 0:
        raise Exception("A non-identity cam0 transform was given, but the caller didn't explicitly say that they support this")

    imagersizes = nps.cat( *[m.imagersize() for m in models_true] )

    # These are perfect
    intrinsics_true         = nps.cat( *[m.intrinsics()[1]         for m in models_true] )
    extrinsics_true_mounted = nps.cat( *[m.extrinsics_rt_fromref() for m in models_true] )
    x_center = -(Ncameras-1)/2.

    # shapes (Nframes, Ncameras, Nh, Nw, 2),
    #        (Nframes, 4,3)
    q_true,Rt_ref_board_true = \
        mrcal.synthesize_board_observations(models_true,
                                            object_width_n, object_height_n, object_spacing,
                                            calobject_warp_true,
                                            np.array((0.,             0.,             0.,             x_center, 0,   4.0)),
                                            np.array((np.pi/180.*30., np.pi/180.*30., np.pi/180.*20., 2.5,      2.5, 2.0)),
                                            Nframes)


    if extra_observation_at:
        c = mrcal.ref_calibration_object(object_width_n,
                                         object_height_n,
                                         object_spacing,
                                         calobject_warp_true)
        Rt_cam0_board_true_far = \
            nps.glue( np.eye(3),
                      np.array((0,0,extra_observation_at)),
                      axis=-2)
        Rt_ref_board_true_far = mrcal.compose_Rt(models_true[0].extrinsics_Rt_toref(),
                                                 Rt_cam0_board_true_far)
        q_true_far = \
            mrcal.project(mrcal.transform_point_Rt(Rt_cam0_board_true_far, c),
                          *models_true[0].intrinsics())

        q_true            = nps.glue( q_true_far, q_true, axis=-5)
        Rt_ref_board_true = nps.glue( Rt_ref_board_true_far, Rt_ref_board_true, axis=-3)

        Nframes += 1

    frames_true = mrcal.rt_from_Rt(Rt_ref_board_true)

    ############# I have perfect observations in q_true. I corrupt them by noise
    # weight has shape (Nframes, Ncameras, Nh, Nw),
    weight01 = (np.random.rand(*q_true.shape[:-1]) + 1.) / 2. # in [0,1]
    weight0 = 0.2
    weight1 = 1.0
    weight = weight0 + (weight1-weight0)*weight01

    if cull_left_of_center:

        imagersize = models_true[0].imagersize()
        for m in models_true[1:]:
            if np.any(m.imagersize() - imagersize):
                raise Exception("I'm assuming all cameras have the same imager size, but this is false")

        weight[q_true[...,0] < imagersize[0]/2.] /= 1000.

    # I want observations of shape (Nframes*Ncameras, Nh, Nw, 3) where each row is
    # (x,y,weight)
    observations_true = nps.clump( nps.glue(q_true,
                                            nps.dummy(weight,-1),
                                            axis=-1),
                                  n=2)


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

    ###########################################################################
    # Now I apply pixel noise, and look at the effects on the resulting calibration.


    # p = mrcal.show_geometry(models_true,
    #                         frames          = frames_true,
    #                         object_width_n  = object_width_n,
    #                         object_height_n = object_height_n,
    #                         object_spacing  = object_spacing)
    # sys.exit()


    # I now reoptimize the perfect-observations problem. Without regularization,
    # this is a no-op: I'm already at the optimum. With regularization, this will
    # move us a certain amount (that the test will evaluate). Then I look at
    # noise-induced motions off this optimization optimum
    optimization_inputs_baseline = \
        dict( intrinsics                                = copy.deepcopy(intrinsics_true),
              points                                    = None,
              observations_board                        = observations_true,
              indices_frame_camintrinsics_camextrinsics = indices_frame_camintrinsics_camextrinsics,
              observations_point                        = None,
              indices_point_camintrinsics_camextrinsics = None,
              lensmodel                                 = lensmodel,
              calobject_warp                            = copy.deepcopy(calobject_warp_true),
              imagersizes                               = imagersizes,
              calibration_object_spacing                = object_spacing,
              verbose                                   = False,
              observed_pixel_uncertainty                = pixel_uncertainty_stdev,
              do_optimize_frames                        = not fixedframes,
              do_optimize_intrinsics_core               = False if model =='splined' else True,
              do_optimize_intrinsics_distortions        = True,
              do_optimize_extrinsics                    = True,
              do_optimize_calobject_warp                = True,
              do_apply_regularization                   = True,
              do_apply_outlier_rejection                = False)

    if fixedframes:
        # Frames are fixed: each camera has an independent pose
        optimization_inputs_baseline['extrinsics_rt_fromref'] = \
            copy.deepcopy(extrinsics_true_mounted)
        optimization_inputs_baseline['frames_rt_toref'] = copy.deepcopy(frames_true)
    else:
        # Frames are NOT fixed: cam0 is fixed as the reference coord system. I
        # transform each optimization extrinsics vector to be relative to cam0
        optimization_inputs_baseline['extrinsics_rt_fromref'] = \
            mrcal.compose_rt(extrinsics_true_mounted[1:,:],
                             mrcal.invert_rt(extrinsics_true_mounted[0,:]))
        optimization_inputs_baseline['frames_rt_toref'] = \
            mrcal.compose_rt(extrinsics_true_mounted[0,:], frames_true)

    mrcal.optimize(**optimization_inputs_baseline)

    models_baseline = \
        [ mrcal.cameramodel( optimization_inputs = optimization_inputs_baseline,
                             icam_intrinsics     = i) \
          for i in range(Ncameras) ]

    return                                                     \
        optimization_inputs_baseline,                          \
        models_true, models_baseline,                          \
        indices_frame_camintrinsics_camextrinsics,             \
        lensmodel, Nintrinsics, imagersizes,                   \
        intrinsics_true, extrinsics_true_mounted, frames_true, \
        observations_true,                                     \
        Nframes


def calibration_sample(Nsamples, Ncameras, Nframes,
                       Nintrinsics,
                       optimization_inputs_baseline,
                       observations_true,
                       pixel_uncertainty_stdev,
                       fixedframes):

    intrinsics_sampled         = np.zeros((Nsamples,Ncameras,Nintrinsics), dtype=float)
    extrinsics_sampled_mounted = np.zeros((Nsamples,Ncameras,6),           dtype=float)
    frames_sampled             = np.zeros((Nsamples,Nframes, 6),           dtype=float)
    calobject_warp_sampled     = np.zeros((Nsamples,2),                    dtype=float)

    for isample in range(Nsamples):
        if (isample+1) % 20 == 0:
            print(f"Sampling {isample+1}/{Nsamples}")

        optimization_inputs = copy.deepcopy(optimization_inputs_baseline)
        optimization_inputs['observations_board'] = \
            sample_dqref(observations_true, pixel_uncertainty_stdev)[1]
        mrcal.optimize(**optimization_inputs)

        intrinsics_sampled    [isample,...] = optimization_inputs['intrinsics']
        frames_sampled        [isample,...] = optimization_inputs['frames_rt_toref']
        calobject_warp_sampled[isample,...] = optimization_inputs['calobject_warp']
        if fixedframes:
            extrinsics_sampled_mounted[isample,   ...] = optimization_inputs['extrinsics_rt_fromref']
        else:
            # the remaining row is already 0
            extrinsics_sampled_mounted[isample,1:,...] = optimization_inputs['extrinsics_rt_fromref']

    return                            \
        ( intrinsics_sampled,         \
          extrinsics_sampled_mounted, \
          frames_sampled,             \
          calobject_warp_sampled )
