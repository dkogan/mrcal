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


def grad(f, x,
         *,
         switch              = None,
         forward_differences = False,
         step                = 1e-6):
    r'''Computes df/dx at x

    f is a function of one argument. If the input has shape Si and the output
    has shape So, the returned gradient has shape So+Si. This computes forward
    differences.

    If the function being evaluated produces bimodal output, the step might move
    us to a different mode, giving a falsely-high difference. Use the "switch"
    argument to switch to the other mode in this case.

    '''

    if switch is not None and not forward_differences:
        raise Exception("switch works ONLY with forward differences")

    d     = np.zeros(x.shape,dtype=float)
    dflat = d.ravel()

    def df_dxi(i, d,dflat):

        if not forward_differences:
            # central differences
            dflat[i] = step
            fplus  = f(x+d)
            fminus = f(x-d)
            j = (fplus-fminus)/(2.*step)
            dflat[i] = 0
            return j

        else:
            # forward differences
            dflat[i] = step
            f0    = f(x)
            fplus = f(x+d)
            if switch is not None and nps.norm2(fplus-f0) > 1.:
                fplus = switch(fplus)
            j = (fplus-f0)/step
            dflat[i] = 0
            return j

    # grad variable is in first dim
    Jflat = nps.cat(*[df_dxi(i, d,dflat) for i in range(len(dflat))])
    # grad variable is in last dim
    Jflat = nps.mv(Jflat, 0, -1)
    return Jflat.reshape( Jflat.shape[:-1] + d.shape )


def calibration_baseline(model, Ncameras, Nframes, extra_observation_at,
                         object_width_n,
                         object_height_n,
                         object_spacing,
                         extrinsics_rt_fromref_true,
                         calobject_warp_true,
                         fixedframes,
                         testdir,
                         cull_left_of_center = False,
                         allow_nonidentity_cam0_transform = False,
                         range_to_boards = 4.0,
                         x_offset = 0.0,
                         x_noiseradius = 2.5,
                         y_noiseradius = 2.5,
                         x_mirror      = False, # half with +x_offset, half with -x_offset
                         report_points = False):
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

I always compute chessboard views. But if report_points: I store each corner
observation as a separate point. if report_points: I ALSO enable the unity_cam01
regularization to make the solve non-singular

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



    def synthesize(z, z_noiseradius, Nframes,
                   x_offset):
        return \
            mrcal.synthesize_board_observations(models_true,
                                                object_width_n                  = object_width_n,
                                                object_height_n                 = object_height_n,
                                                object_spacing                  = object_spacing,
                                                calobject_warp                  = calobject_warp_true,
                                                rt_ref_boardcenter              = np.array((0.,
                                                                                            0.,
                                                                                            0.,
                                                                                            x_center + x_offset,
                                                                                            0,
                                                                                            z)),
                                                rt_ref_boardcenter__noiseradius = np.array((np.pi/180.*30.,
                                                                                            np.pi/180.*30.,
                                                                                            np.pi/180.*20.,
                                                                                            x_noiseradius,
                                                                                            y_noiseradius,
                                                                                            z_noiseradius)),
                                                Nframes                         = Nframes,
                                                pcamera_nominal_ref             = np.array((x_center,0,0), dtype=float),
                                                max_oblique_angle_deg           = 30.)


    if not x_mirror:
        # shapes (Nframes, Ncameras, Nh, Nw, 2),
        #        (Nframes, 4,3)
        q_true,Rt_ref_board_true = \
            synthesize(z             = range_to_boards,
                       z_noiseradius = range_to_boards / 2.0,
                       Nframes       = Nframes,
                       x_offset      = x_offset)
    else:
        if Nframes & 1:
            raise Exception("x_mirror is True, so Nframes must be even")

        # shapes (Nframes//2, Ncameras, Nh, Nw, 2),
        #        (Nframes//2, 4,3)
        q_true0,Rt_ref_board_true0 = \
            synthesize(z             = range_to_boards,
                       z_noiseradius = range_to_boards / 2.0,
                       Nframes       = Nframes//2,
                       x_offset      = x_offset)
        q_true1,Rt_ref_board_true1 = \
            synthesize(z             = range_to_boards,
                       z_noiseradius = range_to_boards / 2.0,
                       Nframes       = Nframes//2,
                       x_offset      = -x_offset)

        q_true            = nps.glue(q_true0,           q_true1,            axis=-5)
        Rt_ref_board_true = nps.glue(Rt_ref_board_true0,Rt_ref_board_true1, axis=-3)


    if extra_observation_at is not None:
        q_true_extra,Rt_ref_board_true_extra = \
            synthesize(z             = extra_observation_at,
                       z_noiseradius = extra_observation_at / 10.0,
                       Nframes       = 1,
                       x_offset      = x_offset)


        q_true            = nps.glue( q_true, q_true_extra,
                                      axis=-5)
        Rt_ref_board_true = nps.glue( Rt_ref_board_true, Rt_ref_board_true_extra,
                                      axis=-3)

        Nframes += 1

    # shape (Nframes, 6)
    frames_true = mrcal.rt_from_Rt(Rt_ref_board_true)

    if not fixedframes:
        # Frames are NOT fixed: cam0 is fixed as the reference coord system. I
        # transform each optimization extrinsics vector to be relative to cam0
        frames_true = mrcal.compose_rt(extrinsics_true_mounted[0,:], frames_true)


    ############# I have perfect observations in q_true.
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
    observations_board_true = \
        nps.clump( nps.glue(q_true,
                            nps.dummy(weight,-1),
                            axis=-1),
                   n=2)

    # Dense observations. All the cameras see all the boards
    indices_frame_camera = np.zeros( (Nframes,Ncameras,2), dtype=np.int32)
    indices_frame_camera[...,0] += nps.transpose(np.arange(Nframes, dtype=np.int32))
    indices_frame_camera[...,1] += np.arange(Ncameras, dtype=np.int32)
    # shape (Nframes*Ncameras, 2)
    indices_frame_camera = nps.clump(indices_frame_camera, n=2)

    # shape (Nframes*Ncameras, 3)
    indices_frame_camintrinsics_camextrinsics = \
        nps.glue(indices_frame_camera,
                 indices_frame_camera[:,(1,)],
                 axis=-1)
    if not fixedframes:
        indices_frame_camintrinsics_camextrinsics[:,2] -= 1



    # I now reoptimize the perfect-observations problem. Without regularization,
    # this is a no-op: I'm already at the optimum. With regularization, this will
    # move us a certain amount (that the test will evaluate). Then I look at
    # noise-induced motions off this optimization optimum
    optimization_inputs_baseline = \
        dict( intrinsics                                = copy.deepcopy(intrinsics_true),
              frames_rt_toref                           = None,
              points                                    = None,
              observations_board                        = None,
              indices_frame_camintrinsics_camextrinsics = None,
              observations_point                        = None,
              indices_point_camintrinsics_camextrinsics = None,
              lensmodel                                 = lensmodel,
              calobject_warp                            = copy.deepcopy(calobject_warp_true),
              imagersizes                               = imagersizes,
              calibration_object_spacing                = object_spacing,
              verbose                                   = False,
              do_optimize_frames                        = not fixedframes,
              do_optimize_intrinsics_core               = False if model =='splined' else True,
              do_optimize_intrinsics_distortions        = True,
              do_optimize_extrinsics                    = True,
              do_optimize_calobject_warp                = True,
              do_apply_regularization                   = True,
              do_apply_outlier_rejection                = False,
              do_apply_regularization_unity_cam01       = report_points)

    if fixedframes:
        # Frames are fixed: each camera has an independent pose
        optimization_inputs_baseline['extrinsics_rt_fromref'] = \
            copy.deepcopy(extrinsics_true_mounted)
    else:
        # Frames are NOT fixed: cam0 is fixed as the reference coord system. I
        # transform each optimization extrinsics vector to be relative to cam0
        optimization_inputs_baseline['extrinsics_rt_fromref'] = \
            mrcal.compose_rt(extrinsics_true_mounted[1:,:],
                             mrcal.invert_rt(extrinsics_true_mounted[0,:]))

    ###########################################################################
    # p = mrcal.show_geometry(models_true,
    #                         frames          = frames_true,
    #                         object_width_n  = object_width_n,
    #                         object_height_n = object_height_n,
    #                         object_spacing  = object_spacing)
    # sys.exit()

    if not report_points:
        optimization_inputs_baseline['indices_frame_camintrinsics_camextrinsics'] = indices_frame_camintrinsics_camextrinsics
        optimization_inputs_baseline['frames_rt_toref'         ]                  = copy.deepcopy(frames_true)
        optimization_inputs_baseline['observations_board']                        = copy.deepcopy(observations_board_true)

    else:

        # I break up the chessboard observations into discrete points

        # shape (Nframes,H,W,Ncameras, 3)
        indices_point_camintrinsics_camextrinsics = np.zeros((Nframes,object_height_n,object_width_n,Ncameras, 3), dtype=np.int32)

        # index_point
        indices_point_camintrinsics_camextrinsics[...,0] += \
            nps.dummy(indices_frame_camintrinsics_camextrinsics[...,0].reshape(Nframes,Ncameras), -2,-2) \
            * object_height_n * object_width_n \
            + nps.dummy(np.arange(object_height_n * object_width_n).reshape(object_height_n,object_width_n), -1)

        # camintrinsics and camextrinsics
        indices_point_camintrinsics_camextrinsics[...,1:] += \
            nps.dummy(indices_frame_camintrinsics_camextrinsics[...,1:].reshape(Nframes,Ncameras,2), -3,-3)

        # shape (Nframes*H*W*Ncameras, 3)
        indices_point_camintrinsics_camextrinsics = \
            nps.clump(indices_point_camintrinsics_camextrinsics, n=4)

        # shape (H,W,3)
        pboard = \
            mrcal.ref_calibration_object(object_width_n,
                                         object_height_n,
                                         object_spacing,
                                         calobject_warp = calobject_warp_true)

        # shape (Nframes,H,W, 3)
        points_true = mrcal.transform_point_Rt(nps.dummy(Rt_ref_board_true,-3,-3),
                                               pboard)
        # shape (Nframes*H*W, 3)
        points_true = nps.clump(points_true, n=3)

        #  shape (Nframes*Nh*Nw*Ncameras, 3)
        observations_point_true = \
            nps.clump(
                nps.mv( observations_board_true.reshape(Nframes,
                                                        Ncameras,
                                                        object_height_n,
                                                        object_width_n,
                                                        3),
                        -4, -2),
                n=4)

        Npoints = points_true.shape[0]

        optimization_inputs_baseline['indices_point_camintrinsics_camextrinsics'] = indices_point_camintrinsics_camextrinsics
        optimization_inputs_baseline['points']                                    = copy.deepcopy(points_true)
        optimization_inputs_baseline['observations_point']                        = copy.deepcopy(observations_point_true)

        optimization_inputs_baseline['point_min_range'] = 1e-3
        optimization_inputs_baseline['point_max_range'] = 1e12

    mrcal.optimize(**optimization_inputs_baseline)

    models_baseline = \
        [ mrcal.cameramodel( optimization_inputs = optimization_inputs_baseline,
                             icam_intrinsics     = i) \
          for i in range(Ncameras) ]

    if not report_points:
        return                                                     \
            optimization_inputs_baseline,                          \
            models_true, models_baseline,                          \
            lensmodel, Nintrinsics, imagersizes,                   \
            intrinsics_true, extrinsics_true_mounted,              \
            indices_frame_camintrinsics_camextrinsics, frames_true, observations_board_true, Nframes

    else:

        return                                                     \
            optimization_inputs_baseline,                          \
            models_true, models_baseline,                          \
            lensmodel, Nintrinsics, imagersizes,                   \
            intrinsics_true, extrinsics_true_mounted,              \
            indices_point_camintrinsics_camextrinsics, points_true, observations_point_true, Npoints


def calibration_sample(Nsamples,
                       optimization_inputs_baseline,
                       pixel_uncertainty_stdev,
                       fixedframes,
                       function_optimize = None):

    r'''Sample calibrations subject to random noise on the input observations

optimization_inputs_baseline['observations_board'] and
optimization_inputs_baseline['observations_point'] are assumed to contain
perfect observations

    '''

    def have(k):
        return k in optimization_inputs_baseline and \
            optimization_inputs_baseline[k] is not None

    intrinsics_sampled = np.zeros((Nsamples,) + optimization_inputs_baseline['intrinsics']    .shape, dtype=float)

    if have('frames_rt_toref'):
        frames_sampled = np.zeros((Nsamples,) + optimization_inputs_baseline['frames_rt_toref'].shape, dtype=float)
        q_noise_board_sampled = np.zeros((Nsamples,) + optimization_inputs_baseline['observations_board'].shape[:-1] + (2,), dtype=float)
    else:
        frames_sampled        = None
        q_noise_board_sampled = None
    if have('points'):
        points_sampled = np.zeros((Nsamples,) + optimization_inputs_baseline['points'].shape, dtype=float)
        q_noise_point_sampled = np.zeros((Nsamples,) + optimization_inputs_baseline['observations_point'].shape[:-1] + (2,), dtype=float)
    else:
        points_sampled        = None
        q_noise_point_sampled = None
    if have('calobject_warp'):
        calobject_warp_sampled = np.zeros((Nsamples,) + optimization_inputs_baseline['calobject_warp'].shape, dtype=float)
    else:
        calobject_warp_sampled = None

    if function_optimize is None:
        b_sampled_unpacked = np.zeros((Nsamples,mrcal.num_states(**optimization_inputs_baseline)),)
    else:
        b_sampled_unpacked = None

    optimization_inputs_sampled = [None] * Nsamples



    Ncameras_extrinsics = optimization_inputs_baseline['extrinsics_rt_fromref'].shape[0]
    if not fixedframes:
        # the first row is fixed at 0
        Ncameras_extrinsics += 1
    extrinsics_sampled_mounted = np.zeros((Nsamples,Ncameras_extrinsics,6), dtype=float)

    for isample in range(Nsamples):
        if (isample+1) % 20 == 0:
            print(f"Sampling {isample+1}/{Nsamples}")

        optimization_inputs_sampled[isample] = copy.deepcopy(optimization_inputs_baseline)
        optimization_inputs = optimization_inputs_sampled[isample]

        if have('observations_board'):
            q_noise_board_sampled[isample],optimization_inputs['observations_board'] = \
                sample_dqref(optimization_inputs['observations_board'],
                             pixel_uncertainty_stdev)
        else:
            q_noise_board = None
        if have('observations_point'):
            q_noise_point_sampled[isample],optimization_inputs['observations_point'] = \
                sample_dqref(optimization_inputs['observations_point'],
                             pixel_uncertainty_stdev)
        else:
            q_noise_point = None

        if function_optimize is None:
            b_sampled_unpacked[isample] = mrcal.optimize(**optimization_inputs)['b_packed']
            mrcal.unpack_state(b_sampled_unpacked[isample], **optimization_inputs)
        else:
            function_optimize(optimization_inputs)

        intrinsics_sampled    [isample,...] = optimization_inputs['intrinsics']
        if fixedframes:
            extrinsics_sampled_mounted[isample,   ...] = optimization_inputs['extrinsics_rt_fromref']
        else:
            # the remaining row is already 0
            extrinsics_sampled_mounted[isample,1:,...] = optimization_inputs['extrinsics_rt_fromref']

        if frames_sampled is not None:
            frames_sampled[isample,...] = optimization_inputs['frames_rt_toref']
        if points_sampled is not None:
            points_sampled[isample,...] = optimization_inputs['points']
        if calobject_warp_sampled is not None:
            calobject_warp_sampled[isample,...] = optimization_inputs['calobject_warp']


    return (intrinsics_sampled,
            extrinsics_sampled_mounted,
            frames_sampled,
            points_sampled,
            calobject_warp_sampled,
            q_noise_board_sampled,
            q_noise_point_sampled,
            b_sampled_unpacked,
            optimization_inputs_sampled)
