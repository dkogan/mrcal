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


def grad__r_from_R(R):

    # This function deals with a flattened Rflat with shape (9,)
    #
    # dr/dRflat isn't well-defined: not all sets of 9 numbers form a valid
    # rotation matrix: the r_from_R mapping locally acts in a 3-dimensional
    # subspace of R. We get this subspace from the opposite, well-defined
    # gradient dRflat/dr. dRflat_dr.shape is (9,3) so dR all lie in a subspace
    # spanned by its columns. I compute a QR decomposition:
    #
    #   dRflat_dr = Q R
    #
    # where Q is orthonormal (9,3) and R is some (3,3). Q is
    # the basis spanned by Rflat in response to perturbations in dr:
    #
    #   dRflat = Q R dr
    #
    # So for some dRflat I can project to the subspace and transform to get dr:
    #
    #   dr = inv(R) Qt dRflat
    #
    # This is a different way of computing the same thing that
    # r_from_R(get_gradients=True) does
    #
    # Note that I'm not computing any numerical gradients here. If R_from_r() is
    # right, then this should be right too
    r = mrcal.r_from_R(R)

    _,dR_dr      = mrcal.R_from_r(r, get_gradients = True)
    dRflat_dr    = nps.clump(dR_dr, n=2)

    Q,R = np.linalg.qr(dRflat_dr)
    return np.linalg.solve(R, Q.T)


def calibration_baseline(model, Ncameras, Nframes, extra_observation_at,
                         object_width_n,
                         object_height_n,
                         object_spacing,
                         # Camera poses in respect to the first camera. Thus
                         # rt_cam_ref_true[0] = zeros(6). This is
                         # checked
                         extrinsics_rt_fromcam0_true,
                         calobject_warp_true,
                         fixedframes,
                         testdir,
                         *,
                         cull_left_of_center = False,
                         range_to_boards = 4.0,
                         x_offset = 0.0,
                         x_noiseradius = 2.5,
                         y_noiseradius = 2.5,
                         x_mirror      = False, # half with +x_offset, half with -x_offset
                         report_points = False,

                         optimize       = True,
                         moving_cameras = False,
                         ref_frame0     = False,

                         # The logic to avoid oblique views was added in
                         #   https://github.com/dkogan/mrcal/commit/b54df5d3
                         #
                         # We want to disable this if we're trying to compare
                         # results to mrcal 2.4, since this commit was made
                         # after that
                         avoid_oblique_views = True):

    r'''Compute a calibration baseline as a starting point for experiments

This is a perfect, noiseless solve. Regularization IS enabled, and the returned
model is at the optimization optimum. So the returned models will not sit
exactly at the ground-truth.

If x_mirror then half of Nframes are shifted in the x direction by x_offset and
the other half by -x_offset

I always compute chessboard views. But if report_points: I store each corner
observation as a separate point. if report_points: I ALSO enable the unity_cam01
regularization to make the solve non-singular

fixedframes may be true in the simple, vanilla case: not moving_cameras and not
ref_frame0. if fixedframes: we lock down the frame poses, and allow ALL the
cameras to move

The moving_cameras and ref_frame0 arguments control what is moving/stationary
and where the reference coordinate system is. Since in the end we look at
relative camera-frame transforms, their absolute poses don't matter, and either
the cameras or frames can be stationary. And the reference coord system can be
defined at either one of the cameras or one of the frames. Some of these
combinations aren't supported at all, and some are supported only if we have a
single camera. An exception will be raised for an un-supported case

ARGUMENTS

- model: string. 'opencv4' or 'opencv8' or 'splined'

- ...

    '''

    if nps.norm2(extrinsics_rt_fromcam0_true[0]) > 0:
        raise Exception("A non-identity cam0 transform was given. This is not supported")

    if fixedframes and \
       (moving_cameras or ref_frame0):
        raise Exception("fixedframes only supported in the simple, vanilla case: not moving_cameras and not ref_frame0")

    if re.match('opencv',model):
        models_true_refcam0 = ( mrcal.cameramodel(f"{testdir}/data/cam0.opencv8.cameramodel"),
                        mrcal.cameramodel(f"{testdir}/data/cam0.opencv8.cameramodel"),
                        mrcal.cameramodel(f"{testdir}/data/cam1.opencv8.cameramodel"),
                        mrcal.cameramodel(f"{testdir}/data/cam1.opencv8.cameramodel") )

        if model == 'opencv4':
            # I have opencv8 models_true_refcam0, but I truncate to opencv4 models_true_refcam0
            for m in models_true_refcam0:
                m.intrinsics( intrinsics = ('LENSMODEL_OPENCV4', m.intrinsics()[1][:8]))
    elif model == 'splined':
        models_true_refcam0 = ( mrcal.cameramodel(f"{testdir}/data/cam0.splined.cameramodel"),
                        mrcal.cameramodel(f"{testdir}/data/cam0.splined.cameramodel"),
                        mrcal.cameramodel(f"{testdir}/data/cam1.splined.cameramodel"),
                        mrcal.cameramodel(f"{testdir}/data/cam1.splined.cameramodel") )
    else:
        raise Exception("Unknown lens being tested")

    models_true_refcam0 = models_true_refcam0[:Ncameras]
    lensmodel   = models_true_refcam0[0].intrinsics()[0]
    Nintrinsics = mrcal.lensmodel_num_params(lensmodel)

    for i in range(Ncameras):
        models_true_refcam0[i].rt_cam_ref(extrinsics_rt_fromcam0_true[i])

    imagersizes = nps.cat( *[m.imagersize() for m in models_true_refcam0] )

    # These are perfect
    intrinsics_true = nps.cat( *[m.intrinsics()[1]         for m in models_true_refcam0] )
    x_center = -(Ncameras-1)/2.



    def synthesize(z, z_noiseradius, Nframes,
                   x_offset):
        if avoid_oblique_views:
            kwargs = \
                dict( pcamera_nominal_ref   = np.array((x_center,0,0), dtype=float),
                      max_oblique_angle_deg = 30. )
        else:
            kwargs = dict()


        return \
            mrcal.synthesize_board_observations(models_true_refcam0,
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
                                                **kwargs)


    if not x_mirror:
        # shapes (Nframes, Ncameras, Nh, Nw, 2),
        #        (Nframes, 4,3)
        q_true,Rt_cam0_board_true = \
            synthesize(z             = range_to_boards,
                       z_noiseradius = range_to_boards / 2.0,
                       Nframes       = Nframes,
                       x_offset      = x_offset)
    else:
        if Nframes & 1:
            raise Exception("x_mirror is True, so Nframes must be even")

        # shapes (Nframes//2, Ncameras, Nh, Nw, 2),
        #        (Nframes//2, 4,3)
        q_true0,Rt_cam0_board_true0 = \
            synthesize(z             = range_to_boards,
                       z_noiseradius = range_to_boards / 2.0,
                       Nframes       = Nframes//2,
                       x_offset      = x_offset)
        q_true1,Rt_cam0_board_true1 = \
            synthesize(z             = range_to_boards,
                       z_noiseradius = range_to_boards / 2.0,
                       Nframes       = Nframes//2,
                       x_offset      = -x_offset)

        q_true            = nps.glue(q_true0,           q_true1,            axis=-5)
        Rt_cam0_board_true = nps.glue(Rt_cam0_board_true0,Rt_cam0_board_true1, axis=-3)


    if extra_observation_at is not None:
        q_true_extra,Rt_cam0_board_true_extra = \
            synthesize(z             = extra_observation_at,
                       z_noiseradius = extra_observation_at / 10.0,
                       Nframes       = 1,
                       x_offset      = x_offset)


        q_true            = nps.glue( q_true, q_true_extra,
                                      axis=-5)
        Rt_cam0_board_true = nps.glue( Rt_cam0_board_true, Rt_cam0_board_true_extra,
                                      axis=-3)

        Nframes += 1

    # shape (Nframes, 6)
    rt_cam0_board_true = mrcal.rt_from_Rt(Rt_cam0_board_true)

    ############# I have perfect observations in q_true.
    # weight has shape (Nframes, Ncameras, Nh, Nw),
    weight01 = (np.random.rand(*q_true.shape[:-1]) + 1.) / 2. # in [0,1]
    weight0 = 0.2
    weight1 = 1.0
    weight = weight0 + (weight1-weight0)*weight01

    if cull_left_of_center:

        imagersize = models_true_refcam0[0].imagersize()
        for m in models_true_refcam0[1:]:
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

    # stationary cameras, so idxci = idxce
    # shape (Nframes*Ncameras, 3)
    indices_frame_camintrinsics_camextrinsics = \
        nps.glue(indices_frame_camera,
                 indices_frame_camera[:,(1,)],
                 axis=-1)
    if not fixedframes:
        # cam0 is at the reference
        indices_frame_camintrinsics_camextrinsics[:,2] -= 1



    # I now reoptimize the perfect-observations problem. Without regularization,
    # this is a no-op: I'm already at the optimum. With regularization, this will
    # move us a certain amount (that the test will evaluate). Then I look at
    # noise-induced motions off this optimization optimum
    optimization_inputs_baseline = \
        dict( intrinsics                                = copy.deepcopy(intrinsics_true),
              rt_cam_ref                                = copy.deepcopy(extrinsics_rt_fromcam0_true if fixedframes \
                                                                        else extrinsics_rt_fromcam0_true[1:,:]),
              rt_ref_frame                              = copy.deepcopy(rt_cam0_board_true),
              points                                    = None,
              observations_board                        = observations_board_true,
              indices_frame_camintrinsics_camextrinsics = indices_frame_camintrinsics_camextrinsics,
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

    ###########################################################################
    # p = mrcal.show_geometry(models_true_refcam0,
    #                         frames          = rt_cam0_board_true,
    #                         object_width_n  = object_width_n,
    #                         object_height_n = object_height_n,
    #                         object_spacing  = object_spacing)
    # sys.exit()


    calibration_make_non_vanilla(optimization_inputs_baseline,
                                 moving_cameras = moving_cameras,
                                 ref_frame0     = ref_frame0)

    if report_points:
        calibration_boards_to_points(optimization_inputs_baseline)

    if optimize:
        mrcal.optimize(**optimization_inputs_baseline)

    if not report_points:
        return                                        \
            optimization_inputs_baseline,             \
            models_true_refcam0,                      \
            rt_cam0_board_true

    else:

        return                                        \
            optimization_inputs_baseline,             \
            models_true_refcam0,                      \
            points_true


def calibration_make_non_vanilla(optimization_inputs,
                                 *,
                                 moving_cameras,
                                 ref_frame0):

    if not moving_cameras and \
       not ref_frame0:

        # stationary camera, moving frame, reference at cam0
        #
        #   For a single camera:
        #
        #   0    state variables for extrinsics
        #   6*Nf state variables for frames (or 3*Np*Nf points)
        #
        #   indices_frame_camintrinsics_camextrinsics =
        #   [ 0 0 -1
        #     1 0 -1
        #     2 0 -1
        #     ...   ]

        # This is the baseline case. The data is already set up like this.
        return


    idxf,idxci,idxce   = optimization_inputs['indices_frame_camintrinsics_camextrinsics'].T
    Ncameras           = len(optimization_inputs['intrinsics'])
    rt_cam0_board_true = optimization_inputs['rt_ref_frame']

    if moving_cameras and \
       ref_frame0:
        # moving camera, stationary frame, reference at the one stationary
        # chessboard
        #
        #   For a single camera:
        #
        #   6*Nf state variables for extrinsics
        #   0    state variables for frames

        #   indices_frame_camintrinsics_camextrinsics =
        #   [ 0 0 0
        #     0 0 1
        #     0 0 2
        #     ... ]
        #
        #   I want to have indices_frame = -1 here, but mrcal does not currently
        #   support this. Instead we set indices_frame = 0, actually store
        #   something into the rt_ref_frame array, and set do_optimize_frames
        #   = False
        #
        #   Today this scenario cannot work with multiple cameras: this would
        #   require a rigid "rig" or cameras moving in unison, which isn't
        #   supported today.
        if Ncameras > 1:
            raise Exception("Scenario (moving-camera, stationary-frame, ref-at-frame) cannot work with multiple cameras: camera-rig implementation is required")

        optimization_inputs['indices_frame_camintrinsics_camextrinsics'] = \
            np.ascontiguousarray(nps.transpose( nps.cat( idxce+1,
                                                         idxci,
                                                         idxf ) ))

        # This is to support "test-projection-uncertainty.py --fixed frames".
        # That tool and this function needs a rework, to take separate arguments
        # to control
        #
        # - what is moving
        # - where the reference is
        # - what is fixed
        # - points/boards
        if np.all(idxce == 0) and \
           np.all(idxf == np.arange(len(idxf))) and \
           optimization_inputs['do_optimize_extrinsics'] and \
           not optimization_inputs['do_optimize_frames']:
            optimization_inputs['indices_frame_camintrinsics_camextrinsics'][:,0] -= 1

        optimization_inputs['rt_cam_ref'            ] = np.array(rt_cam0_board_true)
        optimization_inputs['rt_ref_frame'          ] = np.zeros((1,6), dtype=float)
        optimization_inputs['do_optimize_extrinsics'] = True
        optimization_inputs['do_optimize_frames'    ] = False
        return


    if moving_cameras and \
       not ref_frame0:

        # moving camera, stationary frame, reference at cam0
        #   6*(Nf-1) state variables for extrinsics
        #   6        state variables for frames

        #   indices_frame_camintrinsics_camextrinsics =
        #   [ 0 0 -1
        #     0 0  0
        #     0 0  1
        #     0 0  2
        #     ...   ]
        #
        #   Or if looking_at_points:

        #     if I assume I'm looking at a chessboard, I set the points to the known
        #     geometry, fix that, and I'm done. But what if the point geometry is fixed
        #     between frames, but unknown? I fix cam0 and have a single set of points that
        #     I optimize:

        #     6*(Nf-1) extrinsics
        #     3*Np points

        #     indices_point_camintrinsics_camextrinsics =
        #     [ 0 0 -1
        #       1 0 -1
        #       2 0 -1
        #       ...
        #       0 0  0
        #       1 0  0
        #       2 0  0
        #       ...
        #       0 0  1
        #       1 0  1
        #       2 0  1
        #       ... ]
        #
        #   Today this scenario cannot work with multiple cameras: this would
        #   require a rigid "rig" or cameras moving in unison, which isn't
        #   supported today.
        if Ncameras > 1:
            raise Exception("Scenario (moving-camera, stationary-frame, ref-at-cam0) cannot work with multiple cameras: camera-rig implementation is required")

        optimization_inputs['indices_frame_camintrinsics_camextrinsics'] = \
            np.ascontiguousarray(nps.transpose( nps.cat( idxce+1,
                                                         idxci,
                                                         idxf-1 )))

        rt_cam_cam0 = \
            mrcal.compose_rt(rt_cam0_board_true[1:,:],
                             mrcal.invert_rt(rt_cam0_board_true[0,:]))

        optimization_inputs['rt_cam_ref'            ] = rt_cam_cam0
        optimization_inputs['rt_ref_frame'          ] = rt_cam0_board_true[(0,),:]
        optimization_inputs['do_optimize_extrinsics'] = True
        optimization_inputs['do_optimize_frames'    ] = True
        return


    if not moving_cameras and \
       ref_frame0:

        # stationary camera, moving frame, reference at frame0
        #   6        state variables for extrinsics
        #   6*(Nf-1) state variables for frames

        #   indices_frame_camintrinsics_camextrinsics =
        #   [ -1 0 0
        #      0 0 0
        #      1 0 0
        #     ...   ]
        #
        # mrcal cannot represent this today, so I don't check it. Today mrcal
        # can represent NULL camera transforms (index_camextrinsics < 0), but
        # not NULL frame transforms. Above I could fake a NULL frame transform
        # by setting do_optimize_frames = False, but that only works if I want
        # to lock down ALL the frame transforms, not just a single one, like I
        # need to do here
        raise Exception("Case not supported: not moving_cameras and ref_frame0")

    raise Exception("Unhandled case. Getting here is a bug")


def calibration_boards_to_points(optimization_inputs):

    # I break up the chessboard observations into discrete points

    rt_ref_board                              = optimization_inputs['rt_ref_frame']
    Rt_ref_board                              = mrcal.Rt_from_rt(rt_ref_board)
    Nframes                                   = len(rt_ref_board)
    indices_frame_camintrinsics_camextrinsics = optimization_inputs['indices_frame_camintrinsics_camextrinsics']
    observations_board                        = optimization_inputs['observations_board']
    Nobservations                             = len(observations_board)
    object_height_n,object_width_n            = observations_board.shape[-3:-1]
    object_spacing                            = optimization_inputs['calibration_object_spacing']
    calobject_warp                            = optimization_inputs['calobject_warp']

    # shape (Nobservations,H,W,3)
    indices_point_camintrinsics_camextrinsics = np.zeros((Nobservations,object_height_n,object_width_n,3), dtype=np.int32)

    # shape (Nobservations,)
    iframe = indices_frame_camintrinsics_camextrinsics[...,0]

    # index_point
    # shape (Nobservations,H,W)
    indices_point_camintrinsics_camextrinsics[...,0] = \
        np.arange(object_height_n * object_width_n).reshape(object_height_n,object_width_n) + \
        nps.dummy(iframe * object_height_n * object_width_n,
                  -1,-1)

    # camintrinsics and camextrinsics
    # shape (Nobservations,H,W,2)
    indices_point_camintrinsics_camextrinsics[...,1:] = \
        nps.dummy(indices_frame_camintrinsics_camextrinsics[...,1:], -2,-2)

    # shape (Nobservations*H*W, 3)
    indices_point_camintrinsics_camextrinsics = \
        nps.clump(indices_point_camintrinsics_camextrinsics, n=3)

    # shape (H,W,3)
    pboard = \
        mrcal.ref_calibration_object(object_width_n,
                                     object_height_n,
                                     object_spacing,
                                     calobject_warp = calobject_warp)

    # shape (Nframes,1,1,4,3)
    Rt_ref_board_frames = nps.dummy(Rt_ref_board,-3,-3)

    # shape (Nframes,H,W, 3)
    pref = mrcal.transform_point_Rt(Rt_ref_board_frames,
                                    pboard)
    # shape (Nframes*H*W, 3)
    pref = nps.clump(pref, n=3)

    # shape (Nobservations*Nh*Nw, 3)
    observations_point = nps.clump(observations_board, n=3)

    # At this point the data is conceptually good. However, an implementation
    # detail requires monotonically non-decreasing point indices:
    #   uncertainty.c(983): Unexpected jacobian structure. I'm assuming
    #   non-decreasing point references. The Jcross_t__Jcross computation uses
    #   chunks of Kpackedp; it assumes that once the chunk is computed, it is
    #   DONE, and never revisited. Non-monotonic point indices break that
    # So I re-sort here
    iipoint = np.argsort(indices_point_camintrinsics_camextrinsics[:,0],
                         kind='stable')
    indices_point_camintrinsics_camextrinsics = \
        indices_point_camintrinsics_camextrinsics[iipoint]
    observations_point = observations_point[iipoint]


    optimization_inputs['indices_point_camintrinsics_camextrinsics'] = indices_point_camintrinsics_camextrinsics
    optimization_inputs['points']                                    = pref
    optimization_inputs['observations_point']                        = observations_point

    optimization_inputs['indices_frame_camintrinsics_camextrinsics'] = None
    optimization_inputs['rt_ref_frame']                              = None
    optimization_inputs['observations_board']                        = None

    optimization_inputs['point_min_range'] = 1e-3
    optimization_inputs['point_max_range'] = 1e12

    # point-only solves are unique up-to-scale only, and need the extra regularization
    Ncameras_extrinsics = optimization_inputs['rt_cam_ref'].shape[0]
    if Ncameras_extrinsics <= 1:
        if optimization_inputs['do_optimize_frames']:
            raise Exception("Points-only solves with do_optimize_frames==True need extra regularization, but currently that's only possible with Ncameras_extrinsics>=1")
    else:
        optimization_inputs['do_apply_regularization_unity_cam01'] = True
        # I rescale all geometry to make the unity01 regularization=0 at the start
        s = 1. / nps.norm2(optimization_inputs['rt_cam_ref'][0,3:])
        optimization_inputs['rt_cam_ref'][:,3:] *= s
        optimization_inputs['points'] *= s


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

    if have('rt_ref_frame'):
        frames_sampled = np.zeros((Nsamples,) + optimization_inputs_baseline['rt_ref_frame'].shape, dtype=float)
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



    Ncameras_extrinsics = optimization_inputs_baseline['rt_cam_ref'].shape[0]
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
            extrinsics_sampled_mounted[isample,   ...] = optimization_inputs['rt_cam_ref']
        else:
            # the remaining row is already 0
            extrinsics_sampled_mounted[isample,1:,...] = optimization_inputs['rt_cam_ref']

        if frames_sampled is not None:
            frames_sampled[isample,...] = optimization_inputs['rt_ref_frame']
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
