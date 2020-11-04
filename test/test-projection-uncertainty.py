#!/usr/bin/python3

r'''Uncertainty-quantification test

I run a number of synthetic-data camera calibrations, applying some noise to the
observed inputs each time. I then look at the distribution of projected world
points, and compare that distribution with theoretical predictions.

This test checks two different types of calibrations:

--fixed cam0: we place camera0 at the reference coordinate system. So camera0
  may not move, and has no associated extrinsics vector. The other cameras and
  the frames move. When evaluating at projection uncertainty I pick a point
  referenced off the frames. As the frames move around, so does the point I'm
  projecting. But together, the motion of the frames and the extrinsics and the
  intrinsics should map it to the same pixel in the end.

--fixed frames: the reference coordinate system is attached to the frames,
  which may not move. All cameras may move around and all cameras have an
  associated extrinsics vector. When evaluating at projection uncertainty I also
  pick a point referenced off the frames, but here any point in the reference
  coord system will do. As the cameras move around, so does the point I'm
  projecting. But together, the motion of the extrinsics and the intrinsics
  should map it to the same pixel in the end.

Exactly one of these two arguments is required.

The lens model we're using must appear: either "--model opencv4" or "--model
opencv8" or "--model splined"

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
                        required=True,
                        help='''Are we putting the origin at camera0, or are all the frames at a fixed (and
                        non-optimizeable) pose? One or the other is required.''')
    parser.add_argument('--model',
                        type=str,
                        choices=('opencv4','opencv8','splined'),
                        required = True,
                        help='''Which lens model we're using. Must be one of
                        ('opencv4','opencv8','splined')''')
    parser.add_argument('--Nframes',
                        type=int,
                        default=50,
                        help='''How many chessboard poses to simulate. These are dense observations: every
                        camera sees every corner of every chessboard pose''')
    parser.add_argument('--Nsamples',
                        type=int,
                        default=100,
                        help='''How many random samples to evaluate''')
    parser.add_argument('--distances',
                        type=str,
                        default='5,inf',
                        help='''Comma-separated list of distance where we test the uncertainty predictions.
                        Numbers an "inf" understood. The first value on this
                        list is used for visualization in --show-distribution''')
    parser.add_argument('--no-sampling',
                        action='store_true',
                        help='''By default we check some things, and then generate lots of statistical
                        samples to compare the empirical distributions with
                        analytic predictions. This is slow, so we may want to
                        omit it''')
    parser.add_argument('--show-distribution',
                        action='store_true',
                        help='''If given, we produce plots showing the distribution of samples''')
    parser.add_argument('--write-models',
                        action='store_true',
                        help='''If given, we write the resulting models to disk for further analysis''')
    parser.add_argument('--make-documentation-plots',
                        type=str,
                        help='''If given, we produce plots for the documentation. Takes one argument: a
                        string describing this test. This will be used in the
                        filenames of the resulting plots. The extension of the
                        string selects the output type. To make interactive
                        plots, pass ""''')
    parser.add_argument('--explore',
                        action='store_true',
                        help='''If given, we drop into a REPL at the end''')
    parser.add_argument('--reproject-perturbed',
                        choices=('mean-frames', 'fit-boards-ref', 'diff'),
                        default = 'mean-frames',
                        help='''Which reproject-after-perturbation method to use. This is for experiments.
                        Some of these methods will be probably wrong.''')

    args = parser.parse_args()

    args.distances = args.distances.split(',')
    for i in range(len(args.distances)):
        if args.distances[i] == 'inf':
            args.distances[i] = None
        else:
            if re.match("[0-9deDE.-]+", args.distances[i]):
                s = float(args.distances[i])
            else:
                print('--distances is a comma-separated list of numbers or "inf"', file=sys.stderr)
                sys.exit(1)
            args.distances[i] = s

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

from test_calibration_helpers import sample_dqref,sorted_eig


fixedframes = (args.fixed == 'frames')

import tempfile
import atexit
import shutil
workdir = tempfile.mkdtemp()
def cleanup():
    global workdir
    try:
        shutil.rmtree(workdir)
        workdir = None
    except:
        pass
atexit.register(cleanup)



def gnuplotlib_add_list_option(d, key, value):
    if not key in d:
        d[key] = value
    elif isinstance(d[key], list) or isinstance(d[key], tuple):
        d[key] = list(d[key]) + [value]
    else:
        d[key] = [ d[key], value ]

terminal = None
extraset = None
if args.make_documentation_plots:
    extension = os.path.splitext(args.make_documentation_plots)[1]

    if extension == '.svg':
        terminal = 'svg noenhanced solid dynamic fontscale 0.5'
        extraset = 'pointsize 0.5'
    elif extension == '.pdf':
        terminal = 'pdf noenhanced solid color font ",10" size 11in,6in'
        extraset = 'pointsize 1.'


# I want the RNG to be deterministic
np.random.seed(0)

############# Set up my world, and compute all the perfect positions, pixel
############# observations of everything
if re.match('opencv',args.model):
    models_true = ( mrcal.cameramodel(f"{testdir}/data/cam0.opencv8.cameramodel"),
                    mrcal.cameramodel(f"{testdir}/data/cam0.opencv8.cameramodel"),
                    mrcal.cameramodel(f"{testdir}/data/cam1.opencv8.cameramodel"),
                    mrcal.cameramodel(f"{testdir}/data/cam1.opencv8.cameramodel") )

    if args.model == 'opencv4':
        # I have opencv8 models_true, but I truncate to opencv4 models_true
        for m in models_true:
            m.intrinsics( intrinsics = ('LENSMODEL_OPENCV4', m.intrinsics()[1][:8]))
elif args.model == 'splined':
    models_true = ( mrcal.cameramodel(f"{testdir}/data/cam0.splined.cameramodel"),
                    mrcal.cameramodel(f"{testdir}/data/cam0.splined.cameramodel"),
                    mrcal.cameramodel(f"{testdir}/data/cam1.splined.cameramodel"),
                    mrcal.cameramodel(f"{testdir}/data/cam1.splined.cameramodel") )
else:
    raise Exception("Unknown lens being tested")

lensmodel   = models_true[0].intrinsics()[0]

Nintrinsics = mrcal.lensmodel_num_params(lensmodel)
imagersizes = nps.cat( *[m.imagersize() for m in models_true] )

Ncameras = len(models_true)

models_true[0].extrinsics_rt_fromref(np.zeros((6,), dtype=float))
models_true[1].extrinsics_rt_fromref(np.array((0.08,0.2,0.02, 1., 0.9,0.1)))
models_true[2].extrinsics_rt_fromref(np.array((0.01,0.07,0.2, 2.1,0.4,0.2)))
models_true[3].extrinsics_rt_fromref(np.array((-0.1,0.08,0.08, 4.4,0.2,0.1)))

pixel_uncertainty_stdev = 1.5
object_spacing          = 0.1
object_width_n          = 10
object_height_n         = 9

# These are perfect
intrinsics_true         = nps.cat( *[m.intrinsics()[1]         for m in models_true] )
extrinsics_true_mounted = nps.cat( *[m.extrinsics_rt_fromref() for m in models_true] )
calobject_warp_true     = np.array((0.002, -0.005))

# shapes (Nframes, Ncameras, Nh, Nw, 2),
#        (Nframes, 4,3)
q_true,Rt_cam0_board_true = \
    mrcal.synthesize_board_observations(models_true,
                                        object_width_n, object_height_n, object_spacing,
                                        calobject_warp_true,
                                        np.array((0.,             0.,             0.,             -2,  0,   4.0)),
                                        np.array((np.pi/180.*30., np.pi/180.*30., np.pi/180.*20., 2.5, 2.5, 2.0)),
                                        args.Nframes)
frames_true             = mrcal.rt_from_Rt(Rt_cam0_board_true)

############# I have perfect observations in q_true. I corrupt them by noise
# weight has shape (Nframes, Ncameras, Nh, Nw),
weight01 = (np.random.rand(*q_true.shape[:-1]) + 1.) / 2. # in [0,1]
weight0 = 0.2
weight1 = 1.0
weight = weight0 + (weight1-weight0)*weight01

# I want observations of shape (Nframes*Ncameras, Nh, Nw, 3) where each row is
# (x,y,weight)
observations_true = nps.clump( nps.glue(q_true,
                                        nps.dummy(weight,-1),
                                        axis=-1),
                              n=2)


# Dense observations. All the cameras see all the boards
indices_frame_camera = np.zeros( (args.Nframes*Ncameras, 2), dtype=np.int32)
indices_frame = indices_frame_camera[:,0].reshape(args.Nframes,Ncameras)
indices_frame.setfield(nps.outer(np.arange(args.Nframes, dtype=np.int32),
                                 np.ones((Ncameras,), dtype=np.int32)),
                       dtype = np.int32)
indices_camera = indices_frame_camera[:,1].reshape(args.Nframes,Ncameras)
indices_camera.setfield(nps.outer(np.ones((args.Nframes,), dtype=np.int32),
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
          extrinsics_rt_fromref                     = copy.deepcopy(extrinsics_true_mounted if fixedframes else extrinsics_true_mounted[1:,:]),
          frames_rt_toref                           = copy.deepcopy(frames_true),
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
          do_optimize_intrinsics_core               = False if args.model=='splined' else True,
          do_optimize_intrinsics_distortions        = True,
          do_optimize_extrinsics                    = True,
          do_optimize_calobject_warp                = True,
          do_apply_regularization                   = True,
          do_apply_outlier_rejection                = False)
mrcal.optimize(**optimization_inputs_baseline)

models_baseline = \
    [ mrcal.cameramodel( optimization_inputs = optimization_inputs_baseline,
                         icam_intrinsics     = i) \
      for i in range(Ncameras) ]

# I evaluate the projection uncertainty of this vector. In each camera. I'd like
# it to be center-ish, but not AT the center. So I look at 1/3 (w,h). I want
# this to represent a point in a globally-consistent coordinate system. Here I
# have fixed frames, so using the reference coordinate system gives me that
# consistency. Note that I look at q0 for each camera separately, so I'm going
# to evaluate a different world point for each camera
q0_baseline = imagersizes[0]/3.



if args.make_documentation_plots is not None:
    import gnuplotlib as gp

    if args.make_documentation_plots:
        processoptions_output = dict(wait     = False,
                                     terminal = terminal,
                                     _set     = extraset,
                                     hardcopy = f'simulated-geometry--{args.make_documentation_plots}')
    else:
        processoptions_output = dict(wait = True)

    gnuplotlib_add_list_option(processoptions_output, '_set', 'xyplane relative 0')
    mrcal.show_geometry(models_baseline,
                        unset='key',
                        **processoptions_output)

    if args.make_documentation_plots:
        processoptions_output = dict(wait     = False,
                                     terminal = terminal,
                                     _set     = extraset,
                                     hardcopy = f'simulated-observations--{args.make_documentation_plots}')
    else:
        processoptions_output = dict(wait = True)

    def observed_points(icam):
        obs_cam = observations_true[indices_frame_camintrinsics_camextrinsics[:,1]==icam, ..., :2].ravel()
        return obs_cam.reshape(len(obs_cam)//2,2)

    obs_cam = [ ( (observed_points(icam),),
                  (q0_baseline, dict(_with ='points pt 2 ps 2'))) \
                for icam in range(Ncameras) ]
    gp.plot( *obs_cam,

             tuplesize=-2,
             _with='points',
             square=1,
             _xrange=(0, models_true[0].imagersize()[0]-1),
             _yrange=(models_true[0].imagersize()[1]-1, 0),

             multiplot = 'layout 2,2',
             **processoptions_output)


# These are at the optimum
intrinsics_baseline         = nps.cat( *[m.intrinsics()[1]         for m in models_baseline] )
extrinsics_baseline_mounted = nps.cat( *[m.extrinsics_rt_fromref() for m in models_baseline] )
frames_baseline             = optimization_inputs_baseline['frames_rt_toref']
calobject_warp_baseline     = optimization_inputs_baseline['calobject_warp']

if args.write_models:
    for i in range(Ncameras):
        models_true    [i].write(f"/tmp/models-true-camera{i}.cameramodel")
        models_baseline[i].write(f"/tmp/models-baseline-camera{i}.cameramodel")
    sys.exit()





def reproject_perturbed__mean_frames(q, distance,

                                     # shape (Ncameras, Nintrinsics)
                                     baseline_intrinsics,
                                     # shape (Ncameras, 6)
                                     baseline_rt_cam_ref,
                                     # shape (Nframes, 6)
                                     baseline_rt_ref_frame,
                                     # shape (2)
                                     baseline_calobject_warp,

                                     # shape (..., Ncameras, Nintrinsics)
                                     query_intrinsics,
                                     # shape (..., Ncameras, 6)
                                     query_rt_cam_ref,
                                     # shape (..., Nframes, 6)
                                     query_rt_ref_frame,
                                     # shape (..., 2)
                                     query_calobject_warp):
    r'''Reproject by computing the mean in the space of frames

This is what the uncertainty computation does (as of 2020/10/26). The implied
rotation here is aphysical (it is a mean of multipl rotation matrices)

    '''

    # shape (Ncameras, 3)
    p_cam_baseline = mrcal.unproject(q, lensmodel, baseline_intrinsics,
                                     normalize = True) * distance

    # shape (Ncameras, 3)
    p_ref_baseline = \
        mrcal.transform_point_rt( mrcal.invert_rt(baseline_rt_cam_ref),
                                  p_cam_baseline )

    if fixedframes:
        p_ref_query = p_ref_baseline
    else:
        # shape (Nframes, Ncameras, 3)
        # The point in the coord system of all the frames
        p_frames = mrcal.transform_point_rt( \
            nps.dummy(mrcal.invert_rt(baseline_rt_ref_frame),-2),
                                              p_ref_baseline)

        # shape (..., Ncameras, 3)
        p_ref_query = np.mean( mrcal.transform_point_rt( nps.dummy(query_rt_ref_frame, -2),
                                                         p_frames ),
                               axis = -3)

        # shape (..., Ncameras, 3)
        p_cam_query = \
            mrcal.transform_point_rt(query_rt_cam_ref, p_ref_query)

        # shape (..., Ncameras, 2)
        return mrcal.project(p_cam_query, lensmodel, query_intrinsics)


def reproject_perturbed__fit_boards_ref(q, distance,

                                        # shape (Ncameras, Nintrinsics)
                                        baseline_intrinsics,
                                        # shape (Ncameras, 6)
                                        baseline_rt_cam_ref,
                                        # shape (Nframes, 6)
                                        baseline_rt_ref_frame,
                                        # shape (2)
                                        baseline_calobject_warp,

                                        # shape (..., Ncameras, Nintrinsics)
                                        query_intrinsics,
                                        # shape (..., Ncameras, 6)
                                        query_rt_cam_ref,
                                        # shape (..., Nframes, 6)
                                        query_rt_ref_frame,
                                        # shape (..., 2)
                                        query_calobject_warp):
    r'''Reproject by explicitly computing a procrustes fit to align the reference
    coordinate systems of the two solves. We match up the two sets of chessboard
    points

    '''

    calobject_height,calobject_width = optimization_inputs_baseline['observations_board'].shape[1:3]

    # shape (Nsamples, Nh, Nw, 3)
    if query_calobject_warp.ndim > 1:
        calibration_object_query = \
            nps.cat(*[ mrcal.ref_calibration_object(calobject_width, calobject_height,
                                                    optimization_inputs_baseline['calibration_object_spacing'],
                                                    calobject_warp=calobject_warp) \
                       for calobject_warp in query_calobject_warp] )
    else:
        calibration_object_query = \
            mrcal.ref_calibration_object(calobject_width, calobject_height,
                                         optimization_inputs_baseline['calibration_object_spacing'],
                                         calobject_warp=query_calobject_warp)

    # shape (Nsamples, Nframes, Nh, Nw, 3)
    pcorners_ref_query = \
        mrcal.transform_point_rt( nps.dummy(query_rt_ref_frame, -2, -2),
                                  nps.dummy(calibration_object_query, -4))


    # shape (Nh, Nw, 3)
    calibration_object_baseline = \
        mrcal.ref_calibration_object(calobject_width, calobject_height,
                                     optimization_inputs_baseline['calibration_object_spacing'],
                                     calobject_warp=baseline_calobject_warp)
    # frames_ref.shape is (Nframes, 6)

    # shape (Nframes, Nh, Nw, 3)
    pcorners_ref_baseline = \
        mrcal.transform_point_rt( nps.dummy(baseline_rt_ref_frame, -2, -2),
                                  calibration_object_baseline)

    # shape (Nsamples,4,3)
    Rt_refq_refb = \
        mrcal.align_procrustes_points_Rt01( \
            # shape (Nsamples,N,3)
            nps.mv(nps.clump(nps.mv(pcorners_ref_query, -1,0),n=-3),0,-1),

            # shape (N,3)
            nps.clump(pcorners_ref_baseline, n=3))



    # shape (Ncameras, 3)
    p_cam_baseline = mrcal.unproject(q, lensmodel, baseline_intrinsics,
                                     normalize = True) * distance

    # shape (Ncameras, 3). In the ref coord system
    p_ref_baseline = \
        mrcal.transform_point_rt( mrcal.invert_rt(baseline_rt_cam_ref),
                                  p_cam_baseline )

    # shape (Nsamples,Ncameras,3)
    p_ref_query = \
        mrcal.transform_point_Rt(nps.mv(Rt_refq_refb,-3,-4),
                                 p_ref_baseline)

    # shape (..., Ncameras, 3)
    p_cam_query = \
        mrcal.transform_point_rt(query_rt_cam_ref, p_ref_query)

    # shape (..., Ncameras, 2)
    q1 = mrcal.project(p_cam_query, lensmodel, query_intrinsics)

    if q1.shape[-3] == 1: q1 = q1[0,:,:]
    return q1


# The others broadcast implicitly, while THIS main function really cannot handle
# outer dimensions, and needs an explicit broadcasting loop
@nps.broadcast_define(((2,), (),
                       ('Ncameras', 'Nintrinsics'),
                       ('Ncameras', 6),
                       ('Nframes', 6),
                       (2,),
                       ('Ncameras', 'Nintrinsics'),
                       ('Ncameras', 6),
                       ('Nframes', 6),
                       (2,),),
                      ('Ncameras',2))
def reproject_perturbed__diff(q, distance,
                              # shape (Ncameras, Nintrinsics)
                              baseline_intrinsics,
                              # shape (Ncameras, 6)
                              baseline_rt_cam_ref,
                              # shape (Nframes, 6)
                              baseline_rt_ref_frame,
                              # shape (2)
                              baseline_calobject_warp,

                              # shape (Ncameras, Nintrinsics)
                              query_intrinsics,
                              # shape (Ncameras, 6)
                              query_rt_cam_ref,
                              # shape (Nframes, 6)
                              query_rt_ref_frame,
                              # shape (2)
                              query_calobject_warp):
    r'''Reproject by using the "diff" method to compute a rotation

    '''

    # shape (Ncameras, 3)
    p_cam_baseline = mrcal.unproject(q, lensmodel, baseline_intrinsics,
                                     normalize = True) * distance
    p_cam_query = np.zeros((Ncameras, 3), dtype=float)
    for icam in range (Ncameras):

        # This method only cares about the intrinsics
        model_baseline = \
            mrcal.cameramodel( intrinsics = (lensmodel, baseline_intrinsics[icam]),
                               imagersize = imagersizes[0] )
        model_query = \
            mrcal.cameramodel( intrinsics = (lensmodel, query_intrinsics[icam]),
                               imagersize = imagersizes[0] )

        implied_Rt10_query = \
            mrcal.projection_diff( (model_baseline,
                                    model_query),
                                   distance = distance,
                                   use_uncertainties = False,
                                   focus_center      = None,
                                   focus_radius      = 1000.)[3]
        mrcal.transform_point_Rt( implied_Rt10_query, p_cam_baseline[icam],
                                  out = p_cam_query[icam] )

    # shape (Ncameras, 2)
    return \
        mrcal.project( p_cam_query,
                       lensmodel, query_intrinsics)


# Which implementation we're using. Use the method that matches the uncertainty
# computation. Thus the sampled ellipsoids should match the ellipsoids reported
# by the uncertianty method
if   args.reproject_perturbed == 'mean-frames':
    reproject_perturbed = reproject_perturbed__mean_frames
elif args.reproject_perturbed == 'fit-boards-ref':
    reproject_perturbed = reproject_perturbed__fit_boards_ref
elif args.reproject_perturbed == 'diff':
    reproject_perturbed = reproject_perturbed__diff
else:
    raise Exception("getting here is a bug")




q0_true = dict()
for distance in args.distances:

    # shape (Ncameras, 2)
    q0_true[distance] = \
        reproject_perturbed(q0_baseline,
                            1e5 if distance is None else distance,

                            intrinsics_baseline,
                            extrinsics_baseline_mounted,
                            frames_baseline,
                            calobject_warp_baseline,

                            intrinsics_true,
                            extrinsics_true_mounted,
                            frames_true,
                            calobject_warp_true)

    # I check the bias for cameras 0,1,2. Camera 3 has q0 outside of the
    # observed region, so regularization affects projections there dramatically
    # (it's the only contributor to the projection behavior in that area)
    for icam in range(Ncameras):
        if icam == 3:
            continue
        testutils.confirm_equal(q0_true[distance][icam],
                                q0_baseline,
                                eps = 0.1,
                                worstcase = True,
                                msg = f"Regularization bias small-enough for camera {icam} at distance={'infinity' if distance is None else distance}")

for icam in (0,3):
    # I move the extrinsics of a model, write it to disk, and make sure the same
    # uncertainties come back
    model_moved = mrcal.cameramodel(models_baseline[icam])
    model_moved.extrinsics_rt_fromref([1., 2., 3., 4., 5., 6.])
    model_moved.write(f'{workdir}/out.cameramodel')
    model_read = mrcal.cameramodel(f'{workdir}/out.cameramodel')

    icam_intrinsics_read = model_read.icam_intrinsics()
    icam_extrinsics_read = mrcal.corresponding_icam_extrinsics(icam_intrinsics_read,
                                                               **model_read.optimization_inputs())

    testutils.confirm_equal(icam if fixedframes else icam-1,
                            icam_extrinsics_read,
                            msg = f"corresponding icam_extrinsics reported correctly for camera {icam}")

    p_cam_baseline = mrcal.unproject( q0_baseline, *models_baseline[icam].intrinsics(),
                                      normalize = True)

    Var_dq_ref = \
        mrcal.projection_uncertainty( p_cam_baseline * 1.0,
                                      model = models_baseline[icam] )
    Var_dq_moved_written_read = \
        mrcal.projection_uncertainty( p_cam_baseline * 1.0,
                                      model = model_read )
    testutils.confirm_equal(Var_dq_moved_written_read, Var_dq_ref,
                            eps = 0.001,
                            worstcase = True,
                            relative  = True,
                            msg = f"var(dq) with full rt matches for camera {icam} after moving, writing to disk, reading from disk")

    Var_dq_inf_ref = \
        mrcal.projection_uncertainty( p_cam_baseline * 1.0,
                                      model = models_baseline[icam],
                                      atinfinity = True )
    Var_dq_inf_moved_written_read = \
        mrcal.projection_uncertainty( p_cam_baseline * 1.0,
                                      model = model_read,
                                      atinfinity = True )
    testutils.confirm_equal(Var_dq_inf_moved_written_read, Var_dq_inf_ref,
                            eps = 0.001,
                            worstcase = True,
                            relative  = True,
                            msg = f"var(dq) with rotation-only matches for camera {icam} after moving, writing to disk, reading from disk")

    # the at-infinity uncertainty should be invariant to point scalings (the
    # real scaling used is infinity). The not-at-infinity uncertainty is NOT
    # invariant, so I don't check that
    Var_dq_inf_far_ref = \
        mrcal.projection_uncertainty( p_cam_baseline * 100.0,
                                      model = models_baseline[icam],
                                      atinfinity = True )
    testutils.confirm_equal(Var_dq_inf_far_ref, Var_dq_inf_ref,
                            eps = 0.001,
                            worstcase = True,
                            relative  = True,
                            msg = f"var(dq) (infinity) is invariant to point scale for camera {icam}")

if args.no_sampling:
    testutils.finish()
    sys.exit()


intrinsics_sampled         = np.zeros( (args.Nsamples,Ncameras,Nintrinsics), dtype=float )
extrinsics_sampled_mounted = np.zeros( (args.Nsamples,Ncameras,6),           dtype=float )
frames_sampled             = np.zeros( (args.Nsamples,args.Nframes, 6),      dtype=float )
calobject_warp_sampled     = np.zeros( (args.Nsamples, 2),                   dtype=float )

for isample in range(args.Nsamples):
    print(f"Sampling {isample+1}/{args.Nsamples}")

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


def check_uncertainties_at(q0_baseline, idistance):

    distance = args.distances[idistance]

    # distance of "None" means I'll simulate a large distance, but compare
    # against a special-case distance of "infinity"
    if distance is None:
        distance    = 1e5
        atinfinity  = True
        distancestr = "infinity"
    else:
        atinfinity  = False
        distancestr = str(distance)

    # shape (Ncameras,3)
    p_cam_baseline = mrcal.unproject(q0_baseline, lensmodel, intrinsics_baseline,
                                     normalize = True) * distance

    # shape (Nsamples, Ncameras, 2)
    q_sampled = \
        reproject_perturbed(q0_baseline,
                            distance,

                            intrinsics_baseline,
                            extrinsics_baseline_mounted,
                            frames_baseline,
                            calobject_warp_baseline,

                            intrinsics_sampled,
                            extrinsics_sampled_mounted,
                            frames_sampled,
                            calobject_warp_sampled)

    # shape (Ncameras, 2)
    q_sampled_mean = np.mean(q_sampled, axis=-3)

    # shape (Ncameras, 2,2)
    Var_dq_observed = np.mean( nps.outer(q_sampled-q_sampled_mean,
                                         q_sampled-q_sampled_mean), axis=-4 )

    # shape (Ncameras)
    worst_direction_stdev_observed = mrcal.worst_direction_stdev(Var_dq_observed)

    # shape (Ncameras, 2,2)
    Var_dq = \
        nps.cat(*[ mrcal.projection_uncertainty( \
            p_cam_baseline[icam],
            atinfinity = atinfinity,
            model      = models_baseline[icam]) \
                   for icam in range(Ncameras) ])
    # shape (Ncameras)
    worst_direction_stdev_predicted = mrcal.worst_direction_stdev(Var_dq)


    # q_sampled should be evenly distributed around q0_baseline. I can make eps
    # as tight as I want by increasing Nsamples
    testutils.confirm_equal( nps.mag(q_sampled_mean - q0_baseline),
                             0,
                             eps = 0.3,
                             worstcase = True,
                             msg = f"Sampled projections cluster around the sample point at distance = {distancestr}")

    # I accept 20% error. This is plenty good-enough. And I can get tighter matches
    # if I grab more samples
    testutils.confirm_equal(worst_direction_stdev_observed,
                            worst_direction_stdev_predicted,
                            eps = 0.2,
                            worstcase = True,
                            relative  = True,
                            msg = f"Predicted worst-case projections match sampled observations at distance = {distancestr}")

    # I now compare the variances. The cross terms have lots of apparent error,
    # but it's more meaningful to compare the eigenvectors and eigenvalues, so I
    # just do that

    # First, the thing is symmetric, right?
    testutils.confirm_equal(nps.transpose(Var_dq),
                            Var_dq,
                            worstcase = True,
                            msg = f"Var(dq) is symmetric at distance = {distancestr}")

    for icam in range(Ncameras):
        l_predicted,v = sorted_eig(Var_dq[icam])
        v0_predicted  = v[:,0]

        l_observed,v = sorted_eig(Var_dq_observed[icam])
        v0_observed  = v[:,0]

        testutils.confirm_equal(l_observed,
                                l_predicted,
                                eps = 0.35, # high error tolerance. Nsamples is too low for better
                                worstcase = True,
                                relative  = True,
                                msg = f"Var(dq) eigenvalues match for camera {icam} at distance = {distancestr}")

        if icam == 3:
            # I only check the eigenvectors for camera 3. The other cameras have
            # isotropic covariances, so the eigenvectors aren't well defined. If
            # one isn't isotropic for some reason, the eigenvalue check will
            # fail
            testutils.confirm_equal(np.arcsin(nps.mag(np.cross(v0_observed,v0_predicted))) * 180./np.pi,
                                    0,
                                    eps = 15, # high error tolerance. Nsamples is too low for better
                                    worstcase = True,
                                    msg = f"Var(dq) eigenvectors match for camera {icam} at distance = {distancestr}")

            # I don't bother checking v1. I already made sure the matrix is
            # symmetric. Thus the eigenvectors are orthogonal, so any angle offset
            # in v0 will be exactly the same in v1

    return q_sampled,Var_dq


q_sampled,Var_dq = check_uncertainties_at(q0_baseline, 0)
for idistance in range(1,len(args.distances)):
    check_uncertainties_at(q0_baseline, idistance)

if not (args.explore or \
        args.show_distribution or \
        args.make_documentation_plots is not None):
    testutils.finish()


import gnuplotlib as gp

def get_cov_plot_args(q, Var, what):

    l,v   = sorted_eig(Var)
    l0,l1 = l
    v0,v1 = nps.transpose(v)

    major = np.sqrt(l0)
    minor = np.sqrt(l1)

    return \
      ((q[0], q[1], 2*major, 2*minor, 180./np.pi*np.arctan2(v0[1],v0[0]),
        dict(_with='ellipses', tuplesize=5, legend=what)),)

def get_point_cov_plot_args(q, what):
    q_mean  = np.mean(q,axis=-2)
    q_mean0 = q - q_mean
    Var     = np.mean( nps.outer(q_mean0,q_mean0), axis=0 )
    return get_cov_plot_args(q_mean,Var, what)

def make_plot(icam, report_center_points = True, **kwargs):

    q_sampled_mean = np.mean(q_sampled[:,icam,:],axis=-2)

    def make_tuple(*args): return args

    data_tuples = \
        make_tuple((q_sampled[:,icam,0], q_sampled[:,icam,1],
                    dict(_with = 'points pt 6',
                         tuplesize = 2)),
                   *get_point_cov_plot_args(q_sampled[:,icam,:], "Observed uncertainty"),
                   *get_cov_plot_args(q_sampled_mean, Var_dq[icam], "Predicted uncertainty"),)

    if report_center_points:
        data_tuples += \
            ( (q0_baseline,
               dict(tuplesize = -2,
                    _with     = 'points pt 3 ps 3',
                    legend    = 'Baseline center point')),
              (q0_true[args.distances[0]][icam],
               dict(tuplesize = -2,
                    _with     = 'points pt 3 ps 3',
                    legend    = 'True center point')),
              (q_sampled_mean,
               dict(tuplesize = -2,
                    _with     = 'points pt 3 ps 3',
                    legend    = 'Sampled mean')))

    plot_options = \
        dict(square=1,
             _xrange=(q0_baseline[0]-2,q0_baseline[0]+2),
             _yrange=(q0_baseline[1]-2,q0_baseline[1]+2),
             title=f'Uncertainty reprojection distribution for camera {icam}',
             **kwargs)

    return data_tuples, plot_options


if args.show_distribution:
    plot_distribution = [None] * Ncameras
    for icam in range(Ncameras):
        data_tuples, plot_options = make_plot(icam)
        plot_distribution[icam] = gp.gnuplotlib(**plot_options)
        plot_distribution[icam].plot(*data_tuples)

if args.make_documentation_plots is not None:
    data_tuples_plot_options = \
        [ make_plot(icam, report_center_points=False) \
          for icam in range(Ncameras) ]
    if args.make_documentation_plots:
        processoptions_output = dict(wait     = False,
                                     terminal = terminal,
                                     _set     = extraset,
                                     hardcopy = f'distribution-onepoint--{args.make_documentation_plots}')
    else:
        processoptions_output = dict(wait = True)

    plot_options = data_tuples_plot_options[0][1]
    del plot_options['title']
    gnuplotlib_add_list_option(plot_options, 'unset', 'key')
    data_tuples = [ data_tuples_plot_options[icam][0] for icam in range(Ncameras) ]
    gp.plot( *data_tuples,
             **plot_options,
             multiplot = f'layout 2,2',
             **processoptions_output)



    if args.make_documentation_plots:
        processoptions_output = dict(wait     = False,
                                     terminal = terminal,
                                     _set     = extraset,
                                     hardcopy = f'uncertainty-wholeimage--{args.make_documentation_plots}')
    else:
        processoptions_output = dict(wait = True)
    data_tuples_plot_options = \
        [ mrcal.show_projection_uncertainty( models_baseline[icam],
                                             observations     = False,
                                             distance         = args.distances[0],
                                             return_plot_args = True) \
          for icam in range(Ncameras) ]
    plot_options = data_tuples_plot_options[0][1]
    gnuplotlib_add_list_option(plot_options, '_set', processoptions_output['_set'])
    del processoptions_output['_set']
    del plot_options['title']
    gnuplotlib_add_list_option(plot_options, 'unset', 'key')
    gnuplotlib_add_list_option(plot_options, 'unset', 'xtics')
    gnuplotlib_add_list_option(plot_options, 'unset', 'ytics')
    data_tuples = [ data_tuples_plot_options[icam][0] + \
                    [(q0_baseline[0], q0_baseline[1], 0, \
                      dict(tuplesize = 3,
                           _with ='points pt 2 ps 2 nocontour'))] \
                    for icam in range(Ncameras) ]
    gp.plot( *data_tuples,
             **plot_options,
             multiplot = f'layout 2,2',
             **processoptions_output)


if args.explore:
    import IPython
    IPython.embed()
