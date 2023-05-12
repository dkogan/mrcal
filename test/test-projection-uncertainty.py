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
    parser.add_argument('--Ncameras',
                        type    = int,
                        default = 4,
                        help='''How many cameras to simulate. By default we use 4. The values of those 4 are
                        hard-coded, so --Ncameras must be <= 4''')
    parser.add_argument('--distances',
                        type=str,
                        default='5,inf',
                        help='''Comma-separated list of distance where we test the uncertainty predictions.
                        Numbers and "inf" understood. The first value on this
                        list is used for visualization in --show-distribution''')
    parser.add_argument('--do-sample',
                        action='store_true',
                        help='''By default we don't run the time-intensive
                        samples of the calibration solves. This runs a very
                        limited set of tests, and exits. To perform the full set
                        of tests, pass --do-sample''')
    parser.add_argument('--show-distribution',
                        action='store_true',
                        help='''If given, we produce plots showing the distribution of samples''')
    parser.add_argument('--write-models',
                        action='store_true',
                        help='''If given, we write the resulting models to disk for further analysis''')
    parser.add_argument('--make-documentation-plots',
                        type=str,
                        help='''If given, we produce plots for the
                        documentation. Takes one argument: a string describing
                        this test. This will be used in the filenames of the
                        resulting plots. To make interactive plots, pass ""''')
    parser.add_argument('--terminal-pdf',
                        type=str,
                        help='''The gnuplotlib terminal for --make-documentation-plots .PDFs. Omit this
                        unless you know what you're doing''')
    parser.add_argument('--terminal-svg',
                        type=str,
                        help='''The gnuplotlib terminal for --make-documentation-plots .SVGs. Omit this
                        unless you know what you're doing''')
    parser.add_argument('--terminal-png',
                        type=str,
                        help='''The gnuplotlib terminal for --make-documentation-plots .PNGs. Omit this
                        unless you know what you're doing''')
    parser.add_argument('--explore',
                        action='store_true',
                        help='''If given, we drop into a REPL at the end''')
    parser.add_argument('--extra-observation-at',
                        type=float,
                        help='''Adds one extra observation at the given distance''')
    parser.add_argument('--range-to-boards',
                        type=float,
                        default=4.0,
                        help='''Nominal range to the simulated chessboards''')
    parser.add_argument('--reproject-perturbed',
                        choices=('mean-frames',
                                 'mean-frames-using-meanq',
                                 'mean-frames-using-meanq-penalize-big-shifts',
                                 'fit-boards-ref',
                                 'diff',
                                 'optimize-cross-reprojection-error'),
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

if args.Ncameras <= 0 or args.Ncameras > 4:
    print(f"Ncameras must be in [0,4], but got {args.Ncameras}. Giving up", file=sys.stderr)
    sys.exit(1)
if args.fixed == 'frames' and re.match('mean-frames-using-meanq', args.reproject_perturbed):
    print("--fixed frames currently not implemented together with --reproject-perturbed mean-frames-using-meanq.",
          file = sys.stderr)
    sys.exit(1)




testdir = os.path.dirname(os.path.realpath(__file__))

# I import the LOCAL mrcal since that's what I'm testing
sys.path[:0] = f"{testdir}/..",
import mrcal
import testutils
import copy
import numpy as np
import numpysane as nps

from test_calibration_helpers import calibration_baseline,calibration_sample


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




terminal = dict(pdf = args.terminal_pdf,
                svg = args.terminal_svg,
                png = args.terminal_png,
                gp  = 'gp')
pointscale = dict(pdf = 1,
                  svg = 1,
                  png = 1,
                  gp  = 1)
pointscale[""] = 1.



def shorter_terminal(t):
    # Adjust the terminal string to be less tall. Makes the multiplots look
    # better: less wasted space
    m = re.match("(.*)( size.*?,)([0-9.]+)(.*?)$", t)
    if m is None: return t
    return m.group(1) + m.group(2) + str(float(m.group(3))*0.8) + m.group(4)

if args.make_documentation_plots:

    print(f"Will write documentation plots to {args.make_documentation_plots}-xxxx.pdf and .svg")

    if terminal['svg'] is None: terminal['svg'] = 'svg size 800,600       noenhanced solid dynamic    font ",14"'
    if terminal['pdf'] is None: terminal['pdf'] = 'pdf size 8in,6in       noenhanced solid color      font ",12"'
    if terminal['png'] is None: terminal['png'] = 'pngcairo size 1024,768 transparent noenhanced crop font ",12"'

extraset = dict()
for k in pointscale.keys():
    extraset[k] = f'pointsize {pointscale[k]}'

# I want the RNG to be deterministic
np.random.seed(0)

############# Set up my world, and compute all the perfect positions, pixel
############# observations of everything
pixel_uncertainty_stdev = 1.5
object_spacing          = 0.1
object_width_n          = 10
object_height_n         = 9
calobject_warp_true     = np.array((0.002, -0.005))

extrinsics_rt_fromref_true = \
    np.array(((0,    0,    0,      0,   0,   0),
              (0.08, 0.2,  0.02,   1.,  0.9, 0.1),
              (0.01, 0.07, 0.2,    2.1, 0.4, 0.2),
              (-0.1, 0.08, 0.08,   3.4, 0.2, 0.1), ))

optimization_inputs_baseline,                          \
models_true, models_baseline,                          \
indices_frame_camintrinsics_camextrinsics,             \
lensmodel, Nintrinsics, imagersizes,                   \
intrinsics_true, extrinsics_true_mounted, frames_true, \
observations_true,                                     \
args.Nframes =                                         \
    calibration_baseline(args.model,
                         args.Ncameras,
                         args.Nframes,
                         args.extra_observation_at,
                         object_width_n,
                         object_height_n,
                         object_spacing,
                         extrinsics_rt_fromref_true,
                         calobject_warp_true,
                         fixedframes,
                         testdir,
                         range_to_boards = args.range_to_boards)

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
        for extension in ('pdf','svg','png','gp'):
            processoptions_output = dict(wait     = False,
                                         terminal = terminal[extension],
                                         _set     = extraset[extension],
                                         hardcopy = f'{args.make_documentation_plots}--simulated-geometry.{extension}')
            gp.add_plot_option(processoptions_output, 'set', 'xyplane relative 0')
            mrcal.show_geometry(models_baseline,
                                show_calobjects = True,
                                unset='key',
                                **processoptions_output)
    else:
        processoptions_output = dict(wait = True)

        gp.add_plot_option(processoptions_output, 'set', 'xyplane relative 0')
        mrcal.show_geometry(models_baseline,
                            show_calobjects = True,
                            unset='key',
                            **processoptions_output)




    def observed_points(icam):
        obs_cam = observations_true[indices_frame_camintrinsics_camextrinsics[:,1]==icam, ..., :2].ravel()
        return obs_cam.reshape(len(obs_cam)//2,2)

    if args.make_documentation_plots:
        for extension in ('pdf','svg','png','gp'):
            obs_cam = [ ( (observed_points(icam),),
                          (q0_baseline, dict(_with = f'points pt 3 lw 2 lc "red" ps {2*pointscale[extension]}'))) \
                        for icam in range(args.Ncameras) ]
            processoptions_output = dict(wait     = False,
                                         terminal = shorter_terminal(terminal[extension]),
                                         _set     = extraset[extension],
                                         hardcopy = f'{args.make_documentation_plots}--simulated-observations.{extension}')
            gp.plot( *obs_cam,
                     tuplesize=-2,
                     _with='dots',
                     square=1,
                     _xrange=(0, models_true[0].imagersize()[0]-1),
                     _yrange=(models_true[0].imagersize()[1]-1, 0),
                     multiplot = 'layout 2,2',
                     **processoptions_output)
    else:
        obs_cam = [ ( (observed_points(icam),),
                      (q0_baseline, dict(_with = f'points pt 3 lw 2 lc "red" ps {2*pointscale[""]}'))) \
                    for icam in range(args.Ncameras) ]
        processoptions_output = dict(wait = True)
        gp.plot( *obs_cam,
                 tuplesize=-2,
                 _with='dots',
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
    for i in range(args.Ncameras):
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
                                     # dict
                                     baseline_optimization_inputs,

                                     # shape (..., Ncameras, Nintrinsics)
                                     query_intrinsics,
                                     # shape (..., Ncameras, 6)
                                     query_rt_cam_ref,
                                     # shape (..., Nframes, 6)
                                     query_rt_ref_frame,
                                     # shape (..., 2)
                                     query_calobject_warp,
                                     # list of dicts of length ...
                                     query_optimization_inputs):
    r'''Reproject by computing the mean in the space of frames

This is what the uncertainty computation does (as of 2020/10/26). The implied
rotation here is aphysical (it is a mean of multiple rotation matrices)

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

        # shape (..., Nframes, Ncameras, 3)
        p_ref_query_allframes = \
            mrcal.transform_point_rt( nps.dummy(query_rt_ref_frame, -2),
                                      p_frames )

    if args.reproject_perturbed == 'mean-frames':

        # "Normal" path: I take the mean of all the frame-coord-system
        # representations of my point

        if not fixedframes:
            # shape (..., Ncameras, 3)
            p_ref_query = np.mean( p_ref_query_allframes, axis = -3)

        # shape (..., Ncameras, 3)
        p_cam_query = \
            mrcal.transform_point_rt(query_rt_cam_ref, p_ref_query)

        # shape (..., Ncameras, 2)
        return mrcal.project(p_cam_query, lensmodel, query_intrinsics)


    else:

        # Experimental path: I take the mean of the projections, not the points
        # in the reference frame

        # guaranteed that not fixedframes: I asserted this above

        # shape (..., Nframes, Ncameras, 3)
        p_cam_query_allframes = \
            mrcal.transform_point_rt(nps.dummy(query_rt_cam_ref, -3),
                                     p_ref_query_allframes)

        # shape (..., Nframes, Ncameras, 2)
        q_reprojected = mrcal.project(p_cam_query_allframes, lensmodel, nps.dummy(query_intrinsics,-3))

        if args.reproject_perturbed != 'mean-frames-using-meanq-penalize-big-shifts':
            return np.mean(q_reprojected, axis=-3)
        else:
            # Experiment. Weighted mean to de-emphasize points with huge shifts

            w = 1./nps.mag(q_reprojected - q)
            w = nps.mv(nps.cat(w,w),0,-1)
            return \
                np.sum(q_reprojected*w, axis=-3) / \
                np.sum(w, axis=-3)


def reproject_perturbed__fit_boards_ref(q, distance,

                                        # shape (Ncameras, Nintrinsics)
                                        baseline_intrinsics,
                                        # shape (Ncameras, 6)
                                        baseline_rt_cam_ref,
                                        # shape (Nframes, 6)
                                        baseline_rt_ref_frame,
                                        # shape (2)
                                        baseline_calobject_warp,
                                        # dict
                                        baseline_optimization_inputs,

                                        # shape (..., Ncameras, Nintrinsics)
                                        query_intrinsics,
                                        # shape (..., Ncameras, 6)
                                        query_rt_cam_ref,
                                        # shape (..., Nframes, 6)
                                        query_rt_ref_frame,
                                        # shape (..., 2)
                                        query_calobject_warp,
                                        # list of dicts of length ...
                                        query_optimization_inputs):

    r'''Reproject by explicitly computing a procrustes fit to align the reference
    coordinate systems of the two solves. We match up the two sets of chessboard
    points

    '''

    calobject_height,calobject_width = baseline_optimization_inputs['observations_board'].shape[1:3]

    # shape (Nsamples, Nh, Nw, 3)
    if query_calobject_warp.ndim > 1:
        calibration_object_query = \
            nps.cat(*[ mrcal.ref_calibration_object(calobject_width, calobject_height,
                                                    baseline_optimization_inputs['calibration_object_spacing'],
                                                    calobject_warp=calobject_warp) \
                       for calobject_warp in query_calobject_warp] )
    else:
        calibration_object_query = \
            mrcal.ref_calibration_object(calobject_width, calobject_height,
                                         baseline_optimization_inputs['calibration_object_spacing'],
                                         calobject_warp=query_calobject_warp)

    # shape (Nsamples, Nframes, Nh, Nw, 3)
    pcorners_ref_query = \
        mrcal.transform_point_rt( nps.dummy(query_rt_ref_frame, -2, -2),
                                  nps.dummy(calibration_object_query, -4))


    # shape (Nh, Nw, 3)
    calibration_object_baseline = \
        mrcal.ref_calibration_object(calobject_width, calobject_height,
                                     baseline_optimization_inputs['calibration_object_spacing'],
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
                       (),

                       ('Ncameras', 'Nintrinsics'),
                       ('Ncameras', 6),
                       ('Nframes', 6),
                       (2,),
                       ()),

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
                              # dict
                              baseline_optimization_inputs,

                              # shape (Ncameras, Nintrinsics)
                              query_intrinsics,
                              # shape (Ncameras, 6)
                              query_rt_cam_ref,
                              # shape (Nframes, 6)
                              query_rt_ref_frame,
                              # shape (2)
                              query_calobject_warp,
                              # dict
                              query_optimization_inputs):

    r'''Reproject by using the "diff" method to compute a rotation

    '''

    # shape (Ncameras, 3)
    p_cam_baseline = mrcal.unproject(q, lensmodel, baseline_intrinsics,
                                     normalize = True) * distance
    p_cam_query = np.zeros((args.Ncameras, 3), dtype=float)
    for icam in range (args.Ncameras):

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


def reproject_perturbed__optimize_cross_reprojection_error(q, distance,

                                                           # shape (Ncameras, Nintrinsics)
                                                           baseline_intrinsics,
                                                           # shape (Ncameras, 6)
                                                           baseline_rt_cam_ref,
                                                           # shape (Nframes, 6)
                                                           baseline_rt_ref_frame,
                                                           # shape (2)
                                                           baseline_calobject_warp,
                                                           # dict
                                                           baseline_optimization_inputs,

                                                           # shape (..., Ncameras, Nintrinsics)
                                                           query_intrinsics,
                                                           # shape (..., Ncameras, 6)
                                                           query_rt_cam_ref,
                                                           # shape (..., Nframes, 6)
                                                           query_rt_ref_frame,
                                                           # shape (..., 2)
                                                           query_calobject_warp,
                                                           # list of dicts of length ...
                                                           query_optimization_inputs):

    r'''Reproject by explicitly computing a ref-refperturbed transformation

I have a baseline solve (parameter vector b) and a perturbed solve (parameter
vector bperturbed) obtained from perturbing the observations qref and
re-optimizing. I also have an arbitrary baseline query pixel q and distance d
from which I compute the perturbed reprojection qperturbed.

I need to eventually compute Var(qperturbed). I linearize everything to get
delta_qperturbed ~ dqperturbed/dbperturbed dbperturbed/dqref delta_qref. Let

  L = dqperturbed/dbperturbed
  M = dbperturbed/dqref

so

  delta_qperturbed = L M delta_qref

Then

  Var(qperturbed) = L M Var(qref) Mt Lt.

I have M from the usual uncertainty propagation logic, so I just need L =
dqperturbed/dbperturbed

In my usual least squares solve each chessboard point produces two elements
(error(x), error(y)) of the measurements x:

  x_point =
    project(intrinsics,
           T_cam_ref T_ref_frame p)
    - qref

Here I optimize the "cross reprojection error": I look at the PERTURBED
chessboard,frames,points and the UNPERTURBED camera intrinsics, extrinsics. The
data flows from the top-right to the bottom-left:

  ORIGINAL SOLVE                   PERTURBED SOLVE

  point in                         point in
  chessboard                       chessboard
  frame                            frame

    |                                |
    | Trf                            | Tr+f+
    v                                v

  point in                         point in
  ref frame     <-- Trr+ -->       ref frame

    |                                |
    | Tcr                            | Tc+r+
    v                                v

  point in                         point in
  cam frame                        cam frame

    |                                |
    | project                        | project
    v                                v

  pixel                            pixel

This requires computing a ref transformation to take into account the shifting
reference frame that results when re-optimizing:

  x_cross_perturbed =
    project(intrinsics,
            T_cam_ref T_ref_refperturbed T_refperturbed_frameperturbed p_perturbed)
    - qref

For a given perturbation of the input observations I want to compute
T_ref_refperturbed. So here I look at an operating point T_ref_refperturbed = 0.
At the operating point I have x_cross_perturbed0: affected by the input
perturbation, but not any reference transform.

I reoptimize norm2(x_cross_perturbed) by varying T_ref_refperturbed
(parametrized as rt_ref_refperturbed). Let J_cross_perturbed =
dx_cross_perturbed/drt_ref_refperturbed. I assume everything is locally linear,
as defined by J_cross_perturbed. I minimize

  E = norm2(x_cross_perturbed0 + dx_cross_perturbed)

I set the derivative to 0:

  0 = dE/drt_ref_refperturbed ~ (x_cross_perturbed0 + dx_cross_perturbed)t J_cross_perturbed

-> J_cross_perturbed_t x_cross_perturbed0 = -J_cross_perturbed_t dx_cross_perturbed

Furthermore, dx_cross_perturbed = J_cross_perturbed drt_ref_refperturbed, so

  J_cross_perturbed_t x_cross_perturbed0 = -J_cross_perturbed_t J_cross_perturbed drt_ref_refperturbed

and

  drt_ref_refperturbed = -inv(J_cross_perturbed_t J_cross_perturbed) J_cross_perturbed_t x_cross_perturbed0
                       = -pinv(J_cross_perturbed_t) x_cross_perturbed0

The operating point is at rt_ref_refperturbed=0, so the shift is off 0:

  rt_ref_refperturbed = 0 + drt_ref_refperturbed
                      = -pinv(J_cross_perturbed_t) x_cross_perturbed0

This is good, but implies that J_cross_perturbed needs to be computed directly
by propagating gradients from the projection and the transform composition. We
can do better.

Since everything I'm looking at is near the original solution to the main
optimization problem, I can look at EVERYTHING in the linear space defined by
the optimal measurements x and their gradient J:

  x = x0 +
      J_intrinsics     dintrinsics +
      J_extrinsics     drt_cam_ref +
      J_frame          drt_ref_frame +
      J_calobject_warp dcalobject_warp

Once again, we have this expression:

  x_cross_perturbed =
    project(intrinsics,
            T_cam_ref T_ref_refperturbed T_refperturbed_frameperturbed p_perturbed)
    - qref

Here we use the unperturbed intrinsics and extrinsics, so dintrinsics = 0 and
drt_cam_ref = 0. The shift in the calibration object warp comes directly from
the shift in parameters: dcalobject_warp = M[calobject_warp] delta_qref. That
leaves drt_ref_frame. This represents a shift from the optimized rt_ref_frame to

  rt_ref_frameperturbed = compose_rt(rt_ref_refperturbed,rt_refperturbed_frameperturbed).

For x_cross_perturbed0, I have rt_ref_refperturbed = 0, so there I have
drt_ref_frame = M[frame] delta_qref. So I have

  x_cross_perturbed0 =
              x0 +
              J_frame          M[frame]          delta_qref +
              J_calobject_warp M[calobject_warp] delta_qref

  J_cross_perturbed = dx_cross_perturbed/drt_ref_refperturbed
                    = J_frame drt_ref_frameperturbed/drt_ref_refperturbed

Now that I have rt_ref_refperturbed, I can use it to compute qperturbed. This
can accept arbitrary q, not just those in the solve, so I actually need to
compute projections, rather than looking at a linearized space defined by J

I have

  pcam = unproject(intrinsics, q)
  pcam_perturbed = T_camperturbed_refperturbed T_refperturbed_ref T_ref_cam pcam
  qperturbed = project(intrinsics_perturbed, pcam_perturbed)

  dqperturbed/dbperturbed[intrinsics] comes directly from this project()

  dqperturbed/dbperturbed[extrinsics] = dqperturbed/dpcam_perturbed
                                        (dpcam_perturbed/dextrinsics + dpcam_perturbed/drt_ref_refperturbed drt_ref_refperturbed/dextrinsics)


So I need gradients of rt_ref_refperturbed in respect to p_perturbed

    '''

    if fixedframes:
        raise Exception("reproject_perturbed__optimize_cross_reprojection_error(fixedframes = True) is not yet implemented")


    b_packed_baseline, x_baseline, J_packed_baseline, factorization = \
        mrcal.optimizer_callback(**baseline_optimization_inputs)

    observations_board = \
        baseline_optimization_inputs.get('observations_board')
    indices_frame_camintrinsics_camextrinsics = \
        baseline_optimization_inputs['indices_frame_camintrinsics_camextrinsics']

    object_width_n      = observations_board.shape[-2]
    object_height_n     = observations_board.shape[-3]
    object_spacing      = baseline_optimization_inputs['calibration_object_spacing']
    # shape (Nh,Nw,3)
    calibration_object_baseline = \
        mrcal.ref_calibration_object(object_width_n,
                                     object_height_n,
                                     object_spacing,
                                     calobject_warp = baseline_calobject_warp)

    # shape (...,Nh, Nw,3)
    calibration_object_query = \
        mrcal.ref_calibration_object(object_width_n,
                                     object_height_n,
                                     object_spacing,
                                     calobject_warp = query_calobject_warp)

    weight = observations_board[...,2]

    # shape (Nobservations, 6)
    rt_ref_frame_all = \
        baseline_rt_ref_frame[ ..., indices_frame_camintrinsics_camextrinsics[:,0], :]
    # shape (..., Nobservations, 6)
    rt_refperturbed_frameperturbed_all = \
        query_rt_ref_frame   [ ..., indices_frame_camintrinsics_camextrinsics[:,0], :]
    # shape (Nobservations, Nintrinsics)
    intrinsics_all = \
        baseline_intrinsics[ indices_frame_camintrinsics_camextrinsics[:,1], :]
    if nps.norm2(baseline_rt_cam_ref[0]) > 1e-12:
        raise Exception("I'm assuming a vanilla calibration problem reference at cam0")
    # shape (Nobservations, 6)
    rt_cam_ref_all = \
        baseline_rt_cam_ref[ indices_frame_camintrinsics_camextrinsics[:,2]+1, :]


    def get_rt_ref_refperturbed():

        def transform_point_rt3_withgrad_drt1(rt0, rt1, rt2, p):

            def compose_rt3_withgrad_drt1(rt0, rt1, rt2):
                rt01,drt01_drt0,drt01_drt1 = \
                    mrcal.compose_rt(rt0, rt1, get_gradients = True)
                rt012,drt012_drt01,drt012_drt2 = \
                    mrcal.compose_rt(rt01, rt2, get_gradients = True)
                drt012_drt1 = nps.matmult(drt012_drt01, drt01_drt1)
                return rt012,drt012_drt1


            rt012,drt012_drt1 = compose_rt3_withgrad_drt1(rt0,rt1,rt2)
            pp, dpp_drt012, dpp_dp = mrcal.transform_point_rt(rt012, p, get_gradients = True)
            dpp_drt1 = nps.matmult(dpp_drt012, drt012_drt1)
            return pp, dpp_drt1

        def transform_point_identity_gradient(p):
            r'''Computes dprot/drt where prot = transform(rt,p) and rt = identity

    prot = rotate(p) + t, so clearly dprot/dt = I

    Now let's look at the rotation

    mrcal_transform_point_rt_full() from rotate_point_r_core() defines
    prot = rotate(rt, p) at rt=identity:

      const val_withgrad_t<N> cross[3] =
          {
              (rg[1]*x_ing[2] - rg[2]*x_ing[1]),
              (rg[2]*x_ing[0] - rg[0]*x_ing[2]),
              (rg[0]*x_ing[1] - rg[1]*x_ing[0])
          };
      const val_withgrad_t<N> inner =
          rg[0]*x_ing[0] +
          rg[1]*x_ing[1] +
          rg[2]*x_ing[2];

      // Small rotation. I don't want to divide by 0, so I take the limit
      //   lim(th->0, xrot) =
      //     = x + cross(r, x) + r rt x lim(th->0, (1 - cos(th)) / (th*th))
      //     = x + cross(r, x) + r rt x lim(th->0, sin(th) / (2*th))
      //     = x + cross(r, x) + r rt x/2
      for(int i=0; i<3; i++)
          x_outg[i] =
              x_ing[i] +
              cross[i] +
              rg[i]*inner / 2.;

    So dprot/dr = dcross/dr + d(r*inner / 2)/dr =
                  [  0  p2 -p1]
                = [-p2   0  p0] + (inner(r,p) I + outer(r,p))/2
                  [ p1 -p0   0]

    At r=identity I have r = 0, so

                  [  0  p2 -p1]
      dprot/dr  = [-p2   0  p0]
                  [ p1 -p0   0]

    This is the usual skew-symmetric matrix that appears in the matrix form of the
    cross product

            '''

            # strange-looking implementation to make broadcasting work
            dprot_drt = np.zeros(p.shape[:-1] + (3,6), dtype=float)

            mrcal.skew_symmetric(p, out = dprot_drt[..., :3])
            dprot_drt *= -1.

            dprot_drt[...,0,0+3] = 1.
            dprot_drt[...,1,1+3] = 1.
            dprot_drt[...,2,2+3] = 1.

            _,dprot_drt_reference,_ = \
                mrcal.transform_point_rt(mrcal.identity_rt(), p,
                                         get_gradients=True)
            if nps.norm2((dprot_drt-dprot_drt_reference).ravel()) > 1e-10:
                raise Exception("transform_point_identity_gradient() is computing the wrong thing. This is a bug")

            return dprot_drt

        def compose_rt_tinyr0_gradr0(rt1):
            '''
            R0 (R1 p + t1) + t0 = R0 R1 p + (R0 t1 + t0)
            -> R01 = R0 R1
            -> t01 = R0 t1 + t0

            At rt0 = identity we have

               drt01/drt0 = [ dr01/dr0  dr01/dt0  ] = [ dr01/dr0              0 ]
                            [ dt01/dr0  dt01/dt0  ] = [ -skew_symmetric(t1)   I ]

            I call a function to get dr01_dr0
            '''
            drt01_drt0 = np.zeros(rt1.shape + (6,), dtype=float)

            mrcal.skew_symmetric(rt1[..., 3:],
                                 out = drt01_drt0[..., 3:, :3])
            drt01_drt0 *= -1

            drt01_drt0[..., 0+3, 0+3] = 1.
            drt01_drt0[..., 1+3, 1+3] = 1.
            drt01_drt0[..., 2+3, 2+3] = 1.

            mrcal.compose_r_tinyr0_gradientr0(rt1[..., :3],
                                              out = drt01_drt0[..., :3, :3])

            return drt01_drt0

        def get_cross_operating_point__point_grad(pcam, dpcam_drt_ref_refperturbed):

            r'''Compute (x_cross0,J_cross) directly, from a projection

The expression above is

  x_cross_perturbed =
    project(intrinsics,
            T_cam_ref T_ref_refperturbed T_refperturbed_frameperturbed p_perturbed)
    - qref

The operating point as T_ref_refperturbed = identity: rt_ref_refperturbed = 0. So

  x_cross_perturbed0 =
    project(intrinsics,
            T_cam_ref T_refperturbed_frameperturbed p_perturbed)
    - qref

'''
            # shape (..., Nobservations,Nh,Nw,2)
            #       (..., Nobservations,Nh,Nw,2,3)
            q_cross,dq_dpcam,_ = \
                mrcal.project(pcam,
                              baseline_optimization_inputs['lensmodel'],
                              nps.dummy(intrinsics_all, -2,-2),
                              get_gradients = True)
            q_ref = observations_board[...,:2]
            x_cross0 = (q_cross - q_ref)*nps.dummy(weight,-1)
            x_cross0[...,weight<=0,:] = 0 # outliers
            # shape (..., Nobservations*Nh*Nw*2)
            x_cross0 = nps.clump(x_cross0, n=-4)


            dx_dpcam = dq_dpcam*nps.dummy(weight,-1,-1)
            dx_dpcam[...,weight<=0,:,:] = 0 # outliers
            dx_drt_ref_refperturbed = nps.matmult(dx_dpcam, dpcam_drt_ref_refperturbed)
            # shape (...,Nobservations,Nh,Nw,2,6) ->
            #       (...,Nobservations*Nh*Nw*2,6) ->
            dx_drt_ref_refperturbed = \
                nps.mv(nps.clump(nps.mv(dx_drt_ref_refperturbed, -1, -5),
                                 n = -4),
                       -2, -1)
            J_cross = dx_drt_ref_refperturbed
            return x_cross0, J_cross

        # I broadcast over each sample
        @nps.broadcast_define( ((),),
                               (2,),
                               out_kwarg='out')
        def get_cross_operating_point__internal_compose_and_linearization(query_optimization_inputs, out):
            r'''Compute (x_cross0,J_cross) directly, from the optimized linearization

The docstring above says

x_cross_perturbed0 =
            x0 +
            J_frame          M[frame]          delta_qref +
            J_calobject_warp M[calobject_warp] delta_qref

J_cross_perturbed = dx_cross_perturbed/drt_ref_refperturbed
                  = J_frame drt_ref_frameperturbed/drt_ref_refperturbed

The uncertainty computation in
            http://mrcal.secretsauce.net/uncertainty.html concludes that

  M = inv(JtJ) J[observations]t W

What I actually have is b* and J*: the packed, UNITLESS state and the jacobian
            respectively, where

  b = D b*
            J = J* inv(D)

So

  M = D inv(J*tJ*) J*[observations]t W

To select a subset of b I define the matrix S = [0 eye() 0] and the subset is
            S*b. So

  M[frame] = S D inv(J*tJ*) J*[observations]t W

  J_frame = J* inv(D) St

  J_frame M[frame] = J* inv(D) St S D inv(J*tJ*) J*[observations]t W

  St S = diag([0,0,0,... 1,1,1..., 0,0,0,0])

            '''

            imeas0_observations = mrcal.measurement_index_boards(0, **optimization_inputs_baseline)
            Nmeas_observations  = mrcal.num_measurements_boards(**optimization_inputs_baseline)
            J_packed_baseline_observations = J_packed_baseline[imeas0_observations:imeas0_observations+Nmeas_observations,
                                                               :]

            query_qref    = query_optimization_inputs   ['observations_board'][...,:2].ravel()
            baseline_qref = baseline_optimization_inputs['observations_board'][...,:2].ravel()
            delta_qref = query_qref - baseline_qref
            if len(delta_qref) != Nmeas_observations:
                raise Exception("Mismatched observation counts. This is a bug")
            if mrcal.num_measurements_points(**optimization_inputs_baseline) != 0:
                raise Exception("Uncertainty propagation with points not implemented yet")

            # mask out outliers
            weight = baseline_optimization_inputs['observations_board'][...,2].ravel()
            delta_qref_xy = delta_qref.reshape(len(delta_qref)//2, 2)
            if delta_qref_xy.base is not delta_qref:
                raise Exception("reshape() made new array. This is a bug")
            delta_qref_xy[weight <= 0, :] = 0

            # delta_qref <- W delta_qref
            delta_qref_xy *= nps.transpose(weight)


            Jt_W_qref = np.zeros( (J_packed_baseline_observations.shape[-1],), dtype=float)
            mrcal._mrcal_npsp._Jt_x(J_packed_baseline_observations.indptr,
                                    J_packed_baseline_observations.indices,
                                    J_packed_baseline_observations.data,
                                    delta_qref,
                                    out = Jt_W_qref)

            db_predicted = factorization.solve_xt_JtJ_bt( Jt_W_qref )
            mrcal.unpack_state(db_predicted, **baseline_optimization_inputs)

            if 0:
                ############### compare db_observed, db_predicted
                b_packed_query, x_query, J_packed_query, _ = \
                    mrcal.optimizer_callback(**query_optimization_inputs,
                                             no_factorization = True)
                db_observed = b_packed_query - b_packed_baseline
                mrcal.unpack_state(db_observed, **baseline_optimization_inputs)

                gp.plot( nps.cat(db_predicted, db_observed) )

            x_cross = np.array(x_baseline[imeas0_observations:imeas0_observations+Nmeas_observations])
            x_cross_point = x_cross.reshape(len(x_cross)//2, 2)
            if x_cross_point.base is not x_cross:
                raise Exception("reshape() made new array. This is a bug")
            Ny,Nx = baseline_optimization_inputs['observations_board'].shape[1:3]
            if len(x_cross_point) != len(baseline_optimization_inputs['indices_frame_camintrinsics_camextrinsics'])*Nx*Ny:
                raise Exception("mismatched lengths. This is a bug")

            #### Look only at the effects of frames and calobject_warp when
            #### computing the initial x_cross value
            db_cross = np.array(db_predicted)
            mrcal.pack_state(db_cross, **baseline_optimization_inputs)

            state_mask = np.ones( db_predicted.shape, dtype=bool )

            istate_frame0 = mrcal.state_index_frames(0, **baseline_optimization_inputs)
            Nstates_frame = mrcal.num_states_frames(**baseline_optimization_inputs)
            state_mask[istate_frame0 : istate_frame0+Nstates_frame] = 0

            istate_calobject_warp0 = mrcal.state_index_calobject_warp(**baseline_optimization_inputs)
            Nstates_calobject_warp = mrcal.num_states_calobject_warp(**baseline_optimization_inputs)
            state_mask[istate_calobject_warp0 : istate_calobject_warp0+Nstates_calobject_warp] = 0

            db_cross[state_mask] = 0
            x_cross += J_packed_baseline_observations.dot(db_cross)


            #### Now J_cross = J_frame drt_ref_frame/drt_ref_refperturbed
            b_baseline = np.array(b_packed_baseline)
            mrcal.unpack_state(b_baseline, **baseline_optimization_inputs)

            Nframes = Nstates_frame//6
            rt_ref_frame = b_baseline[istate_frame0 : istate_frame0+Nstates_frame].reshape(Nframes,6)
            drt_ref_frame__drt_ref_refperturbed = compose_rt_tinyr0_gradr0(rt_ref_frame)

            ##### I MANULLY PACK. GOOD-ENOUGH ONLY AS A TEST. HARD-CODES THESE FROM mrcal.c:
            #define SCALE_ROTATION_FRAME          (15.0 * M_PI/180.0)
            #define SCALE_TRANSLATION_FRAME       1.0
            drt_ref_frame__drt_ref_refperturbed[:, :3, :] /= (15.0 * np.pi/180.0)
            drt_ref_frame__drt_ref_refperturbed[:, 3:, :] /= (1.0)

            ##### I ASSUME ONE FRAME POSE PER SET OF OBSERVATIONS. WORKS WITH ONE CAMERA ONLY
            drt_ref_frame__drt_ref_refperturbed = nps.clump(drt_ref_frame__drt_ref_refperturbed, n=2)

            J_cross = J_packed_baseline_observations[:, istate_frame0:istate_frame0+Nstates_frame].dot(drt_ref_frame__drt_ref_refperturbed)

            # Workaround for a numpy bug:
            # https://github.com/numpy/numpy/issues/19470
            out[:] = (x_cross, J_cross)
            return out




        # I look at the un-perturbed data first, to double-check that I'm doing the
        # right thing. This is purely a self-checking step. I don't need to do it
        if 1:

            # shape (Nobservations,Nh,Nw,3),
            pref = mrcal.transform_point_rt(nps.dummy(rt_ref_frame_all, -2,-2),
                                            calibration_object_baseline)
            pcam = mrcal.transform_point_rt(nps.dummy(rt_cam_ref_all, -2,-2),
                                            pref)

            # shape (..., Nobservations,Nh,Nw,2)
            qq = \
                mrcal.project(pcam,
                              baseline_optimization_inputs['lensmodel'],
                              nps.dummy(intrinsics_all, -2,-2))
            x = (qq - observations_board[...,:2])*nps.dummy(weight,-1)
            x[...,weight<=0,:] = 0 # outliers
            x = x.ravel()
            if len(x) != mrcal.num_measurements_boards(**baseline_optimization_inputs):
                raise Exception("Unexpected len(x). This is a bug")
            if nps.norm2(x - x_baseline[:len(x)]) > 1e-12:
                raise Exception("Unexpected x. This is a bug")

            E_baseline = nps.norm2(x)

        # Can select the other get_cross_operating_point__...() implementations
        # here
        if 0:
            # shape (..., Nobservations,Nh,Nw,3),
            #       (..., Nobservations,Nh,Nw,3,6)
            pcam, dpcam_drt_ref_refperturbed = \
                transform_point_rt3_withgrad_drt1(nps.dummy(rt_cam_ref_all, -2,-2),
                                                  mrcal.identity_rt(),
                                                  nps.dummy(rt_refperturbed_frameperturbed_all, -2,-2),
                                                  nps.mv(calibration_object_query,-4,-5))

            x_cross, J_cross = get_cross_operating_point__point_grad(pcam, dpcam_drt_ref_refperturbed)

        elif 1:
            # shape (..., Nobservations,Nh,Nw,3),
            pref = mrcal.transform_point_rt( nps.dummy(rt_refperturbed_frameperturbed_all, -2,-2),
                                             nps.mv(calibration_object_query,-4,-5))
            dpref_drt_ref_refperturbed = transform_point_identity_gradient(pref)
            # shape (..., Nobservations,Nh,Nw,3),
            #       (..., Nobservations,Nh,Nw,3,6)
            pcam, _, dpcam_dpref = \
                mrcal.transform_point_rt(nps.dummy(rt_cam_ref_all, -2,-2),
                                         pref,
                                         get_gradients = True)
            dpcam_drt_ref_refperturbed = \
                nps.matmult(dpcam_dpref, dpref_drt_ref_refperturbed)

            x_cross, J_cross = get_cross_operating_point__point_grad(pcam, dpcam_drt_ref_refperturbed)

        else:
            if query_optimization_inputs is not None:

                xJ_cross = np.empty( (len(query_optimization_inputs),2), dtype=object)
                get_cross_operating_point__internal_compose_and_linearization( np.array(query_optimization_inputs,
                                                                                        dtype=object),
                                                                               out = xJ_cross)
                x_cross = np.array(tuple(xJ_cross[:,0]))
                J_cross = np.array(tuple(xJ_cross[:,1]))

            else:
                print("No query_optimization_inputs available. Cannot compute rt_ref_refperturbed")
                return None


            # These should all match. They do (manual testing), but that should be
            # automated. I'm moving on.


        # need to define the broadcasted function myself
        @nps.broadcast_define((('N',6),('N',)),
                              (6,))
        def lstsq(J,x):
            # inv(JtJ)Jt x0
            return np.linalg.lstsq(J, x, rcond = None)[0]
        E_cross_ref0 = nps.norm2(x_cross)
        rt_ref_refperturbed = -lstsq(J_cross, x_cross)


        # I have a 1-step solve. Let's look at the error to confirm that it's
        # smaller. This is an optional validation step
        if 1:
            # shape (..., Nobservations,Nh,Nw,3)
            pcam = \
                mrcal.transform_point_rt( mrcal.compose_rt(nps.dummy(rt_cam_ref_all, -2,-2),
                                                           nps.mv(rt_ref_refperturbed, -2,-5),
                                                           nps.dummy(rt_refperturbed_frameperturbed_all, -2,-2)),
                                          nps.mv(calibration_object_query,-4,-5))

            # shape (..., Nobservations,Nh,Nw,2)
            q_cross = \
                mrcal.project(pcam,
                              baseline_optimization_inputs['lensmodel'],
                              nps.dummy(intrinsics_all, -2,-2),
                              get_gradients = False)
            x_cross = (q_cross - observations_board[...,:2])*nps.dummy(weight,-1)
            x_cross[...,weight<=0,:] = 0 # outliers
            # shape (..., Nobservations*Nh*Nw*2)
            x_cross = nps.clump(x_cross, n=-4)

            E_cross_solvedref = nps.norm2(x_cross)

            print(f"RMS error baseline            = {np.sqrt(E_baseline        / (x_cross.shape[-1]/2))} pixels")
            print(f"RMS error perturbed           = {np.sqrt(E_cross_ref0      / (x_cross.shape[-1]/2))} pixels")
            print(f"RMS error perturbed_solvedref = {np.sqrt(E_cross_solvedref / (x_cross.shape[-1]/2))} pixels")


        return rt_ref_refperturbed






    rt_ref_refperturbed = get_rt_ref_refperturbed()




    # shape (Ncameras, 3)
    p_cam_baseline = mrcal.unproject(q, lensmodel, baseline_intrinsics,
                                     normalize = True) * distance
    # shape (Ncameras, 3)
    p_ref_baseline = \
        mrcal.transform_point_rt( baseline_rt_cam_ref,
                                  p_cam_baseline,
                                  inverted = True)
    # shape (...,Ncameras, 3)
    p_ref_query = \
        mrcal.transform_point_rt( nps.dummy(rt_ref_refperturbed, -2),
                                  p_ref_baseline,
                                  inverted = True )

    # shape (..., Ncameras, 3)
    p_cam_query = \
        mrcal.transform_point_rt(query_rt_cam_ref, p_ref_query)

    # shape (..., Ncameras, 2)
    #
    # If ... was (), the current code sets it to (1,). So I force the right
    # shape
    return mrcal.project(p_cam_query, lensmodel, query_intrinsics). \
        reshape( *query_intrinsics.shape[:-1],
                 2)



# Which implementation we're using. Use the method that matches the uncertainty
# computation. Thus the sampled ellipsoids should match the ellipsoids reported
# by the uncertianty method
if   re.match('mean-frames', args.reproject_perturbed):
    reproject_perturbed = reproject_perturbed__mean_frames
elif args.reproject_perturbed == 'fit-boards-ref':
    reproject_perturbed = reproject_perturbed__fit_boards_ref
elif args.reproject_perturbed == 'diff':
    reproject_perturbed = reproject_perturbed__diff
elif args.reproject_perturbed == 'optimize-cross-reprojection-error':
    reproject_perturbed = reproject_perturbed__optimize_cross_reprojection_error
else:
    raise Exception("getting here is a bug")




q0_true = dict()
for distance in args.distances:

    # shape (Ncameras, 2)
    q0_true_here = \
        reproject_perturbed(q0_baseline,
                            1e5 if distance is None else distance,

                            intrinsics_baseline,
                            extrinsics_baseline_mounted,
                            frames_baseline,
                            calobject_warp_baseline,
                            optimization_inputs_baseline,

                            intrinsics_true,
                            extrinsics_true_mounted,
                            frames_true,
                            calobject_warp_true,
                            # optimization_inputs_sampled not available here:
                            # the "true" values aren't the result of an
                            # optimization. Subsequent calls to
                            # reproject_perturbed() will have
                            # optimization_inputs_sampled available
                            None)

    if q0_true_here is not None:
        q0_true[distance] = q0_true_here
    else:
        # Couldn't compute. Probably because we needed
        # optimization_inputs_sampled
        q0_true = None
        break

    # I check the bias for cameras 0,1. Cameras 2,3 have q0 outside of the
    # chessboard region, or right on its edge, so regularization DOES affect
    # projections there: it's the only contributor to the projection behavior in
    # that area
    for icam in range(args.Ncameras):
        if icam == 2 or icam == 3:
            continue
        testutils.confirm_equal(q0_true[distance][icam],
                                q0_baseline,
                                eps = 0.1,
                                worstcase = True,
                                msg = f"Regularization bias small-enough for camera {icam} at distance={'infinity' if distance is None else distance}")

for icam in (0,3):
    # I move the extrinsics of a model, write it to disk, and make sure the same
    # uncertainties come back

    if icam >= args.Ncameras: break

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
                                      model = models_baseline[icam],
                                      observed_pixel_uncertainty = pixel_uncertainty_stdev)
    Var_dq_moved_written_read = \
        mrcal.projection_uncertainty( p_cam_baseline * 1.0,
                                      model = model_read,
                                      observed_pixel_uncertainty = pixel_uncertainty_stdev )
    testutils.confirm_equal(Var_dq_moved_written_read, Var_dq_ref,
                            eps = 0.001,
                            worstcase = True,
                            relative  = True,
                            msg = f"var(dq) with full rt matches for camera {icam} after moving, writing to disk, reading from disk")

    Var_dq_inf_ref = \
        mrcal.projection_uncertainty( p_cam_baseline * 1.0,
                                      model = models_baseline[icam],
                                      atinfinity = True,
                                      observed_pixel_uncertainty = pixel_uncertainty_stdev )
    Var_dq_inf_moved_written_read = \
        mrcal.projection_uncertainty( p_cam_baseline * 1.0,
                                      model = model_read,
                                      atinfinity = True,
                                      observed_pixel_uncertainty = pixel_uncertainty_stdev )
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
                                      atinfinity = True,
                                      observed_pixel_uncertainty = pixel_uncertainty_stdev )
    testutils.confirm_equal(Var_dq_inf_far_ref, Var_dq_inf_ref,
                            eps = 0.001,
                            worstcase = True,
                            relative  = True,
                            msg = f"var(dq) (infinity) is invariant to point scale for camera {icam}")

if not args.do_sample:
    testutils.finish()
    sys.exit()


( intrinsics_sampled,         \
  extrinsics_sampled_mounted, \
  frames_sampled,             \
  points_sampled,             \
  calobject_warp_sampled,     \
  optimization_inputs_sampled ) = \
      calibration_sample( args.Nsamples,
                          optimization_inputs_baseline,
                          pixel_uncertainty_stdev,
                          fixedframes)


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
                            optimization_inputs_baseline,

                            intrinsics_sampled,
                            extrinsics_sampled_mounted,
                            frames_sampled,
                            calobject_warp_sampled,
                            optimization_inputs_sampled)

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
            model      = models_baseline[icam],
            observed_pixel_uncertainty = pixel_uncertainty_stdev) \
                   for icam in range(args.Ncameras) ])
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
    for icam in range(args.Ncameras):
        testutils.confirm_covariances_equal(Var_dq[icam],
                                            Var_dq_observed[icam],
                                            what = f"camera {icam} at distance = {distancestr}",
                                            # high error tolerances; Nsamples is too low for better
                                            eps_eigenvalues      = 0.35,
                                            eps_eigenvectors_deg = 15)

    return q_sampled,Var_dq


q_sampled,Var_dq = check_uncertainties_at(q0_baseline, 0)
for idistance in range(1,len(args.distances)):
    check_uncertainties_at(q0_baseline, idistance)

if not (args.explore or \
        args.show_distribution or \
        args.make_documentation_plots is not None):
    testutils.finish()
    sys.exit()

import gnuplotlib as gp

def make_plot(icam, report_center_points = True, **kwargs):

    q_sampled_mean = np.mean(q_sampled[:,icam,:],axis=-2)

    def make_tuple(*args): return args

    data_tuples = \
        make_tuple(*mrcal.utils._plot_args_points_and_covariance_ellipse(q_sampled[:,icam,:], "Observed uncertainty"),
                   mrcal.utils._plot_arg_covariance_ellipse(q_sampled_mean, Var_dq[icam], "Predicted uncertainty"),)

    if report_center_points:

        if q0_true is not None:
            data_tuple_true_center_point = \
                ( (q0_true[args.distances[0]][icam],
                   dict(tuplesize = -2,
                        _with     = 'points pt 3 ps 3',
                        legend    = 'True center point')), )
        else:
            print("q0_true is None; not plotting the true center point")
            data_tuple_true_center_point = ()

        data_tuples += \
            ( (q0_baseline,
               dict(tuplesize = -2,
                    _with     = 'points pt 3 ps 3',
                    legend    = 'Baseline center point')),
              (q_sampled_mean,
               dict(tuplesize = -2,
                    _with     = 'points pt 3 ps 3',
                    legend    = 'Sampled mean')) ) \
            + data_tuple_true_center_point

    plot_options = \
        dict(square=1,
             _xrange=(q0_baseline[0]-2,q0_baseline[0]+2),
             _yrange=(q0_baseline[1]-2,q0_baseline[1]+2),
             title=f'Uncertainty reprojection distribution for camera {icam}',
             **kwargs)

    gp.add_plot_option(plot_options, 'set', ('xtics 1', 'ytics 1'))

    return data_tuples, plot_options


if args.show_distribution:
    plot_distribution = [None] * args.Ncameras
    for icam in range(args.Ncameras):
        data_tuples, plot_options = make_plot(icam)

        if args.extra_observation_at is not None:
            plot_options['title'] += f': boards at {args.range_to_boards}m, extra one at {args.extra_observation_at}m'

        plot_distribution[icam] = gp.gnuplotlib(**plot_options)
        plot_distribution[icam].plot(*data_tuples)

if args.make_documentation_plots is not None:
    data_tuples_plot_options = \
        [ make_plot(icam, report_center_points=False) \
          for icam in range(args.Ncameras) ]
    plot_options = data_tuples_plot_options[0][1]
    del plot_options['title']
    gp.add_plot_option(plot_options, 'unset', 'key')
    data_tuples = [ data_tuples_plot_options[icam][0] for icam in range(args.Ncameras) ]

    if args.make_documentation_plots:
        for extension in ('pdf','svg','png','gp'):
            processoptions_output = dict(wait     = False,
                                         terminal = terminal[extension],
                                         _set     = extraset[extension],
                                         hardcopy = f'{args.make_documentation_plots}--distribution-onepoint.{extension}')
            gp.plot( *data_tuples,
                     **plot_options,
                     multiplot = f'layout 2,2',
                     **processoptions_output)
    else:
        processoptions_output = dict(wait = True)
        gp.plot( *data_tuples,
                 **plot_options,
                 multiplot = f'layout 2,2',
                 **processoptions_output)



    data_tuples_plot_options = \
        [ mrcal.show_projection_uncertainty( models_baseline[icam],
                                             observed_pixel_uncertainty = pixel_uncertainty_stdev,
                                             observations          = 'dots',
                                             distance              = args.distances[0],
                                             contour_increment     = -0.4,
                                             contour_labels_styles = '',
                                             return_plot_args      = True) \
          for icam in range(args.Ncameras) ]
    plot_options = data_tuples_plot_options[0][1]
    del plot_options['title']
    gp.add_plot_option(plot_options, 'unset', 'key')
    gp.add_plot_option(plot_options, 'set',   ('xtics 1000', 'ytics 1000'))

    if args.make_documentation_plots:
        for extension in ('pdf','svg','png','gp'):
            if extension == 'png':
                continue
            data_tuples = [ data_tuples_plot_options[icam][0] + \
                            [(q0_baseline[0], q0_baseline[1], 0, \
                              dict(tuplesize = 3,
                                   _with =f'points pt 3 lw 2 lc "red" ps {2*pointscale[extension]} nocontour'))] \
                            for icam in range(args.Ncameras) ]

            # look through all the plots
            #   look through all the data tuples in each plot
            #     look at the last data tuple
            #     if it's dict(_with = 'dots ...'): ignore it
            def is_datatuple_withdots(t):
                d = t[-1]
                if type(d) is not dict: return False
                if '_with' not in d:    return False
                w = d['_with']
                if type(w) is not str:  return False
                return re.match('dots',w) is not None
            def subplots_noobservations(p):
                return \
                    [ t for t in p if not is_datatuple_withdots(t) ]
            data_tuples_no_observations = \
                [ subplots_noobservations(p) for p in data_tuples ]


            for obs_yesno, dt in (('observations',   data_tuples),
                                  ('noobservations', data_tuples_no_observations)):
                processoptions_output = dict(wait     = False,
                                             terminal = shorter_terminal(terminal[extension]),
                                             _set     = extraset[extension],
                                             hardcopy = f'{args.make_documentation_plots}--uncertainty-wholeimage-{obs_yesno}.{extension}')
                if '_set' in processoptions_output:
                    gp.add_plot_option(plot_options, 'set', processoptions_output['_set'])
                    del processoptions_output['_set']
                gp.plot( *dt,
                         **plot_options,
                         multiplot = f'layout 2,2',
                         **processoptions_output)
    else:
        data_tuples = [ data_tuples_plot_options[icam][0] + \
                        [(q0_baseline[0], q0_baseline[1], 0, \
                          dict(tuplesize = 3,
                               _with =f'points pt 3 lw 2 lc "red" ps {2*pointscale[""]} nocontour'))] \
                        for icam in range(args.Ncameras) ]
        processoptions_output = dict(wait = True)
        if '_set' in processoptions_output:
            gp.add_plot_option(plot_options, 'set', processoptions_output['_set'])
            del processoptions_output['_set']
        gp.plot( *data_tuples,
                 **plot_options,
                 multiplot = f'layout 2,2',
                 **processoptions_output)


if args.explore:
    import IPython
    IPython.embed()
