#!/usr/bin/python3

r'''Simulates different chessboard dances to find the best technique

We want the shortest chessboard dances that produce the most confident results.
In a perfect world would put the chessboard in all the locations where we would
expect to use the visual system, since the best confidence is obtained in
regions where the chessboard was observed.

However, due to geometric constraints it is sometimes impossible to put the
board in the right locations. This tool clearly shows that filling the field of
view produces best results. But very wide lenses require huge chessboards,
displayed very close to the lens in order to fill the field of view. This means
that using a wide lens to look out to infinity will always result in potentially
too little projection confidence. This tool is intended to find the kind of
chessboard dance to get good confidences by simulating different geometries and
dances.

We arrange --Ncameras cameras horizontally, with an identity rotation, evenly
spaced with a spacing of --camera-spacing meters. The left camera is at the
origin.

We show the cameras lots of dense chessboards ("dense" meaning that every camera
sees all the points of all the chessboards). The chessboard come in two
clusters: "near" and "far". Each cluster is centered straight ahead of the
midpoint of all the cameras, with some random noise on the position and
orientation. The distances from the center of the cameras to the center of the
clusters are given by --range. This tool solves the calibration problem, and
generates uncertainty-vs-range curves. Each run of this tool generates a family
of this curve, for different values of Nframes-far, the numbers of chessboard
observations in the "far" cluster.

'''

import sys
import argparse
import re
import os

def parse_args():

    def positive_float(string):
        try:
            value = float(string)
        except:
            raise argparse.ArgumentTypeError("argument MUST be a positive floating-point number. Got '{}'".format(string))
        if value <= 0:
            raise argparse.ArgumentTypeError("argument MUST be a positive floating-point number. Got '{}'".format(string))
        return value
    def positive_int(string):
        try:
            value = int(string)
        except:
            raise argparse.ArgumentTypeError("argument MUST be a positive integer. Got '{}'".format(string))
        if value <= 0 or abs(value-float(string)) > 1e-6:
            raise argparse.ArgumentTypeError("argument MUST be a positive integer. Got '{}'".format(string))
        return value

    parser = \
        argparse.ArgumentParser(description = __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--icam-uncertainty',
                        default = 0,
                        type=int,
                        help='''Which camera to use for the uncertainty reporting. I use the left-most one
                        (camera 0) by default''')
    parser.add_argument('--camera-spacing',
                        default = 0.3,
                        type=positive_float,
                        help='How many meters between adjacent cameras in our synthetic world')
    parser.add_argument('--object-spacing',
                        default=0.077,
                        type=float,
                        help='Width of each square in the calibration board, in meters')
    parser.add_argument('--object-width-n',
                        type=int,
                        default=10,
                        help='''How many points the calibration board has per horizontal side. If omitted we
                        default to 10''')
    parser.add_argument('--object-height-n',
                        type=int,
                        default=10,
                        help='''How many points the calibration board has per vertical side. If omitted, we
                        default to 10''')
    parser.add_argument('--observed-pixel-uncertainty',
                        type=positive_float,
                        default = 1.0,
                        help='''The standard deviation of x and y pixel coordinates of the input observations
                        I generate. The distribution of the inputs is gaussian,
                        with the standard deviation specified by this argument.
                        Note: this is the x and y standard deviation, treated
                        independently. If each of these is s, then the LENGTH of
                        the deviation of each pixel is a Rayleigh distribution
                        with expected value s*sqrt(pi/2) ~ s*1.25''')
    parser.add_argument('--lensmodel',
                        required=False,
                        type=str,
                        help='''Which lens model to use for the simulation. If omitted, we use the model
                        given on the commandline. We may want to use a
                        parametric model to generate data (model on the
                        commandline), but a richer splined model to solve''')

    parser.add_argument('--show-geometry-each-solve',
                        action = 'store_true',
                        help='''If given, display the camera, chessboard geometry after each solve. Used for
                        debugging''')
    parser.add_argument('--show-uncertainty-each-solve',
                        action = 'store_true',
                        help='''If given, display the uncertainty and observations after each solve. Used for
                        debugging''')
    parser.add_argument('--ymax',
                        type=float,
                        default = 10.0,
                        help='''If given, use this as the upper extent of the uncertainty plot.''')
    parser.add_argument('--uncertainty-at-range-sampled-min',
                        type=float,
                        help='''If given, use this as the lower bound of the ranges we look at when
                        evaluating projection confidence''')
    parser.add_argument('--uncertainty-at-range-sampled-max',
                        type=float,
                        help='''If given, use this as the upper bound of the ranges we look at when
                        evaluating projection confidence''')
    parser.add_argument('--explore',
                        action='store_true',
                        help='''Drop into a REPL at the end''')

    parser.add_argument('--scan-ranges',
                        action='store_true',
                        help='''Study the effect of range-to-camera on uncertainty. We will try out ranges
                        between the two --ranges values. --tilt-radius,
                        --Nframes, --Ncameras, control the set point. Exactly
                        one of the --scan-... arguments must be given''')
    parser.add_argument('--scan-tilts',
                        action='store_true',
                        help='''Study the effect of chessboard tilt on uncertainty. We will try out random
                        tilt radiuses between the two --tilt-radius values.
                        --range, --Nframes, --Ncameras control the set point.
                        Exactly one of the --scan-... arguments must be given''')
    parser.add_argument('--scan-Nframes',
                        action='store_true',
                        help='''Study the effect of chessboard observation counts on uncertainty. We will try
                        out Nframes between the two --Nframes values. --range,
                        --tilt-radius, --Ncameras control the set point. Exactly
                        one of the --scan-... arguments must be given''')
    parser.add_argument('--scan-Ncameras',
                        action='store_true',
                        help='''Study the effect of camera counts on uncertainty. We will try out Ncameras
                        between the two --Ncameras values. --range,
                        --tilt-radius, --Nframes control the set point. Exactly
                        one of the --scan-... arguments must be given''')
    parser.add_argument('--scan-num-far-constant-Nframes-near',
                        action='store_true',
                        help='''Study the effect of far-away chessboard observations added to a constant set
                        of near chessboard observations. The "far" and "near"
                        ranges are set by --range. Just like
                        --scan-num-far-constant-Nframes-all, except here "far"
                        observations are ADDED to "near" observations. The
                        "near" and "far" ranges are given in --range. The number
                        of "near" frames is given in --Nframes-near. Exactly one
                        of the --scan-... arguments must be given''')
    parser.add_argument('--scan-num-far-constant-Nframes-all',
                        action='store_true',
                        help='''Study the effect of far-away chessboard observations replacing existing
                        "near" chessboard observations. The "far" and "near"
                        ranges are set by --range. Just like
                        --scan-num-far-constant-Nframes-near, except here "far"
                        observations REPLACE existing "near" observations. The
                        "near" and "far" ranges are given in --range. The number
                        of total frames is given in --Nframes-all. Exactly one
                        of the --scan-... arguments must be given''')

    parser.add_argument('--range',
                        default = '0.5,4.0',
                        type=str,
                        help='''if --scan-num-far-...: this is "NEAR,FAR"; specifying the near and far ranges
                        to the chessboard to evaluate. if --scan-ranges this is
                        "MIN,MAX"; specifying the extents of the ranges to
                        evaluate. Otherwise: this is RANGE, the one range of
                        chessboards to evaluate''')
    parser.add_argument('--tilt-radius',
                        default='30.',
                        type=str,
                        help='''The radius of the uniform distribution used to sample the pitch and yaw of
                        the chessboard observations, in degrees. The default is
                        30, meaning that the chessboard pitch and yaw are
                        sampled from [-30deg,30deg]. if --scan-tilts: this is
                        TILT-MIN,TILT-MAX specifying the bounds of the two
                        tilt-radius values to evaluate. Otherwise: this is the
                        one value we use''')
    parser.add_argument('--Ncameras',
                        default = '1',
                        type=str,
                        help='''How many cameras oberve our synthetic world. By default we just have one
                        camera. if --scan-Ncameras: this is
                        NCAMERAS-MIN,NCAMERAS-MAX specifying the bounds of the
                        camera counts to evaluate. Otherwise: this is the one
                        Ncameras value to use''')
    parser.add_argument('--Nframes',
                        type=str,
                        help='''How many observed frames we have. Ignored if --scan-num-far-... if
                        --scan-Nframes: this is NFRAMES-MIN,NFRAMES-MAX specifying the bounds of the
                        frame counts to evaluate. Otherwise: this is the one Nframes value to use''')

    parser.add_argument('--Nframes-near',
                        type=positive_int,
                        help='''Used if --scan-num-far-constant-Nframes-near. The number of "near" frames is
                        given by this argument, while we look at the effect of
                        adding more "far" frames.''')
    parser.add_argument('--Nframes-all',
                        type=positive_int,
                        help='''Used if --scan-num-far-constant-Nframes-all. The number of "near"+"far"
                        frames is given by this argument, while we look at the
                        effect of replacing "near" frames with "far" frames.''')

    parser.add_argument('--Nscan-samples',
                        type=positive_int,
                        default=8,
                        help='''How many values of the parameter being scanned to evaluate. By default, the
                        scan evaluates 8 different values''')

    parser.add_argument('--hardcopy',
                        help='''Filename to plot into. If omitted, we make an interactive plot''')

    parser.add_argument('model',
                        type = str,
                        help='''Baseline camera model. I use the intrinsics from this model to generate
                        synthetic data. We probably want the "true" model to not
                        be too crazy, so this should probably by a parametric
                        (not splined) model''')

    return parser.parse_args()

args = parse_args()

Nscanargs = 0
if args.scan_ranges:                        Nscanargs += 1
if args.scan_tilts:                         Nscanargs += 1
if args.scan_Ncameras:                      Nscanargs += 1
if args.scan_Nframes:                       Nscanargs += 1
if args.scan_num_far_constant_Nframes_near: Nscanargs += 1
if args.scan_num_far_constant_Nframes_all:  Nscanargs += 1

if Nscanargs != 1:
    print("Exactly one of --scan-... must be given", file=sys.stderr)
    sys.exit(1)


# arg-parsing is done before the imports so that --help works without building
# stuff, so that I can generate the manpages and README


import numpy as np
import numpysane as nps
import gnuplotlib as gp
import copy

sys.path[:0] = '../..',
import mrcal


def split_list(s, t):
    r'''Splits a comma-separated list

    None    -> None
    "A,B,C" -> (A,B,C)
    "A"     -> A
    '''
    if s is None: return None
    l = [t(x) for x in s.split(',')]
    if len(l) == 1: return l[0]
    return l

def first(s):
    r'''first in a list, or the value if a scalar'''
    if hasattr(s, '__iter__'): return s[0]
    return s

def last(s):
    r'''last in a list, or the value if a scalar'''
    if hasattr(s, '__iter__'): return s[-1]
    return s

args.tilt_radius = split_list(args.tilt_radius, float)
args.range       = split_list(args.range,       float)
args.Ncameras    = split_list(args.Ncameras,    int)
args.Nframes     = split_list(args.Nframes,     int)

Nuncertainty_at_range_samples = 80
uncertainty_at_range_sampled_min = args.uncertainty_at_range_sampled_min
if uncertainty_at_range_sampled_min is None:
    uncertainty_at_range_sampled_min = first(args.range)/10.
uncertainty_at_range_sampled_max = args.uncertainty_at_range_sampled_max
if uncertainty_at_range_sampled_max is None:
    uncertainty_at_range_sampled_max = last(args.range) *10.

uncertainty_at_range_samples = \
    np.logspace( np.log10(uncertainty_at_range_sampled_min),
                 np.log10(uncertainty_at_range_sampled_max),
                 Nuncertainty_at_range_samples)


if   args.scan_ranges:
    # --range MIN,MAX
    # --tilt-radius TILT-RAD
    # --Ncameras N
    # --Nframes N
    if args.Nframes is None:
        print("The given --scan-... requires --Nframes", file=sys.stderr)
        sys.exit(1)
    if not hasattr(args.range, '__iter__') or len(args.range) != 2:
        print("The given --scan-... requires --range with 2 arguments", file=sys.stderr)
        sys.exit(1)
    if hasattr(args.tilt_radius, '__iter__'):
        print("The given --scan-... requires --tilt-radius with 1 argument", file=sys.stderr)
        sys.exit(1)
    if hasattr(args.Nframes, '__iter__'):
        print("The given --scan-... requires --Nframes with 1 argument", file=sys.stderr)
        sys.exit(1)
    if hasattr(args.Ncameras, '__iter__'):
        print("The given --scan-... requires --Ncameras with 1 argument", file=sys.stderr)
        sys.exit(1)
    if args.Nframes_near is not None or args.Nframes_all is not None:
        print("The given --scan-... does not use --Nframes-near or --Nframes-all", file=sys.stderr)
        sys.exit(1)

elif args.scan_tilts:
    # --range RANGE
    # --tilt-radius TILT-RAD-MIN,TILT-RAD-MAX
    # --Ncameras N
    # --Nframes N
    if args.Nframes is None:
        print("The given --scan-... requires --Nframes", file=sys.stderr)
        sys.exit(1)
    if hasattr(args.range, '__iter__'):
        print("The given --scan-... requires --range with 1 argument", file=sys.stderr)
        sys.exit(1)
    if not hasattr(args.tilt_radius, '__iter__') or len(args.tilt_radius) != 2:
        print("The given --scan-... requires --tilt-radius with 2 arguments", file=sys.stderr)
        sys.exit(1)
    if hasattr(args.Nframes, '__iter__'):
        print("The given --scan-... requires --Nframes with 1 argument", file=sys.stderr)
        sys.exit(1)
    if hasattr(args.Ncameras, '__iter__'):
        print("The given --scan-... requires --Ncameras with 1 argument", file=sys.stderr)
        sys.exit(1)
    if args.Nframes_near is not None or args.Nframes_all is not None:
        print("The given --scan-... does not use --Nframes-near or --Nframes-all", file=sys.stderr)
        sys.exit(1)

elif args.scan_Ncameras:
    # --range RANGE
    # --tilt-radius TILT-RAD
    # --Ncameras NMIN,NMAX
    # --Nframes N
    if args.Nframes is None:
        print("The given --scan-... requires --Nframes", file=sys.stderr)
        sys.exit(1)
    if hasattr(args.range, '__iter__'):
        print("The given --scan-... requires --range with 1 argument", file=sys.stderr)
        sys.exit(1)
    if hasattr(args.tilt_radius, '__iter__'):
        print("The given --scan-... requires --tilt-radius with 1 argument", file=sys.stderr)
        sys.exit(1)
    if hasattr(args.Nframes, '__iter__'):
        print("The given --scan-... requires --Nframes with 1 argument", file=sys.stderr)
        sys.exit(1)
    if not hasattr(args.Ncameras, '__iter__') or len(args.Ncameras) != 2:
        print("The given --scan-... requires --Ncameras with 2 arguments", file=sys.stderr)
        sys.exit(1)
    if args.Nframes_near is not None or args.Nframes_all is not None:
        print("The given --scan-... does not use --Nframes-near or --Nframes-all", file=sys.stderr)
        sys.exit(1)

elif args.scan_Nframes:
    # --range RANGE
    # --tilt-radius TILT-RAD
    # --Ncameras N
    # --Nframes NMIN,NMAX
    if args.Nframes is None:
        print("The given --scan-... requires --Nframes", file=sys.stderr)
        sys.exit(1)
    if hasattr(args.range, '__iter__'):
        print("The given --scan-... requires --range with 1 argument", file=sys.stderr)
        sys.exit(1)
    if hasattr(args.tilt_radius, '__iter__'):
        print("The given --scan-... requires --tilt-radius with 1 argument", file=sys.stderr)
        sys.exit(1)
    if not hasattr(args.Nframes, '__iter__') or len(args.Nframes) != 2:
        print("The given --scan-... requires --Nframes with 2 arguments", file=sys.stderr)
        sys.exit(1)
    if hasattr(args.Ncameras, '__iter__'):
        print("The given --scan-... requires --Ncameras with 1 argument", file=sys.stderr)
        sys.exit(1)
    if args.Nframes_near is not None or args.Nframes_all is not None:
        print("The given --scan-... does not use --Nframes-near or --Nframes-all", file=sys.stderr)
        sys.exit(1)

elif args.scan_num_far_constant_Nframes_near:
    # --Nframes-near N
    # --range RANGE-NEAR,RANGE-FAR
    # --tilt-radius TILT-RAD
    # --Ncameras N
    if args.Nframes_near is None:
        print("The given --scan-... requires --Nframes-near", file=sys.stderr)
        sys.exit(1)
    if not hasattr(args.range, '__iter__') or len(args.range) != 2:
        print("The given --scan-... requires --range with 2 arguments", file=sys.stderr)
        sys.exit(1)
    if hasattr(args.tilt_radius, '__iter__'):
        print("The given --scan-... requires --tilt-radius with 1 argument", file=sys.stderr)
        sys.exit(1)
    if hasattr(args.Ncameras, '__iter__'):
        print("The given --scan-...requires --Ncameras with 1 argument", file=sys.stderr)
        sys.exit(1)
    if args.Nframes is not None or args.Nframes_all is not None:
        print("The given --scan-... does not use --Nframes or --Nframes-all", file=sys.stderr)
        sys.exit(1)

elif args.scan_num_far_constant_Nframes_all:
    # --Nframes-all N
    # --range RANGE-NEAR,RANGE-FAR
    # --tilt-radius TILT-RAD
    # --Ncameras N
    if args.Nframes_all is None:
        print("The given --scan-... requires --Nframes-all", file=sys.stderr)
        sys.exit(1)
    if not hasattr(args.range, '__iter__') or len(args.range) != 2:
        print("The given --scan-... requires --range with 2 arguments", file=sys.stderr)
        sys.exit(1)
    if hasattr(args.tilt_radius, '__iter__'):
        print("The given --scan-... requires --tilt-radius with 1 argument", file=sys.stderr)
        sys.exit(1)
    if hasattr(args.Ncameras, '__iter__'):
        print("--scan-tilts requires --Ncameras with 1 argument", file=sys.stderr)
        sys.exit(1)
    if args.Nframes is not None or args.Nframes_near is not None:
        print("The given --scan-... does not use --Nframes or --Nframes-near", file=sys.stderr)
        sys.exit(1)

else:
    raise Exception("getting here is a bug")


# I want the RNG to be deterministic
np.random.seed(0)

model_intrinsics = mrcal.cameramodel(args.model)

calobject_warp_true_ref = np.array((0.002, -0.005))


def solve(Ncameras,
          Nframes_near, Nframes_far,
          models_true,

          # q.shape             = (Nframes, Ncameras, object_height, object_width, 2)
          # Rt_cam0_board.shape = (Nframes, 4,3)

          q_true_near, Rt_cam0_board_true_near,
          q_true_far,  Rt_cam0_board_true_far):

    q_true_near             = q_true_near            [:Nframes_near]
    Rt_cam0_board_true_near = Rt_cam0_board_true_near[:Nframes_near]

    if q_true_far is not None:
        q_true_far              = q_true_far             [:Nframes_far ]
        Rt_cam0_board_true_far  = Rt_cam0_board_true_far [:Nframes_far ]
    else:
        q_true_far              = np.zeros( (0,) + q_true_near.shape[1:],
                                            dtype = q_true_near.dtype)
        Rt_cam0_board_true_far  = np.zeros( (0,) + Rt_cam0_board_true_near.shape[1:],
                                            dtype = Rt_cam0_board_true_near.dtype)

    calobject_warp_true = calobject_warp_true_ref.copy()

    Nframes_all = Nframes_near + Nframes_far

    Rt_cam0_board_true = nps.glue( Rt_cam0_board_true_near,
                                   Rt_cam0_board_true_far,
                                   axis = -3 )

    # Dense observations. All the cameras see all the boards
    indices_frame_camera = np.zeros( (Nframes_all*Ncameras, 2), dtype=np.int32)
    indices_frame = indices_frame_camera[:,0].reshape(Nframes_all,Ncameras)
    indices_frame.setfield(nps.outer(np.arange(Nframes_all, dtype=np.int32),
                                     np.ones((Ncameras,), dtype=np.int32)),
                           dtype = np.int32)
    indices_camera = indices_frame_camera[:,1].reshape(Nframes_all,Ncameras)
    indices_camera.setfield(nps.outer(np.ones((Nframes_all,), dtype=np.int32),
                                     np.arange(Ncameras, dtype=np.int32)),
                           dtype = np.int32)
    indices_frame_camintrinsics_camextrinsics = \
        nps.glue(indices_frame_camera,
                 indices_frame_camera[:,(1,)],
                 axis=-1)
    indices_frame_camintrinsics_camextrinsics[:,2] -= 1

    q = nps.glue( q_true_near,
                  q_true_far,
                  axis = -5 )

    # apply noise
    q += np.random.randn(*q.shape) * args.observed_pixel_uncertainty

    # The observations are dense (in the data every camera sees all the
    # chessboards), but some of the observations WILL be out of bounds. I
    # pre-mark those as outliers so that the solve doesn't do weird stuff

    # Set the weights to 1 initially
    # shape (Nframes, Ncameras, object_height_n, object_width_n, 3)
    observations = nps.glue(q,
                            np.ones( q.shape[:-1] + (1,) ),
                            axis = -1)

    # shape (Ncameras, 1, 1, 2)
    imagersizes = nps.mv( nps.cat(*[ m.imagersize() for m in models_true ]),
                          -2, -4 )

    # mark the out-of-view observations as outliers
    observations[ np.any( q              < 0, axis=-1 ), 2 ] = -1.
    observations[ np.any( q-imagersizes >= 0, axis=-1 ), 2 ] = -1.

    # shape (Nobservations, Nh, Nw, 2)
    observations = nps.clump( observations,
                   n = 2 )

    intrinsics = nps.cat( *[m.intrinsics()[1]         for m in models_true]     )
    extrinsics = nps.cat( *[m.extrinsics_rt_fromref() for m in models_true[1:]] )
    if len(extrinsics) == 0: extrinsics = None

    if nps.norm2(models_true[0].extrinsics_rt_fromref()) > 1e-6:
        raise Exception("models_true[0] must sit at the origin")
    imagersizes = nps.cat( *[m.imagersize() for m in models_true] )

    optimization_inputs = \
        dict( # intrinsics filled in later
              extrinsics_rt_fromref                     = extrinsics,
              frames_rt_toref                           = mrcal.rt_from_Rt(Rt_cam0_board_true),
              points                                    = None,
              observations_board                        = observations,
              indices_frame_camintrinsics_camextrinsics = indices_frame_camintrinsics_camextrinsics,
              observations_point                        = None,
              indices_point_camintrinsics_camextrinsics = None,
              # lensmodel filled in later
              calobject_warp                            = copy.deepcopy(calobject_warp_true),
              imagersizes                               = imagersizes,
              calibration_object_spacing                = args.object_spacing,
              verbose                                   = False,
              observed_pixel_uncertainty                = args.observed_pixel_uncertainty,
              # do_optimize_frames filled in later
              # do_optimize_extrinsics filled in later
              # do_optimize_intrinsics_core filled in later
              do_optimize_intrinsics_distortions        = True,
              do_optimize_calobject_warp                = False, # turn this on, and reoptimize later
              do_apply_regularization                   = True,
              do_apply_outlier_rejection                = False)


    if args.lensmodel is None:
        lensmodel = model_intrinsics.intrinsics()[0]
    else:
        lensmodel = args.lensmodel
    Nintrinsics = mrcal.lensmodel_num_params(lensmodel)

    if re.search("SPLINED", lensmodel):

        # These are already mostly right, So I lock them down while I seed the
        # intrinsics
        optimization_inputs['do_optimize_frames']          = False
        optimization_inputs['do_optimize_extrinsics']      = False

        # I pre-optimize the core, and then lock it down
        optimization_inputs['lensmodel']                   = 'LENSMODEL_STEREOGRAPHIC'
        optimization_inputs['intrinsics']                  = intrinsics[:,:4].copy()
        optimization_inputs['do_optimize_intrinsics_core'] = True

        stats = mrcal.optimize(**optimization_inputs)
        print(f"optimized. rms = {stats['rms_reproj_error__pixels']}")

        # core is good. Lock that down, and get an estimate for the control
        # points
        optimization_inputs['do_optimize_intrinsics_core'] = False
        optimization_inputs['lensmodel']                   = lensmodel
        optimization_inputs['intrinsics']                  = nps.glue(optimization_inputs['intrinsics'],
                                                                      np.zeros((Ncameras,Nintrinsics-4),),axis=-1)
        stats = mrcal.optimize(**optimization_inputs)
        print(f"optimized. rms = {stats['rms_reproj_error__pixels']}")

        # Ready for a final reoptimization with the geometry
        optimization_inputs['do_optimize_frames']          = True
        optimization_inputs['do_optimize_extrinsics']      = True

    else:
        optimization_inputs['lensmodel']                   = lensmodel
        if not mrcal.lensmodel_metadata(lensmodel)['has_core'] or \
           not mrcal.lensmodel_metadata(model_intrinsics.intrinsics()[0])['has_core']:
            raise Exception("I'm assuming all the models here have a core. It's just lazy coding. If you see this, feel free to fix.")

        if lensmodel == model_intrinsics.intrinsics()[0]:
            # Same model. Grab the intrinsics. They're 99% right
            optimization_inputs['intrinsics']              = intrinsics.copy()
        else:
            # Different model. Grab the intrinsics core, optimize the rest
            optimization_inputs['intrinsics']              = nps.glue(intrinsics[:,:4],
                                                                      np.zeros((Ncameras,Nintrinsics-4),),axis=-1)
        optimization_inputs['do_optimize_intrinsics_core'] = True
        optimization_inputs['do_optimize_frames']          = True
        optimization_inputs['do_optimize_extrinsics']      = True

    stats = mrcal.optimize(**optimization_inputs)
    print(f"optimized. rms = {stats['rms_reproj_error__pixels']}")

    optimization_inputs['do_optimize_calobject_warp'] = True
    stats = mrcal.optimize(**optimization_inputs)
    print(f"optimized. rms = {stats['rms_reproj_error__pixels']}")

    return optimization_inputs


def observation_centroid(optimization_inputs, icam):
    r'''mean pixel coordinate of all non-outlier points seen by a given camera'''

    ifcice       = optimization_inputs['indices_frame_camintrinsics_camextrinsics']
    observations = optimization_inputs['observations_board']

    # pick the camera I want
    observations = observations[ifcice[:,1] == icam]

    # ignore outliers
    q = observations[ (observations[...,2] > 0), :2]

    return np.mean(q, axis=-2)


def eval_one_rangenear_tilt(models_true,
                            range_near, range_far, tilt_radius,
                            uncertainty_at_range_samples,
                            Ncameras,
                            Nframes_near_samples, Nframes_far_samples):

    # I want the RNG to be deterministic
    np.random.seed(0)

    uncertainties = np.zeros((len(Nframes_far_samples),
                              len(uncertainty_at_range_samples)),
                             dtype=float)


    radius_cameras = (args.camera_spacing * (Ncameras-1)) / 2.

    # shapes (Nframes, Ncameras, Nh, Nw, 2),
    #        (Nframes, 4,3)
    q_true_near, Rt_cam0_board_true_near = \
        mrcal.synthesize_board_observations(models_true,
                                            args.object_width_n,
                                            args.object_height_n,
                                            args.object_spacing,
                                            calobject_warp_true_ref,
                                            np.array((0.,  0., 0., radius_cameras, 0,  range_near,)),
                                            np.array((np.pi/180. * tilt_radius,
                                                      np.pi/180. * tilt_radius,
                                                      np.pi/180. * 20.,
                                                      range_near*2. + radius_cameras,
                                                      range_near*2.,
                                                      range_near/10.)),
                                            np.max(Nframes_near_samples),
                                            which = 'some_cameras_must_see_half_board')
    if range_far is not None:
        q_true_far, Rt_cam0_board_true_far  = \
            mrcal.synthesize_board_observations(models_true,
                                                args.object_width_n,
                                                args.object_height_n,
                                                args.object_spacing,
                                                calobject_warp_true_ref,
                                                np.array((0.,  0., 0., radius_cameras, 0,  range_far,)),
                                                np.array((np.pi/180. * tilt_radius,
                                                          np.pi/180. * tilt_radius,
                                                          np.pi/180. * 20.,
                                                          range_far*2. + radius_cameras,
                                                          range_far*2.,
                                                          range_far/10.)),
                                                np.max(Nframes_far_samples),
                                                which = 'some_cameras_must_see_half_board')
    else:
        q_true_far             = None
        Rt_cam0_board_true_far = None

    for i_Nframes_far in range(len(Nframes_far_samples)):

        Nframes_far  = Nframes_far_samples [i_Nframes_far]
        Nframes_near = Nframes_near_samples[i_Nframes_far]

        optimization_inputs = solve(Ncameras,
                                    Nframes_near, Nframes_far,
                                    models_true,
                                    q_true_near, Rt_cam0_board_true_near,
                                    q_true_far,  Rt_cam0_board_true_far)

        models_out = \
            [ mrcal.cameramodel( optimization_inputs = optimization_inputs,
                                 icam_intrinsics     = icam ) \
              for icam in range(Ncameras) ]

        model = models_out[args.icam_uncertainty]

        if args.show_geometry_each_solve:
            mrcal.show_calibration_geometry(models_out,
                                            wait = True)
        if args.show_uncertainty_each_solve:
            mrcal.show_projection_uncertainty(model,
                                              observations= True,
                                              wait        = True)

        # shape (N,3)
        # I sample the center of the imager
        pcam_samples = \
            mrcal.unproject( observation_centroid(optimization_inputs,
                                                  args.icam_uncertainty),
                             *model.intrinsics(),
                             normalize = True) * \
                             nps.dummy(uncertainty_at_range_samples, -1)

        uncertainties[i_Nframes_far] = \
            mrcal.projection_uncertainty(pcam_samples,
                                         model,
                                         what='worstdirection-stdev')

    return uncertainties






if   args.scan_num_far_constant_Nframes_near or \
     args.scan_num_far_constant_Nframes_all:

    Nfar_samples = args.Nscan_samples
    if   args.scan_num_far_constant_Nframes_near:
        Nframes_far_samples = np.linspace(0,
                                          args.Nframes_near*2,
                                          Nfar_samples, dtype=int)
        Nframes_near_samples = Nframes_far_samples*0 + args.Nframes_near

    else:
        Nframes_far_samples  = np.linspace(0,
                                           args.Nframes_all,
                                           Nfar_samples, dtype=int)
        Nframes_near_samples = args.Nframes_all - Nframes_far_samples

    models_true = \
        [ mrcal.cameramodel(intrinsics          = model_intrinsics.intrinsics(),
                            imagersize          = model_intrinsics.imagersize(),
                            extrinsics_rt_toref = np.array((0,0,0,
                                                            i*args.camera_spacing,
                                                            0,0), dtype=float) ) \
          for i in range(args.Ncameras) ]

    uncertainties = \
        eval_one_rangenear_tilt(models_true,
                                *args.range,
                                args.tilt_radius,
                                uncertainty_at_range_samples,
                                args.Ncameras,
                                Nframes_near_samples, Nframes_far_samples)

    samples = Nframes_far_samples

elif args.scan_ranges:
    Nframes_near_samples = np.array( (args.Nframes,), dtype=int)
    Nframes_far_samples  = np.array( (0,),            dtype=int)

    models_true = \
        [ mrcal.cameramodel(intrinsics          = model_intrinsics.intrinsics(),
                            imagersize          = model_intrinsics.imagersize(),
                            extrinsics_rt_toref = np.array((0,0,0,
                                                            i*args.camera_spacing,
                                                            0,0), dtype=float) ) \
          for i in range(args.Ncameras) ]

    Nrange_samples = args.Nscan_samples
    range_samples = np.linspace(*args.range,
                                Nrange_samples, dtype=float)
    uncertainties = np.zeros((Nrange_samples, Nuncertainty_at_range_samples),
                             dtype=float)
    for i_range in range(Nrange_samples):
        uncertainties[i_range] = \
            eval_one_rangenear_tilt(models_true,
                                    range_samples[i_range], None,
                                    args.tilt_radius,
                                    uncertainty_at_range_samples,
                                    args.Ncameras,
                                    Nframes_near_samples, Nframes_far_samples)[0]

    samples = range_samples

elif args.scan_tilts:
    Nframes_near_samples = np.array( (args.Nframes,), dtype=int)
    Nframes_far_samples  = np.array( (0,),            dtype=int)

    models_true = \
        [ mrcal.cameramodel(intrinsics          = model_intrinsics.intrinsics(),
                            imagersize          = model_intrinsics.imagersize(),
                            extrinsics_rt_toref = np.array((0,0,0,
                                                            i*args.camera_spacing,
                                                            0,0), dtype=float) ) \
          for i in range(args.Ncameras) ]

    Ntilt_rad_samples = args.Nscan_samples
    tilt_rad_samples = np.linspace(*args.tilt_radius,
                                   Ntilt_rad_samples, dtype=float)
    uncertainties = np.zeros((Ntilt_rad_samples, Nuncertainty_at_range_samples),
                             dtype=float)
    for i_tilt in range(Ntilt_rad_samples):
        uncertainties[i_tilt] = \
            eval_one_rangenear_tilt(models_true,
                                    args.range, None,
                                    tilt_rad_samples[i_tilt],
                                    uncertainty_at_range_samples,
                                    args.Ncameras,
                                    Nframes_near_samples, Nframes_far_samples)[0]

    samples = tilt_rad_samples

elif args.scan_Ncameras:

    Nframes_near_samples = np.array( (args.Nframes,), dtype=int)
    Nframes_far_samples  = np.array( (0,),            dtype=int)

    N_Ncameras_samples = args.Nscan_samples
    Ncameras_samples = np.linspace(*args.Ncameras,
                                   N_Ncameras_samples, dtype=int)
    uncertainties = np.zeros((N_Ncameras_samples, Nuncertainty_at_range_samples),
                             dtype=float)
    for i_Ncameras in range(N_Ncameras_samples):

        Ncameras = Ncameras_samples[i_Ncameras]
        models_true = \
            [ mrcal.cameramodel(intrinsics          = model_intrinsics.intrinsics(),
                                imagersize          = model_intrinsics.imagersize(),
                                extrinsics_rt_toref = np.array((0,0,0,
                                                                i*args.camera_spacing,
                                                                0,0), dtype=float) ) \
              for i in range(Ncameras) ]

        uncertainties[i_Ncameras] = \
            eval_one_rangenear_tilt(models_true,
                                    args.range, None,
                                    args.tilt_radius,
                                    uncertainty_at_range_samples,
                                    Ncameras,
                                    Nframes_near_samples, Nframes_far_samples)[0]

    samples = Ncameras_samples

elif args.scan_Nframes:

    Nframes_far_samples  = np.array( (0,),            dtype=int)

    models_true = \
        [ mrcal.cameramodel(intrinsics          = model_intrinsics.intrinsics(),
                            imagersize          = model_intrinsics.imagersize(),
                            extrinsics_rt_toref = np.array((0,0,0,
                                                            i*args.camera_spacing,
                                                            0,0), dtype=float) ) \
          for i in range(args.Ncameras) ]

    N_Nframes_samples = args.Nscan_samples
    Nframes_samples = np.linspace(*args.Nframes,
                                  N_Nframes_samples, dtype=int)
    uncertainties = np.zeros((N_Nframes_samples, Nuncertainty_at_range_samples),
                             dtype=float)
    for i_Nframes in range(N_Nframes_samples):

        Nframes = Nframes_samples[i_Nframes]
        Nframes_near_samples = np.array( (Nframes,), dtype=int)

        uncertainties[i_Nframes] = \
            eval_one_rangenear_tilt(models_true,
                                    args.range, None,
                                    args.tilt_radius,
                                    uncertainty_at_range_samples,
                                    args.Ncameras,
                                    Nframes_near_samples, Nframes_far_samples)[0]

    samples = Nframes_samples

else:
    raise Exception("Getting here is a bug")



if isinstance(args.range, float):
    guides = [ f"arrow nohead dashtype 3 from {args.range},graph 0 to {args.range},graph 1" ]
else:
    guides = [ f"arrow nohead dashtype 3 from {r},graph 0 to {r},graph 1" for r in args.range ]
guides.append(f"arrow nohead dashtype 3 from graph 0,first {args.observed_pixel_uncertainty} to graph 1,first {args.observed_pixel_uncertainty}")


if   args.scan_num_far_constant_Nframes_near:
    title = f"Scanning 'far' observations added to a set of 'near' observations. Have {args.Ncameras} cameras, {args.Nframes_near} 'near' observations, at ranges {args.range}."
    legend_what = 'Nframes_far'
elif args.scan_num_far_constant_Nframes_all:
    title = f"Scanning 'far' observations replacing 'near' observations. Have {args.Ncameras} cameras, {args.Nframes_all} total observations, at ranges {args.range}."
    legend_what = 'Nframes_far'
elif args.scan_Nframes:
    title = f"Scanning Nframes. Have {args.Ncameras} cameras looking out at {args.range}m."
    legend_what = 'Nframes'
elif args.scan_Ncameras:
    title = f"Scanning Ncameras. Observing {args.Nframes} boards at {args.range}m."
    legend_what = 'Ncameras'
elif args.scan_ranges:
    title = f"Scanning the distance to observations. Have {args.Ncameras} cameras looking at {args.Nframes} boards."
    legend_what = 'Range-to-chessboards'
elif args.scan_tilts:
    title = f"Scanning the board tilt. Have {args.Ncameras} cameras looking at {args.Nframes} boards at {args.range}m"
    legend_what = 'Random chessboard tilt radius'
else:
    raise Exception("Getting here is a bug")


if samples.dtype.kind == 'i':
    legend = np.array([ f"{legend_what} = {x}" for x in samples])
else:
    legend = np.array([ f"{legend_what} = {x:.1f}" for x in samples])

gp.plot(uncertainty_at_range_samples,
        uncertainties,
        legend   = legend,
        yrange   = (0, args.ymax),
        _with    = 'lines',
        _set     = guides,
        unset    = 'grid',
        title    = title,
        xlabel   = 'Range (m)',
        ylabel   = 'Expected worst-direction uncertainty (pixels)',
        hardcopy = args.hardcopy,
        wait     = not args.explore and args.hardcopy is None)

if args.explore:
    import IPython
    IPython.embed()
sys.exit()
