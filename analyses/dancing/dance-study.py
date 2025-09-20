#!/usr/bin/env python3

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

This tool scans some parameter (selected by --scan), and reports the
uncertainty-vs-range for the different values of this parameter (as a plot and
as an output vnl).

If we don't want to scan any parameter, and just want a single
uncertainty-vs-range plot, don't pass --scan.

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
                        type=float,
                        help='How many meters between adjacent cameras in our synthetic world')
    parser.add_argument('--object-spacing',
                        default="0.077",
                        type=str,
                        help='''Width of each square in the calibration board, in meters. If --scan
                        object_width_n, this is used as the spacing for the
                        SMALLEST width. As I change object_width_n, I adjust
                        object_spacing accordingly, to keep the total board size
                        constant. If --scan object_spacing, this is MIN,MAX to
                        specify the bounds of the scan. In that case
                        object_width_n stays constant, so the board changes
                        size''')
    parser.add_argument('--object-width-n',
                        type=str,
                        help='''How many points the calibration board has per horizontal side. If both are
                        omitted, we default the width and height to 10 (both
                        must be specified to use a non-default value). If --scan
                        object_width_n, this is MIN,MAX to specify the bounds of
                        the scan. In that case I assume a square object, and
                        ignore --object-height-n. Scanning object-width-n keeps
                        the board size constant''')
    parser.add_argument('--object-height-n',
                        type=int,
                        help='''How many points the calibration board has per vertical side. If both are
                        omitted, we default the width and height to 10 (both
                        must be specified to use a non-default value). If --scan
                        object_width_n, this is ignored, and set equivalent to
                        the object width''')
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
    parser.add_argument('--skip-calobject-warp-solve',
                        action='store_true',
                        default=False,
                        help='''By default we assume the calibration target is
                        slightly deformed, and we compute this deformation. If
                        we want to assume that the chessboard shape is fixed,
                        pass this option. The actual shape of the board is given
                        by --calobject-warp''')
    parser.add_argument('--calobject-warp',
                        type=float,
                        nargs=2,
                        default=(0.002, -0.005),
                        help='''The "calibration-object warp". These specify the
                        flex of the chessboard. By default, the board is
                        slightly warped (as is usually the case in real life).
                        To use a perfectly flat board, specify "0 0" here''')

    parser.add_argument('--show-geometry-first-solve',
                        action = 'store_true',
                        help='''If given, display the camera, chessboard geometry after the first solve, and
                        exit. Used for debugging''')
    parser.add_argument('--show-uncertainty-first-solve',
                        action = 'store_true',
                        help='''If given, display the uncertainty (at infinity) and observations after the
                        first solve, and exit. Used for debugging''')
    parser.add_argument('--write-models-first-solve',
                        action = 'store_true',
                        help='''If given, write the solved models to disk after the first solve, and exit.
                        Used for debugging. Useful to check fov_x_deg when solving for a splined model''')

    parser.add_argument('--which',
                        choices=('all-cameras-must-see-full-board',
                                 'some-cameras-must-see-full-board',
                                 'all-cameras-must-see-half-board',
                                 'some-cameras-must-see-half-board'),
                        default='all-cameras-must-see-half-board',
                        help='''What kind of random poses are accepted. Default
                        is all-cameras-must-see-half-board''')

    parser.add_argument('--fixed-frames',
                        action='store_true',
                        help='''Optimize the geometry keeping the chessboard frames fixed. This reduces the
                        freedom of the solution, and produces more confident
                        calibrations. It's possible to use this in reality by
                        surverying the chessboard poses''')
    parser.add_argument('--method',
                        choices=('mean-pcam', 'bestq', 'cross-reprojection-rrp-Jfp'),
                        default='mean-pcam',
                        help='''Multiple uncertainty quantification methods are available. We default to 'mean-pcam' ''')
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

    parser.add_argument('--scan',
                        type=str,
                        default='',
                        choices=('range',
                                 'tilt_radius',
                                 'Nframes',
                                 'Ncameras',
                                 'object_width_n',
                                 'object_spacing',
                                 'num_far_constant_Nframes_near',
                                 'num_far_constant_Nframes_all'),
                        help='''Study the effect of some parameter on uncertainty. The parameter is given as
                        an argument ot this function. Valid choices:
                        ('range','tilt_radius','Nframes','Ncameras',
                        'object_width_n', 'object_spacing', 'num_far_constant_Nframes_near',
                        'num_far_constant_Nframes_all'). Scanning object-width-n
                        keeps the board size constant''')
    parser.add_argument('--scan-object-spacing-compensate-range-from',
                        type=float,
                        help=f'''Only applies if --scan object-spacing. By
                        default we vary the object spacings without touching
                        anything else: the chessboard grows in size as we
                        increase the spacing. If given, we try to keep the
                        apparent size constant: as the object spacing grows, we
                        increase the range. The nominal range given in this
                        argument''')

    parser.add_argument('--range',
                        default = '0.5,4.0',
                        type=str,
                        help='''if --scan num_far_...: this is "NEAR,FAR"; specifying the near and far ranges
                        to the chessboard to evaluate. if --scan range this is
                        "MIN,MAX"; specifying the extents of the ranges to
                        evaluate. Otherwise: this is RANGE, the one range of
                        chessboards to evaluate''')
    parser.add_argument('--tilt-radius',
                        default='30.',
                        type=str,
                        help='''The radius of the uniform distribution used to
                        sample the pitch and yaw of the chessboard
                        observations, in degrees. The default is 30, meaning
                        that the chessboard pitch and yaw are sampled from
                        [-30deg,30deg]. if --scan tilt_radius: this is
                        TILT-MIN,TILT-MAX specifying the bounds of the two
                        tilt-radius values to evaluate. Otherwise: this is the
                        one value we use''')
    parser.add_argument('--roll-radius',
                        type=float,
                        default=20.,
                        help='''The synthetic chessboard orientation is sampled
                        from a uniform distribution: [-RADIUS,RADIUS]. The
                        pitch,yaw radius is specified by the --tilt-radius. The
                        roll radius is selected here. "Roll" is the rotation
                        around the axis normal to the chessboard plane. Default
                        is 20deg''')
    parser.add_argument('--x-radius',
                        type=float,
                        default=None,
                        help='''The synthetic chessboard position is sampled
                        from a uniform distribution: [-RADIUS,RADIUS]. The x
                        radius is selected here. "x" is direction in the
                        chessboard plane, and is also the axis along which the
                        cameras are distributed. A resonable default (possibly
                        range-dependent) is chosen if omitted. MUST be None if
                        --scan num_far_...''')
    parser.add_argument('--y-radius',
                        type=float,
                        default=None,
                        help='''The synthetic chessboard position is sampled
                        from a uniform distribution: [-RADIUS,RADIUS]. The y
                        radius is selected here. "y" is direction in the
                        chessboard plane, and is also normal to the axis along
                        which the cameras are distributed. A resonable default
                        (possibly range-dependent) is chosen if omitted. MUST be
                        None if --scan num_far_...''')
    parser.add_argument('--z-radius',
                        type=float,
                        default=None,
                        help='''The synthetic chessboard position is sampled
                        from a uniform distribution: [-RADIUS,RADIUS]. The z
                        radius is selected here. "z" is direction normal to the
                        chessboard plane. A resonable default (possibly
                        range-dependent) is chosen if omitted. MUST be None if
                        --scan num_far_...''')
    parser.add_argument('--Ncameras',
                        default = '1',
                        type=str,
                        help='''How many cameras oberve our synthetic world. By default we just have one
                        camera. if --scan Ncameras: this is
                        NCAMERAS-MIN,NCAMERAS-MAX specifying the bounds of the
                        camera counts to evaluate. Otherwise: this is the one
                        Ncameras value to use''')
    parser.add_argument('--Nframes',
                        type=str,
                        help='''How many observed frames we have. Ignored if --scan num_far_... if
                        --scan Nframes: this is NFRAMES-MIN,NFRAMES-MAX specifying the bounds of the
                        frame counts to evaluate. Otherwise: this is the one Nframes value to use''')

    parser.add_argument('--Nframes-near',
                        type=positive_int,
                        help='''Used if --scan num_far_constant_Nframes_near. The number of "near" frames is
                        given by this argument, while we look at the effect of
                        adding more "far" frames.''')
    parser.add_argument('--Nframes-all',
                        type=positive_int,
                        help='''Used if --scan num_far_constant_Nframes_all. The number of "near"+"far"
                        frames is given by this argument, while we look at the
                        effect of replacing "near" frames with "far" frames.''')

    parser.add_argument('--Nscan-samples',
                        type=positive_int,
                        default=8,
                        help='''How many values of the parameter being scanned to evaluate. If we're scanning
                        something (--scan ... given) then by default the scan
                        evaluates 8 different values. Otherwise this is set to 1''')

    parser.add_argument('--hardcopy',
                        help=f'''Filename to plot into. If omitted, we make an interactive plot. This is
                        passed directly to gnuplotlib''')
    parser.add_argument('--terminal',
                        help=f'''gnuplot terminal to use for the plots. This is passed directly to
                        gnuplotlib. Omit this unless you know what you're doing''')
    parser.add_argument('--extratitle',
                        help=f'''Extra title string to add to a plot''')
    parser.add_argument('--title',
                        help=f'''Full title string to use in a plot. Overrides
                        the default and --extratitle''')
    parser.add_argument('--set',
                        type=str,
                        action='append',
                        help='''Extra 'set' directives to gnuplotlib. Can be given multiple times''')
    parser.add_argument('--unset',
                        type=str,
                        action='append',
                        help='''Extra 'unset' directives to gnuplotlib. Can be given multiple times''')

    parser.add_argument('model',
                        type = str,
                        help='''Baseline camera model. I use the intrinsics from this model to generate
                        synthetic data. We probably want the "true" model to not
                        be too crazy, so this should probably by a parametric
                        (not splined) model''')

    args = parser.parse_args()
    if args.title is not None and args.extratitle is not None:
        print("--title and --extratitle are mutually exclusive", file=sys.stderr)
        sys.exit(1)

    return args

args = parse_args()

if args.object_width_n  is None and \
   args.object_height_n is None:
    args.object_width_n  = "10"
    args.object_height_n = 10
elif not ( args.object_width_n  is not None and \
           args.object_height_n is not None) and \
           args.scan != 'object_width_n':
    raise Exception("Either --object-width-n or --object-height-n are given: you must pass both or neither")

# arg-parsing is done before the imports so that --help works without building
# stuff, so that I can generate the manpages and README


import numpy as np
import numpysane as nps
import gnuplotlib as gp
import copy
import os.path

# I import the LOCAL mrcal
scriptdir = os.path.dirname(os.path.realpath(__file__))
sys.path[:0] = f"{scriptdir}/../..",
import mrcal


def split_list(s, t):
    r'''Splits a comma-separated list

    None    -> None
    "A,B,C" -> (A,B,C)
    "A"     -> A
    '''
    if s is None: return None,0
    l = [t(x) for x in s.split(',')]
    if len(l) == 1: return l[0],1
    return l,len(l)

def first(s):
    r'''first in a list, or the value if a scalar'''
    if hasattr(s, '__iter__'): return s[0]
    return s

def last(s):
    r'''last in a list, or the value if a scalar'''
    if hasattr(s, '__iter__'): return s[-1]
    return s



controllable_args          = \
    dict( tilt_radius    = dict(_type = float),
          range          = dict(_type = float),
          Ncameras       = dict(_type = int),
          Nframes        = dict(_type = int),
          object_spacing = dict(_type = float),
          object_width_n = dict(_type = int) )

for a in controllable_args.keys():
    l,n = split_list(getattr(args, a), controllable_args[a]['_type'])
    controllable_args[a]['value']   = l
    controllable_args[a]['listlen'] = n

if any( controllable_args[a]['listlen'] > 2 for a in controllable_args.keys() ):
    raise Exception(f"All controllable args must have either at most 2 values. {a} has {controllable_args[a]['listlen']}")

controllable_arg_0values = [ a for a in controllable_args.keys() if controllable_args[a]['listlen'] == 0 ]
controllable_arg_2values = [ a for a in controllable_args.keys() if controllable_args[a]['listlen'] == 2 ]

if    len(controllable_arg_2values) == 0: controllable_arg_2values = ''
elif  len(controllable_arg_2values) == 1: controllable_arg_2values = controllable_arg_2values[0]
else: raise Exception(f"At most 1 controllable arg may have 2 values. Instead I saw: {controllable_arg_2values}")

if re.match("num_far_constant_Nframes_", args.scan):

    if args.x_radius is not None or \
       args.y_radius is not None or \
       args.z_radius is not None:
        raise Exception("--x-radius and --y-radius and --z-radius are exclusive with --scan num_far_...")

    # special case
    if 'Nframes' not in controllable_arg_0values:
        raise Exception(f"I'm scanning '{args.scan}', so --Nframes must not have been given")
    if 'range' != controllable_arg_2values:
        raise Exception(f"I'm scanning '{args.scan}', so --range must have 2 values")
    if args.scan == "num_far_constant_Nframes_near":
        if args.Nframes_all is not None:
            raise Exception(f"I'm scanning '{args.scan}', so --Nframes-all must not have have been given")
        if args.Nframes_near is None:
            raise Exception(f"I'm scanning '{args.scan}', so --Nframes-near must have have been given")
    else:
        if args.Nframes_near is not None:
            raise Exception(f"I'm scanning '{args.scan}', so --Nframes-near must not have have been given")
        if args.Nframes_all is None:
            raise Exception(f"I'm scanning '{args.scan}', so --Nframes-all must have have been given")

else:
    if args.scan != controllable_arg_2values:
        # This covers the scanning-nothing-no2value-anything case
        raise Exception(f"I'm scanning '{args.scan}', the arg given 2 values is '{controllable_arg_2values}'. They must match")
    if len(controllable_arg_0values):
        raise Exception(f"I'm scanning '{args.scan}', so all controllable args should have some value. Missing: '{controllable_arg_0values}")
    if args.scan == '':
        args.Nscan_samples = 1





Nuncertainty_at_range_samples = 80
uncertainty_at_range_sampled_min = args.uncertainty_at_range_sampled_min
if uncertainty_at_range_sampled_min is None:
    uncertainty_at_range_sampled_min = first(controllable_args['range']['value'])/10.
uncertainty_at_range_sampled_max = args.uncertainty_at_range_sampled_max
if uncertainty_at_range_sampled_max is None:
    uncertainty_at_range_sampled_max = last(controllable_args['range']['value']) *10.

uncertainty_at_range_samples = \
    np.logspace( np.log10(uncertainty_at_range_sampled_min),
                 np.log10(uncertainty_at_range_sampled_max),
                 Nuncertainty_at_range_samples)

# I want the RNG to be deterministic
np.random.seed(0)

model_intrinsics = mrcal.cameramodel(args.model)

calobject_warp_true_ref = np.array(args.calobject_warp)


def solve(Ncameras,
          Nframes_near, Nframes_far,
          object_spacing,
          models_true,

          # q.shape            = (Nframes, Ncameras, object_height, object_width, 2)
          # Rt_ref_board.shape = (Nframes, 4,3)

          q_true_near, Rt_ref_board_true_near,
          q_true_far,  Rt_ref_board_true_far,
          fixed_frames = args.fixed_frames):

    q_true_near            = q_true_near            [:Nframes_near]
    Rt_ref_board_true_near = Rt_ref_board_true_near[:Nframes_near]

    if q_true_far is not None:
        q_true_far             = q_true_far             [:Nframes_far ]
        Rt_ref_board_true_far  = Rt_ref_board_true_far [:Nframes_far ]
    else:
        q_true_far             = np.zeros( (0,) + q_true_near.shape[1:],
                                            dtype = q_true_near.dtype)
        Rt_ref_board_true_far  = np.zeros( (0,) + Rt_ref_board_true_near.shape[1:],
                                            dtype = Rt_ref_board_true_near.dtype)

    calobject_warp_true = calobject_warp_true_ref.copy()

    Nframes_all = Nframes_near + Nframes_far

    Rt_ref_board_true = nps.glue( Rt_ref_board_true_near,
                                  Rt_ref_board_true_far,
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

    # If not fixed_frames: we use camera0 as the reference cordinate system, and
    # we allow the chessboard poses to move around. Else: the reference
    # coordinate system is arbitrary, but all cameras are allowed to move
    # around. The chessboards poses are fixed
    if not fixed_frames:
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

    # If not fixed_frames: we use camera0 as the reference cordinate system, and
    # we allow the chessboard poses to move around. Else: the reference
    # coordinate system is arbitrary, but all cameras are allowed to move
    # around. The chessboards poses are fixed
    if fixed_frames:
        extrinsics = nps.cat( *[m.rt_cam_ref() for m in models_true] )
    else:
        extrinsics = nps.cat( *[m.rt_cam_ref() for m in models_true[1:]] )
    if len(extrinsics) == 0: extrinsics = None

    if nps.norm2(models_true[0].rt_cam_ref()) > 1e-6:
        raise Exception("models_true[0] must sit at the origin")
    imagersizes = nps.cat( *[m.imagersize() for m in models_true] )


    optimization_inputs = \
        dict( # intrinsics filled in later
              rt_cam_ref                                = extrinsics,
              rt_ref_frame                              = mrcal.rt_from_Rt(Rt_ref_board_true),
              points                                    = None,
              observations_board                        = observations,
              indices_frame_camintrinsics_camextrinsics = indices_frame_camintrinsics_camextrinsics,
              observations_point                        = None,
              indices_point_camintrinsics_camextrinsics = None,
              # lensmodel filled in later
              calobject_warp                            = copy.deepcopy(calobject_warp_true),
              imagersizes                               = imagersizes,
              calibration_object_spacing                = object_spacing,
              verbose                                   = False,
              # do_optimize_extrinsics filled in later
              # do_optimize_intrinsics_core filled in later
              do_optimize_frames                        = False,
              do_optimize_intrinsics_distortions        = True,
              do_optimize_calobject_warp                = False, # turn this on, and reoptimize later, if needed
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
        optimization_inputs['do_optimize_extrinsics'] = False

        # I pre-optimize the core, and then lock it down
        optimization_inputs['lensmodel']                   = 'LENSMODEL_STEREOGRAPHIC'
        optimization_inputs['intrinsics']                  = intrinsics[:,:4].copy()
        optimization_inputs['do_optimize_intrinsics_core'] = True

        stats = mrcal.optimize(**optimization_inputs)
        print(f"## optimized. rms = {stats['rms_reproj_error__pixels']}", file=sys.stderr)

        # core is good. Lock that down, and get an estimate for the control
        # points
        optimization_inputs['do_optimize_intrinsics_core'] = False
        optimization_inputs['lensmodel']                   = lensmodel
        optimization_inputs['intrinsics']                  = nps.glue(optimization_inputs['intrinsics'],
                                                                      np.zeros((Ncameras,Nintrinsics-4),),axis=-1)
        stats = mrcal.optimize(**optimization_inputs)
        print(f"## optimized. rms = {stats['rms_reproj_error__pixels']}", file=sys.stderr)

        # Ready for a final reoptimization with the geometry
        optimization_inputs['do_optimize_extrinsics']      = True
        if not fixed_frames:
            optimization_inputs['do_optimize_frames'] = True

    else:
        optimization_inputs['lensmodel']                   = lensmodel
        if not mrcal.lensmodel_metadata_and_config(lensmodel)['has_core'] or \
           not mrcal.lensmodel_metadata_and_config(model_intrinsics.intrinsics()[0])['has_core']:
            raise Exception("I'm assuming all the models here have a core. It's just lazy coding. If you see this, feel free to fix.")

        if lensmodel == model_intrinsics.intrinsics()[0]:
            # Same model. Grab the intrinsics. They're 99% right
            optimization_inputs['intrinsics']              = intrinsics.copy()
        else:
            # Different model. Grab the intrinsics core, optimize the rest
            optimization_inputs['intrinsics']              = nps.glue(intrinsics[:,:4],
                                                                      np.zeros((Ncameras,Nintrinsics-4),),axis=-1)
        optimization_inputs['do_optimize_intrinsics_core'] = True
        optimization_inputs['do_optimize_extrinsics']      = True
        if not fixed_frames:
            optimization_inputs['do_optimize_frames'] = True

    stats = mrcal.optimize(**optimization_inputs)
    print(f"## optimized. rms = {stats['rms_reproj_error__pixels']}", file=sys.stderr)

    if not args.skip_calobject_warp_solve:
        optimization_inputs['do_optimize_calobject_warp'] = True
        stats = mrcal.optimize(**optimization_inputs)
        print(f"## optimized. rms = {stats['rms_reproj_error__pixels']}", file=sys.stderr)

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
                            object_width_n, object_height_n, object_spacing,
                            uncertainty_at_range_samples,
                            Ncameras,
                            Nframes_near_samples, Nframes_far_samples):

    # I want the RNG to be deterministic
    np.random.seed(0)

    uncertainties = np.zeros((len(Nframes_far_samples),
                              len(uncertainty_at_range_samples)),
                             dtype=float)


    radius_cameras = (args.camera_spacing * (Ncameras-1)) / 2.

    x_radius = args.x_radius if args.x_radius is not None else range_near*2. + radius_cameras
    y_radius = args.y_radius if args.y_radius is not None else range_near*2.
    z_radius = args.z_radius if args.z_radius is not None else range_near/10.

    # shapes (Nframes, Ncameras, Nh, Nw, 2),
    #        (Nframes, 4,3)
    q_true_near, Rt_ref_board_true_near = \
        mrcal.synthesize_board_observations(models_true,
                                            object_width_n                  = object_width_n,
                                            object_height_n                 = object_height_n,
                                            object_spacing                  = object_spacing,
                                            calobject_warp                  = calobject_warp_true_ref,
                                            rt_ref_boardcenter              = np.array((0.,  0., 0., radius_cameras, 0,  range_near,)),
                                            rt_ref_boardcenter__noiseradius = \
                                              np.array((np.pi/180. * tilt_radius,
                                                        np.pi/180. * tilt_radius,
                                                        np.pi/180. * args.roll_radius,
                                                        x_radius, y_radius, z_radius)),
                                            Nframes                         = np.max(Nframes_near_samples),
                                            which                           = args.which)
    if range_far is not None:
        q_true_far, Rt_ref_board_true_far  = \
            mrcal.synthesize_board_observations(models_true,
                                                object_width_n                  = object_width_n,
                                                object_height_n                 = object_height_n,
                                                object_spacing                  = object_spacing,
                                                calobject_warp                  = calobject_warp_true_ref,
                                                rt_ref_boardcenter              = np.array((0.,  0., 0., radius_cameras, 0,  range_far,)),
                                                rt_ref_boardcenter__noiseradius = \
                                                  np.array((np.pi/180. * tilt_radius,
                                                            np.pi/180. * tilt_radius,
                                                            np.pi/180. * args.roll_radius,
                                                            range_far*2. + radius_cameras,
                                                            range_far*2.,
                                                            range_far/10.)),
                                                Nframes                         = np.max(Nframes_far_samples),
                                                which                           = args.which)
    else:
        q_true_far            = None
        Rt_ref_board_true_far = None

    for i_Nframes_far in range(len(Nframes_far_samples)):

        Nframes_far  = Nframes_far_samples [i_Nframes_far]
        Nframes_near = Nframes_near_samples[i_Nframes_far]

        optimization_inputs = solve(Ncameras,
                                    Nframes_near, Nframes_far,
                                    object_spacing,
                                    models_true,
                                    q_true_near, Rt_ref_board_true_near,
                                    q_true_far,  Rt_ref_board_true_far)

        models_out = \
            [ mrcal.cameramodel( optimization_inputs = optimization_inputs,
                                 icam_intrinsics     = icam ) \
              for icam in range(Ncameras) ]

        model = models_out[args.icam_uncertainty]

        if args.show_geometry_first_solve:
            mrcal.show_geometry(models_out,
                                show_calobjects = True,
                                wait = True)
            sys.exit()
        if args.show_uncertainty_first_solve:
            mrcal.show_projection_uncertainty(model,
                                              method      = args.method,
                                              observations= True,
                                              wait        = True)
            sys.exit()
        if args.write_models_first_solve:
            for i in range(len(models_out)):
                f = f"/tmp/camera{i}.cameramodel"
                if os.path.exists(f):
                    input(f"File {f} already exists, and I want to overwrite it. Press enter to overwrite. Ctrl-c to exit")
                models_out[i].write(f)
                print(f"Wrote {f}")
            sys.exit()

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
                                         method = args.method,
                                         what   = 'worstdirection-stdev')

    return uncertainties



output_table_legend = 'range_uncertainty_sample Nframes_near Nframes_far Ncameras range_near range_far tilt_radius object_width_n object_spacing uncertainty'
output_table_fmt    = '%f %d %d %d %f %f %f %d %f %f'
output_table_icol__range_uncertainty_sample = 0
output_table_icol__Nframes_near             = 1
output_table_icol__Nframes_far              = 2
output_table_icol__Ncameras                 = 3
output_table_icol__range_near               = 4
output_table_icol__range_far                = 5
output_table_icol__tilt_radius              = 6
output_table_icol__object_width_n           = 7
output_table_icol__object_spacing           = 8
output_table_icol__uncertainty              = 9
output_table_Ncols                          = 10

output_table = np.zeros( (args.Nscan_samples, Nuncertainty_at_range_samples, output_table_Ncols), dtype=float)

output_table[:,:, output_table_icol__range_uncertainty_sample] += uncertainty_at_range_samples

if re.match("num_far_constant_Nframes_", args.scan):

    Nfar_samples = args.Nscan_samples
    if   args.scan == "num_far_constant_Nframes_near":
        Nframes_far_samples = np.linspace(0,
                                          args.Nframes_near//4,
                                          Nfar_samples, dtype=int)
        Nframes_near_samples = Nframes_far_samples*0 + args.Nframes_near

    else:
        Nframes_far_samples  = np.linspace(0,
                                           args.Nframes_all,
                                           Nfar_samples, dtype=int)
        Nframes_near_samples = args.Nframes_all - Nframes_far_samples

    models_true = \
        [ mrcal.cameramodel(intrinsics = model_intrinsics.intrinsics(),
                            imagersize = model_intrinsics.imagersize(),
                            rt_ref_cam = np.array((0,0,0,
                                                   i*args.camera_spacing,
                                                   0,0), dtype=float) ) \
          for i in range(controllable_args['Ncameras']['value']) ]

    # shape (args.Nscan_samples, Nuncertainty_at_range_samples)
    output_table[:,:, output_table_icol__uncertainty] = \
        eval_one_rangenear_tilt(models_true,
                                *controllable_args['range']['value'],
                                controllable_args['tilt_radius']['value'],
                                controllable_args['object_width_n']['value'],
                                args.object_height_n,
                                controllable_args['object_spacing']['value'],
                                uncertainty_at_range_samples,
                                controllable_args['Ncameras']['value'],
                                Nframes_near_samples, Nframes_far_samples)

    output_table[:,:, output_table_icol__Nframes_near]    += nps.transpose(Nframes_near_samples)
    output_table[:,:, output_table_icol__Nframes_far]     += nps.transpose(Nframes_far_samples)
    output_table[:,:, output_table_icol__Ncameras]        = controllable_args['Ncameras']['value']
    output_table[:,:, output_table_icol__range_near]      = controllable_args['range']['value'][0]
    output_table[:,:, output_table_icol__range_far]       = controllable_args['range']['value'][1]
    output_table[:,:, output_table_icol__tilt_radius ]    = controllable_args['tilt_radius']['value']
    output_table[:,:, output_table_icol__object_width_n ] = controllable_args['object_width_n']['value']
    output_table[:,:, output_table_icol__object_spacing ] = controllable_args['object_spacing']['value']

    samples = Nframes_far_samples

elif args.scan == "range":
    Nframes_near_samples = np.array( (controllable_args['Nframes']['value'],), dtype=int)
    Nframes_far_samples  = np.array( (0,),            dtype=int)

    models_true = \
        [ mrcal.cameramodel(intrinsics = model_intrinsics.intrinsics(),
                            imagersize = model_intrinsics.imagersize(),
                            rt_ref_cam = np.array((0,0,0,
                                                   i*args.camera_spacing,
                                                   0,0), dtype=float) ) \
          for i in range(controllable_args['Ncameras']['value']) ]

    Nrange_samples = args.Nscan_samples
    range_samples = np.linspace(*controllable_args['range']['value'],
                                Nrange_samples, dtype=float)
    for i_range in range(Nrange_samples):
        output_table[i_range,:, output_table_icol__uncertainty] = \
            eval_one_rangenear_tilt(models_true,
                                    range_samples[i_range], None,
                                    controllable_args['tilt_radius']['value'],
                                    controllable_args['object_width_n']['value'],
                                    args.object_height_n,
                                    controllable_args['object_spacing']['value'],
                                    uncertainty_at_range_samples,
                                    controllable_args['Ncameras']['value'],
                                    Nframes_near_samples, Nframes_far_samples)[0]

    output_table[:,:, output_table_icol__Nframes_near] = controllable_args['Nframes']['value']
    output_table[:,:, output_table_icol__Nframes_far]  = 0
    output_table[:,:, output_table_icol__Ncameras]     = controllable_args['Ncameras']['value']
    output_table[:,:, output_table_icol__range_near]  += nps.transpose(range_samples)
    output_table[:,:, output_table_icol__range_far]    = -1
    output_table[:,:, output_table_icol__tilt_radius ] = controllable_args['tilt_radius']['value']
    output_table[:,:, output_table_icol__object_width_n ] = controllable_args['object_width_n']['value']
    output_table[:,:, output_table_icol__object_spacing ] = controllable_args['object_spacing']['value']

    samples = range_samples

elif args.scan == "tilt_radius":
    Nframes_near_samples = np.array( (controllable_args['Nframes']['value'],), dtype=int)
    Nframes_far_samples  = np.array( (0,),            dtype=int)

    models_true = \
        [ mrcal.cameramodel(intrinsics = model_intrinsics.intrinsics(),
                            imagersize = model_intrinsics.imagersize(),
                            rt_ref_cam = np.array((0,0,0,
                                                   i*args.camera_spacing,
                                                   0,0), dtype=float) ) \
          for i in range(controllable_args['Ncameras']['value']) ]

    Ntilt_rad_samples = args.Nscan_samples
    tilt_rad_samples = np.linspace(*controllable_args['tilt_radius']['value'],
                                   Ntilt_rad_samples, dtype=float)
    for i_tilt in range(Ntilt_rad_samples):
        output_table[i_tilt,:, output_table_icol__uncertainty] = \
            eval_one_rangenear_tilt(models_true,
                                    controllable_args['range']['value'], None,
                                    tilt_rad_samples[i_tilt],
                                    controllable_args['object_width_n']['value'],
                                    args.object_height_n,
                                    controllable_args['object_spacing']['value'],
                                    uncertainty_at_range_samples,
                                    controllable_args['Ncameras']['value'],
                                    Nframes_near_samples, Nframes_far_samples)[0]

    output_table[:,:, output_table_icol__Nframes_near] = controllable_args['Nframes']['value']
    output_table[:,:, output_table_icol__Nframes_far]  = 0
    output_table[:,:, output_table_icol__Ncameras]     = controllable_args['Ncameras']['value']
    output_table[:,:, output_table_icol__range_near]   = controllable_args['range']['value']
    output_table[:,:, output_table_icol__range_far]    = -1
    output_table[:,:, output_table_icol__tilt_radius] += nps.transpose(tilt_rad_samples)
    output_table[:,:, output_table_icol__object_width_n ] = controllable_args['object_width_n']['value']
    output_table[:,:, output_table_icol__object_spacing ] = controllable_args['object_spacing']['value']

    samples = tilt_rad_samples

elif args.scan == "Ncameras":

    Nframes_near_samples = np.array( (controllable_args['Nframes']['value'],), dtype=int)
    Nframes_far_samples  = np.array( (0,),            dtype=int)

    N_Ncameras_samples = args.Nscan_samples
    Ncameras_samples = np.linspace(*controllable_args['Ncameras']['value'],
                                   N_Ncameras_samples, dtype=int)
    for i_Ncameras in range(N_Ncameras_samples):

        Ncameras = Ncameras_samples[i_Ncameras]
        models_true = \
            [ mrcal.cameramodel(intrinsics = model_intrinsics.intrinsics(),
                                imagersize = model_intrinsics.imagersize(),
                                rt_ref_cam = np.array((0,0,0,
                                                       i*args.camera_spacing,
                                                       0,0), dtype=float) ) \
              for i in range(Ncameras) ]

        output_table[i_Ncameras,:, output_table_icol__uncertainty] = \
            eval_one_rangenear_tilt(models_true,
                                    controllable_args['range']['value'], None,
                                    controllable_args['tilt_radius']['value'],
                                    controllable_args['object_width_n']['value'],
                                    args.object_height_n,
                                    controllable_args['object_spacing']['value'],
                                    uncertainty_at_range_samples,
                                    Ncameras,
                                    Nframes_near_samples, Nframes_far_samples)[0]

    output_table[:,:, output_table_icol__Nframes_near] = controllable_args['Nframes']['value']
    output_table[:,:, output_table_icol__Nframes_far]  = 0
    output_table[:,:, output_table_icol__Ncameras]    += nps.transpose(Ncameras_samples)
    output_table[:,:, output_table_icol__range_near]   = controllable_args['range']['value']
    output_table[:,:, output_table_icol__range_far]    = -1
    output_table[:,:, output_table_icol__tilt_radius]  = controllable_args['tilt_radius']['value']
    output_table[:,:, output_table_icol__object_width_n ] = controllable_args['object_width_n']['value']
    output_table[:,:, output_table_icol__object_spacing ] = controllable_args['object_spacing']['value']

    samples = Ncameras_samples

elif args.scan == "Nframes":

    Nframes_far_samples  = np.array( (0,),            dtype=int)

    models_true = \
        [ mrcal.cameramodel(intrinsics = model_intrinsics.intrinsics(),
                            imagersize = model_intrinsics.imagersize(),
                            rt_ref_cam = np.array((0,0,0,
                                                   i*args.camera_spacing,
                                                   0,0), dtype=float) ) \
          for i in range(controllable_args['Ncameras']['value']) ]

    N_Nframes_samples = args.Nscan_samples
    Nframes_samples = np.linspace(*controllable_args['Nframes']['value'],
                                  N_Nframes_samples, dtype=int)
    for i_Nframes in range(N_Nframes_samples):

        Nframes = Nframes_samples[i_Nframes]
        Nframes_near_samples = np.array( (Nframes,), dtype=int)

        output_table[i_Nframes,:, output_table_icol__uncertainty] = \
            eval_one_rangenear_tilt(models_true,
                                    controllable_args['range']['value'], None,
                                    controllable_args['tilt_radius']['value'],
                                    controllable_args['object_width_n']['value'],
                                    args.object_height_n,
                                    controllable_args['object_spacing']['value'],
                                    uncertainty_at_range_samples,
                                    controllable_args['Ncameras']['value'],
                                    Nframes_near_samples, Nframes_far_samples)[0]

    output_table[:,:, output_table_icol__Nframes_near]+= nps.transpose(Nframes_samples)
    output_table[:,:, output_table_icol__Nframes_far]  = 0
    output_table[:,:, output_table_icol__Ncameras]     = controllable_args['Ncameras']['value']
    output_table[:,:, output_table_icol__range_near]   = controllable_args['range']['value']
    output_table[:,:, output_table_icol__range_far]    = -1
    output_table[:,:, output_table_icol__tilt_radius]  = controllable_args['tilt_radius']['value']
    output_table[:,:, output_table_icol__object_width_n ] = controllable_args['object_width_n']['value']
    output_table[:,:, output_table_icol__object_spacing ] = controllable_args['object_spacing']['value']

    samples = Nframes_samples

elif args.scan == "object_width_n":

    Nframes_near_samples = np.array( (controllable_args['Nframes']['value'],), dtype=int)
    Nframes_far_samples  = np.array( (0,),            dtype=int)

    models_true = \
        [ mrcal.cameramodel(intrinsics = model_intrinsics.intrinsics(),
                            imagersize = model_intrinsics.imagersize(),
                            rt_ref_cam = np.array((0,0,0,
                                                   i*args.camera_spacing,
                                                   0,0), dtype=float) ) \
          for i in range(controllable_args['Ncameras']['value']) ]

    Nsamples = args.Nscan_samples
    samples  = np.linspace(*controllable_args['object_width_n']['value'],
                           Nsamples, dtype=int)

    # As I move the width, I adjust the spacing to keep the total board size
    # constant. The object spacing in the argument applies to the MIN value of
    # the object_width_n.
    W = (controllable_args['object_width_n']['value'][0]-1) * controllable_args['object_spacing']['value']
    object_spacing = W / samples



    for i_sample in range(Nsamples):

        output_table[i_sample,:, output_table_icol__uncertainty] = \
            eval_one_rangenear_tilt(models_true,
                                    controllable_args['range']['value'], None,
                                    controllable_args['tilt_radius']['value'],
                                    samples[i_sample],
                                    samples[i_sample],
                                    object_spacing[i_sample],
                                    uncertainty_at_range_samples,
                                    controllable_args['Ncameras']['value'],
                                    Nframes_near_samples, Nframes_far_samples)[0]

    output_table[:,:, output_table_icol__Nframes_near] = controllable_args['Nframes']['value']
    output_table[:,:, output_table_icol__Nframes_far]  = 0
    output_table[:,:, output_table_icol__Ncameras]     = controllable_args['Ncameras']['value']
    output_table[:,:, output_table_icol__range_near]   = controllable_args['range']['value']
    output_table[:,:, output_table_icol__range_far]    = -1
    output_table[:,:, output_table_icol__tilt_radius]  = controllable_args['tilt_radius']['value']
    output_table[:,:, output_table_icol__object_width_n ]+= nps.transpose(samples)
    output_table[:,:, output_table_icol__object_spacing ]+= nps.transpose(object_spacing)

elif args.scan == "object_spacing":

    Nframes_near_samples = np.array( (controllable_args['Nframes']['value'],), dtype=int)
    Nframes_far_samples  = np.array( (0,),            dtype=int)

    models_true = \
        [ mrcal.cameramodel(intrinsics = model_intrinsics.intrinsics(),
                            imagersize = model_intrinsics.imagersize(),
                            rt_ref_cam = np.array((0,0,0,
                                                   i*args.camera_spacing,
                                                   0,0), dtype=float) ) \
          for i in range(controllable_args['Ncameras']['value']) ]

    Nsamples = args.Nscan_samples
    samples  = np.linspace(*controllable_args['object_spacing']['value'],
                           Nsamples, dtype=float)

    # As I move the spacing, I leave object_width_n, letting the total board
    # size change. The object spacing in the argument applies to the MIN value
    # of the object_width_n.
    for i_sample in range(Nsamples):

        r = controllable_args['range']['value']
        if args.scan_object_spacing_compensate_range_from:
            r *= samples[i_sample]/args.scan_object_spacing_compensate_range_from

        output_table[i_sample,:, output_table_icol__uncertainty] = \
            eval_one_rangenear_tilt(models_true,
                                    r, None,
                                    controllable_args['tilt_radius']['value'],
                                    controllable_args['object_width_n']['value'],
                                    args.object_height_n,
                                    samples[i_sample],
                                    uncertainty_at_range_samples,
                                    controllable_args['Ncameras']['value'],
                                    Nframes_near_samples, Nframes_far_samples)[0]

    output_table[:,:, output_table_icol__Nframes_near] = controllable_args['Nframes']['value']
    output_table[:,:, output_table_icol__Nframes_far]  = 0
    output_table[:,:, output_table_icol__Ncameras]     = controllable_args['Ncameras']['value']
    output_table[:,:, output_table_icol__range_far]    = -1
    output_table[:,:, output_table_icol__tilt_radius]  = controllable_args['tilt_radius']['value']
    output_table[:,:, output_table_icol__object_width_n ] = controllable_args['object_width_n']['value']
    output_table[:,:, output_table_icol__object_spacing ]+= nps.transpose(samples)

    if args.scan_object_spacing_compensate_range_from:
        output_table[:,:, output_table_icol__range_near] += controllable_args['range']['value'] * nps.transpose(samples/samples[0])
    else:
        output_table[:,:, output_table_icol__range_near] = controllable_args['range']['value']

else:
    # no --scan. We just want one sample

    Nframes_near_samples = np.array( (controllable_args['Nframes']['value'],), dtype=int)
    Nframes_far_samples  = np.array( (0,),            dtype=int)

    models_true = \
        [ mrcal.cameramodel(intrinsics = model_intrinsics.intrinsics(),
                            imagersize = model_intrinsics.imagersize(),
                            rt_ref_cam = np.array((0,0,0,
                                                   i*args.camera_spacing,
                                                   0,0), dtype=float) ) \
          for i in range(controllable_args['Ncameras']['value']) ]

    output_table[0,:, output_table_icol__uncertainty] = \
        eval_one_rangenear_tilt(models_true,
                                controllable_args['range']['value'], None,
                                controllable_args['tilt_radius']['value'],
                                controllable_args['object_width_n']['value'],
                                args.object_height_n,
                                controllable_args['object_spacing']['value'],
                                uncertainty_at_range_samples,
                                controllable_args['Ncameras']['value'],
                                Nframes_near_samples, Nframes_far_samples)[0]

    output_table[:,:, output_table_icol__Nframes_near] = controllable_args['Nframes']['value']
    output_table[:,:, output_table_icol__Nframes_far]  = 0
    output_table[:,:, output_table_icol__Ncameras]     = controllable_args['Ncameras']['value']
    output_table[:,:, output_table_icol__range_near]   = controllable_args['range']['value']
    output_table[:,:, output_table_icol__range_far]    = -1
    output_table[:,:, output_table_icol__tilt_radius]  = controllable_args['tilt_radius']['value']
    output_table[:,:, output_table_icol__object_width_n ] = controllable_args['object_width_n']['value']
    output_table[:,:, output_table_icol__object_spacing ] = controllable_args['object_spacing']['value']

    samples = None



if isinstance(controllable_args['range']['value'], float):
    guides = [ f"arrow nohead dashtype 3 from {controllable_args['range']['value']},graph 0 to {controllable_args['range']['value']},graph 1" ]
else:
    guides = [ f"arrow nohead dashtype 3 from {r},graph 0 to {r},graph 1" for r in controllable_args['range']['value'] ]
guides.append(f"arrow nohead dashtype 3 from graph 0,first {args.observed_pixel_uncertainty} to graph 1,first {args.observed_pixel_uncertainty}")

title = args.title

if   args.scan == "num_far_constant_Nframes_near":
    if title is None:
        title = f"Scanning 'far' observations added to a set of 'near' observations. Have {controllable_args['Ncameras']['value']} cameras, {args.Nframes_near} 'near' observations, at ranges {controllable_args['range']['value']}."
    legend_what = 'Nframes_far'
elif args.scan == "num_far_constant_Nframes_all":
    if title is None:
        title = f"Scanning 'far' observations replacing 'near' observations. Have {controllable_args['Ncameras']['value']} cameras, {args.Nframes_all} total observations, at ranges {controllable_args['range']['value']}."
    legend_what = 'Nframes_far'
elif args.scan == "Nframes":
    if title is None:
        title = f"Scanning Nframes. Have {controllable_args['Ncameras']['value']} cameras looking out at {controllable_args['range']['value']:.2f}m."
    legend_what = 'Nframes'
elif args.scan == "Ncameras":
    if title is None:
        title = f"Scanning Ncameras. Observing {controllable_args['Nframes']['value']} boards at {controllable_args['range']['value']:.2f}m."
    legend_what = 'Ncameras'
elif args.scan == "range":
    if title is None:
        title = f"Scanning the distance to observations. Have {controllable_args['Ncameras']['value']} cameras looking at {controllable_args['Nframes']['value']} boards."
    legend_what = 'Range-to-chessboards'
elif args.scan == "tilt_radius":
    if title is None:
        title = f"Scanning the board tilt. Have {controllable_args['Ncameras']['value']} cameras looking at {controllable_args['Nframes']['value']} boards at {controllable_args['range']['value']:.2f}m"
    legend_what = 'Random chessboard tilt radius'
elif args.scan == "object_width_n":
    if title is None:
        title = f"Scanning the calibration object density, keeping the board size constant. Have {controllable_args['Ncameras']['value']} cameras looking at {controllable_args['Nframes']['value']} boards at {controllable_args['range']['value']:.2f}m"
    legend_what = 'Number of chessboard points per side'
elif args.scan == "object_spacing":
    if title is None:
        if args.scan_object_spacing_compensate_range_from:
            title = f"Scanning the calibration object spacing, keeping the point count constant, and letting the board grow. Range grows with spacing. Have {controllable_args['Ncameras']['value']} cameras looking at {controllable_args['Nframes']['value']} boards at {controllable_args['range']['value']:.2f}m"
        else:
            title = f"Scanning the calibration object spacing, keeping the point count constant, and letting the board grow. Range is constant. Have {controllable_args['Ncameras']['value']} cameras looking at {controllable_args['Nframes']['value']} boards at {controllable_args['range']['value']:.2f}m"
    legend_what = 'Distance between adjacent chessboard corners'
else:
    # no --scan. We just want one sample
    if title is None:
        title = f"Have {controllable_args['Ncameras']['value']} cameras looking at {controllable_args['Nframes']['value']} boards at {controllable_args['range']['value']:.2f}m with tilt radius {controllable_args['tilt_radius']['value']}"

if args.extratitle is not None:
    title = f"{title}: {args.extratitle}"

if samples is None:
    legend = None
elif samples.dtype.kind == 'i':
    legend = np.array([ f"{legend_what} = {x}" for x in samples])
else:
    legend = np.array([ f"{legend_what} = {x:.2f}" for x in samples])

np.savetxt(sys.stdout,
           nps.clump(output_table, n=2),
           fmt   = output_table_fmt,
           header= output_table_legend)

plotoptions = \
    dict( yrange   = (0, args.ymax),
          _with    = 'lines',
          _set     = guides,
          unset    = 'grid',
          title    = title,
          xlabel   = 'Range (m)',
          ylabel   = 'Expected worst-direction uncertainty (pixels)',
          hardcopy = args.hardcopy,
          terminal = args.terminal,
          wait     = not args.explore and args.hardcopy is None)
if legend is not None: plotoptions['legend'] = legend

if args.set:
    gp.add_plot_option(plotoptions,
                       _set = args.set)
if args.unset:
    gp.add_plot_option(plotoptions,
                       _unset = args.unset)

gp.plot(uncertainty_at_range_samples,
        output_table[:,:, output_table_icol__uncertainty],
        **plotoptions)

if args.explore:
    import IPython
    IPython.embed()
sys.exit()
