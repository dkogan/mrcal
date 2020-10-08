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
clusters are given by --range-near and --range-far. This tool solves the
calibration problem, and generates uncertainty-vs-range curves. Each run of this
tool generates a family of this curve, for different values of Nfar, the numbers
of chessboard observations in the "far" cluster. If the user specifies --Nnear,
then the "far" observations are added to the set of "near" observations. If the
user instead specifies --Nall, the total number of observations is kept
constant, and we remove a "near" observation for each "far" observation we add.

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

    parser.add_argument('--range-near',
                        default = 0.5,
                        type=float,
                        help='''The "near" range to the calibration object to use in the synthetic data, in
                        meters''')
    parser.add_argument('--range-far',
                        default = 4.0,
                        type=float,
                        help='''The "far" range to the calibration object to use in the synthetic data, in
                        meters''')
    parser.add_argument('--Ncameras',
                        default = 1,
                        type=positive_int,
                        help='How many cameras in our synthetic world')
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
    parser.add_argument('--Nall',
                        type=positive_int,
                        help='''If given, the total number of chessboard observations is kept constant, at
                        this value. Exclusive with --Nnear. Either --Nall or
                        --Nnear are required''')
    parser.add_argument('--Nnear',
                        type=positive_int,
                        help='''If given, the number of "near" chessboard observations is kept constant, at
                        this value. Exclusive with --Nall. Either --Nall or
                        --Nnear are required''')

    parser.add_argument('--explore',
                        action='store_true',
                        help='''Drop into a REPL at the end''')

    parser.add_argument('model',
                        type = str,
                        help='''Baseline camera model. I use the intrinsics from this model to generate
                        synthetic data. We probably want the "true" model to not
                        be too crazy, so this should probably by a parametric
                        (not splined) model''')

    return parser.parse_args()

args = parse_args()

if args.Nall is None and args.Nnear is None:
    print("Exactly one of --Nall and --Nnear must be given", file=sys.stderr)
    sys.exit(1)
if args.Nall is not None and args.Nnear is not None:
    print("Exactly one of --Nall and --Nnear must be given", file=sys.stderr)
    sys.exit(1)
if args.range_near > args.range_far:
    print("--range-near must be < --range-far", file=sys.stderr)
    sys.exit(1)

# arg-parsing is done before the imports so that --help works without building
# stuff, so that I can generate the manpages and README




import numpy as np
import numpysane as nps
import gnuplotlib as gp
import copy

sys.path[:0] = '../..',
import mrcal



model_intrinsics = mrcal.cameramodel(args.model)


def solve(Nframes_near, Nframes_far):

    calobject_warp_true = np.array((0.002, -0.005))

    Nframes_all = Nframes_near + Nframes_far

    rad = (args.camera_spacing * (args.Ncameras-1)) / 2.
    models_true = \
        [ mrcal.cameramodel(intrinsics          = model_intrinsics.intrinsics(),
                            imagersize          = model_intrinsics.imagersize(),
                            extrinsics_rt_toref = np.array((0,0,0,
                                                            i*args.camera_spacing,
                                                            0,0), dtype=float) ) \
          for i in range(args.Ncameras) ]


    # shapes (Nframes_all, Ncameras, Nh, Nw, 2),
    #        (Nframes_all, 4,3)
    q_true_near,Rt_cam0_board_true_near = \
        mrcal.make_synthetic_board_observations(models_true,
                                                args.object_width_n,
                                                args.object_height_n,
                                                args.object_spacing,
                                                calobject_warp_true,
                                                np.array((rad, 0,  args.range_near,
                                                          0.,  0., 0.)),
                                                np.array((args.range_near/3.*2.,
                                                          args.range_near/3.*2.,
                                                          args.range_near/10.,
                                                          40., 30., 30.)),
                                                Nframes_near)
    q_true_far,Rt_cam0_board_true_far = \
        mrcal.make_synthetic_board_observations(models_true,
                                                args.object_width_n,
                                                args.object_height_n,
                                                args.object_spacing,
                                                calobject_warp_true,
                                                np.array((rad, 0,  args.range_far,
                                                          0.,  0., 0.)),
                                                np.array((args.range_far/3.*2.,
                                                          args.range_far/3.*2.,
                                                          args.range_far/10.,
                                                          40., 30., 30.)),
                                                Nframes_far)

    Rt_cam0_board_true = nps.glue( Rt_cam0_board_true_near,
                                   Rt_cam0_board_true_far,
                                   axis = -3 )

    q = nps.clump( nps.glue( q_true_near,
                             q_true_far,
                             axis = -5 ),
                   n = 2 )

    observations = nps.glue(q, q[...,(0,)]*0+1,
                            axis = -1)

    q_noise = np.random.randn(*observations.shape[:-1], 2) * args.observed_pixel_uncertainty
    observations[...,:2] += q_noise

    # Dense observations. All the cameras see all the boards
    indices_frame_camera = np.zeros( (Nframes_all*args.Ncameras, 2), dtype=np.int32)
    indices_frame = indices_frame_camera[:,0].reshape(Nframes_all,args.Ncameras)
    indices_frame.setfield(nps.outer(np.arange(Nframes_all, dtype=np.int32),
                                     np.ones((args.Ncameras,), dtype=np.int32)),
                           dtype = np.int32)
    indices_camera = indices_frame_camera[:,1].reshape(Nframes_all,args.Ncameras)
    indices_camera.setfield(nps.outer(np.ones((Nframes_all,), dtype=np.int32),
                                     np.arange(args.Ncameras, dtype=np.int32)),
                           dtype = np.int32)
    indices_frame_camintrinsics_camextrinsics = \
        nps.glue(indices_frame_camera,
                 indices_frame_camera[:,(1,)],
                 axis=-1)
    indices_frame_camintrinsics_camextrinsics[:,2] -= 1


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
              do_optimize_frames                        = True,
              # do_optimize_intrinsics_core filled in later
              do_optimize_intrinsics_distortions        = True,
              do_optimize_extrinsics                    = True,
              do_optimize_calobject_warp                = True,
              do_apply_regularization                   = True,
              do_apply_outlier_rejection                = False)


    if re.search("SPLINED", args.lensmodel):
        # I pre-optimize the core, and then lock it down
        optimization_inputs['lensmodel']                   = 'LENSMODEL_STEREOGRAPHIC'
        optimization_inputs['intrinsics']                  = intrinsics[:,:4].copy()
        optimization_inputs['do_optimize_intrinsics_core'] = True
        stats = mrcal.optimize(**optimization_inputs)
        print(f"optimized. rms = {stats['rms_reproj_error__pixels']}")

        optimization_inputs['lensmodel']                   = args.lensmodel
        Nintrinsics = mrcal.lensmodel_num_params(optimization_inputs['lensmodel'])
        optimization_inputs['intrinsics']                  = nps.glue(optimization_inputs['intrinsics'],
                                                                      np.zeros((args.Ncameras,Nintrinsics-4),),axis=-1)
        optimization_inputs['do_optimize_intrinsics_core'] = False
    else:
        optimization_inputs['lensmodel']                   = model_intrinsics.intrinsics()[0]
        optimization_inputs['intrinsics']                  = intrinsics.copy()
        optimization_inputs['do_optimize_intrinsics_core'] = True

    stats = mrcal.optimize(**optimization_inputs)
    print(f"optimized. rms = {stats['rms_reproj_error__pixels']}")

    return optimization_inputs



Nrange_samples = 80
range_samples = np.logspace( np.log10(args.range_near/10.),
                             np.log10(args.range_far *10.),
                             Nrange_samples)
# shape (N,3). Each row is (0,0,z)
pcam_samples = \
    nps.glue( nps.transpose(range_samples*0),
              nps.transpose(range_samples*0),
              nps.transpose(range_samples),
              axis = -1)

Nfar_samples = 8

if args.Nall is not None:
    Nframes_far_samples  = np.linspace(0, args.Nall,    Nfar_samples, dtype=int)
    Nframes_near_samples = args.Nall - Nframes_far_samples
else:
    Nframes_far_samples = np.linspace(0, args.Nnear*2, Nfar_samples, dtype=int)
    Nframes_near_samples = Nframes_far_samples*0 + args.Nnear


uncertainties = np.zeros((Nfar_samples, Nrange_samples),
                         dtype=float)

for i_Nframes_far in range(Nfar_samples):

    Nframes_far  = Nframes_far_samples [i_Nframes_far]
    Nframes_near = Nframes_near_samples[i_Nframes_far]

    optimization_inputs = solve(Nframes_near, Nframes_far)

    models_out = \
        [ mrcal.cameramodel( optimization_inputs = optimization_inputs,
                             icam_intrinsics     = icam ) \
          for icam in range(args.Ncameras) ]

    icam = 0
    uncertainties[i_Nframes_far] = \
        mrcal.projection_uncertainty(pcam_samples,
                                     models_out[icam],
                                     what='worstdirection-stdev')


if args.Nall is not None:
    title = f"Simulated {args.Ncameras} cameras. Total {args.Nall} chessboard observations. Adding 'far' observations, removing 'near' observations"
else:
    title = f"Simulated {args.Ncameras} cameras. Have {args.Nnear} 'near' chessboard observations. Adding 'far' observations"

gp.plot(range_samples,
        uncertainties,
        legend = np.array([ f"Nfar = {str(i)}" for i in Nframes_far_samples]),
        ymax   = 10.,
        _with  = 'lines',
        _set = ( f"arrow nohead dashtype 3 from {args.range_near},graph 0 to {args.range_near},graph 1",
                 f"arrow nohead dashtype 3 from {args.range_far}, graph 0 to {args.range_far}, graph 1",
                 f"arrow nohead dashtype 3 from graph 0,first {args.observed_pixel_uncertainty} to graph 1,first {args.observed_pixel_uncertainty}"),
        unset  = 'grid',
        title  = title,
        xlabel = 'Range (m)',
        ylabel = 'Expected worst-direction uncertainty (pixels)',
        wait   = not args.explore)

if args.explore:
    import IPython
    IPython.embed()
sys.exit()








r'''
# first attempt: take a previous real-data solve, and add some number of
# further-out observations. This doesn't work. The further-out observations
# actually reduce our confidence at infinity. I THINK this is because I
# don't have ground-truth, and observations I'm adding come off a different
# "true" solve than the actual "true" solve. Does that mean that I can infer
# which is the "true" solution by looking at the simulated uncertainties at
# infinity?



# dima@shorty:~/jpl/mrcal$ ipython3 --pdb -- ./mrcal-calibrate-cameras --imagersize 5120 3840 --focal 1700 --outdir dance-study/ --object-spacing 0.077 --object-width-n 10 --lensmodel LENSMODEL_SPLINED_STEREOGRAPHIC_order=3_Nx=30_Ny=23_fov_x_deg=145 --corners-cache intrinsics.8/corners.vnl --observed-pixel-uncertainty 1.0 '*-cam'{0,1,2,3}.jpg
model = mrcal.cameramodel('../l2/dance8/joint1/camera0-0.cameramodel')

optimization_inputs = model.optimization_inputs()

extrinsics_Rt_toref = mrcal.invert_Rt(mrcal.Rt_from_rt(optimization_inputs['extrinsics_rt_fromref']))

camera_position_mean = np.mean( nps.glue( extrinsics_Rt_toref[:,3,:],
                                          np.zeros((3,)),
                                          axis=-2 ),
                                axis=-2)
camera_forward_mean =  np.mean( nps.glue( extrinsics_Rt_toref[:,:3,2],
                                          np.array((0,0,1)),
                                          axis=-2 ),
                                axis=-2)
camera_forward_mean /= nps.mag(camera_forward_mean)

# I add Nfar identical dense board observations at args.range_far away. Each one
# provides information, and I don't need them to be different for the sake of
# measuring uncertainty
Nfar      = int(sys.argv[1])
args.range_far = 1.0

Ncameras = optimization_inputs['intrinsics']     .shape[0]
Nframes  = optimization_inputs['frames_rt_toref'].shape[0]

# each has shape (Ncameras,Nfar)
iframe,icam = np.meshgrid( np.arange(Nframes, Nframes+Nfar, dtype=np.int32),
                           np.arange(Ncameras,              dtype=np.int32) )
# [ iframe   0..Ncameras-1 ]
# [ iframe+1 0..Ncameras-1 ]
# ...
indices_frame_cam_new = \
    nps.clump( nps.reorder( nps.cat( iframe, icam ), -1,-2,-3 ), n=2)

indices_frame_camintrinsics_camextrinsics_new = \
    nps.glue( indices_frame_cam_new, indices_frame_cam_new[:,(-1,)]-1,
              axis = -1)

optimization_inputs['indices_frame_camintrinsics_camextrinsics'] = \
    nps.glue( optimization_inputs['indices_frame_camintrinsics_camextrinsics'],
              indices_frame_camintrinsics_camextrinsics_new,
              axis = -2 )

obj = mrcal.ref_calibration_object(optimization_inputs['observations_board'].shape[-2],
                                   optimization_inputs['observations_board'].shape[-3],
                                   optimization_inputs['calibration_object_spacing'],
                                   optimization_inputs['calobject_warp'])
obj_center = (obj[0,0] + obj[-1,-1]) / 2.
frame_center = camera_position_mean - obj_center
frame_center[2] += args.range_far
frames_rt_toref_new = nps.glue( np.zeros((3,)),
                                frame_center,
                                axis = -1)

optimization_inputs['frames_rt_toref'] = \
    nps.glue( optimization_inputs['frames_rt_toref'],
              *((frames_rt_toref_new,) * Nfar),
              axis = -2 )

lensmodel       = optimization_inputs['lensmodel']
intrinsics_data = optimization_inputs['intrinsics']

# shape (H,W,3)
obj_ref = \
    mrcal.transform_point_rt( frames_rt_toref_new, obj )
# shape (Ncameras,1,1,6)
extrinsics_rt_fromref = \
    nps.mv( nps.glue( np.zeros((6,)),
                      optimization_inputs['extrinsics_rt_fromref'],
                      axis = -2 ),
            -2, -4 )
# shape (Ncameras,H,W,3)
obj_cam = mrcal.transform_point_rt(extrinsics_rt_fromref,
                                   obj_ref)
# shape (Ncameras,H,W,2)
q_new = \
    mrcal.project(obj_cam, lensmodel,
                  nps.mv(intrinsics_data,-2, -4))
# add weights. shape (Ncameras,H,W,3)
observations_new = nps.glue( q_new, q_new[...,(0,)]*0+1, axis=-1)

optimization_inputs['observations_board'] = \
    nps.glue(optimization_inputs['observations_board'],
             *((observations_new,) * Nfar),
             axis = -4)


# import IPython
# IPython.embed()
# sys.exit()


optimization_inputs['verbose'] = True
mrcal.optimize(**optimization_inputs)

models = [ mrcal.cameramodel( icam_intrinsics     = i,
                              optimization_inputs = optimization_inputs ) \
           for i in range(Ncameras) ]

for i in range(Ncameras):
    models[i].write(f"far{Nfar}-cam{i}.cameramodel")

# mrcal.show_projection_uncertainty(models[0],
#                                   distance = None)

# import IPython
# IPython.embed()

sys.exit()
'''
