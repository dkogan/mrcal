#!/usr/bin/env python3

r'''Triangulation uncertainty quantification test

We look at the triangulated position computed from a pixel observation in two
cameras. Calibration-time noise and triangulation-time noise both affect the
accuracy of the triangulated result. This tool samples both of these noise
sources to make sure the analytical uncertainty predictions are correct

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
                        default = 'cam0',
                        help='''Are we putting the origin at camera0, or are all the frames at a fixed (and
                        non-optimizeable) pose? One or the other is required.''')
    parser.add_argument('--model',
                        type=str,
                        choices=('opencv4','opencv8','splined'),
                        default = 'opencv4',
                        help='''Which lens model we're using. Must be one of
                        ('opencv4','opencv8','splined')''')
    parser.add_argument('--Nframes',
                        type=int,
                        default=50,
                        help='''How many chessboard poses to simulate. These are dense observations: every
                        camera sees every corner of every chessboard pose''')
    parser.add_argument('--Nsamples',
                        type=int,
                        default=500,
                        help='''How many random samples to evaluate''')
    parser.add_argument('--Ncameras',
                        type    = int,
                        default = 2,
                        help='''How many calibration-time cameras to simulate.
                        We will use 2 of these for triangulation, selected with
                        --cameras''')
    parser.add_argument('--cameras',
                        type = int,
                        nargs = 2,
                        default = (0,1),
                        help='''Which cameras we're using for the triangulation.
                        These need to be different, and in [0,Ncameras-1]. The
                        vanilla case will have Ncameras=2, so the default value
                        for this argument (0,1) is correct''')
    parser.add_argument('--do-sample',
                        action='store_true',
                        help='''By default we don't run the time-intensive
                        samples of the calibration solves. This runs a very
                        limited set of tests, and exits. To perform the full set
                        of tests, pass --do-sample''')
    parser.add_argument('--stabilize-coords',
                        action = 'store_true',
                        help='''Whether we report the triangulation in the camera-0 coordinate system (which
                        is moving due to noise) or in a stabilized coordinate
                        system based on the frame poses''')
    parser.add_argument('--cull-left-of-center',
                        action = 'store_true',
                        help='''If given, the calibration data in the left half of the imager is thrown
                        out''')
    parser.add_argument('--q-calibration-stdev',
                        type    = float,
                        default = 0.0,
                        help='''The observed_pixel_uncertainty of the chessboard
                        observations at calibration time. Defaults to 0.0. At
                        least one of --q-calibration-stdev and
                        --q-observation-stdev MUST be given as > 0''')
    parser.add_argument('--q-observation-stdev',
                        type    = float,
                        default = 0.0,
                        help='''The observed_pixel_uncertainty of the point
                        observations at triangulation time. Defaults to 0.0. At
                        least one of --q-calibration-stdev and
                        --q-observation-stdev MUST be given as > 0''')
    parser.add_argument('--q-observation-stdev-correlation',
                        type    = float,
                        default = 0.0,
                        help='''By default, the noise in the observation-time
                        pixel observations is assumed independent. This isn't
                        entirely realistic: observations of the same feature in
                        multiple cameras originate from an imager correlation
                        operation, so they will have some amount of correlation.
                        If given, this argument specifies how much correlation.
                        This is a value in [0,1] scaling the stdev. 0 means
                        "independent" (the default). 1.0 means "100%%
                        correlated".''')
    parser.add_argument('--baseline',
                        type    = float,
                        default = 2.,
                        help='''The baseline of the camera pair. This is the
                        horizontal distance between each pair of adjacent
                        cameras''')
    parser.add_argument('--observed-point',
                        type    = float,
                        nargs   = 3,
                        required = True,
                        help='''The world coordinate of the observed point. Usually this will be ~(small, 0,
                        large). The code will evaluate two points together: the
                        one passed here, and the same one with a negated x
                        coordinate''')
    parser.add_argument('--cache',
                        type=str,
                        choices=('read','write'),
                        help=f'''A cache file stores the recalibration results;
                        computing these can take a long time. This option allows
                        us to or write the cache instead of sampling. The cache
                        file is hardcoded to a cache file (in /tmp). By default,
                        we do neither: we don't read the cache (we sample
                        instead), and we do not write it to disk when we're
                        done. This option is useful for tests where we reprocess
                        the same scenario repeatedly''')
    parser.add_argument('--make-documentation-plots',
                        type=str,
                        help='''If given, we produce plots for the
                        documentation. Takes one argument: a string describing
                        this test. This will be used in the filenames and titles
                        of the resulting plots. Leading directories will be
                        used; whitespace and funny characters in the filename
                        are allowed: will be replaced with _. To make
                        interactive plots, pass ""''')
    parser.add_argument('--ellipse-plot-radius',
                        type=float,
                        help='''By default, the ellipse plot autoscale to show the data and the ellipses
                        nicely. But that means that plots aren't comparable
                        between runs. This option can be passed to select a
                        constant plot width, which allows such comparisons''')
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

    args = parser.parse_args()

    if args.Ncameras < 2:
        raise Exception("--Ncameras must be given at least 2 cameras")
    if args.cameras[0] == args.cameras[1]:
        raise Exception("--cameras must select two different cameras")
    if args.cameras[0] < 0 or args.cameras[0] >= args.Ncameras:
        raise Exception("--cameras must select two different cameras, each in [0,Ncameras-1]")
    if args.cameras[1] < 0 or args.cameras[1] >= args.Ncameras:
        raise Exception("--cameras must select two different cameras, each in [0,Ncameras-1]")

    if args.q_calibration_stdev <= 0.0 and \
       args.q_observation_stdev <= 0.0:
        raise Exception('At least one of --q-calibration-stdev and --q-observation-stdev MUST be given as > 0')
    return args


args = parse_args()


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

    d,f = os.path.split(args.make_documentation_plots)
    args.make_documentation_plots_extratitle = f
    args.make_documentation_plots_path = os.path.join(d, re.sub(r"[^0-9a-zA-Z_\.\-]", "_", f))

    print(f"Will write documentation plots to {args.make_documentation_plots_path}-xxxx.pdf and .png and .svg")

    if terminal['svg'] is None: terminal['svg'] = 'svg size 800,600       noenhanced solid dynamic    font ",14"'
    if terminal['pdf'] is None: terminal['pdf'] = 'pdf size 8in,6in       noenhanced solid color      font ",16"'
    if terminal['png'] is None: terminal['png'] = 'pngcairo size 1024,768 transparent noenhanced crop font ",12"'
else:
    args.make_documentation_plots_extratitle = None

extraset = dict()
for k in pointscale.keys():
    extraset[k] = f'pointsize {pointscale[k]}'



testdir = os.path.dirname(os.path.realpath(__file__))

# I import the LOCAL mrcal since that's what I'm testing
sys.path[:0] = f"{testdir}/..",
import mrcal
import testutils
import copy
import numpy as np
import numpysane as nps
import pickle

from test_calibration_helpers import calibration_baseline,calibration_sample,grad


############# Set up my world, and compute all the perfect positions, pixel
############# observations of everything
fixedframes = (args.fixed == 'frames')
object_spacing          = 0.1
object_width_n          = 10
object_height_n         = 9
calobject_warp_true     = np.array((0.002, -0.005))

# I want the RNG to be deterministic
np.random.seed(0)


rt_cam_ref_true = np.zeros((args.Ncameras,6), dtype=float)
rt_cam_ref_true[:,:3] = np.random.randn(args.Ncameras,3) * 0.1
rt_cam_ref_true[:, 3] = args.baseline * np.arange(args.Ncameras)
rt_cam_ref_true[:,4:] = np.random.randn(args.Ncameras,2) * 0.1

# cam0 is at the identity. This makes my life easy: I can assume that the
# optimization_inputs returned by calibration_baseline() use the same ref
# coordinate system as these transformations.
rt_cam_ref_true[0] *= 0


# shape (Npoints,3)
p_triangulated_true0 = np.array((args.observed_point,
                                 args.observed_point),
                                dtype=float)
# first point has x<0
p_triangulated_true0[0,0] = -np.abs(p_triangulated_true0[0,0])
# second point is the same, but with a negated x: x>0
p_triangulated_true0[1,0] = -p_triangulated_true0[0,0]

Npoints = p_triangulated_true0.shape[0]



@nps.broadcast_define( (('Nintrinsics',),('Nintrinsics',),
                        (6,),(6,),(6,),
                        ('Nframes',6), ('Nframes',6),
                        ('Npoints',2,2)),
                       ('Npoints',3))
def triangulate_nograd( intrinsics_data0, intrinsics_data1,
                        rt_cam0_ref, rt_cam0_ref_baseline, rt_cam1_ref,
                        rt_ref_frame,
                        rt_ref_frame_baseline,
                        q,
                        lensmodel,
                        stabilize_coords = True):

    q = nps.atleast_dims(q,-3)

    rt01 = mrcal.compose_rt(rt_cam0_ref,
                                     mrcal.invert_rt(rt_cam1_ref))

    # all the v have shape (...,3)
    vlocal0 = \
        mrcal.unproject(q[...,0,:],
                        lensmodel, intrinsics_data0)
    vlocal1 = \
        mrcal.unproject(q[...,1,:],
                        lensmodel, intrinsics_data1)

    v0 = vlocal0
    v1 = \
        mrcal.rotate_point_r(rt01[:3], vlocal1)

    # The triangulated point in the perturbed camera-0 coordinate system.
    # Calibration-time perturbations move this coordinate system, so to get
    # a better estimate of the triangulation uncertainty, we try to
    # transform this to the original camera-0 coordinate system; the
    # stabilization path below does that.
    #
    # shape (..., 3)
    p_triangulated0 = \
        mrcal.triangulate_leecivera_mid2(v0, v1, rt01[3:])

    if not stabilize_coords:
        return p_triangulated0

    # Stabilization path. This uses the "true" solution, so I cannot do
    # this in the field. But I CAN do this in the randomized trials in
    # the test. And I can use the gradients to propagate the uncertainty
    # of this computation in the field
    #
    # Data flow:
    #   point_cam_perturbed -> point_ref_perturbed -> point_frames
    #   point_frames -> point_ref_baseline -> point_cam_baseline

    p_cam0_perturbed = p_triangulated0

    p_ref_perturbed = mrcal.transform_point_rt(rt_cam0_ref, p_cam0_perturbed,
                                               inverted = True)

    # shape (..., Nframes, 3)
    p_frames = \
        mrcal.transform_point_rt(rt_ref_frame,
                                 nps.dummy(p_ref_perturbed,-2),
                                 inverted = True)

    # shape (..., Nframes, 3)
    p_ref_baseline_all = mrcal.transform_point_rt(rt_ref_frame_baseline, p_frames)

    # shape (..., 3)
    p_ref_baseline = np.mean(p_ref_baseline_all, axis=-2)

    # shape (..., 3)
    return mrcal.transform_point_rt(rt_cam0_ref_baseline, p_ref_baseline)



################# Sampling
cache_id = f"{args.fixed}-{args.model}-{args.Nframes}-{args.Nsamples}-{args.Ncameras}-{args.cameras[0]}-{args.cameras[1]}-{1 if args.stabilize_coords else 0}-{args.q_calibration_stdev}-{args.q_observation_stdev}-{args.q_observation_stdev_correlation}"
cache_file = f"/tmp/test-triangulation-uncertainty--{cache_id}.pickle"

if args.cache is None or args.cache == 'write':
    optimization_inputs_baseline, \
    models_true,                  \
    frames_true =                 \
        calibration_baseline(args.model,
                             args.Ncameras,
                             args.Nframes,
                             None,
                             object_width_n,
                             object_height_n,
                             object_spacing,
                             rt_cam_ref_true,
                             calobject_warp_true,
                             fixedframes,
                             testdir,
                             cull_left_of_center = args.cull_left_of_center)

    lensmodel   = optimization_inputs_baseline['lensmodel']
    imagersizes = optimization_inputs_baseline['imagersizes']
    Nintrinsics = mrcal.lensmodel_num_params(lensmodel)

else:
    with open(cache_file,"rb") as f:
        (optimization_inputs_baseline,
         models_true,
         lensmodel,
         Nintrinsics,
         imagersizes,
         frames_true,
         intrinsics_sampled,
         extrinsics_sampled_mounted,
         frames_sampled,
         calobject_warp_sampled) = pickle.load(f)


models_baseline = \
    [ mrcal.cameramodel( optimization_inputs = optimization_inputs_baseline,
                         icam_intrinsics     = i) \
      for i in range(args.Ncameras) ]

baseline_rt_ref_frame = optimization_inputs_baseline['rt_ref_frame']

icam0,icam1 = args.cameras

Rt01_true = mrcal.compose_Rt(mrcal.Rt_from_rt(rt_cam_ref_true[icam0]),
                             mrcal.invert_Rt(mrcal.Rt_from_rt(rt_cam_ref_true[icam1])))
Rt10_true = mrcal.invert_Rt(Rt01_true)

# shape (Npoints,Ncameras,3)
p_triangulated_true_local = nps.xchg( nps.cat( p_triangulated_true0,
                                               mrcal.transform_point_Rt(Rt10_true, p_triangulated_true0) ),
                                      0,1)

# Pixel coords at the perfect intersection
# shape (Npoints,Ncameras,2)
q_true = nps.xchg( np.array([ mrcal.project(p_triangulated_true_local[:,i,:],
                                            lensmodel,
                                            models_true[args.cameras[i]].intrinsics()[1]) \
                            for i in range(2)]),
                 0,1)

# Sanity check. Without noise, the triangulation should report the test point exactly
p_triangulated0 = \
    triangulate_nograd(models_true[icam0].intrinsics()[1],
                       models_true[icam1].intrinsics()[1],
                       models_true[icam0].rt_cam_ref(),
                       models_true[icam0].rt_cam_ref(),
                       models_true[icam1].rt_cam_ref(),
                       frames_true, frames_true,
                       q_true,
                       lensmodel,
                       stabilize_coords = args.stabilize_coords)

testutils.confirm_equal(p_triangulated0, p_triangulated_true0,
                        eps = 1e-6,
                        msg = "Noiseless triangulation should be perfect")

p_triangulated0 = \
    triangulate_nograd(models_baseline[icam0].intrinsics()[1],
                       models_baseline[icam1].intrinsics()[1],
                       models_baseline[icam0].rt_cam_ref(),
                       models_baseline[icam0].rt_cam_ref(),
                       models_baseline[icam1].rt_cam_ref(),
                       baseline_rt_ref_frame, baseline_rt_ref_frame,
                       q_true,
                       lensmodel,
                       stabilize_coords = args.stabilize_coords)

testutils.confirm_equal(p_triangulated0, p_triangulated_true0,
                        worstcase   = True,
                        relative    = True,
                        eps         = 1e-2,
                        reldiff_eps = 0.2,
                        msg = "Re-optimized triangulation should be close to the reference. This checks the regularization bias")


slices = ( (q_true[0], (models_baseline[icam0],models_baseline[icam1])),
           (q_true[1], (models_baseline[icam0],models_baseline[icam1])) )

_,                       \
_,                       \
dp_triangulated_dbstate, \
istate_i0,               \
istate_i1,               \
icam_extrinsics0,        \
icam_extrinsics1,        \
istate_e0,               \
istate_e1 =              \
    mrcal.triangulation._triangulation_uncertainty_internal(slices,
                                                            optimization_inputs_baseline,
                                                            0,0,
                                                            stabilize_coords = args.stabilize_coords)

########## Gradient check
dp_triangulated_di0_empirical = grad(lambda i0: triangulate_nograd(i0, models_baseline[icam1].intrinsics()[1],
                                                                   models_baseline[icam0].rt_cam_ref(),
                                                                   models_baseline[icam0].rt_cam_ref(),
                                                                   models_baseline[icam1].rt_cam_ref(),
                                                                   baseline_rt_ref_frame, baseline_rt_ref_frame,
                                                                   q_true,
                                                                   lensmodel,
                                                                   stabilize_coords=args.stabilize_coords),
                                     models_baseline[icam0].intrinsics()[1],
                                     step = 1e-5)
dp_triangulated_di1_empirical = grad(lambda i1: triangulate_nograd(models_baseline[icam0].intrinsics()[1],i1,
                                                                   models_baseline[icam0].rt_cam_ref(),
                                                                   models_baseline[icam0].rt_cam_ref(),
                                                                   models_baseline[icam1].rt_cam_ref(),
                                                                   baseline_rt_ref_frame, baseline_rt_ref_frame,
                                                                   q_true,
                                                                   lensmodel,
                                                                   stabilize_coords=args.stabilize_coords),
                                     models_baseline[icam1].intrinsics()[1])
dp_triangulated_de1_empirical = grad(lambda e1: triangulate_nograd(models_baseline[icam0].intrinsics()[1], models_baseline[icam1].intrinsics()[1],
                                                                   models_baseline[icam0].rt_cam_ref(),
                                                                   models_baseline[icam0].rt_cam_ref(),
                                                                   e1,
                                                                   baseline_rt_ref_frame, baseline_rt_ref_frame,
                                                                   q_true,
                                                                   lensmodel,
                                                                   stabilize_coords=args.stabilize_coords),
                                     models_baseline[icam1].rt_cam_ref())

dp_triangulated_de0_empirical = grad(lambda e0: triangulate_nograd(models_baseline[icam0].intrinsics()[1], models_baseline[icam1].intrinsics()[1],
                                                                   e0,
                                                                   models_baseline[icam0].rt_cam_ref(),
                                                                   models_baseline[icam1].rt_cam_ref(),
                                                                   baseline_rt_ref_frame, baseline_rt_ref_frame,
                                                                   q_true,
                                                                   lensmodel,
                                                                   stabilize_coords=args.stabilize_coords),
                                     models_baseline[icam0].rt_cam_ref())

dp_triangulated_drtrf_empirical = grad(lambda rtrf: triangulate_nograd(models_baseline[icam0].intrinsics()[1], models_baseline[icam1].intrinsics()[1],
                                                                       models_baseline[icam0].rt_cam_ref(),
                                                                       models_baseline[icam0].rt_cam_ref(),
                                                                       models_baseline[icam1].rt_cam_ref(),
                                                                       rtrf, baseline_rt_ref_frame,
                                                                       q_true,
                                                                       lensmodel,
                                                                       stabilize_coords=args.stabilize_coords),
                                       baseline_rt_ref_frame,
                                       step = 1e-5)

dp_triangulated_dq_empirical = grad(lambda q: triangulate_nograd(models_baseline[icam0].intrinsics()[1], models_baseline[icam1].intrinsics()[1],
                                                                 models_baseline[icam0].rt_cam_ref(),
                                                                 models_baseline[icam0].rt_cam_ref(),
                                                                 models_baseline[icam1].rt_cam_ref(),
                                                                 baseline_rt_ref_frame, baseline_rt_ref_frame,
                                                                 q,
                                                                 lensmodel,
                                                                 stabilize_coords=args.stabilize_coords),
                                    q_true,
                                    step = 1e-3)

testutils.confirm_equal(dp_triangulated_dbstate[...,istate_i0:istate_i0+Nintrinsics],
                        dp_triangulated_di0_empirical,
                        relative = True,
                        worstcase = True,
                        eps = 0.1,
                        reldiff_eps = 1e-5,
                        msg = "Gradient check: dp_triangulated_dbstate[intrinsics0]")
testutils.confirm_equal(dp_triangulated_dbstate[...,istate_i1:istate_i1+Nintrinsics],
                        dp_triangulated_di1_empirical,
                        relative = True,
                        worstcase = True,
                        eps = 0.1,
                        msg = "Gradient check: dp_triangulated_dbstate[intrinsics1]")
if istate_e0 is not None:
    testutils.confirm_equal(dp_triangulated_dbstate[...,istate_e0:istate_e0+6],
                            dp_triangulated_de0_empirical,
                            relative = True,
                            worstcase = True,
                            eps = 5e-3,
                            msg = "Gradient check: dp_triangulated_dbstate[extrinsics0]")
if istate_e1 is not None:
    testutils.confirm_equal(dp_triangulated_dbstate[...,istate_e1:istate_e1+6],
                            dp_triangulated_de1_empirical,
                            relative = True,
                            worstcase = True,
                            eps = 5e-3,
                            msg = "Gradient check: dp_triangulated_dbstate[extrinsics1]")


if optimization_inputs_baseline.get('do_optimize_frames'):
    istate_f0     = mrcal.state_index_frames(0, **optimization_inputs_baseline)
    Nstate_frames = mrcal.num_states_frames(    **optimization_inputs_baseline)
    testutils.confirm_equal(dp_triangulated_dbstate[...,istate_f0:istate_f0+Nstate_frames],
                            nps.clump(dp_triangulated_drtrf_empirical, n=-2),
                            relative = True,
                            worstcase = True,
                            eps = 0.1,
                            msg = "Gradient check: dp_triangulated_drtrf")

# dp_triangulated_dq_empirical has shape (Npoints,3,  Npoints,Ncameras,2)
# The cross terms (p_triangulated(point=A), q(point=B)) should all be zero
dp_triangulated_dq_empirical_cross_only = dp_triangulated_dq_empirical
dp_triangulated_dq_empirical = np.zeros((Npoints,3,2,2), dtype=float)

dp_triangulated_dq = np.zeros((Npoints,3,2,2), dtype=float)
dp_triangulated_dq_flattened = nps.clump(dp_triangulated_dq, n=-2)

for ipt in range(Npoints):
    dp_triangulated_dq_empirical[ipt,...] = dp_triangulated_dq_empirical_cross_only[ipt,:,ipt,:,:]
    dp_triangulated_dq_empirical_cross_only[ipt,:,ipt,:,:] = 0

    p = np.zeros((3,), dtype=float)
    dp_triangulated_dq_flattened[ipt] = \
        mrcal.triangulation._triangulate_grad_simple(*slices[ipt], p)


testutils.confirm_equal(dp_triangulated_dq_empirical_cross_only,
                        0,
                        eps = 1e-6,
                        msg = "Gradient check: dp_triangulated_dq: cross-point terms are 0")
testutils.confirm_equal(dp_triangulated_dq,
                        dp_triangulated_dq_empirical,
                        relative = True,
                        worstcase = True,
                        eps = 5e-3,
                        msg = "Gradient check: dp_triangulated_dq")


p_alone = \
    mrcal.triangulate( q_true, (models_baseline[icam0],models_baseline[icam1]),
                       stabilize_coords = args.stabilize_coords )
p_calnoise,               \
Var_p_calibration_alone = \
    mrcal.triangulate( q_true, (models_baseline[icam0],models_baseline[icam1]),
                       q_calibration_stdev = args.q_calibration_stdev,
                       stabilize_coords    = args.stabilize_coords )
p_obsnoise,               \
Var_p_observation_alone = \
    mrcal.triangulate( q_true, (models_baseline[icam0],models_baseline[icam1]),
                       q_observation_stdev             = args.q_observation_stdev,
                       q_observation_stdev_correlation = args.q_observation_stdev_correlation,
                       stabilize_coords    = args.stabilize_coords )
p,                  \
Var_p_calibration,  \
Var_p_observation,  \
Var_p_joint = \
    mrcal.triangulate( q_true, (models_baseline[icam0],models_baseline[icam1]),
                       q_calibration_stdev             = args.q_calibration_stdev,
                       q_observation_stdev             = args.q_observation_stdev,
                       q_observation_stdev_correlation = args.q_observation_stdev_correlation,
                       stabilize_coords                = args.stabilize_coords)

testutils.confirm_equal(p_alone,
                        p_triangulated0,
                        eps = 1e-6,
                        msg = "triangulate(no noise) returns the right point")
testutils.confirm_equal(p_alone,
                        p_calnoise,
                        eps = 1e-6,
                        msg = "triangulate(cal noise) returns the right point")
testutils.confirm_equal(p_alone,
                        p_obsnoise,
                        eps = 1e-6,
                        msg = "triangulate(obs noise) returns the right point")
testutils.confirm_equal(p_alone,
                        p,
                        eps = 1e-6,
                        msg = "triangulate(both noise) returns the right point")
testutils.confirm_equal(Var_p_calibration_alone,
                        Var_p_calibration,
                        eps = 1e-6,
                        msg = "triangulate(cal noise) returns a consistent Var_p_calibration")
testutils.confirm_equal(Var_p_observation_alone,
                        Var_p_observation,
                        eps = 1e-6,
                        msg = "triangulate(obs noise) returns a consistent Var_p_observation")
testutils.confirm_equal(Var_p_joint.shape,
                        Var_p_calibration.shape,
                        msg = "Var_p_joint.shape matches Var_p_calibration.shape")

for ipt in range(Npoints):
    testutils.confirm_equal(Var_p_joint[ipt,:,ipt,:],
                            Var_p_calibration[ipt,:,ipt,:] + Var_p_observation[ipt,:,:],
                            worstcase = True,
                            eps       = 1e-9,
                            msg       = "Var(joint) should be Var(cal-time-noise) + Var(obs-time-noise)")

if not args.do_sample:
    testutils.finish()
    sys.exit()

try:
    intrinsics_sampled
    did_sample = True
except:
    did_sample = False

if not did_sample:

    ( intrinsics_sampled,         \
      extrinsics_sampled_mounted, \
      frames_sampled,             \
      points_sampled,             \
      calobject_warp_sampled,     \
      q_noise_board_sampled,      \
      q_noise_point_sampled,      \
      b_sampled,                  \
      optimization_inputs_sampled ) = \
          calibration_sample( args.Nsamples,
                              optimization_inputs_baseline,
                              args.q_calibration_stdev,
                              fixedframes)

    if args.cache is not None and args.cache == 'write':
        with open(cache_file,"wb") as f:
            pickle.dump((optimization_inputs_baseline,
                         models_true,
                         models_baseline,
                         lensmodel,
                         Nintrinsics,
                         imagersizes,
                         frames_true,
                         intrinsics_sampled,
                         extrinsics_sampled_mounted,
                         frames_sampled,
                         calobject_warp_sampled),
                        f)
        print(f"Wrote cache to {cache_file}")



# Let's actually apply the noise to compute var(distancep) empirically to compare
# against the var(distancep) prediction I just computed
# shape (Nsamples,Npoints,2,2)
var_qt_onepoint = \
    mrcal.triangulation._compute_Var_q_triangulation(args.q_observation_stdev,
                                                     args.q_observation_stdev_correlation)
var_qt = np.zeros((Npoints*2*2, Npoints*2*2), dtype=float)
for i in range(Npoints):
    var_qt[4*i:4*(i+1), 4*i:4*(i+1)] = var_qt_onepoint
# I want the RNG to be deterministic. If I turn caching on/off I'd be at a
# different place in the RNG here. I reset to keep things consistent
np.random.seed(0)
qt_noise = \
    np.random.multivariate_normal( mean = np.zeros((Npoints*2*2,),),
                                   cov  = var_qt,
                                   size = args.Nsamples ).reshape(args.Nsamples,Npoints,2,2)
q_sampled = q_true + qt_noise


# I have the perfect observation pixel coords. I triangulate them through my
# sampled calibration
if fixedframes:
    extrinsics_sampled_cam0 = extrinsics_sampled_mounted[..., icam_extrinsics0,:]
    extrinsics_sampled_cam1 = extrinsics_sampled_mounted[..., icam_extrinsics1,:]
else:
    # if not fixedframes: extrinsics_sampled_mounted is prepended with the
    # extrinsics for cam0 (selected with icam_extrinsics==-1)
    extrinsics_sampled_cam0 = extrinsics_sampled_mounted[..., icam_extrinsics0+1,:]
    extrinsics_sampled_cam1 = extrinsics_sampled_mounted[..., icam_extrinsics1+1,:]

p_triangulated_sampled0 = triangulate_nograd(intrinsics_sampled[...,icam0,:], intrinsics_sampled[...,icam1,:],
                                             extrinsics_sampled_cam0,
                                             models_baseline[icam0].rt_cam_ref(),
                                             extrinsics_sampled_cam1,
                                             frames_sampled, baseline_rt_ref_frame,
                                             q_sampled,
                                             lensmodel,
                                             stabilize_coords = args.stabilize_coords)


ranges              = nps.mag(p_triangulated0)
ranges_true         = nps.mag(p_triangulated_true0)
ranges_sampled      = nps.transpose(nps.mag(p_triangulated_sampled0))
mean_ranges_sampled = ranges_sampled.mean( axis = -1)
Var_ranges_sampled  = ranges_sampled.var(  axis = -1)
# r = np.mag(p)
# dr_dp = p/r
# Var(r) = dr_dp var(p) dr_dpT
#        = p var(p) pT / norm2(p)
Var_ranges_joint        = np.zeros((Npoints,), dtype=float)
Var_ranges_calibration  = np.zeros((Npoints,), dtype=float)
Var_ranges_observations = np.zeros((Npoints,), dtype=float)
for ipt in range(Npoints):
    Var_ranges_joint[ipt] = \
        nps.matmult(p_triangulated0[ipt],
                    Var_p_joint[ipt,:,ipt,:],
                    nps.transpose(p_triangulated0[ipt]))[0] / nps.norm2(p_triangulated0[ipt])
    Var_ranges_calibration[ipt] = \
        nps.matmult(p_triangulated0[ipt],
                    Var_p_calibration[ipt,:,ipt,:],
                    nps.transpose(p_triangulated0[ipt]))[0] / nps.norm2(p_triangulated0[ipt])
    Var_ranges_observations[ipt] = \
        nps.matmult(p_triangulated0[ipt],
                    Var_p_observation[ipt,:,:],
                    nps.transpose(p_triangulated0[ipt]))[0] / nps.norm2(p_triangulated0[ipt])


diff                  = p_triangulated0[1] - p_triangulated0[0]
distance              = nps.mag(diff)
distance_true         = nps.mag(p_triangulated_true0[:,0] - p_triangulated_true0[:,1])
distance_sampled      = nps.mag(p_triangulated_sampled0[:,1,:] - p_triangulated_sampled0[:,0,:])
mean_distance_sampled = distance_sampled.mean()
Var_distance_sampled  = distance_sampled.var()
# diff = p1-p0
# dist = np.mag(diff)
# ddist_dp01 = [-diff   diff] / dist
# Var(dist) = ddist_dp01 var(p01) ddist_dp01T
#           = [-diff   diff] var(p01) [-diff   diff]T / norm2(diff)
Var_distance = nps.matmult(nps.glue( -diff, diff, axis=-1),
                           Var_p_joint.reshape(Npoints*3,Npoints*3),
                           nps.transpose(nps.glue( -diff, diff, axis=-1),))[0] / nps.norm2(diff)



# I have the observed and predicted distributions, so I make sure things match.
# For some not-yet-understood reason, the distance distribution isn't
# normally-distributed: there's a noticeable fat tail. Thus I'm not comparing
# those two distributions (Var_distance,Var_distance_sampled) in this test.
p_sampled = nps.clump(p_triangulated_sampled0, n=-2)
mean_p_sampled = np.mean(p_sampled, axis=-2)
Var_p_sampled  = nps.matmult( nps.transpose(p_sampled - mean_p_sampled),
                                 p_sampled - mean_p_sampled ) / args.Nsamples

testutils.confirm_equal(mean_p_sampled,
                        p_triangulated_true0.ravel(),
                        worstcase = True,
                        # High threshold. Need many more samples to be able to
                        # reduce it
                        eps = p_triangulated_true0[0,2]/150.,
                        msg = "Triangulated position matches sampled mean")

for ipt in range(2):
    testutils.confirm_covariances_equal(Var_p_joint[ipt,:,ipt,:],
                                        Var_p_sampled[ipt*3:(ipt+1)*3,ipt*3:(ipt+1)*3],
                                        what = f"triangulated point variance for point {ipt}",

                                        # This is a relatively-high threshold. I can tighten
                                        # it, but then I'd need to collect more
                                        # samples. Non-major axes have more
                                        # relative error
                                        eps_eigenvalues      = (0.15, 0.2, 0.3),
                                        eps_eigenvectors_deg = 6)


# It would be great to test the distribution of the difference between two
# points, but I'm not doing that: it doesn't look very gaussian while the
# analytical uncertainty IS gaussian. This needs investigation


if not (args.explore or \
        args.make_documentation_plots is not None):
    testutils.finish()
    sys.exit()



if args.make_documentation_plots is not None:

    import gnuplotlib as gp

    empirical_distributions_xz = \
        [ mrcal.utils._plot_args_points_and_covariance_ellipse(p_triangulated_sampled0[:,ipt,(0,2)],
                                                               'Observed') \
          for ipt in range(Npoints) ]
    # Individual covariances
    Var_p_joint_diagonal = [Var_p_joint[ipt,:,ipt,:][(0,2),:][:,(0,2)] \
                               for ipt in range(Npoints)]
    Var_p_calibration_diagonal = [Var_p_calibration[ipt,:,ipt,:][(0,2),:][:,(0,2)] \
                                     for ipt in range(Npoints)]
    Var_p_observation_diagonal = [Var_p_observation[ipt,:,:][(0,2),:][:,(0,2)] \
                                      for ipt in range(Npoints)]

    max_sigma_points = np.array([ np.max(np.sqrt(np.linalg.eig(V)[0])) for V in Var_p_joint_diagonal ])
    max_sigma = np.max(max_sigma_points)

    if args.ellipse_plot_radius is not None:
        ellipse_plot_radius = args.ellipse_plot_radius
    else:
        ellipse_plot_radius = max_sigma*3

    title_triangulation = 'Triangulation uncertainty'
    title_covariance    = 'Abs(Covariance) of the [p0,p1] vector (m^2)'
    title_range0        = 'Range to the left triangulated point'
    title_distance      = 'Distance between the two triangulated points'

    if args.make_documentation_plots_extratitle is not None:
        title_triangulation += f': {args.make_documentation_plots_extratitle}'
        title_covariance    += f': {args.make_documentation_plots_extratitle}'
        title_range0        += f': {args.make_documentation_plots_extratitle}'
        title_distance      += f': {args.make_documentation_plots_extratitle}'


    subplots = [ [p for p in (empirical_distributions_xz[ipt][1], # points; plot first to not obscure the ellipses
                              mrcal.utils._plot_arg_covariance_ellipse(p_triangulated0[ipt][(0,2),],
                                                                       Var_p_joint_diagonal[ipt],
                                                                       "Predicted-joint"),
                              mrcal.utils._plot_arg_covariance_ellipse(p_triangulated0[ipt][(0,2),],
                                                                       Var_p_calibration_diagonal[ipt],
                                                                       "Predicted-calibration-only"),
                              mrcal.utils._plot_arg_covariance_ellipse(p_triangulated0[ipt][(0,2),],
                                                                       Var_p_observation_diagonal[ipt],
                                                                       "Predicted-observations-only"),
                              empirical_distributions_xz[ipt][0],
                              dict( square = True,
                                    _xrange = [p_triangulated0[ipt,0] - ellipse_plot_radius,
                                               p_triangulated0[ipt,0] + ellipse_plot_radius],
                                    _yrange = [p_triangulated0[ipt,2] - ellipse_plot_radius,
                                               p_triangulated0[ipt,2] + ellipse_plot_radius],
                                    xlabel  = 'Triangulated point x (left/right) (m)',
                                    ylabel  = 'Triangulated point z (forward/back) (m)',)
                              )
                  # ignore all None plot items. This will throw away any
                  # infinitely-small ellipses
                  if p is not None ] \
                 for ipt in range(Npoints) ]

    def makeplots(dohardcopy, processoptions_base):

        processoptions = copy.deepcopy(processoptions_base)
        gp.add_plot_option(processoptions,
                           'set',
                           ('xtics 0.01',
                            'ytics 0.01'))

        # The key should use smaller text than the rest of the plot, if possible
        if 'terminal' in processoptions:
            m = re.search('font ",([0-9]+)"', processoptions['terminal'])
            if m is not None:
                s = int(m.group(1))

                gp.add_plot_option(processoptions,
                                   'set',
                                   ('xtics 0.01',
                                    'ytics 0.01',
                                    f'key font ",{int(s*0.8)}"'))

        if dohardcopy:
            processoptions['hardcopy'] = \
                f'{args.make_documentation_plots_path}--ellipses.{extension}'
            processoptions['terminal'] = shorter_terminal(processoptions['terminal'])

            # Hardcopies do things a little differently, to be nicer for docs
            gp.plot( *subplots,
                     multiplot = f'layout 1,2',
                     unset = 'grid',
                     **processoptions )
            if processoptions.get('hardcopy') and extension == 'pdf':
                os.system(f"pdfcrop {processoptions['hardcopy']}")

        else:
            # Interactive plotting, so no multiplots. Interactive plots
            for p in subplots:
                gp.plot( *p[:-1], **p[-1], **processoptions )

        processoptions = copy.deepcopy(processoptions_base)
        if dohardcopy:
            processoptions['hardcopy'] = \
                f'{args.make_documentation_plots_path}--p0-p1-magnitude-covariance.{extension}'
        processoptions['title'] = title_covariance
        gp.plotimage( np.abs(Var_p_joint.reshape(Npoints*3,Npoints*3)),
                      square = True,
                      xlabel = 'Variable index (left point x,y,z; right point x,y,z)',
                      ylabel = 'Variable index (left point x,y,z; right point x,y,z)',
                      **processoptions)
        if processoptions.get('hardcopy') and extension == 'pdf':
            os.system(f"pdfcrop {processoptions['hardcopy']}")


        processoptions = copy.deepcopy(processoptions_base)
        binwidth = np.sqrt(Var_ranges_joint[0]) / 4.
        equation_range0_observed_gaussian = \
            mrcal.fitted_gaussian_equation(x        = ranges_sampled[0],
                                           binwidth = binwidth,
                                           legend   = "Idealized gaussian fit to data")
        equation_range0_predicted_joint_gaussian = \
            mrcal.fitted_gaussian_equation(mean     = ranges[0],
                                           sigma    = np.sqrt(Var_ranges_joint[0]),
                                           N        = len(ranges_sampled[0]),
                                           binwidth = binwidth,
                                           legend   = "Predicted-joint")
        equation_range0_predicted_calibration_gaussian = \
            mrcal.fitted_gaussian_equation(mean     = ranges[0],
                                           sigma    = np.sqrt(Var_ranges_calibration[0]),
                                           N        = len(ranges_sampled[0]),
                                           binwidth = binwidth,
                                           legend   = "Predicted-calibration")
        equation_range0_predicted_observations_gaussian = \
            mrcal.fitted_gaussian_equation(mean     = ranges[0],
                                           sigma    = np.sqrt(Var_ranges_observations[0]),
                                           N        = len(ranges_sampled[0]),
                                           binwidth = binwidth,
                                           legend   = "Predicted-observations")
        if dohardcopy:
            processoptions['hardcopy'] = \
                f'{args.make_documentation_plots_path}--range-to-p0.{extension}'
        processoptions['title'] = title_range0
        gp.add_plot_option(processoptions, 'set', 'samples 1000')
        gp.plot(ranges_sampled[0],
                histogram       = True,
                binwidth        = binwidth,
                equation_above  = (equation_range0_predicted_joint_gaussian,
                                   equation_range0_predicted_calibration_gaussian,
                                   equation_range0_predicted_observations_gaussian,
                                   equation_range0_observed_gaussian),
                xlabel          = "Range to the left triangulated point (m)",
                ylabel          = "Frequency",
                **processoptions)
        if processoptions.get('hardcopy') and extension == 'pdf':
            os.system(f"pdfcrop {processoptions['hardcopy']}")

        processoptions = copy.deepcopy(processoptions_base)
        binwidth = np.sqrt(Var_distance) / 4.
        equation_distance_observed_gaussian = \
            mrcal.fitted_gaussian_equation(x        = distance_sampled,
                                           binwidth = binwidth,
                                           legend   = "Idealized gaussian fit to data")
        equation_distance_predicted_gaussian = \
            mrcal.fitted_gaussian_equation(mean     = distance,
                                           sigma    = np.sqrt(Var_distance),
                                           N        = len(distance_sampled),
                                           binwidth = binwidth,
                                           legend   = "Predicted")
        if dohardcopy:
            processoptions['hardcopy'] = \
                f'{args.make_documentation_plots_path}--distance-p1-p0.{extension}'
        processoptions['title'] = title_distance
        gp.add_plot_option(processoptions, 'set', 'samples 1000')
        gp.plot(distance_sampled,
                histogram       = True,
                binwidth        = binwidth,
                equation_above  = (equation_distance_predicted_gaussian,
                                   equation_distance_observed_gaussian),
                xlabel          = "Distance between triangulated points (m)",
                ylabel          = "Frequency",
                **processoptions)
        if processoptions.get('hardcopy') and extension == 'pdf':
            os.system(f"pdfcrop {processoptions['hardcopy']}")

    if args.make_documentation_plots:
        for extension in ('pdf','svg','png','gp'):
            makeplots(dohardcopy = True,
                      processoptions_base = dict(wait      = False,
                                                 terminal  = terminal[extension],
                                                 _set      = extraset[extension]))
    else:
        makeplots(dohardcopy = False,
                  processoptions_base = dict(wait = True))

if args.explore:
    import gnuplotlib as gp
    import IPython
    IPython.embed()
