#!/usr/bin/python3

r'''Triangulation uncertainty quantification test

'''

import sys
import argparse
import re
import os

cache_file = "/tmp/test-triangulation-uncertainty.pickle"

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
                        default=100,
                        help='''How many random samples to evaluate''')
    parser.add_argument('--Ncameras',
                        type    = int,
                        default = 2,
                        help='''How many cameras to simulate. At this time, only "2" is supported''')
    parser.add_argument('--stabilize-coords',
                        action = 'store_true',
                        help='''Whether we report the triangulation in the camera-0 coordinate system (which
                        is moving due to noise) or in a stabilized coordinate
                        system based on the frame poses''')
    parser.add_argument('--cull-left-of-center',
                        action = 'store_true',
                        help='''If given, the calibration data in the left half of the imager is thrown
                        out''')
    parser.add_argument('--pixel-uncertainty-stdev-calibration',
                        type    = float,
                        default = 0.5,
                        help='''The observed_pixel_uncertainty of the chessboard observations at calibration
                        time''')
    parser.add_argument('--baseline-calibration',
                        type    = float,
                        default = 2.,
                        help='''The baseline of the camera pair''')
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
                        help=f'''Whether we should read or write the cache instead of sampling. The cache file
                        is hardcoded to {cache_file}. By default, we do neither:
                        we don't read the cache (we sample instead), and we do
                        not write it to disk when we're done. This option is
                        useful for tests where we reprocess the same scenario repeatedly''')
    parser.add_argument('--make-documentation-plots',
                        type=str,
                        help='''If given, we produce plots for the documentation. Takes one argument: a
                        string describing this test. This will be used in the
                        filenames of the resulting plots. To make interactive
                        plots, pass ""''')
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

    if args.fixed != 'cam0':
        raise Exception("'--fixed cam0' is the only supported option at this time")
    if args.Ncameras != 2:
        raise Exception("'--Ncameras 2' is the only supported option at this time")
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

if args.make_documentation_plots:

    print(f"Will write documentation plots to {args.make_documentation_plots}-xxxx.pdf and .svg")

    if terminal['svg'] is None: terminal['svg'] = 'svg size 800,600       noenhanced solid dynamic    font ",14"'
    if terminal['pdf'] is None: terminal['pdf'] = 'pdf size 8in,6in       noenhanced solid color      font ",12"'
    if terminal['png'] is None: terminal['png'] = 'pngcairo size 1024,768 transparent noenhanced crop font ",12"'

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

from test_calibration_helpers import plot_args_points_and_covariance_ellipse,plot_arg_covariance_ellipse,calibration_baseline,calibration_sample,grad


############# Set up my world, and compute all the perfect positions, pixel
############# observations of everything
fixedframes = (args.fixed == 'frames')
object_spacing          = 0.1
object_width_n          = 10
object_height_n         = 9
calobject_warp_true     = np.array((0.002, -0.005))

extrinsics_rt_fromref_true = \
    np.array(((0,    0,    0,                                  0,   0,   0),
              (0.08, 0.2,  0.02,   args.baseline_calibration,  0.09, 0.01), ))

# 1km straight ahead
# shape (Npoints,3)
p_triangulated_true = np.array((args.observed_point,
                                args.observed_point),
                               dtype=float)
# first point has x<0
p_triangulated_true[0,0] = -np.abs(p_triangulated_true[0,0])
# second point is the same, but with a negated x: x>0
p_triangulated_true[1,0] = -p_triangulated_true[0,0]


# I want the RNG to be deterministic
np.random.seed(0)


@nps.broadcast_define( ((2,'Nintrinsics'),
                        (2,6),
                        ('Nframes',6), ('Nframes',6),
                        ('Npoints','Ncameras',2)),
                       ('Npoints',3))
def triangulate_nograd( intrinsics_data,
                        rt_cam_ref,
                        rt_ref_frame, rt_ref_frame_true,
                        q,
                        lensmodel,
                        stabilize_coords = True):
    return _triangulate( intrinsics_data,
                         rt_cam_ref,
                         rt_ref_frame,
                         rt_ref_frame_true,
                         nps.atleast_dims(q,-3),
                         lensmodel,
                         stabilize_coords = stabilize_coords,
                         get_gradients = False)


@nps.broadcast_define( ((2,'Nintrinsics'),
                        (2,6),
                        ('Nframes',6), ('Nframes',6),
                        ('Npoints','Ncameras',2)),
                       (('Npoints',3),
                        (6,6),
                        (6,6),(6,6),
                        ('Npoints',3,2),('Npoints',3,'Nintrinsics'),
                        ('Npoints',3,2),('Npoints',3,'Nintrinsics'),
                        ('Npoints',3,3),('Npoints',3,3),
                        ('Npoints',3,3),('Npoints',3,3),('Npoints',3,3),
                        ('Npoints','Nframes',3,6)))
def triangulate_grad( intrinsics_data,
                      rt_cam_ref,
                      rt_ref_frame, rt_ref_frame_true,
                      q,
                      lensmodel,
                      stabilize_coords = True):
    return _triangulate( intrinsics_data,
                         rt_cam_ref,
                         rt_ref_frame,
                         rt_ref_frame_true,
                         nps.atleast_dims(q,-3),
                         lensmodel,
                         stabilize_coords = stabilize_coords,
                         get_gradients    = True)


def _triangulate(# shape (Ncameras, Nintrinsics)
                 intrinsics_data,
                 # shape (Ncameras, 6)
                 rt_cam_ref,
                 # shape (Nframes,6),
                 rt_ref_frame, rt_ref_frame_true,
                 # shape (..., Ncameras, 2)
                 q,

                 lensmodel,
                 stabilize_coords,
                 get_gradients):

    if not ( intrinsics_data.ndim == 2 and intrinsics_data.shape[0] == 2 and \
             rt_cam_ref.shape == (2,6) and \
             rt_ref_frame.ndim == 2 and rt_ref_frame.shape[-1] == 6 and \
             q.shape[-2:] == (2,2 ) ):
        raise Exception("Arguments must have a consistent Ncameras == 2")

    # I now compute the same triangulation, but just at the un-perturbed baseline,
    # and keeping track of all the gradients
    rt0r = rt_cam_ref[0]
    rt1r = rt_cam_ref[1]

    if not get_gradients:

        rtr1          = mrcal.invert_rt(rt1r)
        rt01_baseline = mrcal.compose_rt(rt0r, rtr1)

        # all the v have shape (...,3)
        vlocal0 = \
            mrcal.unproject(q[...,0,:],
                            lensmodel, intrinsics_data[0])
        vlocal1 = \
            mrcal.unproject(q[...,1,:],
                            lensmodel, intrinsics_data[1])

        v0 = vlocal0
        v1 = \
            mrcal.rotate_point_r(rt01_baseline[:3], vlocal1)

        # p_triangulated has shape (..., 3)
        p_triangulated = \
            mrcal.triangulate_leecivera_mid2(v0, v1, rt01_baseline[3:])

        if stabilize_coords:

            # shape (..., Nframes, 3)
            p_frames_new = \
                mrcal.transform_point_rt(mrcal.invert_rt(rt_ref_frame),
                                         nps.dummy(p_triangulated,-2))

            # shape (..., Nframes, 3)
            p_refs = mrcal.transform_point_rt(rt_ref_frame_true, p_frames_new)

            # shape (..., 3)
            p_triangulated = np.mean(p_refs, axis=-2)

        return p_triangulated
    else:
        rtr1,drtr1_drt1r = mrcal.invert_rt(rt1r,
                                           get_gradients=True)
        rt01_baseline,drt01_drt0r, drt01_drtr1 = mrcal.compose_rt(rt0r, rtr1, get_gradients=True)

        # all the v have shape (...,3)
        vlocal0, dvlocal0_dq0, dvlocal0_dintrinsics0 = \
            mrcal.unproject(q[...,0,:],
                            lensmodel, intrinsics_data[0],
                            get_gradients = True)
        vlocal1, dvlocal1_dq1, dvlocal1_dintrinsics1 = \
            mrcal.unproject(q[...,1,:],
                            lensmodel, intrinsics_data[1],
                            get_gradients = True)

        v0 = vlocal0
        v1, dv1_dr01, dv1_dvlocal1 = \
            mrcal.rotate_point_r(rt01_baseline[:3], vlocal1,
                                 get_gradients=True)

        # p_triangulated has shape (..., 3)
        p_triangulated, dp_triangulated_dv0, dp_triangulated_dv1, dp_triangulated_dt01 = \
            mrcal.triangulate_leecivera_mid2(v0, v1, rt01_baseline[3:],
                                             get_gradients = True)

        Nframes = len(rt_ref_frame)

        if stabilize_coords:

            # shape (Nframes,6)
            rt_frame_ref, drtfr_drtrf = \
                mrcal.invert_rt(rt_ref_frame, get_gradients=True)

            # shape (Nframes,6)
            rt_true_shifted, _, drt_drtfr = \
                mrcal.compose_rt(rt_ref_frame_true, rt_frame_ref,
                                 get_gradients=True)

            # shape (..., Nframes, 3)
            p_refs,dprefs_drt,dprefs_dptriangulated = \
                mrcal.transform_point_rt(rt_true_shifted,
                                         nps.dummy(p_triangulated,-2),
                                         get_gradients = True)

            # shape (..., 3)
            p_triangulated = np.mean(p_refs, axis=-2)

            # I have dpold/dx. dpnew/dx = dpnew/dpold dpold/dx

            # shape (...,3,3)
            dpnew_dpold = np.mean(dprefs_dptriangulated, axis=-3)
            dp_triangulated_dv0  = nps.matmult(dpnew_dpold, dp_triangulated_dv0)
            dp_triangulated_dv1  = nps.matmult(dpnew_dpold, dp_triangulated_dv1)
            dp_triangulated_dt01 = nps.matmult(dpnew_dpold, dp_triangulated_dt01)

            # shape (..., Nframes,3,6)
            dp_triangulated_drtrf = \
                nps.matmult(dprefs_drt, drt_drtfr, drtfr_drtrf) / Nframes
        else:
            shape_leading = dp_triangulated_dv0.shape[:-2]
            dp_triangulated_drtrf = np.zeros(shape_leading + (Nframes,3,6), dtype=float)

        return \
            p_triangulated, \
            drtr1_drt1r, \
            drt01_drt0r, drt01_drtr1, \
            dvlocal0_dq0, dvlocal0_dintrinsics0, \
            dvlocal1_dq1, dvlocal1_dintrinsics1, \
            dv1_dr01, dv1_dvlocal1, \
            dp_triangulated_dv0, dp_triangulated_dv1, dp_triangulated_dt01, \
            dp_triangulated_drtrf



################# Sampling
if args.cache is None or args.cache == 'write':
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
                             None,
                             args.pixel_uncertainty_stdev_calibration,
                             object_width_n,
                             object_height_n,
                             object_spacing,
                             extrinsics_rt_fromref_true,
                             calobject_warp_true,
                             fixedframes,
                             testdir,
                             cull_left_of_center = args.cull_left_of_center)


    ( intrinsics_sampled,         \
      extrinsics_sampled_mounted, \
      frames_sampled,             \
      calobject_warp_sampled ) =  \
          calibration_sample( args.Nsamples, args.Ncameras, args.Nframes,
                              Nintrinsics,
                              optimization_inputs_baseline,
                              observations_true,
                              args.pixel_uncertainty_stdev_calibration,
                              fixedframes)

    if args.cache is not None and args.cache == 'write':
        with open(cache_file,"wb") as f:
            pickle.dump((optimization_inputs_baseline,
                         models_true,
                         models_baseline,
                         indices_frame_camintrinsics_camextrinsics,
                         lensmodel,
                         Nintrinsics,
                         imagersizes,
                         intrinsics_true,
                         extrinsics_true_mounted,
                         frames_true,
                         observations_true,
                         intrinsics_sampled,
                         extrinsics_sampled_mounted,
                         frames_sampled,
                         calobject_warp_sampled),
                        f)
else:
    with open(cache_file,"rb") as f:
        (optimization_inputs_baseline,
         models_true,
         models_baseline,
         indices_frame_camintrinsics_camextrinsics,
         lensmodel,
         Nintrinsics,
         imagersizes,
         intrinsics_true,
         extrinsics_true_mounted,
         frames_true,
         observations_true,
         intrinsics_sampled,
         extrinsics_sampled_mounted,
         frames_sampled,
         calobject_warp_sampled) = pickle.load(f)



Npoints = p_triangulated_true.shape[0]

Rt01_true = mrcal.compose_Rt(mrcal.Rt_from_rt(extrinsics_rt_fromref_true[0]),
                             mrcal.invert_Rt(mrcal.Rt_from_rt(extrinsics_rt_fromref_true[1])))
Rt10_true = mrcal.invert_Rt(Rt01_true)


# shape (Ncameras,Npoints,3)
p_triangulated_true_local = nps.cat( p_triangulated_true,
                                     mrcal.transform_point_Rt(Rt10_true, p_triangulated_true) )

# Pixel coords at the perfect intersection
# shape (Npoints,Ncameras,2)
q_true = nps.xchg( np.array([ mrcal.project(p_triangulated_true_local[i],
                                            lensmodel,
                                            intrinsics_true[i]) \
                            for i in range(len(intrinsics_true))]),
                 0,1)

# I have the perfect observation pixel coords. I triangulate them through my
# sampled calibration
p_triangulated_sampled = triangulate_nograd(intrinsics_sampled,
                                            extrinsics_sampled_mounted,
                                            frames_sampled, frames_true,
                                            q_true,
                                            lensmodel,
                                            stabilize_coords = args.stabilize_coords)



################ At baseline, with gradients
p_triangulated, \
drtr1_drt1r, \
drt01_drt0r, drt01_drtr1, \
dvlocal0_dq0, dvlocal0_dintrinsics0, \
dvlocal1_dq1, dvlocal1_dintrinsics1, \
dv1_dr01, dv1_dvlocal1, \
dp_triangulated_dv0, dp_triangulated_dv1, dp_triangulated_dt01, \
dp_triangulated_drtrf = \
    triangulate_grad([m.intrinsics()[1]         for m in models_baseline],
                     [m.extrinsics_rt_fromref() for m in models_baseline],
                     optimization_inputs_baseline['frames_rt_toref'], frames_true,
                     q_true,
                     lensmodel,
                     stabilize_coords = args.stabilize_coords)

testutils.confirm_equal(p_triangulated, p_triangulated_true,
                        worstcase = True,
                        eps = 1.0,
                        msg = "Re-optimized triangulation should be close to the reference. This checks the regularization bias")

# I have q0,i0           -> v0
#        q1,i1           -> vlocal1
#        vlocal1,r0r,r1r -> v1
#        r0r,r1r,t0r,t1r -> t01
#        v0,v1,t01       -> p_triangulated
ppacked,x,Jpacked,factorization = mrcal.optimizer_callback(**optimization_inputs_baseline)
Nstate = len(ppacked)

# I store dp_triangulated_dp initialy, without worrying about the "packed" part.
# I'll scale the thing when done to pack it
dp_triangulated_dpstate = np.zeros((Npoints,3,Nstate), dtype=float)

istate_i0 = mrcal.state_index_intrinsics(0, **optimization_inputs_baseline)
istate_i1 = mrcal.state_index_intrinsics(1, **optimization_inputs_baseline)

# I'm expecting the layout of a vanilla calibration problem, and I assume that
# camera0 is at the reference below. Here I confirm that this assumption is
# correct
icam_extrinsics0 = mrcal.corresponding_icam_extrinsics(0, **optimization_inputs_baseline)
icam_extrinsics1 = mrcal.corresponding_icam_extrinsics(1, **optimization_inputs_baseline)
if not (icam_extrinsics0 < 0 and icam_extrinsics1 == 0):
    raise Exception("Vanilla calibration problem expected, but got something else instead. Among others, _triangulate() assumes the triangulated result is in cam0, which is the same as the ref coord system")
istate_e1 = mrcal.state_index_extrinsics(icam_extrinsics1, **optimization_inputs_baseline)

istate_f0     = mrcal.state_index_frames(0, **optimization_inputs_baseline)
Nstate_frames = mrcal.num_states_frames(**optimization_inputs_baseline)

# dp_triangulated_di0 = dp_triangulated_dv0              dvlocal0_di0
# dp_triangulated_di1 = dp_triangulated_dv1 dv1_dvlocal1 dvlocal1_di1
nps.matmult( dp_triangulated_dv0,
             dvlocal0_dintrinsics0,
             out = dp_triangulated_dpstate[..., istate_i0:istate_i0+Nintrinsics])
nps.matmult( dp_triangulated_dv1,
             dv1_dvlocal1,
             dvlocal1_dintrinsics1,
             out = dp_triangulated_dpstate[..., istate_i1:istate_i1+Nintrinsics])

# dp_triangulated_de0 doesn't exist: assuming vanilla calibration problem, so
# there is no e0

# dp_triangulated_dr1r =
#   dp_triangulated_dv1 dv1_dr01 dr01_dr1r +
#   dp_triangulated_dt01 dt01_dr1r
dr01_drr1  = drt01_drtr1[:3,:3]
drr1_dr1r  = drtr1_drt1r[:3,:3]
dr01_dr1r  = nps.matmult(dr01_drr1, drr1_dr1r)

dt01_drtr1 = drt01_drtr1[3:,:]
dt01_dr1r  = nps.matmult(dt01_drtr1, drtr1_drt1r[:,:3])
dt01_dt1r  = nps.matmult(dt01_drtr1, drtr1_drt1r[:,3:])
nps.matmult( dp_triangulated_dv1,
             dv1_dr01,
             dr01_dr1r,
             out = dp_triangulated_dpstate[..., istate_e1:istate_e1+3])

dp_triangulated_dpstate[..., istate_e1:istate_e1+3] += \
    nps.matmult(dp_triangulated_dt01, dt01_dr1r)

# dp_triangulated_dt1r =
#   dp_triangulated_dt01 dt01_dt1r
nps.matmult( dp_triangulated_dt01,
             dt01_dt1r,
             out = dp_triangulated_dpstate[..., istate_e1+3:istate_e1+6])

# dp_triangulated_drtrf has shape (Npoints,Nframes,3,6). I reshape to (Npoints,3,Nframes*6)
dp_triangulated_dpstate[..., istate_f0:istate_f0+Nstate_frames] = \
    nps.clump(nps.xchg(dp_triangulated_drtrf,-2,-3), n=-2)

########## Gradient check
dp_triangulated_di0_empirical = grad(lambda i0: triangulate_nograd([i0, models_baseline[1].intrinsics()[1]],
                                                                   [m.extrinsics_rt_fromref() for m in models_baseline],
                                                                   optimization_inputs_baseline['frames_rt_toref'], frames_true,
                                                                   q_true,
                                                                   lensmodel,
                                                                   stabilize_coords=args.stabilize_coords),
                                     models_baseline[0].intrinsics()[1])
dp_triangulated_di1_empirical = grad(lambda i1: triangulate_nograd([models_baseline[0].intrinsics()[1],i1],
                                                                   [m.extrinsics_rt_fromref() for m in models_baseline],
                                                                   optimization_inputs_baseline['frames_rt_toref'], frames_true,
                                                                   q_true,
                                                                   lensmodel,
                                                                   stabilize_coords=args.stabilize_coords),
                                     models_baseline[1].intrinsics()[1])
dp_triangulated_de1_empirical = grad(lambda e1: triangulate_nograd([m.intrinsics()[1]         for m in models_baseline],
                                                                   [models_baseline[0].extrinsics_rt_fromref(),e1],
                                                                   optimization_inputs_baseline['frames_rt_toref'], frames_true,
                                                                   q_true,
                                                                   lensmodel,
                                                                   stabilize_coords=args.stabilize_coords),
                                     models_baseline[1].extrinsics_rt_fromref())
dp_triangulated_drtrf_empirical = grad(lambda rtrf: triangulate_nograd([m.intrinsics()[1]         for m in models_baseline],
                                                                       [m.extrinsics_rt_fromref() for m in models_baseline],
                                                                       rtrf, frames_true,
                                                                       q_true,
                                                                       lensmodel,
                                                                       stabilize_coords=args.stabilize_coords),
                                       optimization_inputs_baseline['frames_rt_toref'])

testutils.confirm_equal(dp_triangulated_dpstate[...,istate_i0:istate_i0+Nintrinsics],
                        dp_triangulated_di0_empirical,
                        relative = True,
                        worstcase = True,
                        eps = 0.05,
                        msg = "Gradient check: dp_triangulated_dpstate[intrinsics0]")
testutils.confirm_equal(dp_triangulated_dpstate[...,istate_i1:istate_i1+Nintrinsics],
                        dp_triangulated_di1_empirical,
                        relative = True,
                        worstcase = True,
                        eps = 0.05,
                        msg = "Gradient check: dp_triangulated_dpstate[intrinsics1]")
testutils.confirm_equal(dp_triangulated_dpstate[...,istate_e1:istate_e1+6],
                        dp_triangulated_de1_empirical,
                        relative = True,
                        worstcase = True,
                        eps = 1e-6,
                        msg = "Gradient check: dp_triangulated_dpstate[extrinsics1]")
testutils.confirm_equal(dp_triangulated_dpstate[...,istate_f0:istate_f0+Nstate_frames],
                        nps.clump(dp_triangulated_drtrf_empirical, n=-2),
                        relative = True,
                        worstcase = True,
                        eps = 0.05,
                        msg = "Gradient check: dp_triangulated_drtrf")

Nmeasurements_observations = mrcal.num_measurements_boards(**optimization_inputs_baseline)
if Nmeasurements_observations == mrcal.num_measurements(**optimization_inputs_baseline):
    # Note the special-case where I'm using all the observations
    Nmeasurements_observations = None

# I look at the two triangulated points together. This is a (6,) vector. And I
# pack the denominator by unpacking the numerator
dp0p1_triangulated_dppacked = copy.deepcopy(dp_triangulated_dpstate)
mrcal.unpack_state(dp0p1_triangulated_dppacked, **optimization_inputs_baseline)
dp0p1_triangulated_dppacked = nps.clump(dp0p1_triangulated_dppacked,n=2)

Var_p0p1_triangulated = \
    mrcal.model_analysis._projection_uncertainty_make_output(factorization, Jpacked,
                                                             dp0p1_triangulated_dppacked,
                                                             Nmeasurements_observations,
                                                             args.pixel_uncertainty_stdev_calibration,
                                                             what = 'covariance')


if not (args.explore or \
        args.make_documentation_plots is not None):
    testutils.finish()
    sys.exit()

import gnuplotlib as gp


if args.make_documentation_plots is not None:

    empirical_distributions_xz = \
        [ plot_args_points_and_covariance_ellipse(p_triangulated_sampled[:,ipt,(0,2)],
                                                  'Triangulation in moving cam0 coord system') \
          for ipt in range(Npoints) ]
    # Individual covariances
    Var_p_diagonal = [Var_p0p1_triangulated[ipt*3:ipt*3+3,ipt*3:ipt*3+3][(0,2),:][:,(0,2)] \
                      for ipt in range(Npoints)]
    max_sigma_points = np.array([ np.max(np.sqrt(np.linalg.eig(V)[0])) for V in Var_p_diagonal ])
    max_sigma = np.max(max_sigma_points)

    title_triangulation = 'Triangulation uncertainty due to calibration-time noise. Cameras at the origin. Equal scaling in both plots'
    title_covariance    = 'Covariance of the [p0,p1] vector. Note the low variance of the y coordinate and the non-zero correlation between the points'

    subplots = [ (*empirical_distributions_xz[ipt],
                  plot_arg_covariance_ellipse(p_triangulated[ipt][(0,2),],
                                              Var_p_diagonal[ipt],
                                              "predicted"),
                  dict( square = True,
                        _xrange = [p_triangulated[ipt,0] - max_sigma*3.,
                                   p_triangulated[ipt,0] + max_sigma*3.],
                        _yrange = [p_triangulated[ipt,2] - max_sigma*3.,
                                   p_triangulated[ipt,2] + max_sigma*3.] )
                  ) \
                 for ipt in range(Npoints) ]

    def makeplot(dohardcopy, processoptions_base):

        processoptions = copy.deepcopy(processoptions_base)

        if dohardcopy:
            processoptions['hardcopy'] = \
                f'{args.make_documentation_plots}--triangulation-uncertainty.{extension}'
        gp.plot( *subplots,
                 multiplot = f'title "{title_triangulation}" layout 1,2',
                 **processoptions )

        if dohardcopy:
            processoptions['hardcopy'] = \
                f'{args.make_documentation_plots}--p0-p1-magnitude-covariance.{extension}'
        processoptions['title'] = title_covariance
        gp.plotimage( np.abs(Var_p0p1_triangulated),
                      square = True,
                      **processoptions)



    if args.make_documentation_plots:
        for extension in ('pdf','svg','png','gp'):
            makeplot(dohardcopy = True,
                     processoptions_base = dict(wait      = False,
                                                terminal  = terminal[extension],
                                                _set      = extraset[extension]))
    else:
        makeplot(dohardcopy = False,
                 processoptions_base = dict(wait = True))

if args.explore:
    import IPython
    IPython.embed()

sys.exit()






r'''
extend to work with non-ref camera

Add pixel noise to look at uncertainty due to calibration AND run-time pixel
noise

fixedframes?

What kind of nice API do I want for this? How much can I reuse the existing
uncertainty code? Can/should I have a separate rotation-only/at-infinity path?
Should finalize the API after I implement the deltapose-propagated uncertainty

Gnuplot: ellipses should move correctly when pressing "7"


tests:

look at distribution due to

- intrinsics/extrinsics and/or pixel observations

- look at mean, variance
'''






import IPython
IPython.embed()
sys.exit()








diffp          = p[0] - p[1]
ddiffp_dp01 = nps.glue(  np.eye(3, dtype=float),
                        -np.eye(3, dtype=float),
                         axis = -1)
distancep      = nps.mag(diffp)
ddistancep_dp0 =  diffp / distancep
ddistancep_dp1 = -diffp / distancep

# I now have all the gradients and all the internal variances, so I can
# propagate everything. The big vector I want to propagate is
#
# - q: pixel noise
# - intrinsics01
# - extrinsics


# For now let's just do q

# I have 4 pixel observations. Let's say qij is the pixel observation of
# point i in camera j



def compute_var_p01(var_x):
    # shape (Npoints, 3, Npoints,Ncameras,2)
    # The trailing (Npoints,Ncameras,2) indexes the x
    dp_dx = np.zeros((2,3,2,2,2), dtype=float)

    # each has shape (3,2)
    dp0_dq00 = dp_dx[0,:,0,0,:]
    dp0_dq01 = dp_dx[0,:,0,1,:]
    dp1_dq10 = dp_dx[1,:,1,0,:]
    dp1_dq11 = dp_dx[1,:,1,1,:]

    # each has shape (Npoints,3,2)
    dv0_dq0 = dvlocal0_dq0
    dv1_dq1 = nps.matmult2(dv1_dvlocal1, dvlocal1_dq1)

    nps.matmult2(dp_dv0[0], dv0_dq0[0], out=dp0_dq00)
    nps.matmult2(dp_dv1[0], dv1_dq1[0], out=dp0_dq01)
    nps.matmult2(dp_dv0[1], dv0_dq0[1], out=dp1_dq10)
    nps.matmult2(dp_dv1[1], dv1_dq1[1], out=dp1_dq11)

    # shape (6,8)
    dpflat_dx = nps.clump(nps.clump(dp_dx, n=-3), n=2)
    return nps.matmult( dpflat_dx, var_x, nps.transpose(dpflat_dx))


def compute_var_range(p, var_p):
    # r = np.mag(p)
    # dr_dp = p/r
    # var(r) = dr_dp var_p dr_dpT
    #        = p var_p pT / norm2(p)
    return nps.matmult(p,var_p,nps.transpose(p))[0] / nps.norm2(p)


def compute_var_distancep_direct(var_x):
    # Distance between 2 3D points

    # I have 4 pixel observations. Let's say qij is the pixel observation of
    # point i in camera j

    ddistancep_dx = np.zeros((1,8), dtype=float)
    ddistancep_dq00 = ddistancep_dx[:,0:2]
    ddistancep_dq01 = ddistancep_dx[:,2:4]
    ddistancep_dq10 = ddistancep_dx[:,4:6]
    ddistancep_dq11 = ddistancep_dx[:,6:8]

    dv00_dq00,dv10_dq10 = dvlocal0_dq0
    dv01_dq01,dv11_dq11 = nps.matmult2(dv1_dvlocal1, dvlocal1_dq1)

    dp0_dv00,dp1_dv10 = dp_dv0
    dp0_dv01,dp1_dv11 = dp_dv1

    ddistancep_dv00 = nps.matmult( ddistancep_dp0, dp0_dv00 )
    ddistancep_dv01 = nps.matmult( ddistancep_dp0, dp0_dv01 )
    ddistancep_dv10 = nps.matmult( ddistancep_dp1, dp1_dv10 )
    ddistancep_dv11 = nps.matmult( ddistancep_dp1, dp1_dv11 )

    nps.matmult2(ddistancep_dv00, dv00_dq00, out=ddistancep_dq00)
    nps.matmult2(ddistancep_dv01, dv01_dq01, out=ddistancep_dq01)
    nps.matmult2(ddistancep_dv10, dv10_dq10, out=ddistancep_dq10)
    nps.matmult2(ddistancep_dv11, dv11_dq11, out=ddistancep_dq11)

    return nps.matmult( ddistancep_dx, var_x, nps.transpose(ddistancep_dx)).ravel()[0]


def compute_var_distancep_from_var_p(var_p):

    ddistancep_dp = nps.glue( ddistancep_dp0, ddistancep_dp1, axis=-1)
    return nps.matmult( ddistancep_dp, var_p, nps.transpose(ddistancep_dp)).ravel()[0]



# Let's say the pixel observations are all independent; this is not
# obviously true. Order in x: q00 q01 q10 q11
Nx = 8
var_x_independent = np.diagflat( (pixel_uncertainty_stdev*pixel_uncertainty_stdev,) * Nx )

var_x = var_x_independent.copy()
var_x[0,2] = var_x[2,0] = pixel_uncertainty_stdev*pixel_uncertainty_stdev*0.9
var_x[1,3] = var_x[3,1] = pixel_uncertainty_stdev*pixel_uncertainty_stdev*0.9
var_x[4,6] = var_x[6,4] = pixel_uncertainty_stdev*pixel_uncertainty_stdev*0.9
var_x[5,7] = var_x[7,5] = pixel_uncertainty_stdev*pixel_uncertainty_stdev*0.9

var_distancep             = compute_var_distancep_direct(var_x)
var_distancep_independent = compute_var_distancep_direct(var_x_independent)
var_p                     = compute_var_p01(var_x)
var_p_independent         = compute_var_p01(var_x_independent)

var_r0             = compute_var_range(pref[0], var_p[:3,:3])
var_r0_independent = compute_var_range(pref[0], var_p_independent[:3,:3])

var_diffp = nps.matmult(ddiffp_dp01, var_p, nps.transpose(ddiffp_dp01))



if np.abs(compute_var_distancep_from_var_p(var_p) - var_distancep) > 1e-6:
    raise Exception("Var(distancep) should identical whether you compute it from Var(p) or not")



# Let's actually apply the noise to compute var(distancep) empirically to compare
# against the var(distancep) prediction I just computed
# shape (Nsamples,Npoints,Ncameras,2)
dq = \
    np.random.multivariate_normal( mean = np.zeros((Nx,),),
                                   cov  = var_x,
                                   size = args.Nsamples ).reshape(args.Nsamples,2,2,2)


vlocal0 = mrcal.unproject(qref[:,0] + dq[:,:,0,:], *models[0].intrinsics())
vlocal1 = mrcal.unproject(qref[:,1] + dq[:,:,1,:], *models[1].intrinsics())
v0      = vlocal0
v1      = mrcal.rotate_point_R(Rt01[:3,:], vlocal1)
p       = mrcal.triangulate_leecivera_mid2(v0, v1, Rt01[3,:])

distancep     = nps.mag(p[:,0,:] - p[:,1,:])
distancep_ref = nps.mag(pref[0] - pref[1])

r0            = nps.mag(p[:,0,:])
r0_ref        = nps.mag(pref[0])

binwidth = 0.4*10

equation_distancep_observed_gaussian = \
    mrcal.fitted_gaussian_equation(x        = distancep,
                                   binwidth = binwidth,
                                   legend   = "Idealized gaussian fit to data")
equation_distancep_predicted_gaussian = \
    mrcal.fitted_gaussian_equation(mean     = distancep_ref,
                                   sigma    = np.sqrt(var_distancep),
                                   N        = len(distancep),
                                   binwidth = binwidth,
                                   legend   = "Predicted")
equation_distancep_predicted_independent_gaussian = \
    mrcal.fitted_gaussian_equation(mean     = distancep_ref,
                                   sigma    = np.sqrt(var_distancep_independent),
                                   N        = len(distancep),
                                   binwidth = binwidth,
                                   legend   = "Predicted, assuming independent noise")

equation_r0_observed_gaussian = \
    mrcal.fitted_gaussian_equation(x        = r0,
                                   binwidth = binwidth,
                                   legend   = "Idealized gaussian fit to data")
equation_r0_predicted_gaussian = \
    mrcal.fitted_gaussian_equation(mean     = r0_ref,
                                   sigma    = np.sqrt(var_r0),
                                   N        = len(r0),
                                   binwidth = binwidth,
                                   legend   = "Predicted")
equation_r0_predicted_independent_gaussian = \
    mrcal.fitted_gaussian_equation(mean     = r0_ref,
                                   sigma    = np.sqrt(var_r0_independent),
                                   N        = len(r0),
                                   binwidth = binwidth,
                                   legend   = "Predicted, assuming independent noise")

gp.plot(distancep,
        histogram       = True,
        binwidth        = binwidth,
        equation_above  = (equation_distancep_predicted_independent_gaussian,
                           equation_distancep_predicted_gaussian,
                           equation_distancep_observed_gaussian),
        xlabel          = "Distance between points",
        ylabel          = "Frequency",
        title           = f"Triangulated distance between points: sensitivity to pixel noise. Predicted stdev: {np.sqrt(var_distancep):.0f}m",
        _set            = 'samples 1000',
        # hardcopy = '/tmp/distance-between.pdf',
        wait=1)



# I look at the xz uncertainty because y is very low. I just look at p[0]
V = var_p[:3,:3]
if np.min((V[0,0],V[2,2]))/V[1,1] < 10:
    raise Exception("Assumption that var(y) << var(xz) is false. I want the worst-case ratio to be >10")

V = V[(0,2),:][:,(0,2)]

ellipse = \
    plot_arg_covariance_ellipse(pref[0,(0,2)], V,
                                'Observed point. Var(p_xz)')

gp.plot( ( nps.glue( np.zeros((2,),),
                     Rt01[3,(0,2)],
                     axis=-2),
           dict(legend    = 'cameras',
                _with     = 'points pt 8 ps 1',
                tuplesize = -2)),
         ellipse,
         square = True,
         xlabel= 'x (m)',
         ylabel= 'y (m)',
         title = 'Top-down view of the world',
         # hardcopy = '/tmp/world.pdf'
         wait = True,
        )

gp.plot( ellipse,
         square = True,
         xlabel= 'x (m)',
         ylabel= 'y (m)',
         title = 'Top-down view of the world; ',
         # hardcopy = '/tmp/uncertainty.pdf',
         wait = True,
        )

gp.plot(r0,
        histogram       = True,
        binwidth        = binwidth,
        equation_above  = (equation_r0_predicted_independent_gaussian,
                           equation_r0_predicted_gaussian,
                           equation_r0_observed_gaussian),
        xlabel          = "Range to triangulated point",
        ylabel          = "Frequency",
        title           = f"Triangulated distance to the observation at 1600m: sensitivity to pixel noise. Predicted stdev: {np.sqrt(var_r0):.0f}m",
        # hardcopy = '/tmp/range0.pdf',
        _set            = 'samples 1000',
        wait = True,
        )



testutils.finish()
