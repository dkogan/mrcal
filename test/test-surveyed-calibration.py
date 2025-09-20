#!/usr/bin/env python3

r'''Surveyed camera-calibration test

I observe, with noise, a number of fixed, known-pose chessboards from a single
camera. And I make sure that I can more or less compute the camera intrinsics
and extrinsics.

This is a monocular solve. Since the chessboards aren't allowed to move,
multiple cameras in one solves wouldn't be interacting with each other, and
there's no reason to have more than one

'''


import sys
import argparse
import re
import os

def parse_args():

    parser = \
        argparse.ArgumentParser(description = __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--write-model',
                        type=str,
                        help='''If given, we write the resulting model to disk
                        for further analysis. The filename is given in this
                        argument''')
    parser.add_argument('--range-board',
                        type=float,
                        default = 10,
                        help='''Distance from the camera to the center of each
                        chessboard. If omitted, we default to 10''')
    parser.add_argument('--range-board-center',
                        type=float,
                        help='''Distance from the camera to the center of the
                        MIDDLE chessboard. If omitted, we use --range-board * 2.
                        Everything at one range causes poor calibrations, and I
                        don't want that to be the default case here''')
    parser.add_argument('--oversample',
                        type=int,
                        default = 1,
                        help='''How many times we observe the stationary scene.
                        Observing multiple times gives us more samples of the
                        noise, resulting in lower uncertainties''')
    parser.add_argument('--distance',
                        type=float,
                        help='''distance for uncertainty computations. If
                        omitted, we use infinity''')
    parser.add_argument('--seed-rng',
                        type=int,
                        default=0,
                        help='''Value to seed the rng with''')
    parser.add_argument('--do-sample',
                        action='store_true',
                        help='''By default we don't run the time-intensive
                        samples of the calibration solves. This runs a very
                        limited set of tests, and exits. To perform the full set
                        of tests, pass --do-sample''')
    parser.add_argument('--only-report-uncertainty',
                        action='store_true',
                        help='''If given, I ONLY report the uncertainty in projection and errz''')
    parser.add_argument('--Nsamples',
                        type=int,
                        default=500,
                        help='''How many random samples to evaluate''')
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

    args = parser.parse_args()

    if args.range_board_center is None:
        args.range_board_center = args.range_board * 2.
    return args


args = parse_args()








import numpy as np
import numpysane as nps
import os

testdir = os.path.dirname(os.path.realpath(__file__))

# I import the LOCAL mrcal since that's what I'm testing
sys.path[:0] = f"{testdir}/..",
import mrcal
import testutils

from test_calibration_helpers import sample_dqref,calibration_sample
import copy

# I Reduce FOV to make it clear that you need data from different ranges
focal_length_scale_from_true = 10.

object_spacing          = 0.03
object_width_n          = 8
object_height_n         = 8
pixel_uncertainty_stdev = 0.5

# I have a 3x3 grid of chessboards
NboardgridX,NboardgridY = 3,1

# We try to recover this in the calibration process
# shape (6,)
rt_cam_ref_true   = np.array((-0.04,  0.05,  -0.1,     1.2, -0.1,  0.1),)

random_radius__r_cam_board_true = 0.1
random_radius__t_cam_board_true = 1.0e-1





############# Plot setup. Very similar to test-projection-uncertainty.py
if args.make_documentation_plots is not None:
    def shorter_terminal(t):
        # Adjust the terminal string to be less tall. Reduces wasted space in
        # some of the plots
        m = re.match("(.*)( size.*?,)([0-9.]+)(.*?)$", t)
        if m is None: return t
        return m.group(1) + m.group(2) + str(float(m.group(3))*0.8) + m.group(4)

    import gnuplotlib as gp

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



# I want the RNG to be deterministic
np.random.seed(args.seed_rng)

############# Set up my world, and compute all the perfect positions, pixel
############# observations of everything
model_true      = mrcal.cameramodel(f"{testdir}/data/cam0.opencv8.cameramodel")
imagersize_true = model_true.imagersize()
W,H             = imagersize_true

# I have opencv8 model_true, but I truncate to opencv4 to keep this simple and
# fast
lensmodel = 'LENSMODEL_OPENCV4'
Nintrinsics = mrcal.lensmodel_num_params(lensmodel)
intrinsics_data_true = model_true.intrinsics()[1][:Nintrinsics]
intrinsics_data_true[:2] *= focal_length_scale_from_true
model_true.intrinsics( (lensmodel, intrinsics_data_true), )


model_true.rt_cam_ref(rt_cam_ref_true)

if False:
    model_true.write("/tmp/true.cameramodel")
    print("wrote /tmp/true.cameramodel")

# Test point. On the left, centered vertically
q_query = np.array((W/4, H/2))

# I want to space out the chessboards across the whole imager. To make this
# simpler I define it in terms of the "boardcentered" coordinate system, whose
# origin is at the center of the chessboard, not in a corner
board_center                  = \
    np.array(( (object_width_n -1)*object_spacing/2.,
               (object_height_n-1)*object_spacing/2.,
               0 ))
rt_boardcentered_board_true     = mrcal.identity_rt()
rt_boardcentered_board_true[3:] = -board_center


# The chessboard centers are arranged in an even grid on the imager
#
# shape (NboardgridY,NboardgridX,2); each row is (qx,qy): the center of each
# chessboard
q_center = \
    nps.mv( \
        nps.cat(*np.meshgrid(np.linspace(0.5, NboardgridX-0.5, NboardgridX, endpoint=True)/NboardgridX * W,
                             np.linspace(0.5, NboardgridY-0.5, NboardgridY, endpoint=True)/NboardgridY * H)),
        0, -1)
v_center = mrcal.unproject(np.ascontiguousarray(q_center),
                           *model_true.intrinsics(),
                           normalize = True)

# shape (NboardgridY,NboardgridX,6)
rt_cam_boardcentered_true = np.zeros(v_center.shape[:-1] + (6,), dtype=float)
rt_cam_boardcentered_true[...,3:] = v_center*args.range_board

if args.range_board_center is not None:
    rt_cam_boardcentered_true[NboardgridY//2, NboardgridX//2,3:] = \
        v_center[NboardgridY//2, NboardgridX//2,:]*args.range_board_center

# shape (NboardgridY,NboardgridX,6)
rt_cam_board_true = mrcal.compose_rt(rt_cam_boardcentered_true,
                                     rt_boardcentered_board_true)
rt_cam_board_true[...,:3] += (np.random.rand(*rt_cam_board_true[...,:3].shape)*2. - 1.) * random_radius__r_cam_board_true
rt_cam_board_true[...,3:] += (np.random.rand(*rt_cam_board_true[...,3:].shape)*2. - 1.) * random_radius__t_cam_board_true

# Nboards = NboardgridX*NboardgridY
# shape (Nboards,6)
rt_cam_board_true = nps.clump(rt_cam_board_true, n=2)


rt_ref_board_true = mrcal.compose_rt( mrcal.invert_rt(rt_cam_ref_true),
                                      rt_cam_board_true)

# We assume that the chessboard poses rt_ref_board_true, rt_cam_board_true were
# measured. Perfectly. This it the ground truth AND we have it available in the
# calibration



# shape (Nh,Nw,3)
cal_object = mrcal.ref_calibration_object(object_width_n,
                                          object_height_n,
                                          object_spacing)
# shape (Nframes,Nh,Nw,3)
pcam_calobjects = \
    mrcal.transform_point_rt(nps.mv(rt_cam_board_true,-2,-4), cal_object)

# shape (N,3)
pcam_true = nps.clump(pcam_calobjects, n=3)
pref_true = mrcal.transform_point_rt(mrcal.invert_rt(rt_cam_ref_true),
                                     pcam_true)
# shape (N,2)
q_true    = mrcal.project(pcam_true, lensmodel, intrinsics_data_true)

Npoints = q_true.shape[0]

if False:
    # show the angles off the optical axis
    import gnuplotlib as gp
    p = pcam_true / nps.dummy(nps.mag(pcam_true),-1)
    c = p[...,2]
    th = np.arccos(c).ravel() * 180./np.pi
    gp.plot(th, yrange=(45,90))

    import IPython
    IPython.embed()
    sys.exit()



# I have perfect observations in q_true. I corrupt them by noise weight has
# shape (N,),
weight01 = (np.random.rand(*q_true.shape[:-1]) + 1.) / 2. # in [0,1]
weight0 = 0.2
weight1 = 1.0
weight = weight0 + (weight1-weight0)*weight01

# out-of-bounds observations are outliers
weight[(q_true[:,0] < 0) + \
       (q_true[:,1] < 0) + \
       (q_true[:,0] > W-1) + \
       (q_true[:,1] > H-1)] = -1.

############# Now I pretend that the noisy observations are all I got, and I run
############# a calibration from those
# shape (Noversample, N, 3)
observations_point = np.zeros((args.oversample,Npoints,3), dtype=float)
# write the same perfect data into each oversampled slice
observations_point += nps.glue(q_true,
                               nps.dummy(weight,-1),
                               axis=-1)
# shape (Noversample*N, 3)
observations_point = nps.clump(observations_point,n=2)

# Dense observations. The cameras see all the boards
indices_point_camintrinsics_camextrinsics = np.zeros( (args.oversample,Npoints, 3), dtype=np.int32)
indices_point_camintrinsics_camextrinsics[...,0] += np.arange(Npoints)
indices_point_camintrinsics_camextrinsics = nps.clump(indices_point_camintrinsics_camextrinsics,n=2)

if False:
    # Compute the seed using the seeding algorithm
    focal_estimate = 2000 * focal_length_scale_from_true
    cxy = np.array(( (W-1.)/2,
                     (H-1.)/2, ), )
    intrinsics_core_estimate = \
        ('LENSMODEL_STEREOGRAPHIC',
         np.array((focal_estimate,
                   focal_estimate,
                   *cxy
                   )))
    # I select data just at the center for seeding. Eventually should move this to
    # the general logic in calibration.py
    i = nps.norm2( observations_point[:Npoints,:2] - cxy ) < 1000**2.
    observations_point_center = np.array(observations_point[:Npoints])
    observations_point_center[~i,2] = -1.
    Rt_camera_ref_estimate = \
        mrcal.calibration._estimate_camera_pose_from_point_observations( \
            indices_point_camintrinsics_camextrinsics[:Npoints,:],
            observations_point_center,
            (intrinsics_core_estimate,),
            pref_true,
            icam_intrinsics = 0)

    rt_camera_ref_estimate = mrcal.rt_from_Rt(Rt_camera_ref_estimate)

    intrinsics_data = np.zeros((1,Nintrinsics), dtype=float)
    intrinsics_data[:,:4] = intrinsics_core_estimate[1]
    intrinsics_data[:,4:] = np.random.random( (1, intrinsics_data.shape[1]-4) ) * 1e-6

else:
    # Use a nearby value to the true seed
    rt_camera_ref_estimate = nps.atleast_dims(np.array(rt_cam_ref_true), -2)
    rt_camera_ref_estimate[..., :3] += np.random.randn(3) * 1e-2 # r
    rt_camera_ref_estimate[..., 3:] += np.random.randn(3) * 0.1  # t

    intrinsics_data = nps.atleast_dims(np.array(intrinsics_data_true), -2)
    intrinsics_data += np.random.randn(*intrinsics_data.shape) * 1e-2



if args.make_documentation_plots is not None:
    def makeplot(**plotoptions):
        mrcal.show_geometry( (rt_camera_ref_estimate,),
                             points      = pref_true, # known. fixed. perfect.
                             show_points = True,
                             title       = 'Surveyed calibration: nominal geometry',
                             **plotoptions)

    if args.make_documentation_plots:
        for extension in ('pdf','svg','png','gp'):
            makeplot(wait     = False,
                     terminal = shorter_terminal(terminal[extension]),
                     _set     = extraset[extension],
                     hardcopy = f'{args.make_documentation_plots}/geometry.{extension}')
    else:
        makeplot(wait = True)

if args.make_documentation_plots is not None:
    def makeplot(**plotoptions):
        gp.plot( observations_point[:,:2],
                     _with     = f'points ps {pointscale[extension]}',
                     tuplesize = -2,
                     square=1,
                 _xrange     = f'0:{W-1}',
                 _yrange     = f'{H-1}:0',
                 title       = 'Surveyed calibration: observations',
                     **plotoptions)

    if args.make_documentation_plots:
        for extension in ('pdf','svg','png','gp'):
            makeplot(wait     = False,
                     terminal = terminal[extension],
                     _set     = extraset[extension],
                     hardcopy = f'{args.make_documentation_plots}/observations.{extension}')
    else:
        makeplot(wait = True)


optimization_inputs_baseline                        = \
    dict( lensmodel                                 = lensmodel,
          intrinsics                                = intrinsics_data,
          rt_cam_ref                                = rt_camera_ref_estimate,
          rt_ref_frame                              = None,
          points                                    = pref_true, # known. fixed. perfect.
          observations_board                        = None,
          indices_frame_camintrinsics_camextrinsics = None,
          observations_point                        = observations_point,
          indices_point_camintrinsics_camextrinsics = indices_point_camintrinsics_camextrinsics,
          calobject_warp                            = None,
          imagersizes                               = nps.atleast_dims(imagersize_true, -2),
          point_min_range                           = 0.01,
          point_max_range                           = 100.,
          verbose                                   = False,
          do_apply_outlier_rejection                = False,
          do_apply_regularization                   = True,
          do_optimize_calobject_warp                = False,
          do_optimize_frames                        = False)

Nstate = mrcal.num_states(**optimization_inputs_baseline)


def optimize(optimization_inputs):
    optimization_inputs['do_optimize_intrinsics_core']        = False
    optimization_inputs['do_optimize_intrinsics_distortions'] = False
    optimization_inputs['do_optimize_extrinsics']             = True
    mrcal.optimize(**optimization_inputs)

    optimization_inputs['do_optimize_intrinsics_core']        = True
    optimization_inputs['do_optimize_intrinsics_distortions'] = True
    optimization_inputs['do_optimize_extrinsics']             = True
    mrcal.optimize(**optimization_inputs)

# Take one solve sample
optimization_inputs = copy.deepcopy(optimization_inputs_baseline)
_,optimization_inputs['observations_point'] = \
    sample_dqref(optimization_inputs['observations_point'], pixel_uncertainty_stdev)
optimize(optimization_inputs)

# Grab the measurements. measurements_point() reports ONLY the point
# reprojection errors: no board observations, no regularization. Also, no
# outliers
rmserr_point = np.std(mrcal.measurements_point(optimization_inputs).ravel())

############# Calibration computed. Now I see how well I did
model_solved = \
    mrcal.cameramodel( optimization_inputs = optimization_inputs,
                       icam_intrinsics     = 0 )

Rt_extrinsics_err = \
    mrcal.compose_Rt( model_solved.Rt_cam_ref(),
                      model_true  .Rt_ref_cam() )

v_query_cam = mrcal.unproject(q_query, *model_solved.intrinsics(), normalize = True)
if args.distance is not None:
    p_query_cam = v_query_cam * args.distance
else:
    p_query_cam = v_query_cam * 1e5 # large distance
p_query_ref = mrcal.transform_point_Rt(model_solved.Rt_ref_cam(),p_query_cam)

def get_variances():

    icam = 0
    i_state = mrcal.state_index_extrinsics(icam, **optimization_inputs)
    rt_extrinsics_err,                    \
    drt_extrinsics_err_drtsolved,         \
    _                                   = \
        mrcal.compose_rt( model_solved.rt_cam_ref(),
                          model_solved.rt_ref_cam(),
                          get_gradients = True)
    derrz_db = np.zeros( (1,Nstate), dtype=float)
    derrz_db[...,i_state:i_state+6] = drt_extrinsics_err_drtsolved[5:,:]
    Var_errz = mrcal.model_analysis._propagate_calibration_uncertainty( \
                 'covariance',
                 dF_db = derrz_db,
                 optimization_inputs = optimization_inputs)
    Var_errz = Var_errz[0,0]

    if args.distance is not None:
        Var_q = \
            mrcal.projection_uncertainty( p_query_cam, model_solved,
                                          atinfinity = False,
                                          what = 'covariance' )
    else:
        Var_q = \
            mrcal.projection_uncertainty( v_query_cam, model_solved,
                                          atinfinity = True,
                                          what       = 'covariance' )

    return Var_errz,Var_q


if args.only_report_uncertainty:

    Var_errz,Var_q = get_variances()
    print("# z-bulk z-center stdev(q) stdev(errz)")

    # using the 'rms-stdev' expression from _propagate_calibration_uncertainty()
    print(f"{args.range_board} {args.range_board_center} {np.sqrt(nps.trace(Var_q)/2.):.4f} {np.sqrt(Var_errz):.4f}")
    sys.exit()



# verify problem layout
testutils.confirm_equal( Nstate,
                         Nintrinsics + 6,
                         msg="num_states()")
testutils.confirm_equal( mrcal.num_states_intrinsics(**optimization_inputs),
                         Nintrinsics,
                         msg="num_states_intrinsics()")
testutils.confirm_equal( mrcal.num_intrinsics_optimization_params(**optimization_inputs),
                         Nintrinsics,
                         msg="num_intrinsics_optimization_params()")
testutils.confirm_equal( mrcal.num_states_extrinsics(**optimization_inputs),
                         6,
                         msg="num_states_extrinsics()")
testutils.confirm_equal( mrcal.num_states_frames(**optimization_inputs),
                         0,
                         msg="num_states_frames()")
testutils.confirm_equal( mrcal.num_states_points(**optimization_inputs),
                         0,
                         msg="num_states_points()")
testutils.confirm_equal( mrcal.num_states_calobject_warp(**optimization_inputs),
                         0,
                         msg="num_states_calobject_warp()")
testutils.confirm_equal( mrcal.num_measurements_boards(**optimization_inputs),
                         0,
                         msg="num_measurements_boards()")
testutils.confirm_equal( mrcal.num_measurements_points(**optimization_inputs),
                         Npoints*args.oversample*2,
                         msg="num_measurements_points()")
testutils.confirm_equal( mrcal.num_measurements_regularization(**optimization_inputs),
                         6,
                         msg="num_measurements_regularization()")
testutils.confirm_equal( mrcal.state_index_intrinsics(0, **optimization_inputs),
                         0,
                         msg="state_index_intrinsics()")
testutils.confirm_equal( mrcal.state_index_extrinsics(0, **optimization_inputs),
                         8,
                         msg="state_index_extrinsics()")
testutils.confirm_equal( mrcal.measurement_index_points(2, **optimization_inputs),
                         2*2,
                         msg="measurement_index_points()")
testutils.confirm_equal( mrcal.measurement_index_regularization(**optimization_inputs),
                         2*args.oversample*Npoints,
                         msg="measurement_index_regularization()")

if args.write_model:
    model_solved.write(args.write_model)
    print(f"Wrote '{args.write_model}'")

testutils.confirm_equal(rmserr_point, 0,
                        eps = 2.5,
                        msg = "Converged to a low RMS error")

# I expect the fitted error (rmserr_point) to be a bit lower than
# pixel_uncertainty_stdev because this solve is going to overfit: I don't have
# enough data to uniquely define this model. And this isn't even a flexible
# splined model!
testutils.confirm_equal( rmserr_point,
                         pixel_uncertainty_stdev - pixel_uncertainty_stdev*0.12,
                         eps = pixel_uncertainty_stdev*0.12,
                         msg = "Residual have the expected distribution" )

testutils.confirm_equal( nps.mag(Rt_extrinsics_err[3,:]),
                         0.0,
                         eps = 0.05,
                         msg = "Recovered extrinsic translation")

testutils.confirm_equal( (np.trace(Rt_extrinsics_err[:3,:]) - 1) / 2.,
                         1.0,
                         eps = np.cos(1. * np.pi/180.0), # 1 deg
                         msg = "Recovered extrinsic rotation")



# Checking the intrinsics. Each intrinsics vector encodes an implicit
# transformation. I compute and apply this transformation when making my
# intrinsics comparisons. I make sure that within some distance of the pixel
# center, the projections match up to within some number of pixels
Nw = 60
def projection_diff(models, max_dist_from_center):
    lensmodels      = [model.intrinsics()[0] for model in models]
    intrinsics_data = [model.intrinsics()[1] for model in models]
    imagersizes     = [model.imagersize()    for model in models]

    # v  shape (...,Ncameras,Nheight,Nwidth,...)
    # q0 shape (...,         Nheight,Nwidth,...)
    v,q0 = \
        mrcal.sample_imager_unproject(Nw,None,
                                      *imagersizes[0],
                                      lensmodels, intrinsics_data,
                                      normalize = True)

    focus_center = None
    focus_radius = -1
    if focus_center is None: focus_center = ((W-1.)/2., (H-1.)/2.)
    if focus_radius < 0:     focus_radius = min(W,H)/6.


    implied_Rt10 = \
        mrcal.implied_Rt10__from_unprojections(q0,
                                               v[0,...], v[1,...],
                                               focus_center = focus_center,
                                               focus_radius = focus_radius)

    q1 = mrcal.project( mrcal.transform_point_Rt(implied_Rt10,
                                                 v[0,...]),
                       lensmodels[1], intrinsics_data[1])
    diff = nps.mag(q1 - q0)

    # zero-out everything too far from the center
    center = (imagersizes[0] - 1.) / 2.
    diff[ nps.norm2(q0 - center) > max_dist_from_center*max_dist_from_center ] = 0

    if False:
        import gnuplotlib as gp
        gp.plot(diff,
                ascii = True,
                using = mrcal.imagergrid_using(imagersizes[0], Nw),
                square=1, _with='image', tuplesize=3, hardcopy='/tmp/yes.gp', cbmax=3)

    return diff


diff = projection_diff( (model_true, model_solved), 800)
testutils.confirm_equal(diff, 0,
                        worstcase = True,
                        eps = 6.,
                        msg = "Recovered intrinsics")


# I test make_perfect_observations(). Doing it here is easy; doing it elsewhere
# it much more work
if True:
    optimization_inputs_perfect = copy.deepcopy(optimization_inputs)

    mrcal.make_perfect_observations(optimization_inputs_perfect,
                                    observed_pixel_uncertainty=0)
    x = mrcal.optimizer_callback(**optimization_inputs_perfect,
                                 no_jacobian      = True,
                                 no_factorization = True)[1]

    Nmeas = mrcal.num_measurements_points(**optimization_inputs_perfect)
    if Nmeas > 0:
        i_meas0 = mrcal.measurement_index_points(0, **optimization_inputs_perfect)
        testutils.confirm_equal( x[i_meas0:i_meas0+Nmeas],
                                 0,
                                 worstcase = True,
                                 eps = 1e-8,
                                 msg = f"make_perfect_observations() works for points")
    else:
        testutils.confirm( False,
                           msg = f"Nmeasurements_points <= 0")





if not args.do_sample:
    testutils.finish()
    sys.exit()


#### We're going to computed a bunch of noisy calibrations, and gather
#### statistics. I look at:
#### - the projection of one point
#### - the error in the z coordinate of the camera position
####
#### For each one I compute the predicted uncertainty and the empirical one,
#### from the samples, and make sure they match
( intrinsics_sampled,         \
  rt_cam_ref_sampled_mounted, \
  frames_sampled,             \
  points_sampled,             \
  calobject_warp_sampled,     \
  q_noise_board_sampled,      \
  q_noise_point_sampled,      \
  b_sampled,                  \
  optimization_inputs_sampled ) = \
calibration_sample(args.Nsamples,
                   optimization_inputs_baseline,
                   pixel_uncertainty_stdev,
                   fixedframes       = True,
                   function_optimize = optimize)


Var_errz,Var_q = get_variances()

####### check errz
if 1:

    rt_extrinsics_sampled_err = \
        mrcal.compose_rt( rt_cam_ref_sampled_mounted,
                          model_solved.rt_ref_cam())

    errz_sampled = rt_extrinsics_sampled_err[:,0,5:]

    testutils.confirm_equal( np.mean(errz_sampled),
                             0,
                             eps = 0.01,
                             msg = f"errz distribution has mean=0")

    testutils.confirm_equal( np.std(errz_sampled),
                             np.sqrt(Var_errz),
                             eps=0.02,
                             relative = True,
                             msg = f"Predicted Var(errz) correct")


    if args.make_documentation_plots is not None:

        errz_sampled = rt_extrinsics_sampled_err[...,0,5]
        binwidth = 0.01
        equation_observed = \
            mrcal.fitted_gaussian_equation(x = errz_sampled,
                                           binwidth = binwidth,
                                           legend   = f'Observed in simulation')
        equation_predicted = \
            mrcal.fitted_gaussian_equation(mean     = np.mean(errz_sampled),
                                           sigma    = np.sqrt(Var_errz),
                                           N        = len(errz_sampled),
                                           binwidth = binwidth,
                                           legend   = "Predicted by mrcal")

        def makeplot(**plotoptions):
            gp.plot(errz_sampled,
                    histogram = True,
                    binwidth = binwidth,
                    equation_above = (equation_observed,
                                      equation_predicted),
                    xlabel = "Error in z (m)",
                    ylabel = "Count",
                    title  = "Surveyed calibration: distribution of error-in-z due to calibration-time noise",
                    **plotoptions)

        if args.make_documentation_plots:
            for extension in ('pdf','svg','png','gp'):
                makeplot(wait     = False,
                         terminal = terminal[extension],
                         _set     = extraset[extension],
                         hardcopy = f'{args.make_documentation_plots}/var-errz.{extension}')

        else:
            makeplot(wait = True)


####### check projection of a point
if 1:

    if 1:
        # Code check. Computing this directly should mimic the
        # mrcal.projection_uncertainty() result exactly since it should be 100% the
        # same computation.
        #
        # EXCEPT when looking at infinity. The projection_uncertainty() path
        # uses atinfinity=True, so it ignores translations completely. THIS path
        # uses a high distance, so the result will be close, but not identical.
        # I use a sloppier bound in that case
        p,dp_drt_cam_ref,_ = \
            mrcal.transform_point_rt(model_solved.rt_cam_ref(),
                                     p_query_ref,
                                     get_gradients = True)
        q,dq_dp,dq_di = \
            mrcal.project( p,
                           model_solved.intrinsics()[0],
                           model_solved.intrinsics()[1],
                           get_gradients = True)
        dq_drt_cam_ref = nps.matmult(dq_dp, dp_drt_cam_ref)

        dq_db = np.zeros( (2,Nstate), dtype=float)

        icam = 0
        i_state = mrcal.state_index_extrinsics(icam, **optimization_inputs)
        dq_db[...,i_state:i_state+6] = dq_drt_cam_ref
        i_state = mrcal.state_index_intrinsics(icam, **optimization_inputs)
        dq_db[...,i_state:i_state+Nintrinsics] = dq_di

        Var_q2 = mrcal.model_analysis._propagate_calibration_uncertainty( \
                     'covariance',
                     dF_db = dq_db,
                     optimization_inputs = optimization_inputs)
        testutils.confirm_equal(Var_q, Var_q2,
                                relative=True,
                                worstcase=True,
                                eps=1e-6 if args.distance is not None else 1e-2,
                                msg="projection_uncertainty() should do the same thing s caling _propagate_calibration_uncertainty() directly")

    p_query_cam_sampled = \
        mrcal.transform_point_rt(rt_cam_ref_sampled_mounted[:,0,:],
                                 p_query_ref)
    q_query_sampled = \
        mrcal.project( p_query_cam_sampled,
                       model_solved.intrinsics()[0],
                       intrinsics_sampled[:,0,:] )

    q_query_sampled_mean = np.mean(q_query_sampled, axis=0)

    # shape (2,2)
    Var_q_observed = np.mean( nps.outer(q_query_sampled-q_query_sampled_mean,
                                        q_query_sampled-q_query_sampled_mean),
                              axis=0 )

    worst_direction_stdev_observed  = mrcal.worst_direction_stdev(Var_q_observed)
    worst_direction_stdev_predicted = mrcal.worst_direction_stdev(Var_q)

    testutils.confirm_equal( nps.mag(q_query_sampled_mean - q_query),
                             0,
                             eps = 5,
                             worstcase = True,
                             msg = "Sampled projections cluster around the sample point")

    testutils.confirm_equal(worst_direction_stdev_observed,
                            worst_direction_stdev_predicted,
                            eps = 0.1,
                            worstcase = True,
                            relative  = True,
                            msg = f"Predicted worst-case projections match sampled observations")

    # I now compare the variances. The cross terms have lots of apparent error,
    # but it's more meaningful to compare the eigenvectors and eigenvalues, so I
    # just do that
    testutils.confirm_covariances_equal(Var_q,
                                        Var_q_observed,
                                        what = "Var_q",
                                        eps_eigenvalues               = 0.05,
                                        eps_eigenvectors_deg          = 10,
                                        check_biggest_eigenvalue_only = True)

    if args.make_documentation_plots is not None:

        def makeplot(**plotoptions):
            gp.plot(*mrcal.utils._plot_args_points_and_covariance_ellipse( \
                        q_query_sampled,
                        "Observed in simulation"),
                    mrcal.utils._plot_arg_covariance_ellipse( \
                        np.mean(q_query_sampled, axis=0),
                        Var_q,
                        "Predicted by mrcal"),
                    square = 1,
                    xlabel = "x (pixels)",
                    ylabel = "y (pixels)",
                    title  = "Surveyed calibration: distribution of point projections due to calibration-time noise",
                    **plotoptions)

        if args.make_documentation_plots:
            for extension in ('pdf','svg','png','gp'):
                makeplot(wait     = False,
                         terminal = shorter_terminal(terminal[extension]),
                         _set     = extraset[extension],
                         hardcopy = f'{args.make_documentation_plots}/var-q.{extension}')

        else:
            makeplot(wait = True)


testutils.finish()
