#!/usr/bin/python3

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
                        default = 2.5,
                        help='''Distance from the camera to the center of each
                        chessboard. If omitted, we default to 2.5''')
    parser.add_argument('--range-board-center',
                        type=float,
                        help='''Distance from the camera to the center of the
                        MIDDLE chessboard. If omitted, we use --range-board''')
    parser.add_argument('--seed-rng',
                        type=int,
                        default=0,
                        help='''Value to seed the rng with''')
    parser.add_argument('--viz-seed-geometry',
                        action='store_true',
                        help='''Show the seed geometry''')
    parser.add_argument('--viz-observations',
                        action='store_true',
                        help='''Show the observations''')
    parser.add_argument('--say-errz',
                        action='store_true',
                        help='''If given, report ONLY the extrinsics error in z.
                        Do not run tests or print anything else''')

    args = parser.parse_args()

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

from test_calibration_helpers import sample_dqref
import copy

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

object_spacing          = 0.1
object_width_n          = 10
object_height_n         = 9
pixel_uncertainty_stdev = 0.5

# We try to recover this in the calibration process
# shape (6,)
rt_cam_ref_true   = np.array((-0.04,  0.05,  -0.1,     1.2, -0.1,  0.1),)
model_true.extrinsics_rt_fromref(rt_cam_ref_true)

if False:
    model_true.write("/tmp/true.cameramodel")
    print("wrote /tmp/true.cameramodel")

# I want to space out the chessboards across the whole imager. To make this
# simpler I define it in terms of the "boardcentered" coordinate system, whose
# origin is at the center of the chessboard, not in a corner
board_center                  = \
    np.array(( (object_width_n -1)*object_spacing/2.,
               (object_height_n-1)*object_spacing/2.,
               0 ))
rt_boardcentered_board_true     = mrcal.identity_rt()
rt_boardcentered_board_true[3:] = -board_center

# I have a 3x3 grid of chessboards
NboardgridX,NboardgridY = 3,1

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
random_radius_r = 0.1
random_radius_t = 0.2
rt_cam_board_true[...,:3] += (np.random.rand(*rt_cam_board_true[...,:3].shape)*2. - 1.) * random_radius_r
rt_cam_board_true[...,3:] += (np.random.rand(*rt_cam_board_true[...,3:].shape)*2. - 1.) * random_radius_t

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

q = \
    q_true \
    +      \
    np.random.randn(*q_true.shape) \
    * pixel_uncertainty_stdev \
    / nps.mv(nps.cat(weight,weight),0,-1)

# out-of-bounds observations are outliers
weight[(q[:,0] < 0) + \
       (q[:,1] < 0) + \
       (q[:,0] > W-1) + \
       (q[:,1] > H-1)] = -1.

############# Now I pretend that the noisy observations are all I got, and I run
############# a calibration from those
# shape (N, 3)
observations_point = nps.glue(q,
                              nps.dummy(weight,-1),
                              axis=-1)

Npoints = q.shape[0]

# Dense observations. The cameras see all the boards
indices_point_camintrinsics_camextrinsics = np.zeros( (Npoints, 3), dtype=np.int32)
indices_point_camintrinsics_camextrinsics[:,0] = np.arange(Npoints)

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
    i = nps.norm2( observations_point[...,:2] - cxy ) < 1000**2.
    observations_point_center = np.array(observations_point)
    observations_point_center[~i,2] = -1.
    Rt_camera_ref_estimate = \
        mrcal.calibration._estimate_camera_pose_from_point_observations( \
            indices_point_camintrinsics_camextrinsics,
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



if args.viz_seed_geometry:
    mrcal.show_geometry( (rt_camera_ref_estimate,),
                         points      = pref_true, # known. fixed. perfect.
                         title       = "Seed geometry",
                         show_points = True,
                         wait        = True)
    sys.exit()

if args.viz_observations:
    import gnuplotlib as gp
    gp.plot( observations_point,
             _with     = 'points palette',
             tuplesize = -3,
             square=1,
             # The rectangle plot needs a later gnuplot to work with
             # the x11 terminal. So I use the 'qt' terminal here
             _set = f'object rectangle from 0,0 to {W-1},{H-1} fs empty border rgb "black"',
             terminal = 'qt',
             wait = True)
    sys.exit()

optimization_inputs                                 = \
    dict( lensmodel                                 = lensmodel,
          intrinsics                                = intrinsics_data,
          extrinsics_rt_fromref                     = rt_camera_ref_estimate,
          frames_rt_toref                           = None,
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

optimization_inputs['do_optimize_intrinsics_core']        = False
optimization_inputs['do_optimize_intrinsics_distortions'] = False
optimization_inputs['do_optimize_extrinsics']             = True
stats = mrcal.optimize(**optimization_inputs)

optimization_inputs['do_optimize_intrinsics_core']        = True
optimization_inputs['do_optimize_intrinsics_distortions'] = True
optimization_inputs['do_optimize_extrinsics']             = True
stats = mrcal.optimize(**optimization_inputs)

# Grab the residuals. residuals_point() reports ONLY the point reprojection
# errors: no range normalization (penalty) terms, no board observations, no
# regularization. Also, no outliers
rmserr_point = np.std(mrcal.residuals_point(optimization_inputs,
                                            residuals = stats['x']).ravel())

############# Calibration computed. Now I see how well I did
model_solved = \
    mrcal.cameramodel( optimization_inputs = optimization_inputs,
                       icam_intrinsics     = 0 )
# Checking the extrinsics.
Rt_extrinsics_err = \
    mrcal.compose_Rt( model_solved.extrinsics_Rt_fromref(),
                      model_true  .extrinsics_Rt_toref() )

if args.say_errz:
    print(Rt_extrinsics_err[3,2])
sys.exit()




# verify problem layout
testutils.confirm_equal( mrcal.num_states(**optimization_inputs),
                         Nintrinsics + 6,
                         "num_states()")
testutils.confirm_equal( mrcal.num_states_intrinsics(**optimization_inputs),
                         Nintrinsics,
                         "num_states_intrinsics()")
testutils.confirm_equal( mrcal.num_intrinsics_optimization_params(**optimization_inputs),
                         Nintrinsics,
                         "num_intrinsics_optimization_params()")
testutils.confirm_equal( mrcal.num_states_extrinsics(**optimization_inputs),
                         6,
                         "num_states_extrinsics()")
testutils.confirm_equal( mrcal.num_states_frames(**optimization_inputs),
                         0,
                         "num_states_frames()")
testutils.confirm_equal( mrcal.num_states_points(**optimization_inputs),
                         0,
                         "num_states_points()")
testutils.confirm_equal( mrcal.num_states_calobject_warp(**optimization_inputs),
                         0,
                         "num_states_calobject_warp()")
testutils.confirm_equal( mrcal.num_measurements_boards(**optimization_inputs),
                         0,
                         "num_measurements_boards()")
testutils.confirm_equal( mrcal.num_measurements_points(**optimization_inputs),
                         Npoints*3,
                         "num_measurements_points()")
testutils.confirm_equal( mrcal.num_measurements_regularization(**optimization_inputs),
                         6,
                         "num_measurements_regularization()")
testutils.confirm_equal( mrcal.state_index_intrinsics(0, **optimization_inputs),
                         0,
                         "state_index_intrinsics()")
testutils.confirm_equal( mrcal.state_index_extrinsics(0, **optimization_inputs),
                         8,
                         "state_index_extrinsics()")
testutils.confirm_equal( mrcal.measurement_index_points(2, **optimization_inputs),
                         3*2,
                         "measurement_index_points()")
testutils.confirm_equal( mrcal.measurement_index_regularization(**optimization_inputs),
                         3*Npoints,
                         "measurement_index_regularization()")

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

testutils.finish()
