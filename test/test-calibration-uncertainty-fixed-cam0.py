#!/usr/bin/python3

r'''Uncertainty-quantification test

I run a number of synthetic-data camera calibrations, applying some noise to the
observed inputs each time. I then look at the distribution of projected world
points, and compare that distribution with theoretical predictions.

This test checks two different types of calibrations:

- fixed-cam0: we place camera0 at the reference coordinate system. So camera0
  may not move, and has no associated extrinsics vector. The other cameras and
  the frames move. When evaluating at projection uncertainty I pick a point
  referenced off the frames. As the frames move around, so does the point I'm
  projecting. But together, the motion of the frames and the extrinsics and the
  intrinsics should map it to the same pixel in the end.

- fixed-frames: the reference coordinate system is attached to the frames, which
  may not move. All cameras may move around and all cameras have an associated
  extrinsics vector. When evaluating at projection uncertainty I also pick a
  point referenced off the frames, but here any point in the reference coord
  system will do. As the cameras move around, so does the point I'm projecting.
  But together, the motion of the extrinsics and the intrinsics should map it to
  the same pixel in the end.

The calibration type is selected by the filename of the script we run. The
expectation is that we have symlinks pointing to this script, and the
calibration type can be chosen by running the correct symlink

ARGUMENTS

By default (no arguments) we run the test and report success/failure as usual.
To test stuff pass any/all of these in any order:

- show-distribution: plot the observed/predicted distributions of the projected
  points

- study: compute a grid of the predicted worst-direction distributions across
  the imager and for different depths

- write-models: write the produced models to disk. The files on disk can then be
  processed with the cmdline tools

Any of the diagnostic modes drop into a REPL when done

'''

import sys
import numpy as np
import numpysane as nps
import os

testdir = os.path.dirname(os.path.realpath(__file__))

# I import the LOCAL mrcal since that's what I'm testing
sys.path[:0] = f"{testdir}/..",
import mrcal
import testutils
import copy

from test_calibration_helpers import optimize,sample_dqref,sorted_eig

import re
if   re.search("fixed-cam0",   sys.argv[0]): fixedframes = False
elif re.search("fixed-frames", sys.argv[0]): fixedframes = True
else:
    raise Exception("This script should contain either 'fixed-cam0' or 'fixed-frames' in the filename")

args = set(sys.argv[1:])

known_args = set(('show-distribution', 'study', 'write-models'))

if not all(arg in known_args for arg in args):
    raise Exception(f"Unknown argument given. I know about {known_args}")

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


# I want the RNG to be deterministic
np.random.seed(0)

############# Set up my world, and compute all the perfect positions, pixel
############# observations of everything
models_ref = ( mrcal.cameramodel(f"{testdir}/data/cam0.opencv8.cameramodel"),
               mrcal.cameramodel(f"{testdir}/data/cam0.opencv8.cameramodel"),
               mrcal.cameramodel(f"{testdir}/data/cam1.opencv8.cameramodel"),
               mrcal.cameramodel(f"{testdir}/data/cam1.opencv8.cameramodel") )

imagersizes = nps.cat( *[m.imagersize() for m in models_ref] )
lensmodel   = models_ref[0].intrinsics()[0]
# I have opencv8 models_ref, but let me truncate to opencv4 models_ref to keep this
# simple and fast
lensmodel = 'LENSMODEL_OPENCV4'
for m in models_ref:
    m.intrinsics( intrinsics = (lensmodel, m.intrinsics()[1][:8]))
Nintrinsics = mrcal.num_lens_params(lensmodel)

Ncameras = len(models_ref)
Ncameras_extrinsics = Ncameras
if not fixedframes: Ncameras_extrinsics -= 1

Nframes  = 50

models_ref[0].extrinsics_rt_fromref(np.zeros((6,), dtype=float))
models_ref[1].extrinsics_rt_fromref(np.array((0.08,0.2,0.02, 1., 0.9,0.1)))
models_ref[2].extrinsics_rt_fromref(np.array((0.01,0.07,0.2, 2.1,0.4,0.2)))
models_ref[3].extrinsics_rt_fromref(np.array((-0.1,0.08,0.08, 4.4,0.2,0.1)))

pixel_uncertainty_stdev = 1.5
object_spacing          = 0.1
object_width_n          = 10
object_height_n         = 9
calobject_warp_ref      = np.array((0.002, -0.005))

# shapes (Nframes, Ncameras, Nh, Nw, 2),
#        (Nframes, 4,3)
q_ref,Rt_cam0_board_ref = \
    mrcal.make_synthetic_board_observations(models_ref,
                                            object_width_n, object_height_n, object_spacing,
                                            calobject_warp_ref,
                                            np.array((-2,   0,  4.0,  0.,  0.,  0.)),
                                            np.array((2.5, 2.5, 2.0, 40., 30., 30.)),
                                            Nframes)

############# I have perfect observations in q_ref. I corrupt them by noise
# weight has shape (Nframes, Ncameras, Nh, Nw),
weight01 = (np.random.rand(*q_ref.shape[:-1]) + 1.) / 2. # in [0,1]
weight0 = 0.2
weight1 = 1.0
weight = weight0 + (weight1-weight0)*weight01

# I want observations of shape (Nframes*Ncameras, Nh, Nw, 3) where each row is
# (x,y,weight)
observations_ref = nps.clump( nps.glue(q_ref,
                                       nps.dummy(weight,-1),
                                       axis=-1),
                              n=2)

# These are perfect
intrinsics_ref = nps.cat( *[m.intrinsics()[1]         for m in models_ref] )
if fixedframes:
    extrinsics_ref = nps.cat( *[m.extrinsics_rt_fromref() for m in models_ref] )
else:
    extrinsics_ref = nps.cat( *[m.extrinsics_rt_fromref() for m in models_ref[1:]] )
if extrinsics_ref.size == 0:
    extrinsics_ref = np.zeros((0,6), dtype=float)
frames_ref     = mrcal.rt_from_Rt(Rt_cam0_board_ref)


# Dense observations. All the cameras see all the boards
indices_frame_camera = np.zeros( (Nframes*Ncameras, 2), dtype=np.int32)
indices_frame = indices_frame_camera[:,0].reshape(Nframes,Ncameras)
indices_frame.setfield(nps.outer(np.arange(Nframes, dtype=np.int32),
                                 np.ones((Ncameras,), dtype=np.int32)),
                       dtype = np.int32)
indices_camera = indices_frame_camera[:,1].reshape(Nframes,Ncameras)
indices_camera.setfield(nps.outer(np.ones((Nframes,), dtype=np.int32),
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
Nsamples = 90


# p = mrcal.show_calibration_geometry(models_ref,
#                                     frames          = frames_ref,
#                                     object_width_n  = object_width_n,
#                                     object_height_n = object_height_n,
#                                     object_spacing  = object_spacing)
# sys.exit()


def sample_reoptimized_parameters(do_optimize_frames, apply_noise=True, **kwargs):
    if apply_noise:
        dqref, observations_perturbed = sample_dqref(observations_ref,
                                                     pixel_uncertainty_stdev)
    else:
        observations_perturbed = observations_ref.copy()
        dqref                  = observations_perturbed[...,:2]*0

    intrinsics_solved,extrinsics_solved,frames_solved,_, \
    idx_outliers, \
    _,  _, _, \
    optimization_inputs =           \
        optimize(intrinsics_ref, extrinsics_ref, frames_ref, observations_perturbed,
                 indices_frame_camintrinsics_camextrinsics,
                 lensmodel,
                 imagersizes,
                 object_spacing, object_width_n, object_height_n,
                 pixel_uncertainty_stdev,
                 calobject_warp                    = calobject_warp_ref,
                 do_optimize_intrinsics_core        = True,
                 do_optimize_intrinsics_distortions = True,
                 do_optimize_extrinsics            = True,
                 do_optimize_frames                = do_optimize_frames,
                 do_optimize_calobject_warp        = True,
                 skip_outlier_rejection            = True,
                 skip_regularization               = True,
                 **kwargs)
    return \
        intrinsics_solved,extrinsics_solved,frames_solved,\
        dqref, observations_perturbed, \
        optimization_inputs


# I want to treat the extrinsics arrays as if all the camera transformations are
# stored there
if fixedframes:
    extrinsics_ref_mounted = extrinsics_ref
else:
    extrinsics_ref_mounted = \
        nps.glue( np.zeros((6,), dtype=float),
                  extrinsics_ref,
                  axis = -2)

# And rebuild a new set of models, BUT, running the optimizer (no noise) before
# storing the models. If the optimization is looking only at the input data,
# then this will be identical to models_ref. But if we also have regularization,
# this will move us off-center
ii,ee,ff,_,_,optimization_inputs = \
    sample_reoptimized_parameters(do_optimize_frames = not fixedframes,
                                  apply_noise=False)

models_ref_optimized = \
    [ mrcal.cameramodel( optimization_inputs = optimization_inputs,
                         icam_intrinsics     = i) \
      for i in range(Ncameras) ]

if 'write-models' in args:
    for i in range(Ncameras):
        models_ref          [i].write(f"/tmp/models-ref-camera{i}.cameramodel")
        models_ref_optimized[i].write(f"/tmp/models-ref-optimized-camera{i}.cameramodel")
    sys.exit()

# I evaluate the projection uncertainty of this vector. In each camera. I'd like
# it to be center-ish, but not AT the center. So I look at 1/3 (w,h). I want
# this to represent a point in a globally-consistent coordinate system. Here I
# have fixed frames, so using the reference coordinate system gives me that
# consistency. Note that I look at q0 for each camera separately, so I'm going
# to evaluate a different world point for each camera
q0 = imagersizes[0]/3.


for icam in (0,3):
    # I move the extrinsics of a model, write it to disk, and make sure the same
    # uncertainties come back
    model_moved = mrcal.cameramodel(models_ref_optimized[icam])
    model_moved.extrinsics_rt_fromref([1., 2., 3., 4., 5., 6.])
    model_moved.write(f'{workdir}/out.cameramodel')
    model_read = mrcal.cameramodel(f'{workdir}/out.cameramodel')

    icam_intrinsics_read = model_read.icam_intrinsics()
    icam_extrinsics_read = mrcal.corresponding_icam_extrinsics(icam_intrinsics_read,
                                                               **model_read.optimization_inputs())

    testutils.confirm_equal(icam if fixedframes else icam-1,
                            icam_extrinsics_read,
                            msg = f"corresponding icam_extrinsics reported correctly for camera {icam}")

    pcam = mrcal.unproject( q0, *models_ref_optimized[icam].intrinsics(),
                            normalize = True)

    Var_dq_ref = \
        mrcal.projection_uncertainty( pcam * 1.0,
                                      model = models_ref_optimized[icam] )
    Var_dq_moved_written_read = \
        mrcal.projection_uncertainty( pcam * 1.0,
                                      model = model_read )
    testutils.confirm_equal(Var_dq_moved_written_read, Var_dq_ref,
                            eps = 0.001,
                            worstcase = True,
                            relative  = True,
                            msg = f"var(dq) with full rt matches for camera {icam} after moving, writing to disk, reading from disk")

    Var_dq_inf_ref = \
        mrcal.projection_uncertainty( pcam * 1.0,
                                      model = models_ref_optimized[icam],
                                      atinfinity = True )
    Var_dq_inf_moved_written_read = \
        mrcal.projection_uncertainty( pcam * 1.0,
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
        mrcal.projection_uncertainty( pcam * 100.0,
                                      model = models_ref_optimized[icam],
                                      atinfinity = True )
    testutils.confirm_equal(Var_dq_inf_far_ref, Var_dq_inf_ref,
                            eps = 0.001,
                            worstcase = True,
                            relative  = True,
                            msg = f"var(dq) (infinity) is invariant to point scale for camera {icam}")

intrinsics_sampled = np.zeros( (Nsamples,Ncameras,Nintrinsics),  dtype=float )
extrinsics_sampled = np.zeros( (Nsamples,Ncameras_extrinsics,6), dtype=float )
frames_sampled     = np.zeros( (Nsamples,Nframes, 6),            dtype=float )

for isample in range(Nsamples):
    print(f"Sampling {isample}/{Nsamples}")
    ii,ee,ff = sample_reoptimized_parameters(do_optimize_frames = not fixedframes)[:3]

    intrinsics_sampled[isample,...] = ii
    extrinsics_sampled[isample,...] = ee
    frames_sampled    [isample,...] = ff


def check_uncertainties_at(q0, distance):

    # distance of "None" means I'll simulate a large distance, but compare
    # against a special-case distance of "infinity"
    if distance is None:
        distance    = 1e4
        atinfinity  = True
        distancestr = "infinity"
    else:
        atinfinity  = False
        distancestr = str(distance)

    # I come up with ONE global point per camera. And I project that one point for
    # each sample
    # shape (Ncameras, 3)
    v0_cam = mrcal.unproject(q0, lensmodel, intrinsics_ref,
                             normalize = True)

    # shape (Ncameras, 3). In the ref coord system
    p0_ref = \
        mrcal.transform_point_rt( mrcal.invert_rt(extrinsics_ref_mounted),
                                  v0_cam * distance )

    if fixedframes:
        p0_frames = p0_ref
    else:
        # shape (Nframes, Ncameras, 3)
        # The point in the coord system of all the frames
        p0_frames = mrcal.transform_point_rt( nps.dummy(mrcal.invert_rt(frames_ref),-2),
                                              p0_ref)


    ###############################################################################
    # Now I have the projected point in the coordinate system of the frames. I
    # project that back to each sampled camera, and gather the projection statistics
    if fixedframes:
        extrinsics_sampled_mounted = extrinsics_sampled
        p0_sampleref               = p0_ref
    else:
        # I want to treat the extrinsics arrays as if ALL the camera
        # transformations are stored there. Cam0 is at the reference, so prepend
        # its identity transformation
        extrinsics_sampled_mounted = \
            nps.glue( np.zeros((Nsamples,1,6), dtype=float),
                      extrinsics_sampled,
                      axis = -2)

        # shape (Nsamples, Nframes, Ncameras, 3)
        p0_sampleref_allframes = mrcal.transform_point_rt( nps.dummy(frames_sampled, -2),
                                                           p0_frames )
        # shape (Nsamples, Ncameras, 3)
        p0_sampleref = np.mean(p0_sampleref_allframes, axis=-3)


    # shape (Nsamples, Ncameras, 2)
    q_sampled = \
        mrcal.project( \
            mrcal.transform_point_rt(extrinsics_sampled_mounted,
                                     p0_sampleref),
            lensmodel,
            intrinsics_sampled )


    # shape (Ncameras, 2)
    q_sampled_mean = np.mean(q_sampled, axis=-3)
    Var_dq_observed = np.mean( nps.outer(q_sampled-q_sampled_mean,
                                         q_sampled-q_sampled_mean), axis=-4 )

    worst_direction_stdev_observed = mrcal.worst_direction_stdev(Var_dq_observed)

    Var_dq = \
        nps.cat(*[ mrcal.projection_uncertainty( \
            v0_cam[icam] * (distance if not atinfinity else 1.0),
            atinfinity = atinfinity,
            model      = models_ref_optimized[icam]) \
                   for icam in range(Ncameras) ])
    worst_direction_stdev_predicted = mrcal.worst_direction_stdev(Var_dq)


    testutils.confirm_equal(q_sampled_mean,
                            nps.matmult(np.ones((Ncameras,1)), q0),
                            eps = 0.5,
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


q_sampled,Var_dq = check_uncertainties_at(q0, 5.)
check_uncertainties_at(q0, None)

if len(args) == 0:
    testutils.finish()


import gnuplotlib as gp

if 'study' in args:
    gridn_width  = 16
    gridn_height = 12
    # shape (gridn_height,gridn_width,2)
    qxy = mrcal.sample_imager(gridn_width, gridn_height, *imagersizes[0])

    ranges = np.linspace(1,30,10)

    # shape (gridn_height,gridn_width,Nranges,3)
    pcam = \
        nps.dummy(nps.cat(*[mrcal.unproject( qxy, *models_ref_optimized[icam].intrinsics(),
                                             normalize = True) for icam in range(Ncameras)]), -2) * \
        nps.dummy(ranges, -1)

    # shape (Ncameras, gridn_height, gridn_width, Nranges, 2,2)
    Var_dq_grid = \
        nps.cat(*[ mrcal.projection_uncertainty( \
            pcam[icam],
            model = models_ref_optimized[icam] ) \
                   for icam in range(Ncameras) ])
    # shape (Ncameras, gridn_height, gridn_width, Nranges)
    worst_direction_stdev_grid = mrcal.worst_direction_stdev(Var_dq_grid)

    # shape (Ncameras, gridn_height, gridn_width, 2,2)
    Var_dq_infinity = \
        nps.cat(*[ mrcal.projection_uncertainty( \
            pcam[icam,:,:,0,:], # any range works here
            atinfinity = True,
            model = models_ref_optimized[icam] ) \
                   for icam in range(Ncameras) ])

    # shape (Ncameras, gridn_height, gridn_width)
    worst_direction_stdev_infinity = mrcal.worst_direction_stdev(Var_dq_infinity)


    # Can plot like this:
    if 0:
        grid__x_y_ranges = \
            np.meshgrid(qxy[0,:,0],
                        qxy[:,0,1],
                        ranges,
                        indexing = 'ij')
        grid__x_y = \
            np.meshgrid(qxy[0,:,0],
                        qxy[:,0,1],
                        indexing = 'ij')
        icam = 0

        gp.plot( *[g.ravel() for g in grid__x_y_ranges],
                 nps.xchg(worst_direction_stdev_grid[icam],0,1).ravel(),
                 nps.xchg(worst_direction_stdev_grid[icam],0,1).ravel(),
                 tuplesize = 5,
                 _with = 'points pt 7 ps variable palette',
                 _3d = True,
                 squarexy = True,
                 xlabel = 'pixel x',
                 ylabel = 'pixel y',
                 zlabel = 'range',
                 wait = True)

        gp.plot( *[g.ravel() for g in grid__x_y],
                 nps.xchg(worst_direction_stdev_infinity[icam],0,1).ravel(),
                 nps.xchg(worst_direction_stdev_infinity[icam],0,1).ravel(),
                 tuplesize = 4,
                 _with = 'points pt 7 ps variable palette',
                 square = True,
                 yinv   = True,
                 xlabel = 'pixel x',
                 ylabel = 'pixel y')


def get_cov_plot_args(q, Var, what):

    l,v   = sorted_eig(Var)
    l0,l1 = l
    v0,v1 = nps.transpose(v)

    major       = np.sqrt(l0)
    minor       = np.sqrt(l1)
    isotropic   = np.sqrt( np.trace(Var) / 2. )

    return \
      ((q[0], q[1], 2*major, 2*minor, 180./np.pi*np.arctan2(v0[1],v0[0]),
        dict(_with='ellipses', tuplesize=5, legend=f'{what} 1-sigma, full covariance')),
       (q[0], q[1], 2.*isotropic, 2.*isotropic,
        dict(_with='ellipses dt 2', tuplesize=4, legend=f'{what} 1-sigma; isotropic')))

def get_point_cov_plot_args(q, what):
    q_mean  = np.mean(q,axis=-2)
    q_mean0 = q - q_mean
    Var     = np.mean( nps.outer(q_mean0,q_mean0), axis=0 )
    return get_cov_plot_args(q_mean,Var, what)

def make_plot(icam, **kwargs):

    q_sampled_mean = np.mean(q_sampled[:,icam,:],axis=-2)

    p = gp.gnuplotlib(square=1,
                      _xrange=(q_sampled_mean[0]-2,q_sampled_mean[0]+2),
                      _yrange=(q_sampled_mean[1]-2,q_sampled_mean[1]+2),
                      title=f'Uncertainty reprojection distribution for camera {icam}',
                      **kwargs)
    p.plot( (q_sampled[:,icam,0], q_sampled[:,icam,1],
             dict(_with = 'points pt 6',
                  tuplesize = 2)),
            *get_point_cov_plot_args(q_sampled[:,icam,:], "Observed"),
            *get_cov_plot_args(q_sampled_mean, Var_dq[icam], "Propagating intrinsics, extrinsics uncertainties"),
            (q0,
             dict(tuplesize = -2,
                  _with     = 'points pt 1 ps 2',
                  legend    = 'Nominal center point. Regularization may bias the distribution off this point')))
    return p

if 'show-distribution' in args:
    plot_distribution = [None] * Ncameras
    for icam in range(Ncameras):
        plot_distribution[icam] = make_plot(icam)

import IPython
IPython.embed()
