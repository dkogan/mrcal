#!/usr/bin/env python3

r'''Linearization test

Make sure the linearization assumptions used in the uncertainty computations
hold. The expressions are derived in the docstring for
mrcal.projection_uncertainty()

'''

import sys
import numpy as np
import numpysane as nps
import os

testdir = os.path.dirname(os.path.realpath(__file__))

# I import the LOCAL mrcal since that's what I'm testing
sys.path[:0] = f"{testdir}/..",
import mrcal
import copy
import testutils

from test_calibration_helpers import sample_dqref

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
Nintrinsics = mrcal.lensmodel_num_params(lensmodel)

Ncameras = len(models_ref)
Ncameras_extrinsics = Ncameras - 1

Nframes  = 50

models_ref[0].rt_cam_ref(np.zeros((6,), dtype=float))
models_ref[1].rt_cam_ref(np.array((0.08,0.2,0.02, 1., 0.9,0.1)))
models_ref[2].rt_cam_ref(np.array((0.01,0.07,0.2, 2.1,0.4,0.2)))
models_ref[3].rt_cam_ref(np.array((-0.1,0.08,0.08, 4.4,0.2,0.1)))

pixel_uncertainty_stdev = 1.5
object_spacing          = 0.1
object_width_n          = 10
object_height_n         = 9
calobject_warp_ref      = np.array((0.002, -0.005))

# shapes (Nframes, Ncameras, Nh, Nw, 2),
#        (Nframes, 4,3)
q_ref,Rt_ref_board_ref = \
    mrcal.synthesize_board_observations(models_ref,
                                        object_width_n                  = object_width_n,
                                        object_height_n                 = object_height_n,
                                        object_spacing                  = object_spacing,
                                        calobject_warp                  = calobject_warp_ref,
                                        rt_ref_boardcenter              = np.array((0.,  0.,  0., -2,   0,  4.0)),
                                        rt_ref_boardcenter__noiseradius = np.array((np.pi/180.*30., np.pi/180.*30., np.pi/180.*20., 2.5, 2.5, 2.0)),
                                        Nframes                         = Nframes)

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
extrinsics_ref = nps.cat( *[m.rt_cam_ref() for m in models_ref[1:]] )
if extrinsics_ref.size == 0:
    extrinsics_ref = np.zeros((0,6), dtype=float)
frames_ref     = mrcal.rt_from_Rt(Rt_ref_board_ref)


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
             indices_frame_camera[:,(1,)] - 1,
             axis=-1)


# Add a bit of noise to make my baseline not perfect
_,observations_baseline = sample_dqref(observations_ref,
                                       pixel_uncertainty_stdev)
baseline = \
    dict(intrinsics                                = intrinsics_ref,
         rt_cam_ref                                = extrinsics_ref,
         rt_ref_frame                              = frames_ref,
         points                                    = None,
         observations_board                        = observations_baseline,
         indices_frame_camintrinsics_camextrinsics = indices_frame_camintrinsics_camextrinsics,
         observations_point                        = None,
         indices_point_camintrinsics_camextrinsics = None,
         lensmodel                                 = lensmodel,
         do_optimize_calobject_warp                = True,
         calobject_warp                            = calobject_warp_ref,
         do_optimize_intrinsics_core               = True,
         do_optimize_intrinsics_distortions        = True,
         do_optimize_extrinsics                    = True,
         imagersizes                               = imagersizes,
         calibration_object_spacing                = object_spacing,
         do_apply_regularization                   = True)

mrcal.optimize(**baseline,
               do_apply_outlier_rejection = False)


# Done setting up. I'll be looking at tiny motions off the baseline
Nframes     = len(frames_ref)
Ncameras    = len(intrinsics_ref)
lensmodel   = baseline['lensmodel']
Nintrinsics = mrcal.lensmodel_num_params(lensmodel)

Nmeasurements_boards         = mrcal.num_measurements_boards(**baseline)
Nmeasurements_regularization = mrcal.num_measurements_regularization(**baseline)

b0,x0,J0 = mrcal.optimizer_callback(no_factorization = True,
                                    **baseline)[:3]
J0 = J0.toarray()


###########################################################################
# First a very basic gradient check. Looking at an arbitrary camera's
# intrinsics. The test-gradients tool does this much more thoroughly
optimization_inputs = copy.deepcopy(baseline)
db_packed           = np.random.randn(len(b0)) * 1e-9

mrcal.ingest_packed_state(b0 + db_packed,
                          **optimization_inputs)

x1 = mrcal.optimizer_callback(no_factorization = True,
                              no_jacobian      = True,
                              **optimization_inputs)[1]

dx_observed = x1 - x0

dx_predicted = nps.inner(J0, db_packed)
testutils.confirm_equal( dx_predicted, dx_observed,
                         eps = 1e-1,
                         worstcase = True,
                         relative  = True,
                         msg = "Trivial, sanity-checking gradient check")

if 0:
    import gnuplotlib as gp
    gp.plot( nps.cat(dx_predicted, dx_observed,),
             _with='lines',
             legend=np.arange(2),
             _set = mrcal.plotoptions_measurement_boundaries(**optimization_inputs),
             wait=1)

###########################################################################
# We're supposed to be at the optimum. E = norm2(x) ~ norm2(x0 + J db) =
# norm2(x0) + 2 dbt Jt x0 + norm2(J db). At the optimum Jt x0 = 0 -> E =
# norm2(x0) + norm2(J db). dE = norm2(J db) = norm2(dx_predicted)
x_predicted  = x0 + dx_predicted
dE           = nps.norm2(x1) - nps.norm2(x0)
dE_predicted = nps.norm2(dx_predicted)
testutils.confirm_equal( dE_predicted, dE,
                         eps = 1e-3,
                         relative = True,
                         msg = "diff(E) predicted")

# At the optimum dE/db = 0 -> xtJ = 0
xtJ0 = nps.inner(nps.transpose(J0),x0)
mrcal.pack_state(xtJ0, **optimization_inputs)
testutils.confirm_equal( xtJ0, 0,
                         eps = 1.5e-2,
                         worstcase = True,
                         msg = "dE/db = 0 at the optimum: original")

###########################################################################
# I perturb my input observation vector qref by dqref.
noise_for_gradients = 1e-3
dqref,observations_perturbed = sample_dqref(baseline['observations_board'],
                                            noise_for_gradients)
optimization_inputs = copy.deepcopy(baseline)
optimization_inputs['observations_board'] = observations_perturbed

mrcal.optimize(**optimization_inputs, do_apply_outlier_rejection=False)
b1,x1,J1 = mrcal.optimizer_callback(no_factorization = True,
                                    **optimization_inputs)[:3]
J1 = J1.toarray()

dx_observed = x1-x0
db_observed = b1-b0
w           = observations_perturbed[...,2]
w[w < 0]    = 0 # outliers have weight=0
w           = np.ravel(nps.mv(nps.cat(w,w),0,-1)) # each weight controls x,y

xtJ1 = nps.inner(nps.transpose(J1),x1)
mrcal.pack_state(xtJ0, **optimization_inputs)
testutils.confirm_equal( xtJ1, 0,
                         eps = 1e-2,
                         worstcase = True,
                         msg = "dE/db = 0 at the optimum: perturbed")

# I added noise reoptimized, did dx do the expected thing?
# I should have
#   x(b+db, qref+dqref) = x + J db + dx/dqref dqref
# -> x1-x0 ~ J db + dx/dqref dqref
#   x[measurements] = (q - qref) * weight
# -> dx/dqref = -diag(weight)
dx_predicted = nps.inner(J0,db_observed)
dx_predicted[:Nmeasurements_boards] -= w * dqref.ravel()

# plot_dx = gp.gnuplotlib( title = "dx predicted,observed",
#                          _set  = f'arrow nohead from {Nmeasurements_boards},graph 0 to {Nmeasurements_boards},graph 1')
# plot_dx.plot( (nps.cat(dx_observed,dx_predicted),
#                dict(legend = np.array(('observed','predicted')),
#                     _with  = 'lines')),
#               (dx_observed-dx_predicted,
#                dict(legend = "err",
#                     _with  = "lines lw 2",
#                     y2=1)))
testutils.confirm_equal( dx_predicted, dx_observed,
                         eps = 1e-6,
                         worstcase = True,
                         msg = "dx follows the prediction")

# The effect on the
# parameters should be db = M dqref. Where M = inv(JtJ) Jobservationst W

M = np.linalg.solve( nps.matmult(nps.transpose(J0),J0),
                     nps.transpose(J0[:Nmeasurements_boards, :]) ) * w
db_predicted = nps.matmult( dqref.ravel(), nps.transpose(M)).ravel()

istate0_frames         = mrcal.state_index_frames (0, **baseline)
istate0_calobject_warp = mrcal.state_index_calobject_warp(**baseline)
istate0_extrinsics = mrcal.state_index_extrinsics(0, **baseline)
if istate0_extrinsics is None:
    istate0_extrinsics = istate0_frames

slice_intrinsics = slice(0, istate0_extrinsics)
slice_extrinsics = slice(istate0_extrinsics, istate0_frames)
slice_frames     = slice(istate0_frames, istate0_calobject_warp)

# These thresholds look terrible. And they are. But I'm pretty sure this is
# working properly. Look at the plots:
if 0:
    import gnuplotlib as gp
    plot_db = gp.gnuplotlib( title = "db predicted,observed",
                             _set  = mrcal.plotoptions_state_boundaries(**optimization_inputs))
    plot_db.plot( (nps.cat(db_observed,db_predicted),
                   dict(legend = np.array(('observed','predicted')),
                        _with  = 'linespoints')),
                  (db_observed-db_predicted,
                   dict(legend = "err",
                        _with  = "lines lw 2",
                        y2=1)))
    plot_db.wait()

testutils.confirm_equal( db_predicted[slice_intrinsics],
                         db_observed [slice_intrinsics],
                         percentile = 80,
                         eps        = 0.2,
                         msg        = f"Predicted db from dqref: intrinsics")
testutils.confirm_equal( db_predicted[slice_extrinsics],
                         db_observed [slice_extrinsics],
                         relative   = True,
                         percentile = 80,
                         eps        = 0.2,
                         msg        = f"Predicted db from dqref: extrinsics")
testutils.confirm_equal( db_predicted[slice_frames],
                         db_observed [slice_frames],
                         percentile = 80,
                         eps        = 0.2,
                         msg        = f"Predicted db from dqref: frames")

testutils.finish()
