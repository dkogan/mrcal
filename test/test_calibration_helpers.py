#!/usr/bin/python3

import sys
import numpy as np
import numpysane as nps
import copy
import os

# I import the LOCAL mrcal since that's what I'm testing
testdir = os.path.dirname(os.path.realpath(__file__))
sys.path[:0] = f"{testdir}/..",
import mrcal



def optimize( intrinsics,
              extrinsics_rt_fromref,
              frames_rt_toref,
              observations,
              indices_frame_camintrinsics_camextrinsics,
              lensmodel, imagersizes,
              object_spacing, object_width_n, object_height_n,
              pixel_uncertainty_stdev,

              calobject_warp                    = None,
              do_optimize_intrinsic_core        = False,
              do_optimize_intrinsic_distortions = False,
              do_optimize_extrinsics            = False,
              do_optimize_frames                = False,
              do_optimize_calobject_warp        = False,
              skip_outlier_rejection            = True,
              skip_regularization               = False,
              **kwargs):
    r'''Run the optimizer

    Function arguments are read-only. The optimization results, in various
    forms, are returned.

    '''

    intrinsics            = copy.deepcopy(intrinsics)
    extrinsics_rt_fromref = copy.deepcopy(extrinsics_rt_fromref)
    frames_rt_toref       = copy.deepcopy(frames_rt_toref)
    calobject_warp        = copy.deepcopy(calobject_warp)
    observations          = copy.deepcopy(observations)

    solver_context = mrcal.SolverContext()
    stats = mrcal.optimize( intrinsics, extrinsics_rt_fromref, frames_rt_toref, None,
                            observations, indices_frame_camintrinsics_camextrinsics,
                            None, None, lensmodel,
                            calobject_warp              = calobject_warp,
                            imagersizes                 = imagersizes,
                            calibration_object_spacing  = object_spacing,
                            calibration_object_width_n  = object_width_n,
                            calibration_object_height_n = object_height_n,
                            verbose                     = False,

                            observed_pixel_uncertainty  = pixel_uncertainty_stdev,

                            do_optimize_frames                = do_optimize_frames,
                            do_optimize_intrinsic_core        = do_optimize_intrinsic_core,
                            do_optimize_intrinsic_distortions = do_optimize_intrinsic_distortions,
                            do_optimize_extrinsics            = do_optimize_extrinsics,
                            do_optimize_calobject_warp        = do_optimize_calobject_warp,
                            skip_regularization               = skip_regularization,
                            skip_outlier_rejection            = skip_outlier_rejection,
                            solver_context                    = solver_context,
                            **kwargs)

    p_packed = solver_context.p().copy()

    return \
        intrinsics, extrinsics_rt_fromref, frames_rt_toref, calobject_warp,   \
        observations[...,2] < 0.0, \
        p_packed, stats['x'], stats['rms_reproj_error__pixels'], \
        solver_context

def sample_dqref(observations,
                 pixel_uncertainty_stdev,
                 make_outliers = False):

    weight  = observations[...,-1]
    q_noise = np.random.randn(*observations.shape[:-1], 2) * pixel_uncertainty_stdev / nps.mv(nps.cat(weight,weight),0,-1)

    if make_outliers:
        if not hasattr(sample_dqref, 'idx_outliers_ref_flat'):
            NobservedPoints = observations.size // 3
            sample_dqref.idx_outliers_ref_flat = \
                np.random.choice( NobservedPoints,
                                  (NobservedPoints//100,), # 1% outliers
                                  replace = False )
        nps.clump(q_noise, n=3)[sample_dqref.idx_outliers_ref_flat, :] *= 20

    observations_perturbed = observations.copy()
    observations_perturbed[...,:2] += q_noise
    return q_noise, observations_perturbed

def get_var_ief(icam_intrinsics, icam_extrinsics,
                did_optimize_extrinsics, did_optimize_frames,
                Nstate_intrinsics_onecam, Nframes,
                pixel_uncertainty_stdev,
                rotation_only, solver_context,
                cache_var_full):
    r'''Computes Var(intrinsics,extrinsics,frames)

    If we're optimizing extrinsics, BUT some camera is sitting at the reference,
    I include the rows, columns for the extrinsics arrays, but I set them to 0

    '''

    def apply_slice(arr, i):
        if isinstance(i, int):
            return np.zeros( (i,arr.shape[-1]), dtype=arr.dtype)
        return arr[i]

    def apply_slices(var_full, slices):
        r'''Use the slices[] to cut along both dimensions'''

        cut_vert  = nps.glue( *[apply_slice(var_full,                s) \
                                for s in slices], axis=-2 )
        cut_horiz = nps.glue( *[apply_slice(nps.transpose(cut_vert), s) \
                                for s in slices], axis=-2 )
        return cut_horiz



    if cache_var_full is not None and cache_var_full[0] is not None:
        var_full = cache_var_full[0]
    else:

        # The docstring of projection_uncertainty() has the full derivation of
        # the uncertainty expressions:
        #
        #   Var(p) = inv(JtJ) J[observations]t J[observations] inv(JtJ)
        #
        # In the special case where the measurements consist entirely of
        # observation discrepancies (i.e. no regularization terms) we can
        # simplify:
        #
        #   Var(p) = inv(JtJ) J[observations]t J[observations] inv(JtJ)
        #          = inv(JtJ) JtJ inv(JtJ)
        #          = inv(JtJ)
        #
        # These expressions all use unitless optimizer-internal state, but I
        # want it in the full state. So using * to denote unitless state I
        # actually have:
        #
        #   Var(p*) = inv(J*tJ*) J*[observations]t J*[observations] inv(J*tJ*)
        #
        # or
        #
        #   Var(p*) = inv(J*tJ*)
        #
        # I want Var(p) = Var( D p* ) =
        #               = D Var(p*) D =
        #               = D inv(J*t J*) D =
        #               = D inv( (JD)t JD ) D =
        #               = D inv(D) inv( JtJ ) inv(D) D =
        #               = inv(JtJ)
        #
        # Similarly for the more complex with-regularization expression. So I
        # make J from J*, and I'm good to go

        Jpacked = solver_context.J().toarray()
        J       = Jpacked.copy()
        solver_context.pack(J) # pack(), not unpack() because the packed variables are in the denominator

        Nmeasurements_regularization = solver_context.num_measurements_dict()['regularization']
        if Nmeasurements_regularization == 0:
            var_full = np.linalg.inv( nps.matmult(nps.transpose(J), J) )
        else:
            M = \
                np.linalg.solve(nps.matmult(nps.transpose(J), J),
                                nps.transpose(J[:-Nmeasurements_regularization]))
            var_full = nps.matmult(M, nps.transpose(M))

        if cache_var_full is not None:
            cache_var_full[0] = var_full

    if not did_optimize_extrinsics:

        slices = [ slice(solver_context.state_index_intrinsics(icam_intrinsics ),
                         solver_context.state_index_intrinsics(icam_intrinsics) + Nstate_intrinsics_onecam) ]

        if did_optimize_frames:
            if not rotation_only:
                slices.append( slice(solver_context.state_index_frame_rt(0),
                                     solver_context.state_index_frame_rt(0) + 6*Nframes) )
            else:
                # just the rotation
                for i in range(Nframes):
                    slices.append( slice(solver_context.state_index_frame_rt(i),
                                         solver_context.state_index_frame_rt(i)+3) )

    else:
        slices = [ slice(solver_context.state_index_intrinsics(icam_intrinsics ),
                         solver_context.state_index_intrinsics(icam_intrinsics) + Nstate_intrinsics_onecam) ]
        if not rotation_only:
            if icam_extrinsics >= 0:
                slices.append( slice(solver_context.state_index_camera_rt (icam_extrinsics),
                                     solver_context.state_index_camera_rt (icam_extrinsics) + 6) )
            else:
                # this camera is at the reference. A slice of int(x) means "fill
                # in with x 0s"
                slices.append(6)
        else:
            # just the rotation
            if icam_extrinsics >= 0:
                slices.append( slice(solver_context.state_index_camera_rt (icam_extrinsics),
                                     solver_context.state_index_camera_rt (icam_extrinsics) + 3) )
            else:
                # this camera is at the reference. A slice of int(x) means "fill
                # in with x 0s"
                slices.append(3)

        if did_optimize_frames:
            if not rotation_only:
                slices.append( slice(solver_context.state_index_frame_rt(0),
                                     solver_context.state_index_frame_rt(0) + 6*Nframes) )
            else:
                # just the rotation
                for i in range(Nframes):
                    slices.append( slice(solver_context.state_index_frame_rt(i),
                                         solver_context.state_index_frame_rt(i)+3) )
    return \
        pixel_uncertainty_stdev*pixel_uncertainty_stdev * \
        apply_slices(var_full, slices)

def sorted_eig(C):
    'like eig(), but the results are sorted by eigenvalue'
    l,v = np.linalg.eig(C)
    i = np.argsort(l)
    return l[i], v[:,i]
