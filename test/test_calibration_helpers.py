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

    optimization_inputs = \
        dict( intrinsics                                = intrinsics,
              extrinsics_rt_fromref                     = extrinsics_rt_fromref,
              frames_rt_toref                           = frames_rt_toref,
              points                                    = None,
              observations_board                        = observations,
              indices_frame_camintrinsics_camextrinsics = indices_frame_camintrinsics_camextrinsics,
              observations_point                        = None,
              indices_point_camintrinsics_camextrinsics = None,
              lensmodel                                 = lensmodel,
              calobject_warp                            = calobject_warp,
              imagersizes                               = imagersizes,
              calibration_object_spacing                = object_spacing,
              calibration_object_width_n                = object_width_n,
              calibration_object_height_n               = object_height_n,
              verbose                                   = False,
              observed_pixel_uncertainty                = pixel_uncertainty_stdev,
              do_optimize_frames                        = do_optimize_frames,
              do_optimize_intrinsic_core                = do_optimize_intrinsic_core,
              do_optimize_intrinsic_distortions         = do_optimize_intrinsic_distortions,
              do_optimize_extrinsics                    = do_optimize_extrinsics,
              do_optimize_calobject_warp                = do_optimize_calobject_warp,
              skip_regularization                       = skip_regularization,
              **kwargs)

    stats = mrcal.optimize(**optimization_inputs,
                           skip_outlier_rejection = skip_outlier_rejection)

    p_packed = stats['p_packed']

    return \
        intrinsics, extrinsics_rt_fromref, frames_rt_toref, calobject_warp,   \
        observations[...,2] < 0.0, \
        p_packed, stats['x'], stats['rms_reproj_error__pixels'], \
        optimization_inputs

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

def sorted_eig(C):
    'like eig(), but the results are sorted by eigenvalue'
    l,v = np.linalg.eig(C)
    i = np.argsort(l)
    return l[i], v[:,i]
