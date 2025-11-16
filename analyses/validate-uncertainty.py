#!/usr/bin/env python3

# Copyright (c) 2017-2023 California Institute of Technology ("Caltech"). U.S.
# Government sponsorship acknowledged. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

r'''Study the uncertainty as a predictor of cross-validation diffs

SYNOPSIS

  $ validate-uncertainty.py  \
      [01].cameramodel

  ... plots pop up, showing the uncertainty prediction from the given models,
  ... and cross-validation diff obtained from creating perfect data corrupted
  ... ONLY with perfect gaussian noise. If the noise on the inputs was the ONLY
  ... source of error (what the uncertainty modeling expects), then the
  ... uncertainty plots would predict the cross-validation plots well

A big feature of mrcal is the ability to gauge the accuracy of the solved
intrinsics: by computing the projection uncertainty. This measures the
sensitivity of the solution to noise in the inputs. So using this as a measure
of calibration accuracy makes a core assumption: this input noise is the only
source of error. This assumption is often false, so cross-validation diffs can
be computed to sample the full set of error sources, not just this one.

Sometimes we see cross-validation results higher than what the uncertainties
promise us, and figuring out the reason can be challenging. This tool serves to
validate the techniques to help in that debugging.

'''

import sys
import argparse
import re
import os

def parse_args():

    parser = \
        argparse.ArgumentParser(description = __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--validate',
                        choices=('uncertainty',
                                 'input-noise',
                                 'noncentral'),
                        default = 'uncertainty',
                        help='''What we're testing: uncertainty predicting the
                        cross-validation samples, the input-noise-estimation
                        expressions, the noncentral assumption respectively''')

    parser.add_argument('--uncertainty-num-samples',
                        type = int,
                        default = 10,
                        help='''How many noisy-data samples to process with
                        --validate uncertainty''')

    parser.add_argument('--noncentral-mode',
                        choices=('too-close', 'too-far-from-center'),
                        default='too-close',
                        help='''What --validate noncentral does. "too-close": we
                        throw out some points that were too close to the camera.
                        "too-far-from-center": we throw out some points that
                        were observed too far from the imager center''')

    parser.add_argument('--noncentral-cull-percentile',
                        type=float,
                        default=10,
                        help='''The percentile of worst (too close, too far from
                        center of imager, ...) points to throw out for
                        --validate noncentral''')

    parser.add_argument('--observed-pixel-uncertainty',
                        type = float,
                        default = 0.3,
                        help='''How much noise to inject into perfect solves''')
    parser.add_argument('--gridn-width',
                        type = int,
                        default = 60,
                        help='''How densely we should sample the imager for the
                        uncertainty and cross-validation visualizations. Here we
                        just take the "width". The height will be automatically
                        computed based on the imager aspect ratio''')

    parser.add_argument('models',
                        type=str,
                        nargs='+',
                        help='''The camera models to process. '--validate
                        input-noise' and '--validate uncertainty' use each of
                        these individually. '--validate noncentral' requires an
                        even number of models, at least 2. If we are given
                        exactly two models, we use those two. If we are given
                        N>2 models, we join data from the first N/2 models and
                        the second N/2 models into two bigger sets of
                        calibration data, and we process those''')

    args = parser.parse_args()

    Nmodels = len(args.models)
    if args.validate == 'noncentral' and (Nmodels % 2):
        print("'--validate noncentral' requires an EVEN number of models",
              file=sys.stderr)
        sys.exit(1)

    return args

args = parse_args()



# I import the LOCAL mrcal
sys.path[:0] = f"{os.path.dirname(os.path.realpath(__file__))}/..",


import mrcal
import mrcal.model_analysis
import numpy as np
import numpysane as nps

import copy
import gnuplotlib as gp



def join_inputs(*optimization_inputs_all):
    r'''Combines multiple calibration datasets into one

Intrinsics from the first input'''

    if not all(o['intrinsics'].shape[-2] == 1 for o in optimization_inputs_all):
        raise Exception('Everything must be MONOCULAR chessboard observations')
    if not all(o.get('rt_cam_ref') is None or \
               o['rt_cam_ref'].size == 0 \
               for o in optimization_inputs_all):
        raise Exception('Everything must be monocular chessboard observations with a STATIONARY camera')
    if not all(o.get('points') is None or \
               o['points'].size == 0 \
               for o in optimization_inputs_all):
        raise Exception('Everything must be monocular CHESSBOARD observations')

    optimization_inputs = copy.deepcopy(optimization_inputs_all[0])

    optimization_inputs['rt_ref_frame'] = \
        nps.glue( *[ o['rt_ref_frame'] \
                     for o in optimization_inputs_all],
                  axis = -2 )

    if not all( not np.any( o['indices_frame_camintrinsics_camextrinsics'][:,0] - np.arange(len(o['rt_ref_frame']))) \
                for o in optimization_inputs_all ):
        raise Exception("I assume frame indices starting at 0 and incrementing by 1")

    Nobservations = \
        sum(len(o['indices_frame_camintrinsics_camextrinsics']) \
            for o in optimization_inputs_all)

    optimization_inputs['indices_frame_camintrinsics_camextrinsics'] = \
        np.zeros((Nobservations,3), dtype=np.int32)
    optimization_inputs['indices_frame_camintrinsics_camextrinsics'][:,0] = \
        np.arange(Nobservations, dtype=np.int32)
    optimization_inputs['indices_frame_camintrinsics_camextrinsics'][:,2] = -1

    optimization_inputs['observations_board'] = \
        nps.glue( *[ o['observations_board'] \
                     for o in optimization_inputs_all],
                  axis = -4 )

    optimization_inputs['imagepaths'] = \
        nps.glue( *[ o['imagepaths'] \
                     for o in optimization_inputs_all],
                  axis = -1 )

    return optimization_inputs


def apply_noise(optimization_inputs,
                *,
                observed_pixel_uncertainty):
    noise_nominal = \
        observed_pixel_uncertainty * \
        np.random.randn(*optimization_inputs['observations_board'][...,:2].shape)

    weight = nps.dummy( optimization_inputs['observations_board'][...,2],
                        axis = -1 )
    weight[ weight<=0 ] = 1. # to avoid dividing by 0

    optimization_inputs['observations_board'][...,:2] += \
        noise_nominal / weight


def validate_input_noise(model,
                         *,
                         observed_pixel_uncertainty):

    optimization_inputs = model.optimization_inputs()
    mrcal.make_perfect_observations(optimization_inputs,
                                    observed_pixel_uncertainty = observed_pixel_uncertainty)

    mrcal.optimize(**optimization_inputs)

    # The ratio of observed noise to what I expected. Should be ~ 1.0
    noise_observed_ratio = \
        mrcal.model_analysis._observed_pixel_uncertainty_from_inputs(optimization_inputs) / observed_pixel_uncertainty

    Nstates       = mrcal.num_states(**optimization_inputs)
    Nmeasurements = mrcal.num_measurements(**optimization_inputs)


    # This correction is documented here:
    #   https://mrcal.secretsauce.net/docs-3.0/formulation.html#estimating-input-noise
    # This probably should be added to
    # _observed_pixel_uncertainty_from_inputs(). Today (2025/11/15) it has not
    # yet been. Because very're usually VERY overdetermined, which can be
    # validated by this script. I will add this factor later, if I discover that this is necessary
    f = np.sqrt(1 - Nstates/Nmeasurements)
    noise_predicted_ratio = noise_observed_ratio/f

    print("Noise ratios measured/actual. Should be ~ 1.0")
    print(f"  observed, by looking at the distribution of residulas: {noise_observed_ratio:.3f}")
    print(f"  predicted, by correcting the above by sqrt(1-Nstates/Nmeasurements_observed): {noise_predicted_ratio:.3f}")


def validate_uncertainty(model,
                         *,
                         Nsamples,
                         observed_pixel_uncertainty):

    plots = []

    plots.append( \
        mrcal.show_projection_uncertainty(model,
                                          gridn_width = args.gridn_width,
                                          observed_pixel_uncertainty = observed_pixel_uncertainty,
                                          cbmax                      = 0.3,
                                          title                      = 'Baseline uncertainty with a perfect-noise solve')
                 )

    optimization_inputs_perfect = model.optimization_inputs()
    mrcal.make_perfect_observations(optimization_inputs_perfect,
                                    observed_pixel_uncertainty = 0)

    def model_sample():
        optimization_inputs = copy.deepcopy(optimization_inputs_perfect)
        apply_noise(optimization_inputs,
                    observed_pixel_uncertainty = observed_pixel_uncertainty)
        mrcal.optimize(**optimization_inputs)
        return mrcal.cameramodel(optimization_inputs = optimization_inputs,
                                 icam_intrinsics     = 0)


    model0 = model_sample()

    def diff_sample():
        model1 = model_sample()
        return \
            mrcal.show_projection_diff((model0,model1),
                                       gridn_width       = args.gridn_width,
                                       use_uncertainties = False,
                                       focus_radius      = 100,
                                       cbmax             = 1.,
                                       title             = 'Simulated cross-validation diff: comparing two perfectly-noised models')[0]

    plots.extend([ diff_sample() for i in range(Nsamples)])

    return plots


def validate_noncentral(models,
                        *,
                        percentile,
                        mode = 'too-close'):

    r'''Check for noncentral effects by re-optimizing without too-close points

I take the optimization_inputs as they are, WITHOUT making perfect data, and I
    re-solve the problem after throwing out points the percentile nearest
    points. If noncentrality was an issue, this new solve would match reality
    better, and the two solves would be closer to each other than the original poor
    cross-validation

    '''

    known_modes = set(('too-far-from-center',
                       'too-close'),)
    if mode not in known_modes:
        raise Exception(f"{mode=} must be in {known_modes=}")

    if mode == 'too-close':
        what_culling = f'{percentile}% nearest'
        what         = 'range'
        binwidth     = 0.01
        cull_nearest = True
    elif mode == 'too-far-from-center':
        what_culling = f'{percentile}% off-center'
        what         = 'pixel distance off-center'
        binwidth     = 20
        cull_nearest = False
        percentile   = 100 - percentile
    else:
        # can't happen; checked above
        raise

    def reoptimize(imodel, model):
        print('')

        optimization_inputs = model.optimization_inputs()
        observations_board  = optimization_inputs['observations_board']

        Noutliers = \
            np.count_nonzero(observations_board[...,2] <= 0)
        print(f"Before culling the {what_culling} points: {Noutliers=}")

        if mode == 'too-close':
            p = mrcal.hypothesis_board_corner_positions(**optimization_inputs)[0]
            r = nps.mag(p)

        elif mode == 'too-far-from-center':

            if not mrcal.lensmodel_metadata_and_config(model.intrinsics()[0])['has_core']:
                raise Exception("Here I'm assuming the model has an fxycxy core")
            qcenter = model.intrinsics()[1][2:4]
            r = nps.mag(observations_board[...,:2] - qcenter)

        else:
            # can't happen; checked above
            raise

        rthreshold = np.percentile(r.ravel(), percentile)
        print(f"{what.capitalize()} at {percentile}-th percentile: {rthreshold:.2f}")


        if False:
            # This is a "directions" plot off residuals, with a
            # range,qdiff_off_center domain. Hopefully I'll be able to see
            # model-error patterns off this

            x_board = mrcal.measurements_board(optimization_inputs)
            p       = mrcal.hypothesis_board_corner_positions(**optimization_inputs)[2]
            r       = nps.mag(p)

            qcenter = model.intrinsics()[1][2:4]
            idx_inliers = observations_board[...,2].ravel() > 0.
            qobs_off_center     = \
                nps.clump(observations_board[...,:2], n=3)[idx_inliers] - \
                qcenter
            mag_qobs_off_center = nps.mag(qobs_off_center)

            qobs_dir_off_center = np.array(qobs_off_center)
            # to avoid /0
            idx = mag_qobs_off_center>0
            qobs_dir_off_center[idx] /= nps.dummy(mag_qobs_off_center[idx],
                                                  axis = -1)
            x_board_radial_off_center = nps.inner(x_board, qobs_dir_off_center)

            th = 180./np.pi * np.arctan2(x_board[...,1], x_board[...,0])


            # hoping to see low-range points imply clustering in the residual
            # direction
            gp.plot( r,
                     mag_qobs_off_center,
                     th,
                     cbrange = [-180.,180.],
                     _with = 'points pt 7 palette',
                     _tuplesize = 3,
                     _set = 'palette defined ( 0 "#00ffff", 0.5 "#80ffff", 1 "#ffffff") model HSV')

            # Hoping to see low ranges imply a non-zero bias on x_board_radial_off_center
            gp.plot(r, x_board_radial_off_center, _with='points')


            import IPython
            IPython.embed()
            sys.exit()





        histogram = gp.gnuplotlib()
        histogram.plot(r.ravel(),
                       histogram = True,
                       binwidth  = binwidth,
                       _set = f'arrow from {rthreshold},graph 0 to {rthreshold},graph 1 nohead front',
                       title = f'Histogram of {what}, with the {what_culling} points marked: camera {imodel}')

        if cull_nearest:
            i_cull = r.ravel() < rthreshold
        else:
            i_cull = r.ravel() > rthreshold

        nps.clump(observations_board, n=3)[i_cull, 2] = -1

        Noutliers = \
            np.count_nonzero(observations_board[...,2] <= 0)
        print(f"After culling the {what_culling} points: {Noutliers=}")

        mrcal.optimize(**optimization_inputs)

        return (mrcal.cameramodel(optimization_inputs = optimization_inputs,
                                  icam_intrinsics     = 0),
                histogram)


    plots = []

    models_plots_reoptimized = [ reoptimize(i,m) for i,m in enumerate(models) ]
    models_reoptimized = [ m for (m,p) in models_plots_reoptimized ]
    plots.extend([ p for (m,p) in models_plots_reoptimized ])

    plots.extend( \
        [ mrcal.show_projection_uncertainty(m,
                                            gridn_width = args.gridn_width,
                                            cbmax       = 0.3,
                                            title       = f'Uncertainty after cutting the {what_culling} points: camera {i}') \
          for i,m in enumerate(models_reoptimized) ] )


    plots.extend( \
        [ mrcal.show_projection_diff((models[i],models_reoptimized[i]),
                                     gridn_width       = args.gridn_width,
                                     use_uncertainties = False,
                                     focus_radius      = 100,
                                     cbmax             = 1.,
                                     title             = f'Reoptimizing after cutting the {what_culling} points: resulting diff for camera {i}')[0] \
          for i in range(len(models)) ] )

    if len(models) != 2:
        print("WARNING: validate_noncentral() is intended to work with exactly two models. Got something different; not showing the new cross-validation diff")
    else:
        plots.extend( \
            [ mrcal.show_projection_diff((models[0],models[1]),
                                         gridn_width       = args.gridn_width,
                                         use_uncertainties = False,
                                         focus_radius      = 100,
                                         cbmax             = 1.,
                                         title             = f'Original, poor cross-validation diff')[0],
              mrcal.show_projection_diff((models_reoptimized[0],models_reoptimized[1]),
                                         gridn_width       = args.gridn_width,
                                         use_uncertainties = False,
                                         focus_radius      = 100,
                                         cbmax             = 1.,
                                         title             = f'Cross-validation diff after cutting the {what_culling} points')[0]
             ])

    return plots




models = [mrcal.cameramodel(f) for f in args.models]

if len(models) > 2:

    Nmodels = args.models

    o0 = join_inputs( *[models[i].optimization_inputs() for i in range(0,Nmodels//2)] )
    o1 = join_inputs( *[models[i].optimization_inputs() for i in range(Nmodels//2,Nmodels)] )

    mrcal.optimize(**o0)
    mrcal.optimize(**o1)
    models = ( mrcal.cameramodel(optimization_inputs = o0,
                                 icam_intrinsics     = 0),
               mrcal.cameramodel(optimization_inputs = o1,
                                 icam_intrinsics     = 0) )


if args.validate == 'input-noise':
    for model in models:
        validate_input_noise(model,
                             observed_pixel_uncertainty = args.observed_pixel_uncertainty)
    plots = None

if args.validate == 'uncertainty':
    plots = \
        [  plot \
           for model in models for plot in \
           validate_uncertainty(models[0],
                                Nsamples                   = args.uncertainty_num_samples,
                                observed_pixel_uncertainty = args.observed_pixel_uncertainty) ]

if args.validate == 'noncentral':
    plots = \
        validate_noncentral(models,
                            percentile = args.noncentral_cull_percentile,
                            mode       = args.noncentral_mode)

if plots:
    # Needs gnuplotlib >= 0.42
    gp.wait(*plots)
