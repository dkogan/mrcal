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

  $ validate-uncertainty.py camera.cameramodel

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

    parser.add_argument('--num-samples',
                        type = int,
                        default = 16,
                        help='''How many noisy-data samples to process''')

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
    parser.add_argument('model',
                        type=str,
                        help='''The camera model to process''')

    return parser.parse_args()

args = parse_args()



# I import the LOCAL mrcal
sys.path[:0] = f"{os.path.dirname(os.path.realpath(__file__))}/..",


import mrcal
import mrcal.model_analysis
import numpy as np
import numpysane as nps
import gnuplotlib as gp
import copy

# Today I hardcode this. I'm mostly thinking of monocular chessboard solves
# only. Presumably other behave the same way
icam_intrinsics = 0





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



model = mrcal.cameramodel(args.model)
observed_pixel_uncertainty = args.observed_pixel_uncertainty

plots = []

plots.append( \
    mrcal.show_projection_uncertainty(model,
                                      gridn_width = args.gridn_width,
                                      observed_pixel_uncertainty = observed_pixel_uncertainty,
                                      cbmax                      = 0.3,
                                      title                      = f'Baseline uncertainty with {observed_pixel_uncertainty=}') )

optimization_inputs_perfect = model.optimization_inputs()
mrcal.make_perfect_observations(optimization_inputs_perfect,
                                observed_pixel_uncertainty = 0)


def model_sample():
    optimization_inputs = copy.deepcopy(optimization_inputs_perfect)
    apply_noise(optimization_inputs,
                observed_pixel_uncertainty = observed_pixel_uncertainty)
    mrcal.optimize(**optimization_inputs)
    return mrcal.cameramodel(optimization_inputs = optimization_inputs,
                             icam_intrinsics     = icam_intrinsics)

model0  = model_sample()
models1 = [model_sample() for _ in range(args.num_samples)]

plots.append( \
    mrcal.show_projection_uncertainty(model0,
                                      gridn_width = args.gridn_width,
                                      observed_pixel_uncertainty = observed_pixel_uncertainty,
                                      cbmax                      = 0.3,
                                      title                      = 'Uncertainty with perfect observations + noise; should be very close to baseline') )


def gnuplotlib_normalize_options_dict(d):
    d2 = {}
    for key in d:
        gp.add_plot_option(d2, key, d[key])
    return d2

gridn_plot = int( np.ceil(np.sqrt(len(models1)) ) )

diff_multiplot_args = []
for model1 in models1:
    data_tuples,plot_options = \
        mrcal.show_projection_diff((model0,model1),
                                   gridn_width       = args.gridn_width,
                                   use_uncertainties = False,
                                   focus_radius      = 100,
                                   cbmax             = 1.,
                                   title             = '',
                                   unset=('key',
                                          'xtics',
                                          'ytics',),
                                   contour_labels_styles = None, # no label
                                   _set=('lmargin 0',
                                         'tmargin 0',
                                         'rmargin 0',
                                         'bmargin 0',
                                         ),
                                   return_plot_args  = True)[0]
    plot_options = gnuplotlib_normalize_options_dict(plot_options)
    subplot_options = dict()
    for o in plot_options:
        if o in gp.knownSubplotOptions:
            subplot_options[o] = plot_options[o]
    diff_multiplot_args.append( (*data_tuples,subplot_options) )

# massage one of the plot_options from the subplots; they're probably all the same
_plot_options = copy.copy(plot_options)
plot_options = dict()
for o in _plot_options:
    if o not in gp.knownSubplotOptions:
        plot_options[o] = _plot_options[o]
diff_multiplot = gp.gnuplotlib( title     = 'Simulated cross-validation diff samples comparing two perfectly-noised models',
                                multiplot = f'layout {gridn_plot},{gridn_plot}',
                                **plot_options )
diff_multiplot.plot( *diff_multiplot_args )
plots.append(diff_multiplot)

plots.append( mrcal.show_projection_diff([model0, *models1],
                                         gridn_width       = args.gridn_width,
                                         use_uncertainties = False,
                                         focus_radius      = 100,
                                         cbmax             = 1.,
                                         title             = 'Simulated cross-validation diff: stdev of of ALL the samples')[0] )

# Needs gnuplotlib >= 0.42
gp.wait(*plots)
