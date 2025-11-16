#!/usr/bin/env python3

# Copyright (c) 2017-2023 California Institute of Technology ("Caltech"). U.S.
# Government sponsorship acknowledged. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

r'''Validate deriving the input noise from the solve residuals

SYNOPSIS

  $ validate-input-noise.py [01].cameramodel

  Noise ratios measured/actual. Should be ~ 1.0
    observed, by looking at the distribution of residulas: 0.998
    predicted, by correcting the above by sqrt(1-Nstates/Nmeasurements_observed): 1.000


'''

import sys
import argparse
import re
import os

def parse_args():

    parser = \
        argparse.ArgumentParser(description = __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--observed-pixel-uncertainty',
                        type = float,
                        default = 0.3,
                        help='''How much noise to inject into perfect solves''')
    parser.add_argument('models',
                        type=str,
                        nargs='+',
                        help='''The camera models to process. Each is handled
                        individually''')

    return parser.parse_args()

args = parse_args()



# I import the LOCAL mrcal
sys.path[:0] = f"{os.path.dirname(os.path.realpath(__file__))}/..",


import mrcal
import mrcal.model_analysis
import numpy as np
import numpysane as nps


models = [mrcal.cameramodel(f) for f in args.models]

for model in models:

    optimization_inputs = model.optimization_inputs()
    mrcal.make_perfect_observations(optimization_inputs,
                                    observed_pixel_uncertainty = args.observed_pixel_uncertainty)

    mrcal.optimize(**optimization_inputs)

    # The ratio of observed noise to what I expected. Should be ~ 1.0
    noise_observed_ratio = \
        mrcal.model_analysis._observed_pixel_uncertainty_from_inputs(optimization_inputs) / args.observed_pixel_uncertainty

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
