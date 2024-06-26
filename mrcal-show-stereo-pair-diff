#!/usr/bin/env python3

# Copyright (c) 2017-2023 California Institute of Technology ("Caltech"). U.S.
# Government sponsorship acknowledged. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

r'''Visualize the difference in projection between N models

SYNOPSIS

  $ mrcal-show-stereo-pair-diff before.cameramodel after.cameramodel
  ... a plot pops up showing how these two models differ in their projections

The operation of this tool is documented at

  https://mrcal.secretsauce.net/differencing.html

This tool visualizes the results of mrcal.stereo_pair_diff()

It is often useful to compare the projection behavior of two camera models. For
instance, one may want to validate a calibration by comparing the results of two
different chessboard dances. Or one may want to evaluate the stability of the
intrinsics in response to mechanical or thermal stresses. This tool makes these
comparisons, and produces a visualization of the results.

In the most common case we're given exactly 2 models to compare. We then display
the projection difference as either a vector field or a heat map. If we're given
more than 2 models, then a vector field isn't possible and we instead display as
a heatmap the standard deviation of the differences between models 1..N and
model0.

The top-level operation of this tool:

- Grid the imager
- Unproject each point in the grid using one camera model
- Apply a transformation to map this point from one camera's coord system to the
  other. How we obtain this transformation is described below
- Project the transformed points to the other camera
- Look at the resulting pixel difference in the reprojection

The details of how the comparison is computed, and the meaning of the arguments
controlling this, are in the docstring of mrcal.stereo_pair_diff().

'''

import sys
import argparse
import re
import os

def parse_args():

    parser = \
        argparse.ArgumentParser(description = __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--gridn',
                        type=int,
                        default = (60,40),
                        nargs = 2,
                        help='''How densely we should sample the imager. By default we use a 60x40 grid''')
    parser.add_argument('--distance',
                        type=float,
                        help='''The projection difference varies depending on
                        the range to the observed world points, with the queried
                        range set in this argument. If omitted we look out to
                        infinity.''')

    parser.add_argument('--observations',
                        action='store_true',
                        default=False,
                        help='''If given, I show where the chessboard corners were observed at calibration
                        time. These should correspond to the low-diff regions.''')
    parser.add_argument('--valid-intrinsics-region',
                        action='store_true',
                        default=False,
                        help='''If given, I overlay the valid-intrinsics regions onto the plot''')
    parser.add_argument('--cbmax',
                        type=float,
                        default=4,
                        help='''Maximum range of the colorbar''')

    parser.add_argument('--title',
                        type=str,
                        default = None,
                        help='''Title string for the plot. Overrides the default
                        title. Exclusive with --extratitle''')
    parser.add_argument('--extratitle',
                        type=str,
                        default = None,
                        help='''Additional string for the plot to append to the
                        default title. Exclusive with --title''')

    parser.add_argument('--vectorfield',
                        action = 'store_true',
                        default = False,
                        help='''Plot the diff as a vector field instead of as a heat map. The vector field
                        contains more information (magnitude AND direction), but
                        is less clear at a glance''')

    parser.add_argument('--vectorscale',
                        type = float,
                        default = 1.0,
                        help='''If plotting a vectorfield, scale all the vectors by this factor. Useful to
                        improve legibility if the vectors are too small to
                        see''')

    parser.add_argument('--directions',
                        action = 'store_true',
                        help='''If given, the plots are color-coded by the direction of the error, instead of
                        the magnitude''')

    parser.add_argument('--hardcopy',
                        type=str,
                        help='''Write the output to disk, instead of making an interactive plot''')
    parser.add_argument('--terminal',
                        type=str,
                        help=r'''gnuplotlib terminal. The default is good almost always, so most people don't
                        need this option''')
    parser.add_argument('--set',
                        type=str,
                        action='append',
                        help='''Extra 'set' directives to gnuplotlib. Can be given multiple times''')
    parser.add_argument('--unset',
                        type=str,
                        action='append',
                        help='''Extra 'unset' directives to gnuplotlib. Can be given multiple times''')

    parser.add_argument('models',
                        type=str,
                        nargs=4,
                        help='''Camera models to diff''')

    args = parser.parse_args()

    if len(args.models) < 4:
        print(f"I need at least 4 models to diff. Instead got '{args.models}'", file=sys.stderr)
        sys.exit(1)

    if args.title      is not None and \
       args.extratitle is not None:
        print("--title and --extratitle are exclusive", file=sys.stderr)
        sys.exit(1)

    return args

args = parse_args()

# arg-parsing is done before the imports so that --help works without building
# stuff, so that I can generate the manpages and README

if args.vectorscale != 1.0 and not args.vectorfield:
    print("Error: --vectorscale only makes sense with --vectorfield",
          file = sys.stderr)
    sys.exit(1)

# if len(args.models) > 2:
#     if args.vectorfield:
#         print("Error: --vectorfield works only with exactly 2 models",
#               file = sys.stderr)
#         sys.exit(1)
#     if args.directions:
#         print("Error: --directions works only with exactly 2 models",
#               file = sys.stderr)
#         sys.exit(1)


import mrcal
import numpy as np
import numpysane as nps


plotkwargs_extra = {}
if args.set is not None:
    plotkwargs_extra['set'] = args.set
if args.unset is not None:
    plotkwargs_extra['unset'] = args.unset

if args.title is not None:
    plotkwargs_extra['title'] = args.title
if args.extratitle is not None:
    plotkwargs_extra['extratitle'] = args.extratitle

def openmodel(f):
    try:
        return mrcal.cameramodel(f)
    except Exception as e:
        print(f"Couldn't load camera model '{f}': {e}",
              file=sys.stderr)
        sys.exit(1)

models = [openmodel(modelfilename) for modelfilename in args.models]

model_pairs = [ (models[i0], models[i0+1]) \
                for i0 in range(0,len(models),2)]


# if args.observations:
#     optimization_inputs = [ model_pair[0].optimization_inputs() for model_pair in model_pairs ]
#     if any( oi is None for oi in optimization_inputs ):
#         print("mrcal-show-stereo-pair-diff --observations requires optimization_inputs to be available for all models, but this is missing for some models",
#               file=sys.stderr)
#         sys.exit(1)

plot = mrcal.show_stereo_pair_diff(model_pairs,
                                   gridn_width             = args.gridn[0],
                                   gridn_height            = args.gridn[1],
                                   observations            = args.observations,
                                   valid_intrinsics_region = args.valid_intrinsics_region,
                                   distance                = args.distance,
                                   vectorfield             = args.vectorfield,
                                   vectorscale             = args.vectorscale,
                                   hardcopy                = args.hardcopy,
                                   terminal                = args.terminal,
                                   cbmax                   = args.cbmax,
                                   **plotkwargs_extra)

if args.hardcopy is None:
    plot.wait()
