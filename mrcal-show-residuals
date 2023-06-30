#!/usr/bin/env python3

# Copyright (c) 2017-2023 California Institute of Technology ("Caltech"). U.S.
# Government sponsorship acknowledged. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

r'''Visualize calibration residuals in an imager

SYNOPSIS

  $ mrcal-show-residuals --vectorfield left.cameramodel

  ... a plot pops up showing the vector field of residuals for this camera

This tool supports several different modes, selected by the commandline
arguments. Exactly one of these mode options must be given:

  --vectorfield Visualize the optimized residuals as a vector field. Each
                vector runs from the observed chessboard corner to its
                prediction at the optimal solution.

  --magnitudes  Visualize the optimized residual magnitudes as color-coded
                points. Similar to --vectorfield, but instead of a vector, each
                residual is plotted as a colored circle, coded by the MAGNITUDE
                of the error. This is usually more legible.

  --directions  Visualize the optimized residual directions as color-coded
                points. Similar to --vectorfield, but instead of a vector, each
                residual is plotted as a colored circle, coded by the DIRECTION
                of the error. This is very useful in detecting biases caused by
                a poorly-fitting lens model: these show up as clusters of
                similar color, instead of a random distribution.

  --regional    Visualize the optimized residuals, broken up by region. The imager
                of a camera is subdivided into bins. The residual statistics are
                then computed for each bin separately. We can then clearly see
                areas of insufficient data (observation counts will be low). And
                we can clearly see lens-model-induced biases (non-zero mean) and
                we can see heteroscedasticity (uneven standard deviation). The
                mrcal-calibrate-cameras tool uses these metrics to construct a
                valid-intrinsics region for the models it computes. This serves as
                a quick/dirty method of modeling projection reliability, which can
                be used even if projection uncertainty cannot be computed.

  --histogram   Visualize the distribution of the optimized residuals. We display
                a histogram of residuals and overlay it with an idealized
                gaussian distribution.

'''

import sys
import argparse
import re
import os

def parse_args():

    parser = \
        argparse.ArgumentParser(description = __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    mode_group = parser.add_mutually_exclusive_group(required = True)
    mode_group.add_argument('--vectorfield',
                            action='store_true',
                            help='''Visualize the optimized residuals as a vector field''')
    mode_group.add_argument('--magnitudes',
                            action='store_true',
                            help='''Visualize the optimized residual magnitudes as color-coded points''')
    mode_group.add_argument('--directions',
                            action='store_true',
                            help='''Visualize the optimized residual directions as color-coded points''')
    mode_group.add_argument('--regional',
                            action='store_true',
                            help='''Visualize the optimized residuals, broken up by region''')
    mode_group.add_argument('--histogram',
                            action='store_true',
                            help='''Visualize the distribution of the optimized residuals''')
    mode_group.add_argument('--histogram-this-camera',
                            action='store_true',
                            help='''If given, we show the histogram for residuals for THIS camera only. Otherwise
                            (by default) we display the residuals for all the
                            cameras in this solve. Implies --histogram''')


    parser.add_argument('--valid-intrinsics-region',
                        action='store_true',
                        help='''If given, I overlay the valid-intrinsics regions onto the plot. Applies to
                        all the modes except --histogram''')
    parser.add_argument('--gridn',
                        type=int,
                        default = (20,14),
                        nargs = 2,
                        help='''How densely we should bin the imager. By default we use a 20x14 grid of bins.
                        Applies only if --regional''')
    parser.add_argument('--binwidth',
                        type=float,
                        default=0.02,
                        help='''The width of binds used for the histogram. Defaults to 0.02 pixels. Applies
                        only if --histogram''')
    parser.add_argument('--vectorscale',
                        type = float,
                        default = 1.0,
                        help='''Scale all the vectors by this factor. Useful to improve legibility if the
                        vectors are too small to see. Applies only if --vectorfield''')
    parser.add_argument('--cbmax',
                        type=float,
                        help='''Maximum range of the colorbar used in
                        --vectorfield, --magnitudes. If omitted, we autoscale''')
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
    parser.add_argument('--hardcopy',
                        type=str,
                        help='''Write the output to disk, instead of making an
                        interactive plot. If --regional, then several plots are
                        made, and the --hardcopy argument is a base name:
                        "--hardcopy /a/b/c/d.pdf" will produce plots in
                        "/a/b/c/d.XXX.pdf" where XXX is the type of plot being
                        made''')
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

    parser.add_argument('model',
                        type=str,
                        help=r'''Camera model that contains the optimization_inputs that describe the solve.
                        The displayed observations may come from ANY of the
                        cameras in the solve, not necessarily the one given by
                        this model''')

    args = parser.parse_args()

    if args.histogram_this_camera:
        args.histogram = True

    if args.title      is not None and \
       args.extratitle is not None:
        print("--title and --extratitle are exclusive", file=sys.stderr)
        sys.exit(1)

    return args

args = parse_args()

# arg-parsing is done before the imports so that --help works without building
# stuff, so that I can generate the manpages and README


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

try:
    model = mrcal.cameramodel(args.model)
except Exception as e:
    print(f"Couldn't load camera model '{args.model}': {e}", file=sys.stderr)
    sys.exit(1)

optimization_inputs = model.optimization_inputs()
if optimization_inputs is None:
    print(f"Camera model '{args.model}' does not contain optimization inputs. Residuals aren't available", file=sys.stderr)
    sys.exit(1)

residuals = mrcal.optimizer_callback(**optimization_inputs)[1]

if not args.hardcopy:
    plotkwargs_extra['wait'] = True


kwargs = dict(residuals   = residuals,
              hardcopy    = args.hardcopy,
              terminal    = args.terminal,
              **plotkwargs_extra)

if args.vectorfield:
    mrcal.show_residuals_vectorfield(model,
                                     vectorscale = args.vectorscale,
                                     valid_intrinsics_region = args.valid_intrinsics_region,
                                     cbmax                   = args.cbmax,
                                     **kwargs)

elif args.magnitudes:
    mrcal.show_residuals_magnitudes(model,
                                    valid_intrinsics_region = args.valid_intrinsics_region,
                                    cbmax                   = args.cbmax,
                                    **kwargs)

elif args.directions:
    mrcal.show_residuals_directions(model,
                                    valid_intrinsics_region = args.valid_intrinsics_region,
                                    **kwargs)

elif args.histogram:
    mrcal.show_residuals_histogram(optimization_inputs,
                                   icam_intrinsics = model.icam_intrinsics() if args.histogram_this_camera else None,
                                   binwidth = args.binwidth,
                                   **kwargs)

elif args.regional:

    plotargs = \
        mrcal.show_residuals_regional(model,
                                      valid_intrinsics_region = args.valid_intrinsics_region,
                                      gridn_width             = args.gridn[0],
                                      gridn_height            = args.gridn[1],
                                      return_plot_args        = args.hardcopy is None,
                                      **kwargs)

    if args.hardcopy:
        sys.exit(0)

    # This produces 3 plots. I want to pop up the 3 separate windows at the same
    # time, and I want to exit when all 3 are done
    import gnuplotlib as gp

    pids = [0,0,0]
    for i in range(3):
        pid = os.fork()
        if pid == 0:
            # child
            # make the plot, and wait for it to be closed by the user
            (data_tuples,plot_options) = plotargs[i]
            p = gp.gnuplotlib(**plot_options)
            p.plot(*data_tuples)
            sys.exit()

        # parent
        pids[i] = pid

    for i in range(3):
        os.waitpid(pids[i], 0)
    sys.exit()



else:
    print("Unknown mode. Exactly one of the mutually-exclusive variables should have gotten through the argument parser. Is this an argparse bug?",
          file=sys.stderr)
    sys.exit(1)
