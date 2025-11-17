#!/usr/bin/env python3

# Copyright (c) 2017-2023 California Institute of Technology ("Caltech"). U.S.
# Government sponsorship acknowledged. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

r'''Study the noncentral effects of this solve

SYNOPSIS

  $ validate-noncentral.py [01].cameramodel

  ... plots pop up, showing the effect of removing points that are close to the
  ... lens. If they were causing poor fits due to noncentrality, we'd see
  improved ... cross-validation

I take the optimization_inputs as they are, WITHOUT making perfect data, and I
re-solve the problem after throwing out points the percentile nearest
points. If noncentrality was an issue, this new solve would match reality
better, and the two solves would be closer to each other than the original poor
cross-validation

'''

import sys
import argparse
import re
import os

def parse_args():

    parser = \
        argparse.ArgumentParser(description = __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--mode',
                        choices=('too-close', 'too-far-from-center'),
                        default='too-close',
                        help='''What this tool does. "too-close": we throw out
                        some points that were too close to the camera.
                        "too-far-from-center": we throw out some points that
                        were observed too far from the imager center''')
    parser.add_argument('--cull-percentile',
                        type=float,
                        default=10,
                        help='''The percentile of worst (too close, too far from
                        center of imager, ...) points to throw out''')
    parser.add_argument('--gridn-width',
                        type = int,
                        default = 60,
                        help='''How densely we should sample the imager for the
                        uncertainty and cross-validation visualizations. Here we
                        just take the "width". The height will be automatically
                        computed based on the imager aspect ratio''')
    parser.add_argument('--cbmax-diff',
                        type=float,
                        help='''The max-color to use for the diff plots. If
                        omitted, we use the default in
                        mrcal.show_projection_diff()''')
    parser.add_argument('--cbmax-uncertainty',
                        type=float,
                        help='''The max-color to use for the uncertainty plots.
                        If omitted, we use the default in
                        mrcal.show_projection_uncertainty()''')
    parser.add_argument('--hardcopy',
                        type=str,
                        help='''If given, we write the output plots to this
                        path. This path is given as DIR/FILE.EXTENSION. Multiple
                        plots will be made, to DIR/FILE-thing.EXTENSION''')
    parser.add_argument('--terminal',
                        type=str,
                        help='''The gnuplot terminal to use for plots''')
    parser.add_argument('models',
                        type=str,
                        nargs='+',
                        help='''The camera models to process. This requires an
                        even number of models, at least 2. If we are given
                        exactly two models, we use those two. If we are given
                        N>2 models, we join data from the first N/2 models and
                        the second N/2 models into two bigger sets of
                        calibration data, and we process those''')

    args = parser.parse_args()

    Nmodels = len(args.models)
    if (Nmodels % 2):
        print("We require an EVEN number of models", file=sys.stderr)
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





percentile = args.cull_percentile
mode       = args.mode

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





kwargs_show_uncertainty = dict()
if args.cbmax_uncertainty is not None:
    kwargs_show_uncertainty['cbmax'] = args.cbmax_uncertainty
kwargs_show_diff = dict()
if args.cbmax_diff is not None:
    kwargs_show_diff['cbmax'] = args.cbmax_diff

if args.hardcopy is None:
    filename = None
else:
    hardcopy_base,hardcopy_extension = os.path.splitext(args.hardcopy)



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





    if args.hardcopy is not None:
        filename = f"{hardcopy_base}-histogram-measurements-cull-camera{imodel}{hardcopy_extension}"
    else:
        filename = None
    histogram = gp.gnuplotlib()
    histogram.plot(r.ravel(),
                   histogram = True,
                   binwidth  = binwidth,
                   _set = f'arrow from {rthreshold},graph 0 to {rthreshold},graph 1 nohead front',
                   title = f'Histogram of {what}, with the {what_culling} points marked: camera {imodel}',
                   hardcopy = filename,
                   terminal = args.terminal)
    if args.hardcopy is not None:
        print(f"Wrote '{filename}'")

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

for i,m in enumerate(models_reoptimized):
    if args.hardcopy is not None:
        filename = f"{hardcopy_base}-uncertainty-post-cull-camera{i}{hardcopy_extension}"
    plots.append( \
        mrcal.show_projection_uncertainty(
                      m,
                      gridn_width = args.gridn_width,
                      title       = f'Uncertainty after cutting the {what_culling} points: camera {i}',
                      hardcopy    = filename,
                      terminal    = args.terminal,
                      **kwargs_show_uncertainty) )
    if args.hardcopy is not None:
        print(f"Wrote '{filename}'")

for i in range(len(models)):
    if args.hardcopy is not None:
        filename = f"{hardcopy_base}-diff-from-cull-camera{i}{hardcopy_extension}"
    plots.append( \
        mrcal.show_projection_diff( \
                      (models[i],models_reoptimized[i]),
                      gridn_width       = args.gridn_width,
                      use_uncertainties = False,
                      focus_radius      = 100,
                      title             = f'Reoptimizing after cutting the {what_culling} points: resulting diff for camera {i}',
                      hardcopy          = filename,
                      terminal          = args.terminal,
                      **kwargs_show_diff)[0] )
    if args.hardcopy is not None:
        print(f"Wrote '{filename}'")


if len(models) != 2:
    print("WARNING: validate_noncentral() is intended to work with exactly two models. Got something different; not showing the new cross-validation diff")
else:
    if args.hardcopy is not None:
        filename = f"{hardcopy_base}-cross-validation-pre-cull-camera{i}{hardcopy_extension}"
    plots.append( \
          mrcal.show_projection_diff((models[0],models[1]),
                                     gridn_width       = args.gridn_width,
                                     use_uncertainties = False,
                                     focus_radius      = 100,
                                     title             = f'Original, poor cross-validation diff',
                                     hardcopy          = filename,
                                     terminal          = args.terminal,
                                     **kwargs_show_diff)[0])
    if args.hardcopy is not None:
        print(f"Wrote '{filename}'")

    if args.hardcopy is not None:
        filename = f"{hardcopy_base}-cross-validation-post-cull-camera{i}{hardcopy_extension}"
    plots.append( \
          mrcal.show_projection_diff((models_reoptimized[0],models_reoptimized[1]),
                                     gridn_width       = args.gridn_width,
                                     use_uncertainties = False,
                                     focus_radius      = 100,
                                     title             = f'Cross-validation diff after cutting the {what_culling} points',
                                     hardcopy          = filename,
                                     terminal          = args.terminal,
                                     **kwargs_show_diff)[0])
    if args.hardcopy is not None:
        print(f"Wrote '{filename}'")

# Needs gnuplotlib >= 0.42
gp.wait(*plots)
