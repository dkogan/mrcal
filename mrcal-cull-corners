#!/usr/bin/env python3

# Copyright (c) 2017-2023 California Institute of Technology ("Caltech"). U.S.
# Government sponsorship acknowledged. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

r'''Filters a corners.vnl on stdin to cut out some points

SYNOPSIS

  $ < corners.vnl mrcal-cull-corners --cull-left-of 1000 > corners.culled.vnl

This tool reads a set of corner detections on stdin, throws some of them out,
and writes the result to stdout. This is useful for testing and evaluating the
performance of the mrcal calibration tools.

The specific operation of this tool is defined on which --cull-... option is
given. Exactly one is required:

  --cull-left-of X: throw away all corner observations to the left of the given
    X coordinate

  --cull-rad-off-center D: throw away all corner observations further than D
    away from the center. --imagersize or --where must be given also so that we
    know where the center is. If D < 0: we cull the points -D or closer to the
    corners: we use a radius of sqrt(width^2 + height^2)/2. - abs(D)

  --cull-random-observations-ratio R: throws away a ratio R object observations
    at random. To throw out half of all object observations, pass R = 0.5.
    --object-width-n and --object-height-n are then required to make the parsing
    work

--cull-left-of X and --cull-rad-off-center throw out individual points. This is
  done by keeping the point in the output data stream, but setting its
  decimation level to '-'. The downstream tools then know to ignore those points

--cull-random-observations-ratio throws out whole object observations, not just
  individual points. These removed observations do not appear in the output data
  stream at all


This tool exists primarily for testing, and probably you don't want to use it.
The filtering is crude, and the tool might report chessboard observations with
very few remaining points. You PROBABLY want to post-process the output to keep
only observations with enough points. For instance:

  mrcal-cull-corners ... > culled-raw.vnl

  vnl-join --vnl-sort - -j filename culled-raw.vnl \
     <(< culled-raw.vnl vnl-filter -p filename --has level |
                        vnl-uniq -c |
                        vnl-filter 'count > 20' -p filename ) \
  > culled.vnl

'''


import sys
import argparse
import re
import os

def parse_args():

    parser = \
        argparse.ArgumentParser(description = __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--object-width-n',
                        type=int,
                        help='''How many points the calibration board has per horizontal side. This is required
                        if --cull-random-observation-ratio''')
    parser.add_argument('--object-height-n',
                        type=int,
                        help='''How many points the calibration board has per vertical side. If omitted, I
                        assume a square object and use the same value as --object-width-n''')
    parser.add_argument('--imagersize',
                        nargs=2,
                        type=int,
                        help='''Size of the imager. If --cull-rad-off-center is
                        given: we require --imagersize or --where''')

    parser.add_argument('--cull-left-of',
                        required=False,
                        type=float,
                        help='''Throw out all observations with x < the given value. Exclusive with the other
                        --cull-... options''')
    parser.add_argument('--cull-rad-off-center',
                        required=False,
                        type=float,
                        help='''Throw out all points that appear outside the
                        given radius from the --where of the imager. Exclusive
                        with the other --cull-... options. If given: we require
                        --imagersize or --where''')
    parser.add_argument('--cull-rad-off-center-board',
                        required=False,
                        type=float,
                        help='''Throw out all observations beyond the given
                        radius from the center of the board. The units are
                        "board squares". Requires --objects-width-n''')
    parser.add_argument('--cull-random-observations-ratio',
                        required=False,
                        type=float,
                        help='''Throw out a random number of board observations.
                        The ratio of observations is given as the argument. 1.0
                        = throw out ALL the observations; 0.0 = throw out NONE
                        of the observations. Exclusive with the other --cull-...
                        options. Requires --objects-width-n''')
    parser.add_argument('--where',
                        type=float,
                        nargs=2,
                        help='''Used with --cull-rad-off-center. Specifies the
                        location of the "center" point. If omitted, we use the
                        center of the imager. May NOT be given if
                        --cull-rad-off-center < 0. If --cull-rad-off-center is
                        given: we require --imagersize or --where''')
    parser.add_argument('--filename',
                        type=str,
                        action='append',
                        help='''Apply the filtering only to observations where
                        the filename matches the given regex. May be given
                        multiple times: filenames that match ANY of the given
                        regexen are culled. If omitted, we cull ALL the
                        observations. Exclusive with
                        --cull-random-observations-ratio''')
    return parser.parse_args()

args = parse_args()

Nculloptions = 0
if args.cull_left_of                   is not None: Nculloptions += 1
if args.cull_rad_off_center            is not None: Nculloptions += 1
if args.cull_rad_off_center_board      is not None: Nculloptions += 1
if args.cull_random_observations_ratio is not None: Nculloptions += 1

if Nculloptions != 1:
    print("Exactly one --cull-... option must be given", file=sys.stderr)
    sys.exit(1)

if args.object_width_n is not None and args.object_height_n is None:
    args.object_height_n = args.object_width_n
if args.cull_random_observations_ratio or args.cull_rad_off_center_board:
    if (args.object_width_n is None or args.object_height_n is None):
        print("--cull-random-observation-ratio and --cull-rad-off-center-board requires --object-width-n",
              file=sys.stderr)
        sys.exit(1)
if args.cull_random_observations_ratio:
    if args.filename is not None:
        print("--cull-random-observation-ratio is exclusive with --filename",
              file=sys.stderr)
        sys.exit(1)

if args.cull_rad_off_center is not None:
    if args.imagersize is None and args.where is None:
        print("--cull-rad-off-center requires --imagersize or --where", file=sys.stderr)
        sys.exit(1)
    if args.cull_rad_off_center < 0 and args.where is not None:
        print("--cull-rad-off-center < 0 may not be given with --where", file=sys.stderr)
        sys.exit(1)




import re
import numpy as np
import numpysane as nps
import mrcal
import time


print(f"## generated on {time.strftime('%Y-%m-%d %H:%M:%S')} with   {' '.join(mrcal.shellquote(s) for s in sys.argv)}")

if args.cull_left_of              is not None or \
   args.cull_rad_off_center       is not None or \
   args.cull_rad_off_center_board is not None:
    # Simple file parsing.

    if args.filename is None:
        filename_filter = None
    else:
        filename_filter_string = '|'.join(f"(?:{s})" for s in args.filename)
        filename_filter = re.compile(filename_filter_string)

    if args.cull_rad_off_center is not None:
        if args.cull_rad_off_center >= 0:
            if args.where is None:
                c  = (np.array(args.imagersize, dtype=float) - 1.) / 2.
            else:
                c = np.array(args.where, dtype=float)
            r = args.cull_rad_off_center
        else:
            dims = np.array(args.imagersize, dtype=float)
            c  = (dims - 1.) / 2.
            r = nps.mag(dims)/2. + args.cull_rad_off_center

    if args.cull_rad_off_center_board is not None:
        cboard = (np.array((args.object_width_n, args.object_height_n), dtype=float) - 1.) / 2.


    for l in sys.stdin:
        if re.match(r'\s*(?:##|$)',l):
            sys.stdout.write(l)
            continue

        if l == '# filename x y level\n':
            sys.stdout.write(l)
            break

        print("This tool REQUIRES a vnlog with legend matching EXACTLY '# filename x y level'. Giving up", file=sys.stderr)
        sys.exit(1)

    xy_pt            = np.array((0,0), dtype=int)
    filename_current = ''

    for l in sys.stdin:
        if re.match(r'\s*(?:#|$)',l):
            sys.stdout.write(l)
            continue

        f = l.split()
        if f[1] == '-':
            sys.stdout.write(l)
            continue

        if filename_filter is not None and \
           not filename_filter.search(f[0]):
            sys.stdout.write(l)
            continue

        if args.cull_left_of        is not None and float(f[1]) > args.cull_left_of:
            sys.stdout.write(l)
            continue
        if args.cull_rad_off_center is not None and \
           nps.norm2(np.array((float(f[1]), float(f[2]))) - c) < r*r:
            sys.stdout.write(l)
            continue
        if args.cull_rad_off_center_board is not None:
            if filename_current != f[0]:
                filename_current = f[0]
                xy_pt *= 0

            accept_pt = nps.norm2(xy_pt - cboard) < args.cull_rad_off_center_board*args.cull_rad_off_center_board

            xy_pt[0] += 1
            if xy_pt[0] == args.object_width_n:
                xy_pt[0] = 0
                xy_pt[1] += 1
            if accept_pt:
                sys.stdout.write(l)
                continue

        # Cull the point!
        f[3] = '-'
        sys.stdout.write(' '.join(f) + '\n')

    sys.exit()



# --cull-random-observation-ratio

observations, _, paths = \
    mrcal.compute_chessboard_corners(W                  = args.object_width_n,
                                     H                  = args.object_height_n,
                                     corners_cache_vnl  = sys.stdin,
                                     weight_column_kind = 'weight')

N_keep       = int(round((1.0 - args.cull_random_observations_ratio) * len(observations)))
indices_keep = np.sort(np.random.choice(len(observations), N_keep, replace=False))

# shape (N, Nh,Nw,3)
observations = observations[indices_keep]
paths        = [paths[i] for i in indices_keep]

# shape (N, Nh*Nw, 3)
observations = nps.mv( nps.clump(nps.mv(observations, -1,-3),
                                 n = -2),
                       -2, -1)

# I cut out the data. Now I reconstitute the corners.vnl
print('# filename x y level')
for i in range(len(paths)):
    path   = paths[i]
    for j in range(observations.shape[-2]):
        l = observations[i][j][2]
        if l < 0: l = '-'
        else:     l = int(l)
        print(f"{path} {observations[i][j][0]} {observations[i][j][1]} {l}")
