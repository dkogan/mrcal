#!/usr/bin/env python3

# Copyright (c) 2017-2023 California Institute of Technology ("Caltech"). U.S.
# Government sponsorship acknowledged. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

r'''Visualize the valid-intrinsics region

SYNOPSIS

  $ mrcal-show-valid-intrinsics-region --write-image --image image.png left.cameramodel
  Wrote image-valid-intrinsics-region.png

Given a camera model (or models), this tool displays the valid-intrinsics
region(s). All the given models MUST contain a valid-intrinsics region. Empty
regions are handled properly.

If an image is given, the region is rendered overlaid onto the image.

If --points then we also read x,y points from STDIN, and plot those too.

By default, we use gnuplotlib to make an interactive plot. Alternately, pass
--write-image to annotate a given image, and write the new image on disk.

--write-image is not supported together with --points.

'''


import sys
import argparse
import re
import os

def parse_args():

    parser = \
        argparse.ArgumentParser(description = __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--write-image',
                        action='store_true',
                        help='''By default I make a plot. If --write-image is
                        given, I output an annotated image instead. The image to
                        annotate is then required, and is given in --image''')
    parser.add_argument('--force', '-f',
                        action='store_true',
                        default=False,
                        help='''With --write-image we refuse to overwrite any existing images. Pass --force to
                        allow overwriting''')
    parser.add_argument('--points',
                        action='store_true',
                        help='''If given, I read a set of xy points from STDIN, and include them in the plot.
                        This applies ONLY if not --write-image''')
    parser.add_argument('--title',
                        type=str,
                        default = None,
                        help='''Extra title string for the plot''')
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
    parser.add_argument('--image',
                        type=str,
                        required = False,
                        help='''Image to annotate. Used both with and without --write-image''')
    parser.add_argument('models',
                        type=str,
                        nargs='+',
                        help='''Input camera model(s)''')

    return parser.parse_args()

args = parse_args()

# arg-parsing is done before the imports so that --help works without building
# stuff, so that I can generate the manpages and README


if args.write_image:
    if args.image is None:
        print("--write-image NEEDS an image to annotate",
              file=sys.stderr)
        sys.exit(1)

    if args.title    is not None or \
       args.hardcopy is not None or \
       args.terminal is not None or \
       args.set      is not None or \
       args.unset    is not None:
        print("--title and --hardcopy and --terminal and --set and --unset are only valid without --write-image",
              file=sys.stderr)
        sys.exit(1)

    if args.points:
        print("Currently --points is implemented ONLY if not --write-image",
              file=sys.stderr)
        sys.exit(1)




import mrcal
import numpy as np

def openmodel(f):
    try:
        return mrcal.cameramodel(f)
    except Exception as e:
        print(f"Couldn't load camera model '{f}': {e}",
              file=sys.stderr)
        sys.exit(1)

models = [openmodel(modelfilename) for modelfilename in args.models]

if any( m.valid_intrinsics_region() is None for m in models ):
    print("Not all given models have a valid-intrinsics contour! Giving up",
          file=sys.stderr)
    sys.exit(1)

if args.write_image:

    # this function stolen from mrcal-reproject-image. Please consolidate
    def target_image_filename(filename_in, suffix):

        base,extension = os.path.splitext(filename_in)
        if len(extension) != 4:
            print(f"imagefile must end in .xxx where 'xxx' is some image extension. Instead got '{filename_in}'",
                  file=sys.stderr)
            sys.exit(1)
        filename_out = f"{base}-{suffix}{extension}"
        if not args.force and os.path.exists(filename_out):
            print(f"Target image '{filename_out}' already exists. Doing nothing, and giving up. Pass -f to overwrite",
                  file=sys.stderr)
            sys.exit(1)
        return filename_out

    def target_image_filename(filename_in, suffix):

        base,extension = os.path.splitext(filename_in)
        if len(extension) != 4:
            print(f"imagefile must end in .xxx where 'xxx' is some image extension. Instead got '{filename_in}'",
                  file=sys.stderr)
            sys.exit(1)
        filename_out = f"{base}-{suffix}{extension}"
        return filename_out


    imagefile_out = target_image_filename(args.image, 'valid-intrinsics-region')
    image_out = mrcal.load_image(args.image)
    for m in models:
        mrcal.annotate_image__valid_intrinsics_region(image_out, m)
    mrcal.save_image(imagefile_out, image_out)
    sys.stderr.write("Wrote {}\n".format(imagefile_out))

else:

    points = None
    if args.points:
        points = np.loadtxt(sys.stdin)

    plotkwargs_extra = {}
    if args.set is not None:
        plotkwargs_extra['set'] = args.set
    if args.unset is not None:
        plotkwargs_extra['unset'] = args.unset

    plot = mrcal.show_valid_intrinsics_region( \
               models,
               cameranames = args.models,
               image       = args.image,
               points      = points,
               hardcopy    = args.hardcopy,
               terminal    = args.terminal,
               title       = args.title,
               **plotkwargs_extra)

    if args.hardcopy is None:
        plot.wait()
