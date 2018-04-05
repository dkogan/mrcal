#!/usr/bin/python2

r'''Converts distorted points from one model to another

Synopsis:

  $ redistort.py --from pinhole.cameramodel --to fisheye.cameramodel < input.vnl > output.vnl

This tool takes a set of pixel observations corresponding to one camera model,
and converts them to corresponding observations in another model. This is useful
in conjunction with the undistort.py tool. An envisioned usage:

- undistort.py --model fisheye.cameramodel input.png
  This produces an undistorted image and a corresponding pinhole camera model.

- Run some sort of feature-detection thing on the input_undistorted.png thing we
  just made. This feature-detection thing can make geometric assumptions that
  wouldn't hold in the distorted image

- redistort.py to convert the pixel coords we got from the feature detector back
  into the space of the original image

The input data comes in on standard input, and the output data goes out on
standard output. Both are vnlog data: human-readable text with 2 columns: x and
y pixel coord. Comments are allowed, and start with the '#' character.

'''


import numpy as np
import numpysane as nps
import sys
import argparse
import os
import re

from mrcal import cameramodel
from mrcal import cahvor
from mrcal import projections






def parse_args():

    parser = \
        argparse.ArgumentParser(description = __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--from',
                        type=lambda f: f if os.path.isfile(f) else \
                                parser.error("The cameramodel must be an existing readable file, but got '{}'".format(f)),
                        required=True,
                        nargs=1,
                        help='''Camera model for the INPUT points. Assumed to be mrcal native, Unless the name is xxx.cahvor,
                        in which case the cahvor format is assumed''')

    parser.add_argument('--to',
                        type=lambda f: f if os.path.isfile(f) else \
                                parser.error("The cameramodel must be an existing readable file, but got '{}'".format(f)),
                        required=True,
                        nargs=1,
                        help='''Camera model for the OUTPUT points. Assumed to be mrcal native, Unless the name is xxx.cahvor,
                        in which case the cahvor format is assumed''')

    return parser.parse_args()







args = parse_args()

# 'from' is reserved in python
args_from = getattr(args, 'from')
if re.match(".*\.cahvor$", args_from[0]):
    model_from = cahvor.read(args_from[0])
else:
    model_from = cameramodel(args_from[0])

if re.match(".*\.cahvor$", args.to[0]):
    model_to = cahvor.read(args.to[0])
else:
    model_to = cameramodel(args.to[0])


p = np.genfromtxt(sys.stdin)

v = projections.unproject(p, model_from.intrinsics())
p = projections.project  (v, model_to  .intrinsics())

np.savetxt(sys.stdout, p, fmt='%f', header='x y')
