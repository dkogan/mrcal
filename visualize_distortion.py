#!/usr/bin/python2

import numpy as np
import numpysane as nps
import gnuplotlib as gp
import sys
import cPickle as pickle
import re
import cv2

sys.path[:0] = ('/home/dima/src_boats/stereo-server/analyses',)
import camera_models

import utils

cahvorfile = sys.argv[1]
try:
    imagefile = sys.argv[2]
except:
    imagefile = None


if imagefile is None:
    # no image file is given. Draw the vector field
    utils.visualize_distortion_vector_field(cahvorfile)
else:
    m = re.match("(.*)\.([a-z][a-z][a-z])$", imagefile)
    if not m:
        raise Exception("imagefile must end in .xxx where 'xxx' is some image extension. Instead got '{}'".format(imagefile))

    image_corrected = \
        utils.undistort_image(cahvorfile, imagefile)

    imagefile_corrected = "{}_undistorted.{}".format(m.group(1),m.group(2))
    cv2.imwrite(imagefile_corrected, image_corrected)
    print "Wrote {}".format(imagefile_corrected)
