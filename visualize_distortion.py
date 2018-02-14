#!/usr/bin/python2

import numpy as np
import numpysane as nps
import gnuplotlib as gp
import sys
import cPickle as pickle
import re
import cv2

from mrcal import cameramodel
from mrcal import cahvor
from mrcal import projections




def visualize_distortion_vector_field(model):
    r'''Visualize the distortion effect of a set of intrinsic

    This function renders the distortion vector field
    '''

    intrinsics = model.intrinsics()

    N = 20
    W,H = [2*center for center in intrinsics[1][2:4]]

    # get the input and output grids of shape Nwidth,Nheight,2
    grid, dgrid = projections.distortion_map__to_warped(intrinsics,
                                                        np.linspace(0,W,N),
                                                        np.linspace(0,H,N))

    # shape: N*N,2
    grid  = nps.clump(grid,  n=2)
    dgrid = nps.clump(dgrid, n=2)

    delta = dgrid-grid
#    delta *= 1000
    gp.plot( (grid[:,0], grid[:,1], delta[:,0], delta[:,1],
              {'with': 'vectors size screen 0.01,20 fixed filled',
               'tuplesize': 4,
               }),
             (grid[:,0], grid[:,1],
              {'with': 'points',
               'tuplesize': 2,
               }),
             _xrange=(0,W), _yrange=(H,0))

    import time
    time.sleep(100000)



modelfile = sys.argv[1]
if re.match(".*\.cahvor$", modelfile):
    model = cahvor.read(modelfile)
else:
    model = cameramodel(modelfile)

try:
    imagefile = sys.argv[2]
except:
    imagefile = None


if imagefile is None:
    # no image file is given. Draw the vector field
    visualize_distortion_vector_field(model)
else:
    m = re.match("(.*)\.([a-z][a-z][a-z])$", imagefile)
    if not m:
        raise Exception("imagefile must end in .xxx where 'xxx' is some image extension. Instead got '{}'".format(imagefile))

    image_corrected = \
        projections.undistort_image(model, imagefile)

    imagefile_corrected = "{}_undistorted.{}".format(m.group(1),m.group(2))
    cv2.imwrite(imagefile_corrected, image_corrected)
    print "Wrote {}".format(imagefile_corrected)
