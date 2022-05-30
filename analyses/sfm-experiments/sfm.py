#!/usr/bin/python3

r'''
'''

import sys
import numpy as np
import numpysane as nps
import gnuplotlib as gp
import cv2
import glob


def imread(filename, decimation):
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, image[::decimation, ::decimation]

def plot_flow(filename,
              image, flow, decimation):
    H,W = flow.shape[:2]

    # each has shape (H,W)
    xx,yy = np.meshgrid(np.arange(0,W,decimation),
                        np.arange(0,H,decimation))

    # shape (H,W,4)
    vectors = \
        nps.glue( nps.dummy(xx,-1),
                  nps.dummy(yy,-1),
                  flow[::decimation, ::decimation, :],
                  axis = -1);
    vectors = nps.clump(vectors, n=2)

    gp.plot( (image, dict(_with='image',
                          tuplesize = 3)),
             (vectors,
              dict(_with='vectors lc "red"',
                   tuplesize=-4)),

             _set  = 'palette gray',
             unset = 'colorbox',
             square = True,
             _xrange = (0,W),
             _yrange = (H,0),
             hardcopy = filename)

    print(f"Wrote {filename}")






directory       = "/home/dima/data/xxxxx/delta/*.jpg"
outdir          = "/tmp"
decimation      = 20
decimation_extra_plot = 5
Nimages         = 10 #None # all of them


l = sorted(glob.glob(directory))


image0,image0_decimated = imread(l[0], decimation)

i=0
for f in l[1:Nimages]:
    image1,image1_decimated = imread(f, decimation)

    # shape (Hdecimated,Wdecimated,2)
    flow = -cv2.calcOpticalFlowFarneback(image0_decimated, image1_decimated,
                                         flow       = None, # for in-place output
                                         pyr_scale  = 0.5,
                                         levels     = 3,
                                         winsize    = 15,
                                         iterations = 3,
                                         poly_n     = 5,
                                         poly_sigma = 1.2,
                                         flags      = 0# cv2.OPTFLOW_USE_INITIAL_FLOW
                                         )

    plot_flow(f"{outdir}/flow{i:03d}.png",
              image0_decimated, flow,
              decimation_extra_plot)

    image0_decimated = image1_decimated
    i += 1
