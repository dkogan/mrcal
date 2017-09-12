#!/usr/bin/python2

import numpy as np
import numpysane as nps
import gnuplotlib as gp
import sys
import cPickle as pickle

sys.path[:0] = ('/home/dima/jpl/stereo-server/analyses',)
import camera_models



def visualize_distortion(model):
    r'''Visualize the distortion effect of a set of intrinsic

The input should either be a file containing a CAHVOR model, a python file
object from which such a model could be read, the dict representation you get
when you parse_cahvor() on such a file OR a numpy array containing the
intrinsics

    '''

    need_close = False

    if type(model) is str:
        model = open(model, 'r')
        need_close = True

    if type(model) is file:
        model = camera_models.parse_cahvor(model)

    if type(model) is dict:
        model, extrinsics = camera_models.factor_cahvor(model)

    if type(model) is not np.ndarray:
        raise Exception("Input must be a string, a file, a dict or a numpy array")

    if len(model) == 4:
        print "Pinhole camera. No distortion"
        sys.exit(0)
    if len(model) != 9:
        raise Exception("Intrinsics vector MUST have length 4 or 9. Instead got {}".format(len(model)))

    N = 20
    W,H = [2*center for center in model[2:4]]
    w    = np.linspace(0,W,N)
    h    = np.linspace(0,H,N)
    grid = nps.cat( *np.meshgrid(w,h) )        # shape: 2,N,N
    grid = nps.transpose(nps.clump(grid, n=2)) # shape: N*N,2

    @nps.broadcast_define( ((2,),),
                           (2,), )
    def proj(p):
        return camera_models.cahvor_warp(p, *model)

    pgrid = proj(grid)


    delta = pgrid-grid
    gp.plot( (grid[:,0], grid[:,1], delta[:,0], delta[:,1],
              {'with': 'vectors size screen 0.01,20 fixed filled',
               'tuplesize': 4,
               }),
             (grid[:,0], grid[:,1],
              {'with': 'points',
               'tuplesize': 2,
               }),
             _xrange=(0,W), _yrange=(0,H),)

    import time
    time.sleep(100000)
