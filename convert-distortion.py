#!/usr/bin/python2

import sys
import numpy as np
import numpysane as nps
import cv2
import re
import argparse
import cPickle as pickle
import os

from mrcal import cahvor
from mrcal import projections
from mrcal import cameramodel
import mrcal.optimizer as optimizer

r'''Converts a camera model from one distortion model to another

Synopsis:

  $ convert_distortion --viz --to DISTORTION_OPENCV4 left.cameramodel > left.opencv4.cameramodel

  ... lots of output as the solve runs ...
  libdogleg at dogleg.c:1064: success! took 10 iterations
  RMS error of the solution: 3.40256580058 pixels.

  ... a plot pops up showing the vector field of the difference ...
'''




def parse_args():

    parser = \
        argparse.ArgumentParser(description = \
r'''This tool solves the special-case-but-common problem of calibrating ONE PAIR of cameras
given a time-series of chessboard observations''')
    parser.add_argument('--to',
                        required=True,
                        type=str,
                        help='The target distortion model')

    parser.add_argument('--viz',
                        action='store_true',
                        help='''Visualize the difference''')

    parser.add_argument('--margin',
                        required=False,
                        default=0,
                        type=int,
                        help='''Where to run the solve. A (usually positive) value of x means look in a
                        window from (x,x) to (W-x,H-x)''')

    parser.add_argument('cameramodel',
                        type=lambda f: f if os.path.isfile(f) else \
                                parser.error("datafile must be an existing readable file, but got '{}".format(f)),
                        nargs=1,
                        help='''Input camera model. Assumed to be mrcal native, Unless the name is xxx.cahvor,
                        in which case the cahvor format is assumed''')

    return parser.parse_args()

def writemodel(m):
    print("========== CAMERAMODEL =========")
    m.write(sys.stdout)
    print("")

    print("========== CAHVOR MODEL =========")
    cahvor.write(sys.stdout, m)



args = parse_args()

distortionmodel_to = args.to
try:
    Ndistortions = optimizer.getNdistortionParams(distortionmodel_to)
except:
    raise Exception("Unknown distortion model: '{}'".format(distortionmodel_to))


iscahvor = re.match(".*\.cahvor$", args.cameramodel[0])
if iscahvor:
    m = cahvor.read(args.cameramodel[0])
else:
    m = cameramodel(args.cameramodel[0])

intrinsics_from = m.intrinsics()
distortionmodel_from = intrinsics_from[0]

if distortionmodel_from == distortionmodel_to:
    sys.stderr.write("Input and output have the same distortion model: {}. Returning the input\n".format(distortionmodel_to))
    sys.stderr.write("RMS error of the solution: 0 pixels.\n")
    writemodel(m)
    sys.exit(0)


if distortionmodel_to == 'DISTORTION_CAHVORE':
    raise Exception("I don't know how to solve for CAHVORE. Need gradients and stuff")




# Alrighty. Let's actually do the work. I do this:
#
# 1. Sample the imager space with the known model
# 2. Unproject to get the 3d observation vectors
# 3. Solve a new model that fits those vectors to the known observations, but
#    using the new model
dims = m.dimensions()
if dims is None:
    sys.stderr.write("Warning: imager dimensions not available. Using centerpixel*2\n")
    dims = m.intrinsics()[1][2:4] * 2


### I sample the pixels in an NxN grid
N       = 6
Npoints = N*N

px = np.linspace(args.margin, dims[0]-1-args.margin, N)
py = np.linspace(args.margin, dims[1]-1-args.margin, N)

# pxy is (N*N, 2). Each slice of pxy[:] is an (x,y) pixel coord
pxy = nps.transpose(nps.clump( nps.cat(*np.meshgrid(px,py)), n=-2))

### I unproject this, with broadcasting
v = projections.unproject( pxy, m.intrinsics() )

### Solve!
# random seed for the new intrinsics
intrinsics_core = m.intrinsics()[1][:4]
distortions     = (np.random.rand(Ndistortions) - 0.5) * 1e-6
intrinsics_to_values = nps.dummy(nps.glue(intrinsics_core, distortions, axis=-1),
                                 axis=-2)

# range-less points
observations_points = nps.glue(pxy, -np.ones((Npoints, 1),), axis=-1)
observations_points = np.ascontiguousarray(observations_points) # must be contiguous. optimizer.optimize() should really be more lax here

# Which points we're observing. This is dense and kinda silly for this
# application. Each slice is (i_point,i_camera)
indices_points = nps.transpose(nps.glue(np.arange(Npoints,    dtype=np.int32),
                                        np.zeros ((Npoints,), dtype=np.int32), axis=-2))
indices_points = np.ascontiguousarray(indices_points) # must be contiguous. optimizer.optimize() should really be more lax here

optimizer.optimize(intrinsics_to_values,
                   None, # no extrinsics. Just one camera
                   None, # no frames. Just points
                   v,
                   None, # no board observations
                   None, # no board observations
                   observations_points,
                   indices_points,
                   distortionmodel_to,

                   do_optimize_intrinsic_core        = True,
                   do_optimize_intrinsic_distortions = True,
                   do_optimize_extrinsics            = False,
                   do_optimize_frames                = False)

m_to = cameramodel( intrinsics            = (distortionmodel_to, intrinsics_to_values.ravel()),
                    extrinsics_rt_fromref = m.extrinsics_rt(False),
                    dimensions            = dims )

pxy_solved = projections.project( v,
                                  (distortionmodel_to, intrinsics_to_values))
diff = pxy_solved - pxy

sys.stderr.write("RMS error of the solution: {} pixels.\n". \
                 format(np.sqrt(np.mean(nps.inner(diff, diff)))))


writemodel(m_to)

if args.viz:

    import gnuplotlib as gp

    gp.plot( pxy[:,0], pxy[:,1], diff[:,0], diff[:,1],
             _with='vectors size screen 0.005,10 fixed lw 2',
             tuplesize=4,
             xrange=[-50,dims[0]+50],
             yrange=[dims[1]+50, -50])
    import time
    time.sleep(100000)
