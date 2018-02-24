#!/usr/bin/python2


r'''Converts a camera model from one distortion model to another

Synopsis:

  $ convert-distortion --viz --to DISTORTION_OPENCV4 left.cameramodel > left.opencv4.cameramodel

  ... lots of output as the solve runs ...
  libdogleg at dogleg.c:1064: success! took 10 iterations
  RMS error of the solution: 3.40256580058 pixels.

  ... a plot pops up showing the vector field of the difference ...


Description:

This is a tool to convert a given camera model from one distortion models to
another. The input and output models have identical extrinsics and an identical
intrinsic core (focal lengths, center pixel coords). The ONLY differing part is
the distortion coefficients.

While the distortion models all exist to solve the same problem, the different
representations don't map to one another perfectly, so this tool seeks to find
the best fit only. It does this by sampling a number of points in the imager,
converting them to observation vectors in the camera coordinate system (using
the given camera model), and then fitting a new camera model (with a different
distortions) that matches the observation vectors to the source imager
coordinates.

Note that the distortion model implementations are usually optimized in the
'undistort' direction, not the 'distort' direction, so the step of converting
the target imager coordinates to observation vectors can be slow. This is highly
dependent on the camera model specifically. CAHVORE especially is glacial. This
can be mitigated somewhat by a better implementation, but in the meantime,
please be patient.

Camera models have originally been computed by a calibration procedure that
takes as input a number of point observations, and the resulting models are only
valid in an area where those observations were available; it's an extrapolation
everywhere else. This is generally OK, and we try to cover the whole imager when
calibrating cameras. Models with high distortions (CAHVORE, OPENCV8) generally
have quickly-increasing effects towards the edges of the imager, and the
distortions represented by these models at the extreme edges of the imager are
often not reliable, since the initial calibration data is rarely available at
the extreme edges. Thus using points at the extreme edges to fit another model
is often counterproductive, and I provide the --margin commandline option for
this case. convert-distortion --margin N will avoid N pixels at the edge of the
imager for fitting purposes.

'''



import sys
import numpy as np
import numpysane as nps
import cv2
import re
import argparse
import os

from mrcal import cahvor
from mrcal import projections
from mrcal import cameramodel
import mrcal.optimizer as optimizer



def parse_args():

    parser = \
        argparse.ArgumentParser(description = __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
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
                                parser.error("The cameramodel must be an existing readable file, but got '{}'".format(f)),
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
N       = 20
Npoints = N*N

px = np.linspace(args.margin, dims[0]-1-args.margin, N)
py = np.linspace(args.margin, dims[1]-1-args.margin, N)

# pxy is (N*N, 2). Each slice of pxy[:] is an (x,y) pixel coord
pxy = nps.transpose(nps.clump( nps.cat(*np.meshgrid(px,py)), n=-2))

### I unproject this, with broadcasting
v = projections.unproject( pxy, m.intrinsics() )


### Solve!

### I solve the optimization a number of times with different random seed
### values, taking the best-fitting results. This is required for the richer
### models such as DISTORTION_OPENCV8
err_rms_best = 1e10
intrinsics_values_best = np.array(())
for i in xrange(40): # this many trials
    # random seed for the new intrinsics
    intrinsics_core = m.intrinsics()[1][:4]
    distortions     = (np.random.rand(Ndistortions) - 0.5) * 1e-8 # random initial seed
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

                       do_optimize_intrinsic_core        = False,
                       do_optimize_intrinsic_distortions = True,
                       do_optimize_extrinsics            = False,
                       do_optimize_frames                = False)

    pxy_solved = projections.project( v,
                                      (distortionmodel_to, intrinsics_to_values))
    diff = pxy_solved - pxy
    err_rms = np.sqrt(np.mean(nps.inner(diff, diff)))
    sys.stderr.write("RMS error of this solution: {} pixels.\n".format(err_rms))
    if err_rms < err_rms_best:
        err_rms_best = err_rms
        intrinsics_values_best = np.array(intrinsics_to_values)



sys.stderr.write("RMS error of the BEST solution: {} pixels.\n".format(err_rms_best))
m_to = cameramodel( intrinsics            = (distortionmodel_to, intrinsics_values_best.ravel()),
                    extrinsics_rt_fromref = m.extrinsics_rt(False),
                    dimensions            = dims )
writemodel(m_to)

if args.viz:

    import gnuplotlib as gp

    gp.plot( pxy[:,0], pxy[:,1], diff[:,0], diff[:,1],
             _with='vectors size screen 0.005,10 fixed filled',
             tuplesize=4,
             xrange=(-50,dims[0]+50),
             yrange=(dims[1]+50, -50),
             _set='object 1 rectangle from 0,0 to {},{} fillstyle empty'. format(*dims))
    import time
    time.sleep(100000)
