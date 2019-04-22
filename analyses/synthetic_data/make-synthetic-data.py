#!/usr/bin/python3

r'''Generate synthetic data for calibration experiments
'''

from __future__ import print_function

import sys
import argparse
import re
import os

def parse_args():

    def positive_float(string):
        try:
            value = float(string)
        except:
            raise argparse.ArgumentTypeError("argument MUST be a positive floating-point number. Got '{}'".format(string))
        if value <= 0:
            raise argparse.ArgumentTypeError("argument MUST be a positive floating-point number. Got '{}'".format(string))
        return value

    parser = \
        argparse.ArgumentParser(description = __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--Nframes',
                        required=True,
                        type=int,
                        help='''How many frames of data to generate. A frame includes an observation from each
                        camera, so there are a total of Nframes * Ncameras
                        observations of the board''')
    parser.add_argument('--observed-pixel-uncertainty',
                        type=positive_float,
                        required=True,
                        help='''The standard deviation of x and y pixel coordinates in the generated data.
                        The noise is assumed to be gaussian, with the standard
                        deviation specified by this argument. Note: this is the
                        x and y standard deviation, treated independently. If
                        each of these is s, then the LENGTH of the deviation of
                        each pixel is a Rayleigh distribution with expected
                        value s*sqrt(pi/2) ~ s*1.25''')
    parser.add_argument('--object-spacing',
                        required=True,
                        type=float,
                        help='Width of each square in the calibration board, in meters')
    parser.add_argument('--object-width-n',
                        type=int,
                        default=10,
                        help='How many points the calibration board has per side')
    parser.add_argument('--bowing',
                        type=float,
                        default=0.0,
                        help='''How bowed the calibration board is. By default, this is 0: the board is
                        perfectly flat. If >0, the board is convex; if <0 it is
                        concave. The value (given in meters) specifies the size
                        of the off-flat gap in the center of the board''')
    parser.add_argument('--at-xyz-rpydeg',
                        type=float,
                        required=True,
                        nargs=6,
                        help='''The pose of the calibration object is a normally-distributed random variable,
                        independent over x,y,z,roll,pitch,yaw. It is given in
                        the coordinate system of camera0. This is 6 values:
                        x,y,z,roll,pitch,yaw, with the roll, pitch, yaw are
                        given in degrees. This argument is the mean of each
                        component.''')
    parser.add_argument('--stdev-xyz-rpydeg',
                        type=float,
                        required=True,
                        nargs=6,
                        help='''The pose of the calibration object is a normally-distributed random variable,
                        independent over x,y,z,roll,pitch,yaw. It is given in
                        the coordinate system of camera0. This is 6 values:
                        x,y,z,roll,pitch,yaw, with the roll, pitch, yaw are
                        given in degrees. This argument is the standard
                        deviation of each component.''')

    parser.add_argument('models',
                        nargs='+',
                        type=lambda f: f if os.path.isfile(f) else \
                                parser.error("Each cameramodel must be an existing readable file, but got '{}'".format(f)),
                        help='''Camera models to use in ''')
    return parser.parse_args()

args = parse_args()

# arg-parsing is done before the imports so that --help works without building
# stuff, so that I can generate the manpages and README




import numpy as np
import numpysane as nps
import time

# look in the project root
sys.path[:0] = (os.path.dirname(os.path.abspath(sys.argv[0])) + '/../..'),
import mrcal


# I move the board, and keep the cameras stationary.
#
# Camera coords: x,y with pixels, z forward
# Board coords:  x,y along. z forward (i.e. back towards the camera)
#                rotation around (x,y,z) is (pitch,yaw,roll)
#                respectively




def make_calibration_board(N, w, bowing):
    '''Returns a calibration object in its reference coord system

    The output has shape (N,N,2)'''

    board = mrcal.get_ref_calibration_object(N, N, w)

    # I shift the board so that the center in at the origin
    board -= (board[0,0,:]+board[-1,-1,:]) / 2.0

    # I did some simple math to evaluate the bowing. If I assume that I live in
    # 2d only and that the shape is a circular arc (until I make
    # very-small-bowing assumptions that is), I get dz = (1 - x^2/u^2) z0. Where
    # dz is the off-plane offset at x meters from the center, u is width/2 and
    # z0 is the offset at the center. In my 3d world, I put my unbowed edges at
    # the corners
    u2 = nps.norm2(board[0,0,:] - board[-1,-1,:]) / 4.
    x2 = nps.norm2(board)
    dz = (1. - x2/u2) * bowing

    board[:,:,2] += dz
    return board


# shape: (N,N,3)
board_reference = make_calibration_board(args.object_width_n,
                                         args.object_spacing,
                                         args.bowing)
# shape: (N*N,3)
board_reference = nps.clump(board_reference, n=2)


Nframes  = args.Nframes
Ncameras = len(args.models)

models = [ mrcal.cameramodel(modelfile) for modelfile in args.models ]
Rt_r0 = models[0].extrinsics_Rt(toref=True)
Rt_xr = [ m.extrinsics_Rt(toref=False) for m in models ]
Rt_x0 = [ mrcal.compose_Rt( Rt_xr[i], Rt_r0 ) \
          for i in range(Ncameras) ]

pixel_noise_xy_1stdev = args.observed_pixel_uncertainty



def get_observation_chunk():
    '''Evaluates Nframes-worth of observations

    But many of these will produce out-of-bounds views, and will be thrown out.
    So <Nframes observations will be returned.

    Returns array of shape (Nframes_inview,Ncameras,N*N,2)

    '''

    xyz       = np.array( args.at_xyz_rpydeg   [:3] )
    rpy       = np.array( args.at_xyz_rpydeg   [3:] ) * np.pi/180.
    xyz_stdev = np.array( args.stdev_xyz_rpydeg[:3] )
    rpy_stdev = np.array( args.stdev_xyz_rpydeg[3:] ) * np.pi/180.

    # shape (Nframes,3)
    xyz = xyz + np.random.randn(Nframes,3) * xyz_stdev
    rpy = rpy + np.random.randn(Nframes,3) * rpy_stdev

    roll,pitch,yaw = nps.transpose(rpy)

    sr,cr = np.sin(roll), np.cos(roll)
    sp,cp = np.sin(pitch),np.cos(pitch)
    sy,cy = np.sin(yaw),  np.cos(yaw)

    Rp = np.zeros((Nframes,3,3), dtype=float)
    Ry = np.zeros((Nframes,3,3), dtype=float)
    Rr = np.zeros((Nframes,3,3), dtype=float)

    Rp[:,0,0] =   1
    Rp[:,1,1] =  cp
    Rp[:,2,1] =  sp
    Rp[:,1,2] = -sp
    Rp[:,2,2] =  cp

    Ry[:,1,1] =   1
    Ry[:,0,0] =  cy
    Ry[:,2,0] =  sy
    Ry[:,0,2] = -sy
    Ry[:,2,2] =  cy

    Rr[:,2,2] =   1
    Rr[:,0,0] =  cr
    Rr[:,1,0] =  sr
    Rr[:,0,1] = -sr
    Rr[:,1,1] =  cr

    # I didn't think about the order too hard; it might be backwards. It also
    # probably doesn't really matter
    R = nps.matmult(Rr, Ry, Rp)

    # shape = (Nframes, N*N, 3)
    boards_cam0 = nps.matmult( # shape (        N*N,1,3)
                               nps.mv(board_reference, 0,-3),

                               # shape (Nframes,1,  3,3)
                               nps.mv(R,               0,-4),
                             )[..., 0, :] + \
                  nps.mv(xyz, 0,-3) # shape (Nframes,1,3)

    # I project everything. Shape: (Nframes,Ncameras,N*N,2)
    p = nps.mv( nps.cat( *[ mrcal.project( mrcal.transform_point_Rt(Rt_x0[i], boards_cam0),
                                           *models[i].intrinsics()) \
                            for i in range(Ncameras) ]),
                0,1 )

    # I pick only those frames where all observations (over all the cameras) are
    # in view
    iframe = \
        np.all(nps.clump(p,         n=-3) >= 0,   axis=-1)
    for i in range(Ncameras):
        W,H = models[i].imagersize()
        iframe *= \
            np.all(nps.clump(p[..., 0], n=-2) <= W-1, axis=-1) * \
            np.all(nps.clump(p[..., 1], n=-2) <= H-1, axis=-1)
    p = p[iframe, ...]

    # p now has shape (Nframes_inview,Ncameras,N*N,2)
    return p


# shape (Nframes_inview,Ncameras,N*N,2)
p = np.zeros((0,Ncameras,args.object_width_n*args.object_width_n,2), dtype=float)

# I keep creating data, until I get Nframes-worth of in-view observations
while True:
    p = nps.glue(p, get_observation_chunk(), axis=-4)
    if p.shape[0] >= Nframes:
        p = p[:,:Nframes,...]
        break

p += np.random.randn(*p.shape) * pixel_noise_xy_1stdev

note = \
    "generated on {} with   {}".format(time.strftime("%Y-%m-%d %H:%M:%S"),
                                         ' '.join(mrcal.shellquote(s) for s in sys.argv))
sys.stdout.write("## " + note + "\n")
sys.stdout.write("# filename x y\n")
for iframe in range(Nframes):
    for icam in range(Ncameras):
        np.savetxt(sys.stdout, p[iframe,icam,...],
                   fmt='frame{:06d}-cam{:01d}.xxx %.3f %.3f'.format(iframe, icam))
