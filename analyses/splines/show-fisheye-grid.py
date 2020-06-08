#!/usr/bin/python3

r'''Evaluate 2D griddings of a camera view

I'm interested in gridding the observation vectors for projection. These
observation vectors have length-1, so they're 3D quantities that live in a 2D
manifold. I need some sort of a 2D representation so that I can convert between
this representation and the 3D vectors without worrying about constraints.

I'd like a rectangular 2D gridding of observation vectors to more-or-less map to
a gridding of projected pixel coordinates: I want both to be as rectangular as
possible. This tool grids the imager, and plots the unprojected grid for a
number of different 3d->2d schemes.

I take in an existing camera mode, and I grid its imager. I unproject each
pixel, and reproject back using the scheme I'm looking at. Most schemes are
fisheye projections described at

  https://en.wikipedia.org/wiki/Fisheye_lens

mrcal-show-distortions --radial also compares a given model with fisheye
projections

'''


import sys
import argparse
import re
import os

def parse_args():

    parser = \
        argparse.ArgumentParser(description = __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('scheme',
                        type=str,
                        choices=('pinhole', 'stereographic', 'equidistant', 'equisolidangle', 'orthographic'),
                        help='''Type of fisheye model to visualize. For a description of the choices see
                        https://en.wikipedia.org/wiki/Fisheye_lens''')

    parser.add_argument('model',
                        type=str,
                        help='''Camera model to grid''')

    args = parser.parse_args()

    return args

args = parse_args()

# arg-parsing is done before the imports so that --help works without building
# stuff, so that I can generate the manpages and README





import numpy as np
import numpysane as nps
import gnuplotlib as gp

testdir = os.path.dirname(os.path.realpath(__file__))

# I import the LOCAL mrcal since that's what I'm testing
sys.path[:0] = f"{testdir}/../..",
import mrcal





@nps.broadcast_define(((3,),), (2,))
def project_simple(v, d):
    k = d[4:]
    fxy = d[:2]
    cxy = d[2:4]
    x,y = v[:2]/v[2]
    r2 = x*x + y*y
    r4 = r2*r2
    r6 = r4*r2
    a1 = 2*x*y
    a2 = r2 + 2*x*x
    a3 = r2 + 2*y*y
    cdist = 1 + k[0]*r2 + k[1]*r4 + k[4]*r6
    icdist2 = 1./(1 + k[5]*r2 + k[6]*r4 + k[7]*r6)
    return np.array(( x*cdist*icdist2 + k[2]*a1 + k[3]*a2,
                      y*cdist*icdist2 + k[2]*a3 + k[3]*a1 )) * fxy + cxy
@nps.broadcast_define(((3,),), (2,))
def project_radial_numdenom(v, d):
    k = d[4:]
    fxy = d[:2]
    cxy = d[2:4]
    x,y = v[:2]/v[2]
    r2 = x*x + y*y
    r4 = r2*r2
    r6 = r4*r2
    a1 = 2*x*y
    a2 = r2 + 2*x*x
    a3 = r2 + 2*y*y
    return np.array((1 + k[0]*r2 + k[1]*r4 + k[4]*r6,
                     1 + k[5]*r2 + k[6]*r4 + k[7]*r6))



try:
    m = mrcal.cameramodel(args.model)
except:
    print(f"Couldn't read '{args.model}' as a camera model", file=sys.stderr)
    sys.exit(1)


W,H = m.imagersize()
Nw  = 40
Nh  = 30
# shape (Nh,Nw,2)
xy = \
    nps.mv(nps.cat(*np.meshgrid( np.linspace(0,W-1,Nw),
                                 np.linspace(0,H-1,Nh) )),
           0,-1)
fxy = m.intrinsics()[1][0:2]
cxy = m.intrinsics()[1][2:4]

# shape (Nh,Nw,2)
v  = mrcal.unproject(np.ascontiguousarray(xy), *m.intrinsics())
v0 = mrcal.unproject(cxy, *m.intrinsics())

# shape (Nh,Nw)
costh = nps.inner(v,v0) / (nps.mag(v) * nps.mag(v0))
th = np.arccos(costh)

# shape (Nh,Nw,2)
xy_rel = xy-cxy
# shape (Nh,Nw)
az = np.arctan2( xy_rel[...,1], xy_rel[..., 0])

if   args.scheme == 'stereographic':  r = np.tan(th/2.) * 2.
elif args.scheme == 'equidistant':    r = th
elif args.scheme == 'equisolidangle': r = np.sin(th/2.) * 2.
elif args.scheme == 'orthographic':   r = np.sin(th)
elif args.scheme == 'pinhole':        r = np.tan(th)
else: print("Unknown scheme {args.scheme}. Shouldn't happen. argparse should have taken care of it")

mapped = xy_rel * nps.dummy(r/nps.mag(xy_rel),-1)
gp.plot(mapped, tuplesize=-2,
        _with  = 'linespoints',
        title  = f"Gridded model '{args.model}' looking at pinhole unprojection with z=1",
        xlabel = f'Normalized {args.scheme} x',
        ylabel = f'Normalized {args.scheme} y',
        square = True,
        wait   = True)
