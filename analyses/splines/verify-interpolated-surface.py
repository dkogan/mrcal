#!/usr/bin/python3

f'''Observe the interpolation grid implemented in the C code
'''

import sys
import os

import numpy as np
import numpysane as nps
import gnuplotlib as gp

testdir = os.path.dirname(os.path.realpath(__file__))

# I import the LOCAL mrcal since that's what I'm testing
sys.path[:0] = f"{testdir}/../..",
import mrcal


order      = 3
Nx         = 11
Ny         = 8
fov_x_deg  = 200 # more than 180deg
imagersize = np.array((3000,2000))
cxy        = (imagersize.astype(float) - 1.) / 2.

# I want random control points, with some (different) bias on x and y
controlpoints_fx = np.random.rand(Ny, Nx) * 50 + 2000. + 100. * np.arange(Nx)/Nx
controlpoints_fy = np.random.rand(Ny, Nx) * 50 + 1200. - 100. * np.arange(Nx)/Nx


# The parameters vector dimensions appear (in order from slowest-changing to
# fastest-changing):
# - y coord
# - x coord
# - fx/fy
parameters = nps.mv(nps.cat(controlpoints_fx,controlpoints_fy),
                    0, -1).ravel()


# Now I produce a grid of observation vectors indexed on the coords of the
# control-point arrays

# The index into my spline.
# Indexes on (x,y) and contains (x,y) tuples. Note that this is different from
# the numpy (y,x) convention
x_sampled = np.linspace(1,Nx-2,Nx*5)
y_sampled = np.linspace(1,Ny-2,Ny*5)
ixy = \
    nps.reorder( nps.cat(*np.meshgrid( x_sampled, y_sampled )),
                 -1, -2, -3)

# Stereographic projection function:
#   p   = xyz
#   rxy = mag(xy)
#   th = atan2(rxy, z)
#   xy_stereographic = tan(th/2) * 2. * xy/mag(xy) * f + cxy
#
# I look up f in the splined surface. The index of that lookup is linear (and
# cartesian) with
#
#   tan(th/2) * 2. * xy/mag(xy)
#
# So ix = tan(th/2) * 2. * x/mag(xy) * k + (Nx-1)/2
#    iy = tan(th/2) * 2. * y/mag(xy) * k + (Ny-1)/2
#
# Note that the same scale k is applied for both the x and y indices. The
# constants set the center of the spline surface to x=0 and y=0
#
# The scale k is set by fov_x_deg (this is at y=0):
#
#   ix_margin = tan(-fov_x_deg/2/2) * 2. * k + (Nx-1)/2 --->
#   k = (ix_margin - (Nx-1)/2) / (tan(fov_x_deg/2/2) * 2)
#
# I want to compute p from (ix,iy). I transform these in known ways to get
#
#   jx = tan(th/2) x/mag(xy)
#   jy = tan(th/2) y/mag(xy)
#
# Let mag(xy) = 1. This will be singular at one point at the center, but
# supports points at the edges well, even behind the camera
#
#   jxy = tan(th/2) xy
#       = tan(atan2(1, z)/2) xy
#       = sin(atan2(1, z)) / (1 + cos(atan2(1, z))) xy
#       = (1 - cos()) / sin() xy
#       = (1 - z/sqrt(z^2+1)) / (1/sqrt(z^2+1)) xy =
#       = (sqrt(z^2+1) - z) xy
#
#   mag(jxy) = sqrt(z^2+1) - z
#
#   Let q = sqrt(z^2+1) + z ->
#     mag(jxy) q   = 1
#     q - mag(jxy) = 2z
#   ---> z = (q - mag(jxy)) / 2 = (1/mag(jxy) - mag(jxy)) / 2

if order == 3:
    # cubic splines. There's exactly one extra control point on each side past
    # my fov. So ix_margin = 1
    ix_margin = 1
else:
    raise Exception("Only order==3 supported for now")

k = (ix_margin - (Nx-1)/2) / (np.tan(-fov_x_deg*np.pi/180./2/2) * 2)

jxy  = (ixy - (np.array( (Nx, Ny), dtype=float) - 1.)/2.) / k / 2.
mjxy = nps.mag(jxy)
z    = (1./mjxy - mjxy) / 2.
xy   = jxy / nps.dummy(mjxy, -1) # singular at the center. Do I care?
p    = nps.glue(xy, nps.dummy(z,-1), axis=-1)

mxy = nps.mag(xy)

# Bam. I have applied a stereographic unprojection to get 3D vectors that would
# stereographically project to given spline grid locations. I use the mrcal
# internals to project the unprojection, and to get the focal lengths it ended
# up using. If the internals were implemented correctly, the dense surface of
# focal lengths should follow the sparse surface of spline control points
lens_model_type = f'LENSMODEL_SPLINED_STEREOGRAPHIC_{order}_{Nx}_{Ny}_{fov_x_deg}_{cxy[0]}_{cxy[1]}'
q = mrcal.project(np.ascontiguousarray(p), lens_model_type, parameters)

fxy = (q-cxy) / ( nps.dummy(np.tan(np.arctan2(mxy,z)/2.)/mxy, -1) * 2. * xy)

fx,fy = nps.mv(fxy, -1,0)


gp.plot3d( (nps.transpose(fy),
            dict( _with='lines',
                  using=f'($1*{x_sampled[1]-x_sampled[0]}+{x_sampled[0]}):($2*{y_sampled[1]-y_sampled[0]}+{y_sampled[0]}):3' )),
           (controlpoints_fy,
            dict( _with='points pt 7 ps 2' )),
           xlabel='x control point index',
           ylabel='y control point index',
           title='Focal-y',
           squarexy=True,
           ascii=True,
           wait=True)
