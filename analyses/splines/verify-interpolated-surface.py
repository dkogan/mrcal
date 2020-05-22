#!/usr/bin/python3

f'''Observe the interpolation grid implemented in the C code

This is a validation of mrcal.project()

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
fxy        = np.array((2000., 1900.))
cxy        = (imagersize.astype(float) - 1.) / 2.

# I want random control points, with some (different) bias on x and y
controlpoints_x = np.random.rand(Ny, Nx) * 1 + 0.5 * np.arange(Nx)/Nx
controlpoints_y = np.random.rand(Ny, Nx) * 1 - 0.9 * np.arange(Nx)/Nx

# to test a delta function
# controlpoints_y*=0
# controlpoints_y[4,5] = 1

# The parameters vector dimensions appear (in order from slowest-changing to
# fastest-changing):
# - y coord
# - x coord
# - fx/fy
parameters = nps.glue( np.array(( fxy[0], fxy[1], cxy[0], cxy[1])),
                       nps.mv(nps.cat(controlpoints_x,controlpoints_y),
                              0, -1).ravel(),
                       axis = -1 )


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

##### this has mostly been implemented in mrcal_project_stereographic() and
##### mrcal_unproject_stereographic()
# Stereographic projection function:
#   p   = xyz
#   rxy = mag(xy)
#   th = atan2(rxy, z)
#   u  = tan(th/2) * 2. * xy/mag(xy)
#   q = (u + deltau) * fxy + cxy
#
# I look up deltau in the splined surface. The index of that lookup is linear
# (and cartesian) with u
#
# So ixy = u * k + (Nxy-1)/2
#
# ix is in [0,Nx] (modulo edges). one per control point
#
# Note that the same scale k is applied for both the x and y indices. The
# constants set the center of the spline surface to x=0 and y=0
#
# The scale k is set by fov_x_deg (this is at y=0):
#
#   ix_margin = tan(-fov_x_deg/2/2) * 2. * k + (Nx-1)/2 --->
#   k = (ix_margin - (Nx-1)/2) / (tan(fov_x_deg/2/2) * 2)
#
# I want to compute p from (ix,iy). p is unique up-to scale. So let me
# arbitrarily set mag(xy) = 1. I define a new var
#
#   jxy = tan(th/2) xy --->
#   jxy = (ixy - (Nxy-1)/2) / (2k)
#
#   jxy = tan(th/2) xy
#       = (1 - cos(th)) / sin(th) xy
#       = (1 - cos(atan2(1, z))) / sin(atan2(1, z)) xy
#       = (1 - z/mag(xyz)) / (1/mag(xyz)) xy =
#       = (mag(xyz) - z) xy =
#
#   mag(jxy) = (mag(xyz) - z)
#            = sqrt(z^2+1) - z
#
#   Let h = sqrt(z^2+1) + z ->
#     mag(jxy) h   = 1
#     h - mag(jxy) = 2z
#   ---> z = (h - mag(jxy)) / 2 = (1/mag(jxy) - mag(jxy)) / 2

if order == 3:
    # cubic splines. There's exactly one extra control point on each side past
    # my fov. So ix_margin = 1
    ix_margin = 1
else:
    # quadratic splines. 1/2 control points on each side past my fov
    ix_margin = 0.5

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
lensmodel_type = f'LENSMODEL_SPLINED_STEREOGRAPHIC_{order}_{Nx}_{Ny}_{fov_x_deg}'
q = mrcal.project(np.ascontiguousarray(p), lensmodel_type, parameters)
th = np.arctan2( nps.mag(p[..., :2]), p[..., 2])
uxy = p[..., :2] * nps.dummy(np.tan(th/2)*2/nps.mag(p[..., :2]), -1)
deltau = (q-cxy) / fxy - uxy

deltaux,deltauy = nps.mv(deltau, -1,0)


gp.plot3d( (nps.transpose(deltauy),
            dict( _with='lines',
                  using=f'($1*{x_sampled[1]-x_sampled[0]}+{x_sampled[0]}):($2*{y_sampled[1]-y_sampled[0]}+{y_sampled[0]}):3' )),
           (controlpoints_y,
            dict( _with='points pt 7 ps 2' )),
           xlabel='x control point index',
           ylabel='y control point index',
           title='Deltau-y',
           squarexy=True,
           ascii=True,
           wait=True)
