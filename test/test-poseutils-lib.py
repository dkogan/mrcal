#!/usr/bin/python3

import sys
import numpy as np
import numpysane as nps
import os
testdir = os.path.dirname(os.path.realpath(__file__))

# I import the LOCAL mrcal since that's what I'm testing
sys.path[:0] = f"{testdir}/..",
import mrcal
import mrcal._poseutils as _poseutils

import cv2
from testutils import *

def R_from_r(r):
    r'''Rotation matrix from a Rodrigues vector

    Simple reference implementation from wikipedia:

    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
'''

    th = nps.mag(r)
    Kth = np.array(((    0, -r[2],  r[1]),
                     ( r[2],    0, -r[0]),
                     (-r[1], r[0],     0)))
    if th > 1e-10:
        # normal path
        s = np.sin(th)
        c = np.cos(th)
        K = Kth  / th
        return np.eye(3) + s*K + (1. - c)*nps.matmult(K,K)

    # small th. Can't divide by it. But I can look at the limit.
    #
    # s*K = Kth * sin(th)/th -> Kth
    #
    # (1-c)*nps.matmult(K,K) = (1-c) Kth^2/th^2 -> Kth^2 s / 2th -> Kth^2/2
    return np.eye(3) + Kth + nps.matmult(Kth,Kth) / 2.


def r_from_R(R):
    r'''Rodrigues vector from a Rotation matrix

    Simple reference implementation from wikipedia:

    https://en.wikipedia.org/wiki/Rotation_matrix#Conversion_from_and_to_axis%E2%80%93angle

    I assume the input is a valid rotation matrix

    '''

    costh = (np.trace(R) - 1.)/2.
    th = np.arccos(costh)
    axis = np.array((R[2,1] - R[1,2],
                     R[0,2] - R[2,0],
                     R[1,0] - R[0,1] ))

    if th > 1e-10:
        # normal path
        return axis / nps.mag(axis) * th

    # small th. Can't divide by it. But I can look at the limit.
    #
    # axis / (2 sinth)*th = axis/2 *th/sinth ~ axis/2
    return axis/2.


def Rt_from_rt(rt):
    r'''Simple reference implementation'''
    return nps.glue(R_from_r(rt[:3]), rt[3:], axis=-2)

def rt_from_Rt(Rt):
    r'''Simple reference implementation'''
    return nps.glue(r_from_R(Rt[:3,:]), Rt[3,:], axis=-1)


def invert_rt(rt):
    r'''Simple reference implementation

    b = Ra + t  -> a = R'b - R't
'''
    r = rt[:3]
    t = rt[3:]
    R = R_from_r(r)
    tinv = -nps.matmult(t, R)
    return nps.glue( -r, tinv.ravel(), axis=-1)

def invert_Rt(Rt):
    r'''Simple reference implementation

    b = Ra + t  -> a = R'b - R't
'''
    R = Rt[:3,:]
    tinv = -nps.matmult(Rt[3,:], R)
    return nps.glue( nps.transpose(R), tinv.ravel(), axis=-2)

def compose_Rt(Rt0, Rt1):
    r'''Simple reference implementation

    b = R0 (R1 x + t1) + t0 =
      = R0 R1 x + R0 t1 + t0
'''
    R0 = Rt0[:3,:]
    t0 = Rt0[ 3,:]
    R1 = Rt1[:3,:]
    t1 = Rt1[ 3,:]
    R2 = nps.matmult(R0,R1)
    t2 = nps.matmult(t1, nps.transpose(R0)) + t0
    return nps.glue( R2, t2.ravel(), axis=-2)

def compose_rt(rt0, rt1):
    r'''Simple reference implementation'''

    return rt_from_Rt(compose_Rt( Rt_from_rt(rt0),
                                  Rt_from_rt(rt1)))



# Big array. I'm going to slice this thing for my working arrays to produce
# interesting non-contiguous input, output
base   = np.zeros((7,11,13,5), dtype=float)



base[1,1,0:3,1] = np.array((1., 2., 0.1))
r0_ref = base[1,1,0:3,1]

base[1,1,3:6,1] = np.array((3., 5., -2.4))
t0_ref = base[1,1,3:6,1]

base[1,1,6:9,1] = np.array((-.3, -.2, 1.1))
r1_ref = base[1,1,6:9,1]

base[1,1,9:12,1] = np.array((-8.,  .5, -.4))
t1_ref = base[1,1,9:12,1]

base[1,2,0:3,1] = np.array((-10., -108., 3.))
x = base[1,2,0:3,1]

base[1,:3,:3,2] = R_from_r(r0_ref)
R0_ref = base[1,:3,:3,2]

base[1,3:7,:3,2]= nps.glue(R0_ref, t0_ref, axis=-2)
Rt0_ref = base[1,3:7,:3,2]

base[1,2,3:9,1]= nps.glue(r0_ref, t0_ref, axis=-1)
rt0_ref = base[1,2,3:9,1]

base[1,7:10,:3,2] = R_from_r(r1_ref)
R1_ref = base[1,7:10,:3,2]

base[2,:4,:3,2]= nps.glue(R1_ref, t1_ref, axis=-2)
Rt1_ref = base[2,:4,:3,2]

base[1,3,:6,1]= nps.glue(r1_ref, t1_ref, axis=-1)
rt1_ref = base[1,3,:6,1]

# the implementation has a separate path for tiny R, so I test it separately
base[1,5,0:3,1] = np.array((-2.e-18, 3.e-19, -5.e-18))
r0_ref_tiny = base[1,5,0:3,1]

base[5,:3,:3,2] = R_from_r(r0_ref_tiny)
R0_ref_tiny = base[5,:3,:3,2]



out333 = base[:3, :3,   :3,   3]
out343 = base[:3,3:7,   :3,   3]
out43  = base[2,  4:8,  :3,   2]
out33  = base[2,  8:11, :3,   2]
out33a = base[3,    :3, :3,   2]
out33b = base[3,   3:6, :3,   2]
out33c = base[3,   6:9, :3,   2]
out33d = base[4,   :3,  :3,   2]
out36  = base[6,   :3,  :6,   2]
out3   = base[1,  3,    6:9,  1]
out6   = base[1,  4,    :6,   1]
out66  = base[5,3:9,    3:9,  2]
out66a = base[6,3:9,    3:9,  2]

confirm_equal( mrcal.identity_R(out=out33),
               np.eye(3),
               msg='identity_R')
confirm_equal( mrcal.identity_Rt(out=out43),
               nps.glue(np.eye(3), np.zeros((3,),), axis=-2),
               msg='identity_Rt')
confirm_equal( mrcal.identity_r(out=out3),
               np.zeros((3,)),
               msg='identity_r')
confirm_equal( mrcal.identity_rt(out=out6),
               np.zeros((6,)),
               msg='identity_rt')

y = \
    mrcal.rotate_point_R(R0_ref, x, out = out3)
confirm_equal( y,
               nps.matmult(x, nps.transpose(R0_ref)),
               msg='rotate_point_R result')

y, J_R, J_x = \
    mrcal.rotate_point_R(R0_ref, x, get_gradients=True,
                         out = (out3,out333,out33))
J_R_ref = grad(lambda R: nps.matmult(x, nps.transpose(R)),
               R0_ref)
J_x_ref = R0_ref
confirm_equal( y,
               nps.matmult(x, nps.transpose(R0_ref)),
               msg='rotate_point_R result')
confirm_equal( J_R,
               J_R_ref,
               msg='rotate_point_R J_R')
confirm_equal( J_x,
               J_x_ref,
               msg='rotate_point_R J_x')

y = mrcal.rotate_point_r(r0_ref, x, out = out3)
confirm_equal( y,
               nps.matmult(x, nps.transpose(R_from_r(r0_ref))),
               msg='rotate_point_r result')

y, J_r, J_x = mrcal.rotate_point_r(r0_ref, x, get_gradients=True,
                                   out = (out3, out33, out33a))
J_r_ref = grad(lambda r: nps.matmult(x, nps.transpose(R_from_r(r))),
               r0_ref)
J_x_ref = grad(lambda x: nps.matmult(x, nps.transpose(R_from_r(r0_ref))),
               x)
confirm_equal( y,
               nps.matmult(x, nps.transpose(R_from_r(r0_ref))),
               msg='rotate_point_r result')
confirm_equal( J_r,
               J_r_ref,
               msg='rotate_point_r J_r')
confirm_equal( J_x,
               J_x_ref,
               msg='rotate_point_r J_x')

y = mrcal.transform_point_Rt(Rt0_ref, x, out = out3)
confirm_equal( y,
               nps.matmult(x, nps.transpose(R0_ref))+t0_ref,
               msg='transform_point_Rt result')

y, J_Rt, J_x = mrcal.transform_point_Rt(Rt0_ref, x, get_gradients=True,
                                        out = (out3, out343, out33))
J_Rt_ref = grad(lambda Rt: nps.matmult(x, nps.transpose(Rt[:3,:])) + Rt[3,:],
                Rt0_ref)
J_x_ref = R0_ref
confirm_equal( y,
               nps.matmult(x, nps.transpose(R0_ref))+t0_ref,
               msg='transform_point_Rt result')
confirm_equal( J_Rt,
               J_Rt_ref,
               msg='transform_point_Rt J_Rt')
confirm_equal( J_x,
               J_x_ref,
               msg='transform_point_Rt J_x')

y = mrcal.transform_point_rt(rt0_ref, x, out = out3)
confirm_equal( y,
               nps.matmult(x, nps.transpose(R0_ref))+t0_ref,
               msg='transform_point_rt result')

y, J_rt, J_x = mrcal.transform_point_rt(rt0_ref, x, get_gradients=True,
                                        out = (out3,out36,out33a))
J_rt_ref = grad(lambda rt: nps.matmult(x, nps.transpose(R_from_r(rt[:3])))+rt[3:],
                rt0_ref)
J_x_ref = grad(lambda x: nps.matmult(x, nps.transpose(R0_ref))+t0_ref,
               x)
confirm_equal( y,
               nps.matmult(x, nps.transpose(R0_ref))+t0_ref,
               msg='transform_point_rt result')
confirm_equal( J_rt,
               J_rt_ref,
               msg='transform_point_rt J_rt')
confirm_equal( J_x,
               J_x_ref,
               msg='transform_point_rt J_x')



r = mrcal.r_from_R(R0_ref, out = out3)
confirm_equal( r,
               r0_ref,
               msg='r_from_R result')

r, J_R = mrcal.r_from_R(R0_ref, get_gradients=True,
                        out = (out3,out333))
J_R_ref = grad(r_from_R,
               R0_ref)
confirm_equal( r,
               r0_ref,
               msg='r_from_R result')
confirm_equal( J_R,
               J_R_ref,
               msg='r_from_R J_R')

# Do it again, actually calling opencv. This is both a test, and shows how to
# migrate old code
r, J_R = mrcal.r_from_R(R0_ref, get_gradients=True,
                        out = (out3,out333))
rref,J_R_ref = cv2.Rodrigues(R0_ref)
confirm_equal( r,
               rref,
               msg='r_from_R result, comparing with cv2.Rodrigues')

# I'm not comparing with opencv's gradient report or dr/dR. It doesn't match. I
# know my gradient is correct because I numerically checked it above. Maybe
# opencv is doing something different because of the constraints placed on R.
# Figuring this out would be good
# J_R_ref = nps.transpose(J_R_ref) # fix opencv's weirdness. Now shape=(3,9)
# J_R_ref = J_R_ref.reshape(3,3,3)
# confirm_equal( J_R,
#                J_R_ref,
#                msg='r_from_R J_R, comparing with cv2.Rodrigues')



R = mrcal.R_from_r(r0_ref, out = out33)
confirm_equal( R,
               R0_ref,
               msg='R_from_r result')

R, J_r = mrcal.R_from_r(r0_ref, get_gradients=True,
                        out = (out33,out333))
J_r_ref = grad(R_from_r,
               r0_ref)
confirm_equal( R,
               R0_ref,
               msg='R_from_r result')
confirm_equal( J_r,
               J_r_ref,
               msg='R_from_r J_r')

# the implementation has a separate path for tiny R, so I test it separately
R = mrcal.R_from_r(r0_ref_tiny, out = out33)
confirm_equal( R,
               R0_ref_tiny,
               msg='R_from_r result for tiny r0')

R, J_r = mrcal.R_from_r(r0_ref_tiny, get_gradients=True,
                        out = (out33,out333))
J_r_ref = grad(R_from_r,
               r0_ref_tiny)
confirm_equal( R,
               R0_ref_tiny,
               msg='R_from_r result for tiny r0')
confirm_equal( J_r,
               J_r_ref,
               msg='R_from_r J_r for tiny r0')

# Do it again, actually calling opencv. This is both a test, and shows how to
# migrate old code
R, J_r = mrcal.R_from_r(r0_ref, get_gradients=True,
                        out = (out33,out333))
Rref,J_r_ref = cv2.Rodrigues(r0_ref)
J_r_ref = nps.transpose(J_r_ref) # fix opencv's weirdness. Now shape=(9,3)
J_r_ref = J_r_ref.reshape(3,3,3)
confirm_equal( R,
               Rref,
               msg='R_from_r result, comparing with cv2.Rodrigues')
confirm_equal( J_r,
               J_r_ref,
               msg='R_from_r J_r, comparing with cv2.Rodrigues')



rt = mrcal.rt_from_Rt(Rt0_ref, out = out6)
confirm_equal( rt,
               rt0_ref,
               msg='rt_from_Rt result')

rt, J_R = mrcal.rt_from_Rt(Rt0_ref, get_gradients = True,
                           out = (out6,out333))
J_R_ref = grad(r_from_R,
               Rt0_ref[:3,:])
confirm_equal( rt,
               rt0_ref,
               msg='rt_from_Rt result')
confirm_equal( J_R,
               J_R_ref,
               msg='rt_from_Rt grad result')

Rt = mrcal.Rt_from_rt(rt0_ref, out=out43)
confirm_equal( Rt,
               Rt0_ref,
               msg='Rt_from_rt result')

Rt, J_r = mrcal.Rt_from_rt(rt0_ref, get_gradients = True,
                           out = (out43,out333))
J_r_ref = grad(R_from_r,
               rt0_ref[:3])
confirm_equal( Rt,
               Rt0_ref,
               msg='Rt_from_rt result')
confirm_equal( J_r,
               J_r_ref,
               msg='Rt_from_rt grad result')

Rt = mrcal.invert_Rt(Rt0_ref, out=out43)
confirm_equal( Rt,
               invert_Rt(Rt0_ref),
               msg='invert_Rt result')

rt = mrcal.invert_rt(rt0_ref, out=out6)
confirm_equal( rt,
               invert_rt(rt0_ref),
               msg='invert_rt result')

rt,drt_drt  = mrcal.invert_rt(rt0_ref, get_gradients = True,
                                 out=(out6,out66))
drt_drt_ref = grad(invert_rt,
                   rt0_ref)
confirm_equal( rt,
               invert_rt(rt0_ref),
               msg='invert_rt with grad result')
confirm_equal( drt_drt,
               drt_drt_ref,
               msg='invert_rt drt/drt result')

Rt2 = mrcal.compose_Rt(Rt0_ref, Rt1_ref,
                       out=out43)
confirm_equal( Rt2,
               compose_Rt(Rt0_ref, Rt1_ref),
               msg='compose_Rt result')

rt2 = mrcal.compose_rt(rt0_ref, rt1_ref, out = out6)
confirm_equal( rt2,
               compose_rt(rt0_ref, rt1_ref),
               msg='compose_rt result')

# _compose_rt() is not excercised by the python library, so I explicitly test it
# here
rt2 = _poseutils._compose_rt(rt0_ref, rt1_ref, out=out6)
confirm_equal( rt2,
               compose_rt(rt0_ref, rt1_ref),
               msg='compose_rt result; calling _compose_rt() directly')

rt2,drt2_drt0,drt2_drt1 = \
    mrcal.compose_rt(rt0_ref, rt1_ref, get_gradients=True,
                     out = (out6, out66, out66a))

drt2_drt0_ref = grad(lambda rt0: compose_rt( rt0, rt1_ref), rt0_ref)
drt2_drt1_ref = grad(lambda rt1: compose_rt( rt0_ref, rt1), rt1_ref)
confirm_equal( rt2,
               compose_rt(rt0_ref, rt1_ref),
               msg='compose_rt result')
confirm_equal( drt2_drt0,
               drt2_drt0_ref,
               msg='compose_rt drt2_drt0')
confirm_equal( drt2_drt1,
               drt2_drt1_ref,
               msg='compose_rt drt2_drt1')

Rt2 = mrcal.compose_Rt(Rt0_ref, Rt1_ref,Rt0_ref,
                       out=out43)
confirm_equal( Rt2,
               compose_Rt(compose_Rt(Rt0_ref, Rt1_ref), Rt0_ref),
               msg='compose_Rt with 3 inputs; associate one way')
confirm_equal( Rt2,
               compose_Rt(Rt0_ref, compose_Rt(Rt1_ref, Rt0_ref)),
               msg='compose_Rt with 3 inputs; associate the other way')

rt2 = mrcal.compose_rt(rt0_ref, rt1_ref,rt0_ref,
                       out=out6)
confirm_equal( rt2,
               compose_rt(compose_rt(rt0_ref, rt1_ref), rt0_ref),
               msg='compose_rt with 3 inputs; associate one way')
confirm_equal( rt2,
               compose_rt(rt0_ref, compose_rt(rt1_ref, rt0_ref)),
               msg='compose_rt with 3 inputs; associate the other way')

finish()
