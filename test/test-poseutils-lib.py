#!/usr/bin/python3

import sys
import numpy as np
import numpysane as nps
import os
testdir = os.path.dirname(os.path.realpath(__file__))

# I import the LOCAL mrcal since that's what I'm testing
sys.path[:0] = f"{testdir}/..",
import mrcal
import mrcal._poseutils_npsp as _poseutils

import cv2
from testutils import *
from test_calibration_helpers import grad


def R_from_r(r):
    r'''Rotation matrix from a Rodrigues vector

    Simple reference implementation from wikipedia:

    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
'''

    th_sq = nps.norm2(r)
    Kth = np.array(((    0, -r[2],  r[1]),
                     ( r[2],    0, -r[0]),
                     (-r[1], r[0],     0)))
    if th_sq > 1e-20:
        # normal path
        th = np.sqrt(th_sq)
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

    axis = np.array((R[2,1] - R[1,2],
                     R[0,2] - R[2,0],
                     R[1,0] - R[0,1] ))

    if nps.norm2(axis) > 1e-20:
        # normal path
        costh = (np.trace(R) - 1.)/2.
        th = np.arccos(costh)
        return axis / nps.mag(axis) * th

    # small mag(axis). Can't divide by it. But I can look at the limit.
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

def invert_R(R):
    r'''Simple reference implementation

'''
    return nps.transpose(R)

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

def compose_r(r0, r1):
    r'''Simple reference implementation'''
    return r_from_R(nps.matmult( R_from_r(r0),
                                 R_from_r(r1)))

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

################# rotate_point_R
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

# In-place
R0_ref_copy = np.array(R0_ref)
x_copy      = np.array(x)
y = \
    mrcal.rotate_point_R(R0_ref_copy, x_copy, out = x_copy)
confirm_equal( y,
               nps.matmult(x, nps.transpose(R0_ref)),
               msg='rotate_point_R result written in-place into x')

# inverted
y = \
    mrcal.rotate_point_R(R0_ref, x, out = out3, inverted=True)
confirm_equal( y,
               nps.matmult(x, R0_ref),
               msg='rotate_point_R(inverted) result')

y, J_R, J_x = \
    mrcal.rotate_point_R(R0_ref, x, get_gradients=True,
                         out = (out3,out333,out33),
                         inverted = True)
J_R_ref = grad(lambda R: nps.matmult(x, R),
               R0_ref)
J_x_ref = nps.transpose(R0_ref)
confirm_equal( y,
               nps.matmult(x, R0_ref),
               msg='rotate_point_R(inverted) result')
confirm_equal( J_R,
               J_R_ref,
               msg='rotate_point_R(inverted) J_R')
confirm_equal( J_x,
               J_x_ref,
               msg='rotate_point_R(inverted) J_x')

# inverted in-place
R0_ref_copy = np.array(R0_ref)
x_copy      = np.array(x)
y = \
    mrcal.rotate_point_R(R0_ref_copy, x_copy, out = x_copy, inverted = True)
confirm_equal( y,
               nps.matmult(x, R0_ref),
               msg='rotate_point_R result written in-place into x')

################# rotate_point_r

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

r0_ref_copy  = np.array(r0_ref)
x_copy       = np.array(x)
y = mrcal.rotate_point_r(r0_ref_copy, x_copy,
                         out = x_copy)
confirm_equal( y,
               nps.matmult(x, nps.transpose(R_from_r(r0_ref))),
               msg='rotate_point_r result written in-place into x')

r0_ref_copy  = np.array(r0_ref)
x_copy       = np.array(x)
out33_copy   = np.array(out33)
out33a_copy  = np.array(out33a)
y, J_r, J_x = mrcal.rotate_point_r(r0_ref_copy, x_copy, get_gradients=True,
                                   out = (x_copy, out33, out33a))
confirm_equal( y,
               nps.matmult(x, nps.transpose(R_from_r(r0_ref))),
               msg='rotate_point_r (with gradients) result written in-place into x')
confirm_equal( J_r,
               J_r_ref,
               msg='rotate_point_r (with gradients) result written in-place into x: J_r')
confirm_equal( J_x,
               J_x_ref,
               msg='rotate_point_r (with gradients) result written in-place into x: J_x')

# inverted
y = mrcal.rotate_point_r(r0_ref, x, out = out3, inverted=True)
confirm_equal( y,
               nps.matmult(x, R_from_r(r0_ref)),
               msg='rotate_point_r(inverted) result')

y, J_r, J_x = mrcal.rotate_point_r(r0_ref, x, get_gradients=True,
                                   out = (out3, out33, out33a),
                                   inverted = True)
J_r_ref = grad(lambda r: nps.matmult(x, R_from_r(r)),
               r0_ref)
J_x_ref = grad(lambda x: nps.matmult(x, R_from_r(r0_ref)),
               x)
confirm_equal( y,
               nps.matmult(x, R_from_r(r0_ref)),
               msg='rotate_point_r(inverted) result')
confirm_equal( J_r,
               J_r_ref,
               msg='rotate_point_r(inverted) J_r')
confirm_equal( J_x,
               J_x_ref,
               msg='rotate_point_r(inverted) J_x')

# inverted, in-place
r0_ref_copy  = np.array(r0_ref)
x_copy       = np.array(x)
y = mrcal.rotate_point_r(r0_ref_copy, x_copy, inverted = True,
                         out = x_copy)
confirm_equal( y,
               nps.matmult(x, R_from_r(r0_ref)),
               msg='rotate_point_r(inverted) result written in-place into x')
r0_ref_copy  = np.array(r0_ref)
x_copy       = np.array(x)
out33_copy   = np.array(out33)
out33a_copy  = np.array(out33a)
y, J_r, J_x = mrcal.rotate_point_r(r0_ref_copy, x_copy, get_gradients=True, inverted = True,
                                   out = (x_copy, out33, out33a))
confirm_equal( y,
               nps.matmult(x, R_from_r(r0_ref)),
               msg='rotate_point_r(inverted, with-gradients) result written in-place into x')
confirm_equal( J_r,
               J_r_ref,
               msg='rotate_point_r(inverted, with-gradients result written in-place into x) J_r')
confirm_equal( J_x,
               J_x_ref,
               msg='rotate_point_r(inverted, with-gradients result written in-place into x) J_x')


################# transform_point_Rt


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

# In-place. I can imagine wanting to write the result in-place into t. But not
# into any of R
Rt0_ref_copy = np.array(Rt0_ref)
x_copy       = np.array(x)
y = mrcal.transform_point_Rt(Rt0_ref_copy, x_copy, out = Rt0_ref_copy[3,:])
confirm_equal( y,
               nps.matmult(x, nps.transpose(R0_ref))+t0_ref,
               msg='transform_point_Rt result written in-place into t')
Rt0_ref_copy = np.array(Rt0_ref)
x_copy       = np.array(x)
y = mrcal.transform_point_Rt(Rt0_ref_copy, x_copy, out = x_copy)
confirm_equal( y,
               nps.matmult(x, nps.transpose(R0_ref))+t0_ref,
               msg='transform_point_Rt result written in-place into x')

Rt0_ref_copy = np.array(Rt0_ref)
x_copy       = np.array(x)
y, J_Rt, J_x = mrcal.transform_point_Rt(Rt0_ref_copy, x_copy,
                                        out = (Rt0_ref_copy[3,:], out343, out33),
                                        get_gradients = True)
confirm_equal( y,
               nps.matmult(x, nps.transpose(R0_ref))+t0_ref,
               msg='transform_point_Rt (with gradients) result written in-place into t')
confirm_equal( J_Rt,
               J_Rt_ref,
               msg='transform_point_Rt (with gradients) result written in-place into t: J_Rt')
confirm_equal( J_x,
               J_x_ref,
               msg='transform_point_Rt (with gradients) result written in-place into t: J_x')
Rt0_ref_copy = np.array(Rt0_ref)
x_copy       = np.array(x)
y, J_Rt, J_x = mrcal.transform_point_Rt(Rt0_ref_copy, x_copy,
                                        out = (x_copy, out343, out33),
                                        get_gradients = True)
confirm_equal( y,
               nps.matmult(x, nps.transpose(R0_ref))+t0_ref,
               msg='transform_point_Rt (with gradients) result written in-place into x')
confirm_equal( J_Rt,
               J_Rt_ref,
               msg='transform_point_Rt (with gradients) result written in-place into x: J_Rt')
confirm_equal( J_x,
               J_x_ref,
               msg='transform_point_Rt (with gradients) result written in-place into x: J_x')

# inverted
y = mrcal.transform_point_Rt(Rt0_ref, x, out = out3,
                             inverted = True)
confirm_equal( y,
               nps.matmult(x, R0_ref) - nps.matmult(t0_ref, R0_ref),
               msg='transform_point_Rt(inverted) result')

y, J_Rt, J_x = mrcal.transform_point_Rt(Rt0_ref, x, get_gradients=True,
                                        out = (out3, out343, out33),
                                        inverted = True)
J_Rt_ref = grad(lambda Rt: nps.matmult(x, Rt[:3,:]) - nps.matmult(Rt[3,:], Rt[:3,:]),
                Rt0_ref)
J_x_ref = nps.transpose(R0_ref)
confirm_equal( y,
               nps.matmult(x, R0_ref)-nps.matmult(t0_ref, R0_ref),
               msg='transform_point_Rt(inverted) result')
confirm_equal( J_Rt,
               J_Rt_ref,
               msg='transform_point_Rt(inverted) J_Rt')
confirm_equal( J_x,
               J_x_ref,
               msg='transform_point_Rt(inverted) J_x')

# inverted in-place. I can imagine wanting to write the result in-place into t. But not
# into any of R
Rt0_ref_copy = np.array(Rt0_ref)
x_copy       = np.array(x)
y = mrcal.transform_point_Rt(Rt0_ref_copy, x_copy, out = Rt0_ref_copy[3,:],
                             inverted = True)
confirm_equal( y,
               nps.matmult(x, R0_ref)-nps.matmult(t0_ref, R0_ref),
               msg='transform_point_Rt(inverted) result written in-place into t')
Rt0_ref_copy = np.array(Rt0_ref)
x_copy       = np.array(x)
y = mrcal.transform_point_Rt(Rt0_ref_copy, x_copy, out = x_copy,
                             inverted = True)
confirm_equal( y,
               nps.matmult(x, R0_ref)-nps.matmult(t0_ref, R0_ref),
               msg='transform_point_Rt(inverted) result written in-place into x')

Rt0_ref_copy = np.array(Rt0_ref)
x_copy       = np.array(x)
y, J_Rt, J_x = mrcal.transform_point_Rt(Rt0_ref_copy, x_copy,
                                        out = (Rt0_ref_copy[3,:], out343, out33),
                                        get_gradients = True,
                                        inverted = True)
confirm_equal( y,
               nps.matmult(x, R0_ref)-nps.matmult(t0_ref, R0_ref),
               msg='transform_point_Rt(inverted) (with gradients) result written in-place into t')
confirm_equal( J_Rt,
               J_Rt_ref,
               msg='transform_point_Rt(inverted) (with gradients) result written in-place into t: J_Rt')
confirm_equal( J_x,
               J_x_ref,
               msg='transform_point_Rt(inverted) (with gradients) result written in-place into t: J_x')
Rt0_ref_copy = np.array(Rt0_ref)
x_copy       = np.array(x)
y, J_Rt, J_x = mrcal.transform_point_Rt(Rt0_ref_copy, x_copy,
                                        out = (x_copy, out343, out33),
                                        get_gradients = True,
                                        inverted = True)
confirm_equal( y,
               nps.matmult(x, R0_ref)-nps.matmult(t0_ref, R0_ref),
               msg='transform_point_Rt(inverted) (with gradients) result written in-place into x')
confirm_equal( J_Rt,
               J_Rt_ref,
               msg='transform_point_Rt(inverted) (with gradients) result written in-place into x: J_Rt')
confirm_equal( J_x,
               J_x_ref,
               msg='transform_point_Rt(inverted) (with gradients) result written in-place into x: J_x')


################# transform_point_rt


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

# In-place. I can imagine wanting to write the result in-place into t. But not
# into any of r or J
rt0_ref_copy = np.array(rt0_ref)
x_copy       = np.array(x)
y = mrcal.transform_point_rt(rt0_ref_copy, x_copy,
                             out = rt0_ref_copy[3:])
confirm_equal( y,
               nps.matmult(x, nps.transpose(R0_ref))+t0_ref,
               msg='transform_point_rt result written in-place into t')
rt0_ref_copy = np.array(rt0_ref)
x_copy       = np.array(x)
y = mrcal.transform_point_rt(rt0_ref_copy, x_copy,
                             out = x_copy)
confirm_equal( y,
               nps.matmult(x, nps.transpose(R0_ref))+t0_ref,
               msg='transform_point_rt result written in-place into x')

rt0_ref_copy = np.array(rt0_ref)
x_copy       = np.array(x)
out36_copy   = np.array(out36)
out33a_copy  = np.array(out33a)
y, J_rt, J_x = mrcal.transform_point_rt(rt0_ref_copy, x_copy, get_gradients=True,
                                        out = (rt0_ref_copy[3:],out36_copy,out33a_copy))
confirm_equal( y,
               nps.matmult(x, nps.transpose(R0_ref))+t0_ref,
               msg='transform_point_rt (with gradients) result written in-place into t')
confirm_equal( J_rt,
               J_rt_ref,
               msg='transform_point_rt (with gradients) result written in-place into t: J_rt')
confirm_equal( J_x,
               J_x_ref,
               msg='transform_point_rt (with gradients) result written in-place into t: J_x')
rt0_ref_copy = np.array(rt0_ref)
x_copy       = np.array(x)
out36_copy   = np.array(out36)
out33a_copy  = np.array(out33a)
y, J_rt, J_x = mrcal.transform_point_rt(rt0_ref_copy, x_copy, get_gradients=True,
                                        out = (x_copy, out36_copy,out33a_copy))
confirm_equal( y,
               nps.matmult(x, nps.transpose(R0_ref))+t0_ref,
               msg='transform_point_rt (with gradients) result written in-place into x')
confirm_equal( J_rt,
               J_rt_ref,
               msg='transform_point_rt (with gradients) result written in-place into x: J_rt')
confirm_equal( J_x,
               J_x_ref,
               msg='transform_point_rt (with gradients) result written in-place into x: J_x')

# Inverted
y = mrcal.transform_point_rt(rt0_ref, x, out = out3, inverted=True)
confirm_equal( y,
               nps.matmult(x-t0_ref, R0_ref),
               msg='transform_point_rt(inverted) result')

y, J_rt, J_x = mrcal.transform_point_rt(rt0_ref, x, get_gradients=True,
                                        out = (out3,out36,out33a),
                                        inverted = True)
J_rt_ref = grad(lambda rt: nps.matmult(x-rt[3:], R_from_r(rt[:3])),
                rt0_ref)
J_x_ref = grad(lambda x: nps.matmult(x-t0_ref, R0_ref),
               x)
confirm_equal( y,
               nps.matmult(x-t0_ref, R0_ref),
               msg='transform_point_rt(inverted) result')
confirm_equal( J_rt,
               J_rt_ref,
               msg='transform_point_rt(inverted) J_rt')
confirm_equal( J_x,
               J_x_ref,
               msg='transform_point_rt(inverted) J_x')

# Inverted in-place. I can imagine wanting to write the result in-place into t.
# But not into any of r or J
rt0_ref_copy = np.array(rt0_ref)
x_copy       = np.array(x)
y = mrcal.transform_point_rt(rt0_ref_copy, x_copy, inverted=True,
                             out = rt0_ref_copy[3:])
confirm_equal( y,
               nps.matmult(x-t0_ref, R0_ref),
               msg='transform_point_rt(inverted) result written in-place into t')
rt0_ref_copy = np.array(rt0_ref)
x_copy       = np.array(x)
y = mrcal.transform_point_rt(rt0_ref_copy, x_copy, inverted=True,
                             out = x_copy)
confirm_equal( y,
               nps.matmult(x-t0_ref, R0_ref),
               msg='transform_point_rt(inverted) result written in-place into x')

rt0_ref_copy = np.array(rt0_ref)
x_copy       = np.array(x)
out36_copy   = np.array(out36)
out33a_copy  = np.array(out33a)
y, J_rt, J_x = mrcal.transform_point_rt(rt0_ref_copy, x_copy, get_gradients=True, inverted=True,
                                        out = (rt0_ref_copy[3:],out36_copy,out33a_copy))
confirm_equal( y,
               nps.matmult(x-t0_ref, R0_ref),
               msg='transform_point_rt(inverted, with gradients) result written in-place into t')
confirm_equal( J_rt,
               J_rt_ref,
               msg='transform_point_rt(inverted, with gradients) result written in-place into t: J_rt')
confirm_equal( J_x,
               J_x_ref,
               msg='transform_point_rt(inverted, with gradients) result written in-place into t: J_x')
rt0_ref_copy = np.array(rt0_ref)
x_copy       = np.array(x)
out36_copy   = np.array(out36)
out33a_copy  = np.array(out33a)
y, J_rt, J_x = mrcal.transform_point_rt(rt0_ref_copy, x_copy, get_gradients=True, inverted=True,
                                        out = (x_copy,out36_copy,out33a_copy))
confirm_equal( y,
               nps.matmult(x-t0_ref, R0_ref),
               msg='transform_point_rt(inverted, with gradients) result written in-place into x')
confirm_equal( J_rt,
               J_rt_ref,
               msg='transform_point_rt(inverted, with gradients) result written in-place into x: J_rt')
confirm_equal( J_x,
               J_x_ref,
               msg='transform_point_rt(inverted, with gradients) result written in-place into x: J_x')


################# r_from_R


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


# I've seen this show up in the wild. r_from_R() was producing [nan nan nan]
R_fuzzed_I = \
    np.array([[ 0.9999999999999999              , -0.000000000000000010408340855861,  0.                              ],
              [-0.000000000000000010408340855861,  0.9999999999999999              ,  0.000000000000000013877787807814],
              [ 0.                              ,  0.000000000000000013877787807814,  0.9999999999999999              ]])
confirm_equal( mrcal.r_from_R(R_fuzzed_I), np.zeros((3,)),
               msg = 'r_from_R() can handle numerical fuzz')

################# R_from_r
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

# in-place
Rt0_ref_copy = np.array(Rt0_ref)
Rt = mrcal.invert_Rt(Rt0_ref_copy, out=Rt0_ref_copy)
confirm_equal( Rt,
               invert_Rt(Rt0_ref),
               msg='invert_Rt result written in-place')

R = mrcal.invert_R(R0_ref, out=out33)
confirm_equal( R,
               invert_R(R0_ref),
               msg='invert_R result')

# in-place
R0_ref_copy = np.array(R0_ref)
R = mrcal.invert_R(R0_ref_copy, out=R0_ref_copy)
confirm_equal( R,
               invert_R(R0_ref),
               msg='invert_R result written in-place')

rt = mrcal.invert_rt(rt0_ref, out=out6)
confirm_equal( rt,
               invert_rt(rt0_ref),
               msg='invert_rt result')

# in-place
rt0_ref_copy = np.array(rt0_ref)
rt = mrcal.invert_rt(rt0_ref_copy, out=rt0_ref_copy)
confirm_equal( rt,
               invert_rt(rt0_ref),
               msg='invert_rt result written in-place')

rt,drt_drt  = mrcal.invert_rt(rt0_ref,
                              get_gradients = True,
                              out=(out6,out66))
drt_drt_ref = grad(invert_rt,
                   rt0_ref)
confirm_equal( rt,
               invert_rt(rt0_ref),
               msg='invert_rt with grad result')
confirm_equal( drt_drt,
               drt_drt_ref,
               msg='invert_rt drt/drt result')

# in-place
rt0_ref_copy = np.array(rt0_ref)
drt_drt_copy = np.array(drt_drt)
rt,drt_drt = mrcal.invert_rt(rt0_ref_copy, out=(rt0_ref_copy,drt_drt_copy),
                             get_gradients=True)
confirm_equal( rt,
               invert_rt(rt0_ref),
               msg='invert_rt with grad result written in-place')
confirm_equal( drt_drt,
               drt_drt_ref,
               msg='invert_rt with grad drt/drt result written in-place')

Rt2 = mrcal.compose_Rt(Rt0_ref, Rt1_ref,
                       out=out43)
confirm_equal( Rt2,
               compose_Rt(Rt0_ref, Rt1_ref),
               msg='compose_Rt result')

# in-place
Rt0_ref_copy = np.array(Rt0_ref)
Rt1_ref_copy = np.array(Rt1_ref)
Rt2 = mrcal.compose_Rt(Rt0_ref_copy, Rt1_ref_copy,
                       out=Rt0_ref_copy)
confirm_equal( Rt2,
               compose_Rt(Rt0_ref, Rt1_ref),
               msg='compose_Rt result written in-place to Rt0')
Rt0_ref_copy = np.array(Rt0_ref)
Rt1_ref_copy = np.array(Rt1_ref)
Rt2 = mrcal.compose_Rt(Rt0_ref_copy, Rt1_ref_copy,
                       out=Rt1_ref_copy)
confirm_equal( Rt2,
               compose_Rt(Rt0_ref, Rt1_ref),
               msg='compose_Rt result written in-place to Rt1')




rt2 = mrcal.compose_rt(rt0_ref, rt1_ref, out = out6)
confirm_equal( rt2,
               compose_rt(rt0_ref, rt1_ref),
               msg='compose_rt result')

# in-place
rt0_ref_copy = np.array(rt0_ref)
rt1_ref_copy = np.array(rt1_ref)
rt2 = mrcal.compose_rt(rt0_ref_copy, rt1_ref_copy,
                       out=rt0_ref_copy)
confirm_equal( rt2,
               compose_rt(rt0_ref, rt1_ref),
               msg='compose_rt result written in-place to rt0')
rt0_ref_copy = np.array(rt0_ref)
rt1_ref_copy = np.array(rt1_ref)
rt2 = mrcal.compose_rt(rt0_ref_copy, rt1_ref_copy,
                       out=rt1_ref_copy)
confirm_equal( rt2,
               compose_rt(rt0_ref, rt1_ref),
               msg='compose_rt result written in-place to rt1')

rt2 = _poseutils._compose_rt(rt0_ref, rt1_ref, out=out6)
confirm_equal( rt2,
               compose_rt(rt0_ref, rt1_ref),
               msg='compose_rt result; calling _compose_rt() directly')

# in-place
rt0_ref_copy = np.array(rt0_ref)
rt1_ref_copy = np.array(rt1_ref)
rt2 = _poseutils._compose_rt(rt0_ref_copy, rt1_ref_copy,
                             out=rt0_ref_copy)
confirm_equal( rt2,
               compose_rt(rt0_ref, rt1_ref),
               msg='compose_rt (calling _compose_rt() directly) written in-place to rt0')
rt0_ref_copy = np.array(rt0_ref)
rt1_ref_copy = np.array(rt1_ref)
rt2 = _poseutils._compose_rt(rt0_ref_copy, rt1_ref_copy,
                             out=rt1_ref_copy)
confirm_equal( rt2,
               compose_rt(rt0_ref, rt1_ref),
               msg='compose_rt (calling _compose_rt() directly) written in-place to rt1')

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

# in-place
rt0_ref_copy = np.array(rt0_ref)
rt1_ref_copy = np.array(rt1_ref)
drt2_drt0_copy = np.array(drt2_drt0)
drt2_drt1_copy = np.array(drt2_drt1)
rt2,drt2_drt0,drt2_drt1 = \
    mrcal.compose_rt(rt0_ref_copy, rt1_ref_copy, get_gradients=True,
                     out=(rt0_ref_copy,drt2_drt0_copy,drt2_drt1_copy))
confirm_equal( rt2,
               compose_rt(rt0_ref, rt1_ref),
               msg='compose_rt (with gradients) result written in-place to rt0')
confirm_equal( drt2_drt0,
               drt2_drt0_ref,
               msg='compose_rt (with gradients) result written in-place to rt0: drt2_drt0')
confirm_equal( drt2_drt1,
               drt2_drt1_ref,
               msg='compose_rt (with gradients) result written in-place to rt0: drt2_drt1')
rt0_ref_copy = np.array(rt0_ref)
rt1_ref_copy = np.array(rt1_ref)
rt2,drt2_drt0,drt2_drt1 = \
    mrcal.compose_rt(rt0_ref_copy, rt1_ref_copy, get_gradients=True,
                     out=(rt1_ref_copy,drt2_drt0_copy,drt2_drt1_copy))
confirm_equal( rt2,
               compose_rt(rt0_ref, rt1_ref),
               msg='compose_rt (with gradients) result written in-place to rt1')
confirm_equal( drt2_drt0,
               drt2_drt0_ref,
               msg='compose_rt (with gradients) result written in-place to rt1: drt2_drt0')
confirm_equal( drt2_drt1,
               drt2_drt1_ref,
               msg='compose_rt (with gradients) result written in-place to rt1: drt2_drt1')

Rt2 = mrcal.compose_Rt(Rt0_ref, Rt1_ref,Rt0_ref,
                       out=out43)
confirm_equal( Rt2,
               compose_Rt(compose_Rt(Rt0_ref, Rt1_ref), Rt0_ref),
               msg='compose_Rt with 3 inputs')

rt2 = mrcal.compose_rt(rt0_ref, rt1_ref,rt0_ref,
                       out=out6)
confirm_equal( rt2,
               compose_rt(compose_rt(rt0_ref, rt1_ref), rt0_ref),
               msg='compose_rt with 3 inputs')

################# compose_r()
#
# I check big, almost-0 and 0 arrays in both positions, and all the gradients.
r0big      = base[:3,0,0,0]
r0nearzero = base[:3,1,0,0]
r0zero     = base[:3,2,0,0]
r1big      = base[:3,3,0,0]
r1nearzero = base[:3,4,0,0]
r1zero     = base[:3,5,0,0]
r01        = base[:3,6,0,0]
dr01_dr0   = base[3:6,:3,0,0]
dr01_dr1   = base[3:6,3:6,0,0]

r0big[:] = np.array(( 1.0,  3.0, 1.1)) * 1e-1
r1big[:] = np.array((-2.0, -1.2, 0.3)) * 1e-1

r0nearzero[:] = np.array(( 1.7, -2.0, -5.1 )) * 1e-12
r1nearzero[:] = np.array((-1.2,  5.2,  0.03)) * 1e-12

r0zero[:] *= 0.
r1zero[:] *= 0.

confirm_equal( mrcal.compose_r(r0big, r1big),
               compose_r      (r0big, r1big),
               msg='compose_r basic operation')

mrcal.compose_r(r0big, r1big, out=r01)
confirm_equal( r01,
               compose_r      (r0big, r1big),
               msg='compose_r in-place output')

r01_notinplace, dr01_dr0_notinplace, dr01_dr1_notinplace = \
    mrcal.compose_r(r0big, r1big, get_gradients=True)
dr01_dr0_ref = grad(lambda r0: compose_r( r0, r1big), r0big, step=1e-5)
dr01_dr1_ref = grad(lambda r1: compose_r( r0big, r1), r1big, step=1e-5)
confirm_equal( r01_notinplace,
               compose_r      (r0big, r1big),
               msg='compose_r gradients: r01')
confirm_equal( dr01_dr0_notinplace,
               dr01_dr0_ref,
               msg='compose_r gradients: dr01_dr0')
confirm_equal( dr01_dr1_notinplace,
               dr01_dr1_ref,
               msg='compose_r gradients: dr01_dr1')

mrcal.compose_r(r0big, r1big,
                get_gradients=True,
                out = (r01, dr01_dr0, dr01_dr1))
confirm_equal( r01,
               compose_r      (r0big, r1big),
               msg='compose_r in-place gradients: r01')
confirm_equal( dr01_dr0,
               dr01_dr0_ref,
               msg='compose_r in-place gradients: dr01_dr0')
confirm_equal( dr01_dr1,
               dr01_dr1_ref,
               msg='compose_r in-place gradients: dr01_dr1')


mrcal.compose_r(r0big, r1nearzero,
                get_gradients=True,
                out = (r01, dr01_dr0, dr01_dr1))
dr01_dr0_ref = grad(lambda r0: compose_r( r0, r1nearzero), r0big, step=1e-5)
dr01_dr1_ref = grad(lambda r1: compose_r( r0big, r1), r1nearzero, step=1e-5)
confirm_equal( r01,
               compose_r      (r0big, r1nearzero),
               msg='compose_r in-place r1nearzero gradients: r01')
confirm_equal( dr01_dr0,
               dr01_dr0_ref,
               msg='compose_r in-place r1nearzero gradients: dr01_dr0')
confirm_equal( dr01_dr1,
               dr01_dr1_ref,
               msg='compose_r in-place r1nearzero gradients: dr01_dr1')

mrcal.compose_r(r0big, r1zero,
                get_gradients=True,
                out = (r01, dr01_dr0, dr01_dr1))
dr01_dr0_ref = grad(lambda r0: compose_r( r0, r1zero), r0big, step=1e-5)
dr01_dr1_ref = grad(lambda r1: compose_r( r0big, r1), r1zero, step=1e-5)
confirm_equal( r01,
               compose_r      (r0big, r1zero),
               msg='compose_r in-place r1zero gradients: r01')
confirm_equal( dr01_dr0,
               dr01_dr0_ref,
               msg='compose_r in-place r1zero gradients: dr01_dr0')
confirm_equal( dr01_dr1,
               dr01_dr1_ref,
               msg='compose_r in-place r1zero gradients: dr01_dr1')

mrcal.compose_r(r0nearzero, r1big,
                get_gradients=True,
                out = (r01, dr01_dr0, dr01_dr1))
dr01_dr0_ref = grad(lambda r0: compose_r( r0, r1big), r0nearzero, step=1e-5)
dr01_dr1_ref = grad(lambda r1: compose_r( r0nearzero, r1), r1big, step=1e-5)
confirm_equal( r01,
               compose_r      (r0nearzero, r1big),
               msg='compose_r in-place r1nearzero gradients: r01')
confirm_equal( dr01_dr0,
               dr01_dr0_ref,
               msg='compose_r in-place r1nearzero gradients: dr01_dr0')
confirm_equal( dr01_dr1,
               dr01_dr1_ref,
               msg='compose_r in-place r1nearzero gradients: dr01_dr1')

mrcal.compose_r(r0zero, r1big,
                get_gradients=True,
                out = (r01, dr01_dr0, dr01_dr1))
dr01_dr0_ref = grad(lambda r0: compose_r( r0, r1big), r0zero, step=1e-5)
dr01_dr1_ref = grad(lambda r1: compose_r( r0zero, r1), r1big, step=1e-5)
confirm_equal( r01,
               compose_r      (r0zero, r1big),
               msg='compose_r in-place r1zero gradients: r01')
confirm_equal( dr01_dr0,
               dr01_dr0_ref,
               msg='compose_r in-place r1zero gradients: dr01_dr0')
confirm_equal( dr01_dr1,
               dr01_dr1_ref,
               msg='compose_r in-place r1zero gradients: dr01_dr1')

mrcal.compose_r(r0nearzero, r1nearzero,
                get_gradients=True,
                out = (r01, dr01_dr0, dr01_dr1))
dr01_dr0_ref = grad(lambda r0: compose_r( r0, r1nearzero), r0nearzero, step=1e-5)
dr01_dr1_ref = grad(lambda r1: compose_r( r0nearzero, r1), r1nearzero, step=1e-5)
confirm_equal( r01,
               compose_r      (r0nearzero, r1nearzero),
               msg='compose_r in-place r0nearzero,r1nearzero gradients: r01')
confirm_equal( dr01_dr0,
               dr01_dr0_ref,
               msg='compose_r in-place r0nearzero,r1nearzero gradients: dr01_dr0')
confirm_equal( dr01_dr1,
               dr01_dr1_ref,
               msg='compose_r in-place r0nearzero,r1nearzero gradients: dr01_dr1')

mrcal.compose_r(r0nearzero, r1zero,
                get_gradients=True,
                out = (r01, dr01_dr0, dr01_dr1))
dr01_dr0_ref = grad(lambda r0: compose_r( r0, r1zero), r0nearzero, step=1e-5)
dr01_dr1_ref = grad(lambda r1: compose_r( r0nearzero, r1), r1zero, step=1e-5)
confirm_equal( r01,
               compose_r      (r0nearzero, r1zero),
               msg='compose_r in-place r0nearzero,r1zero gradients: r01')
confirm_equal( dr01_dr0,
               dr01_dr0_ref,
               msg='compose_r in-place r0nearzero,r1zero gradients: dr01_dr0')
confirm_equal( dr01_dr1,
               dr01_dr1_ref,
               msg='compose_r in-place r0nearzero,r1zero gradients: dr01_dr1')

mrcal.compose_r(r0zero, r1nearzero,
                get_gradients=True,
                out = (r01, dr01_dr0, dr01_dr1))
dr01_dr0_ref = grad(lambda r0: compose_r( r0, r1nearzero), r0zero, step=1e-5)
dr01_dr1_ref = grad(lambda r1: compose_r( r0zero, r1), r1nearzero, step=1e-5)
confirm_equal( r01,
               compose_r      (r0zero, r1nearzero),
               msg='compose_r in-place r0zero,r1nearzero gradients: r01')
confirm_equal( dr01_dr0,
               dr01_dr0_ref,
               msg='compose_r in-place r0zero,r1nearzero gradients: dr01_dr0')
confirm_equal( dr01_dr1,
               dr01_dr1_ref,
               msg='compose_r in-place r0zero,r1nearzero gradients: dr01_dr1')

mrcal.compose_r(r0zero, r1zero,
                get_gradients=True,
                out = (r01, dr01_dr0, dr01_dr1))
dr01_dr0_ref = grad(lambda r0: compose_r( r0, r1zero), r0zero, step=1e-5)
dr01_dr1_ref = grad(lambda r1: compose_r( r0zero, r1), r1zero, step=1e-5)
confirm_equal( r01,
               compose_r      (r0zero, r1zero),
               msg='compose_r in-place r0zero,r1zero gradients: r01')
confirm_equal( dr01_dr0,
               dr01_dr0_ref,
               msg='compose_r in-place r0zero,r1zero gradients: dr01_dr0')
confirm_equal( dr01_dr1,
               dr01_dr1_ref,
               msg='compose_r in-place r0zero,r1zero gradients: dr01_dr1')

# Finally, let's look at rotation composition when the result is 0
mrcal.compose_r(r0big, -r0big,
                get_gradients=True,
                out = (r01, dr01_dr0, dr01_dr1))
dr01_dr0_ref = grad(lambda r0: compose_r( r0, -r0big), r0big, step=1e-5)
dr01_dr1_ref = grad(lambda r1: compose_r( r0big, r1), -r0big, step=1e-5)
confirm_equal( r01,
               compose_r      (r0big, -r0big),
               msg='compose_r in-place r0big,-r0big gradients: r01')
confirm_equal( dr01_dr0,
               dr01_dr0_ref,
               msg='compose_r in-place r0big,-r0big gradients: dr01_dr0')
confirm_equal( dr01_dr1,
               dr01_dr1_ref,
               msg='compose_r in-place r0big,-r0big gradients: dr01_dr1')

mrcal.compose_r(r0nearzero, -r0nearzero,
                get_gradients=True,
                out = (r01, dr01_dr0, dr01_dr1))
dr01_dr0_ref = grad(lambda r0: compose_r( r0, -r0nearzero), r0nearzero, step=1e-5)
dr01_dr1_ref = grad(lambda r1: compose_r( r0nearzero, r1), -r0nearzero, step=1e-5)
confirm_equal( r01,
               compose_r      (r0nearzero, -r0nearzero),
               msg='compose_r in-place r0nearzero,-r0nearzero gradients: r01')
confirm_equal( dr01_dr0,
               dr01_dr0_ref,
               msg='compose_r in-place r0nearzero,-r0nearzero gradients: dr01_dr0')
confirm_equal( dr01_dr1,
               dr01_dr1_ref,
               msg='compose_r in-place r0nearzero,-r0nearzero gradients: dr01_dr1')

mrcal.compose_r(r0zero, -r0zero,
                get_gradients=True,
                out = (r01, dr01_dr0, dr01_dr1))
dr01_dr0_ref = grad(lambda r0: compose_r( r0, -r0zero), r0zero, step=1e-5)
dr01_dr1_ref = grad(lambda r1: compose_r( r0zero, r1), -r0zero, step=1e-5)
confirm_equal( r01,
               compose_r      (r0zero, -r0zero),
               msg='compose_r in-place r0zero,-r0zero gradients: r01')
confirm_equal( dr01_dr0,
               dr01_dr0_ref,
               msg='compose_r in-place r0zero,-r0zero gradients: dr01_dr0')
confirm_equal( dr01_dr1,
               dr01_dr1_ref,
               msg='compose_r in-place r0zero,-r0zero gradients: dr01_dr1')

################# compose_r_tinyr0_gradientr0()
# These are a subset of the compose_r() tests just above
r1big      = base[:3,3,0,0]
r1nearzero = base[:3,4,0,0]
r0zero     = base[:3,2,0,0]
r1zero     = base[:3,5,0,0]
r01        = base[:3,6,0,0]
dr01_dr0   = base[3:6,:3,0,0]

r1big      [:] = np.array((-2.0, -1.2, 0.3)) * 1e-1
r1nearzero [:] = np.array((-1.2,  5.2,  0.03)) * 1e-12
r0zero     [:] *= 0.
r1zero     [:] *= 0.


mrcal.compose_r_tinyr0_gradientr0(r1big,
                                  out = dr01_dr0)
dr01_dr0_ref = grad(lambda r0: compose_r( r0, r1big), r0zero, step=1e-5)
confirm_equal( dr01_dr0,
               dr01_dr0_ref,
               msg='compose_r_tinyr0_gradientr0 in-place r1zero gradients: dr01_dr0')

mrcal.compose_r_tinyr0_gradientr0(r1nearzero,
                                  out = dr01_dr0)
dr01_dr0_ref = grad(lambda r0: compose_r( r0, r1nearzero), r0zero, step=1e-5)
confirm_equal( dr01_dr0,
               dr01_dr0_ref,
               msg='compose_r_tinyr0_gradientr0 in-place r1nearzero gradients: dr01_dr0')

mrcal.compose_r_tinyr0_gradientr0(r1zero,
                                  out = dr01_dr0)
dr01_dr0_ref = grad(lambda r0: compose_r( r0, r1zero), r0zero, step=1e-5)
confirm_equal( dr01_dr0,
               dr01_dr0_ref,
               msg='compose_r_tinyr0_gradientr0 in-place r1zero gradients: dr01_dr0')

mrcal.compose_r_tinyr0_gradientr0(-r0zero,
                                  out = dr01_dr0)
dr01_dr0_ref = grad(lambda r0: compose_r( r0, -r0zero), r0zero, step=1e-5)
confirm_equal( dr01_dr0,
               dr01_dr0_ref,
               msg='compose_r_tinyr0_gradientr0 in-place r0zero,-r0zero gradients: dr01_dr0')

finish()
