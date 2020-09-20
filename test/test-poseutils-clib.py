#!/usr/bin/python3

import sys
import numpy as np
import numpysane as nps
import os
testdir = os.path.dirname(os.path.realpath(__file__))

# I import the LOCAL mrcal since that's what I'm testing
sys.path[:0] = f"{testdir}/..",
import mrcal
import cv2
from testutils import *

import mrcal._poseutils as _poseutils

def grad(f, x, step=1e-6):
    r'''Computes df/dx at x

    f is a function of one argument. If the input has shape Si and the output
    has shape So, the returned gradient has shape So+Si. This applies central
    differences

    '''

    d     = x*0
    dflat = d.ravel()

    def df_dxi(i, d,dflat):

        dflat[i] = step
        fplus  = f(x+d)
        fminus = f(x-d)
        j = (fplus-fminus)/(2.*step)
        dflat[i] = 0
        return j

    # grad variable is in first dim
    Jflat = nps.cat(*[df_dxi(i, d,dflat) for i in range(len(dflat))])
    # grad variable is in last dim
    Jflat = nps.mv(Jflat, 0, -1)
    return Jflat.reshape( Jflat.shape[:-1] + d.shape )


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



r0_ref = np.array((1., 2., 0.1))
t0_ref = np.array((3., 5., -2.4))

r1_ref = np.array((-.3, -.2, 1.1))
t1_ref = np.array((-8.,  .5, -.4))

x      = np.array((-10., -108., 3.))

R0_ref = R_from_r(r0_ref)
Rt0_ref= nps.glue(R0_ref, t0_ref, axis=-2)
rt0_ref= nps.glue(r0_ref, t0_ref, axis=-1)

R1_ref = R_from_r(r1_ref)
Rt1_ref= nps.glue(R1_ref, t1_ref, axis=-2)
rt1_ref= nps.glue(r1_ref, t1_ref, axis=-1)




confirm_equal( mrcal.identity_R(),
               np.eye(3),
               msg='identity_R')
confirm_equal( mrcal.identity_Rt(),
               nps.glue(np.eye(3), np.zeros((3,),), axis=-2),
               msg='identity_Rt')
confirm_equal( mrcal.identity_r(),
               np.zeros((3,)),
               msg='identity_r')
confirm_equal( mrcal.identity_rt(),
               np.zeros((6,)),
               msg='identity_rt')

y, J_R, J_x = mrcal.rotate_point_R(R0_ref, x, get_gradients=True)
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

y, J_r, J_x = mrcal.rotate_point_r(r0_ref, x, get_gradients=True)
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

y, J_R, J_t, J_x = mrcal.transform_point_Rt(Rt0_ref, x, get_gradients=True)
J_R_ref = grad(lambda R: nps.matmult(x, nps.transpose(R))+t0_ref,
               R0_ref)
J_t_ref = np.identity(3)
J_x_ref = R0_ref
confirm_equal( y,
               nps.matmult(x, nps.transpose(R0_ref))+t0_ref,
               msg='transform_point_Rt result')
confirm_equal( J_R,
               J_R_ref,
               msg='transform_point_Rt J_R')
confirm_equal( J_t,
               J_t_ref,
               msg='transform_point_Rt J_t')
confirm_equal( J_x,
               J_x_ref,
               msg='transform_point_Rt J_x')

y, J_r, J_t, J_x = mrcal.transform_point_rt(rt0_ref, x, get_gradients=True)
J_r_ref = grad(lambda r: nps.matmult(x, nps.transpose(R_from_r(r)))+t0_ref,
               r0_ref)
J_t_ref = np.identity(3)
J_x_ref = grad(lambda x: nps.matmult(x, nps.transpose(R0_ref))+t0_ref,
               x)
confirm_equal( y,
               nps.matmult(x, nps.transpose(R0_ref))+t0_ref,
               msg='transform_point_rt result')
confirm_equal( J_r,
               J_r_ref,
               msg='transform_point_rt J_r')
confirm_equal( J_t,
               J_t_ref,
               msg='transform_point_rt J_t')
confirm_equal( J_x,
               J_x_ref,
               msg='transform_point_rt J_x')



r, J_R = mrcal.r_from_R(R0_ref, get_gradients=True)
J_R_ref = grad(r_from_R,
               R0_ref)
confirm_equal( r,
               r0_ref,
               msg='r_from_R result')
confirm_equal( J_R,
               J_R_ref,
               msg='r_from_R J_R')

r   = np.eye(3, dtype=float)[:,0]
J_R = np.zeros((3,3,3), dtype=float).transpose()
R1_ref_noncontiguous = nps.glue(R1_ref, np.ones((3,1), dtype=float), axis=-1)[:,:3]
_poseutils._r_from_R_withgrad(R1_ref_noncontiguous, out=(r,J_R))
J_R_ref = grad(r_from_R,
               R1_ref)
confirm_equal( r,
               r1_ref,
               msg='r_from_R non-contiguous result')
confirm_equal( J_R,
               J_R_ref,
               msg='r_from_R non-contiguous J_R')

# Do it again, actually calling opencv. This is both a test, and shows how to
# migrate old code
r, J_R = mrcal.r_from_R(R0_ref, get_gradients=True)
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



R, J_r = mrcal.R_from_r(r0_ref, get_gradients=True)
J_r_ref = grad(R_from_r,
               r0_ref)
confirm_equal( R,
               R0_ref,
               msg='R_from_r result')
confirm_equal( J_r,
               J_r_ref,
               msg='R_from_r J_r')

R   = np.zeros((3,3),   dtype=float).transpose()
J_r = np.zeros((3,3,3), dtype=float).transpose()
r1_ref_noncontiguous = nps.outer(r1_ref,np.ones((3,),dtype=float)).transpose()[0,:]
_poseutils._R_from_r_withgrad(r1_ref_noncontiguous, out=(R,J_r))
J_r_ref = grad(R_from_r,
               r1_ref)
confirm_equal( R,
               R1_ref,
               msg='R_from_r non-contiguous result')
confirm_equal( J_r,
               J_r_ref,
               msg='R_from_r non-contiguous J_r')

# Do it again, actually calling opencv. This is both a test, and shows how to
# migrate old code
R, J_r = mrcal.R_from_r(r0_ref, get_gradients=True)
Rref,J_r_ref = cv2.Rodrigues(r0_ref)
J_r_ref = nps.transpose(J_r_ref) # fix opencv's weirdness. Now shape=(9,3)
J_r_ref = J_r_ref.reshape(3,3,3)
confirm_equal( R,
               Rref,
               msg='R_from_r result, comparing with cv2.Rodrigues')
confirm_equal( J_r,
               J_r_ref,
               msg='R_from_r J_r, comparing with cv2.Rodrigues')



rt = mrcal.rt_from_Rt(Rt0_ref)
confirm_equal( rt,
               rt0_ref,
               msg='rt_from_Rt result')

rt, J_R = mrcal.rt_from_Rt(Rt0_ref, get_gradients = True)
J_R_ref = grad(r_from_R,
               Rt0_ref[:3,:])
confirm_equal( rt,
               rt0_ref,
               msg='rt_from_Rt result')
confirm_equal( J_R,
               J_R_ref,
               msg='rt_from_Rt grad result')

Rt = mrcal.Rt_from_rt(rt0_ref)
confirm_equal( Rt,
               Rt0_ref,
               msg='Rt_from_rt result')

Rt, J_r = mrcal.Rt_from_rt(rt0_ref, get_gradients = True)
J_r_ref = grad(R_from_r,
               rt0_ref[:3])
confirm_equal( Rt,
               Rt0_ref,
               msg='Rt_from_rt result')
confirm_equal( J_r,
               J_r_ref,
               msg='Rt_from_rt grad result')

Rt = mrcal.invert_Rt(Rt0_ref)
confirm_equal( Rt,
               invert_Rt(Rt0_ref),
               msg='invert_Rt result')

rt = mrcal.invert_rt(rt0_ref)
confirm_equal( rt,
               invert_rt(rt0_ref),
               msg='invert_rt result')

rt,dt_dr,dt_dt = mrcal.invert_rt(rt0_ref, get_gradients = True)
drt_drt_ref = grad(invert_rt,
                   rt0_ref)
confirm_equal( rt,
               invert_rt(rt0_ref),
               msg='invert_rt with grad result')
confirm_equal( dt_dr,
               drt_drt_ref[3:,:3],
               msg='invert_rt dt/dr result')
confirm_equal( dt_dt,
               drt_drt_ref[3:,3:],
               msg='invert_rt dt/dt result')
confirm_equal( -np.eye(3),
               drt_drt_ref[:3,:3],
               msg='invert_rt dr/dr assumption')
confirm_equal( np.zeros((3,3)),
               drt_drt_ref[:3,3:],
               msg='invert_rt dr/dt assumption')

Rt2 = mrcal.compose_Rt(Rt0_ref, Rt1_ref)
confirm_equal( Rt2,
               compose_Rt(Rt0_ref, Rt1_ref),
               msg='compose_Rt result')

rt2,dr2_dr0,dr2_dr1,dt2_dr0,dt2_dt1 = mrcal.compose_rt(rt0_ref, rt1_ref, get_gradients=True)
dr2_dr0_ref = grad(lambda r0: compose_rt( nps.glue(r0,rt0_ref[3:], axis=-1), rt1_ref)[:3], rt0_ref[:3])
dr2_dr1_ref = grad(lambda r1: compose_rt( rt0_ref, nps.glue(r1,rt1_ref[3:], axis=-1))[:3], rt1_ref[:3])
dt2_dr0_ref = grad(lambda r0: compose_rt( nps.glue(r0,rt0_ref[3:], axis=-1), rt1_ref)[3:], rt0_ref[:3])
dt2_dt1_ref = grad(lambda t1: compose_rt( rt0_ref, nps.glue(rt1_ref[:3],t1, axis=-1))[3:], rt0_ref[3:])
confirm_equal( rt2,
               compose_rt(rt0_ref, rt1_ref),
               msg='compose_rt result')
confirm_equal( dr2_dr0,
               dr2_dr0_ref,
               msg='compose_rt dr2_dr0')
confirm_equal( dr2_dr1,
               dr2_dr1_ref,
               msg='compose_rt dr2_dr1')
confirm_equal( dt2_dr0,
               dt2_dr0_ref,
               msg='compose_rt dt2_dr0')
confirm_equal( dt2_dt1,
               dt2_dt1_ref,
               msg='compose_rt dt2_dt1')
finish()
