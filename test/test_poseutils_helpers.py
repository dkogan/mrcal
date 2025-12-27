r'''
Simple reference implementations of poseutils functions

I compare the mrcal results against these
'''

import sys
import numpy as np
import numpysane as nps
import os
testdir = os.path.dirname(os.path.realpath(__file__))

# I import the LOCAL mrcal since that's what I'm testing
sys.path[:0] = f"{testdir}/..",
import mrcal

import scipy.linalg


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

    # see the comment in r_from_R_core() for lots of detail

    #       [ R21 - R12 ]
    #   u = [ R02 - R20 ]
    #       [ R10 - R01 ]
    #   u = 2 sin(th) axis
    u = np.array((R[2,1] - R[1,2],
                  R[0,2] - R[2,0],
                  R[1,0] - R[0,1] ))

    costh = (np.trace(R) - 1.)/2.

    if nps.norm2(u) > 1e-12:
        # normal path
        if   costh >  1.0: th = 0
        elif costh < -1.0: th = np.pi
        else:              th = np.arccos(costh)
        return u / nps.mag(u) * th

    # small mag(u). Can't divide by it. But I can look at the limit.
    # th ~ 0 or th ~ 180
    if costh > 0:
        # th ~ 0
        # r = axis th = u / (2 sinth)*th ~ u/2
        return u/2.

    # th ~ 180

    # I need to set rcond because grad() might cause this function to be called
    # with not-quite-rotation matrices
    axis = scipy.linalg.null_space(R - mrcal.identity_R(),
                                   rcond = 1e-6).ravel()
    if axis.size != 3:
        raise Exception("Reference r_from_R implementation did something wrong...")

    # r_from_R_core() has comments. I use this: R - Rt = 2 sin(th) V where V =
    # skew_symmetric(axis)
    #
    # -> sin(th) ~ (R - Rt) / 2V
    #
    # The diagonals of both are 0. Both are anti-symmetric. So I need to look at
    # the 3 off-diagonal elements only.
    #
    # The top-right corner of V is    {-axis[2] axis[1] -axis[0]}
    # The top-right corner of R-Rt is {R01-R10  R02-R20 R12-R21}
    Roffdiag = np.array( (R[0,1]-R[1,0],
                          R[0,2]-R[2,0],
                          R[1,2]-R[2,1]))
    Voffdiag = np.array((-axis[2], axis[1], -axis[0]))
    i = np.abs(Voffdiag) > 0.1
    sinth = np.mean( Roffdiag[i] / Voffdiag[i] ) / 2.
    th = np.arctan2(sinth,costh)
    return axis*th


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

def normalize_r(r):
    r'''If abs(rot) > 180deg -> flip direction, abs(rot) <- 180-abs(rot)'''
    norm2 = nps.norm2(r)
    if norm2 < np.pi*np.pi:
        return r

    mag = np.sqrt(norm2)
    r_unit = r / mag

    mag = mag % (np.pi*2.)
    if mag < np.pi:
        # same direction, but fewer full rotations
        return r_unit*mag

    return -r_unit * (np.pi*2. - mag)

def normalize_rt(rt):
    return nps.glue(normalize_r(rt[:3]),
                    rt[3:],
                    axis=-1)

def compose_r(r0, r1):
    r'''Simple reference implementation'''
    return r_from_R(nps.matmult( R_from_r(r0),
                                 R_from_r(r1)))

def compose_rt(rt0, rt1):
    r'''Simple reference implementation'''
    return rt_from_Rt(compose_Rt( Rt_from_rt(rt0),
                                  Rt_from_rt(rt1)))
