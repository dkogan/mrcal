#!/usr/bin/python2

import numpy as np
import numpysane as nps
import cv2


@nps.broadcast_define( ((6,),),
                       (4,3,), )
def Rt_from_rt(rt):
    r'''Convert an rt pose to an Rt pose'''

    r = rt[:3]
    t = rt[3:]
    R = cv2.Rodrigues(r)[0]
    return nps.glue( R, t, axis=-2)

@nps.broadcast_define( ((4,3),),
                       (6,), )
def rt_from_Rt(Rt):
    r'''Convert an Rt pose to an rt pose'''

    R = Rt[:3,:]
    t = Rt[ 3,:]
    r = cv2.Rodrigues(R)[0].ravel()
    return nps.glue( r, t, axis=-1)

@nps.broadcast_define( ((4,3,),),
                       (4,3,), )
def invert_Rt(Rt):
    r'''Given a (R,t) transformation, return the inverse transformation

    I need to reverse the transformation:
      b = Ra + t  -> a = R'b - R't

    '''

    R = Rt[:3,:]
    t = Rt[ 3,:]

    t = -nps.matmult(t, R)
    R = nps.transpose(R)
    return nps.glue(R,t, axis=-2)

@nps.broadcast_define( ((6,),),
                       (6,), )
def invert_rt(rt):
    r'''Given a (r,t) transformation, return the inverse transformation
    '''

    return rt_from_Rt(invert_Rt(Rt_from_rt(rt)))

@nps.broadcast_define( ((4,3),(4,3)),
                       (4,3,), )
def compose_Rt(Rt1, Rt2):
    r'''Composes two Rt transformations

    y = R1(R2 x + t2) + t1 = R1 R2 x + R1 t2 + t1
    '''
    R1 = Rt1[:3,:]
    t1 = Rt1[ 3,:]
    R2 = Rt2[:3,:]
    t2 = Rt2[ 3,:]

    R = nps.matmult(R1,R2)
    t = nps.matmult(t2, nps.transpose(R1)) + t1
    return nps.glue(R,t, axis=-2)

@nps.broadcast_define( ((6,),(6,)),
                       (6,), )
def compose_rt(rt1, rt2):
    r'''Composes two rt transformations'''
    return rt_from_Rt( compose_Rt( Rt_from_rt(rt1),
                                   Rt_from_rt(rt2)))

def identity_Rt():
    r'''Returns an identity Rt transform'''
    return nps.glue( np.eye(3), np.zeros(3), axis=-2)

def identity_rt():
    r'''Returns an identity rt transform'''
    return np.zeros(6, dtype=float)

@nps.broadcast_define( ((4,3),(3,)),
                       (3,), )
def transform_point_Rt(Rt, x):
    r'''Transforms a given point by a given Rt transformation'''
    R = Rt[:3,:]
    t = Rt[ 3,:]
    return nps.matmult(x, nps.transpose(R)) + t

@nps.broadcast_define( ((6,),(3,)),
                       (3,), )
def transform_point_rt(rt, x):
    r'''Transforms a given point by a given rt transformation'''
    return transform_point_Rt(Rt_from_rt(rt))


def R_from_quat(q):
    r'''Rotation matrix from a unit quaternion

    This is mostly for compatibility with some old stuff. I don't really use
    quaternions

    '''

    # From the expression in wikipedia
    r,i,j,k = q[:]

    ii = i*i
    ij = i*j
    ik = i*k
    ir = i*r
    jj = j*j
    jk = j*k
    jr = j*r
    kk = k*k
    kr = k*r

    return np.array((( 1-2*(jj+kk),   2*(ij-kr),   2*(ik+jr)),
                     (   2*(ij+kr), 1-2*(ii+kk),   2*(jk-ir)),
                     (   2*(ik-jr),   2*(jk+ir), 1-2*(ii+jj))))
