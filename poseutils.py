#!/usr/bin/python2

import numpy as np
import numpysane as nps
import cv2


def Rt_from_rt(rt):
    r'''Convert an rt pose to an Rt pose'''

    r = rt[:3]
    t = rt[3:]
    R = cv2.Rodrigues(r)[0]
    return nps.glue( R, t, axis=-2)

def rt_from_Rt(Rt):
    r'''Convert an Rt pose to an rt pose'''

    R = Rt[:3,:]
    t = Rt[ 3,:]
    r = cv2.Rodrigues(R)[0].ravel()
    return nps.glue( r, t, axis=-1)

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

def invert_rt(rt):
    r'''Given a (r,t) transformation, return the inverse transformation
    '''

    return rt_from_Rt(invert_Rt(Rt_from_rt(rt)))

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

def compose_rt(rt1, rt2):
    r'''Composes two rt transformations
    '''
    return rt_from_Rt( compose_Rt( Rt_from_rt(rt1),
                                   Rt_from_rt(rt2)))
