#!/usr/bin/python3

import sys
import mrcal
import numpy as np
import numpysane as nps




@nps.broadcast_define( ( (), (3,)),
                       ())
def _az_from_qx_sparse(qx,
                       c12l2r,
                       scale,
                       Naz):
    r'''For a discrete set of points. Broadcasts in Python'''
    qxmid = (Naz-1)/2
    xs = (qx - qxmid) / scale
    c1,c2l,c2r = c12l2r
    return \
        (xs * (c1 + xs*c2l))*(1 - (np.sign(xs)>0)) + \
        (xs * (c1 + xs*c2r))*(    (np.sign(xs)>0))


def _az_from_qx_dense(qx,     # shape (Nel,Naz)
                      c12l2r, # shape (Nel,3)
                      scale,  # scalar
                      Naz,    # scalar
                      ):
    r'''For processing a whole image at a time. Broadcasts in C'''
    qxmid = (Naz-1)/2

    # shape (Nel,Naz)
    xs = (qx - qxmid) / scale

    # shape (Nel,1)
    c1  = c12l2r[:,0,np.newaxis]
    c2l = c12l2r[:,1,np.newaxis]
    c2r = c12l2r[:,2,np.newaxis]

    return \
        (xs * (c1 + xs*c2l))*(1 - (np.sign(xs)>0)) + \
        (xs * (c1 + xs*c2r))*(    (np.sign(xs)>0))

def az_from_qx(qx,
               *,
               # cookie
               c12l2r,
               scale,
               Naz,
               **extra):

    if c12l2r.ndim == 2              and \
       c12l2r.shape[1:] == (3,)      and \
       qx.shape[-1] == Naz:
        return _az_from_qx_dense(qx,c12l2r,scale,Naz)
    return _az_from_qx_sparse(qx,c12l2r,scale,Naz)


@nps.broadcast_define( ( (), (3,)),
                       ())
def _qx_from_az_sparse(az,
                       c12l2r,
                       scale,
                       Naz):
    r'''For a discrete set of points. Broadcasts in Python'''
    # I have az = c1 xs + c2 xs*xs. Around (xs=0,az=0) I have daz/dxs > 0. So I
    # can pick which parabola I'm looking at based on xs>0 or az>0. At xs=0
    # daz/dxs = c1, so c1>0.
    #
    #   xs = (-c1 +- sqrt(c1^2 + 4*c2*az)) / 2*c2
    #
    # Generally (from empirical plot visualization) with az>0 I have c2<0 and
    # with az<0 I have c2>0, so 4*c2*az<0. So if a real solution exists, both
    # will be >0 or both will be <0. Of the two I pick the one that has abs(xs)
    # closer to 0. This is (-c1 + sqrt(c1^2 + 4*c2*az)) / 2*c2
    c1,c2l,c2r = c12l2r
    xs = \
        (-c1 + np.sqrt(c1l*c1l + 4.*c2l*az)) / (2.*c2l)*(1 - (np.sign(az)>0)) + \
        (-c1 + np.sqrt(c1r*c1r + 4.*c2r*az)) / (2.*c2r)*(    (np.sign(az)>0))

    qxmid = (Naz-1)/2
    qx = xs*scale + qxmid
    return qx

def _qx_from_az_dense(az,   # shape (Nel,Naz)
                      c12l2r,  # shape (Nel,2,2)
                      scale,# scalar
                      Naz,  # scalar
                      ):
    # shape (Nel,1)
    c1  = c12l2r[:,0,np.newaxis]
    c2l = c12l2r[:,1,np.newaxis]
    c2r = c12l2r[:,2,np.newaxis]

    # shape (Nel,Naz)
    xs = \
        (-c1 + np.sqrt(c1l*c1l + 4.*c2l*az)) / (2.*c2l)*(1 - (np.sign(az)>0)) + \
        (-c1 + np.sqrt(c1r*c1r + 4.*c2r*az)) / (2.*c2r)*(    (np.sign(az)>0))

    qxmid = (Naz-1)/2

    # shape (Nel,Naz)
    qx = xs*scale + qxmid
    return qx


def qx_from_az(az,
               *,
               # cookie
               c12l2r,
               scale,
               Naz,
               **extra):

    if c12l2r.ndim == 2              and \
       c12l2r.shape[1:] == (3,)      and \
       az.shape[-1] == Naz:
        return _qx_from_az_dense(az,c12l2r,scale,Naz)
    return _qx_from_az_sparse(az,c12l2r,scale,Naz)


def undo_adaptive_rectification(cookie,
                                qrect     = None, # input and output (unless qrect is None)
                                disparity = None,
                                # shape (Nel,Naz)
                                daz       = None):

    fx,fy,cx,cy = cookie['fxycxy']
    Naz         = cookie['Naz']
    Nel         = cookie['Nel']

    if qrect is None:
        # full imager
        qrect = np.zeros((Nel,Naz,2),
                         dtype=float)
        if disparity is None:
            disparity = np.zeros((Naz,))
        # shape (Nel,Naz)
        az = az_from_qx(np.arange(Naz)-disparity, **cookie)
        if daz is not None:
            az += daz
        qrect[...,0] = az*fx + cx
        qrect[...,1] = nps.transpose(np.arange(Nel,dtype=float))
        return qrect

    if disparity is None:
        disparity = np.zeros(qrect.shape[:-1])

    i0 = qrect[...,1].astype(int)
    si = qrect[...,1] - i0

    cookie_c12l2r_floor = dict(cookie)
    cookie_c12l2r_ceil  = dict(cookie)
    cookie_c12l2r_floor['c12l2r'] = cookie['c12l2r'][i0,    ...]
    cookie_c12l2r_ceil ['c12l2r'] = cookie['c12l2r'][i0+1,  ...]
    az = \
        az_from_qx(qrect[...,0]-disparity, **cookie_c12l2r_floor) * (1-si) + \
        az_from_qx(qrect[...,0]-disparity, **cookie_c12l2r_ceil ) * (  si)
    if daz is not None:
        j0 = qrect[...,0].astype(int)
        sj = qrect[...,0] - j0

        az += \
            daz[i0,  j0  ]*(1-si)*(1-sj) + \
            daz[i0+1,j0  ]*   si *(1-sj) + \
            daz[i0  ,j0+1]*(1-si)*   sj  + \
            daz[i0+1,j0+1]*si    *   sj
    qrect[...,0] = az*fx + cx
    return qrect
