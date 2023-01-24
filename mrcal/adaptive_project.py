#!/usr/bin/python3

import sys
import mrcal
import numpy as np
import numpysane as nps




@nps.broadcast_define( ( (), (2,2)),
                       ())
def _az_from_qx_sparse(qx,
                       c12,
                       scale,
                       Naz):
    r'''For a discrete set of points. Broadcasts in Python'''
    qxmid = (Naz-1)/2
    xs = (qx - qxmid) / scale
    c1l = c12[0,0]
    c2l = c12[0,1]
    c1r = c12[1,0]
    c2r = c12[1,1]
    return \
        (xs * (c1l + xs*c2l))*(1 - (np.sign(xs)>0)) + \
        (xs * (c1r + xs*c2r))*(    (np.sign(xs)>0))


def _az_from_qx_dense(qx,    # shape (Nel,Naz)
                      c12,   # shape (Nel,2,2)
                      scale, # scalar
                      Naz,   # scalar
                      ):
    r'''For processing a whole image at a time. Broadcasts in C'''
    qxmid = (Naz-1)/2

    # shape (Nel,Naz)
    xs = (qx - qxmid) / scale

    # shape (Nel,1)
    c1l = c12[:,0,0,np.newaxis]
    c2l = c12[:,0,1,np.newaxis]
    c1r = c12[:,1,0,np.newaxis]
    c2r = c12[:,1,1,np.newaxis]

    return \
        (xs * (c1l + xs*c2l))*(1 - (np.sign(xs)>0)) + \
        (xs * (c1r + xs*c2r))*(    (np.sign(xs)>0))

def az_from_qx(qx,
               *,
               # cookie
               c12,
               scale,
               Naz,
               **extra):

    if c12.ndim == 3               and \
       c12.shape[1:] == (2,2)      and \
       qx.shape[-1] == Naz:
        return _az_from_qx_dense(qx,c12,scale,Naz)
    return _az_from_qx_sparse(qx,c12,scale,Naz)


@nps.broadcast_define( ( (), (2,2)),
                       ())
def _qx_from_az_sparse(az,
                       c12,
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
    c1l = c12[0,0]
    c2l = c12[0,1]
    c1r = c12[1,0]
    c2r = c12[1,1]
    xs = \
        (-c1l + np.sqrt(c1l*c1l + 4.*c2l*az)) / (2.*c2l)*(1 - (np.sign(az)>0)) + \
        (-c1r + np.sqrt(c1r*c1r + 4.*c2r*az)) / (2.*c2r)*(    (np.sign(az)>0))

    qxmid = (Naz-1)/2
    qx = xs*scale + qxmid
    return qx

def _qx_from_az_dense(az,   # shape (Nel,Naz)
                      c12,  # shape (Nel,2,2)
                      scale,# scalar
                      Naz,  # scalar
                      ):
    # shape (Nel,1)
    c1l = c12[:,0,0,np.newaxis]
    c2l = c12[:,0,1,np.newaxis]
    c1r = c12[:,1,0,np.newaxis]
    c2r = c12[:,1,1,np.newaxis]

    # shape (Nel,Naz)
    xs = \
        (-c1l + np.sqrt(c1l*c1l + 4.*c2l*az)) / (2.*c2l)*(1 - (np.sign(az)>0)) + \
        (-c1r + np.sqrt(c1r*c1r + 4.*c2r*az)) / (2.*c2r)*(    (np.sign(az)>0))

    qxmid = (Naz-1)/2

    # shape (Nel,Naz)
    qx = xs*scale + qxmid
    return qx


def qx_from_az(az,
               *,
               # cookie
               c12,
               scale,
               Naz,
               **extra):

    if c12.ndim == 3               and \
       c12.shape[1:] == (2,2)      and \
       az.shape[-1] == Naz:
        return _qx_from_az_dense(az,c12,scale,Naz)
    return _qx_from_az_sparse(az,c12,scale,Naz)


def project_adaptive_rectification(p, cookie):

    fy   = cookie['fy']
    cy   = cookie['cy']
    daz1 = cookie['daz1']

    if daz1:
        raise Exception(f"project_adaptive_rectification(daz1 != 0) not implemented")

    azel  = mrcal.project_latlon(p)
    az,el = nps.mv(azel, -1, 0)

    q = np.zeros(azel.shape,
                 dtype=float)

    q[...,1] = el * fy + cy
    q[...,0] = qx_from_az(az, **cookie)

    return q


def unproject_adaptive_rectification(q,
                                     cookie,
                                     disparity = None):

    fy   = cookie['fy']
    cy   = cookie['cy']
    daz1 = cookie['daz1']
    Naz  = cookie['Naz']
    Nel  = cookie['Nel']

    if daz1:
        raise Exception(f"unproject_adaptive_rectification(daz1 != 0) not implemented")

    if q is None:
        # full imager
        azel = np.zeros((Nel,Naz,2),
                        dtype=float)
        qy = np.arange(Nel)
        nps.transpose(azel[...,1])[:] += (qy - cy) / fy

        if disparity is None:
            disparity = np.zeros((Naz,))
        azel[...,0] = az_from_qx(np.arange(Naz)-disparity, **cookie)

        return mrcal.unproject_latlon(azel)

    azel = np.zeros(q.shape,
                    dtype=float)
    azel[...,1] = (q[...,1] - cy) / fy
    if disparity is None:
        disparity = np.zeros(q.shape[:-1])

    i0 = q[...,1].astype(int)
    s  = q[...,1] - i0
    cookie_c12_floor = dict(cookie)
    cookie_c12_ceil  = dict(cookie)
    cookie_c12_floor['c12'] = cookie['c12'][i0,    ...]
    cookie_c12_ceil ['c12'] = cookie['c12'][i0+1,  ...]
    azel[...,0] = \
        az_from_qx(q[...,0]-disparity, **cookie_c12_floor) * (1-s) + \
        az_from_qx(q[...,0]-disparity, **cookie_c12_ceil ) * (  s)

    return mrcal.unproject_latlon(azel)


if __name__ == '__main__':

    import pickle

    filename = "/tmp/cookie"
    with open(filename, "rb") as f:
        (qx,
         az_domain,
         fy,cy) = \
             pickle.load(f)

    # q_nominal = ((1574.738,855.411))
    p = np.array([0.57652792, 0.0982791 , 0.81114535])
    # q_adaptive = ((1096.8,855.7))

    q = \
        project_adaptive_rectification(p,
                                       qx          = qx,
                                       az_domain   = az_domain,
                                       fy          = fy,
                                       cy          = cy,
                                       daz1      = 0)
    pp = \
        unproject_adaptive_rectification(q,
                                         qx          = qx,
                                         az_domain   = az_domain,
                                         fy          = fy,
                                         cy          = cy,
                                         daz1      = 0)

    print(p)
    print(pp)
    print(q)
