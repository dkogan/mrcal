#!/usr/bin/python3

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
def _compose_Rt(Rt1, Rt2):
    r'''Composes exactly 2 Rt transformations

    Given 2 Rt transformations, returns their composition. This is an internal
    function used by mrcal.compose_Rt(), which supports >2 input transformations

    '''
    R1 = Rt1[:3,:]
    t1 = Rt1[ 3,:]
    R2 = Rt2[:3,:]
    t2 = Rt2[ 3,:]

    R = nps.matmult(R1,R2)
    t = nps.matmult(t2, nps.transpose(R1)) + t1
    return nps.glue(R,t, axis=-2)

def compose_Rt(*args):
    r'''Composes Rt transformations

    Given some number (2 or more, presumably) of Rt transformations, returns
    their composition

    '''
    return reduce( _compose_Rt, args, np.array(((1,0,0),
                                                (0,1,0),
                                                (0,0,1),
                                                (0,0,0)), dtype=float) )

def compose_rt(*args):
    r'''Composes rt transformations

    Given some number (2 or more, presumably) of rt transformations, returns
    their composition

    '''

    return rt_from_Rt( compose_Rt( *[Rt_from_rt(rt) for rt in args] ) )

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
def _transform_point_rt(rt, x):
    r'''Transforms a given point by a given rt transformation'''
    return transform_point_Rt(Rt_from_rt(rt), x)

@nps.broadcast_define( ((6,),(3,)),
                       (3,10), )
def _transform_point_rt_withgradient(rt, x):
    r'''Transforms a given point by a given rt transformation

    Returns a projection result AND a gradient'''


    # Test. I have vref.shape=(16,3) and I have some rt. This prints the
    # worst-case relative errors. Both should be ~0
    #
    #     vfit,dvfit_drt,dvfit_dvref = mrcal.transform_point_rt(rt, vref, get_gradients=True)
    #     dvref = np.random.random(vref.shape)*1e-5
    #     drt   = np.random.random(rt.shape)*1e-5
    #     vfit1 = mrcal.transform_point_rt(rt, vref + dvref)
    #     dvfit_observed = vfit1-vfit
    #     dvfit_expected = nps.matmult(dvfit_dvref, nps.dummy(dvref,-1))[...,0]
    #     print np.max(np.abs( (dvfit_expected - dvfit_observed) / ( (np.abs(dvfit_expected) + np.abs(dvfit_observed))/2.)))
    #     vfit1 = mrcal.transform_point_rt(rt + drt, vref)
    #     dvfit_observed = vfit1-vfit
    #     dvfit_expected = nps.matmult(dvfit_drt, nps.dummy(drt,-1))[...,0]
    #     print np.max(np.abs( (dvfit_expected - dvfit_observed) / ( (np.abs(dvfit_expected) + np.abs(dvfit_observed))/2.)))
    #     sys.exit()

    R,dRdr = cv2.Rodrigues(rt[:3])
    dRdr = nps.transpose(dRdr) # fix opencv's weirdness. Now shape=(9,3)

    xx = nps.matmult(x, nps.transpose(R)) + rt[3:]
    d_dx = R

    # d_dr = nps.matmult(dxx_dRflat, dRdr)
    # where dxx_dRflat =
    #   ( x0 x1 x2  0  0  0 0  0  0  )
    #   (  0  0  0 x0 x1 x2 0  0  0  )
    #   (  0  0  0  0  0  0 x0 x1 x2 )
    # I don't actually deal with the 0s
    d_dr = nps.glue(nps.matmult(x, dRdr[0:3,:]),
                    nps.matmult(x, dRdr[3:6,:]),
                    nps.matmult(x, dRdr[6:9,:]),
                    axis = -2)
    d_dt = np.eye(3)
    return nps.glue(nps.transpose(xx), d_dx, d_dr, d_dt,
                    axis = -1)

def transform_point_rt(rt, x, get_gradients=False):
    r'''Transforms a given point by a given rt transformation

    if get_gradients: return a tuple of (transform_result,d_drt,d_dx)

    This function supports broadcasting fully

    '''
    if not get_gradients:
        return _transform_point_rt(rt,x)

    # We're getting gradients. The inner broadcastable function returns a single
    # packed array. I unpack it into components before returning. Note that to
    # make things line up, the inner function uses a COLUMN vector to store the
    # projected result, not a ROW vector
    result = _transform_point_rt_withgradient(rt,x)
    x      = result[..., 0  ]
    d_dx   = result[..., 1:4]
    d_drt  = result[..., 4: ]
    return x,d_drt,d_dx

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
