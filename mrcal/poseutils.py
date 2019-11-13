#!/usr/bin/python3

from __future__ import print_function

import numpy as np
import numpysane as nps
import cv2

# for python3
from functools import reduce


@nps.broadcast_define( ((3,3),),
                       (3,), )
def r_from_R(R):
    r'''Broadcasting-aware wrapper cvRodrigues

    This handles the R->r direction, and does not report the gradient

    '''

    return cv2.Rodrigues(R)[0].ravel()


@nps.broadcast_define( ((3,),),
                       (3,3), )
def R_from_r(r):
    r'''Broadcasting-aware wrapper cvRodrigues

    This handles the r->R direction, and does not report the gradient

    '''

    return cv2.Rodrigues(r)[0]

@nps.broadcast_define( ((6,),),
                       (4,3,), )
def Rt_from_rt(rt):
    r'''Convert an rt pose to an Rt pose'''

    r = rt[:3]
    t = rt[3:]
    R = cv2.Rodrigues(r.astype(float))[0]
    return nps.glue( R, t, axis=-2)

@nps.broadcast_define( ((4,3),),
                       (6,), )
def rt_from_Rt(Rt):
    r'''Convert an Rt pose to an rt pose'''

    R = Rt[:3,:]
    t = Rt[ 3,:]
    r = cv2.Rodrigues(R.astype(float))[0].ravel()
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
    #     print(np.max(np.abs( (dvfit_expected - dvfit_observed) / ( (np.abs(dvfit_expected) + np.abs(dvfit_observed))/2.))))
    #     vfit1 = mrcal.transform_point_rt(rt + drt, vref)
    #     dvfit_observed = vfit1-vfit
    #     dvfit_expected = nps.matmult(dvfit_drt, nps.dummy(drt,-1))[...,0]
    #     print(np.max(np.abs( (dvfit_expected - dvfit_observed) / ( (np.abs(dvfit_expected) + np.abs(dvfit_observed))/2.))))
    #     sys.exit()

    R,dRdr = cv2.Rodrigues(rt[:3].astype(float))
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

@nps.broadcast_define( ((4,),),
                       (3,3) )
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

def quat_from_R(R):
    r'''Unit quaternion from a rotation matrix

    This is mostly for compatibility with some old stuff. I don't really use
    quaternions

    This comes directly from the scipy project, the from_dcm() function in

      https://github.com/scipy/scipy/blob/master/scipy/spatial/transform/rotation.py

    Commit: 1169d27ad47a29abafa8a3d2cb5b67ff0df80a8f

    License:

    Copyright (c) 2001-2002 Enthought, Inc.  2003-2019, SciPy Developers.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:

    1. Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    3. Neither the name of the copyright holder nor the names of its
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    '''

    R = nps.dummy(R, 0)
    num_rotations = R.shape[0]

    decision_matrix = np.empty((num_rotations, 4))
    decision_matrix[:, :3] = R.diagonal(axis1=1, axis2=2)
    decision_matrix[:, -1] = decision_matrix[:, :3].sum(axis=1)
    choices = decision_matrix.argmax(axis=1)

    quat = np.empty((num_rotations, 4))

    ind = np.nonzero(choices != 3)[0]
    i = choices[ind]
    j = (i + 1) % 3
    k = (j + 1) % 3

    quat[ind, i+1] = 1 - decision_matrix[ind, -1] + 2 * R[ind, i, i]
    quat[ind, j+1] = R[ind, j, i] + R[ind, i, j]
    quat[ind, k+1] = R[ind, k, i] + R[ind, i, k]
    quat[ind, 0  ] = R[ind, k, j] - R[ind, j, k]

    ind = np.nonzero(choices == 3)[0]
    quat[ind, 1] = R[ind, 2, 1] - R[ind, 1, 2]
    quat[ind, 2] = R[ind, 0, 2] - R[ind, 2, 0]
    quat[ind, 3] = R[ind, 1, 0] - R[ind, 0, 1]
    quat[ind, 0] = 1 + decision_matrix[ind, -1]

    quat /= np.linalg.norm(quat, axis=1)[:, None]
    return quat[0]
