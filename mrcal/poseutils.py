#!/usr/bin/python3

from __future__ import print_function

import numpy as np
import numpysane as nps

# for python3
from functools import reduce

from . import _poseutils

def r_from_R(R, get_gradients=False):
    r'''Compute a Rodrigues vector from a rotation matrix

    if not get_gradients: return r
    else:                 return (r,dr/dR)

    dr/dR is a (3,3,3) array where the first dimension selects the element of r
    in question, and the last 2 dimensions selects the element of R

    This function supports broadcasting fully

    '''
    if get_gradients:
        return _poseutils._r_from_R_withgrad(R)
    return _poseutils._r_from_R(R)

def R_from_r(r, get_gradients=False):
    r'''Compute a rotation matrix from a Rodrigues vector

    if not get_gradients: return R
    else:                 return (R,dR/dr)

    dR/dr is a (3,3,3) array where the first two dimensions select the element
    of R in question, and the last dimension selects the element of r

    This function supports broadcasting fully

    '''
    if get_gradients:
        return _poseutils._R_from_r_withgrad(r)
    return _poseutils._R_from_r(r)


def compose_Rt(*args):
    r'''Composes Rt transformations

    Given some number (2 or more, presumably) of Rt transformations, returns
    their composition

    '''
    return reduce( _poseutils._compose_Rt, args, _poseutils.identity_Rt() )

def compose_rt(*args):
    r'''Composes rt transformations

    Given some number (2 or more, presumably) of rt transformations, returns
    their composition

    '''

    return _poseutils.rt_from_Rt( compose_Rt( *[_poseutils.Rt_from_rt(rt) for rt in args] ) )

def rotate_point_r(r, x, get_gradients=False):
    r'''Rotates a point by a given Rodrigues rotation

    if not get_gradients: return u=r(x)
    else:                 return (u=r(x),du/dr,du/dx)

    This function supports broadcasting fully

    '''
    if not get_gradients:
        return _poseutils._rotate_point_r(r,x)
    return _poseutils._rotate_point_r_withgrad(r,x)

def rotate_point_R(R, x, get_gradients=False):
    r'''Rotates a point by a given rotation matrix

    if not get_gradients: return u=R(x)
    else:                 return (u=R(x),du/dR,du/dx)

    du/dR is a (3,3,3) array where the first dimension selects the element of u,
    and the last 2 dimensions select the element of R

    This function supports broadcasting fully.

    '''
    if not get_gradients:
        return _poseutils._rotate_point_R(R,x)
    return _poseutils._rotate_point_R_withgrad(R,x)

def transform_point_rt(rt, x, get_gradients=False):
    r'''Transforms a point by a given rt transformation

    if not get_gradients: return u=rt(x)
    else:                 return (u=rt(x),du/dr,du/dt,du/dx)

    This function supports broadcasting fully

    '''

    # Should do this nicer in the C code. But for the time being, this will do
    rt = np.ascontiguousarray(rt)
    x  = np.ascontiguousarray(x)

    if not get_gradients:
        return _poseutils._transform_point_rt(rt,x)
    return _poseutils._transform_point_rt_withgrad(rt,x)

def transform_point_Rt(Rt, x, get_gradients=False):
    r'''Transforms a point by a given Rt transformation

    if not get_gradients: return u=Rt(x)
    else:                 return (u=Rt(x),du/dR,du/dt,du/dx)

    du/dR is a (3,3,3) array where the first dimension selects the element of u,
    and the last 2 dimensions select the element of R

    This function supports broadcasting fully.

    '''

    # Should do this nicer in the C code. But for the time being, this will do
    Rt = np.ascontiguousarray(Rt)
    x  = np.ascontiguousarray(x)

    if not get_gradients:
        return _poseutils._transform_point_Rt(Rt,x)
    return _poseutils._transform_point_Rt_withgrad(Rt,x)

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
