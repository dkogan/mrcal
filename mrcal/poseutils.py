#!/usr/bin/python3


import numpy as np
import numpysane as nps

# for python3
from functools import reduce

from . import _poseutils

def r_from_R(R, get_gradients=False):
    r"""Compute a Rodrigues vector from a rotation matrix

SYNOPSIS

    r = mrcal.r_from_R(R)

    rotation_magnitude = nps.mag(r)
    rotation_axis      = r / rotation_magnitude

Given a rotation specified as a (3,3) rotation matrix, converts it to a
Rodrigues vector, a unit rotation axis scaled by the rotation magnitude, in
radians.

By default this function returns the Rodrigues vector(s) only. If we also want
gradients, pass get_gradients=True. Logic:

    if not get_gradients: return r
    else:                 return (r, dr/dR)

This function supports broadcasting fully.

ARGUMENTS

- R: array of shape (3,3). This matrix defines the rotation. It is assumed that
  this is a valid rotation (matmult(R,transpose(R)) = I, det(R) = 1), but that
  is not checked

- get_gradients: optional boolean. By default (get_gradients=False) we return an
  array of Rodrigues vectors. Otherwise we return a tuple of arrays of Rodrigues
  vectors and their gradients.

RETURNED VALUE

If not get_gradients: we return an array of Rodrigues vector(s). Each
broadcasted slice has shape (3,)

If get_gradients: we return a tuple of arrays containing the Rodrigues vector(s)
and the gradients (r, dr/dR)

1. The Rodrigues vector. Each broadcasted slice has shape (3,)

2. The gradient dr/dR. Each broadcasted slice has shape (3,3,3). The first
   dimension selects the element of r, and the last two dimensions select the
   element of R

    """

    if get_gradients:
        return _poseutils._r_from_R_withgrad(R)
    return _poseutils._r_from_R(R)

def R_from_r(r, get_gradients=False):
    r"""Compute a rotation matrix from a Rodrigues vector

SYNOPSIS

    r  = rotation_axis * rotation_magnitude

    R = mrcal.R_from_r(r)

Given a rotation specified as a Rodrigues vector (a unit rotation axis scaled by
the rotation magnitude, in radians.), converts it to a rotation matrix.

By default this function returns the rotation matrices only. If we also want
gradients, pass get_gradients=True. Logic:

    if not get_gradients: return R
    else:                 return (R, dR/dr)

This function supports broadcasting fully.

ARGUMENTS

- r: array of shape (3,). The Rodrigues vector that defines the rotation. This is
  a unit rotation axis scaled by the rotation magnitude, in radians

- get_gradients: optional boolean. By default (get_gradients=False) we return an
  array of rotation matrices. Otherwise we return a tuple of arrays of rotation
  matrices and their gradients.

RETURNED VALUE

If not get_gradients: we return an array of rotation matrices. Each broadcasted
slice has shape (3,3)

If get_gradients: we return a tuple of arrays containing the rotation matrices
and the gradients (R, dR/dr):

1. The rotation matrix. Each broadcasted slice has shape (3,3). This is a valid
   rotation: matmult(R,transpose(R)) = I, det(R) = 1

2. The gradient dR/dr. Each broadcasted slice has shape (3,3,3). The first two
   dimensions select the element of R, and the last dimension selects the
   element of r

    """

    if get_gradients:
        return _poseutils._R_from_r_withgrad(r)
    return _poseutils._R_from_r(r)


def compose_Rt(*Rt):
    r"""Compose Rt transformations

SYNOPSIS

    Rt10 = nps.glue(rotation_matrix10,translation10, axis=-2)
    Rt21 = nps.glue(rotation_matrix21,translation21, axis=-2)
    Rt32 = nps.glue(rotation_matrix32,translation32, axis=-2)

    print(Rt10.shape)
    ===>
    (4,3)

    Rt30 = mrcal.compose_Rt( Rt32, Rt21, Rt10 )

    print(x0.shape)
    ===>
    (3,)

    print( mrcal.transform_point_Rt(Rt30, x0) -
           mrcal.transform_point_Rt(Rt32,
             mrcal.transform_point_Rt(Rt21,
               mrcal.transform_point_Rt(Rt10, x0))) )
    ===>
    0

Given some number (2 or more, presumably) of Rt transformations, returns
their composition. An Rt transformation is a (4,3) array formed by
nps.glue(R,t, axis=-2) where R is a (3,3) rotation matrix and t is a (3,)
translation vector. This transformation is defined by a matrix multiplication
and an addition. x and t are stored as a row vector (that's how numpy stores
1-dimensional arrays), but the multiplication works as if x was a column vector
(to match linear algebra conventions):

    transform_point_Rt(Rt, x) = transpose( matmult(Rt[:3,:], transpose(x)) +
                                           transpose(Rt[3,:]) ) =
                              = matmult(x, transpose(Rt[:3,:])) +
                                Rt[3,:]

This function supports broadcasting fully, so we can compose lots of
transformations at the same time.

ARGUMENTS

- *Rt: a list of transformations to compose. Usually we'll be composing two
  transformations, but any number could be given here. Each broadcasted slice
  has shape (4,3).

RETURNED VALUE

An array of composed Rt transformations. Each broadcasted slice has shape (4,3)

    """
    return reduce( _poseutils._compose_Rt, Rt, _poseutils.identity_Rt() )


def compose_rt(*rt, get_gradients=False):
    r"""Compose rt transformations

SYNOPSIS

    r10 = rotation_axis10 * rotation_magnitude10
    r21 = rotation_axis21 * rotation_magnitude21
    r32 = rotation_axis32 * rotation_magnitude32

    print(rt10.shape)
    ===>
    (6,)

    rt30 = mrcal.compose_rt( rt32, rt21, rt10 )

    print(x0.shape)
    ===>
    (3,)

    print( mrcal.transform_point_rt(rt30, x0) -
           mrcal.transform_point_rt(rt32,
             mrcal.transform_point_rt(rt21,
               mrcal.transform_point_rt(rt10, x0))) )
    ===>
    0

    print( [arr.shape for arr in mrcal.compose_rt(rt21,rt10,
                                                  get_gradients = True)] )
    ===>
    [(6,), (3,3), (3,3), (3,3), (3,3)]

Given some number (2 or more, presumably) of rt transformations, returns their
composition. An rt transformation is a (6,) array formed by nps.glue(r,t,
axis=-1) where r is a (3,) Rodrigues vector and t is a (3,) translation vector.
This transformation is defined by a matrix multiplication and an addition. x and
t are stored as a row vector (that's how numpy stores 1-dimensional arrays), but
the multiplication works as if x was a column vector (to match linear algebra
conventions):

    transform_point_rt(rt, x) = transpose( matmult(R_from_r(rt[:3]), transpose(x)) +
                                           transpose(rt[3,:]) ) =
                              = matmult(x, transpose(R_from_r(rt[:3]))) +
                                rt[3:]

By default this function returns the composed transformations only. If we also
want gradients, pass get_gradients=True. This is supported ONLY if we have
EXACTLY 2 transformations to compose. Logic:

    if not get_gradients: return rt=compose(rt0,rt1)
    else:                 return (rt=compose(rt0,rt1), dr/dr0,dr/dr1,dt/dr0,dt/dt1)

Note that:

- dr/dt0 is not returned: it is always 0
- dr/dt1 is not returned: it is always 0
- dt/dr1 is not returned: it is always 0
- dt/dt0 is not returned: it is always the identity matrix

This function supports broadcasting fully, so we can compose lots of
transformations at the same time.

ARGUMENTS

- *rt: a list of transformations to compose. Usually we'll be composing two
  transformations, but any number could be given here. Each broadcasted slice
  has shape (6,)

- get_gradients: optional boolean. By default (get_gradients=False) we return an
  array of composed transformations. Otherwise we return a tuple of arrays of
  composed transformations and their gradients. Gradient reporting is only
  supported when exactly two transformations are given

RETURNED VALUE

If not get_gradients: we return an array of composed rt transformations. Each
broadcasted slice has shape (4,3)

If get_gradients: we return a tuple of arrays containing the composed
transformations and the gradients (rt=compose(rt0,rt1),
dr/dr0,dr/dr1,dt/dr0,dt/dt1):

1. The composed transformation. Each broadcasted slice has shape (6,)

2. The gradient dr/dr0. Each broadcasted slice has shape (3,3). The first
   dimension selects the element of r, and the last dimension selects the
   element of r0

3. The gradient dr/dr1. Each broadcasted slice has shape (3,3). The first
   dimension selects the element of r, and the last dimension selects the
   element of r1

4. The gradient dt/dr0. Each broadcasted slice has shape (3,3). The first
   dimension selects the element of t, and the last dimension selects the
   element of r0

5. The gradient dt/dt1. Each broadcasted slice has shape (3,3). The first
   dimension selects the element of t, and the last dimension selects the
   element of t1

Note that:

- dr/dt0 is not returned: it is always 0
- dr/dt1 is not returned: it is always 0
- dt/dr1 is not returned: it is always 0
- dt/dt0 is not returned: it is always the identity matrix

    """

    if get_gradients:
        if len(rt) != 2:
            raise Exception("compose_rt(get_gradients=True) is supported only if exactly 2 inputs are given")
        return _poseutils._compose_rt_withgrad(*rt)

    return _poseutils.rt_from_Rt( compose_Rt( *[_poseutils.Rt_from_rt(_rt) for _rt in rt] ) )

def rotate_point_r(r, x, get_gradients=False):
    r"""Rotate point(s) using a Rodrigues vector

SYNOPSIS

    r = rotation_axis * rotation_magnitude

    print(r.shape)
    ===>
    (3,)

    print(x.shape)
    ===>
    (10,3)

    print(mrcal.rotate_point_r(r, x).shape)
    ===>
    (10,3)

    print( [arr.shape for arr in mrcal.rotate_point_r(r, x,
                                                      get_gradients = True)] )
    ===>
    [(10,3), (10,3,3), (10,3,3)]

Rotate point(s) by a rotation matrix. The Rodrigues vector is converted to a
rotation matrix internally, and then this function is a matrix multiplication. x
is stored as a row vector (that's how numpy stores 1-dimensional arrays), but
the multiplication works as if x was a column vector (to match linear algebra
conventions):

    rotate_point_r(r,x) = transpose( matmult(R(r), transpose(x))) =
                        = matmult(x, transpose(R(r)))

By default this function returns the rotated points only. If we also want
gradients, pass get_gradients=True. Logic:

    if not get_gradients: return u=r(x)
    else:                 return (u=r(x),du/dr,du/dx)

This function supports broadcasting fully, so we can rotate lots of points at
the same time and/or apply lots of different rotations at the same time

ARGUMENTS

- r: array of shape (3,). The Rodrigues vector that defines the rotation. This is
  a unit rotation axis scaled by the rotation magnitude, in radians

- x: array of shape (3,). The point being rotated

- get_gradients: optional boolean. By default (get_gradients=False) we return an
  array of rotated points. Otherwise we return a tuple of arrays of rotated
  points and their gradients.

RETURNED VALUE

If not get_gradients: we return an array of rotated point(s). Each broadcasted
slice has shape (3,)

If get_gradients: we return a tuple of arrays containing the rotated points and
the gradients (u=r(x),du/dr,du/dx):

A tuple (u=r(x),du/dr,du/dx):

1. The rotated point(s). Each broadcasted slice has shape (3,)

2. The gradient du/dr. Each broadcasted slice has shape (3,3). The first
   dimension selects the element of u, and the last dimension selects the
   element of r

3. The gradient du/dx. Each broadcasted slice has shape (3,3). The first
   dimension selects the element of u, and the last dimension selects the
   element of x

    """
    if not get_gradients:
        return _poseutils._rotate_point_r(r,x)
    return _poseutils._rotate_point_r_withgrad(r,x)

def rotate_point_R(R, x, get_gradients=False):
    r"""Rotate point(s) using a rotation matrix

SYNOPSIS

    r = rotation_axis * rotation_magnitude
    R = mrcal.R_from_r(r)

    print(R.shape)
    ===>
    (3,3)

    print(x.shape)
    ===>
    (10,3)

    print( mrcal.rotate_point_R(R, x).shape )
    ===>
    (10,3)

    print( [arr.shape for arr in mrcal.rotate_point_R(R, x,
                                                      get_gradients = True)] )
    ===>
    [(10,3), (10,3,3,3), (10,3,3)]

Rotate point(s) by a rotation matrix. This is a matrix multiplication. x is
stored as a row vector (that's how numpy stores 1-dimensional arrays), but the
multiplication works as if x was a column vector (to match linear algebra
conventions):

    rotate_point_R(R,x) = transpose( matmult(R, transpose(x))) =
                        = matmult(x, transpose(R))

By default this function returns the rotated points only. If we also want
gradients, pass get_gradients=True. Logic:

    if not get_gradients: return u=R(x)
    else:                 return (u=R(x),du/dR,du/dx)

This function supports broadcasting fully, so we can rotate lots of points at
the same time and/or apply lots of different rotations at the same time

ARGUMENTS

- R: array of shape (3,3). This matrix defines the rotation. It is assumed that
  this is a valid rotation (matmult(R,transpose(R)) = I, det(R) = 1), but that
  is not checked

- x: array of shape (3,). The point being rotated

- get_gradients: optional boolean. By default (get_gradients=False) we return an
  array of rotated points. Otherwise we return a tuple of arrays of rotated
  points and their gradients.

RETURNED VALUE

If not get_gradients: we return an array of rotated point(s). Each broadcasted
slice has shape (3,)

If get_gradients: we return a tuple of arrays containing the rotated points and
the gradients (u=R(x),du/dR,du/dx):

1. The rotated point(s). Each broadcasted slice has shape (3,)

2. The gradient du/dR. Each broadcasted slice has shape (3,3,3). The first
   dimension selects the element of u, and the last 2 dimensions select the
   element of R

3. The gradient du/dx. Each broadcasted slice has shape (3,3). The first
   dimension selects the element of u, and the last dimension selects the
   element of x

    """

    # Should do this nicer in the C code. But for the time being, this will do
    R = np.ascontiguousarray(R)
    x = np.ascontiguousarray(x)

    if not get_gradients:
        return _poseutils._rotate_point_R(R,x)
    return _poseutils._rotate_point_R_withgrad(R,x)

def transform_point_rt(rt, x, get_gradients=False):
    r"""Transform point(s) using an rt transformation

SYNOPSIS

    r  = rotation_axis * rotation_magnitude
    rt = nps.glue(r,t, axis=-1)

    print(rt.shape)
    ===>
    (6,)

    print(x.shape)
    ===>
    (10,3)

    print( mrcal.transform_point_rt(rt, x).shape )
    ===>
    (10,3)

    print( [arr.shape
            for arr in mrcal.transform_point_rt(rt, x,
                                                get_gradients = True)] )
    ===>
    [(10,3), (10,3,3), (10,3,3), (10,3,3)]

Transform point(s) by an rt transformation: a (6,) array formed by
nps.glue(r,t, axis=-1) where r is a (3,) Rodrigues vector and t is a (3,)
translation vector. This transformation is defined by a matrix multiplication
and an addition. x and t are stored as a row vector (that's how numpy stores
1-dimensional arrays), but the multiplication works as if x was a column vector
(to match linear algebra conventions):

    transform_point_rt(rt, x) = transpose( matmult(R_from_r(rt[:3]), transpose(x)) +
                                           transpose(rt[3,:]) ) =
                              = matmult(x, transpose(R_from_r(rt[:3]))) +
                                rt[3:]

By default this function returns the transformed points only. If we also want
gradients, pass get_gradients=True. Logic:

    if not get_gradients: return u=rt(x)
    else:                 return (u=rt(x),du/dr,du/dt,du/dx)

This function supports broadcasting fully, so we can transform lots of points at
the same time and/or apply lots of different transformations at the same time

ARGUMENTS

- rt: array of shape (6,). This vector defines the transformation. rt[:3] is a
  rotation defined as a Rodrigues vector; rt[3:] is a translation.

- x: array of shape (3,). The point being transformed

- get_gradients: optional boolean. By default (get_gradients=False) we return an
  array of transformed points. Otherwise we return a tuple of arrays of
  transformed points and their gradients.

RETURNED VALUE

If not get_gradients: we return an array of transformed point(s). Each
broadcasted slice has shape (3,)

If get_gradients: we return a tuple of arrays of transformed points and the
gradients (u=rt(x),du/dr,du/dt,du/dx):

1. The transformed point(s). Each broadcasted slice has shape (3,)

2. The gradient du/dr. Each broadcasted slice has shape (3,3). The first
   dimension selects the element of u, and the last dimension selects the
   element of r

3. The gradient du/dt. Each broadcasted slice has shape (3,3). The first
   dimension selects the element of u, and the last dimension selects the
   element of t

4. The gradient du/dx. Each broadcasted slice has shape (3,3). The first
   dimension selects the element of u, and the last dimension selects the
   element of x

    """

    # Should do this nicer in the C code. But for the time being, this will do
    rt = np.ascontiguousarray(rt)
    x  = np.ascontiguousarray(x)

    if not get_gradients:
        return _poseutils._transform_point_rt(rt,x)
    return _poseutils._transform_point_rt_withgrad(rt,x)

def transform_point_Rt(Rt, x, get_gradients=False):
    r"""Transform point(s) using an Rt transformation

SYNOPSIS

    Rt = nps.glue(rotation_matrix,translation, axis=-2)

    print(Rt.shape)
    ===>
    (4,3)

    print(x.shape)
    ===>
    (10,3)

    print( mrcal.transform_point_Rt(Rt, x).shape )
    ===>
    (10,3)

    print( [arr.shape
            for arr in mrcal.transform_point_Rt(Rt, x,
                                                get_gradients = True)] )
    ===>
    [(10,3), (10,3,3,3), (10,3,3), (10,3,3)]

Transform point(s) by an Rt transformation: a (4,3) array formed by
nps.glue(R,t, axis=-2) where R is a (3,3) rotation matrix and t is a (3,)
translation vector. This transformation is defined by a matrix multiplication
and an addition. x and t are stored as a row vector (that's how numpy stores
1-dimensional arrays), but the multiplication works as if x was a column vector
(to match linear algebra conventions):

    transform_point_Rt(Rt, x) = transpose( matmult(Rt[:3,:], transpose(x)) +
                                           transpose(Rt[3,:]) ) =
                              = matmult(x, transpose(Rt[:3,:])) +
                                Rt[3,:]

By default this function returns the transformed points only. If we also want
gradients, pass get_gradients=True. Logic:

    if not get_gradients: return u=Rt(x)
    else:                 return (u=Rt(x),du/dR,du/dt,du/dx)

This function supports broadcasting fully, so we can transform lots of points at
the same time and/or apply lots of different transformations at the same time

ARGUMENTS

- Rt: array of shape (4,3). This matrix defines the transformation. Rt[:3,:] is
    a rotation matrix; Rt[3,:] is a translation. It is assumed that the rotation
    matrix is a valid rotation (matmult(R,transpose(R)) = I, det(R) = 1), but that
    is not checked

- x: array of shape (3,). The point being transformed

- get_gradients: optional boolean. By default (get_gradients=False) we return an
  array of transformed points. Otherwise we return a tuple of arrays of
  transformed points and their gradients.

RETURNED VALUE

If not get_gradients: we return an array of transformed point(s). Each
broadcasted slice has shape (3,)

If get_gradients: we return a tuple of arrays of transformed points and the
gradients (u=Rt(x),du/dR,du/dt,du/dx):

1. The transformed point(s). Each broadcasted slice has shape (3,)

2. The gradient du/dR. Each broadcasted slice has shape (3,3,3). The first
   dimension selects the element of u, and the last 2 dimensions select the
   element of R

3. The gradient du/dt. Each broadcasted slice has shape (3,3). The first
   dimension selects the element of u, and the last dimension selects the
   element of t

4. The gradient du/dx. Each broadcasted slice has shape (3,3). The first
   dimension selects the element of u, and the last dimension selects the
   element of x

    """

    # Should do this nicer in the C code. But for the time being, this will do
    Rt = np.ascontiguousarray(Rt)
    x  = np.ascontiguousarray(x)

    if not get_gradients:
        return _poseutils._transform_point_Rt(Rt,x)
    return _poseutils._transform_point_Rt_withgrad(Rt,x)

@nps.broadcast_define( ((4,),),
                       (3,3) )
def R_from_quat(q):
    r"""Convert a rotation defined as a unit quaternion rotation to a rotation matrix

SYNOPSIS

    s    = np.sin(rotation_magnitude/2.)
    c    = np.cos(rotation_magnitude/2.)
    quat = nps.glue( c, s*rotation_axis, axis = -1)

    print(quat.shape)
    ===>
    (4,)

    R = mrcal.R_from_quat(quat)

    print(R.shape)
    ===>
    (3,3)

This is mostly for compatibility with some old stuff. mrcal doesn't use
quaternions anywhere. Test this thoroughly before using.

    """

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
    r"""Convert a rotation defined as a rotation matrix to a unit quaternion

SYNOPSIS

    print(R.shape)
    ===>
    (3,3)

    quat = mrcal.quat_from_R(R)

    print(quat.shape)
    ===>
    (4,)

    c = quat[0]
    s = nps.mag(quat[1:])

    rotation_magnitude = 2. * np.arctan2(s,c)

    rotation_axis = quat[1:] / s

This is mostly for compatibility with some old stuff. mrcal doesn't use
quaternions anywhere. Test this thoroughly before using.

The implementation comes directly from the scipy project, the from_dcm()
function in

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

    """

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
