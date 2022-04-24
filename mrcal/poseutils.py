#!/usr/bin/python3

'''Routines to manipulate poses, transformations and points

Most of these are Python wrappers around the written-in-C Python extension
module mrcal._poseutils_npsp. Most of the time you want to use this module
instead of touching mrcal._poseutils_npsp directly.

All functions are exported into the mrcal module. So you can call these via
mrcal.poseutils.fff() or mrcal.fff(). The latter is preferred.

'''

import numpy as np
import numpysane as nps

# for python3
from functools import reduce

from . import _poseutils_npsp
from . import _poseutils_scipy

def r_from_R(R, get_gradients=False, out=None):
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

- out: optional argument specifying the destination. By default, new numpy
  array(s) are created and returned. To write the results into existing (and
  possibly non-contiguous) arrays, specify them with the 'out' kwarg. If not
  get_gradients: 'out' is the one numpy array we will write into. Else: 'out' is
  a tuple of all the output numpy arrays. If 'out' is given, we return the 'out'
  that was passed in. This is the standard behavior provided by
  numpysane_pywrap.

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
        return _poseutils_npsp._r_from_R_withgrad(R, out=out)
    return _poseutils_npsp._r_from_R(R, out=out)

def R_from_r(r, get_gradients=False, out=None):
    r"""Compute a rotation matrix from a Rodrigues vector

SYNOPSIS

    r = rotation_axis * rotation_magnitude
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

- out: optional argument specifying the destination. By default, new numpy
  array(s) are created and returned. To write the results into existing (and
  possibly non-contiguous) arrays, specify them with the 'out' kwarg. If not
  get_gradients: 'out' is the one numpy array we will write into. Else: 'out' is
  a tuple of all the output numpy arrays. If 'out' is given, we return the 'out'
  that was passed in. This is the standard behavior provided by
  numpysane_pywrap.

RETURNED VALUE

If not get_gradients: we return an array of rotation matrices. Each broadcasted
slice has shape (3,3)

If get_gradients: we return a tuple of arrays containing the rotation matrices
and the gradient (R, dR/dr):

1. The rotation matrix. Each broadcasted slice has shape (3,3). This is a valid
   rotation: matmult(R,transpose(R)) = I, det(R) = 1

2. The gradient dR/dr. Each broadcasted slice has shape (3,3,3). The first two
   dimensions select the element of R, and the last dimension selects the
   element of r

    """

    if get_gradients:
        return _poseutils_npsp._R_from_r_withgrad(r, out=out)
    return _poseutils_npsp._R_from_r(r, out=out)

def invert_R(R, out=None):
    r"""Invert a rotation matrix

SYNOPSIS

    print(R.shape)
    ===>
    (3,3)

    R10 = mrcal.invert_R(R01)

    print(x1.shape)
    ===>
    (3,)

    x0 = mrcal.rotate_point_R(R01, x1)

    print( nps.norm2( x1 - \
                      mrcal.rotate_point_R(R10, x0) ))
    ===>
    0

Given a rotation specified as a (3,3) rotation matrix outputs another rotation
matrix that has the opposite effect. This is simply a matrix transpose.

This function supports broadcasting fully.

In-place operation is supported; the output array may be the same as the input
array to overwrite the input.

ARGUMENTS

- R: array of shape (3,3), a rotation matrix. It is assumed that this is a valid
  rotation (matmult(R,transpose(R)) = I, det(R) = 1), but that is not checked

- out: optional argument specifying the destination. By default, a new numpy
  array is created and returned. To write the results into an existing (and
  possibly non-contiguous) array, specify it with the 'out' kwarg. If 'out' is
  given, we return the 'out' that was passed in. This is the standard behavior
  provided by numpysane_pywrap.

RETURNED VALUE

The inverse rotation matrix in an array of shape (3,3).

    """
    return _poseutils_npsp._invert_R(R, out=out)

def rt_from_Rt(Rt, get_gradients=False, out=None):
    r"""Compute an rt transformation from a Rt transformation

SYNOPSIS

    Rt = nps.glue(rotation_matrix,translation, axis=-2)

    print(Rt.shape)
    ===>
    (4,3)

    rt = mrcal.rt_from_Rt(Rt)

    print(rt.shape)
    ===>
    (6,)

    translation        = rt[3:]
    rotation_magnitude = nps.mag(rt[:3])
    rotation_axis      = rt[:3] / rotation_magnitude

Converts an Rt transformation to an rt transformation. Both specify a rotation
and translation. An Rt transformation is a (4,3) array formed by nps.glue(R,t,
axis=-2) where R is a (3,3) rotation matrix and t is a (3,) translation vector.
An rt transformation is a (6,) array formed by nps.glue(r,t, axis=-1) where r is
a (3,) Rodrigues vector and t is a (3,) translation vector.

Applied to a point x the transformed result is rotate(x)+t. Given a matrix R,
the rotation is defined by a matrix multiplication. x and t are stored as a row
vector (that's how numpy stores 1-dimensional arrays), but the multiplication
works as if x was a column vector (to match linear algebra conventions). See the
docs for mrcal._transform_point_Rt() for more detail.

By default this function returns the rt transformations only. If we also want
gradients, pass get_gradients=True. Logic:

    if not get_gradients: return rt
    else:                 return (rt, dr/dR)

Note that the translation gradient isn't returned: it is always the identity

This function supports broadcasting fully.

ARGUMENTS

- Rt: array of shape (4,3). This matrix defines the transformation. Rt[:3,:] is
  a rotation matrix; Rt[3,:] is a translation. It is assumed that the rotation
  matrix is a valid rotation (matmult(R,transpose(R)) = I, det(R) = 1), but that
  is not checked

- get_gradients: optional boolean. By default (get_gradients=False) we return an
  array of rt transformations. Otherwise we return a tuple of arrays of rt
  transformations and their gradients.

- out: optional argument specifying the destination. By default, new numpy
  array(s) are created and returned. To write the results into existing (and
  possibly non-contiguous) arrays, specify them with the 'out' kwarg. If not
  get_gradients: 'out' is the one numpy array we will write into. Else: 'out' is
  a tuple of all the output numpy arrays. If 'out' is given, we return the 'out'
  that was passed in. This is the standard behavior provided by
  numpysane_pywrap.

RETURNED VALUE

If not get_gradients: we return the rt transformation. Each broadcasted slice
has shape (6,). rt[:3] is a rotation defined as a Rodrigues vector; rt[3:] is a
translation.

If get_gradients: we return a tuple of arrays containing the rt transformation
and the gradient (rt, dr/dR):

1. The rt transformation. Each broadcasted slice has shape (6,)

2. The gradient dr/dR. Each broadcasted slice has shape (3,3,3). The first
   dimension selects the element of r, and the last two dimension select the
   element of R

    """
    if get_gradients:
        return _poseutils_npsp._rt_from_Rt_withgrad(Rt, out=out)
    return _poseutils_npsp._rt_from_Rt(Rt, out=out)

def Rt_from_rt(rt, get_gradients=False, out=None):
    r"""Compute an Rt transformation from a rt transformation

SYNOPSIS

    r  = rotation_axis * rotation_magnitude
    rt = nps.glue(r,t, axis=-1)

    print(rt.shape)
    ===>
    (6,)

    Rt = mrcal.Rt_from_rt(rt)

    print(Rt.shape)
    ===>
    (4,3)

    translation     = Rt[3,:]
    rotation_matrix = Rt[:3,:]

Converts an rt transformation to an Rt transformation. Both specify a rotation
and translation. An Rt transformation is a (4,3) array formed by nps.glue(R,t,
axis=-2) where R is a (3,3) rotation matrix and t is a (3,) translation vector.
An rt transformation is a (6,) array formed by nps.glue(r,t, axis=-1) where r is
a (3,) Rodrigues vector and t is a (3,) translation vector.

Applied to a point x the transformed result is rotate(x)+t. Given a matrix R,
the rotation is defined by a matrix multiplication. x and t are stored as a row
vector (that's how numpy stores 1-dimensional arrays), but the multiplication
works as if x was a column vector (to match linear algebra conventions). See the
docs for mrcal._transform_point_Rt() for more detail.

By default this function returns the Rt transformations only. If we also want
gradients, pass get_gradients=True. Logic:

    if not get_gradients: return Rt
    else:                 return (Rt, dR/dr)

Note that the translation gradient isn't returned: it is always the identity

This function supports broadcasting fully.

ARGUMENTS

- rt: array of shape (6,). This vector defines the input transformation. rt[:3]
  is a rotation defined as a Rodrigues vector; rt[3:] is a translation.

- get_gradients: optional boolean. By default (get_gradients=False) we return an
  array of Rt transformations. Otherwise we return a tuple of arrays of Rt
  transformations and their gradients.

- out: optional argument specifying the destination. By default, new numpy
  array(s) are created and returned. To write the results into existing (and
  possibly non-contiguous) arrays, specify them with the 'out' kwarg. If not
  get_gradients: 'out' is the one numpy array we will write into. Else: 'out' is
  a tuple of all the output numpy arrays. If 'out' is given, we return the 'out'
  that was passed in. This is the standard behavior provided by
  numpysane_pywrap.

RETURNED VALUE

If not get_gradients: we return the Rt transformation. Each broadcasted slice
has shape (4,3). Rt[:3,:] is a rotation matrix; Rt[3,:] is a translation. The
matrix R is a valid rotation: matmult(R,transpose(R)) = I and det(R) = 1

If get_gradients: we return a tuple of arrays containing the Rt transformation
and the gradient (Rt, dR/dr):

1. The Rt transformation. Each broadcasted slice has shape (4,3,)

2. The gradient dR/dr. Each broadcasted slice has shape (3,3,3). The first two
   dimensions select the element of R, and the last dimension selects the
   element of r

    """
    if get_gradients:
        return _poseutils_npsp._Rt_from_rt_withgrad(rt, out=out)
    return _poseutils_npsp._Rt_from_rt(rt, out=out)

def invert_Rt(Rt, out=None):
    r"""Invert an Rt transformation

SYNOPSIS

    Rt01 = nps.glue(rotation_matrix,translation, axis=-2)

    print(Rt01.shape)
    ===>
    (4,3)

    Rt10 = mrcal.invert_Rt(Rt01)

    print(x1.shape)
    ===>
    (3,)

    x0 = mrcal.transform_point_Rt(Rt01, x1)

    print( nps.norm2( x1 - \
                      mrcal.transform_point_Rt(Rt10, x0) ))
    ===>
    0

Given an Rt transformation to convert a point representated in coordinate system
1 to coordinate system 0 (let's call it Rt01), returns a transformation that
does the reverse: converts a representation in coordinate system 0 to coordinate
system 1 (let's call it Rt10).

Thus if you have a point in coordinate system 1 (let's call it x1), we can
convert it to a representation in system 0, and then back. And we'll get the
same thing out:

  x1 == mrcal.transform_point_Rt( mrcal.invert_Rt(Rt01),
          mrcal.transform_point_Rt( Rt01, x1 ))

An Rt transformation represents a rotation and a translation. It is a (4,3)
array formed by nps.glue(R,t, axis=-2) where R is a (3,3) rotation matrix and t
is a (3,) translation vector.

Applied to a point x the transformed result is rotate(x)+t. Given a matrix R,
the rotation is defined by a matrix multiplication. x and t are stored as a row
vector (that's how numpy stores 1-dimensional arrays), but the multiplication
works as if x was a column vector (to match linear algebra conventions). See the
docs for mrcal._transform_point_Rt() for more detail.

This function supports broadcasting fully.

In-place operation is supported; the output array may be the same as the input
array to overwrite the input.

ARGUMENTS

- Rt: array of shape (4,3). This matrix defines the transformation. Rt[:3,:] is
  a rotation matrix; Rt[3,:] is a translation. It is assumed that the rotation
  matrix is a valid rotation (matmult(R,transpose(R)) = I, det(R) = 1), but that
  is not checked

- out: optional argument specifying the destination. By default, a new numpy
  array is created and returned. To write the results into an existing (and
  possibly non-contiguous) array, specify it with the 'out' kwarg. If 'out' is
  given, we return the 'out' that was passed in. This is the standard behavior
  provided by numpysane_pywrap.

RETURNED VALUE

The inverse Rt transformation in an array of shape (4,3).

    """
    return _poseutils_npsp._invert_Rt(Rt, out=out)

def invert_rt(rt, get_gradients=False, out=None):
    r"""Invert an rt transformation

SYNOPSIS

    r    = rotation_axis * rotation_magnitude
    rt01 = nps.glue(r,t, axis=-1)

    print(rt01.shape)
    ===>
    (6,)

    rt10 = mrcal.invert_rt(rt01)

    print(x1.shape)
    ===>
    (3,)

    x0 = mrcal.transform_point_rt(rt01, x1)

    print( nps.norm2( x1 -
                      mrcal.transform_point_rt(rt10, x0) ))
    ===>
    0

Given an rt transformation to convert a point representated in coordinate system
1 to coordinate system 0 (let's call it rt01), returns a transformation that
does the reverse: converts a representation in coordinate system 0 to coordinate
system 1 (let's call it rt10).

Thus if you have a point in coordinate system 1 (let's call it x1), we can
convert it to a representation in system 0, and then back. And we'll get the
same thing out:

  x1 == mrcal.transform_point_rt( mrcal.invert_rt(rt01),
          mrcal.transform_point_rt( rt01, x1 ))

An rt transformation represents a rotation and a translation. It is a (6,) array
formed by nps.glue(r,t, axis=-1) where r is a (3,) Rodrigues vector and t is a
(3,) translation vector.

Applied to a point x the transformed result is rotate(x)+t. x and t are stored
as a row vector (that's how numpy stores 1-dimensional arrays). See the docs for
mrcal._transform_point_rt() for more detail.

By default this function returns the rt transformation only. If we also want
gradients, pass get_gradients=True. Logic:

    if not get_gradients: return rt
    else:                 return (rt, drtout_drtin)

Note that the poseutils C API returns only

- dtout_drin
- dtout_dtin

because

- drout_drin is always -I
- drout_dtin is always 0

This Python function, however fills in those constants to return the full (and
more convenient) arrays.

This function supports broadcasting fully.

In-place operation is supported; the output array may be the same as the input
array to overwrite the input.

ARGUMENTS

- rt: array of shape (6,). This vector defines the input transformation. rt[:3]
  is a rotation defined as a Rodrigues vector; rt[3:] is a translation.

- get_gradients: optional boolean. By default (get_gradients=False) we return an
  array of rt translation. Otherwise we return a tuple of arrays of rt
  translations and their gradients.

- out: optional argument specifying the destination. By default, new numpy
  array(s) are created and returned. To write the results into existing (and
  possibly non-contiguous) arrays, specify them with the 'out' kwarg. If not
  get_gradients: 'out' is the one numpy array we will write into. Else: 'out' is
  a tuple of all the output numpy arrays. If 'out' is given, we return the 'out'
  that was passed in. This is the standard behavior provided by
  numpysane_pywrap.

RETURNED VALUE

If not get_gradients: we return an array of rt transformation(s). Each
broadcasted slice has shape (6,)

If get_gradients: we return a tuple of arrays containing the rt transformation(s)
and the gradients (rt, drtout/drtin)

1. The rt transformation. Each broadcasted slice has shape (6,)

2. The gradient drtout/drtin. Each broadcasted slice has shape (6,6). The first
   dimension selects elements of rtout, and the last dimension selects elements
   of rtin

    """
    if get_gradients:
        return _poseutils_npsp._invert_rt_withgrad(rt, out=out)
    return _poseutils_npsp._invert_rt(rt, out=out)

def compose_Rt(*Rt, out=None):
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

    print( nps.norm2( mrcal.transform_point_Rt(Rt30, x0) -
                      mrcal.transform_point_Rt(Rt32,
                        mrcal.transform_point_Rt(Rt21,
                          mrcal.transform_point_Rt(Rt10, x0)))))
    ===>
    0

Given 2 or more Rt transformations, returns their composition. An Rt
transformation is a (4,3) array formed by nps.glue(R,t, axis=-2) where R is a
(3,3) rotation matrix and t is a (3,) translation vector. This transformation is
defined by a matrix multiplication and an addition. x and t are stored as a row
vector (that's how numpy stores 1-dimensional arrays), but the multiplication
works as if x was a column vector (to match linear algebra conventions):

    transform_point_Rt(Rt, x) = transpose( matmult(Rt[:3,:], transpose(x)) +
                                           transpose(Rt[3,:]) ) =
                              = matmult(x, transpose(Rt[:3,:])) +
                                Rt[3,:]

This function supports broadcasting fully, so we can compose lots of
transformations at the same time.

In-place operation is supported; the output array may be the same as either of
the input arrays to overwrite the input.

ARGUMENTS

- *Rt: a list of transformations to compose. Usually we'll be composing two
  transformations, but any number could be given here. Each broadcasted slice
  has shape (4,3).

- out: optional argument specifying the destination. By default, a new numpy
  array is created and returned. To write the results into an existing (and
  possibly non-contiguous) array, specify it with the 'out' kwarg. If 'out' is
  given, we return the 'out' that was passed in. This is the standard behavior
  provided by numpysane_pywrap.

RETURNED VALUE

An array of composed Rt transformations. Each broadcasted slice has shape (4,3)

    """
    Rt1onwards = reduce( _poseutils_npsp._compose_Rt, Rt[1:], _poseutils_npsp.identity_Rt() )
    return _poseutils_npsp._compose_Rt(Rt[0], Rt1onwards, out=out)

def compose_r(*r, get_gradients=False, out=None):
    r"""Compose angle-axis rotations

SYNOPSIS

    r10 = rotation_axis10 * rotation_magnitude10
    r21 = rotation_axis21 * rotation_magnitude21
    r32 = rotation_axis32 * rotation_magnitude32

    print(r10.shape)
    ===>
    (3,)

    r30 = mrcal.compose_r( r32, r21, r10 )

    print(x0.shape)
    ===>
    (3,)

    print( nps.norm2( mrcal.rotate_point_r(r30, x0) -
                      mrcal.rotate_point_r(r32,
                        mrcal.rotate_point_r(r21,
                          mrcal.rotate_point_r(r10, x0)))))
    ===>
    0

    print( [arr.shape for arr in mrcal.compose_r(r21,r10,
                                                 get_gradients = True)] )
    ===>
    [(3,), (3,3), (3,3)]

Given 2 or more axis-angle rotations, returns their composition. By default this
function returns the composed rotation only. If we also want gradients, pass
get_gradients=True. This is supported ONLY if we have EXACTLY 2 rotations to
compose. Logic:

    if not get_gradients: return r=compose(r0,r1)
    else:                 return (r=compose(r0,r1), dr/dr0, dr/dr1)

This function supports broadcasting fully, so we can compose lots of
rotations at the same time.

In-place operation is supported; the output array may be the same as either of
the input arrays to overwrite the input.

ARGUMENTS

- *r: a list of rotations to compose. Usually we'll be composing two rotations,
  but any number could be given here. Each broadcasted slice has shape (3,)

- get_gradients: optional boolean. By default (get_gradients=False) we return an
  array of composed rotations. Otherwise we return a tuple of arrays of composed
  rotations and their gradients. Gradient reporting is only supported when
  exactly two rotations are given

- out: optional argument specifying the destination. By default, new numpy
  array(s) are created and returned. To write the results into existing (and
  possibly non-contiguous) arrays, specify them with the 'out' kwarg. If not
  get_gradients: 'out' is the one numpy array we will write into. Else: 'out' is
  a tuple of all the output numpy arrays. If 'out' is given, we return the 'out'
  that was passed in. This is the standard behavior provided by
  numpysane_pywrap.

RETURNED VALUE

If not get_gradients: we return an array of composed rotations. Each broadcasted
slice has shape (3,)

If get_gradients: we return a tuple of arrays containing the composed rotations
and the gradients (r=compose(r0,r1), dr/dr0, dr/dr1):

1. The composed rotation. Each broadcasted slice has shape (3,)

2. The gradient dr/dr0. Each broadcasted slice has shape (3,3). The first
   dimension selects the element of r, and the last dimension selects the
   element of r0

3. The gradient dr/dr1. Each broadcasted slice has shape (3,3). The first
   dimension selects the element of r, and the last dimension selects the
   element of r1

    """

    if get_gradients:
        if len(r) != 2:
            raise Exception("compose_r(..., get_gradients=True) is supported only if exactly 2 inputs are given")
        return _poseutils_npsp._compose_r_withgrad(*r, out=out)

    r1onwards = reduce( _poseutils_npsp._compose_r, r[1:], _poseutils_npsp.identity_r() )
    return _poseutils_npsp._compose_r(r[0], r1onwards, out=out)

def compose_rt(*rt, get_gradients=False, out=None):
    r"""Compose rt transformations

SYNOPSIS

    r10 = rotation_axis10 * rotation_magnitude10
    r21 = rotation_axis21 * rotation_magnitude21
    r32 = rotation_axis32 * rotation_magnitude32

    rt10 = nps.glue(r10,t10, axis=-1)
    rt21 = nps.glue(r21,t21, axis=-1)
    rt32 = nps.glue(r32,t32, axis=-1)

    print(rt10.shape)
    ===>
    (6,)

    rt30 = mrcal.compose_rt( rt32, rt21, rt10 )

    print(x0.shape)
    ===>
    (3,)

    print( nps.norm2( mrcal.transform_point_rt(rt30, x0) -
                      mrcal.transform_point_rt(rt32,
                        mrcal.transform_point_rt(rt21,
                          mrcal.transform_point_rt(rt10, x0)))))
    ===>
    0

    print( [arr.shape for arr in mrcal.compose_rt(rt21,rt10,
                                                  get_gradients = True)] )
    ===>
    [(6,), (6,6), (6,6)]

Given 2 or more rt transformations, returns their composition. An rt
transformation is a (6,) array formed by nps.glue(r,t, axis=-1) where r is a
(3,) Rodrigues vector and t is a (3,) translation vector. This transformation is
defined by a matrix multiplication and an addition. x and t are stored as a row
vector (that's how numpy stores 1-dimensional arrays), but the multiplication
works as if x was a column vector (to match linear algebra conventions):

    transform_point_rt(rt, x) = transpose( matmult(R_from_r(rt[:3]), transpose(x)) +
                                           transpose(rt[3,:]) ) =
                              = matmult(x, transpose(R_from_r(rt[:3]))) +
                                rt[3:]

By default this function returns the composed transformation only. If we also
want gradients, pass get_gradients=True. This is supported ONLY if we have
EXACTLY 2 transformations to compose. Logic:

    if not get_gradients: return rt=compose(rt0,rt1)
    else:                 return (rt=compose(rt0,rt1), dr/drt0, dr/drt1)

Note that the poseutils C API returns only

- dr_dr0
- dr_dr1
- dt_dr0
- dt_dt1

because

- dr/dt0 is always 0
- dr/dt1 is always 0
- dt/dr1 is always 0
- dt/dt0 is always the identity matrix

This Python function, however fills in those constants to return the full (and
more convenient) arrays.

This function supports broadcasting fully, so we can compose lots of
transformations at the same time.

In-place operation is supported; the output array may be the same as either of
the input arrays to overwrite the input.

ARGUMENTS

- *rt: a list of transformations to compose. Usually we'll be composing two
  transformations, but any number could be given here. Each broadcasted slice
  has shape (6,)

- get_gradients: optional boolean. By default (get_gradients=False) we return an
  array of composed transformations. Otherwise we return a tuple of arrays of
  composed transformations and their gradients. Gradient reporting is only
  supported when exactly two transformations are given

- out: optional argument specifying the destination. By default, new numpy
  array(s) are created and returned. To write the results into existing (and
  possibly non-contiguous) arrays, specify them with the 'out' kwarg. If not
  get_gradients: 'out' is the one numpy array we will write into. Else: 'out' is
  a tuple of all the output numpy arrays. If 'out' is given, we return the 'out'
  that was passed in. This is the standard behavior provided by
  numpysane_pywrap.

RETURNED VALUE

If not get_gradients: we return an array of composed rt transformations. Each
broadcasted slice has shape (6,)

If get_gradients: we return a tuple of arrays containing the composed
transformations and the gradients (rt=compose(rt0,rt1),
drt/drt0,drt/drt1):

1. The composed transformation. Each broadcasted slice has shape (6,)

2. The gradient drt/dr0. Each broadcasted slice has shape (6,6). The first
   dimension selects the element of rt, and the last dimension selects the
   element of rt0

3. The gradient drt/drt1. Each broadcasted slice has shape (6,6). The first
   dimension selects the element of rt, and the last dimension selects the
   element of rt1

    """

    if get_gradients:
        if len(rt) != 2:
            raise Exception("compose_rt(..., get_gradients=True) is supported only if exactly 2 inputs are given")
        return _poseutils_npsp._compose_rt_withgrad(*rt, out=out)

    # I convert them all to Rt and compose for efficiency. Otherwise each
    # internal composition will convert to Rt, compose, and then convert back to
    # rt. The way I'm doing it I convert to rt just once, at the end. This will
    # save operations if I'm composing more than 2 transformations
    Rt = compose_Rt(*[_poseutils_npsp._Rt_from_rt(_rt) for _rt in rt])
    return _poseutils_npsp._rt_from_Rt( Rt, out=out)

def rotate_point_r(r, x, get_gradients=False, out=None, inverted=False):
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

In-place operation is supported; the output array may be the same as the input
arrays to overwrite the input.

ARGUMENTS

- r: array of shape (3,). The Rodrigues vector that defines the rotation. This is
  a unit rotation axis scaled by the rotation magnitude, in radians

- x: array of shape (3,). The point being rotated

- get_gradients: optional boolean. By default (get_gradients=False) we return an
  array of rotated points. Otherwise we return a tuple of arrays of rotated
  points and their gradients.

- inverted: optional boolean, defaulting to False. If True, the opposite
  rotation is computed. The gradient du/dr is returned in respect to the input r

- out: optional argument specifying the destination. By default, new numpy
  array(s) are created and returned. To write the results into existing (and
  possibly non-contiguous) arrays, specify them with the 'out' kwarg. If not
  get_gradients: 'out' is the one numpy array we will write into. Else: 'out' is
  a tuple of all the output numpy arrays. If 'out' is given, we return the 'out'
  that was passed in. This is the standard behavior provided by
  numpysane_pywrap.

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
        return _poseutils_npsp._rotate_point_r(r,x, out=out, inverted=inverted)
    return _poseutils_npsp._rotate_point_r_withgrad(r,x, out=out, inverted=inverted)

def rotate_point_R(R, x, get_gradients=False, out=None, inverted=False):
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

In-place operation is supported; the output array may be the same as the input
arrays to overwrite the input.

ARGUMENTS

- R: array of shape (3,3). This matrix defines the rotation. It is assumed that
  this is a valid rotation (matmult(R,transpose(R)) = I, det(R) = 1), but that
  is not checked

- x: array of shape (3,). The point being rotated

- get_gradients: optional boolean. By default (get_gradients=False) we return an
  array of rotated points. Otherwise we return a tuple of arrays of rotated
  points and their gradients.

- inverted: optional boolean, defaulting to False. If True, the opposite
  rotation is computed. The gradient du/dR is returned in respect to the input R

- out: optional argument specifying the destination. By default, new numpy
  array(s) are created and returned. To write the results into existing (and
  possibly non-contiguous) arrays, specify them with the 'out' kwarg. If not
  get_gradients: 'out' is the one numpy array we will write into. Else: 'out' is
  a tuple of all the output numpy arrays. If 'out' is given, we return the 'out'
  that was passed in. This is the standard behavior provided by
  numpysane_pywrap.

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

    if not get_gradients:
        return _poseutils_npsp._rotate_point_R(R,x, out=out, inverted=inverted)
    return _poseutils_npsp._rotate_point_R_withgrad(R,x, out=out, inverted=inverted)

def transform_point_rt(rt, x, get_gradients=False, out=None, inverted=False):
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
    [(10,3), (10,3,6), (10,3,3)]

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
    else:                 return (u=rt(x),du/drt,du/dx)

This function supports broadcasting fully, so we can transform lots of points at
the same time and/or apply lots of different transformations at the same time

In-place operation is supported; the output array may be the same as the input
arrays to overwrite the input.

ARGUMENTS

- rt: array of shape (6,). This vector defines the transformation. rt[:3] is a
  rotation defined as a Rodrigues vector; rt[3:] is a translation.

- x: array of shape (3,). The point being transformed

- get_gradients: optional boolean. By default (get_gradients=False) we return an
  array of transformed points. Otherwise we return a tuple of arrays of
  transformed points and their gradients.

- inverted: optional boolean, defaulting to False. If True, the opposite
  transformation is computed. The gradient du/drt is returned in respect to the
  input rt

- out: optional argument specifying the destination. By default, new numpy
  array(s) are created and returned. To write the results into existing (and
  possibly non-contiguous) arrays, specify them with the 'out' kwarg. If not
  get_gradients: 'out' is the one numpy array we will write into. Else: 'out' is
  a tuple of all the output numpy arrays. If 'out' is given, we return the 'out'
  that was passed in. This is the standard behavior provided by
  numpysane_pywrap.

RETURNED VALUE

If not get_gradients: we return an array of transformed point(s). Each
broadcasted slice has shape (3,)

If get_gradients: we return a tuple of arrays of transformed points and the
gradients (u=rt(x),du/drt,du/dx):

1. The transformed point(s). Each broadcasted slice has shape (3,)

2. The gradient du/drt. Each broadcasted slice has shape (3,6). The first
   dimension selects the element of u, and the last dimension selects the
   element of rt

3. The gradient du/dx. Each broadcasted slice has shape (3,3). The first
   dimension selects the element of u, and the last dimension selects the
   element of x

    """

    if not get_gradients:
        return _poseutils_npsp._transform_point_rt(rt,x, out=out, inverted=inverted)
    return _poseutils_npsp._transform_point_rt_withgrad(rt,x, out=out, inverted=inverted)

def transform_point_Rt(Rt, x, get_gradients=False, out=None, inverted=False):
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
    [(10,3), (10,3,4,3), (10,3,3)]

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
    else:                 return (u=Rt(x),du/dRt,du/dx)

This function supports broadcasting fully, so we can transform lots of points at
the same time and/or apply lots of different transformations at the same time

In-place operation is supported; the output array may be the same as the input
arrays to overwrite the input.

ARGUMENTS

- Rt: array of shape (4,3). This matrix defines the transformation. Rt[:3,:] is
    a rotation matrix; Rt[3,:] is a translation. It is assumed that the rotation
    matrix is a valid rotation (matmult(R,transpose(R)) = I, det(R) = 1), but that
    is not checked

- x: array of shape (3,). The point being transformed

- get_gradients: optional boolean. By default (get_gradients=False) we return an
  array of transformed points. Otherwise we return a tuple of arrays of
  transformed points and their gradients.

- inverted: optional boolean, defaulting to False. If True, the opposite
  transformation is computed. The gradient du/dRt is returned in respect to the
  input Rt

- out: optional argument specifying the destination. By default, new numpy
  array(s) are created and returned. To write the results into existing (and
  possibly non-contiguous) arrays, specify them with the 'out' kwarg. If not
  get_gradients: 'out' is the one numpy array we will write into. Else: 'out' is
  a tuple of all the output numpy arrays. If 'out' is given, we return the 'out'
  that was passed in. This is the standard behavior provided by
  numpysane_pywrap.

RETURNED VALUE

If not get_gradients: we return an array of transformed point(s). Each
broadcasted slice has shape (3,)

If get_gradients: we return a tuple of arrays of transformed points and the
gradients (u=Rt(x),du/dRt,du/dx):

1. The transformed point(s). Each broadcasted slice has shape (3,)

2. The gradient du/dRt. Each broadcasted slice has shape (3,4,3). The first
   dimension selects the element of u, and the last 2 dimensions select the
   element of Rt

3. The gradient du/dx. Each broadcasted slice has shape (3,3). The first
   dimension selects the element of u, and the last dimension selects the
   element of x

    """

    if not get_gradients:
        return _poseutils_npsp._transform_point_Rt(Rt,x, out=out, inverted=inverted)
    return _poseutils_npsp._transform_point_Rt_withgrad(Rt,x, out=out, inverted=inverted)


from . import _poseutils_scipy
quat_from_R = _poseutils_scipy.quat_from_R

def qt_from_Rt(Rt, out=None):
    r"""Compute a qt transformation from a Rt transformation

SYNOPSIS

    Rt = nps.glue(rotation_matrix,translation, axis=-2)

    print(Rt.shape)
    ===>
    (4,3)

    qt = mrcal.qt_from_Rt(Rt)

    print(qt.shape)
    ===>
    (7,)

    quat        = qt[:4]
    translation = qt[4:]

Converts an Rt transformation to a qt transformation. Both specify a rotation
and translation. An Rt transformation is a (4,3) array formed by nps.glue(R,t,
axis=-2) where R is a (3,3) rotation matrix and t is a (3,) translation vector.
A qt transformation is a (7,) array formed by nps.glue(q,t, axis=-1) where q is
a (4,) unit quaternion and t is a (3,) translation vector.

Applied to a point x the transformed result is rotate(x)+t. Given a matrix R,
the rotation is defined by a matrix multiplication. x and t are stored as a row
vector (that's how numpy stores 1-dimensional arrays), but the multiplication
works as if x was a column vector (to match linear algebra conventions). See the
docs for mrcal._transform_point_Rt() for more detail.

This function supports broadcasting fully.

Note: mrcal does not use unit quaternions anywhere to represent rotations. This
function is provided for convenience, but isn't thoroughly tested.

ARGUMENTS

- Rt: array of shape (4,3). This matrix defines the transformation. Rt[:3,:] is
  a rotation matrix; Rt[3,:] is a translation. It is assumed that the rotation
  matrix is a valid rotation (matmult(R,transpose(R)) = I, det(R) = 1), but that
  is not checked

- out: optional argument specifying the destination. By default, new numpy
  array(s) are created and returned. To write the results into existing (and
  possibly non-contiguous) arrays, specify them with the 'out' kwarg.

RETURNED VALUE

We return the qt transformation. Each broadcasted slice has shape (7,). qt[:4]
is a rotation defined as a unit quaternion; qt[4:] is a translation.

    """
    if out is not None:
        qt = out
    else:
        qt = np.zeros(Rt.shape[:-2] + (7,), dtype=float)

    _poseutils_scipy.quat_from_R(Rt[..., :3, :], out=qt[..., :4])
    qt[..., 4:] = Rt[..., 3, :]
    return qt


def Rt_from_qt(qt, out=None):
    r"""Compute an Rt transformation from a qt transformation

SYNOPSIS

    qt = nps.glue(q,t, axis=-1)

    print(qt.shape)
    ===>
    (7,)

    Rt = mrcal.Rt_from_qt(qt)

    print(Rt.shape)
    ===>
    (4,3)

    translation     = Rt[3,:]
    rotation_matrix = Rt[:3,:]

Converts a qt transformation to an Rt transformation. Both specify a rotation
and translation. An Rt transformation is a (4,3) array formed by nps.glue(R,t,
axis=-2) where R is a (3,3) rotation matrix and t is a (3,) translation vector.
A qt transformation is a (7,) array formed by nps.glue(q,t, axis=-1) where q is
a (4,) unit quaternion and t is a (3,) translation vector.

Applied to a point x the transformed result is rotate(x)+t. Given a matrix R,
the rotation is defined by a matrix multiplication. x and t are stored as a row
vector (that's how numpy stores 1-dimensional arrays), but the multiplication
works as if x was a column vector (to match linear algebra conventions). See the
docs for mrcal._transform_point_Rt() for more detail.

This function supports broadcasting fully.

Note: mrcal does not use unit quaternions anywhere to represent rotations. This
function is provided for convenience, but isn't thoroughly tested.

ARGUMENTS

- qt: array of shape (7,). This vector defines the input transformation. qt[:4]
  is a rotation defined as a unit quaternion; qt[4:] is a translation.

- out: optional argument specifying the destination. By default, new numpy
  array(s) are created and returned. To write the results into existing (and
  possibly non-contiguous) arrays, specify them with the 'out' kwarg.

RETURNED VALUE

We return the Rt transformation. Each broadcasted slice has shape (4,3).
Rt[:3,:] is a rotation matrix; Rt[3,:] is a translation. The matrix R is a valid
rotation: matmult(R,transpose(R)) = I and det(R) = 1

    """
    if out is not None:
        Rt = out
    else:
        Rt = np.zeros(qt.shape[:-1] + (4,3), dtype=float)

    _poseutils_npsp.R_from_quat(qt[..., :4], out=Rt[..., :3, :])
    Rt[..., 3, :] = qt[..., 4:]
    return Rt
