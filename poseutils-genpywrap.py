#!/usr/bin/python3

r'''Python-wrap the mrcal geometry routines

'''

import sys
import os

import numpy as np
import numpysane as nps

import numpysane_pywrap as npsp

m = npsp.module( name      = "_poseutils",
                 docstring = "geometry utils",
                 header    = r'''
#include "poseutils.h"
#include <string.h>
''')

m.function( "identity_R",
            """Return an identity rotation matrix

SYNOPSIS

    print( mrcal.identity_R() )
    ===>
    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]
""",

            args_input       = (),
            prototype_input  = (),
            prototype_output = (3,3),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
            *(double*)(data_slice__output + i*strides_slice__output[0] + j*strides_slice__output[1]) =
                 (i==j) ? 1.0 : 0.0;
    return true;
'''})

m.function( "identity_r",
            """Return an identity Rodrigues vector

SYNOPSIS

    print( mrcal.identity_r() )
    ===>
    [0. 0. 0.]
""",

            args_input       = (),
            prototype_input  = (),
            prototype_output = (3,),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    for(int i=0; i<3; i++)
        *(double*)(data_slice__output + i*strides_slice__output[0]) = 0.0;
    return true;
'''})

m.function( "identity_Rt",
            """Return an identity Rt transformation

SYNOPSIS

    print( mrcal.identity_Rt() )
    ===>
    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]
     [0. 0. 0.]]
""",

            args_input       = (),
            prototype_input  = (),
            prototype_output = (4,3),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
            *(double*)(data_slice__output + i*strides_slice__output[0] + j*strides_slice__output[1]) =
                 (i==j) ? 1.0 : 0.0;
    for(int j=0; j<3; j++)
        *(double*)(data_slice__output + 3*strides_slice__output[0] + j*strides_slice__output[1]) = 0.0;
    return true;
'''})

m.function( "identity_rt",
            """Return an identity rt transformation

SYNOPSIS

    print( mrcal.identity_rt() )
    ===>
    [0. 0. 0. 0. 0. 0.]
""",

            args_input       = (),
            prototype_input  = (),
            prototype_output = (6,),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    for(int i=0; i<6; i++)
        *(double*)(data_slice__output + i*strides_slice__output[0]) = 0.0;
    return true;
'''})

m.function( "_rotate_point_R",
            """Rotate a point using a rotation matrix

SYNOPSIS

    r = rotation_axis * rotation_magnitude
    R = mrcal.R_from_r(r)

    print(R.shape)
    ===>
    (3,3)

    print(x.shape)
    ===>
    (10,3)

    print( mrcal._rotate_point_R(R, x).shape )
    ===>
    (10,3)

This is an internal function. You probably want mrcal.rotate_point_R()

Rotate point(s) by a rotation matrix. This is a matrix multiplication. x is
stored as a row vector (that's how numpy stores 1-dimensional arrays), but the
multiplication works as if x was a column vector (to match linear algebra
conventions):

_rotate_point_R(R,x) = transpose( matmult(R, transpose(x))) =
                     = matmult(x, transpose(R))

This function supports broadcasting fully, so we can rotate lots of points at
the same time and/or apply lots of different rotations at the same time

ARGUMENTS

- R: array of shape (3,3). This matrix defines the rotation. It is assumed that
  this is a valid rotation (matmult(R,transpose(R)) = I, det(R) = 1), but that
  is not checked

- x: array of shape (3,). The point being rotated

RETURNED VALUE

The rotated point(s). Each broadcasted slice has shape (3,)

""",
            args_input       = ('R', 'x'),
            prototype_input  = ((3,3), (3,)),
            prototype_output = (3,),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_rotate_point_R( (double*)data_slice__output,
                          NULL,NULL,
                          (const double*)data_slice__R,
                          (const double*)data_slice__x );
    return true;
'''},
            Ccode_validate = r'''
            return \
              CHECK_CONTIGUOUS_AND_SETERROR_ALL();
'''
)

m.function( "_rotate_point_R_withgrad",
            """Rotate a point using a rotation matrix; report the result and gradients

SYNOPSIS

    r = rotation_axis * rotation_magnitude
    R = mrcal.R_from_r(r)

    print(R.shape)
    ===>
    (3,3)

    print(x.shape)
    ===>
    (10,3)

    print( [arr.shape for arr in mrcal._rotate_point_R_withgrad(R, x)] )
    ===>
    [(10,3), (10,3,3,3), (10,3,3)]

This is an internal function. You probably want mrcal.rotate_point_R()

Rotate point(s) by a rotation matrix. Unlike _rotate_point_R(), this returns a
tuple of the result and the gradients: (u=R(x),du/dR,du/dx).

This is a matrix multiplication. x is stored as a row vector (that's how numpy
stores 1-dimensional arrays), but the multiplication works as if x was a column
vector (to match linear algebra conventions):

_rotate_point_R(R,x) = transpose( matmult(R, transpose(x))) =
                     = matmult(x, transpose(R))

This function supports broadcasting fully, so we can rotate lots of points at
the same time and/or apply lots of different rotations at the same time

ARGUMENTS

- R: array of shape (3,3). This matrix defines the rotation. It is assumed that
  this is a valid rotation (matmult(R,transpose(R)) = I, det(R) = 1), but that
  is not checked

- x: array of shape (3,). The point being rotated

RETURNED VALUE

A tuple (u=R(x),du/dR,du/dx):

1. The rotated point(s). Each broadcasted slice has shape (3,)

2. The gradient du/dR. Each broadcasted slice has shape (3,3,3,). The first
   dimension selects the element of u, and the last 2 dimensions select the
   element of R

3. The gradient du/dx. Each broadcasted slice has shape (3,3). The first
   dimension selects the element of u, and the last dimension selects the
   element of x """,

            args_input       = ('R', 'x'),
            prototype_input  = ((3,3), (3,)),
            prototype_output = ((3,), (3,3,3), (3,3)),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_rotate_point_R( (double*)data_slice__output0,
                          (double*)data_slice__output1,
                          (double*)data_slice__output2,
                          (const double*)data_slice__R,
                          (const double*)data_slice__x );
    return true;
'''},
            Ccode_validate = r'''
            return \
              CHECK_CONTIGUOUS_AND_SETERROR_ALL();
'''
)

m.function( "_rotate_point_r",
            """Rotate a point using a Rodrigues vector

SYNOPSIS

    r = rotation_axis * rotation_magnitude

    print(r.shape)
    ===>
    (3,)

    print(x.shape)
    ===>
    (10,3)

    print(mrcal._rotate_point_r(r, x).shape)
    ===>
    (10,3)

This is an internal function. You probably want mrcal.rotate_point_r()

Rotate point(s) by a rotation matrix. The Rodrigues vector is converted to a
rotation matrix internally, and then this function is a matrix multiplication. x
is stored as a row vector (that's how numpy stores 1-dimensional arrays), but
the multiplication works as if x was a column vector (to match linear algebra
conventions):

_rotate_point_r(r,x) = transpose( matmult(R(r), transpose(x))) =
                     = matmult(x, transpose(R(r)))

This function supports broadcasting fully, so we can rotate lots of points at
the same time and/or apply lots of different rotations at the same time

ARGUMENTS

- r: array of shape (3,). The Rodriges vector that defines the rotation. This is
  a rotation axis scaled by the rotation magnitude, in radians

- x: array of shape (3,). The point being rotated

RETURNED VALUE

The rotated point(s). Each broadcasted slice has shape (3,)
""",
            args_input       = ('r', 'x'),
            prototype_input  = ((3,), (3,)),
            prototype_output = (3,),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_rotate_point_r( (double*)data_slice__output,
                          NULL,NULL,
                          (const double*)data_slice__r,
                          (const double*)data_slice__x );
    return true;
'''},
            Ccode_validate = r'''
            return \
              CHECK_CONTIGUOUS_AND_SETERROR_ALL();
'''
)

m.function( "_rotate_point_r_withgrad",
            """Rotate a point using a Rodrigues vector; report the result and gradients

SYNOPSIS

    r = rotation_axis * rotation_magnitude

    print(r.shape)
    ===>
    (3,3)

    print(x.shape)
    ===>
    (10,3)

    print( [arr.shape for arr in mrcal._rotate_point_r_withgrad(r, x)] )
    ===>
    [(10,3), (10,3,3), (10,3,3)]

This is an internal function. You probably want mrcal.rotate_point_r()

Rotate point(s) by a rotation matrix. Unlike _rotate_point_r(), this returns a
tuple of the result and the gradients: (u=r(x),du/dr,du/dx).

The Rodrigues vector is converted to a rotation matrix internally, and then this
function is a matrix multiplication. x is stored as a row vector (that's how
numpy stores 1-dimensional arrays), but the multiplication works as if x was a
column vector (to match linear algebra conventions):

_rotate_point_r(r,x) = transpose( matmult(R(r), transpose(x))) =
                     = matmult(x, transpose(R(r)))

This function supports broadcasting fully, so we can rotate lots of points at
the same time and/or apply lots of different rotations at the same time

ARGUMENTS

- r: array of shape (3,). The Rodriges vector that defines the rotation. This is
  a rotation axis scaled by the rotation magnitude, in radians

- x: array of shape (3,). The point being rotated

RETURNED VALUE

A tuple (u=r(x),du/dr,du/dx):

1. The rotated point(s). Each broadcasted slice has shape (3,)

2. The gradient du/dr. Each broadcasted slice has shape (3,3,). The first
   dimension selects the element of u, and the last dimension selects the
   element of r

3. The gradient du/dx. Each broadcasted slice has shape (3,3). The first
   dimension selects the element of u, and the last dimension selects the
   element of x """,
            args_input       = ('r', 'x'),
            prototype_input  = ((3,), (3,)),
            prototype_output = ((3,), (3,3), (3,3)),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_rotate_point_r( (double*)data_slice__output0,
                          (double*)data_slice__output1,
                          (double*)data_slice__output2,
                          (const double*)data_slice__r,
                          (const double*)data_slice__x );
    return true;
'''},
            Ccode_validate = r'''
            return \
              CHECK_CONTIGUOUS_AND_SETERROR_ALL();
'''
)

m.function( "_transform_point_Rt",
            """Transform a point using an Rt transformation

SYNOPSIS

    r  = rotation_axis * rotation_magnitude
    rt = nps.glue(r,t, axis=-1)
    Rt = mrcal.Rt_from_rt(rt)

    print(Rt.shape)
    ===>
    (4,3)

    print(x.shape)
    ===>
    (10,3)

    print( mrcal._transform_point_Rt(Rt, x).shape )
    ===>
    (10,3)

This is an internal function. You probably want mrcal.transform_point_Rt()

Transform point(s) by an Rt transformation: a (4,3) array formed by
nps.glue(R,t, axis=-2) where R is a (3,3) rotation matrix and t is a (3,)
translation vector. This transformation is defined by a matrix multiplication
and an addition. x and t are stored as a row vector (that's how numpy stores
1-dimensional arrays), but the multiplication works as if x was a column vector
(to match linear algebra conventions):

_transform_point_Rt(Rt, x) = transpose( matmult(Rt[:3,:], transpose(x)) +
                                        transpose(Rt[3,:]) ) =
                           = matmult(x, transpose(Rt[:3,:])) +
                             transpose(Rt[3,:])

This function supports broadcasting fully, so we can transform lots of points at
the same time and/or apply lots of different transformations at the same time

ARGUMENTS

- Rt: array of shape (4,3). This matrix defines the transformation. Rt[:3,:] is
  a rotation matrix; Rt[3,:] is a translation. It is assumed that the rotation
  matrix is a valid rotation (matmult(R,transpose(R)) = I, det(R) = 1), but that
  is not checked

- x: array of shape (3,). The point being transformed

RETURNED VALUE

The transformed point(s). Each broadcasted slice has shape (3,)

""",
            args_input       = ('Rt', 'x'),
            prototype_input  = ((4,3), (3,)),
            prototype_output = (3,),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_transform_point_Rt( (double*)data_slice__output,
                              NULL,NULL,NULL,
                              (const double*)data_slice__Rt,
                              (const double*)data_slice__x );
    return true;
'''},
            Ccode_validate = r'''
            return \
              CHECK_CONTIGUOUS_AND_SETERROR_ALL();
'''
)

m.function( "_transform_point_Rt_withgrad",
            """Transform a point using an Rt transformation; report the result and gradients

SYNOPSIS

    r  = rotation_axis * rotation_magnitude
    rt = nps.glue(r,t, axis=-1)
    Rt = mrcal.Rt_from_rt(rt)

    print(Rt.shape)
    ===>
    (4,3)

    print(x.shape)
    ===>
    (10,3)

    print( [arr.shape for arr in mrcal._transform_point_Rt_withgrad(Rt, x)] )
    ===>
    [(10,3), (10,3,3,3), (10,3,3), (10,3,3)]

This is an internal function. You probably want mrcal.transform_point_Rt()

Transform point(s) by an Rt transformation: a (4,3) array formed by
nps.glue(R,t, axis=-2) where R is a (3,3) rotation matrix and t is a (3,)
translation vector. Unlike _transform_point_Rt(), this returns a
tuple of the result and the gradients: (u=Rt(x),du/dR,du/dt,du/dx)

This transformation is defined by a matrix multiplication
and an addition. x and t are stored as a row vector (that's how numpy stores
1-dimensional arrays), but the multiplication works as if x was a column vector
(to match linear algebra conventions):

_transform_point_Rt(Rt, x) = transpose( matmult(Rt[:3,:], transpose(x)) +
                                        transpose(Rt[3,:]) ) =
                           = matmult(x, transpose(Rt[:3,:])) +
                             transpose(Rt[3,:])

This function supports broadcasting fully, so we can transform lots of points at
the same time and/or apply lots of different transformations at the same time

ARGUMENTS

- Rt: array of shape (4,3). This matrix defines the transformation. Rt[:3,:] is
  a rotation matrix; Rt[3,:] is a translation. It is assumed that the rotation
  matrix is a valid rotation (matmult(R,transpose(R)) = I, det(R) = 1), but that
  is not checked

- x: array of shape (3,). The point being transformed

RETURNED VALUE

A tuple (u=Rt(x),du/dR,du/dt,du/dx)

1. The transformed point(s). Each broadcasted slice has shape (3,)

2. The gradient du/dR. Each broadcasted slice has shape (3,3,3,). The first
   dimension selects the element of u, and the last 2 dimensions select the
   element of R

3. The gradient du/dt. Each broadcasted slice has shape (3,3). The first
   dimension selects the element of u, and the last dimension selects the
   element of t

4. The gradient du/dx. Each broadcasted slice has shape (3,3). The first
   dimension selects the element of u, and the last dimension selects the
   element of x
""",
            args_input       = ('Rt', 'x'),
            prototype_input  = ((4,3), (3,)),
            prototype_output = ((3,), (3,3,3), (3,3), (3,3)),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_transform_point_Rt( (double*)data_slice__output0,
                              (double*)data_slice__output1,
                              (double*)data_slice__output2,
                              (double*)data_slice__output3,
                              (const double*)data_slice__Rt,
                              (const double*)data_slice__x );
    return true;
'''},
            Ccode_validate = r'''
            return \
              CHECK_CONTIGUOUS_AND_SETERROR_ALL();
'''
)

m.function( "_transform_point_rt",
            """Transform a point using an rt transformation

SYNOPSIS

    r  = rotation_axis * rotation_magnitude
    rt = nps.glue(r,t, axis=-1)

    print(rt.shape)
    ===>
    (6,)

    print(x.shape)
    ===>
    (10,3)

    print( mrcal._transform_point_rt(rt, x).shape )
    ===>
    (10,3)

This is an internal function. You probably want mrcal.transform_point_rt()

Transform point(s) by an rt transformation: a (6,) array formed by
nps.glue(r,t, axis=-1) where r is a (3,) Rodrigues vector and t is a (3,)
translation vector. This transformation is defined by a matrix multiplication
and an addition. x and t are stored as a row vector (that's how numpy stores
1-dimensional arrays), but the multiplication works as if x was a column vector
(to match linear algebra conventions):

_transform_point_rt(rt, x) = transpose( matmult(R_from_r(rt[:3]), transpose(x)) +
                                        transpose(rt[3,:]) ) =
                           = matmult(x, transpose(R_from_r(rt[:3]))) +
                             transpose(rt[3:])

This function supports broadcasting fully, so we can transform lots of points at
the same time and/or apply lots of different transformations at the same time

ARGUMENTS

- rt: array of shape (6,). This vector defines the transformation. rt[:3] is a
  rotation defined as a Rodrigues vector; rt[3:] is a translation.

- x: array of shape (3,). The point being transformed

RETURNED VALUE

The transformed point(s). Each broadcasted slice has shape (3,)

""",
            args_input       = ('rt', 'x'),
            prototype_input  = ((6,), (3,)),
            prototype_output = (3,),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_transform_point_rt( (double*)data_slice__output,
                              NULL,NULL,NULL,
                              (const double*)data_slice__rt,
                              (const double*)data_slice__x );
    return true;
'''},
            Ccode_validate = r'''
            return \
              CHECK_CONTIGUOUS_AND_SETERROR_ALL();
'''
)

m.function( "_transform_point_rt_withgrad",
            """Transform a point using an rt transformation; report the result and gradients

SYNOPSIS

    r  = rotation_axis * rotation_magnitude
    rt = nps.glue(r,t, axis=-1)

    print(rt.shape)
    ===>
    (6,)

    print(x.shape)
    ===>
    (10,3)

    print( [arr.shape for arr in mrcal._transform_point_rt_withgrad(rt, x)] )
    ===>
    [(10,3), (10,3,3), (10,3,3), (10,3,3)]

This is an internal function. You probably want mrcal.transform_point_rt()

Transform point(s) by an rt transformation: a (6,) array formed by nps.glue(r,t,
axis=-1) where r is a (3,) Rodrigues vector and t is a (3,) translation vector.
Unlike _transform_point_Rt(), this returns a tuple of the result and the
gradients: (u=rt(x),du/dr,du/dt,du/dx)

This transformation is defined by a matrix multiplication and an addition. x and
t are stored as a row vector (that's how numpy stores 1-dimensional arrays), but
the multiplication works as if x was a column vector (to match linear algebra
conventions):

_transform_point_rt(rt, x) = transpose( matmult(R_from_r(rt[:3]), transpose(x)) +
                                        transpose(rt[3,:]) ) =
                           = matmult(x, transpose(R_from_r(rt[:3]))) +
                             transpose(rt[3:])

This function supports broadcasting fully, so we can transform lots of points at
the same time and/or apply lots of different transformations at the same time

ARGUMENTS

- rt: array of shape (6,). This vector defines the transformation. rt[:3] is a
  rotation defined as a Rodrigues vector; rt[3:] is a translation.

- x: array of shape (3,). The point being transformed

RETURNED VALUE

A tuple (u=Rt(x),du/dR,du/dt,du/dx)

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

""",
            args_input       = ('rt', 'x'),
            prototype_input  = ((6,), (3,)),
            prototype_output = ((3,), (3,3), (3,3), (3,3)),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_transform_point_rt( (double*)data_slice__output0,
                              (double*)data_slice__output1,
                              (double*)data_slice__output2,
                              (double*)data_slice__output3,
                              (const double*)data_slice__rt,
                              (const double*)data_slice__x );
    return true;
'''},
            Ccode_validate = r'''
            return \
              CHECK_CONTIGUOUS_AND_SETERROR_ALL();
'''
)

m.function( "_r_from_R",
            "Compute a Rodrigues vector from a rotation matrix",

            args_input       = ('R',),
            prototype_input  = ((3,3),),
            prototype_output = (3,),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_r_from_R_noncontiguous(
                 (double*)data_slice__output,strides_slice__output[0],
                 NULL,0,0,0,
                 (const double*)data_slice__R,strides_slice__R[0], strides_slice__R[1] );
    return true;
'''}
)

m.function( "_r_from_R_withgrad",
            "Compute a Rodrigues vector and a gradient from a rotation matrix",

            args_input       = ('R',),
            prototype_input  = ((3,3),),
            prototype_output = ((3,),(3,3,3)),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_r_from_R_noncontiguous(
                 (double*)data_slice__output0,strides_slice__output0[0],
                 (double*)data_slice__output1,strides_slice__output1[0], strides_slice__output1[1],strides_slice__output1[2],
                 (const double*)data_slice__R,strides_slice__R[0], strides_slice__R[1] );
    return true;
'''}
)

m.function( "_R_from_r",
            "Compute a rotation matrix from a Rodrigues vector",

            args_input       = ('r',),
            prototype_input  = ((3,),),
            prototype_output = (3,3),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_R_from_r_noncontiguous(
                 (double*)data_slice__output, strides_slice__output[0], strides_slice__output[1],
                 NULL,0,0,0,
                 (const double*)data_slice__r, strides_slice__r[0] );
    return true;
'''}
)

m.function( "_R_from_r_withgrad",
            "Compute a rotation matrix and a gradient from a Rodrigues vector",

            args_input       = ('r',),
            prototype_input  = ((3,),),
            prototype_output = ((3,3), (3,3,3)),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_R_from_r_noncontiguous(
                 (double*)data_slice__output0,strides_slice__output0[0], strides_slice__output0[1],
                 (double*)data_slice__output1,strides_slice__output1[0], strides_slice__output1[1],strides_slice__output1[2],
                 (const double*)data_slice__r, strides_slice__r[0] );
    return true;
'''}
)

m.function( "rt_from_Rt",
            "Compute an rt transformation from a Rt transformation",

            args_input       = ('Rt',),
            prototype_input  = ((4,3),),
            prototype_output = (6,),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_rt_from_Rt_noncontiguous(
                 (double*)data_slice__output,strides_slice__output[0],
                 (const double*)data_slice__Rt,strides_slice__Rt[0], strides_slice__Rt[1] );
    return true;
'''}
)

m.function( "Rt_from_rt",
            "Compute an Rt transformation from a rt transformation",

            args_input       = ('rt',),
            prototype_input  = ((6,),),
            prototype_output = (4,3),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_Rt_from_rt_noncontiguous(
                 (double*)data_slice__output, strides_slice__output[0],strides_slice__output[1],
                 (const double*)data_slice__rt, strides_slice__rt[0] );
    return true;
'''}
)

m.function( "invert_Rt",
            "invert an Rt transformation",

            args_input       = ('Rt',),
            prototype_input  = ((4,3),),
            prototype_output = (4,3),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_invert_Rt( (double*)data_slice__output,
                     (const double*)data_slice__Rt );
    return true;
'''},
            Ccode_validate = r'''
            return \
              CHECK_CONTIGUOUS_AND_SETERROR_ALL();
'''
)

m.function( "invert_rt",
            "invert an rt transformation",

            args_input       = ('rt',),
            prototype_input  = ((6,),),
            prototype_output = (6,),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_invert_rt( (double*)data_slice__output,
                     (const double*)data_slice__rt );
    return true;
'''},
            Ccode_validate = r'''
            return \
              CHECK_CONTIGUOUS_AND_SETERROR_ALL();
'''
)

m.function( "_compose_Rt",
            "Composes exactly 2 Rt transformations\n"
            "\n"
            "Given 2 Rt transformations, returns their composition. This is an internal\n"
            "function used by mrcal.compose_Rt(), which supports >2 input transformations\n",

            args_input       = ('Rt0', 'Rt1'),
            prototype_input  = ((4,3,), (4,3,)),
            prototype_output = (4,3),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_compose_Rt( (double*)data_slice__output,
                      (const double*)data_slice__Rt0,
                      (const double*)data_slice__Rt1 );
    return true;
'''},
            Ccode_validate = r'''
            return \
              CHECK_CONTIGUOUS_AND_SETERROR_ALL();
'''
)

m.function( "_compose_rt_withgrad",
            "Compose 2 rt transformations; Return (rt,dr/dr0,dr/dr1,dt/dr0,dt/dt1)\n"
            "\n"
            "Given 2 Rt transformations, return their composition. This is an internal\n"
            "function used by mrcal.compose_rt(), which supports >2 input transformations.\n"
            "THIS path is used from Python only if we need gradients"
            "\n"
            "dt_dr1 is not returned: it is always 0\n"
            "dt_dt0 is not returned: it is always the identity matrix\n",

            args_input       = ('rt0', 'rt1'),
            prototype_input  = ((6,), (6,)),
            prototype_output = ((6,), (3,3),(3,3),(3,3),(3,3)),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_compose_rt( (double*)data_slice__output0,
                      (double*)data_slice__output1,
                      (double*)data_slice__output2,
                      (double*)data_slice__output3,
                      (double*)data_slice__output4,
                      (const double*)data_slice__rt0,
                      (const double*)data_slice__rt1 );
    return true;
'''},
            Ccode_validate = r'''
            return \
              CHECK_CONTIGUOUS_AND_SETERROR_ALL();
'''
)

m.write()
