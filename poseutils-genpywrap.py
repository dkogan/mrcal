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

This is an internal function. You probably want mrcal.rotate_point_R(). See the
docs for that function for details.

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

This is an internal function. You probably want mrcal.rotate_point_R(). See the
docs for that function for details.
""",

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

This is an internal function. You probably want mrcal.rotate_point_r(). See the
docs for that function for details.
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

This is an internal function. You probably want mrcal.rotate_point_r(). See the
docs for that function for details.
""",
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

This is an internal function. You probably want mrcal.transform_point_Rt(). See
the docs for that function for details.
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

This is an internal function. You probably want mrcal.transform_point_Rt(). See
the docs for that function for details.
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

This is an internal function. You probably want mrcal.transform_point_Rt(). See
the docs for that function for details.
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

This is an internal function. You probably want mrcal.transform_point_Rt(). See
the docs for that function for details.
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
            """Compute a Rodrigues vector from a rotation matrix

This is an internal function. You probably want mrcal.r_from_R(). See the docs
for that function for details.
""",
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
            """Compute a Rodrigues vector from a rotation matrix

This is an internal function. You probably want mrcal.r_from_R(). See the docs
for that function for details.
""",
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
            """Compute a rotation matrix from a Rodrigues vector

This is an internal function. You probably want mrcal.R_from_r(). See the docs
for that function for details.
""",
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
            """Compute a rotation matrix from a Rodrigues vector

This is an internal function. You probably want mrcal.R_from_r(). See the docs
for that function for details.
""",
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

m.function( "_rt_from_Rt",
            """Compute an rt transformation from a Rt transformation

This is an internal function. You probably want mrcal.rt_from_Rt(). See the docs
for that function for details.
""",
            args_input       = ('Rt',),
            prototype_input  = ((4,3),),
            prototype_output = (6,),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_rt_from_Rt_noncontiguous(
                 (double*)data_slice__output,strides_slice__output[0],
                 NULL,0,0,0,
                 (const double*)data_slice__Rt,strides_slice__Rt[0], strides_slice__Rt[1] );
    return true;
'''}
)

m.function( "_rt_from_Rt_withgrad",
            """Compute an rt transformation from a Rt transformation

This is an internal function. You probably want mrcal.rt_from_Rt(). See the docs
for that function for details.
""",
            args_input       = ('Rt',),
            prototype_input  = ((4,3),),
            prototype_output = ((6,), (3,3,3)),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_rt_from_Rt_noncontiguous(
                 (double*)data_slice__output0,strides_slice__output0[0],
                 (double*)data_slice__output1,strides_slice__output1[0], strides_slice__output1[1],strides_slice__output1[2],
                 (const double*)data_slice__Rt,strides_slice__Rt[0], strides_slice__Rt[1] );
    return true;
'''}
)

m.function( "_Rt_from_rt",
            """Compute an Rt transformation from a rt transformation

This is an internal function. You probably want mrcal.Rt_from_rt(). See the docs
for that function for details.
""",
            args_input       = ('rt',),
            prototype_input  = ((6,),),
            prototype_output = (4,3),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_Rt_from_rt_noncontiguous(
                 (double*)data_slice__output, strides_slice__output[0],strides_slice__output[1],
                 NULL,0,0,0,
                 (const double*)data_slice__rt, strides_slice__rt[0] );
    return true;
'''}
)

m.function( "_Rt_from_rt_withgrad",
            """Compute an Rt transformation from a rt transformation

This is an internal function. You probably want mrcal.Rt_from_rt(). See the docs
for that function for details.
""",
            args_input       = ('rt',),
            prototype_input  = ((6,),),
            prototype_output = ((4,3), (3,3,3)),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_Rt_from_rt_noncontiguous(
                 (double*)data_slice__output0, strides_slice__output0[0],strides_slice__output0[1],
                 (double*)data_slice__output1,strides_slice__output1[0], strides_slice__output1[1],strides_slice__output1[2],
                 (const double*)data_slice__rt, strides_slice__rt[0] );
    return true;
'''}
)

m.function( "invert_Rt",
            """Invert an Rt transformation

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

ARGUMENTS

- Rt: array of shape (4,3). This matrix defines the transformation. Rt[:3,:] is
  a rotation matrix; Rt[3,:] is a translation. It is assumed that the rotation
  matrix is a valid rotation (matmult(R,transpose(R)) = I, det(R) = 1), but that
  is not checked

RETURNED VALUE

The inverse Rt transformation in an array of shape (4,3).
""",
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

m.function( "_invert_rt",
            """Invert an rt transformation

This is an internal function. You probably want mrcal.invert_rt(). See the docs
for that function for details.
""",
            args_input       = ('rt',),
            prototype_input  = ((6,),),
            prototype_output = (6,),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_invert_rt( (double*)data_slice__output,
                     NULL,NULL,
                     (const double*)data_slice__rt );
    return true;
'''},
            Ccode_validate = r'''
            return \
              CHECK_CONTIGUOUS_AND_SETERROR_ALL();
'''
)

m.function( "_invert_rt_withgrad",
            """Invert an rt transformation

This is an internal function. You probably want mrcal.invert_rt(). See the docs
for that function for details.
""",
            args_input       = ('rt',),
            prototype_input  = ((6,),),
            prototype_output = ((6,), (3,3), (3,3)),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_invert_rt( (double*)data_slice__output0,
                     (double*)data_slice__output1,
                     (double*)data_slice__output2,
                     (const double*)data_slice__rt );
    return true;
'''},
            Ccode_validate = r'''
            return \
              CHECK_CONTIGUOUS_AND_SETERROR_ALL();
'''
)

m.function( "_compose_Rt",
            """Composes two Rt transformations

This is an internal function. You probably want mrcal.compose_Rt(). See the docs
for that function for details. This internal function differs from compose_Rt():

- It supports exactly two arguments, while compose_Rt() can compose N
  transformations
""",

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
            """Compose two rt transformations; return (rt,dr/dr0,dr/dr1,dt/dr0,dt/dt1)

This is an internal function. You probably want mrcal.compose_rt(). See the docs
for that function for details. This internal function differs from compose_rt():

- It supports exactly two arguments, while compose_rt() can compose N
  transformations

- It always reports gradients

dr/dt0 is not returned: it is always 0
dr/dt1 is not returned: it is always 0
dt/dr1 is not returned: it is always 0
dt/dt0 is not returned: it is always the identity matrix
""",

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
