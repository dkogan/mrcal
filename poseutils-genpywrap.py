#!/usr/bin/env python3

# Copyright (c) 2017-2023 California Institute of Technology ("Caltech"). U.S.
# Government sponsorship acknowledged. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

r'''Python-wrap the mrcal geometry routines

'''

import sys
import os

import numpy as np
import numpysane as nps

import numpysane_pywrap as npsp



docstring_module = '''Low-level routines to manipulate poses, transformations and points

This is the written-in-C Python extension module. Most of the time you want to
use the mrcal.poseutils wrapper module instead of this module directly. Any
functions not prefixed with "_" are meant to be called directly, without the
wrapper.

All functions are exported into the mrcal module. So you can call these via
mrcal._poseutils.fff() or mrcal.fff(). The latter is preferred.

'''


m = npsp.module( name      = "_poseutils_npsp",
                 docstring = docstring_module,
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

As with all the poseutils functions, the output can be written directly into a
(possibly-non-contiguous) array, by specifying the destination in the 'out'
kwarg """,

            args_input       = (),
            prototype_input  = (),
            prototype_output = (3,3),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_identity_R_full( (double*)data_slice__output,
                           strides_slice__output[0],
                           strides_slice__output[1] );
    return true;
'''})

m.function( "identity_r",
            """Return an identity Rodrigues rotation

SYNOPSIS

    print( mrcal.identity_r() )
    ===>
    [0. 0. 0.]

As with all the poseutils functions, the output can be written directly into a
(possibly-non-contiguous) array, by specifying the destination in the 'out'
kwarg""",

            args_input       = (),
            prototype_input  = (),
            prototype_output = (3,),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_identity_r_full( (double*)data_slice__output,
                           strides_slice__output[0] );
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

As with all the poseutils functions, the output can be written directly into a
(possibly-non-contiguous) array, by specifying the destination in the 'out'
kwarg""",

            args_input       = (),
            prototype_input  = (),
            prototype_output = (4,3),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_identity_Rt_full( (double*)data_slice__output,
                            strides_slice__output[0],
                            strides_slice__output[1] );
    return true;
'''})

m.function( "identity_rt",
            """Return an identity rt transformation

SYNOPSIS

    print( mrcal.identity_rt() )
    ===>
    [0. 0. 0. 0. 0. 0.]

As with all the poseutils functions, the output can be written directly into a
(possibly-non-contiguous) array, by specifying the destination in the 'out'
kwarg""",

            args_input       = (),
            prototype_input  = (),
            prototype_output = (6,),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_identity_rt_full( (double*)data_slice__output,
                            strides_slice__output[0] );
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
            extra_args = (("int", "inverted", "false", "p"),),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_rotate_point_R_full( (double*)data_slice__output,
                               strides_slice__output[0],
                               NULL,0,0,0,
                               NULL,0,0,
                               (const double*)data_slice__R,
                               strides_slice__R[0],
                               strides_slice__R[1],
                               (const double*)data_slice__x,
                               strides_slice__x[0],
                               *inverted );
    return true;
'''},
)

m.function( "_rotate_point_R_withgrad",
            """Rotate a point using a rotation matrix; report the result and gradients

This is an internal function. You probably want mrcal.rotate_point_R(). See the
docs for that function for details.
""",

            args_input       = ('R', 'x'),
            prototype_input  = ((3,3), (3,)),
            prototype_output = ((3,), (3,3,3), (3,3)),
            extra_args = (("int", "inverted", "false", "p"),),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_rotate_point_R_full( (double*)data_slice__output0,
                               strides_slice__output0[0],
                               (double*)data_slice__output1,
                               strides_slice__output1[0],
                               strides_slice__output1[1],
                               strides_slice__output1[2],
                               (double*)data_slice__output2,
                               strides_slice__output2[0],
                               strides_slice__output2[1],
                               (const double*)data_slice__R,
                               strides_slice__R[0],
                               strides_slice__R[1],
                               (const double*)data_slice__x,
                               strides_slice__x[0],
                               *inverted );
    return true;
'''},
)

m.function( "_rotate_point_r",
            """Rotate a point using a Rodrigues vector

This is an internal function. You probably want mrcal.rotate_point_r(). See the
docs for that function for details.
""",
            args_input       = ('r', 'x'),
            prototype_input  = ((3,), (3,)),
            prototype_output = (3,),
            extra_args = (("int", "inverted", "false", "p"),),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_rotate_point_r_full( (double*)data_slice__output,
                               strides_slice__output[0],
                               NULL,0,0,
                               NULL,0,0,
                               (const double*)data_slice__r,
                               strides_slice__r[0],
                               (const double*)data_slice__x,
                               strides_slice__x[0],
                               *inverted);
    return true;
'''},
)

m.function( "_rotate_point_r_withgrad",
            """Rotate a point using a Rodrigues vector; report the result and gradients

This is an internal function. You probably want mrcal.rotate_point_r(). See the
docs for that function for details.
""",
            args_input       = ('r', 'x'),
            prototype_input  = ((3,), (3,)),
            prototype_output = ((3,), (3,3), (3,3)),
            extra_args = (("int", "inverted", "false", "p"),),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_rotate_point_r_full( (double*)data_slice__output0,
                               strides_slice__output0[0],
                               (double*)data_slice__output1,
                               strides_slice__output1[0],
                               strides_slice__output1[1],
                               (double*)data_slice__output2,
                               strides_slice__output2[0],
                               strides_slice__output2[1],
                               (const double*)data_slice__r,
                               strides_slice__r[0],
                               (const double*)data_slice__x,
                               strides_slice__x[0],
                               *inverted);
    return true;
'''},
)

m.function( "_transform_point_Rt",
            """Transform a point using an Rt transformation

This is an internal function. You probably want mrcal.transform_point_Rt(). See
the docs for that function for details.
""",
            args_input       = ('Rt', 'x'),
            prototype_input  = ((4,3), (3,)),
            prototype_output = (3,),
            extra_args = (("int", "inverted", "false", "p"),),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_transform_point_Rt_full( (double*)data_slice__output,
                                   strides_slice__output[0],
                                   NULL,0,0,0,
                                   NULL,0,0,
                                   (const double*)data_slice__Rt,
                                   strides_slice__Rt[0],
                                   strides_slice__Rt[1],
                                   (const double*)data_slice__x,
                                   strides_slice__x[0],
                                   *inverted );
    return true;
'''},
)

m.function( "_transform_point_Rt_withgrad",
            """Transform a point using an Rt transformation; report the result and gradients

This is an internal function. You probably want mrcal.transform_point_Rt(). See
the docs for that function for details.
""",
            args_input       = ('Rt', 'x'),
            prototype_input  = ((4,3), (3,)),
            prototype_output = ((3,), (3,4,3), (3,3)),
            extra_args = (("int", "inverted", "false", "p"),),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_transform_point_Rt_full( (double*)data_slice__output0,
                                   strides_slice__output0[0],
                                   (double*)data_slice__output1,
                                   strides_slice__output1[0],
                                   strides_slice__output1[1],
                                   strides_slice__output1[2],
                                   (double*)data_slice__output2,
                                   strides_slice__output2[0],
                                   strides_slice__output2[1],
                                   (const double*)data_slice__Rt,
                                   strides_slice__Rt[0],
                                   strides_slice__Rt[1],
                                   (const double*)data_slice__x,
                                   strides_slice__x[0],
                                   *inverted );
    return true;
'''},
)

m.function( "_transform_point_rt",
            """Transform a point using an rt transformation

This is an internal function. You probably want mrcal.transform_point_rt(). See
the docs for that function for details.
""",
            args_input       = ('rt', 'x'),
            prototype_input  = ((6,), (3,)),
            prototype_output = (3,),
            extra_args = (("int", "inverted", "false", "p"),),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_transform_point_rt_full( (double*)data_slice__output,
                                   strides_slice__output[0],
                                   NULL,0,0,
                                   NULL,0,0,
                                   (const double*)data_slice__rt,
                                   strides_slice__rt[0],
                                   (const double*)data_slice__x,
                                   strides_slice__x[0],
                                   *inverted );
    return true;
'''},
)

m.function( "_transform_point_rt_withgrad",
            """Transform a point using an rt transformation; report the result and gradients

This is an internal function. You probably want mrcal.transform_point_rt(). See
the docs for that function for details.
""",
            args_input       = ('rt', 'x'),
            prototype_input  = ((6,), (3,)),
            prototype_output = ((3,), (3,6), (3,3)),
            extra_args = (("int", "inverted", "false", "p"),),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_transform_point_rt_full( (double*)data_slice__output0,
                                   strides_slice__output0[0],
                                   (double*)data_slice__output1,
                                   strides_slice__output1[0],
                                   strides_slice__output1[1],
                                   (double*)data_slice__output2,
                                   strides_slice__output2[0],
                                   strides_slice__output2[1],
                                   (const double*)data_slice__rt,
                                   strides_slice__rt[0],
                                   (const double*)data_slice__x,
                                   strides_slice__x[0],
                                   *inverted );
    return true;
'''},
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
    mrcal_r_from_R_full(
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
    mrcal_r_from_R_full(
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
    mrcal_R_from_r_full(
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
    mrcal_R_from_r_full(
        (double*)data_slice__output0,strides_slice__output0[0], strides_slice__output0[1],
        (double*)data_slice__output1,strides_slice__output1[0], strides_slice__output1[1],strides_slice__output1[2],
        (const double*)data_slice__r, strides_slice__r[0] );
    return true;
'''}
)

m.function( "_invert_R",
            """Invert a rotation matrix

This is an internal function. You probably want mrcal.invert_R(). See the docs
for that function for details.
""",
            args_input       = ('R',),
            prototype_input  = ((3,3),),
            prototype_output = (3,3),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_invert_R_full( (double*)data_slice__output,
                         strides_slice__output[0], strides_slice__output[1],
                         (const double*)data_slice__R,
                         strides_slice__R[0], strides_slice__R[1] );
    return true;
'''},
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
    mrcal_rt_from_Rt_full(
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
    mrcal_rt_from_Rt_full(
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
    mrcal_Rt_from_rt_full(
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
    mrcal_Rt_from_rt_full(
        (double*)data_slice__output0, strides_slice__output0[0],strides_slice__output0[1],
        (double*)data_slice__output1,strides_slice__output1[0], strides_slice__output1[1],strides_slice__output1[2],
        (const double*)data_slice__rt, strides_slice__rt[0] );
    return true;
'''}
)

m.function( "_invert_Rt",
            """Invert an Rt transformation

This is an internal function. You probably want mrcal.invert_Rt(). See the docs
for that function for details.
""",
            args_input       = ('Rt',),
            prototype_input  = ((4,3),),
            prototype_output = (4,3),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_invert_Rt_full( (double*)data_slice__output,
                          strides_slice__output[0], strides_slice__output[1],
                          (const double*)data_slice__Rt,
                          strides_slice__Rt[0], strides_slice__Rt[1] );
    return true;
'''},
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
    mrcal_invert_rt_full( (double*)data_slice__output,
                          strides_slice__output[0],
                          NULL,0,0,
                          NULL,0,0,
                          (const double*)data_slice__rt,
                          strides_slice__rt[0] );
    return true;
'''},
)

m.function( "_invert_rt_withgrad",
            """Invert an rt transformation

This is an internal function. You probably want mrcal.invert_rt(). See the docs
for that function for details.

Note that the C library returns limited gradients:

- It returns dtout_drin,dtout_dtin only because

- drout_drin always -I
- drout_dtin always 0

THIS function combines these into a full drtout_drtin array

""",
            args_input       = ('rt',),
            prototype_input  = ((6,),),
            prototype_output = ((6,), (6,6)),

            # output1 is drtout/drtin = [ drout/drin drout/dtin ]
            #                           [ dtout/drin dtout/dtin ]
            #
            #                         = [     -I        0       ]
            #                           [ dtout/drin dtout/dtin ]
            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_invert_rt_full( (double*)data_slice__output0,
                          strides_slice__output0[0],

                          &item__output1(3,0),
                          strides_slice__output1[0], strides_slice__output1[1],

                          &item__output1(3,3),
                          strides_slice__output1[0], strides_slice__output1[1],

                          (const double*)data_slice__rt,
                          strides_slice__rt[0] );
    for(int i=0; i<3; i++)
        for(int j=0; j<6; j++)
            item__output1(i,j) = 0;
    item__output1(0,0) = -1.;
    item__output1(1,1) = -1.;
    item__output1(2,2) = -1.;

    return true;
'''},
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
    mrcal_compose_Rt_full( (double*)data_slice__output,
                           strides_slice__output[0], strides_slice__output[1],
                           (const double*)data_slice__Rt0,
                           strides_slice__Rt0[0], strides_slice__Rt0[1],
                           (const double*)data_slice__Rt1,
                           strides_slice__Rt1[0], strides_slice__Rt1[1] );
    return true;
'''},
)

m.function( "_compose_r",
            """Compose two angle-axis rotations

This is an internal function. You probably want mrcal.compose_r(). See the docs
for that function for details. This internal function differs from compose_r():

- It supports exactly two arguments, while compose_r() can compose N rotations

- It never reports gradients
""",

            args_input       = ('r0', 'r1'),
            prototype_input  = ((3,), (3,)),
            prototype_output = (3,),
            extra_args = (("int", "inverted0", "false", "p"),
                          ("int", "inverted1", "false", "p"),),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_compose_r_full( (double*)data_slice__output,
                           strides_slice__output[0],
                           NULL,0,0,
                           NULL,0,0,
                           (const double*)data_slice__r0,
                           strides_slice__r0[0],
                           (const double*)data_slice__r1,
                           strides_slice__r1[0],
                           *inverted0, *inverted1);
    return true;
'''},
)

m.function( "_compose_r_withgrad",
            """Compose two angle-axis rotations; return (r,dr/dr0,dr/dr1)

This is an internal function. You probably want mrcal.compose_r(). See the docs
for that function for details. This internal function differs from compose_r():

- It supports exactly two arguments, while compose_r() can compose N rotations

- It always reports gradients

""",

            args_input       = ('r0', 'r1'),
            prototype_input  = ((3,), (3,)),
            prototype_output = ((3,), (3,3),(3,3)),
            extra_args = (("int", "inverted0", "false", "p"),
                          ("int", "inverted1", "false", "p"),),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_compose_r_full( (double*)data_slice__output0,
                           strides_slice__output0[0],

                           // dr/dr0
                           &item__output1(0,0),
                           strides_slice__output1[0], strides_slice__output1[1],

                           // dr/dr1
                           &item__output2(0,0),
                           strides_slice__output2[0], strides_slice__output2[1],

                           (const double*)data_slice__r0,
                           strides_slice__r0[0],
                           (const double*)data_slice__r1,
                           strides_slice__r1[0],
                           *inverted0, *inverted1);

    return true;
'''},
)

m.function( "compose_r_tinyr0_gradientr0",
    r"""Special-case rotation composition for the uncertainty computation

SYNOPSIS

    r1 = rotation_axis1 * rotation_magnitude1

    dr01_dr0 = compose_r_tinyr0_gradientr0(r1)

    ### Another way to get the same thing (but possibly less efficiently)
     _,dr01_dr0,_ = compose_r(np.zeros((3,),),
                              r1,
                              get_gradients=True)

This is a special-case subset of compose_r(). It is the same, except:

- r0 is assumed to be 0, so we don't ingest it, and we don't report the
  composition result
- we ONLY report the dr01/dr0 gradient

This special-case function is a part of the projection uncertainty computation,
so it exists separate from compose_r(). See the documentation for compose_r()
for all the details.

This function supports broadcasting fully.

ARGUMENTS

- r1: the second of the two rotations being composed. The first rotation is an
  identity, so it's not given

- out: optional argument specifying the destination. By default, a new numpy
  array is created and returned. To write the results into an existing (and
  possibly non-contiguous) array, specify it with the 'out' kwarg

RETURNED VALUE

We return a single array of shape (...,3,3): dr01/dr0

""",

            args_input       = ('r1',),
            prototype_input  = ((3,),),
            prototype_output = (3,3),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_compose_r_tinyr0_gradientr0_full(
        // dr/dr0
        &item__output(0,0),
        strides_slice__output[0], strides_slice__output[1],
        (const double*)data_slice__r1,
        strides_slice__r1[0] );

    return true;
'''},
)

m.function( "compose_r_tinyr1_gradientr1",
    r"""Special-case rotation composition for the uncertainty computation

SYNOPSIS

    r0 = rotation_axis0 * rotation_magnitude0

    dr01_dr1 = compose_r_tinyr1_gradientr1(r0)

    ### Another way to get the same thing (but possibly less efficiently)
     _,_,dr01_dr1 = compose_r(r0,
                              np.zeros((3,),),
                              get_gradients=True)

This is a special-case subset of compose_r(). It is the same, except:

- r1 is assumed to be 0, so we don't ingest it, and we don't report the
  composition result
- we ONLY report the dr01/dr1 gradient

This special-case function is a part of the projection uncertainty computation,
so it exists separate from compose_r(). See the documentation for compose_r()
for all the details.

This function supports broadcasting fully.

ARGUMENTS

- r0: the first of the two rotations being composed. The second rotation is an
  identity, so it's not given

- out: optional argument specifying the destination. By default, a new numpy
  array is created and returned. To write the results into an existing (and
  possibly non-contiguous) array, specify it with the 'out' kwarg

RETURNED VALUE

We return a single array of shape (...,3,3): dr01/dr1

""",

            args_input       = ('r0',),
            prototype_input  = ((3,),),
            prototype_output = (3,3),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_compose_r_tinyr1_gradientr1_full(
        // dr/dr1
        &item__output(0,0),
        strides_slice__output[0], strides_slice__output[1],
        (const double*)data_slice__r0,
        strides_slice__r0[0] );

    return true;
'''},
)

m.function( "_compose_rt",
            """Compose two rt transformations

This is an internal function. You probably want mrcal.compose_rt(). See the docs
for that function for details. This internal function differs from compose_rt():

- It supports exactly two arguments, while compose_rt() can compose N
  transformations

- It never reports gradients
""",

            args_input       = ('rt0', 'rt1'),
            prototype_input  = ((6,), (6,)),
            prototype_output = (6,),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_compose_rt_full( (double*)data_slice__output,
                           strides_slice__output[0],
                           NULL,0,0,
                           NULL,0,0,
                           NULL,0,0,
                           NULL,0,0,
                           (const double*)data_slice__rt0,
                           strides_slice__rt0[0],
                           (const double*)data_slice__rt1,
                           strides_slice__rt1[0] );
    return true;
'''},
)

m.function( "_compose_rt_withgrad",
            """Compose two rt transformations; return (rt,drt/drt0,drt/drt1)

This is an internal function. You probably want mrcal.compose_rt(). See the docs
for that function for details. This internal function differs from compose_rt():

- It supports exactly two arguments, while compose_rt() can compose N
  transformations

- It always reports gradients

Note that the C library returns limited gradients:

- dr/dt0 is not returned: it is always 0
- dr/dt1 is not returned: it is always 0
- dt/dr1 is not returned: it is always 0
- dt/dt0 is not returned: it is always the identity matrix

THIS function combines these into the full drtout_drt0,drtout_drt1 arrays

""",

            args_input       = ('rt0', 'rt1'),
            prototype_input  = ((6,), (6,)),
            prototype_output = ((6,), (6,6),(6,6)),

            # output1 is drt/drt0 = [ dr/dr0 dr/dt0 ]
            #                       [ dt/dr0 dt/dt0 ]
            #
            #                     = [ dr/dr0   0    ]
            #                       [ dt/dr0   I    ]
            #
            # output2 is drt/drt1 = [ dr/dr1 dr/dt1 ]
            #                       [ dt/dr1 dt/dt1 ]
            #
            #                     = [ dr/dr1   0    ]
            #                       [   0    dt/dt1 ]

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_compose_rt_full( (double*)data_slice__output0,
                           strides_slice__output0[0],

                           // dr/dr0
                           &item__output1(0,0),
                           strides_slice__output1[0], strides_slice__output1[1],

                           // dr/dr1
                           &item__output2(0,0),
                           strides_slice__output2[0], strides_slice__output2[1],

                           // dt/dr0
                           &item__output1(3,0),
                           strides_slice__output1[0], strides_slice__output1[1],

                           // dt/dt1
                           &item__output2(3,3),
                           strides_slice__output2[0], strides_slice__output2[1],

                           (const double*)data_slice__rt0,
                           strides_slice__rt0[0],
                           (const double*)data_slice__rt1,
                           strides_slice__rt1[0] );
    for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
        {
            item__output1(i,  j+3) = 0;
            item__output1(i+3,j+3) = 0;
            item__output2(i,  j+3) = 0;
            item__output2(i+3,j  ) = 0;
        }

    item__output1(3,3) = 1.;
    item__output1(4,4) = 1.;
    item__output1(5,5) = 1.;

    return true;
'''},
)

m.function( "R_from_quat",
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

This function supports broadcasting fully.

ARGUMENTS

- quat: array of shape (4,). The unit quaternion that defines the rotation. The
  values in the array are (u,i,j,k)

- out: optional argument specifying the destination. By default, new numpy
  array(s) are created and returned. To write the results into existing (and
  possibly non-contiguous) arrays, specify them with the 'out' kwarg. If 'out'
  is given, we return the 'out' that was passed in. This is the standard
  behavior provided by numpysane_pywrap.

RETURNED VALUE

We return an array of rotation matrices. Each broadcasted slice has shape (3,3)

    """,
            args_input       = ('q',),
            prototype_input  = ((4,),),
            prototype_output = (3,3),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    // From the expression in wikipedia
    const double r = item__q(0);
    const double i = item__q(1);
    const double j = item__q(2);
    const double k = item__q(3);

    const double ii = i*i;
    const double ij = i*j;
    const double ik = i*k;
    const double ir = i*r;
    const double jj = j*j;
    const double jk = j*k;
    const double jr = j*r;
    const double kk = k*k;
    const double kr = k*r;

    item__output(0,0) = 1. - 2.*(jj+kk);
    item__output(0,1) =      2.*(ij-kr);
    item__output(0,2) =      2.*(ik+jr);

    item__output(1,0) =      2.*(ij+kr);
    item__output(1,1) = 1. - 2.*(ii+kk);
    item__output(1,2) =      2.*(jk-ir);

    item__output(2,0) =      2.*(ik-jr);
    item__output(2,1) =      2.*(jk+ir);
    item__output(2,2) = 1. - 2.*(ii+jj);

    return true;
'''}
)

m.function( "skew_symmetric",
            r"""Return the skew-symmetric matrix used in a cross product

SYNOPSIS

    a = np.array(( 1.,  5.,  7.))
    b = np.array(( 3., -.1, -10.))

    A = mrcal.skew_symmetric(a)

    print( nps.inner(A,b) )
    ===>
    [-49.3  31.  -15.1]

    print( np.cross(a,b) )
    ===>
    [-49.3  31.  -15.1]

A vector cross-product a x b can be represented as a matrix multiplication A*b
where A is a skew-symmetric matrix based on the vector a. This function computes
this matrix A from the vector a.

This function supports broadcasting fully.

ARGUMENTS

- a: array of shape (3,)

- out: optional argument specifying the destination. By default, new numpy
  array(s) are created and returned. To write the results into existing (and
  possibly non-contiguous) arrays, specify them with the 'out' kwarg. If 'out'
  is given, we return the 'out' that was passed in. This is the standard
  behavior provided by numpysane_pywrap.

RETURNED VALUE

We return the matrix A in a (3,3) numpy array

    """,
            args_input       = ('a',),
            prototype_input  = ((3,),),
            prototype_output = (3,3),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    // diagonal is zero
    item__output(0,0) = 0.0;
    item__output(1,1) = 0.0;
    item__output(2,2) = 0.0;

    item__output(0,1) = -item__a(2);
    item__output(0,2) =  item__a(1);
    item__output(1,0) =  item__a(2);
    item__output(1,2) = -item__a(0);
    item__output(2,0) = -item__a(1);
    item__output(2,1) =  item__a(0);

    return true;
'''}
)

for w in ('weights', 'noweights'):
    for kind in ('R01', 'Rt01'):

        if w == 'weights':
            args_input      = ('v0','v1','weights')
            prototype_input = (('N',3,), ('N',3,), ('N',))
            weightarg = "(double*)data_slice__weights"
        else:
            args_input      = ('v0','v1')
            prototype_input = (('N',3,), ('N',3,))
            weightarg = "NULL"

        if kind == 'R01':
            what             = 'vectors'
            prototype_output = (3,3)
            Nelements_output = 9
        else:
            what             = 'points'
            prototype_output = (4,3)
            Nelements_output = 12

        m.function( f"_align_procrustes_{what}_{kind}_{w}",
            r"""Compute a rotation to align two sets of direction vectors or points

        This is the written-in-C Python extension module. Most of the time you want to
        use the mrcal.poseutils wrapper module instead of this module directly. Any
        functions not prefixed with "_" are meant to be called directly, without the
        wrapper.

        All functions are exported into the mrcal module. So you can call these via
        mrcal._poseutils.fff() or mrcal.fff(). The latter is preferred.

            """,
                    args_input       = args_input,
                    prototype_input  = prototype_input,
                    prototype_output = prototype_output,

                    Ccode_validate = r'''
                    return CHECK_CONTIGUOUS_AND_SETERROR_ALL();''',

                    Ccode_slice_eval = \
                        {np.float64:
                         rf'''
            bool result =
            mrcal_align_procrustes_{what}_{kind}((double*)data_slice__output,
                                                 dims_slice__v0[0],
                                                 (double*)data_slice__v0,
                                                 (double*)data_slice__v1,
                                                 {weightarg});

            if(!result && 0.0 == *(double*)data_slice__output)
            {{

                // Poorly-defined problem. I indicate this with an all-zero
                // output, but I return true. This allows us to process
                // lots of data via broadcasting, without breaking ALL
                // the slices if one slice is broken
                memset((double*)data_slice__output, 0, {Nelements_output}*sizeof(double));
                return true;
            }}
            return result;
        '''}
        )

m.function( f"R_aligned_to_vector",
    r'''Compute a rotation to map a given vector to [0,0,1]

SYNOPSIS

    # I have a plane that passes through a point p, and has a normal n. I
    # compute a transformation from the world to a coord system aligned to the
    # plane, with p at the origin. R_plane_world p + t_plane_world = 0:

    Rt_plane_world = np.zeros((4,3), dtype=float)
    Rt_plane_world[:3,:] = mrcal.R_aligned_to_vector(n)
    Rt_plane_world[ 3,:] = -mrcal.rotate_point_R(Rt_plane_world[:3,:],p)

This rotation is not unique: adding any rotation around v still maps v to
[0,0,1]. An arbitrary acceptable rotation is returned.

ARGUMENTS

- v: a numpy array of shape (3,). The vector that the computed rotation maps to
  [0,0,1]. Does not need to be normalized. Must be non-0

RETURNED VALUES

The rotation in a (3,3) array

    ''',
            args_input       = ('v',),
            prototype_input  = ( (3,), ),
            prototype_output = (3,3),

            Ccode_validate = r'''
            return CHECK_CONTIGUOUS_AND_SETERROR_ALL();''',

            Ccode_slice_eval = \
                {np.float64:
                 rf'''
    mrcal_R_aligned_to_vector((double*)data_slice__output,
                              (double*)data_slice__v);
    return true;
'''}
)

m.write()
