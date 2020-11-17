#!/usr/bin/python3

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


m = npsp.module( name      = "_poseutils",
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
    mrcal_identity_R_noncontiguous( (double*)data_slice__output,
                                    strides_slice__output[0],
                                    strides_slice__output[1] );
    return true;
'''})

m.function( "identity_r",
            """Return an identity Rodrigues vector

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
    mrcal_identity_r_noncontiguous( (double*)data_slice__output,
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
    mrcal_identity_Rt_noncontiguous( (double*)data_slice__output,
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
    mrcal_identity_rt_noncontiguous( (double*)data_slice__output,
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

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_rotate_point_R_noncontiguous( (double*)data_slice__output,
                                        strides_slice__output[0],
                                        NULL,0,0,0,
                                        NULL,0,0,
                                        (const double*)data_slice__R,
                                        strides_slice__R[0],
                                        strides_slice__R[1],
                                        (const double*)data_slice__x,
                                        strides_slice__x[0] );
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

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_rotate_point_R_noncontiguous( (double*)data_slice__output0,
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
                                        strides_slice__x[0] );
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

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_rotate_point_r_noncontiguous( (double*)data_slice__output,
                                        strides_slice__output[0],
                                        NULL,0,0,
                                        NULL,0,0,
                                        (const double*)data_slice__r,
                                        strides_slice__r[0],
                                        (const double*)data_slice__x,
                                        strides_slice__x[0]);
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

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_rotate_point_r_noncontiguous( (double*)data_slice__output0,
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
                                        strides_slice__x[0]);
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

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_transform_point_Rt_noncontiguous( (double*)data_slice__output,
                                            strides_slice__output[0],
                                            NULL,0,0,0,
                                            NULL,0,0,
                                            (const double*)data_slice__Rt,
                                            strides_slice__Rt[0],
                                            strides_slice__Rt[1],
                                            (const double*)data_slice__x,
                                            strides_slice__x[0] );
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

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_transform_point_Rt_noncontiguous( (double*)data_slice__output0,
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
                                            strides_slice__x[0] );
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

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_transform_point_rt_noncontiguous( (double*)data_slice__output,
                                            strides_slice__output[0],
                                            NULL,0,0,
                                            NULL,0,0,
                                            (const double*)data_slice__rt,
                                            strides_slice__rt[0],
                                            (const double*)data_slice__x,
                                            strides_slice__x[0] );
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

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_transform_point_rt_noncontiguous( (double*)data_slice__output0,
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
                                            strides_slice__x[0] );
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
    mrcal_invert_Rt_noncontiguous( (double*)data_slice__output,
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
    mrcal_invert_rt_noncontiguous( (double*)data_slice__output,
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
    mrcal_invert_rt_noncontiguous( (double*)data_slice__output0,
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
    mrcal_compose_Rt_noncontiguous( (double*)data_slice__output,
                                    strides_slice__output[0], strides_slice__output[1],
                                    (const double*)data_slice__Rt0,
                                    strides_slice__Rt0[0], strides_slice__Rt0[1],
                                    (const double*)data_slice__Rt1,
                                    strides_slice__Rt1[0], strides_slice__Rt1[1] );
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
    mrcal_compose_rt_noncontiguous( (double*)data_slice__output,
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
    mrcal_compose_rt_noncontiguous( (double*)data_slice__output0,
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

m.write()
