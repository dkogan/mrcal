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
            "Returns an identity rotation matrix",

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
            "Returns an identity rotation Rodrigues vector",

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
            "Returns an identity Rt transformation",

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
            "Returns an identity rt transformation",

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

m.function( "rotate_point_R",
            "Rotate a point using a rotation matrix",

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

m.function( "rotate_point_r",
            "Rotate a point using a Rodrigues vector",

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

m.function( "transform_point_Rt",
            "Transform a point using an Rt transformation",

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

m.function( "transform_point_rt",
            "Transform a point using an rt transformation",

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

m.function( "r_from_R",
            "Compute a Rodrigues vector from a rotation matrix",

            args_input       = ('R',),
            prototype_input  = ((3,3),),
            prototype_output = ((3,), (3,3,3)),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_r_from_R( (double*)data_slice__output0,
                    (double*)data_slice__output1,
                    (const double*)data_slice__R );
    return true;
'''},
            Ccode_validate = r'''
            return \
              CHECK_CONTIGUOUS_AND_SETERROR_ALL();
'''
)

m.function( "R_from_r",
            "Compute a rotation matrix from a Rodrigues vector",

            args_input       = ('r',),
            prototype_input  = ((3,),),
            prototype_output = ((3,3), (3,3,3)),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_R_from_r( (double*)data_slice__output0,
                    (double*)data_slice__output1,
                    (const double*)data_slice__r );
    return true;
'''},
            Ccode_validate = r'''
            return \
              CHECK_CONTIGUOUS_AND_SETERROR_ALL();
'''
)

m.function( "rt_from_Rt",
            "Compute an rt transformation from a Rt transformation",

            args_input       = ('Rt',),
            prototype_input  = ((4,3),),
            prototype_output = (6,),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_rt_from_Rt( (double*)data_slice__output,
                      (const double*)data_slice__Rt );
    return true;
'''},
            Ccode_validate = r'''
            return \
              CHECK_CONTIGUOUS_AND_SETERROR_ALL();
'''
)

m.function( "Rt_from_rt",
            "Compute an Rt transformation from a rt transformation",

            args_input       = ('rt',),
            prototype_input  = ((6,),),
            prototype_output = (4,3),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_Rt_from_rt( (double*)data_slice__output,
                      (const double*)data_slice__rt );
    return true;
'''},
            Ccode_validate = r'''
            return \
              CHECK_CONTIGUOUS_AND_SETERROR_ALL();
'''
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

m.function( "compose_Rt",
            "compose 2 Rt transformations",

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

m.function( "compose_rt",
            "compose 2 rt transformations",

            args_input       = ('rt0', 'rt1'),
            prototype_input  = ((6,), (6,)),
            prototype_output = (6,),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    mrcal_compose_rt( (double*)data_slice__output,
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
