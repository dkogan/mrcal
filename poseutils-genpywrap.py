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

m.function( "_rotate_point_R",
            "Rotate a point using a rotation matrix",

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
            "Rotate a point using a rotation matrix; Return (u=R(x),du/dR,du/dx)",

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
            "Rotate a point using a Rodrigues vector",

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
            "Rotate a point using a Rodrigues vector; Return (u=r(x),du/dr,du/dx)",

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
            "Transform a point using an Rt transformation",

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
            "Transform a point using an Rt transformation; Return (u=Rt(x),du/dR,du/dt,du/dx)",

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
            "Transform a point using an rt transformation",

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
            "Transform a point using an rt transformation. Return (u=rt(x),du/dr,du/dt,du/dx)",

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
