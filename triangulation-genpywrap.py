#!/usr/bin/python3

r'''Python-wrap the triangulation routines

'''

import sys
import os

import numpy as np
import numpysane as nps

import numpysane_pywrap as npsp


docstring_module = '''Internal triangulation routines

This is the written-in-C Python extension module that underlies the
triangulation routines. The user-facing functions are available in
mrcal.triangulation module in mrcal/triangulation.py

All functions are exported into the mrcal module. So you can call these via
mrcal._triangulation_npsp.fff() or mrcal.fff(). The latter is preferred.

'''

m = npsp.module( name      = "_triangulation_npsp",
                 docstring = docstring_module,
                 header    = r'''
#include "mrcal.h"
''')



# All the triangulation routines except Lindstrom have an identical structure.
# Lindstrom is slightly different: it takes LOCAL v1 instead of a cam0-coords v1


NAME = "_triangulate_{WHAT}"
DOCS = r"""Internal {LONGNAME} triangulation routine

This is the internals for mrcal.triangulate_{WHAT}(get_gradients = False). As a
user, please call THAT function, and see the docs for that function. The
differences:

- This is just the no-gradients function. The internal function that returns
  gradients is _triangulate_{WHAT}_withgrad

- This function is wrapped with numpysane_pywrap, so the arguments broadcast as
  expected

"""
DOCS_WITHGRAD = r"""Internal {LONGNAME} triangulation routine (with gradients)

This is the internals for mrcal.triangulate_{WHAT}(get_gradients = True). As a
user, please call THAT function, and see the docs for that function. The
differences:

- This is just the gradients-returning function. The internal function that
  skips those is _triangulate_{WHAT}

- This function is wrapped with numpysane_pywrap, so the arguments broadcast as
  expected

"""
BODY_SLICE =  r'''
                const mrcal_point3_t* v0  = (const mrcal_point3_t*)data_slice__v0;
                const mrcal_point3_t* v1  = (const mrcal_point3_t*)data_slice__v1;
                const mrcal_point3_t* t01 = (const mrcal_point3_t*)data_slice__t01;

                *(mrcal_point3_t*)data_slice__output =
                  mrcal_triangulate_{WHAT}(NULL, NULL, NULL,
                                           v0, v1, t01);
                return true;
'''
BODY_SLICE_WITHGRAD =  r'''
                const mrcal_point3_t* v0  = (const mrcal_point3_t*)data_slice__v0;
                const mrcal_point3_t* v1  = (const mrcal_point3_t*)data_slice__v1;
                const mrcal_point3_t* t01 = (const mrcal_point3_t*)data_slice__t01;

                mrcal_point3_t* dm_dv0  = (mrcal_point3_t*)data_slice__output1;
                mrcal_point3_t* dm_dv1  = (mrcal_point3_t*)data_slice__output2;
                mrcal_point3_t* dm_dt01 = (mrcal_point3_t*)data_slice__output3;

                *(mrcal_point3_t*)data_slice__output0 =
                  mrcal_triangulate_{WHAT}( dm_dv0, dm_dv1, dm_dt01,
                                            v0, v1, t01);
                return true;
'''
common_kwargs = dict( args_input       = ('v0', 'v1', 't01'),
                      prototype_input  = ((3,), (3,), (3,)),
                      prototype_output = (3,),
                      Ccode_validate = r'''
                      return CHECK_CONTIGUOUS_AND_SETERROR_ALL();''' )

common_kwargs_withgrad = dict( args_input       = ('v0', 'v1', 't01'),
                               prototype_input  = ((3,), (3,), (3,)),
                               prototype_output = ((3,), (3,3), (3,3), (3,3)),
                               Ccode_validate = r'''
                               return CHECK_CONTIGUOUS_AND_SETERROR_ALL();''')

for WHAT,LONGNAME in (('geometric',       'geometric'),
                      ('leecivera_l1',    'Lee-Civera L1'),
                      ('leecivera_linf',  'Lee-Civera L-infinity'),
                      ('leecivera_mid2',  'Lee-Civera Mid2'),
                      ('leecivera_wmid2', 'Lee-Civera wMid2')):

    m.function( NAME.format(WHAT     = WHAT),
                DOCS.format(WHAT     = WHAT,
                            LONGNAME = LONGNAME),
                Ccode_slice_eval = { (np.float64,np.float64,np.float64,
                                      np.float64):
                                     BODY_SLICE.format(WHAT = WHAT) },
                **common_kwargs
    )

    m.function( NAME.format(         WHAT     = WHAT) + "_withgrad",
                DOCS_WITHGRAD.format(WHAT     = WHAT,
                                     LONGNAME = LONGNAME),
                Ccode_slice_eval = { (np.float64,np.float64,np.float64,
                                      np.float64,np.float64,np.float64,np.float64):
                                     BODY_SLICE_WITHGRAD.format(WHAT = WHAT) },
                **common_kwargs_withgrad
    )



# Lindstrom's triangulation. Takes a local v1, so the arguments are a bit
# different
m.function( "_triangulate_lindstrom",
            f"""Internal lindstrom's triangulation routine

This is the internals for mrcal.triangulate_lindstrom(). As a user, please call
THAT function, and see the docs for that function. The differences:

- This is just the no-gradients function. The internal function that returns
  gradients is _triangulate_lindstrom_withgrad

- This function is wrapped with numpysane_pywrap, so the arguments broadcast as
  expected

""",

            args_input       = ('v0_local', 'v1_local', 'Rt01'),
            prototype_input  = ((3,), (3,), (4,3),),
            prototype_output = ((3,) ),

            Ccode_validate = r'''
            return CHECK_CONTIGUOUS_AND_SETERROR_ALL();''',

            Ccode_slice_eval = { (np.float64,np.float64,np.float64,
                                  np.float64):
                                 r'''
                const mrcal_point3_t* v0  = (const mrcal_point3_t*)data_slice__v0_local;
                const mrcal_point3_t* v1  = (const mrcal_point3_t*)data_slice__v1_local;
                const mrcal_point3_t* Rt01= (const mrcal_point3_t*)data_slice__Rt01;

                *(mrcal_point3_t*)data_slice__output =
                  mrcal_triangulate_lindstrom(NULL,NULL,NULL,
                                              v0, v1, Rt01);
                return true;
'''},
)

m.function( "_triangulate_lindstrom_withgrad",
            f"""Internal lindstrom's triangulation routine

This is the internals for mrcal.triangulate_lindstrom(). As a user, please call
THAT function, and see the docs for that function. The differences:

- This is just the gradient-returning function. The internal function that skips those
  is _triangulate_lindstrom

- This function is wrapped with numpysane_pywrap, so the arguments broadcast as
  expected

""",

            args_input       = ('v0_local', 'v1_local', 'Rt01'),
            prototype_input  = ((3,), (3,), (4,3),),
            prototype_output = ((3,), (3,3), (3,3), (3,4,3) ),

            Ccode_validate = r'''
            return CHECK_CONTIGUOUS_AND_SETERROR_ALL();''',

            Ccode_slice_eval = { (np.float64,np.float64,np.float64,
                                  np.float64,np.float64,np.float64,np.float64):
                  r'''
                const mrcal_point3_t* v0  = (const mrcal_point3_t*)data_slice__v0_local;
                const mrcal_point3_t* v1  = (const mrcal_point3_t*)data_slice__v1_local;
                const mrcal_point3_t* Rt01= (const mrcal_point3_t*)data_slice__Rt01;

                mrcal_point3_t* dm_dv0  = (mrcal_point3_t*)data_slice__output1;
                mrcal_point3_t* dm_dv1  = (mrcal_point3_t*)data_slice__output2;
                mrcal_point3_t* dm_dRt01= (mrcal_point3_t*)data_slice__output3;

                *(mrcal_point3_t*)data_slice__output0 =
                  mrcal_triangulate_lindstrom(dm_dv0, dm_dv1, dm_dRt01,
                                              v0, v1, Rt01);
                return true;
''' },
)

m.write()
