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


m.function( "_triangulate_geometric",
            r"""Internal geometric triangulation routine

This is the internals for mrcal.triangulate_geometric(). As a user, please call
THAT function, and see the docs for that function. The differences:

- This is just the no-gradients function. The internal function that returns
  gradients is _triangulate_geometric_withgrad

- This function is wrapped with numpysane_pywrap, so the arguments broadcast as
  expected

""",

            args_input       = ('v0', 'v1', 't01'),
            prototype_input  = ((3,), (3,), (3,)),
            prototype_output = (3,),

            Ccode_validate = r'''
            return CHECK_CONTIGUOUS_AND_SETERROR_ALL();''',

            Ccode_slice_eval = \
                { (np.float64,np.float64,np.float64,
                   np.float64): r'''
                const mrcal_point3_t* v0  = (const mrcal_point3_t*)data_slice__v0;
                const mrcal_point3_t* v1  = (const mrcal_point3_t*)data_slice__v1;
                const mrcal_point3_t* t01 = (const mrcal_point3_t*)data_slice__t01;

                *(mrcal_point3_t*)data_slice__output =
                  mrcal_triangulate_geometric(NULL, NULL, NULL,
                                        v0, v1, t01);
                return true;
''' },
)

m.function( "_triangulate_geometric_withgrad",
            r"""Internal geometric triangulation routine

This is the internals for mrcal.triangulate_geometric(). As a user, please call
THAT function, and see the docs for that function. The differences:

- This is just the gradients-returning function. The internal function that
  skips those is _triangulate_geometric

- This function is wrapped with numpysane_pywrap, so the arguments broadcast as
  expected

""",

            args_input       = ('v0', 'v1', 't01'),
            prototype_input  = ((3,), (3,), (3,)),
            prototype_output = ((3,), (3,3), (3,3), (3,3)),

            Ccode_validate = r'''
            return CHECK_CONTIGUOUS_AND_SETERROR_ALL();''',

            Ccode_slice_eval = \
                { (np.float64,np.float64,np.float64,
                   np.float64,np.float64,np.float64,np.float64): r'''
                const mrcal_point3_t* v0  = (const mrcal_point3_t*)data_slice__v0;
                const mrcal_point3_t* v1  = (const mrcal_point3_t*)data_slice__v1;
                const mrcal_point3_t* t01 = (const mrcal_point3_t*)data_slice__t01;

                mrcal_point3_t* dm_dv0  = (mrcal_point3_t*)data_slice__output1;
                mrcal_point3_t* dm_dv1  = (mrcal_point3_t*)data_slice__output2;
                mrcal_point3_t* dm_dt01 = (mrcal_point3_t*)data_slice__output3;

                *(mrcal_point3_t*)data_slice__output0 =
                  mrcal_triangulate_geometric(dm_dv0, dm_dv1, dm_dt01,
                                        v0, v1, t01);
                return true;
''' },
)

m.function( "_triangulate_lindstrom",
            f"""Internal Lindstrom's triangulation routine

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

            Ccode_slice_eval = \
                { (np.float64,np.float64,np.float64,
                   np.float64): r'''
                const mrcal_point3_t* v0  = (const mrcal_point3_t*)data_slice__v0_local;
                const mrcal_point3_t* v1  = (const mrcal_point3_t*)data_slice__v1_local;
                const mrcal_point3_t* Rt01= (const mrcal_point3_t*)data_slice__Rt01;

                *(mrcal_point3_t*)data_slice__output =
                  mrcal_triangulate_lindstrom(NULL,NULL,NULL,
                                        v0, v1, Rt01);
                return true;
''' },
)

m.function( "_triangulate_lindstrom_withgrad",
            f"""Internal Lindstrom's triangulation routine

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

            Ccode_slice_eval = \
                { (np.float64,np.float64,np.float64,
                   np.float64,np.float64,np.float64,np.float64): r'''
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

m.function( "_triangulate_leecivera_l1",
            r"""Internal Lee-Civera L1 triangulation routine

This is the internals for mrcal.triangulate_leecivera_l1(). As a user, please call
THAT function, and see the docs for that function. The differences:

- This is just the no-gradients function. The internal function that returns
  gradients is _triangulate_leecivera_l1_withgrad

- This function is wrapped with numpysane_pywrap, so the arguments broadcast as
  expected

""",

            args_input       = ('v0', 'v1', 't01'),
            prototype_input  = ((3,), (3,), (3,)),
            prototype_output = (3,),

            Ccode_validate = r'''
            return CHECK_CONTIGUOUS_AND_SETERROR_ALL();''',

            Ccode_slice_eval = \
                { (np.float64,np.float64,np.float64,
                   np.float64): r'''
                const mrcal_point3_t* v0  = (const mrcal_point3_t*)data_slice__v0;
                const mrcal_point3_t* v1  = (const mrcal_point3_t*)data_slice__v1;
                const mrcal_point3_t* t01 = (const mrcal_point3_t*)data_slice__t01;

                *(mrcal_point3_t*)data_slice__output =
                  mrcal_triangulate_leecivera_l1( NULL, NULL, NULL,
                                            v0, v1, t01);
                return true;
''' },
)

m.function( "_triangulate_leecivera_l1_withgrad",
            r"""Internal Lee-Civera L1 triangulation routine

This is the internals for mrcal.triangulate_leecivera_l1(). As a user, please call
THAT function, and see the docs for that function. The differences:

- This is just the gradients-returning function. The internal function that
  skips those is _triangulate_leecivera_l1

- This function is wrapped with numpysane_pywrap, so the arguments broadcast as
  expected

""",

            args_input       = ('v0', 'v1', 't01'),
            prototype_input  = ((3,), (3,), (3,)),
            prototype_output = ((3,), (3,3), (3,3), (3,3)),

            Ccode_validate = r'''
            return CHECK_CONTIGUOUS_AND_SETERROR_ALL();''',

            Ccode_slice_eval = \
                { (np.float64,np.float64,np.float64,
                   np.float64,np.float64,np.float64,np.float64): r'''
                const mrcal_point3_t* v0  = (const mrcal_point3_t*)data_slice__v0;
                const mrcal_point3_t* v1  = (const mrcal_point3_t*)data_slice__v1;
                const mrcal_point3_t* t01 = (const mrcal_point3_t*)data_slice__t01;

                mrcal_point3_t* dm_dv0  = (mrcal_point3_t*)data_slice__output1;
                mrcal_point3_t* dm_dv1  = (mrcal_point3_t*)data_slice__output2;
                mrcal_point3_t* dm_dt01 = (mrcal_point3_t*)data_slice__output3;

                *(mrcal_point3_t*)data_slice__output0 =
                  mrcal_triangulate_leecivera_l1( dm_dv0, dm_dv1, dm_dt01,
                                            v0, v1, t01);
                return true;
''' },
)

m.function( "_triangulate_leecivera_linf",
            r"""Internal Lee-Civera L-infinity triangulation routine

This is the internals for mrcal.triangulate_leecivera_linf(). As a user, please call
THAT function, and see the docs for that function. The differences:

- This is just the no-gradients function. The internal function that returns
  gradients is _triangulate_leecivera_linf_withgrad

- This function is wrapped with numpysane_pywrap, so the arguments broadcast as
  expected

""",

            args_input       = ('v0', 'v1', 't01'),
            prototype_input  = ((3,), (3,), (3,)),
            prototype_output = (3,),

            Ccode_validate = r'''
            return CHECK_CONTIGUOUS_AND_SETERROR_ALL();''',

            Ccode_slice_eval = \
                { (np.float64,np.float64,np.float64,
                   np.float64): r'''
                const mrcal_point3_t* v0  = (const mrcal_point3_t*)data_slice__v0;
                const mrcal_point3_t* v1  = (const mrcal_point3_t*)data_slice__v1;
                const mrcal_point3_t* t01 = (const mrcal_point3_t*)data_slice__t01;

                *(mrcal_point3_t*)data_slice__output =
                  mrcal_triangulate_leecivera_linf( NULL, NULL, NULL,
                                              v0, v1, t01);
                return true;
''' },
)

m.function( "_triangulate_leecivera_linf_withgrad",
            r"""Internal Lee-Civera L-infinity triangulation routine

This is the internals for mrcal.triangulate_leecivera_linf(). As a user, please call
THAT function, and see the docs for that function. The differences:

- This is just the gradients-returning function. The internal function that
  skips those is _triangulate_leecivera_linf

- This function is wrapped with numpysane_pywrap, so the arguments broadcast as
  expected

""",

            args_input       = ('v0', 'v1', 't01'),
            prototype_input  = ((3,), (3,), (3,)),
            prototype_output = ((3,), (3,3), (3,3), (3,3)),

            Ccode_validate = r'''
            return CHECK_CONTIGUOUS_AND_SETERROR_ALL();''',

            Ccode_slice_eval = \
                { (np.float64,np.float64,np.float64,
                   np.float64,np.float64,np.float64,np.float64): r'''
                const mrcal_point3_t* v0  = (const mrcal_point3_t*)data_slice__v0;
                const mrcal_point3_t* v1  = (const mrcal_point3_t*)data_slice__v1;
                const mrcal_point3_t* t01 = (const mrcal_point3_t*)data_slice__t01;

                mrcal_point3_t* dm_dv0  = (mrcal_point3_t*)data_slice__output1;
                mrcal_point3_t* dm_dv1  = (mrcal_point3_t*)data_slice__output2;
                mrcal_point3_t* dm_dt01 = (mrcal_point3_t*)data_slice__output3;

                *(mrcal_point3_t*)data_slice__output0 =
                  mrcal_triangulate_leecivera_linf( dm_dv0, dm_dv1, dm_dt01,
                                              v0, v1, t01);
                return true;
''' },
)

m.function( "_triangulate_leecivera_mid2",
            r"""Internal Lee-Civera Mid2 triangulation routine

This is the internals for mrcal.triangulate_leecivera_mid2(). As a user, please call
THAT function, and see the docs for that function. The differences:

- This is just the no-gradients function. The internal function that returns
  gradients is _triangulate_leecivera_mid2_withgrad

- This function is wrapped with numpysane_pywrap, so the arguments broadcast as
  expected

""",

            args_input       = ('v0', 'v1', 't01'),
            prototype_input  = ((3,), (3,), (3,)),
            prototype_output = (3,),

            Ccode_validate = r'''
            return CHECK_CONTIGUOUS_AND_SETERROR_ALL();''',

            Ccode_slice_eval = \
                { (np.float64,np.float64,np.float64,
                   np.float64): r'''
                const mrcal_point3_t* v0  = (const mrcal_point3_t*)data_slice__v0;
                const mrcal_point3_t* v1  = (const mrcal_point3_t*)data_slice__v1;
                const mrcal_point3_t* t01 = (const mrcal_point3_t*)data_slice__t01;

                *(mrcal_point3_t*)data_slice__output =
                  mrcal_triangulate_leecivera_mid2( NULL, NULL, NULL,
                                              v0, v1, t01);
                return true;
''' },
)

m.function( "_triangulate_leecivera_mid2_withgrad",
            r"""Internal Lee-Civera Mid2 triangulation routine

This is the internals for mrcal.triangulate_leecivera_mid2(). As a user, please call
THAT function, and see the docs for that function. The differences:

- This is just the gradients-returning function. The internal function that
  skips those is _triangulate_leecivera_mid2

- This function is wrapped with numpysane_pywrap, so the arguments broadcast as
  expected

""",

            args_input       = ('v0', 'v1', 't01'),
            prototype_input  = ((3,), (3,), (3,)),
            prototype_output = ((3,), (3,3), (3,3), (3,3)),

            Ccode_validate = r'''
            return CHECK_CONTIGUOUS_AND_SETERROR_ALL();''',

            Ccode_slice_eval = \
                { (np.float64,np.float64,np.float64,
                   np.float64,np.float64,np.float64,np.float64): r'''
                const mrcal_point3_t* v0  = (const mrcal_point3_t*)data_slice__v0;
                const mrcal_point3_t* v1  = (const mrcal_point3_t*)data_slice__v1;
                const mrcal_point3_t* t01 = (const mrcal_point3_t*)data_slice__t01;

                mrcal_point3_t* dm_dv0  = (mrcal_point3_t*)data_slice__output1;
                mrcal_point3_t* dm_dv1  = (mrcal_point3_t*)data_slice__output2;
                mrcal_point3_t* dm_dt01 = (mrcal_point3_t*)data_slice__output3;

                *(mrcal_point3_t*)data_slice__output0 =
                  mrcal_triangulate_leecivera_mid2( dm_dv0, dm_dv1, dm_dt01,
                                              v0, v1, t01);
                return true;
''' },
)

m.function( "_triangulate_leecivera_wmid2",
            r"""Internal Lee-Civera wMid2 triangulation routine

This is the internals for mrcal.triangulate_leecivera_wmid2(). As a user, please call
THAT function, and see the docs for that function. The differences:

- This is just the no-gradients function. The internal function that returns
  gradients is _triangulate_leecivera_wmid2_withgrad

- This function is wrapped with numpysane_pywrap, so the arguments broadcast as
  expected

""",

            args_input       = ('v0', 'v1', 't01'),
            prototype_input  = ((3,), (3,), (3,)),
            prototype_output = (3,),

            Ccode_validate = r'''
            return CHECK_CONTIGUOUS_AND_SETERROR_ALL();''',

            Ccode_slice_eval = \
                { (np.float64,np.float64,np.float64,
                   np.float64): r'''
                const mrcal_point3_t* v0  = (const mrcal_point3_t*)data_slice__v0;
                const mrcal_point3_t* v1  = (const mrcal_point3_t*)data_slice__v1;
                const mrcal_point3_t* t01 = (const mrcal_point3_t*)data_slice__t01;

                *(mrcal_point3_t*)data_slice__output =
                  mrcal_triangulate_leecivera_wmid2( NULL, NULL, NULL,
                                              v0, v1, t01);
                return true;
''' },
)

m.function( "_triangulate_leecivera_wmid2_withgrad",
            r"""Internal Lee-Civera wMid2 triangulation routine

This is the internals for mrcal.triangulate_leecivera_wmid2(). As a user, please call
THAT function, and see the docs for that function. The differences:

- This is just the gradients-returning function. The internal function that
  skips those is _triangulate_leecivera_wmid2

- This function is wrapped with numpysane_pywrap, so the arguments broadcast as
  expected

""",

            args_input       = ('v0', 'v1', 't01'),
            prototype_input  = ((3,), (3,), (3,)),
            prototype_output = ((3,), (3,3), (3,3), (3,3)),

            Ccode_validate = r'''
            return CHECK_CONTIGUOUS_AND_SETERROR_ALL();''',

            Ccode_slice_eval = \
                { (np.float64,np.float64,np.float64,
                   np.float64,np.float64,np.float64,np.float64): r'''
                const mrcal_point3_t* v0  = (const mrcal_point3_t*)data_slice__v0;
                const mrcal_point3_t* v1  = (const mrcal_point3_t*)data_slice__v1;
                const mrcal_point3_t* t01 = (const mrcal_point3_t*)data_slice__t01;

                mrcal_point3_t* dm_dv0  = (mrcal_point3_t*)data_slice__output1;
                mrcal_point3_t* dm_dv1  = (mrcal_point3_t*)data_slice__output2;
                mrcal_point3_t* dm_dt01 = (mrcal_point3_t*)data_slice__output3;

                *(mrcal_point3_t*)data_slice__output0 =
                  mrcal_triangulate_leecivera_wmid2( dm_dv0, dm_dv1, dm_dt01,
                                              v0, v1, t01);
                return true;
''' },
)

m.write()
