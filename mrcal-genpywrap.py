#!/usr/bin/python3

r'''Python-wrap the mrcal routines that require broadcasting

'''

import sys
import os

import numpy as np
import numpysane as nps

import numpysane_pywrap as npsp

m = npsp.module( name      = "_mrcal_broadcasted",
                 docstring = "Internal python wrappers for broadcasting functions",
                 header    = r'''
#include "mrcal.h"

static
bool validate_lensmodel(// out; valid if we returned true
                        lensmodel_t* lensmodel,

                        // in
                        const char* lensmodel_str,
                        int Nintrinsics_in_arg, bool is_project)
{
    if(lensmodel_str == NULL)
    {
        PyErr_Format(PyExc_RuntimeError,
                     "The 'lensmodel' argument is required");
        return false;
    }

    *lensmodel = mrcal_lensmodel_from_name(lensmodel_str);
    if( !mrcal_lensmodel_type_is_valid(lensmodel->type) )
    {
        PyErr_Format(PyExc_RuntimeError,
                     "Couldn't parse 'lensmodel' argument '%s'", lensmodel_str);
        return false;
    }

    if( !is_project && lensmodel->type == LENSMODEL_CAHVORE )
    {
        PyErr_Format(PyExc_RuntimeError,
                     "The internal _unproject() routine does not support CAHVORE. Use the Python mrcal.unproject() for that");
        return false;
    }

    int NlensParams = mrcal_getNlensParams(*lensmodel);
    if( NlensParams != Nintrinsics_in_arg )
    {
        PyErr_Format(PyExc_RuntimeError,
                     "Lens model '%s' has %d parameters, but the given array has %d",
                     lensmodel_str, NlensParams, Nintrinsics_in_arg);
        return false;
    }

    return true;
}
''')


# NOTE: these projection functions are not as fast as they SHOULD be. In fact
# using numpysane_pywrap() is currently causing a performance hit of about 10%.
# This should be improved by speeding up the main broadcasting loop in
# numpysane_pywrap. I recall it being more complex than it should be. This bit
# of python works to benchmark:
r"""
import numpy as np
import mrcal
import time
m = mrcal.cameramodel('test/data/cam.splined.cameramodel')
v = np.random.random((2000,3000,3))
v[..., 2] += 10.
t0 = time.time()
mapxy = mrcal.project( v, *m.intrinsics() )
print(time.time()-t0)
"""
# To test the broadcast-using-the-mrcal-loop, apply this patch:
r"""
diff --git a/mrcal-genpywrap.py b/mrcal-genpywrap.py
index 666f48e..2a4edff 100644
--- a/mrcal-genpywrap.py
+++ b/mrcal-genpywrap.py
@@ -89,7 +93,7 @@
 
             args_input       = ('points', 'intrinsics'),
-            prototype_input  = ((3,), ('Nintrinsics',)),
-            prototype_output = (2,),
+            prototype_input  = (('N',3,), ('Nintrinsics',)),
+            prototype_output = ('N',2,),
 
             extra_args = (("const char*", "lensmodel", "NULL", "s"),),
 
@@ -113,7 +117,7 @@ _project_withgrad() in mrcal-genpywrap.py. Please keep them in sync
             Ccode_slice_eval = \
                 {np.float64:
                  r'''
-                 const int N = 1;
+                 const int N = dims_slice__points[0];
 
                  if(cookie->lensmodel.type == LENSMODEL_CAHVORE)
                      return _mrcal_project_internal_cahvore(
"""
# I see 0.9 sec with the code as is, and 0.8 sec with the patch. Or before I
# moved on to this whole npsp thing in 482728c. As it stands, the patch is not
# committable. It assumes contiguous memory, and it'll produce incorrect output
# shapes if we try to broadcast on intrinsics_data. These are all fixable, and
# I'm moving on for now
m.function( "_project",
            """Internal point-projection routine

This is the internals for mrcal.project(). As a user, please call THAT function,
and see the docs for that function. The differences:

- This is just the no-gradients function. The internal function that reports the
  gradients also is _project_withgrad

- This function is wrapped with numpysane_pywrap, so the points and the
  intrinsics broadcast as expected

- To make the broadcasting work, the argument order in this function is
  different. numpysane_pywrap broadcasts the leading arguments, so this function
  takes the lensmodel (the one argument that does not broadcast) last

- To speed things up, this function doesn't call the C mrcal_project(), but uses
  the _mrcal_project_internal...() functions instead. That allows as much as
  possible of the outer init stuff to be moved outside of the slice computation
  loop

The outer logic (outside the loop-over-N-points) is duplicated in
mrcal_project() and in the python wrapper definition in _project() and
_project_withgrad() in mrcal-genpywrap.py. Please keep them in sync

""",

            args_input       = ('points', 'intrinsics'),
            prototype_input  = ((3,), ('Nintrinsics',)),
            prototype_output = (2,),

            extra_args = (("const char*", "lensmodel", "NULL", "s"),),

            Ccode_cookie_struct = '''
              lensmodel_t                    lensmodel;
              int                            Nintrinsics;
              mrcal_projection_precomputed_t precomputed;
            ''',

            Ccode_validate = r'''
              if( !( validate_lensmodel(&cookie->lensmodel,
                                        lensmodel, dims_slice__intrinsics[0], true) &&
                     CHECK_CONTIGUOUS_AND_SETERROR_ALL()))
                  return false;

              cookie->Nintrinsics = mrcal_getNlensParams(cookie->lensmodel);
              _mrcal_precompute_lensmodel_data(&cookie->precomputed, cookie->lensmodel);
              return true;
''',

            Ccode_slice_eval = \
                {np.float64:
                 r'''
                 const int N = 1;

                 if(cookie->lensmodel.type == LENSMODEL_CAHVORE)
                     return _mrcal_project_internal_cahvore(
                                (point2_t*)data_slice__output,
                                (const point3_t*)data_slice__points,
                                N,
                                (const double*)data_slice__intrinsics);

                 if(LENSMODEL_IS_OPENCV(cookie->lensmodel.type) ||
                    cookie->lensmodel.type == LENSMODEL_PINHOLE)
                 {
                     _mrcal_project_internal_opencv(
                                (point2_t*)data_slice__output,
                                NULL, NULL,
                                (const point3_t*)data_slice__points,
                                N,
                                (const double*)data_slice__intrinsics,
                                cookie->Nintrinsics);
                     return true;
                 }

                 return
                     _mrcal_project_internal((point2_t*)data_slice__output,
                                             NULL, NULL,
                                             (const point3_t*)data_slice__points,
                                             N,
                                             cookie->lensmodel,
                                             // core, distortions concatenated
                                             (const double*)data_slice__intrinsics,
                                             cookie->Nintrinsics, &cookie->precomputed);
'''},
)

m.function( "_project_withgrad",
            """Internal point-projection routine

This is the internals for mrcal.project(). As a user, please call THAT function,
and see the docs for that function. The differences:

- This is just the gradients-returning function. The internal function that
  skips those is _project

- This function is wrapped with numpysane_pywrap, so the points and the
  intrinsics broadcast as expected

- To make the broadcasting work, the argument order in this function is
  different. numpysane_pywrap broadcasts the leading arguments, so this function
  takes the lensmodel (the one argument that does not broadcast) last

- To speed things up, this function doesn't call the C mrcal_project(), but uses
  the _mrcal_project_internal...() functions instead. That allows as much as
  possible of the outer init stuff to be moved outside of the slice computation
  loop

The outer logic (outside the loop-over-N-points) is duplicated in
mrcal_project() and in the python wrapper definition in _project() and
_project_withgrad() in mrcal-genpywrap.py. Please keep them in sync

""",

            args_input       = ('points', 'intrinsics'),
            prototype_input  = ((3,), ('Nintrinsics',)),
            prototype_output = ((2,), (2,3), (2,'Nintrinsics')),

            extra_args = (("const char*", "lensmodel", "NULL", "s"),),

            Ccode_cookie_struct = '''
              lensmodel_t                    lensmodel;
              int                            Nintrinsics;
              mrcal_projection_precomputed_t precomputed;
            ''',

            Ccode_validate = r'''
              if( !( validate_lensmodel(&cookie->lensmodel,
                                        lensmodel, dims_slice__intrinsics[0], true) &&
                     CHECK_CONTIGUOUS_AND_SETERROR_ALL()))
                  return false;

              if(cookie->lensmodel.type == LENSMODEL_CAHVORE)
              {
                  PyErr_Format(PyExc_RuntimeError,
                               "_project(LENSMODEL_CAHVORE) is not yet implemented if we're asking for gradients");
                  return false;
              }
              cookie->Nintrinsics = mrcal_getNlensParams(cookie->lensmodel);
              _mrcal_precompute_lensmodel_data(&cookie->precomputed, cookie->lensmodel);
              return true;
''',

            Ccode_slice_eval = \
                {np.float64:
                 r'''
                 const int N = 1;

                 // Some models have sparse gradients, but I'm returning a dense array here.
                 // So I init everything at 0
                 memset(data_slice__output2, 0, N*2*cookie->Nintrinsics*sizeof(double));

                 return
                     _mrcal_project_internal((point2_t*)data_slice__output0,
                                             (point3_t*)data_slice__output1,
                                             (double*)  data_slice__output2,
                                             (const point3_t*)data_slice__points,
                                             N,
                                             cookie->lensmodel,
                                             // core, distortions concatenated
                                             (const double*)data_slice__intrinsics,
                                             cookie->Nintrinsics, &cookie->precomputed);
'''},
)

m.function( "_unproject",
            """Internal point-unprojection routine

This is the internals for mrcal.unproject(). As a user, please call THAT
function, and see the docs for that function. The differences:

- This function is wrapped with numpysane_pywrap, so the points and the
  intrinsics broadcast as expected

- To make the broadcasting work, the argument order in this function is
  different. numpysane_pywrap broadcasts the leading arguments, so this function
  takes the lensmodel (the one argument that does not broadcast) last

- This function does NOT support CAHVORE

- To speed things up, this function doesn't call the C mrcal_unproject(), but
  uses the _mrcal_unproject_internal...() functions instead. That allows as much
  as possible of the outer init stuff to be moved outside of the slice
  computation loop

The outer logic (outside the loop-over-N-points) is duplicated in
mrcal_unproject() and in the python wrapper definition in _unproject()
mrcal-genpywrap.py. Please keep them in sync """,

            args_input       = ('points', 'intrinsics'),
            prototype_input  = ((2,), ('Nintrinsics',)),
            prototype_output = (3,),

            extra_args = (("const char*", "lensmodel", "NULL", "s"),),

            Ccode_cookie_struct = '''
              lensmodel_t lensmodel;
              mrcal_projection_precomputed_t precomputed;
            ''',

            Ccode_validate = r'''
              if( !( validate_lensmodel(&cookie->lensmodel,
                                        lensmodel, dims_slice__intrinsics[0], false) &&
                     CHECK_CONTIGUOUS_AND_SETERROR_ALL()))
                  return false;

              if(cookie->lensmodel.type == LENSMODEL_CAHVORE)
              {
                  PyErr_Format(PyExc_RuntimeError,
                               "_unproject(LENSMODEL_CAHVORE) is not yet implemented: we need gradients. The python mrcal.unproject() should work; slowly.");
                  return false;
              }
              _mrcal_precompute_lensmodel_data(&cookie->precomputed, cookie->lensmodel);
              return true;
''',

            Ccode_slice_eval = \
                {np.float64:
                 r'''
                 const int N = 1;
                 if( cookie->lensmodel.type == LENSMODEL_PINHOLE ||
                     cookie->lensmodel.type == LENSMODEL_STEREOGRAPHIC )
                     mrcal_unproject((point3_t*)data_slice__output,
                                     (const point2_t*)data_slice__points,
                                     N,
                                     cookie->lensmodel,
                                     // core, distortions concatenated
                                     (const double*)data_slice__intrinsics);
                 return
                     _mrcal_unproject_internal((point3_t*)data_slice__output,
                                               (const point2_t*)data_slice__points,
                                               N,
                                               cookie->lensmodel,
                                               // core, distortions concatenated
                                               (const double*)data_slice__intrinsics,
                                               &cookie->precomputed);
'''},
)

m.function( "_A_Jt_J_At",
            """Computes matmult(A,Jt,J,At) for a sparse J

This is used in the internals of projection_uncertainty().

A has shape (2,Nstate)

J has shape (Nmeasurements,Nstate). J is large and sparse

We use the Nleading_rows_J leading rows of J. This integer is passed-in as an
argument.

matmult(A, Jt, J, At) has shape (2,2)

The input matrices are large, but the result is very small. I can't see a way to
do this efficiently in pure Python, so I'm writing this.

J is sparse, stored by row. This is the scipy.sparse.csr_matrix representation,
and is also how CHOLMOD stores Jt (CHOLMOD stores by column, so the same data
looks like Jt to CHOLMOD). The sparse J is given here as the p,i,x arrays from
CHOLMOD, equivalent to the indptr,indices,data members of
scipy.sparse.csr_matrix respectively.
 """,

            args_input       = ('A', 'Jp', 'Ji', 'Jx'),
            prototype_input  = ((2,'Nstate'), ('Np',), ('Nix',), ('Nix',)),
            prototype_output = (2,2),

            extra_args = (("int", "Nleading_rows_J", "-1", "i"),),

            Ccode_validate = r'''
            if(*Nleading_rows_J <= 0)
            {
                PyErr_Format(PyExc_RuntimeError,
                             "Nleading_rows_J must be passed, and must be > 0");
                return false;
            }
            return CHECK_CONTIGUOUS_AND_SETERROR_ALL();''',

            Ccode_slice_eval = \
                { (np.float64, np.int32, np.int32, np.float64, np.float64):
                 r'''

                 // I'm computing A Jt J At = sum(outer(ja,ja)) where ja is each
                 // row of matmult(J,At). Rows of matmult(J,At) are
                 // matmult(jt,At) where jt are rows of J. So the logic is:
                 //
                 //   For each row jt of J:
                 //     jta = matmult(jt,At); // jta has shape (2,)
                 //     accumulate( outer(jta,jta) )


                 int32_t Nstate = dims_slice__A[1];

                 const double* A   = (const double* )data_slice__A;
                 const int32_t* Jp = (const int32_t*)data_slice__Jp;
                 const int32_t* Ji = (const int32_t*)data_slice__Ji;
                 const double*  Jx = (const double* )data_slice__Jx;
                 double* out       = (      double* )data_slice__output;

                 // zero out the accumulator
                 memset(((double*)data_slice__output), 0, 2*2*sizeof(double));

                 for(int irow=0; irow<*Nleading_rows_J; irow++)
                 {
                     double jta[2] = {};

                     for(int32_t i = Jp[irow]; i < Jp[irow+1]; i++)
                     {
                         int32_t icol = Ji[i];
                         double x     = Jx[i];

                         jta[0] += A[icol + 0*Nstate] * x;
                         jta[1] += A[icol + 1*Nstate] * x;

                     }
                     out[0] += jta[0]*jta[0];
                     out[1] += jta[1]*jta[0];
                     out[2] += jta[1]*jta[0];
                     out[3] += jta[1]*jta[1];
                 }
                 return true;
'''},
)

m.write()
