#!/usr/bin/python3

r'''Python-wrap the mrcal routines that can be done with numpysane_pywrap

'''

import sys
import os

import numpy as np
import numpysane as nps

import numpysane_pywrap as npsp


docstring_module = '''Low-level routines for core mrcal operations

This is the written-in-C Python extension module that underlies the core
(un)project routines, and several low-level operations. Most of the functions in
this module (those prefixed with "_") are not meant to be called directly, but
have Python wrappers that should be used instead.

All functions are exported into the mrcal module. So you can call these via
mrcal._mrcal_npsp.fff() or mrcal.fff(). The latter is preferred.

'''

m = npsp.module( name      = "_mrcal_npsp",
                 docstring = docstring_module,
                 header    = r'''
#include "mrcal.h"

static
bool validate_lensmodel_un_project(// out; valid if we returned true
                        mrcal_lensmodel_t* lensmodel,

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

    mrcal_lensmodel_from_name(lensmodel, lensmodel_str);
    if( !mrcal_lensmodel_type_is_valid(lensmodel->type) )
    {
        PyErr_Format(PyExc_RuntimeError,
                     "Couldn't parse 'lensmodel' argument '%s'", lensmodel_str);
        return false;
    }

    mrcal_lensmodel_metadata_t meta = mrcal_lensmodel_metadata(lensmodel);
    if( !is_project && !meta.has_gradients )
    {
        PyErr_Format(PyExc_RuntimeError,
                     "The internal _unproject() routine requires lens models that have gradients implemented. Use the Python mrcal.unproject() as a workaround");
        return false;
    }

    int NlensParams = mrcal_lensmodel_num_params(lensmodel);
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
m = mrcal.cameramodel('test/data/cam0.splined.cameramodel')
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
 
                  if(cookie->lensmodel.type == MRCAL_LENSMODEL_CAHVORE)
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
              mrcal_lensmodel_t              lensmodel;
              int                            Nintrinsics;
              mrcal_projection_precomputed_t precomputed;
            ''',

            Ccode_validate = r'''
              if( !( validate_lensmodel_un_project(&cookie->lensmodel,
                                        lensmodel, dims_slice__intrinsics[0], true) &&
                     CHECK_CONTIGUOUS_AND_SETERROR_ALL()))
                  return false;

              cookie->Nintrinsics = mrcal_lensmodel_num_params(&cookie->lensmodel);
              _mrcal_precompute_lensmodel_data(&cookie->precomputed, &cookie->lensmodel);
              return true;
''',

            Ccode_slice_eval = \
                {np.float64:
                 r'''
                 const int N = 1;

                 if(cookie->lensmodel.type == MRCAL_LENSMODEL_CAHVORE)
                     return _mrcal_project_internal_cahvore(
                                (mrcal_point2_t*)data_slice__output,
                                (const mrcal_point3_t*)data_slice__points,
                                N,
                                (const double*)data_slice__intrinsics,
                                cookie->lensmodel.LENSMODEL_CAHVORE__config.linearity);

                 if(MRCAL_LENSMODEL_IS_OPENCV(cookie->lensmodel.type) ||
                    cookie->lensmodel.type == MRCAL_LENSMODEL_PINHOLE)
                 {
                     _mrcal_project_internal_opencv(
                                (mrcal_point2_t*)data_slice__output,
                                NULL, NULL,
                                (const mrcal_point3_t*)data_slice__points,
                                N,
                                (const double*)data_slice__intrinsics,
                                cookie->Nintrinsics);
                     return true;
                 }

                 return
                     _mrcal_project_internal((mrcal_point2_t*)data_slice__output,
                                             NULL, NULL,
                                             (const mrcal_point3_t*)data_slice__points,
                                             N,
                                             &cookie->lensmodel,
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
              mrcal_lensmodel_t              lensmodel;
              int                            Nintrinsics;
              mrcal_projection_precomputed_t precomputed;
            ''',

            Ccode_validate = r'''
              if( !( validate_lensmodel_un_project(&cookie->lensmodel,
                                        lensmodel, dims_slice__intrinsics[0], true) &&
                     CHECK_CONTIGUOUS_AND_SETERROR_ALL()))
                  return false;

              mrcal_lensmodel_metadata_t meta = mrcal_lensmodel_metadata(&cookie->lensmodel);
              if(!meta.has_gradients)
              {
                  PyErr_Format(PyExc_RuntimeError,
                               "_project(get_gradients=True) requires a lens model that has gradient support");
                  return false;
              }
              cookie->Nintrinsics = mrcal_lensmodel_num_params(&cookie->lensmodel);
              _mrcal_precompute_lensmodel_data(&cookie->precomputed, &cookie->lensmodel);
              return true;
''',

            Ccode_slice_eval = \
                {np.float64:
                 r'''
                 const int N = 1;

                 return
                     _mrcal_project_internal((mrcal_point2_t*)data_slice__output0,
                                             (mrcal_point3_t*)data_slice__output1,
                                             (double*)  data_slice__output2,
                                             (const mrcal_point3_t*)data_slice__points,
                                             N,
                                             &cookie->lensmodel,
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

- This function requires gradients, so it does not support some lens models;
  CAHVORE for instance

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
              mrcal_lensmodel_t lensmodel;
              mrcal_projection_precomputed_t precomputed;
            ''',

            Ccode_validate = r'''
              if( !( validate_lensmodel_un_project(&cookie->lensmodel,
                                        lensmodel, dims_slice__intrinsics[0], false) &&
                     CHECK_CONTIGUOUS_AND_SETERROR_ALL()))
                  return false;

              _mrcal_precompute_lensmodel_data(&cookie->precomputed, &cookie->lensmodel);
              return true;
''',

            Ccode_slice_eval = \
                {np.float64:
                 r'''
                 const int N = 1;
                 return
                     _mrcal_unproject_internal((mrcal_point3_t*)data_slice__output,
                                               (const mrcal_point2_t*)data_slice__points,
                                               N,
                                               &cookie->lensmodel,
                                               // core, distortions concatenated
                                               (const double*)data_slice__intrinsics,
                                               &cookie->precomputed);
'''},
)

project_simple_doc = """Internal projection routine

This is the internals for mrcal.project_{what}(). As a user, please call
THAT function, and see the docs for that function. The differences:

- This is just the no-gradients function. The internal function that reports the
  gradients also is _project_{what}_withgrad()

- This function is wrapped with numpysane_pywrap, so the points broadcast as
  expected

"""

project_withgrad_simple_doc = """Internal projection routine with gradients

This is the internals for mrcal.project_{what}(). As a user, please call
THAT function, and see the docs for that function. The differences:

- This is just the gradient-reporting function. The internal function that
  does not report the gradients is _project_{what}()

- This function is wrapped with numpysane_pywrap, so the points broadcast as
  expected

"""

unproject_simple_doc = """Internal unprojection routine

This is the internals for mrcal.unproject_{what}(). As a user, please
call THAT function, and see the docs for that function. The differences:

- This is just the no-gradients function. The internal function that reports the
  gradients also is _unproject_{what}_withgrad()

- This function is wrapped with numpysane_pywrap, so the points broadcast as
  expected

"""

unproject_withgrad_simple_doc = """Internal unprojection routine

This is the internals for mrcal.unproject_{what}(). As a user, please
call THAT function, and see the docs for that function. The differences:

- This is just the gradient-reporting function. The internal function that does
  not report the gradients is _unproject_{what}()

- This function is wrapped with numpysane_pywrap, so the points broadcast as
  expected

"""

simple_kwargs_base = \
    dict(args_input = ('points','fxycxy'),
         Ccode_validate = r'''return CHECK_CONTIGUOUS_AND_SETERROR_ALL();''')
project_simple_kwargs = \
    dict( prototype_input  = ((3,),(4,)),
          prototype_output = (2,))
project_withgrad_simple_kwargs = \
    dict( prototype_input  = ((3,),(4,)),
          prototype_output = ((2,), (2,3)))
unproject_simple_kwargs = \
    dict( prototype_input  = ((2,),(4,)),
          prototype_output = (3,))
unproject_withgrad_simple_kwargs = \
    dict( prototype_input  = ((2,),(4,)),
          prototype_output = ((3,), (3,2)))

project_simple_code = '''
            const double* fxycxy = (const double*)data_slice__fxycxy;
            mrcal_project_{what}((mrcal_point2_t*)data_slice__output,
                                        NULL,
                                        (const mrcal_point3_t*)data_slice__points,
                                        1,
                                        fxycxy);
            return true;'''
project_withgrad_simple_code = '''
            const double* fxycxy = (const double*)data_slice__fxycxy;
            mrcal_project_{what}((mrcal_point2_t*)data_slice__output0,
                                        (mrcal_point3_t*)data_slice__output1,
                                        (const mrcal_point3_t*)data_slice__points,
                                        1,
                                        fxycxy);
            return true;'''
unproject_simple_code = '''
            const double* fxycxy = (const double*)data_slice__fxycxy;
            mrcal_unproject_{what}((mrcal_point3_t*)data_slice__output,
                                        NULL,
                                        (const mrcal_point2_t*)data_slice__points,
                                        1,
                                        fxycxy);
            return true;'''
unproject_withgrad_simple_code = '''
            const double* fxycxy = (const double*)data_slice__fxycxy;
            mrcal_unproject_{what}((mrcal_point3_t*)data_slice__output0,
                                        (mrcal_point2_t*)data_slice__output1,
                                        (const mrcal_point2_t*)data_slice__points,
                                        1,
                                        fxycxy);
            return true;'''

for what in ('pinhole', 'stereographic', 'lonlat', 'latlon'):
    m.function( f"_project_{what}",
                project_simple_doc.format(what = what),
                Ccode_slice_eval = {np.float64:
                                    project_simple_code.format(what = what)},
                **project_simple_kwargs,
                **simple_kwargs_base )
    m.function( f"_project_{what}_withgrad",
                project_withgrad_simple_doc.format(what = what),
                Ccode_slice_eval = {np.float64:
                                    project_withgrad_simple_code.format(what = what)},
                **project_withgrad_simple_kwargs,
                **simple_kwargs_base )
    m.function( f"_unproject_{what}",
                unproject_simple_doc.format(what = what),
                Ccode_slice_eval = {np.float64:
                                    unproject_simple_code.format(what = what)},
                **unproject_simple_kwargs,
                **simple_kwargs_base )
    m.function( f"_unproject_{what}_withgrad",
                unproject_withgrad_simple_doc.format(what = what),
                Ccode_slice_eval = {np.float64:
                                    unproject_withgrad_simple_code.format(what = what)},
                **unproject_withgrad_simple_kwargs,
                **simple_kwargs_base )

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
            prototype_input  = (('Nx','Nstate'), ('Np',), ('Nix',), ('Nix',)),
            prototype_output = ('Nx','Nx'),

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
                 //     jta = matmult(jt,At); // jta has shape (Nx,)
                 //     accumulate( outer(jta,jta) )


                 int32_t Nx     = dims_slice__A[0];
                 int32_t Nstate = dims_slice__A[1];

                 const double*   A = (const double* )data_slice__A;
                 const int32_t* Jp = (const int32_t*)data_slice__Jp;
                 const int32_t* Ji = (const int32_t*)data_slice__Ji;
                 const double*  Jx = (const double* )data_slice__Jx;
                 double* out       = (      double* )data_slice__output;

                 for(int i=0; i<Nx*Nx; i++)
                     out[i] = 0.0;

                 for(int irow=0; irow<*Nleading_rows_J; irow++)
                 {
                     double jta[Nx];
                     for(int i=0; i<Nx; i++)
                         jta[i] = 0.0;

                     for(int32_t i = Jp[irow]; i < Jp[irow+1]; i++)
                     {
                         int32_t icol = Ji[i];
                         double x     = Jx[i];

                         for(int j=0; j<Nx; j++)
                             jta[j] += A[icol + j*Nstate] * x;
                     }
                     for(int i=0; i<Nx; i++)
                     {
                         out[i*Nx + i] += jta[i]*jta[i];
                         for(int j=i+1; j<Nx; j++)
                         {
                             out[i*Nx + j] += jta[i]*jta[j];
                             out[j*Nx + i] += jta[i]*jta[j];
                         }
                     }
                 }
                 return true;
'''},
)

m.function( "_Jt_x",
            """Computes matrix-vector multiplication Jt*xt

SYNOPSIS

    Jt_x = np.zeros( (J.shape[-1],), dtype=float)
    mrcal._mrcal_npsp._Jt_x(J.indptr,
                            J.indices,
                            J.data,
                            x,
                            out = Jt_x)

Jt is the transpose of a (possibly very large) sparse array and x is a dense
column vector. We pass in

- J: the sparse array
- xt: the row vector transpose of x

The output is a dense row vector, the transpose of the multiplication

J is sparse, stored by row. This is the scipy.sparse.csr_matrix representation,
and is also how CHOLMOD stores Jt (CHOLMOD stores by column, so the same data
looks like Jt to CHOLMOD). The sparse J is given here as the p,i,x arrays from
CHOLMOD, equivalent to the indptr,indices,data members of
scipy.sparse.csr_matrix respectively.

Note: The output array MUST be passed-in because there's no way to know its
shape beforehand. For the same reason, we cannot verify that its shape is
correct, and the caller MUST do that, or else the program can crash.

""",

            args_input       = ('Jp', 'Ji', 'Jx', 'xt'),
            prototype_input  = (('Np',), ('Nix',), ('Nix',), ('Nrows',)),
            prototype_output = ('Ny',),

            Ccode_validate = r'''
            int32_t Np    = dims_slice__Jp[0];
            int32_t Nrows = dims_slice__xt[0];
            if( Nrows != Np-1 )
            {
                PyErr_Format(PyExc_RuntimeError,
                             "len(xt) must match the number of rows in J");
                return false;
            }
            return CHECK_CONTIGUOUS_AND_SETERROR_ALL();''',

            Ccode_slice_eval = \
                { (np.int32, np.int32, np.float64, np.float64, np.float64, ):
                 r'''

                 int32_t Np        = dims_slice__Jp[0];
                 const int32_t* Jp = (const int32_t*)data_slice__Jp;
                 const int32_t* Ji = (const int32_t*)data_slice__Ji;
                 const double*  Jx = (const double* )data_slice__Jx;

                 const double* x   = (const double* )data_slice__xt;


                 int32_t Ny = dims_slice__output[0];
                 double* y  = (double*)data_slice__output;

                 const int32_t Nrows = Np-1;
                 for(int i=0; i<Ny; i++)
                     y[i] = 0.0;

                 for(int irow=0; irow<Nrows; irow++)
                 {
                     for(int32_t i = Jp[irow]; i < Jp[irow+1]; i++)
                     {
                         int32_t icol = Ji[i];
                         double j     = Jx[i];

                         y[icol] += j*x[irow];
                     }
                 }
                 return true;
'''},
)


apply_homography_body = \
r'''
    ctype__v xyz[3] = {
        item__H(0,0)*item__v(0) + item__H(0,1)*item__v(1) + item__H(0,2),
        item__H(1,0)*item__v(0) + item__H(1,1)*item__v(1) + item__H(1,2),
        item__H(2,0)*item__v(0) + item__H(2,1)*item__v(1) + item__H(2,2)
     };
     item__output(0) = xyz[0]/xyz[2];
     item__output(1) = xyz[1]/xyz[2];
     return true;
'''
m.function( "apply_homography",
            r'''Apply a homogeneous-coordinate homography to a set of 2D points

SYNOPSIS

    print( H.shape )
    ===> (3,3)

    print( q0.shape )
    ===> (100, 2)

    q1 = mrcal.apply_homography(H10, q0)

    print( q1.shape )
    ===> (100, 2)

A homography maps from pixel coordinates observed in one camera to pixel
coordinates in another. For points represented in homogeneous coordinates ((k*x,
k*y, k) to represent a pixel (x,y) for any k) a homography is a linear map H.
Since homogeneous coordinates are unique only up-to-scale, the homography matrix
H is also unique up to scale.

If two pinhole cameras are observing a planar surface, there exists a homography
that relates observations of the plane in the two cameras.

This function supports broadcasting fully.

ARGUMENTS

- H: an array of shape (..., 3,3). This is the homography matrix. This is unique
  up-to-scale, so a homography H is functionally equivalent to k*H for any
  non-zero scalar k

- q: an array of shape (..., 2). The pixel coordinates we are mapping

RETURNED VALUE

An array of shape (..., 2) containing the pixels q after the homography was
applied

    ''',

            args_input       = ('H', 'v'),
            prototype_input  = ((3,3), (2,)),
            prototype_output = (2,),

            Ccode_slice_eval = \
                { np.float64: apply_homography_body,
                  np.float32: apply_homography_body },
)

m.write()
