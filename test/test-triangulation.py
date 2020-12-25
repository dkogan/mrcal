#!/usr/bin/python3

import sys
import numpy as np
import numpysane as nps
import os

testdir = os.path.dirname(os.path.realpath(__file__))

# I import the LOCAL mrcal since that's what I'm testing
sys.path[:0] = f"{testdir}/..",
import mrcal
import testutils
from test_calibration_helpers import grad,plot_args_points_and_covariance_ellipse,plot_arg_covariance_ellipse


# I want the RNG to be deterministic
np.random.seed(0)




def triangulate_geometric_ref(v0,v1,t01):
    r'''reference implementation to compare against'''

    # Let's say I have a ray from the origin to v0, and another ray from t01
    # to v1 (v0 and v1 aren't necessarily normal). Let the nearest points on
    # each ray be k0 and k1 along each ray respectively: E = norm2(t01 + k1*v1
    # - k0*v0):
    #
    #   dE/dk0 = 0 = inner(t01 + k1*v1 - k0*v0, -v0)
    #   dE/dk1 = 0 = inner(t01 + k1*v1 - k0*v0,  v1)
    #
    # ->    t01.v0 + k1 v0.v1 = k0 v0.v0
    #      -t01.v1 + k0 v0.v1 = k1 v1.v1
    #
    # -> [  v0.v0   -v0.v1] [k0] = [ t01.v0]
    #    [ -v0.v1    v1.v1] [k1] = [-t01.v1]
    #
    # -> [k0] = 1/(v0.v0 v1.v1 -(v0.v1)**2) [ v1.v1   v0.v1][ t01.v0]
    #    [k1]                               [ v0.v1   v0.v0][-t01.v1]
    #
    # The midpoint:
    #
    #   x = (k0 v0 + t01 + k1 v1)/2
    M = nps.cat(v1,v0)
    det = nps.norm2(v0) * nps.norm2(v1) - nps.inner(v0,v1)*nps.inner(v0,v1)
    k = nps.inner( nps.matmult(M,nps.transpose(M)),
                   np.array((   nps.inner(t01,v0),
                               -nps.inner(t01,v1))) ) / det
    return (k[0]*v0 + k[1]*v1 + t01) / 2.


############### Test geometry
t01  = np.array(( 1.,   0.1,  -0.2))
R01  = mrcal.R_from_r(np.array((0.01, -0.02, -0.03)))
Rt01 = nps.glue(R01, t01, axis=-2)

# camera0 is the "reference"
model0 = mrcal.cameramodel( intrinsics = ('LENSMODEL_PINHOLE',
                                          np.array((1000., 1000., 500., 500.))),
                            imagersize = np.array((1000,1000)) )
model1 = mrcal.cameramodel( intrinsics = ('LENSMODEL_PINHOLE',
                                          np.array((1100., 1100., 500., 500.))),
                            imagersize = np.array((1000,1000)),
                            extrinsics_Rt_toref = Rt01 )

d        = 200.
sigma    = 0.5
Nsamples = 2000

p = np.array((10., 20., 200.))
q0 = mrcal.project( mrcal.transform_point_Rt( model0.extrinsics_Rt_fromref(), p),
                    *model0.intrinsics() )
q1 = mrcal.project( mrcal.transform_point_Rt( model1.extrinsics_Rt_fromref(), p),
                    *model1.intrinsics() )

def noisy_observation_vectors(N, sigma = 0.1):
    q_noise = np.random.randn(2,N,2) * sigma

    # All have shape (N,2)
    v0_local_noisy = mrcal.unproject( q0 + q_noise[0], *model0.intrinsics() )
    v1_local_noisy = mrcal.unproject( q1 + q_noise[1], *model1.intrinsics() )
    v0_ref_noisy   = v0_local_noisy
    v1_ref_noisy   = mrcal.rotate_point_R(Rt01[:3,:], v1_local_noisy)

    # All have shape (N,3)
    return v0_local_noisy,v1_local_noisy,v0_ref_noisy,v1_ref_noisy

v0_local_noisy,v1_local_noisy,v0_ref_noisy,v1_ref_noisy = \
    [v[0] for v in noisy_observation_vectors(1)]


############# Test case is set up. Let's run the tests!
f = mrcal._mrcal_npsp._triangulate_geometric_withgrad
result, m, dm_dv0, dm_dv1, dm_dt01 = f( v0_ref_noisy, v1_ref_noisy, t01 )

dm_dv0_empirical  = \
  grad( lambda v0:  f(v0,           v1_ref_noisy, t01)[1], v0_ref_noisy)
dm_dv1_empirical  = \
  grad( lambda v1:  f(v0_ref_noisy, v1,           t01)[1], v1_ref_noisy)
dm_dt01_empirical = \
  grad( lambda t01: f(v0_ref_noisy, v1_ref_noisy, t01)[1], t01)


testutils.confirm_equal( m, triangulate_geometric_ref(v0_ref_noisy,v1_ref_noisy,t01),
                         relative  = True,
                         worstcase = True,
                         msg = f"Geometric triangulation: computation is correct",
                         eps = 1e-6)
testutils.confirm_equal( m, p,
                         relative  = True,
                         worstcase = True,
                         msg = f"Geometric triangulation: noisy intersection is about right",
                         eps = 0.1)
testutils.confirm_equal( dm_dv0, dm_dv0_empirical,
                         relative  = True,
                         worstcase = True,
                         msg = f"Geometric triangulation, dm/dv0",
                         eps = 1e-6)
testutils.confirm_equal( dm_dv1, dm_dv1_empirical,
                         relative  = True,
                         worstcase = True,
                         msg = f"Geometric triangulation, dm/dv1",
                         eps = 1e-6)
testutils.confirm_equal( dm_dt01, dm_dt01_empirical,
                         relative  = True,
                         worstcase = True,
                         msg = f"Geometric triangulation, dm/dt01",
                         eps = 1e-6)


f = mrcal._mrcal_npsp._triangulate_lindstrom_withgrad
result, m, dm_dv0, dm_dv1, dm_dRt01 = f( v0_local_noisy, v1_local_noisy, Rt01 )

dm_dv0_empirical   = \
  grad( lambda v0:  f(v0,             v1_local_noisy, Rt01)[1], v0_local_noisy)
dm_dv1_empirical   = \
  grad( lambda v1:  f(v0_local_noisy, v1,             Rt01)[1], v1_local_noisy)
dm_dRt01_empirical = \
  grad( lambda Rt01:f(v0_local_noisy, v1_local_noisy, Rt01)[1], Rt01)

testutils.confirm_equal( m, p,
                         relative  = True,
                         worstcase = True,
                         msg = f"Lindstrom triangulation: noisy intersection is about right",
                         eps = 0.1)
testutils.confirm_equal( dm_dv0, dm_dv0_empirical,
                         relative  = True,
                         worstcase = True,
                         msg = f"Lindstrom triangulation, dm/dv0",
                         eps = 1e-6)
testutils.confirm_equal( dm_dv1, dm_dv1_empirical,
                         relative  = True,
                         worstcase = True,
                         msg = f"Lindstrom triangulation, dm/dv1",
                         eps = 1e-6)
testutils.confirm_equal( dm_dRt01, dm_dRt01_empirical,
                         relative  = True,
                         worstcase = True,
                         msg = f"Lindstrom triangulation, dm/dRt01",
                         eps = 1e-6)


############ bias test
#
# The lindstrom method is supposed to have minimal bias, while the geometric
# method should be biased. Let's do some random sampling to confirm
v0_local_noisy,v1_local_noisy,v0_ref_noisy,v1_ref_noisy = \
    noisy_observation_vectors(Nsamples, sigma = sigma)

print("sampling: p_sampled_geometric")
p_sampled_geometric = \
    mrcal._mrcal_npsp._triangulate_geometric( v0_ref_noisy, v1_ref_noisy, t01 )[1]
print("sampling: p_sampled_lindstrom")
p_sampled_lindstrom = \
    mrcal._mrcal_npsp._triangulate_lindstrom( v0_local_noisy, v1_local_noisy, Rt01 )[1]
print("sampling: p_sampled_leecivera_l1")
p_sampled_leecivera_l1 = \
    mrcal._mrcal_npsp._triangulate_leecivera_l1( v0_local_noisy, v1_local_noisy, t01 )[1]
print("sampling: p_sampled_leecivera_linf")
p_sampled_leecivera_linf = \
    mrcal._mrcal_npsp._triangulate_leecivera_linf( v0_local_noisy, v1_local_noisy, t01 )[1]

q0_sampled_geometric      = mrcal.project(p_sampled_geometric,      *model0.intrinsics())
q0_sampled_lindstrom      = mrcal.project(p_sampled_lindstrom,      *model0.intrinsics())
q0_sampled_leecivera_l1   = mrcal.project(p_sampled_leecivera_l1,   *model0.intrinsics())
q0_sampled_leecivera_linf = mrcal.project(p_sampled_leecivera_linf, *model0.intrinsics())


import gnuplotlib as gp
gp.plot( *plot_args_points_and_covariance_ellipse( q0_sampled_geometric,      'geometric' ),
         *plot_args_points_and_covariance_ellipse( q0_sampled_lindstrom,      'lindstrom' ),
         *plot_args_points_and_covariance_ellipse( q0_sampled_leecivera_l1,   'lee-civera-l1' ),
         *plot_args_points_and_covariance_ellipse( q0_sampled_leecivera_linf, 'lee-civera-linf' ),
         ( q0, dict(_with     = 'points pt 7 ps 2',
                    tuplesize = -2,
                    legend    = 'Ground truth')),
         square = True,
         wait = True
        )




import IPython
IPython.embed()





# Try r grad via r (not R)
# need python wrapper to set the result, and I should call the python wrapper here
# wrapper should do no-gradients by default
# should check result here
# docs


testutils.finish()
