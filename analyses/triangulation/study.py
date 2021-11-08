#!/usr/bin/python3

r'''Study the precision and accuracy of the various triangulation routines'''


import sys
import numpy as np
import numpysane as nps
import gnuplotlib as gp

import os.path

# I import the LOCAL mrcal
scriptdir = os.path.dirname(os.path.realpath(__file__))
sys.path[:0] = f"{scriptdir}/../..",
import mrcal

############ bias visualization
#
# I simulate pixel noise, and see what that does to the triangulation. Play with
# the geometric details to get a sense of how these behave

model0 = mrcal.cameramodel( intrinsics = ('LENSMODEL_PINHOLE',
                                          np.array((1000., 1000., 500., 500.))),
                            imagersize = np.array((1000,1000)) )
model1 = mrcal.cameramodel( intrinsics = ('LENSMODEL_PINHOLE',
                                          np.array((1100., 1100., 500., 500.))),
                            imagersize = np.array((1000,1000)) )


# square camera layout
t01  = np.array(( 1.,   0.1,  -0.2))
R01  = mrcal.R_from_r(np.array((0.001, -0.002, -0.003)))
Rt01 = nps.glue(R01, t01, axis=-2)

p  = np.array(( 5000., 300.,  2000.))
q0 = mrcal.project(p, *model0.intrinsics())

Nsamples = 100000
sigma    = 0.1

v0local_noisy, v1local_noisy,v0_noisy,v1_noisy,_,_,_,_ = \
    mrcal.synthetic_data. \
    _noisy_observation_vectors_for_triangulation(p,Rt01,
                                                 model0.intrinsics(),
                                                 model1.intrinsics(),
                                                 Nsamples,
                                                 sigma = sigma)

p_sampled_geometric       = mrcal.triangulate_geometric(      v0_noisy,      v1_noisy,      t01 )
p_sampled_lindstrom       = mrcal.triangulate_lindstrom(      v0local_noisy, v1local_noisy, Rt01 )
p_sampled_leecivera_l1    = mrcal.triangulate_leecivera_l1(   v0_noisy,      v1_noisy,      t01 )
p_sampled_leecivera_linf  = mrcal.triangulate_leecivera_linf( v0_noisy,      v1_noisy,      t01 )
p_sampled_leecivera_mid2  = mrcal.triangulate_leecivera_mid2( v0_noisy,      v1_noisy,      t01 )
p_sampled_leecivera_wmid2 = mrcal.triangulate_leecivera_wmid2(v0_noisy,      v1_noisy,      t01 )

q0_sampled_geometric       = mrcal.project(p_sampled_geometric,      *model0.intrinsics())
q0_sampled_lindstrom       = mrcal.project(p_sampled_lindstrom,      *model0.intrinsics())
q0_sampled_leecivera_l1    = mrcal.project(p_sampled_leecivera_l1,   *model0.intrinsics())
q0_sampled_leecivera_linf  = mrcal.project(p_sampled_leecivera_linf, *model0.intrinsics())
q0_sampled_leecivera_mid2  = mrcal.project(p_sampled_leecivera_mid2, *model0.intrinsics())
q0_sampled_leecivera_wmid2 = mrcal.project(p_sampled_leecivera_wmid2, *model0.intrinsics())

range_sampled_geometric       = nps.mag(p_sampled_geometric)
range_sampled_lindstrom       = nps.mag(p_sampled_lindstrom)
range_sampled_leecivera_l1    = nps.mag(p_sampled_leecivera_l1)
range_sampled_leecivera_linf  = nps.mag(p_sampled_leecivera_linf)
range_sampled_leecivera_mid2  = nps.mag(p_sampled_leecivera_mid2)
range_sampled_leecivera_wmid2 = nps.mag(p_sampled_leecivera_wmid2)

if False:
    # Plot the reprojected pixel
    gp.plot( *mrcal.utils._plot_args_points_and_covariance_ellipse( q0_sampled_geometric,      'geometric' ),
             *mrcal.utils._plot_args_points_and_covariance_ellipse( q0_sampled_lindstrom,      'lindstrom' ),
             *mrcal.utils._plot_args_points_and_covariance_ellipse( q0_sampled_leecivera_l1,   'lee-civera-l1' ),
             *mrcal.utils._plot_args_points_and_covariance_ellipse( q0_sampled_leecivera_linf, 'lee-civera-linf' ),
             *mrcal.utils._plot_args_points_and_covariance_ellipse( q0_sampled_leecivera_mid2, 'lee-civera-mid2' ),
             *mrcal.utils._plot_args_points_and_covariance_ellipse( q0_sampled_leecivera_wmid2,'lee-civera-wmid2' ),
             ( q0,
               dict(_with     = 'points pt 3 ps 2',
                    tuplesize = -2,
                    legend    = 'Ground truth')),
             square = True,
             wait   = True,
             title  = 'Reprojected triangulated point')

else:
    # Plot the range distribution
    range_ref = nps.mag(p)
    gp.plot( nps.cat( range_sampled_geometric,
                      range_sampled_lindstrom,
                      range_sampled_leecivera_l1,
                      range_sampled_leecivera_linf,
                      range_sampled_leecivera_mid2,
                      range_sampled_leecivera_wmid2 ),
             legend = np.array(( 'range_sampled_geometric',
                                 'range_sampled_lindstrom',
                                 'range_sampled_leecivera_l1',
                                 'range_sampled_leecivera_linf',
                                 'range_sampled_leecivera_mid2',
                                 'range_sampled_leecivera_wmid2' )),
             histogram=True,
             binwidth=200,
             _with='lines',
             _set = f'arrow from {range_ref},graph 0 to {range_ref},graph 1 nohead lw 5',
             wait = True,
             title = "Range distribution")


