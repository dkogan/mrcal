#!/usr/bin/env python3

r'''Study the precision and accuracy of the various triangulation routines'''


import sys
import argparse
import re
import os


def parse_args():

    parser = \
        argparse.ArgumentParser(description = __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--Nsamples',
                        type=int,
                        default=100000,
                        help='''How many random samples to evaluate. 100000 by
                        default''')

    group = parser.add_mutually_exclusive_group(required = True)
    group.add_argument('--ellipses',
                       action='store_true',
                       help='''Display the ellipses and samples in the xy plane''')
    group.add_argument('--ranges',
                       action='store_true',
                       help='''Display the distribution of the range''')


    parser.add_argument('--samples',
                       action='store_true',
                       help='''If --ellipses, plot the samples ALSO. Usually
                       this doesn't clarify anything, so the default is to omit
                       them''')

    parser.add_argument('--cache',
                        type=str,
                        choices=('read','write'),
                        help=f'''A cache file stores the recalibration results;
                        computing these can take a long time. This option allows
                        us to or write the cache instead of sampling. The cache
                        file is hardcoded to a cache file (in /tmp). By default,
                        we do neither: we don't read the cache (we sample
                        instead), and we do not write it to disk when we're
                        done. This option is useful for tests where we reprocess
                        the same scenario repeatedly''')

    parser.add_argument('--observed-point',
                        type    = float,
                        nargs   = 3,
                        default = ( 5000., 300.,  2000.),
                        help='''The camera0 coordinate of the observed point.
                        Default is ( 5000., 300., 2000.)''')


    parser.add_argument('--title',
                        type=str,
                        default = None,
                        help='''Title string for the plot. Overrides the default
                        title. Exclusive with --extratitle''')
    parser.add_argument('--extratitle',
                        type=str,
                        default = None,
                        help='''Additional string for the plot to append to the
                        default title. Exclusive with --title''')

    parser.add_argument('--hardcopy',
                        type=str,
                        help='''Write the output to disk, instead of an interactive plot''')
    parser.add_argument('--terminal',
                        type=str,
                        help=r'''gnuplotlib terminal. The default is good almost always, so most people don't
                        need this option''')
    parser.add_argument('--set',
                        type=str,
                        action='append',
                        help='''Extra 'set' directives to gnuplotlib. Can be given multiple times''')
    parser.add_argument('--unset',
                        type=str,
                        action='append',
                        help='''Extra 'unset' directives to gnuplotlib. Can be given multiple times''')

    args = parser.parse_args()


    if args.title      is not None and \
       args.extratitle is not None:
        print("--title and --extratitle are exclusive", file=sys.stderr)
        sys.exit(1)

    return args


args = parse_args()




import numpy as np
import numpysane as nps
import gnuplotlib as gp
import pickle
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

p  = np.array(args.observed_point)

q0 = mrcal.project(p, *model0.intrinsics())

sigma    = 0.1




cache_file = "/tmp/triangulation-study-cache.pickle"
if args.cache is None or args.cache == 'write':
    v0local_noisy, v1local_noisy,v0_noisy,v1_noisy,_,_,_,_ = \
        mrcal.synthetic_data. \
        _noisy_observation_vectors_for_triangulation(p,Rt01,
                                                     model0.intrinsics(),
                                                     model1.intrinsics(),
                                                     args.Nsamples,
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

    if args.cache is not None:
        with open(cache_file,"wb") as f:
            pickle.dump((v0local_noisy,
                         v1local_noisy,
                         v0_noisy,
                         v1_noisy,
                         p_sampled_geometric,
                         p_sampled_lindstrom,
                         p_sampled_leecivera_l1,
                         p_sampled_leecivera_linf,
                         p_sampled_leecivera_mid2,
                         p_sampled_leecivera_wmid2,
                         q0_sampled_geometric,
                         q0_sampled_lindstrom,
                         q0_sampled_leecivera_l1,
                         q0_sampled_leecivera_linf,
                         q0_sampled_leecivera_mid2,
                         q0_sampled_leecivera_wmid2,
                         range_sampled_geometric,
                         range_sampled_lindstrom,
                         range_sampled_leecivera_l1,
                         range_sampled_leecivera_linf,
                         range_sampled_leecivera_mid2,
                         range_sampled_leecivera_wmid2),
                        f)
        print(f"Wrote cache to {cache_file}")
else:
    with open(cache_file,"rb") as f:
        (v0local_noisy,
         v1local_noisy,
         v0_noisy,
         v1_noisy,
         p_sampled_geometric,
         p_sampled_lindstrom,
         p_sampled_leecivera_l1,
         p_sampled_leecivera_linf,
         p_sampled_leecivera_mid2,
         p_sampled_leecivera_wmid2,
         q0_sampled_geometric,
         q0_sampled_lindstrom,
         q0_sampled_leecivera_l1,
         q0_sampled_leecivera_linf,
         q0_sampled_leecivera_mid2,
         q0_sampled_leecivera_wmid2,
         range_sampled_geometric,
         range_sampled_lindstrom,
         range_sampled_leecivera_l1,
         range_sampled_leecivera_linf,
         range_sampled_leecivera_mid2,
         range_sampled_leecivera_wmid2) = \
            pickle.load(f)



plot_options = {}
if args.set is not None:
    plot_options['set'] = args.set
if args.unset is not None:
    plot_options['unset'] = args.unset

if args.hardcopy is not None:
    plot_options['hardcopy'] = args.hardcopy
if args.terminal is not None:
    plot_options['terminal'] = args.terminal

if args.ellipses:
    # Plot the reprojected pixels and the fitted ellipses

    data_tuples = \
        [ *mrcal.utils._plot_args_points_and_covariance_ellipse( q0_sampled_geometric,      'geometric' ),
          *mrcal.utils._plot_args_points_and_covariance_ellipse( q0_sampled_lindstrom,      'lindstrom' ),
          *mrcal.utils._plot_args_points_and_covariance_ellipse( q0_sampled_leecivera_l1,   'lee-civera-l1' ),
          *mrcal.utils._plot_args_points_and_covariance_ellipse( q0_sampled_leecivera_linf, 'lee-civera-linf' ),
          *mrcal.utils._plot_args_points_and_covariance_ellipse( q0_sampled_leecivera_mid2, 'lee-civera-mid2' ),
          *mrcal.utils._plot_args_points_and_covariance_ellipse( q0_sampled_leecivera_wmid2,'lee-civera-wmid2' ), ]
    if not args.samples:
        # Not plotting samples. Get rid of all the "dots" I'm plotting
        data_tuples = [ t for t in data_tuples if \
                        not (isinstance(t[-1], dict) and \
                             '_with' in t[-1] and \
                             t[-1]['_with'] == 'dots') ]

    if args.title is not None:
        title = args.title
    else:
        title = 'Reprojected triangulated point'
    if args.extratitle is not None:
        title += ': ' + args.extratitle

    gp.plot( *data_tuples,
             ( q0,
               dict(_with     = 'points pt 3 ps 2',
                    tuplesize = -2,
                    legend    = 'Ground truth')),
             square = True,
             wait   = 'hardcopy' not in plot_options,
             title  = title,
             **plot_options)

elif args.ranges:
    # Plot the range distribution
    range_ref = nps.mag(p)


    if args.title is not None:
        title = args.title
    else:
        title = "Range distribution"
    if args.extratitle is not None:
        title += ': ' + args.extratitle
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
             wait   = 'hardcopy' not in plot_options,
             title = title,
             **plot_options)

else:
    raise Exception("Getting here is a bug")

