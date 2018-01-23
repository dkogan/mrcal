#!/usr/bin/python2

import numpy as np
import numpysane as nps
import sys
import argparse
import re
import os.path

from mrcal import cameramodel
from mrcal import cahvor
from mrcal import poseutils
from mrcal import utils


# I need at least gnuplotlib 0.16. That version fixed label plotting. Don't know
# how to ask python to check. Tried
# pkg_resources.get_distribution("gnuplotlib").version, but that reports the
# wrong value
import gnuplotlib as gp




r'''This tool reads in stereo calibration files (the .cahvor files and the
transforms.txt) and makes a plot that shows the world, the cameras, the ins and
the vehicle coord systems. This should simplify debugging'''

def parse_args():
    parser = \
        argparse.ArgumentParser(description= \
    r'''Visualize stereo calibration geometry. The calibration files can be given as
a directory or as a set of files directly. By default, a visualization in the
coord system of the vehicle is given. If --wld_from_ins or --wld_from_pair is
given then we use the world coord system instead''')
    parser.add_argument('--dir',
                        nargs=1,
                        help='Directory that contains the calibration')
    parser.add_argument('--wld_from_ins',
                        nargs=7,
                        help=r'''ins->world transform. Given as 7 numbers: pos.{xyz} rot.{uxyz}. Exclusive with --wld_from_pair''')
    parser.add_argument('--wld_from_pair',
                        nargs=8,
                        help=r'''camera_pair->world transform. Given as 8 numbers: pair_index pos.{xyz}
rot.{uxyz}. Exclusive with --wld_from_ins''')
    parser.add_argument('cal_file',
                        nargs='*',
                        type=file,
                        help='.cahvor and tranforms.txt files')

    args = parser.parse_args()

    if args.dir and args.cal_file:
        raise Exception("A directory OR calibration files could be given; not both")
    if not args.dir and not args.cal_file:
        raise Exception("A directory OR calibration files must given; not neither")

    if args.dir:
        transforms = '{}/transforms.txt'.format(args.dir[0])
        if not os.path.isfile(transforms):
            transforms = None

        cahvors = { 0: ['{}/camera0-0.cahvor'.format(args.dir[0]),
                        '{}/camera0-1.cahvor'.format(args.dir[0])],
                    1: ['{}/camera1-0.cahvor'.format(args.dir[0]),
                        '{}/camera1-1.cahvor'.format(args.dir[0])] }

    else:
        transforms = [f for f in args.cal_file if re.match('(?:.*/)?transforms.txt$', f.name)]
        if len(transforms) > 1:
            raise Exception("At most one transforms.txt should have been given")
        try:
            transforms = transforms[0]
        except:
            transforms = None

        cahvors = {}
        for f in args.cal_file:
            if re.match('(?:.*/)?transforms.txt$', f.name):
                continue

            m = re.match('(?:.*/)?camera(\d+)-([01]).cahvor$', f.name)
            if not m:
                raise Exception("All cal_file args must be either 'transforms.txt' or 'cameraA-B.cahvor', but '{}' is neither".format(f.name))

            p,i = int(m.group(1)),int(m.group(2))
            if cahvors.get(p) is None:
                cahvors[p] = [None, None]
            if cahvors[p][i] is not None:
                raise Exception("cahvor for pair {}, camera {} given multiple times".format(p,i))
            cahvors[p][i] = f


    if args.wld_from_ins and args.wld_from_pair:
        raise Exception("At most one of --wld_from_ins and --wld_from_pair can be given")

    i_wld_from_pair,wld_from_pair,wld_from_ins = None,None,None
    if args.wld_from_pair:
        i_wld_from_pair = int(args.wld_from_pair[0])
        p = np.array([float(x) for x in args.wld_from_pair[1:4]])
        q = np.array([float(x) for x in args.wld_from_pair[4:]])

        if np.abs(nps.inner(q,q) - 1) > 1e-5:
            raise Exception("wld_from_pair given a non-unit quaternion rotation: {}".format(q))
        wld_from_pair = cahvor.Rt_from_pq(nps.glue(p,q, axis=-1))

    if args.wld_from_ins:
        p = np.array([float(x) for x in args.wld_from_ins[:3]])
        q = np.array([float(x) for x in args.wld_from_ins[3:]])
        if np.abs(nps.inner(q,q) - 1) > 1e-5:
            raise Exception("wld_from_ins given a non-unit quaternion rotation: {}".format(q))
        wld_from_ins = cahvor.Rt_from_pq(nps.glue(p,q, axis=-1))

    return transforms,cahvors,wld_from_ins,i_wld_from_pair,wld_from_pair


def parse_and_consolidate(transforms, models):
    if transforms is not None:
        transforms = cahvor.read_transforms(transforms)
    else:
        Rt_ident = nps.glue( np.eye(3), np.zeros(3), axis=-2)
        # no transforms.txt. Use an identity transform for everything
        transforms = {'ins_from_camera': {},
                      'veh_from_ins': Rt_ident }
        for i_pair in models.keys():
            transforms['ins_from_camera'][i_pair] = Rt_ident

    for i_pair in models.keys():
        models[i_pair] = [cahvor.read(m) for m in models[i_pair]]

    pair_ids = sorted(transforms['ins_from_camera'].keys())
    if pair_ids != sorted(models.keys()):
        raise Exception("Mismatched camera pair IDs. transforms.txt knows about pairs {}, but I have models for pairs {}".format(pair_ids,models.keys()))

    pairs = {}
    for i in pair_ids:
        pair = {'ins_from_camera': transforms['ins_from_camera'][i]}

        for icam in range(len(models[i])):
            pair[icam] = models[i][icam]
        pairs[i] = pair

    veh_from_ins = transforms['veh_from_ins']

    return pairs,veh_from_ins


def extend_axes_for_plotting(axes):
    r'''Input is a 4x3 axes array: center, center+x, center+y, center+z. I transform
this into a 3x6 array that can be gnuplotted "with vectors"

    '''

    # first, copy the center 3 times
    out = nps.cat( axes[0,:],
                   axes[0,:],
                   axes[0,:] )

    # then append just the deviations to each row containing the center
    out = nps.glue( out, axes[1:,:] - axes[0,:], axis=-1)
    return out

def gen_pair_axes(pairs, global_from_ins):
    r'''Given all my camera pairs, generate a list of tuples that can be passed to
gnuplotlib to plot my world'''


    def gen_one_pair_axes(ipair, pair):
        r'''Given ONE camera pair, generate a list of tuples that can be passed to
gnuplotlib to plot my world

        '''

        def gen_one_cam_axes(icam, cam, ins_from_camera):

            return utils.gen_plot_axes( (global_from_ins,
                                         ins_from_camera,
                                         cam.extrinsics_Rt(True)),

                                        'pair{}-camera{}'.format(ipair, icam),
                                        color = ipair+1,
                                        scale = 0.5,
                                        label_offset=0.05)


        individual_cam_axes = (e for icam in (0,1) for e in gen_one_cam_axes(icam, pair[icam], pair['ins_from_camera']))
        pair_axes           = \
            utils.gen_plot_axes( (global_from_ins, pair['ins_from_camera']),
                                 'pair{}'.format(ipair),
                                 color = ipair+1,
                                 scale = 0.75)

        return (e for axes in (individual_cam_axes,pair_axes) for e in axes)

    return (a for ipair,pair in pairs.items() for a in gen_one_pair_axes(ipair, pair))

def gen_ins_axes(global_from_ins):
    return utils.gen_plot_axes( (global_from_ins,), "INS")






transforms,cahvors, \
    wld_from_ins,i_wld_from_pair,wld_from_pair = parse_args()

# if I don't do --dir but specify the stuff individually, transforms becomes [<file>], and parse_and_consolidate() then barfs
pairs,veh_from_ins = parse_and_consolidate(transforms, cahvors)
if wld_from_pair is not None:
    wld_from_ins = pose3_mul(wld_from_pair, pose3_inv(pairs[i_wld_from_pair]['ins_from_camera']))

global_from_ins = veh_from_ins if wld_from_ins is None else wld_from_ins
plot_pairs      = gen_pair_axes(pairs, global_from_ins)
plot_ins        = gen_ins_axes(global_from_ins)

# flatten the lists
allplots = [ e for p in plot_pairs,plot_ins for e in p ]


gp.plot3d( *allplots, square=1, zinv=1, xinv=1, yinv=1, ascii=1,
           xlabel='x', ylabel='y', zlabel='z',
           title="{} coordinate system".format( "VEHICLE" if wld_from_ins is None else "WORLD"))

import time
time.sleep(100000)
