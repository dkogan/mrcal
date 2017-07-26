#!/usr/bin/python2

import numpy as np
import numpysane as nps
import sys
import argparse
import re
from scipy import weave

# I need at least gnuplotlib 0.16 (got {}). That version fixed label plotting.
# Don't know how to ask python to check. Tried
# pkg_resources.get_distribution("gnuplotlib").version, but that reports the
# wrong value
import gnuplotlib as gp




r'''This tool reads in stereo calibration files (the .cahvor files and the
transforms.txt) and makes a plot that shows the world, the cameras, the ins and
the vehicle coord systems. This should simplify debugging'''


def transform(p,q, X, do_inverse=False):
    """Invoke a C program to apply a pose transformation. p is a 3d position vector
and q is a 4d unit quaternion to represent the rotation. 'X' is an Nx3 numpy
array (N vectors) and transformed N vectors in a new Nx3 are returned

    """

    p = p.astype(float)
    q = q.astype(float)
    X = X.astype(float)
    code = r'''

    pose3_t transform = {.pos = {.x = P1(0),
                                 .y = P1(1),
                                 .z = P1(2)},
                         .rot = {.u = Q1(0),
                                 .x = Q1(1),
                                 .y = Q1(2),
                                 .z = Q1(3)}};

    if( PyObject_IsTrue(do_inverse) ) transform = pose3_inv(transform);

    assert( DX >= 2);        // at least 2D
    assert( NX[DX-1] == 3 ); // last dimension has 3D points

    PyArrayObject* out = (PyArrayObject*)PyArray_SimpleNew(DX, NX, NPY_DOUBLE);


    // I wanted to have a recursive function to loop through ALL of X,
    // regardless of how many dimensions we have (one recursion level per
    // dimensions). But apparently this thing is in c++, so I can't have these
    // nested functions. Fine. I then hardcode working for 2D arrays only
    #if 0
    void transform_dim(npy_intp* i, int idim)
    {
        if(idim == DX-1)
        {
            // last dimension. transform
            vec3_t vin;
            i[idim] = 0; vin.x = *(double*)PyArray_GetPtr( X_array, i );
            i[idim] = 1; vin.y = *(double*)PyArray_GetPtr( X_array, i );
            i[idim] = 2; vin.z = *(double*)PyArray_GetPtr( X_array, i );

            vec3_t vout = vec3_transform(transform, vin);

            i[idim] = 0; *(double*)PyArray_GetPtr( out, i ) = vout.x;
            i[idim] = 1; *(double*)PyArray_GetPtr( out, i ) = vout.y;
            i[idim] = 2; *(double*)PyArray_GetPtr( out, i ) = vout.z;
        }
        else
            for(i[idim]=0; i[idim]<NX[idim]; i[idim]++)
                transform_dim(i, idim+1);
    }
    npy_intp i[DX];
    transform_dim(i, 0);


    #else

    assert(DX == 2);
    npy_intp i[2];
    for(int i=0; i<NX[0]; i++)
    {
        vec3_t vin;
        vin.x = X2(i,0);
        vin.y = X2(i,1);
        vin.z = X2(i,2);

        vec3_t vout = vec3_transform(transform, vin);

        (*((double*)(out->data + i*out->strides[0] + 0*out->strides[1]))) = vout.x;
        (*((double*)(out->data + i*out->strides[0] + 1*out->strides[1]))) = vout.y;
        (*((double*)(out->data + i*out->strides[0] + 2*out->strides[1]))) = vout.z;
    }

    #endif

    return_val = (PyObject*)out;
'''

    return \
        weave.inline(code,
                     ['p','q','X','do_inverse'],
                     include_dirs=["/home/dima/src_boats/mr-pose/include/maritime_robotics/pose/"],
                     headers=['"pose3.h"'],

                     # I get important-looking warnings about using a deprecated
                     # API, and I don't want to see them.
                     extra_compile_args=["-Wno-cpp -Wno-unused-variable"])

def rot2quat(R):
    """Invoke a C program to convert a 3x3 rotation matrix to a unit quaternion

    """

    R = R.astype(float)

    code = r'''

    assert( DR == 2);                  // 2d
    assert( NR[0] == 3 && NR[1] == 3); // must be 3x3

    PyArrayObject* R_array_contiguous = PyArray_GETCONTIGUOUS( R_array );
    quat_t q = quat_from_mat33d((double(*)[3])R_array_contiguous->data);
    Py_DECREF(R_array_contiguous);

    npy_intp out_dims[] = {4};
    PyArrayObject* out = (PyArrayObject*)PyArray_SimpleNew(1, out_dims, NPY_DOUBLE);
    ((double*)out->data)[0] = q.u;
    ((double*)out->data)[1] = q.x;
    ((double*)out->data)[2] = q.y;
    ((double*)out->data)[3] = q.z;

    return_val = (PyObject*)out;
'''

    return \
        weave.inline(code,
                     ['R'],
                     include_dirs=["/usr/include/maritime_robotics/pose/"],
                     headers=['"quat.h"'],

                     # I get important-looking warnings about using a deprecated
                     # API, and I don't want to see them.
                     extra_compile_args=["-Wno-cpp -Wno-unused-variable"])

def parse_args():
    parser = \
        argparse.ArgumentParser(description= \
    r'''Visualize stereo calibration geometry. The calibration files can be given as
    a directory or as a set of files directly''')
    parser.add_argument('--dir',
                        nargs=1,
                        help='Directory that contains the calibration')
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
        transforms = open('{}/transforms.txt'.format(args.dir[0]), 'r')
        cahvors = { 0: [open('{}/camera{}-{}.cahvor'.format(args.dir[0], 0, 0), 'r'),
                        open('{}/camera{}-{}.cahvor'.format(args.dir[0], 0, 1), 'r')],
                    1: [open('{}/camera{}-{}.cahvor'.format(args.dir[0], 1, 0), 'r'),
                        open('{}/camera{}-{}.cahvor'.format(args.dir[0], 1, 1), 'r')] }

    else:
        transforms = [f for f in args.cal_file if f.name.endswith('/transforms.txt')]
        if len(transforms) != 1:
            raise Exception("Exactly one transforms.txt should have been given")

        cahvors = {}
        for f in args.cal_file:
            if f.name.endswith('/transforms.txt'):
                continue

            m = re.match('.*/camera(\d+)-([01]).cahvor$', f.name)
            if not m:
                raise Exception("All cal_file args must be either 'transforms.txt' or 'camera-A-B.cahvor', but '{}' is neither".format(f.name))

            p,i = int(m.group(1)),int(m.group(2))
            if cahvors.get(p) is None:
                cahvors[p] = [None, None]
            if cahvors[p][i] is not None:
                raise Exception("cahvor for pair {}, camera {} given multiple times".format(p,i))
            cahvors[p][i] = f


    return transforms,cahvors

def parse_transforms(transforms):

    x = { 'ins2veh': None,
          'cam2ins': {} }

    for l in transforms:
        if l[0] == '\n' or l[0] == '#':
            continue

        re_f = '[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
        re_u = '\d+'
        re_d = '[-+]?\d+'
        re_s = '.+'

        re_pos  = '\(\s*({f})\s+({f})\s+({f})\s*\)'        .format(f=re_f)
        re_quat = '\(\s*({f})\s+({f})\s+({f})\s+({f})\s*\)'.format(f=re_f)
        m = re.match('\s*ins2veh\s*=\s*{p}\s*{q}\s*\n?$'.
                     format(u=re_u, p=re_pos, q=re_quat),
                     l)
        if m:
            if x['ins2veh'] is not None:
                raise("'{}' is corrupt: more than one ins2veh".format(transforms.name))

            x['ins2veh'] = ( np.array((float(m.group(1)),float(m.group(2)),float(m.group(3)))),
                             np.array((float(m.group(4)),float(m.group(5)),float(m.group(6)),float(m.group(7)))))
            continue

        m = re.match('\s*cam2ins\s*\[({u})\]\s*=\s*{p}\s*{q}\s*\n?$'.
                     format(u=re_u, p=re_pos, q=re_quat),
                     l)
        if m:
            i = int(m.group(1))
            if x['cam2ins'].get(i) is not None:
                raise("'{}' is corrupt: more than one cam2ins[{}]".format(transforms.name, i))

            x['cam2ins'][i] = ( np.array((float(m.group(2)),float(m.group(3)),float(m.group(4)))),
                                np.array((float(m.group(5)),float(m.group(6)),float(m.group(7)),float(m.group(8)))))
            continue

        raise Exception("'transforms.txt': I only know about 'ins2veh' and 'cam2ins' lines. Got '{}'".
                        format(l))

    if not all(e for e in x.values()):
        raise Exception("Transforms file '{}' incomplete. Missing values for: {}",
                        transforms.name,
                        [k for k in x.keys() if not x[k]])
    return x

def parse_cahvor(cahvor):

    x = { 'dimensions':  None,
          'C':           None,
          'A':           None,
          'H':           None,
          'V':           None,
          'O':           None,
          'R':           None,
          'E':           None }

    for l in cahvor:
        if l[0] == '\n' or l[0] == '#':
            continue

        re_f = '[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
        re_u = '\d+'
        re_d = '[-+]?\d+'
        re_s = '.+'

        m = re.match('\s*dimensions\s*=\s*({u})\s+({u})\s*\n?$'.format(u=re_u),
                     l,
                     flags=re.I)
        if m:
            if x['dimensions'] is not None:
                raise("'{}' is corrupt: more than one set of dimensions given".format(cahvor.name))

            x['dimensions'] = (int(m.group(1)),int(m.group(2)))
            continue

        m = re.match('\s*([cahvore])\s*=\s*({f})\s+({f})\s+({f})\s*\n?$'.format(f=re_f),
                     l,
                     flags=re.I)
        if m:
            what = m.group(1).upper()
            if x.get(what) is not None:
                raise("'{}' is corrupt: more than one '{}'".format(cahvor.name, what))

            x[what] = np.array((float(m.group(2)),float(m.group(3)),float(m.group(4))))
            continue

    # read through the whole file. Did I get all the things I want
    if not all(e is not None for e in x.values()):
        raise Exception("Cahvor file '{}' incomplete. Missing values for: {}".
                        format(cahvor.name,
                               [k for k in x.keys() if x[k] is None]))

    x['Hc'] = nps.inner(x['H'], x['A'])
    hshp = x['H'] - x['Hc'] * x['A']
    x['Hs'] = np.sqrt(nps.inner(hshp,hshp))
    x['Hp'] = hshp / x['Hs']

    x['Vc'] = nps.inner(x['V'], x['A'])
    vsvp = x['V'] - x['Vc'] * x['A']
    x['Vs'] = np.sqrt(nps.inner(vsvp,vsvp))
    x['Vp'] = vsvp / x['Vs']

    x['cam2pair_p'] = x['C']
    x['cam2pair_q'] = rot2quat( nps.transpose(np.array( (x['Hp'], x['Vp'], x['A'] ))) )

    return x

def parse_and_consolidate(transforms, cahvors):
    transforms = parse_transforms(transforms)
    for cahvor_pair_id in cahvors.keys():
        cahvors[cahvor_pair_id] = [ parse_cahvor(c) for c in cahvors[cahvor_pair_id] ]

    pair_ids = sorted(transforms['cam2ins'].keys())
    if pair_ids != sorted(cahvors.keys()):
        raise Exception("Mismatched camera pair IDs")

    pairs = {}
    for i in pair_ids:
        pair = {'cam2ins': transforms['cam2ins'][i]}

        for icam in range(len(cahvors[i])):
            pair[icam] = cahvors[i][icam]
        pairs[i] = pair

    ins2veh = transforms['ins2veh']

    return pairs,ins2veh

def extend_axes_for_plotting(axes):
    r'''Input is a 4x3 axes array: center, center+x, center+y, center+z. I transform
this into a 3x6 array that can be gnuplotted "with vectors", and into a 

    '''

    # first, copy the center 3 times
    out = nps.cat( axes[0,:],
                   axes[0,:],
                   axes[0,:] )

    # then append just the deviations to each row containing the center
    out = nps.glue( out, axes[1:,:] - axes[0,:], axis=-1)
    return out

def gen_plot_axes(transforms, label):
    r'''Given a list of transforms (applied to the reference set of axes in order)
and a label, return a list of plotting directives gnuplotlib understands

    '''
    axes = np.array( ((0,0,0),
                      (1,0,0),
                      (0,1,0),
                      (0,0,1),), dtype=float )

    for xform in transforms:
        axes = transform( *xform, X=axes )

    axes_forplotting = extend_axes_for_plotting(axes)

    l_axes = tuple(nps.transpose(axes_forplotting)) + \
        ({'with': 'vectors', 'tuplesize': 6},)
    l_labels = tuple(nps.transpose(axes*1.01)) + \
        (np.array((label,
                   'x', 'y', 'z')),
         {'with': 'labels', 'tuplesize': 4},)
    return l_axes, l_labels



def gen_pair_axes(pairs):
    r'''Given all my camera pairs, generate a list of tuples that can be passed to
gnuplotlib to plot my world'''


    def gen_one_pair_axes(ipair, pair):
        r'''Given ONE camera pair, generate a list of tuples that can be passed to
gnuplotlib to plot my world

        '''

        def gen_one_cam_axes(icam, cam, cam2ins):

            return gen_plot_axes( ((cam['cam2pair_p'], cam['cam2pair_q']),
                                   (cam2ins[0], cam2ins[1])),
                                  'pair{}-camera{}'.format(ipair, icam))



        return (a for icam in (0,1) for a in gen_one_cam_axes(icam, pair[icam], pair['cam2ins']))

    return (a for ipair,pair in pairs.items() for a in gen_one_pair_axes(ipair, pair))


transforms,cahvors = parse_args()
pairs,ins2veh      = parse_and_consolidate(transforms, cahvors)
plot_pairs         = gen_pair_axes(pairs)
plot_ins           = gen_plot_axes( (), "INS")

# flatten the lists
allplots = [ e for p in plot_pairs,plot_ins for e in p ]

gp.plot3d( *allplots, square=1, zinv=1, ascii=1, xlabel='x', ylabel='y', zlabel='z')

import time
time.sleep(100000)
