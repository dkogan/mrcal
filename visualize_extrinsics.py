#!/usr/bin/python2

import numpy as np
import numpysane as nps
import sys
import argparse
import re
from scipy import weave
import camera_models

# I need at least gnuplotlib 0.16 (got {}). That version fixed label plotting.
# Don't know how to ask python to check. Tried
# pkg_resources.get_distribution("gnuplotlib").version, but that reports the
# wrong value
import gnuplotlib as gp




r'''This tool reads in stereo calibration files (the .cahvor files and the
transforms.txt) and makes a plot that shows the world, the cameras, the ins and
the vehicle coord systems. This should simplify debugging'''


def transform(pq, X, do_inverse=False):
    """Invoke a C program to apply a pose transformation. p is a 3d position vector
and q is a 4d unit quaternion to represent the rotation. 'X' is an Nx3 numpy
array (N vectors) and transformed N vectors in a new Nx3 are returned

    """

    p,q = pq[0].astype(float), pq[1].astype(float)
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
                     include_dirs=["/usr/include/maritime_robotics/pose/"],
                     headers=['"pose3.h"'],

                     # I get important-looking warnings about using a deprecated
                     # API, and I don't want to see them.
                     extra_compile_args=["-Wno-cpp -Wno-unused-variable"])

def pose_inv(pq):
    r'''Inverts a pose. Wrapper for pose3_inv()'''

    p,q = pq[0].astype(float), pq[1].astype(float)

    code = r'''

    pose3_t pose = {.pos = {.x = P1(0),
                            .y = P1(1),
                            .z = P1(2)},
                    .rot = {.u = Q1(0),
                            .x = Q1(1),
                            .y = Q1(2),
                            .z = Q1(3)}};

    pose = pose3_inv(pose);

    // Done. Now return the new pose

    npy_intp shape3[] = {3};
    npy_intp shape4[] = {4};
    PyArrayObject* pout = (PyArrayObject*)PyArray_SimpleNew(1, shape3, NPY_DOUBLE);
    PyArrayObject* qout = (PyArrayObject*)PyArray_SimpleNew(1, shape4, NPY_DOUBLE);

    *(double*)(pout->data + 0*pout->strides[0]) = pose.pos.x;
    *(double*)(pout->data + 1*pout->strides[0]) = pose.pos.y;
    *(double*)(pout->data + 2*pout->strides[0]) = pose.pos.z;

    *(double*)(qout->data + 0*qout->strides[0]) = pose.rot.u;
    *(double*)(qout->data + 1*qout->strides[0]) = pose.rot.x;
    *(double*)(qout->data + 2*qout->strides[0]) = pose.rot.y;
    *(double*)(qout->data + 3*qout->strides[0]) = pose.rot.z;

    PyObject* out = Py_BuildValue("NN", pout, qout);
    return_val = out;
'''

    return \
        weave.inline(code,
                     ['p','q'],
                     include_dirs=["/usr/include/maritime_robotics/pose/"],
                     headers=['"pose3.h"'],

                     # I get important-looking warnings about using a deprecated
                     # API, and I don't want to see them.
                     extra_compile_args=["-Wno-cpp -Wno-unused-variable"])

def pose_mul(pq1,  pq2):
    r'''Multiplies two pose. Wrapper for pose_mul()'''

    p1,q1 = (pq1[0].astype(float), pq1[1].astype(float))
    p2,q2 = (pq2[0].astype(float), pq2[1].astype(float))

    code = r'''

    pose3_t pose1 = {.pos = {.x = P11(0),
                             .y = P11(1),
                             .z = P11(2)},
                     .rot = {.u = Q11(0),
                             .x = Q11(1),
                             .y = Q11(2),
                             .z = Q11(3)}};
    pose3_t pose2 = {.pos = {.x = P21(0),
                             .y = P21(1),
                             .z = P21(2)},
                     .rot = {.u = Q21(0),
                             .x = Q21(1),
                             .y = Q21(2),
                             .z = Q21(3)}};

    pose1 = pose3_mul(pose1, pose2);

    // Done. Now return the new pose

    npy_intp shape3[] = {3};
    npy_intp shape4[] = {4};
    PyArrayObject* pout = (PyArrayObject*)PyArray_SimpleNew(1, shape3, NPY_DOUBLE);
    PyArrayObject* qout = (PyArrayObject*)PyArray_SimpleNew(1, shape4, NPY_DOUBLE);

    *(double*)(pout->data + 0*pout->strides[0]) = pose1.pos.x;
    *(double*)(pout->data + 1*pout->strides[0]) = pose1.pos.y;
    *(double*)(pout->data + 2*pout->strides[0]) = pose1.pos.z;

    *(double*)(qout->data + 0*qout->strides[0]) = pose1.rot.u;
    *(double*)(qout->data + 1*qout->strides[0]) = pose1.rot.x;
    *(double*)(qout->data + 2*qout->strides[0]) = pose1.rot.y;
    *(double*)(qout->data + 3*qout->strides[0]) = pose1.rot.z;

    PyObject* out = Py_BuildValue("NN", pout, qout);
    return_val = out;
'''

    return \
        weave.inline(code,
                     ['p1','q1','p2','q2'],
                     include_dirs=["/usr/include/maritime_robotics/pose/"],
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


    if args.wld_from_ins and args.wld_from_pair:
        raise Exception("At most one of --wld_from_ins and --wld_from_pair can be given")

    i_wld_from_pair,wld_from_pair,wld_from_ins = None,None,None
    if args.wld_from_pair:
        i_wld_from_pair = int(args.wld_from_pair[0])
        p = np.array([float(x) for x in args.wld_from_pair[1:4]])
        q = np.array([float(x) for x in args.wld_from_pair[4:]])
        if np.abs(nps.inner(q,q) - 1) > 1e-5:
            raise Exception("wld_from_pair given a non-unit quaternion rotation: {}".format(q))
        wld_from_pair = (p,q)

    if args.wld_from_ins:
        p = np.array([float(x) for x in args.wld_from_ins[:3]])
        q = np.array([float(x) for x in args.wld_from_ins[3:]])
        if np.abs(nps.inner(q,q) - 1) > 1e-5:
            raise Exception("wld_from_ins given a non-unit quaternion rotation: {}".format(q))
        wld_from_ins = (p,q)

    return transforms,cahvors,wld_from_ins,i_wld_from_pair,wld_from_pair

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

def gen_plot_axes(transforms, label, scale = 1.0, label_offset = None):
    r'''Given a list of transforms (applied to the reference set of axes in order)
and a label, return a list of plotting directives gnuplotlib understands

    '''
    axes = np.array( ((0,0,0),
                      (1,0,0),
                      (0,1,0),
                      (0,0,1),), dtype=float ) * scale

    for xform in transforms:
        axes = transform( xform, X=axes )

    axes_forplotting = extend_axes_for_plotting(axes)

    l_axes = tuple(nps.transpose(axes_forplotting)) + \
        ({'with': 'vectors', 'tuplesize': 6},)

    l_labels = tuple(nps.transpose(axes*1.01 + \
                                   (label_offset if label_offset is not None else 0))) + \
        (np.array((label,
                   'x', 'y', 'z')),
         {'with': 'labels', 'tuplesize': 4},)
    return l_axes, l_labels

def gen_pair_axes(pairs, ins2global):
    r'''Given all my camera pairs, generate a list of tuples that can be passed to
gnuplotlib to plot my world'''


    def gen_one_pair_axes(ipair, pair):
        r'''Given ONE camera pair, generate a list of tuples that can be passed to
gnuplotlib to plot my world

        '''

        def gen_one_cam_axes(icam, cam, cam2ins):

            return gen_plot_axes( (cahvor_pair_from_camera(cam), cam2ins, ins2global),
                                  'pair{}-camera{}'.format(ipair, icam),
                                  scale = 0.5,
                                  label_offset=0.05)


        individual_cam_axes = (e for icam in (0,1) for e in gen_one_cam_axes(icam, pair[icam], pair['cam2ins']))
        pair_axes           = gen_plot_axes( (pair['cam2ins'], ins2global),
                                             'pair{}'.format(ipair),
                                             scale = 0.75)

        return (e for axes in (individual_cam_axes,pair_axes) for e in axes)

    return (a for ipair,pair in pairs.items() for a in gen_one_pair_axes(ipair, pair))

def gen_ins_axes(ins2global):
    return gen_plot_axes( (ins2global,), "INS")






transforms,cahvors, \
    wld_from_ins,i_wld_from_pair,wld_from_pair = parse_args()

pairs,veh_from_ins = camera_models.parse_and_consolidate(transforms, cahvors)

if wld_from_pair is not None:
    wld_from_ins = pose_mul(wld_from_pair, pose_inv(pairs[i_wld_from_pair]['ins_from_camera']))

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
