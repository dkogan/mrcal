#!/usr/bin/python2

import numpy as np
import numpysane as nps
import mrpose
import re
import sys
import copy
import cv2


def cahvor_HVs_HVc_HVp(cahvor):
    r'''Given a cahvor dict returns a tuple containing (Hs,Vs,Hc,Vc,Hp,Vp)'''

    Hc   = nps.inner(cahvor['H'], cahvor['A'])
    hshp = cahvor['H'] - Hc * cahvor['A']
    Hs   = np.sqrt(nps.inner(hshp,hshp))

    Vc   = nps.inner(cahvor['V'], cahvor['A'])
    vsvp = cahvor['V'] - Vc * cahvor['A']
    Vs   = np.sqrt(nps.inner(vsvp,vsvp))

    Hp   = hshp / Hs
    Vp   = vsvp / Vs

    return Hs,Vs,Hc,Vc,Hp,Vp

def cahvor_fxy_cxy(cahvor):
    r'''Given a cahvor dict returns a tuple containing (fx,fy,cx,cy)'''
    return cahvor_HVs_HVc_HVp(cahvor)[:4]

def _validate_cahvor(cahvor):
    r'''Confirm that a given model is valid'''

    if abs(np.linalg.norm(cahvor['A']) - 1) > 1e-8:
        raise Exception("cahvor.['A'] must be a unit vector. Instead head {} of length {}". \
                        format(cahvor['A'],
                               np.linalg.norm(cahvor['A'])))
    if 'O' in cahvor and \
       abs(np.linalg.norm(cahvor['O']) - 1) > 1e-8:
        raise Exception("cahvor.['O'] must be a unit vector. Instead head {} of length {}". \
                        format(cahvor['O'],
                               np.linalg.norm(cahvor['O'])))
    Hp,Vp = cahvor_HVs_HVc_HVp(cahvor)[-2:]
    if abs(nps.inner(Hp,Vp)) > 1e-8:
        raise Exception("Hp must be perpendicular to Vp. Instead inner(Hp,Vp) = {}. Full model: {}". \
                        format(nps.inner(Hp,Vp), cahvor))

    if np.linalg.norm(np.cross(Hp,Vp) - cahvor['A']) > 0.1:
        raise Exception("Hp,Vp,A must for a right-handed orthonormal basis. It looks left-handed. I have Hp={}, Vp={}, A={}; model={}". \
                        format(Hp,Vp,cahvor['A'],cahvor))
    return True

def parse_transforms(f):
    r'''Reads a file (an iterable containing lines of text) into a transforms dict,
    and returns the dict

    '''

    x = { 'veh_from_ins': None,

          # this is actually "pair" to ins
          'ins_from_camera': {} }

    for l in f:
        if re.match('^\s*#|^\s*$', l):
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
            if x['veh_from_ins'] is not None:
                raise("'{}' is corrupt: more than one 'ins2veh'".format(f.name))

            x['veh_from_ins'] = np.array((float(m.group(1)),float(m.group(2)),float(m.group(3)),
                                          float(m.group(4)),float(m.group(5)),float(m.group(6)),float(m.group(7))))
            continue

        m = re.match('\s*cam2ins\s*\[({u})\]\s*=\s*{p}\s*{q}\s*\n?$'.
                     format(u=re_u, p=re_pos, q=re_quat),
                     l)
        if m:
            i = int(m.group(1))
            if x['ins_from_camera'].get(i) is not None:
                raise("'{}' is corrupt: more than one 'cam2ins'[{}]".format(f.name, i))

            x['ins_from_camera'][i] = np.array((float(m.group(2)),float(m.group(3)),float(m.group(4)),
                                                float(m.group(5)),float(m.group(6)),float(m.group(7)),float(m.group(8))))
            continue

        raise Exception("'transforms.txt': I only know about 'ins2veh' and 'cam2ins' lines. Got '{}'".
                        format(l))

    if not all(e is not None for e in x.values()):
        raise Exception("Transforms file '{}' incomplete. Missing values for: {}",
                        f.name,
                        [k for k in x.keys() if not x[k]])
    return x

def parse_cahvor(f):
    r'''Reads a file (a filename string, or a file-like object: an iterable
    containing lines of text) into a cahvor dict, and returns the dict

    '''

    re_f = '[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
    re_u = '\d+'
    re_d = '[-+]?\d+'
    re_s = '.+'

    needclose = False
    if type(f) is str:
        filename = f
        f = open(filename, 'r')
        needclose = True

    # I parse all key=value lines into my dict as raw text. Further I
    # post-process some of these raw lines. The covariances don't fit into this
    # mold, since the values span multiple lines. But since I don't care about
    # covariances, I happily ignore them
    x = {}
    for l in f:
        if re.match('^\s*#|^\s*$', l):
            continue

        m = re.match('\s*(\w+)\s*=\s*(.+?)\s*\n?$'.format(u=re_u),
                     l, flags=re.I)
        if m:
            key = m.group(1).title()
            if key in x:
                raise Exception("Reading '{}': key '{}' seen more than once".format(f.name,
                                                                                    m.group(1)))
            x[key] = m.group(2)


    # Done reading. Any values that look like numbers, I convert to numbers.
    for i in x:
        if re.match('{}$'.format(re_f), x[i]):
            x[i] = float(x[i])

    # I parse the fields I know I care about into numpy arrays
    for i in ('Dimensions','C','A','H','V','O','R','E'):
        if i in x:
            if re.match('[0-9\s]+$', x[i]): totype = int
            else:                           totype = float
            x[i] = np.array( [ totype(v) for v in re.split('\s+', x[i])])

    # Now I sanity-check the results and call it done
    for k in ('Dimensions','C','A','H','V'):
        if not k in x:
            raise Exception("Cahvor file '{}' incomplete. Missing values for: {}".
                            format(f.name, k))

    # I want to cut back on derived data in this structure to prevent confusion,
    # so I delete things I KNOW I want to skip
    for k in ('Hs','Vs','Hc','Vc','Theta'):
        if k in x:
            del x[k]

    if needclose:
        f.close()

    return x

def parse_and_consolidate(transforms, cahvors):
    transforms = parse_transforms(transforms)
    for i_pair in cahvors.keys():
        for i_cam in cahvors[i_pair]:
            cahvors[i_pair][i_cam] = parse_cahvor(cahvors[i_pair][i_cam])

    pair_ids = sorted(transforms['ins_from_camera'].keys())
    if pair_ids != sorted(cahvors.keys()):
        raise Exception("Mismatched camera pair IDs. transforms.txt knows about pairs {}, but I have cahvors for pairs {}".format(pair_ids,cahvors.keys()))

    pairs = {}
    for i in pair_ids:
        pair = {'ins_from_camera': transforms['ins_from_camera'][i]}

        for icam in range(len(cahvors[i])):
            pair[icam] = cahvors[i][icam]
        pairs[i] = pair

    veh_from_ins = transforms['veh_from_ins']

    return pairs,veh_from_ins

def write_cahvor(f, cahvor):
    r'''Given a (filename or a writeable file) and a cahvor dict, spits out the
    cahvor

    '''

    _validate_cahvor(cahvor)

    needclose = False
    if type(f) is str:
        f = open(f, 'w+')
        needclose = True

    for i in ('Model','Dimensions','C','A','H','V','O','R','E'):
        if i in cahvor:
            N = len(cahvor[i])
            f.write(("{} =" + (" {:15.10f}" * N) + "\n").format(i, *cahvor[i]))

    Hs,Vs,Hc,Vc = cahvor_HVs_HVc_HVp(cahvor)[:4]
    f.write("Hs = {}\n".format(Hs))
    f.write("Hc = {}\n".format(Hc))
    f.write("Vs = {}\n".format(Vs))
    f.write("Vc = {}\n".format(Vc))
    f.write("Theta = {} (-90.0 deg) # this is hard-coded\n".format(-np.pi/2))

    if needclose:
        f.close()

def get_intrinsics(cahvor):
    r'''Pull the intrinsics from a cahvor dict

    The intrinsics are a numpy array containing
      - fx
      - fy
      - cx
      - cy
      - theta (for computing O)
      - phi   (for computing O)
      - R0
      - R1
      - R2

    If O == A and R == 0, we have a distortion-less camera, and the distortion
    parameters (theta, phi, R0, R1, R2) are omitted.

    '''

    fx,fy,cx,cy = cahvor_fxy_cxy(cahvor)

    Rt = get_extrinsics_Rt_toref(cahvor)
    R  = Rt[:3,:]

    if 'O' not in cahvor:
        theta = 0
        phi   = 0
    else:
        o = nps.matmult( cahvor['O'], R )
        norm2_oxy = o[0]*o[0] + o[1]*o[1]
        if norm2_oxy < 1e-8:
            theta = 0
            phi   = 0
        else:
            theta = np.arctan2(o[1], o[0])
            phi   = np.arcsin( np.sqrt( norm2_oxy ) )

    if abs(phi) < 1e-8 and \
       ( 'R' not in cahvor or np.linalg.norm(cahvor['R']) < 1e-8):
        # pinhole
        intrinsics = np.array((fx,fy,cx,cy))
    else:
        R0,R1,R2 = cahvor['R'].ravel()
        intrinsics = np.array((fx,fy,cx,cy,theta,phi,R0,R1,R2))

    return intrinsics

def get_extrinsics_Rt_toref(cahvor):
    r'''Pull the extrinsics from a cahvor dict

    Returns the extrinsics in a Rt representation: a (4,3)-shape numpy array of
    a rotation represented as a 3x3 matrix in the [:3,:] slice, concatenated
    with the translation in the [3,:] slice. This is a transformation FROM
    points in this camera TO the ref coordinate system

    '''

    Hp,Vp = cahvor_HVs_HVc_HVp(cahvor)[-2:]

    R = nps.transpose( nps.cat( Hp,
                                Vp,
                                cahvor['A'] ))
    t = cahvor['C']

    return nps.glue(R,t, axis=-2);

def get_extrinsics_pq_toref(cahvor):
    r'''Pull the extrinsics from a cahvor dict

    Returns the extrinsics in a pq representation: a 7-long numpy array of 3D
    translation followed by a 4D unit quaternion rotation. This is a
    transformation FROM points in this camera TO the ref coordinate system

    '''

    Rt = get_extrinsics_Rt_toref(cahvor)
    R  = Rt[:3,:]
    t  = Rt[ 3,:]

    p = t
    q = mrpose.quat_from_mat33d( R )
    return nps.glue(p,q, axis=-1)

def get_extrinsics_rt_fromref(cahvor):
    r'''Pull the extrinsics from a cahvor dict

    Returns the extrinsics in a rt representation: a 6-long numpy array of a 3D
    Rodrigues rotation followed by a 3D translation. This is a transformation
    FROM the ref coordinate system TO this camera

    '''

    Rt = get_extrinsics_Rt_toref(cahvor)
    R  = Rt[:3,:]
    t  = Rt[ 3,:]

    # My transformation is in the reverse direction from what I want.
    # a = Rb + t -> b = R'(a-t) = R'a - R't
    t = -nps.matmult(t, R)
    R = nps.transpose(R)

    r = cv2.Rodrigues(R)[0]
    return nps.glue(r,t, axis=-1)

def assemble_cahvor( intrinsics, extrinsics = None ):
    r'''Produces a CAHVOR model from separate intrinsics and extrinsics

    The inputs:

    - intrinsics: a numpy array containing
      - fx
      - fy
      - cx
      - cy
      - theta (for computing O)
      - phi   (for computing O)
      - R0
      - R1
      - R2

      The distortion parameters (theta, phi, R0, R1, R2) is optional.

    - extrinsics: a transformation, the interpretation of which depends on the
      argument. Supported options are:

      - None: the identity transform is assumed

      - a numpy array of shape (4,3): Rt_toref. A rotation represented as a 3x3
        matrix in the [:3,:] slice, concatenated with the translation in the
        [3,:] slice. This is a transformation FROM points in this camera TO the
        ref coordinate system

      - a numpy array of shape (7,): pq_toref. A 3D translation followed by a 4D
        unit quaternion rotation. This is a transformation FROM points in this
        camera TO the ref coordinate system

      - a numpy array of shape (6,): rt_fromref. A 3D Rodrigues rotation
        followed by a 3D translation. This is a transformation FROM the ref
        coordinate system TO this camera

    '''
    if len(intrinsics) == 4:
        pinhole = True
    elif len(intrinsics) == 9:
        pinhole = False
    else:
        raise Exception("I know how to deal with an ideal camera (4 intrinsics) or a cahvor camera (9 intrinsics), but I got {} intrinsics intead".format(len(intrinsics)))

    # I parse the extrinsics, and convert them into Rt_toref form
    if extrinsics is None:
        R = np.eye(3)
        t = np.zeros(3)
    elif isinstance( extrinsics, np.ndarray ):
        if extrinsics.shape == (4,3):
            # Rt_toref
            R = extrinsics[:3,:]
            t = extrinsics[ 3,:]
        elif extrinsics.shape == (7,):
            # pq_toref
            p = extrinsics[:3]
            q = extrinsics[3:]

            R = mrpose.quat_to_mat33d(q)
            t = p
        elif extrinsics.shape == (6,):
            # rt_fromref
            r = extrinsics[:3]
            t = extrinsics[3:]

            # My transformation is in the reverse direction from what I want.
            # a = Rb + t -> b = R'(a-t) = R'a - R't
            R = cv2.Rodrigues(r)[0]
            t = -nps.matmult(t, R)
            R = nps.transpose(R)
        else:
            raise Exception("extrinsics must have shape (4,3) or (7,) or (6,) but got {}".format(extrinsics.shape))
    else:
        raise Exception("Extrinsics MUST be None of a numpy array. Instead got value of type '{}'".format(type(extrinsics)))

    fx,fy,cx,cy = intrinsics[:4]

    # Now we can establish the geometry. C is the camera origin, A is the +z
    # direction, Hp is the +x direction and Vp is the +y direction, all in the
    # ref-coord-system
    out      = {}
    out['C'] = t
    out['A'] = R[:,2]
    Hp       = R[:,0]
    Vp       = R[:,1]
    out['H'] = fx*Hp + out['A']*cx
    out['V'] = fy*Vp + out['A']*cy

    if not pinhole:
        theta,phi,R0,R1,R2 = intrinsics[4:]

        sth,cth,sph,cph = np.sin(theta),np.cos(theta),np.sin(phi),np.cos(phi)
        out['O'] = nps.matmult( R, nps.transpose(np.array(( sph*cth, sph*sth,  cph ))) ).ravel()
        out['R'] = np.array((R0, R1, R2))

    _validate_cahvor(out)
    return out

