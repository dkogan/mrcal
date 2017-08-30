#!/usr/bin/python2

import numpy as np
import numpysane as nps
import mrpose
import re
import sys
import copy

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

def write_cahvor(f, cahvor):
    r'''Given a filename or a writeable file and a cahvor dict, spits out the
cahvor'''

    needclose = False
    if type(f) is str:
        f = open(f, 'w+')
        needclose = True

    for i in ('Model','Dimensions','C','A','H','V','O','R','E'):
        if i in cahvor:
            f.write("{} = {}\n".format(i, cahvor[i] if type(cahvor[i]) is not np.ndarray else ' '.join(str(x) for x in cahvor[i])))

    Hs,Vs,Hc,Vc = cahvor_HVs_HVc(cahvor)
    f.write("Hs = {}\n".format(Hs))
    f.write("Hc = {}\n".format(Hc))
    f.write("Vs = {}\n".format(Vs))
    f.write("Vc = {}\n".format(Vc))

    if needclose:
        f.close()

def parse_and_consolidate(transforms, cahvors):
    transforms = parse_transforms(transforms)
    for cahvor_pair_id in cahvors.keys():
        cahvors[cahvor_pair_id] = [ parse_cahvor(c) for c in cahvors[cahvor_pair_id] ]

    pair_ids = sorted(transforms['ins_from_camera'].keys())
    if pair_ids != sorted(cahvors.keys()):
        raise Exception("Mismatched camera pair IDs")

    pairs = {}
    for i in pair_ids:
        pair = {'ins_from_camera': transforms['ins_from_camera'][i]}

        for icam in range(len(cahvors[i])):
            pair[icam] = cahvors[i][icam]
        pairs[i] = pair

    veh_from_ins = transforms['veh_from_ins']

    return pairs,veh_from_ins

def cahvor_HVs_HVc(cahvor):
    r'''Given a cahvor dict returns a tuple containing (Hs,Vs,Hc,Vc)'''

    Hc   = nps.inner(cahvor['H'], cahvor['A'])
    hshp = cahvor['H'] - Hc * cahvor['A']
    Hs   = np.sqrt(nps.inner(hshp,hshp))

    Vc   = nps.inner(cahvor['V'], cahvor['A'])
    vsvp = cahvor['V'] - Vc * cahvor['A']
    Vs   = np.sqrt(nps.inner(vsvp,vsvp))

    return Hs,Vs,Hc,Vc

def cahvor_fxy_cxy(cahvor):
    r'''Given a cahvor dict returns a tuple containing (fx,fy,cx,cy)'''
    return cahvor_HVs_HVc(cahvor)

def cahvor_pair_from_camera(cahvor):
    r'''Given a cahvor dict returns a pair_from_camera transformation'''
    Hs,Vs,Hc,Vc = cahvor_HVs_HVc(cahvor)

    Hp   = (cahvor['H'] - Hc * cahvor['A']) / Hs
    Vp   = (cahvor['V'] - Vc * cahvor['A']) / Vs

    return nps.glue( np.array( cahvor['C'] ),
                     mrpose.quat_from_mat33d( nps.transpose(np.array((Hp, Vp, cahvor['A'] )))),
                     axis=-1 )

def set_extrinsics(cahvor, pq):
    r'''Given a cahvor dict and a transformation, return another cahvor dict that
represents the first (as far as the intrinsics are concerned), but with the
extrinsics replaced by the given transformation. The transformation is given as
a (p,q) camera->pair tuple as usual

    '''

    p = pq[:3]
    q = pq[3:]
    R_pair_from_camera = mrpose.quat_to_mat33d(q)
    Hp,Vp,A            = nps.transpose(R_pair_from_camera)

    Hs,Vs,Hc,Vc = cahvor_HVs_HVc(cahvor)
    H           = Hs*Hp + Hc*A
    V           = Vs*Vp + Vc*A

    out = copy.deepcopy(cahvor)

    out['C'] = p
    out['A'] = A
    out['H'] = H
    out['V'] = V

    # I need to keep the intrinsics. This is simple, except the 'O' parameter
    # needs special attention. The O vector is an intrinsic parameter
    # representing the optical axis direction. In a distortion-less camera this
    # is identical to A. I need to separate the intrinsics and extrinsics I want
    # some sort of deviation-from-A representation. Thus I transform O into the
    # camera coord system, which is by definition extrinsic-less. When I'm done
    # moving the extrinsics, I put O back into the new coord system
    pq_pair_from_camera = cahvor_pair_from_camera(cahvor)
    q_pair_from_camera = pq_pair_from_camera[3:]
    if 'O' in cahvor:
        O_pair_old = cahvor['O']
        O_camera   = mrpose.vec3_rotate(mrpose.quat_conj(q_pair_from_camera),
                                        O_pair_old)
        O_pair_new = mrpose.vec3_rotate(q, O_camera)
        out['O'] = O_pair_new

    return out
