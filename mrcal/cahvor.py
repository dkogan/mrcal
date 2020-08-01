#!/usr/bin/python3

from __future__ import print_function

import re

import numpy     as np
import numpysane as nps

import mrcal

r'''A wrapper around mrcal.cameramodel to interface with JPL's CAHVOR files and
transforms.txt files

The O in CAHVOR is an optical axis: 3 numbers representing a 2DOF quantity. I
store this as an unconstrainted 2-vector (alpha, beta)

I parametrize the optical axis such that
- o(alpha=0, beta=0) = (0,0,1) i.e. the optical axis is at the center
  if both parameters are 0
- The gradients are cartesian. I.e. do/dalpha and do/dbeta are both
  NOT 0 at (alpha=0,beta=0). This would happen at the poles (gimbal
  lock), and that would make my solver unhappy

So o = { s_al*c_be, s_be,  c_al*c_be }
'''



def _HVs_HVc_HVp(cahvor):
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

def _fxy_cxy(cahvor):
    r'''Given a cahvor dict returns a tuple containing (fx,fy,cx,cy)'''
    return _HVs_HVc_HVp(cahvor)[:4]

def _read(s, name):
    r'''Reads a .cahvor file into a cameramodel

    The input is the .cahvor file contents as a string'''


    re_f = '[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
    re_u = '\d+'
    re_d = '[-+]?\d+'
    re_s = '.+'

    # I parse all key=value lines into my dict as raw text. Further I
    # post-process some of these raw lines.
    x = {}
    for l in s.splitlines():
        if re.match('^\s*#|^\s*$', l):
            continue

        m = re.match('\s*(\w+)\s*=\s*(.+?)\s*\n?$',
                     l, flags=re.I)
        if m:
            key = m.group(1)
            if key in x:
                raise Exception("Reading '{}': key '{}' seen more than once".format(name,
                                                                                    m.group(1)))
            value = m.group(2)

            # for compatibility
            if re.match('^DISTORTION', key):
                key = key.replace('DISTORTION', 'LENSMODEL')

            x[key] = value


    # Done reading. Any values that look like numbers, I convert to numbers.
    for i in x:
        if re.match('{}$'.format(re_f), x[i]):
            x[i] = float(x[i])

    # I parse the fields I know I care about into numpy arrays
    for i in ('Dimensions','C','A','H','V','O','R','E',
              'LENSMODEL_OPENCV4',
              'LENSMODEL_OPENCV5',
              'LENSMODEL_OPENCV8',
              'LENSMODEL_OPENCV12',
              'VALID_INTRINSICS_REGION'):
        if i in x:
            # Any data that's composed only of digits and whitespaces (no "."),
            # use integers
            if re.match('[0-9\s]+$', x[i]): totype = int
            else:                           totype = float
            x[i] = np.array( [ totype(v) for v in re.split('\s+', x[i])], dtype=totype)

    # Now I sanity-check the results and call it done
    for k in ('Dimensions','C','A','H','V'):
        if not k in x:
            raise Exception("Cahvor file '{}' incomplete. Missing values for: {}".
                            format(name, k))


    is_cahvor_or_cahvore = False
    if 'LENSMODEL_OPENCV12' in x:
        distortions = x["LENSMODEL_OPENCV12"]
        lens_model = 'LENSMODEL_OPENCV12'
    elif 'LENSMODEL_OPENCV8' in x:
        distortions = x["LENSMODEL_OPENCV8"]
        lens_model = 'LENSMODEL_OPENCV8'
    elif 'LENSMODEL_OPENCV5' in x:
        distortions = x["LENSMODEL_OPENCV5"]
        lens_model = 'LENSMODEL_OPENCV5'
    elif 'LENSMODEL_OPENCV4' in x:
        distortions = x["LENSMODEL_OPENCV4"]
        lens_model = 'LENSMODEL_OPENCV4'
    elif 'R' not              in x:
        distortions = np.array(())
        lens_model = 'LENSMODEL_PINHOLE'
    else:
        is_cahvor_or_cahvore = True

    if 'VALID_INTRINSICS_REGION' in x:
        x['VALID_INTRINSICS_REGION'] = \
            x['VALID_INTRINSICS_REGION'].reshape( len(x['VALID_INTRINSICS_REGION'])//2, 2)

    # get extrinsics from cahvor
    if 'Model' not in x:
        x['Model'] = ''

    m = re.match('CAHVORE3,([0-9\.e-]+)\s*=\s*general',x['Model'])
    if m:
        is_cahvore = True
        cahvore_linearity = float(m.group(1))
    else:
        is_cahvore = False

    Hp,Vp = _HVs_HVc_HVp(x)[-2:]
    R_toref = nps.transpose( nps.cat( Hp,
                                      Vp,
                                      x['A'] ))
    t_toref = x['C']

    if is_cahvor_or_cahvore:
        if 'O' not in x:
            alpha = 0
            beta  = 0
        else:
            o     = nps.matmult( x['O'], R_toref )
            alpha = np.arctan2(o[0], o[2])
            beta  = np.arcsin( o[1] )

        if is_cahvore:
            # CAHVORE
            if 'E' not in x:
                raise Exception('Cahvor file {} LOOKS like a cahvore, but lacks the E'.format(name))
            R0,R1,R2 = x['R'].ravel()
            E0,E1,E2 = x['E'].ravel()

            distortions      = np.array((alpha,beta,R0,R1,R2,E0,E1,E2,cahvore_linearity), dtype=float)
            lens_model = 'LENSMODEL_CAHVORE'

        else:
            # CAHVOR
            if 'E' in x:
                raise Exception('Cahvor file {} LOOKS like a cahvor, but has an E'.format(name))

            if abs(beta) < 1e-8 and \
               ( 'R' not in x or np.linalg.norm(x['R']) < 1e-8):
                # pinhole
                alpha = 0
                beta  = 0
            else:
                R0,R1,R2 = x['R'].ravel()

            if alpha == 0 and beta == 0:
                distortions = np.array(())
                lens_model = 'LENSMODEL_PINHOLE'
            else:
                distortions = np.array((alpha,beta,R0,R1,R2), dtype=float)
                lens_model = 'LENSMODEL_CAHVOR'

    m = mrcal.cameramodel(imagersize = x['Dimensions'].astype(np.int32),
                          intrinsics = (lens_model, nps.glue( np.array(_fxy_cxy(x), dtype=float),
                                                                    distortions,
                                                                    axis = -1)),
                          valid_intrinsics_region = x.get('VALID_INTRINSICS_REGION'),
                          extrinsics_Rt_toref = np.ascontiguousarray(nps.glue(R_toref,t_toref, axis=-2)))
    return m

def read(f):
    r'''Reads a .cahvor file into a cameramodel

    The input is a filename or an opened file'''

    if type(f) is mrcal.cameramodel:
        return f

    if type(f) is str:
        with open(f, 'r') as openedfile:
            return _read(openedfile.read(), f)

    return _read(f.read(), f.name)

def read_from_string(s):
    return _read(s, "<string>")

def _write(f, m, note=None):
    r'''Writes a cameramodel as a .cahvor to a writeable file object'''

    if note is not None:
        for l in note.splitlines():
            f.write('# ' + l + '\n')
    d = m.imagersize()
    f.write('Dimensions = {} {}\n'.format(int(d[0]), int(d[1])))

    lens_model,intrinsics = m.intrinsics()
    if lens_model == 'LENSMODEL_CAHVOR':
        f.write("Model = CAHVOR = perspective, distortion\n")
    elif lens_model == 'LENSMODEL_CAHVORE':
        f.write("Model = CAHVORE3,{} = general\n".format(intrinsics[4+5+3]))
    elif re.match('LENSMODEL_(OPENCV.*|PINHOLE)', lens_model):
        f.write("Model = CAHV = perspective, linear\n")
    else:
        raise Exception("Don't know how to handle lens model '{}'".format(lens_model))


    fx,fy,cx,cy = intrinsics[:4]
    Rt_toref = m.extrinsics_Rt_toref()
    R_toref = Rt_toref[:3,:]
    t_toref = Rt_toref[ 3,:]

    C  = t_toref
    A  = R_toref[:,2]
    Hp = R_toref[:,0]
    Vp = R_toref[:,1]
    H  = fx*Hp + A*cx
    V  = fy*Vp + A*cy

    f.write(("{} =" + (" {:15.10f}" * 3) + "\n").format('C', *C))
    f.write(("{} =" + (" {:15.10f}" * 3) + "\n").format('A', *A))
    f.write(("{} =" + (" {:15.10f}" * 3) + "\n").format('H', *H))
    f.write(("{} =" + (" {:15.10f}" * 3) + "\n").format('V', *V))

    if re.match('LENSMODEL_CAHVOR', lens_model):
        # CAHVOR(E)
        alpha,beta,R0,R1,R2 = intrinsics[4:9]

        s_al,c_al,s_be,c_be = np.sin(alpha),np.cos(alpha),np.sin(beta),np.cos(beta)
        O = nps.matmult( R_toref, nps.transpose(np.array(( s_al*c_be, s_be, c_al*c_be ), dtype=float)) ).ravel()
        R = np.array((R0, R1, R2), dtype=float)
        f.write(("{} =" + (" {:15.10f}" * 3) + "\n").format('O', *O))
        f.write(("{} =" + (" {:15.10f}" * 3) + "\n").format('R', *R))

        if 'LENSMODEL_CAHVORE' == lens_model:
            E = intrinsics[9:]
            f.write(("{} =" + (" {:15.10f}" * 3) + "\n").format('E', *E))

    elif re.match('LENSMODEL_OPENCV*', lens_model):
        Ndistortions = mrcal.num_lens_params(lens_model) - 4
        f.write(("{} =" + (" {:15.10f}" * Ndistortions) + "\n").format(lens_model, *intrinsics[4:]))
    elif len(intrinsics) != 4:
        raise Exception("Somehow ended up with unwritten distortions. Nintrinsics={}, lens_model={}".format(len(intrinsics), lens_model))

    c = m.valid_intrinsics_region()
    if c is not None:
        f.write("VALID_INTRINSICS_REGION = ")
        np.savetxt(f, c.ravel(), fmt='%.2f', newline=' ')
        f.write('\n')

    Hs,Vs,Hc,Vc = intrinsics[:4]
    f.write("Hs = {}\n".format(Hs))
    f.write("Hc = {}\n".format(Hc))
    f.write("Vs = {}\n".format(Vs))
    f.write("Vc = {}\n".format(Vc))
    f.write("# this is hard-coded\nTheta = {} (-90.0 deg)\n".format(-np.pi/2))

    return True

def write(f, m, note=None):
    r'''Writes a cameramodel as a .cahvor to a filename or a writeable file object'''

    if type(f) is str:
        with open(f, 'w') as openedfile:
            return _write(openedfile, m, note)

    return _write(f, m)

def Rt_from_pq(pq):
    r'''Converts a pq transformation to an Rt transformation

    pq is a 7-long array: a 3-long translation followed by a 4-long unit
    quaternion.

    Rt is a (4,3) array: a (3,3) rotation matrix with a 3-long translation in
    the last row

    Broadcasting is supported

    '''

    p = pq[..., :3]
    q = pq[..., 3:]
    R = mrcal.R_from_quat(q)
    return nps.glue(R,
                    nps.dummy(p,-2),
                    axis=-2)

def pq_from_Rt(Rt):
    r'''Converts an Rt transformation to an pq transformation

    pq is a 7-long array: a 3-long translation followed by a 4-long unit
    quaternion.

    Rt is a (4,3) array: a (3,3) rotation matrix with a 3-long translation in
    the last row

    '''

    R = Rt[:3,:]
    t = Rt[ 3,:]
    q = mrcal.quat_from_R(R)
    return nps.glue(t,q, axis=-1)

def read_transforms(f):
    r'''Reads a file (a filename string, or a file-like object: an iterable
    containing lines of text) into a transforms dict, and returns the dict

    '''

    needclose = False
    if type(f) is str:
        filename = f
        f = open(filename, 'r')
        needclose = True

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

            x['veh_from_ins'] = Rt_from_pq( np.array((float(m.group(1)),float(m.group(2)),float(m.group(3)),
                                                      float(m.group(4)),float(m.group(5)),float(m.group(6)),float(m.group(7)))))
            continue

        m = re.match('\s*cam2ins\s*\[({u})\]\s*=\s*{p}\s*{q}\s*\n?$'.
                     format(u=re_u, p=re_pos, q=re_quat),
                     l)
        if m:
            i = int(m.group(1))
            if x['ins_from_camera'].get(i) is not None:
                raise("'{}' is corrupt: more than one 'cam2ins'[{}]".format(f.name, i))

            x['ins_from_camera'][i] = Rt_from_pq( np.array((float(m.group(2)),float(m.group(3)),float(m.group(4)),
                                                            float(m.group(5)),float(m.group(6)),float(m.group(7)),float(m.group(8)))))
            continue

        raise Exception("'transforms.txt': I only know about 'ins2veh' and 'cam2ins' lines. Got '{}'".
                        format(l))

    if not all(e is not None for e in x.values()):
        raise Exception("Transforms file '{}' incomplete. Missing values for: {}",
                        f.name,
                        [k for k in x.keys() if not x[k]])
    if needclose:
        f.close()

    return x

