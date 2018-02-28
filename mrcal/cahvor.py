#!/usr/bin/python2

import re

import numpy     as np
import numpysane as nps

import cameramodel
import poseutils
import optimizer


r'''A wrapper around mrcal.cameramodel to interface with JPL's CAHVOR files and
transforms.txt files'''



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

def _read(f):
    r'''Reads a .cahvor file into a cameramodel

    The input is a an opened file'''


    re_f = '[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
    re_u = '\d+'
    re_d = '[-+]?\d+'
    re_s = '.+'

    # I parse all key=value lines into my dict as raw text. Further I
    # post-process some of these raw lines. The covariances don't fit into this
    # mold, since the values span multiple lines. But since I don't care about
    # covariances, I happily ignore them
    x = {}
    for l in f:
        if re.match('^\s*#|^\s*$', l):
            continue

        m = re.match('\s*(\w+)\s*=\s*(.+?)\s*\n?$',
                     l, flags=re.I)
        if m:
            key = m.group(1)
            if key in x:
                raise Exception("Reading '{}': key '{}' seen more than once".format(f.name,
                                                                                    m.group(1)))
            x[key] = m.group(2)


    # Done reading. Any values that look like numbers, I convert to numbers.
    for i in x:
        if re.match('{}$'.format(re_f), x[i]):
            x[i] = float(x[i])

    # I parse the fields I know I care about into numpy arrays
    for i in ('Dimensions','C','A','H','V','O','R','E',
              'DISTORTION_OPENCV4', 'DISTORTION_OPENCV5', 'DISTORTION_OPENCV8'):
        if i in x:
            if re.match('[0-9\s]+$', x[i]): totype = int
            else:                           totype = float
            x[i] = np.array( [ totype(v) for v in re.split('\s+', x[i])])

    # Now I sanity-check the results and call it done
    for k in ('Dimensions','C','A','H','V'):
        if not k in x:
            raise Exception("Cahvor file '{}' incomplete. Missing values for: {}".
                            format(f.name, k))


    if   'DISTORTION_OPENCV8' in x:
        distortions = x["DISTORTION_OPENCV8"]
        distortion_model = 'DISTORTION_OPENCV8'
    elif 'DISTORTION_OPENCV5' in x:
        distortions = x["DISTORTION_OPENCV5"]
        distortion_model = 'DISTORTION_OPENCV5'
    elif 'DISTORTION_OPENCV4' in x:
        distortions = x["DISTORTION_OPENCV4"]
        distortion_model = 'DISTORTION_OPENCV4'
    elif 'R' not              in x:
        distortions = np.array(())
        distortion_model = 'DISTORTION_NONE'
    else:
        # CAHVOR(E)

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

        if 'O' not in x:
            theta = 0
            phi   = 0
        else:
            o = nps.matmult( x['O'], R_toref )
            norm2_oxy = o[0]*o[0] + o[1]*o[1]
            if norm2_oxy < 1e-8:
                theta = 0
                phi   = 0
            else:
                theta = np.arctan2(o[1], o[0])
                phi   = np.arcsin( np.sqrt( norm2_oxy ) )

        if is_cahvore:
            # CAHVORE
            if 'E' not in x:
                raise Exception('Cahvor file {} LOOKS like a cahvore, but lacks the E'.format(f.name))
            R0,R1,R2 = x['R'].ravel()
            E0,E1,E2 = x['E'].ravel()

            distortions      = np.array((theta,phi,R0,R1,R2,E0,E1,E2,cahvore_linearity))
            distortion_model = 'DISTORTION_CAHVORE'

        else:
            # CAHVOR
            if 'E' in x:
                raise Exception('Cahvor file {} LOOKS like a cahvor, but has an E'.format(f.name))

            if abs(phi) < 1e-8 and \
               ( 'R' not in x or np.linalg.norm(x['R']) < 1e-8):
                # pinhole
                theta = 0
                phi   = 0
            else:
                R0,R1,R2 = x['R'].ravel()

            if theta == 0 and phi == 0:
                distortions = np.array(())
                distortion_model = 'DISTORTION_NONE'
            else:
                distortions = np.array((theta,phi,R0,R1,R2))
                distortion_model = 'DISTORTION_CAHVOR'


    m = cameramodel.cameramodel()
    m.intrinsics( (distortion_model,
                   nps.glue( np.array(_fxy_cxy(x)),
                             distortions,
                             axis = -1)))
    m.extrinsics_Rt(True, nps.glue(R_toref,t_toref, axis=-2))
    m.dimensions(x['Dimensions'])

    # I write the whole thing into my structure so that I can pull it out later
    m.set_cookie(x)

    return m

def read(f):
    r'''Reads a .cahvor file into a cameramodel

    The input is a filename or an opened file'''

    if f is None:
        return cameramodel.cameramodel()

    if type(f) is cameramodel:
        return f

    if type(f) is str:
        with open(f, 'r') as openedfile:
            return _read(openedfile)

    return _read(f)

def _write(f, m, note=None):
    r'''Writes a cameramodel as a .cahvor to a writeable file object'''

    if note is not None:
        f.write('# ' + note + '\n')
    d = m.dimensions()
    if d is not None:
        f.write('Dimensions = {} {}\n'.format(int(d[0]), int(d[1])))
    else:
        f.write('# this is arbitrary and hard-coded:\nDimensions = 3904 3904\n')

    distortion_model,intrinsics = m.intrinsics()
    if distortion_model == 'DISTORTION_CAHVOR':
        f.write("Model = CAHVOR = perspective, distortion\n")
    elif distortion_model == 'DISTORTION_CAHVORE':
        f.write("Model = CAHVORE3,{} = general\n".format(intrinsics[4+5+3]))
    elif re.match('DISTORTION_(OPENCV.*|NONE)', distortion_model):
        f.write("Model = CAHV = perspective, linear\n")
    else:
        raise Exception("Don't know how to handle distortion model '{}'".format(distortion_model))


    fx,fy,cx,cy = intrinsics[:4]
    Rt_toref = m.extrinsics_Rt(True)
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

    if re.match('DISTORTION_CAHVOR', distortion_model):
        # CAHVOR(E)
        theta,phi,R0,R1,R2 = intrinsics[4:9]

        sth,cth,sph,cph = np.sin(theta),np.cos(theta),np.sin(phi),np.cos(phi)
        O = nps.matmult( R_toref, nps.transpose(np.array(( sph*cth, sph*sth,  cph ))) ).ravel()
        R = np.array((R0, R1, R2))
        f.write(("{} =" + (" {:15.10f}" * 3) + "\n").format('O', *O))
        f.write(("{} =" + (" {:15.10f}" * 3) + "\n").format('R', *R))

        if 'DISTORTION_CAHVORE' == distortion_model:
            E = intrinsics[9:]
            f.write(("{} =" + (" {:15.10f}" * 3) + "\n").format('E', *E))

    elif re.match('DISTORTION_OPENCV*', distortion_model):
        Ndistortions = optimizer.getNdistortionParams(distortion_model)
        f.write(("{} =" + (" {:15.10f}" * Ndistortions) + "\n").format(distortion_model, *intrinsics[4:]))
    elif len(intrinsics) != 4:
        raise Exception("Somehow ended up with unwritten distortions. Nintrinsics={}, distortion_model={}".format(len(intrinsics), distortion_model))
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

    '''

    p = pq[:3]
    q = pq[3:]
    R = poseutils.R_from_quat(q)
    return nps.glue(R,p, axis=-2)


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

