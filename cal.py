#!/usr/bin/python2

import sys
sys.path[:0] = ('/home/dima/projects/numpysane/',)

import numpy as np
import numpysane as nps
import gnuplotlib as gp
import cv2
import re



def align3d_procrustes(A, B):
    r"""Given two sets of 3d points in numpy arrays of shape (3,N), find the optimal
rotation to align these sets of points. I.e. minimize

  E = sum( norm2( a_i - (R b_i + t)))

We can expand this to get

  E = sum( norm2(a_i) - 2 inner(a_i, R b_i + t ) + norm2(b_i) + 2 inner(R b_i,t) + norm2(t) )

ignoring factors not depending on the optimization variables (R,t)

  E ~ sum( - 2 inner(a_i, R b_i + t ) + 2 inner(R b_i,t) + norm2(t) )

  dE/dt = sum( - 2 a_i + 2 R b_i + 2 t ) = 0
  -> sum(a_i) = R sum(b_i) + N t -> t = mean(a) - R mean(b)

I can shift my a_i and b_i so that they have 0-mean. In this case, the optimal t
= 0 and

  E ~ sum( inner(a_i, R b_i )  )

This is the classic procrustes problem

  E = tr( At R B ) = tr( R B At ) = tr( R U S Vt ) = tr( Vt R U S )

So the critical points are at Vt R U = I and R = V Ut, modulo a tweak to make
sure that R is in SO(3) not just in SE(3)

    """

    # allow (3,N) and (N,3)
    if A.shape[0] != 3 and A.shape[1] == 3:
        A = nps.transpose(A)
    if B.shape[0] != 3 and B.shape[1] == 3:
        B = nps.transpose(B)

    M = nps.matmult(               B - np.mean(B, axis=-1)[..., np.newaxis],
                     nps.transpose(A - np.mean(A, axis=-1)[..., np.newaxis]) )
    U,S,Vt = np.linalg.svd(M)

    R = nps.matmult(U, Vt)

    # det(R) is now +1 or -1. If it's -1, then this contains a mirror, and thus
    # is not a physical rotation. I compensate by negating the least-important
    # pair of singular vectors
    if np.linalg.det(R) < 0:
        U[:,2] *= -1
        R = nps.matmult(U, Vt)

    # I wanted V Ut, not U Vt
    R = nps.transpose(R)

    # Now that I have my optimal R, I compute the optimal t. From before:
    #
    #   t = mean(a) - R mean(b)
    t = np.mean(A, axis=-1)[..., np.newaxis] - nps.matmult( R, np.mean(B, axis=-1)[..., np.newaxis] )
    t = t.ravel()

    return R,t

def get_full_object(W, H, dot_spacing):
    xx,yy       = np.meshgrid( np.arange(W,dtype=float), np.arange(H,dtype=float))
    full_object = nps.mv( nps.cat(xx,yy), 0, -1) # shape (H,W,2)
    full_object = nps.mv( full_object,-3, -2)    # shape (W,H,2)
    return full_object * dot_spacing

def read_observations_from_file(filename, which):
    r"""Given a xxx.dots file, read the observations into a numpy array. Returns this
numpy array and a list of metadata.

The array has axes: (iframe, idot3d_x, idot3d_y, idot2d_xy)

So as an example, the observed pixel coord of the dot (3,4) in frame index 5 is
the 2-vector dots[5,3,4,:]

The metadata is a dictionary, containing the dimensions of the imager, and the
indices of frames that the numpy array contains

    """

    re_f = '[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
    re_u = '\d+'
    re_d = '[-+]?\d+'
    re_s = '.+'

    with open(filename, 'r') as f:
        for l in f:
            if re.match('# Format: jplv$',
                        l):
                break
        else:
            raise Exception('No explicit "Format: jplv" when reading {}'.format(filename))

        # Data. Axes: (iframe, idot3d_x, idot3d_y, idot2d_xy)
        # So the observed pixel coord of the dot (3,4) in frame index 5 is
        # the 2-vector dots[5,3,4,:]
        dots       = np.array( (), dtype=float)
        metadata   = {'frames': [],
                      'imager_size': None,
                      'dot_spacing': None}
        cur_frame  = None
        cur_iframe = None

        for l in f:
            if l[0] == '\n' or l[0] == '#':
                continue

            m = re.match('IMAGE ({u}) ({u})'.format(u=re_u),
                         l)
            if m:
                if metadata['imager_size'] is not None:
                    raise Exception('Got line "{}", but already have width, height'.format(l))
                metadata['imager_size'] = (int(m.group(1)), int(m.group(2)))
                continue

            m = re.match('DOT ({s}) (stcal-({u})-({s}).pnm) FIX ({u}) ({u}) ({u}) ({f}) ({f}) IMG ({f}) ({f})'.format(f=re_f, u=re_u, d=re_d, s=re_s), l)
            if m:
                if which != m.group(1):
                    raise Exception("Reading file {}: expected '{}' frames, but got '{}". \
                                    format(filename, which, m.group(1)))
                if which != m.group(4):
                    raise Exception("Reading file {}: expected '{}' frames, but got image file '{}". \
                                    format(filename, which, m.group(2)))
                frame  = int(m.group(3))
                iframe = int(m.group(5))
                idot3d = (int(   m.group(6)),  int(  m.group(7)))
                dot3d  = (float( m.group(8)),  float(m.group(9)))
                dot2d  = np.array(( float(m.group(10)), float(m.group(11))))

                if cur_iframe == iframe and \
                   cur_frame  != frame:
                    raise Exception('frame changed, but iframe did not')
                if cur_frame  == frame and \
                   cur_iframe != iframe:
                    raise Exception('iframe changed, but frame did not')
                if cur_iframe is None and iframe != 0:
                    raise Exception('iframe should start at 0')

                if cur_iframe != iframe:
                    if cur_iframe is not None and cur_iframe+1 != iframe:
                        raise Exception('non-consecutive frame index...')
                    if cur_frame is not None and cur_frame >= frame:
                        raise Exception('non-increasing frame number...')

                    cur_frame,cur_iframe = frame,iframe
                    dots = nps.glue( dots,
                                     np.zeros((10,10,2), dtype=float) - 1,
                                     axis=-4 )
                    metadata['frames'].append(frame)

                dot_spacing = np.array(dot3d, dtype=float) / np.array(idot3d, dtype=float)
                if metadata['dot_spacing'] is None:
                    metadata['dot_spacing'] = dot_spacing
                else:
                    if np.max( np.abs(metadata['dot_spacing'] - dot_spacing) ) > 1e-4:
                        raise Exception("Inconsistent dot spacing. Previously saw {} but now have {}". \
                                        format(metadata['dot_spacing'], dot_spacing))

                dots[-1, idot3d[0]-1, idot3d[1]-1,:] = dot2d
                continue

            raise Exception('Got unknown line "{}"'.format(l))

    return dots, metadata

def estimate_local_calobject_poses( dots, dot_spacing, focal, imager_size ):
    r"""Given observations, and an estimate of camera intrinsics (focal lengths,
imager size) computes an estimate of the pose of the calibration object in
respect to the camera. This assumes a pinhole camera, and all the work is done
by the solvePnP() openCV call.

The observations are given in a numpy array with axes:

  (iframe, idot3d_x, idot3d_y, idot2d_xy)

So as an example, the observed pixel coord of the dot (3,4) in frame index 5 is
the 2-vector dots[5,3,4,:]

Missing observations are given as negative pixel coords

    """

    camera_matrix = np.array((( focal[0], 0,        imager_size[0]/2), \
                              (        0, focal[1], imager_size[1]/2), \
                              (        0,        0,                 1)))

    full_object = get_full_object(10, 10, dot_spacing)

    # look through each frame
    Rall = np.array(())
    tall = np.array(())

    for d in dots:
        d = nps.transpose(nps.clump( nps.mv( nps.glue(d, full_object, axis=-1), -1, -3 ), n=2 ))
        # d is (100,4). I pick off those rows where the observations are both >=
        # 0. Result should be (N,4) where N <= 100
        i = (d[..., 0] >= 0) * (d[..., 1] >= 0)
        d = d[i,:]

        observations = d[:,:2]
        ref_object   = nps.glue(d[:,2:], np.zeros((d.shape[0],1)), axis=-1)
        result,rvec,tvec = cv2.solvePnP(ref_object  [..., np.newaxis],
                                        observations[..., np.newaxis],
                                        camera_matrix, None)
        if not result:
            raise Exception("solvePnP failed!")
        if tvec[2] <= 0:
            raise Exception("solvePnP says that tvec.z <= 0. Maybe needs a flip, but please examine this")

        R,_ = cv2.Rodrigues(rvec)
        Rall = nps.glue(Rall, R,            axis=-3)
        tall = nps.glue(tall, tvec.ravel(), axis=-2)

    return Rall,tall

def estimate_camera_pose( calobject_poses, dots, dot_spacing, metadata ):

    if sorted(metadata.keys()) != ['left', 'right']:
        raise Exception("metadata dict has unknown keys: {}".format(metadata.keys()))

    f0 = metadata['left' ]['frames']
    f1 = metadata['right']['frames']
    if f0 != f1:
        raise Exception("Inconsistent frame sets. This is from my quick implementation. Please fix")

    Nframes = len(f0)

    A = np.array(())
    B = np.array(())

    R0 = calobject_poses['left' ][0]
    t0 = calobject_poses['left' ][1]
    R1 = calobject_poses['right'][0]
    t1 = calobject_poses['right'][1]

    full_object = get_full_object(10, 10, dot_spacing)

    for iframe in xrange(Nframes):

        # d looks at one frame and has shape (10,10,6)
        d = nps.glue( dots['left'][iframe], dots['right'][iframe], full_object, axis=-1 )

        # squash dims so that d is (100,6)
        d = nps.transpose(nps.clump(nps.mv(d, -1, -3), n=2))

        # I pick out those points that have observations in both frames
        i = (d[..., 0] >= 0) * (d[..., 1] >= 0) * (d[..., 2] >= 0) * (d[..., 3] >= 0)
        d = d[i,:]

        # ref_object is (3,N)
        ref_object = nps.transpose(nps.glue(d[:,4:], np.zeros((d.shape[0],1)), axis=-1))

        A = nps.glue(A, nps.transpose(nps.matmult( R0[iframe], ref_object)) + t0[iframe],
                     axis = -2)
        B = nps.glue(B, nps.transpose(nps.matmult( R1[iframe], ref_object)) + t1[iframe],
                     axis = -2)


    return align3d_procrustes(A, B)



focal_estimate = 1950 # pixels

calobject_poses = {}
dots            = {}
metadata        = {}

for fil in ('/home/dima/data/cal_data_2017_07_14/lfc4/stcal-left.dots',
            '/home/dima/data/cal_data_2017_07_14/lfc4/stcal-right.dots'):
    m = re.match( '.*-(left|right)\.dots$', fil)
    if not m: raise Exception("Can't tell if '{}' is left or right".format(fil))
    which = m.group(1)

    d,m             = read_observations_from_file( fil, which )
    dots    [which] = d
    metadata[which] = m

    # I compute an estimate of the poses of the calibration object in the local
    # coord system of each camera for each frame. This is done for each frame
    # and for each camera separately. This isn't meant to be precise, and is
    # only used for seeding.
    #
    # I get rotation, translation such that R*calobject + t produces the
    # calibration object points in the coord system of the camera. Here R,t have
    # dimensions (N,3,3) and (N,3) respectively
    R,t = estimate_local_calobject_poses( dots[which],
                                          metadata[which]['dot_spacing'],
                                          (focal_estimate, focal_estimate),
                                          metadata[which]['imager_size'] )
    calobject_poses[which] = (R,t)

if np.linalg.norm(metadata['left']['dot_spacing'] - metadata['right']['dot_spacing']) > 1e-5:
    raise Exception("Mismatched dot spacing")

# I now have a rough estimate of calobject poses in the coord system of each
# frame. One can think of these as two sets of point clouds, each attached to
# their camera. I can move around the two sets of point clouds to try to match
# them up, and this will give me an estimate of the relative pose of the two
# cameras in respect to each other. I need to set up the correspondences, and
# align3d_procrustes() does the rest
#
# I get transformations that map points in 1-Nth camera coord system to 0th
# camera coord system. R,t have dimensions (N-1,3,3) and (N-1,3) respectively
R,t = estimate_camera_pose( calobject_poses,
                            dots,
                            metadata['left']['dot_spacing'],
                            metadata )

# I now have estimates of all parameters, and can run the full optimization
