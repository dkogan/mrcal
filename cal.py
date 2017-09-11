#!/usr/bin/python2

import sys
sys.path[:0] = ('/home/dima/projects/numpysane/',)

import numpy as np
import numpysane as nps
import gnuplotlib as gp
import cv2
import re
import glob
import cPickle as pickle

sys.path[:0] = ('/home/dima/src_boats/stereo-server/analyses',)
import camera_models

sys.path[:0] = ('build/lib.linux-x86_64-2.7/',)
import mrcal



# stuff the user may want to set
pair            = 1
read_cache_dots = True

focal_estimate    = 1950 # pixels
imager_w_estimate = 3904

# This is a concatenation of all the individual files in
# /to_be_filed/datasets/2017-08-08-usv-swarm-test-3/output/calibration/stereo-2017-08-02-Wed-19-30-23/dots/opencv-only
datafile_asciilog='viet_norfolk_joint.asciilog'


# if defined, we will use this:
# datadir_stcal = '/home/dima/data/cal_data_2017_07_14/lfc4/' # for old dot files











re_f = '[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
re_u = '\d+'
re_d = '[-+]?\d+'
re_s = '.+'


@nps.broadcast_define( ((3,3),),
                       (3,), )
def Rodrigues_tor_broadcasted(R):
    r'''Broadcasting-aware wrapper cvRodrigues

This handles the R->r direction, and does not report the gradient'''

    return cv2.Rodrigues(R)[0].ravel()


@nps.broadcast_define( ((3,),),
                       (3,3), )
def Rodrigues_toR_broadcasted(r):
    r'''Broadcasting-aware wrapper cvRodrigues

This handles the r->R direction, and does not report the gradient'''

    return cv2.Rodrigues(r)[0]


@nps.broadcast_define( (('N',3), ('N',3),),
                       (4,3), )
def align3d_procrustes(A, B):
    r"""Computes an optimal (R,t) to match points in B to points in A

Given two sets of 3d points in numpy arrays of shape (N,3), find the optimal
rotation, translation to align these sets of points. Returns array of shape
(4,3): [ R ]
       [ t ]

We minimize

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

    # I don't check dimensionality. The broadcasting-aware wrapper will do that

    A = nps.transpose(A)
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

    return nps.glue( R, t.ravel(), axis=-2)

def get_full_object(W, H, dot_spacing):
    r'''Returns the geometry of the calibration object in its own coordinate frame

Shape is (H,W,3). I.e. the x index varies the fastest and each xyz coordinate
lives at (y,x,:)

    '''

    xx,yy       = np.meshgrid( np.arange(W,dtype=float), np.arange(H,dtype=float))
    full_object = nps.glue(nps.mv( nps.cat(xx,yy), 0, -1),
                           np.zeros((H,W,1)),
                           axis=-1) # shape (H,W,3)
    return full_object * dot_spacing

def estimate_local_calobject_poses( indices_frame_camera, \
                                    dots, dot_spacing, focal, imager_size ):
    r"""Estimates pose of observed object in a single-camera view

Given observations, and an estimate of camera intrinsics (focal lengths, imager
size) computes an estimate of the pose of the calibration object in respect to
the camera for each frame. This assumes that all frames are independent and all
cameras are independent. This assumes a pinhole camera.

This function is a wrapper around the solvePnP() openCV call, which does all the
work.

The observations are given in a numpy array with axes:

  (iframe, idot_x, idot_y, idot2d_xy)

So as an example, the observed pixel coord of the dot (3,4) in frame index 5 is
the 2-vector dots[5,3,4,:]

Missing observations are given as negative pixel coords.

This function returns an (Nobservations,4,3) array, with the observations
aligned with the dots and indices_frame_camera arrays. Each observation slice is
(4,3) in glue(R, t, axis=-2)

    """

    Nobservations = indices_frame_camera.shape[0]

    # this wastes memory, but makes it easier to keep track of which data goes
    # with what
    Rt_all = np.zeros( (Nobservations, 4, 3), dtype=float)
    camera_matrix = np.array((( focal[0], 0,        imager_size[0]/2), \
                              (        0, focal[1], imager_size[1]/2), \
                              (        0,        0,                 1)))

    full_object = get_full_object(10, 10, dot_spacing)

    for i_observation in xrange(dots.shape[0]):
        d = dots[i_observation, ...]

        d = nps.transpose(nps.clump( nps.mv( nps.glue(d, full_object, axis=-1), -1, -3 ), n=2 ))
        # d is (100,5); each row is an xy pixel observation followed by the xyz
        # coord of the point in the calibration object. I pick off those rows
        # where the observations are both >= 0. Result should be (N,5) where N
        # <= 100
        i = (d[..., 0] >= 0) * (d[..., 1] >= 0)
        d = d[i,:]

        observations = d[:,:2]
        ref_object   = d[:,2:]
        result,rvec,tvec = cv2.solvePnP(ref_object  [..., np.newaxis],
                                        observations[..., np.newaxis],
                                        camera_matrix, None)
        if not result:
            raise Exception("solvePnP failed!")
        if tvec[2] <= 0:
            raise Exception("solvePnP says that tvec.z <= 0. Maybe needs a flip, but please examine this")

        Rt_all[i_observation, :3, :] = Rodrigues_toR_broadcasted(rvec.ravel())
        Rt_all[i_observation,  3, :] = tvec.ravel()

    return Rt_all

def estimate_camera_poses( calobject_poses_Rt, indices_frame_camera, \
                           dots, dot_spacing, Ncameras):
    r'''Estimates camera poses in respect to each other

We are given poses of the calibration object in respect to each observing
camera. We also have multiple cameras observing the same calibration object at
the same time, and we have local poses for each. We can thus compute the
relative camera pose from these observations.

We have many frames that have different observations from the same set of
fixed-relative-pose cameras, so we compute the relative camera pose to optimize
the observations

    '''

    # This is a bit of a hack. I look at the correspondence of camera0 to camera
    # i for i in 1:N-1. I ignore all correspondences between cameras i,j if i!=0
    # and j!=0. Good enough for now
    full_object = get_full_object(10, 10, dot_spacing)
    Rt = np.array(())
    for i_camera in xrange(1,Ncameras):
        A = np.array(())
        B = np.array(())

        # I traverse my observation list, and pick out observations from frames
        # that had data from both camera 0 and camera i
        i_frame_last = -1
        d0  = None
        d1  = None
        Rt0 = None
        Rt1 = None
        for i_observation in xrange(dots.shape[0]):
            i_frame_this,i_camera_this = indices_frame_camera[i_observation, ...]
            if i_frame_this != i_frame_last:
                d0  = None
                d1  = None
                Rt0 = None
                Rt1 = None
                i_frame_last = i_frame_this

            if i_camera_this == 0:
                if Rt0 is not None:
                    raise Exception("Saw multiple camera0 observations in frame {}".format(i_frame_this))
                Rt0 = calobject_poses_Rt[i_observation, ...]
                d0  = dots[i_observation, ...]
            if i_camera_this == i_camera:
                if Rt1 is not None:
                    raise Exception("Saw multiple camera{} observations in frame {}".format(i_camera_this,
                                                                                            i_frame_this))
                Rt1 = calobject_poses_Rt[i_observation, ...]
                d1  = dots[i_observation, ...]

                if Rt0 is None: # have camera1 observation, but not camera0
                    continue

                # It's possible that I could have incomplete views of the
                # calibration object, so I pull out only those point
                # observations that have a complete view. In reality, I
                # currently don't accept any incomplete views, and much outside
                # code would need an update to support that. This doesn't hurt, however

                # d looks at one frame and has shape (10,10,7). Each row is
                #   xy pixel observation in left camera
                #   xy pixel observation in right camera
                #   xyz coord of dot in the calibration object coord system
                d = nps.glue( d0, d1, full_object, axis=-1 )

                # squash dims so that d is (100,7)
                d = nps.transpose(nps.clump(nps.mv(d, -1, -3), n=2))

                # I pick out those points that have observations in both frames
                i = (d[..., 0] >= 0) * (d[..., 1] >= 0) * (d[..., 2] >= 0) * (d[..., 3] >= 0)
                d = d[i,:]

                # ref_object is (N,3)
                ref_object = d[:,4:]
                A = nps.glue(A, nps.matmult( ref_object, nps.transpose(Rt0[:3,:])) + Rt0[3,:],
                             axis = -2)
                B = nps.glue(B, nps.matmult( ref_object, nps.transpose(Rt1[:3,:])) + Rt1[3,:],
                             axis = -2)
        Rt = nps.glue(Rt, align3d_procrustes(A, B),
                      axis=-3)

    return Rt

def estimate_frame_poses(calobject_poses_Rt, camera_poses_Rt, indices_frame_camera):
    r'''We're given

calobject_poses_Rt:

  an array of dimensions (Nobservations,4,3) that contains a
  calobject-to-camera transformation estimate, for each observation of the board

camera_poses_Rt:

  an array of dimensions (Ncameras-1,4,3) that contains a camerai-to-camera0
  transformation estimate. camera0-to-camera0 is the identity, so this isn't
  stored

indices_frame_camera:

  an array of shape (Nobservations,2) that indicates which frame and which
  camera has observed the board

With this data, I return an array of shape (Nframes,6) that contains an estimate
of the pose of each frame, in the camera0 coord system. Each row is (r,t) where
r is a Rodrigues rotation and t is a translation that map points in the
calobject coord system to that of camera 0

    '''

    frame_poses_rt = np.array(())

    # frame poses should map FROM the frame coord system TO the ref coord system
    # (camera 0). I have an estimate of these for each camera. Should merge them,
    # but for now, let me simply take the first estimate
    i_frame_last     = -1

    for i_observation in xrange(indices_frame_camera.shape[0]):
        i_frame,i_camera = indices_frame_camera[i_observation, ...]

        if i_frame != i_frame_last:
            if i_camera == 0:
                R = calobject_poses_Rt[i_observation, :3, :]
                t = calobject_poses_Rt[i_observation,  3, :]
            else:
                # cameraiTcamera0 camera0Tboard = cameraiTboard
                # I need camera0Tboard = inv(cameraiTcamera0) cameraiTboard
                # camera_poses_Rt    is inv(cameraiTcamera0)
                # calobject_poses_Rt is cameraiTboard
                Rtcam = camera_poses_Rt[i_camera-1, ...]
                Rcam  = Rtcam[:3,:]
                tcam  = Rtcam[ 3,:]
                Rf = calobject_poses_Rt[i_observation, :3, :]
                tf = calobject_poses_Rt[i_observation,  3, :]

                # Rcam( Rframe *x + tframe) + tcam = Rcam Rframe x + Rcam tframe + tcam
                R = nps.matmult(Rcam, Rf)
                t = nps.matmult( Rcam, nps.transpose(tf)).ravel() + tcam

            frame_poses_rt = nps.glue(frame_poses_rt,
                                      nps.glue( Rodrigues_tor_broadcasted(R),
                                                t,
                                                axis=-1 ),
                                      axis=-2)
            i_frame_last = i_frame

    return frame_poses_rt

def project_points(intrinsics, extrinsics, frames, dot_spacing):
    r'''Takes in the same arguments as mrcal.optimize(), and returns all the
projections. Output has shape (Nframes,Ncameras,10,10,2)'''

    object_ref = get_full_object(10, 10, dot_spacing)
    Rf = Rodrigues_toR_broadcasted(frames[:,:3])
    Rf = nps.mv(Rf,           0, -5)
    tf = nps.mv(frames[:,3:], 0, -5)

    # object in the cam0 coord system. shape=(Nframes, 1, 10, 10, 3)
    object_cam0 = nps.matmult( object_ref, nps.transpose(Rf)) + tf

    Rc = Rodrigues_toR_broadcasted(extrinsics[:,:3])
    Rc = nps.mv(Rc,               0, -4)
    tc = nps.mv(extrinsics[:,3:], 0, -4)

    # object in the OTHER camera coord systems. shape=(Nframes, Ncameras-1, 10, 10, 3)
    object_cam_others = nps.matmult( object_cam0, nps.transpose(Rc)) + tc

    # object in the ALL camera coord systems. shape=(Nframes, Ncameras, 10, 10, 3)
    object_cam = nps.glue(object_cam0, object_cam_others, axis=-4)

    # I now project all of these
    intrinsics = nps.mv(intrinsics, 0, -4)

    # projected points. shape=(Nframes, Ncameras, 10, 10, 2)
    return camera_models.project( object_cam, intrinsics )

def compute_reproj_error(projected, observations, indices_frame_camera):
    r'''Given

- projected (shape [Nframes,Ncameras,10,10,2])
- observations (shape [Nframes,10,10,2])
- indices_frame_camera (shape [Nobservations,2])

Return the reprojection error for each point: shape [Nobservations,10,10,2]

    '''

    Nframes               = projected.shape[0]
    Nobservations         = indices_frame_camera.shape[0]
    err                   = np.zeros((Nobservations,10,10,2))
    for i_observation in xrange(Nobservations):
        i_frame, i_camera = indices_frame_camera[i_observation]
        err[i_observation] = observations[i_observation] - projected[i_frame,i_camera]

    return err


def _read_dots_stcal(datadir):

    raise Exception("This is still coded to assume stereo PAIRS. Update to use discrete cameras")
    def read_observations_from_file__old_dot(filename, which):
        r"""Parses the xxx.dots from the old stcal tool

Given a xxx.dots file produced with stcal, read the observations into a numpy
array. Returns this numpy array and a list of metadata.

The array has axes: (iframe, idot_y, idot_x, idot2d_xy)

So as an example, the observed pixel coord of the dot (3,4) in frame index 5 is
the 2-vector dots[5,3,4,:]

The metadata is a dictionary, containing the dimensions of the imager, and the
indices of frames that the numpy array contains

        """

        with open(filename, 'r') as f:
            for l in f:
                if re.match('# Format: jplv$',
                            l):
                    break
            else:
                raise Exception('No explicit "Format: jplv" when reading {}'.format(filename))

            # Data. Axes: (iframe, idot_y, idot_x, idot2d_xy)
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
                    idot   = (int(   m.group(6)),  int(  m.group(7)))
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

                    dot_spacing = np.array(dot3d, dtype=float) / np.array(idot, dtype=float)
                    if metadata['dot_spacing'] is None:
                        metadata['dot_spacing'] = dot_spacing
                    else:
                        if np.max( np.abs(metadata['dot_spacing'] - dot_spacing) ) > 1e-4:
                            raise Exception("Inconsistent dot spacing. Previously saw {} but now have {}". \
                                            format(metadata['dot_spacing'], dot_spacing))

                    dots[-1, idot[1]-1, idot[0]-1,:] = dot2d
                    continue

                raise Exception('Got unknown line "{}"'.format(l))

        return dots, metadata




    dots            = {}
    metadata        = {}

    for fil in ('{}/stcal-left.dots' .format(datadir),
                '{}/stcal-right.dots'.format(datadir)):
        m = re.match( '.*-(left|right)\.dots$', fil)
        if not m: raise Exception("Can't tell if '{}' is left or right".format(fil))
        which = m.group(1)

        d,m             = read_observations_from_file__old_dot( fil, which )
        dots    [which] = d
        metadata[which] = m

    return dots,metadata

def _read_dots_asciilog(datafile):
    r'''Reads an asciilog dots file produced by cdas-find-dots

cdas-find-dots lives in the cdas-core project.

This function parses a single data file that contains ALL the observations.
There are no assumptions of any joint observations. I.e. during each instant in
time anywhere from 1 to N>2 cameras could have been observed (this is a generic
N-camera calibration, NOT a camera pair calibration).

Each board observation lives in a slice of the returned 'dots' array. The frame
and camera indices responsible for that observation live in a corresponding
slice of the metadata['indices_frame_camera'] array

    '''
    def get_next_dots(f):

        def parse_image_header(l):
            # Two image formats are possible:
            #   frame00002-pair1-cam0.jpg
            #   input-right-0-02093.jpg
            m = re.match('([^ ]*/frame([0-9]+)-pair([01])-cam([0-9]+)\.[a-z][a-z][a-z]) ({f}) ({f}) 10 10 ({d}) - - - - - -\n$'.format(f=re_f, d=re_d), l)
            if m:
                path        = m.group(1)
                i_frame     = int(m.group(2))
                i_pair      = int(m.group(3))
                i_camera    = int(m.group(4))
                dot_spacing = float(m.group(6))
                Ndetected   = int(m.group(7))
                return path,i_frame,i_pair,i_camera,dot_spacing,Ndetected
            m = re.match('([^ ]*/input-(left|right)-([01])-([0-9]+)\.[a-z][a-z][a-z]) ({f}) ({f}) 10 10 ({d}) - - - - - -\n$'.format(f=re_f, d=re_d), l)
            if m:
                path        = m.group(1)
                i_frame     = int(m.group(4))
                i_pair      = int(m.group(3))
                i_camera    = 0 if m.group(2) == 'left' else 1
                dot_spacing = float(m.group(6))
                Ndetected   = int(m.group(7))
                return path,i_frame,i_pair,i_camera,dot_spacing,Ndetected
            raise Exception("Couldn't parse image header line '{}'".format(l))





        # Keep going until I get a full frame's worth of data or until there's
        # nothing else to read
        while True:

            # Grab the next non-comment line
            while True:
                try:
                    l = next(f)
                except:
                    return None,None,None,None,None,None
                if l[0] != '#':
                    break

            path,i_frame,i_pair,i_camera,dot_spacing,Ndetected = parse_image_header(l)

            if Ndetected != 10*10:
                if Ndetected != 0:
                    raise Exception("I can't handle incomplete board observations yet")
                continue

            # OK then. I have dots to look at
            dots = np.zeros((10,10,2), dtype=float)

            for point_index in xrange(Ndetected):
                l = next(f)
                lf = l.split()
                if lf[0] != path:
                    raise Exception("Unexpected incomplete observation. Expected path '{}' but got '{}'".
                                    format(path, lf[0]))
                idot_x,idot_y     = [int(x)   for x in lf[6 :8 ]]
                dot2d_x, dot2d_y  = [float(x) for x in lf[10:12]]

                # I only accept complete observations of the cal board for now
                idot_y_want = int(point_index / 10)
                idot_x_want = point_index - idot_y_want*10
                if idot_x != idot_x_want or idot_y != idot_y_want:
                    raise Exception("Unexpected dot index")

                dots[idot_y,idot_x,:] = (dot2d_x,dot2d_y)

            return path,i_frame,i_pair,i_camera,dot_spacing,dots










    # dimensions (Nobservations, 10,10, 2)
    dots                 = np.array(())
    # dimension (Nobservations, 2). Each row is (i_frame_consecutive, i_camera)
    indices_frame_camera = np.array((), dtype=np.int32)

    metadata = { 'imager_size':    (imager_w_estimate, imager_w_estimate),
                 'focal_estimate': (focal_estimate, focal_estimate)}

    i_frame_consecutive   = -1
    i_frame_last          = -1
    seen_cameras          = set()

    # I want the data to come in order:
    # frames - pairs - cameras - dots

    # Data. Axes: (idot_y, idot_x, idot2d_xy)
    # So the observed pixel coord of the dot (3,4) is
    # the 2-vector dots[3,4,:]
    dots        = np.array( (), dtype=float)
    dot_spacing = None

    point_index = 0

    with open(datafile, 'r') as f:

        l = next(f)
        if l != '# path fixture_size_m fixture_space_m fixture_cols fixture_rows num_dots_detected dot_fixture_col dot_fixture_row dot_fixture_physical_x dot_fixture_physical_y dot_image_col dot_image_row\n':
            raise Exception("Unexpected legend in '{}".format(datafile))

        while True:
            path,i_frame,i_pair,i_camera,dot_spacing,dots_here = get_next_dots(f)
            if i_frame is None:
                break

            if i_frame != i_frame_last:
                new_frame = True

            # make sure I get the ordering I want: frames - pairs - cameras
            if i_frame != i_frame_last:
                # if i_frame < i_frame_last:
                #     raise Exception("Non-consecutive i_frame: got {} and then {}".
                #                     format(i_frame_last, i_frame))
                # commented out because I have different directories and non-consecutive frames will thus result

                i_frame_last         = i_frame
                i_pair_last          = i_pair
                i_camera_last        = i_camera
            elif i_pair != i_pair_last:
                if i_pair < i_pair_last:
                    raise Exception("Non-consecutive i_pair: got {} and then {}".
                                    format(i_pair_last, i_pair))
                i_pair_last          = i_pair
                i_camera_last        = i_camera
            elif i_camera != i_camera_last:
                if i_camera < i_camera_last:
                    raise Exception("Non-consecutive i_camera: got {} and then {}".
                                    format(i_camera_last, i_camera))
                i_camera_last        = i_camera


            if i_pair != pair_want:
                continue

            if new_frame:
                i_frame_consecutive += 1
                new_frame = False


            seen_cameras.add(i_camera)

            if not 'dot_spacing' in metadata:
                metadata['dot_spacing'] = dot_spacing
            else:
                if dot_spacing != metadata['dot_spacing']:
                    raise Exception("Inconsistent dot spacing")

            dots = nps.glue(dots, dots_here, axis = -4)
            indices_frame_camera = \
                nps.glue(indices_frame_camera,
                         np.array((i_frame_consecutive, i_camera), dtype=np.int32),
                         axis=-2)


        if min(seen_cameras) != 0:
            raise Exception("Min camera index must be 0, but got {}".format(min(seen_cameras)))
        metadata['Ncameras'] = max(seen_cameras) + 1
        if metadata['Ncameras'] != len(seen_cameras):
            raise Exception("Non-consecutive cam indices: min: {} max: {} len: {}". \
                            format(min(seen_cameras),max(seen_cameras),len(seen_cameras)))
        metadata['indices_frame_camera'] = indices_frame_camera
        return dots,metadata
def read_dots():
    try:
        return _read_dots_stcal(datadir_stcal)
    except:
        return _read_dots_asciilog(datafile_asciilog)







cachefile_dots = 'mrcal.dots.pair{}.pickle'.format(pair)
if( read_cache_dots ):
    with open(cachefile_dots, 'r') as f:
        dots,metadata = pickle.load(f)
else:
    dots,metadata = read_dots()
    with open(cachefile_dots, 'w') as f:
        pickle.dump( (dots,metadata), f, protocol=2)


# I now have estimates of all parameters, and can run the full optimization
def make_intrinsics_vector(i_camera, metadata):
    imager_w,imager_h = metadata['imager_size']
    return np.array( (metadata['focal_estimate'][0], metadata['focal_estimate'][1],
                      float(imager_w-1)/2.,
                      float(imager_h-1)/2.))

intrinsics = nps.cat( *[make_intrinsics_vector(i_camera, metadata) \
                        for i_camera in xrange(metadata['Ncameras'])] )


# I compute an estimate of the poses of the calibration object in the local
# coord system of each camera for each frame. This is done for each frame
# and for each camera separately. This isn't meant to be precise, and is
# only used for seeding.
#
# I get rotation, translation in a (4,3) array, such that R*calobject + t
# produces the calibration object points in the coord system of the camera.
# The result has dimensions (N,4,3)
calobject_poses_Rt = \
    estimate_local_calobject_poses( metadata['indices_frame_camera'],
                                    dots,
                                    metadata['dot_spacing'],
                                    metadata['focal_estimate'],
                                    metadata['imager_size'] )
# these map FROM the coord system of the calibration object TO the coord
# system of this camera

# I now have a rough estimate of calobject poses in the coord system of each
# frame. One can think of these as two sets of point clouds, each attached to
# their camera. I can move around the two sets of point clouds to try to match
# them up, and this will give me an estimate of the relative pose of the two
# cameras in respect to each other. I need to set up the correspondences, and
# align3d_procrustes() does the rest
#
# I get transformations that map points in 1-Nth camera coord system to 0th
# camera coord system. R,t have dimensions (N-1,3,3) and (N-1,3) respectively
camera_poses_Rt = estimate_camera_poses( calobject_poses_Rt,
                                         metadata['indices_frame_camera'],
                                         dots,
                                         metadata['dot_spacing'],
                                         metadata['Ncameras'] )

# extrinsics should map FROM the ref coord system TO the coord system of the
# camera in question. This is backwards from what I have. To flip:
#
# R*x + t = x'    ->     x = Rt x' - Rt t
R = camera_poses_Rt[..., :3, :]
t = camera_poses_Rt[...,  3, :]
extrinsics = nps.atleast_dims( nps.glue( Rodrigues_tor_broadcasted(nps.transpose(R)),
                                         -nps.matmult( nps.dummy(t,-2), R )[..., 0, :],
                                         axis=-1 ),
                               -2)

frames = \
    estimate_frame_poses(calobject_poses_Rt, camera_poses_Rt,
                         metadata['indices_frame_camera'])
observations = dots





# done with everything. Run the calibration, in several passes.
projected = \
    project_points(intrinsics, extrinsics, frames,
                   metadata['dot_spacing'])
err = compute_reproj_error(projected, observations,
                           metadata['indices_frame_camera'])

norm2_err = nps.inner(err.ravel(), err.ravel())
rms_err   = np.sqrt( norm2_err / (err.ravel().shape[0]/2) )
print "initial norm2 err: {}, rms: {}".format(norm2_err, rms_err )

distortion_model = "DISTORTION_NONE"
mrcal.optimize(intrinsics, extrinsics, frames,
               observations, metadata['indices_frame_camera'], distortion_model, False)

distortion_model = "DISTORTION_NONE"
mrcal.optimize(intrinsics, extrinsics, frames,
               observations, metadata['indices_frame_camera'], distortion_model, True)

distortion_model = "DISTORTION_CAHVOR"
Ndistortions = mrcal.getNdistortionParams(distortion_model)
intrinsics = nps.glue( intrinsics, np.random.random((metadata['Ncameras'], Ndistortions))*1e-6, axis=-1 )
mrcal.optimize(intrinsics, extrinsics, frames,
               observations, metadata['indices_frame_camera'], distortion_model, True)

# Done! Write out a cache of the solution
cachefile_solution = 'mrcal.solution.pair{}.pickle'.format(pair)
with open(cachefile_solution, 'w') as f:
    pickle.dump( (intrinsics, extrinsics, frames, observations), f, protocol=2)

# and write out the resulting cahvor files
cahvor0 = camera_models.make_cahvor( intrinsics[0] )
cahvor1 = camera_models.make_cahvor( intrinsics[1], extrinsics[0] )

camera_models.write_cahvor( "camera{}-0.cahvor".format(pair), cahvor0 )
camera_models.write_cahvor( "camera{}-1.cahvor".format(pair), cahvor1 )
