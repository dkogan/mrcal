#!/usr/bin/python2

import numpy as np
import numpysane as nps
import gnuplotlib as gp
import sys
import cv2
import cPickle as pickle
import scipy.optimize

sys.path[:0] = ('/home/dima/jpl/stereo-server/analyses',)
import camera_models



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

    I can shift my a_i and b_i so that they have 0-mean. In this case, the
    optimal t = 0 and

      E ~ sum( inner(a_i, R b_i )  )

    This is the classic procrustes problem

      E = tr( At R B ) = tr( R B At ) = tr( R U S Vt ) = tr( Vt R U S )

    So the critical points are at Vt R U = I and R = V Ut, modulo a tweak to
    make sure that R is in SO(3) not just in SE(3)

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

def cahvor_warp_distort(p, fx, fy, cx, cy, *distortions):
    r'''Apply a CAHVOR warp to an un-distorted point

    Given intrinsic parameters of a CAHVOR model and a pinhole-projected
    point(s) numpy array of shape (..., 2), return the projected point(s) that
    we'd get with distortion. We ASSUME THE SAME fx,fy,cx,cy

    This function can broadcast the points array

    '''

    if not len(distortions):
        return p
    theta, phi, r0, r1, r2 = distortions

    # p is a 2d point. Temporarily convert to a 3d point
    p = nps.mv( nps.cat((p[..., 0] - cx)/fx,
                        (p[..., 1] - cy)/fy,
                        np.ones( p.shape[:-1])),
                0, -1 )
    o = np.array( (np.sin(phi) * np.cos(theta),
                   np.sin(phi) * np.sin(theta),
                   np.cos(phi) ))

    # cos( angle between p and o ) = inner(p,o) / (norm(o) * norm(p)) =
    # omega/norm(p)
    omega = nps.inner(p,o)

    # tau = 1/cos^2 - 1 = inner(p,p)/(omega*omega) - 1 =
    #     = tan^2
    tau   = nps.inner(p,p) / (omega*omega) - 1.0
    mu    = r0 + tau*r1 + tau*tau*r2
    p     = p*(nps.dummy(mu,-1)+1.0) - nps.dummy(mu*omega, -1)*o

    # now I apply a normal projection to the warped 3d point p
    return np.array((fx,fy)) * p[..., :2] / p[..., (2,)] + np.array((cx,cy))

def cahvor_warp_undistort(p, fx, fy, cx, cy, *distortions):
    r'''Un-apply a CAHVOR warp: undistort a point

    Given intrinsic parameters of a CAHVOR model and a pinhole-projected
    point(s) numpy array of shape (..., 2), return the projected point(s) that
    we'd get without distortion. We ASSUME THE SAME fx,fy,cx,cy

    This function can broadcast the points array.

    Note that this function has an iterative solver and is thus SLOW. This is
    the "backwards" direction. Most of the time you want cahvor_warp_distort().

    '''

    if not len(distortions):
        return p

    # I could make this much more efficient: precompute lots of stuff, use
    # gradients, etc, etc. I can also optimize each point separately. But that
    # would make the code messy and require more work. This functions decently
    # well and I leave it.
    def f(p0):
        '''Optimization functions'''
        N = len(p0.ravel()) / 2
        p1 = cahvor_warp_distort(p0.reshape(N,2), fx,fy,cx,cy, *distortions)
        return p1.ravel() - p.ravel()

    p1 = scipy.optimize.leastsq(f, p.ravel())[0]
    return np.array(p1).reshape(p.shape)

def project(p, intrinsics):
    r'''Projects 3D point(s) using the given camera intrinsics

    This function is broadcastable over p only: you're meant to use it to
    project a number of points at the same time, but only a single set of
    intrinsics is supported.

    I can easily add that support with

      @nps.broadcast_define( ((3,),('Nintrinsics',)),
                             (2,), )

    but then I'd have a python loop iterating over all my points, which is slow.
    The computations here are simple enough for numpy to handle all the
    broadcasting at the C level, so I let it do that.

    Inputs:

    - p a 3D point in the camera coord system

    - intrinsics: a numpy array containing
      - fx
      - fy
      - cx
      - cy
      - CAHVOR theta (for computing O)
      - CAHVOR phi   (for computing O)
      - CAHVOR R0
      - CAHVOR R1
      - CAHVOR R2

      The CAHVOR distortion stuff is optional.

    '''
    intrinsics = intrinsics.ravel()

    if len(intrinsics) == 4:
        pinhole = True
    elif len(intrinsics) == 9:
        pinhole = False
    else:
        raise Exception("I know how to deal with an ideal camera (4 intrinsics) or a cahvor camera (9 intrinsics), but I got {} intrinsics intead".format(len(intrinsics)))


    p2d = p[..., :2]/p[..., (2,)] * intrinsics[:2] + intrinsics[2:4]
    if pinhole: return p2d

    return cahvor_warp_distort(p2d, *intrinsics)

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


def ingest_cahvor(model):
    r'''Reutnrs a cahvor model from a variety of representations

    The input should be any of

    - a file containing a CAHVOR model
    - a python file object from which such a model could be read
    - a dict representation you get when you parse_cahvor() such a file

    The output is a cahvor dict
    '''

    model_file = None

    if isinstance(model, str):
        model = open(model, 'r')
        model_file = model

    if isinstance(model, file):
        model = camera_models.parse_cahvor(model)

    if model_file is not None:
        model_file.close()

    if isinstance(model, dict):
        return model

    raise Exception("Input must be a string, a file, a dict or a numpy array.")


def ingest_intrinsics(model):
    r'''Reads cahvor intrinsics from a variety of representations

    The input should be any of

    - a file containing a CAHVOR model
    - a python file object from which such a model could be read
    - a dict representation you get when you parse_cahvor() such a file
    - a numpy array containing the intrinsics.

    The output is a numpy array containing the intrinsics
    '''

    if not isinstance(model, np.ndarray):
        model = ingest_cahvor(model)
        model = camera_models.get_intrinsics(model)

    if not isinstance(model, np.ndarray):
        raise Exception("Input must be a string, a file, a dict or a numpy array")

    if len(model) == 4:
        return model
    if len(model) == 9:
        return model

    raise Exception("Intrinsics vector MUST have length 4 or 9. Instead got {}".format(len(model)))


def distortion_map__to_warped(model, w, h):
    r'''Returns the pre and post distortion map of a model

    Takes in

    - a cahvor model (in a number of representations)
    - a sampling spacing on the x axis
    - a sampling spacing on the y axis

    Produces two arrays of shape Nwidth,Nheight,2:

    - grid: a meshgrid of the sampling spacings given as inputs. This refers to the
      UNDISTORTED image

    - dgrid: a meshgrid where each point in grid had the distortion corrected.
      This refers to the DISTORTED image

    '''

    intrinsics = ingest_intrinsics(model)

    # shape: Nwidth,Nheight,2
    grid  = nps.reorder(nps.cat(*np.meshgrid(w,h)), -1, -2, -3)
    dgrid = cahvor_warp_distort(grid, *intrinsics)
    return grid, dgrid

def visualize_distortion_vector_field(model):
    r'''Visualize the distortion effect of a set of intrinsic

    This function renders the distortion vector field

    The input should either be a file containing a CAHVOR model, a python file
    object from which such a model could be read, the dict representation you
    get when you parse_cahvor() on such a file OR a numpy array containing the
    intrinsics

    '''

    intrinsics = ingest_intrinsics(model)

    N = 20
    W,H = [2*center for center in intrinsics[2:4]]

    # get the input and output grids of shape Nwidth,Nheight,2
    grid, dgrid = distortion_map__to_warped(intrinsics,
                                            np.linspace(0,W,N),
                                            np.linspace(0,H,N))

    # shape: N*N,2
    grid  = nps.clump(grid,  n=2)
    dgrid = nps.clump(dgrid, n=2)

    delta = dgrid-grid
    gp.plot( (grid[:,0], grid[:,1], delta[:,0], delta[:,1],
              {'with': 'vectors size screen 0.01,20 fixed filled',
               'tuplesize': 4,
               }),
             (grid[:,0], grid[:,1],
              {'with': 'points',
               'tuplesize': 2,
               }),
             _xrange=(0,W), _yrange=(H,0))

    import time
    time.sleep(100000)

def undistort_image(model, image):
    r'''Visualize the distortion effect of a set of intrinsic

    This function warps an image to remove the distortion.

    The input should either be a file containing a CAHVOR model, a python file
    object from which such a model could be read, the dict representation you
    get when you parse_cahvor() on such a file OR a numpy array containing the
    intrinsics.

    An image is also input (could be a filename or an array). An array image is
    output.

    '''

    intrinsics = ingest_intrinsics(model)

    if not isinstance(image, np.ndarray):
        image = cv2.imread(image)

    H,W = image.shape[:2]
    _,mapxy = distortion_map__to_warped(intrinsics,
                                        np.arange(W), np.arange(H))
    mapx = mapxy[:,:,0].astype(np.float32)
    mapy = mapxy[:,:,1].astype(np.float32)

    remapped = cv2.remap(image,
                         nps.transpose(mapx),
                         nps.transpose(mapy),
                         cv2.INTER_LINEAR)
    return remapped

def homography_atinfinity_map( w, h, m0, m1 ):
    r'''Compute that 1<-0 at-infinity homography for two cahvor models

    The initial inputs are the output image width grid and height grid.

    The following inputs should either be a file containing a CAHVOR model, a
    python file object from which such a model could be read or the dict
    representation you get when you parse_cahvor() on such a file.

    THIS FUNCTION IGNORES DISTORTIONS.

    I have an observation x0 in camera0. If x0 is in homogeneous coordinates,
    then the corresponding observation in the other camera is like so:

        x1 = A1 * (R10 (k * inv(A0)*x0) + t10)

    At infinity, the k dominates:

        x1 = A1 * (R10 * k * inv(A0)*x0)

    And since I have homogeneous coordinates, I can drop the k:

        x1 = A1 * R10 * inv(A0)*x0.

    I.e:

        x1 = H10 x0    where   H10 = A1 * R10 * inv(A0)

    R10 is R 1<-0

    '''

    def get_components(m):
        m = ingest_cahvor(m)
        e = camera_models.get_extrinsics_Rt_toref(m)
        fx,fy,cx,cy = camera_models.cahvor_fxy_cxy(m)
        return e[:3,:],fx,fy,cx,cy

    Rr0,fx0,fy0,cx0,cy0 = get_components(m0)
    Rr1,fx1,fy1,cx1,cy1 = get_components(m1)

    # R10 = R1r * Rr0 = inv(Rr1) * Rr0
    R10 = nps.matmult( nps.transpose(Rr1), Rr0 )

    # I can compute the homography:
    #   A0 = np.array(((fx0,   0, cx0), \
    #                  (  0, fy0, cy0), \
    #                  (  0,   0,   1)))
    #   A1 = np.array(((fx1,   0, cx1), \
    #                  (  0, fy1, cy1), \
    #                  (  0,   0,   1)))
    #   H = nps.matmult(A1, R10, np.linalg.inv(A0))
    #
    # To apply this homography I'd need a broadcast-multiply of a bunch of
    # points by H, which is currently a python loop in numpysane, so it's very
    # slow. Instead I compute a map here using core numpy components, so it's
    # fast.

    # I'm computing a map TO the remapped coords, so I need the INVERSE
    # homography: A0 R01 inv(A1)


    # Output image grid. shape: Nwidth,Nheight,2
    p1 = nps.reorder(nps.cat(*np.meshgrid(w,h)), -1, -2, -3)

    # Output image grid in homogeneous coords; without the extra "1". shape:
    # Nwidth,Nheight,2
    p1xy3d = (p1 - np.array((cx1,cy1))) / np.array((fx1,fy1))

    # Input image grid in homogeneous coords. Each has shape: Nwidth,Nheight
    p03d_x = nps.inner(R10[:2,0], p1xy3d) + R10[2,0]
    p03d_y = nps.inner(R10[:2,1], p1xy3d) + R10[2,1]
    p03d_z = nps.inner(R10[:2,2], p1xy3d) + R10[2,2]

    # Project. shape: Nwidth,Nheight,2
    p0xy = nps.mv(nps.cat(p03d_x,p03d_y) / p03d_z, 0,-1)

    # Input Pixel coords. shape: Nwidth,Nheight,2
    p0xy = p0xy * np.array((fx0,fy0)) + np.array((cx0,cy0))
    return p1, p0xy


# from visualize_extrinsic. please unduplicate
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

# from visualize_extrinsic. please unduplicate. I changed the length of the
# drawn x axis. I'm coloring the pairs separately in the other function
def gen_plot_axes(transforms, label, scale = 1.0, label_offset = None):
    r'''Given a list of transforms (applied to the reference set of axes in reverse
order) and a label, return a list of plotting directives gnuplotlib understands.

Transforms are in reverse order so a point x being transformed as A*B*C*x can be
represented as a transforms list (A,B,C)

    '''
    axes = np.array( ((0,0,0),
                      (1,0,0),
                      (0,1,0),
                      (0,0,2),), dtype=float ) * scale

    transform = mrpose.pose3_ident()

    for x in transforms:
        transform = mrpose.pose3_mul(transform, x)
    axes = np.array([ mrpose.vec3_transform(transform, x) for x in axes ])

    axes_forplotting = extend_axes_for_plotting(axes)

    l_axes = tuple(nps.transpose(axes_forplotting)) + \
        ({'with': 'vectors', 'tuplesize': 6},)

    l_labels = tuple(nps.transpose(axes*1.01 + \
                                   (label_offset if label_offset is not None else 0))) + \
        (np.array((label,
                   'x', 'y', 'z')),
         {'with': 'labels', 'tuplesize': 4},)
    return l_axes, l_labels



@nps.broadcast_define( ((6,),),
                       (7,))
def pose__pq_from_rt(rt):
    r'''Converts a pose from an rt to a pq representation

Input is a (6,) numpy array containing a 3d rodrigues rotation and a 3d
translation

Output is a pose_t object numpy array containing a 3d translation and a 4d quaternion
    '''
    p = rt[3:]
    r = rt[:3]
    R = Rodrigues_toR_broadcasted(r)
    q = mrpose.quat_from_mat33d(R)
    return mrpose.pose3_set(p,q)


def project_points(intrinsics, extrinsics, frames, dot_spacing, Nwant):
    r'''Takes in the same arguments as mrcal.optimize(), and returns all the
projections. Output has shape (Nframes,Ncameras,Nwant,Nwant,2)'''

    object_ref = get_full_object(Nwant, Nwant, dot_spacing)
    Rf = Rodrigues_toR_broadcasted(frames[:,:3])
    Rf = nps.mv(Rf,           0, -5)
    tf = nps.mv(frames[:,3:], 0, -5)

    # object in the cam0 coord system. shape=(Nframes, 1, Nwant, Nwant, 3)
    object_cam0 = nps.matmult( object_ref, nps.transpose(Rf)) + tf

    Rc = Rodrigues_toR_broadcasted(extrinsics[:,:3])
    Rc = nps.mv(Rc,               0, -4)
    tc = nps.mv(extrinsics[:,3:], 0, -4)

    # object in the OTHER camera coord systems. shape=(Nframes, Ncameras-1, Nwant, Nwant, 3)
    object_cam_others = nps.matmult( object_cam0, nps.transpose(Rc)) + tc

    # object in the ALL camera coord systems. shape=(Nframes, Ncameras, Nwant, Nwant, 3)
    object_cam = nps.glue(object_cam0, object_cam_others, axis=-4)

    # I now project all of these
    intrinsics = nps.mv(intrinsics, 0, -4)

    # projected points. shape=(Nframes, Ncameras, Nwant, Nwant, 2)
    return project( object_cam, intrinsics )

def compute_reproj_error(projected, observations, indices_frame_camera, Nwant):
    r'''Given

- projected (shape [Nframes,Ncameras,Nwant,Nwant,2])
- observations (shape [Nframes,Nwant,Nwant,2])
- indices_frame_camera (shape [Nobservations,2])

Return the reprojection error for each point: shape [Nobservations,Nwant,Nwant,2]

    '''

    Nframes               = projected.shape[0]
    Nobservations         = indices_frame_camera.shape[0]
    err                   = np.zeros((Nobservations,Nwant,Nwant,2))
    for i_observation in xrange(Nobservations):
        i_frame, i_camera = indices_frame_camera[i_observation]

        err[i_observation] = observations[i_observation] - projected[i_frame,i_camera]

    return err


def visualize_solution(intrinsics, extrinsics, frames, observations,
                       indices_frame_camera, dot_spacing, Nwant, i_camera=None):
    r'''Plot the best-estimate 3d poses of a hypothesis calibration

The inputs are the same as to mrcal.optimize(). If i_camera is not None, the
visualization is colored by the reprojection-error-quality of the fit

    '''

    Nobservations = len(indices_frame_camera)
    if i_camera is not None:
        i_observations_frames = [(i,indices_frame_camera[i,0]) for i in xrange(Nobservations) if indices_frame_camera[i,1] == i_camera]
        i_observations, i_frames = nps.transpose(np.array(i_observations_frames))
        frames = frames[i_frames, ...]


    object_ref = get_full_object(Nwant, Nwant, dot_spacing)

    Rf = Rodrigues_toR_broadcasted(frames[..., :3])
    Rf = nps.mv(Rf,           0, -4)
    tf = nps.mv(frames[..., 3:], 0, -4)

    # object in the cam0 coord system. shape=(Nframes, Nwant, Nwant, 3)
    object_cam0 = nps.matmult( object_ref, nps.transpose(Rf)) + tf
    if i_camera is not None:
        # shape=(Nobservations, Nwant, Nwant, 2)
        if i_camera == 0:
            object_cam = object_cam0
        else:
            Rc = Rodrigues_toR_broadcasted(extrinsics[i_camera-1,:3])
            tc = extrinsics[i_camera-1,3:]

            object_cam = nps.matmult( object_cam0, nps.transpose(Rc)) + tc

        err = observations[i_observations, ...] - project(object_cam, intrinsics[i_camera, ...])
        err = nps.clump(err, n=-3)
        rms = np.sqrt(nps.inner(err,err) / (Nwant*Nwant))
        object_cam0 = nps.glue( object_cam0,
                                nps.dummy( nps.mv(rms, -1, -3) * np.ones((Nwant,Nwant)),
                                           -1 ),
                                axis = -1)

    object_cam0 = nps.clump( nps.mv(object_cam0, -1, -4), n=-2)
    cam0_axes_labels = gen_plot_axes((mrpose.pose3_ident(),), 'cam0')
    cam_axes_labels  = [gen_plot_axes( (mrpose.pose3_inv(pose__pq_from_rt(extrinsics[i])),),
                                        'cam{}'.format(i+1)) for i in range(0,extrinsics.shape[-2])]


    curves = list(cam0_axes_labels) + [ca for x in cam_axes_labels for ca in x]

    if i_camera is not None:
        object_curveopts = {'with':'lines palette',
                            'tuplesize': 4}
    else:
        object_curveopts = {'with':'lines',
                            'tuplesize': 3}

    curves.append( tuple(list(object_cam0) + [object_curveopts,]))

    # Need ascii=1 because I'm plotting labels.
    gp.plot(*curves, _3d=1, square=1, ascii=1 )

    import time
    time.sleep(100000)

@nps.broadcast_define( (('Nw','Nh',2),),
                       ())
def get_observation_size(obs):
    r'''Computes an observed 'size' of an observation.

Given an observed calibration object, this returns the max(delta_x, delta_y).
From this we can get a very rough initial estimate of the range to the object.

The observation is an array of shape (N,N,2)
'''

    # corners
    c = nps.cat( obs[ 0, 0, ...],
                 obs[-1, 0, ...],
                 obs[ 0,-1, ...],
                 obs[-1,-1, ...] )

    dx = c[:,0].max() - c[:,0].min()
    dy = c[:,1].max() - c[:,1].min()

    return max(dx,dy)
