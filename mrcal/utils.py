#!/usr/bin/python2

import numpy as np
import numpysane as nps
import gnuplotlib as gp
import sys
import re
import cv2

sys.path[:0] = ('build/lib.linux-x86_64-2.7/',)
import mrcal

import mrpose



@nps.broadcast_define( (('N',3), ('N',3),),
                       (4,3), )
def align3d_procrustes(A, B, vectors=False):
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


    This can ALSO be used to find the optimal rotation to align a set of unit
    vectors. The math is exactly the same, but subtracting the mean should be
    skipped. And returning t is non-sensical, so in this case we just return R. To
    do this, pass vectors=True as a kwarg

    """

    # I don't check dimensionality. The broadcasting-aware wrapper will do that

    A = nps.transpose(A)
    B = nps.transpose(B)

    if vectors:
        M = nps.matmult( B, nps.transpose(A) )
    else:
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

    if vectors:
        return R

    # Now that I have my optimal R, I compute the optimal t. From before:
    #
    #   t = mean(a) - R mean(b)
    t = np.mean(A, axis=-1)[..., np.newaxis] - nps.matmult( R, np.mean(B, axis=-1)[..., np.newaxis] )

    return nps.glue( R, t.ravel(), axis=-2)



@nps.broadcast_define( ((3,3),),
                       (3,), )
def Rodrigues_tor_broadcasted(R):
    r'''Broadcasting-aware wrapper cvRodrigues

    This handles the R->r direction, and does not report the gradient

    '''

    return cv2.Rodrigues(R)[0].ravel()


@nps.broadcast_define( ((3,),),
                       (3,3), )
def Rodrigues_toR_broadcasted(r):
    r'''Broadcasting-aware wrapper cvRodrigues

    This handles the r->R direction, and does not report the gradient

    '''

    return cv2.Rodrigues(r)[0]

def get_full_object(W, H, dot_spacing):
    r'''Returns the geometry of the calibration object in its own coordinate frame

    Shape is (H,W,3). I.e. the x index varies the fastest and each xyz
    coordinate lives at (y,x,:)

    '''

    xx,yy       = np.meshgrid( np.arange(W,dtype=float), np.arange(H,dtype=float))
    full_object = nps.glue(nps.mv( nps.cat(xx,yy), 0, -1),
                           np.zeros((H,W,1)),
                           axis=-1) # shape (H,W,3)
    return full_object * dot_spacing


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
        m = cahvor(m)
        e = m.extrinsics_Rt(True)
        fx,fy,cx,cy = m.intrinsics()[1][:4]
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

    # project. shape: Nwidth,Nheight,2
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
    order) and a label, return a list of plotting directives gnuplotlib
    understands.

    Transforms are in reverse order so a point x being transformed as A*B*C*x
    can be represented as a transforms list (A,B,C)

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

    Output is a pose_t object numpy array containing a 3d translation and a 4d
    quaternion

    '''
    p = rt[3:]
    r = rt[:3]
    R = Rodrigues_toR_broadcasted(r)
    q = mrpose.quat_from_mat33d(R)
    return mrpose.pose3_set(p,q)


def visualize_solution(distortion_model, intrinsics, extrinsics, frames, observations,
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

        err = observations[i_observations, ...] - project(object_cam, distortion_model, intrinsics[i_camera, ...])
        err = nps.clump(err, n=-3)
        rms = np.sqrt(nps.inner(err,err) / (Nwant*Nwant))
        # igood = rms <  0.4
        # ibad  = rms >= 0.4
        # rms[igood] = 0
        # rms[ibad] = 1
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

    print frames.shape
    print object_cam0.shape
    print observations.shape
    # # object in the OTHER camera coord systems. shape=(Nframes, Ncameras-1, Nwant, Nwant, 3)
    # object_cam_others = nps.matmult( object_cam0, nps.transpose(Rc)) + tc

    # # object in the ALL camera coord systems. shape=(Nframes, Ncameras, Nwant, Nwant, 3)
    # object_cam = nps.glue(object_cam0, object_cam_others, axis=-4)

    # # I now project all of these
    # intrinsics = nps.mv(intrinsics, 0, -4)

    # # projected points. shape=(Nframes, Ncameras, Nwant, Nwant, 2)
    # return project( object_cam, intrinsics )



    # obj = get_full_object()
    # convert to cam0
    # convert to cam n
    # colors = get_fit


@nps.broadcast_define( (('Nw','Nh',2),),
                       ())
def get_observation_size(obs):
    r'''Computes an observed 'size' of an observation.

    Given an observed calibration object, this returns the max(delta_x,
    delta_y). From this we can get a very rough initial estimate of the range to
    the object.

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
