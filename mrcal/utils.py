#!/usr/bin/python2

import numpy as np
import numpysane as nps
import sys
import re
import cv2
import warnings

import mrcal

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


def get_ref_calibration_object(W, H, dot_spacing):
    r'''Returns the geometry of the calibration object in its own coordinate frame

    Shape is (H,W,3). I.e. the x index varies the fastest and each xyz
    coordinate lives at (y,x,:)

    '''

    xx,yy       = np.meshgrid( np.arange(W,dtype=float), np.arange(H,dtype=float))
    full_object = nps.glue(nps.mv( nps.cat(xx,yy), 0, -1),
                           np.zeros((H,W,1)),
                           axis=-1) # shape (H,W,3)
    return full_object * dot_spacing


def show_solution(intrinsics_data, extrinsics, frames, points,
                  observations_board, indices_frame_camera_board,
                  observations_point,  indices_frame_camera_points,
                  distortion_model,

                  axis_scale = 1.0,
                  dot_spacing = 0, Nwant = 10, i_camera=None):
    r'''Plot what a hypothetical 3d calibrated world looks like

    Can be used to visualize the output (or input) of mrcal.optimize(). Not
    coindicentally, the geometric parameters are all identical to those
    mrcal.optimize() takes.

    If we don't have any observed calibration boards, observations_board and
    indices_frame_camera_board should be None

    If we don't have any observed points, observations_point and
    indices_frame_camera_points should be None

    We should always have at least one camera observing the world, so
    intrinsics_data should never be None.

    If we have only one camera, extrinsics will not be referenced.



    COLOR CODING



    The inputs are the same as to mrcal.optimize(). If i_camera is not None, the
    visualization is colored by the reprojection-error-quality of the fit

    '''

    import gnuplotlib as gp

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


    def gen_plot_axes(transforms, label, color = 0, scale = 1.0, label_offset = None):
        r'''Given a list of transforms (applied to the reference set of axes in reverse
        order) and a label, return a list of plotting directives gnuplotlib
        understands. Each transform is an Rt (4,3) matrix

        Transforms are in reverse order so a point x being transformed as A*B*C*x
        can be represented as a transforms list (A,B,C).

        '''
        axes = np.array( ((0,0,0),
                          (1,0,0),
                          (0,1,0),
                          (0,0,2),), dtype=float ) * scale

        transform = mrcal.identity_Rt()

        for x in transforms:
            transform = mrcal.compose_Rt(transform, x)
        axes = np.array([ mrcal.transform_point_Rt(x, transform) for x in axes ])

        axes_forplotting = extend_axes_for_plotting(axes)

        l_axes = tuple(nps.transpose(axes_forplotting)) + \
            ({'with': 'vectors linecolor {}'.format(color), 'tuplesize': 6},)

        l_labels = tuple(nps.transpose(axes*1.01 + \
                                       (label_offset if label_offset is not None else 0))) + \
            (np.array((label,
                       'x', 'y', 'z')),
             {'with': 'labels', 'tuplesize': 4},)
        return l_axes, l_labels




    # I need to plot 3 things:
    #
    # - Cameras
    # - Calibration object poses
    # - Observed points
    def gen_curves_cameras():

        cam0_axes_labels = gen_plot_axes((mrcal.identity_Rt(),), 'cam0', scale=axis_scale)
        cam_axes_labels  = [gen_plot_axes((mrcal.invert_Rt(mrcal.Rt_from_rt(extrinsics[i])),),
                                           'cam{}'.format(i+1),
                                          scale=axis_scale) for i in range(0,extrinsics.shape[-2])]

        return list(cam0_axes_labels) + [ca for x in cam_axes_labels for ca in x]


    def gen_curves_calobjects():

        if observations_board              is None or \
           indices_frame_camera_board      is None or \
           len(observations_board)         == 0    or \
           len(indices_frame_camera_board) == 0:
            return []


        Nobservations = len(indices_frame_camera_board)

        # if i_camera is not None:
        #     i_observations_frames = [(i_observation,indices_frame_camera_board[i_observation,0]) \
        #                              for i_observation in xrange(Nobservations) \
        #                              if indices_frame_camera_board[i_observation,1] == i_camera]

        #     i_observations, i_frames = nps.transpose(np.array(i_observations_frames))
        #     frames = frames[i_frames, ...]


        calobject_ref = get_ref_calibration_object(Nwant, Nwant, dot_spacing)

        Rf = mrcal.utils.Rodrigues_toR_broadcasted(frames[..., :3])
        Rf = nps.mv(Rf,              0, -4)
        tf = nps.mv(frames[..., 3:], 0, -4)

        # object in the cam0 coord system. shape=(Nframes, Nwant, Nwant, 3)
        calobject_cam0 = nps.matmult( calobject_ref, nps.transpose(Rf)) + tf

        # if i_camera is not None:
        #     # shape=(Nobservations, Nwant, Nwant, 2)
        #     if i_camera == 0:
        #         calobject_cam = calobject_cam0
        #     else:
        #         Rc = mrcal.utils.Rodrigues_toR_broadcasted(extrinsics[i_camera-1,:3])
        #         tc = extrinsics[i_camera-1,3:]

        #         calobject_cam = nps.matmult( calobject_cam0, nps.transpose(Rc)) + tc

        #     print "double-check this. I don't broadcast over the intrinsics anymore"
        #     err = observations[i_observations, ...] - mrcal.project(calobject_cam, distortion_model, intrinsics_data[i_camera, ...])
        #     err = nps.clump(err, n=-3)
        #     rms = np.sqrt(nps.inner(err,err) / (Nwant*Nwant))
        #     # igood = rms <  0.4
        #     # ibad  = rms >= 0.4
        #     # rms[igood] = 0
        #     # rms[ibad] = 1
        #     calobject_cam0 = nps.glue( calobject_cam0,
        #                             nps.dummy( nps.mv(rms, -1, -3) * np.ones((Nwant,Nwant)),
        #                                        -1 ),
        #                             axis = -1)

        calobject_cam0 = nps.clump( nps.mv(calobject_cam0, -1, -4), n=-2)

        # if i_camera is not None:
        #     calobject_curveopts = {'with':'lines palette', 'tuplesize': 4}
        # else:
        calobject_curveopts = {'with':'lines', 'tuplesize': 3}

        return tuple(list(calobject_cam0) + [calobject_curveopts,])


    def gen_curves_points():

        if observations_point               is None or \
           indices_frame_camera_points      is None or \
           len(observations_point)          == 0    or \
           len(indices_frame_camera_points) == 0:
            return []

        curveopts = {'with':'points pt 7 ps 2'}
        return [tuple(list(nps.transpose(points)) + [curveopts,])]



    curves_cameras    = gen_curves_cameras()
    curves_calobjects = gen_curves_calobjects()
    curves_points     = gen_curves_points()

    # Need ascii=1 because I'm plotting labels.
    plot = gp.gnuplotlib(_3d=1, square=1, ascii=1,
                         xlabel='x',
                         ylabel='y',
                         zlabel='z')



    plot.plot(*(curves_cameras + curves_calobjects + curves_points))
    return plot


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


def _sample_imager_unproject(gridn_x, gridn_y, distortion_model, intrinsics_data, W, H):
    r'''Reports 3d observation vectors that regularly sample the imager

    This is a utility function for the various visualization routines.
    Broadcasts on the distortion_model and intrinsics_data (if they're lists,
    not numpy arrays. Couldn't do it with numpy arrays because the intrinsics
    have varying sizes)

    Note: the returned matrices index on X and THEN on Y. This is opposite of
    how numpy does it: Y then X. A consequence is that plotting the matrices
    directly will produce transposed images

    '''

    w = np.linspace(0,W-1,gridn_x)
    h = np.linspace(0,H-1,gridn_y)

    # shape: Nwidth,Nheight,2
    grid = nps.reorder(nps.cat(*np.meshgrid(w,h)), -1, -2, -3)

    if type(distortion_model) is list or type(intrinsics_data) is list:
        # shape: Ncameras,Nwidth,Nheight,3
        return np.array([mrcal.unproject(grid,
                                         distortion_model[i],
                                         intrinsics_data[i]) \
                         for i in xrange(len(distortion_model))]), \
               grid
    else:
        # shape: Ncameras,Nwidth,Nheight,3
        return \
            mrcal.unproject(grid,
                            distortion_model,
                            intrinsics_data), \
            grid

# NEED TO DOCUMENT R FIT
def compute_Rcorrected_dq_dintrinsics(q, v, dq_dintrinsics, dq_dv,
                                      imagersize,

                                      # fit everywhere by default
                                      focus_center = None,
                                      focus_radius = 1.e6, # effectively an infinite
                                                           # number of pixels
):
    '''THIS IS ALL WRONG

       Computes D, the solved matrix used to apply an R correction

    I use a subset of my sample points here. The R fit is based on this
    subset. I then use the fit I computed here for all the data.

    '''

    if focus_radius is None or focus_radius <= 0:
        # We're not fitting a rotation to compensate for shifted intrinsics.
        return dq_dp


    # We're fitting the rotation to compensate for shifted intrinsics.

    @nps.broadcast_define( ((3,),), (3,3))
    def skew_symmetric(v):
        return np.array(((   0,  -v[2],  v[1]),
                         ( v[2],    0,  -v[0]),
                         (-v[1],  v[0],    0)))
    V = skew_symmetric(v)

    W,H = imagersize
    if focus_center is None: focus_center = ((W-1.)/2., (H-1.)/2.)



    def clump_leading_dims(x):
        '''clump leading dims to leave 2 trailing ones.

        input shape (..., a,b). Output shape: (N, a,b)
        '''
        return nps.clump(x, n=len(x.shape)-2)


    # everything by default
    V_c     = clump_leading_dims(V)
    dq_dp_c = clump_leading_dims(dq_dp)
    dq_dv_c = clump_leading_dims(dq_dv)

    if focus_radius < 2*(W+H):
        delta_q = nps.clump(q, n=2) - focus_center
        i = nps.norm2(delta_q) < focus_radius*focus_radius
        if np.count_nonzero(i)<3:
            warnings.warn("Focus region contained too few points; I need at least 3. Fitting EVERYWHERE across the imager")
        else:
            V_c     = V_c             [i, ...]
            dq_dv_c = dq_dv_c         [i, ...]
            dq_dp_c = dq_dp_c[i, ...]

    # shape (3,Nintrinsics)
    C_Vvp  = np.sum(nps.matmult( V_c,
                                 nps.transpose(dq_dv_c),
                                 dq_dp_c ),
                    axis=0)

    # shape (3,3)
    C_VvvV  = np.sum(nps.matmult( V_c,
                                  nps.transpose(dq_dv_c),
                                  dq_dv_c,
                                  V_c ),
                     axis=0)

    # shape (3,Nintrinsics)
    D = np.linalg.solve(C_VvvV, C_Vvp)

    # I have D. D is a constant. Used for ALL the samples v. I return the
    # correction; this uses the D, but also a different V for each sample
    return dq_dp - nps.matmult(dq_dv, V, D)


def compute_intrinsics_uncertainty( distortion_model, intrinsics_data,
                                    imagersize,
                                    observed_pixel_uncertainty,
                                    invJtJ_intrinsics,
                                    outlierness  = False,
                                    gridn_x      = 60,
                                    gridn_y      = 40,

                                    # fit a "reasonable" area in the center by
                                    # default
                                    focus_center = None,
                                    focus_radius = -1.
                              ):
    r'''Computes the uncertainty in a projection of a 3D point

    Given a (broadcastable) 3D vector, and the inv(JtJ) matrix for the
    intrinsics, returns the uncertainty of the projection of that vector,
    measured in pixels. This function implements two different methods:

    - input-noise-based (default; selected with outlierness=False)
    - outlierness-based (selected with outlierness=True)

    The approaches for the two methods are different, but the implementation
    ends up being similar.

    We may assume the rotation of this camera is uncertain, and that shifts in
    intrinsics could be compensated for by applying a rotation. For a given
    perturbation in intrinsics I compute the optimal rotation to reduce the
    reprojection error. For this alignment I can use an arbitrary subset of the
    data:

        if focus_radius > 0:
           I use observation vectors within focus_radius pixels of the
           focus_center. To use ALL the data, pass in a very large focus_radius.

        if focus_radius < 0:
           I fit a compensating rotation using a "reasonable" area in the center
           of the imager. I use focus_radius = min(width,height)/6.

        if focus_radius == 0:
           I do NOT fit a compensating rotation. Rationale: with radius == 0, I
           have no fitting data, so I do not fit anything at all.

        if focus_center is None:
           focus_center is at the center of the imager


    *** input-noise-based method

      The pixel observations input to the calibration method are noisy. I assume
      they are zero-mean, independent and gaussian. I treat the x and y
      coordinates of the observations as two independent measurements. I propagate
      this noise through the optimization to compute the resulting noise in the
      parameters. Then I propagate that through a projection to compute the
      resulting uncertainty in projected pixels. Areas with a high expected
      projection error are unreliable for further work.

      Details (from comment for computeUncertaintyMatrices() in mrcal.c):

      I minimize a cost function E = norm2(x) where x is the measurements. Some
      elements of x depend on inputs, and some don't (regularization for instance).
      I perturb the inputs, reoptimize (assuming everything is linear) and look
      what happens to the state p. I'm at an optimum p*:

        dE/dp (p=p*) = 2 Jt x (p=p*) = 0

      I perturb the inputs:

        E(x(p+dp, m+dm)) = norm2( x + J dp + dx/dm dm)

      And I reoptimize:

        dE/ddp ~ ( x + J dp + dx/dm dm)t J = 0

      I'm at an optimum, so Jtx = 0, so

        -Jt dx/dm dm = JtJ dp

      So if I perturb my input observation vector m by dm, the resulting effect on
      the parameters is dp = M dm

        where M = -inv(JtJ) Jt dx/dm

      In order to be useful I need to do something with M. I want to quantify
      how precise our optimal intrinsics are. Ultimately these are always used
      in a projection operation. So given a 3d observation vector v, I project
      it onto our image plane:

        q = project(v, intrinsics)

      I assume an independent, gaussian noise on my input observations, and for a
      set of given observation vectors v, I compute the effect on the projection.

        dq = dproj/dintrinsics dintrinsics
           = dproj/dintrinsics Mi dm

      dprojection/dintrinsics comes from cvProjectPoints2(). I'm assuming
      everything is locally linear, so this is a constant matrix for each v.
      dintrinsics is the shift in the intrinsics of this camera. Mi
      is the subset of M that corresponds to these intrinsics

      If dm represents noise of the zero-mean, independent, gaussian variety,
      then dp and dq are also zero-mean gaussian, but no longer independent

        Var(dq) = (dproj/dintrinsics Mi) Var(dm) (dproj/dintrinsics Mi)t =
                = (dproj/dintrinsics Mi) (dproj/dintrinsics Mi)t s^2
                = dproj/dintrinsics (Mi Mit) dproj/dintrinsicst s^2

      where s is the standard deviation of the noise of each parameter in dm.

      For mrcal, the measurements are

      1. reprojection errors of chessboard grid observations
      2. reprojection errors of individual point observations
      3. range errors for points with known range
      4. regularization terms

      The observed pixel measurements come into play directly into 1 and 2 above,
      but NOT 3 and 4. Let's say I'm doing ordinary least squares, so x = f(p) - m

        dx/dm = [ -I ]
                [  0 ]

      I thus ignore measurements past the observation set.

          [ ... ]           [ ...  ...    ... ]
      M = [  Mi ] and MMt = [ ...  MiMit  ... ]
          [ ... ]           [ ...  ...    ... ]

      The MiMit is the covariance input to this function

      MMt = inv(JtJ) Jt dx/dm dx/dmt J inv(JtJ)

    *** outlierness-based

      This is completely different. I computed a calibration with some input
      data. Let's pretend we use this calibrated camera to observe a 3D vector
      v. The projection of this vector has the same expected observation noise
      as above: diag(s^2).

      If I add this new observation to the optimization, the solution will
      shift, and the reprojection error of this new observation will improve. By
      how much? If it improves by a lot, then we have little confidence in the
      intrinsics in that area. If it doesn't improve by a lot, then we have much
      confidence.

      The big comment in dogleg.c describes the derivation currently. Here I
      implement the "interesting Dima's self-only query outlierness" metric:

      Let p,x,J represent the solution. The new feature we're adding is x* with
      jacobian J*. The solution would move by dp to get to the new optimum.

      Original solution is an optimum: Jt x = 0

      If we add x* and move by dp, we get

        E1 = norm2(x + J dp) + norm2( x* + J* dp )

      The problem including the new point is also at an optimum:

        dE1/dp = 0 -> 0 = Jt x + JtJ dp + J*t x* + J*tJ*dp =
                        =        JtJ dp + J*t x* + J*tJ*dp
      -> dp = -inv(JtJ + J*tJ*) J*t x*

      Woodbury identity:

        -inv(JtJ + J*t J*) =
        = -inv(JtJ) + inv(JtJ) J*t inv(I + J* inv(JtJ) J*t) J* inv(JtJ)

      Let
        A = J* inv(JtJ) J*t   (same as before)
        B = inv(A + I)        (NOT the same as before)

      So
        AB = BA = I-B

      Thus
        -inv(JtJ + J*t J*) =
        = -inv(JtJ) + inv(JtJ) J*t B J* inv(JtJ)

      and

        dp = -inv(JtJ + J*tJ*) J*t x* =
           = -inv(JtJ)J*t x* + inv(JtJ) J*t B J* inv(JtJ) J*t x* =
           = -inv(JtJ)J*t x* + inv(JtJ) J*t B A x*
           = -inv(JtJ)J*t(I - B A) x*
           = -inv(JtJ)J*t B x*   (same as before, but with a different B!)

      Then

        norm2(J dp) = x*t ( B J* inv() Jt J inv() J*t B ) x*
                    = x*t ( B J* inv() J*t B ) x*
                    = x*t ( B A B ) x*
                    = x*t ( B - B*B ) x*

        2 x*t J* dp = -2 x*t J* inv(JtJ)J*t B x* =
                    = -2 x*t A B x* =
                    = x*t (-2AB) x* =
                    = x*t (-2I + 2B) x*

        norm2(J* dp) = x*t ( B J* inv() J*tJ* inv() J*t B ) x* =
                     = x*t ( B A A B ) x* =
                     = x*t ( I - 2B + B*B ) x*

        norm2(x*)    = x*t ( I ) x*

      How do I compute "Dima's self" factor? The "simple" flavor from above looks at
      the new measurement only: norm2(x*). The "interesting" flavor, look at what
      happens to the measurements' error when they're added to the optimization set.
      State moves by dp. x* moves by J* dp. I look at

        dE = norm2(x* + J*dp) - norm2(x*) =
             2 x*t J* dp + norm2(J* dp) =
             x*t (-2I + 2B + I - 2B + B*B) x* =
             x*t (B*B - I) x*

      I expect that adding a point to the optimization would make it fit better: dE <
      0. Let's check. Let's say that there's v,l such that

        (B*B-I)v = l v, norm2(v) = 1
        -->
        BBv      = (l+1)v
        vBBv     = l+1

        Let u = Bv ->
        norm2(u) = l+1

        A = J* inv(JtJ) J*t
        B = inv(A + I) ->

        v = (A+I)u
        norm2(v) = 1 = norm2(Au) + 2ut A u + norm2(u) ->
        -> l = -2ut A u - norm2(Au)

        Let w = J*t u
        -> Au = J* inv(JtJ) w
        -> ut A u = wt inv(JtJ) w ->
        l = -2 wt inv(JtJ) w - norm2(Au)

        Since inv(JtJ) > 0 -> l < 0. As expected

      So B*B-I is negative definite: adding measurements to the optimization set makes
      them fit better

      Putting this together I expect norm2(x*) to improve by x*t (I - B*B) x*.
      x* is a random variable and E(x*t (I - B*B) x*) = tr( (I - B*B) Var(x*) )

      As before, Var(x*) = s^2 I

      So E() = s^2 tr( I - B*B )

    '''

    W,H = imagersize
    if focus_center is None: focus_center = ((W-1.)/2., (H-1.)/2.)
    if focus_radius < 0:     focus_radius = min(W,H)/6.

    v,_ = _sample_imager_unproject(gridn_x, gridn_y,
                                   distortion_model, intrinsics_data,
                                   *imagersize)
    q,dq_dp,dq_dv = \
        mrcal.project(v, distortion_model, intrinsics_data, get_gradients=True)

    dq_dp_corrected = \
        compute_Rcorrected_dq_dintrinsics(q, v, dq_dp,dq_dv,
                                          imagersize,
                                          focus_center, focus_radius)

    if outlierness:

        A  = nps.matmult( dq_dp_corrected, invJtJ_intrinsics, nps.transpose(dq_dp_corrected))
        B  = np.linalg.inv(A + np.eye(2))
        tr = 2 - np.sum( nps.clump(B*B, n=-2),
                         axis = -1)

        # I'm going to pretend that sqrt(E0) - sqrt(E1) = sqrt(E0 - E1). This is
        # true if E0 ~ E1, which maybe is OK here
        return observed_pixel_uncertainty * np.sqrt(tr)

    else:

        # Let x be a 0-mean normally-distributed 2-vector with covariance V. I want
        # E(sqrt(norm2(x))). This is somewhat like a Rayleigh distribution, but with
        # an arbitrary covariance, instead of sI (which is what the Rayleigh
        # distribution expects). I thus compute sqrt(E(norm2(x))) instead of
        # E(sqrt(norm2(x))). Hopefully that's close enough
        #
        # E(norm2(x)) = E(x0*x0 + x1*x1) = E(x0*x0) + E(x1*x1) = trace(V)
        #
        # trace(V) = trace(covariance_intrinsics * dqdpt_dqdp) =
        #          = sum(elementwise_product(covariance_intrinsics, dqdpt_dqdp))
        dqdpt_dqdp = \
            nps.matmult(nps.transpose(dq_dp_corrected),
                        dq_dp_corrected)
        return observed_pixel_uncertainty * \
            np.sqrt(np.sum(nps.clump(invJtJ_intrinsics * dqdpt_dqdp,
                                     n = -2),
                           axis = -1))
def show_intrinsics_uncertainty(distortion_model, intrinsics_data,
                                imagersize,
                                observed_pixel_uncertainty,
                                invJtJ_intrinsics,
                                outlierness      = False,
                                gridn_x          = 60,
                                gridn_y          = 40,

                                # fit a "reasonable" area in the center by
                                # default
                                focus_center = None,
                                focus_radius = -1.,

                                extratitle       = None,
                                hardcopy         = None,
                                cbmax            = None,
                                kwargs = None):
    r'''Visualizes the uncertainty in the intrinsics of a camera

    There are two distinct methods supported by this function:

    - input-noise-based (default; selected with outlierness=False)
    - outlierness-based (selected with outlierness=True)

    This routine uses the covariance of observed inputs. See
    compute_intrinsics_uncertainty() for a description of both routines

    '''

    if kwargs is None: kwargs = {}

    import gnuplotlib as gp
    W,H=imagersize
    err = compute_intrinsics_uncertainty(distortion_model, intrinsics_data,
                                         imagersize,
                                         observed_pixel_uncertainty,
                                         invJtJ_intrinsics,
                                         outlierness,
                                         gridn_x, gridn_y,
                                         focus_center = focus_center,
                                         focus_radius = focus_radius)

    if 'title' not in kwargs:
        if focus_radius < 0:
            focus_radius = min(W,H)/6.

        if focus_radius == 0:
            where = "NOT fitting an implied rotation"
        elif focus_radius > 2*(W+H):
            where = "implied rotation fitted everywhere"
        else:
            where = "implied rotation fit looking at {} with radius {}". \
                format('the imager center' if focus_center is None else focus_center,
                       focus_radius)
        title = "Projection uncertainty (in pixels) based on {}; {}".format("outlierness" if outlierness else "calibration input noise", where)
        if extratitle is not None:
            title += ": " + extratitle
        kwargs['title'] = title

    if 'hardcopy' not in kwargs and hardcopy is not None:
        kwargs['hardcopy'] = hardcopy

    if 'set' not in kwargs:
        kwargs['set'] = []
    elif type(kwargs['set']) is not list:
        kwargs['set'] = [kwargs['set']]
    kwargs['set'].extend(['xrange [:] noextend',
                                   'yrange [:] noextend reverse',
                                   'view equal xy',
                                   'view map',
                                   'contour surface',
                                   'cntrparam levels incremental 10,-0.2,0'])
    plot = \
        gp.gnuplotlib(_3d=1,
                      unset='grid',

                      _xrange=[0,W],
                      _yrange=[H,0],
                      cbrange=[0,cbmax],
                      ascii=1,
                      **kwargs)

    # err has shape (W,H), but the plotter wants what numpy wants: (H,W)
    err = nps.transpose(err)

    using='($1*{}):($2*{}):3'.format(float(W-1)/(gridn_x-1), float(H-1)/(gridn_y-1))

    # Currently "with image" can't produce contours. I work around this, by
    # plotting the data a second time.
    # Yuck.
    # https://sourceforge.net/p/gnuplot/mailman/message/36371128/
    plot.plot( (err, dict(tuplesize=3, _with='image',           using=using)),
               (err, dict(tuplesize=3, _with='lines nosurface', using=using)))
    return plot


def _intrinsics_diff_get_reprojected_grid(grid0, v0, v1,
                                          focus_center,
                                          focus_radius,
                                          distortion_models, intrinsics_data,

                                          # these are for testing mostly. When I figure out
                                          # what I'm doing here, I can probably toss these
                                          imagersizes):

    r'''Computes a undistorted grid for camera1 that corresponds to camera0

    I sample the imager grid in all my cameras, and compute the rotation
    that maps the vectors to each other as closely as possible. Then I
    produce a difference map by projecting the matched-up vectors

    The simplest way to compute this rotation is with a procrustes fit:

        R = align3d_procrustes( nps.clump(v0,n=2),
                                nps.clump(v1,n=2), vectors=True)

    This works, but it minimizes a norm2() metric, and is sensitive to outliers.
    If my distortion model doesn't fit perfectly, I can fit well only in some
    areas. So I try to handle the outliers in two ways:

    - I compute a reasonable seed using a procrustes fit using data in the area
      of interest

    - The residuals in the area of interest will be low-ish. Outside of it they
      may or may not be low, depending on how well the model fits

    - I pick a loose threshold, and throw away all data outside of a
      low-enough-error region around the center of the region of interest

    - Then I run the solve again using a nonlinear optimizer and a robust loss
      function

    The optimization routine I'm using doesn't map to my problem very well, so
    I'm doing something ugly: least squares with the residual vector composed of
    the angle errors. cos(angle) = inner(v0,v1) ~ 1 - x*x -> x = sqrt(1-inner())
    I.e. I compute sqrt() to get the residual, and the optimizer will then
    square it again. I'm guessing this uglyness won't affect me adversely in any
    way

    '''

    # my state vector is a rodrigues rotation, seeded with the identity
    # rotation
    cache = {'r': None}

    def angle_err(v0,v1,R):
        # cos(x) = inner(v0,v1) ~ 1 - x*x
        c = nps.inner(nps.matmult(v0,R), v1)
        return np.sqrt(1-c)

    def residual_jacobian(r):
        R,dRdr = cv2.Rodrigues(r)
        dRdr = nps.transpose(dRdr) # fix opencv's weirdness. Now shape=(9,3)

        x = angle_err(V0fit, V1fit, R)

        # dx/dr = 1/(2x) d(1-c)/dr = -1/(2x) V1ct dV0R/dr
        dV0R_dr = \
            nps.dummy(V0fit[..., (0,)], axis=-1) * dRdr[0:3,:] + \
            nps.dummy(V0fit[..., (1,)], axis=-1) * dRdr[3:6,:] + \
            nps.dummy(V0fit[..., (2,)], axis=-1) * dRdr[6:9,:]

        J = -nps.matmult(nps.dummy(V1fit, -2), dV0R_dr)[..., 0, :] / (2. * nps.dummy(x, -1))
        return x,J

    def residual(r, cache, **kwargs):
        if cache['r'] is None or not np.array_equal(r,cache['r']):
            cache['r'] = r
            cache['x'],cache['J'] = residual_jacobian(r)
        return cache['x']
    def jacobian(r, cache, **kwargs):
        if cache['r'] is None or not np.array_equal(r,cache['r']):
            cache['r'] = r
            cache['x'],cache['J'] = residual_jacobian(r)
        return cache['J']



    W,H = imagersizes[0,:]
    if focus_center is None: focus_center = ((W-1.)/2., (H-1.)/2.)
    if focus_radius < 0:     focus_radius = min(W,H)/6.
    if focus_radius == 0:
        # We assume the geometry is fixed across the two models, and we fit
        # nothing
        R = np.eye(3)

    else:

        V0cut   = nps.clump(v0,n=2)
        V1cut   = nps.clump(v1,n=2)
        icenter = np.array((v0.shape[:2]))/2
        if focus_radius < 2*(W+H):
            # We try to match the geometry in a particular region

            grid_off_center = grid0 - focus_center
            i = nps.norm2(grid_off_center) < focus_radius*focus_radius
            if np.count_nonzero(i)<3:
                warnings.warn("Focus region contained too few points; I need at least 3. Fitting EVERYWHERE across the imager")
            else:
                V0cut = v0[i, ...]
                V1cut = v1[i, ...]

                # get the nearest index on my grid to the requested center
                icenter_flat = np.argmin(nps.norm2(grid_off_center))

                # This looks funny, but it's right. My grid is set up that you index
                # with the x-coord and then the y-coord. This is opposite from the
                # matrix convention that numpy uses: y then x.
                ix = icenter_flat/v0.shape[1]
                iy = icenter_flat - ix*v0.shape[1]
                icenter = np.array((ix,iy))

        # I compute a procrustes fit using ONLY data in the region of interest.
        # This is used to seed the nonlinear optimizer
        R_procrustes = align3d_procrustes( V0cut, V1cut, vectors=True)
        r_procrustes,_ = cv2.Rodrigues(R_procrustes)
        r_procrustes = r_procrustes.ravel()

        e = angle_err(v0,v1,R_procrustes)

        # throw away everything that's k times as wrong as the center of
        # interest. I look at a connected component around the center. I pick a
        # large k here, and use a robust error function further down
        k = 10
        angle_err_at_center = e[icenter[0],icenter[1]]
        threshold = angle_err_at_center*k
        import scipy.ndimage
        regions,_ = scipy.ndimage.label(e < threshold)
        mask = regions==regions[icenter[0],icenter[1]]
        V0fit = v0[mask, ...]
        V1fit = v1[mask, ...]
        # V01fit are used by the optimization cost function

        # Seed from the procrustes solve
        r = r_procrustes

        # gradient check
        # r0 = r
        # x0,J0 = residual_jacobian(r0)
        # dr = np.random.random(3) * 1e-7
        # r1 = r+dr
        # x1,J1 = residual_jacobian(r1)
        # dx_theory = nps.matmult(J0, nps.transpose(dr)).ravel()
        # dx_got    = x1-x0
        # relerr = (dx_theory-dx_got) / ( (np.abs(dx_theory)+np.abs(dx_got))/2. )
        # import gnuplotlib as gp
        # gp.plot(relerr)

        import scipy.optimize
        res = scipy.optimize.least_squares(residual, r, jac=jacobian,
                                           method='dogbox',

                                           loss='soft_l1',
                                           f_scale=angle_err_at_center*3.0,
                                           # max_nfev=1,
                                           args=(cache,),
                                           verbose=0)

        r_fit = res.x
        R_fit,_ = cv2.Rodrigues(r_fit)

        R = R_fit


        # # A simpler routine to JUST move pitch/yaw to align the optical axes
        # r,_ = cv2.Rodrigues(R)
        # r[2] = 0
        # R,_ = cv2.Rodrigues(r)
        # dth_x = \
        #     np.arctan2( intrinsics_data[i,2] - imagersizes[i,0],
        #                 intrinsics_data[i,0] ) - \
        #     np.arctan2( intrinsics_data[0,2] - imagersizes[0,0],
        #                 intrinsics_data[0,0] )
        # dth_y = \
        #     np.arctan2( intrinsics_data[i,3] - imagersizes[i,1],
        #                 intrinsics_data[i,1] ) - \
        #     np.arctan2( intrinsics_data[0,3] - imagersizes[1,1],
        #                 intrinsics_data[0,1] )
        # r = np.array((-dth_y, dth_x, 0))
        # R,_ = cv2.Rodrigues(r)



    # Great. Got R. Reproject.
    return mrcal.project(nps.matmult(v0,R),
                         distortion_models,
                         intrinsics_data)



# show_intrinsics_diff() takes models while
# show_intrinsics_uncertainty_outlierness() takes raw intrinsics data.
# get_intrinsics_uncertainty() does something too
# Yuck. It should be one or the other consistently)
def show_intrinsics_diff(models,
                         gridn_x          = 60,
                         gridn_y          = 40,

                         # fit a "reasonable" area in the center by
                         # default
                         focus_center     = None,
                         focus_radius     = -1.,

                         vectorfield      = False,
                         extratitle       = None,
                         hardcopy         = None,
                         cbmax            = None,
                         kwargs = None):
    r'''Visualize the different between N intrinsic models

    If we're given exactly 2 models then I can either show a vector field of a
    heat map of the differences. I N > 2 then a vector field isn't possible and
    we show a heat map of the standard deviation of the differences. Note that
    for N=2 the difference shows in a-b, which is NOT the standard deviation
    (that is (a-b)/2). I use the standard deviation for N > 2

    This routine fits the implied camera rotation to align the models as much as
    possible. This is required because a camera pitch/yaw motion looks a lot
    like a shift in the camera optical axis (cx,cy). So I could be comparing two
    sets of intrinsics that both represent the same lens faithfully, but imply
    different rotations: the rotation would be compensated for by a shift in
    cx,cy. If I compare the two sets of intrinsics by IGNORING rotations, I
    would get a large diff because of the cx,cy difference. I select the fitting
    region like this:

        if focus_radius > 0:
           I use observation vectors within focus_radius pixels of the
           focus_center. To use ALL the data, pass in a very large focus_radius.

        if focus_radius < 0:
           I fit a compensating rotation using a "reasonable" area in the center
           of the imager. I use focus_radius = min(width,height)/6.

        if focus_radius == 0:
           I do NOT fit a compensating rotation. Rationale: with radius == 0, I
           have no fitting data, so I do not fit anything at all.

        if focus_center is None:
           focus_center is at the center of the imager

    When fitting a rotation, I try to find the largest matching region around
    the center of the area of interest. So the recommentation is to specify
    focus_center and to use a modest focus_radius

    '''

    if kwargs is None: kwargs = {}

    if len(models) > 2 and vectorfield:
        raise Exception("I can only plot a vectorfield when looking at exactly 2 models. Instead I have {}". \
                        format(len(models)))

    import gnuplotlib as gp

    imagersizes = np.array([model.imagersize() for model in models])
    if np.linalg.norm(np.std(imagersizes, axis=-2)) != 0:
        raise Exception("The diff function needs all the imager dimensions to match. Instead got {}". \
                        format(imagersizes))
    W,H=imagersizes[0]

    distortion_models = [model.intrinsics()[0] for model in models]
    intrinsics_data   = [model.intrinsics()[1] for model in models]


    v,grid0 = _sample_imager_unproject(gridn_x, gridn_y,
                                       distortion_models, intrinsics_data,
                                       W, H)

    if len(models) == 2:
        # Two models. Take the difference and call it good
        grid1   = _intrinsics_diff_get_reprojected_grid(grid0,
                                                        v[0,...], v[1,...],
                                                        focus_center, focus_radius,
                                                        distortion_models[1], intrinsics_data[1],
                                                        imagersizes)
        diff    = grid1 - grid0
        difflen = np.sqrt(nps.inner(diff, diff))

    else:
        # Many models. Look at the stdev
        grids = nps.cat(*[_intrinsics_diff_get_reprojected_grid(grid0,
                                                                v[0,...], v[i,...],
                                                                focus_center, focus_radius,
                                                                distortion_models[i], intrinsics_data[i],
                                                                imagersizes) for i in xrange(1,len(v))])
        difflen = np.sqrt(np.mean(nps.norm2(grids-grid0),axis=0))

    if 'title' not in kwargs:
        if focus_radius < 0:
            focus_radius = min(W,H)/6.

        if focus_radius == 0:
            where = "NOT fitting an implied rotation"
        elif focus_radius > 2*(W+H):
            where = "implied rotation fitted everywhere"
        else:
            where = "implied rotation fit looking at {} with radius {}". \
                format('the imager center' if focus_center is None else focus_center,
                       focus_radius)
        title = "Diff looking at {} models; {}".format(len(models), where)
        if extratitle is not None:
            title += ": " + extratitle
        kwargs['title'] = title

    if 'hardcopy' not in kwargs and hardcopy is not None:
        kwargs['hardcopy'] = hardcopy

    if vectorfield:
        plot = gp.gnuplotlib(square=1,
                             _xrange=[0,W],
                             _yrange=[H,0],
                             cbrange=[0,cbmax],
                             **kwargs)

        p0      = nps.clump(grid0,   n=2)
        p1      = nps.clump(grid1,   n=2)
        diff    = nps.clump(diff,    n=2)
        difflen = nps.clump(difflen, n=2)

        plot.plot( p0  [:,0], p0  [:,1],
                   diff[:,0], diff[:,1],
                   difflen,
                   _with='vectors size screen 0.005,10 fixed filled palette',
                   tuplesize=5)
    else:
        if 'set' not in kwargs:
            kwargs['set'] = []
        elif type(kwargs['set']) is not list:
            kwargs['set'] = [kwargs['set']]
        kwargs['set'].extend(['xrange [:] noextend',
                                       'yrange [:] noextend reverse',
                                       'view equal xy',
                                       'view map',
                                       'contour surface',
                                       'cntrparam levels incremental 10,-1,0'])
        plot = \
            gp.gnuplotlib(_3d=1,
                          unset='grid',
                          _xrange=[0,W],
                          _yrange=[H,0],
                          cbrange=[0,cbmax],
                          ascii=1,
                          **kwargs)

        # difflen has shape (W,H), but the plotter wants what numpy wants: (H,W)
        difflen = nps.transpose(difflen)

        using='($1*{}):($2*{}):3'.format(float(W-1)/(gridn_x-1), float(H-1)/(gridn_y-1))
        # Currently "with image" can't produce contours. I work around this, by
        # plotting the data a second time.
        # Yuck.
        # https://sourceforge.net/p/gnuplot/mailman/message/36371128/
        plot.plot( (difflen, dict(               tuplesize=3, _with='image',           using=using)),
                   (difflen, dict(legend="diff", tuplesize=3, _with='lines nosurface', using=using)))
    return plot


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


def _get_correspondences_from_hugin(f):
    r'''Reads correspondences from a hugin .pto file

    Returns an (N,4) numpy array containing (x1,y1, x2,y2) rows

    The input is an opened file'''

    p = np.array((), dtype=float)
    for l in f:
        m = re.match('^c .* x([0-9e\.-]+) y([0-9e\.-]+) X([0-9e\.-]+) Y([0-9e\.-]+)', l)
        if m:
            p = nps.glue(p, np.array([m.group(i+1) for i in xrange(4)], dtype=float),
                         axis=-2)
    return p

def get_correspondences_from_hugin(f):
    r'''Reads correspondences from a hugin .pto file

    Returns an (N,4) numpy array containing (x1,y1, x2,y2) rows

    The input is a filename or an opened file'''

    if type(f) is str:
        with open(f, 'r') as openedfile:
            return _get_correspondences_from_hugin(openedfile)

    return _get_correspondences_from_hugin(f)


def get_mapping_file_framecamera(files_per_camera):
    r'''Parse image filenames to get the frame numbers

    I take in a list of image paths per camera. I return a list:

    - a dict that maps each image filename to (framenumber,cameraindex)
    - a string for the prefix of the FIRST set of images
    - a string for the suffix of the FIRST set of images

    '''

    def get_longest_leading_trailing_substrings(strings):
        r'''Given a list of strings, returns the length of the longest leading and
        trailing substring common to all the strings

        Main use case is to take in strings such as

          a/b/c/frame001.png
          a/b/c/frame002.png
          a/b/c/frame003.png

        and return ("a/b/c/frame00", ".png")

        '''

        # These feel inefficient, especially being written in python. There's
        # probably some built-in primitive I'm not seeing
        def longest_leading_substring(a,b):
            for i in xrange(len(a)):
                if i >= len(b) or a[i] != b[i]:
                    return a[:i]
            return a
        def longest_trailing_substring(a,b):
            for i in xrange(len(a)):
                if i >= len(b) or a[-i-1] != b[-i-1]:
                    if i == 0:
                        return ''
                    return a[-i:]
            return a

        if not strings:
            return (None,None)

        leading  = strings[0]
        trailing = strings[0]

        for s in strings[1:]:
            leading  = longest_leading_substring (leading,s)
            trailing = longest_trailing_substring(trailing,s)
        return leading,trailing

    def pull_framenumbers(files):

        leading,trailing = get_longest_leading_trailing_substrings(files)
        Nleading  = len(leading)
        Ntrailing = len(trailing)

        # I now have leading and trailing substrings. I make sure that all the stuff
        # between the leading and trailing strings is numeric

        # needed because I want s[i:-0] to mean s[i:], but that doesn't work, but
        # s[i:None] does
        Itrailing = -Ntrailing if Ntrailing > 0 else None
        for f in files:
            if not re.match("^[0-9]+$", f[Nleading:Itrailing]):
                raise Exception(("Image filenames MUST be of the form 'something..number..something'\n" +   \
                                 "where the somethings are common to all the filenames. File '{}'\n" + \
                                 "has a non-numeric middle: '{}'. The somethings are: '{}' and '{}'\n" + \
                                 "Did you forget to pass globs for each camera separately?"). \
                                format(f, f[Nleading:Itrailing],
                                       leading, trailing))

        # Alrighty. The centers are all numeric. I gather all the digits around the
        # centers, and I'm done
        m = re.match("^(.*?)([0-9]*)$", leading)
        if m:
            pre_numeric = m.group(2)
            leading     = m.group(1)
        else:
            pre_numeric = ''

        m = re.match("^([0-9]*)(.*?)$", trailing)
        if m:
            post_numeric = m.group(1)
            trailing     = m.group(2)
        else:
            post_numeric = ''

        return [int(pre_numeric + f[Nleading:Itrailing] + post_numeric) for f in files], leading, trailing




    Ncameras = len(files_per_camera)
    mapping = {}
    prefix0 = None
    suffix0 = None
    for icamera in xrange(Ncameras):
        if len(files_per_camera[icamera]) <= 1:
            raise Exception("Camera {} has <=1 images".format(icamera))
        framenumbers, leading, trailing = pull_framenumbers(files_per_camera[icamera])
        if framenumbers is not None:
            if prefix0 is None: prefix0 = leading
            if suffix0 is None: suffix0 = trailing
            mapping.update(zip(files_per_camera[icamera], [(iframe,icamera) for iframe in framenumbers]))
    return mapping, prefix0, suffix0
