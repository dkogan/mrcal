#!/usr/bin/python2

import numpy as np
import numpysane as nps
import sys
import re
import cv2

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


def visualize_solution(intrinsics_data, extrinsics, frames, points,
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


def sample_imager_unproject(gridn_x, gridn_y, distortion_model, intrinsics_data, W, H):
    r'''Reports 3d observation vectors that regularly sample the imager

    Synopsis:

        vectors,pixelgrid = sample_imager_unproject(gridn_x, gridn_y,
                                                    distortion_model, intrinsics_data,
                                                    *imagersize)
        Expected_projection_shift = \
          get_projection_uncertainty(vectors,
                                     distortion_model, intrinsics_data,
                                     covariance_intrinsics)

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
        return mrcal.unproject(grid,
                               distortion_model,
                               intrinsics_data), \
                            grid


def get_projection_uncertainty(V, distortion_model, intrinsics_data, covariance_intrinsics):
    r'''Computes the uncertainty in a projection of a 3D point

    Given a (broadcastable) 3D vector, and the covariance matrix for the
    intrinsics, returns the expected value of the deviation from center of the
    projected point that would result from noise in the calibration
    observations. See comment in visualize_intrinsics_uncertainty() for
    derivation

    '''
    imagePoints,dp_dintrinsics,_ = \
        mrcal.project(V, distortion_model, intrinsics_data, get_gradients=True)
    F = dp_dintrinsics[..., 0:2]
    C = dp_dintrinsics[..., 2:4]

    Cff = covariance_intrinsics[..., 0:2, 0:2]
    Cfc = covariance_intrinsics[..., 0:2, 2:4]
    Ccf = covariance_intrinsics[..., 2:4, 0:2]
    Ccc = covariance_intrinsics[..., 2:4, 2:4]

    # shape Nwidth,Nheight,2,2: each slice is a 2x2 covariance
    Vdprojection = \
        nps.matmult(F,Cff,nps.transpose(F)) + \
        nps.matmult(F,Cfc,nps.transpose(C)) + \
        nps.matmult(C,Ccf,nps.transpose(F)) + \
        nps.matmult(C,Ccc,nps.transpose(C))


    if distortion_model != 'DISTORTION_NONE':
        D = dp_dintrinsics[..., 4:]
        Cfd = covariance_intrinsics[..., 0:2, 4: ]
        Ccd = covariance_intrinsics[..., 2:4, 4: ]
        Cdc = covariance_intrinsics[..., 4:,  2:4]
        Cdd = covariance_intrinsics[..., 4:,  4: ]
        Cdf = covariance_intrinsics[..., 4:,  0:2]
        Vdprojection += \
            nps.matmult(F,Cfd,nps.transpose(D)) + \
            nps.matmult(C,Ccd,nps.transpose(D)) + \
            nps.matmult(D,Cdf,nps.transpose(F)) + \
            nps.matmult(D,Cdc,nps.transpose(C)) + \
            nps.matmult(D,Cdd,nps.transpose(D))

    # Let x be a 0-mean normally-distributed 2-vector with covariance V. I want
    # E(sqrt(norm2(x))). This is somewhat like a Rayleigh distribution, but with
    # an arbitrary covariance, instead of sI (which is what the Rayleigh
    # distribution expects). I thus compute sqrt(E(norm2(x))) instead of
    # E(sqrt(norm2(x))). Hopefully that's close enough
    #
    # E(norm2(x)) = E(x0*x0 + x1*x1) = E(x0*x0) + E(x1*x1) = trace(V)
    @nps.broadcast_define( (('n','n'),), ())
    def trace(x):
        return np.trace(x)
    Expected_projection_shift = np.sqrt(trace(Vdprojection))
    return Expected_projection_shift


def visualize_intrinsics_uncertainty(distortion_model, intrinsics_data,
                                     covariance_intrinsics, imagersize,
                                     gridn_x = 60,
                                     gridn_y = 40,
                                     extratitle = None,
                                     hardcopy = None,
                                     extraplotkwargs = {}):
    r'''Visualizes the uncertainty in the intrinsics of a camera

    This routine uses the covariance of observed inputs

    A calibration process produces the best-fitting camera parameters (intrinsics
    and extrinsics) and a covariance matrix representing the uncertainty in
    these parameters. When we use the intrinsics to project 3D points into the
    image plane, this intrinsics uncertainty creates an uncertainty in the
    resulting projection point. This tool plots the expected value of this
    projection error across the imager. Areas with a high expected projection
    error are unreliable for further work.

    Comment from the mrcal core:

        This is part of sensitivity analysis to quantify how much errors in the
        input pixel observations affect our solution. A "good" solution will not be
        very sensitive: measurement noise doesn't affect the solution very much.

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

        In order to be useful I need to do something with M. Let's say I want to
        quantify how precise our optimal intrinsics are. Ultimately these are always
        used in a projection operation. So given a 3d observation vector v, I project
        it onto our image plane:

          q = project(v, intrinsics)

        I assume an independent, gaussian noise on my input observations, and for a
        set of given observation vectors v, I compute the effect on the projection.

          dq = dprojection/dintrinsics dintrinsics

        dprojection/dintrinsics comes from cvProjectPoints2()
        dintrinsics is the shift in our optimal state: M dm

        If dm represents noise of the zero-mean, independent, gaussian variety, then
        dp is also zero-mean gaussian, but no longer independent.

          Var(dp) = M Var(dm) Mt = M Mt s^2

        where s is the standard deviation of the noise of each parameter in dm.

        The intrinsics of each camera have 3 components:

        - f: focal lengths
        - c: center pixel coord
        - d: distortion parameters

        Let me define dprojection/df = F, dprojection/dc = C, dprojection/dd = D.
        These all come from cvProjectPoints2().

        Rewriting the projection equation I get

          q = project(v,  f,c,d)
          dq = F df + C dc + D dd

        df,dc,dd are random variables that come from dp.

          Var(dq) = F Covar(df,df) Ft +
                    C Covar(dc,dc) Ct +
                    D Covar(dd,dd) Dt +
                    F Covar(df,dc) Ct +
                    F Covar(df,dd) Dt +
                    C Covar(dc,df) Ft +
                    C Covar(dc,dd) Dt +
                    D Covar(dd,df) Ft +
                    D Covar(dd,dc) Ct

        Covar(dx,dy) are all submatrices of the larger Var(dp) matrix we computed
        above: M Mt s^2.

        Here I look ONLY at the interactions of intrinsic parameters for a particular
        camera with OTHER intrinsic parameters of the same camera. I ignore
        cross-camera interactions and interactions with other parameters, such as the
        frame poses and extrinsics.

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

        My matrices are large and sparse. Thus I compute the blocks of M Mt that I
        need here, and return these densely to the upper levels (python). These
        callers will then use these dense matrices to finish the computation

          M Mt = sum(outer(col(M), col(M)))
          col(M) = solve(JtJ, row(J))

    '''

    import gnuplotlib as gp

    W,H=imagersize
    V,_ = sample_imager_unproject(gridn_x, gridn_y,
                                  distortion_model, intrinsics_data,
                                  W, H)
    Expected_projection_shift = get_projection_uncertainty(V, distortion_model, intrinsics_data, covariance_intrinsics)

    if 'title' not in extraplotkwargs:
        title = "Projection uncertainty"
        if extratitle is not None:
            title += ": " + extratitle
        extraplotkwargs['title'] = title

    if 'hardcopy' not in extraplotkwargs and hardcopy is not None:
        extraplotkwargs['hardcopy'] = hardcopy

    if 'set' not in extraplotkwargs:
        extraplotkwargs['set'] = []
    elif type(extraplotkwargs['set']) is not list:
        extraplotkwargs['set'] = [extraplotkwargs['set']]
    extraplotkwargs['set'].extend(['xrange [:] noextend',
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
                      cbrange=[0,5],
                      ascii=1,
                      **extraplotkwargs)

    # Expected_projection_shift has shape (W,H), but the plotter wants what numpy wants: (H,W)
    Expected_projection_shift = nps.transpose(Expected_projection_shift)

    using='($1*{}):($2*{}):3'.format(float(W-1)/(gridn_x-1), float(H-1)/(gridn_y-1))

    # Currently "with image" can't produce contours. I work around this, by
    # plotting the data a second time.
    # Yuck.
    # https://sourceforge.net/p/gnuplot/mailman/message/36371128/
    plot.plot( (Expected_projection_shift, dict(                                    tuplesize=3, _with='image',           using=using)),
               (Expected_projection_shift, dict(legend="Expected_projection_shift", tuplesize=3, _with='lines nosurface', using=using)))
    return plot


def visualize_intrinsics_uncertainty_outlierness(distortion_model, intrinsics_data,
                                                 solver_context, i_camera, observed_pixel_uncertainty,
                                                 imagersize,
                                                 Noutliers,
                                                 gridn_x = 60,
                                                 gridn_y = 40,
                                                 extratitle = None,
                                                 hardcopy = None,
                                                 extraplotkwargs = {}):
    r'''Visualizes the uncertainty in the intrinsics of a camera

    This routine uses the outlierness factor of hypothetical query points

    A calibration process produces the best-fitting camera parameters
    (intrinsics and extrinsics). I throw out outliers based on an "outlierness"
    metric. I use the same metric for the uncertainty computation here: if I add
    a hypothetical observation, and it easily looks outliery, then I have strong
    consensus in the solution in that region, and my confidence there is high.
    Conversely, if it takes a lot to make a query point look like an outlier,
    then the solution is uncertain in that area.

    Comment from the mrcal core:

        I add a hypothetical new measurement, projecting a 3d vector v in the
        coord system of the camera

          x = project(v) - observation

        I compute the projection now, so I know what the observation should be,
        and I can set it such that x=0 here. If I do that, x fits the existing
        data perfectly, and is very un-outliery looking.

        But everything is noisy, so observation will move around, and thus x
        moves around. I'm assuming the observations are mean-0 gaussian, so I let
        my x correspondingly also be mean-0 gaussian.

        I then have a quadratic form outlierness_factor = xt B x
        for some known constant N and known symmetric matrix B. I compute the
        expected value of this quadratic form: E = tr(B * Var(x))

        I get B from libdogleg. See
        dogleg_getOutliernessTrace_newFeature_sparse() for a derivation.

        I'm assuming the noise on the x is independent, so

          Var(x) = observed-pixel-uncertainty^2 I

        And thus E = tr(B) * observed-pixel-uncertainty^2

    '''

    import gnuplotlib as gp

    W,H=imagersize
    V,_ = sample_imager_unproject(gridn_x, gridn_y,
                                  distortion_model, intrinsics_data,
                                  W, H)

    Expected_outlierness = mrcal.queryIntrinsicOutliernessAt( V.copy(), i_camera, solver_context, Noutliers) * \
        observed_pixel_uncertainty * observed_pixel_uncertainty

    if 'title' not in extraplotkwargs:
        title = "Projection uncertainty outlierness"
        if extratitle is not None:
            title += ": " + extratitle
        extraplotkwargs['title'] = title

    if 'hardcopy' not in extraplotkwargs and hardcopy is not None:
        extraplotkwargs['hardcopy'] = hardcopy

    if 'set' not in extraplotkwargs:
        extraplotkwargs['set'] = []
    elif type(extraplotkwargs['set']) is not list:
        extraplotkwargs['set'] = [extraplotkwargs['set']]
    extraplotkwargs['set'].extend(['xrange [:] noextend',
                                   'yrange [:] noextend reverse',
                                   'view equal xy',
                                   'view map',
                                   'contour surface',
                                   'cntrparam levels incremental 0,0.5,10'])
    plot = \
        gp.gnuplotlib(_3d=1,
                      unset='grid',

                      _xrange=[0,W],
                      _yrange=[H,0],
                      cbrange=[0,2],
                      ascii=1,
                      **extraplotkwargs)

    # Expected_outlierness has shape (W,H), but the plotter wants what numpy wants: (H,W)
    Expected_outlierness = nps.transpose(Expected_outlierness)

    using='($1*{}):($2*{}):3'.format(float(W-1)/(gridn_x-1), float(H-1)/(gridn_y-1))

    # Currently "with image" can't produce contours. I work around this, by
    # plotting the data a second time.
    # Yuck.
    # https://sourceforge.net/p/gnuplot/mailman/message/36371128/
    plot.plot( (Expected_outlierness, dict(                                    tuplesize=3, _with='image',           using=using)),
               (Expected_outlierness, dict(legend="Expected_projection_shift", tuplesize=3, _with='lines nosurface', using=using)))
    return plot


def visualize_intrinsics_diff(models,
                              gridn_x = 60,
                              gridn_y = 40,
                              vectorfield = False,
                              extratitle = None,
                              hardcopy = None,
                              extraplotkwargs = {}):
    r'''Visualize the different between N intrinsic models

    If we're given exactly 2 models then I can either show a vector field of a
    heat map of the differences. I N > 2 then a vector field isn't possible and
    we show a heat map of the standard deviation of the differences. Note that
    for N=2 the difference shows in a-b, which is NOT the standard deviation
    (that is (a-b)/2). I use the standard deviation for N > 2

    '''

    if len(models) > 2 and vectorfield:
        raise Exception("I can only plot a vectorfield when looking at exactly 2 models. Instead I have {}". \
                        format(len(models)))

    import gnuplotlib as gp

    imagersizes = np.array([model.dimensions() for model in models])
    if np.linalg.norm(np.std(imagersizes, axis=-2)) != 0:
        raise Exception("The diff function needs all the imager dimensions to match. Instead got {}". \
                        format(imagersizes))
    W,H=imagersizes[0]

    distortion_models = [model.intrinsics()[0] for model in models]
    intrinsics_data   = [model.intrinsics()[1] for model in models]


    V,grid = sample_imager_unproject(gridn_x, gridn_y,
                                     distortion_models, intrinsics_data,
                                     W, H)

    def compute_grid1(i):

        # When I generated the calibration, the rotation of the camera was never
        # fully defined. A small camera rotation looks like a type of distortion
        # since the observed frame poses can compensate for the rotation. Thus
        # here I compute an optimal rotation, and undo its effect before showing
        # the intrinsics differences
        R = align3d_procrustes( nps.clump(V[0,...],n=2),
                                nps.clump(V[i,...],n=2), vectors=True)

        # shape: Nwidth,Nheight,2
        return mrcal.project(nps.matmult(V[0,...],R),
                             distortion_models[i],
                             intrinsics_data[i])



    if len(models) == 2:
        # Two models. Take the difference and call it good
        grid1   = compute_grid1(1)
        diff    = grid1 - grid
        difflen = np.sqrt(nps.inner(diff, diff))
    else:
        # Many models. Look at the stdev
        grids = nps.cat(grid,
                        *[compute_grid1(i) for i in xrange(1,len(V))])
        stdevs  = np.std(grids, axis=0)
        difflen = np.sqrt(nps.inner(stdevs, stdevs))

    if 'title' not in extraplotkwargs:
        title = "Model diff"
        if extratitle is not None:
            title += ": " + extratitle
        extraplotkwargs['title'] = title

    if 'hardcopy' not in extraplotkwargs and hardcopy is not None:
        extraplotkwargs['hardcopy'] = hardcopy

    if vectorfield:
        plot = gp.gnuplotlib(square=1,
                             _xrange=[0,W],
                             _yrange=[H,0],
                             cbrange=[0,10],
                             **extraplotkwargs)

        p0      = nps.clump(grid,    n=2)
        p1      = nps.clump(grid1,   n=2)
        diff    = nps.clump(diff,    n=2)
        difflen = nps.clump(difflen, n=2)

        plot.plot( p0  [:,0], p0  [:,1],
                   diff[:,0], diff[:,1],
                   difflen,
                   _with='vectors size screen 0.005,10 fixed filled palette',
                   tuplesize=5)
    else:
        if 'set' not in extraplotkwargs:
            extraplotkwargs['set'] = []
        elif type(extraplotkwargs['set']) is not list:
            extraplotkwargs['set'] = [extraplotkwargs['set']]
        extraplotkwargs['set'].extend(['xrange [:] noextend',
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
                          cbrange=[0,10],
                          ascii=1,
                          **extraplotkwargs)

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
