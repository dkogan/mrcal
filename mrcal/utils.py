#!/usr/bin/python2

import numpy as np
import numpysane as nps
import sys
import re
import cv2

import poseutils
import projections


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
    understands.

    Transforms are in reverse order so a point x being transformed as A*B*C*x
    can be represented as a transforms list (A,B,C)

    '''
    axes = np.array( ((0,0,0),
                      (1,0,0),
                      (0,1,0),
                      (0,0,2),), dtype=float ) * scale

    transform = poseutils.identity_Rt()

    for x in transforms:
        transform = poseutils.compose_Rt(transform, x)
    axes = np.array([ poseutils.transform_point_Rt(transform, x) for x in axes ])

    axes_forplotting = extend_axes_for_plotting(axes)

    l_axes = tuple(nps.transpose(axes_forplotting)) + \
        ({'with': 'vectors linecolor {}'.format(color), 'tuplesize': 6},)

    l_labels = tuple(nps.transpose(axes*1.01 + \
                                   (label_offset if label_offset is not None else 0))) + \
        (np.array((label,
                   'x', 'y', 'z')),
         {'with': 'labels', 'tuplesize': 4},)
    return l_axes, l_labels


def visualize_solution(distortion_model, intrinsics_data, extrinsics, frames, observations,
                       indices_frame_camera, dot_spacing, Nwant, i_camera=None):
    r'''Plot the best-estimate 3d poses of a hypothesis calibration

    The inputs are the same as to mrcal.optimize(). If i_camera is not
    None, the visualization is colored by the reprojection-error-quality of the
    fit

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

        print "double-check this. I don't broadcast over the intrinsics anymore"
        err = observations[i_observations, ...] - projections.project(object_cam, distortion_model, intrinsics_data[i_camera, ...])
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
    cam0_axes_labels = gen_plot_axes((poseutils.identity_Rt(),), 'cam0')
    cam_axes_labels  = [gen_plot_axes( (poseutils.invert_Rt(Rt_from_rt(extrinsics[i])),),
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
    import gnuplotlib as gp
    plot = gp.gnuplotlib(_3d=1, square=1, ascii=1 )
    plot.plot(*curves)
    return plot


def get_projection_uncertainty(V, distortion_model, intrinsics_data, covariance_intrinsics):
    r'''Computes the uncertainty in a projection of a 3D point

    Given a (broadcastable) 3D vector, and the covariance matrix for the
    intrinsics, returns the expected value of the deviation from center of the
    projected point that would result from noise in the calibration
    observations. See comment in visualize_intrinsics_uncertainty() for
    derivation

    '''
    imagePoints,dp_dintrinsics,_ = \
        projections.project(V, distortion_model, intrinsics_data, get_gradients=True)
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
                                     gridn = 40,
                                     extratitle = None,
                                     hardcopy = None):
    r'''A calibration process produces the best-fitting camera parameters (intrinsics
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
    w = np.linspace(0,W-1,gridn)
    h = np.linspace(0,H-1,gridn)
    # shape: Nwidth,Nheight,2
    grid = nps.reorder(nps.cat(*np.meshgrid(w,h)), -1, -2, -3)

    # shape: Nwidth,Nheight,3
    V = projections.unproject(grid, distortion_model, intrinsics_data)
    Expected_projection_shift = get_projection_uncertainty(V, distortion_model, intrinsics_data, covariance_intrinsics)

    title = "Projection uncertainty"
    if extratitle is not None:
        title += ": " + extratitle

    extraplotkwargs = dict(title = title)
    if hardcopy is not None:
        extraplotkwargs['hardcopy'] = hardcopy

    plot = \
        gp.gnuplotlib(_3d=1,
                      unset='grid',
                      set=['xrange [:] noextend',
                           'yrange [:] noextend reverse',
                           'view equal xy',
                           'view map',
                           'contour surface',
                           'cntrparam levels incremental 10,-0.2,0'],
                      _xrange=[0,W],
                      _yrange=[H,0],
                      cbrange=[0,5],
                      ascii=1,
                      **extraplotkwargs)

    using='($1*{}):($2*{}):3'.format((W-1)/(gridn-1), (H-1)/(gridn-1))

    # Currently "with image" can't produce contours. I work around this, by
    # plotting the data a second time.
    # Yuck.
    # https://sourceforge.net/p/gnuplot/mailman/message/36371128/
    plot.plot( (Expected_projection_shift, dict(                                    tuplesize=3, _with='image',           using=using)),
               (Expected_projection_shift, dict(legend="Expected_projection_shift", tuplesize=3, _with='lines nosurface', using=using)))
    return plot


def visualize_intrinsics_diff(distortion_model0, intrinsics_data0,
                              distortion_model1, intrinsics_data1,
                              imagersize,
                              gridn = 40,
                              vectorfield = False,
                              extratitle = None,
                              hardcopy = None):
    r'''Given two intrinsics, show their projection differences

    We do this either with a vectorfield or a colormap. The former carries more
    information (direction and magnitude), but is less clear at a glance. I
    default to showing a colormap

    '''

    import gnuplotlib as gp

    W,H=imagersize
    w = np.linspace(0,W-1,gridn)
    h = np.linspace(0,H-1,gridn)
    # shape: Nwidth,Nheight,2
    p0 = nps.reorder(nps.cat(*np.meshgrid(w,h)), -1, -2, -3)

    # shape: Nwidth,Nheight,3
    V  = projections.unproject(p0, distortion_model0, intrinsics_data0)

    # shape: Nwidth,Nheight,2
    p1 = projections.project(V, distortion_model1, intrinsics_data1)

    diff    = p1-p0
    difflen = np.sqrt(nps.inner(diff, diff))

    title = "Model diff"
    if extratitle is not None:
        title += ": " + extratitle

    extraplotkwargs = dict(title = title)
    if hardcopy is not None:
        extraplotkwargs['hardcopy'] = hardcopy

    if vectorfield:
        plot = gp.gnuplotlib(square=1, _xrange=[0,W], yrange=[H,0],
                             **extraplotkwargs)

        p0      = nps.clump(p0,      n=2)
        p1      = nps.clump(p1,      n=2)
        diff    = nps.clump(diff,    n=2)
        difflen = nps.clump(difflen, n=2)

        plot.plot( p0  [:,0], p0  [:,1],
                   diff[:,0], diff[:,1],
                   difflen,
                   _with='vectors size screen 0.005,10 fixed filled palette',
                   tuplesize=5)
    else:
        plot = \
            gp.gnuplotlib(_3d=1,
                          unset='grid',
                          set=['xrange [:] noextend',
                               'yrange [:] noextend reverse',
                               'view equal xy',
                               'view map',
                               'contour surface',
                               'cntrparam levels incremental 10,-1,0'],
                          _xrange=[0,W],
                          _yrange=[H,0],
                          cbrange=[0,10],
                          ascii=1,
                          **extraplotkwargs)

        using='($1*{}):($2*{}):3'.format((W-1)/(gridn-1), (H-1)/(gridn-1))
        # Currently "with image" can't produce contours. I work around this, by
        # plotting the data a second time.
        # Yuck.
        # https://sourceforge.net/p/gnuplot/mailman/message/36371128/
        plot.plot( (difflen, dict(                                    tuplesize=3, _with='image',           using=using)),
                   (difflen, dict(legend="Expected_projection_shift", tuplesize=3, _with='lines nosurface', using=using)))
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
