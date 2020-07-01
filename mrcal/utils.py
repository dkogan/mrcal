#!/usr/bin/python3

from __future__ import print_function

import numpy as np
import numpysane as nps
import sys
import re
import cv2
import warnings

import mrcal

@nps.broadcast_define( (('N',3,), ('N',3,),),
                       (4,3), )
def _align3d_procrustes_points(A, B):
    A = nps.transpose(A)
    B = nps.transpose(B)

    # I process Mt instead of M to not need to transpose anything later, and to
    # end up with contiguous-memory results
    Mt = nps.matmult(               A - np.mean(A, axis=-1)[..., np.newaxis],
                      nps.transpose(B - np.mean(B, axis=-1)[..., np.newaxis]))
    V,S,Ut = np.linalg.svd(Mt)

    R = nps.matmult(V, Ut)

    # det(R) is now +1 or -1. If it's -1, then this contains a mirror, and thus
    # is not a physical rotation. I compensate by negating the least-important
    # pair of singular vectors
    if np.linalg.det(R) < 0:
        V[:,2] *= -1
        R = nps.matmult(V, Ut)

    # Now that I have my optimal R, I compute the optimal t. From before:
    #
    #   t = mean(a) - R mean(b)
    t = np.mean(A, axis=-1)[..., np.newaxis] - nps.matmult( R, np.mean(B, axis=-1)[..., np.newaxis] )

    return nps.glue( R, t.ravel(), axis=-2)


@nps.broadcast_define( (('N',3,), ('N',3,),),
                       (3,3), )
def _align3d_procrustes_vectors(A, B):
    A = nps.transpose(A)
    B = nps.transpose(B)

    # I process Mt instead of M to not need to transpose anything later, and to
    # end up with contiguous-memory results
    Mt = nps.matmult( A, nps.transpose(B) )
    V,S,Ut = np.linalg.svd(Mt)

    R = nps.matmult(V, Ut)

    # det(R) is now +1 or -1. If it's -1, then this contains a mirror, and thus
    # is not a physical rotation. I compensate by negating the least-important
    # pair of singular vectors
    if np.linalg.det(R) < 0:
        V[:,2] *= -1
        R = nps.matmult(V, Ut)

    return R


# I use _align3d_procrustes_...() to do the work. Those are separate functions
# with separate broadcasting prototypes
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
    if vectors: return _align3d_procrustes_vectors(A,B)
    else:       return _align3d_procrustes_points (A,B)


def get_ref_calibration_object(W, H, object_spacing, calobject_warp=None):
    r'''Returns the geometry of the calibration object in its own coordinate frame

    Shape is (H,W,3). The corner of the board is at the origin, so
    get_ref_calibration_object(...)[0,0,:] will be a vector of 0

    By default the calibration object is flat: z = 0 for all points. If we want
    to add bowing to the board, pass a length-2 iterable in calobject_warp.
    These describe additive flex along the x axis and along the y axis, in that
    order. In each direction the flex is a parabola, with the parameter k
    describing the max deflection at the center. If the ends are at +- 1 I have
    d = k*(1 - x^2). If the ends are at (0,N-1) the equivalent expression is: d
    = k*( 1 - 4*x^2/(N-1)^2 + 4*x/(N-1) - 1 ) = d = 4*k*(x/(N-1) - x^2/(N-1)^2)
    = d = 4.*k*x*r(1. - x*r)

    '''

    xx,yy       = np.meshgrid( np.arange(W,dtype=float), np.arange(H,dtype=float))
    full_object = nps.glue(nps.mv( nps.cat(xx,yy), 0, -1),
                           np.zeros((H,W,1)),
                           axis=-1) # shape (H,W,3)
    full_object *= object_spacing

    if calobject_warp is not None:
        xr = xx / (W-1)
        yr = yy / (H-1)
        dx = 4. * xr * (1. - xr)
        dy = 4. * yr * (1. - yr)

        full_object[..., 2] += calobject_warp[0] * dx
        full_object[..., 2] += calobject_warp[1] * dy

    return full_object


def make_synthetic_board_observations(models,
                                      object_width_n,object_height_n,
                                      object_spacing, calobject_warp,
                                      at_xyz_rpydeg, noiseradius_xyz_rpydeg,
                                      Nframes):
    r'''Produces synthetic observations of a chessboard

SYNOPSIS

    models = [mrcal.cameramodel("0.cameramodel"),
              mrcal.cameramodel("1.cameramodel"),]
    p = mrcal.make_synthetic_board_observations(models,

                                                # board geometry
                                                10,12,0.1,None,

                                                # Mean board pose
                                                at_xyz_rpydeg,

                                                # Noise radius of the board pose
                                                noiseradius_xyz_rpydeg,

                                                # How many frames we want
                                                100)

    print(p.shape)
    [100, 2, 12, 10, 2]

Given a description of a calibration object and of the cameras observing it,
produces pixel observations of the objects by those cameras. Exactly Nframes
frames of data will be returned. In each frame ALL the cameras will see ALL the
points in the calibration object

The "models" provides the intrinsics and extrinsics.

The calibration objects are nominally have pose at_xyz_rpydeg in the reference
coordinate system, with each pose perturbed uniformly with radius
noiseradius_xyz_rpydeg. I'd like control over roll,pitch,yaw, so this isn't a
normal rt transformation.

Returns a tuple:

- The point observations p:
  array of shape (Nframes, Ncameras, object_height, object_width, 2)
- The pose of the chessboards Rt_cam0_boardref:
  array of shape (Nframes, Ncameras, 4,3). This transforms an object returned by
  make_synthetic_board_observations() to the pose that was projected

ARGUMENTS

- models: an array of mrcal.cameramodel objects, one for each camera we're
  simulating. This is the intrinsics and the extrinsics. Ncameras = len(models)

- object_width_n:  the number of horizontal points in the calibration object grid

- object_height_n: the number of vertical   points in the calibration object grid

- object_spacing: the distance between adjacent points in the calibration object grid

- calobject_warp: a description of the calibration board warping. None means "no
  warping": the object is flat. Otherwise this is an array of shape (2,). See
  the docs for get_ref_calibration_object() for a description.

- at_xyz_rpydeg: the nominal pose of the calibration object, in the reference
  coordinate system. This is an array of shape (6,): the position of the center
  of the object, followed by the roll-pitch-yaw orientation, in degrees

- noiseradius_xyz_rpydeg: the deviation-from-nominal for the chessboard for each
  frame. This is the uniform distribution radius; the elements have the same
  meaning as at_xyz_rpydeg

- Nframes: how many frames of observations to return

    '''

    Ncameras = len(models)

    # I move the board, and keep the cameras stationary.
    #
    # Camera coords: x,y with pixels, z forward
    # Board coords:  x,y along. z forward (i.e. back towards the camera)
    #                rotation around (x,y,z) is (pitch,yaw,roll)
    #                respectively


    # shape: (Nh,Nw,3)
    # The center of the board is at the origin (ignoring warping)
    board_translation = \
        np.array(( (object_height_n-1)*object_spacing/2.,
                   (object_width_n -1)*object_spacing/2.,
                   0 ))
    board_reference = \
        mrcal.get_ref_calibration_object(object_width_n,object_height_n,
                                         object_spacing,calobject_warp) - \
        board_translation

    # Transformation from the board returned by get_ref_calibration_object() to
    # the one I use here. It's a shift to move the origin to the center of the
    # board
    Rt_boardref_origboardref = mrcal.identity_Rt()
    Rt_boardref_origboardref[3,:] = -board_translation

    def get_observation_chunk():
        '''Get some number of observations. I make Nframes of them, but keep only the
        in-bounds ones'''

        xyz             = np.array( at_xyz_rpydeg         [:3] )
        rpy             = np.array( at_xyz_rpydeg         [3:] ) * np.pi/180.
        xyz_noiseradius = np.array( noiseradius_xyz_rpydeg[:3] )
        rpy_noiseradius = np.array( noiseradius_xyz_rpydeg[3:] ) * np.pi/180.

        # shape (Nframes,3)
        xyz = xyz + np.random.uniform(low=-1.0, high=1.0, size=(Nframes,3)) * xyz_noiseradius
        rpy = rpy + np.random.uniform(low=-1.0, high=1.0, size=(Nframes,3)) * rpy_noiseradius

        roll,pitch,yaw = nps.transpose(rpy)

        sr,cr = np.sin(roll), np.cos(roll)
        sp,cp = np.sin(pitch),np.cos(pitch)
        sy,cy = np.sin(yaw),  np.cos(yaw)

        Rp = np.zeros((Nframes,3,3), dtype=float)
        Ry = np.zeros((Nframes,3,3), dtype=float)
        Rr = np.zeros((Nframes,3,3), dtype=float)

        Rp[:,0,0] =   1
        Rp[:,1,1] =  cp
        Rp[:,2,1] =  sp
        Rp[:,1,2] = -sp
        Rp[:,2,2] =  cp

        Ry[:,1,1] =   1
        Ry[:,0,0] =  cy
        Ry[:,2,0] =  sy
        Ry[:,0,2] = -sy
        Ry[:,2,2] =  cy

        Rr[:,2,2] =   1
        Rr[:,0,0] =  cr
        Rr[:,1,0] =  sr
        Rr[:,0,1] = -sr
        Rr[:,1,1] =  cr

        # I didn't think about the order too hard; it might be backwards. It
        # also doesn't really matter. shape (Nframes,3,3)
        R = nps.matmult(Rr, Ry, Rp)

        # shape(Nframes,4,3)
        Rt_cam0_boardref = nps.glue(R, nps.dummy(xyz, -2), axis=-2)

        # shape = (Nframes, Nh,Nw,3)
        boards_cam0 = mrcal.transform_point_Rt( # shape (Nframes, 1,1,4,3)
                                                nps.mv(Rt_cam0_boardref, 0, -5),

                                                # shape ( Nh,Nw,3)
                                                board_reference )

        # I project everything. Shape: (Nframes,Ncameras,Nh,Nw,2)
        p = \
          nps.mv( \
            nps.cat( \
              *[ mrcal.project( mrcal.transform_point_Rt(models[i].extrinsics_Rt_fromref(), boards_cam0),
                                *models[i].intrinsics()) \
                 for i in range(Ncameras) ]),
                  0,1 )

        # I pick only those frames where all observations (over all the cameras) are
        # in view
        iframe = \
            np.all(nps.clump(p,n=-4) >= 0, axis=-1)
        for i in range(Ncameras):
            W,H = models[i].imagersize()
            iframe *= \
                np.all(nps.clump(p[..., 0], n=-3) <= W-1, axis=-1) * \
                np.all(nps.clump(p[..., 1], n=-3) <= H-1, axis=-1)

        # p                has shape (Nframes_inview,Ncameras,Nh*Nw,2)
        # Rt_cam0_boardref has shape (Nframes_inview,4,3)
        return p[iframe, ...], Rt_cam0_boardref[iframe, ...]




    # shape (Nframes_sofar,Ncameras,Nh,Nw,2)
    p = np.zeros((0,
                  Ncameras,
                  object_height_n,object_width_n,
                  2),
                 dtype=float)
    # shape (Nframes_sofar,4,3)
    Rt_cam0_boardref = np.zeros((0,4,3), dtype=float)

    # I keep creating data, until I get Nframes-worth of in-view observations
    while True:
        p_here, Rt_cam0_boardref_here = get_observation_chunk()

        p = nps.glue(p, p_here, axis=-5)
        Rt_cam0_boardref = nps.glue(Rt_cam0_boardref, Rt_cam0_boardref_here, axis=-3)
        if p.shape[0] >= Nframes:
            p                = p               [:Nframes,...]
            Rt_cam0_boardref = Rt_cam0_boardref[:Nframes,...]
            break

    return p, mrcal.compose_Rt(Rt_cam0_boardref, Rt_boardref_origboardref)


def show_calibration_geometry(models_or_extrinsics_rt_fromref,
                              cameranames                 = None,
                              cameras_Rt_plot_ref         = None,
                              frames                      = None,
                              points                      = None,

                              axis_scale         = 1.0,
                              object_spacing     = 0,
                              object_width_n     = None,
                              object_height_n    = None,
                              calobject_warp     = None,
                              i_camera_highlight = None,
                              point_labels       = None,

                              **kwargs):

    r'''Plot what a hypothetical 3d calibrated world looks like

    Can be used to visualize the output (or input) of mrcal.optimize(). Not
    coindicentally, the geometric parameters are all identical to those
    mrcal.optimize() takes.

    If we don't have any observed calibration boards, frames should be None

    If we don't have any observed points, points should be None

    point_labels is a dict from a point index to a name string. Points in this
    dict are plotted with this legend; all other points are plotted under a
    generic "points" legend. This can be omitted to plot all points together

    object_spacing may be omitted ONLY if we are not observing any calibration
    boards

    models_or_extrinsics_rt_fromref is an iterable of cameramodel or (6,) rt
    arrays. An (N,6) array works

    need to check for object_width_n is None

    '''

    import gnuplotlib as gp

    if i_camera_highlight is not None:
        raise Exception("This isn't done yet. Sorry")


    def get_extrinsics_Rt_toref_one(m):
        if isinstance(m, mrcal.cameramodel):
            return m.extrinsics_Rt_toref()
        else:
            return mrcal.invert_Rt(mrcal.Rt_from_rt(m))

    extrinsics_Rt_toref = \
        nps.cat(*[get_extrinsics_Rt_toref_one(m) \
                  for m in models_or_extrinsics_rt_fromref])
    extrinsics_Rt_toref = nps.atleast_dims(extrinsics_Rt_toref, -3)

    if frames is not None: frames = nps.atleast_dims(frames, -2)
    if points is not None: points = nps.atleast_dims(points, -2)


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


    def gen_plot_axes(transforms, legend, scale = 1.0):
        r'''Given a list of transforms (applied to the reference set of axes in reverse
        order) and a legend, return a list of plotting directives gnuplotlib
        understands. Each transform is an Rt (4,3) matrix

        Transforms are in reverse order so a point x being transformed as A*B*C*x
        can be represented as a transforms list (A,B,C).

        if legend is not None: plot ONLY the axis vectors. If legend is None: plot
        ONLY the axis labels

        '''
        axes = np.array( ((0,0,0),
                          (1,0,0),
                          (0,1,0),
                          (0,0,2),), dtype=float ) * scale

        transform = mrcal.identity_Rt()

        for x in transforms:
            transform = mrcal.compose_Rt(transform, x)

        if legend is not None:
            return \
                (extend_axes_for_plotting(mrcal.transform_point_Rt(transform, axes)),
                 dict(_with     = 'vectors',
                      tuplesize = -6,
                      legend    = legend), )
        return tuple(nps.transpose(mrcal.transform_point_Rt(transform,
                                                    axes[1:,:]*1.01))) + \
                                   (np.array(('x', 'y', 'z')),
                                    dict(_with     = 'labels',
                                         tuplesize = 4,
                                         legend    = 'labels'),)




    # I need to plot 3 things:
    #
    # - Cameras
    # - Calibration object poses
    # - Observed points
    def gen_curves_cameras():

        def camera_Rt_toplotcoords(i):
            Rt_ref_cam = extrinsics_Rt_toref[i]
            try:
                Rt_plot_ref = cameras_Rt_plot_ref[i]
                return mrcal.compose_Rt(Rt_plot_ref,
                                        Rt_ref_cam)
            except:
                return Rt_ref_cam

        def camera_name(i):
            try:
                return cameranames[i]
            except:
                return 'cam{}'.format(i)

        cam_axes  = \
            [gen_plot_axes( ( camera_Rt_toplotcoords(i), ),
                            legend = camera_name(i),
                            scale=axis_scale) for i in range(0,len(extrinsics_Rt_toref))]
        cam_axes_labels = \
            [gen_plot_axes( ( camera_Rt_toplotcoords(i), ),
                            legend = None,
                            scale=axis_scale) for i in range(0,len(extrinsics_Rt_toref))]

        # I collapse all the labels into one gnuplotlib dataset. Thus I'll be
        # able to turn them all on/off together
        return cam_axes + [(np.ravel(nps.cat(*[l[0] for l in cam_axes_labels])),
                            np.ravel(nps.cat(*[l[1] for l in cam_axes_labels])),
                            np.ravel(nps.cat(*[l[2] for l in cam_axes_labels])),
                            np.tile(cam_axes_labels[0][3], len(extrinsics_Rt_toref))) + \
                            cam_axes_labels[0][4:]]


    def gen_curves_calobjects():

        if frames is None or len(frames) == 0:
            return []

        if object_spacing <= 0:
            raise Exception("We're observing calibration boards, but their spacing is 0: please pass a valid object_spacing")

        # if observations_board              is None or \
        #    indices_frame_camera_board      is None or \
        #    len(observations_board)         == 0    or \
        #    len(indices_frame_camera_board) == 0:
        #     return []
        # Nobservations = len(indices_frame_camera_board)

        # if i_camera_highlight is not None:
        #     i_observations_frames = [(i_observation,indices_frame_camera_board[i_observation,0]) \
        #                              for i_observation in range(Nobservations) \
        #                              if indices_frame_camera_board[i_observation,1] == i_camera_highlight]

        #     i_observations, i_frames = nps.transpose(np.array(i_observations_frames))
        #     frames = frames[i_frames, ...]


        calobject_ref = get_ref_calibration_object(object_width_n, object_height_n,
                                                   object_spacing, calobject_warp)

        Rf = mrcal.R_from_r(frames[..., :3])
        Rf = nps.mv(Rf,              0, -4)
        tf = nps.mv(frames[..., 3:], 0, -4)

        # object in the cam0 coord system. shape=(Nframes, object_height_n, object_width_n, 3)
        calobject_cam0 = nps.matmult( calobject_ref, nps.transpose(Rf)) + tf

        # if i_camera_highlight is not None:
        #     # shape=(Nobservations, object_height_n, object_width_n, 2)
        #     calobject_cam = nps.transform_point_Rt( models[i_camera_highlight].extrinsics_Rt_fromref(), calobject_cam0 )

        #     print("double-check this. I don't broadcast over the intrinsics anymore")
        #     err = observations[i_observations, ...] - mrcal.project(calobject_cam, *models[i_camera_highlight].intrinsics())
        #     err = nps.clump(err, n=-3)
        #     rms = np.mag(err) / (object_height_n*object_width_n))
        #     # igood = rms <  0.4
        #     # ibad  = rms >= 0.4
        #     # rms[igood] = 0
        #     # rms[ibad] = 1
        #     calobject_cam0 = nps.glue( calobject_cam0,
        #                             nps.dummy( nps.mv(rms, -1, -3) * np.ones((object_height_n,object_width_n)),
        #                                        -1 ),
        #                             axis = -1)

        # calobject_cam0 shape: (3, Nframes, object_height_n*object_width_n).
        # This will broadcast nicely
        calobject_cam0 = nps.clump( nps.mv(calobject_cam0, -1, -4), n=-2)

        # if i_camera_highlight is not None:
        #     calobject_curveopts = {'with':'lines palette', 'tuplesize': 4}
        # else:
        calobject_curveopts = {'with':'lines', 'tuplesize': 3}

        return [tuple(list(calobject_cam0) + [calobject_curveopts,])]


    def gen_curves_points():
        if points is None or len(points) == 0:
            return []

        if point_labels is not None:

            # all the non-fixed point indices
            ipoint_not = np.ones( (len(points),), dtype=bool)
            ipoint_not[np.array(list(point_labels.keys()))] = False

            return \
                [ (points[ipoint_not],
                   dict(tuplesize = -3,
                        _with = 'points',
                        legend = 'points')) ] + \
                [ (points[ipoint],
                   dict(tuplesize = -3,
                        _with = 'points',
                        legend = point_labels[ipoint])) \
                  for ipoint in point_labels.keys() ]

        else:
            return [ (points, dict(tuplesize = -3,
                                   _with = 'points',
                                   legend = 'points')) ]

    curves_cameras    = gen_curves_cameras()
    curves_calobjects = gen_curves_calobjects()
    curves_points     = gen_curves_points()

    plot = gp.gnuplotlib(_3d=1,
                         square=1,
                         xlabel='x',
                         ylabel='y',
                         zlabel='z',
                         **kwargs)



    plot.plot(*(curves_points + curves_cameras + curves_calobjects))
    return plot


def sample_imager(gridn_width, gridn_height, imager_width, imager_height):
    r'''Returns regularly-sampled, gridded pixels coordinates across the imager

SYNOPSIS

    q = sample_imager( 60, 40, 640,480 )

    print(q.shape)
    ===>
    (40,60,2)

Note that the arguments are given in width,height order, as is customary when
generally talking about images and indexing. However, the output is in
height,width order, as is customary when talking about matrices and numpy
arrays.

If we ask for gridding dimensions (gridn_width, gridn_height), the output has
shape (gridn_height,gridn_width,2) where each row is an (x,y) pixel coordinate.

The top-left corner is at [0,0,:]:

    sample_imager(...)[0,0] = [0,0]

The the bottom-right corner is at [-1,-1,:]:

     sample_imager(...)[            -1,           -1,:] =
     sample_imager(...)[gridn_height-1,gridn_width-1,:] =
     (imager_width-1,imager_height-1)

When making plots you probably want to call mrcal.imagergrid_using(). See the
that docstring for details.

ARGUMENTS

- gridn_width: how many points along the horizontal gridding dimension

- gridn_height: how many points along the vertical gridding dimension. If None,
  we compute an integer gridn_height to maintain a square-ish grid:
  gridn_height/gridn_width ~ imager_height/imager_width

- imager_width,imager_height: the width, height of the imager. With a
  mrcal.cameramodel object this is *model.imagersize()

RETURNED VALUES

We return an array of shape (gridn_height,gridn_width,2). Each row is an (x,y)
pixel coordinate.

    '''

    if gridn_height is None:
        gridn_height = int(round(imager_height/imager_width*gridn_width))

    w = np.linspace(0,imager_width -1,gridn_width)
    h = np.linspace(0,imager_height-1,gridn_height)
    return np.ascontiguousarray(nps.mv(nps.cat(*np.meshgrid(w,h)),
                                       0,-1))


def sample_imager_unproject(gridn_width,  gridn_height,
                            imager_width, imager_height,
                            lensmodel, intrinsics_data,
                            normalize = False):
    r'''Reports 3D observation vectors that regularly sample the imager

SYNOPSIS

    import gnuplotlib as gp
    import mrcal

    ...

    Nwidth  = 60
    Nheight = 40

    # shape (Nheight,Nwidth,3)
    v,q = \
        mrcal.sample_imager_unproject(Nw, Nh,
                                      *model.imagersize(),
                                      *model.intrinsics())

    # shape (Nheight,Nwidth)
    f = interesting_quantity(v)

    gp.plot(f,
            tuplesize = 3,
            ascii     = True,
            using     = mrcal.imagergrid_using(model.imagersize, Nw, Nh),
            square    = True,
            _with     = 'image')

This is a utility function used by functions that evalute some interesting
quantity for various locations across the imager. Grid dimensions and lens
parameters are passed in, and the grid points and corresponding unprojected
vectors are returned. The unprojected vectors are unique only up-to-length, and
the returned vectors aren't normalized by default. If we want them to be
normalized, pass normalize=True.

This function has two modes of operation:

- One camera. lensmodel is a string, and intrinsics_data is a 1-dimensions numpy
  array. With a mrcal.cameramodel object together these are *model.intrinsics().
  We return (v,q) where v is a shape (Nheight,Nwidth,3) array of observation
  vectors, and q is a (Nheight,Nwidth,2) array of corresponding pixel
  coordinates (the grid returned by sample_imager())

- Multiple cameras. lensmodel is a list or tuple of strings; intrinsics_data is
  an iterable of 1-dimensional numpy arrays (a list/tuple or a 2D array). We
  return the same q as before (only one camera is gridded), but the unprojected
  array v has shape (Ncameras,Nheight,Nwidth,3) where Ncameras is the leading
  dimension of lensmodel. The gridded imager appears in camera0: v[0,...] =
  unproject(q)

ARGUMENTS

- gridn_width: how many points along the horizontal gridding dimension

- gridn_height: how many points along the vertical gridding dimension. If None,
  we compute an integer gridn_height to maintain a square-ish grid:
  gridn_height/gridn_width ~ imager_height/imager_width

- imager_width,imager_height: the width, height of the imager. With a
  mrcal.cameramodel object this is *model.imagersize()

- lensmodel, intrinsics_data: the lens parameters. With a single camera,
  lensmodel is a string, and intrinsics_data is a 1-dimensions numpy array; with
  a mrcal.cameramodel object together these are *model.intrinsics(). With
  multiple cameras, lensmodel is a list/tuple of strings. And intrinsics_data is
  an iterable of 1-dimensional numpy arrays (a list/tuple or a 2D array).

- normalize: optional boolean defaults to False. If True: normalize the output
  vectors

RETURNED VALUES

We return a tuple:

- v: the unprojected vectors. If we have a single camera this has shape
  (Nheight,Nwidth,3). With multiple cameras this has shape
  (Ncameras,Nheight,Nwidth,3). These are NOT normalized by default. To get
  normalized vectors, pass normalize=True

- q: the imager-sampling grid. This has shape (Nheight,Nwidth,2) regardless of
  how many cameras were given (we always sample just one camera). This is what
  sample_imager() returns

    '''

    def is_list_or_tuple(l):
        return isinstance(l,list) or isinstance(l,tuple)


    # shape: (Nheight,Nwidth,2). Contains (x,y) rows
    grid = sample_imager(gridn_width, gridn_height, imager_width, imager_height)

    if is_list_or_tuple(lensmodel):
        # shape: Ncameras,Nwidth,Nheight,3
        return np.array([mrcal.unproject(np.ascontiguousarray(grid),
                                         lensmodel[i],
                                         intrinsics_data[i]) \
                         for i in range(len(lensmodel))]), \
               grid
    else:
        # shape: Nheight,Nwidth,3
        return \
            mrcal.unproject(np.ascontiguousarray(grid),
                            lensmodel, intrinsics_data), \
            grid


def compute_Rcorrected_dq_dintrinsics( # The gridded points used to compute the R correction
                                       q_grid, v_grid, dqgrid_dp, dqgrid_dv,

                                       # The query points. We return the gradient for thes
                                       v, dq_dp, dq_dv,
                                       imagersize,

                                       # fit everywhere by default
                                       focus_center = None,
                                       focus_radius = -1.):
    '''Returns dq/dintrinsics with a compensating rotation applied

    When we evaluate the uncertainty of a solution, we want to see what would
    happen to a projection in a specific location if the optimization vector
    (intrinsics, extrinsics, poses of the observed objects) were perturbed. A
    naive way to do this is to

    - take a point-of-interest on the imager q
    - unproject it into a vector v in the 3D camera coordinates
    - reproject v using the perturbed intrinsics
    - compare that projection to q

    This works, but makes an incorrect assumption: it assumes the pose of the
    camera is fixed, and well-known. This is never true. Perturbing the full
    optimization vector moves the camera, so by comparing projections of a
    single vector v in the camera coord system, we're actually comparing
    projections of two different points because the camera coord system is
    perturbed also. Note that this happens EVEN IN A MONOCULAR CALIBRATION. In
    this case the camera coord system is the reference, so we have no explicit
    representation of this coord system in our optimization vector, but we DO
    represent observed poses, and those can all move in unison to effectively
    move the camera.

    The less naive way of evaluating the uncertainty of projection is to

    - take a point-of-interest on the imager q
    - unproject it into a vector v in the 3D camera coordinates
    - infer a rotation R that describes the motion of the camera due to the
      optimization vector perturbation
    - Apply this rotation R to the observation vector v
    - reproject the rotated v using the perturbed intrinsics
    - compare that projection to q

    In theory I can infer a full 6DOF transformation R,t instead of just R, but
    this would affect points at different ranges differently, so I use just R
    for now. This is similar in concept to what we do in show_intrinsics_diff(),
    but there we explicitly compute R with a Procrustes fit, while here we have
    to do this analytically

    This procedure is clear, but we have to decide how to compute the rotation
    R. I do this by

    - sampling the imager in a regular grid
    - unprojecting the 2D observations q in this grid using the intrinsics
      before AND after the perturbation of the intrinsics
    - computing an optimal rotation to match the two sets of vectors v

    Since this allows me to pick the set of vectors that I use to compute the
    rotation, I can focus this fit in any areas I choose. This means that I can
    make the projection uncertainty as low as I want in any area (by matching
    the rotations very well), but not ALL areas at the same time. This makes
    sense since if I care about only a tiny-enough area in the imager, then all
    calibrations are correct. A truly good calibration in general is one that
    fits well in a large area.

    The area is specified using the focus_radius and focus_center arguments, as
    with the other functions:

        if focus_radius > 0:
           I use observation vectors within focus_radius pixels of the
           focus_center. To use ALL the data, pass in a very large focus_radius.

        if focus_radius < 0 (this is the default):
           I fit a compensating rotation using a "reasonable" area in the center
           of the imager. I use focus_radius = min(width,height)/6.

        if focus_radius == 0:
           I do NOT fit a compensating rotation. Rationale: with radius == 0, I
           have no fitting data, so I do not fit anything at all.

        if focus_center is None (this is the default):
           focus_center is at the center of the imager

    Derivation:

    I'm looking at projections of some points v in the camera coord system;
    these v project to some coordinates q in the imager. I perturb the
    intrinsics of the camera, which moves where these v are projected to. I find
    a rotation R that minimizes these differences in projection as much as
    possible. For any single vector v I have an error

      e = proj(Rv, intrinsics+dintrinsics) - proj(v, intrinsics)

    Let's say the rotation R is of length th around an axis k (unit vector).
    This is a small perturbation, so I assume that th ~ 0:

      Rv = v cos(th) + cross(k,v) sin(th) + (1-cos(th))dot(k,v)k
         ~ v + cross(k,v) th

    Let my rotation be represented as a Rodrigues vector r = k th. And let V be
    a skew-symmetric matrix representing a cross product:

      cross(k,v) = -cross(v,k) = -Vk ->

      Rv ~ v + cross(k,v) th = v - V k th = v - V r

    So

      e_i = proj(v_i - V_i r, intrinsics+dintrinsics) - proj(v_i, intrinsics)
          ~ dproj_i/dintrinsics dintrinsics - dproj_i/dv_i V_i r

    I have dproj_i/dwhatever for each point from the projection function.
    dintrinsics is my perturbation. I have V (directly from the vector I'm
    projecting). r represents the rotation I'm looking for. I find the r that
    minimizes a joint error for a number of projections.

      E = sum( norm2(e_i) )
      dE/dr = 0 ~ sum( V_it dproj_i/dv_it (dproj_i/dintrinsics dintrinsics - dproj_i/dv_i V_i r) )
            = sum( V_it dproj_i/dv_it dproj_i/dintrinsics dintrinsics) -
              sum( V_it dproj_i/dv_it dproj_i/dv_i V_i r)

    ---> r = inv( sum( V_it dproj_i/dv_it dproj_i/dv_i V_i r) ) sum( V_it dproj_i/dv_it dproj_i/dintrinsics ) dintrinsics

    For simplicity, let's define matrices I can compute explicitly:

      C_VvvV = sum( V_i dproj_i/dv_it dproj_i/dv_i V_i r)
      C_Vvp  = sum( V_i dproj_i/dv_it dproj_i/dintrinsics )

    Then (I used the fact that Vt = -V)

      r = inv(C_VvvV) C_Vvp dintrinsics

    Defining

      M = inv(C_VvvV) C_Vvp

    I get

      r = M dintrinsics

    Cool! Given some projections in an area I want to match, I can explicitly
    compute M. Then for any perturbation in intrinsics, I can compute the
    compensating rotation r. Let's take the next step, and use this.

    Ultimately the question we'll be asking is "what is the shift in projection
    due to a perturbation in intrinsics?" This is e_i from above:

      e_i ~ dproj_i/dintrinsics dintrinsics - dproj_i/dv_i V_i r

    Except we already have r = M dintrinsics, so

      e_i ~ (dproj_i/dintrinsics - dproj_i/dv_i V_i M) dintrinsics

    Thus this function returns the CORRECTED dproj_i/dintrinsics:

      dproj_i/dintrinsics - dproj_i/dv_i V_i M

    '''

    if focus_radius is None or focus_radius <= 0:
        # We're not fitting a rotation to compensate for shifted intrinsics.
        return dq_dp

    @nps.broadcast_define( ((3,),), (3,3))
    def skew_symmetric(v):
        return np.array(((   0,  -v[2],  v[1]),
                         ( v[2],    0,  -v[0]),
                         (-v[1],  v[0],    0)))
    V_grid = skew_symmetric(v_grid)

    W,H = imagersize
    if focus_center is None: focus_center = ((W-1.)/2., (H-1.)/2.)



    def clump_leading_dims(x):
        '''clump leading dims to leave 2 trailing ones.

        input shape (..., a,b). Output shape: (N, a,b)
        '''
        return nps.clump(x, n=len(x.shape)-2)


    # everything by default
    V_c_grid = clump_leading_dims(V_grid)
    dqgrid_dp_c  = clump_leading_dims(dqgrid_dp)
    dqgrid_dv_c  = clump_leading_dims(dqgrid_dv)

    if focus_radius < 2*(W+H):
        delta_q = nps.clump(q_grid, n=2) - focus_center
        i = nps.norm2(delta_q) < focus_radius*focus_radius
        if np.count_nonzero(i)<3:
            warnings.warn("Focus region contained too few points; I need at least 3. Fitting EVERYWHERE across the imager")
        else:
            V_c_grid     = V_c_grid   [i, ...]
            dqgrid_dv_c  = dqgrid_dv_c[i, ...]
            dqgrid_dp_c  = dqgrid_dp_c[i, ...]

    # shape (3,Nintrinsics)
    C_Vvp  = np.sum(nps.matmult( V_c_grid,
                                 nps.transpose(dqgrid_dv_c),
                                 dqgrid_dp_c ),
                    axis=0)

    # shape (3,3)
    C_VvvV  = np.sum(nps.matmult( V_c_grid,
                                  nps.transpose(dqgrid_dv_c),
                                  dqgrid_dv_c,
                                  V_c_grid ),
                     axis=0)

    # shape (3,Nintrinsics)
    M = np.linalg.solve(C_VvvV, C_Vvp)

    # I have M. M is a constant. Used for ALL the samples v. I now apply this
    # correction to the query points in dq_dp, dq_dv; these may not be the same
    # points as the gridded ones used to compute the correction M.
    V = skew_symmetric(v)
    return dq_dp - nps.matmult(dq_dv, V, M)


def imagergrid_using(imagersize, gridn_width, gridn_height = None):
    '''Get a 'using' expression for imager colormap plots

SYNOPSIS

    import gnuplotlib as gp
    import mrcal

    ...

    Nwidth  = 60
    Nheight = 40

    # shape (Nheight,Nwidth,3)
    v,_ = \
        mrcal.sample_imager_unproject(Nw, Nh,
                                      *model.imagersize(),
                                      *model.intrinsics())

    # shape (Nheight,Nwidth)
    f = interesting_quantity(v)

    gp.plot(f,
            tuplesize = 3,
            ascii     = True,
            using     = mrcal.imagergrid_using(model.imagersize, Nw, Nh),
            square    = True,
            _with     = 'image')

We often want to plot some quantity at every location on the imager (intrinsics
uncertainties for instance). This is done by gridding the imager, computing the
quantity at each grid point, and sending this to gnuplot. This involves a few
non-obvious plotting idioms, with the full usage summarized in the above
example.

Due to peculiarities of gnuplot, the 'using' expression produced by this
function can only be used in plots using ascii data commands (i.e. pass
'ascii=True' to gnuplotlib).

ARGUMENTS

- imagersize: a (width,height) tuple for the size of the imager. With a
  mrcal.cameramodel object this is model.imagersize()

- gridn_width: how many points along the horizontal gridding dimension

- gridn_height: how many points along the vertical gridding dimension. If
  omitted or None, we compute an integer gridn_height to maintain a square-ish
  grid: gridn_height/gridn_width ~ imager_height/imager_width

    '''

    W,H = imagersize
    if gridn_height is None:
        gridn_height = int(round(H/W*gridn_width))
    return '($1*{}):($2*{}):3'.format(float(W-1)/(gridn_width-1), float(H-1)/(gridn_height-1))


def compute_projection_stdev( model,
                              v            = None,
                              gridn_width  = None,
                              gridn_height = None,

                              # fit a "reasonable" area in the center by
                              # default
                              focus_center = None,
                              focus_radius = -1.):
    r'''Computes the uncertainty in a projection of a 3D point

    Given a (broadcastable) 3D vector, and the covariance matrix for the
    intrinsics, returns the uncertainty of the projection of that vector,
    measured in pixels. This function broadcasts over v.

    This function implements two different methods:

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

        if focus_radius < 0 (this is the default):
           I fit a compensating rotation using a "reasonable" area in the center
           of the imager. I use focus_radius = min(width,height)/6.

        if focus_radius == 0:
           I do NOT fit a compensating rotation. Rationale: with radius == 0, I
           have no fitting data, so I do not fit anything at all.

        if focus_center is None (this is the default):
           focus_center is at the center of the imager


    *** input-noise-based method

      The below derivation is double-checked in test-calibration.py

      The pixel observations input to the calibration method are noisy. I assume
      they are zero-mean, independent and gaussian. I treat the x and y
      coordinates of the observations as two independent measurements. I propagate
      this noise through the optimization to compute the resulting noise in the
      parameters. Then I propagate that through a projection to compute the
      resulting uncertainty in projected pixels. Areas with a high expected
      projection error are unreliable for further work.

      Details:

      I minimize a cost function E = norm2(x) where x is the measurements. In
      the full optimization some elements of x depend on inputs, and some don't
      (regularization for instance). For most measurements we have a weighted
      reprojection error: xi = wi (qi - qrefi). Uncertain measurements (high
      Var(qrefi)) are weighted less (lower wi), so the noise on qrefi (on x and
      on y separately) is assumed to be mean-0 gaussian with stdev
      observed_pixel_uncertainty/wi. So the noise on xi has stdev
      observed_pixel_uncertainty. I perturb the inputs, reoptimize (assuming
      everything is linear) and look what happens to the state p. I'm at an
      optimum p*:

        dE/dp (p=p*) = 2 Jt x (p=p*) = 0

      I perturb the inputs:

        E(x(p+dp, qref+dqref)) = norm2( x + J dp + dx/dqref dqref)

      And I reoptimize:

        dE/ddp ~ ( x + J dp + dx/dqref dqref)t J = 0

      I'm at an optimum, so Jtx = 0, so

        -Jt dx/dqref dqref = JtJ dp

      For the reprojection errors I have xobservation = W ( f(p) - qref ), so

        dx/dqref = [ -W ]
                   [  0 ]

        Jobservationst W dqref = JtJ dp

      So if I perturb my input observation vector qref by dqref, the resulting
      effect on the parameters is dp = M dqref. Where

        M = inv(JtJ) Jobservationst W

      So

        Var(p) = M Var(dqref) Mt

      In particular, if we want the variance of the intrinsics for a camera in
      isolation (assuming these aren't correlated with the extrinsics and the
      parameters of the other cameras), I get Mi: the row-subset of M that
      corresponds to the intrinsics I care about, and I have

        Var(intrinsics) = Mi Var(dqref) Mit
                        = Mi W^-2 s^2 Mit

      where s is observed_pixel_uncertainty

      Mi W^-1 = inv(JtJ)[intrinsicsrows] Jobservationst W W^-1 =
              = inv(JtJ)[intrinsicsrows] Jobservationst

      -> Var(intrinsics) = (inv(JtJ)[intrinsicsrows] Jobservationst)
                           (inv(JtJ)[intrinsicsrows] Jobservationst)t
                           s^2

      If Jobservations was J (no regularization terms) then this would simplify
      even more. inv(JtJ)[intrinsics] is a row-subset of inv(JtJ). Let X = JtJ,
      B = inv(JtJ)[intrinsics]. I would then have

        Var(intrinsics)/s^2 = B X Bt

                          [ A ]         [ AX ]
        I = inv(JtJ)JtJ = [ B ] [ X ] = [ BX ] -> B X = [0 I 0]
                          [ C ]         [ CX ]
      -> B X Bt = row,col slice of inv(JtJ)

      But I DO have regularization terms, so J != Jobservations. I thus leave it
      as is:

      Var(intrinsics)/s^2 =
         inv(JtJ)[intrinsicsrows] Jobservationst
         Jobservations inv(JtJ)[intrinsicsrows]t


      Ultimately the intrinsics are always used in a projection operation. So
      given a 3d observation vector v, I project it onto our image plane:

        q = project(v, intrinsics)

        dq = dproj/dintrinsics dintrinsics

        Var(q) = dproj/dintrinsics Var(intrinsics) dproj/dintrinsicst

      dproj/dintrinsics comes from mrcal_project(). I'm assuming everything is
      locally linear, so this is a constant matrix for each v. dintrinsics is
      the shift in the intrinsics of this camera.

      I want to convert Var(q) into a single number that describes my projection
      uncertainty at q. The x,y components of q will be roughly independent,
      with roughly the same stdev, so I estimate this stdev:

      stdev(q) ~ sqrt( trace(Var(q))/2 )

      I can also look at the full ellipse defined by the variance, and take the
      worst-case defiation: the major axis:

      eig (a b) --> (a-l)*(c-l)-b^2 = 0 --> l^2 - (a+c) l + ac-b^2 = 0
          (b c)

      --> l = (a+c +- sqrt( a^2+2ac+c^2 - 4ac + 4b^2)) / 2 =
            = (a+c +- sqrt( a^2-2ac+c^2 + 4b^2)) / 2 =
            = (a+c)/2 +- sqrt( (a-c)^2/4 + b^2)

      So the worst-case stdev(q) is sqrt((a+c)/2 + sqrt( (a-c)^2/4 + b^2))

      tr(AB) = tr(BA) ->
      trace(Var(q)) = tr( Var(intrinsics) dproj/dintrinsicst dproj/dintrinsics )
                    = sum(elementwise_product(Var(intrinsics),
                                              dqdpt_dqdp))

    *** outlierness-based

      The below was written before weighted measurements were implemented. It's
      mostly all unused, so I'm not updating this to handle weighting

      This is a completely different technique of estimating uncertainty, but
      produces a very similar result. I computed a calibration with some input
      data. Let's pretend we use this calibrated camera to observe a 3D vector v.
      The projection of this vector has the same expected observation noise as
      above.

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

      Let's say the weights on x* are 1.0, so Var(x*) = s^2 I

      So E() = s^2 tr( I - B*B )

    *** extrinsics uncertainty

      How do I compute the uncertainty in my extrinsics? This is similar to the
      intrinsics procedures above, but it's simpler on some level: the
      uncertainty in the parameters themselves is meaningful, I don't need to
      propagate that to projection. The derivation works exactly the same way as
      before, and

      -> Var(extrinsics) = (inv(JtJ)[extrinsicsrows] Jobservationst)
                           (inv(JtJ)[extrinsicsrows] Jobservationst)t
                           s^2

      The frame poses aren't known, however, so the extrinsics aren't defined
      globally and I'm going to want to look at the extrinsics uncertainties of
      some cameras in respect to other cameras. If I don't do this then any
      camera at the coordinate-system reference has infinitely-confident
      extrinsics; which is wrong. An example of what I might want to do: let's
      say I have 4 cameras, with camera0 at the reference (3 sets of
      extrinsics). How do I compute Var(yaw between cameras 2,3)? For simplicity
      let's assume the cameras have relative transformation ~ identity, so yaw ~
      rodrigues[1]. I have relative extrinsics for my cameras: rt20, rt30 ->
      rt23[1] = compose(rt20, invert(rt30))[1]. I linearize this, so that
      locally rt23[1] ~ rt23[1](0) + drt23[1]/drt20 drt20 + drt23[1]/drt30 drt30
      = rt23[1](0) + A drt20 + B drt30 ~ AB drt2030

      -> Var(rt23[1]) = AB Var(drt2030) ABt =
                      = AB Mrt2030 Var(qref) Mrt2030t ABt =
                      = AB Mrt2030 W^-2 s^2 Mrt2030t ABt =
                      = AB inv(JtJ)_2030 Jobservationst W W^-2 s^2 W Jobservations inv(JtJ)_2030t ABt =
                      = AB inv(JtJ)_2030 Jobservationst s^2 Jobservations inv(JtJ)_2030t ABt

      Thus I make available the full covariance of all the extrinsics variables,
      so that all the pairwise relationships can be observed.

    '''

    if (gridn_width is     None and gridn_height is not None) or \
       (gridn_width is not None and gridn_height is     None):
        raise Exception("gridn_width,gridn_height must both be None or neither may be None")
    if gridn_width is None and v is None:
        raise Exception("gridn_width,gridn_height may not both be None")

    lensmodel, intrinsics_data = model.intrinsics()
    imagersize                 = model.imagersize()
    W,H = imagersize
    if focus_center is None: focus_center = ((W-1.)/2., (H-1.)/2.)
    if focus_radius < 0:     focus_radius = min(W,H)/6.

    if gridn_width is not None:
        v_grid,_ = \
            mrcal.sample_imager_unproject(gridn_width, gridn_height,
                                          *imagersize, lensmodel, intrinsics_data)
    else:
        # Don't have the grid parameters. Assume my v is already gridded
        v_grid = v
    if v is None:
        # Don't have v. Use the grid for that
        v = v_grid

    q,dq_dv,dq_dp = \
        mrcal.project(v, lensmodel, intrinsics_data, get_gradients=True)
    if v_grid is v:
        q_grid,dqgrid_dv,dqgrid_dp = \
            q,dq_dv,dq_dp
    else:
        q_grid,dqgrid_dv,dqgrid_dp = \
            mrcal.project(v_grid, lensmodel, intrinsics_data, get_gradients=True)

    intrinsics_covariance = model.covariance_intrinsics()
    if intrinsics_covariance is None:
        raise Exception("The given camera model doesn't have the intrinsics covariance. Can't compute uncertainty.")

    dq_dp_corrected = \
        compute_Rcorrected_dq_dintrinsics(q_grid, v_grid, dqgrid_dp,dqgrid_dv,
                                          v, dq_dp,dq_dv,
                                          imagersize,
                                          focus_center, focus_radius)
    dqdpt_dqdp = \
        nps.matmult(nps.transpose(dq_dp_corrected),
                    dq_dp_corrected)
    return \
        np.sqrt(np.sum(nps.clump(intrinsics_covariance * dqdpt_dqdp,
                                 n = -2),
                       axis = -1) \
                / 2.)

def compute_projection_covariance_from_solve( q, distance,
                                              icam_intrinsics, icam_extrinsics,
                                              lensmodel, intrinsics_data, extrinsics, frames,
                                              solver_context,
                                              pixel_uncertainty_stdev,

                                              cachelist_invJtJ_JobstJobs = None):
    r'''Computes the uncertainty in a projection of a 3D point

    Unlike compute_projection_stdev(), does take into account the variances of
    the extrinsics and of the frames, i.e. it uses the full jacobian. This data
    isn't available in the .cameramodel, so this has to be called right after we
    optimized a new model.

    No compensating rotations need to be computed: the extra bit of geometry is
    provided by the extrinsics and frames variances.

    Computed off the Jacobian returned by the solver. The C layer does no
    computation here. It would be faster if it did, though

    Currently this function more or less assumes a vanilla calibration problem:
    lots of stationary cameras are observing a moving chessboard, no point
    observations, camera0 defines the reference coordinate system

    everything we care about broadcasts: q, intrinsics_data, extrinsics, distance
    except you have to pass in nonstandard broadcasting dimensions

    document cachelist_invJtJ_JobstJobs

    distance is None: process just one distance: infinity

    todo:

    - Make this work in the general case. Observed points or moving cameras make
      this break. Outliers?

    - Make this efficient. I'm using dumb, dense linear algebra here, and I'm not
      taking advantage of matrix symmetries. For small problems and simple, parametric
      camera models this is ok. A splined model would probably bring this
      implementation to its knees

    - This function transforms points to the coord system of the frames, and back; I
      do that for the gradients. Do I need to? Can I grab the gradients in one
      direction, and then not go back?

    - Need to support a fixed-frames case. This is simple, but I need to do it

    - Need to write a test

    - Broadcasting works in a funny way here. I'm not using broadcast_define()
      because all the functions I'm using support broadcasting internally in C. This
      is great, but it means that the caller must line up the dimensions in a
      non-standard way

    - Documentation!




    #   pref = ( transform( frames[0], p0_frames[0] ) +
    #            transform( frames[1], p0_frames[1] ) +
    #            ... ) / Nframes
    #   pcam = transform(extrinsics, pref)
    #   q    = project( pcam, intrinsics_data )

    # q depends on the intrinsics and on the extrinsics and on the frames

    #  dq/di comes from project()
    #  dq/de = dq/dpcam dpcam/de
    #  dq/df = dq/dpcam dpcam/dpref (dpref/dframes0 dpref/dframes1 ...) / Nframes

    # Linearization: dq ~ dq/dief dief ->
    # -> Var(dq) = dq/dief Var(dief) transpose(dq/dief)

    # What is Var(die)? Derivation in docstring for compute_projection_stdev()

    '''

    Nintrinsics = intrinsics_data.shape[-1]
    Nframes     = len(frames)

    def get_var_ief(rotation_only):
        if cachelist_invJtJ_JobstJobs is not None and cachelist_invJtJ_JobstJobs[0] is not None:
            invJtJ,JobstJobs = cachelist_invJtJ_JobstJobs
        else:
            J      = solver_context.J().toarray()
            solver_context.pack(J)
            invJtJ = np.linalg.inv(nps.matmult(nps.transpose(J), J))

            if solver_context.num_measurements_dict()['points'] > 0:
                raise Exception("This has been thought about with board observations only. I use the board frames to define the global coord system. Think about what points mean here")
            Nboard_measurements = solver_context.num_measurements_dict()['boards']

            imeas0 = 0
            imeas1 = Nboard_measurements
            Jobservations = J[imeas0:imeas1, :]
            JobstJobs = nps.matmult(nps.transpose(Jobservations),Jobservations)

            if cachelist_invJtJ_JobstJobs is not None:
                cachelist_invJtJ_JobstJobs[0] = invJtJ
                cachelist_invJtJ_JobstJobs[1] = JobstJobs

        if extrinsics is None:

            invJtJ_i  = invJtJ[solver_context.state_index_intrinsics(icam_intrinsics ):
                               solver_context.state_index_intrinsics(icam_intrinsics) + Nintrinsics,
                               :]

            invJtJ_f  = invJtJ[solver_context.state_index_frame_rt(0):
                               solver_context.state_index_frame_rt(0) + 6*Nframes,
                               :]
            if rotation_only:
                # pull out just the rotation terms
                invJtJ_f = invJtJ_f.reshape(Nframes,2,3,len(invJtJ))[:,0,:,:].reshape(Nframes*3,len(invJtJ))

            invJtJ_if = nps.glue(invJtJ_i, invJtJ_f, axis=-2)
            Var_ief = \
                pixel_uncertainty_stdev*pixel_uncertainty_stdev * \
                nps.matmult( invJtJ_if,
                             JobstJobs,
                             nps.transpose(invJtJ_if))
        else:
            invJtJ_i  = invJtJ[solver_context.state_index_intrinsics(icam_intrinsics ):
                               solver_context.state_index_intrinsics(icam_intrinsics) + Nintrinsics,
                               :]
            invJtJ_e  = invJtJ[solver_context.state_index_camera_rt (icam_extrinsics):
                               solver_context.state_index_camera_rt (icam_extrinsics) + 6,
                               :]
            invJtJ_f  = invJtJ[solver_context.state_index_frame_rt(0):
                               solver_context.state_index_frame_rt(0) + 6*Nframes,
                               :]
            if rotation_only:
                # pull out just the rotation terms
                invJtJ_e = invJtJ_e[:3,:]
                invJtJ_f = invJtJ_f.reshape(Nframes,2,3,len(invJtJ))[:,0,:,:].reshape(Nframes*3,len(invJtJ))
            invJtJ_ief = nps.glue(invJtJ_i, invJtJ_e, invJtJ_f, axis=-2)

            Var_ief = \
                pixel_uncertainty_stdev*pixel_uncertainty_stdev * \
                nps.matmult( invJtJ_ief,
                             JobstJobs,
                             nps.transpose(invJtJ_ief))
        return Var_ief


    # I have pixel coordinates q. I unproject and transform to the frames'
    # coordinate system. Then I transform and project back to the same q, but
    # keeping track of the gradients. Which I then use to compute the
    # sensitivities
    v_cam = mrcal.unproject(q, lensmodel, intrinsics_data,
                            normalize = True)


    # Two distinct paths here that are very similar, but different-enough to not
    # share any code. If distance is None, I'm looking at infinity, so I ignore
    # all translations
    if distance is not None:
        p_cam = v_cam*distance

        if extrinsics is not None:
            p_ref = \
                mrcal.transform_point_rt( mrcal.invert_rt(extrinsics),
                                          p_cam )
        else:
            p_ref = p_cam

        # The point in the coord system of all the frames. I index the frames on
        # axis -2
        p_frames = mrcal.transform_point_rt( mrcal.invert_rt(frames),
                                             nps.dummy(p_ref,-2) )

        # I now have the observed point represented in the coordinate system of the
        # frames. This is indendent of any intrinsics-implied rotation, or anything
        # of the sort. I project this point back to pixels, through noisy estimates
        # of the frames, extrinsics and intrinsics.
        #
        # I transform each frame-represented point back to the reference coordinate
        # system, and I average out each estimate to get the one p_ref I will use. I
        # already have p_ref, so I don't actually need to compute the value; I just
        # need the gradients

        _, \
        dprefallframes_dframesr, \
        dprefallframes_dframest, \
        _ = mrcal.transform_point_rt( frames, p_frames,
                                      get_gradients = True)

        # shape (..., Nframes,3,6)
        dprefallframes_dframes = nps.glue(dprefallframes_dframesr,
                                          dprefallframes_dframest,
                                          axis=-1)
        # shape (..., 3,6*Nframes)
        # /Nframes because I compute the mean over all the frames
        dpref_dframes = nps.clump( nps.mv(dprefallframes_dframes, -3, -2),
                                   n = -2 ) / Nframes

        _, dq_dpcam, dq_dintrinsics = \
            mrcal.project( p_cam, lensmodel, intrinsics_data,
                           get_gradients = True)

        Var_ief = get_var_ief(rotation_only = False)

        if extrinsics is not None:
            _, dpcam_dr, dpcam_dt, dpcam_dpref = \
                mrcal.transform_point_rt(extrinsics, p_ref,
                                         get_gradients = True)
            dq_dframes = nps.matmult(dq_dpcam, dpcam_dpref, dpref_dframes)

            dq_dr = nps.matmult(dq_dpcam, dpcam_dr)
            dq_dt = nps.matmult(dq_dpcam, dpcam_dt)
            dq_dief = nps.glue(dq_dintrinsics,
                               dq_dr,
                               dq_dt,
                               dq_dframes,
                               axis=-1)
            return \
                nps.matmult(dq_dief,
                            Var_ief,
                            nps.transpose(dq_dief))
        else:
            dq_dframes = nps.matmult(dq_dpcam, dpref_dframes)

            dq_dif = nps.glue(dq_dintrinsics,
                              dq_dframes,
                              axis=-1)

            return \
                nps.matmult(dq_dif,
                            Var_ief,
                            nps.transpose(dq_dif))

    else:

        # distance is None. I'm looking at infinity. Ignore all translations
        p_cam = v_cam

        if extrinsics is not None:
            p_ref = \
                mrcal.rotate_point_r( -extrinsics[..., :3], p_cam )
        else:
            p_ref = p_cam

        # The point in the coord system of all the frames. I index the frames on
        # axis -2
        p_frames = mrcal.rotate_point_r( -frames[...,:3],
                                         nps.dummy(p_ref,-2) )

        # I now have the observed point represented in the coordinate system of the
        # frames. This is indendent of any intrinsics-implied rotation, or anything
        # of the sort. I project this point back to pixels, through noisy estimates
        # of the frames, extrinsics and intrinsics.
        #
        # I transform each frame-represented point back to the reference coordinate
        # system, and I average out each estimate to get the one p_ref I will use. I
        # already have p_ref, so I don't actually need to compute the value; I just
        # need the gradients

        _, \
        dprefallframes_dframesr, \
        _ = mrcal.rotate_point_r( frames[...,:3], p_frames,
                                  get_gradients = True)

        # shape (..., 3,3*Nframes)
        # /Nframes because I compute the mean over all the frames
        dpref_dframes = nps.clump( nps.mv(dprefallframes_dframesr, -3, -2),
                                   n = -2 ) / Nframes

        _, dq_dpcam, dq_dintrinsics = \
            mrcal.project( p_cam, lensmodel, intrinsics_data,
                           get_gradients = True)

        Var_ief = get_var_ief(rotation_only = True)

        if extrinsics is not None:
            _, dpcam_dr, dpcam_dpref = \
                mrcal.rotate_point_r(extrinsics[...,:3], p_ref,
                                     get_gradients = True)
            dq_dframes = nps.matmult(dq_dpcam, dpcam_dpref, dpref_dframes)

            dq_dr = nps.matmult(dq_dpcam, dpcam_dr)
            dq_dief = nps.glue(dq_dintrinsics,
                               dq_dr,
                               dq_dframes,
                               axis=-1)
            return \
                nps.matmult(dq_dief,
                            Var_ief,
                            nps.transpose(dq_dief))
        else:
            dq_dframes = nps.matmult(dq_dpcam, dpref_dframes)

            dq_dif = nps.glue(dq_dintrinsics,
                              dq_dframes,
                              axis=-1)

            return \
                nps.matmult(dq_dif,
                            Var_ief,
                            nps.transpose(dq_dif))


def show_intrinsics_uncertainty(model,
                                gridn_width  = 60,
                                gridn_height = None,

                                # fit a "reasonable" area in the center by
                                # default
                                focus_center = None,
                                focus_radius = -1.,

                                extratitle       = None,
                                hardcopy         = None,
                                cbmax            = 3,
                                kwargs           = None):
    r'''Visualizes the uncertainty in the intrinsics of a camera

    This routine uses the covariance of observed inputs. See
    compute_projection_stdev() for a description of both routines and of the
    arguments

    '''

    if kwargs is None: kwargs = {}

    import gnuplotlib as gp
    W,H=model.imagersize()
    if gridn_height is None:
        gridn_height = int(round(H/W*gridn_width))

    lensmodel, intrinsics_data = model.intrinsics()
    imagersize                 = model.imagersize()
    err = compute_projection_stdev(model,
                                   gridn_width  = gridn_width,
                                   gridn_height = gridn_height,
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
        title = f"Projection uncertainty (in pixels) based on calibration input noise; {where}"
        if extratitle is not None:
            title += ": " + extratitle
        kwargs['title'] = title

    if 'hardcopy' not in kwargs and hardcopy is not None:
        kwargs['hardcopy'] = hardcopy

    if 'set' not in kwargs:
        kwargs['set'] = []
    elif type(kwargs['set']) is not list:
        kwargs['set'] = [kwargs['set']]
    kwargs['set'].extend(['view equal xy',
                          'view map',
                          'contour surface',
                          'key box opaque',
                          'cntrparam levels incremental 3,-0.5,0'])

    plot_data_args = [(err,
                       dict( tuplesize=3,
                             legend = "", # needed to force contour labels
                             using = imagergrid_using(model.imagersize(), gridn_width, gridn_height),

                             # Currently "with image" can't produce contours. I work around this, by
                             # plotting the data a second time. Yuck.
                             # https://sourceforge.net/p/gnuplot/mailman/message/36371128/
                             _with=np.array(('image','lines nosurface'),),

))]

    valid_intrinsics_region = model.valid_intrinsics_region()
    if valid_intrinsics_region is not None:
        plot_data_args.append( (valid_intrinsics_region[:,0],
                                valid_intrinsics_region[:,1],
                                np.zeros(valid_intrinsics_region.shape[-2]),
                                dict(_with  = 'lines lw 3 nocontour',
                                     legend = "Valid-intrinsics region")) )
    plot = \
        gp.gnuplotlib(_3d=1,
                      unset='grid',

                      _xrange=[0,W],
                      _yrange=[H,0],
                      cbrange=[0,cbmax],
                      ascii=1,
                      **kwargs)

    plot.plot(*plot_data_args)
    return plot

def report_residual_statistics( obs, err,
                                imagersize,
                                gridn_width  = 20,
                                gridn_height = None):
    '''Reports statistics about the fit resudial across the imager

    If everything fits well, the residual distributions in each area of the
    imager should be identical. If the model doesn't fit well, the statistics
    will not be consistent. This function returns a tuple
    (mean,stdev,count,imagergrid_using). imagergrid_using is a "using" keyword
    for plotting this data in a heatmap

    '''

    W,H=imagersize

    if gridn_height is None:
        gridn_height = int(round(H/W*gridn_width))

    # shape: (Nheight,Nwidth,2). Contains (x,y) rows
    c = sample_imager(gridn_width, gridn_height, W, H)

    wcell = float(W-1) / (gridn_width -1)
    hcell = float(H-1) / (gridn_height-1)
    rcell = np.array((wcell,hcell), dtype=float) / 2.

    @nps.broadcast_define( (('N',2), ('N',), (2,)),
                           (3,) )
    def residual_stats(xy, z, center):
        r'''Generates localized residual statistics

        Takes in an array of residuals of shape (N,3) (contains (x,y,error)
        slices)

        '''

        # boolean (x,y separately) map of observations that are within a cell
        idx = np.abs(xy-center) < rcell

        # join x,y: both the x,y must be within a cell for the observation to be
        # within a cell
        idx = idx[:,0] * idx[:,1]

        z   = z[idx, ...]
        if z.shape[0] <= 5:
            # we have too little data in this cell
            return np.array((0.,0.,z.shape[0]), dtype=float)

        mean   = np.mean(z)
        stdev  = np.std(z)
        return np.array((mean,stdev,z.shape[0]))


    # obs,err each have shape (N,2): each slice is xy. I have 2N
    # measurements, so I flatten the errors, and double-up the observations
    errflat = err.ravel()
    obsflat = nps.clump(nps.mv(nps.cat(obs,obs), -3, -2), n=2)

    # Each has shape (2,Nheight,Nwidth)
    mean,stdev,count = nps.mv( residual_stats(obsflat, errflat, c),
                               -1, 0)
    return mean,stdev,count,imagergrid_using(imagersize, gridn_width, gridn_height)


def show_distortion(model,
                    mode,
                    scale        = 1.,
                    cbmax        = 25.0,
                    gridn_width  = 60,
                    gridn_height = None,

                    extratitle       = None,
                    hardcopy         = None,
                    kwargs           = None):
    r'''Visualizes the distortion of a lens

    "Distortion" means "deviation from some norm". This function takes the
    "norm" to be a pinhole lens. So wide lenses will have a lot of reported
    distortion.

    For lens models based on corrections to a pinhole projection (those that
    have an intrinsic core), the baseline pinhole model is used as the
    reference. For models not based on an intrinsic we either estimate the
    pinhole model at the center, or we simply do not support this function.

    This function has 3 modes of operation, specified as a string in the 'mode'
    argument:

      'heatmap': the imager is gridded, as specified by the
      gridn_width,gridn_height arguments. For each point in the grid, we
      evaluate the difference in projection between the given model, and a
      pinhole model with the same core intrinsics (focal lengths, center pixel
      coords). This difference is color-coded and a heat map is displayed

      'vectorfield': this is the same as 'heatmap', except we display a vector
      for each point, intead of a color-coded cell. If legibility requires the
      vectors being larger or smaller, pass an appropriate value in the 'scale'
      argument

      'radial': Looks at radial distortion only. Plots a curve showing the
      magnitude of the radial distortion as a function of the distance to the
      center

    This function creates a plot and returns the corresponding gnuplotlib object

    '''

    import gnuplotlib as gp

    lensmodel, intrinsics_data = model.intrinsics()
    imagersize                  = model.imagersize()

    if kwargs is None: kwargs = {}
    if 'title' not in kwargs:

        title = "Distortion of {}".format(lensmodel)
        if extratitle is not None:
            title += ": " + extratitle
        kwargs['title'] = title

    if 'hardcopy' not in kwargs and hardcopy is not None:
        kwargs['hardcopy'] = hardcopy

    if 'set' not in kwargs:
        kwargs['set'] = []
    elif type(kwargs['set']) is not list:
        kwargs['set'] = [kwargs['set']]





    W,H = imagersize
    if gridn_height is None:
        gridn_height = int(round(H/W*gridn_width))

    if not mrcal.getLensModelMeta(lensmodel)['has_core']:
        raise Exception("This currently works only with models that have an fxfycxcy core. It might not be required. Take a look at the following code if you want to add support")
    fxy = intrinsics_data[ :2]
    cxy = intrinsics_data[2:4]

    if mode == 'radial':

        # plot the radial distortion. For now I only deal with opencv here
        m = re.search("OPENCV([0-9]+)", lensmodel)
        if not m:
            raise Exception("Radial distortion visualization implemented only for OpenCV distortions; for now")
        N = int(m.group(1))

        # OpenCV does this:
        #
        # This is the opencv distortion code in cvProjectPoints2 in
        # calibration.cpp Here x,y are x/z and y/z. OpenCV applies distortion to
        # x/z, y/z and THEN does the ...*f + c thing. The distortion factor is
        # based on r, which is ~ x/z ~ tan(th) where th is the deviation off
        # center
        #
        #         z = z ? 1./z : 1;
        #         x *= z; y *= z;
        #         r2 = x*x + y*y;
        #         r4 = r2*r2;
        #         r6 = r4*r2;
        #         a1 = 2*x*y;
        #         a2 = r2 + 2*x*x;
        #         a3 = r2 + 2*y*y;
        #         cdist = 1 + k[0]*r2 + k[1]*r4 + k[4]*r6;
        #         icdist2 = 1./(1 + k[5]*r2 + k[6]*r4 + k[7]*r6);
        #         xd = x*cdist*icdist2 + k[2]*a1 + k[3]*a2 + k[8]*r2+k[9]*r4;
        #         yd = y*cdist*icdist2 + k[2]*a3 + k[3]*a1 + k[10]*r2+k[11]*r4;
        #         ...
        #         m[i].x = xd*fx + cx;
        #         m[i].y = yd*fy + cy;
        distortions = intrinsics_data[4:]
        k2 = distortions[0]
        k4 = distortions[1]
        k6 = 0
        if N >= 5:
            k6 = distortions[4]
        numerator = '1. + r*r * ({} + r*r * ({} + r*r * {}))'.format(k2,k4,k6)
        numerator = numerator.replace('r', 'tan(x*pi/180.)')

        if N >= 8:
            denominator = '1. + r*r * ({} + r*r * ({} + r*r * {}))'.format(*distortions[5:8])
            denominator = denominator.replace('r', 'tan(x*pi/180.)')
            scale = '({})/({})'.format(numerator,denominator)
        else:
            scale = numerator


        x0,x1 = 0, W-1
        y0,y1 = 0, H-1
        q_corners = np.array(((x0,y0),
                              (x0,y1),
                              (x1,y0),
                              (x1,y1)), dtype=float)
        q_centersx  = np.array(((cxy[0],y0),
                                (cxy[0],y1)), dtype=float)
        q_centersy  = np.array(((x0,cxy[1]),
                                (x1,cxy[1])), dtype=float)

        v_corners  = mrcal.unproject( q_corners,  lensmodel, intrinsics_data)
        v_centersx = mrcal.unproject( q_centersx, lensmodel, intrinsics_data)
        v_centersy = mrcal.unproject( q_centersy, lensmodel, intrinsics_data)

        th_corners  = 180./np.pi * np.arctan2(nps.mag(v_corners [..., :2]), v_corners [..., 2])
        th_centersx = 180./np.pi * np.arctan2(nps.mag(v_centersx[..., :2]), v_centersx[..., 2])
        th_centersy = 180./np.pi * np.arctan2(nps.mag(v_centersy[..., :2]), v_centersy[..., 2])

        # Now the equations. The 'x' value here is "pinhole pixels off center",
        # which is f*tan(th). I plot this model's radial relationship, and that
        # from other common fisheye projections (formulas mostly from
        # https://en.wikipedia.org/wiki/Fisheye_lens)
        equations = [f'180./pi*atan(tan(x*pi/180.) * ({scale})) with lines lw 2 title "THIS model"',
                     'x title "pinhole"',
                     f'180./pi*atan(2. * tan( x*pi/180. / 2.)) title "stereographic"',
                     f'180./pi*atan(x*pi/180.) title "equidistant"',
                     f'180./pi*atan(2. * sin( x*pi/180. / 2.)) title "equisolid angle"',
                     f'180./pi*atan( sin( x*pi/180. )) title "orthogonal"']
        sets = \
            ['arrow from {th}, graph 0 to {th}, graph 1 nohead lc "red"'  . \
             format(th=th) for th in th_centersy] + \
            ['arrow from {th}, graph 0 to {th}, graph 1 nohead lc "green"'. \
             format(th=th) for th in th_centersx] + \
            ['arrow from {th}, graph 0 to {th}, graph 1 nohead lc "blue"' . \
             format(th=th) for th in th_corners ]
        if 'set' in kwargs:
            if type(kwargs['set']) is list: sets.extend(kwargs['set'])
            else:                           sets.append(kwargs['set'])
            del kwargs['set']
        if '_set' in kwargs:
            if type(kwargs['set']) is list: sets.extend(kwargs['_set'])
            else:                           sets.append(kwargs['_set'])
            del kwargs['_set']

        if N >= 8:
            equations.extend( [numerator   + ' axis x1y2 title "numerator (y2)"',
                               denominator + ' axis x1y2 title "denominator (y2)"',
                               '0 axis x1y2 with lines lw 2' ] )
            sets.append('y2tics')
            kwargs['y2label'] = 'Rational correction numerator, denominator'
        kwargs['title'] += ': radial distortion. Red: x edges. Green: y edges. Blue: corners'
        plot = gp.gnuplotlib(equation = equations,
                             _set=sets,
                             _xrange = [0,np.max(th_corners) * 1.01],
                             xlabel = 'Angle off the projection center (deg)',
                             ylabel = 'Distorted angle off the projection center',
                             **kwargs)
        plot.plot()
        return plot


    if not ( mode == 'heatmap' or mode == 'vectorfield' ):
        raise Exception("Unknown mode '{}'. I only know about 'heatmap','vectorfield','radial'".format(mode))


    # shape: (Nheight,Nwidth,2). Contains (x,y) rows
    grid  = np.ascontiguousarray(nps.mv(nps.cat(*np.meshgrid(np.linspace(0,W-1,gridn_width),
                                                             np.linspace(0,H-1,gridn_height))),
                                        0,-1),
                                 dtype = float)

    dgrid =  mrcal.project( nps.glue( (grid-cxy)/fxy,
                                    np.ones(grid.shape[:-1] + (1,), dtype=float),
                                    axis = -1 ),
                          lensmodel, intrinsics_data )

    if mode == 'heatmap':

        if 'set' not in kwargs:
            kwargs['set'] = []
        elif type(kwargs['set']) is not list:
            kwargs['set'] = [kwargs['set']]
        kwargs['set'].extend([ 'view equal xy',
                               'view map',
                               'contour surface',
                               'cntrparam levels incremental {},-1,0'.format(cbmax)])

        delta = dgrid-grid
        delta *= scale

        # shape: gridn_height,gridn_width. Because numpy (and thus gnuplotlib) want it that
        # way
        distortion = nps.mag(delta)

        # Currently "with image" can't produce contours. I work around this, by
        # plotting the data a second time.
        # Yuck.
        # https://sourceforge.net/p/gnuplot/mailman/message/36371128/
        plot = gp.gnuplotlib(_3d=1,
                             unset='grid',
                             _xrange=[0,W],
                             _yrange=[H,0],
                             cbrange=[0,cbmax],
                             ascii=1,
                             **kwargs)
        plot.plot(distortion,
                  tuplesize=3,
                  _with=np.array(('image','lines nosurface'),),
                  legend = "", # needed to force contour labels
                  using = imagergrid_using(imagersize, gridn_width, gridn_height))
        return plot

    else:
        # vectorfield

        # shape: gridn_height*gridn_width,2
        grid  = nps.clump(grid,  n=2)
        dgrid = nps.clump(dgrid, n=2)

        delta = dgrid-grid
        delta *= scale

        kwargs['_xrange']=(-50,W+50)
        kwargs['_yrange']=(H+50, -50)
        kwargs['_set'   ]=['object 1 rectangle from 0,0 to {},{} fillstyle empty'.format(W,H)]
        kwargs['square' ]=True

        if 'set' in kwargs:
            if type(kwargs['set']) is list: kwargs['_set'].extend(kwargs['set'])
            else:                           kwargs['_set'].append(kwargs['set'])
            del kwargs['set']

        plot = gp.gnuplotlib( **kwargs )
        plot.plot( (grid[:,0], grid[:,1], delta[:,0], delta[:,1],
                    {'with': 'vectors size screen 0.01,20 fixed filled',
                     'tuplesize': 4,
                    }),
                   (grid[:,0], grid[:,1],
                    {'with': 'points',
                     'tuplesize': 2,
                    }))
        return plot


def splined_stereographic_valid_region(lensmodel):
    '''Return the bounds of the valid region for these models

    Splined stereographic models are defined by a splined surface. This
    surface has some finite domain, beyond which it carries no information.
    This function reports a piecewise linear contour reporting this region.

    We return the contour in whatever domain is being used to index the surface.
    At this time this is the stereographic projection uxy.

    '''

    if not re.match('LENSMODEL_SPLINED_STEREOGRAPHIC', lensmodel):
        raise Exception(f"This only makes sense with splined models. Input uses {lensmodel}")

    ux,uy = mrcal.getKnotsForSplinedModels(lensmodel)
    # shape (Ny,Nx,2)
    u = np.ascontiguousarray(nps.mv(nps.cat(*np.meshgrid(ux,uy)), 0, -1))

    meta = mrcal.getLensModelMeta(lensmodel)
    if meta['order'] == 2:
        # spline order is 3. The valid region is 1/2 segments inwards from the
        # outer contour
        return \
            nps.glue( (u[0,1:-2] + u[1,1:-2]) / 2.,
                      (u[0,-2] + u[1,-2] + u[0,-1] + u[1,-1]) / 4.,

                      (u[1:-2, -2] + u[1:-2, -1]) / 2.,
                      (u[-2,-2] + u[-1,-2] + u[-2,-1] + u[-1,-1]) / 4.,

                      (u[-2, -2:1:-1] + u[-1, -2:1:-1]) / 2.,
                      (u[-2, 1] + u[-1, 1] + u[-2, 0] + u[-1, 0]) / 4.,

                      (u[-2:0:-1, 0] +u[-2:0:-1, 1]) / 2.,
                      (u[0, 0] +u[0, 1] + u[1, 0] +u[1, 1]) / 4.,

                      (u[0,1] + u[1,1]) / 2.,
                      axis = -2 )

    elif meta['order'] == 3:
        # spline order is 3. The valid region is the outer contour, leaving one
        # knot out
        return \
            nps.glue( u[1,1:-2], u[1:-2, -2], u[-2, -2:1:-1], u[-2:0:-1, 1],
                      axis=-2 )
    else:
        raise Exception("I only support cubic (order==3) and quadratic (order==2) models")


def polygon_difference(positive, negative):
    r'''Returns the difference of two closed polygons

    The polygons are represented as (.,2) arrays.

    The result is represented as a list of (.,2) arrays, to be interpreted as a
    union. Each of the constituent resulting arrays is guaranteed to not have
    holes. If any holes are found when computing the difference, we cut apart
    the resulting shape until no holes remain.

    '''

    from shapely.geometry import Polygon,MultiPolygon,GeometryCollection,LineString
    import shapely.ops


    diff = Polygon(positive).difference(Polygon(negative))
    if isinstance(diff, (MultiPolygon,GeometryCollection)):
        diff = list(diff)
    elif isinstance(diff, Polygon):
        diff = [diff]
    else:
        raise Exception(f"I only know how to deal with MultiPolygon or Polygon, but instead got type '{type(diff)}")

    def split_polygon_to_remove_holes(p):
        if not isinstance(p, Polygon):
            raise Exception(f"Expected a 'Polygon' type, but got {type(p)}")

        if not (p.interiors and len(p.interiors)):
            # No hole. Return the coords, if they exist
            try:
                return [np.array(p.exterior.coords)]
            except:
                return []

        # There's a hole! We need to split this polygon. I cut the polygon by a
        # line between the centroid and some vertex. Which one doesn't matter; I
        # keep trying until some cut works
        hole = p.interiors[0]
        for i in range(0,len(hole.coords)):

            l0 = np.array((hole.centroid))
            l1 = np.array((hole.coords[i]))
            l0,l1 = (l1 + 100*(l0-l1)),(l0 + 100*(l1-l0))
            line = LineString( (l0,l1) )

            s = shapely.ops.split(p, line)
            if len(s) > 1:
                # success. split into multiple pieces. I process each one
                # recursively, and I'm done. I return a flattened list
                return [subpiece for piece in s for subpiece in split_polygon_to_remove_holes(piece)]
            # Split didn't work. Try the next vertex

        print("WARNING: Couldn't split the region. Ignoring",
              file = sys.stderr)
        return []

    return \
        [subpiece for p in diff for subpiece in split_polygon_to_remove_holes(p)]


def show_splined_model_surface(model, ixy,
                               imager_domain    = True,
                               extratitle       = None,
                               hardcopy         = None,
                               **kwargs):

    r'''Visualizes the surface represented in a splined model

    Splined models are built with a splined surface that we index to compute the
    projection. The meaning of what indexes the surface and the values of the
    surface varies by model, but in all cases, visualizing the surface is useful.

    The surface is defined by control points. The value of the control points is
    set in the intrinsics vector, but their locations (called "knots") are
    fixed, as defined by the model configuration. The configuration selects the
    control point density AND the expected field of view of the lens. This field
    of view should roughly match the actual lens+camera we're using, and this
    fit can be visualized with this tool.

    If the field of view is too small, some parts of the imager will lie outside
    of the region that the splined surface covers. This tool throws a warning in
    that case, and displays the offending regions.

    This function can produce a plot in the imager domain or in the spline index
    domain. Both are useful, and this is controlled by the imager_domain
    argument; the default is True.

    This function creates a plot and returns the corresponding gnuplotlib object

    '''

    lensmodel,intrinsics_data = model.intrinsics()
    W,H                       = model.imagersize()

    if not re.match('LENSMODEL_SPLINED_STEREOGRAPHIC', lensmodel):
        raise Exception(f"This only makes sense with splined models. Input uses {lensmodel}")


    import gnuplotlib as gp

    if 'title' not in kwargs:

        title = f"Surface for {lensmodel}. Looking at deltau{'y' if ixy else 'x'}"
        if extratitle is not None:
            title += ": " + extratitle
        kwargs['title'] = title

    if 'hardcopy' not in kwargs and hardcopy is not None:
        kwargs['hardcopy'] = hardcopy

    if 'set' not in kwargs:
        kwargs['set'] = []
    elif type(kwargs['set']) is not list:
        kwargs['set'] = [kwargs['set']]


    ux_knots,uy_knots = mrcal.getKnotsForSplinedModels(lensmodel)
    meta = mrcal.getLensModelMeta(lensmodel)
    Nx = meta['Nx']
    Ny = meta['Ny']

    if imager_domain:
        # Shape (Ny,Nx,2); contains (x,y) rows
        q = \
            nps.mv( nps.cat(*np.meshgrid( np.linspace(0, W-1, 60),
                                          np.linspace(0, H-1, 40) )),
                    0,-1)
        v = mrcal.unproject(np.ascontiguousarray(q), lensmodel, intrinsics_data)
        u = mrcal.project_stereographic(v)
    else:

        # In the splined_stereographic models, the spline is indexed by u. So u is
        # linear with the knots. I can thus get u at the edges, and linearly
        # interpolate between

        # Shape (Ny,Nx,2); contains (x,y) rows
        u = \
            nps.mv( nps.cat(*np.meshgrid( np.linspace(ux_knots[0], ux_knots[-1],Nx*5),
                                          np.linspace(uy_knots[0], uy_knots[-1],Ny*5) )),
                    0,-1)

        # My projection is q = (u + deltau) * fxy + cxy. deltau is queried from the
        # spline surface
        v = mrcal.unproject_stereographic(np.ascontiguousarray(u))
        q = mrcal.project(v, lensmodel, intrinsics_data)

    fxy = intrinsics_data[0:2]
    cxy = intrinsics_data[2:4]
    deltau = (q - cxy) / fxy - u

    # the imager boundary
    imager_boundary_sparse = \
        np.array(((0,   0),
                  (W-1, 0),
                  (W-1, H-1),
                  (0,   H-1),
                  (0,   0)), dtype=float)

    if imager_domain:
        imager_boundary = imager_boundary_sparse
    else:
        imager_boundary = \
            mrcal.project_stereographic(
                mrcal.unproject(
                    _densify_polyline(imager_boundary_sparse,
                                      spacing = 50),
                    lensmodel, intrinsics_data ))

    plotoptions = dict(kwargs,
                       zlabel   = f"Deltau{'y' if ixy else 'x'} (unitless)")
    surface_curveoptions = dict()
    if imager_domain:
        plotoptions['xlabel'] = 'X pixel coord'
        plotoptions['ylabel'] = 'Y pixel coord'
        surface_curveoptions['using'] = \
            f'($1/({deltau.shape[1]-1})*({W-1})):' + \
            f'($2/({deltau.shape[0]-1})*({H-1})):' + \
            '3'
    else:
        plotoptions['xlabel'] = 'Stereographic ux'
        plotoptions['ylabel'] = 'Stereographic uy'
        surface_curveoptions['using'] = \
            f'({ux_knots[0]}+$1/({deltau.shape[1]-1})*({ux_knots[-1]-ux_knots[0]})):' + \
            f'({uy_knots[0]}+$2/({deltau.shape[0]-1})*({uy_knots[-1]-uy_knots[0]})):' + \
            '3'

    plotoptions['square']   = True
    plotoptions['yinv']     = True
    plotoptions['ascii']    = True
    surface_curveoptions['_with']     = 'image'
    surface_curveoptions['tuplesize'] = 3

    plot = gp.gnuplotlib(**plotoptions)

    data = [ ( deltau[..., ixy],
               surface_curveoptions ) ]

    valid_region_contour_u = splined_stereographic_valid_region(lensmodel)
    knots_u = nps.clump(nps.mv(nps.cat(*np.meshgrid(ux_knots,uy_knots)),
                               0, -1),
                        n = 2)
    if imager_domain:
        valid_region_contour = \
            mrcal.project(
                mrcal.unproject_stereographic( valid_region_contour_u),
                lensmodel, intrinsics_data)
        knots = \
            mrcal.project(
                mrcal.unproject_stereographic( np.ascontiguousarray(knots_u)),
                lensmodel, intrinsics_data)
    else:
        valid_region_contour = valid_region_contour_u
        knots = knots_u

    data.extend( [ ( imager_boundary,
                     dict(_with     = 'lines lw 2',
                          tuplesize = -2,
                          legend    = 'imager boundary')),
                   ( valid_region_contour,
                     dict(_with     = 'lines lw 1',
                          tuplesize = -2,
                          legend    = 'Valid projection region')),
                   ( knots,
                     dict(_with     = 'points pt 2 ps 2',
                          tuplesize = -2,
                          legend    = 'knots'))] )


    # Anything outside the valid region contour but inside the imager is an
    # invalid area: the field-of-view of the camera needs to be increased. I
    # plot this area
    imager_boundary_nonan = \
        imager_boundary[ np.isfinite(imager_boundary[:,0]) *
                         np.isfinite(imager_boundary[:,1]),:]

    try:
        invalid_regions = polygon_difference(imager_boundary_nonan,
                                             valid_region_contour)
    except Exception as e:
        # sometimes the valid_region_contour self-intersects, and this makes us
        # barf
        print(f"WARNING: Couldn't compute invalid projection region. Exception: {e}")
        invalid_regions = []

    if len(invalid_regions) > 0:
        print("WARNING: some parts of the imager cannot be projected from a region covered by the spline surface! You should increase the field-of-view of the model")

        data.extend( [ ( r,
                         dict( tuplesize = -2,
                               _with     = 'filledcurves closed fillcolor "red"',
                               legend    = 'Invalid regions'))
                       for r in invalid_regions] )


    plot.plot( *data )
    return plot


def is_within_valid_intrinsics_region(q, model):
    r'''Which of the pixel coordinates fall within the valid-intrinsics region?

SYNOPSIS

    mask = mrcal.is_within_valid_intrinsics_region(q, model)
    q_trustworthy = q[mask]

mrcal camera models may have an estimate of the region of the imager where the
intrinsics are trustworthy (originally computed with a low-enough error and
uncertainty). When using a model, we may want to process points that fall
outside of this region differently from points that fall within this region.
This function returns a mask that indicates whether each point is within the
region or not.

If no valid-intrinsics region is defined in the model, returns None.

ARGUMENTS

- q: an array of shape (..., 2) of pixel coordinates

- model: the model we're interrogating

    '''

    r = model.valid_intrinsics_region()
    if r is None:
        return None

    from shapely.geometry import Polygon,Point

    r = Polygon(r)

    mask = np.zeros(q.shape[:-1], dtype=bool)
    mask_flat = mask.ravel()
    q_flat = q.reshape(q.size//2, 2)
    for i in range(q.size // 2):
        if r.contains(Point(q_flat[i])):
            mask_flat[i] = True
    return mask


def compute_Rcompensating(q0, v0, v1,
                          focus_center, focus_radius,
                          imagersizes):

    r'''Computes a compensating rotation to fit two cameras' projections

    I sample the imager grid in all my cameras, and compute the rotation that
    maps the vectors to each other as closely as possible. Then I produce a
    difference map by projecting the matched-up vectors. This is very similar in
    spirit to what compute_Rcorrected_dq_dintrinsics() does, but that function
    has to work analytically, while this one explicitly computes the rotation by
    matching up known vectors.


    I compute the rotation using a Procrustes fit:

        R = align3d_procrustes( nps.clump(v0,n=2),
                                nps.clump(v1,n=2), vectors=True)

    This works, but it minimizes a norm2() metric, and is sensitive to outliers.
    If my lens model doesn't fit perfectly, I can fit well only in some
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

    # I THINK this function expects normalized vectors, so I normalize them here
    def normalize(x):
        return x / nps.dummy(nps.mag(x), axis=-1)
    v0 = normalize(v0)
    v1 = normalize(v1)

    # my state vector is a rodrigues rotation, seeded with the identity
    # rotation
    cache = {'r': None}

    def angle_err_sq(v0,v1,R):
        # cos(x) = inner(v0,v1) ~ 1 - x*x
        return 1 - nps.inner(nps.matmult(v0,R), v1)

    def residual_jacobian(r):
        R,dRdr = mrcal.R_from_r(r, get_gradients=True)
        # dRdr has shape (3,3,3). First 2 for R, last 1 for r

        x = angle_err_sq(V0fit, V1fit, R)

        # dx/dr = d(1-c)/dr = - V1ct dV0R/dr
        dV0R_dr = \
            nps.dummy(V0fit[..., (0,)], axis=-1) * dRdr[0,:,:] + \
            nps.dummy(V0fit[..., (1,)], axis=-1) * dRdr[1,:,:] + \
            nps.dummy(V0fit[..., (2,)], axis=-1) * dRdr[2,:,:]

        J = -nps.matmult(nps.dummy(V1fit, -2), dV0R_dr)[..., 0, :]
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
        return np.eye(3)


    V0cut   = nps.clump(v0,n=2)
    V1cut   = nps.clump(v1,n=2)
    icenter = np.array((v0.shape[:2])) // 2
    if focus_radius < 2*(W+H):
        # We try to match the geometry in a particular region

        q_off_center = q0 - focus_center
        i = nps.norm2(q_off_center) < focus_radius*focus_radius
        if np.count_nonzero(i)<3:
            warnings.warn("Focus region contained too few points; I need at least 3. Fitting EVERYWHERE across the imager")
        else:
            V0cut = v0[i, ...]
            V1cut = v1[i, ...]

            # get the nearest index on my grid to the requested center
            icenter_flat = np.argmin(nps.norm2(q_off_center))

            # This looks funny, but it's right. My grid is set up that you index
            # with the x-coord and then the y-coord. This is opposite from the
            # matrix convention that numpy uses: y then x.
            ix = icenter_flat // v0.shape[1]
            iy = icenter_flat - ix*v0.shape[1]
            icenter = np.array((ix,iy))

    # I compute a procrustes fit using ONLY data in the region of interest.
    # This is used to seed the nonlinear optimizer
    R_procrustes = align3d_procrustes( V0cut, V1cut, vectors=True)
    r_procrustes = mrcal.r_from_R(R_procrustes)
    r_procrustes = r_procrustes.ravel()

    esq = angle_err_sq(v0,v1,R_procrustes)

    # throw away everything that's k times as wrong as the center of
    # interest. I look at a connected component around the center. I pick a
    # large k here, and use a robust error function further down
    k = 10
    angle_err_sq_at_center = esq[icenter[0],icenter[1]]
    thresholdsq = angle_err_sq_at_center*k*k
    import scipy.ndimage
    regions,_ = scipy.ndimage.label(esq < thresholdsq)
    mask = regions==regions[icenter[0],icenter[1]]
    V0fit = v0[mask, ...]
    V1fit = v1[mask, ...]
    # V01fit are used by the optimization cost function

    # gradient check
    # r = r_procrustes
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
    # Seed from the procrustes solve
    res = scipy.optimize.least_squares(residual, r_procrustes, jac=jacobian,
                                       method='dogbox',

                                       loss='soft_l1',
                                       f_scale=angle_err_sq_at_center*3.0,
                                       # max_nfev=1,
                                       args=(cache,),
                                       verbose=0)
    r_fit = res.x
    R_fit = mrcal.R_from_r(r_fit)

    # # A simpler routine to JUST move pitch/yaw to align the optical axes
    # r = mrcal.r_from_R(R)
    # r[2] = 0
    # R = mrcal.R_from_r(r)
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
    # R = mrcal.R_from_r(r)

    return R_fit


def _densify_polyline(p, spacing):
    r'''Returns the input polyline, but resampled more densely
    The input and output polylines are a numpy array of shape (N,2). The output
    is resampled such that each input point is hit, but each linear segment is
    also sampled with at least the given spacing

    '''

    p1 = np.array(p[0,:], dtype=p.dtype)

    for i in range(1,len(p)):
        a = p[i-1,:]
        b = p[i,  :]
        d = b-a
        l = nps.mag(d)

        # A hacky method of rounding up
        N = int(l/spacing - 1e-6 + 1.)

        for j in range(N):
            p1 = nps.glue(p1,
                          float(j+1) / N * d + a,
                          axis=-2)
    return p1


def show_intrinsics_diff(models,
                         gridn_width  = 60,
                         gridn_height = None,

                         # fit a "reasonable" area in the center by
                         # default
                         focus_center     = None,
                         focus_radius     = -1.,

                         vectorfield      = False,
                         vectorscale      = 1.0,
                         extratitle       = None,
                         hardcopy         = None,
                         cbmax            = 4,
                         kwargs = None):
    r'''Visualize the difference in projection between N models


    If we're given exactly 2 models then I show the projection DIFFERENCE. I
    show this as either a vector field or a heat map. If N > 2 then a vector
    field isn't possible and we show a heat map of the STANDARD DEVIATION of the
    differences.

    This routine takes into account the potential variability of camera rotation by
    fitting this implied camera rotation to align the models as much as possible.
    This is required because a camera pitch/yaw motion looks a lot like a shift in
    the camera optical axis (cx,cy). So I could be comparing two sets of intrinsics
    that both represent the same lens faithfully, but imply different rotations: the
    rotation would be compensated for by a shift in cx,cy. If I compare the two sets
    of intrinsics by IGNORING the rotations, the cx,cy difference would produce a
    large diff despite both models being right.

    The implied rotation is fit using a subset of the imager data:

      if focus_radius < 0 (the default):
         I fit a compensating rotation using a "reasonable" area in the center of
         the imager. I use focus_radius = min(width,height)/6.

      if focus_radius > 0:
         I use observation vectors within focus_radius pixels of focus_center.
         To use ALL the data, pass in a very large focus_radius.

      if focus_radius == 0:
         I do NOT fit a compensating rotation. Rationale: with radius == 0, I have
         no fitting data, so I do not fit anything at all.

      if focus_center is omitted (the default):
         focus_center is at the center of the imager

    Generally the computation isn't very sensitive to choices of focus_radius
    and focus_center, so omitting these is recommended.

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

    lensmodels      = [model.intrinsics()[0] for model in models]
    intrinsics_data = [model.intrinsics()[1] for model in models]


    # shape (...,Nheight,Nwidth,...)
    v,q0 = sample_imager_unproject(gridn_width, gridn_height,
                                   W, H,
                                   lensmodels, intrinsics_data)

    if len(models) == 2:
        # Two models. Take the difference and call it good

        Rcompensating01 = \
            compute_Rcompensating(q0,
                                  v[0,...], v[1,...],
                                  focus_center, focus_radius,
                                  imagersizes)
        q1 = mrcal.project(nps.matmult(v[0,...],Rcompensating01),
                           lensmodels[1], intrinsics_data[1])

        diff    = q1 - q0
        difflen = nps.mag(diff)

    else:
        # Many models. Look at the stdev
        def get_reprojections(q0, v0, v1,
                              focus_center,
                              focus_radius,
                              lensmodel, intrinsics_data,
                              imagersizes):
            R = compute_Rcompensating(q0, v0, v1,
                                      focus_center,
                                      focus_radius,
                                      imagersizes)
            return mrcal.project(nps.matmult(v0,R),
                                 lensmodel, intrinsics_data)

        grids = nps.cat(*[get_reprojections(q0,
                                            v[0,...], v[i,...],
                                            focus_center, focus_radius,
                                            lensmodels[i], intrinsics_data[i],
                                            imagersizes) for i in range(1,len(v))])

        # I look at synthetic data with calibrations fit off ONLY the right half
        # of the image. I look to see how well I'm doing on the left half of the
        # image. I expect this wouldn't work well, and indeed it doesn't: diffs
        # off the reference look poor. BUT cross-validation says we're doing
        # well: diffs off the various solutions look good. Here I look at a
        # point in the diff map the is clearly wrong (diff-off-reference says it
        # is), but that the cross-validation says is fine.
        #
        # Command:
        #   dima@fatty:~/src_boats/mrcal/studies/syntheticdata/scans.LENSMODEL_OPENCV4.cull_leftof2000$ ../../../mrcal-show-intrinsics-diff  --cbmax 2 *.cameramodel ../../../analyses/synthetic_data/reference.cameramodel  --where 800 1080
        #
        # I plot the projection points for the reference and for each of my
        # models. I can see that the models cluster: things look consistent so
        # cross-validation says we're confident. But I can also see the
        # reference off to the side: they're all wrong. Hmmm.

        # g = grids[:,10,19,:]
        # r = g[-1,:]
        # g = nps.glue(q0[10,19], g[:-1,:], axis=-2)
        # gp.plot((r[0],r[1],               dict(legend='reference')),
        #         (g[...,0,None],g[...,1,None], dict(legend=np.arange(len(g)))),
        #         square=1, _with='points', wait=1)
        # sys.exit()

        # I also had this: is it better?
        # stdevs  = np.std(grids, axis=0)
        # difflen = nps.mag(stdevs)

        difflen = np.sqrt(np.mean(nps.norm2(grids-q0),axis=0))

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

        q0      = nps.clump(q0,      n=2)
        q1      = nps.clump(q1,      n=2)
        diff    = nps.clump(diff,    n=2)
        difflen = nps.clump(difflen, n=2)

        plot_data_args = \
            [ (q0  [:,0], q0  [:,1],
               diff[:,0] * vectorscale, diff[:,1] * vectorscale,
               difflen,
               dict(_with='vectors size screen 0.01,20 fixed filled palette',
                    tuplesize=5)) ]

    else:
        if 'set' not in kwargs:
            kwargs['set'] = []
        elif type(kwargs['set']) is not list:
            kwargs['set'] = [kwargs['set']]
        kwargs['set'].extend(['view equal xy',
                              'view map',
                              'contour surface',
                              'key box opaque',
                              'cntrparam levels incremental 4,-0.5,0'])
        plot = \
            gp.gnuplotlib(_3d=1,
                          unset='grid',
                          _xrange=[0,W],
                          _yrange=[H,0],
                          cbrange=[0,cbmax],
                          ascii=1,
                          **kwargs)

        # Currently "with image" can't produce contours. I work around this, by
        # plotting the data a second time.
        # Yuck.
        # https://sourceforge.net/p/gnuplot/mailman/message/36371128/
        plot_data_args = [ (difflen,
                            dict( tuplesize=3,
                                  _with=np.array(('image','lines nosurface'),),
                                  legend = "", # needed to force contour labels
                                  using = imagergrid_using(imagersizes[0], gridn_width, gridn_height)
                            )) ]

    valid_region0 = models[0].valid_intrinsics_region()
    if valid_region0 is not None:
        if vectorfield:
            # 2d plot
            plot_data_args.append( (valid_region0[:,0], valid_region0[:,1],
                                    dict(_with = 'lines lw 3 nocontour',
                                         legend = "valid region of 1st camera")) )
        else:
            # 3d plot
            plot_data_args.append( (valid_region0[:,0], valid_region0[:,1], valid_region0[:,0]*0,
                                    dict(_with = 'lines lw 3 nocontour',
                                         legend = "valid region of 1st camera")) )

    valid_region1 = models[1].valid_intrinsics_region()
    if len(models) == 2 and valid_region1 is not None:
        # The second camera has a valid region, and I should plot it. This has
        # more complexity: each point on the contour of the valid region of the
        # second camera needs to be transformed to the coordinate system of the
        # first camera to make sense. The transformation is complex, and
        # straight lines will not remain straight. I thus resample the polyline
        # more densely.
        v1 = mrcal.unproject(_densify_polyline(valid_region1, spacing = 50),
                             lensmodels[1], intrinsics_data[1])
        valid_region1 = mrcal.project( nps.matmult( v1, nps.transpose(Rcompensating01) ),
                                       lensmodels[0], intrinsics_data[0] )
        if vectorfield:
            # 2d plot
            plot_data_args.append( (valid_region1[:,0], valid_region1[:,1],
                                    dict(_with = 'lines lw 3 nocontour',
                                         legend = "valid region of 2nd camera")) )
        else:
            # 3d plot
            plot_data_args.append( (valid_region1[:,0], valid_region1[:,1], valid_region1[:,0]*0,
                                    dict(_with = 'lines lw 3 nocontour',
                                         legend = "valid region of 2nd camera")) )

    plot.plot( *plot_data_args )

    return plot


def show_valid_intrinsics_region(models,
                                 image    = None,
                                 points   = None,
                                 title    = None,
                                 hardcopy = None,
                                 kwargs   = None):
    r'''Annotates a given image with a valid-intrinsics region

    This function takes in a camera model (or a list of models) and an image. It
    then makes a plot with the valid-intrinsics region(s) drawn on top of the
    image. The image can be a filename or a numpy array. The camera model(s)
    MUST contain the valid-intrinsics region(s).

    If given, points is a (2,N) numpy array of points to draw onto the image
    also

    This is similar to mrcal.annotate_image__valid_intrinsics_region(), but
    instead of writing an image, makes a plot

    '''
    if isinstance(models, mrcal.cameramodel):
        models = (models,)

    W,H = models[0].imagersize()
    for m in models[1:]:
        WH1 = m.imagersize()
        if W != WH1[0] or H != WH1[1]:
            raise Exception("All given models MUST have the same imagersize. Got {} and {}".format((W,H), WH1))

    try:
        valid_regions = [m.valid_intrinsics_region() for m in models]
        if any(r is None for r in valid_regions): raise
    except:
        raise Exception("Some given models have no valid-intrinsics region defined")

    if kwargs is None: kwargs = {}
    else:              kwargs = dict(kwargs)

    if kwargs.get('set') is not None:
        if isinstance(kwargs['set'], str):
            kwargs['set'] = [kwargs['set']]
        else:
            kwargs['set'] = list(kwargs['set'])
    else:
        kwargs['set'] = []
    kwargs['set'].append('key opaque')

    import gnuplotlib as gp

    if 'title' not in kwargs and title is not None:
        kwargs['title'] = title

    if 'hardcopy' not in kwargs and hardcopy is not None:
        kwargs['hardcopy'] = hardcopy

    plot_data_args = []

    if image is not None:
        if isinstance(image, np.ndarray):
            plot_data_args.append( (image, dict(_with='image',
                                                tuplesize = 3)))
        else:
            kwargs['rgbimage'] = image

    plot_data_args.extend( (r[:,0], r[:,1],
                            dict(_with = 'lines lw 3',
                                 legend = 'Model {}'.format(i))) \
                           for i,r in enumerate(valid_regions) )

    if points is not None:
        plot_data_args.append( (points, dict(tuplesize = -2,
                                             _with = 'points pt 7 ps 1')))

    plot = gp.gnuplotlib(square=1,
                         _xrange=[0,W],
                         _yrange=[H,0],
                         **kwargs)

    plot.plot(*plot_data_args)

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
            p = nps.glue(p, np.array([m.group(i+1) for i in range(4)], dtype=float),
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


# mrcal.shellquote is either pipes.quote or shlex.quote, depending on
# python2/python3
try:
    import pipes
    shellquote = pipes.quote
except:
    # python3 puts this into a different module
    import shlex
    shellquote = shlex.quote


def get_mapping_file_framenocameraindex(*files_per_camera):
    r'''Parse image filenames to get the frame numbers

SYNOPSIS

    mapping_file_framenocameraindex = \
      get_mapping_file_framenocameraindex( ('img5-cam2.jpg', 'img6-cam2.jpg'),
                                           ('img6-cam3.jpg', 'img7-cam3.jpg'),)

    print(mapping_file_framenocameraindex)
    ===>
    { 'frame5-cam2.jpg': (5, 0),
      'frame6-cam2.jpg': (6, 0),
      'frame6-cam3.jpg': (6, 1),
      'frame7-cam3.jpg': (7, 1) }


Prior to this call we already applied a glob to some images, so we already know
which images belong to which camera. This function further classifies the images
to find the frame number of each image. This is done by looking at the filenames
of images in each camera, removing common prefixes and suffixes, and using the
central varying filename component as the frame number. This varying component
should be numeric. If it isn't and we have multiple cameras, then we barf. If it
isn't, but we only have one camera, we fallback on sequential frame numbers.

If we have just one image for a camera, I can't tell what is constant in the
filenames, so I return framenumber=0.

ARGUMENTS

- *files_per_camera: one argument per camera. Each argument is a list of strings
   of filenames of images observed by that camera

RETURNED VALUES

We return a dict from filenames to (framenumber, cameraindex) tuples. The
"cameraindex" is a sequential index counting up from 0. cameraindex==0
corresponds to files_per_camera[0] and so on.

The "framenumber" may not be sequential OR starting from 0: this comes directly
from the filename.

    '''

    i_empty = [i for i in range(len(files_per_camera)) if len(files_per_camera[i]) == 0]
    if len(i_empty) > 0:
        raise Exception("These camera globs matched no files: {}".format(i_empty))


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
            for i in range(len(a)):
                if i >= len(b) or a[i] != b[i]:
                    return a[:i]
            return a
        def longest_trailing_substring(a,b):
            for i in range(len(a)):
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

        if len(files) == 1:
            # special case where only one file is given. In this case I can't
            # tell where the frame number is, but I don't really care. I just
            # say that the frame number is 0
            return [0]

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
        else:
            pre_numeric = ''

        m = re.match("^([0-9]*)(.*?)$", trailing)
        if m:
            post_numeric = m.group(1)
        else:
            post_numeric = ''

        return [int(pre_numeric + f[Nleading:Itrailing] + post_numeric) for f in files]




    Ncameras = len(files_per_camera)
    mapping = {}
    for icamera in range(Ncameras):
        try:
            framenumbers = pull_framenumbers(files_per_camera[icamera])
        except:
            # If we couldn't parse out the frame numbers, but there's only one
            # camera, then I just use a sequential list of integers. Since it
            # doesn't matter
            if Ncameras == 1:
                framenumbers = range(len(files_per_camera[icamera]))
            else:
                raise
        if framenumbers is not None:
            mapping.update(zip(files_per_camera[icamera], [(iframe,icamera) for iframe in framenumbers]))
    return mapping


def get_chessboard_observations(Nw, Nh, globs=('*',), corners_cache_vnl=None, jobs=1,
                                exclude_images=set(),
                                weighted=True,
                                keep_level=False):
    r'''Compute the chessboard observations and returns them in a usable form

SYNOPSIS

  observations, indices_frame_camera, paths = \
      mrcal.get_chessboard_observations(10, 10,
                                        ('frame*-cam0.jpg','frame*-cam1.jpg'),
                                        "corners.vnl")

The input to a calibration problem is a set of images of a calibration object
from different angles and positions. This function ingests these images, and
outputs the detected chessboard corner coordinates in a form usable by the mrcal
optimization routines. The "corners_cache_vnl" argument specifies a file to read
from, or write to. This is a vnlog with legend

  # filename x y level

Each record is a chessboard corner. The image filename and corner coordinates
are given. The "level" is a decimation level of the detected corner. If we
needed to cut down the image resolution to detect a corner, its coordinates are
known less precisely, and we use that information to weight the errors
appropriately later. We are able to read data that is missing the "level" field:
a level of 0 (weight = 1) will be filled in. If we read data from a file,
records with a "-" level mean "skip this point". We'll report the point with
weight < 0. Any images with fewer than 3 points will be ignored entirely.

if keep_level: then instead of converting a decimation level to a weight, we'll
just write the decimation level into the returned observations array. We write
<0 to mean "skip this point".

ARGUMENTS

- Nw, Nh: the width and height of the point grid in the calibration object we're
  using

- globs: a list of strings, one per camera, containing globs matching the image
  filenames for that camera. The filenames are expected to encode the
  instantaneous frame numbers, with identical frame numbers implying
  synchronized images. A common scheme is to name an image taken by frame C at
  time T "frameT-camC.jpg". Then images frame10-cam0.jpg and frame10-cam1.jpg
  are assumed to have been captured at the same moment in time by cameras 0 and
  1. With this scheme, if you wanted to calibrate these two cameras together,
  you'd pass ('frame*-cam0.jpg','frame*-cam1.jpg') in the "globs" argument.

  The "globs" argument may be omitted. In this case all images are mapped to the
  same camera.

- corners_cache_vnl: the name of a file to use to read/write the detected
  corners; or a python file object to read data from. If the given file exists
  or a python file object is given, we read the detections from it, and do not
  run the detector. If the given file does NOT exist (which is what happens the
  first time), mrgingham will be invoked to compute the corners from the images,
  and the results will be written to that file. So the same function call can be
  used to both compute the corners initially, and to reuse the pre-computed
  corners with subsequent calls. This exists to save time where re-analyzing the
  same data multiple times.

- jobs: a GNU-Make style parallelization flag. Indicates how many parallel
  processes should be invoked when computing the corners. If given, a numerical
  argument is required. This flag does nothing if the corners-cache file already
  exists

- exclude_images: a set of filenames to exclude from reported results

- weighted: corner detectors can report an uncertainty in the coordinates of
  each corner, and we use that by default. To ignore this, and to weigh all the
  corners equally, call with weighted=True

- keep_level: if True, we write the decimation level into the observations
  array, instead of converting it to a weight first. If keep_level: then
  "weighted" has no effect. The default is False.

RETURNED VALUES

This function returns a tuple (observations, indices_frame_camera, files_sorted)

- observations: an ordered (N,object-height-n,object-width-n,3) array describing
  N board observations where the board has dimensions
  (object-height-n,object-width-n) and each point is an (x,y,weight) pixel
  observation. A weight<0 means "ignore this point". Incomplete chessboard
  observations can be specified in this way. if keep_level: then the decimation
  level appears in the last column, instead of the weight

- indices_frame_camera is an (N,2) array of contiguous, sorted integers where
  each observation is (index_frame,index_camera)

- files_sorted is a list of paths of images corresponding to the observations

Note that this assumes we're solving a calibration problem (stationary cameras)
observing a moving object, so this returns indices_frame_camera. It is the
caller's job to convert this into indices_frame_camintrinsics_camextrinsics,
which mrcal.optimize() expects

    '''

    import os
    import fnmatch
    import subprocess
    import shutil
    from tempfile import mkstemp
    import io
    import copy

    def get_corner_observations(Nw, Nh, globs, corners_cache_vnl, exclude_images=set()):
        r'''Return dot observations, from a cache or from mrgingham

        Returns a dict mapping from filename to a numpy array with a full grid
        of dot observations. If no grid was observed in a particular image, the
        relevant dict entry is empty

        The corners_cache_vnl argument is for caching corner-finder results.
        This can be None if we want to ignore this. Otherwise, this is treated
        as a path to a file on disk or a python file object. If this file
        exists:

            The corner coordinates are read from this file instead of being
            computed. We don't need to actually have the images stored on disk.
            Any image filenames mentioned in this cache file are matched against
            the globs to decide which camera the image belongs to. If it matches
            none of the globs, that image filename is silently ignored

        If this file does not exist:

            We process the images to compute the corner coordinates. Before we
            compute the calibration off these coordinates, we create the cache
            file and store this data there. Thus a subsequent identical
            invocation of mrcal-calibrate-cameras will see this file as
            existing, and will automatically use the data it contains instead of
            recomputing the corner coordinates

        '''

        # Expand any ./ and // etc
        globs = [os.path.normpath(g) for g in globs]

        Ncameras = len(globs)
        files_per_camera = []
        for i in range(Ncameras):
            files_per_camera.append([])

        # images in corners_cache_vnl have paths relative to where the
        # corners_cache_vnl lives
        corners_dir = None

        reading_pipe = isinstance(corners_cache_vnl, io.IOBase)

        if corners_cache_vnl is not None and not reading_pipe:
            corners_dir = os.path.dirname( corners_cache_vnl )

        def accum_files(f):
            for i_camera in range(Ncameras):
                g = globs[i_camera]
                if g[0] != '/':
                    g = '*/' + g
                if fnmatch.fnmatch(os.path.abspath(f), g):
                    files_per_camera[i_camera].append(f)
                    return True
            return False


        pipe_corners_write_fd          = None
        pipe_corners_write_tmpfilename = None
        if corners_cache_vnl is not None and \
           not reading_pipe              and \
           os.path.isdir(corners_cache_vnl):
            raise Exception("Given cache path '{}' is a directory. Must be a file or must not exist". \
                            format(corners_cache_vnl))

        if corners_cache_vnl is None or \
           ( not reading_pipe and \
             not os.path.isfile(corners_cache_vnl) ):
            # Need to compute the dot coords. And maybe need to save them into a
            # cache file too
            if Nw != 10 or Nh != 10:
                raise Exception("mrgingham currently accepts ONLY 10x10 grids")

            args_mrgingham = ['mrgingham', '--jobs',
                              str(jobs)]
            args_mrgingham.extend(globs)

            sys.stderr.write("Computing chessboard corners by running:\n   {}\n". \
                             format(' '.join(shellquote(s) for s in args_mrgingham)))
            if corners_cache_vnl is not None:
                # need to save the corners into a cache. I want to do this
                # atomically: if the dot-finding is interrupted I don't want to
                # be writing incomplete results, so I write to a temporary file
                # and then rename when done
                pipe_corners_write_fd,pipe_corners_write_tmpfilename = mkstemp('.vnl')
                sys.stderr.write("Will save corners to '{}'\n".format(corners_cache_vnl))

            corners_output = subprocess.Popen(args_mrgingham, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                              encoding='ascii')
            pipe_corners_read = corners_output.stdout
        else:
            # Have an existing cache file. Just read it
            if reading_pipe:
                pipe_corners_read = corners_cache_vnl
            else:
                pipe_corners_read = open(corners_cache_vnl, 'r', encoding='ascii')
            corners_output    = None


        mapping = {}
        context0 = dict(f            = '',
                        igrid        = 0,
                        Nvalidpoints = 0)
        # The default weight is 1; the default decimation level is 0
        if keep_level: context0['grid'] = np.zeros((Nh*Nw,3), dtype=float)
        else:          context0['grid'] = np.ones( (Nh*Nw,3), dtype=float)

        context = copy.deepcopy(context0)

        def finish_chessboard_observation():
            nonlocal context
            if context['igrid']:
                if Nw*Nh != context['igrid']:
                    raise Exception("File '{}' expected to have {} points, but got {}". \
                                    format(context['f'], Nw*Nh, context['igrid']))
                if context['f'] not in exclude_images and \
                   context['Nvalidpoints'] > 3:
                    # There is a bit of ambiguity here. The image path stored in
                    # the 'corners_cache_vnl' file is relative to what? It could be
                    # relative to the directory the corners_cache_vnl lives in, or it
                    # could be relative to the current directory. The image
                    # doesn't necessarily need to exist. I implement a few
                    # heuristics to figure this out
                    if corners_dir is None          or \
                       context['f'][0] == '/'       or \
                       os.path.isfile(context['f']):
                        filename_canonical = os.path.normpath(context['f'])
                    else:
                        filename_canonical = os.path.join(corners_dir, context['f'])
                    if accum_files(filename_canonical):
                        mapping[filename_canonical] = context['grid'].reshape(Nh,Nw,3)
                context = copy.deepcopy(context0)

        for line in pipe_corners_read:
            if pipe_corners_write_fd is not None:
                os.write(pipe_corners_write_fd, line.encode())

            if line[0] == '#':
                continue
            m = re.match('(\S+)\s+(.*?)$', line)
            if m is None:
                raise Exception("Unexpected line in the corners output: '{}'".format(line))
            if m.group(2)[:2] == '- ':
                # No observations for this image. Done with this image; move on
                finish_chessboard_observation()
                continue
            if context['f'] != m.group(1):
                # Got data for the next image. Finish out this one
                finish_chessboard_observation()
                context['f'] = m.group(1)

            # The row may have 2 or 3 values: if 3, it contains a decimation
            # level of the corner observation (used for the weight). If 2, a
            # weight of 1.0 is assumed. The weight array is pre-filled with 1.0.
            # A decimation level of - will be used to set weight <0 which means
            # "ignore this point"
            fields = m.group(2).split()
            if len(fields) < 2:
                raise Exception("'corners.vnl' data rows must contain a filename and 2 or 3 values. Instead got line '{}'".format(line))
            else:
                context['grid'][context['igrid'],:2] = (float(fields[0]),float(fields[1]))
                context['Nvalidpoints'] += 1
                if len(fields) == 3:
                    if fields[2] == '-':
                        # ignore this point
                        context['grid'][context['igrid'],2] = -1.0
                        context['Nvalidpoints'] -= 1
                    elif keep_level:
                        context['grid'][context['igrid'],2] = float(fields[2])
                    elif weighted:
                        # convert decimation level to weight. The weight is
                        # 2^(-level). I.e. level-0 -> weight=1, level-1 ->
                        # weight=0.5, etc
                        context['grid'][context['igrid'],2] = 1. / (1 << int(fields[2]))
                    # else use the 1.0 that's already there

            context['igrid'] += 1

        finish_chessboard_observation()

        if corners_output is not None:
            sys.stderr.write("Done computing chessboard corners\n")

            if corners_output.wait() != 0:
                err = corners_output.stderr.read()
                raise Exception("mrgingham failed: {}".format(err))
            if pipe_corners_write_fd is not None:
                os.close(pipe_corners_write_fd)
                shutil.move(pipe_corners_write_tmpfilename, corners_cache_vnl)
        elif not reading_pipe:
            pipe_corners_read.close()

        # If I have multiple cameras, I use the filenames to figure out what
        # indexes the frame and what indexes the camera, so I need at least
        # two images for each camera to figure that out. Example:
        #
        #   I have two cameras, with one image each:
        #   - frame2-cam0.jpg
        #   - frame3-cam1.jpg
        #
        # If this is all I had, it'd be impossible for me to tell whether
        # the images correspond to the same frame or not. But if cam0 also
        # had "frame4-cam0.jpg" then I could look at the same-camera cam0
        # filenames, find the common prefixes,suffixes, and conclude that
        # the frame indices are 2 and 4.
        #
        # If I only have one camera, however, then the details of the
        # filenames don't matter, and I just make sure I have at least one
        # image to look at
        min_num_images = 2 if len(files_per_camera) > 1 else 1
        for i_camera in range(len(files_per_camera)):
            N = len(files_per_camera[i_camera])

            if N < min_num_images:
                raise Exception("Found too few ({}; need at least {}) images containing a calibration pattern in camera {}; glob '{}'". \
                                format(N, min_num_images, i_camera, globs[i_camera]))

        return mapping,files_per_camera


    indices_frame_camera = np.array((), dtype=np.int32)
    observations         = np.array((), dtype=float)

    # basic logic is this:
    #   for frames:
    #       for cameras:
    #           if have observation:
    #               push observations
    #               push indices_frame_camera
    mapping_file_corners,files_per_camera = get_corner_observations(Nw, Nh, globs, corners_cache_vnl, exclude_images)
    mapping_file_framenocameraindex       = get_mapping_file_framenocameraindex(*files_per_camera)

    # I create a file list sorted by frame and then camera. So my for(frames)
    # {for(cameras) {}} loop will just end up looking at these files in order
    files_sorted = sorted(mapping_file_corners.keys(), key=lambda f: mapping_file_framenocameraindex[f][1])
    files_sorted = sorted(files_sorted,                key=lambda f: mapping_file_framenocameraindex[f][0])

    i_observation = 0

    i_frame_last = None
    index_frame  = -1
    for f in files_sorted:
        # The frame indices I return are consecutive starting from 0, NOT the
        # original frame numbers
        i_frame,i_camera = mapping_file_framenocameraindex[f]
        if i_frame_last == None or i_frame_last != i_frame:
            index_frame += 1
            i_frame_last = i_frame

        indices_frame_camera = nps.glue(indices_frame_camera,
                                        np.array((index_frame, i_camera), dtype=np.int32),
                                        axis=-2)
        observations = nps.glue(observations,
                                mapping_file_corners[f],
                                axis=-4)

    return observations, indices_frame_camera, files_sorted


def estimate_local_calobject_poses( indices_frame_camera,
                                    observations,
                                    object_spacing,
                                    object_width_n,object_height_n,
                                    models_or_intrinsics ):
    r"""Estimates pose of observed object in a single-camera view

    Given observations, and an estimate of camera intrinsics (focal
    lengths/imager size/model) computes an estimate of the pose of the
    calibration object in respect to the camera for each frame. This assumes
    that all frames are independent and all cameras are independent. This
    assumes a pinhole camera.

    This function is a wrapper around the solvePnP() openCV call, which does all
    the work.

    The observations are given in a numpy array with axes:

      (iframe, icorner_x, icorner_y, idot2d_xyweight)

    So as an example, the observed pixel coord of the dot (3,4) in frame index 5
    is the 2-vector observations[5,3,4,:2] with weight observations[5,3,4,2]

    Missing observations are given as negative pixel coords.

    This function returns an (Nobservations,4,3) array, with the observations
    aligned with the observations and indices_frame_camera arrays. Each observation
    slice is (4,3) in glue(R, t, axis=-2)

    The camera models are given in the "models_or_intrinsics" argument as a list
    of either:

    - cameramodel object
    - (lensmodel,intrinsics_data) tuple

    Note that this assumes we're solving a calibration problem (stationary
    cameras) observing a moving object, so uses indices_frame_camera, not
    indices_frame_camintrinsics_camextrinsics, which mrcal.optimize() expects

    """

    # For now I ignore all the weights
    observations = observations[..., :2]

    # I'm given models. I remove the distortion so that I can pass the data
    # on to solvePnP()
    lensmodels_intrinsics_data = [ m.intrinsics() if type(m) is mrcal.cameramodel else m for m in models_or_intrinsics ]
    lensmodels     = [di[0] for di in lensmodels_intrinsics_data]
    intrinsics_data = [di[1] for di in lensmodels_intrinsics_data]

    if not all([mrcal.getLensModelMeta(m)['has_core'] for m in lensmodels]):
        raise Exception("this currently works only with models that have an fxfycxcy core. It might not be required. Take a look at the following code if you want to add support")

    fx = [ i[0] for i in intrinsics_data ]
    fy = [ i[1] for i in intrinsics_data ]
    cx = [ i[2] for i in intrinsics_data ]
    cy = [ i[3] for i in intrinsics_data ]

    observations = observations.copy()
    for i_observation in range(len(observations)):
        i_camera = indices_frame_camera[i_observation,1]

        v = mrcal.unproject(observations[i_observation,...],
                            lensmodels[i_camera], intrinsics_data[i_camera])
        observations[i_observation,...] = mrcal.project(v, 'LENSMODEL_PINHOLE',
                                                        intrinsics_data[i_camera][:4])


    Nobservations = indices_frame_camera.shape[0]

    # this wastes memory, but makes it easier to keep track of which data goes
    # with what
    Rt_cf_all = np.zeros( (Nobservations, 4, 3), dtype=float)

    # No calobject_warp. Good-enough for the seeding
    full_object = mrcal.get_ref_calibration_object(object_width_n, object_height_n, object_spacing)

    for i_observation in range(Nobservations):

        i_camera = indices_frame_camera[i_observation,1]
        camera_matrix = np.array((( fx[i_camera], 0,            cx[i_camera]), \
                                  ( 0,            fy[i_camera], cy[i_camera]), \
                                  ( 0,            0,            1.)))
        d = observations[i_observation, ...]

        d = nps.clump( nps.glue(d, full_object, axis=-1), n=2)
        # d is (object_height_n*object_width_n,5); each row is an xy pixel observation followed by the xyz
        # coord of the point in the calibration object. I pick off those rows
        # where the observations are both >= 0. Result should be (N,5) where N
        # <= object_height_n*object_width_n
        i = (d[..., 0] >= 0) * (d[..., 1] >= 0)
        d = d[i,:]

        # copying because cv2.solvePnP() requires contiguous memory apparently
        observations_local = np.array(d[:,:2][..., np.newaxis])
        ref_object         = np.array(d[:,2:][..., np.newaxis])
        result,rvec,tvec   = cv2.solvePnP(np.array(ref_object),
                                        np.array(observations_local),
                                        camera_matrix, None)
        if not result:
            raise Exception("solvePnP failed!")
        if tvec[2] <= 0:

            # The object ended up behind the camera. I flip it, and try to solve
            # again
            result,rvec,tvec = cv2.solvePnP(np.array(ref_object),
                                            np.array(observations_local),
                                            camera_matrix, None,
                                            rvec, -tvec,
                                            useExtrinsicGuess = True)
            if not result:
                raise Exception("retried solvePnP failed!")
            if tvec[2] <= 0:
                raise Exception("retried solvePnP says that tvec.z <= 0")


        Rt_cf = mrcal.Rt_from_rt(nps.glue(rvec.ravel(), tvec.ravel(), axis=-1))

        # visualize the fit
        # x_cam    = nps.matmult(Rt_cf[:3,:],ref_object)[..., 0] + Rt_cf[3,:]
        # x_imager = x_cam[...,:2]/x_cam[...,(2,)] * focal + (imagersize-1)/2
        # import gnuplotlib as gp
        # gp.plot( (x_imager[:,0],x_imager[:,1], dict(legend='solved')),
        #          (observations_local[:,0,0],observations_local[:,1,0], dict(legend='observed')),
        #          square=1,xrange=(0,4000),yrange=(4000,0),
        #          wait=1)
        # import IPython
        # IPython.embed()
        # sys.exit()

        Rt_cf_all[i_observation, :, :] = Rt_cf


    return Rt_cf_all


def _estimate_camera_poses( calobject_poses_local_Rt_cf, indices_frame_camera, \
                            observations, Ncameras,
                            object_spacing,
                            object_width_n, object_height_n):
    r'''Estimate camera poses in respect to each other

    We are given poses of the calibration object in respect to each observing
    camera. We also have multiple cameras observing the same calibration object
    at the same time, and we have local poses for each. We can thus compute the
    relative camera pose from these observations.

    We have many frames that have different observations from the same set of
    fixed-relative-pose cameras, so we compute the relative camera pose to
    optimize the observations

    Note that this assumes we're solving a calibration problem (stationary
    cameras) observing a moving object, so uses indices_frame_camera, not
    indices_frame_camintrinsics_camextrinsics, which mrcal.optimize() expects
    '''

    import heapq


    # I need to compute an estimate of the pose of each camera in the coordinate
    # system of camera0. This is only possible if there're enough overlapping
    # observations. For instance if camera1 has overlapping observations with
    # camera2, but neight overlap with camera0, then I can't relate camera1,2 to
    # camera0. However if camera2 has overlap with camera2, then I can compute
    # the relative pose of camera2 from its overlapping observations with
    # camera0. And I can compute the camera1-camera2 pose from its overlapping
    # data, and then transform to the camera0 coord system using the
    # previously-computed camera2-camera0 pose
    #
    # I do this by solving a shortest-path problem using Dijkstra's algorithm to
    # find a set of pair overlaps between cameras that leads to camera0. I favor
    # edges with large numbers of shared observed frames

    # list of camera-i to camera-0 transforms. I keep doing stuff until this
    # list is full of valid data
    Rt_0c = [None] * (Ncameras-1)

    def compute_pairwise_Rt(icam_to, icam_from):

        # I want to assume that icam_from > icam_to. If it's not true, compute the
        # opposite transform, and invert
        if icam_to > icam_from:
            Rt = compute_pairwise_Rt(icam_from, icam_to)
            return mrcal.invert_Rt(Rt)

        if icam_to == icam_from:
            raise Exception("Got icam_to == icam_from ( = {} ). This was probably a mistake".format(icam_to))

        # Now I KNOW that icam_from > icam_to


        Nobservations = indices_frame_camera.shape[0]

        # This is a hack. I look at the correspondence of camera0 to camera i for i
        # in 1:N-1. I ignore all correspondences between cameras i,j if i!=0 and
        # j!=0. Good enough for now
        #
        # No calobject_warp. Good-enough for the seeding
        full_object = mrcal.get_ref_calibration_object(object_width_n,object_height_n,
                                                       object_spacing)

        A = np.array(())
        B = np.array(())

        # I traverse my observation list, and pick out observations from frames
        # that had data from both my cameras
        i_frame_last = -1
        d0  = None
        d1  = None
        Rt0 = None
        Rt1 = None
        for i_observation in range(Nobservations):
            i_frame_this,i_camera_this = indices_frame_camera[i_observation, ...]
            if i_frame_this != i_frame_last:
                d0  = None
                d1  = None
                Rt0 = None
                Rt1 = None
                i_frame_last = i_frame_this

            # The cameras appear in order. And above I made sure that icam_from >
            # icam_to, so I take advantage of that here
            if i_camera_this == icam_to:
                if Rt0 is not None:
                    raise Exception("Saw multiple camera{} observations in frame {}".format(i_camera_this,
                                                                                            i_frame_this))
                Rt0 = calobject_poses_local_Rt_cf[i_observation, ...]
                d0  = observations[i_observation, ..., :2]
            elif i_camera_this == icam_from:
                if Rt0 is None: # have camera1 observation, but not camera0
                    continue

                if Rt1 is not None:
                    raise Exception("Saw multiple camera{} observations in frame {}".format(i_camera_this,
                                                                                            i_frame_this))
                Rt1 = calobject_poses_local_Rt_cf[i_observation, ...]
                d1  = observations[i_observation, ..., :2]



                # d looks at one frame and has shape (object_height_n,object_width_n,7). Each row is
                #   xy pixel observation in left camera
                #   xy pixel observation in right camera
                #   xyz coord of dot in the calibration object coord system
                d = nps.glue( d0, d1, full_object, axis=-1 )

                # squash dims so that d is (object_height_n*object_width_n,7)
                d = nps.clump(d, n=2)

                ref_object = nps.clump(full_object, n=2)

                # # It's possible that I could have incomplete views of the
                # # calibration object, so I pull out only those point
                # # observations that have a complete view. In reality, I
                # # currently don't accept any incomplete views, and much outside
                # # code would need an update to support that. This doesn't hurt, however

                # # d looks at one frame and has shape (10,10,7). Each row is
                # #   xy pixel observation in left camera
                # #   xy pixel observation in right camera
                # #   xyz coord of dot in the calibration object coord system
                # d = nps.glue( d0, d1, full_object, axis=-1 )

                # # squash dims so that d is (object_height_n*object_width_n,7)
                # d = nps.transpose(nps.clump(nps.mv(d, -1, -3), n=2))

                # # I pick out those points that have observations in both frames
                # i = (d[..., 0] >= 0) * (d[..., 1] >= 0) * (d[..., 2] >= 0) * (d[..., 3] >= 0)
                # d = d[i,:]

                # # ref_object is (N,3)
                # ref_object = d[:,4:]

                A = nps.glue(A, nps.matmult( ref_object, nps.transpose(Rt0[:3,:])) + Rt0[3,:],
                             axis = -2)
                B = nps.glue(B, nps.matmult( ref_object, nps.transpose(Rt1[:3,:])) + Rt1[3,:],
                             axis = -2)

        return mrcal.align3d_procrustes(A, B)


    def compute_connectivity_matrix():
        r'''Returns a connectivity matrix of camera observations

        Returns a symmetric (Ncamera,Ncamera) matrix of integers, where each
        entry contains the number of frames containing overlapping observations
        for that pair of cameras

        '''

        camera_connectivity = np.zeros( (Ncameras,Ncameras), dtype=int )
        def finish_frame(i0, i1):
            for ic0 in range(i0, i1):
                for ic1 in range(ic0+1, i1+1):
                    camera_connectivity[indices_frame_camera[ic0,1], indices_frame_camera[ic1,1]] += 1
                    camera_connectivity[indices_frame_camera[ic1,1], indices_frame_camera[ic0,1]] += 1

        f_current       = -1
        i_start_current = -1

        for i in range(len(indices_frame_camera)):
            f,c = indices_frame_camera[i]
            if f < f_current:
                raise Exception("I'm assuming the frame indices are increasing monotonically")
            if f > f_current:
                # first camera in this observation
                f_current = f
                if i_start_current >= 0:
                    finish_frame(i_start_current, i-1)
                i_start_current = i
        finish_frame(i_start_current, len(indices_frame_camera)-1)
        return camera_connectivity


    shared_frames = compute_connectivity_matrix()

    class Node:
        def __init__(self, camera_idx):
            self.camera_idx    = camera_idx
            self.from_idx      = -1
            self.cost_to_node  = None

        def __lt__(self, other):
            return self.cost_to_node < other.cost_to_node

        def visit(self):
            '''Dijkstra's algorithm'''
            self.finish()

            for neighbor_idx in range(Ncameras):
                if neighbor_idx == self.camera_idx                  or \
                   shared_frames[neighbor_idx,self.camera_idx] == 0:
                    continue
                neighbor = nodes[neighbor_idx]

                if neighbor.visited():
                    continue

                cost_edge = Node.compute_edge_cost(shared_frames[neighbor_idx,self.camera_idx])

                cost_to_neighbor_via_node = self.cost_to_node + cost_edge
                if not neighbor.seen():
                    neighbor.cost_to_node = cost_to_neighbor_via_node
                    neighbor.from_idx     = self.camera_idx
                    heapq.heappush(heap, neighbor)
                else:
                    if cost_to_neighbor_via_node < neighbor.cost_to_node:
                        neighbor.cost_to_node = cost_to_neighbor_via_node
                        neighbor.from_idx     = self.camera_idx
                        heapq.heapify(heap) # is this the most efficient "update" call?

        def finish(self):
            '''A shortest path was found'''
            if self.camera_idx == 0:
                # This is the reference camera. Nothing to do
                return

            Rt_fc = compute_pairwise_Rt(self.from_idx, self.camera_idx)

            if self.from_idx == 0:
                Rt_0c[self.camera_idx-1] = Rt_fc
                return

            Rt_0f = Rt_0c[self.from_idx-1]
            Rt_0c[self.camera_idx-1] = mrcal.compose_Rt( Rt_0f, Rt_fc)

        def visited(self):
            '''Returns True if this node went through the heap and has then been visited'''
            return self.camera_idx == 0 or Rt_0c[self.camera_idx-1] is not None

        def seen(self):
            '''Returns True if this node has been in the heap'''
            return self.cost_to_node is not None

        @staticmethod
        def compute_edge_cost(shared_frames):
            # I want to MINIMIZE cost, so I MAXIMIZE the shared frames count and
            # MINIMIZE the hop count. Furthermore, I really want to minimize the
            # number of hops, so that's worth many shared frames.
            cost = 100000 - shared_frames
            assert(cost > 0) # dijkstra's algorithm requires this to be true
            return cost



    nodes = [Node(i) for i in range(Ncameras)]
    nodes[0].cost_to_node = 0
    heap = []

    nodes[0].visit()
    while heap:
        node_top = heapq.heappop(heap)
        node_top.visit()

    if any([x is None for x in Rt_0c]):
        raise Exception("ERROR: Don't have complete camera observations overlap!\n" +
                        ("Past-camera-0 Rt:\n{}\n".format(Rt_0c))                   +
                        ("Shared observations matrix:\n{}\n".format(shared_frames)))


    return np.ascontiguousarray(nps.cat(*Rt_0c))


def estimate_frame_poses_from_monocular_views(calobject_poses_local_Rt_cf,
                                              extrinsics_rt_fromref,
                                              indices_frame_camera, object_spacing,
                                              object_width_n, object_height_n):
    r'''Estimate poses of the calibration object using no extrinsic information

    We're given

    calobject_poses_local_Rt_cf:

      an array of dimensions (Nobservations,4,3) that contains a
      camera-from-calobject transformation estimate, for each observation of the
      board

    extrinsics_rt_fromref:

      an array of dimensions (Ncameras-1,6) that contains a camerai-from-camera0
      transformation estimate. camera0-from-camera0 is the identity, so this isn't
      stored

    indices_frame_camera:

      an array of shape (Nobservations,2) that indicates which frame and which
      camera has observed the board

    With this data, I return an array of shape (Nframes,6) that contains an
    estimate of the pose of each frame, in the camera0 coord system. Each row is
    (r,t) where r is a Rodrigues rotation and t is a translation that map points
    in the calobject coord system to that of camera 0

    Note that this assumes we're solving a calibration problem (stationary
    cameras) observing a moving object, so uses indices_frame_camera, not
    indices_frame_camintrinsics_camextrinsics, which mrcal.optimize() expects

    '''

    Rt_0c = mrcal.invert_Rt( mrcal.Rt_from_rt( extrinsics_rt_fromref ))


    def process(i_observation0, i_observation1):
        R'''Given a range of observations corresponding to the same frame, estimate the
        frame pose'''

        def T_camera_board(i_observation):
            r'''Transform from the board coords to the camera coords'''
            i_frame,i_camera = indices_frame_camera[i_observation, ...]

            Rt_cf = calobject_poses_local_Rt_cf[i_observation, :,:]
            if i_camera == 0:
                return Rt_cf

            # T_cami_cam0 T_cam0_board = T_cami_board
            return mrcal.compose_Rt( Rt_0c[i_camera-1, ...], Rt_cf)


        # frame poses should map FROM the frame coord system TO the ref coord
        # system (camera 0).

        # special case: if there's a single observation, I just use it
        if i_observation1 - i_observation0 == 1:
            return T_camera_board(i_observation0)

        # Multiple cameras have observed the object for this frame. I have an
        # estimate of these for each camera. I merge them in a lame way: I
        # average out the positions of each point, and fit the calibration
        # object into the mean point cloud
        #
        # No calobject_warp. Good-enough for the seeding
        obj = mrcal.get_ref_calibration_object(object_width_n, object_height_n,
                                               object_spacing)

        sum_obj_unproj = obj*0
        for i_observation in range(i_observation0, i_observation1):
            Rt = T_camera_board(i_observation)
            sum_obj_unproj += mrcal.transform_point_Rt(Rt, obj)

        mean = sum_obj_unproj / (i_observation1 - i_observation0)

        # Got my point cloud. fit

        # transform both to shape = (N*N, 3)
        obj  = nps.clump(obj,  n=2)
        mean = nps.clump(mean, n=2)
        return mrcal.align3d_procrustes( mean, obj )





    frame_poses_rt = np.array(())

    i_frame_current          = -1
    i_observation_framestart = -1;

    for i_observation in range(indices_frame_camera.shape[0]):
        i_frame,i_camera = indices_frame_camera[i_observation, ...]

        if i_frame != i_frame_current:
            if i_observation_framestart >= 0:
                Rt = process(i_observation_framestart, i_observation)
                frame_poses_rt = nps.glue(frame_poses_rt, mrcal.rt_from_Rt(Rt), axis=-2)

            i_observation_framestart = i_observation
            i_frame_current = i_frame

    if i_observation_framestart >= 0:
        Rt = process(i_observation_framestart, indices_frame_camera.shape[0])
        frame_poses_rt = nps.glue(frame_poses_rt, mrcal.rt_from_Rt(Rt), axis=-2)

    return frame_poses_rt


def make_seed_no_distortion( imagersizes,
                             focal_estimate,
                             Ncameras,
                             indices_frame_camera,
                             observations,
                             object_spacing,
                             object_width_n,
                             object_height_n):
    r'''Generate a solution seed for a given input

    Note that this assumes we're solving a calibration problem (stationary
    cameras) observing a moving object, so uses indices_frame_camera, not
    indices_frame_camintrinsics_camextrinsics, which mrcal.optimize() expects
    '''


    def make_intrinsics_vector(i_camera):
        imager_w,imager_h = imagersizes[i_camera]
        return np.array( (focal_estimate, focal_estimate,
                          float(imager_w-1)/2.,
                          float(imager_h-1)/2.))

    intrinsics_data = nps.cat( *[make_intrinsics_vector(i_camera) \
                                 for i_camera in range(Ncameras)] )

    # I compute an estimate of the poses of the calibration object in the local
    # coord system of each camera for each frame. This is done for each frame
    # and for each camera separately. This isn't meant to be precise, and is
    # only used for seeding.
    #
    # I get rotation, translation in a (4,3) array, such that R*calobject + t
    # produces the calibration object points in the coord system of the camera.
    # The result has dimensions (N,4,3)
    intrinsics = [('LENSMODEL_PINHOLE', np.array((focal_estimate,focal_estimate, (imagersize[0]-1.)/2,(imagersize[1]-1.)/2,))) \
                  for imagersize in imagersizes]
    calobject_poses_local_Rt_cf = \
        mrcal.estimate_local_calobject_poses( indices_frame_camera,
                                              observations,
                                              object_spacing,
                                              object_width_n, object_height_n,
                                              intrinsics)
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
    # camera coord system. Rt have dimensions (N-1,4,3)
    camera_poses_Rt01 = _estimate_camera_poses( calobject_poses_local_Rt_cf,
                                                indices_frame_camera,
                                                observations,
                                                Ncameras,
                                                object_spacing,
                                                object_width_n, object_height_n)

    if len(camera_poses_Rt01):
        # extrinsics should map FROM the ref coord system TO the coord system of the
        # camera in question. This is backwards from what I have
        extrinsics = nps.atleast_dims( mrcal.rt_from_Rt(mrcal.invert_Rt(camera_poses_Rt01)),
                                       -2 )
    else:
        extrinsics = np.zeros((0,6))

    frames = \
        mrcal.estimate_frame_poses_from_monocular_views(
            calobject_poses_local_Rt_cf, extrinsics,
            indices_frame_camera,
            object_spacing,
            object_width_n, object_height_n)

    return intrinsics_data,extrinsics,frames


def close_contour(c):
    r'''If a given polyline isn't closed, close it

    Takes in a numpy array of shape (N,2): a sequence of 2d points. If the first
    point and the last point are identical, returns the input. Otherwise returns
    the same array as the input, except the first point is duplicated at the end
    '''
    if c is None: return None
    if np.linalg.norm( c[0,:] - c[-1,:]) < 1e-6:
        return c
    return nps.glue(c, c[0,:], axis=-2)


def get_homography_headon_view(intrinsics0, intrinsics1,
                               q0, q1 = None,
                               Rt10   = None,
                               range0 = None):
    r'''Compute a local homogeneous-coordinate homography

    Let's say I'm observing a small planar patch in the world, parametrized into
    some uv coordinates. uv is an orthonormal coordinate system. Let's say uv0 =
    [u v 0]t and p = Ruv uv0 + tuv = Ruv01 uv + tuv

    dq0/duv = dq0/dp0         dp0/duv = dq0/dp0     Ruv01
    dq1/duv = dq1/dp1 dp1/dp0 dp0/duv = dq1/dp1 R10 Ruv01

    I can compute the local relationship: dq1/dq0 = dq1/duv duv/dq0

    And I then combine this local relationship with a q0->q1 translation into a
    homography that I return.

    If I know the geometry of the cameras and the geometry of the object I'm
    looking at, I can compute this directly. Otherwise, I can make some
    assumptions to fill in the missing information. This function does not know
    what we're looking at, and assumes that it's a plane perpendicular to the
    viewing vector.

    This function REQUIRES q0: the pixel observation in camera0. If q1 is None,
    we must have the range to the object in range0. If no extrinsics are
    available, we assume either no relative rotation or that we're looking very
    far away.

    '''

    # This is temporary. mrcal should not depend on deltapose_lite
    import deltapose_lite



    def get_R_0_uv_headon(p0, p1 = None, Rt10 = None):
        r'''Returns a rotation between head-on plane and world coords

        Two cameras are observing a point. I assume this point lies on a plane
        that's observed as head-on as possible by both cameras. The plane is
        parametrized in some 2D uv coordinates. Let uv0 = [u v 0]t and p = Ruv uv0 +
        tuv = Ruv01 uv + tuv. Here I return Ruv. p is assumed to lie in the
        coordinates of camera 0

        If we don't have reliable extrinsics, set Rt10 to None, and we'll return
        a head-on rotation looking just at camera0. For far-away objects viewer
        by mostly-aligned cameras this should be ok

        '''

        def mean_direction(n0, n1):
            r'''Given two unit vectors, returns an "average"'''

            v = n0+n1
            return v / nps.mag(v)

        def get_R_abn(n):
            r'''Return a rotation with the given n as the last column

            n is assumed to be a normal vector: nps.norm2(n) = 1

            Returns a rotation matrix where the 3rd column is the given vector n. The
            first two vectors are arbitrary, but are guaranteed to produce a valid
            rotation: RtR = I, det(R) = 1
            '''

            # arbitrary, but can't be exactly n
            a = np.array((1., 0, 0, ))
            proj = nps.inner(a, n)
            if abs(proj) > 0.8:
                # Any other orthogonal vector will do. If this projection was >
                # 0.8, I'm guaranteed that all others will be smaller
                a = np.array((0, 1., 0, ))
                proj = nps.inner(a, n)

            a -= proj*n
            a /= nps.mag(a)
            b = np.cross(n,a)
            return nps.transpose(nps.cat(a,b,n))


        n0 = p0/nps.mag(p0)

        if Rt10 is None:
            return get_R_abn(n0)

        if p1 is None:
            p1 = mrcal.transform_point_Rt(Rt10, p0)
        n1 = p1/nps.mag(p1)   # n1 in cam1 coords
        n1 = nps.matmult(n1, Rt10[:3,:]) # n1 in cam0 coords
        n = mean_direction(n0, n1)

        return get_R_abn(n)





    if (q1 is     None and range0 is     None) or \
       (q1 is not None and range0 is not None):
        raise Exception("I need exactly one of (q1,range0) to be given")

    if Rt10 is None:
        if q1 is None:
            # I don't know anything. Assume an identity homography
            return np.eye(3)

        # I have no extrinsics, but I DO have a set of pixel observations.
        # Assume we're looking at faraway objects
        if range0 is not None:
            v0 = mrcal.unproject(q0, *intrinsics0)
            p0 = v0 / nps.mag(v0) * range0
            p1 = p0
        else:
            v0 = mrcal.unproject(q0, *intrinsics0)
            p0 = v0 / nps.mag(v0) * 1000.
            v1 = mrcal.unproject(q1, *intrinsics1)
            p1 = v1 / nps.mag(v1) * 1000.
    else:

        if range0 is not None:
            v0 = mrcal.unproject(q0, *intrinsics0)
            p0 = v0 / nps.mag(v0) * range0
        else:
            p0 = deltapose_lite. \
                compute_3d_intersection_lindstrom( mrcal.rt_from_Rt(Rt10),
                                                   intrinsics0, intrinsics1,
                                                   q0, q1).ravel()

        p1 = mrcal.transform_point_Rt(Rt10, p0)


    _,          dq0_dp0,_ = mrcal.project(p0, *intrinsics0, get_gradients=True)
    q1_estimate,dq1_dp1,_ = mrcal.project(p1, *intrinsics1, get_gradients=True)

    if q1 is None:
        q1 = q1_estimate

    # To transform from the uv0 coord system to cam0 coord system: p0 = R_0_uv0
    # puv0 + t0_uv0 -> dp0/dpuv0 = R_0_uv0. And if we constrain ourselves to the
    # uv surface we have dp0/dpuv = R_0_uv
    #
    # Similarly, p1 = R10 p0 + t10 -> dp1/dp0 = R10
    if Rt10 is not None:
        R10 = Rt10[:3,:]
    else:
        R10 = np.eye(3)

    R_0_uv0 = get_R_0_uv_headon(p0, p1, Rt10)
    R_0_uv  = R_0_uv0[:,:2]
    dq0_duv = nps.matmult(dq0_dp0,      R_0_uv)
    dq1_duv = nps.matmult(dq1_dp1, R10, R_0_uv)

    dq0_dq1 = nps.matmult( dq0_duv,
                           np.linalg.inv(dq1_duv) )
    # I now have the relative pixel homography dq0/dq1 now. This is a 2x2
    # matrix. I embed it into a homogeneous-coordinate homography in a 3x3
    # matrix. And I make sure that q1 maps to q0, so this becomes an
    # absolute-coordinate mapping
    H01 = nps.glue( nps.glue( dq0_dq1,
                              nps.transpose(q0) - nps.matmult(dq0_dq1,nps.transpose(q1)),
                              axis=-1),
                    np.array((0,0,1)), axis=-2)
    return H01


def apply_homography(H, q):
    r'''Apply a homogeneous-coordinate homography to a set of 2d points

    Broadcasting is fully supported

    The homography is a (...., 3,3) matrix: a homogeneous-coordinate
    transformation. The point(s) are in a (...., 2) array.

    '''

    Hq = nps.matmult( nps.dummy(q, -2), nps.transpose(H[..., :,:2]))[..., 0, :] + H[..., 2]
    return Hq[..., :2] / Hq[..., (2,)]
