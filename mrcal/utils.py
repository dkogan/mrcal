#!/usr/bin/python3


import numpy as np
import numpysane as nps
import sys
import re
import cv2
import warnings

import mrcal

def align_procrustes_points_Rt01(p0, p1, weights=None):
    r"""Compute a rigid transformation to align two point clouds

SYNOPSIS

    print(points0.shape)
    ===>
    (100,3)

    print(points1.shape)
    ===>
    (100,3)

    Rt01 = mrcal.align_procrustes_points_Rt01(points0, points1)

    print( np.sum(nps.norm2(mrcal.transform_point_Rt(Rt01, points1) -
                            points0)) )
    ===>
    [The fit error from applying the optimal transformation. If the two point
     clouds match up, this will be small]

Given two sets of 3D points in numpy arrays of shape (N,3), we find the optimal
rotation, translation to align these sets of points. This is done with a
well-known direct method. See:

- https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
- https://en.wikipedia.org/wiki/Kabsch_algorithm

We return a transformation that minimizes the sum 2-norm of the misalignment:

    cost = sum( norm2( w[i] (a[i] - transform(b[i])) ))

We return an Rt transformation to map points in set 1 to points in set 0.

A similar computation can be performed to instead align a set of UNIT VECTORS to
compute an optimal rotation matrix R by calling align_procrustes_vectors_R01().

ARGUMENTS

- p0: an array of shape (..., N, 3). Each row is a point in the coordinate
  system we're transforming TO

- p1: an array of shape (..., N, 3). Each row is a point in the coordinate
  system we're transforming FROM

- weights: optional array of shape (..., N). Specifies the relative weight of
  each point. If omitted, all the given points are weighted equally

RETURNED VALUES

The Rt transformation in an array of shape (4,3). We return the optimal
transformation to align the given point clouds. The transformation maps points
TO coord system 0 FROM coord system 1.

    """
    if weights is None:
        weights = np.ones(p0.shape[:-1], dtype=float)
    return _align_procrustes_points_Rt01(p0,p1,weights)


@nps.broadcast_define( (('N',3,), ('N',3,), ('N',)),
                       (4,3), )
def _align_procrustes_points_Rt01(p0, p1, weights):

    p0 = nps.transpose(p0)
    p1 = nps.transpose(p1)

    # I process Mt instead of M to not need to transpose anything later, and to
    # end up with contiguous-memory results
    Mt = nps.matmult(              (p0 - np.mean(p0, axis=-1)[..., np.newaxis])*weights,
                      nps.transpose(p1 - np.mean(p1, axis=-1)[..., np.newaxis]))
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
    t = np.mean(p0, axis=-1)[..., np.newaxis] - nps.matmult( R, np.mean(p1, axis=-1)[..., np.newaxis] )

    return nps.glue( R, t.ravel(), axis=-2)


def align_procrustes_vectors_R01(v0, v1, weights=None):
    r"""Compute a rotation to align two sets of direction vectors

SYNOPSIS

    print(vectors0.shape)
    ===>
    (100,3)

    print(vectors1.shape)
    ===>
    (100,3)

    R01 = mrcal.align_procrustes_vectors_R01(vectors0, vectors1)

    print( np.mean(1. - nps.inner(mrcal.rotate_point_R(R01, vectors1),
                                  vectors0)) )
    ===>
    [The fit error from applying the optimal rotation. If the two sets of
     vectors match up, this will be small]

Given two sets of normalized direction vectors in 3D (stored in numpy arrays of
shape (N,3)), we find the optimal rotation to align them. This is done with a
well-known direct method. See:

- https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
- https://en.wikipedia.org/wiki/Kabsch_algorithm

We return a rotation that minimizes the weighted sum of the cosine of the
misalignment:

    cost = -sum( w[i] inner(a[i], rotate(b[i])) )

We return a rotation to map vectors in set 1 to vectors in set 0.

A similar computation can be performed to instead align a set of POINTS to
compute an optimal transformation Rt by calling align_procrustes_points_Rt01().

ARGUMENTS

- v0: an array of shape (..., N, 3). Each row is a vector in the coordinate
  system we're transforming TO

- v1: an array of shape (..., N, 3). Each row is a vector in the coordinate
  system we're transforming FROM

- weights: optional array of shape (..., N). Specifies the relative weight of
  each vector. If omitted, everything is weighted equally

RETURNED VALUES

The rotation matrix in an array of shape (3,3). We return the optimal rotation
to align the given vector sets. The rotation maps vectors TO coord system 0 FROM
coord system 1.

    """

    if weights is None:
        weights = np.ones(v0.shape[:-1], dtype=float)
    return _align_procrustes_vectors_R01(v0,v1,weights)


@nps.broadcast_define( (('N',3,), ('N',3,), ('N',)),
                       (3,3), )
def _align_procrustes_vectors_R01(v0, v1, weights):

    v0 = nps.transpose(v0)
    v1 = nps.transpose(v1)

    # I process Mt instead of M to not need to transpose anything later, and to
    # end up with contiguous-memory results
    Mt = nps.matmult( v0*weights, nps.transpose(v1) )
    V,S,Ut = np.linalg.svd(Mt)

    R = nps.matmult(V, Ut)

    # det(R) is now +1 or -1. If it's -1, then this contains a mirror, and thus
    # is not a physical rotation. I compensate by negating the least-important
    # pair of singular vectors
    if np.linalg.det(R) < 0:
        V[:,2] *= -1
        R = nps.matmult(V, Ut)

    return R


def ref_calibration_object(W, H, object_spacing, calobject_warp=None):
    r'''Return the geometry of the calibration object

SYNOPSIS

    import gnuplotlib as gp
    import numpysane as nps

    obj = mrcal.ref_calibration_object( 10,6, 0.1 )

    print(obj.shape)
    ===> (6, 10, 3)

    gp.plot( nps.clump( obj[...,:2], n=2),
             tuplesize = -2,
             _with     = 'points',
             _xrange   = (-0.1,1.0),
             _yrange   = (-0.1,0.6),
             unset     = 'grid',
             square    = True,
             terminal  = 'dumb 74,45')

     0.6 +---------------------------------------------------------------+
         |     +          +           +           +          +           |
         |                                                               |
     0.5 |-+   A     A    A     A     A     A     A     A    A     A   +-|
         |                                                               |
         |                                                               |
     0.4 |-+   A     A    A     A     A     A     A     A    A     A   +-|
         |                                                               |
         |                                                               |
     0.3 |-+   A     A    A     A     A     A     A     A    A     A   +-|
         |                                                               |
         |                                                               |
     0.2 |-+   A     A    A     A     A     A     A     A    A     A   +-|
         |                                                               |
         |                                                               |
     0.1 |-+   A     A    A     A     A     A     A     A    A     A   +-|
         |                                                               |
         |                                                               |
       0 |-+   A     A    A     A     A     A     A     A    A     A   +-|
         |                                                               |
         |     +          +           +           +          +           |
    -0.1 +---------------------------------------------------------------+
               0         0.2         0.4         0.6        0.8          1

Returns the geometry of a calibration object in its own reference coordinate
system in a (H,W,3) array. Only a grid-of-points calibration object is
supported, possibly with some bowing (i.e. what the internal mrcal solver
supports). Each row of the output is an (x,y,z) point. The origin is at the
corner of the grid, so ref_calibration_object(...)[0,0,:] is
np.array((0,0,0)). The grid spans x and y, with z representing the depth: z=0
for a flat calibration object.

A simple parabolic board warping model is supported by passing a (2,) array in
calobject_warp. These 2 values describe additive flex along the x axis and along
the y axis, in that order. In each direction the flex is a parabola, with the
parameter k describing the max deflection at the center. If the edges were at
+-1 we'd have

    z = k*(1 - x^2)

The edges we DO have are at (0,N-1), so the equivalent expression is

    xr = x / (N-1)
    z = k*( 1 - 4*xr^2 + 4*xr - 1 ) =
        4*k*(xr - xr^2) =

ARGUMENTS

- W: how many points we have in the horizontal direction

- H: how many points we have in the vertical direction

- object_spacing: the distance between adjacent points in the calibration
  object. A square object is assumed, so the vertical and horizontal distances
  are assumed to be identical.

- calobject_warp: optional array of shape (2,) defaults to None. Describes the
  warping of the calibration object. If None, the object is flat. If an array is
  given, the values describe the maximum additive deflection along the x and y
  axes

RETURNED VALUES

The calibration object geometry in a (H,W,3) array

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
    r'''Produce synthetic observations of a chessboard

SYNOPSIS

    models = [mrcal.cameramodel("0.cameramodel"),
              mrcal.cameramodel("1.cameramodel"),]
    p,Rt_cam0_boardref = \
        mrcal.make_synthetic_board_observations(models,

                                                # board geometry
                                                10,12,0.1,None,

                                                # Mean board pose
                                                at_xyz_rpydeg,

                                                # Noise radius of the board pose
                                                noiseradius_xyz_rpydeg,

                                                # How many frames we want
                                                100)

    print(p.shape)
    ===> (100, 2, 12, 10, 2)

    print(Rt_cam0_boardref.shape)
    ===> (100, 4, 3)

Given a description of a calibration object and of the cameras observing it,
produces pixel observations of the objects by those cameras. Exactly Nframes
frames of data will be returned. In each frame ALL the cameras will see ALL the
points in the calibration object

The "models" provides the intrinsics and extrinsics.

The calibration objects are nominally have pose at_xyz_rpydeg in the reference
coordinate system, with each pose perturbed uniformly with radius
noiseradius_xyz_rpydeg. I'd like control over roll,pitch,yaw, so this isn't a
normal rt transformation.

Returns the point observations and the poses of the chessboards

ARGUMENTS

- models: an array of mrcal.cameramodel objects, one for each camera we're
  simulating. This is the intrinsics and the extrinsics. Ncameras = len(models)

- object_width_n:  the number of horizontal points in the calibration object grid

- object_height_n: the number of vertical points in the calibration object grid

- object_spacing: the distance between adjacent points in the calibration
  object. A square object is assumed, so the vertical and horizontal distances
  are assumed to be identical.

- calobject_warp: a description of the calibration board warping. None means "no
  warping": the object is flat. Otherwise this is an array of shape (2,). See
  the docs for ref_calibration_object() for a description.

- at_xyz_rpydeg: the nominal pose of the calibration object, in the reference
  coordinate system. This is an array of shape (6,): the position of the center
  of the object, followed by the roll-pitch-yaw orientation, in degrees

- noiseradius_xyz_rpydeg: the deviation-from-nominal for the chessboard for each
  frame. This is the uniform distribution radius; the elements have the same
  meaning as at_xyz_rpydeg

- Nframes: how many frames of observations to return

RETURNED VALUES

We return a tuple:

- The point observations p:
  array of shape (Nframes, Ncameras, object_height, object_width, 2)
- The pose of the chessboards Rt_cam0_boardref:
  array of shape (Nframes, 4,3). This transforms an object returned by
  make_synthetic_board_observations() to the pose that was projected

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
        mrcal.ref_calibration_object(object_width_n,object_height_n,
                                     object_spacing,calobject_warp) - \
        board_translation

    # Transformation from the board returned by ref_calibration_object() to
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

        # I compute the full random block in one shot. This is useful for
        # simulations that want to see identical poses when asking for N-1
        # random poses and when asking for the first N-1 of a set of N random
        # poses

        # shape (Nframes,6)
        randomblock = np.random.uniform(low=-1.0, high=1.0, size=(Nframes,6))

        # shape (Nframes,3)
        xyz = xyz + randomblock[:,:3] * xyz_noiseradius
        rpy = rpy + randomblock[:,3:] * rpy_noiseradius

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
                              frames_rt_toref             = None,
                              points                      = None,

                              axis_scale         = 1.0,
                              object_width_n     = None,
                              object_height_n    = None,
                              object_spacing     = 0,
                              calobject_warp     = None,
                              point_labels       = None,

                              **kwargs):

    r'''Visualize the world resulting from a calibration run

SYNOPSIS

    # Visualize the geometry from some models on disk
    models = [mrcal.cameramodel(m) for m in model_filenames]
    plot1 = mrcal.show_calibration_geometry(models)

    # Solve a calibration problem. Visualize the resulting geometry AND the
    # observed calibration objects and points
    ...
    mrcal.optimize(intrinsics,
                   extrinsics_rt_fromref,
                   frames_rt_toref,
                   points,
                   ...)
    plot2 = \
      mrcal.show_calibration_geometry(extrinsics_rt_fromref,
                                      frames_rt_toref = frames_rt_toref,
                                      points          = points)

This function can visualize the world described by a set of camera models on
disk. It can also be used to visualize the output (or input) of
mrcal.optimize(); the relevant parameters are all identical to those
mrcal.optimize() takes. Cameras are always rendered. If given, the observed
calibration objects and/or the observed points are rendered as well.

This function does all the work for the mrcal-show-calibration-geometry tool.

All arguments except models_or_extrinsics_rt_fromref are optional.

ARGUMENTS

- models_or_extrinsics_rt_fromref: an iterable of mrcal.cameramodel objects or
  (6,) rt arrays. A array of shape (N,6) works to represent N cameras

- cameranames: optional array of strings of labels for the cameras. If omitted,
  we use generic labels. If given, the array must have the same length as
  models_or_extrinsics_rt_fromref

- cameras_Rt_plot_ref: optional transformation(s). If omitted, we plot
  everything in the camera reference coordinate system. If given, we use a
  "plot" coordinate system with the transformation TO plot coordinates FROM the
  reference coordinates given in this argument. This argument can be given as an
  iterable of Rt transformations to use a different one for each camera (None
  means "identity"). Or a single Rt transformation can be given to use that one
  for ALL the cameras

- frames_rt_toref: optional array of shape (N,6). If given, each row of shape
  (6,) is an rt transformation representing the transformation TO the reference
  coordinate system FROM the calibration object coordinate system. The
  calibration object then MUST be defined by passing in valid object_width_n,
  object_height_n, object_spacing parameters. If frames_rt_toref is omitted or
  None, we look for this data in the given camera models. I look at the given
  models in order, and grab the frames from the first model that has them. If
  none of the models have this data and frames_rt_toref is omitted or NULL, then
  I don't plot any frames at all


- object_width_n: the number of horizontal points in the calibration object
  grid. Required only if frames_rt_toref is not None

- object_height_n: the number of vertical points in the calibration object grid.
  Required only if frames_rt_toref is not None

- object_spacing: the distance between adjacent points in the calibration
  object. A square object is assumed, so the vertical and horizontal distances
  are assumed to be identical. Required only if frames_rt_toref is not None

- calobject_warp: optional (2,) array describing the calibration board warping.
  None means "no warping": the object is flat. Used only if frames_rt_toref is
  not None. See the docs for ref_calibration_object() for a description.

- points: optional array of shape (N,3). If omitted, we don't plot the observed
  points. If given, each row of shape (3,) is a point in the reference
  coordinate system.

- point_labels: optional dict from a point index to a string describing it.
  Points in this dict are plotted with this legend; all other points are plotted
  under a generic "points" legend. As many or as few of the points may be
  labelled in this way. If omitted, none of the points will be labelled
  specially. This is used only if points is not None

- axis_scale: optional scale factor for the size of the axes used to represent
  the cameras. Can be omitted to use some reasonable default size, but for very
  large or very small problems, this may be required to make the plot look right

- **kwargs: optional arguments passed verbatim as plot options to gnuplotlib.
  Useful to make hardcopies, set the plot title, etc.

RETURNED VALUES

The gnuplotlib plot object. The plot disappears when this object is destroyed
(by the garbage collection, for instance), so do save this returned plot object
into a variable, even if you're not going to be doing anything with this object

    '''

    import gnuplotlib as gp

    def get_extrinsics_Rt_toref_one(m):
        if isinstance(m, mrcal.cameramodel):
            return m.extrinsics_Rt_toref()
        else:
            return mrcal.invert_Rt(mrcal.Rt_from_rt(m))

    extrinsics_Rt_toref = \
        nps.cat(*[get_extrinsics_Rt_toref_one(m) \
                  for m in models_or_extrinsics_rt_fromref])
    extrinsics_Rt_toref = nps.atleast_dims(extrinsics_Rt_toref, -3)

    if frames_rt_toref is None:
        # No frames were given. I grab them from the first .cameramodel that has
        # them. If none of the models have this data, I don't plot any frames at
        # all
        for m in models_or_extrinsics_rt_fromref:
            _frames_rt_toref = None
            _object_spacing  = None
            _object_width_n  = None
            _object_height_n = None
            _calobject_warp  = None
            if not isinstance(m, mrcal.cameramodel):
                continue
            try:
                optimization_inputs = m.optimization_inputs()
                _frames_rt_toref = optimization_inputs['frames_rt_toref']
                _object_spacing  = optimization_inputs['calibration_object_spacing']
                _object_width_n  = optimization_inputs['observations_board'].shape[-2]
                _object_height_n = optimization_inputs['observations_board'].shape[-3]
                _calobject_warp  = optimization_inputs['calobject_warp']
            except:
                _frames_rt_toref = None
                _object_spacing  = None
                _object_width_n  = None
                _object_height_n = None
                _calobject_warp  = None
                continue
            break

        # Use the data from the model if everything I need was valid
        if _frames_rt_toref is not None:
            frames_rt_toref = _frames_rt_toref
            object_width_n  = _object_width_n
            object_height_n = _object_height_n
            object_spacing  = _object_spacing
            calobject_warp  = _calobject_warp

    if frames_rt_toref is not None:
        frames_rt_toref = nps.atleast_dims(frames_rt_toref, -2)
    if points          is not None:
        points          = nps.atleast_dims(points,          -2)

    try:
        if cameras_Rt_plot_ref.shape == (4,3):
            cameras_Rt_plot_ref = \
                np.repeat(nps.atleast_dims(cameras_Rt_plot_ref,-3),
                          len(extrinsics_Rt_toref),
                          axis=-3)
    except:
        pass

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

        if frames_rt_toref is None or len(frames_rt_toref) == 0:
            return []

        if object_spacing <= 0     or \
           object_width_n  is None or \
           object_height_n is None:
            raise Exception("We're observing calibration boards, so object_spacing and object_width_n and object_height_n must be valid")

        # if observations_board              is None or \
        #    indices_frame_camera_board      is None or \
        #    len(observations_board)         == 0    or \
        #    len(indices_frame_camera_board) == 0:
        #     return []
        # Nobservations = len(indices_frame_camera_board)

        # if icam_highlight is not None:
        #     i_observations_frames = [(i_observation,indices_frame_camera_board[i_observation,0]) \
        #                              for i_observation in range(Nobservations) \
        #                              if indices_frame_camera_board[i_observation,1] == icam_highlight]

        #     i_observations, iframes = nps.transpose(np.array(i_observations_frames))
        #     frames_rt_toref = frames_rt_toref[iframes, ...]


        calobject_ref = ref_calibration_object(object_width_n, object_height_n,
                                               object_spacing, calobject_warp)

        Rf = mrcal.R_from_r(frames_rt_toref[..., :3])
        Rf = nps.mv(Rf,                       0, -4)
        tf = nps.mv(frames_rt_toref[..., 3:], 0, -4)

        # object in the cam0 coord system. shape=(Nframes, object_height_n, object_width_n, 3)
        calobject_cam0 = nps.matmult( calobject_ref, nps.transpose(Rf)) + tf

        # if icam_highlight is not None:
        #     # shape=(Nobservations, object_height_n, object_width_n, 2)
        #     calobject_cam = nps.transform_point_Rt( models[icam_highlight].extrinsics_Rt_fromref(), calobject_cam0 )

        #     print("double-check this. I don't broadcast over the intrinsics anymore")
        #     err = observations[i_observations, ...] - mrcal.project(calobject_cam, *models[icam_highlight].intrinsics())
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

        # if icam_highlight is not None:
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

    q = sample_imager( 60, 40, *model.imagersize() )

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
                                         intrinsics_data[i],
                                         normalize = normalize) \
                         for i in range(len(lensmodel))]), \
               grid
    else:
        # shape: Nheight,Nwidth,3
        return \
            mrcal.unproject(np.ascontiguousarray(grid),
                            lensmodel, intrinsics_data,
                            normalize = normalize), \
            grid


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

RETURNED VALUE

The 'using' string.

    '''

    W,H = imagersize
    if gridn_height is None:
        gridn_height = int(round(H/W*gridn_width))
    return '($1*{}):($2*{}):3'.format(float(W-1)/(gridn_width-1), float(H-1)/(gridn_height-1))


def worst_direction_stdev(cov):
    r'''Compute the worst-direction standard deviation from a 2x2 covariance matrix

SYNOPSIS

    # A covariance matrix
    print(cov)
    ===>
    [[ 1.  -0.4]
     [-0.4  0.5]]

    # Sample 1000 0-mean points using this covariance
    x = np.random.multivariate_normal(mean = np.array((0,0)),
                                      cov  = cov,
                                      size = (1000,))

    # Compute the worst-direction standard deviation of the sampled data
    print(np.sqrt(np.max(np.linalg.eig(np.mean(nps.outer(x,x),axis=0))[0])))
    ===>
    1.1102510878087053

    # The predicted worst-direction standard deviation
    print(mrcal.worst_direction_stdev(cov))
    ===> 1.105304960905736

The covariance of a (2,) random variable can be described by a (2,2)
positive-definite symmetric matrix. The 1-sigma contour of this random variable
is described by an ellipse with its axes aligned with the eigenvectors of the
covariance, and the semi-major and semi-minor axis lengths specified as the sqrt
of the corresponding eigenvalues. This function returns the worst-case standard
deviation of the given covariance: the sqrt of the larger of the two
eigenvalues.

This function supports broadcasting fully.

DERIVATION

Let cov = (a b). If l is an eigenvalue of the covariance then
          (b c)

    (a-l)*(c-l) - b^2 = 0 --> l^2 - (a+c) l + ac-b^2 = 0

    --> l = (a+c +- sqrt( a^2 + 2ac + c^2 - 4ac + 4b^2)) / 2 =
          = (a+c +- sqrt( a^2 - 2ac + c^2 + 4b^2)) / 2 =
          = (a+c)/2 +- sqrt( (a-c)^2/4 + b^2)

So the worst-direction standard deviation is

    sqrt((a+c)/2 + sqrt( (a-c)^2/4 + b^2))

ARGUMENTS

- cov: the covariance matrices given as a (..., 2,2) array. Valid covariances
  are positive-semi-definite (symmetric with eigenvalues >= 0), but this is not
  checked

RETURNED VALUES

The worst-direction standard deviation. This is a scalar or an array, if we're
broadcasting

    '''

    a = cov[..., 0,0]
    b = cov[..., 1,0]
    c = cov[..., 1,1]
    return np.sqrt((a+c)/2 + np.sqrt( (a-c)*(a-c)/4 + b*b))


def _projection_uncertainty_make_output( factorization, Jpacked, dq_dpief_packed,
                                         Nmeasurements_observations,
                                         observed_pixel_uncertainty, what ):
    r'''Helper for projection uncertainty functions

    The given factorization uses the packed, unitless state: p*.

    The given Jpacked uses the packed, unitless state: p*. Jpacked applies to
    all observations. The leading Nmeasurements_observations rows apply to the
    observations of the calibration object, and we use just those for the input
    noise propagation. if Nmeasurements_observations is None: assume that ALL
    the measurements come from the calibration object observations; a simplifed
    expression can be used in this case

    The given dq_dpief_packed uses the packed, unitless state p*, so it already
    includes the multiplication by D in the expressions below. It's sparse, but
    stored densely, so it already includes the multiplication by S



    The docstring of projection_uncertainty() has the derivation that
    concludes that

      Var(p*) = observed_pixel_uncertainty^2 inv(J*tJ*) J*[observations]t J*[observations] inv(J*tJ*)

    where p* and J* are the UNITLESS state and the jacobian respectively.

    In the special case where all the measurements come from
    observations, this simplifies to

      Var(p*) = observed_pixel_uncertainty^2 inv(J*tJ*)

    My factorization is of packed (scaled, unitless) flavors of J (J*). So

      Var(p) = D Var(p*) D

    I want Var(q) = dq/dp[ief] Var(p[ief]) dq/dp[ief]t. Let S = [I 0] where the
    specific nonzero locations specify the locations of [ief]:

      Var(p[ief]) = S Var(p) St

    So

      Var(q) = dq/dp[ief] S D Var(p*) D St dq/dp[ief]t

    In the regularized case I have

      Var(q) = dq/dp[ief] S D inv(J*tJ*) J*[observations]t J*[observations] inv(J*tJ*) D St dq/dp[ief]t observed_pixel_uncertainty^2

    It is far more efficient to compute inv(J*tJ*) D St dq/dp[ief]t than
    inv(J*tJ*) J*[observations]t: there's far less to compute, and the matrices
    are far smaller. Thus I don't compute the covariances directly.

    In the non-regularized case:

      Var(q) = dq/dp[ief] S D inv(J*tJ*) D St dq/dp[ief]t

      1. solve( J*tJ*, D St dq/dp[ief]t)
         The result has shape (Nstate,2)

      2. pre-multiply by dq/dp[ief] S D

      3. multiply by observed_pixel_uncertainty^2

    In the regularized case:

      Var(q) = dq/dp[ief] S D inv(J*tJ*) J*[observations]t J*[observations] inv(J*tJ*) D St dq/dp[ief]t

      1. solve( J*tJ*, D St dq/dp[ief]t)
         The result has shape (Nstate,2)

      2. Pre-multiply by J*[observations]
         The result has shape (Nmeasurements_observations,2)

      3. Compute the sum of the outer products of each row

      4. multiply by observed_pixel_uncertainty^2

    '''

    # shape (2,Nstate)
    A = factorization.solve_xt_JtJ_bt( dq_dpief_packed )
    if Nmeasurements_observations is not None:
        # I have regularization. Use the more complicated expression

        # I see no python way to do matrix multiplication with sparse matrices,
        # so I have my own routine in C. AND the C routine does the outer
        # product, so there's no big temporary expression. It's much faster
        Var_dq = mrcal._mrcal_broadcasted._A_Jt_J_At(A, Jpacked.indptr, Jpacked.indices, Jpacked.data,
                                                     Nleading_rows_J = Nmeasurements_observations)
    else:
        # No regularization. Use the simplified expression
        Var_dq = nps.matmult(dq_dpief_packed, nps.transpose(A))

    if what == 'covariance':           return Var_dq * observed_pixel_uncertainty*observed_pixel_uncertainty
    if what == 'worstdirection-stdev': return worst_direction_stdev(Var_dq) * observed_pixel_uncertainty
    if what == 'rms-stdev':            return np.sqrt(nps.trace(Var_dq)/2.) * observed_pixel_uncertainty
    else: raise Exception("Shouldn't have gotten here. There's a bug")


def _projection_uncertainty( p_cam,
                             lensmodel, intrinsics_data,
                             extrinsics_rt_fromref, frames_rt_toref,
                             factorization, Jpacked, optimization_inputs,
                             istate_intrinsics, istate_extrinsics, istate_frames,
                             slice_optimized_intrinsics,
                             Nmeasurements_observations,
                             observed_pixel_uncertainty,
                             what):
    r'''Helper for projection_uncertainty()

    See docs for _projection_uncertainty_make_output() and
    projection_uncertainty()

    This function does all the work when observing points with a finite range

    '''

    Nstate = Jpacked.shape[-1]
    dq_dpief = np.zeros(p_cam.shape[:-1] + (2,Nstate), dtype=float)

    if frames_rt_toref is not None: Nframes = len(frames_rt_toref)

    if extrinsics_rt_fromref is not None:
        p_ref = \
            mrcal.transform_point_rt( mrcal.invert_rt(extrinsics_rt_fromref),
                                      p_cam )
    else:
        p_ref = p_cam

    if frames_rt_toref is not None:
        # The point in the coord system of all the frames. I index the frames on
        # axis -2
        p_frames = mrcal.transform_point_rt( mrcal.invert_rt(frames_rt_toref),
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

        # dprefallframes_dframes has shape (..., Nframes,3,6)
        _, \
        dprefallframes_dframes, \
        _ = mrcal.transform_point_rt( frames_rt_toref, p_frames,
                                      get_gradients = True)

        # shape (..., 3,6*Nframes)
        # /Nframes because I compute the mean over all the frames
        dpref_dframes = nps.clump( nps.mv(dprefallframes_dframes, -3, -2),
                                   n = -2 ) / Nframes

    _, dq_dpcam, dq_dintrinsics = \
        mrcal.project( p_cam, lensmodel, intrinsics_data,
                       get_gradients = True)

    if istate_intrinsics is not None:
        dq_dintrinsics_optimized = dq_dintrinsics[..., slice_optimized_intrinsics]
        Nintrinsics = dq_dintrinsics_optimized.shape[-1]
        dq_dpief[..., istate_intrinsics:istate_intrinsics+Nintrinsics] = \
            dq_dintrinsics_optimized

    if extrinsics_rt_fromref is not None:
        _, dpcam_drt, dpcam_dpref = \
            mrcal.transform_point_rt(extrinsics_rt_fromref, p_ref,
                                     get_gradients = True)

        dq_dpief[..., istate_extrinsics:istate_extrinsics+6] = \
            nps.matmult(dq_dpcam, dpcam_drt)

        if frames_rt_toref is not None:
            dq_dpief[..., istate_frames:istate_frames+Nframes*6] = \
                nps.matmult(dq_dpcam, dpcam_dpref, dpref_dframes)
    else:
        if frames_rt_toref is not None:
            dq_dpief[..., istate_frames:istate_frames+Nframes*6] = \
                nps.matmult(dq_dpcam, dpref_dframes)

    # Make dq_dpief use the packed state. I call "unpack_state" because the
    # state is in the denominator
    mrcal.unpack_state(dq_dpief, **optimization_inputs)
    return \
        _projection_uncertainty_make_output( factorization, Jpacked,
                                             dq_dpief, Nmeasurements_observations,
                                             observed_pixel_uncertainty,
                                             what)


def _projection_uncertainty_rotationonly( p_cam,
                                          lensmodel, intrinsics_data,
                                          extrinsics_rt_fromref, frames_rt_toref,
                                          factorization, Jpacked, optimization_inputs,
                                          istate_intrinsics, istate_extrinsics, istate_frames,
                                          slice_optimized_intrinsics,
                                          Nmeasurements_observations,
                                          observed_pixel_uncertainty,
                                          what):
    r'''Helper for projection_uncertainty()

    See docs for _projection_uncertainty_make_output() and
    projection_uncertainty()

    This function does all the work when observing points at infinity

    '''

    Nstate = Jpacked.shape[-1]
    dq_dpief = np.zeros(p_cam.shape[:-1] + (2,Nstate), dtype=float)

    if frames_rt_toref is not None: Nframes = len(frames_rt_toref)

    if extrinsics_rt_fromref is not None:
        p_ref = \
            mrcal.rotate_point_r( -extrinsics_rt_fromref[..., :3], p_cam )
    else:
        p_ref = p_cam

    if frames_rt_toref is not None:
        # The point in the coord system of all the frames. I index the frames on
        # axis -2
        p_frames = mrcal.rotate_point_r( -frames_rt_toref[...,:3],
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

        # dprefallframes_dframesr has shape (..., Nframes,3,3)
        _, \
        dprefallframes_dframesr, \
        _ = mrcal.rotate_point_r( frames_rt_toref[...,:3], p_frames,
                                  get_gradients = True)

    _, dq_dpcam, dq_dintrinsics = \
        mrcal.project( p_cam, lensmodel, intrinsics_data,
                       get_gradients = True)

    if istate_intrinsics is not None:
        dq_dintrinsics_optimized = dq_dintrinsics[..., slice_optimized_intrinsics]
        Nintrinsics = dq_dintrinsics_optimized.shape[-1]
        dq_dpief[..., istate_intrinsics:istate_intrinsics+Nintrinsics] = \
            dq_dintrinsics_optimized

    if extrinsics_rt_fromref is not None:
        _, dpcam_dr, dpcam_dpref = \
            mrcal.rotate_point_r(extrinsics_rt_fromref[...,:3], p_ref,
                                 get_gradients = True)
        dq_dpief[..., istate_extrinsics:istate_extrinsics+3] = \
            nps.matmult(dq_dpcam, dpcam_dr)

        if frames_rt_toref is not None:

            dq_dpref = nps.matmult(dq_dpcam, dpcam_dpref)

            # dprefallframes_dframesr has shape (..., Nframes,3,3)
            for i in range(Nframes):
                dq_dpief[..., istate_frames+6*i:istate_frames+6*i+3] = \
                    nps.matmult(dq_dpref, dprefallframes_dframesr[...,i,:,:]) / Nframes
    else:
        if frames_rt_toref is not None:
            # dprefallframes_dframesr has shape (..., Nframes,3,3)
            for i in range(Nframes):
                dq_dpief[..., istate_frames+6*i:istate_frames+6*i+3] = \
                    nps.matmult(dq_dpcam, dprefallframes_dframesr[...,i,:,:]) / Nframes

    # Make dq_dpief use the packed state. I call "unpack_state" because the
    # state is in the denominator
    mrcal.unpack_state(dq_dpief, **optimization_inputs)
    return \
        _projection_uncertainty_make_output( factorization, Jpacked,
                                             dq_dpief, Nmeasurements_observations,
                                             observed_pixel_uncertainty,
                                             what)


def projection_uncertainty( p_cam, model,
                            atinfinity = False,

                            # what we're reporting
                            what = 'covariance'):
    r'''Compute the projection uncertainty of a camera-referenced point

SYNOPSIS

    model = mrcal.cameramodel("xxx.cameramodel")

    q        = np.array((123., 443.))
    distance = 10.0

    pcam = distance * mrcal.unproject(q, *model.intrinsics(), normalize=True)

    print(mrcal.projection_uncertainty(pcam,
                                       model = model,
                                       what  = 'worstdirection-stdev'))
    ===> 0.5

    # So if we have observed a world point at pixel coordinates q, and we know
    # it's 10m out, then we know that the standard deviation of the noise of the
    # pixel obsevation is 0.5 pixels, in the worst direction

After a camera model is computed via a calibration process, the model is
ultimately used in projection/unprojection operations to map between world
coordinates and projected pixel coordinates. We never know the parameters of the
model perfectly, and it is VERY useful to know the resulting uncertainty of
projection. This can be used, among other things, to

- propagate the projection noise down to whatever is using the observed pixels
  to do stuff

- evaluate the quality of calibrations, to know whether a given calibration
  should be accepted, or rejected

- evaluate the stability of a computed model

I quantify uncertainty by propagating expected noise on observed chessboard
corners through the optimization problem we're solving during calibration time
to the solved parameters. And then propagating the noise on the parameters
through projection.

The below derivation is double-checked via simulated noise in
test-calibration-uncertainty.py

The uncertainties can be visualized with the mrcal-show-projection-uncertainty
tool.

DERIVATION

I solve the calibration problem using Ordinary Least Squares, minimizing the
discrepancies between pixel observations and their predictions. The pixel
observations are noisy, and I assume they are zero-mean, independent and
normally-distributed. I treat the x and y coordinates of the observations as two
independent measurements. Thus I minimize a cost function norm2(x). I choose the
weights on the measurements x to get homoscedasticity, and thus the optimal
parameter vector is the maximum-likelihood estimate of the true solution.

Some elements of x depend on the pixel observations, and some don't
(regularization: often needed to help convergence; small-enough to stay close to
the minimum-likelihood result). We care about the measurements that depend on
pixel observations. These are a weighted reprojection error:

    x[i] = w[i] (q[i] - qref[i])

where w[i] is the weight, q[i] is the predicted x or y projection of point i,
and qref[i] is the observation.

The vision algorithm that produced the pixel observation qref[i] knows how
precisely it was able to localize qref[i], and can report some sense of
Var(qref[i]). I assume that we have some baseline Var(qref[i]) =
observed_pixel_uncertainty^2, and that the corner finder reports a scale factor
for each point: Var(qref[i]) = (k[i] observed_pixel_uncertainty)^2. I set w[i] =
1. / k[i], so Var(x[i]) = observed_pixel_uncertainty^2 for all i, and we get
homoscedasticity: an even error distribution on each measurement. This makes
conceptual sense also: low-accuracy qref[i] are weighted less in the
optimization.

I apply a perturbation to the observations qref, reoptimize (assuming everything
is linear) and look what happens to the state p. I start out at an optimum p*:

    E = norm2(x)
    dE/dp (p=p*) = 2 Jt x (p=p*) = 0

I perturb the inputs:

    E(x(p+dp, qref+dqref)) = norm2( x + J dp + dx/dqref dqref)

And I reoptimize:

    dE/ddp ~ ( x + J dp + dx/dqref dqref)t J = 0

I started at an optimum, so Jt x = 0, and thus

    JtJ dp = -Jt dx/dqref dqref

As stated above, for reprojection errors I have

    x[observations] = W (q - qref)

where W is diag(w), a diagonal matrix of observation weights. Some elements of x
don't depend on the observations (let's assume these are trailing elements of
x), so

    dx/dqref = [ -W ]
               [  0 ]

and thus

    JtJ dp = J[observations]t W dqref

So if I perturb my input observation vector qref by dqref, the resulting effect
on the optimal parameters is dp = M dqref. Where

    M = inv(JtJ) J[observations]t W

So

    Var(p) = M Var(qref) Mt

As stated before, I'm assuming independent noise on all observed pixels, with a
standard deviation inversely proportional to the weight:

    Var(qref) = observed_pixel_uncertainty^2 W^-2

and

    Var(p) = observed_pixel_uncertainty^2 M W^-2 Mt=
           = observed_pixel_uncertainty^2 inv(JtJ) J[observations]t W W^-2 W J[observations] inv(JtJ)=
           = observed_pixel_uncertainty^2 inv(JtJ) J[observations]t J[observations] inv(JtJ)

If we have no regularization, and all measurements are pixel errors, then
J[observations] = J and

    Var(p) = observed_pixel_uncertainty^2 inv(JtJ) J[observations]t J[observations] inv(JtJ)
           = observed_pixel_uncertainty^2 inv(JtJ) JtJ inv(JtJ)
           = observed_pixel_uncertainty^2 inv(JtJ)

This is the variance of the full optimization state p. This contains the
intrinsics and extrinsics of ALL the cameras. And it contains ALL the poses of
observed chessboards, and everything else, like the chessboard warp terms.

Note that this does not explicitly depend on W. However, the weight is a part of
the jacobian J. If we were to lose our faith that our measurements are precise,
then Var(qref[i]) would go up, w[i] would go down, x[i] would go down and thus J
would go down as well. And as a result, Var(p) would increase, as expected.

Ultimately the parameters are used in a projection operation. So given a point
in camera coordinates pcam, I project it onto the image plane:

    q = project(pcam, intrinsics)

Propagating the uncertainties from this expression alone is insufficient. We
want to know the projection uncertainty of points in the world coordinate
system, while the above projects points in the local camera coordinate system.
And on top of that, the transform between this local coordinate system and the
world is not fixed: it's an unknown random quantity, just like the lens
parameters. THE ORIGIN AND ORIENTATION OF THE LOCAL CAMERA COORDINATE SYSTEM
INSIDE THE CAMERA HOUSING ARE RANDOM VARIABLES.

I use the poses of the observed chessboards in aggregate to define the world,
since these are the most stationary thing I have.

Let's look at ONE observed chessboard frame: frame i. I want to know the
uncertainty at a pixel coordinate q0. I unproject to the camera coordinate
system, and then transform to the reference coord system and then to the framei
coordinate system:

  pcam0 = unproject(intrinsics0, q0)
  pf0   = T_fr0[i] T_rc0 pcam0

Here T_ab is a transformation of a point from coord system b to the same point
being represented in coord system a. Using homogeneous coordinates T can contain
both a rotation and a translation, and remain linear:

  T = [R t]
      [0 1]

I perturb my optimization state, which changes all the geometric trasformations
and the intrinsics. But the framei coordinate system is an estimate of the
global coordinate system, so I transform and project that point back to look at
the deviation:

  pcam1 = T_cr1 T_rf1[i] T_fr0[i] T_rc0 pcam0
  q1    = project(intrinsics1,  pcam1)

And q1-q0 then represents my uncertainty. How do I combine all the different
estimates from the different chessboard observations? I can take the mean of all
the q1 I get from each frame to use mean(q1)-q0 as the uncertainty. I will
linearize the projection, so everything is linear, and I will have

  T_r1r0 = mean( T_rf1[i] T_rf0[i] )

This "T" is not a valid transformation since mean(R) is not a rotation matrix.
But for small perturbations, this should be close-enough.

I computed Var(p) earlier, which contains the variance of ALL the optimization
parameters together. The noise on the chessboard poses is coupled to the noise
on the extrinsics and to the noise on the intrinsics. And we can apply all these
together to propagate the uncertainty.

Let's define some variables:

- p_i: the intrinsics of a camera
- p_e: the extrinsics of that camera (T_cr)
- p_f: ALL the chessboard poses (T_fr)
- p_ief: the concatenation of p_i, p_e and p_f

I have

    dq = q0 + dq/dp_ief dp_ief

    Var(q) = dq/dp_ief Var(p_ief) (dq/dp_ief)t

    Var(p_ief) is a subset of Var(p), computed above.

    dq/dp_ief = [dq/dp_i dq/dp_e dq/dp_f]

    dq/dp_e = dq/dpcam dpcam/dp_e

    dq/dp_f = dq/dpcam dpcam/dpref dpref/dp_f / Nframes

dq/dp_i and all the constituent expressions comes directly from the project()
and transform calls above. Depending on the details of the optimization problem,
some of these may not exist. For instance, if we're looking at a camera that is
sitting at the reference coordinate system, then there is no p_e, and Var_ief is
smaller: it's just Var_if. If we somehow know the poses of the frames, then
there's no Var_f. If we want to know the uncertainty at distance=infinity, then
we ignore all the translation components of p_e and p_f.

And note that this all assumes a vanilla calibration setup: we're calibration a
number of stationary cameras by observing a moving object. If we're instead
moving the cameras, then there're multiple extrinsics vectors for each set of
intrinsics, and it's not clear what projection uncertainty even means.

Note a surprising consequence of all this: projecting k*pcam in camera
coordinates always maps to the same pixel coordinate q for any non-zero scalar
k. However, the uncertainty DOES depend on k. If a calibration was computed with
lots of chessboard observations at some distance from the camera, then the
uncertainty of projections at THAT distance will be much lower than the
uncertanties of projections at any other distance. And as we get closer and
closer to the camera, the uncertainty grows to infinity as the translation
uncertainty in the extrinsics begins to dominate.

Alright, so we have Var(q). We could claim victory at that point. But it'd be
nice to convert Var(q) into a single number that describes my projection
uncertainty at q. Empirically I see that Var(dq) often describes an eccentric
ellipse, so I want to look at the length of the major axis of the 1-sigma
ellipse:

    eig (a b) --> (a-l)*(c-l)-b^2 = 0 --> l^2 - (a+c) l + ac-b^2 = 0
        (b c)

    --> l = (a+c +- sqrt( a^2+2ac+c^2 - 4ac + 4b^2)) / 2 =
          = (a+c +- sqrt( a^2-2ac+c^2 + 4b^2)) / 2 =
          = (a+c)/2 +- sqrt( (a-c)^2/4 + b^2)

So the worst-case stdev(q) is

    sqrt((a+c)/2 + sqrt( (a-c)^2/4 + b^2))

ARGUMENTS

This function accepts an array of camera-referenced points p_cam and some
representation of parameters and uncertainties (either a single
mrcal.cameramodel object or all of
(lensmodel,intrinsics_data,extrinsics_rt_fromref,frames_rt_toref,Var_ief)). And
a few meta-parameters that describe details of the behavior. This function
broadcasts on p_cam only. We accept

- p_cam: a numpy array of shape (..., 3). This is the set of camera-coordinate
  points where we're querying uncertainty. if not atinfinity: then the full 3D
  coordinates of p_cam are significant, even distance to the camera. if
  atinfinity: the distance to the camera is ignored.

- model: a mrcal.cameramodel object containing the intrinsics, extrinsics, frame
  poses and their covariance. If this isn't given, then each of these MUST be
  given in a separate argument

- lensmodel: a string describing which lens model we're using. This is something
  like 'LENSMODEL_OPENCV4'. This is required if and only if model is None

- intrinsics_data: a numpy array of shape (Nintrinsics,) where Nintrinsics is
  the number of parameters in the intrinsics vector for this lens model,
  returned by mrcal.lensmodel_num_params(lensmodel). This is required if and only if
  model is None

- extrinsics_rt_fromref: a numpy array of shape (6,) or None. This is an rt
  transformation from the reference coordinate system to the camera coordinate
  system. If None: the camera is at the reference coordinate system. Note that
  these are the extrinsics AT CALIBRATION TIME. If we moved the camera after
  calibrating, then this is OK, but for the purposes of uncertainty
  computations, we care about where the camera used to be. This is required if
  and only if model is None

- frames_rt_toref: a numpy array of shape (Nframes,6). These are rt
  transformations from the coordinate system of each calibration object coord
  system to the reference coordinate system. This array represents ALL the
  observed chessboards in a calibration optimization problem. This is required
  if and only if model is None

- Var_ief: a square numpy array with the intrinsics, extrinsics, frame
  covariance. It is the caller's responsibility to make sure that the dimensions
  match the frame counts and whether extrinsics_rt_fromref is None or not. This
  is required if and only if model is None

- atinfinity: optional boolean, defaults to False. If True, we want to know the
  projection uncertainty, looking at a point infinitely-far away. We propagate
  all the uncertainties, ignoring the translation components of the poses

- what: optional string, defaults to 'covariance'. This chooses what kind of
  output we want. Known options are:

  - 'covariance':           return a full (2,2) covariance matrix Var(q) for
                            each p_cam
  - 'worstdirection-stdev': return the worst-direction standard deviation for
                            each p_cam

  - 'rms-stdev':            return the RMS of the worst and best direction
                            standard deviations

RETURN VALUE

A numpy array of uncertainties. If p_cam has shape (..., 3) then:

if what == 'covariance': we return an array of shape (..., 2,2)
else:                    we return an array of shape (...)

    '''

    what_known = set(('covariance', 'worstdirection-stdev', 'rms-stdev'))
    if not what in what_known:
        raise Exception(f"'what' kwarg must be in {what_known}, but got '{what}'")


    lensmodel = model.intrinsics()[0]

    optimization_inputs = model.optimization_inputs()
    if optimization_inputs is None:
        raise Exception("optimization_inputs are unavailable in this model. Uncertainty cannot be computed")

    if not optimization_inputs.get('do_optimize_extrinsics'):
        raise Exception("Computing uncertainty if !do_optimize_extrinsics not supported currently. This is possible, but not implemented. _projection_uncertainty...() would need a path for fixed extrinsics like they already do for fixed frames")

    Jpacked,factorization = \
        mrcal.optimizer_callback( **optimization_inputs )[2:]

    if factorization is None:
        raise Exception("Cannot compute the uncertainty: factorization computation failed")

    # The intrinsics,extrinsics,frames MUST come from the solve when
    # evaluating the uncertainties. The user is allowed to update the
    # extrinsics in the model after the solve, as long as I use the
    # solve-time ones for the uncertainty computation. Updating the
    # intrinsics invalidates the uncertainty stuff so I COULD grab those
    # from the model. But for good hygiene I get them from the solve as
    # well

    # which calibration-time camera we're looking at
    icam_intrinsics = model.icam_intrinsics()
    icam_extrinsics = mrcal.corresponding_icam_extrinsics(icam_intrinsics, **optimization_inputs)

    intrinsics_data   = optimization_inputs['intrinsics'][icam_intrinsics]

    if not optimization_inputs.get('do_optimize_intrinsics_core') and \
       not optimization_inputs.get('do_optimize_intrinsics_distortions'):
        istate_intrinsics          = None
        slice_optimized_intrinsics = None
    else:
        istate_intrinsics = mrcal.state_index_intrinsics(icam_intrinsics, **optimization_inputs)

        i0,i1 = None,None # everything by default

        has_core     = mrcal.lensmodel_meta(lensmodel)['has_core']
        Ncore        = 4 if has_core else 0
        Ndistortions = mrcal.lensmodel_num_params(lensmodel) - Ncore

        if not optimization_inputs.get('do_optimize_intrinsics_core'):
            i0 = Ncore
        if not optimization_inputs.get('do_optimize_intrinsics_distortions'):
            i1 = -Ndistortions

        slice_optimized_intrinsics  = slice(i0,i1)

    try:
        istate_frames = mrcal.state_index_frames(0, **optimization_inputs)
    except:
        istate_frames = None

    if icam_extrinsics < 0:
        extrinsics_rt_fromref = None
        istate_extrinsics     = None
    else:
        extrinsics_rt_fromref = optimization_inputs['extrinsics_rt_fromref'][icam_extrinsics]
        istate_extrinsics     = mrcal.state_index_extrinsics (icam_extrinsics, **optimization_inputs)

    frames_rt_toref = None
    if optimization_inputs.get('do_optimize_frames'):
        frames_rt_toref = optimization_inputs.get('frames_rt_toref')


    Nmeasurements_observations = mrcal.num_measurements_boards(**optimization_inputs)
    if Nmeasurements_observations == mrcal.num_measurements(**optimization_inputs):
        # Note the special-case where I'm using all the observations
        Nmeasurements_observations = None

    observed_pixel_uncertainty = optimization_inputs['observed_pixel_uncertainty']

    # Two distinct paths here that are very similar, but different-enough to not
    # share any code. If atinfinity, I ignore all translations
    if not atinfinity:
        return \
            _projection_uncertainty(p_cam,
                                    lensmodel, intrinsics_data,
                                    extrinsics_rt_fromref, frames_rt_toref,
                                    factorization, Jpacked, optimization_inputs,
                                    istate_intrinsics, istate_extrinsics, istate_frames,
                                    slice_optimized_intrinsics,
                                    Nmeasurements_observations,
                                    observed_pixel_uncertainty,
                                    what)
    else:
        return \
            _projection_uncertainty_rotationonly(p_cam,
                                                 lensmodel, intrinsics_data,
                                                 extrinsics_rt_fromref, frames_rt_toref,
                                                 factorization, Jpacked, optimization_inputs,
                                                 istate_intrinsics, istate_extrinsics, istate_frames,
                                                 slice_optimized_intrinsics,
                                                 Nmeasurements_observations,
                                                 observed_pixel_uncertainty,
                                                 what)


def hypothesis_corner_positions(icam_intrinsics = None,
                                idx_inliers     = None,
                                **optimization_inputs):
    '''Reports the 3D chessboard points observed by a camera at calibration time

SYNOPSIS

    model = mrcal.cameramodel("xxx.cameramodel")

    optimization_inputs = model.optimization_inputs()

    indices = optimization_inputs['indices_frame_camintrinsics_camextrinsics']

    # shape (Nobservations, Nheight, Nwidth, 3)
    pcam = mrcal.hypothesis_corner_positions(**optimization_inputs)

    # shape (Nobservations,1,1,Nintrinsics)
    intrinsics = nps.mv(optimization_inputs['intrinsics'][indices[:,1]],-2,-4)

    optimization_inputs['observations_board'][...,:2] = \
        mrcal.project( p,
                       optimization_inputs['lensmodel'],
                       intrinsics )

    # optimization_inputs now contains perfect, noiseless board observations

    x = mrcal.optimizer_callback(**optimization_inputs)[1]
    print(nps.norm2(x[:mrcal.num_measurements_boards(**optimization_inputs)]))
    ==>
    0

The optimization routine generates hypothetical observations from a set of
parameters being evaluated, trying to match these hypothetical observations to
real observations. To facilitate analysis, this routine returns these
hypothetical coordinates of the chessboard corners being observed. This routine
reports the 3D points in the coordinate system of the observing camera.

The hypothetical points are constructed from

- The calibration object geometry
- The calibration object-reference transformation in
  optimization_inputs['frames_rt_toref']
- The camera extrinsics (reference-camera transformation) in
  optimization_inputs['extrinsics_rt_fromref']
- The table selecting the camera and calibration object frame for each
  observation in
  optimization_inputs['indices_frame_camintrinsics_camextrinsics']

This function knows to return 3 types of output:

- ALL the points observed by ALL cameras together (returned always)
- The points observed by only a specific camera, inliers only (returned if
  icam_intrinsics is not None)
- The points observed by only a specific camera, outliers only (returned if
  icam_intrinsics is not None)

ARGUMENTS

- icam_intrinsics: optional integer specifying which intrinsic camera in the
  optimization_inputs we're looking at. If omitted (or None), I return a single
  numpy array containing the points for all the cameras. Otherwise I return a
  3-tuple with this array in the first element, and the camera-specific arrays
  in the last two elements

- idx_inliers: optional numpy array of booleans of shape
  (Nobservations,object_height,object_width) to select the outliers manually. If
  omitted (or None), the outliers are selected automatically: idx_inliers =
  observations_board[...,2] > 0. This argument is available to pick common
  inliers from two separate solves.

- **optimization_inputs: a dict() of arguments passable to mrcal.optimize() and
  mrcal.optimizer_callback(). We use the geometric data. This dict is obtainable
  from a cameramodel object by calling cameramodel.optimization_inputs()

RETURNED VALUE

if icam_intrinsics is None: returns only the array containing ALL the points
observed by ALL cameras. Otherwise returns a tuple, with that array as the first
element:

- An array of shape (Nobservations, Nheight, Nwidth, 3) containing the
  coordinates (in the coordinate system of each camera) of the chessboard
  corners. These correspond to the observations in
  optimization_inputs['observations_board'], which also have shape
  (Nobservations, Nheight, Nwidth, 3)

- an (N,3) array containing camera-frame 3D points observed at calibration time,
  and accepted by the solver as inliers. Returned only if icam_intrinsics is not
  None

- an (N,3) array containing camera-frame 3D points observed at calibration time,
  but rejected by the solver as outliers. Returned only if icam_intrinsics is
  not None

    '''

    observations_board = optimization_inputs.get('observations_board')
    if observations_board is None:
        return Exception("No board observations available")

    indices_frame_camintrinsics_camextrinsics = \
        optimization_inputs['indices_frame_camintrinsics_camextrinsics']

    object_width_n      = observations_board.shape[-2]
    object_height_n     = observations_board.shape[-3]
    object_spacing      = optimization_inputs['calibration_object_spacing']
    calobject_warp      = optimization_inputs.get('calobject_warp')
    # shape (Nh,Nw,3)
    full_object         = mrcal.ref_calibration_object(object_width_n,
                                                       object_height_n,
                                                       object_spacing,
                                                       calobject_warp)
    frames_Rt_toref = \
        mrcal.Rt_from_rt( optimization_inputs['frames_rt_toref'] )\
        [ indices_frame_camintrinsics_camextrinsics[:,0] ]
    extrinsics_Rt_fromref = \
        nps.glue( mrcal.identity_Rt(),
                  mrcal.Rt_from_rt(optimization_inputs['extrinsics_rt_fromref']),
                  axis = -3 ) \
        [ indices_frame_camintrinsics_camextrinsics[:,2]+1 ]

    Rt_cam_frame = mrcal.compose_Rt( extrinsics_Rt_fromref,
                                     frames_Rt_toref )

    p_cam_calobjects = \
        mrcal.transform_point_Rt(nps.mv(Rt_cam_frame,-3,-5), full_object)

    if icam_intrinsics is None:
        return p_cam_calobjects

    # shape (Nobservations,)
    idx_observations = indices_frame_camintrinsics_camextrinsics[:,1]==icam_intrinsics
    # shape (Nobservations,Nheight,Nwidth)
    if idx_inliers is None:
        idx_inliers = observations_board[...,2] > 0

    idx_outliers = ~idx_inliers

    idx_inliers [~idx_observations] = False
    idx_outliers[~idx_observations] = False

    return \
        p_cam_calobjects, \
        p_cam_calobjects[idx_inliers,  ...], \
        p_cam_calobjects[idx_outliers, ...]


def _options_heatmap_with_contours( # update these
                                    plotoptions,

                                    contour_max, contour_increment,
                                    imagersize, gridn_width, gridn_height):
    r'''Update plotoptions, return curveoptions for a contoured heat map'''


    if 'set' not in plotoptions:
        plotoptions['set'] = []
    elif not isinstance(plotoptions['set'], list):
        plotoptions['set'] = [plotoptions['set']]
    if 'unset' not in plotoptions:
        plotoptions['unset'] = []
    elif not isinstance(plotoptions['unset'], list):
        plotoptions['unset'] = [plotoptions['unset']]

    plotoptions['set'].extend( ['view equal xy',
                                'view map',
                                'contour base',
                                'key box opaque',
                                'style textbox opaque',
                                f'cntrparam levels incremental {contour_max},{contour_increment},0'] )

    plotoptions['_3d']     = True
    plotoptions['_xrange'] = [0,             imagersize[0]]
    plotoptions['_yrange'] = [imagersize[1], 0]
    plotoptions['cbrange'] = [0,             contour_max]
    plotoptions['ascii']   = True # needed for imagergrid_using to work

    plotoptions['unset'].extend(['grid'])

    return \
        dict( tuplesize=3,
              legend = "", # needed to force contour labels
              using = imagergrid_using(imagersize, gridn_width, gridn_height),

              # I plot 3 times:
              # - to make the heat map
              # - to make the contours
              # - to make the contour labels
              _with=np.array(('image',
                              'lines nosurface',
                              'labels boxed nosurface')))


def show_projection_uncertainty(model,
                                gridn_width  = 60,
                                gridn_height = None,

                                observations = False,
                                distance     = None,
                                isotropic    = False,
                                extratitle   = None,
                                cbmax        = 3,
                                **kwargs):
    r'''Visualize the uncertainty in camera projection

SYNOPSIS

    model = mrcal.cameramodel('xxx.cameramodel')

    mrcal.show_projection_uncertainty(model)

    ... A plot pops up displaying the expected projection uncertainty across the
    ... imager

This function uses the expected noise of the calibration-time observations to
estimate the uncertainty of projection of the final model. At calibration time we estimate

- The intrinsics (lens paramaters) of a number of cameras
- The extrinsics (geometry) of a number of cameras in respect to some reference
  coordinate system
- The poses of observed chessboards, also in respect to some reference
  coordinate system

All the coordinate systems move around, and all 3 of these sets of data have
some uncertainty. This tool takes into account all the uncertainties to report
an estimated uncertainty metric. See the docstring for projection_uncertainty()
for a detailed description of the computation.

This function grids the imager, and reports an uncertainty for each point on the
grid. The resulting plot contains a heatmap of the uncertainties for each cell
in the grid, and corresponding contours.

Since the projection uncertainty is based on calibration-time observation
uncertainty, it is sometimes useful to see where the calibration-time
observations were. Pass observations=True to do that.

Since the projection uncertainty is based partly on the uncertainty of the
camera pose, points at different distance from the camera will have different
reported uncertainties EVEN IF THEY PROJECT TO THE SAME PIXEL. The queried
distance is passed in the distance argument. If distance is None (the default)
then we look out to infinity.

To see a 3D view showing the calibration-time observations AND uncertainties for
a set of distances at the same time, call show_projection_uncertainty_xydist()
instead.

For each cell we compute the covariance matrix of the projected (x,y) coords,
and by default we report the worst-direction standard deviation. If isotropic:
we report the RMS standard deviation instead.

ARGUMENTS

- model: the mrcal.cameramodel object being evaluated

- gridn_width: optional value, defaulting to 60. How many points along the
  horizontal gridding dimension

- gridn_height: how many points along the vertical gridding dimension. If None,
  we compute an integer gridn_height to maintain a square-ish grid:
  gridn_height/gridn_width ~ imager_height/imager_width

- observations: optional boolean, defaulting to False. If True, we overlay
  calibration-time observations on top of the uncertainty plot. We should then
  see that more data produces more confident results.

- distance: optional value, defaulting to None. The projection uncertainty
  varies depending on the range to the observed point, with the queried range
  set in this 'distance' argument. If None (the default) we look out to
  infinity.

- isotropic: optional boolean, defaulting to False. We compute the full 2x2
  covariance matrix of the projection. The 1-sigma contour implied by this
  matrix is an ellipse, and we use the worst-case direction by default. If we
  want the RMS size of the ellipse instead of the worst-direction size, pass
  isotropic=True.

- extratitle: optional string to include in the title of the resulting plot

- cbmax: optional value, defaulting to 3.0. Sets the maximum range of the color
  map

- **kwargs: optional arguments passed verbatim as plot options to gnuplotlib.
  Useful to make hardcopies, etc

RETURNED VALUE

The gnuplotlib plot object. The plot disappears when this object is destroyed
(by the garbage collection, for instance), so do save this returned plot object
into a variable, even if you're not going to be doing anything with this object

    '''

    import gnuplotlib as gp
    W,H=model.imagersize()
    if gridn_height is None:
        gridn_height = int(round(H/W*gridn_width))

    lensmodel, intrinsics_data = model.intrinsics()

    q    = sample_imager( gridn_width, gridn_height, *model.imagersize() )
    pcam = mrcal.unproject(q, *model.intrinsics(),
                           normalize = True)

    err = projection_uncertainty(pcam * (distance if distance is not None else 1.0),
                                 model           = model,
                                 atinfinity      = distance is None,
                                 what            = 'rms-stdev' if isotropic else 'worstdirection-stdev')
    if 'title' not in kwargs:
        if distance is None:
            distance_description = ". Looking out to infinity"
        else:
            distance_description = f". Looking out to {distance}m"

        if not isotropic:
            what_description = "Projection"
        else:
            what_description = "Isotropic projection"

        title = f"{what_description} uncertainty (in pixels) based on calibration input noise{distance_description}"
        if extratitle is not None:
            title += ": " + extratitle
        kwargs['title'] = title

    curveoptions = \
        _options_heatmap_with_contours( # update these plot options
                                        kwargs,

                                        cbmax,-0.1,
                                        model.imagersize(),
                                        gridn_width, gridn_height)

    plot_data_args = [(err, curveoptions)]

    valid_intrinsics_region = model.valid_intrinsics_region()
    if valid_intrinsics_region is not None:
        plot_data_args.append( (valid_intrinsics_region[:,0],
                                valid_intrinsics_region[:,1],
                                np.zeros(valid_intrinsics_region.shape[-2]),
                                dict(_with  = 'lines lw 3 nocontour',
                                     legend = "Valid-intrinsics region")) )

    if observations:
        p_cam_calobjects_inliers, p_cam_calobjects_outliers = \
            hypothesis_corner_positions(model.icam_intrinsics(),
                                        **model.optimization_inputs())[1:]
        q_cam_calobjects_inliers = \
            mrcal.project( p_cam_calobjects_inliers, *model.intrinsics() )
        q_cam_calobjects_outliers = \
            mrcal.project( p_cam_calobjects_outliers, *model.intrinsics() )

        if len(q_cam_calobjects_inliers):
            plot_data_args.append( ( q_cam_calobjects_inliers[...,0],
                                     q_cam_calobjects_inliers[...,1],
                                     np.zeros(q_cam_calobjects_inliers.shape[:-1]),
                                     dict( tuplesize = 3,
                                           _with  = 'points nocontour',
                                           legend = 'inliers')) )
        if len(q_cam_calobjects_outliers):
            plot_data_args.append( ( q_cam_calobjects_outliers[...,0],
                                     q_cam_calobjects_outliers[...,1],
                                     np.zeros(q_cam_calobjects_outliers.shape[:-1]),
                                     dict( tuplesize = 3,
                                           _with  = 'points nocontour',
                                           legend = 'outliers')) )

    plot = gp.gnuplotlib(**kwargs)
    plot.plot(*plot_data_args)
    return plot


def show_projection_uncertainty_xydist(model,
                                       gridn_width  = 15,
                                       gridn_height = None,

                                       extratitle   = None,
                                       hardcopy     = None,
                                       cbmax        = 3,
                                       **kwargs):
    r'''Visualize in 3D the uncertainty in camera projection

SYNOPSIS

    model = mrcal.cameramodel('xxx.cameramodel')

    mrcal.show_projection_uncertainty_xydist(model)

    ... A plot pops up displaying the calibration-time observations and the
    ... expected projection uncertainty for various ranges and for various
    ... locations on the imager

This function is similar to show_projection_uncertainty(), but it visualizes the
uncertainty at multiple ranges at the same time.

This function uses the expected noise of the calibration-time observations to
estimate the uncertainty of projection of the final model. At calibration time
we estimate

- The intrinsics (lens paramaters) of a number of cameras
- The extrinsics (geometry) of a number of cameras in respect to some reference
  coordinate system
- The poses of observed chessboards, also in respect to some reference
  coordinate system

All the coordinate systems move around, and all 3 of these sets of data have
some uncertainty. This tool takes into account all the uncertainties to report
an estimated uncertainty metric. See the docstring for projection_uncertainty()
for a detailed description of the computation.

This function grids the imager and a set of observation distances, and reports
an uncertainty for each point on the grid. It also plots the calibration-time
observations. The expectation is that the uncertainty will be low in the areas
where we had observations at calibration time.

ARGUMENTS

- model: the mrcal.cameramodel object being evaluated

- gridn_width: optional value, defaulting to 15. How many points along the
  horizontal gridding dimension

- gridn_height: how many points along the vertical gridding dimension. If None,
  we compute an integer gridn_height to maintain a square-ish grid:
  gridn_height/gridn_width ~ imager_height/imager_width

- extratitle: optional string to include in the title of the resulting plot

- **kwargs: optional arguments passed verbatim as plot options to gnuplotlib.
  Useful to make hardcopies, etc

RETURNED VALUE

The gnuplotlib plot object. The plot disappears when this object is destroyed
(by the garbage collection, for instance), so do save this returned plot object
into a variable, even if you're not going to be doing anything with this object

    '''

    import gnuplotlib as gp
    W,H=model.imagersize()
    if gridn_height is None:
        gridn_height = int(round(H/W*gridn_width))


    p_cam_calobjects_inliers, p_cam_calobjects_outliers = \
        hypothesis_corner_positions(model.icam_intrinsics(),
                                    **model.optimization_inputs())[1:]
    q_cam_calobjects_inliers = \
        mrcal.project( p_cam_calobjects_inliers, *model.intrinsics() )
    q_cam_calobjects_outliers = \
        mrcal.project( p_cam_calobjects_outliers, *model.intrinsics() )
    dist_inliers     = nps.mag(p_cam_calobjects_inliers)
    dist_outliers    = nps.mag(p_cam_calobjects_outliers)
    xydist_inliers   = nps.glue(q_cam_calobjects_inliers,  nps.dummy(dist_inliers, -1), axis=-1)
    xydist_outliers  = nps.glue(q_cam_calobjects_outliers, nps.dummy(dist_outliers,-1), axis=-1)

    # shape (gridn_height,gridn_width,2)
    qxy = mrcal.sample_imager(gridn_width, gridn_height, *model.imagersize())

    ranges = np.linspace( np.min( nps.glue( xydist_inliers [...,2].ravel(),
                                            xydist_outliers[...,2].ravel(),
                                            axis=-1) )/3.0,
                          np.max( nps.glue( xydist_inliers [...,2].ravel(),
                                            xydist_outliers[...,2].ravel(),
                                            axis=-1) )*2.0,
                          10)

    # shape (gridn_height,gridn_width,Nranges,3)
    pcam = \
        nps.dummy(mrcal.unproject( qxy, *model.intrinsics()), -2) * \
        nps.dummy(ranges, -1)

    # shape (gridn_height, gridn_width, Nranges)
    worst_direction_stdev_grid = \
        mrcal.projection_uncertainty( pcam,
                                      model = model,
                                      what = 'worstdirection-stdev')

    grid__x_y_ranges = \
        np.meshgrid(qxy[0,:,0],
                    qxy[:,0,1],
                    ranges,
                    indexing = 'ij')

    if 'title' not in kwargs:
        title = f"Projection uncertainty (in pixels) based on calibration input noise"
        if extratitle is not None:
            title += ": " + extratitle
        kwargs['title'] = title

    if 'set' not in kwargs:
        kwargs['set'] = []
    elif type(kwargs['set']) is not list:
        kwargs['set'] = [kwargs['set']]

    plot = gp.gnuplotlib( _3d      = True,
                          squarexy = True,
                          xlabel   = 'Pixel x',
                          ylabel   = 'Pixel y',
                          zlabel   = 'Range',
                          **kwargs )

    plotargs = [ (grid__x_y_ranges[0].ravel(), grid__x_y_ranges[1].ravel(), grid__x_y_ranges[2].ravel(),
                  nps.xchg(worst_direction_stdev_grid,0,1).ravel().clip(max=3),
                  nps.xchg(worst_direction_stdev_grid,0,1).ravel(),
                  dict(tuplesize = 5,
                       _with = 'points pt 7 ps variable palette',))]
    if len(xydist_inliers):
        plotargs.append( ( xydist_inliers,
                           dict(tuplesize = -3,
                                _with = 'points',
                                legend = 'inliers')) )
    if len(xydist_outliers):
        plotargs.append( ( xydist_outliers,
                           dict(tuplesize = -3,
                                _with = 'points',
                                legend = 'outliers')) )
    plot.plot( *plotargs )

    return plot


def report_residual_statistics( observations, reprojection_error,
                                imagersize,
                                gridn_width  = 20,
                                gridn_height = None):

    r'''Reports fit statistics for regions across the imager

SYNOPSIS

    print( observations.shape )
    ===> (100, 2)

    print( reprojection_error.shape )
    ===> (100,)

    mean, stdev, count, using = \
        mrcal.report_residual_statistics(observations,
                                         reprojection_error,
                                         imagersize,
                                         gridn_width = 30)

    import gnuplotlib as gp
    W,H = imagersize
    gp.plot( np.abs(mean),
             tuplesize = 3,
             _with     = 'image',
             ascii     = True,
             square    = True,
             using     = using)

The mrcal solver optimizes reprojection errors for ALL the observations in ALL
cameras at the same time. It is useful to evaluate the optimal solution by
examining reprojection errors in subregions of the imager, which is accomplished
by this function. All the observations and reprojection errors and subregion
gridding are given. The mean and standard derivation of the reprojection errors
and a point count are returned for each subregion cell. A "using" expression for
plotting is reported as well.

After a problem-free solve, the error distributions in each area of the imager
should be similar, and should match the error distribution of the pixel
observations. If the lens model doesn't fit the data, the statistics will not be
consistent across the region: the residuals would be heteroscedastic.

ARGUMENTS

- observations: an array of shape (..., 2) of observed points. Each row is an
  (x,y) pixel coordinate. This is an input to the optimization

- reprojection_error: an array of reprojection error values corresponding to
  each row in the "observations". This is an output from the optimization

- imagersize: a len-2 iterable: width,height of the imager. With a
  mrcal.cameramodel object this is model.imagersize()

- gridn_width: how many points along the horizontal gridding dimension

- gridn_height: how many points along the vertical gridding dimension. If None,
  we compute an integer gridn_height to maintain a square-ish grid:
  gridn_height/gridn_width ~ imager_height/imager_width

RETURNED VALUES

This function returns a tuple

- mean: an array of shape (gridn_height,gridn_width). Contains the mean of
  reprojection errors in the corresponding cell

- stdev: an array of shape (gridn_height,gridn_width). Contains the standard
  deviation of reprojection errors in the corresponding cell

- count: an array of shape (gridn_height,gridn_width). Contains the count of
  observations in the corresponding cell

- using: is a "using" keyword for plotting the output matrices with gnuplotlib.
  See the docstring for imagergrid_using() for details

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


    # observations,reprojection_error each have shape (N,2): each slice is xy. I
    # have 2N measurements, so I flatten the errors, and double-up the
    # observations
    errflat = reprojection_error.ravel()
    obsflat = nps.clump(nps.mv(nps.cat(observations,observations), -3, -2), n=2)

    # Each has shape (2,Nheight,Nwidth)
    mean,stdev,count = nps.mv( residual_stats(obsflat, errflat, c),
                               -1, 0)
    return mean,stdev,count,imagergrid_using(imagersize, gridn_width, gridn_height)


def show_projection_behavior(model,
                             mode,
                             scale        = 1.,
                             cbmax        = 25.0,
                             gridn_width  = 60,
                             gridn_height = None,
                             extratitle   = None,
                             **kwargs):

    r'''Visualize the behavior of a lens

SYNOPSIS

    model = mrcal.cameramodel('xxx.cameramodel')

    mrcal.show_projection_behavior( model, 'heatmap' )

    ... A plot pops up displaying how much this model deviates from a pinhole
    ... model across the imager

This function treats a pinhole projection as a baseline, and visualizes
deviations from this baseline. So wide lenses will have a lot of reported
"distortion".

This function has 3 modes of operation, specified as a string in the 'mode'
argument.

ARGUMENTS

- model: the mrcal.cameramodel object being evaluated

- mode: this function can produce several kinds of visualizations, with the
  specific mode selected as a string in this argument. Known values:

  - 'heatmap': the imager is gridded, as specified by the
    gridn_width,gridn_height arguments. For each point in the grid, we evaluate
    the difference in projection between the given model, and a pinhole model
    with the same core intrinsics (focal lengths, center pixel coords). This
    difference is color-coded and a heat map is displayed.

  - 'vectorfield': this is the same as 'heatmap', except we display a vector for
    each point, intead of a color-coded cell. If legibility requires the vectors
    being larger or smaller, pass an appropriate value in the 'scale' argument

  - 'radial': Looks at radial distortion only. Plots a curve showing the
    magnitude of the radial distortion as a function of the distance to the
    center

- scale: optional value, defaulting to 1.0. Used to scale the rendered vectors
  if mode=='vectorfield'

- cbmax: optional value, defaulting to 25.0. Sets the maximum range of the color
  map and of the contours if mode=='heatmap'

- gridn_width: how many points along the horizontal gridding dimension. Used if
  mode=='vectorfield' or mode=='heatmap'

- gridn_height: how many points along the vertical gridding dimension. If None,
  we compute an integer gridn_height to maintain a square-ish grid:
  gridn_height/gridn_width ~ imager_height/imager_width. Used if
  mode=='vectorfield' or mode=='heatmap'

- extratitle: optional string to include in the title of the resulting plot

- **kwargs: optional arguments passed verbatim as plot options to gnuplotlib.
  Useful to make hardcopies, etc

RETURNED VALUE

The gnuplotlib plot object. The plot disappears when this object is destroyed
(by the garbage collection, for instance), so do save this returned plot object
into a variable, even if you're not going to be doing anything with this object

    '''

    import gnuplotlib as gp

    lensmodel, intrinsics_data = model.intrinsics()
    imagersize                  = model.imagersize()

    if 'title' not in kwargs:

        title = "Effects of {}".format(lensmodel)
        if extratitle is not None:
            title += ": " + extratitle
        kwargs['title'] = title

    if 'set' not in kwargs:
        kwargs['set'] = []
    elif type(kwargs['set']) is not list:
        kwargs['set'] = [kwargs['set']]





    W,H = imagersize
    if gridn_height is None:
        gridn_height = int(round(H/W*gridn_width))

    if not mrcal.lensmodel_meta(lensmodel)['has_core']:
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
                             # any of the unprojections could be nan, so I do the best I can
                             _xrange = [0,np.max(np.nan_to_num(nps.glue(th_corners,
                                                                        th_centersx,
                                                                        th_centersy,
                                                                        axis=-1)))
                                        * 1.01],
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

        curveoptions = \
            _options_heatmap_with_contours( # update these plot options
                                            kwargs,

                                            cbmax,-1,
                                            imagersize,
                                            gridn_width, gridn_height)
        delta = dgrid-grid

        # shape: gridn_height,gridn_width. Because numpy (and thus gnuplotlib) want it that
        # way
        distortion = nps.mag(delta)

        plot = gp.gnuplotlib( **kwargs)
        plot.plot(distortion, **curveoptions)
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


def _splined_stereographic_domain(lensmodel):
    '''Return the stereographic domain for splined-stereographic lens models

SYNOPSIS

    model = mrcal.cameramodel(model_filename)

    lensmodel = model.intrinsics()[0]

    domain_contour = mrcal._splined_stereographic_domain(lensmodel)

Splined stereographic models are defined by a splined surface. This surface is
indexed by normalized stereographic-projected points. This surface is defined in
some finite area, and this function reports a piecewise linear contour reporting
this region.

This function only makes sense for splined stereographic models.

RETURNED VALUE

An array of shape (N,2) containing a contour representing the projection domain.

    '''

    if not re.match('LENSMODEL_SPLINED_STEREOGRAPHIC', lensmodel):
        raise Exception(f"This only makes sense with splined models. Input uses {lensmodel}")

    ux,uy = mrcal.knots_for_splined_models(lensmodel)
    # shape (Ny,Nx,2)
    u = np.ascontiguousarray(nps.mv(nps.cat(*np.meshgrid(ux,uy)), 0, -1))

    meta = mrcal.lensmodel_meta(lensmodel)
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
    r'''Return the difference of two closed polygons

SYNOPSIS

    import numpy as np
    import numpysane as nps
    import gnuplotlib as gp

    A = np.array(((-1,-1),( 1,-1),( 1, 1),(-1, 1),(-1,-1)))
    B = np.array(((-.1,-1.1),( .1,-1.1),( .1, 1.1),(-.1, 1.1),(-.1,-1.1)))

    diff = mrcal.polygon_difference(A, B)

    gp.plot( (A, dict(legend = 'A', _with = 'lines')),
             (B, dict(legend = 'B', _with = 'lines')),
             *[ ( r, dict( _with     = 'filledcurves closed fillcolor "red"',
                           legend    = 'difference'))
                for r in diff],
             tuplesize = -2,
             square    = True,
             wait      = True)

Given two polygons specified as a point sequence in arrays of shape (N,2) this
function computes the topological difference: all the regions contained in the
positive polygon, but missing in the negative polygon. The result could be
empty, or it could contain any number of disconnected polygons, so a list of
polygons is returned. Each of the constituent resulting polygons is guaranteed
to not have holes. If any holes are found when computing the difference, we cut
apart the resulting shape until no holes remain.

ARGUMENTS

- positive: a polygon specified by a sequence of points in an array of shape
  (N,2). The resulting difference describes regions contained inside the
  positive polygon

- negative: a polygon specified by a sequence of points in an array of shape
  (N,2). The resulting difference describes regions outside the negative polygon

RETURNED VALUE

A list of arrays of shape (N,2). Each array in the list describes a hole-free
polygon as a sequence of points. The difference is a union of all these
constituent polygons. This list could have 0 elements (empty difference) or N
element (difference consists of N separate polygons)

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
                coords = p.exterior.coords
                if len(coords) == 0:
                    return []
                return [np.array(coords)]
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


def show_splined_model_surface(model, xy,
                               imager_domain = True,
                               extratitle    = None,
                               **kwargs):

    r'''Visualize the surface represented by a splined model

SYNOPSIS

    model = mrcal.cameramodel(model_filename)

    mrcal.show_splined_model_surface( model, 'x' )

    ... A plot pops up displaying the spline knots, the spline surface (for the
    ... "x" coordinate), the spline domain and the imager boundary

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
domain. Both are useful, and this is controlled by the imager_domain argument
(the default is True). The spline is defined in the stereographic projection
domain, so in the imager domain the knot grid and the domain boundary become
skewed. At this time the spline representation can cross itself, which is
visible as a kink in the domain boundary.

One use for this function is to check that the field-of-view we're using for
this model is reasonable. We'd like the field of view to be wide-enough to cover
the whole imager, but not much wider, since representing invisible areas isn't
useful. Ideally the surface domain boundary (that this tool displays) is just
wider than the imager edges (which this tool also displays).

ARGUMENTS

- model: the mrcal.cameramodel object being evaluated

- xy: 'x' or 'y': selects the surface we're looking at. We have a separate
  surface for the x and y coordinates, with the two sharing the knot positions

- imager_domain: optional boolean defaults to True. If False: we plot everything
  against normalized stereographic coordinates; in this representation the knots
  form a regular grid, and the surface domain is a rectangle, but the imager
  boundary is curved. If True: we plot everything against the rendered pixel
  coordinates; the imager boundary is a rectangle, while the knots and domain
  become curved

- extratitle: optional string to include in the title of the resulting plot

- **kwargs: optional arguments passed verbatim as plot options to gnuplotlib.
  Useful to make hardcopies, etc

RETURNED VALUES

The gnuplotlib plot object. The plot disappears when this object is destroyed
(by the garbage collection, for instance), so do save this returned plot object
into a variable, even if you're not going to be doing anything with this object

    '''

    if   xy == 'x': ixy = 0
    elif xy == 'y': ixy = 1
    else:
        raise Exception("xy should be either 'x' or 'y'")

    lensmodel,intrinsics_data = model.intrinsics()
    W,H                       = model.imagersize()

    if not re.match('LENSMODEL_SPLINED_STEREOGRAPHIC', lensmodel):
        raise Exception(f"This only makes sense with splined models. Input uses {lensmodel}")


    import gnuplotlib as gp

    if 'title' not in kwargs:

        title = f"Surface for {lensmodel}. Looking at deltau{xy}"
        if extratitle is not None:
            title += ": " + extratitle
        kwargs['title'] = title

    if 'set' not in kwargs:
        kwargs['set'] = []
    elif type(kwargs['set']) is not list:
        kwargs['set'] = [kwargs['set']]


    ux_knots,uy_knots = mrcal.knots_for_splined_models(lensmodel)
    meta = mrcal.lensmodel_meta(lensmodel)
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
                       zlabel   = f"Deltau{xy} (unitless)")
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

    domain_contour_u = _splined_stereographic_domain(lensmodel)
    knots_u = nps.clump(nps.mv(nps.cat(*np.meshgrid(ux_knots,uy_knots)),
                               0, -1),
                        n = 2)
    if imager_domain:
        domain_contour = \
            mrcal.project(
                mrcal.unproject_stereographic( domain_contour_u),
                lensmodel, intrinsics_data)
        knots = \
            mrcal.project(
                mrcal.unproject_stereographic( np.ascontiguousarray(knots_u)),
                lensmodel, intrinsics_data)
    else:
        domain_contour = domain_contour_u
        knots = knots_u

    data.extend( [ ( imager_boundary,
                     dict(_with     = 'lines lw 2',
                          tuplesize = -2,
                          legend    = 'imager boundary')),
                   ( domain_contour,
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
                                             domain_contour)
    except Exception as e:
        # sometimes the domain_contour self-intersects, and this makes us
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

RETURNED VALUE

The mask that indicates whether each point is within the region

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


def implied_Rt10__from_unprojections(q0, v0, v1,
                                     weights      = None,
                                     atinfinity   = True,
                                     focus_center = np.zeros((2,), dtype=float),
                                     focus_radius = 1.0e8):

    r'''Compute the implied-by-the-intrinsics transformation to fit two cameras' projections

SYNOPSIS

    models = ( mrcal.cameramodel('cam0-dance0.cameramodel'),
               mrcal.cameramodel('cam0-dance1.cameramodel') )

    lensmodels      = [model.intrinsics()[0] for model in models]
    intrinsics_data = [model.intrinsics()[1] for model in models]

    # v  shape (...,Ncameras,Nheight,Nwidth,...)
    # q0 shape (...,         Nheight,Nwidth,...)
    v,q0 = \
        mrcal.sample_imager_unproject(60, None,
                                      *models[0].imagersize(),
                                      lensmodels, intrinsics_data,
                                      normalize = True)
    implied_Rt10 = \
        mrcal.implied_Rt10__from_unprojections(q0, v[0,...], v[1,...])

    q1 = mrcal.project( mrcal.transform_point_Rt(implied_Rt10, v[0,...]),
                        *models[1].intrinsics())

    projection_diff = q1 - q0

When comparing projections from two lens models, it is usually necessary to
align the geometry of the two cameras, to cancel out any transformations implied
by the intrinsics of the lenses. This transformation is computed by this
function, used primarily by show_projection_diff() and the
mrcal-show-projection-diff tool.

What are we comparing? We project the same world point into the two cameras, and
report the difference in projection. Usually, the lens intrinsics differ a bit,
and the implied origin of the camera coordinate systems and their orientation
differ also. These geometric uncertainties are baked into the intrinsics. So
when we project "the same world point" we must apply a geometric transformation
to compensate for the difference in the geometry of the two cameras. This
transformation is unknown, but we can estimate it by fitting projections across
the imager: the "right" transformation would result in apparent low projection
diffs in a wide area.

The primary inputs are unprojected gridded samples of the two imagers, obtained
with something like sample_imager_unproject(). We grid the two imagers, and
produce normalized observation vectors for each grid point. We pass the pixel
grid from camera0 in q0, and the two unprojections in v0, v1. This function then
tries to find a transformation to minimize

  norm2( project(camera1, transform(v0)) - q1 )

We return an Rt transformation to map points in the camera0 coordinate system to
the camera1 coordinate system. Some details about this general formulation are
significant:

- The subset of points we use for the optimization
- What kind of transformation we use

In most practical usages, we would not expect a good fit everywhere in the
imager: areas where no chessboards were observed will not fit well, for
instance. From the point of view of the fit we perform, those ill-fitting areas
should be treated as outliers, and they should NOT be a part of the solve. How
do we specify the well-fitting area? The best way is to use the model
uncertainties to pass the weights in the "weights" argument (see
show_projection_diff() for an implementation). If uncertainties aren't
available, or if we want a faster solve, the focus region can be passed in the
focus_center, focus_radius arguments. By default, these are set to encompass the
whole imager, since the uncertainties would take care of everything, but without
uncertainties (weights = None), these should be set more discriminately. It is
possible to pass both a focus region and weights, but it's probably not very
useful.

Unlike the projection operation, the diff operation is NOT invariant under
geometric scaling: if we look at the projection difference for two points at
different locations along a single observation ray, there will be a variation in
the observed diff. This is due to the geometric difference in the two cameras.
If the models differed only in their intrinsics parameters, then this would not
happen. Thus this function needs to know how far from the camera it should look.
By default (atinfinity = True) we look out to infinity. In this case, v0 is
expected to contain unit vectors. To use any other distance, pass atinfinity =
False, and pass POINTS in v0 instead of just observation directions. v1 should
always be normalized. Generally the most confident distance will be where the
chessboards were observed at calibration time.

Practically, it is very easy for the unprojection operation to produce nan or
inf values. And the weights could potentially have some invalid values also.
This function explicitly checks for such illegal data in v0, v1 and weights, and
ignores those points.

ARGUMENTS

- q0: an array of shape (Nh,Nw,2). Gridded pixel coordinates covering the imager
  of both cameras

- v0: an array of shape (...,Nh,Nw,3). An unprojection of q0 from camera 0. If
  atinfinity, this should contain unit vectors, else it should contain points in
  space at the desired distance from the camera. This array may have leading
  dimensions that are all used in the fit. These leading dimensions correspond
  to those in the "weights" array

- v1: an array of shape (Nh,Nw,3). An unprojection of q0 from camera 1. This
  should always contain unit vectors, regardless of the value of atinfinity

- weights: optional array of shape (...,Nh,Nw); None by default. If given, these
  are used to weigh each fitted point differently. Usually we use the projection
  uncertainties to apply a stronger weight to more confident points. If omitted
  or None, we weigh each point equally. This array may have leading dimensions
  that are all used in the fit. These leading dimensions correspond to those in
  the "v0" array

- atinfinity: optional boolean; True by default. If True, we're looking out to
  infinity, and I compute a rotation-only fit; a full Rt transformation is still
  returned, but Rt[3,:] is 0; v0 should contain unit vectors. If False, I'm
  looking out to a finite distance, and v0 should contain 3D points specifying
  the positions of interest.

- focus_center: optional array of shape (2,); (0,0) by default. Used to indicate
  that we're interested only in a subset of pixels q0, a distance focus_radius
  from focus_center. By default focus_radius is LARGE, so we use all the points.
  This is intended to be used if no uncertainties are available, and we need to
  manually select the focus region.

- focus_radius: optional value; LARGE by default. Used to indicate that we're
  interested only in a subset of pixels q0, a distance focus_radius from
  focus_center. By default focus_radius is LARGE, so we use all the points. This
  is intended to be used if no uncertainties are available, and we need to
  manually select the focus region.

RETURNED VALUE

An array of shape (4,3), representing an Rt transformation from camera0 to
camera1. If atinfinity then we're computing a rotation-fit only, but we still
report a full Rt transformation with the t component set to 0

    '''

    # This is very similar in spirit to what compute_Rcorrected_dq_dintrinsics() did
    # (removed in commit 4240260), but that function worked analytically, while this
    # one explicitly computes the rotation by matching up known vectors.

    import scipy.optimize

    if weights is None:
        weights = np.ones(v0.shape[:-1], dtype=float)
    else:
        # Any inf/nan weight or vector are set to 0
        weights = weights.copy()
        weights[ ~np.isfinite(weights) ] = 0.0

    v0 = v0.copy()
    v1 = v1.copy()

    # v0 had shape (..., Nh,Nw,3). Collapse all the leading dimensions into one
    # And do the same for weights
    v0      = nps.clump(v0,      n = len(v0.shape)     -3)
    weights = nps.clump(weights, n = len(weights.shape)-2)

    i_nan_v0 = ~np.isfinite(v0)
    v0[i_nan_v0] = 0.
    weights[i_nan_v0[...,0]] = 0.0
    weights[i_nan_v0[...,1]] = 0.0
    weights[i_nan_v0[...,2]] = 0.0

    i_nan_v1 = ~np.isfinite(v1)
    v1[i_nan_v1] = 0.
    weights[..., i_nan_v1[...,0]] = 0.0
    weights[..., i_nan_v1[...,1]] = 0.0
    weights[..., i_nan_v1[...,2]] = 0.0

    # We try to match the geometry in a particular region
    q_off_center = q0 - focus_center
    i = nps.norm2(q_off_center) < focus_radius*focus_radius
    if np.count_nonzero(i)<3:
        raise Exception("Focus region contained too few points")

    v0_cut  = v0     [...,i, :]
    v1_cut  = v1     [    i, :]
    weights = weights[...,i   ]

    def residual_jacobian_rt(rt):

        # rtp0 has shape (...,N,3)
        rtp0, drtp0_drt, _ = \
            mrcal.transform_point_rt(rt, v0_cut,
                                     get_gradients = True)

        # inner(a,b)/(mag(a)*mag(b)) ~ cos(x) ~ 1 - x^2/2
        # Each of these has shape (...,N)
        mag_rtp0 = nps.mag(rtp0)
        inner    = nps.inner(rtp0, v1_cut)
        th2      = 2.* (1.0 - inner / mag_rtp0)
        x        = th2 * weights

        # shape (...,N,6)
        dmag_rtp0_drt = nps.matmult( nps.dummy(rtp0, -2),   # shape (...,N,1,3)
                                     drtp0_drt              # shape (...,N,3,6)
                                     # matmult has shape (...,N,1,6)
                                   )[...,0,:] / \
                                   nps.dummy(mag_rtp0, -1)  # shape (...,N,1)
        # shape (..., N,6)
        dinner_drt    = nps.matmult( nps.dummy(v1_cut, -2), # shape (    N,1,3)
                                     drtp0_drt              # shape (...,N,3,6)
                                     # matmult has shape (...,N,1,6)
                                   )[...,0,:]

        # dth2 = 2 (inner dmag_rtp0 - dinner mag_rtp0)/ mag_rtp0^2
        # shape (...,N,6)
        J = 2. * \
            (nps.dummy(inner,    -1) * dmag_rtp0_drt - \
             nps.dummy(mag_rtp0, -1) * dinner_drt) / \
             nps.dummy(mag_rtp0*mag_rtp0, -1) * \
             nps.dummy(weights,-1)
        return x.ravel(), nps.clump(J, n=len(J.shape)-1)


    def residual_jacobian_r(r):

        # rv0     has shape (N,3)
        # drv0_dr has shape (N,3,3)
        rv0, drv0_dr, _ = \
            mrcal.rotate_point_r(r, v0_cut,
                                 get_gradients = True)

        # inner(a,b)/(mag(a)*mag(b)) ~ cos(x) ~ 1 - x^2/2
        # Each of these has shape (N)
        inner = nps.inner(rv0, v1_cut)
        th2   = 2.* (1.0 - inner)
        x     = th2 * weights

        # shape (N,3)
        dinner_dr = nps.matmult( nps.dummy(v1_cut, -2), # shape (N,1,3)
                                 drv0_dr                # shape (N,3,3)
                                 # matmult has shape (N,1,3)
                               )[:,0,:]

        J = -2. * dinner_dr * nps.dummy(weights,-1)
        return x, J


    cache = {'rt': None}
    def residual(rt, f):
        if cache['rt'] is None or not np.array_equal(rt,cache['rt']):
            cache['rt'] = rt
            cache['x'],cache['J'] = f(rt)
        return cache['x']
    def jacobian(rt, f):
        if cache['rt'] is None or not np.array_equal(rt,cache['rt']):
            cache['rt'] = rt
            cache['x'],cache['J'] = f(rt)
        return cache['J']


    # # gradient check
    # import gnuplotlib as gp
    # rt0 = np.random.random(6)*1e-3
    # x0,J0 = residual_jacobian_rt(rt0)
    # drt = np.random.random(6)*1e-7
    # rt1 = rt0+drt
    # x1,J1 = residual_jacobian_rt(rt1)
    # dx_theory = nps.matmult(J0, nps.transpose(drt)).ravel()
    # dx_got    = x1-x0
    # relerr = (dx_theory-dx_got) / ( (np.abs(dx_theory)+np.abs(dx_got))/2. )
    # gp.plot(relerr, wait=1, title='rt')
    # r0 = np.random.random(3)*1e-3
    # x0,J0 = residual_jacobian_r(r0)
    # dr = np.random.random(3)*1e-7
    # r1 = r0+dr
    # x1,J1 = residual_jacobian_r(r1)
    # dx_theory = nps.matmult(J0, nps.transpose(dr)).ravel()
    # dx_got    = x1-x0
    # relerr = (dx_theory-dx_got) / ( (np.abs(dx_theory)+np.abs(dx_got))/2. )
    # gp.plot(relerr, wait=1, title='r')
    # sys.exit()

    if atinfinity:

        r = np.random.random(3) * 1e-3

        res = scipy.optimize.least_squares(residual,
                                           r,
                                           jac=jacobian,
                                           method='trf',

                                           loss='soft_l1',
                                           f_scale = (1.0e-1 * np.pi/180.)**2., # 0.1 deg^2
                                           # max_nfev=1,
                                           args=(residual_jacobian_r,),

                                           # Without this, the optimization was
                                           # ending too quickly, and I was
                                           # seeing not-quite-optimal solutions.
                                           # Especially for
                                           # very-nearly-identical rotations
                                           # requires scipy >= 1.3.0
                                           gtol = None,
                                           verbose=0)
        Rt = np.zeros((4,3), dtype=float)
        Rt[:3,:] = mrcal.R_from_r(res.x)
        return Rt

    else:

        rt = np.random.random(6) * 1e-3

        res = scipy.optimize.least_squares(residual,
                                           rt,
                                           jac=jacobian,
                                           method='trf',

                                           loss='soft_l1',
                                           f_scale = (1.0e-1 * np.pi/180.)**2., # 0.1 deg^2
                                           # max_nfev=1,
                                           args=(residual_jacobian_rt,),

                                           # Without this, the optimization was
                                           # ending too quickly, and I was
                                           # seeing not-quite-optimal solutions.
                                           # Especially for
                                           # very-nearly-identical rotations
                                           gtol = None,
                                           verbose=0)
        return mrcal.Rt_from_rt(res.x)


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


def projection_diff(models,
                    gridn_width  = 60,
                    gridn_height = None,

                    distance     = None,

                    use_uncertainties= True,
                    focus_center     = None,
                    focus_radius     = -1.,
                    implied_Rt10     = None):
    r'''Compute the difference in projection between N models

SYNOPSIS

    models = ( mrcal.cameramodel('cam0-dance0.cameramodel'),
               mrcal.cameramodel('cam0-dance1.cameramodel') )

    difference,_,q0,_ = mrcal.projection_diff(models)

    print(q0.shape)
    ==> (40,60)

    print(difference.shape)
    ==> (40,60)

    # The differences are computed across a grid. 'q0' is the pixel centers of
    # each grid cell. 'difference' is the projection variation between the two
    # models at each cell

It is often useful to compare the projection behavior of two camera models. For
instance, one may want to validate a calibration by comparing the results of two
different chessboard dances. Or one may want to evaluate the stability of the
intrinsics in response to mechanical or thermal stresses. This function makes
these comparisons, and returns the results. mrcal.show_projection_diff() ALSO
produces a visualization.

In the most common case we're given exactly 2 models to compare, and we compute
the xy differences in projection at each point. If we're given more than 2
models, we compute the STANDARD DEVIATION of all the differences instead.

We do this:

- grid the imager
- unproject each point in the grid from one camera to produce a world point
- apply a transformation we compute to match up the two camera geometries
- reproject the transformed points to the other camera
- look at the resulting pixel difference in the reprojection

When looking at multiple cameras, their lens intrinsics differ. Less obviously,
the position and orientation of the camera coordinate system in respect to the
physical camera housing differ also. These geometric uncertainties are baked
into the intrinsics. So when we project "the same world point" into both
cameras, we must apply a geometric transformation because we want to be
comparing projections of world points (relative to the camera housing), not
projections relative to the (floating) camera coordinate systems. This
transformation is unknown, but we can estimate it by fitting projections across
the imager: the "right" transformation would result in apparent low projection
differences in a wide area.

This transformation is computed by implied_Rt10__from_unprojections(), and some
details of its operation are significant:

- The imager area we use for the fit
- Which world points we're looking at

In most practical usages, we would not expect a good fit everywhere in the
imager: areas where no chessboards were observed will not fit well, for
instance. From the point of view of the fit we perform, those ill-fitting areas
should be treated as outliers, and they should NOT be a part of the solve. How
do we specify the well-fitting area? The best way is to use the model
uncertainties: these can be used to emphasize the confident regions of the
imager. This behavior is selected with use_uncertainties=True, which is the
default. If uncertainties aren't available, or if we want a faster solve, pass
use_uncertainties=False. The well-fitting region can then be passed using the
focus_center,focus_radius arguments to indicate the circle in the imager we care
about.

If use_uncertainties then the defaults for focus_center,focus_radius are set to
utilize all the data in the imager. If not use_uncertainties, then the defaults
are to use a more reasonable circle of radius min(width,height)/6 at the center
of the imager. Usually this is sufficiently correct, and we don't need to mess
with it. If we aren't guided to the correct focus region, the
implied-by-the-intrinsics solve will try to fit lots of outliers, which would
result in an incorrect transformation, which in turn would produce overly-high
reported diffs. A common case when this happens is if the chessboard
observations used in the calibration were concentrated to the side of the image
(off-center), no uncertainties were used, and the focus_center was not pointed
to that area.

If we KNOW that there is no geometric difference between our cameras, and we
thus should look at the intrinsics differences only, then we don't need to
estimate the transformation. Indicate this case by passing focus_radius=0.

Unlike the projection operation, the diff operation is NOT invariant under
geometric scaling: if we look at the projection difference for two points at
different locations along a single observation ray, there will be a variation in
the observed diff. This is due to the geometric difference in the two cameras.
If the models differed only in their intrinsics parameters, then this variation
would not appear. Thus we need to know how far from the camera to look, and this
is specified by the "distance" argument. By default (distance = None) we look
out to infinity. If we care about the projection difference at some other
distance, pass that here. Multiple distances can be passed in an iterable. We'll
then fit the implied-by-the-intrinsics transformation using all the distances,
and we'll display the best-fitting difference for each pixel. Generally the most
confident distance will be where the chessboards were observed at calibration
time.

ARGUMENTS

- models: iterable of mrcal.cameramodel objects we're comparing. Usually there
  will be 2 of these, but more than 2 is possible. The intrinsics are used; the
  extrinsics are NOT.

- gridn_width: optional value, defaulting to 60. How many points along the
  horizontal gridding dimension

- gridn_height: how many points along the vertical gridding dimension. If None,
  we compute an integer gridn_height to maintain a square-ish grid:
  gridn_height/gridn_width ~ imager_height/imager_width

- distance: optional value, defaulting to None. The projection difference varies
  depending on the range to the observed world points, with the queried range
  set in this 'distance' argument. If None (the default) we look out to
  infinity. We can compute the implied-by-the-intrinsics transformation off
  multiple distances if they're given here as an iterable. This is especially
  useful if we have uncertainties, since then we'll emphasize the best-fitting
  distances.

- use_uncertainties: optional boolean, defaulting to True. If True we use the
  whole imager to fit the implied-by-the-intrinsics transformation, using the
  uncertainties to emphasize the confident regions. If False, it is important to
  select the confident region using the focus_center and focus_radius arguments.
  If use_uncertainties is True, but that data isn't available, we report a
  warning, and try to proceed without.

- focus_center: optional array of shape (2,); the imager center by default. Used
  to indicate that the implied-by-the-intrinsics transformation should use only
  those pixels a distance focus_radius from focus_center. This is intended to be
  used if no uncertainties are available, and we need to manually select the
  focus region.

- focus_radius: optional value. If use_uncertainties then the default is LARGE,
  to use the whole imager. Else the default is min(width,height)/6. Used to
  indicate that the implied-by-the-intrinsics transformation should use only
  those pixels a distance focus_radius from focus_center. This is intended to be
  used if no uncertainties are available, and we need to manually select the
  focus region. Pass focus_radius=0 to avoid computing the transformation, and
  to use the identity. This would mean there're no geometric differences, and
  we're comparing the intrinsics only

- implied_Rt10: optional Rt transformation (numpy array of shape (4,3)). If
  given, I use the given value for the implied-by-the-intrinsics transformation
  instead of fitting it. If omitted, I compute the transformation. Exclusive
  with focus_center, focus_radius. Valid only if exactly two models are given.

RETURNED VALUE

A tuple

- difflen: a numpy array of shape (gridn_height,gridn_width) containing the
  magnitude of differences at each cell, or the standard deviation of
  differences if len(models)>1. if len(models)==1: this is nps.mag(diff)

- diff: a numpy array of shape (gridn_height,gridn_width,2) containing the
  vector of differences at each cell. If len(models)>1 this isn't well-defined,
  and we set this to None

- q0: a numpy array of shape (gridn_height,gridn_width,2) containing the pixel
  coordinates of each grid cell

- implied_Rt10: the geometric Rt transformation in an array of shape (...,4,3).
  This is either whatever was passed into this function (if anything was), or
  the identity if focus_radius==0 or the fitted results. if len(models)>1: this
  is an array of shape (len(models)-1,4,3), with slice i representing the
  transformation between camera 0 and camera i+1.

    '''

    if implied_Rt10 is not None:
        if len(models) != 2:
            raise Exception("implied_Rt10 may be given ONLY if I have exactly two models")
        if focus_center is not None:
            raise Exception("implied_Rt10 is given, so focus_center, focus_radius shouldn't be")

        use_uncertainties = False

    if distance is None:
        atinfinity = True
        distance   = 1.0
    else:
        atinfinity = False
        distance   = nps.atleast_dims(np.array(distance), -1)
        distance   = nps.mv(distance.ravel(), -1,-4)

    imagersizes = np.array([model.imagersize() for model in models])
    if np.linalg.norm(np.std(imagersizes, axis=-2)) != 0:
        raise Exception("The diff function needs all the imager dimensions to match. Instead got {}". \
                        format(imagersizes))
    W,H=imagersizes[0]

    lensmodels      = [model.intrinsics()[0] for model in models]
    intrinsics_data = [model.intrinsics()[1] for model in models]

    # v  shape (Ncameras,Nheight,Nwidth,3)
    # q0 shape (         Nheight,Nwidth,2)
    v,q0 = sample_imager_unproject(gridn_width, gridn_height,
                                   W, H,
                                   lensmodels, intrinsics_data,
                                   normalize = True)

    if focus_radius == 0:
        use_uncertainties = False

    if use_uncertainties:
        try:
            # len(uncertainties) = Ncameras. Each has shape (len(distance),Nh,Nw)
            uncertainties = \
                [ mrcal.projection_uncertainty(v[i] * distance,
                                               models[i],
                                               atinfinity = atinfinity,
                                               what       = 'worstdirection-stdev') \
                  for i in range(len(models)) ]
        except Exception as e:
            print(f"WARNING: show_projection_diff() was asked to use uncertainties, but they aren't available/couldn't be computed. Falling back on the region-based-only logic\nException: {e}",
                  file = sys.stderr)
            use_uncertainties = False
            uncertainties     = None
    else:
        use_uncertainties = False
        uncertainties     = None

    if focus_center is None: focus_center = ((W-1.)/2., (H-1.)/2.)
    if focus_radius < 0:
        if use_uncertainties:
            focus_radius = max(W,H) * 100 # whole imager
        else:
            focus_radius = min(W,H)/6.

    if len(models) == 2:
        # Two models. Take the difference and call it good

        if implied_Rt10 is not None:
            # I already have the transformation, so no need to compute it
            pass
        elif focus_radius == 0:
            implied_Rt10 = mrcal.identity_Rt()
        else:
            # weights has shape (len(distance),Nh,Nw))
            if uncertainties is not None:
                weights = 1.0 / (uncertainties[0]*uncertainties[1])
            else:
                weights = None

            # weight may be inf or nan. implied_Rt10__from_unprojections() will
            # clean those up, as well as any inf/nan in v (from failed
            # unprojections)
            implied_Rt10 = \
                implied_Rt10__from_unprojections(q0,
                                                 v[0,...] * distance,
                                                 v[1,...],
                                                 weights,
                                                 atinfinity,
                                                 focus_center, focus_radius)

        q1 = mrcal.project( mrcal.transform_point_Rt(implied_Rt10,
                                                     v[0,...] * distance),
                            lensmodels[1], intrinsics_data[1])
        # shape (len(distance),Nheight,Nwidth,2)
        q1 = nps.atleast_dims(q1, -4)

        diff    = q1 - q0
        difflen = nps.mag(diff)
        difflen = np.min( difflen, axis=-3)
    else:

        # Many models. Look at the stdev
        def get_implied_Rt10( i0, i1,
                              focus_center, focus_radius):
            v0 = v[i0,...]
            v1 = v[i1,...]

            if focus_radius == 0:
                return mrcal.identity_Rt()

            if uncertainties is not None:
                weights = 1.0 / (uncertainties[i0]*uncertainties[i1])
            else:
                weights = None

            return \
                implied_Rt10__from_unprojections(q0, v0*distance, v1,
                                                 weights, atinfinity,
                                                 focus_center, focus_radius)
        def get_reprojections(q0, implied_Rt10,
                              lensmodel, intrinsics_data):
            q1 = mrcal.project(mrcal.transform_point_Rt(implied_Rt10,
                                                        v[0,...]*distance),
                               lensmodel, intrinsics_data)
            # returning shape (len(distance),Nheight,Nwidth,2)
            return nps.atleast_dims(q1, -4)

        implied_Rt10 = nps.cat(*[ get_implied_Rt10(0,i,
                                                   focus_center, focus_radius) \
                                  for i in range(1,len(v))])

        # shape (Ncameras-1,len(distance),Nheight,Nwidth,2)
        grids = nps.cat(*[get_reprojections(q0, implied_Rt10[i-1],
                                            lensmodels[i], intrinsics_data[i]) \
                          for i in range(1,len(v))])

        diff    = None
        difflen = np.sqrt(np.mean( np.min(nps.norm2(grids-q0),
                                          axis=-3),
                                   axis=0))

    return difflen, diff, q0, implied_Rt10


def show_projection_diff(models,
                         gridn_width  = 60,
                         gridn_height = None,

                         observations = False,
                         distance     = None,

                         use_uncertainties= True,
                         focus_center     = None,
                         focus_radius     = -1.,
                         implied_Rt10     = None,

                         vectorfield      = False,
                         vectorscale      = 1.0,
                         extratitle       = None,
                         cbmax            = 4,
                         **kwargs):
    r'''Visualize the difference in projection between N models

SYNOPSIS

    models = ( mrcal.cameramodel('cam0-dance0.cameramodel'),
               mrcal.cameramodel('cam0-dance1.cameramodel') )

    mrcal.show_projection_diff(models)

    # A plot pops up displaying the projection difference between the two models

It is often useful to compare the projection behavior of two camera models. For
instance, one may want to evaluate the quality of a calibration by comparing the
results of two different chessboard dances. Or one may want to evaluate the
stability of the intrinsics in response to mechanical or thermal stresses. This
function makes these comparisons, and produces a visualization of the results.
mrcal.projection_diff() computes the differences, and returns the results
WITHOUT making plots.

In the most common case we're given exactly 2 models to compare. We then show
the projection DIFFERENCE as either a vector field or a heat map. If we're given
more than 2 models, then a vector field isn't possible and we show a heat map of
the STANDARD DEVIATION of all the differences.

The details of how the comparison is computed, and the meaning of the arguments
controlling this, are in the docstring of mrcal.projection_diff().

ARGUMENTS

- models: iterable of mrcal.cameramodel objects we're comparing. Usually there
  will be 2 of these, but more than 2 is possible. The intrinsics are used; the
  extrinsics are NOT.

- gridn_width: optional value, defaulting to 60. How many points along the
  horizontal gridding dimension

- gridn_height: how many points along the vertical gridding dimension. If None,
  we compute an integer gridn_height to maintain a square-ish grid:
  gridn_height/gridn_width ~ imager_height/imager_width

- observations: optional boolean, defaulting to False. If True, we overlay
  calibration-time observations on top of the difference plot. We should then
  see that more data produces more consistent results.

- distance: optional value, defaulting to None. The projection difference varies
  depending on the range to the observed world points, with the queried range
  set in this 'distance' argument. If None (the default) we look out to
  infinity. We can compute the implied-by-the-intrinsics transformation off
  multiple distances if they're given here as an iterable. This is especially
  useful if we have uncertainties, since then we'll emphasize the best-fitting
  distances.

- use_uncertainties: optional boolean, defaulting to True. If True we use the
  whole imager to fit the implied-by-the-intrinsics transformation, using the
  uncertainties to emphasize the confident regions. If False, it is important to
  select the confident region using the focus_center and focus_radius arguments.
  If use_uncertainties is True, but that data isn't available, we report a
  warning, and try to proceed without.

- focus_center: optional array of shape (2,); the imager center by default. Used
  to indicate that the implied-by-the-intrinsics transformation should use only
  those pixels a distance focus_radius from focus_center. This is intended to be
  used if no uncertainties are available, and we need to manually select the
  focus region.

- focus_radius: optional value. If use_uncertainties then the default is LARGE,
  to use the whole imager. Else the default is min(width,height)/6. Used to
  indicate that the implied-by-the-intrinsics transformation should use only
  those pixels a distance focus_radius from focus_center. This is intended to be
  used if no uncertainties are available, and we need to manually select the
  focus region. Pass focus_radius=0 to avoid computing the transformation, and
  to use the identity. This would mean there're no geometric differences, and
  we're comparing the intrinsics only

- implied_Rt10: optional Rt transformation (numpy array of shape (4,3)). If
  given, I use the given value for the implied-by-the-intrinsics transformation
  instead of fitting it. If omitted, I compute the transformation. Exclusive
  with focus_center, focus_radius. Valid only if exactly two models are given.

- vectorfield: optional boolean, defaulting to False. By default we produce a
  heat map of the projection differences. If vectorfield: we produce a vector
  field instead. This is more busy, and is often less clear at first glance, but
  unlike a heat map, this shows the directions of the differences in addition to
  the magnitude. This is only valid if we're given exactly two models to compare

- vectorscale: optional value, defaulting to 1.0. Applicable only if
  vectorfield. The magnitude of the errors displayed in the vector field is
  often very small, and impossible to make out when looking at the whole imager.
  This argument can be used to scale all the displayed vectors to improve
  legibility.

- extratitle: optional string to include in the title of the resulting plot

- cbmax: optional value, defaulting to 4.0. Sets the maximum range of the color
  map

- **kwargs: optional arguments passed verbatim as plot options to gnuplotlib.
  Useful to make hardcopies, etc

RETURNED VALUE

The gnuplotlib plot object. The plot disappears when this object is destroyed
(by the garbage collection, for instance), so do save this returned plot object
into a variable, even if you're not going to be doing anything with this object

    '''

    import gnuplotlib as gp

    if 'title' not in kwargs:
        if implied_Rt10 is not None:
            title_note = "using given extrinsics transform"
        elif focus_radius == 0:
            title_note = "using an identity extrinsics transform"
        else:
            distance_string = "infinity" if distance is None else f"distance={distance}"

            using_uncertainties_string = f"{'' if use_uncertainties else 'not '}using uncertainties"
            title_note = f"computing the extrinsics transform {using_uncertainties_string} from data at {distance_string}"

        # should say something about the focus too, but it's already too long
        # elif focus_radius > 2*(W+H):
        #     where = "extrinsics transform fitted everywhere"
        # else:
        #     where = "extrinsics transform fit looking at {} with radius {}". \
        #         format('the imager center' if focus_center is None else focus_center,
        #                focus_radius)

        title = f"Diff looking at {len(models)} models, {title_note}"
        if extratitle is not None:
            title += ": " + extratitle
        kwargs['title'] = title

    if vectorfield:
        if len(models) > 2:
            raise Exception("I can only plot a vectorfield when looking at exactly 2 models. Instead I have {}". \
                            format(len(models)))
        if not (distance is None or \
                isinstance(distance,float) or \
                len(distance) == 1):
            raise Exception("I don't know how to plot multiple-distance diff with vectorfields")

    # Now do all the actual work
    difflen,diff,q0,implied_Rt10 = projection_diff(models,
                                                   gridn_width, gridn_height,
                                                   distance,
                                                   use_uncertainties,
                                                   focus_center,focus_radius,
                                                   implied_Rt10)
    if vectorfield:

        # The projection_diff() call made sure they're the same for all the
        # models
        W,H=models[0].imagersize()

        plot = gp.gnuplotlib(square=1,
                             _xrange=[0,W],
                             _yrange=[H,0],
                             cbrange=[0,cbmax],
                             **kwargs)

        q0      = nps.clump(q0,      n=len(q0     .shape)-1)
        diff    = nps.clump(diff,    n=len(diff   .shape)-1)
        difflen = nps.clump(difflen, n=len(difflen.shape)  )

        plot_data_args = \
            [ (q0  [:,0], q0  [:,1],
               diff[:,0] * vectorscale, diff[:,1] * vectorscale,
               difflen,
               dict(_with='vectors size screen 0.01,20 fixed filled palette',
                    tuplesize=5)) ]

    else:
        curveoptions = \
            _options_heatmap_with_contours( # update these plot options
                                            kwargs,

                                            cbmax, -0.5,
                                            models[0].imagersize(),
                                            gridn_width, gridn_height)
        plot = gp.gnuplotlib(**kwargs)
        plot_data_args = [ (difflen, curveoptions) ]


    valid_region0 = models[0].valid_intrinsics_region()
    if valid_region0 is not None:
        if vectorfield:
            # 2d plot
            plot_data_args.append( (valid_region0[:,0], valid_region0[:,1],
                                    dict(_with = 'lines lw 3',
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
                             *models[1].intrinsics())
        valid_region1 = mrcal.project( mrcal.transform_point_Rt( mrcal.invert_Rt(implied_Rt10),
                                                                 v1 ),
                                       *models[0].intrinsics() )
        if vectorfield:
            # 2d plot
            plot_data_args.append( (valid_region1[:,0], valid_region1[:,1],
                                    dict(_with = 'lines lw 3',
                                         legend = "valid region of 2nd camera")) )
        else:
            # 3d plot
            plot_data_args.append( (valid_region1[:,0], valid_region1[:,1], valid_region1[:,0]*0,
                                    dict(_with = 'lines lw 3 nocontour',
                                         legend = "valid region of 2nd camera")) )

    if observations:

        # "nocontour" only for 3d plots
        _2d = bool(vectorfield)
        if _2d:
            _with     = 'points'
            tuplesize = 2
        else:
            _with     = 'points nocontour'
            tuplesize = 3

        for i in range(len(models)):

            m = models[i]

            p_cam_calobjects_inliers, p_cam_calobjects_outliers = \
                hypothesis_corner_positions(m.icam_intrinsics(),
                                            **m.optimization_inputs())[1:]
            q_cam_calobjects_inliers = \
                mrcal.project( p_cam_calobjects_inliers, *m.intrinsics() )
            q_cam_calobjects_outliers = \
                mrcal.project( p_cam_calobjects_outliers, *m.intrinsics() )

            if len(q_cam_calobjects_inliers):
                plot_data_args.append( ( q_cam_calobjects_inliers[...,0],
                                         q_cam_calobjects_inliers[...,1] ) +
                                       ( () if _2d else ( np.zeros(q_cam_calobjects_inliers.shape[:-1]), )) +
                                       ( dict( tuplesize = tuplesize,
                                               _with     = _with,
                                               legend    = f'Camera {i} inliers'), ))
            if len(q_cam_calobjects_outliers):
                plot_data_args.append( ( q_cam_calobjects_outliers[...,0],
                                         q_cam_calobjects_outliers[...,1] ) +
                                       ( () if _2d else ( np.zeros(q_cam_calobjects_outliers.shape[:-1]), )) +
                                       ( dict( tuplesize = tuplesize,
                                               _with     = _with,
                                               legend    = f'Camera {i} outliers'), ))

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


# mrcal.shellquote is either pipes.quote or shlex.quote, depending on
# python2/python3
try:
    import pipes
    shellquote = pipes.quote
except:
    # python3 puts this into a different module
    import shlex
    shellquote = shlex.quote


def mapping_file_framenocameraindex(*files_per_camera):
    r'''Parse image filenames to get the frame numbers

SYNOPSIS

    mapping_file_framenocameraindex = \
      mapping_file_framenocameraindex( ('img5-cam2.jpg', 'img6-cam2.jpg'),
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


def chessboard_observations(Nw, Nh, globs=('*',), corners_cache_vnl=None, jobs=1,
                            exclude_images=set(),
                            weighted=True,
                            keep_level=False):
    r'''Compute the chessboard observations and returns them in a usable form

SYNOPSIS

  observations, indices_frame_camera, paths = \
      mrcal.chessboard_observations(10, 10,
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

- observations: an ordered (N,object_height_n,object_width_n,3) array describing
  N board observations where the board has dimensions
  (object_height_n,object_width_n) and each point is an (x,y,weight) pixel
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
            for icam in range(Ncameras):
                g = globs[icam]
                if g[0] != '/':
                    g = '*/' + g
                if fnmatch.fnmatch(os.path.abspath(f), g):
                    files_per_camera[icam].append(f)
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
        for icam in range(len(files_per_camera)):
            N = len(files_per_camera[icam])

            if N < min_num_images:
                raise Exception("Found too few ({}; need at least {}) images containing a calibration pattern in camera {}; glob '{}'". \
                                format(N, min_num_images, icam, globs[icam]))

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
    file_framenocameraindex               = mapping_file_framenocameraindex(*files_per_camera)

    # I create a file list sorted by frame and then camera. So my for(frames)
    # {for(cameras) {}} loop will just end up looking at these files in order
    files_sorted = sorted(mapping_file_corners.keys(), key=lambda f: file_framenocameraindex[f][1])
    files_sorted = sorted(files_sorted,                key=lambda f: file_framenocameraindex[f][0])

    i_observation = 0

    iframe_last = None
    index_frame  = -1
    for f in files_sorted:
        # The frame indices I return are consecutive starting from 0, NOT the
        # original frame numbers
        iframe,icam = file_framenocameraindex[f]
        if iframe_last == None or iframe_last != iframe:
            index_frame += 1
            iframe_last = iframe

        indices_frame_camera = nps.glue(indices_frame_camera,
                                        np.array((index_frame, icam), dtype=np.int32),
                                        axis=-2)
        observations = nps.glue(observations,
                                mapping_file_corners[f],
                                axis=-4)

    return observations, indices_frame_camera, files_sorted


def estimate_monocular_calobject_poses_Rt_tocam( indices_frame_camera,
                                                 observations,
                                                 object_spacing,
                                                 models_or_intrinsics ):
    r"""Estimate camera-referenced poses of the calibration object from monocular views

SYNOPSIS

    print( indices_frame_camera.shape )
    ===>
    (123, 2)

    print( observations.shape )
    ===>
    (123, 3)

    models = [mrcal.cameramodel(f) for f in ("cam0.cameramodel",
                                             "cam1.cameramodel")]

    # Estimated poses of the calibration object from monocular observations
    Rt_camera_frame = \
        mrcal.estimate_monocular_calobject_poses_Rt_tocam( indices_frame_camera,
                                                           observations,
                                                           object_spacing,
                                                           models)

    print( Rt_camera_frame.shape )
    ===>
    (123, 4, 3)

    i_observation = 10
    icam = indices_frame_camera[i_observation,1]

    # The calibration object in its reference coordinate system
    calobject = mrcal.ref_calibration_object(object_width_n,
                                                 object_height_n,
                                                 object_spacing)

    # The estimated calibration object points in the observing camera coordinate
    # system
    pcam = mrcal.transform_point_Rt( Rt_camera_frame[i_observation],
                                     calobject )

    # The pixel observations we would see if the calibration object pose was
    # where it was estimated to be
    q = mrcal.project(pcam, *models[icam].intrinsics())

    # The reprojection error, comparing these hypothesis pixel observations from
    # what we actually observed. We estimated the calibration object pose from
    # the observations, so this should be small
    err = q - observations[i_observation][:2]

    print( np.linalg.norm(err) )
    ===>
    [something small]

mrcal solves camera calibration problems by iteratively optimizing a nonlinear
least squares problem to bring the pixel observation predictions in line with
actual pixel observations. This requires an initial "seed", an initial estimate
of the solution. This function is a part of that computation. Since this is just
an initial estimate that will be refined, the results of this function do not
need to be exact.

We have pixel observations of a known calibration object, and we want to
estimate the pose of this object in the coordinate system of the camera that
produced these observations. This function ingests a number of such
observations, and solves this "PnP problem" separately for each one. The
observations may come from any lens model; everything is reprojected to a
pinhole model first. This function is a wrapper around the solvePnP() openCV
call, which does all the work.

ARGUMENTS

- indices_frame_camera: an array of shape (Nobservations,2) and dtype
  numpy.int32. Each row (iframe,icam) represents an observation of a
  calibration object by camera icam. iframe is not used by this function

- observations: an array of shape
  (Nobservations,object_height_n,object_width_n,3). Each observation corresponds
  to a row in indices_frame_camera, and contains a row of shape (3,) for each
  point in the calibration object. Each row is (x,y,weight) where x,y are the
  observed pixel coordinates. Any point where x<0 or y<0 or weight<0 is ignored.
  This is the only use of the weight in this function.

- object_spacing: the distance between adjacent points in the calibration
  object. A square object is assumed, so the vertical and horizontal distances
  are assumed to be identical. Usually we need the object dimensions in the
  object_height_n,object_width_n arguments, but here we get those from the shape
  of the observations array

- models_or_intrinsics: either

  - a list of mrcal.cameramodel objects from which we use the intrinsics
  - a list of (lensmodel,intrinsics_data) tuples

  These are indexed by icam from indices_frame_camera

RETURNED VALUE

An array of shape (Nobservations,4,3). Each slice is an Rt transformation TO the
camera coordinate system FROM the calibration object coordinate system.

    """

    # I'm given models. I remove the distortion so that I can pass the data
    # on to solvePnP()
    lensmodels_intrinsics_data = [ m.intrinsics() if type(m) is mrcal.cameramodel else m for m in models_or_intrinsics ]
    lensmodels      = [di[0] for di in lensmodels_intrinsics_data]
    intrinsics_data = [di[1] for di in lensmodels_intrinsics_data]

    if not all([mrcal.lensmodel_meta(m)['has_core'] for m in lensmodels]):
        raise Exception("this currently works only with models that have an fxfycxcy core. It might not be required. Take a look at the following code if you want to add support")

    fx = [ i[0] for i in intrinsics_data ]
    fy = [ i[1] for i in intrinsics_data ]
    cx = [ i[2] for i in intrinsics_data ]
    cy = [ i[3] for i in intrinsics_data ]

    Nobservations = indices_frame_camera.shape[0]

    # Reproject all the observations to a pinhole model
    observations = observations.copy()
    for i_observation in range(Nobservations):
        icam = indices_frame_camera[i_observation,1]

        v = mrcal.unproject(observations[i_observation,...,:2],
                            lensmodels[icam], intrinsics_data[icam])
        observations[i_observation,...,:2] = \
            mrcal.project(v, 'LENSMODEL_PINHOLE',
                          intrinsics_data[icam][:4])

    # this wastes memory, but makes it easier to keep track of which data goes
    # with what
    Rt_cf_all = np.zeros( (Nobservations, 4, 3), dtype=float)

    object_height_n,object_width_n = observations.shape[-3:-1]

    # No calobject_warp. Good-enough for the seeding
    full_object = mrcal.ref_calibration_object(object_width_n, object_height_n, object_spacing)

    for i_observation in range(Nobservations):

        icam = indices_frame_camera[i_observation,1]
        camera_matrix = np.array((( fx[icam], 0,        cx[icam]), \
                                  ( 0,        fy[icam], cy[icam]), \
                                  ( 0,        0,          1.)))

        # shape (Nh,Nw,3)
        d = observations[i_observation, ...]

        # shape (Nh,Nw,6); each row is an x,y,weight pixel observation followed
        # by the xyz coord of the point in the calibration object
        d = nps.glue(d, full_object, axis=-1)

        # shape (Nh*Nw,6)
        d = nps.clump( d, n=2)

        # I pick off those rows where the point observation is valid. Result
        # should be (N,6) where N <= object_height_n*object_width_n
        i = (d[..., 0] >= 0) * (d[..., 1] >= 0) * (d[..., 2] >= 0)
        d = d[i,:]

        # copying because cv2.solvePnP() requires contiguous memory apparently
        observations_local = np.array(d[:,:2][..., np.newaxis])
        ref_object         = np.array(d[:,3:][..., np.newaxis])
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
                            observations,
                            object_spacing):
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


    object_height_n,object_width_n = observations.shape[-3:-1]
    Ncameras = np.max(indices_frame_camera[:,1]) + 1

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
        full_object = mrcal.ref_calibration_object(object_width_n,object_height_n,
                                                       object_spacing)

        A = np.array(())
        B = np.array(())

        # I traverse my observation list, and pick out observations from frames
        # that had data from both my cameras
        iframe_last = -1
        d0  = None
        d1  = None
        Rt0 = None
        Rt1 = None
        for i_observation in range(Nobservations):
            iframe_this,icam_this = indices_frame_camera[i_observation, ...]
            if iframe_this != iframe_last:
                d0  = None
                d1  = None
                Rt0 = None
                Rt1 = None
                iframe_last = iframe_this

            # The cameras appear in order. And above I made sure that icam_from >
            # icam_to, so I take advantage of that here
            if icam_this == icam_to:
                if Rt0 is not None:
                    raise Exception("Saw multiple camera{} observations in frame {}".format(icam_this,
                                                                                            iframe_this))
                Rt0 = calobject_poses_local_Rt_cf[i_observation, ...]
                d0  = observations[i_observation, ..., :2]
            elif icam_this == icam_from:
                if Rt0 is None: # have camera1 observation, but not camera0
                    continue

                if Rt1 is not None:
                    raise Exception("Saw multiple camera{} observations in frame {}".format(icam_this,
                                                                                            iframe_this))
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

        return mrcal.align_procrustes_points_Rt01(A, B)


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


def estimate_joint_frame_poses(calobject_Rt_camera_frame,
                               extrinsics_Rt_fromref,
                               indices_frame_camera,
                               object_width_n, object_height_n,
                               object_spacing):

    r'''Estimate world-referenced poses of the calibration object

SYNOPSIS

    print( calobject_Rt_camera_frame.shape )
    ===>
    (123, 4,3)

    print( extrinsics_Rt_fromref.shape )
    ===>
    (2, 4,3)
    # We have 3 cameras. The first one is at the reference coordinate system,
    # the pose estimates of the other two are in this array

    print( indices_frame_camera.shape )
    ===>
    (123, 2)

    frames_rt_toref = \
        mrcal.estimate_joint_frame_poses(calobject_Rt_camera_frame,
                                         extrinsics_Rt_fromref,
                                         indices_frame_camera,
                                         object_width_n, object_height_n,
                                         object_spacing)

    print( frames_rt_toref.shape )
    ===>
    (87, 6)

    # We have 123 observations of the calibration object by ANY camera. 87
    # instances of time when the object was observed. Most of the time it was
    # observed by multiple cameras simultaneously, hence 123 > 87

    i_observation = 10
    iframe,icam = indices_frame_camera[i_observation, :]

    # The calibration object in its reference coordinate system
    calobject = mrcal.ref_calibration_object(object_width_n,
                                                 object_height_n,
                                                 object_spacing)

    # The estimated calibration object points in the reference coordinate
    # system, for this one observation
    pref = mrcal.transform_point_rt( frames_rt_toref[iframe],
                                     calobject )

    # The estimated calibration object points in the camera coord system. Camera
    # 0 is at the reference
    if icam >= 1:
        pcam = mrcal.transform_point_Rt( extrinsics_Rt_fromref[icam-1],
                                         pref )
    else:
        pcam = pref

    # The pixel observations we would see if the pose estimates were correct
    q = mrcal.project(pcam, *models[icam].intrinsics())

    # The reprojection error, comparing these hypothesis pixel observations from
    # what we actually observed. This should be small
    err = q - observations[i_observation][:2]

    print( np.linalg.norm(err) )
    ===>
    [something small]

mrcal solves camera calibration problems by iteratively optimizing a nonlinear
least squares problem to bring the pixel observation predictions in line with
actual pixel observations. This requires an initial "seed", an initial estimate
of the solution. This function is a part of that computation. Since this is just
an initial estimate that will be refined, the results of this function do not
need to be exact.

This function ingests an estimate of the camera poses in respect to each other,
and the estimate of the calibration objects in respect to the observing camera.
Most of the time we have simultaneous calibration object observations from
multiple cameras, so this function consolidates all this information to produce
poses of the calibration object in the reference coordinate system, NOT the
observing-camera coordinate system poses we already have.

By convention, we have a "reference" coordinate system that ties the poses of
all the frames (calibration objects) and the cameras together. And by
convention, this "reference" coordinate system is the coordinate system of
camera 0. Thus the array of camera poses extrinsics_Rt_fromref holds Ncameras-1
transformations: the first camera has an identity transformation, by definition.

This function assumes we're observing a moving object from stationary cameras
(i.e. a vanilla camera calibration problem). The mrcal solver is more general,
and supports moving cameras, hence it uses a more general
indices_frame_camintrinsics_camextrinsics array instead of the
indices_frame_camera array used here.

ARGUMENTS

- calobject_Rt_camera_frame: an array of shape (Nobservations,4,3). Each slice
  is an Rt transformation TO the observing camera coordinate system FROM the
  calibration object coordinate system. This is returned by
  estimate_monocular_calobject_poses_Rt_tocam()

- extrinsics_Rt_fromref: an array of shape (Ncameras-1,4,3). Each slice is an Rt
  transformation TO the camera coordinate system FROM the reference coordinate
  system. By convention camera 0 defines the reference coordinate system, so
  that camera's extrinsics are the identity, by definition, and we don't store
  that data in this array

- indices_frame_camera: an array of shape (Nobservations,2) and dtype
  numpy.int32. Each row (iframe,icam) represents an observation at time
  instant iframe of a calibration object by camera icam

- object_width_n: number of horizontal points in the calibration object grid

- object_height_n: number of vertical points in the calibration object grid

- object_spacing: the distance between adjacent points in the calibration
  object. A square object is assumed, so the vertical and horizontal distances
  are assumed to be identical

RETURNED VALUE

An array of shape (Nframes,6). Each slice represents the pose of the calibration
object at one instant in time: an rt transformation TO the reference coordinate
system FROM the calibration object coordinate system.

    '''

    Rt_ref_cam = mrcal.invert_Rt( extrinsics_Rt_fromref )


    def Rt_ref_frame(i_observation0, i_observation1):
        R'''Given a range of observations corresponding to the same frame, estimate the
        pose of that frame

        '''

        def Rt_ref_frame__single_observation(i_observation):
            r'''Transform from the board coords to the reference coords'''
            iframe,icam = indices_frame_camera[i_observation, ...]

            Rt_cam_frame = calobject_Rt_camera_frame[i_observation, :,:]
            if icam == 0:
                return Rt_cam_frame

            return mrcal.compose_Rt( Rt_ref_cam[icam-1, ...], Rt_cam_frame)


        # frame poses should map FROM the frame coord system TO the ref coord
        # system (camera 0).

        # special case: if there's a single observation, I just use it
        if i_observation1 - i_observation0 == 1:
            return Rt_ref_frame__single_observation(i_observation0)

        # Multiple cameras have observed the object for this frame. I have an
        # estimate of these for each camera. I merge them in a lame way: I
        # average out the positions of each point, and fit the calibration
        # object into the mean point cloud
        #
        # No calobject_warp. Good-enough for the seeding
        obj = mrcal.ref_calibration_object(object_width_n, object_height_n,
                                               object_spacing)

        sum_obj_unproj = obj*0
        for i_observation in range(i_observation0, i_observation1):
            Rt = Rt_ref_frame__single_observation(i_observation)
            sum_obj_unproj += mrcal.transform_point_Rt(Rt, obj)

        mean_obj_ref = sum_obj_unproj / (i_observation1 - i_observation0)

        # Got my point cloud. fit

        # transform both to shape = (N*N, 3)
        obj          = nps.clump(obj,  n=2)
        mean_obj_ref = nps.clump(mean_obj_ref, n=2)
        return mrcal.align_procrustes_points_Rt01( mean_obj_ref, obj )




    frames_rt_toref = np.array(())

    iframe_current          = -1
    i_observation_framestart = -1;

    for i_observation in range(indices_frame_camera.shape[0]):
        iframe,icam = indices_frame_camera[i_observation, ...]

        if iframe != iframe_current:
            if i_observation_framestart >= 0:
                Rt = Rt_ref_frame(i_observation_framestart,
                                  i_observation)
                frames_rt_toref = nps.glue(frames_rt_toref,
                                           mrcal.rt_from_Rt(Rt),
                                           axis=-2)

            i_observation_framestart = i_observation
            iframe_current          = iframe

    if i_observation_framestart >= 0:
        Rt = Rt_ref_frame(i_observation_framestart,
                          indices_frame_camera.shape[0])
        frames_rt_toref = nps.glue(frames_rt_toref,
                                   mrcal.rt_from_Rt(Rt),
                                   axis=-2)

    return frames_rt_toref


def make_seed_pinhole( imagersizes,
                       focal_estimate,
                       indices_frame_camera,
                       observations,
                       object_spacing):
    r'''Compute an optimization seed for a camera calibration

SYNOPSIS

    print( imagersizes.shape )
    ===>
    (4, 2)

    print( indices_frame_camera.shape )
    ===>
    (123, 2)

    print( observations.shape )
    ===>
    (123, 3)

    intrinsics_data,       \
    extrinsics_rt_fromref, \
    frames_rt_toref =      \
        mrcal.make_seed_pinhole(imagersizes          = imagersizes,
                                focal_estimate       = 1500,
                                indices_frame_camera = indices_frame_camera,
                                observations         = observations,
                                object_spacing       = object_spacing)

    ....

    mrcal.optimize(intrinsics_data, extrinsics_rt_fromref, frames_rt_toref,
                   lensmodel = 'LENSMODEL_PINHOLE',
                   ...)

mrcal solves camera calibration problems by iteratively optimizing a nonlinear
least squares problem to bring the pixel observation predictions in line with
actual pixel observations. This requires an initial "seed", an initial estimate
of the solution. This function computes a usable seed, and its results can be
fed to mrcal.optimize(). The output of this function is just an initial estimate
that will be refined, so the results of this function do not need to be exact.

This function assumes we have pinhole lenses, and the returned intrinsics apply
to LENSMODEL_PINHOLE. This is usually good-enough to serve as a seed. The
returned intrinsics can be expanded to whatever lens model we actually want to
use prior to invoking the optimizer.

By convention, we have a "reference" coordinate system that ties the poses of
all the frames (calibration objects) and the cameras together. And by
convention, this "reference" coordinate system is the coordinate system of
camera 0. Thus the array of camera poses extrinsics_rt_fromref holds Ncameras-1
transformations: the first camera has an identity transformation, by definition.

This function assumes we're observing a moving object from stationary cameras
(i.e. a vanilla camera calibration problem). The mrcal solver is more general,
and supports moving cameras, hence it uses a more general
indices_frame_camintrinsics_camextrinsics array instead of the
indices_frame_camera array used here.

See test/test-calibration-basic.py and mrcal-calibrate-cameras for usage
examples.

ARGUMENTS

- imagersizes: an iterable of (imager_width,imager_height) iterables. Defines
  the imager dimensions for each camera we're calibrating. May be an array of
  shape (Ncameras,2) or a tuple of tuples or a mix of the two

- focal_estimate: an initial estimate of the focal length of the cameras, in
  pixels. For the purposes of the initial estimate we use the same focal length
  value for both the x and y focal length of ALL the cameras

- indices_frame_camera: an array of shape (Nobservations,2) and dtype
  numpy.int32. Each row (iframe,icam) represents an observation of a
  calibration object by camera icam. iframe is not used by this function

- observations: an array of shape
  (Nobservations,object_height_n,object_width_n,3). Each observation corresponds
  to a row in indices_frame_camera, and contains a row of shape (3,) for each
  point in the calibration object. Each row is (x,y,weight) where x,y are the
  observed pixel coordinates. Any point where x<0 or y<0 or weight<0 is ignored.
  This is the only use of the weight in this function.

- object_spacing: the distance between adjacent points in the calibration
  object. A square object is assumed, so the vertical and horizontal distances
  are assumed to be identical. Usually we need the object dimensions in the
  object_height_n,object_width_n arguments, but here we get those from the shape
  of the observations array

RETURNED VALUES

We return a tuple:

- intrinsics_data: an array of shape (Ncameras,4). Each slice contains the
  pinhole intrinsics for the given camera. These intrinsics are
  (focal_x,focal_y,centerpixel_x,centerpixel_y), and define LENSMODEL_PINHOLE
  model. mrcal refers to these 4 values as the "intrinsics core". For models
  that have such a core (currently, ALL supported models), the core is the first
  4 parameters of the intrinsics vector. So to calibrate some cameras, call
  make_seed_pinhole(), append to intrinsics_data the proper number of parameters
  to match whatever lens model we're using, and then invoke the optimizer.

- extrinsics_rt_fromref: an array of shape (Ncameras-1,6). Each slice is an rt
  transformation TO the camera coordinate system FROM the reference coordinate
  system. By convention camera 0 defines the reference coordinate system, so
  that camera's extrinsics are the identity, by definition, and we don't store
  that data in this array

- frames_rt_toref: an array of shape (Nframes,6). Each slice represents the pose
  of the calibration object at one instant in time: an rt transformation TO the
  reference coordinate system FROM the calibration object coordinate system.

    '''

    def make_intrinsics_vector(imagersize):
        imager_w,imager_h = imagersize
        return np.array( (focal_estimate, focal_estimate,
                          float(imager_w-1)/2.,
                          float(imager_h-1)/2.))

    intrinsics_data = nps.cat( *[make_intrinsics_vector(imagersize) \
                                 for imagersize in imagersizes] )

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
        mrcal.estimate_monocular_calobject_poses_Rt_tocam( indices_frame_camera,
                                                           observations,
                                                           object_spacing,
                                                           intrinsics)
    # these map FROM the coord system of the calibration object TO the coord
    # system of this camera

    # I now have a rough estimate of calobject poses in the coord system of each
    # camera. One can think of these as two sets of point clouds, each attached
    # to their camera. I can move around the two sets of point clouds to try to
    # match them up, and this will give me an estimate of the relative pose of
    # the two cameras in respect to each other. I need to set up the
    # correspondences, and align_procrustes_points_Rt01() does the rest
    #
    # I get transformations that map points in camera-cami coord system to 0th
    # camera coord system. Rt have dimensions (N-1,4,3)
    camera_poses_Rt_0_cami = \
        _estimate_camera_poses( calobject_poses_local_Rt_cf,
                                indices_frame_camera,
                                observations,
                                object_spacing)

    if len(camera_poses_Rt_0_cami):
        # extrinsics should map FROM the ref coord system TO the coord system of the
        # camera in question. This is backwards from what I have
        extrinsics_Rt_fromref = \
            nps.atleast_dims( mrcal.invert_Rt(camera_poses_Rt_0_cami),
                              -3 )
    else:
        extrinsics_Rt_fromref = np.zeros((0,4,3))

    object_height_n,object_width_n = observations.shape[-3:-1]

    frames_rt_toref = \
        mrcal.estimate_joint_frame_poses(
            calobject_poses_local_Rt_cf,
            extrinsics_Rt_fromref,
            indices_frame_camera,
            object_width_n, object_height_n,
            object_spacing)

    return \
        intrinsics_data, \
        nps.atleast_dims(mrcal.rt_from_Rt(extrinsics_Rt_fromref), -2), \
        frames_rt_toref


def close_contour(c):
    r'''Close a polyline, if it isn't already closed

SYNOPSIS

    print( a.shape )
    ===>
    (5, 2)

    print( a[0] )
    ===>
    [844 204]

    print( a[-1] )
    ===>
    [886 198]

    b = mrcal.close_contour(a)

    print( b.shape )
    ===>
    (6, 2)

    print( b[0] )
    ===>
    [844 204]

    print( b[-2:] )
    ===>
    [[886 198]
     [844 204]]

This function works with polylines represented as arrays of shape (N,2). The
polygon represented by such a polyline is "closed" if its first and last points
sit at the same location. This function ingests a polyline, and returns the
corresponding, closed polygon. If the first and last points of the input match,
the input is returned. Otherwise, the first point is appended to the end, and
this extended polyline is returned.

None is accepted as an empty polygon: we return None.

ARGUMENTS

- c: an array of shape (N,2) representing the polyline to be closed. None is
  accepted as well

RETURNED VALUE

An array of shape (N,2) representing the closed polygon. None is returned if the
input was None

    '''
    if c is None: return None
    if np.linalg.norm( c[0,:] - c[-1,:]) < 1e-6:
        return c
    return nps.glue(c, c[0,:], axis=-2)




    Hq = nps.matmult( nps.dummy(q, -2), nps.transpose(H[..., :,:2]))[..., 0, :] + H[..., 2]
    return Hq[..., :2] / Hq[..., (2,)]


def plotoptions_state_boundaries(**optimization_inputs):
    r'''Return the 'set' plot options for gnuplotlib to show the state boundaries

SYNOPSIS

    import numpy as np
    import gnuplotlib as gp
    import mrcal

    model               = mrcal.cameramodel('xxx.cameramodel')
    optimization_inputs = model.optimization_inputs()

    J = mrcal.optimizer_callback(**optimization_inputs)[2]

    gp.plot( np.sum(np.abs(J.toarray()), axis=-2),
             _set = mrcal.plotoptions_state_boundaries(**optimization_inputs) )

    # a plot pops up showing the magnitude of the effects of each element of the
    # packed state (as seen by the optimizer), with boundaries between the
    # different state variables denoted

When plotting the state vector (or anything relating to it, such as rows of the
Jacobian), it is usually very useful to infer at a glance the meaning of each
part of the plot. This function returns a list of 'set' directives passable to
gnuplotlib that show the boundaries inside the state vector.

ARGUMENTS

**optimization_inputs: a dict() of arguments passable to mrcal.optimize() and
mrcal.optimizer_callback(). These define the full optimization problem, and can
be obtained from the optimization_inputs() method of mrcal.cameramodel

RETURNED VALUE

A list of 'set' directives passable as plot options to gnuplotlib

    '''
    istate0 = []

    try:    istate0.append(mrcal.state_index_intrinsics    (0, **optimization_inputs))
    except: pass
    try:    istate0.append(mrcal.state_index_extrinsics    (0, **optimization_inputs))
    except: pass
    try:    istate0.append(mrcal.state_index_frames        (0, **optimization_inputs))
    except: pass
    try:    istate0.append(mrcal.state_index_points        (0, **optimization_inputs))
    except: pass
    try:    istate0.append(mrcal.state_index_calobject_warp(   **optimization_inputs))
    except: pass

    return [f"arrow nohead from {x},graph 0 to {x},graph 1" for x in istate0]


def plotoptions_measurement_boundaries(**optimization_inputs):
    r'''Return the 'set' plot options for gnuplotlib to show the measurement boundaries

SYNOPSIS

    import numpy as np
    import gnuplotlib as gp
    import mrcal

    model               = mrcal.cameramodel('xxx.cameramodel')
    optimization_inputs = model.optimization_inputs()

    x = mrcal.optimizer_callback(**optimization_inputs)[1]

    gp.plot( np.abs(x),
             _set = mrcal.plotoptions_measurement_boundaries(**optimization_inputs) )

    # a plot pops up showing the magnitude of each measurement, with boundaries
    # between the different measurements denoted

When plotting the measurement vector (or anything relating to it, such as
columns of the Jacobian), it is usually very useful to infer at a glance the
meaning of each part of the plot. This function returns a list of 'set'
directives passable to gnuplotlib that show the boundaries inside the
measurement vector.

ARGUMENTS

**optimization_inputs: a dict() of arguments passable to mrcal.optimize() and
mrcal.optimizer_callback(). These define the full optimization problem, and can
be obtained from the optimization_inputs() method of mrcal.cameramodel

RETURNED VALUE

A list of 'set' directives passable as plot options to gnuplotlib

    '''

    imeas0 = []

    try:    imeas0.append(mrcal.measurement_index_boards        (0, **optimization_inputs))
    except: pass
    try:    imeas0.append(mrcal.measurement_index_points        (0, **optimization_inputs))
    except: pass
    try:    imeas0.append(mrcal.measurement_index_regularization(0, **optimization_inputs))
    except: pass

    return [f"arrow nohead from {x},graph 0 to {x},graph 1" for x in imeas0]


def ingest_packed_state(p_packed,
                        **optimization_inputs):
    r'''Read a given packed state into optimization_inputs

SYNOPSIS

    # A simple gradient check

    model               = mrcal.cameramodel('xxx.cameramodel')
    optimization_inputs = model.optimization_inputs()

    p0,x0,J = mrcal.optimizer_callback(no_factorization = True,
                                       **optimization_inputs)[:3]

    dp = np.random.randn(len(p0)) * 1e-9

    mrcal.ingest_packed_state(p0 + dp,
                              **optimization_inputs)

    x1 = mrcal.optimizer_callback(no_factorization = True,
                                  no_jacobian      = True,
                                  **optimization_inputs)[1]

    dx_observed  = x1 - x0
    dx_predicted = nps.inner(J, dp_packed)

This is the converse of mrcal.optimizer_callback(). One thing
mrcal.optimizer_callback() does is to convert the expanded (intrinsics,
extrinsics, ...) arrays into a 1-dimensional scaled optimization vector
p_packed. mrcal.ingest_packed_state() allows updates to p_packed to be absorbed
back into the (intrinsics, extrinsics, ...) arrays for further evaluation with
mrcal.optimizer_callback() and others.

ARGUMENTS

- p_packed: a numpy array of shape (Nstate,) containing the input packed state

- **optimization_inputs: a dict() of arguments passable to mrcal.optimize() and
  mrcal.optimizer_callback(). The arrays in this dict are updated


RETURNED VALUE

None

    '''

    intrinsics     = optimization_inputs.get("intrinsics")
    extrinsics     = optimization_inputs.get("extrinsics_rt_fromref")
    frames         = optimization_inputs.get("frames_rt_toref")
    points         = optimization_inputs.get("points")
    calobject_warp = optimization_inputs.get("calobject_warp")

    Npoints_fixed  = optimization_inputs.get('Npoints_fixed', 0)

    Nvars_intrinsics     = mrcal.num_states_intrinsics    (**optimization_inputs)
    Nvars_extrinsics     = mrcal.num_states_extrinsics    (**optimization_inputs)
    Nvars_frames         = mrcal.num_states_frames        (**optimization_inputs)
    Nvars_points         = mrcal.num_states_points        (**optimization_inputs)
    Nvars_calobject_warp = mrcal.num_states_calobject_warp(**optimization_inputs)

    Nvars_expected = \
        Nvars_intrinsics + \
        Nvars_extrinsics + \
        Nvars_frames     + \
        Nvars_points     + \
        Nvars_calobject_warp

    # Defaults MUST match those in OPTIMIZER_ARGUMENTS_OPTIONAL in
    # mrcal_pywrap_nonbroadcasted.c. Or better yet, this whole function should
    # come from the C code instead of being reimplemented here in Python
    do_optimize_intrinsics_core        = optimization_inputs.get('do_optimize_intrinsics_core',        True)
    do_optimize_intrinsics_distortions = optimization_inputs.get('do_optimize_intrinsics_distortions', True)
    do_optimize_extrinsics             = optimization_inputs.get('do_optimize_extrinsics',             True)
    do_optimize_frames                 = optimization_inputs.get('do_optimize_frames',                 True)
    do_optimize_calobject_warp         = optimization_inputs.get('do_optimize_calobject_warp',         True)


    if p_packed.ravel().size != Nvars_expected:
        raise Exception(f"Mismatched array size: p_packed.size={p_packed.ravel().size} while the optimization problem expects {Nvars_expected}")

    p = p_packed.copy()
    mrcal.unpack_state(p, **optimization_inputs)

    if do_optimize_intrinsics_core or \
       do_optimize_intrinsics_distortions:

        try:    ivar0 = mrcal.state_index_intrinsics(0, **optimization_inputs)
        except: ivar0 = None
        if ivar0 is not None:
            iunpacked0,iunpacked1 = None,None # everything by default

            lensmodel    = optimization_inputs['lensmodel']
            has_core     = mrcal.lensmodel_meta(lensmodel)['has_core']
            Ncore        = 4 if has_core else 0
            Ndistortions = mrcal.lensmodel_num_params(lensmodel) - Ncore

            if not do_optimize_intrinsics_core:
                iunpacked0 = Ncore
            if not do_optimize_intrinsics_distortions:
                iunpacked1 = -Ndistortions

            intrinsics[:, iunpacked0:iunpacked1].ravel()[:] = \
                p[ ivar0:Nvars_intrinsics ]

    if do_optimize_extrinsics:
        try:    ivar0 = mrcal.state_index_extrinsics(0, **optimization_inputs)
        except: ivar0 = None
        if ivar0 is not None:
            extrinsics.ravel()[:] = p[ivar0:ivar0+Nvars_extrinsics]

    if do_optimize_frames:
        try:    ivar0 = mrcal.state_index_frames(0, **optimization_inputs)
        except: ivar0 = None
        if ivar0 is not None:
            frames.ravel()[:] = p[ivar0:ivar0+Nvars_frames]

    if do_optimize_frames:
        try:    ivar0 = mrcal.state_index_points(0, **optimization_inputs)
        except: ivar0 = None
        if ivar0 is not None:
            points.ravel()[:-Npoints_fixed*3] = p[ivar0:ivar0+Nvars_points]

    if do_optimize_calobject_warp:
        try:    ivar0 = mrcal.state_index_calobject_warp(**optimization_inputs)
        except: ivar0 = None
        if ivar0 is not None:
            calobject_warp.ravel()[:] = p[ivar0:ivar0+Nvars_calobject_warp]

