#!/usr/bin/python3

'''Routines useful in generation and processing of synthetic data

These are very useful in analyzing the behavior or cameras and lenses.

All functions are exported into the mrcal module. So you can call these via
mrcal.synthetic_data.fff() or mrcal.fff(). The latter is preferred.

'''

import numpy as np
import numpysane as nps
import sys
import mrcal

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

        # Layout must match mrcal_calobject_warp_t in mrcal.h
        cx,cy,x2,xy,y2 = calobject_warp

        # Logic must match project() in mrcal.c

        # [0..1]
        xr = xx / (W-1)
        yr = yy / (H-1)

        xr -= cx + 0.5
        yr -= cy + 0.5
        full_object[..., 2] += \
            xr*xr * x2 + \
            xr*yr * xy + \
            yr*yr * y2

    return full_object


def synthesize_board_observations(models,
                                  object_width_n,object_height_n,
                                  object_spacing, calobject_warp,
                                  rt_ref_boardcenter,
                                  rt_ref_boardcenter__noiseradius,
                                  Nframes,

                                  which = 'all_cameras_must_see_full_board'):
    r'''Produce synthetic chessboard observations

SYNOPSIS

    models = [mrcal.cameramodel("0.cameramodel"),
              mrcal.cameramodel("1.cameramodel"),]

    # shapes (Nframes, Ncameras, object_height_n, object_width_n, 2) and
    #        (Nframes, 4, 3)
    q,Rt_ref_boardref = \
        mrcal.synthesize_board_observations( \
          models,

          # board geometry
          10,12,0.1,None,

          # mean board pose and the radius of the added uniform noise
          rt_ref_boardcenter,
          rt_ref_boardcenter__noiseradius,

          # How many frames we want
          100,

          which = 'some_cameras_must_see_half_board')

    # q now contains the synthetic pixel observations, but some of them will be
    # out of view. I construct an (x,y,weight) observations array, as expected
    # by the optimizer, and I set the weight for the out-of-view points to -1 to
    # tell the optimizer to ignore those points


    # Set the weights to 1 initially
    # shape (Nframes, Ncameras, object_height_n, object_width_n, 3)
    observations = nps.glue(q,
                            np.ones( q.shape[:-1] + (1,) ),
                            axis = -1)

    # shape (Ncameras, 1, 1, 2)
    imagersizes = nps.mv( nps.cat(*[ m.imagersize() for m in models ]),
                          -2, -4 )

    observations[ np.any( q              < 0, axis=-1 ), 2 ] = -1.
    observations[ np.any( q-imagersizes >= 0, axis=-1 ), 2 ] = -1.

Given a description of a calibration object and of the cameras observing it,
produces perfect pixel observations of the objects by those cameras. We return a
dense observation array: every corner observation from every chessboard pose
will be reported for every camera. Some of these observations MAY be
out-of-view, depending on the value of the 'which' argument; see description
below. The example above demonstrates how to mark such out-of-bounds
observations as outliers to tell the optimization to ignore these.

The "models" provides the intrinsics and extrinsics.

The calibration objects are nominally have pose rt_ref_boardcenter in the
reference coordinate system, with each pose perturbed uniformly with radius
rt_ref_boardcenter__noiseradius. This is nonstandard since here I'm placing the
board origin at its center instead of the corner (as
mrcal.ref_calibration_object() does). But this is more appropriate to the usage
of this function. The returned Rt_ref_boardref transformation DOES use the
normal corner-referenced board geometry

Returns the point observations and the chessboard poses that produced these
observations.

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
  the docs for ref_calibration_object() for the meaning of the values in this
  array.

- rt_ref_boardcenter: the nominal pose of the calibration object, in the
  reference coordinate system. This is an rt transformation from a
  center-referenced calibration object to the reference coordinate system

- rt_ref_boardcenter__noiseradius: the deviation-from-nominal for the chessboard
  pose for each frame. I add uniform noise to rt_ref_boardcenter, with each
  element sampled independently, with the radius given here.

- Nframes: how many frames of observations to return

- which: a string, defaulting to 'all_cameras_must_see_full_board'. Controls the
  requirements on the visibility of the returned points. Valid values:

  - 'all_cameras_must_see_full_board': We return only those chessboard poses
    that produce observations that are FULLY visible by ALL the cameras.

  - 'some_cameras_must_see_full_board': We return only those chessboard poses
    that produce observations that are FULLY visible by AT LEAST ONE camera.

  - 'all_cameras_must_see_half_board': We return only those chessboard poses
    that produce observations that are AT LEAST HALF visible by ALL the cameras.

  - 'some_cameras_must_see_half_board': We return only those chessboard poses
    that produce observations that are AT LEAST HALF visible by AT LEAST ONE
    camera.

RETURNED VALUES

We return a tuple:

- q: an array of shape (Nframes, Ncameras, object_height, object_width, 2)
  containing the pixel coordinates of the generated observations

- Rt_ref_boardref: an array of shape (Nframes, 4,3) containing the poses of the
  chessboards. This transforms the object returned by ref_calibration_object()
  to the pose that was projected, in the ref coord system

    '''


    # Can visualize results with this script:
    r'''
    r = np.array((30, 0, 0,), dtype=float) * np.pi/180.

    model = mrcal.cameramodel( intrinsics = ('LENSMODEL_PINHOLE',
                                             np.array((1000., 1000., 1000., 1000.,))),
                               imagersize = np.array((2000,2000)) )
    Rt_ref_boardref = \
        mrcal.synthesize_board_observations([model],
                                            5,20,0.1,None,
                                            nps.glue(r, np.array((0,0,3.)), axis=-1),
                                            np.array((0,0,0., 0,0,0)),
                                            1) [1]
    mrcal.show_geometry( models_or_extrinsics_rt_fromref = np.zeros((1,1,6), dtype=float),
                         frames_rt_toref                 = mrcal.rt_from_Rt(Rt_ref_boardref),
                         object_width_n                  = 20,
                         object_height_n                 = 5,
                         object_spacing                  = 0.1,
                         _set = 'xyplane 0',
                         wait = 1 )
    '''

    which_valid = ( 'all_cameras_must_see_full_board',
                    'some_cameras_must_see_full_board',
                    'all_cameras_must_see_half_board',
                    'some_cameras_must_see_half_board', )

    if not which in which_valid:
        raise Exception(f"'which' argument must be one of {which_valid}")

    Ncameras = len(models)

    # I move the board, and keep the cameras stationary.
    #
    # Camera coords: x,y with pixels, z forward
    # Board coords:  x,y in-plane. z forward (i.e. back towards the camera)

    # The center of the board is at the origin (ignoring warping)
    board_center = \
        np.array(( (object_width_n -1)*object_spacing/2.,
                   (object_height_n-1)*object_spacing/2.,
                   0 ))

    # shape: (Nh,Nw,3)
    board_reference = \
        mrcal.ref_calibration_object(object_width_n,object_height_n,
                                     object_spacing,calobject_warp) - \
        board_center

    # Transformation from the board returned by ref_calibration_object() to
    # the one I use here. It's a shift to move the origin to the center of the
    # board
    Rt_boardref_origboardref = mrcal.identity_Rt()
    Rt_boardref_origboardref[3,:] = -board_center

    def get_observation_chunk():
        '''Make Nframes observations, and return them all, even the out-of-view ones'''

        # I compute the full random block in one shot. This is useful for
        # simulations that want to see identical poses when asking for N-1
        # random poses and when asking for the first N-1 of a set of N random
        # poses

        # shape (Nframes,6)
        randomblock = np.random.uniform(low=-1.0, high=1.0, size=(Nframes,6))

        # shape(Nframes,4,3)
        Rt_ref_boardref = \
            mrcal.Rt_from_rt( rt_ref_boardcenter + randomblock * rt_ref_boardcenter__noiseradius )

        # shape = (Nframes, Nh,Nw,3)
        boards_ref = mrcal.transform_point_Rt( # shape (Nframes, 1,1,4,3)
                                               nps.mv(Rt_ref_boardref, 0, -5),

                                               # shape ( Nh,Nw,3)
                                               board_reference )

        # I project full_board. Shape: (Nframes,Ncameras,Nh,Nw,2)
        q = \
          nps.mv( \
            nps.cat( \
              *[ mrcal.project( mrcal.transform_point_Rt(models[i].extrinsics_Rt_fromref(), boards_ref),
                                *models[i].intrinsics()) \
                 for i in range(Ncameras) ]),
                  0,1 )

        return q,Rt_ref_boardref

    def cull_out_of_view(q,Rt_ref_boardref,
                         which):

        # q               has shape (Nframes,Ncameras,Nh,Nw,2)
        # Rt_ref_boardref has shape (Nframes,4,3)

        # I pick only those frames where at least one cameras sees the whole
        # board

        # shape (Nframes,Ncameras,Nh,Nw)
        mask_visible = (q[..., 0] >= 0) * (q[..., 1] >= 0)
        for i in range(Ncameras):
            W,H = models[i].imagersize()
            mask_visible[:,i,...] *= \
                (q[:,i,:,:,0] <= W-1) * \
                (q[:,i,:,:,1] <= H-1)

        # shape (Nframes, Ncameras)
        Nvisible = np.count_nonzero(mask_visible, axis=(-1,-2) )

        Nh,Nw = q.shape[2:4]
        if   which == 'all_cameras_must_see_full_board':
            iframe = np.all(Nvisible == Nh*Nw, axis=-1)
        elif which == 'some_cameras_must_see_full_board':
            iframe = np.any(Nvisible == Nh*Nw, axis=-1)
        elif which == 'all_cameras_must_see_half_board':
            iframe = np.all(Nvisible > Nh*Nw//2, axis=-1)
        elif which == 'some_cameras_must_see_half_board':
            iframe = np.any(Nvisible > Nh*Nw//2, axis=-1)
        else:
            raise Exception("Unknown 'which' argument. This is a bug. I checked for the valid options at the top of this function")

        # q               has shape (Nframes_inview,Ncameras,Nh*Nw,2)
        # Rt_ref_boardref has shape (Nframes_inview,4,3)
        return q[iframe, ...], Rt_ref_boardref[iframe, ...]

    # shape (Nframes_sofar,Ncameras,Nh,Nw,2)
    q = np.zeros((0,
                  Ncameras,
                  object_height_n,object_width_n,
                  2),
                 dtype=float)
    # shape (Nframes_sofar,4,3)
    Rt_ref_boardref = np.zeros((0,4,3), dtype=float)

    # I keep creating data, until I get Nframes-worth of in-view observations
    while True:
        q_here, Rt_ref_boardref_here = get_observation_chunk()

        q_here, Rt_ref_boardref_here = \
            cull_out_of_view(q_here, Rt_ref_boardref_here,
                             which)

        q = nps.glue(q, q_here, axis=-5)
        Rt_ref_boardref = nps.glue(Rt_ref_boardref, Rt_ref_boardref_here, axis=-3)
        if q.shape[0] >= Nframes:
            q               = q               [:Nframes,...]
            Rt_ref_boardref = Rt_ref_boardref[:Nframes,...]
            break

    return q, mrcal.compose_Rt(Rt_ref_boardref, Rt_boardref_origboardref)


