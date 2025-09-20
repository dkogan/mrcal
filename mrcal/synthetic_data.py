#!/usr/bin/python3

# Copyright (c) 2017-2023 California Institute of Technology ("Caltech"). U.S.
# Government sponsorship acknowledged. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

'''Routines useful in generation and processing of synthetic data

These are very useful in analyzing the behavior or cameras and lenses.

All functions are exported into the mrcal module. So you can call these via
mrcal.synthetic_data.fff() or mrcal.fff(). The latter is preferred.

'''

import numpy as np
import numpysane as nps
import sys
import mrcal

def ref_calibration_object(W              = None,
                           H              = None,
                           object_spacing = None,
                           *,
                           optimization_inputs = None,

                           calobject_warp      = None,
                           x_corner0           = 0,
                           x_corner1           = None,
                           Nx                  = None,
                           y_corner0           = 0,
                           y_corner1           = None,
                           Ny                  = None):
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
supported, possibly with some deformation (i.e. what the internal mrcal solver
supports). Each row of the output is an (x,y,z) point. The origin is at the
corner of the grid, so ref_calibration_object(...)[0,0,:] is np.array((0,0,0)).
The grid spans x and y, with z representing the depth: z=0 for a flat
calibration object.

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
        4*k*xr*(1 - xr)

By default we return the coordinates of the chessboard CORNERS only, but this
function can return the position of ANY point on the chessboard. This can be
controlled by passing the x_corner0,x_corner1,Nx arguments (and/or their y-axis
versions). This selects the grid of points we return, in chessboard-corner
coordinates (0 is the first corner, 1 is the second corner, etc). We use
np.linspace(x_corner0, x_corner1, Nx). By default we have

- x_corner0 = 0
- x_corner1 = W-1
- Nx        = W

So we only return the coordinates of the corners by default. The points returned
along the y axis work similarly, using their variables.

If optimization_inputs is given, we get H,W,object_spacing and calobject_warp
from the inputs. In this case, (H,W,object_spacing) must all be None. Otherwise
all of (H,W,object_spacing) must be given. Thus it's possible to call this
function like this:

    model = mrcal.cameramodel('calibration.cameramodel')
    obj = mrcal.ref_calibration_object(optimization_inputs =
                                       optimization_inputs)

ARGUMENTS

- W: how many chessboard corners we have in the horizontal direction

- H: how many chessboard corners we have in the vertical direction

- object_spacing: the distance between adjacent points in the calibration
  object. If a scalar is given, a square object is assumed, and the vertical and
  horizontal distances are assumed to be identical. An array of shape (..., 2)
  can be given: the last dimension is (spacing_h, spacing_w), and the preceding
  dimensions are used for broadcasting

- calobject_warp: optional array of shape (2,) defaults to None. Describes the
  warping of the calibration object. If None, the object is flat. If an array is
  given, the values describe the maximum additive deflection along the x and y
  axes. Extended array can be given for broadcasting

- optimization_inputs: the input from a calibrated model. Usually the output of
  mrcal.cameramodel.optimization_inputs() call. If given,
  (H,W,object_spacing,calobject_warp) are all read from these inputs, and must
  not be given separately.

- x_corner0: optional value, defaulting to 0. Selects the first point in the
  linear horizontal grid we're returning. This indexes the chessboard corners,
  and we start with the first corner by default

- x_corner1: optional value, defaulting to W-1. Selects the last point in the
  linear horizontal grid we're returning. This indexes the chessboard corners,
  and we end with the last corner by default

- Nx: optional value, defaulting to W. Selects the number of points we return in
  the horizontal direction, between x_corner0 and x_corner1 inclusive.

- y_corner0,y_corner1,Ny: same as x_corner0,x_corner1,Nx but acting in the
  vertical direction

This function supports broadcasting across object_spacing and calobject_warp

RETURNED VALUES

The calibration object geometry in a (..., Ny,Nx,3) array, with the leading
dimensions set by the broadcasting rules. Usually Ny = H and Nx = W

    '''
    Noptions_base = 0
    options_base = ('W', 'H', 'object_spacing')
    for o in options_base:
        if locals()[o] is not None: Noptions_base += 1
    if not (Noptions_base == 0 or \
            Noptions_base == len(options_base)):
        raise Exception(f"Options '{options_base}': ALL must be given, or NONE must be given")
    if Noptions_base > 0 and optimization_inputs is not None:
        raise Exception(f"Options '{options_base}' and 'optimization_inputs' cannot both be given")
    if Noptions_base == 0 and optimization_inputs is None:
        raise Exception(f"One of options '{options_base}' and 'optimization_inputs' MUST be given")

    if optimization_inputs is not None:
        H,W = optimization_inputs['observations_board'].shape[-3:-1]

        object_spacing = optimization_inputs['calibration_object_spacing']
        calobject_warp = optimization_inputs['calobject_warp']


    if Nx        is None: Nx        = W
    if Ny        is None: Ny        = H
    if x_corner1 is None: x_corner1 = W-1
    if y_corner1 is None: y_corner1 = H-1

    # shape (Ny,Nx)
    xx,yy = np.meshgrid( np.linspace(x_corner0, x_corner1, Nx),
                         np.linspace(y_corner0, y_corner1, Ny))

    # shape (Ny,Nx,3)
    full_object = nps.glue(nps.mv( nps.cat(xx,yy), 0, -1),
                           np.zeros(xx.shape + (1,)),
                           axis=-1)

    # object_spacing has shape (..., 2)
    object_spacing = np.array(object_spacing)
    if object_spacing.ndim == 0:
        object_spacing = np.array((1,1))*object_spacing
    object_spacing = nps.dummy(object_spacing, -2,-2)
    # object_spacing now has shape (..., 1,1,2)

    if object_spacing.ndim > 3:
        # extend full_object to the output shape I want
        full_object = full_object * np.ones( object_spacing.shape[:-3] + (1,1,1) )
    full_object[..., :2] *= object_spacing

    if calobject_warp is not None:
        xr = xx / (W-1)
        yr = yy / (H-1)
        dx = 4. * xr * (1. - xr)
        dy = 4. * yr * (1. - yr)

        # To allow broadcasting over calobject_warp
        if calobject_warp.ndim > 1:
            # shape (..., 1,1,2)
            calobject_warp = nps.dummy(calobject_warp, -2,-2)
            # extend full_object to the output shape I want
            full_object = full_object * np.ones( calobject_warp.shape[:-3] + (1,1,1) )
        full_object[..., 2] += calobject_warp[...,0] * dx
        full_object[..., 2] += calobject_warp[...,1] * dy

    return full_object


def synthesize_board_observations(models,
                                  *,
                                  object_width_n,
                                  object_height_n,
                                  object_spacing,
                                  calobject_warp,
                                  rt_ref_boardcenter,
                                  rt_ref_boardcenter__noiseradius,
                                  Nframes,
                                  max_oblique_angle_deg = None,
                                  pcamera_nominal_ref   = np.array((0,0,0), dtype=float),

                                  which = 'all-cameras-must-see-full-board'):
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
          object_width_n  = 10,
          object_height_n = 12,
          object_spacing  = 0.1,
          calobject_warp  = None,

          # mean board pose and the radius of the added uniform noise
          rt_ref_boardcenter              = rt_ref_boardcenter,
          rt_ref_boardcenter__noiseradius = rt_ref_boardcenter__noiseradius,

          # How many frames we want
          Nframes = 100,

          which = 'some-cameras-must-see-half-board')

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

- which: a string, defaulting to 'all-cameras-must-see-full-board'. Controls the
  requirements on the visibility of the returned points. Valid values:

  - 'all-cameras-must-see-full-board': We return only those chessboard poses
    that produce observations that are FULLY visible by ALL the cameras.

  - 'some-cameras-must-see-full-board': We return only those chessboard poses
    that produce observations that are FULLY visible by AT LEAST ONE camera.

  - 'all-cameras-must-see-half-board': We return only those chessboard poses
    that produce observations that are AT LEAST HALF visible by ALL the cameras.

  - 'some-cameras-must-see-half-board': We return only those chessboard poses
    that produce observations that are AT LEAST HALF visible by AT LEAST ONE
    camera.

- max_oblique_angle_deg: optional value, defaulting to None. If non-None, we
  only return observations where the board normal is within this angle of the
  vector to the nominal camera (at pcamera_nominal_ref). This ensures that the
  boards all "face" the camera to a certain degree

- pcamera_nominal_ref: optional vector, defaulting to (0,0,0). Used in
  conjunction with max_oblique_angle_deg to make sure the observation angle
  isn't too oblique

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
                                            object_width_n                  = 5,
                                            object_height_n                 = 20,
                                            object_spacing                  = 0.1,
                                            calobject_warp                  = None,
                                            rt_ref_boardcenter              = nps.glue(r, np.array((0,0,3.)), axis=-1),
                                            rt_ref_boardcenter__noiseradius = np.array((0,0,0., 0,0,0)),
                                            Nframes                         = 1) [1]
    mrcal.show_geometry( models_or_rt_cam_ref = np.zeros((1,1,6), dtype=float),
                         rt_ref_frame         = mrcal.rt_from_Rt(Rt_ref_boardref),
                         object_width_n       = 20,
                         object_height_n      = 5,
                         object_spacing       = 0.1,
                         _set = 'xyplane 0',
                         wait = 1 )
    '''

    which_valid = ( 'all-cameras-must-see-full-board',
                    'some-cameras-must-see-full-board',
                    'all-cameras-must-see-half-board',
                    'some-cameras-must-see-half-board', )

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
                                     object_spacing,
                                     calobject_warp = calobject_warp) - \
        board_center

    # Transformation from the board returned by ref_calibration_object() to
    # the one I use here. It's a shift to move the origin to the center of the
    # board
    Rt_boardref_origboardref = mrcal.identity_Rt()
    Rt_boardref_origboardref[3,:] = -board_center

    if max_oblique_angle_deg is not None:
        max_cos_oblique_angle = np.cos(max_oblique_angle_deg * np.pi/180.)
    else:
        max_cos_oblique_angle = None


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
              *[ mrcal.project( mrcal.transform_point_Rt(models[i].Rt_cam_ref(), boards_ref),
                                *models[i].intrinsics()) \
                 for i in range(Ncameras) ]),
                  0,1 )

        return q,Rt_ref_boardref

    def cull(q,Rt_ref_boardref,
             which):

        # q               has shape (Nframes,Ncameras,Nh,Nw,2)
        # Rt_ref_boardref has shape (Nframes,4,3)

        ######## Throw out extreme oblique views
        if max_cos_oblique_angle is not None:
            nref_position = Rt_ref_boardref[...,3,:] - pcamera_nominal_ref
            nref_position /= nps.dummy(nps.mag(nref_position), -1)
            nref_orientation = Rt_ref_boardref[...,:3,2]
            costh = np.abs(nps.inner(nref_position, nref_orientation))
            i = costh > max_cos_oblique_angle

            q               = q[i]
            Rt_ref_boardref = Rt_ref_boardref[i]


        ######## I pick only those frames where at least one cameras sees the
        ######## whole board

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
        if   which == 'all-cameras-must-see-full-board':
            iframe = np.all(Nvisible == Nh*Nw, axis=-1)
        elif which == 'some-cameras-must-see-full-board':
            iframe = np.any(Nvisible == Nh*Nw, axis=-1)
        elif which == 'all-cameras-must-see-half-board':
            iframe = np.all(Nvisible > Nh*Nw//2, axis=-1)
        elif which == 'some-cameras-must-see-half-board':
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
            cull(q_here, Rt_ref_boardref_here,
                 which)

        q = nps.glue(q, q_here, axis=-5)
        Rt_ref_boardref = nps.glue(Rt_ref_boardref, Rt_ref_boardref_here, axis=-3)
        if q.shape[0] >= Nframes:
            q               = q               [:Nframes,...]
            Rt_ref_boardref = Rt_ref_boardref[:Nframes,...]
            break

    return q, mrcal.compose_Rt(Rt_ref_boardref, Rt_boardref_origboardref)


def _noisy_observation_vectors_for_triangulation(p,
                                                 Rt01,
                                                 intrinsics0, intrinsics1,
                                                 Nsamples, sigma):

    # p has shape (...,3)

    # shape (..., 2)
    q0 = mrcal.project( p,
                        *intrinsics0 )
    q1 = mrcal.project( mrcal.transform_point_Rt( mrcal.invert_Rt(Rt01), p),
                        *intrinsics1 )

    # shape (..., 1,2). Each has x,y
    q0 = nps.dummy(q0,-2)
    q1 = nps.dummy(q1,-2)

    q_noise = np.random.randn(*p.shape[:-1], Nsamples,2,2) * sigma
    # shape (..., Nsamples,2). Each has x,y
    q0_noise = q_noise[...,:,0,:]
    q1_noise = q_noise[...,:,1,:]

    q0_noisy = q0 + q0_noise
    q1_noisy = q1 + q1_noise

    # shape (..., Nsamples, 3)
    v0local_noisy = mrcal.unproject( q0_noisy, *intrinsics0 )
    v1local_noisy = mrcal.unproject( q1_noisy, *intrinsics1 )
    v0_noisy      = v0local_noisy
    v1_noisy      = mrcal.rotate_point_R(Rt01[:3,:], v1local_noisy)

    # All have shape (..., Nsamples,3)
    return \
        v0local_noisy, v1local_noisy, v0_noisy,v1_noisy, \
        q0,q1, q0_noisy, q1_noisy


def make_perfect_observations(optimization_inputs,
                              *,
                              observed_pixel_uncertainty = None):

    r'''Write perfect observations with perfect noise into the optimization_inputs

SYNOPSIS

    model = mrcal.cameramodel("0.cameramodel")
    optimization_inputs = model.optimization_inputs()

    optimization_inputs['calobject_warp'] = np.array((1e-3, -1e-3))
    mrcal.make_perfect_observations(optimization_inputs)

    # We now have perfect data assuming a slightly WARPED chessboard. Let's use
    # this data to compute a calibration assuming a FLAT chessboard
    optimization_inputs['calobject_warp'] *= 0.
    optimization_inputs['do_optimize_calobject_warp'] = False

    mrcal.optimize(**optimization_inputs)

    model = mrcal.cameramodel(optimization_inputs = optimization_inputs,
                              icam_intrinsics     = model.icam_intrinsics())
    model.write("reoptimized.cameramodel")

    # We can now look at the residuals and diffs to see how much a small
    # chessboard deformation affects our results

Tracking down all the sources of error in real-world models computed by mrcal is
challenging: the models never fit perfectly, and the noise never follows the
assumed distribution exactly. It is thus really useful to be able to run
idealized experiments where both the models and the noise are perfect. We can
then vary only one variable to judge its effects. Since everything else is
perfect, we can be sure that any imperfections in the results are due only to
the variable we tweaked. In the sample above we evaluated the effect of a small
chessboard deformation.

This function ingests optimization_inputs from a completed calibration. It then
assumes that all the geometry and intrinsics are perfect, and sets the
observations to projections of that perfect geometry. If requested, perfect
gaussian noise is then added to the observations.

THIS FUNCTION MODIFIES THE INPUT OPTIMIZATION_INPUTS

ARGUMENTS

- optimization_inputs: the input from a calibrated model. Usually the output of
  mrcal.cameramodel.optimization_inputs() call. The output is written into
  optimization_inputs['observations_board'] and
  optimization_inputs['observations_point']

- observed_pixel_uncertainty: optional standard deviation of the noise to apply.
  By default the noise applied has same variance as the noise in the input
  optimization_inputs. If we want to omit the noise, pass
  observed_pixel_uncertainty = 0

RETURNED VALUES

None

    '''

    import mrcal.model_analysis

    x = mrcal.optimizer_callback(**optimization_inputs,
                                 no_jacobian      = True,
                                 no_factorization = True)[1]

    if observed_pixel_uncertainty is None:
        observed_pixel_uncertainty = \
            mrcal.model_analysis._observed_pixel_uncertainty_from_inputs(optimization_inputs,
                                                                         x = x)

    if 'indices_frame_camintrinsics_camextrinsics' in optimization_inputs and \
       optimization_inputs['indices_frame_camintrinsics_camextrinsics'] is not None and \
       optimization_inputs['indices_frame_camintrinsics_camextrinsics'].size:

        # shape (Nobservations, Nheight, Nwidth, 3)
        pcam = mrcal.hypothesis_board_corner_positions(**optimization_inputs)[0]
        i_intrinsics = optimization_inputs['indices_frame_camintrinsics_camextrinsics'][:,1]
        # shape (Nobservations,1,1,Nintrinsics)
        intrinsics = nps.mv(optimization_inputs['intrinsics'][i_intrinsics],-2,-4)
        optimization_inputs['observations_board'][...,:2] = \
            mrcal.project( pcam,
                           optimization_inputs['lensmodel'],
                           intrinsics )

    if 'indices_point_camintrinsics_camextrinsics' in optimization_inputs and \
       optimization_inputs['indices_point_camintrinsics_camextrinsics'] is not None and \
       optimization_inputs['indices_point_camintrinsics_camextrinsics'].size:

        indices_point_camintrinsics_camextrinsics = \
            optimization_inputs['indices_point_camintrinsics_camextrinsics']

        # shape (Nobservations,3)
        pref = optimization_inputs['points'][ indices_point_camintrinsics_camextrinsics[:,0] ]

        # shape (Nobservations,4,3)
        Rt_cam_ref = \
            nps.glue( mrcal.identity_Rt(),
                      mrcal.Rt_from_rt(optimization_inputs['rt_cam_ref']),
                      axis = -3 ) \
            [ indices_point_camintrinsics_camextrinsics[:,2]+1 ]

        # shape (Nobservations,3)
        pcam = mrcal.transform_point_Rt(Rt_cam_ref, pref)

        # shape (Nobservations,Nintrinsics)
        intrinsics = optimization_inputs['intrinsics'][ indices_point_camintrinsics_camextrinsics[:,1] ]
        optimization_inputs['observations_point'][...,:2] = \
            mrcal.project( pcam,
                           optimization_inputs['lensmodel'],
                           intrinsics )

    ########### The perfect observations have been written. Make sure we get
    ########### perfect residuals
    # I don't actually do that here, and rely on the tests to make sure it works properly
    if False:
        x = mrcal.optimizer_callback(**optimization_inputs,
                                     no_jacobian      = True,
                                     no_factorization = True)[1]

        Nmeas = mrcal.num_measurements_boards(**optimization_inputs)
        if Nmeas > 0:
            i_meas0 = mrcal.measurement_index_boards(0, **optimization_inputs)
            err = nps.norm2(x[i_meas0:i_meas0+Nmeas])
            if err > 1e-16:
                raise Exception("Perfect observations produced nonzero error for boards. This is a bug")

        Nmeas = mrcal.num_measurements_points(**optimization_inputs)
        if Nmeas > 0:
            i_meas0 = mrcal.measurement_index_points(0, **optimization_inputs)
            err = nps.norm2(x[i_meas0:i_meas0+Nmeas])
            if err > 1e-16:
                raise Exception("Perfect observations produced nonzero error for points. This is a bug")


    ########### I have perfect data. Now add perfect noise
    if observed_pixel_uncertainty == 0:
        return

    for what in ('observations_board','observations_point'):

        if what in optimization_inputs and \
           optimization_inputs[what] is not None and \
           optimization_inputs[what].size:

            noise_nominal = \
                observed_pixel_uncertainty * \
                np.random.randn(*optimization_inputs[what][...,:2].shape)

            weight = nps.dummy( optimization_inputs[what][...,2],
                                axis = -1 )
            weight[ weight<=0 ] = 1. # to avoid dividing by 0

            optimization_inputs[what][...,:2] += \
                noise_nominal / weight
