#!/usr/bin/python3

'''Routines to (un)project points using any camera model

Most of these are Python wrappers around the written-in-C Python extension
module mrcal._mrcal_npsp. Most of the time you want to use this module
instead of touching mrcal._mrcal_npsp directly.

All functions are exported into the mrcal module. So you can call these via
mrcal.projections.fff() or mrcal.fff(). The latter is preferred.

'''

import numpy as np
import numpysane as nps
import sys
import scipy.optimize

import mrcal


def project(v, lensmodel, intrinsics_data,
            get_gradients = False,
            out           = None):
    r'''Projects a set of 3D camera-frame points to the imager

SYNOPSIS

    points = mrcal.project( # (...,3) array of 3d points we're projecting
                            v,

                            lensmodel, intrinsics_data)

    # points is a (...,2) array of pixel coordinates

Given a shape-(...,3) array of points in the camera frame (x,y aligned with the
imager coords, z 'forward') and an intrinsic model, this function computes the
projection, optionally with gradients.

Projecting out-of-bounds points (beyond the field of view) returns undefined
values. Generally things remain continuous even as we move off the imager
domain. Pinhole-like projections will work normally if projecting a point behind
the camera. Splined projections clamp to the nearest spline segment: the
projection will fly off to infinity quickly since we're extrapolating a
polynomial, but the function will remain continuous.

Broadcasting is fully supported across v and intrinsics_data

ARGUMENTS

- points: array of dims (...,3); the points we're projecting

- lensmodel: a string such as

  LENSMODEL_PINHOLE
  LENSMODEL_OPENCV4
  LENSMODEL_CAHVOR
  LENSMODEL_SPLINED_STEREOGRAPHIC_order=3_Nx=16_Ny=12_fov_x_deg=100

- intrinsics: array of dims (Nintrinsics):

    (focal_x, focal_y, center_pixel_x, center_pixel_y, distortion0, distortion1,
    ...)

  The focal lengths are given in pixels.

- get_gradients: optional boolean that defaults to False. Whether we should
  compute and report the gradients. This affects what we return

- out: optional argument specifying the destination. By default, new numpy
  array(s) are created and returned. To write the results into existing arrays,
  specify them with the 'out' kwarg. If get_gradients: 'out' is the one numpy
  array we will write into. Else: 'out' is a tuple of all the output numpy
  arrays. If 'out' is given, we return the same arrays passed in. This is the
  standard behavior provided by numpysane_pywrap.

RETURNED VALUE

if not get_gradients:

  we return an (...,2) array of projected pixel coordinates

if get_gradients: we return a tuple:

  - (...,2) array of projected pixel coordinates
  - (...,2,3) array of the gradients of the pixel coordinates in respect to
    the input 3D point positions
  - (...,2,Nintrinsics) array of the gradients of the pixel coordinates in
    respect to the intrinsics

The unprojected observation vector of shape (..., 3).

    '''

    # Internal function must have a different argument order so
    # that all the broadcasting stuff is in the leading arguments
    if not get_gradients:
        return mrcal._mrcal_npsp._project(v, intrinsics_data, lensmodel=lensmodel, out=out)
    return mrcal._mrcal_npsp._project_withgrad(v, intrinsics_data, lensmodel=lensmodel, out=out)


def unproject(q, lensmodel, intrinsics_data,
              normalize = False,
              out       = None):
    r'''Unprojects pixel coordinates to observation vectors

SYNOPSIS

    v = mrcal.unproject( # (...,2) array of pixel observations
                         q,
                         lensmodel, intrinsics_data )

Maps a set of 2D imager points q to a 3d vector in camera coordinates that
produced these pixel observations. The 3d vector is unique only up-to-length,
and the returned vectors aren't normalized by default. If we want them to be
normalized, pass normalize=True.

This is the "reverse" direction, so an iterative nonlinear optimization is
performed internally to compute this result. This is much slower than
mrcal_project. For OpenCV distortions specifically, OpenCV has
cvUndistortPoints() (and cv2.undistortPoints()), but these are inaccurate:
https://github.com/opencv/opencv/issues/8811

Broadcasting is fully supported across q and intrinsics_data

ARGUMENTS

- q: array of dims (...,2); the pixel coordinates we're unprojecting

- lensmodel: a string such as

  LENSMODEL_PINHOLE
  LENSMODEL_OPENCV4
  LENSMODEL_CAHVOR
  LENSMODEL_SPLINED_STEREOGRAPHIC_order=3_Nx=16_Ny=12_fov_x_deg=100

- intrinsics_data: array of dims (Nintrinsics):

    (focal_x, focal_y, center_pixel_x, center_pixel_y, distortion0, distortion1,
    ...)

  The focal lengths are given in pixels.

- normalize: optional boolean defaults to False. If True: normalize the output
  vectors

- out: optional argument specifying the destination. By default, new numpy
  array(s) are created and returned. To write the results into existing arrays,
  specify them with the 'out' kwarg. If get_gradients: 'out' is the one numpy
  array we will write into. Else: 'out' is a tuple of all the output numpy
  arrays. If 'out' is given, we return the same arrays passed in. This is the
  standard behavior provided by numpysane_pywrap.

RETURNED VALUE

The unprojected observation vector of shape (..., 3). These are NOT normalized
by default. To get normalized vectors, pass normalize=True

    '''

    if lensmodel != 'LENSMODEL_CAHVORE':
        # Main path. Internal function must have a different argument order so
        # that all the broadcasting stuff is in the leading arguments
        v = mrcal._mrcal_npsp._unproject(q, intrinsics_data, lensmodel=lensmodel, out=out)
        if normalize:
            v /= nps.dummy(nps.mag(v), -1)
        return v

    # CAHVORE. This is a reimplementation of the C code. It's barely maintained,
    # and here for legacy compatibility only

    if q is None: return q
    if q.size == 0:
        s = q.shape
        return np.zeros(s[:-1] + (3,))

    if out is not None:
        raise Exception("unproject(..., out) is unsupported if out is not None and lensmodel == 'LENSMODEL_CAHVORE'")

    fxy = intrinsics_data[ :2]
    cxy = intrinsics_data[2:4]

    # undistort the q, by running an optimizer

    # I optimize each point separately because the internal optimization
    # algorithm doesn't know that each point is independent, so if I optimized
    # it all together, it would solve a dense linear system whose size is linear
    # in Npoints. The computation time thus would be much slower than
    # linear(Npoints)
    @nps.broadcast_define( ((2,),), )
    def undistort_this(q0):

        def cost_no_gradients(vxy, *args, **kwargs):
            '''Optimization functions'''
            return \
                mrcal.project(np.array((vxy[0],vxy[1],1.)), lensmodel, intrinsics_data) - \
                q0

        # seed assuming distortions aren't there
        vxy_seed = (q0 - cxy) / fxy

        # no gradients available
        result = scipy.optimize.least_squares(cost_no_gradients, vxy_seed,
                                              '3-point')

        vxy = result.x

        # This needs to be precise; if it isn't, I barf. Shouldn't happen
        # very often
        if np.sqrt(result.cost/2.) > 1e-3:
            if not unproject.__dict__.get('already_complained'):
                sys.stderr.write("WARNING: unproject() wasn't able to precisely compute some points. Returning nan for those. Will complain just once\n")
                unproject.already_complained = True
            return np.array((np.nan,np.nan))
        return vxy

    vxy = undistort_this(q)

    # I append a 1. shape = (..., 3)
    v = nps.glue(vxy, np.ones( vxy.shape[:-1] + (1,) ), axis=-1)
    if normalize:
        v /= nps.dummy(nps.mag(v), -1)
    return v

