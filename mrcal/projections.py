#!/usr/bin/python3


import numpy as np
import numpysane as nps
import sys
import re
import cv2
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
        return mrcal._mrcal_broadcasted._project(v, intrinsics_data, lensmodel=lensmodel, out=out)
    return mrcal._mrcal_broadcasted._project_withgrad(v, intrinsics_data, lensmodel=lensmodel, out=out)


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
        v = mrcal._mrcal_broadcasted._unproject(q, intrinsics_data, lensmodel=lensmodel, out=out)
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


def scale_focal__best_pinhole_fit(model, fit):
    r'''Compute the optimal focal-length scale for reprojection to a pinhole lens

SYNOPSIS

    model = mrcal.cameramodel('from.cameramodel')

    lensmodel,intrinsics_data = model.intrinsics()

    scale_focal = mrcal.scale_focal__best_pinhole_fit(model,
                                                      'centers-horizontal')

    intrinsics_data[:2] *= scale_focal

    model_pinhole = \
        mrcal.cameramodel(intrinsics = ('LENSMODEL_PINHOLE',
                                        intrinsics_data[:4]),
                          imagersize = model.imagersize(),
                          extrinsics_rt_fromref = model.extrinsics_rt_fromref() )

Many algorithms work with images assumed to have been captured with a pinhole
camera, even though real-world lenses never fit a pinhole model. mrcal provides
several functions to remap images captured with non-pinhole lenses into images
of the same scene as if they were observed by a pinhole lens. When doing this,
we're free to choose all of the parameters of this pinhole lens model, and this
function allows us to pick the best scaling on the focal-length parameter.

The focal length parameters serve as a "zoom" factor: changing these parameters
can increase the resolution of the center of the image, at the expense of
cutting off the edges. This function computes the best focal-length scaling, for
several possible meanings of "best".

I assume an output pinhole model that has pinhole parameters

    (k*fx, k*fy, cx, cy)

where (fx, fy, cx, cy) are the parameters from the input model, and k is the
scaling we compute.

This function looks at some points on the edge of the input image. I choose k so
that all of these points end up inside the pinhole-reprojected image, leaving
the worst one at the edge. The set of points I look at are specified in the
"fit" argument.

ARGUMENTS

- model: a mrcal.cameramodel object for the input lens model

- fit: which pixel coordinates in the input image must project into the output
  pinhole-projected image. The 'fit' argument must be one of

  - a numpy array of shape (N,2) where each row is a pixel coordinate in the
    input image

  - "corners": each of the 4 corners of the input image must project into the
    output image

  - "centers-horizontal": the two points at the left and right edges of the
    input image, half-way vertically, must both project into the output image

  - "centers-vertical": the two points at the top and bottom edges of the input
    image, half-way horizontally, must both project into the output image

RETURNED VALUE

A scalar scale_focal that can be passed to pinhole_model_for_reprojection()

    '''

    if fit is None: return 1.0

    WH  = np.array(model.imagersize(), dtype=float)
    W,H = WH

    if type(fit) is np.ndarray:
        q_edges = fit
    elif type(fit) is str:
        if fit == 'corners':
            q_edges = np.array(((  0.,   0.),
                                (  0., H-1.),
                                (W-1., H-1.),
                                (W-1.,   0.)))
        elif fit == 'centers-horizontal':
            q_edges = np.array(((0,    (H-1.)/2.),
                                (W-1., (H-1.)/2.)))
        elif fit == 'centers-vertical':
            q_edges = np.array((((W-1.)/2., 0,   ),
                                ((W-1.)/2., H-1.,)))
        else:
            raise Exception("fit must be either None or a numpy array or one of ('corners','centers-horizontal','centers-vertical')")
    else:
        raise Exception("fit must be either None or a numpy array or one of ('corners','centers-horizontal','centers-vertical')")

    lensmodel,intrinsics_data = model.intrinsics()

    v_edges = mrcal.unproject(q_edges, lensmodel, intrinsics_data)

    if not mrcal.lensmodel_meta(lensmodel)['has_core']:
        raise Exception("This currently works only with models that have an fxfycxcy core")
    fxy = intrinsics_data[ :2]
    cxy = intrinsics_data[2:4]

    # I have points in space now. My scaled pinhole camera would map these to
    # (k*fx*x/z+cx, k*fy*y/z+cy). I pick a k to make sure that this is in-bounds
    # for all my points, and that the points occupy as large a chunk of the
    # imager as possible. I can look at just the normalized x and y. Just one of
    # the query points should land on the edge; the rest should be in-bounds

    normxy_edges = v_edges[:,:2] / v_edges[:,(2,)]
    normxy_min   = (      - cxy) / fxy
    normxy_max   = (WH-1. - cxy) / fxy

    # Each query point will imply a scale to just fit into the imager I take the
    # most conservative of these. For each point I look at the normalization sign to
    # decide if I should be looking at the min or max edge. And for each I pick the
    # more conservative scale

    # start out at an unreasonably high scale. The data will cut this down
    scale = 1e6
    for p in normxy_edges:
        for ixy in range(2):
            if p[ixy] > 0: scale = np.min((scale,normxy_max[ixy]/p[ixy]))
            else:          scale = np.min((scale,normxy_min[ixy]/p[ixy]))
    return scale


def pinhole_model_for_reprojection(model_from,
                                   fit         = None,
                                   scale_focal = None,
                                   scale_image = None):

    r'''Generate a pinhole model suitable for reprojecting an image

SYNOPSIS

    model_orig = mrcal.cameramodel("xxx.cameramodel")
    image_orig = cv2.imread("image.jpg")

    model_pinhole = mrcal.pinhole_model_for_reprojection(model_orig,
                                                         fit = "corners")

    mapxy = mrcal.compute_map_for_reprojection(model_orig, model_pinhole)

    image_undistorted = mrcal.transform_image(image_orig, mapxy)

Many algorithms work with images assumed to have been captured with a pinhole
camera, even though real-world lenses never fit a pinhole model. mrcal provides
several functions to remap images captured with non-pinhole lenses into images
of the same scene as if they were observed by a pinhole lens. When doing this,
we're free to choose all of the parameters of this pinhole lens model. THIS
function produces the pinhole camera model based on some guidance in the
arguments, and this model can then be used to "undistort" images.

ARGUMENTS

- model_from: the mrcal.cameramodel object used to build the pinhole model. We
  use the intrinsics as the baseline, and we copy the extrinsics to the
  resulting pinhole model.

- fit: optional specification for focal-length scaling. By default we use the
  focal length values from the input model. This is either a numpy array of
  shape (...,2) containing pixel coordinates that the resulting pinhole model
  must represent, or one of ("corners","centers-horizontal","centers-vertical").
  See the docstring for scale_focal__best_pinhole_fit() for details. Exclusive
  with 'scale_focal'

- scale_focal: optional specification for focal-length scaling. By default we
  use the focal length values from the input model. If given, we scale the input
  focal lengths by the given value. Exclusive with 'fit'

- scale_image: optional specification for the scaling of the image size. By
  default the output model represents an image of the same resolution as the
  input model. If we want something else, the scaling can be given here.

RETURNED VALUE

A mrcal.cameramodel object with lensmodel = LENSMODEL_PINHOLE corresponding to
the input model.

    '''

    if scale_focal is None:

        if fit is not None:
            if isinstance(fit, np.ndarray):
                if fit.shape[-1] != 2:
                    raise Exception("'fit' is an array, so it must have shape (...,2)")
                fit = nps.atleast_dims(fit, -2)
                fit = nps.clump(fit, n=len(fit.shape)-1)
                # fit now has shape (N,2)

            elif re.match("^(corners|centers-horizontal|centers-vertical)$", fit):
                # this is valid. nothing to do
                pass
            else:
                raise Exception("'fit' must be an array of shape (...,2) or one of ('corners','centers-horizontal','centers-vertical')",
                      file=sys.stderr)
                sys.exit(1)

        scale_focal = mrcal.scale_focal__best_pinhole_fit(model_from, fit)

    else:
        if fit is not None:
            raise Exception("At most one of 'scale_focal' and 'fit' may be non-None")

    # I have some scale_focal now. I apply it
    lensmodel,intrinsics_data = model_from.intrinsics()
    imagersize                = model_from.imagersize()

    if not mrcal.lensmodel_meta(lensmodel)['has_core']:
        raise Exception("This currently works only with models that have an fxfycxcy core")
    cx,cy = intrinsics_data[2:4]
    intrinsics_data[:2] *= scale_focal

    if scale_image is not None:
        # Now I apply the imagersize scale. The center of the imager should
        # unproject to the same point:
        #
        #   (q0 - cxy0)/fxy0 = v = (q1 - cxy1)/fxy1
        #   ((WH-1)/2 - cxy) / fxy = (((ki*WH)-1)/2 - kc*cxy) / (kf*fxy)
        #
        # The focal lengths scale directly: kf = ki
        #   ((WH-1)/2 - cxy) / fxy = (((ki*WH)-1)/2 - kc*cxy) / (ki*fxy)
        #   (WH-1)/2 - cxy = (((ki*WH)-1)/2 - kc*cxy) / ki
        #   (WH-1)/2 - cxy = (WH-1/ki)/2 - kc/ki*cxy
        #   -1/2 - cxy = (-1/ki)/2 - kc/ki*cxy
        #   1/2 + cxy = 1/(2ki) + kc/ki*cxy
        # -> kc = (1/2 + cxy - 1/(2ki)) * ki / cxy
        #       = (ki + 2*cxy*ki - 1) / (2 cxy)
        #
        # Sanity check: cxy >> 1: ki+2*cxy*ki = ki*(1+2cxy) ~ 2*cxy*ki
        #                         2*cxy*ki - 1 ~ 2*cxy*ki
        #               -> kc ~ 2*cxy*ki /( 2 cxy ) = ki. Yes.
        # Looks like I scale cx and cy separately.
        imagersize[0] = round(imagersize[0]*scale_image)
        imagersize[1] = round(imagersize[1]*scale_image)
        kfxy = scale_image
        kcx  = (kfxy + 2.*cx*kfxy - 1.) / (2. * cx)
        kcy  = (kfxy + 2.*cy*kfxy - 1.) / (2. * cy)

        intrinsics_data[:2] *= kfxy
        intrinsics_data[2]  *= kcx
        intrinsics_data[3]  *= kcy

    return \
        mrcal.cameramodel( intrinsics            = ('LENSMODEL_PINHOLE',intrinsics_data[:4]),
                           extrinsics_rt_fromref = model_from.extrinsics_rt_fromref(),
                           imagersize            = imagersize )


def compute_map_for_reprojection(model_from, model_to,

                                 use_rotation = False,
                                 plane_n      = None,
                                 plane_d      = None):

    r'''Compute a reprojection map between two models

SYNOPSIS

    model_orig = mrcal.cameramodel("xxx.cameramodel")
    image_orig = cv2.imread("image.jpg")

    model_pinhole = mrcal.pinhole_model_for_reprojection(model_orig,
                                                         fit = "corners")

    mapxy = mrcal.compute_map_for_reprojection(model_orig, model_pinhole)

    image_undistorted = mrcal.transform_image(image_orig, mapxy)

    # image_undistorted is now a pinhole-reprojected version of image_orig

Returns the transformation that describes a mapping

- from pixel coordinates of an image of a scene observed by model_to
- to pixel coordinates of an image of the same scene observed by model_from

This transformation can then be applied to a whole image by calling
transform_image().

This function returns a transformation map in an (Nheight,Nwidth,2) array. The
image made by model_to will have shape (Nheight,Nwidth). Each pixel (x,y) in
this image corresponds to a pixel mapxy[y,x,:] in the image made by model_from.

This function has 3 modes of operation:

- intrinsics-only

  This is the default. Selected if

  - use_rotation = False
  - plane_n      = None
  - plane_d      = None

  All of the extrinsics are ignored. If the two cameras have the same
  orientation, then their observations of infinitely-far-away objects will line
  up exactly

- rotation

  This can be selected explicitly with

  - use_rotation = True
  - plane_n      = None
  - plane_d      = None

  Here we use the rotation component of the relative extrinsics. The relative
  translation is impossible to use without knowing what we're looking at, so IT
  IS ALWAYS IGNORED. If the relative orientation in the models matches reality,
  then the two cameras' observations of infinitely-far-away objects will line up
  exactly

- plane

  This is selected if

  - use_rotation = True
  - plane_n is not None
  - plane_d is not None

  We map observations of a given plane in camera FROM coordinates
  coordinates to where this plane would be observed by camera TO. This uses
  ALL the intrinsics, extrinsics and the plane representation. If all of
  these are correct, the observations of this plane would line up exactly in
  the remapped-camera-fromimage and the camera-to image. The plane is
  represented in camera-from coordinates by a normal vector plane_n, and the
  distance to the normal plane_d. The plane is all points p such that
  inner(p,plane_n) = plane_d. plane_n does not need to be normalized; any
  scaling is compensated in plane_d.

ARGUMENTS

- model_from: the mrcal.cameramodel object describing the camera used to capture
  the input image

- model_to: the mrcal.cameramodel object describing the camera that would have
  captured the image we're producing

- use_rotation: optional boolean, defaulting to False. If True: we respect the
  relative rotation in the extrinsics of the camera models.

- plane_n: optional numpy array of shape (3,); None by default. If given, we
  produce a transformation to map observations of a given plane to the same
  pixels in the source and target images. This argument describes the normal
  vector in the coordinate system of model_from. The plane is all points p such
  that inner(p,plane_n) = plane_d. plane_n does not need to be normalized; any
  scaling is compensated in plane_d. If given, plane_d should be given also, and
  use_rotation should be True. if given, we use the full intrinsics and
  extrinsics of both camera models

- plane_d: optional floating-point valud; None by default. If given, we produce
  a transformation to map observations of a given plane to the same pixels in
  the source and target images. The plane is all points p such that
  inner(p,plane_n) = plane_d. plane_n does not need to be normalized; any
  scaling is compensated in plane_d. If given, plane_n should be given also, and
  use_rotation should be True. if given, we use the full intrinsics and
  extrinsics of both camera models

RETURNED VALUE

A numpy array of shape (Nheight,Nwidth,2) where Nheight and Nwidth represent the
imager dimensions of model_to. This array contains 32-bit floats, as required by
cv2.remap() (the function providing the internals of mrcal.transform_image()).
This array can be passed to mrcal.transform_image()

    '''

    if (plane_n is      None and plane_d is not None) or \
       (plane_n is not  None and plane_d is     None):
        raise Exception("plane_n and plane_d should both be None or neither should be None")
    if plane_n is not None and plane_d is not None and \
       not use_rotation:
        raise Exception("We're looking at remapping a plane (plane_d, plane_n are not None), so use_rotation should be True")

    Rt_to_from = None
    if use_rotation:
        Rt_to_r    = model_to.  extrinsics_Rt_fromref()
        Rt_r_from  = model_from.extrinsics_Rt_toref()
        Rt_to_from = mrcal.compose_Rt(Rt_to_r, Rt_r_from)

    lensmodel_from,intrinsics_data_from = model_from.intrinsics()
    lensmodel_to,  intrinsics_data_to   = model_to.  intrinsics()

    if re.match("LENSMODEL_OPENCV",lensmodel_from) and \
       lensmodel_to == "LENSMODEL_PINHOLE"         and \
       plane_n is None:

        # This is a common special case. This branch works identically to the
        # other path (the other side of this "if" can always be used instead),
        # but the opencv-specific code is optimized and at one point ran faster
        # than the code on the other side
        fxy_from = intrinsics_data_from[0:2]
        cxy_from = intrinsics_data_from[2:4]
        cameraMatrix_from = np.array(((fxy_from[0],          0, cxy_from[0]),
                                      ( 0,         fxy_from[1], cxy_from[1]),
                                      ( 0,                   0,           1)))

        fxy_to = intrinsics_data_to[0:2]
        cxy_to = intrinsics_data_to[2:4]
        cameraMatrix_to = np.array(((fxy_to[0],        0, cxy_to[0]),
                                    ( 0,       fxy_to[1], cxy_to[1]),
                                    ( 0,               0,         1)))

        output_shape = model_to.imagersize()
        distortion_coeffs = intrinsics_data_from[4: ]

        if Rt_to_from is not None:
            R_to_from = Rt_to_from[:3,:]
            if np.trace(R_to_from) > 3. - 1e-12:
                R_to_from = None # identity, so I pass None
        else:
            R_to_from = None

        return nps.glue( *[ nps.dummy(arr,-1) for arr in \
                            cv2.initUndistortRectifyMap(cameraMatrix_from, distortion_coeffs,
                                                        R_to_from,
                                                        cameraMatrix_to, tuple(output_shape),
                                                        cv2.CV_32FC1)],
                         axis = -1)

    W_from,H_from = model_from.imagersize()
    W_to,  H_to   = model_to.  imagersize()

    # shape: (Nheight,Nwidth,2). Contains (x,y) rows
    grid = np.ascontiguousarray(nps.mv(nps.cat(*np.meshgrid(np.arange(W_to),
                                                            np.arange(H_to))),
                                       0,-1),
                                dtype = float)
    if lensmodel_to == "LENSMODEL_PINHOLE":
        # Faster path for the unproject. Nice, simple closed-form solution
        fxy_to = intrinsics_data_to[0:2]
        cxy_to = intrinsics_data_to[2:4]
        v = np.zeros( (grid.shape[0], grid.shape[1], 3), dtype=float)
        v[..., :2] = (grid-cxy_to)/fxy_to
        v[...,  2] = 1
    elif lensmodel_to == "LENSMODEL_STEREOGRAPHIC":
        # Faster path for the unproject. Nice, simple closed-form solution
        v = mrcal.unproject_stereographic(grid, *intrinsics_data_to[:4])
    else:
        v = mrcal.unproject(grid, lensmodel_to, intrinsics_data_to)

    if plane_n is not None:

        R_to_from = Rt_to_from[:3,:]
        t_to_from = Rt_to_from[ 3,:]

        # The homography definition. Derived in many places. For instance in
        # "Motion and structure from motion in a piecewise planar environment"
        # by Olivier Faugeras, F. Lustman.
        A_to_from = plane_d * R_to_from + nps.outer(t_to_from, plane_n)
        A_from_to = np.linalg.inv(A_to_from)
        v = nps.matmult( v, nps.transpose(A_from_to) )

    else:
        if Rt_to_from is not None:
            R_to_from = Rt_to_from[:3,:]
            if np.trace(R_to_from) < 3. - 1e-12:
                # rotation isn't identity. apply
                v = nps.matmult(v, R_to_from)

    mapxy = mrcal.project( v, lensmodel_from, intrinsics_data_from )

    return mapxy.astype(np.float32)


def transform_image(image, mapxy):
    r'''Transforms a given image using a given map

SYNOPSIS

    model_orig = mrcal.cameramodel("xxx.cameramodel")
    image_orig = cv2.imread("image.jpg")

    model_pinhole = mrcal.pinhole_model_for_reprojection(model_orig,
                                                         fit = "corners")

    mapxy = mrcal.compute_map_for_reprojection(model_orig, model_pinhole)

    image_undistorted = mrcal.transform_image(image_orig, mapxy)

    # image_undistorted is now a pinhole-reprojected version of image_orig

Given an array of pixel mappings this function can be used to transform one
image to another. If we want to convert a scene image observed by one camera
model to the image of the same scene using a different model, we can produce a
suitable transformation map with mrcal.compute_map_for_reprojection(). An
example of this common usage appears above in the synopsis.

At this time this function is a thin wrapper around cv2.remap()

ARGUMENTS

- image: a numpy array containing an image we're transforming

- mapxy: a numpy array of shape (Nheight,Nwidth,2) where Nheight and Nwidth
  represent the dimensions of the target image. This array is expected to have
  dtype=np.float32, since the internals of this function are provided by
  cv2.remap()

RETURNED VALUE

A numpy array of shape (..., Nheight, Nwidth) containing the transformed image.

    '''
    return cv2.remap(image, mapxy, None,
                     cv2.INTER_LINEAR)


def annotate_image__valid_intrinsics_region(model, image, color=(0,0,255)):
    r'''Annotates a given image with a valid-intrinsics region

    This function takes in a camera model and an image, and returns a numpy
    array with the valid-intrinsics region drawn on top of the image. The image
    is a numpy array. The camera model should contain the valid-intrinsics
    region; if not, the image is returned as is.

    This is similar to mrcal.show_valid_intrinsics_region(), but instead of
    making a plot, it creates an image

    '''
    valid_intrinsics_region = model.valid_intrinsics_region()
    if valid_intrinsics_region is not None:
        cv2.polylines(image, [valid_intrinsics_region], True, color, 3)


def project_boards(lensmodel, intrinsics,
                   extrinsics_rt_fromref,
                   frames_rt_toref,
                   calibration_object_spacing,
                   calibration_object_width_n  = None,
                   calibration_object_height_n = None,
                   observations_board          = None,
                   calobject_warp              = None,
                   **kwargs):

    r'''Project all calibration-time observations of the calibration object

SYNOPSIS

    model               = mrcal.cameramodel("xxx.cameramodel")
    optimization_inputs = model.optimization_inputs()

    q = mrcal.project_boards( **optimization_inputs )

    # q is an array of pixel coordinates of all the chessboard corners over all
    # the cameras and all the instances in time

Note: This function returns dense observations, which is probably not what you
want. Consider using mrcal.hypothesis_corner_positions() instead.

A camera calibration based on observations of a known object consists of
minimizing the difference between observed pixels and the projections of the
object using a hypothetical set of parameters. These hypothetical projections
can be obtained by calling this function, usually as part of an evaluation of
calibration results.

An actual calibration problem usually has sparse observations: some cameras will
observe the calibration object sometimes, but not all cameras will observe it
all the time. This hypothetical projection function isn't constrained by
reality, however, so it reports dense results: it says where each camera would
have seen the object at every point in time.

The arguments to this function are a subset of the arguments to
mrcal.optimize(), so we can pass in **optimization_inputs. The actual arguments
we use are listed below. Normally the dimensions of the calibration object are
inferred from the shape of the 'observations_board' array. Since this function
doesn't need that data, we can pass those dimensions in explicitly, in the
calibration_object_width_n, calibration_object_height_n arguments. Either those
arguments OR observations_board must be given.

ARGUMENTS

- lensmodel: the lens model we're using. This is a string "LENSMODEL_..."

- intrinsics: a numpy array of shape (Ncameras_intrinsics,Nintrinsics)
  describing the lens parameters

- extrinsics_rt_fromref: a numpy array of shape (Ncameras_extrinsics,6)
  describing the relative camera geometry

- frames_rt_toref: a numpy array of shape (Nframes,6) describing the pose of the
  calibration object at each instance in time

- calibration_object_spacing: the distance between adjacent corners in the
  calibration object

- calibration_object_width_n: optional integer describing the horizontal corner
  count in the calibration object. If omitted, we get this from the shape of
  'observations_board'

- calibration_object_height_n: optional integer describing the vertical corner
  count in the calibration object. If omitted, we get this from the shape of
  'observations_board'

- observations_board: optional numpy array used to provide the shape of the
  calibration object if calibration_object_width_n and/or
  calibration_object_height_n are not given

- calobject_warp: optional numpy array of shape (2,) used to describe the
  warping of the calibration object. If omitted, we assume a flat object

RETURNED VALUE

We return a dense numpy array of pixel observations of shape
(Nframes,Ncameras,Nheight,Nwidth,2)

    '''

    if calibration_object_height_n is None: calibration_object_height_n = observations_board.shape[-3]
    if calibration_object_width_n  is None: calibration_object_width_n  = observations_board.shape[-2]


    object_ref = mrcal.ref_calibration_object(calibration_object_width_n,
                                              calibration_object_height_n,
                                              calibration_object_spacing,
                                              calobject_warp)
    Rf = mrcal.R_from_r(frames_rt_toref[:,:3])
    Rf = nps.mv(Rf,           0, -5)
    tf = nps.mv(frames_rt_toref[:,3:], 0, -5)

    # object in the cam0 coord system. shape=(Nframes, 1, calibration_object_height_n, calibration_object_width_n, 3)
    object_cam0 = nps.matmult( object_ref, nps.transpose(Rf)) + tf

    Rc = mrcal.R_from_r(extrinsics_rt_fromref[:,:3])
    Rc = nps.mv(Rc,               0, -4)
    tc = nps.mv(extrinsics_rt_fromref[:,3:], 0, -4)

    # object in the OTHER camera coord systems.
    # shape=(Nframes, Ncameras-1, calibration_object_height_n, calibration_object_width_n, 3)
    object_cam_others = nps.matmult( object_cam0, nps.transpose(Rc)) + tc

    # object in the ALL camera coord systems.
    # shape=(Nframes, Ncameras, calibration_object_height_n, calibration_object_width_n, 3)
    object_cam = nps.glue(object_cam0, object_cam_others, axis=-4)

    # projected points.
    # shape=(Nframes, Ncameras, calibration_object_height_n, calibration_object_width_n, 2)

    # loop over Ncameras. project() will broadcast over the points
    intrinsics = nps.atleast_dims(intrinsics, -2)
    Ncameras = intrinsics.shape[-2]
    return nps.mv( nps.cat(*[mrcal.project( np.ascontiguousarray(object_cam[...,i_camera,:,:,:]),
                                            lensmodel,
                                            intrinsics[i_camera,:] ) for \
                             i_camera in range(Ncameras)]),
                   0,-4)


def calibration_residuals(projected_boards, observations_board, indices_frame_camera):
    r'''Report reprojection errors at a calibration solution

SYNOPSIS

    model               = mrcal.cameramodel("xxx.cameramodel")
    optimization_inputs = model.optimization_inputs()

    q = mrcal.project_boards( **optimization_inputs )

    observations_board   = optimization_inputs['observations_board']
    indices_frame_camera = optimization_inputs['indices_frame_camintrinsics_camextrinsics'][:,:2]

    err_everything,err_zero_for_outliers = \
        mrcal.calibration_residuals(q,
                                    observations_board,
                                    indices_frame_camera)

A camera calibration based on observations of a known object consists of
minimizing the difference between observed pixels and the projections of the
object using a hypothetical set of parameters. These differences can be obtained
by calling this function, usually as part of an evaluation of calibration
results.

Unlike mrcal.optimize(), this function assumes we're solving a vanilla
calibration problem: stationary cameras are observing a moving calibration
object, with the first camera defining the reference coordinate system. Thus
this function accepts 'indices_frame_camera' instead of the more general
'indices_frame_camintrinsics_camextrinsics' accepted by mrcal.optimize(). For
this function, the "camera" refers to the intrinsics, so

    indices_frame_camera = indices_frame_camintrinsics_camextrinsics[:,:2]

We return two versions of the residuals: one reporting everything, including the
outliers, and the other with 0 in place of the outlier residuals. The outliers
aren't a part of the optimization, so the first version would show very high
errors for those observations. The second version is what is actually seen by
the optimization algorithm.

The optimization weighs the observations differently, depending on the value of
the weights stored in observations_board[..., 2]. This function returns pure
pixel errors, and does NOT apply the weights.

ARGUMENTS

- projected_boards: a numpy array of shape
  (Nframes,Ncameras,object_height_n,object_width_n,2). These are the dense
  projections that we would be observing if the parameters in this solve
  described the world perfectly. Unlike this dense array, the actual
  observations are sparse: some cameras don't observe the chessboard in some
  frames, with the pattern described in indices_frame_camera. This array is
  returned by mrcal.project_boards()

- observations_board: a numpy array of shape
  (Nobservations,object_height_n,object_width_n,3). The observed pixel
  coordinates of the calibration object corners. Each row of this array is
  (x,y,weight). This function does NOT apply the weight to the pixel errors.
  Outliers are indicated by weight<0, and this function DOES respect that.

- indices_frame_camera: a numpy array of shape (Nobservations,2). For each
  observation in observations_board the frame and camera indices are given in
  each row of indices_frame_camera. The frame indicates a point in time.

RETURNED VALUE

A tuple:

- err_everything: a numpy array of shape
  (Nobservations,object_height_n,object_width_n,2) describing the x,y pixel
  errors of each corner in each observation. This array contains errors even for
  outliers. Those were not used in the optimization, so large errors would
  appear in this array for the outlier observations

- err_zero_for_outliers: a numpy array of shape
  (Nobservations,object_height_n,object_width_n,2) describing the x,y pixel
  errors of each corner in each observation. This array contains 0 for outlier
  observations

    '''

    Nobservations = indices_frame_camera.shape[0]

    object_height_n = observations_board.shape[-3]
    object_width_n  = observations_board.shape[-2]

    err_everything        = np.zeros((Nobservations,object_height_n,object_width_n,2))
    err_zero_for_outliers = np.zeros((Nobservations,object_height_n,object_width_n,2))

    for i_observation in range(Nobservations):
        i_frame, i_camera = indices_frame_camera[i_observation]

        err = projected_boards[i_frame,i_camera] - observations_board[i_observation, ..., :2]

        err_everything[i_observation] = err

        mask_outliers = (observations_board[i_observation, ..., 2] < 0.0)
        err[mask_outliers, :] = 0
        err_zero_for_outliers[i_observation] = err

    return err_everything,err_zero_for_outliers

