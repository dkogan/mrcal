#!/usr/bin/python3

# Copyright (c) 2017-2023 California Institute of Technology ("Caltech"). U.S.
# Government sponsorship acknowledged. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

'''Routines for transformation of images

All functions are exported into the mrcal module. So you can call these via
mrcal.image_transforms.fff() or mrcal.fff(). The latter is preferred.

'''

import numpy as np
import numpysane as nps
import sys
import re
import mrcal

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
                          rt_cam_ref = model.rt_cam_ref() )

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

    if not mrcal.lensmodel_metadata_and_config(lensmodel)['has_core']:
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
                                   *,
                                   scale_focal = None,
                                   scale_image = None):

    r'''Generate a pinhole model suitable for reprojecting an image

SYNOPSIS

    model_orig = mrcal.cameramodel("xxx.cameramodel")
    image_orig = mrcal.load_image("image.jpg")

    model_pinhole = mrcal.pinhole_model_for_reprojection(model_orig,
                                                         fit = "corners")

    mapxy = mrcal.image_transformation_map(model_orig, model_pinhole,
                                           intrinsics_only = True)

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

    if not mrcal.lensmodel_metadata_and_config(lensmodel)['has_core']:
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
                           rt_cam_ref = model_from.rt_cam_ref(),
                           imagersize            = imagersize )


def image_transformation_map(model_from, model_to,
                             *,
                             intrinsics_only                   = False,
                             distance                          = None,
                             plane_n                           = None,
                             plane_d                           = None,
                             mask_valid_intrinsics_region_from = False):

    r'''Compute a reprojection map between two models

SYNOPSIS

    model_orig = mrcal.cameramodel("xxx.cameramodel")
    image_orig = mrcal.load_image("image.jpg")

    model_pinhole = mrcal.pinhole_model_for_reprojection(model_orig,
                                                         fit = "corners")

    mapxy = mrcal.image_transformation_map(model_orig, model_pinhole,
                                           intrinsics_only = True)

    image_undistorted = mrcal.transform_image(image_orig, mapxy)

    # image_undistorted is now a pinhole-reprojected version of image_orig

Returns the transformation that describes a mapping

- from pixel coordinates of an image of a scene observed by model_to
- to pixel coordinates of an image of the same scene observed by model_from

This transformation can then be applied to a whole image by calling
mrcal.transform_image().

This function returns a transformation map in an (Nheight,Nwidth,2) array. The
image made by model_to will have shape (Nheight,Nwidth). Each pixel (x,y) in
this image corresponds to a pixel mapxy[y,x,:] in the image made by model_from.

One application of this function is to validate the models in a stereo pair. For
instance, reprojecting one camera's image at distance=infinity should produce
exactly the same image that is observed by the other camera when looking at very
far objects, IF the intrinsics and rotation are correct. If the images don't
line up well, we know that some part of the models is off. Similarly, we can use
big planes (such as observations of the ground) and plane_n, plane_d to validate.

This function has several modes of operation:

- intrinsics, extrinsics

  Used if not intrinsics_only and \
          plane_n is None     and \
          plane_d is None

  This is the default. For each pixel in the output, we use the full model to
  unproject a given distance out, and then use the full model to project back
  into the other camera.

- intrinsics only

  Used if intrinsics_only and \
          plane_n is None and \
          plane_d is None

  Similar, but the extrinsics are ignored. We unproject the pixels in one model,
  and project the into the other camera. The two camera coordinate systems are
  assumed to line up perfectly

- plane

  Used if plane_n is not None and
          plane_d is not None

  We map observations of a given plane in camera FROM coordinates coordinates to
  where this plane would be observed by camera TO. We unproject each pixel in
  one camera, compute the intersection point with the plane, and project that
  intersection point back to the other camera. This uses ALL the intrinsics,
  extrinsics and the plane representation. The plane is represented by a normal
  vector plane_n, and the distance to the normal plane_d. The plane is all
  points p such that inner(p,plane_n) = plane_d. plane_n does not need to be
  normalized; any scaling is compensated in plane_d.

ARGUMENTS

- model_from: the mrcal.cameramodel object describing the camera used to capture
  the input image. We always use the intrinsics. if not intrinsics_only: we use
  the extrinsics also

- model_to: the mrcal.cameramodel object describing the camera that would have
  captured the image we're producing. We always use the intrinsics. if not
  intrinsics_only: we use the extrinsics also

- intrinsics_only: optional boolean, defaulting to False. If False: we respect
  the relative transformation in the extrinsics of the camera models.

- distance: optional value, defaulting to None. Used only if not
  intrinsics_only. We reproject points in space a given distance out. If
  distance is None (the default), we look out to infinity. This is equivalent to
  using only the rotation component of the extrinsics, ignoring the translation.

- plane_n: optional numpy array of shape (3,); None by default. If given, we
  produce a transformation to map observations of a given plane to the same
  pixels in the source and target images. This argument describes the normal
  vector in the coordinate system of model_from. The plane is all points p such
  that inner(p,plane_n) = plane_d. plane_n does not need to be normalized; any
  scaling is compensated in plane_d. If given, plane_d should be given also, and
  intrinsics_only should be False. if given, we use the full intrinsics and
  extrinsics of both camera models

- plane_d: optional floating-point value; None by default. If given, we produce
  a transformation to map observations of a given plane to the same pixels in
  the source and target images. The plane is all points p such that
  inner(p,plane_n) = plane_d. plane_n does not need to be normalized; any
  scaling is compensated in plane_d. If given, plane_n should be given also, and
  intrinsics_only should be False. if given, we use the full intrinsics and
  extrinsics of both camera models

- mask_valid_intrinsics_region_from: optional boolean defaulting to False. If
  True, points outside the valid-intrinsics region in the FROM image are set to
  black, and thus do not appear in the output image

RETURNED VALUE

A numpy array of shape (Nheight,Nwidth,2) where Nheight and Nwidth represent the
imager dimensions of model_to. This array contains 32-bit floats, as required by
cv2.remap() (the function providing the internals of mrcal.transform_image()).
This array can be passed to mrcal.transform_image()

    '''

    if (plane_n is      None and plane_d is not None) or \
       (plane_n is not  None and plane_d is     None):
        raise Exception("plane_n and plane_d should both be None or neither should be None")
    if plane_n is not None and \
       intrinsics_only:
        raise Exception("We're looking at remapping a plane (plane_d, plane_n are not None), so intrinsics_only should be False")

    if distance is not None and \
       (plane_n is not None or intrinsics_only):
        raise Exception("'distance' makes sense only without plane_n/plane_d and without intrinsics_only")

    if intrinsics_only:
        Rt_to_from = None
    else:
        Rt_to_from = mrcal.compose_Rt(model_to.  Rt_cam_ref(),
                                      model_from.Rt_ref_cam())

    lensmodel_from,intrinsics_data_from = model_from.intrinsics()
    lensmodel_to,  intrinsics_data_to   = model_to.  intrinsics()

    if re.match("LENSMODEL_OPENCV",lensmodel_from) and \
       lensmodel_to == "LENSMODEL_PINHOLE"         and \
       plane_n is None                             and \
       not mask_valid_intrinsics_region_from       and \
       distance is None:

        # This is a common special case. This branch works identically to the
        # other path (the other side of this "if" can always be used instead),
        # but the opencv-specific code is optimized and at one point ran faster
        # than the code on the other side.
        #
        # The mask_valid_intrinsics_region_from isn't implemented in this path.
        # It COULD be, then this faster path could be used
        import cv2

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
            if distance is not None:
                v = mrcal.transform_point_Rt(mrcal.invert_Rt(Rt_to_from),
                                             v/nps.dummy(nps.mag(v),-1) * distance)
            else:
                R_to_from = Rt_to_from[:3,:]
                v = nps.matmult(v, R_to_from)

    mapxy = mrcal.project( v, lensmodel_from, intrinsics_data_from )

    if mask_valid_intrinsics_region_from:

        # Using matplotlib to compute the out-of-bounds points. It doesn't
        # support broadcasting, so I do that manually with a clump/reshape
        from matplotlib.path import Path
        region = Path(model_from.valid_intrinsics_region())
        is_inside = region.contains_points(nps.clump(mapxy,n=2)).reshape(mapxy.shape[:2])
        mapxy[ ~is_inside, :] = -1


    return mapxy.astype(np.float32)


def transform_image(image, mapxy,
                    *,
                    out = None,
                    borderMode    = None,
                    borderValue   = 0,
                    interpolation = None):

    r'''Transforms a given image using a given map

SYNOPSIS

    model_orig = mrcal.cameramodel("xxx.cameramodel")
    image_orig = mrcal.load_image("image.jpg")

    model_pinhole = mrcal.pinhole_model_for_reprojection(model_orig,
                                                         fit = "corners")

    mapxy = mrcal.image_transformation_map(model_orig, model_pinhole,
                                           intrinsics_only = True)

    image_undistorted = mrcal.transform_image(image_orig, mapxy)

    # image_undistorted is now a pinhole-reprojected version of image_orig

Given an array of pixel mappings this function can be used to transform one
image to another. If we want to convert a scene image observed by one camera
model to the image of the same scene using a different model, we can produce a
suitable transformation map with mrcal.image_transformation_map(). An example of
this common usage appears above in the synopsis.

At this time this function is a thin wrapper around cv2.remap()

ARGUMENTS

- image: a numpy array containing an image we're transforming. May be grayscale:
  shape (Nheight_input, Nwidth_input) or RGB: shape (Nheight_input,
  Nwidth_input, 3)

- mapxy: a numpy array of shape (Nheight,Nwidth,2) where Nheight and Nwidth
  represent the dimensions of the target image. This array is expected to have
  dtype=np.float32, since the internals of this function are provided by
  cv2.remap()

- out: optional numpy array of shape (Nheight,Nwidth) or (Nheight,Nwidth,3) to
  receive the result. If omitted, a new array is allocated and returned.

- borderMode: optional constant defining out-of-bounds behavior. Defaults to
  cv2.BORDER_TRANSPARENT, and is passed directly to cv2.remap(). Please see the
  docs for that function for details. This option may disappear if mrcal stops
  relying on opencv

- interpolation: optional constant defining pixel interpolation behavior.
  Defaults to cv2.INTER_LINEAR, and is passed directly to cv2.remap(). Please
  see the docs for that function for details. This option may disappear if mrcal
  stops relying on opencv

RETURNED VALUE

A numpy array of shape (Nheight, Nwidth) if grayscale or (Nheight, Nwidth, 3) if
RGB. Contains the transformed image.

    '''

    # necessary to avoid opencv crashing
    if not isinstance(image, np.ndarray): raise Exception("'image' must be a numpy array")
    if not isinstance(mapxy, np.ndarray): raise Exception("'mapxy' must be a numpy array")

    import cv2
    if borderMode is None:
        borderMode = cv2.BORDER_CONSTANT
    if interpolation is None:
        interpolation = cv2.INTER_LINEAR
    return cv2.remap(image, mapxy, None,
                     borderMode    = borderMode,
                     interpolation = interpolation,
                     borderValue   = borderValue,
                     dst           = out)
