#!/usr/bin/python3

from __future__ import print_function

import numpy as np
import numpysane as nps
import sys
import re
import cv2
import scipy.optimize

import mrcal


def unproject(q, lens_model, intrinsics_data):
    r'''Removes distortion from pixel observations

SYNOPSIS

    v = mrcal.unproject( # (N,2) array of pixel observations
                         q,
                         lens_model, intrinsics_data )

This is the python wrapper to the internal written-in-C _mrcal._unproject(). This
wrapper that has a slow path to handle CAHVORE. Otherwise, the it just calls
_mrcal._unproject(), which does NOT support CAHVORE

Maps a set of 2D imager points q to a 3d vector in camera coordinates that
produced these pixel observations. The 3d vector is unique only up-to-length.
The returned vectors aren't normalized.

This is the "reverse" direction, so an iterative nonlinear optimization is
performed internally to compute this result. This is much slower than
mrcal_project. For OpenCV distortions specifically, OpenCV has
cvUndistortPoints() (and cv2.undistortPoints()), but these are inaccurate:
https://github.com/opencv/opencv/issues/8811

This function does NOT support CAHVORE.

ARGUMENTS

- q: array of dims (...,2); the pixel coordinates we're unprojecting. This
  supports broadcasting fully, and any leading dimensions are allowed, including
  none

- lens_model: a string such as

  LENSMODEL_PINHOLE
  LENSMODEL_OPENCV4
  LENSMODEL_OPENCV5
  LENSMODEL_OPENCV8
  LENSMODEL_OPENCV12 (if we have OpenCV >= 3.0.0)
  LENSMODEL_OPENCV14 (if we have OpenCV >= 3.1.0)
  LENSMODEL_CAHVOR

- intrinsics_data: array of dims (Nintrinsics):

    (focal_x, focal_y, center_pixel_x, center_pixel_y, distorion0, distortion1,
    ...)

  The focal lengths are given in pixels.

    '''

    if lens_model != 'LENSMODEL_CAHVORE':
        return mrcal._mrcal._unproject(q, lens_model, intrinsics_data)

    # CAHVORE. This is a reimplementation of the C code. It's barely maintained,
    # and here for legacy compatibility only

    if q is None: return q
    if q.size == 0:
        s = q.shape
        return np.zeros(s[:-1] + (3,))

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
                mrcal.project(np.array((vxy[0],vxy[1],1.)), lens_model, intrinsics_data) - \
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
    return  nps.glue(vxy, np.ones( vxy.shape[:-1] + (1,) ), axis=-1)


def compute_scale_f_pinhole_for_fit(model, fit):
    r'''Compute best value of scale_f_pinhole for mrcal-reproject-image

    mrcal-reproject-image can produce images obtained with an arbitrary
    focal-length scale. Different scaling values can either zoom in to cut off
    sections of the original image, or zoom out to waste pixels with no-data
    available at the edges. This functions computes optimal values for
    scale_f_pinhole, based on the "fit" parameter. This is one of

    - None to keep the focal lengths the same for the input and output image
      models. This is a scaling of 1.0

    - A numpy array of imager points that must fill the imager as much as
      possible, without going out of bounds

    - A string that indicates which points must remain in bounds. One of
      "corners", "centers-horizontal", "centers-vertical"

    '''

    # I create an output pinhole model that has pinhole parameters
    # (k*fx,k*fy,cx,cy) where (fx,fy,cx,cy) are the parameters from the input
    # model. Note the scaling k
    #
    # I look at a number of points on the edge of the input image. Of the points
    # I look at I choose k so that all the points end up inside the undistorted
    # image, leaving the worst one at the edge
    if fit is None: return 1.0

    W,H = model.imagersize()
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
            raise Exception("fit must be either None or a numpy array of one of ('corners','centers-horizontal','centers-vertical')")
    else:
        raise Exception("fit must be either None or a numpy array of one of ('corners','centers-horizontal','centers-vertical')")

    lens_model,intrinsics_data = model.intrinsics()

    v_edges = mrcal.unproject(q_edges, lens_model, intrinsics_data)

    if not mrcal.getLensModelMeta(lens_model)['has_core']:
        raise Exception("This currently works only with models that have an fxfycxcy core")
    fxy = intrinsics_data[ :2]
    cxy = intrinsics_data[2:4]

    # I have points in space now. My scaled pinhole camera would map these to
    # (k*fx*x/z+cx, k*fy*y/z+cy). I pick a k to make sure that this is in-bounds
    # for all my points, and that the points occupy as large a chunk of the
    # imager as possible. I can look at just the normalized x and y. Just one of
    # the query points should land on the edge; the rest should be in-bounds

    WH = np.array(model.imagersize(), dtype=float)

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


def make_target_pinhole_model_for_reprojection(model_from,
                                               fit         = None,
                                               scale_focal = None,
                                               scale_image = None):
    r'''Generate a pinhole model for mrcal-reproject-image

    mrcal-reproject-image can produce images obtained with a pihole model with
    particular focal-length and image scalings. The target model used by that
    routine is generated by this function from

    - A base model. This defines the extrinsics, the base imager size, and the
      base pinhole parameters

    - The "fit" spec. This defines the focal-length scaling, computd by
      compute_scale_f_pinhole_for_fit()

    - The imager size scaling

    A pinhole model with all the scalings applied is returned by this function

    '''

    if scale_focal is None:

        if fit is not None:
            if re.match(r"^[0-9\.e-]+(,[0-9\.e-]+)*$", fit):
                xy = np.array([int(x) for x in fit.split(',')], dtype=float)
                Nxy = len(xy)
                if Nxy % 2 or Nxy < 4:
                    print(f"If passing pixel coordinates to --fit, I need at least 2 x,y pairs. Instead got {Nxy} values",
                          file=sys.stderr)
                    sys.exit(1)
                fit = xy.reshape(Nxy//2, 2)
            elif re.match("^(corners|centers-horizontal|centers-vertical)$", fit):
                # this is valid. nothing to do
                pass
            else:
                print("--fit must be a comma-separated list of numbers or one of ('corners','centers-horizontal','centers-vertical')",
                      file=sys.stderr)
                sys.exit(1)

        scale_focal = mrcal.compute_scale_f_pinhole_for_fit(model_from, fit)

    # I have some scale_focal now. I apply it
    lensmodel,intrinsics_data = model_from.intrinsics()
    imagersize                = model_from.imagersize()

    if not mrcal.getLensModelMeta(lensmodel)['has_core']:
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


def reproject_image__compute_map(model_from, model_to,

                                 use_rotation = False,
                                 plane_n      = None,
                                 plane_d      = None):

    r'''Computes a reprojection map between two models

    Synopsis:

        # To transform an image to a pinhole projection
        model_pinhole = make_target_pinhole_model_for_reprojection(model)
        mapxy = mrcal.reproject_image__compute_map(model, model_pinhole)
        image_transformed = mrcal.transform_image(image, mapxy)

    Returns the transformation that describes a mapping

    - from pixel coordinates of an image of a scene observed by model_to
    - to pixel coordinates of an image of the same scene observed by model_from

    This transformation can then be applied to a whole image by calling
    transform_image().

    The return transformation map is an (2,Nheight,Nwidth) array. The image made
    by model_to will have shape (Nheight,Nwidth). Each pixel (x,y) in this image
    corresponds to a pixel (mapxy[0,y,x],mapxy[1,y,x]) in the image made by
    model_from. Note the strange xy/yx ordering here.

    This function has 3 modes of operation:

    - intrinsics-only

      This is the default. Selected if

      - use_rotation = False
      - plane_n      = None
      - plane_d      = None

      All of the extrinsics are ignored. If both cameras have the same
      orientation, then their observations of infinitely-far-away objects will
      line up exactly

    - rotation

      This can be selected explicitly with

      - use_rotation = True
      - plane_n      = None
      - plane_d      = None

      Here we use the rotation component of the relative extrinsics. The
      relative translation is impossible to use, so IT IS ALWAYS IGNORED. If the
      relative orientation in the models matches reality, then the two cameras'
      observations of infinitely-far-away objects will line up exactly

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
      scaling is compensated in plane_d

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

    lens_model_from,intrinsics_data_from = model_from.intrinsics()
    lens_model_to,  intrinsics_data_to   = model_to.  intrinsics()

    if re.match("LENSMODEL_OPENCV",lens_model_from) and \
       lens_model_to == "LENSMODEL_PINHOLE"         and \
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

        return nps.cat( *cv2.initUndistortRectifyMap(cameraMatrix_from, distortion_coeffs,
                                                     R_to_from,
                                                     cameraMatrix_to, tuple(output_shape),
                                                     cv2.CV_32FC1) )

    W_from,H_from = model_from.imagersize()
    W_to,  H_to   = model_to.  imagersize()

    # shape: Nwidth,Nheight,2
    grid  = np.ascontiguousarray(nps.reorder(nps.cat(*np.meshgrid(np.arange(W_to),
                                                                  np.arange(H_to))),
                                             -1, -2, -3),
                                 dtype = float)
    if lens_model_to == "LENSMODEL_PINHOLE":
        # faster path for the unproject; probably
        fxy_to = intrinsics_data_to[0:2]
        cxy_to = intrinsics_data_to[2:4]
        v = np.zeros( (grid.shape[0], grid.shape[1], 3), dtype=float)
        v[..., :2] = (grid-cxy_to)/fxy_to
        v[...,  2] = 1
    else:
        v = mrcal.unproject(grid, lens_model_to, intrinsics_data_to)

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
        R_to_from = Rt_to_from[:3,:]
        if np.trace(R_to_from) < 3. - 1e-12:
            # rotation isn't identity. apply
            v = nps.matmult(v, R_to_from)

    mapxy = mrcal.project( v, lens_model_from, intrinsics_data_from )
    # Reorder to (2,Nheight,Nwidth)
    return nps.reorder(mapxy, -1, -2, -3).astype(np.float32)


def transform_image(image, mapxy):
    r'''Transforms a given image using a given map

    The map is an (2,Nheight,Nwidth) array; this can be generated by
    reproject_image__compute_map(). The returned output image will have shape
    (Nheight,Nwidth). Each pixel (x,y) in this output image corresponds to a
    pixel (mapxy[0,y,x],mapxy[1,y,x]) in the input image. Note the strange xy/yx
    ordering here.

    '''
    return cv2.remap(image, mapxy[0], mapxy[1],
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


def calobservations_project(lens_model, intrinsics, extrinsics, frames, dot_spacing, Nwant, calobject_warp):
    r'''Takes in the same arguments as mrcal.optimize(), and returns all
    the projections. Output has shape (Nframes,Ncameras,Nwant,Nwant,2)

    '''

    object_ref = mrcal.get_ref_calibration_object(Nwant, Nwant, dot_spacing, calobject_warp)
    Rf = mrcal.R_from_r(frames[:,:3])
    Rf = nps.mv(Rf,           0, -5)
    tf = nps.mv(frames[:,3:], 0, -5)

    # object in the cam0 coord system. shape=(Nframes, 1, Nwant, Nwant, 3)
    object_cam0 = nps.matmult( object_ref, nps.transpose(Rf)) + tf

    Rc = mrcal.R_from_r(extrinsics[:,:3])
    Rc = nps.mv(Rc,               0, -4)
    tc = nps.mv(extrinsics[:,3:], 0, -4)

    # object in the OTHER camera coord systems. shape=(Nframes, Ncameras-1, Nwant, Nwant, 3)
    object_cam_others = nps.matmult( object_cam0, nps.transpose(Rc)) + tc

    # object in the ALL camera coord systems. shape=(Nframes, Ncameras, Nwant, Nwant, 3)
    object_cam = nps.glue(object_cam0, object_cam_others, axis=-4)

    # projected points. shape=(Nframes, Ncameras, Nwant, Nwant, 2)

    # loop over Ncameras. project() will broadcast over the points
    intrinsics = nps.atleast_dims(intrinsics, -2)
    Ncameras = intrinsics.shape[-2]
    return nps.mv( nps.cat(*[mrcal.project( np.ascontiguousarray(object_cam[...,i_camera,:,:,:]),
                                            lens_model,
                                            intrinsics[i_camera,:] ) for \
                             i_camera in range(Ncameras)]),
                   0,-4)

def calobservations_compute_reproj_error(projected, observations, indices_frame_camera, Nwant,
                                         outlier_indices = None):
    r'''Computes reprojection errors when calibrating with board observations

    Given

    - projected (shape [Nframes,Ncameras,Nwant,Nwant,2])
    - observations (shape [Nframes,Nwant,Nwant,3])
    - indices_frame_camera (shape [Nobservations,2])
    - outlier_indices, a list of point indices that were deemed to be outliers.
      These are plain integers indexing the flattened observations array, but
      one per POINT, not (x,y) independently

    Return (err_all_points,err_ignoring_outliers). Each is the reprojection
    error for each point: shape (Nobservations,Nwant,Nwant,2). One includes the
    outliers in the returned errors, and the other does not

    '''

    if outlier_indices is None: outlier_indices = np.array(())

    Nframes               = projected.shape[0]
    Nobservations         = indices_frame_camera.shape[0]
    err_all_points        = np.zeros((Nobservations,Nwant,Nwant,2))

    for i_observation in range(Nobservations):
        i_frame, i_camera = indices_frame_camera[i_observation]

        err_all_points[i_observation] = projected[i_frame,i_camera] - observations[i_observation, ..., :2]

    err_ignoring_outliers = err_all_points.copy()
    err_ignoring_outliers.ravel()[outlier_indices*2  ] = 0
    err_ignoring_outliers.ravel()[outlier_indices*2+1] = 0

    return err_all_points,err_ignoring_outliers

