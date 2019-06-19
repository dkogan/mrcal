#!/usr/bin/python3

from __future__ import print_function

import numpy as np
import numpysane as nps
import sys
import re
import cv2
import scipy.optimize

import mrcal


# Broadcasting available for points only; handled internally by the C layer
def project(v, distortion_model, intrinsics_data, get_gradients=False):
    r'''Projects 3D point(s) using the given camera intrinsics

    Most of the time this invokes _mrcal.project() directly UNLESS we're using
    CAHVORE. _mrcal.project() does not support CAHVORE, so we implement our own
    path here. gradients are NOT implemented for CAHVORE

    This function is broadcastable over points only.

    Inputs:

    - v 3D point(s) in the camera coord system. This is unaffected by scale, so
      from the point of view of a camera, this is an observation vector

    - distortion_model: a string that says what the values in the intrinsics
      array mean. The supported values are reported by
      mrcal.getSupportedDistortionModels(). At the time of this
      writing, the supported values are

        DISTORTION_NONE
        DISTORTION_OPENCV4
        DISTORTION_OPENCV5
        DISTORTION_OPENCV8
        DISTORTION_OPENCV12 (if we have OpenCV >= 3.0.0)
        DISTORTION_OPENCV14 (if we have OpenCV >= 3.1.0)
        DISTORTION_CAHVOR
        DISTORTION_CAHVORE

    - intrinsics_data: a numpy array containing
      - fx
      - fy
      - cx
      - cy
      - distortion-specific values

    if get_gradients: instead of returning the projected points I return a tuple

    - (...,2) array of projected pixel coordinates
    - (...,2,3) array of the gradients of the pixel coordinates in respect to
      the input 3D point positions
    - (...,2,Nintrinsics) array of the gradients of the pixel coordinates in
      respect to the intrinsics

    '''

    if v is None: return v
    if v.size == 0:
        if get_gradients:
            Nintrinsics = intrinsics_data.shape[-1]
            s = v.shape
            return np.zeros(s[:-1] + (2,)), np.zeros(s[:-1] + (2,3)), np.zeros(s[:-1] + (2,Nintrinsics))
        else:
            s = v.shape
            return np.zeros(s[:-1] + (2,))

    if distortion_model != 'DISTORTION_CAHVORE':
        return mrcal._mrcal.project(np.ascontiguousarray(v),
                                    distortion_model,
                                    intrinsics_data,
                                    get_gradients=get_gradients)

    # oof. CAHVORE. Legacy code follows
    if get_gradients:
        raise Exception("Gradients not implemented for CAHVORE")

    def project_one_cam(intrinsics, v):
        fxy = intrinsics[:2]
        cxy = intrinsics[2:4]
        q    = v[..., :2]/v[..., (2,)] * fxy + cxy
        return mrcal.distort(q, 'DISTORTION_CAHVORE', intrinsics)


    # manually broadcast over intrinsics[]. The broadcast over v happens
    # implicitly. Note that I'm not implementing broadcasting over intrinsics in
    # the non-cahvore path above. I might at some point
    #
    # intrinsics shape I support: (a,b,c,..., Nintrinsics)
    # In my use case, at most one of a,b,c,... is != 1
    idims_not1 = [ i for i in range(len(intrinsics_data.shape)-1) if intrinsics_data.shape[i] != 1 ]
    if len(idims_not1) > 1:
        raise Exception("More than 1D worth of broadcasting for the intrinsics not implemented")

    if len(idims_not1) == 0:
        return project_one_cam(intrinsics_data.ravel(), v)

    idim_broadcast = idims_not1[0] - len(intrinsics_data.shape)
    Nbroadcast = intrinsics_data.shape[idim_broadcast]
    if v.shape[idim_broadcast] != Nbroadcast:
        raise Exception("Inconsistent dimensionality for broadcast at idim {}. v.shape: {} and intrinsics_data.shape: {}".format(idim_broadcast, v.shape, intrinsics_data.shape))

    vsplit          = nps.mv(v,               idim_broadcast, 0)
    intrinsics_data = nps.mv(intrinsics_data, idim_broadcast, 0)

    return \
        nps.mv( nps.cat(*[ project_one_cam(intrinsics_data[i].ravel(),
                                           vsplit[i])
                           for i in range(Nbroadcast)]),
                0, idim_broadcast )

def unproject(q, distortion_model, intrinsics_data):
    r'''Computes unit vector(s) corresponding to pixel observation(s)

    This function is broadcastable over q (using numpy primitives intead of
    nps.broadcast_define() to avoid a slow python broadcasting loop).

    This function is NOT broadcastable over the intrinsics

    Inputs:

    - q 2D pixel coordinate(s)

    - distortion_model: a string that says what the values in the intrinsics
      array mean. The supported values are reported by
      mrcal.getSupportedDistortionModels(). At the time of this
      writing, the supported values are

        DISTORTION_NONE
        DISTORTION_OPENCV4
        DISTORTION_OPENCV5
        DISTORTION_OPENCV8
        DISTORTION_OPENCV12 (if we have OpenCV >= 3.0.0)
        DISTORTION_OPENCV14 (if we have OpenCV >= 3.1.0)
        DISTORTION_CAHVOR
        DISTORTION_CAHVORE

    - intrinsics_data: a numpy array containing
      - fx
      - fy
      - cx
      - cy
      - distortion-specific values

    '''

    if q is None: return q
    if q.size == 0:
        s = q.shape
        return np.zeros(s[:-1] + (3,))

    fxy = intrinsics_data[ :2]
    cxy = intrinsics_data[2:4]

    if distortion_model == "DISTORTION_NONE":

        # undistorted q is the same as distorted q. I skip some steps

        # shape = (..., 2)
        vxy = (q - cxy) / fxy

    else:
        # undistort the q, by running an optimizer

        # I optimize each point separately because the internal optimization
        # algorithm doesn't know that each point is independent, so if I optimized
        # it all together, it would solve a dense linear system whose size is linear
        # in Npoints. The computation time thus would be much slower than
        # linear(Npoints)
        @nps.broadcast_define( ((2,),), )
        def undistort_this(q0):

            def cost_have_gradients(vxy):
                '''Optimization functions'''
                q,dqdxyz,_ = mrcal.project(np.array((vxy[0],vxy[1],1.)), distortion_model, intrinsics_data,
                                           get_gradients=True)
                return q - q0, dqdxyz[:,:2]
            def cost_no_gradients(vxy, *args, **kwargs):
                '''Optimization functions'''
                return \
                    mrcal.project(np.array((vxy[0],vxy[1],1.)), distortion_model, intrinsics_data) - \
                    q0

            # scipy.optimize.least_squares() has separate callbacks for the
            # function value and jacobians, while I compute them at the same
            # time. I have a caching mechanism to avoid recomputation
            undistort_this.cache_pfJ = [None,None,None]
            def f_callback_have_gradients(p, *args, **kwargs):
                p_cache = undistort_this.cache_pfJ[0]
                if p_cache is None or np.linalg.norm(p_cache - p) > 1e-12:
                    undistort_this.cache_pfJ[0]  = np.array(p)
                    undistort_this.cache_pfJ[1:] = cost_have_gradients(p)
                return undistort_this.cache_pfJ[1]
            def J_callback_have_gradients(p, *args, **kwargs):
                p_cache = undistort_this.cache_pfJ[0]
                if p_cache is None or np.linalg.norm(p_cache - p) > 1e-12:
                    undistort_this.cache_pfJ[0]  = np.array(p)
                    undistort_this.cache_pfJ[1:] = cost_have_gradients(p)
                return undistort_this.cache_pfJ[2]

            # seed assuming distortions aren't there
            vxy_seed = (q0 - cxy) / fxy
            if distortion_model == 'DISTORTION_CAHVORE':
                # no gradients available
                result = scipy.optimize.least_squares(cost_no_gradients, vxy_seed,
                                                      '3-point')
            else:
                result = scipy.optimize.least_squares(f_callback_have_gradients, vxy_seed,
                                                      J_callback_have_gradients)

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

    # normalize each vector
    return v / nps.dummy(np.sqrt(nps.inner(v,v)), -1)


def compute_scale_f_pinhole_for_fit(model, fit, scale_imagersize_pinhole = 1.0):
    r'''Compute best value of scale_f_pinhole for undistort_image()

    undistort_image() can produce undistorted images obtained with an arbitrary
    scale parameter. Different scaling values can either zoom in to cut off
    sections of the original image, or zoom out to waste pixels with no-data
    available at the edges. Here I try to compute optimal values for
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

    distortion_model,intrinsics_data = model.intrinsics()
    v_edges = mrcal.unproject(q_edges, distortion_model, intrinsics_data)

    # I have points in space now. My scaled pinhole camera would map these to
    # (k*fx*x/z+cx, fy*y/z+cy). I pick a k sure that this is in-bounds for all my
    # points, and that the points occupy as large a chunk of the imager as possible.
    # I can look at just the normalized x and y. Just one of the query points should
    # land on the edge; the rest should be in-bounds
    fxy = intrinsics_data[ :2]
    cxy = intrinsics_data[2:4]
    fx,fy,cx,cy  = intrinsics_data[:4]

    W1 = int(W*scale_imagersize_pinhole + 0.5)
    H1 = int(H*scale_imagersize_pinhole + 0.5)
    cx1 = cx / float(W - 1) * float(W1 - 1)
    cy1 = cy / float(H - 1) * float(H1 - 1)

    normxy_edges = v_edges[:,:2] / v_edges[:,(2,)]
    normxy_min   = (                               - np.array((cx1,cy1))) / fxy
    normxy_max   = (np.array((W1,H1), dtype=float) - 1.0 - np.array((cx1,cy1))) / fxy

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


def undistort_image__compute_map(model,
                                 scale_f_pinhole          = 1.0,
                                 scale_imagersize_pinhole = 1.0):
    r'''Computes a distortion map useful in undistort_image()

    Synopsis:

        mapxy, model_pinhole = mrcal.undistort_image__compute_map(model)
        image_undistorted =
            mrcal.undistort_image(model, imagefile, mapxy = mapxy)

    Returns a tuple:

    - mapxy: the undistortion map. This is a numpy array of shape
          (2,Nheight,Nwidth) because this fits into what cv2.remap() wants.

    - model_pinhole: the pinhole model used to construct this undistortion map.
          This model has the same extrinsics as the input model

    Results of this could be passed to undistort_image() in the "mapxy"
    argument. This routine uses opencv-specific functions if possible, for
    performance

    '''

    distortion_model,intrinsics_data = model.intrinsics()
    W,H                              = model.imagersize()
    fx,fy,cx,cy                      = intrinsics_data[ :4]


    W1 = int(W*scale_imagersize_pinhole + 0.5)
    H1 = int(H*scale_imagersize_pinhole + 0.5)
    output_shape = (W1, H1)

    cx1 = cx / float(W - 1) * float(W1 - 1)
    cy1 = cy / float(H - 1) * float(H1 - 1)
    fx1 = fx * scale_f_pinhole
    fy1 = fy * scale_f_pinhole

    if re.match("DISTORTION_OPENCV",distortion_model):
        # OpenCV models have a special-case path here. This works
        # identically to the other path (the other side of this "if" can
        # always be used instead), but the opencv-specific code is 100%
        # written in C (not Python) so it runs much faster
        distortion_coeffs = intrinsics_data[4: ]
        cameraMatrix     = np.array(((fx,  0, cx),
                                     ( 0, fy, cy),
                                     ( 0,  0,  1)))
        cameraMatrix_new = np.array(((fx1,   0, cx1),
                                     ( 0,  fy1, cy1),
                                     ( 0,    0,   1)))

        # opencv has cv2.undistort(), but this apparently only works if the
        # input and output dimensions match 100%. If they don't I see a
        # black image on output, which isn't useful. So I split this into
        # the map creation and the remapping. Which is fine. I can then
        # cache the map, and I can unify the remap code for all the model
        # types

        mapxy = nps.cat( *cv2.initUndistortRectifyMap(cameraMatrix, distortion_coeffs, None,
                                                      cameraMatrix_new, output_shape,
                                                      cv2.CV_32FC1) )
    else:

        # shape: Nwidth,Nheight,2
        grid  = np.ascontiguousarray(nps.reorder(nps.cat(*np.meshgrid(np.arange(W1),
                                                                      np.arange(H1))),
                                                 -1, -2, -3),
                                     dtype = float)
        mapxy = mrcal.distort(grid, distortion_model, intrinsics_data,
                              fx1,fy1,cx1,cy1)

        mapxy = nps.reorder(mapxy, -1, -2, -3).astype(np.float32)


    model_pinhole = mrcal.cameramodel( model )
    model_pinhole.intrinsics(np.array((W1,H1)),
                             ('DISTORTION_NONE', np.array((fx1,fy1,cx1,cy1))))
    return mapxy, model_pinhole


def annotate_image__valid_intrinsics_region(model, image):
    r'''Annotates a given image with a valid-intrinsics region

    This function takes in a camera model and an image, and returns a numpy
    array with the valid-intrinsics region drawn on top of the image. The image
    can be a filename or a numpy array. The camera model should contain the
    valid-intrinsics region; if not, the image is returned as is.

    This is similar to mrcal.show_valid_intrinsics_region(), but instead of
    making a plot, it creates an image

    '''

    if not isinstance(image, np.ndarray):
        image = cv2.imread(image)

    valid_intrinsics_region = model.valid_intrinsics_region()
    if valid_intrinsics_region is not None:
        cv2.polylines(image, [valid_intrinsics_region], True, (0,0,255), 3)
    return image


def undistort_image(model, image,
                    scale_f_pinhole          = 1.0,
                    scale_imagersize_pinhole = 1.0,
                    mapxy                    = None):
    r'''Removes the distortion from a given image

    Given an image and a distortion model (and optionally, a scaling), generates
    a new image that would be produced by the same scene, but with a perfect
    pinhole camera. This undistorted model is a pinhole camera with

    - the same center pixel coord as the distorted camera the distorted-camera
    - focal length scaled by a factor of scale_f_pinhole. Good values for this
    - can be obtained with compute_scale_f_pinhole_for_fit()

    The input image can be a filename or an array.

    Returns:

    - The undistorted image, in an array

    '''

    # Testing code that does opencv stuff more directly. Don't need it, but
    # could be handy for future debugging
    #
    # if re.match("DISTORTION_OPENCV",distortion_model):
    #     if not isinstance(image, np.ndarray):
    #         image = cv2.imread(image)
    #     fx,fy,cx,cy       = intrinsics_data[ :4]
    #     distortion_coeffs = intrinsics_data[4: ]
    #     cameraMatrix     = np.array(((fx,  0, cx),
    #                                  ( 0, fy, cy),
    #                                  ( 0,  0,  1)))
    #
    #     # # equivalent compute-map-and-undistort in a single call
    #     # remapped = np.zeros(image.shape, dtype = image.dtype)
    #     # cv2.undistort(image, cameraMatrix, distortion_coeffs, remapped, cameraMatrix)
    #
    #     mapx,mapy = cv2.initUndistortRectifyMap(cameraMatrix, distortion_coeffs, None,
    #                                             cameraMatrix, (W,H),
    #                                             cv2.CV_32FC1)
    #     return cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

    if mapxy is None:
        mapxy,_ = undistort_image__compute_map(model,
                                               scale_f_pinhole,
                                               scale_imagersize_pinhole)
    elif type(mapxy) is not np.ndarray:
        raise Exception('mapxy_cache MUST either be None or a numpy array')

    image = annotate_image__valid_intrinsics_region(model, image)

    return cv2.remap(image, mapxy[0], mapxy[1],
                     cv2.INTER_LINEAR)


def remap_via_atinfinity_homography(model0, model1, image0):
    r'''Remaps a given image into another model, assuming no translation

    Takes in

    - two camera models (filenames or objects)
    - an image captured with model0 (filename or array)

    Ignoring the translation between the two cameras produces an image of the
    same scene that would have been observed by camera1. If the intrinsics and
    the rotation were correct, this remapped camera0 image woudl match the
    camera1 image for objects infinitely-far away. This is thus a good
    validation function

    '''

    m = [mrcal.cameramodel(model0),
         mrcal.cameramodel(model1)]

    W,H = m[1].imagersize()
    distortion_model, intrinsics_data = m[1].intrinsics()

    R01 = nps.matmult( m[0].extrinsics_Rt_fromref()[:3,:],
                       m[1].extrinsics_Rt_toref  ()[:3,:] )

    # I want to do this:
    #   v,_ = mrcal.utils._sample_imager_unproject(W, H, distortion_model, intrinsics_data, W, H)
    # but that would involve a python loop that's REALLY slow, so I do this instead
    mapxy,m1_pinhole = undistort_image__compute_map(m[1])
    # shape (2,H,W) -> (H,W,2)
    mapxy = nps.mv(mapxy, 0, -1)
    v = mrcal.unproject( mapxy, *m1_pinhole.intrinsics())

    v = nps.matmult(R01, nps.dummy(v, -1))[..., 0]
    q = mrcal.project(v, *m[0].intrinsics())

    if not isinstance(image0, np.ndarray):
        image0 = cv2.imread(image0)

    return cv2.remap(image0,
                     q[..., 0].astype(np.float32),
                     q[..., 1].astype(np.float32),
                     cv2.INTER_LINEAR)


def calobservations_project(distortion_model, intrinsics, extrinsics, frames, dot_spacing, Nwant, calobject_warp):
    r'''Takes in the same arguments as mrcal.optimize(), and returns all
    the projections. Output has shape (Nframes,Ncameras,Nwant,Nwant,2)

    '''

    object_ref = mrcal.get_ref_calibration_object(Nwant, Nwant, dot_spacing, calobject_warp)
    Rf = mrcal.Rodrigues_toR_broadcasted(frames[:,:3])
    Rf = nps.mv(Rf,           0, -5)
    tf = nps.mv(frames[:,3:], 0, -5)

    # object in the cam0 coord system. shape=(Nframes, 1, Nwant, Nwant, 3)
    object_cam0 = nps.matmult( object_ref, nps.transpose(Rf)) + tf

    Rc = mrcal.Rodrigues_toR_broadcasted(extrinsics[:,:3])
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
    return nps.mv( nps.cat(*[project( object_cam[...,i_camera,:,:,:],
                                      distortion_model,
                                      intrinsics[i_camera,:] ) for \
                             i_camera in range(Ncameras)]),
                   0,-4)

def calobservations_compute_reproj_error(projected, observations, indices_frame_camera, Nwant,
                                         outlier_indices = None):
    r'''Computes reprojection errors when calibrating with board observations

    Given

    - projected (shape [Nframes,Ncameras,Nwant,Nwant,2])
    - observations (shape [Nframes,Nwant,Nwant,2])
    - indices_frame_camera (shape [Nobservations,2])
    - outlier_indices, a list of point indices that were deemed to be outliers.
      These are plain integers indexing the flattened observations array, but
      one per POINT, not (x,y) independently

    Return (err_all_points,err_ignoring_outliers). Each is the reprojection
    error for each point: shape [Nobservations,Nwant,Nwant,2]. One includes the
    outliers in the returned errors, and the other does not

    '''

    if outlier_indices is None: outlier_indices = np.array(())

    Nframes               = projected.shape[0]
    Nobservations         = indices_frame_camera.shape[0]
    err_all_points        = np.zeros((Nobservations,Nwant,Nwant,2))

    for i_observation in range(Nobservations):
        i_frame, i_camera = indices_frame_camera[i_observation]

        err_all_points[i_observation] = projected[i_frame,i_camera] - observations[i_observation]

    err_ignoring_outliers = err_all_points.copy()
    err_ignoring_outliers.ravel()[outlier_indices*2  ] = 0
    err_ignoring_outliers.ravel()[outlier_indices*2+1] = 0

    return err_all_points,err_ignoring_outliers

