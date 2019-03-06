#!/usr/bin/python2

import numpy as np
import numpysane as nps
import sys
import re
import cv2
import scipy.optimize

import mrcal
import _mrcal


def _undistort(q, distortion_model, intrinsics):
    r'''Un-apply a CAHVOR warp: undistort a point

    This is a model-generic function. We use the given distortion_model: a
    string that says what the values in 'distortions' mean. The supported values
    are defined in mrcal.h. At the time of this writing, the supported values
    are

      DISTORTION_NONE
      DISTORTION_OPENCV4
      DISTORTION_OPENCV5
      DISTORTION_OPENCV8
      DISTORTION_OPENCV12 (if we have OpenCV >= 3.0.0)
      DISTORTION_OPENCV14 (if we have OpenCV >= 3.1.0)
      DISTORTION_CAHVOR
      DISTORTION_CAHVORE

    Given intrinsic parameters of a model and a projected and distorted point(s)
    numpy array of shape (..., 2), return the projected point(s) that we'd get
    without distortion. We ASSUME THE SAME fx,fy,cx,cy

    This function can broadcast the points array.

    Note that this function has an iterative solver and is thus SLOW. This is
    the "backwards" direction. Most of the time you want distort().

    '''

    if q is None or q.size == 0: return q


    Ndistortions = mrcal.getNdistortionParams(distortion_model)
    if len(intrinsics)-4 != Ndistortions:
        raise Exception("Inconsistent distortion_model/values. Model '{}' expects {} distortion parameters, but got {} distortion values", distortion_model, Ndistortions, len(intrinsics)-4)

    if distortion_model == "DISTORTION_NONE":
        return q

    # I could make this much more efficient: precompute lots of stuff, use
    # gradients, etc, etc. This functions decently well and I leave it.

    # I optimize each point separately because the internal optimization
    # algorithm doesn't know that each point is independent, so if I optimized
    # it all together, it would solve a dense linear system whose size is linear
    # in Npoints. The computation time thus would be much slower than
    # linear(Npoints)
    @nps.broadcast_define( ((2,),), )
    def undistort_this(q0):
        def f(qundistorted):
            '''Optimization functions'''
            qdistorted = mrcal.distort(qundistorted, distortion_model, intrinsics)
            return qdistorted - q0
        q1 = scipy.optimize.leastsq(f, q0)[0]
        return np.array(q1)

    return undistort_this(q)

# @nps.broadcast_define( ((3,),('Nintrinsics',)),
#                        (2,), )
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
    - (...,2,Nintrinsics) array of the gradients of the pixel coordinates in
      respect to the intrinsics
    - (...,2,3) array of the gradients of the pixel coordinates in respect to
      the input 3D point positions

    '''

    if v is None: return v
    if v.size == 0:
        if get_gradients:
            Nintrinsics = intrinsics_data.shape[-1]
            s = v.shape
            return np.zeros(s[:-1] + (2,)), np.zeros(s[:-1] + (2,Nintrinsics)), np.zeros(s[:-1] + (2,3))
        else:
            s = v.shape
            return np.zeros(s[:-1] + (2,))

    if distortion_model != 'DISTORTION_CAHVORE':
        return _mrcal.project(np.ascontiguousarray(v),
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
    idims_not1 = [ i for i in xrange(len(intrinsics_data.shape)-1) if intrinsics_data.shape[i] != 1 ]
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
                           for i in xrange(Nbroadcast)]),
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

    q = _undistort(q, distortion_model, intrinsics_data)

    (fx, fy, cx, cy) = intrinsics_data[:4]

    # shape = (..., 2)
    v = (q - np.array((cx,cy))) / np.array((fx,fy))

    # I append a 1. shape = (..., 3)
    v = nps.glue(v, np.ones( v.shape[:-1] + (1,) ), axis=-1)

    # normalize each vector
    return v / nps.dummy(np.sqrt(nps.inner(v,v)), -1)

def distortion_map__to_warped(distortion_model, intrinsics_data, w, h, scale_f_pinhole=1.0):
    r'''Returns the pre and post distortion map of a model

    Takes in

    - a cahvor model (in a number of representations)
    - a sampling spacing on the x axis
    - a sampling spacing on the y axis

    Produces two arrays of shape Nwidth,Nheight,2:

    - grid: a meshgrid of the sampling spacings given as inputs. This refers to the
      UNDISTORTED image

    - dgrid: a meshgrid where each point in grid had the distortion corrected.
      This refers to the DISTORTED image

    '''

    # shape: Nwidth,Nheight,2
    grid  = np.ascontiguousarray(nps.reorder(nps.cat(*np.meshgrid(w,h)), -1, -2, -3),
                                 dtype = float)
    dgrid = mrcal.distort(grid, distortion_model, intrinsics_data, scale_f_pinhole)

    return grid, dgrid

def undistort_image(model, image, scale_f_pinhole=1.0):
    r'''Removes the distortion from a given image

    Given an image and a distortion model (and optionally, a scaling), generates
    a new image that would be produced by the same scene, but with a perfect
    pinhole camera. This undistorted model is a pinhole camera with

    - the same center pixel coord as the distorted camera
    - the distorted-camera focal length scaled by a factor of scale_f_pinhole

    The input image can be a filename or an array.

    Returns:

    - The undistorted image, in an array

    '''

    distortion_model,intrinsics_data = model.intrinsics()

    if not isinstance(image, np.ndarray):
        image = cv2.imread(image)

    if re.match("DISTORTION_OPENCV",distortion_model):
        # OpenCV models have a special-case path here. This works identically to
        # the other path (this "if" block can be removed entirely), but the
        # opencv-specific code is 100% written in C (not Python) so it runs much
        # faster
        fx,fy,cx,cy       = intrinsics_data[ :4]
        distortion_coeffs = intrinsics_data[4: ]
        cameraMatrix     = np.array(((fx,  0, cx),
                                     ( 0, fy, cy),
                                     ( 0,  0,  1)))
        fx *= scale_f_pinhole
        fy *= scale_f_pinhole
        cameraMatrix_new = np.array(((fx,  0, cx),
                                     ( 0, fy, cy),
                                     ( 0,  0,  1)))
        remapped = np.zeros(image.shape, dtype = image.dtype)
        cv2.undistort(image, cameraMatrix, distortion_coeffs, remapped, cameraMatrix_new)
        return remapped

    H,W = image.shape[:2]

    _,mapxy = distortion_map__to_warped(distortion_model,intrinsics_data,
                                        np.arange(W), np.arange(H),
                                        scale_f_pinhole = scale_f_pinhole)
    mapx = mapxy[:,:,0].astype(np.float32)
    mapy = mapxy[:,:,1].astype(np.float32)

    remapped = cv2.remap(image,
                         nps.transpose(mapx),
                         nps.transpose(mapy),
                         cv2.INTER_LINEAR)
    return remapped

def calobservations_project(distortion_model, intrinsics, extrinsics, frames, dot_spacing, Nwant):
    r'''Takes in the same arguments as mrcal.optimize(), and returns all
    the projections. Output has shape (Nframes,Ncameras,Nwant,Nwant,2)

    '''

    object_ref = mrcal.get_ref_calibration_object(Nwant, Nwant, dot_spacing)
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
                             i_camera in xrange(Ncameras)]),
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

    for i_observation in xrange(Nobservations):
        i_frame, i_camera = indices_frame_camera[i_observation]

        err_all_points[i_observation] = projected[i_frame,i_camera] - observations[i_observation]

    err_ignoring_outliers = err_all_points.copy()
    err_ignoring_outliers.ravel()[outlier_indices*2  ] = 0
    err_ignoring_outliers.ravel()[outlier_indices*2+1] = 0

    return err_all_points,err_ignoring_outliers

