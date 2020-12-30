#!/usr/bin/python3

'''Routines for analysis of camera projection

This is largely dealing with uncertainty and projection diff operations.

All functions are exported into the mrcal module. So you can call these via
mrcal.model_analysis.fff() or mrcal.fff(). The latter is preferred.

'''

import numpy as np
import numpysane as nps
import sys
import mrcal
import mrcal._mrcal_npsp

def _parse_args(v1,
                t01,
                get_gradients,
                v_are_local,
                Rt01):
    r'''Parse arguments to triangulation functions that take camera-0-referenced v
    AND t01'''

    if Rt01 is not None and t01 is not None:
        raise Exception("Exactly one of Rt01 and t01 must be None. Both were non-None")

    if Rt01 is None     and t01 is None:
        raise Exception("Exactly one of Rt01 and t01 must be None. Both were None")

    if v_are_local:
        if get_gradients:
            raise Exception("get_gradients is True, so v_are_local MUST be the default: False")
        if Rt01 is None:
            raise Exception("v_are_local is True, so Rt01 MUST have been given")
        v1 = mrcal.rotate_point_R(Rt01[:3,:], v1)
        t01 = Rt01[3,:]
    else:
        # Normal path
        if t01 is None:
            t01 = Rt01[3,:]
            if get_gradients:
                raise Exception("get_gradients is True, so t01 MUST have been given")
        else:
            # Normal path
            pass

    return v1, t01


def triangulate_geometric(v0, v1,
                          t01           = None,
                          get_gradients = False,
                          v_are_local   = False,
                          Rt01          = None):

    r'''Simple geometric triangulation

SYNOPSIS

    models = ( mrcal.cameramodel('cam0.cameramodel'),
               mrcal.cameramodel('cam1.cameramodel') )

    images = (cv2.imread('image0.jpg', cv2.IMREAD_GRAYSCALE),
              cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE))

    Rt01 = mrcal.compose_Rt( models[0].extrinsics_Rt_fromref(),
                             models[1].extrinsics_Rt_toref() )

    R01 = Rt01[:3,:]
    t01 = Rt01[ 3,:]

    # pixel observation in camera0
    q0 = np.array((1233, 2433), dtype=np.float32)

    # corresponding pixel observation in camera1
    q1, _ = \
        mrcal.match_feature( *images,
                             template_size = (17,17),
                             method        = cv2.TM_CCORR_NORMED,
                             search_radius = 20,
                             q0            = q0,
                             H10           = H10, # homography mapping q0 to q1
                           )

    v0 = mrcal.unproject(q0, *models[0].intrinsics())
    v1 = mrcal.rotate_point_R(R01, mrcal.unproject(q1, *models[1].intrinsics()))

    # Estimated 3D position in camera-0 coordinates of the feature observed in
    # the two cameras
    p = mrcal.triangulate_geometric( v0, v1, t01 )

This function implements a very simple closest-approach-in-3D routine. It finds
the point on each ray that's nearest to the other ray, and returns the mean of
these two points.

If the triangulated point lies behind either camera (i.e. if the observation
rays are parallel or divergent), (0,0,0) is returned.

This routine is provided because it is simple and because it's useful for
testing the other routines. Use the other mrcal.triangulate_...() routines for
processing real data.

This function supports broadcasting fully.

By default, this function takes a translation t01 instead of a full
transformation Rt01. This is consistent with most, but not all of the
triangulation routines. For API compatibility with ALL triangulation routines,
the full Rt01 may be passed as a kwarg.

Also, by default this function takes v1 in the camera-0-local coordinate system
like most, but not all the other triangulation routines. If v_are_local: then v1
is interpreted in the camera-1 coordinate system instead. This makes it simple
to compare the triangulation routines against one another.

The invocation compatible across all the triangulation routines omits t01, and
passes Rt01 and v_are_local:

  triangulate_...( v0, v1,
                   Rt01        = Rt01,
                   v_are_local = False )

Gradient reporting is possible in the default case of Rt01 is None and not
v_are_local.

ARGUMENTS

- v0: (3,) numpy array containing a not-necessarily-normalized observation
  vector of a feature observed in camera-0, described in the camera-0 coordinate
  system

- v1: (3,) numpy array containing a not-necessarily-normalized observation
  vector of a feature observed in camera-1, described in the camera-0 coordinate
  system. Note that this vector is represented in the SAME coordinate system as
  v0

- t01: (3,) numpy array containing the position of the camera-1 origin in the
  camera-0 coordinate system.

- get_gradients: optional boolean that defaults to False. Whether we should
  compute and report the gradients. This affects what we return

- v_are_local: optional boolean that defaults to False. If True: v1 is
  represented in the local coordinate system of camera-1. The default is
  consistent with most, but not all of the triangulation routines.

- Rt01: optional (4,3) numpy array, defaulting to None. If given, we use this
  transformation from camera-1 coordinates to camera-0 coordinates instead of
  t01. This exists for API compatibility with the other triangulation routines.

RETURNED VALUE

if not get_gradients:

  we return an (...,3) array of triangulated point positions in the camera-0
  coordinate system

if get_gradients: we return a tuple:

  - (...,3) array of triangulated point positions
  - (...,3,3) array of the gradients of the triangulated positions in respect to
    v0
  - (...,3,3) array of the gradients of the triangulated positions in respect to
    v1
  - (...,3,3) array of the gradients of the triangulated positions in respect to
    t01

    '''

    v1, t01 = _parse_args(v1, t01,
                          get_gradients, v_are_local, Rt01)

    if not get_gradients:
        return mrcal._mrcal_npsp._triangulate_geometric(v0, v1, t01)
    else:
        return mrcal._mrcal_npsp._triangulate_geometric_withgrad(v0, v1, t01)


def triangulate_leecivera_l1(v0, v1,
                             t01           = None,
                             get_gradients = False,
                             v_are_local   = False,
                             Rt01          = None):

    r'''Triangulation minimizing the L1-norm of angle differences

SYNOPSIS

    models = ( mrcal.cameramodel('cam0.cameramodel'),
               mrcal.cameramodel('cam1.cameramodel') )

    images = (cv2.imread('image0.jpg', cv2.IMREAD_GRAYSCALE),
              cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE))

    Rt01 = mrcal.compose_Rt( models[0].extrinsics_Rt_fromref(),
                             models[1].extrinsics_Rt_toref() )

    R01 = Rt01[:3,:]
    t01 = Rt01[ 3,:]

    # pixel observation in camera0
    q0 = np.array((1233, 2433), dtype=np.float32)

    # corresponding pixel observation in camera1
    q1, _ = \
        mrcal.match_feature( *images,
                             template_size = (17,17),
                             method        = cv2.TM_CCORR_NORMED,
                             search_radius = 20,
                             q0            = q0,
                             H10           = H10, # homography mapping q0 to q1
                           )

    v0 = mrcal.unproject(q0, *models[0].intrinsics())
    v1 = mrcal.rotate_point_R(R01, mrcal.unproject(q1, *models[1].intrinsics()))

    # Estimated 3D position in camera-0 coordinates of the feature observed in
    # the two cameras
    p = mrcal.triangulate_leecivera_l1( v0, v1, t01 )

This function implements a triangulation routine minimizing the L1 norm of
angular errors. This is described in

  "Closed-Form Optimal Two-View Triangulation Based on Angular Errors", Seong
  Hun Lee and Javier Civera. ICCV 2019.

If the triangulated point lies behind either camera (i.e. if the observation
rays are parallel or divergent), (0,0,0) is returned.

This function supports broadcasting fully.

By default, this function takes a translation t01 instead of a full
transformation Rt01. This is consistent with most, but not all of the
triangulation routines. For API compatibility with ALL triangulation routines,
the full Rt01 may be passed as a kwarg.

Also, by default this function takes v1 in the camera-0-local coordinate system
like most, but not all the other triangulation routines. If v_are_local: then v1
is interpreted in the camera-1 coordinate system instead. This makes it simple
to compare the triangulation routines against one another.

The invocation compatible across all the triangulation routines omits t01, and
passes Rt01 and v_are_local:

  triangulate_...( v0, v1,
                   Rt01        = Rt01,
                   v_are_local = False )

Gradient reporting is possible in the default case of Rt01 is None and not
v_are_local.

ARGUMENTS

- v0: (3,) numpy array containing a not-necessarily-normalized observation
  vector of a feature observed in camera-0, described in the camera-0 coordinate
  system

- v1: (3,) numpy array containing a not-necessarily-normalized observation
  vector of a feature observed in camera-1, described in the camera-0 coordinate
  system. Note that this vector is represented in the SAME coordinate system as
  v0

- t01: (3,) numpy array containing the position of the camera-1 origin in the
  camera-0 coordinate system.

- get_gradients: optional boolean that defaults to False. Whether we should
  compute and report the gradients. This affects what we return

- v_are_local: optional boolean that defaults to False. If True: v1 is
  represented in the local coordinate system of camera-1. The default is
  consistent with most, but not all of the triangulation routines.

- Rt01: optional (4,3) numpy array, defaulting to None. If given, we use this
  transformation from camera-1 coordinates to camera-0 coordinates instead of
  t01. This exists for API compatibility with the other triangulation routines.

RETURNED VALUE

if not get_gradients:

  we return an (...,3) array of triangulated point positions in the camera-0
  coordinate system

if get_gradients: we return a tuple:

  - (...,3) array of triangulated point positions
  - (...,3,3) array of the gradients of the triangulated positions in respect to
    v0
  - (...,3,3) array of the gradients of the triangulated positions in respect to
    v1
  - (...,3,3) array of the gradients of the triangulated positions in respect to
    t01

    '''

    v1, t01 = _parse_args(v1, t01,
                          get_gradients, v_are_local, Rt01)

    if not get_gradients:
        return mrcal._mrcal_npsp._triangulate_leecivera_l1(v0, v1, t01)
    else:
        return mrcal._mrcal_npsp._triangulate_leecivera_l1_withgrad(v0, v1, t01)


def triangulate_leecivera_linf(v0, v1,
                               t01           = None,
                               get_gradients = False,
                               v_are_local   = False,
                               Rt01          = None):

    r'''Triangulation minimizing the infinity-norm of angle differences

SYNOPSIS

    models = ( mrcal.cameramodel('cam0.cameramodel'),
               mrcal.cameramodel('cam1.cameramodel') )

    images = (cv2.imread('image0.jpg', cv2.IMREAD_GRAYSCALE),
              cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE))

    Rt01 = mrcal.compose_Rt( models[0].extrinsics_Rt_fromref(),
                             models[1].extrinsics_Rt_toref() )

    R01 = Rt01[:3,:]
    t01 = Rt01[ 3,:]

    # pixel observation in camera0
    q0 = np.array((1233, 2433), dtype=np.float32)

    # corresponding pixel observation in camera1
    q1, _ = \
        mrcal.match_feature( *images,
                             template_size = (17,17),
                             method        = cv2.TM_CCORR_NORMED,
                             search_radius = 20,
                             q0            = q0,
                             H10           = H10, # homography mapping q0 to q1
                           )

    v0 = mrcal.unproject(q0, *models[0].intrinsics())
    v1 = mrcal.rotate_point_R(R01, mrcal.unproject(q1, *models[1].intrinsics()))

    # Estimated 3D position in camera-0 coordinates of the feature observed in
    # the two cameras
    p = mrcal.triangulate_leecivera_linf( v0, v1, t01 )

This function implements a triangulation routine minimizing the infinity norm of
angular errors (it minimizes the larger of the two angle errors). This is
described in

  "Closed-Form Optimal Two-View Triangulation Based on Angular Errors", Seong
  Hun Lee and Javier Civera. ICCV 2019.

If the triangulated point lies behind either camera (i.e. if the observation
rays are parallel or divergent), (0,0,0) is returned.

This function supports broadcasting fully.

By default, this function takes a translation t01 instead of a full
transformation Rt01. This is consistent with most, but not all of the
triangulation routines. For API compatibility with ALL triangulation routines,
the full Rt01 may be passed as a kwarg.

Also, by default this function takes v1 in the camera-0-local coordinate system
like most, but not all the other triangulation routines. If v_are_local: then v1
is interpreted in the camera-1 coordinate system instead. This makes it simple
to compare the triangulation routines against one another.

The invocation compatible across all the triangulation routines omits t01, and
passes Rt01 and v_are_local:

  triangulate_...( v0, v1,
                   Rt01        = Rt01,
                   v_are_local = False )

Gradient reporting is possible in the default case of Rt01 is None and not
v_are_local.

ARGUMENTS

- v0: (3,) numpy array containing a not-necessarily-normalized observation
  vector of a feature observed in camera-0, described in the camera-0 coordinate
  system

- v1: (3,) numpy array containing a not-necessarily-normalized observation
  vector of a feature observed in camera-1, described in the camera-0 coordinate
  system. Note that this vector is represented in the SAME coordinate system as
  v0

- t01: (3,) numpy array containing the position of the camera-1 origin in the
  camera-0 coordinate system.

- get_gradients: optional boolean that defaults to False. Whether we should
  compute and report the gradients. This affects what we return

- v_are_local: optional boolean that defaults to False. If True: v1 is
  represented in the local coordinate system of camera-1. The default is
  consistent with most, but not all of the triangulation routines.

- Rt01: optional (4,3) numpy array, defaulting to None. If given, we use this
  transformation from camera-1 coordinates to camera-0 coordinates instead of
  t01. This exists for API compatibility with the other triangulation routines.

RETURNED VALUE

if not get_gradients:

  we return an (...,3) array of triangulated point positions in the camera-0
  coordinate system

if get_gradients: we return a tuple:

  - (...,3) array of triangulated point positions
  - (...,3,3) array of the gradients of the triangulated positions in respect to
    v0
  - (...,3,3) array of the gradients of the triangulated positions in respect to
    v1
  - (...,3,3) array of the gradients of the triangulated positions in respect to
    t01

    '''

    v1, t01 = _parse_args(v1, t01,
                          get_gradients, v_are_local, Rt01)

    if not get_gradients:
        return mrcal._mrcal_npsp._triangulate_leecivera_linf(v0, v1, t01)
    else:
        return mrcal._mrcal_npsp._triangulate_leecivera_linf_withgrad(v0, v1, t01)


def triangulate_lindstrom(v0, v1, Rt01,
                          get_gradients = False,
                          v_are_local   = True):

    r'''Triangulation minimizing the 2-norm of pinhole reprojection errors

SYNOPSIS

    models = ( mrcal.cameramodel('cam0.cameramodel'),
               mrcal.cameramodel('cam1.cameramodel') )

    images = (cv2.imread('image0.jpg', cv2.IMREAD_GRAYSCALE),
              cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE))

    Rt01 = mrcal.compose_Rt( models[0].extrinsics_Rt_fromref(),
                             models[1].extrinsics_Rt_toref() )

    R01 = Rt01[:3,:]
    t01 = Rt01[ 3,:]

    # pixel observation in camera0
    q0 = np.array((1233, 2433), dtype=np.float32)

    # corresponding pixel observation in camera1
    q1, _ = \
        mrcal.match_feature( *images,
                             template_size = (17,17),
                             method        = cv2.TM_CCORR_NORMED,
                             search_radius = 20,
                             q0            = q0,
                             H10           = H10, # homography mapping q0 to q1
                           )

    # observation vectors in the LOCAL coordinate system of the two cameras
    v0 = mrcal.unproject(q0, *models[0].intrinsics())
    v1 = mrcal.unproject(q1, *models[1].intrinsics())

    # Estimated 3D position in camera-0 coordinates of the feature observed in
    # the two cameras
    p = mrcal.triangulate_lindstrom( v0, v1, Rt01 )

This function implements a triangulation routine minimizing the 2-norm of
reprojection errors, ASSUMING a pinhole projection. This is described in

  "Triangulation Made Easy", Peter Lindstrom, IEEE Conference on Computer Vision
  and Pattern Recognition, 2010.

The assumption of a pinhole projection is a poor one when using a wide lens, and
looking away from the optical center. The Lee-Civera triangulation functions
don't have this problem, and are generally faster. See the Lee, Civera paper for
details.

If the triangulated point lies behind either camera (i.e. if the observation
rays are parallel or divergent), (0,0,0) is returned.

This function supports broadcasting fully.

This function takes a full transformation Rt01, instead of t01 like most of the
other triangulation functions do by default. The other function may take Rt01,
for API compatibility.

Also, by default this function takes v1 in the camera-1-local coordinate system
unlike most of the other triangulation routines. If not v_are_local: then v1 is
interpreted in the camera-0 coordinate system instead. This makes it simple to
compare the routines against one another.

The invocation compatible across all the triangulation routines omits t01, and
passes Rt01 and v_are_local:

  triangulate_...( v0, v1,
                   Rt01        = Rt01,
                   v_are_local = False )

Gradient reporting is possible in the default case of v_are_local is True

ARGUMENTS

- v0: (3,) numpy array containing a not-necessarily-normalized observation
  vector of a feature observed in camera-0, described in the camera-0 coordinate
  system

- v1: (3,) numpy array containing a not-necessarily-normalized observation
  vector of a feature observed in camera-1, described in the camera-1 coordinate
  system. Note that this vector is represented in the camera-local coordinate
  system, unlike the representation in all the other triangulation routines

- Rt01: (4,3) numpy array describing the transformation from camera-1
  coordinates to camera-0 coordinates

- get_gradients: optional boolean that defaults to False. Whether we should
  compute and report the gradients. This affects what we return

- v_are_local: optional boolean that defaults to True. If True: v1 is
  represented in the local coordinate system of camera-1. This is different from
  the other triangulation routines. Set v_are_local to False to make this
  function interpret v1 similarly to the other triangulation routines

RETURNED VALUE

if not get_gradients:

  we return an (...,3) array of triangulated point positions in the camera-0
  coordinate system

if get_gradients: we return a tuple:

  - (...,3) array of triangulated point positions
  - (...,3,3) array of the gradients of the triangulated positions in respect to
    v0
  - (...,3,3) array of the gradients of the triangulated positions in respect to
    v1
  - (...,3,4,3) array of the gradients of the triangulated positions in respect
    to Rt01

    '''

    if not v_are_local:
        if get_gradients:
            raise Exception("get_gradients is True, so v_are_local MUST be True")
        v1 = mrcal.rotate_point_R(nps.transpose(Rt01[:3,:]), v1)

    if not get_gradients:
        return mrcal._mrcal_npsp._triangulate_lindstrom(v0, v1, Rt01)
    else:
        return mrcal._mrcal_npsp._triangulate_lindstrom_withgrad(v0, v1, Rt01)
