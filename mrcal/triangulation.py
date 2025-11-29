#!/usr/bin/python3

# Copyright (c) 2017-2023 California Institute of Technology ("Caltech"). U.S.
# Government sponsorship acknowledged. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

'''Triangulation routines

Various ways to convert two rays in 3D into a 3D point those rays represent

All functions are exported into the mrcal module. So you can call these via
mrcal.triangulation.fff() or mrcal.fff(). The latter is preferred.

'''

import numpy as np
import numpysane as nps
import sys
import mrcal
import mrcal._triangulation_npsp
import mrcal.model_analysis

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
        v1 = mrcal.rotate_point_R(Rt01[...,:3,:], v1)
        t01 = Rt01[...,3,:]
    else:
        # Normal path
        if t01 is None:
            t01 = Rt01[...,3,:]
            if get_gradients:
                raise Exception("get_gradients is True, so t01 MUST have been given")
        else:
            # Normal path
            pass

    return v1, t01


def triangulate_geometric(v0, v1,
                          t01           = None,
                          *,
                          get_gradients = False,
                          v_are_local   = False,
                          Rt01          = None,
                          out           = None):

    r'''Simple geometric triangulation

SYNOPSIS

    models = ( mrcal.cameramodel('cam0.cameramodel'),
               mrcal.cameramodel('cam1.cameramodel') )

    images = (mrcal.load_image('image0.jpg', bits_per_pixel=8, channels=1),
              mrcal.load_image('image1.jpg', bits_per_pixel=8, channels=1))

    Rt01 = mrcal.compose_Rt( models[0].Rt_cam_ref(),
                             models[1].Rt_ref_cam() )

    # pixel observation in camera0
    q0 = np.array((1233, 2433), dtype=np.float32)

    # corresponding pixel observation in camera1
    q1, _ = \
        mrcal.match_feature( *images,
                             q0            = q0,
                             template_size = (17,17),
                             method        = cv2.TM_CCORR_NORMED,
                             search_radius = 20,
                             H10           = H10, # homography mapping q0 to q1
                           )

    v0 = mrcal.unproject(q0, *models[0].intrinsics())
    v1 = mrcal.unproject(q1, *models[1].intrinsics())

    # Estimated 3D position in camera-0 coordinates of the feature observed in
    # the two cameras
    p = mrcal.triangulate_geometric( v0, v1,
                                     Rt01        = Rt01,
                                     v_are_local = True )

This is the lower-level triangulation routine. For a richer function that can be
used to propagate uncertainties, see mrcal.triangulate()

This function implements a very simple closest-approach-in-3D routine. It finds
the point on each ray that's nearest to the other ray, and returns the mean of
these two points. This is the "Mid" method in the paper

  "Triangulation: Why Optimize?", Seong Hun Lee and Javier Civera.
  https://arxiv.org/abs/1907.11917

This paper compares many methods. This routine is simplest and fastest, but it
has the highest errors.

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
  v0 in the default case (v_are_local is False)

- t01: (3,) numpy array containing the position of the camera-1 origin in the
  camera-0 coordinate system. Exclusive with Rt01.

- get_gradients: optional boolean that defaults to False. Whether we should
  compute and report the gradients. This affects what we return. If
  get_gradients: v_are_local must have the default value

- v_are_local: optional boolean that defaults to False. If True: v1 is
  represented in the local coordinate system of camera-1. The default is
  consistent with most, but not all of the triangulation routines. Must have the
  default value if get_gradients

- Rt01: optional (4,3) numpy array, defaulting to None. Exclusive with t01. If
  given, we use this transformation from camera-1 coordinates to camera-0
  coordinates instead of t01. If v_are_local: then Rt01 MUST be given instead of
  t01. This exists for API compatibility with the other triangulation routines.

- out: optional argument specifying the destination. By default, new numpy
  array(s) are created and returned. To write the results into existing (and
  possibly non-contiguous) arrays, specify them with the 'out' kwarg. If not
  get_gradients: 'out' is the one numpy array we will write into. Else: 'out' is
  a tuple of all the output numpy arrays. If 'out' is given, we return the 'out'
  that was passed in. This is the standard behavior provided by
  numpysane_pywrap.

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
        return mrcal._triangulation_npsp._triangulate_geometric(v0, v1, t01, out=out)
    else:
        return mrcal._triangulation_npsp._triangulate_geometric_withgrad(v0, v1, t01, out=out)


def triangulate_leecivera_l1(v0, v1,
                             t01           = None,
                             *,
                             get_gradients = False,
                             v_are_local   = False,
                             Rt01          = None,
                             out           = None):

    r'''Triangulation minimizing the L1-norm of angle differences

SYNOPSIS

    models = ( mrcal.cameramodel('cam0.cameramodel'),
               mrcal.cameramodel('cam1.cameramodel') )

    images = (mrcal.load_image('image0.jpg', bits_per_pixel=8, channels=1),
              mrcal.load_image('image1.jpg', bits_per_pixel=8, channels=1))

    Rt01 = mrcal.compose_Rt( models[0].Rt_cam_ref(),
                             models[1].Rt_ref_cam() )

    # pixel observation in camera0
    q0 = np.array((1233, 2433), dtype=np.float32)

    # corresponding pixel observation in camera1
    q1, _ = \
        mrcal.match_feature( *images,
                             q0            = q0,
                             template_size = (17,17),
                             method        = cv2.TM_CCORR_NORMED,
                             search_radius = 20,
                             H10           = H10, # homography mapping q0 to q1
                           )

    v0 = mrcal.unproject(q0, *models[0].intrinsics())
    v1 = mrcal.unproject(q1, *models[1].intrinsics())

    # Estimated 3D position in camera-0 coordinates of the feature observed in
    # the two cameras
    p = mrcal.triangulate_leecivera_l1( v0, v1,
                                        Rt01        = Rt01,
                                        v_are_local = True )

This is the lower-level triangulation routine. For a richer function that can be
used to propagate uncertainties, see mrcal.triangulate()

This function implements a triangulation routine minimizing the L1 norm of
angular errors. This is described in

  "Closed-Form Optimal Two-View Triangulation Based on Angular Errors", Seong
  Hun Lee and Javier Civera. ICCV 2019.

This is the "L1 ang" method in the paper

  "Triangulation: Why Optimize?", Seong Hun Lee and Javier Civera.
  https://arxiv.org/abs/1907.11917

This paper compares many methods. This routine works decently well, but it isn't
the best. triangulate_leecivera_mid2() (or triangulate_leecivera_wmid2() if
we're near the cameras) are preferred, according to the paper.

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
  v0 in the default case (v_are_local is False)

- t01: (3,) numpy array containing the position of the camera-1 origin in the
  camera-0 coordinate system. Exclusive with Rt01.

- get_gradients: optional boolean that defaults to False. Whether we should
  compute and report the gradients. This affects what we return. If
  get_gradients: v_are_local must have the default value

- v_are_local: optional boolean that defaults to False. If True: v1 is
  represented in the local coordinate system of camera-1. The default is
  consistent with most, but not all of the triangulation routines. Must have the
  default value if get_gradients

- Rt01: optional (4,3) numpy array, defaulting to None. Exclusive with t01. If
  given, we use this transformation from camera-1 coordinates to camera-0
  coordinates instead of t01. If v_are_local: then Rt01 MUST be given instead of
  t01. This exists for API compatibility with the other triangulation routines.

- out: optional argument specifying the destination. By default, new numpy
  array(s) are created and returned. To write the results into existing (and
  possibly non-contiguous) arrays, specify them with the 'out' kwarg. If not
  get_gradients: 'out' is the one numpy array we will write into. Else: 'out' is
  a tuple of all the output numpy arrays. If 'out' is given, we return the 'out'
  that was passed in. This is the standard behavior provided by
  numpysane_pywrap.

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
        return mrcal._triangulation_npsp._triangulate_leecivera_l1(v0, v1, t01, out=out)
    else:
        return mrcal._triangulation_npsp._triangulate_leecivera_l1_withgrad(v0, v1, t01, out=out)


def triangulate_leecivera_linf(v0, v1,
                               t01           = None,
                               *,
                               get_gradients = False,
                               v_are_local   = False,
                               Rt01          = None,
                               out           = None):

    r'''Triangulation minimizing the infinity-norm of angle differences

SYNOPSIS

    models = ( mrcal.cameramodel('cam0.cameramodel'),
               mrcal.cameramodel('cam1.cameramodel') )

    images = (mrcal.load_image('image0.jpg', bits_per_pixel=8, channels=1),
              mrcal.load_image('image1.jpg', bits_per_pixel=8, channels=1))

    Rt01 = mrcal.compose_Rt( models[0].Rt_cam_ref(),
                             models[1].Rt_ref_cam() )

    # pixel observation in camera0
    q0 = np.array((1233, 2433), dtype=np.float32)

    # corresponding pixel observation in camera1
    q1, _ = \
        mrcal.match_feature( *images,
                             q0            = q0,
                             template_size = (17,17),
                             method        = cv2.TM_CCORR_NORMED,
                             search_radius = 20,
                             H10           = H10, # homography mapping q0 to q1
                           )

    v0 = mrcal.unproject(q0, *models[0].intrinsics())
    v1 = mrcal.unproject(q1, *models[1].intrinsics())

    # Estimated 3D position in camera-0 coordinates of the feature observed in
    # the two cameras
    p = mrcal.triangulate_leecivera_linf( v0, v1,
                                          Rt01        = Rt01,
                                          v_are_local = True )

This is the lower-level triangulation routine. For a richer function that can be
used to propagate uncertainties, see mrcal.triangulate()

This function implements a triangulation routine minimizing the infinity norm of
angular errors (it minimizes the larger of the two angle errors). This is
described in

  "Closed-Form Optimal Two-View Triangulation Based on Angular Errors", Seong
  Hun Lee and Javier Civera. ICCV 2019.

This is the "L-infinity ang" method in the paper

  "Triangulation: Why Optimize?", Seong Hun Lee and Javier Civera.
  https://arxiv.org/abs/1907.11917

This paper compares many methods. This routine works decently well, but it isn't
the best. triangulate_leecivera_mid2() (or triangulate_leecivera_wmid2() if
we're near the cameras) are preferred, according to the paper.

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
  v0 in the default case (v_are_local is False)

- t01: (3,) numpy array containing the position of the camera-1 origin in the
  camera-0 coordinate system. Exclusive with Rt01.

- get_gradients: optional boolean that defaults to False. Whether we should
  compute and report the gradients. This affects what we return. If
  get_gradients: v_are_local must have the default value

- v_are_local: optional boolean that defaults to False. If True: v1 is
  represented in the local coordinate system of camera-1. The default is
  consistent with most, but not all of the triangulation routines. Must have the
  default value if get_gradients

- Rt01: optional (4,3) numpy array, defaulting to None. Exclusive with t01. If
  given, we use this transformation from camera-1 coordinates to camera-0
  coordinates instead of t01. If v_are_local: then Rt01 MUST be given instead of
  t01. This exists for API compatibility with the other triangulation routines.

- out: optional argument specifying the destination. By default, new numpy
  array(s) are created and returned. To write the results into existing (and
  possibly non-contiguous) arrays, specify them with the 'out' kwarg. If not
  get_gradients: 'out' is the one numpy array we will write into. Else: 'out' is
  a tuple of all the output numpy arrays. If 'out' is given, we return the 'out'
  that was passed in. This is the standard behavior provided by
  numpysane_pywrap.

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
        return mrcal._triangulation_npsp._triangulate_leecivera_linf(v0, v1, t01, out=out)
    else:
        return mrcal._triangulation_npsp._triangulate_leecivera_linf_withgrad(v0, v1, t01, out=out)


def triangulate_leecivera_mid2(v0, v1,
                               t01           = None,
                               *,
                               get_gradients = False,
                               v_are_local   = False,
                               Rt01          = None,
                               out           = None):

    r'''Triangulation using Lee and Civera's alternative midpoint method

SYNOPSIS

    models = ( mrcal.cameramodel('cam0.cameramodel'),
               mrcal.cameramodel('cam1.cameramodel') )

    images = (mrcal.load_image('image0.jpg', bits_per_pixel=8, channels=1),
              mrcal.load_image('image1.jpg', bits_per_pixel=8, channels=1))

    Rt01 = mrcal.compose_Rt( models[0].Rt_cam_ref(),
                             models[1].Rt_ref_cam() )

    # pixel observation in camera0
    q0 = np.array((1233, 2433), dtype=np.float32)

    # corresponding pixel observation in camera1
    q1, _ = \
        mrcal.match_feature( *images,
                             q0            = q0,
                             template_size = (17,17),
                             method        = cv2.TM_CCORR_NORMED,
                             search_radius = 20,
                             H10           = H10, # homography mapping q0 to q1
                           )

    v0 = mrcal.unproject(q0, *models[0].intrinsics())
    v1 = mrcal.unproject(q1, *models[1].intrinsics())

    # Estimated 3D position in camera-0 coordinates of the feature observed in
    # the two cameras
    p = mrcal.triangulate_leecivera_mid2( v0, v1,
                                          Rt01        = Rt01,
                                          v_are_local = True )

This is the lower-level triangulation routine. For a richer function that can be
used to propagate uncertainties, see mrcal.triangulate()

This function implements the "Mid2" triangulation routine in

  "Triangulation: Why Optimize?", Seong Hun Lee and Javier Civera.
  https://arxiv.org/abs/1907.11917

This paper compares many methods. This routine works decently well, but it isn't
the best. The method in this function should be a good tradeoff between accuracy
(in 3D and 2D) and performance. If we're looking at objects very close to the
cameras, where the distances to the two cameras are significantly different, use
triangulate_leecivera_wmid2() instead.

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
  v0 in the default case (v_are_local is False)

- t01: (3,) numpy array containing the position of the camera-1 origin in the
  camera-0 coordinate system. Exclusive with Rt01.

- get_gradients: optional boolean that defaults to False. Whether we should
  compute and report the gradients. This affects what we return. If
  get_gradients: v_are_local must have the default value

- v_are_local: optional boolean that defaults to False. If True: v1 is
  represented in the local coordinate system of camera-1. The default is
  consistent with most, but not all of the triangulation routines. Must have the
  default value if get_gradients

- Rt01: optional (4,3) numpy array, defaulting to None. Exclusive with t01. If
  given, we use this transformation from camera-1 coordinates to camera-0
  coordinates instead of t01. If v_are_local: then Rt01 MUST be given instead of
  t01. This exists for API compatibility with the other triangulation routines.

- out: optional argument specifying the destination. By default, new numpy
  array(s) are created and returned. To write the results into existing (and
  possibly non-contiguous) arrays, specify them with the 'out' kwarg. If not
  get_gradients: 'out' is the one numpy array we will write into. Else: 'out' is
  a tuple of all the output numpy arrays. If 'out' is given, we return the 'out'
  that was passed in. This is the standard behavior provided by
  numpysane_pywrap.

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
        return mrcal._triangulation_npsp._triangulate_leecivera_mid2(v0, v1, t01, out=out)
    else:
        return mrcal._triangulation_npsp._triangulate_leecivera_mid2_withgrad(v0, v1, t01, out=out)


def triangulate_leecivera_wmid2(v0, v1,
                                t01           = None,
                                *,
                                get_gradients = False,
                                v_are_local   = False,
                                Rt01          = None,
                                out           = None):

    r'''Triangulation using Lee and Civera's weighted alternative midpoint method

SYNOPSIS

    models = ( mrcal.cameramodel('cam0.cameramodel'),
               mrcal.cameramodel('cam1.cameramodel') )

    images = (mrcal.load_image('image0.jpg', bits_per_pixel=8, channels=1),
              mrcal.load_image('image1.jpg', bits_per_pixel=8, channels=1))

    Rt01 = mrcal.compose_Rt( models[0].Rt_cam_ref(),
                             models[1].Rt_ref_cam() )

    # pixel observation in camera0
    q0 = np.array((1233, 2433), dtype=np.float32)

    # corresponding pixel observation in camera1
    q1, _ = \
        mrcal.match_feature( *images,
                             q0            = q0,
                             template_size = (17,17),
                             method        = cv2.TM_CCORR_NORMED,
                             search_radius = 20,
                             H10           = H10, # homography mapping q0 to q1
                           )

    v0 = mrcal.unproject(q0, *models[0].intrinsics())
    v1 = mrcal.unproject(q1, *models[1].intrinsics())

    # Estimated 3D position in camera-0 coordinates of the feature observed in
    # the two cameras
    p = mrcal.triangulate_leecivera_wmid2( v0, v1,
                                           Rt01        = Rt01,
                                           v_are_local = True )

This is the lower-level triangulation routine. For a richer function that can be
used to propagate uncertainties, see mrcal.triangulate()

This function implements the "wMid2" triangulation routine in

  "Triangulation: Why Optimize?", Seong Hun Lee and Javier Civera.
  https://arxiv.org/abs/1907.11917

This paper compares many methods. This routine works decently well, but it isn't
the best. The preferred method, according to the paper, is
triangulate_leecivera_mid2. THIS method (wMid2) is better if we're looking at
objects very close to the cameras, where the distances to the two cameras are
significantly different.

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
  v0 in the default case (v_are_local is False)

- t01: (3,) numpy array containing the position of the camera-1 origin in the
  camera-0 coordinate system. Exclusive with Rt01.

- get_gradients: optional boolean that defaults to False. Whether we should
  compute and report the gradients. This affects what we return. If
  get_gradients: v_are_local must have the default value

- v_are_local: optional boolean that defaults to False. If True: v1 is
  represented in the local coordinate system of camera-1. The default is
  consistent with most, but not all of the triangulation routines. Must have the
  default value if get_gradients

- Rt01: optional (4,3) numpy array, defaulting to None. Exclusive with t01. If
  given, we use this transformation from camera-1 coordinates to camera-0
  coordinates instead of t01. If v_are_local: then Rt01 MUST be given instead of
  t01. This exists for API compatibility with the other triangulation routines.

- out: optional argument specifying the destination. By default, new numpy
  array(s) are created and returned. To write the results into existing (and
  possibly non-contiguous) arrays, specify them with the 'out' kwarg. If not
  get_gradients: 'out' is the one numpy array we will write into. Else: 'out' is
  a tuple of all the output numpy arrays. If 'out' is given, we return the 'out'
  that was passed in. This is the standard behavior provided by
  numpysane_pywrap.

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
        return mrcal._triangulation_npsp._triangulate_leecivera_wmid2(v0, v1, t01, out=out)
    else:
        return mrcal._triangulation_npsp._triangulate_leecivera_wmid2_withgrad(v0, v1, t01, out=out)


def triangulate_lindstrom(v0, v1,
                          # The other routines take t01 here
                          Rt01,
                          *,
                          get_gradients = False,
                          v_are_local   = True,
                          out           = None):

    r'''Triangulation minimizing the 2-norm of pinhole reprojection errors

SYNOPSIS

    models = ( mrcal.cameramodel('cam0.cameramodel'),
               mrcal.cameramodel('cam1.cameramodel') )

    images = (mrcal.load_image('image0.jpg', bits_per_pixel=8, channels=1),
              mrcal.load_image('image1.jpg', bits_per_pixel=8, channels=1))

    Rt01 = mrcal.compose_Rt( models[0].Rt_cam_ref(),
                             models[1].Rt_ref_cam() )

    # pixel observation in camera0
    q0 = np.array((1233, 2433), dtype=np.float32)

    # corresponding pixel observation in camera1
    q1, _ = \
        mrcal.match_feature( *images,
                             q0            = q0,
                             template_size = (17,17),
                             method        = cv2.TM_CCORR_NORMED,
                             search_radius = 20,
                             H10           = H10, # homography mapping q0 to q1
                           )

    # observation vectors in the LOCAL coordinate system of the two cameras
    v0 = mrcal.unproject(q0, *models[0].intrinsics())
    v1 = mrcal.unproject(q1, *models[1].intrinsics())

    # Estimated 3D position in camera-0 coordinates of the feature observed in
    # the two cameras
    p = mrcal.triangulate_lindstrom( v0, v1, Rt01 = Rt01 )

This is the lower-level triangulation routine. For a richer function that can be
used to propagate uncertainties, see mrcal.triangulate()

This function implements a triangulation routine minimizing the 2-norm of
reprojection errors, ASSUMING a pinhole projection. This is described in

  "Triangulation Made Easy", Peter Lindstrom, IEEE Conference on Computer Vision
  and Pattern Recognition, 2010.

This is the "L2 img 5-iteration" method in the paper

  "Triangulation: Why Optimize?", Seong Hun Lee and Javier Civera.
  https://arxiv.org/abs/1907.11917

but with only 2 iterations (Lindstrom's paper recommends 2 iterations). This
Lee, Civera paper compares many methods. This routine works decently well, but
it isn't the best. The angular methods should work better than this one for wide
lenses. triangulate_leecivera_mid2() (or triangulate_leecivera_wmid2() if we're
near the cameras) are preferred, according to the paper.

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
  system by default (v_are_local is True). Note that this vector is represented
  in the camera-local coordinate system, unlike the representation in all the
  other triangulation routines

- Rt01: (4,3) numpy array describing the transformation from camera-1
  coordinates to camera-0 coordinates

- get_gradients: optional boolean that defaults to False. Whether we should
  compute and report the gradients. This affects what we return. If
  get_gradients: v_are_local must have the default value

- v_are_local: optional boolean that defaults to True. If True: v1 is
  represented in the local coordinate system of camera-1. This is different from
  the other triangulation routines. Set v_are_local to False to make this
  function interpret v1 similarly to the other triangulation routines. Must have
  the default value if get_gradients

- out: optional argument specifying the destination. By default, new numpy
  array(s) are created and returned. To write the results into existing (and
  possibly non-contiguous) arrays, specify them with the 'out' kwarg. If not
  get_gradients: 'out' is the one numpy array we will write into. Else: 'out' is
  a tuple of all the output numpy arrays. If 'out' is given, we return the 'out'
  that was passed in. This is the standard behavior provided by
  numpysane_pywrap.

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
        return mrcal._triangulation_npsp._triangulate_lindstrom(v0, v1, Rt01, out=out)
    else:
        return mrcal._triangulation_npsp._triangulate_lindstrom_withgrad(v0, v1, Rt01, out=out)


def _triangulated_error(v0, v1,
                        t01           = None,
                        get_gradients = False,
                        v_are_local   = False,
                        Rt01          = None,
                        out           = None):

    r'''Triangulated error function used in the optimization loop

SYNOPSIS

    models = ( mrcal.cameramodel('cam0.cameramodel'),
               mrcal.cameramodel('cam1.cameramodel') )

    images = (cv2.imread('image0.jpg', cv2.IMREAD_GRAYSCALE),
              cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE))

    Rt01 = mrcal.compose_Rt( models[0].Rt_cam_ref(),
                             models[1].Rt_ref_cam() )

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
    v1 = mrcal.unproject(q1, *models[1].intrinsics())

    # Estimated 3D position in camera-0 coordinates of the feature observed in
    # the two cameras
    err = mrcal.triangulation._triangulated_error( v0, v1,
                                                   Rt01        = Rt01,
                                                   v_are_local = True )

This exposes the triangulation-based error function used in the optimization of
triangulated points. It's made available in Python primarily for testing. Unlike
the rest of the triangulation routines:

- This function returns an "error" scalar instead of a triangulated point in
  space

- Special logic is implemented for divergent rays. Please see the implementation
  in triangulation.cc for details

- If get_gradients: we return derr/dv1 and derr/dt01 only. derr/dv0 is NOT
  computed or returned

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
  v0 in the default case (v_are_local is False)

- t01: (3,) numpy array containing the position of the camera-1 origin in the
  camera-0 coordinate system. Exclusive with Rt01.

- get_gradients: optional boolean that defaults to False. Whether we should
  compute and report the gradients. This affects what we return. If
  get_gradients: v_are_local must have the default value

- v_are_local: optional boolean that defaults to False. If True: v1 is
  represented in the local coordinate system of camera-1. The default is
  consistent with most, but not all of the triangulation routines. Must have the
  default value if get_gradients

- Rt01: optional (4,3) numpy array, defaulting to None. Exclusive with t01. If
  given, we use this transformation from camera-1 coordinates to camera-0
  coordinates instead of t01. If v_are_local: then Rt01 MUST be given instead of
  t01. This exists for API compatibility with the other triangulation routines.

- out: optional argument specifying the destination. By default, new numpy
  array(s) are created and returned. To write the results into existing (and
  possibly non-contiguous) arrays, specify them with the 'out' kwarg. If not
  get_gradients: 'out' is the one numpy array we will write into. Else: 'out' is
  a tuple of all the output numpy arrays. If 'out' is given, we return the 'out'
  that was passed in. This is the standard behavior provided by
  numpysane_pywrap.

RETURNED VALUE

if not get_gradients:

  we return an (...) array of triangulated errors. Each slice is a scalar.

if get_gradients: we return a tuple:

  - (...) array of triangulated errors
  - (...,3) array of the gradients of the triangulated errors in respect to
    v1
  - (...,3) array of the gradients of the triangulated errors in respect to
    t01

    '''

    v1, t01 = _parse_args(v1, t01,
                          get_gradients, v_are_local, Rt01)

    if not get_gradients:
        return mrcal._triangulation_npsp._triangulated_error(v0, v1, t01, out=out)
    else:
        return mrcal._triangulation_npsp._triangulated_error_withgrad(v0, v1, t01, out=out)


def _compute_Var_q_triangulation(sigma, stdev_cross_camera_correlation):
    r'''Compute triangulation variance due to observation noise

This is an internal piece of mrcal.triangulate(). It's available separately for
the benefit of the test

    '''

    # For each triangulation we ingest one pixel observation per camera.
    # This is 4 numbers: 2 cameras, with (x,y) for each one
    Ncameras = 2
    Nxy      = 2
    var_q = np.eye(Ncameras*Nxy) * sigma*sigma

    # When computing dense stereo we generally assume that q0 is fixed, and we
    # only think about the effects of Var(q1): Var(q0) = 0, sigma_cross = 0
    # var_q[0,0]  = 0
    # var_q[1,1]  = 0
    # sigma_cross = 0


    var_q_reshaped = var_q.reshape( Ncameras, Nxy,
                                    Ncameras, Nxy )

    sigma_cross = sigma*stdev_cross_camera_correlation
    var_cross   = sigma_cross*sigma_cross

    # cam0-cam1 correlations
    var_q_reshaped[0,0, 1,0] = var_cross
    var_q_reshaped[0,1, 1,1] = var_cross

    # cam1-cam0 correlations
    var_q_reshaped[1,0, 0,0] = var_cross
    var_q_reshaped[1,1, 0,1] = var_cross

    return var_q


def _triangulate_grad_simple(q, models,
                             out,
                             method = triangulate_leecivera_mid2):
    r'''Compute a single triangulation, reporting a single gradient

This is an internal piece of mrcal.triangulate(). It's available separately for
the benefit of the test

    '''

    # Simplified path. We don't need most of the gradients

    rt01 = \
        mrcal.compose_rt(models[0].rt_cam_ref(),
                         models[1].rt_ref_cam())

    # all the v have shape (3,)
    vlocal0, dvlocal0_dq0, _ = \
        mrcal.unproject(q[0,:],
                        *models[0].intrinsics(),
                        get_gradients = True)
    vlocal1, dvlocal1_dq1, _ = \
        mrcal.unproject(q[1,:],
                        *models[1].intrinsics(),
                        get_gradients = True)

    v0 = vlocal0
    v1, _, dv1_dvlocal1 = \
        mrcal.rotate_point_r(rt01[:3], vlocal1,
                             get_gradients=True)

    dp_triangulated_dv0  = np.zeros( out.shape + (3,), dtype=float)
    dp_triangulated_dv1  = np.zeros( out.shape + (3,), dtype=float)
    dp_triangulated_dt01 = np.zeros( out.shape + (3,), dtype=float)
    if method is mrcal.triangulate_lindstrom:
        raise Exception("Triangulation gradients not supported (yet?) with method=triangulate_lindstrom. It has slightly different inputs and slightly different gradients")
    method(v0, v1, rt01[3:],
           out = (out,
                  dp_triangulated_dv0,
                  dp_triangulated_dv1,
                  dp_triangulated_dt01),
           get_gradients = True)

    dp_triangulated_dq = np.zeros((3,) + q.shape[-2:], dtype=float)
    nps.matmult( dp_triangulated_dv0,
                 dvlocal0_dq0,
                 out = dp_triangulated_dq[..., 0, :])
    nps.matmult( dp_triangulated_dv1,
                 dv1_dvlocal1,
                 dvlocal1_dq1,
                 out = dp_triangulated_dq[..., 1, :])

    # shape (3,4)
    return nps.clump(dp_triangulated_dq, n=-2)


def _triangulation_uncertainty_internal(slices,
                                        optimization_inputs, # if None: we're not propagating calibration-time noise
                                        q_observation_stdev,
                                        q_observation_stdev_correlation,
                                        method = triangulate_leecivera_mid2,
                                        stabilize_coords = True):
    r'''Compute most of the triangulation uncertainty logic

This is an internal piece of mrcal.triangulate(). It's available separately to
allow the test suite to validate some of the internals.

if optimization_inputs is None and q_observation_stdev is None:
    We're not propagating any noise. Just return the triangulated point

    '''

    def _triangulate_grad(models, q, out, method):

        # Full path. Compute and return the gradients for most things
        rt_ref1,drt_ref1_drt_1ref = \
            mrcal.invert_rt(models[1].rt_cam_ref(),
                            get_gradients=True)
        rt01,drt01_drt_0ref,drt01_drt_ref1 = \
            mrcal.compose_rt(models[0].rt_cam_ref(), rt_ref1, get_gradients=True)

        # all the v have shape (3,)
        vlocal0, dvlocal0_dq0, dvlocal0_dintrinsics0 = \
            mrcal.unproject(q[0,:],
                            *models[0].intrinsics(),
                            get_gradients = True)
        vlocal1, dvlocal1_dq1, dvlocal1_dintrinsics1 = \
            mrcal.unproject(q[1,:],
                            *models[1].intrinsics(),
                            get_gradients = True)

        v0 = vlocal0
        v1, dv1_dr01, dv1_dvlocal1 = \
            mrcal.rotate_point_r(rt01[:3], vlocal1,
                                 get_gradients=True)

        dp_triangulated_dv0  = np.zeros(out.shape + (3,), dtype=float)
        dp_triangulated_dv1  = np.zeros(out.shape + (3,), dtype=float)
        dp_triangulated_dt01 = np.zeros(out.shape + (3,), dtype=float)

        if method is mrcal.triangulate_lindstrom:
            raise Exception("Triangulation gradients not supported (yet?) with method=triangulate_lindstrom. It has slightly different inputs and slightly different gradients")
        method(v0, v1, rt01[3:],
               out = (out,
                      dp_triangulated_dv0,
                      dp_triangulated_dv1,
                      dp_triangulated_dt01),
               get_gradients = True)

        dp_triangulated_dq = np.zeros((3,) + q.shape[-2:], dtype=float)
        nps.matmult( dp_triangulated_dv0,
                     dvlocal0_dq0,
                     out = dp_triangulated_dq[..., 0, :])
        nps.matmult( dp_triangulated_dv1,
                     dv1_dvlocal1,
                     dvlocal1_dq1,
                     out = dp_triangulated_dq[..., 1, :])

        # shape (3,4)
        dp_triangulated_dq = nps.clump(dp_triangulated_dq, n=-2)

        return                     \
            dp_triangulated_dq,    \
            drt_ref1_drt_1ref,     \
            drt01_drt_0ref,        \
            drt01_drt_ref1,        \
            dvlocal0_dintrinsics0, \
            dvlocal1_dintrinsics1, \
            dv1_dr01,              \
            dv1_dvlocal1,          \
            dp_triangulated_dv0,   \
            dp_triangulated_dv1,   \
            dp_triangulated_dt01


    def stabilize(p_cam0,
                  rt_cam0_ref,
                  rt_ref_frame):

        # The triangulated point is reported in the coordinate system of
        # camera0. If we perturb the calibration inputs, the coordinate
        # system itself moves, and without extra effort, the reported
        # triangulation uncertainty incorporates this extra coordinate
        # system motion. Here, the stabilization logic is available to try
        # to compensate for the effects of the shifting coordinate system.
        # This is done very similarly to how we do this when computing the
        # projection uncertainty.
        #
        # Let's say I have a triangulated point collected after a
        # perturbation. I transform it to the coordinate systems of the
        # frames. Those represent fixed objects in space, so THESE
        # coordinate systems do not shift after a calibration-time
        # perturbation. I then project the point in the coordinate systems
        # of the frames back, using the unperturbed geometry. This gives me
        # the triangulation in the UNPERTURBED (baseline) camera0 frame.
        #
        # The data flow:
        #   point_cam_perturbed -> point_ref_perturbed -> point_frames
        #   point_frames -> point_ref_baseline -> point_cam_baseline
        #
        # The final quantity point_cam_baseline depends on calibration
        # parameters in two ways:
        #
        # 1. Indirectly, via point_cam_perturbed
        # 2. Directly, via each of the transformations in the above data flow
        #
        # For the indirect dependencies, we have the unstabilized
        # dpoint_cam_perturbed/dparam for a number of parameters, and we
        # want to propagate that to the stabilized quantity:
        #
        #   dpoint_cam_baseline/dparam =
        #     dpoint_cam_baseline/dpoint_ref_baseline
        #     dpoint_ref_baseline/dpoint_frames
        #     dpoint_frames/dpoint_ref_perturbed
        #     dpoint_ref_perturbed/dpoint_cam_perturbed
        #     dpoint_cam_perturbed/dparam
        #
        # We only consider small perturbations, and we assume that
        # everything is locally linear. Thus the gradients in this
        # expression all cancel out, and we get simply
        #
        #   dpoint_cam_baseline/dparam = dpoint_cam_perturbed/dparam
        #
        # Thus there's nothing for this function to do to handle these indirect
        # dependencies.
        #
        # For the direct dependencies, we consider point_frames to be the
        # nominal representation of the point. So for the purpose of
        # computing gradients we don't look at the baseline parameter
        # gradients:
        #
        #   dpoint_cam_baseline/dparam =
        #     dpoint_cam_baseline/dpoint_frames
        #     dpoint_frames/dparam =
        #
        # Simplifying notation:
        #
        #   dpc/dparam = dpc/dpf dpf/dparam
        #
        # Thus we have exactly two transformations whose parameters should
        # be propagated:
        #
        # 1. rt_cam_ref
        # 2. rt_ref_frame
        #
        # Propagating rt_cam_ref:
        #
        #   dpc/drt_cr = dpc/dpf dpf/drt_cr
        #              = dpc/dpr dpr/dpf dpf/dpr dpr/drt_cr
        #              = dpc/dpr dpr/drt_cr
        #
        # Propagating rt_ref_frame:
        #
        #   dpc/drt_rf = dpc/dpf dpf/drt_rf
        #              = dpc/dpr dpr/dpf dpf/drt_rf

        #
        # If the frames are fixed, the same logic applies, with some
        # simplifications. The cameras move in respect to the ref frame, but the
        # frames are fixed in the ref frame. So the ref frame is the nominal
        # representation
        #
        #   dpoint_cam_baseline/dparam =
        #     dpoint_cam_baseline/dpoint_ref
        #     dpoint_ref/dparam

        # triangulated point in the perturbed reference coordinate system
        p_ref,                     \
        dp_ref_drt_0ref,           \
        dp_ref_dp_cam0 = \
            mrcal.transform_point_rt(rt_cam0_ref, p_cam0,
                                     get_gradients = True,
                                     inverted      = True)

        dp_triangulated_drt_0ref = \
            np.linalg.solve( dp_ref_dp_cam0,
                             dp_ref_drt_0ref)


        if rt_ref_frame is not None:

            # we're optimizing the frames

            # dp_frames_drtrf  has shape (..., Nframes, 3,6)
            # dp_frames_dp_ref has shape (..., Nframes, 3,3)
            _,                 \
            dp_frames_drtrf,   \
            dp_frames_dp_ref = \
                mrcal.transform_point_rt(rt_ref_frame,
                                         nps.dummy(p_ref,-2),
                                         get_gradients = True,
                                         inverted      = True)

            dp_frames_dp_cam0 = \
                nps.matmult( dp_frames_dp_ref,
                             nps.dummy(dp_ref_dp_cam0, -3))

            Nframes = len(rt_ref_frame)

            # shape (..., 3,6)
            dp_triangulated_drtrf = np.linalg.solve(dp_frames_dp_cam0,
                                                    dp_frames_drtrf) / Nframes
        else:
            # the frames are fixed; not subject to optimization
            dp_triangulated_drtrf = None

        return \
            dp_triangulated_drtrf, \
            dp_triangulated_drt_0ref




    Npoints = len(slices)

    # Output goes here. This function fills in the observation-time stuff.
    # Otherwise this function just returns the array of 0s, which the callers
    # will fill using the dp_triangulated_db data this function returns
    p = np.zeros((Npoints,3), dtype=float)

    if optimization_inputs is not None:

        Nintrinsics = mrcal.num_intrinsics_optimization_params(**optimization_inputs)
        Nstate      = mrcal.num_states(**optimization_inputs)

        # I store dp_triangulated_db initially, without worrying about the
        # "packed" part. I'll scale the thing when done to pack it
        dp_triangulated_db = np.zeros((Npoints,3,Nstate), dtype=float)

        if stabilize_coords and optimization_inputs.get('do_optimize_frames'):
            # We're re-optimizing (looking at calibration uncertainty) AND we
            # are optimizing the frames AND we have stabilization enabled.
            # Without stabilization, there's no dependence on rt_ref_frame
            rt_ref_frame  = optimization_inputs['rt_ref_frame']
            istate_f0     = mrcal.state_index_frames(0, **optimization_inputs)
            Nstate_frames = mrcal.num_states_frames(    **optimization_inputs)
        else:
            rt_ref_frame  = None
            istate_f0     = None
            Nstate_frames = None

    else:
        # We don't need to evaluate the calibration-time noise.
        dp_triangulated_db = None
        istate_i0          = None
        istate_i1          = None
        icam_extrinsics0   = None
        icam_extrinsics1   = None
        istate_e1          = None
        istate_e0          = None

    if q_observation_stdev is not None:
        # observation-time variance of each observed pair of points
        # shape (Ncameras*Nxy, Ncameras*Nxy) = (4,4)
        Var_q_observation_flat = \
            _compute_Var_q_triangulation(q_observation_stdev,
                                         q_observation_stdev_correlation)
        Var_p_observation = np.zeros((Npoints,3,3), dtype=float)
    else:
        Var_p_observation = None


    for ipt in range(Npoints):
        q,models01 = slices[ipt]

        if optimization_inputs is None:
            # shape (3,Ncameras*Nxy=4)
            dp_triangulated_dq = \
                _triangulate_grad_simple(q, models01,
                                         out = p[ipt],
                                         method = method)

        else:
            dp_triangulated_dq,    \
            drt_ref1_drt_1ref,     \
            drt01_drt_0ref,        \
            drt01_drt_ref1,        \
            dvlocal0_dintrinsics0, \
            dvlocal1_dintrinsics1, \
            dv1_dr01,              \
            dv1_dvlocal1,          \
            dp_triangulated_dv0,   \
            dp_triangulated_dv1,   \
            dp_triangulated_dt01 = \
                _triangulate_grad(models01, q,
                                  out = p[ipt],
                                  method = method)

        # triangulation-time uncertainty
        if q_observation_stdev is not None:
            nps.matmult( dp_triangulated_dq,
                         Var_q_observation_flat,
                         nps.transpose(dp_triangulated_dq),
                         out = Var_p_observation[ipt,...])

        if optimization_inputs is None:
            # Not evaluating calibration-time uncertainty. Nothing else to do.
            continue

        # calibration-time uncertainty
        if stabilize_coords:
            dp_triangulated_drtrf,     \
            dp_triangulated_drt_0ref = \
                stabilize(p[ipt],
                          models01[0].rt_cam_ref(),
                          rt_ref_frame)
        else:
            dp_triangulated_drtrf    = None
            dp_triangulated_drt_0ref = None


        # Do the right thing is we're optimizing partial intrinsics only
        i0,i1 = None,None # everything by default
        has_core     = mrcal.lensmodel_metadata_and_config(optimization_inputs['lensmodel'])['has_core']
        Ncore        = 4 if has_core else 0
        Ndistortions = mrcal.lensmodel_num_params(optimization_inputs['lensmodel']) - Ncore
        if not optimization_inputs.get('do_optimize_intrinsics_core'):
            i0 = Ncore
        if not optimization_inputs.get('do_optimize_intrinsics_distortions'):
            i1 = -Ndistortions
        slice_optimized_intrinsics  = slice(i0,i1)
        dvlocal0_dintrinsics0 = dvlocal0_dintrinsics0[...,slice_optimized_intrinsics]
        dvlocal1_dintrinsics1 = dvlocal1_dintrinsics1[...,slice_optimized_intrinsics]

        ### Sensitivities
        # The data flow:
        #   q0,i0                       -> v0 (same as vlocal0; I'm working in the cam0 coord system)
        #   q1,i1                       -> vlocal1
        #   r_0ref,r_1ref               -> r01
        #   r_0ref,r_1ref,t_0ref,t_1ref -> t01
        #   vlocal1,r01                 -> v1
        #   v0,v1,t01                   -> p_triangulated
        icam_intrinsics0 = models01[0].icam_intrinsics()
        icam_intrinsics1 = models01[1].icam_intrinsics()

        istate_i0 = mrcal.state_index_intrinsics(icam_intrinsics0, **optimization_inputs)
        istate_i1 = mrcal.state_index_intrinsics(icam_intrinsics1, **optimization_inputs)
        if istate_i0 is not None:
            # dp_triangulated_di0 = dp_triangulated_dv0              dvlocal0_di0
            # dp_triangulated_di1 = dp_triangulated_dv1 dv1_dvlocal1 dvlocal1_di1
            nps.matmult( dp_triangulated_dv0,
                         dvlocal0_dintrinsics0,
                         out = dp_triangulated_db[ipt, :, istate_i0:istate_i0+Nintrinsics])
        if istate_i1 is not None:
            nps.matmult( dp_triangulated_dv1,
                         dv1_dvlocal1,
                         dvlocal1_dintrinsics1,
                         out = dp_triangulated_db[ipt, :, istate_i1:istate_i1+Nintrinsics])


        icam_extrinsics0 = mrcal.corresponding_icam_extrinsics(icam_intrinsics0, **optimization_inputs)
        icam_extrinsics1 = mrcal.corresponding_icam_extrinsics(icam_intrinsics1, **optimization_inputs)

        if icam_extrinsics0 >= 0:
            istate_e0 = mrcal.state_index_extrinsics(icam_extrinsics0, **optimization_inputs)
        else:
            istate_e0 = None
        if icam_extrinsics1 >= 0:
            istate_e1 = mrcal.state_index_extrinsics(icam_extrinsics1, **optimization_inputs)
        else:
            istate_e1 = None

        if istate_e1 is not None:
            # dp_triangulated_dr_0ref = dp_triangulated_dv1  dv1_dr01 dr01_dr_0ref +
            #                           dp_triangulated_dt01          dt01_dr_0ref
            # dp_triangulated_dr_1ref = dp_triangulated_dv1  dv1_dr01 dr01_dr_1ref +
            #                           dp_triangulated_dt01          dt01_dr_1ref
            # dp_triangulated_dt_0ref = dp_triangulated_dt01          dt01_dt_0ref
            # dp_triangulated_dt_1ref = dp_triangulated_dt01          dt01_dt_1ref
            dr01_dr_ref1    = drt01_drt_ref1[:3,:3]
            dr_ref1_dr_1ref = drt_ref1_drt_1ref[:3,:3]
            dr01_dr_1ref    = nps.matmult(dr01_dr_ref1, dr_ref1_dr_1ref)

            dt01_drt_ref1 = drt01_drt_ref1[3:,:]
            dt01_dr_1ref  = nps.matmult(dt01_drt_ref1, drt_ref1_drt_1ref[:,:3])
            dt01_dt_1ref  = nps.matmult(dt01_drt_ref1, drt_ref1_drt_1ref[:,3:])

            nps.matmult( dp_triangulated_dv1,
                         dv1_dr01,
                         dr01_dr_1ref,
                         out = dp_triangulated_db[ipt, :, istate_e1:istate_e1+3])
            dp_triangulated_db[ipt, :, istate_e1:istate_e1+3] += \
                nps.matmult(dp_triangulated_dt01, dt01_dr_1ref)

            nps.matmult( dp_triangulated_dt01,
                         dt01_dt_1ref,
                         out = dp_triangulated_db[ipt, :, istate_e1+3:istate_e1+6])

        if istate_e0 is not None:
            dr01_dr_0ref = drt01_drt_0ref[:3,:3]
            dt01_dr_0ref = drt01_drt_0ref[3:,:3]
            dt01_dt_0ref = drt01_drt_0ref[3:,3:]

            nps.matmult( dp_triangulated_dv1,
                         dv1_dr01,
                         dr01_dr_0ref,
                         out = dp_triangulated_db[ipt, :, istate_e0:istate_e0+3])
            dp_triangulated_db[ipt, :, istate_e0:istate_e0+3] += \
                nps.matmult(dp_triangulated_dt01, dt01_dr_0ref)

            nps.matmult( dp_triangulated_dt01,
                         dt01_dt_0ref,
                         out = dp_triangulated_db[ipt, :, istate_e0+3:istate_e0+6])

            if dp_triangulated_drt_0ref is not None:
                dp_triangulated_db[ipt, :, istate_e0:istate_e0+6] += dp_triangulated_drt_0ref

        if dp_triangulated_drtrf is not None:
            # We're re-optimizing (looking at calibration uncertainty) AND we
            # are optimizing the frames AND we have stabilization enabled.
            # Without stabilization, there's no dependence on rt_ref_frame

            # dp_triangulated_drtrf has shape (Npoints,Nframes,3,6). I reshape to (Npoints,3,Nframes*6)
            dp_triangulated_db[ipt, :, istate_f0:istate_f0+Nstate_frames] = \
                nps.clump(nps.xchg(dp_triangulated_drtrf,-2,-3), n=-2)

    # Returning the istate stuff for the test suite. These are the istate_...
    # and icam_... for the last slice only. This is good-enough for the test
    # suite
    return p, Var_p_observation, dp_triangulated_db, \
        istate_i0,                            \
        istate_i1,                            \
        icam_extrinsics0,                     \
        icam_extrinsics1,                     \
        istate_e0,                            \
        istate_e1


def triangulate( q,
                 models,
                 *,
                 q_calibration_stdev             = None,
                 q_observation_stdev             = None,
                 q_observation_stdev_correlation = 0,
                 method                          = triangulate_leecivera_mid2,
                 stabilize_coords                = True):

    r'''Triangulate N points with uncertainty propagation

SYNOPSIS

    p,                  \
    Var_p_calibration,  \
    Var_p_observation,  \
    Var_p_joint =       \
        mrcal.triangulate( nps.cat(q0, q),
                           (model0, model1),
                           q_calibration_stdev             = q_calibration_stdev,
                           q_observation_stdev             = q_observation_stdev,
                           q_observation_stdev_correlation = q_observation_stdev_correlation )

    # p is now the triangulated point, in camera0 coordinates. The uncertainties
    # of p from different sources are returned in the Var_p_... arrays

DESCRIPTION

This is the interface to the triangulation computations described in
https://mrcal.secretsauce.net/triangulation.html

Let's say two cameras observe a point p in space. The pixel observations of this
point in the two cameras are q0 and q1 respectively. If the two cameras are
calibrated (both intrinsics and extrinsics), I can reconstruct the observed
point p from the calibration and the two pixel observations. This is the
"triangulation" operation implemented by this function.

If the calibration and the pixel observations q0,q1 were perfect, computing the
corresponding point p would be trivial: unproject q0 and q1, and find the
intersection of the resulting rays. Since everything is perfect, the rays will
intersect exactly at p.

But in reality, both the calibration and the pixel observations are noisy. This
function propagates these sources of noise through the triangulation, to produce
covariance matrices of p.

Two kinds of noise are propagated:

- Calibration-time noise. This is the noise in the pixel observations of the
  chessboard corners, propagated through the calibration to the triangulation
  result. The normal use case is to calibrate once, and then use the same
  calibration result many times. So each time we use a given calibration, we
  have the same calibration-time noise, resulting in correlated errors between
  each such triangulation operation. This is a source of bias: averaging many
  different triangulation results will NOT push the errors to 0.

  Here, only the calibration-time input noise is taken into account. Other
  sources of calibration-time errors (bad input data, outliers, non-fitting
  model) are ignored.

  Furthermore, currently a vanilla calibration is assumed: both cameras in the
  camera pair must have been calibrated together, and the cameras were not moved
  in respect to each other after the calibration.

- Observation-time noise. This is the noise in the pixel observations of the
  point being propagated: q0, q1. Unlike the calibration-time noise, each sample
  of observations (q0,q1) IS independent from every other sample. So if we
  observe the same point p many times, this observation noise will average out.

Both sources of noise are assumed normal with the noise on the x and y
components of the observation being independent. The standard deviation of the
noise is given by the q_calibration_stdev and q_observation_stdev arguments.
Since q0 and q1 often arise from an image correlation operation, they are
usually correlated with each other. It's not yet clear to me how to estimate
this correlation, but it can be specified in this function with the
q_observation_stdev_correlation argument:

- q_observation_stdev_correlation = 0: the noise on q0 and q1 is independent
- q_observation_stdev_correlation = 1: the noise on q0 and q1 is 100% correlated

The (q0x,q1x) and (q0y,q1y) cross terms of the covariance matrix are
(q_observation_stdev_correlation*q_observation_stdev)^2

Since the distribution of the calibration-time input noise is given at
calibration time, we can just use that distribution instead of specifying it
again: pass q_calibration_stdev<0 to do that.

COORDINATE STABILIZATION

We're analyzing the effects of calibration-time noise. As with projection
uncertainty, varying calibration-time noise affects the pose of the camera
coordinate system in respect to the physical camera housing. So Var(p) in the
camera coord system includes this coordinate-system fuzz, which is usually not
something we want to include. To compensate for this fuzz pass
stabilize_coords=True to return Var(p) in the physical camera housing coords.
The underlying method is exactly the same as how this is done with projection
uncertainty:

  https://mrcal.secretsauce.net/uncertainty.html#propagating-through-projection

In the usual case, the translation component of this extra transformation is
negligible, but the rotation (even a small one) produces lateral uncertainty
that isn't really there. Enabling stabilization usually reduces the size of the
uncertainty ellipse in the lateral direction.

BROADCASTING

Broadcasting is fully supported on models and q. Each slice has 2 models and 2
pixel observations (q0,q1). If multiple models and/or observation pairs are
given, we compute the covariances with all the cross terms, so
Var_p_calibration.size grows quadratically with the number of broadcasted slices.

ARGUMENTS

- q: (..., 2,2) numpy array of pixel observations. Each broadcasted slice
  describes a pixel observation from each of the two cameras

- models: iterable of shape (..., 2). Complex shapes may be represented in a
  numpy array of dtype=np.object. Each row is two mrcal.cameramodel objects
  describing the left and right cameras

- q_calibration_stdev: optional value describing the calibration-time noise. If
  omitted or None, we do not compute or return the uncertainty resulting from
  this noise. The noise in the observations of chessboard corners is assumed to
  be normal and independent for each corner and for the x and y components. To
  use the optimization residuals, pass any q_calibration_stdev < 0

- q_observation_stdev: optional value describing the observation-time noise. If
  omitted or None, we do not compute or return the uncertainty resulting from
  this noise. The noise in the observations is assumed to be normal and
  independent for the x and y components

- q_observation_stdev_correlation: optional value, describing the correlation
  between the pair of pixel coordinates observing the same point in space. Since
  q0 and q1 often arise from an image correlation operation, they are usually
  correlated with each other. This argument linearly scales q_observation_stdev:
  0 = "independent", 1 = "100% correlated". The default is 0

- method: optional value selecting the triangulation method. This is one of the
  mrcal.triangulate_... functions. If omitted, we select
  mrcal.triangulate_leecivera_mid2. At this time, mrcal.triangulate_lindstrom is
  usable only if we do not propagate any uncertainties

- stabilize_coords: optional boolean, defaulting to True. We always return the
  triangulated point in camera-0 coordinates. If we're propagating
  calibration-time noise, then the origin of those coordinates moves around
  inside the housing of the camera. Characterizing this extra motion is
  generally not desired in Var_p_calibration. To compensate for this motion, and
  return Var_p_calibration in the coordinate system of the HOUSING of camera-0,
  pass stabilize_coords = True.

RETURN VALUES

We always return p: the coordinates of the triangulated point(s) in the camera-0
coordinate system. Depending on the input arguments, we may also return
uncertainties. The general logic is:

    if q_xxx_stdev is None:
        don't propagate or return that source of uncertainty

    if q_xxx_stdev == 0:
        don't propagate that source of uncertainty, but return
        Var_p_xxx = np.zeros(...)

    if both q_xxx_stdev are not None:
        we compute and report the two separate uncertainty components AND a
        joint covariance combining the two

If we need to return Var_p_calibration, it has shape (...,3, ...,3) where ... is
the broadcasting shape. If a single triangulation is being performed, there's no
broadcasting, and Var_p_calibration.shape = (3,3). This representation allows us
to represent the correlated covariances (non-zero cross terms) that arise due to
calibration-time uncertainty.

If we need to return Var_p_observation, it has shape (...,3,3) where ... is the
broadcasting shape. If a single triangulation is being performed, there's no
broadcasting, and Var_p_observation.shape = (3,3). This representation is more
compact than that for Var_p_calibration because it assumes independent
covariances (zero cross terms) that result when propagating observation-time
uncertainty.

If we need to return Var_p_joint, it has shape (...,3, ...,3) where ... is the
broadcasting shape. If a single triangulation is being performed, there's no
broadcasting, and Var_p_joint.shape = (3,3). This representation allows us to
represent the correlated covariances (non-zero cross terms) that arise due to
calibration-time uncertainty.

Complete logic:

    if q_calibration_stdev is None and
       q_observation_stdev is None:
        # p.shape = (...,3)
        return p

    if q_calibration_stdev is not None and
       q_observation_stdev is None:
        # p.shape = (...,3)
        # Var_p_calibration.shape = (...,3,...,3)
        return p, Var_p_calibration

    if q_calibration_stdev is None and
       q_observation_stdev is not None:
        # p.shape = (...,3)
        # Var_p_observation.shape = (...,3,3)
        return p, Var_p_observation

    if q_calibration_stdev is not None and
       q_observation_stdev is not None:
        # p.shape = (...,3)
        # Var_p_calibration.shape = (...,3,...,3)
        # Var_p_observation.shape = (...,3,    3)
        # Var_p_joint.shape       = (...,3,...,3)
        return p, Var_p_calibration, Var_p_observation, Var_p_joint

    '''


    # I'm propagating noise in the input vector
    #
    #   x = [q_cal q_obs0 q_obs1 q_obs2 ...]
    #
    # This is the noise in the pixel observations at calibration-time and at
    # triangulation time. All the triangulated points are assumed to originate
    # from cameras calibrated using this one calibration run (using observations
    # q_cal). For each triangulation we want to compute, we have a separate set
    # of observations. I want to propagate the noise in x to some function f(x).
    # As usual
    #
    #   Var(f) = df/dx Var(x) (df/dx)T.
    #
    # The noise on all the triangulation-time points is independent, as is the
    # calibration-time noise. Thus Var(x) is block-diagonal and
    #
    #   Var(f) = df/dq_cal  Var(q_cal)  (df/dq_cal)T  +
    #            df/dq_obs0 Var(q_obs0) (df/dq_obs0)T +
    #            df/dq_obs1 Var(q_obs1) (df/dq_obs1)T + ...
    if q_observation_stdev is not None and \
       q_observation_stdev < 0:
        raise Exception("q_observation_stdev MUST be None or >= 0")

    if not isinstance(models, np.ndarray):
        models = np.array(models, dtype=object)


    slices            = tuple(nps.broadcast_generate(   ((2,2),(2,)), (q, models) ) )
    broadcasted_shape = tuple(nps.broadcast_extra_dims( ((2,2),(2,)), (q, models) ))

    if (q_calibration_stdev is None or q_calibration_stdev == 0) and \
       (q_observation_stdev is None or q_observation_stdev == 0):

        # I don't need to propagate any noise

        @nps.broadcast_define(((2,2),(2,)), (3,))
        def triangulate_slice(q01, m01):
            Rt01 = \
                mrcal.compose_Rt(m01[0].Rt_cam_ref(),
                                 m01[1].Rt_ref_cam())

            # all the v have shape (3,)
            vlocal0 = mrcal.unproject(q01[0,:], *m01[0].intrinsics())
            vlocal1 = mrcal.unproject(q01[1,:], *m01[1].intrinsics())

            return method(vlocal0, vlocal1,
                          v_are_local = True,
                          Rt01        = Rt01)

        p = triangulate_slice(q, models)

        if q_calibration_stdev is None and \
           q_observation_stdev is None:
            return p

        if q_calibration_stdev is not None:
            Var_p_calibration = np.zeros(broadcasted_shape + (3,) +
                                         broadcasted_shape + (3,),
                                         dtype=float)
        if q_observation_stdev is not None:
            Var_p_observation = np.zeros(broadcasted_shape + (3,3), dtype=float)

        if q_calibration_stdev is not None:
            if q_observation_stdev is not None:
                return p, Var_p_calibration, Var_p_observation, Var_p_calibration
            else:
                return p, Var_p_calibration
        else:
            return p, Var_p_observation



    # SOMETHING is non-zero, so we need to do noise propagation

    if q_calibration_stdev is not None and \
       q_calibration_stdev != 0:
        # we're propagating calibration-time noise

        models_flat = models.ravel()

        optimization_inputs = models_flat[0].optimization_inputs()

        if optimization_inputs is None:
            raise Exception("optimization_inputs are not available, so I cannot propagate calibration-time noise")

        for i0 in range(len(models_flat)):
            for i1 in range(i0):
                if not models_flat[i0]._optimization_inputs_match(models_flat[i1]):
                    raise Exception("The optimization_inputs for all of the given models must be identical")

            if models_flat[i0]._extrinsics_moved_since_calibration():
                raise Exception(f"The given models must have been fixed inside the initial calibration. Model {i0} has been moved")

    else:
        optimization_inputs = None



    # p has shape (Npoints + (3,))
    # Var_p_observation_flat has shape (Npoints + (3,3))
    # dp_triangulated_db has shape (Npoints*3 + (Nstate,))
    p,                      \
    Var_p_observation_flat, \
    dp_triangulated_db = \
        _triangulation_uncertainty_internal(
                        slices,
                        optimization_inputs,
                        q_observation_stdev,
                        q_observation_stdev_correlation,
                        method           = method,
                        stabilize_coords = stabilize_coords)[:3]

    # Done looping through all the triangulated points. I have computed the
    # observation-time noise contributions in Var_p_observation. And I have all
    # the gradients in dp_triangulated_db

    if optimization_inputs is not None:

        # reshape dp_triangulated_db to (Npoints*3, Nstate)
        # So the Var(p) will end up with shape (Npoints*3, Npoints*3)
        dp_triangulated_db = nps.clump(dp_triangulated_db,n=2)

        if q_calibration_stdev > 0:
            # Calibration-time noise is given. Use it.
            observed_pixel_uncertainty = q_calibration_stdev
        else:
            # Infer the expected calibration-time noise from the calibration data
            observed_pixel_uncertainty = None

        # Var_p_calibration_flat has shape (Npoints*3,Npoints*3)
        Var_p_calibration_flat = \
            mrcal.model_analysis._propagate_calibration_uncertainty(
                                               'covariance',
                                               dF_dbunpacked              = dp_triangulated_db,
                                               observed_pixel_uncertainty = observed_pixel_uncertainty,
                                               optimization_inputs        = optimization_inputs)

    else:
        Var_p_calibration_flat = None

    # I used broadcast_generate() to implement the broadcasting logic. This
    # function flattened the broadcasted output, so at this time I have
    #   p.shape                      = (Npoints,3)
    #   Var_p_calibration_flat.shape = (Npoints*3, Npoints*3)
    #   Var_p_observation_flat.shape = (Npoints,3,3)
    #
    # I now reshape the output into its proper shape. I have the leading shape I
    # want in broadcasted_shape (I know that reduce(broadcast_extra_dims, *) =
    # Npoints)
    p = p.reshape(broadcasted_shape + (3,))
    if Var_p_calibration_flat is not None:
        Var_p_calibration = \
            Var_p_calibration_flat.reshape(broadcasted_shape + (3,) +
                                           broadcasted_shape + (3,))
    elif  q_calibration_stdev is not None and \
          q_calibration_stdev == 0:
        Var_p_calibration = \
            np.zeros(broadcasted_shape + (3,) +
                     broadcasted_shape + (3,))
    else:
        Var_p_calibration = None
    if Var_p_observation_flat is not None:
        Var_p_observation = \
            Var_p_observation_flat.reshape(broadcasted_shape + (3,3))
    else:
        Var_p_observation = None

    if Var_p_observation is None:
        return p, Var_p_calibration
    if Var_p_calibration is None:
        return p, Var_p_observation

    # Propagating both types of noise. I create a joint covariance matrix
    if Var_p_calibration is not None:
        Var_p_joint = Var_p_calibration.copy()
    else:
        Var_p_joint = np.zeros((broadcasted_shape + (3,) +
                                broadcasted_shape + (3,)), dtype=float)
    if Var_p_observation is not None:
        Var_p_joint_flat = Var_p_joint.reshape(len(slices)*3,
                                               len(slices)*3)
        for ipt in range(len(slices)):
            Var_p_joint_flat[ipt*3:(ipt+1)*3,ipt*3:(ipt+1)*3] += \
                Var_p_observation_flat[ipt,...]

    return p, Var_p_calibration, Var_p_observation, Var_p_joint
