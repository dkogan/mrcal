#!/usr/bin/python3

# Copyright (c) 2017-2023 California Institute of Technology ("Caltech"). U.S.
# Government sponsorship acknowledged. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

'''Routines for stereo processing: rectification and ranging

All functions are exported into the mrcal module. So you can call these via
mrcal.stereo.fff() or mrcal.fff(). The latter is preferred.
'''

import sys
import numpy as np
import numpysane as nps
import mrcal
import re

def rectified_resolution(model0,
                         *,
                         az_fov_deg,
                         el_fov_deg,
                         az0_deg,
                         el0_deg,
                         R_cam0_rect0,
                         pixels_per_deg_az   = -1.,
                         pixels_per_deg_el   = -1.,
                         rectification_model = 'LENSMODEL_LATLON'):

    r'''Compute the resolution to be used for the rectified system

SYNOPSIS

    pixels_per_deg_az,  \
    pixels_per_deg_el = \
        mrcal.rectified_resolution(model0,
                                   az_fov_deg = 120,
                                   el_fov_deg = 100,
                                   az0_deg    = 0,
                                   el0_deg    = 0
                                   R_cam0_rect0)

This is usually called from inside mrcal.rectified_system() only, and usually
there's no reason for the end-user to call this function. If the final
resolution used in the rectification is needed, call
mrcal.rectified_system(return_metadata = True)

This function also supports LENSMODEL_LONLAT (not for stereo rectification, but
for a 360deg around-the-horizon view), and is useful to compute the image
resolution in those applications.

Similar to mrcal.rectified_system(), this functions takes in rectified-image
pan,zoom values and a desired resolution in pixels_per_deg_.... If
pixels_per_deg_... < 0: we compute and return a scaled factor of the input image
resolution at the center of the rectified field of view. pixels_per_deg_... = -1
means "full resolution", pixels_per_deg_... = -0.5 means "half resolution" and
so on.

If pixels_per_deg_... > 0: then we use the intended value as is.

In either case, we adjust the returned pixels_per_deg_... to evenly fit into the
requested field of view, to match the integer pixel count in the rectified
image. This is only possible for LENSMODEL_LATLON and LENSMODEL_LONLAT.

ARGUMENTS

- model0: the model of the camera that captured the image that will be
  reprojected. In a stereo pair this is the FIRST camera. Used to determine the
  angular resolution of the input image. Only the intrinsics are used.

- az_fov_deg: required value for the azimuth (along-the-baseline) field-of-view
  of the desired rectified system, in degrees

- el_fov_deg: required value for the elevation (across-the-baseline)
  field-of-view of the desired rectified system, in degrees

- az0_deg: required value for the azimuth center of the rectified images.

- el0_deg: required value for the elevation center of the rectified system.

- R_cam0_rect0: required rotation matrix relating the input-camera and
  rectified-camera systems

- pixels_per_deg_az: optional value for the azimuth resolution of the rectified
  image. If a resolution of >0 is requested, the value is used as is. If a
  resolution of <0 is requested, we use this as a scale factor on the resolution
  of the first input image. For instance, to downsample by a factor of 2, pass
  pixels_per_deg_az = -0.5. By default, we use -1: the resolution of the input
  image at the center of the rectified system.

- pixels_per_deg_el: same as pixels_per_deg_az but in the elevation direction

- rectification_model: optional string that selects the projection function to
  use in the resulting rectified system. This is a string selecting the mrcal
  lens model. Currently supported are "LENSMODEL_LATLON" (the default) and
  "LENSMODEL_LONLAT" and "LENSMODEL_PINHOLE"

RETURNED VALUES

A tuple (pixels_per_deg_az,pixels_per_deg_el)

    '''
    # The guts of this function are implemented in C. Call that
    return mrcal._mrcal._rectified_resolution(*model0.intrinsics(),
                                              R_cam0_rect0        = np.ascontiguousarray(R_cam0_rect0),
                                              az_fov_deg          = az_fov_deg,
                                              el_fov_deg          = el_fov_deg,
                                              az0_deg             = az0_deg,
                                              el0_deg             = el0_deg,
                                              pixels_per_deg_az   = pixels_per_deg_az,
                                              pixels_per_deg_el   = pixels_per_deg_el,
                                              rectification_model = rectification_model)

def _rectified_resolution_python(model0,
                                 *,
                                 az_fov_deg,
                                 el_fov_deg,
                                 az0_deg,
                                 el0_deg,
                                 R_cam0_rect0,
                                 pixels_per_deg_az   = -1.,
                                 pixels_per_deg_el   = -1.,
                                 rectification_model = 'LENSMODEL_LATLON'):

    r'''Reference implementation of mrcal_rectified_resolution() in python

The main implementation is written in C in stereo.c:

  mrcal_rectified_resolution()

This should be identical to the rectified_resolution() function above. Tests
compare the two implementations to make sure.

    '''

    if pixels_per_deg_az < 0 or \
       pixels_per_deg_el < 0:

        azel0 = np.array((az0_deg,el0_deg)) * np.pi/180.

        # I need to compute the resolution of the rectified images. I try to
        # match the resolution of the cameras. I just look at camera0. If your
        # two cameras are different, pass in the pixels_per_deg yourself
        #
        # I look at the center of the stereo field of view. There I have qrect =
        # project(vrect) where vrect is a unit projection vector. I compute
        # dqrect/dth where th is an angular perturbation applied to vrect.
        if rectification_model == 'LENSMODEL_LATLON':
            q0rect_normalized = azel0
            vrect,dvrect_dazel = \
                mrcal.unproject_latlon( q0rect_normalized,
                                        get_gradients = True )
        elif rectification_model == 'LENSMODEL_LONLAT':
            q0rect_normalized = azel0
            vrect,dvrect_dazel = \
                mrcal.unproject_lonlat( q0rect_normalized,
                                        get_gradients = True )
        elif rectification_model == 'LENSMODEL_PINHOLE':
            q0rect_normalized = np.tan(azel0)
            vrect,dv_dq0normalized = \
                mrcal.unproject_pinhole( q0rect_normalized,
                                         get_gradients = True )
            # dqrect/dth = dtanth/dth = 1/cos^2(th)
            dvrect_dazel = dv_dq0normalized

            cos_azel0 = np.cos(azel0)
            dvrect_dazel /= cos_azel0*cos_azel0

        else:
            raise Exception("Unsupported rectification model")

        v0         = mrcal.rotate_point_R(R_cam0_rect0, vrect)
        dv0_dazel  = nps.matmult(R_cam0_rect0, dvrect_dazel)

        _,dq_dv0,_ = mrcal.project(v0, *model0.intrinsics(), get_gradients = True)

        # More complex method that's probably not any better
        #
        # if False:
        #     # I rotate my v to a coordinate system where u = rotate(v) is [0,0,1].
        #     # Then u = [a,b,0] are all orthogonal to v. So du/dth = [cos, sin, 0].
        #     # I then have dq/dth = dq/dv dv/du [cos, sin, 0]t
        #     # ---> dq/dth = dq/dv dv/du[:,:2] [cos, sin]t = M [cos,sin]t
        #     #
        #     # norm2(dq/dth) = [cos,sin] MtM [cos,sin]t is then an ellipse with the
        #     # eigenvalues of MtM giving me the best and worst sensitivities. I can
        #     # use mrcal.worst_direction_stdev() to find the densest direction. But I
        #     # actually know the directions I care about, so I evaluate them
        #     # independently for the az and el directions
        #     Ruv = mrcal.R_aligned_to_vector(v0)
        #     M = nps.matmult(dq_dv0, nps.transpose(Ruv[:2,:]))
        #     # I pick the densest direction: highest |dq/dth|
        #     pixels_per_rad = mrcal.worst_direction_stdev( nps.matmult( nps.transpose(M),M) )

        dq_dazel = nps.matmult(dq_dv0, dv0_dazel)

        if pixels_per_deg_az < 0:
            pixels_per_deg_az_have = nps.mag(dq_dazel[:,0])*np.pi/180.
            pixels_per_deg_az *= -pixels_per_deg_az_have

        if pixels_per_deg_el < 0:
            pixels_per_deg_el_have = nps.mag(dq_dazel[:,1])*np.pi/180.
            pixels_per_deg_el *= -pixels_per_deg_el_have

    # I now have the desired pixels_per_deg
    #
    # With LENSMODEL_LATLON or LENSMODEL_LONLAT we have even angular spacing, so
    # q = f th + c -> dq/dth = f everywhere. I can thus compute the rectified
    # image size and adjust the resolution accordingly
    #
    # With LENSMODEL_PINHOLE this is much more complex, so this function just
    # leaves the desired pixels_per_deg as it is
    if rectification_model == 'LENSMODEL_LATLON' or \
       rectification_model == 'LENSMODEL_LONLAT':

        Naz = round(az_fov_deg*pixels_per_deg_az)
        Nel = round(el_fov_deg*pixels_per_deg_el)

        pixels_per_deg_az = Naz/az_fov_deg
        pixels_per_deg_el = Nel/el_fov_deg

    return \
        pixels_per_deg_az, \
        pixels_per_deg_el

def rectified_system(models,
                     *,
                     az_fov_deg,
                     el_fov_deg,
                     az0_deg             = None,
                     el0_deg             = 0,
                     az_edge_margin_deg  = 10.,
                     pixels_per_deg_az   = -1.,
                     pixels_per_deg_el   = -1.,
                     rectification_model = 'LENSMODEL_LATLON',
                     return_metadata     = False):

    r'''Build rectified models for stereo rectification

SYNOPSIS

    import sys
    import mrcal
    import cv2
    import numpy as np
    import numpysane as nps

    models = [ mrcal.cameramodel(f) \
               for f in ('left.cameramodel',
                         'right.cameramodel') ]

    images = [ mrcal.load_image(f) \
               for f in ('left.jpg', 'right.jpg') ]

    models_rectified = \
        mrcal.rectified_system(models,
                               az_fov_deg = 120,
                               el_fov_deg = 100)

    rectification_maps = mrcal.rectification_maps(models, models_rectified)

    images_rectified = [ mrcal.transform_image(images[i], rectification_maps[i]) \
                         for i in range(2) ]

    # Find stereo correspondences using OpenCV
    block_size = 3
    max_disp   = 160 # in pixels
    matcher = \
        cv2.StereoSGBM_create(minDisparity      = 0,
                              numDisparities    = max_disp,
                              blockSize         = block_size,
                              P1                = 8 *3*block_size*block_size,
                              P2                = 32*3*block_size*block_size,
                              uniquenessRatio   = 5,

                              disp12MaxDiff     = 1,
                              speckleWindowSize = 50,
                              speckleRange      = 1)
    disparity16 = matcher.compute(*images_rectified) # in pixels*16

    # Point cloud in rectified camera-0 coordinates
    # shape (H,W,3)
    p_rect0 = mrcal.stereo_unproject( disparity16,
                                      models_rectified,
                                      disparity_scale = 16 )

    Rt_cam0_rect0 = mrcal.compose_Rt( models          [0].Rt_cam_ref(),
                                      models_rectified[0].Rt_ref_cam() )

    # Point cloud in camera-0 coordinates
    # shape (H,W,3)
    p_cam0 = mrcal.transform_point_Rt(Rt_cam0_rect0, p_rect0)

This function computes the parameters of a rectified system from two
cameramodels in a stereo pair. The output is a pair of "rectified" models. Each
of these is a normal mrcal.cameramodel object describing a "camera" somewhere in
space, with some particular projection behavior. The pair of models returned
here have the desired property that each row of pixels represents a plane in
space AND each corresponding row in the pair of rectified images represents the
SAME plane: the epipolar lines are aligned. We can use the rectified models
returned by this function to transform the input images into "rectified" images,
and then we can perform stereo matching efficiently, by treating each row
independently.

This function is generic: the two input cameras may use any lens models, any
resolution and any geometry. They don't even have to match. As long as there's
some non-zero baseline and some overlapping views, we can run stereo processing.

The two rectified models describe the poses of the rectified cameras. Each
rectified camera sits at the same position as the input camera, but with a
different orientation. The orientations of the two cameras in the rectified
system are identical. The only difference between the poses of the two rectified
cameras is a translation of the second camera along the x axis. The axes of the
rectified coordinate system:

- x: from the origin of the first camera to the origin of the second camera (the
  baseline direction)

- y: completes the system from x,z

- z: the mean "forward" direction of the two input cameras, with the component
  parallel to the baseline subtracted off

In a nominal geometry (the two cameras are square with each other, the second
camera strictly to the right of the first camera), the geometry of the rectified
models exactly matches the geometry of the input models. The above formulation
supports any geometry, however, including vertical and/or forward/backward
shifts. Vertical stereo is supported: we still run stereo matching on ROWS of
the rectified images, but the rectification transformation will rotate the
images by 90 degrees.

Several projection functions may be used in the rectified system. These are
selectable using the "rectification_model" keyword argument; they're a string
representing the lensmodel that will be used in the cameramodel objects we
return. Two projections are currently supported:

- "LENSMODEL_LATLON": the default projection that utilizes a transverse
  equirectangular map. This projection has even angular spacing between pixels,
  so it works well even with wide lenses. The documentation has more information:
  https://mrcal.secretsauce.net/lensmodels.html#lensmodel-latlon

- "LENSMODEL_PINHOLE": the traditional projection function that utilizes a
  pinhole camera. This works badly with wide lenses, and is recommended if
  compatibility with other stereo processes is desired

ARGUMENTS

- models: an iterable of two mrcal.cameramodel objects representing the cameras
  in the stereo pair. Any sane combination of lens models and resolutions and
  geometries is valid

- az_fov_deg: required value for the azimuth (along-the-baseline) field-of-view
  of the desired rectified system, in degrees

- el_fov_deg: required value for the elevation (across-the-baseline)
  field-of-view of the desired rectified system, in degrees. Note that this
  applies at the center of the rectified system: az = 0. With a skewed stereo
  system (if we have a forward/backward shift or if a nonzero az0_deg is given),
  this rectification center will not be at the horizontal center of the image,
  and may not be in-bounds of the image at all.

- az0_deg: optional value for the azimuth center of the rectified images. This
  is especially significant in a camera system with some forward/backward shift.
  That causes the baseline to no longer be perpendicular with the view axis of
  the cameras, and thus the azimuth=0 vector no longer points "forward". If
  omitted, we compute az0_deg to align the center of the rectified system with
  the center of the two cameras' views. This computed value can be retrieved in
  the metadata dict by passing return_metadata = True

- el0_deg: optional value for the elevation center of the rectified system.
  Defaults to 0.

- az_edge_margin_deg: optional angle, the nearest the rectified view is allowed
  to approach az = -90 or az = +90. Defaults to 10deg, for a maximum az field of
  view of 160deg. Applies ONLY to LENSMODEL_LATLON rectification

- pixels_per_deg_az: optional value for the azimuth resolution of the rectified
  image. If a resolution of >0 is requested, the value is used as is. If a
  resolution of <0 is requested, we use this as a scale factor on the resolution
  of the first input image. For instance, to downsample by a factor of 2, pass
  pixels_per_deg_az = -0.5. By default, we use -1: the resolution of the input
  image at the center of the rectified system. The value we end up with can be
  retrieved in the metadata dict by passing return_metadata = True

- pixels_per_deg_el: same as pixels_per_deg_az but in the elevation direction

- rectification_model: optional string that selects the projection function to
  use in the resulting rectified system. This is a string selecting the mrcal
  lens model. Currently supported are "LENSMODEL_LATLON" (the default) and
  "LENSMODEL_PINHOLE"

- return_metadata: optional boolean, defaulting to False. If True, we return a
  dict of metadata describing the rectified system in addition to the rectified
  models. This is useful to retrieve any of the autodetected values. At this
  time, the metadata dict contains keys:

    - az_fov_deg
    - el_fov_deg
    - az0_deg
    - el0_deg
    - pixels_per_deg_az
    - pixels_per_deg_el
    - baseline

RETURNED VALUES

We compute a tuple of mrcal.cameramodels describing the two rectified cameras.
These two models are identical, except for a baseline translation in the +x
direction in rectified coordinates.

if not return_metadata: we return this tuple of models
else:                   we return this tuple of models, dict of metadata

    '''

    for m in models:
        lensmodel = m.intrinsics()[0]
        meta = mrcal.lensmodel_metadata_and_config(lensmodel)
        if meta['noncentral']:
            if re.match("^LENSMODEL_CAHVORE", lensmodel):
                if nps.norm2(m.intrinsics()[1][-3:]) > 0:
                    raise Exception("Stereo rectification is only possible with a central projection. Please centralize your models. This is CAHVORE, so set E=0 to centralize. This will ignore all noncentral effects near the lens")
            else:
                raise Exception("Stereo rectification is only possible with a central projection. Please centralize your models")

    # The guts of this function are implemented in C. Call that
    pixels_per_deg_az, \
    pixels_per_deg_el, \
    Naz, Nel,          \
    fxycxy_rectified,  \
    rt_rect0_ref,      \
    baseline,          \
    az_fov_deg,        \
    el_fov_deg,        \
    az0_deg,           \
    el0_deg =          \
        mrcal._mrcal._rectified_system(*models[0].intrinsics(),
                                       models[0].rt_cam_ref(),
                                       models[1].rt_cam_ref(),
                                       az_fov_deg          = az_fov_deg,
                                       el_fov_deg          = el_fov_deg,
                                       az0_deg             = az0_deg if az0_deg is not None else 1e7,
                                       el0_deg             = el0_deg,
                                       az_edge_margin_deg  = az_edge_margin_deg,
                                       pixels_per_deg_az   = pixels_per_deg_az,
                                       pixels_per_deg_el   = pixels_per_deg_el,
                                       rectification_model = rectification_model)


    ######## The geometry

    # rect1 coord system has the same orientation as rect0, but translated by
    # baseline in the x direction (its origin is at the origin of cam1)
    #   rt_rect0_rect1 = (0,0,0, baseline,0,0)
    #   rt_rect1_ref = rt_rect1_rect0 rt_rect0_ref
    rt_rect1_ref = rt_rect0_ref.copy()
    rt_rect1_ref[3] -= baseline

    models_rectified = \
        ( mrcal.cameramodel( intrinsics = (rectification_model, fxycxy_rectified),
                             imagersize = (Naz, Nel),
                             rt_cam_ref = rt_rect0_ref),

          mrcal.cameramodel( intrinsics = (rectification_model, fxycxy_rectified),
                             imagersize = (Naz, Nel),
                             rt_cam_ref = rt_rect1_ref) )

    if not return_metadata:
        return models_rectified

    metadata = \
        dict( az_fov_deg        = az_fov_deg,
              el_fov_deg        = el_fov_deg,
              az0_deg           = az0_deg,
              el0_deg           = el0_deg,
              pixels_per_deg_az = pixels_per_deg_az,
              pixels_per_deg_el = pixels_per_deg_el,
              baseline          = baseline )

    return models_rectified, metadata

def _rectified_system_python(models,
                             *,
                             az_fov_deg,
                             el_fov_deg,
                             az0_deg             = None,
                             el0_deg             = 0,
                             pixels_per_deg_az   = -1.,
                             pixels_per_deg_el   = -1.,
                             rectification_model = 'LENSMODEL_LATLON',
                             return_metadata     = False):

    r'''Reference implementation of mrcal_rectified_system() in python

The main implementation is written in C in stereo.c:

  mrcal_rectified_system()

This should be identical to the rectified_system() function above. Tests compare
the two implementations to make sure.
    '''


    if not (rectification_model == 'LENSMODEL_LATLON' or \
            rectification_model == 'LENSMODEL_PINHOLE'):
        raise(f"Unsupported rectification model '{rectification_model}'. Only LENSMODEL_LATLON and LENSMODEL_PINHOLE are supported.")

    if len(models) != 2:
        raise Exception("I need exactly 2 camera models")

    if pixels_per_deg_az == 0:
        raise Exception("pixels_per_deg_az == 0 is illegal. Must be >0 if we're trying to specify a value, or <0 to autodetect")
    if pixels_per_deg_el == 0:
        raise Exception("pixels_per_deg_el == 0 is illegal. Must be >0 if we're trying to specify a value, or <0 to autodetect")

    if az_fov_deg is None or el_fov_deg is None or \
       az_fov_deg <= 0.   or el_fov_deg <= 0.:
        raise Exception("az_fov_deg, el_fov_deg must be > 0. No auto-detection implemented yet")

    ######## Compute the geometry of the rectified stereo system. This is a
    ######## rotation, centered at camera0. More or less we have axes:
    ########
    ######## x: from camera0 to camera1
    ######## y: completes the system from x,z
    ######## z: component of the cameras' viewing direction
    ########    normal to the baseline
    Rt01 = mrcal.compose_Rt( models[0].Rt_cam_ref(),
                             models[1].Rt_ref_cam())

    # Rotation relating camera0 coords to the rectified camera coords. I fill in
    # each row separately
    Rt_rect0_cam0 = np.zeros((4,3), dtype=float)
    R_rect0_cam0 = Rt_rect0_cam0[:3,:]

    # Axes of the rectified system, in the cam0 coord system
    right       = R_rect0_cam0[0,:]
    down        = R_rect0_cam0[1,:]
    forward     = R_rect0_cam0[2,:]

    # "right" of the rectified coord system: towards the origin of camera1 from
    # camera0, in camera0 coords
    right[:] = Rt01[3,:]
    baseline = nps.mag(right)
    right   /= baseline

    # "forward" for each of the two cameras, in the cam0 coord system
    forward0 = np.array((0,0,1.))
    forward1 = Rt01[:3,2]

    # "forward" of the rectified coord system, in camera0 coords. The mean
    # optical-axis direction of the two cameras: component orthogonal to "right"
    forward01 = forward0 + forward1
    forward01_proj_right = nps.inner(forward01,right)
    forward[:] = forward01 - forward01_proj_right*right
    forward /= nps.mag(forward)

    # "down" of the rectified coord system, in camera0 coords. Completes the
    # right,down,forward coordinate system
    down[:] = np.cross(forward,right)

    ######## Done with the geometry! Now to get the az/el grid. I need to figure
    ######## out the resolution and the extents

    if az0_deg is not None:
        az0 = az0_deg * np.pi/180.

    else:
        # In the rectified system az=0 sits perpendicular to the baseline.
        # Normally the cameras are looking out perpendicular to the baseline
        # also, so I center my azimuth samples around 0 to match the cameras'
        # field of view. But what if the geometry isn't square, and one camera
        # is behind the other? Like this:
        #
        #    camera
        #     view
        #       ^
        #       |
        #     \ | /
        #      \_/
        #        .    /
        #         .  /az=0
        #          ./
        #           .
        #  baseline  .
        #             .
        #            \   /
        #             \_/
        #
        # Here the center-of-view axis of each camera is not at all
        # perpendicular to the baseline. Thus I compute the mean "forward"
        # direction of the cameras in the rectified system, and set that as the
        # center azimuth az0.
        az0 = np.arcsin( forward01_proj_right / nps.mag(forward01) )
        az0_deg = az0 * 180./np.pi

    el0 = el0_deg * np.pi/180.

    pixels_per_deg_az,  \
    pixels_per_deg_el = \
        mrcal.rectified_resolution(models[0],
                                   az_fov_deg          = az_fov_deg,
                                   el_fov_deg          = el_fov_deg,
                                   az0_deg             = az0_deg,
                                   el0_deg             = el0_deg,
                                   R_cam0_rect0        = nps.transpose(R_rect0_cam0),
                                   pixels_per_deg_az   = pixels_per_deg_az,
                                   pixels_per_deg_el   = pixels_per_deg_el,
                                   rectification_model = rectification_model)

    # How do we apply the desired pixels_per_deg?
    #
    # With LENSMODEL_LATLON we have even angular spacing, so q = f th + c ->
    # dq/dth = f everywhere.
    #
    # With LENSMODEL_PINHOLE the angular resolution changes across the image: q
    # = f tan(th) + c -> dq/dth = f/cos^2(th). So at the center, th=0 and we
    # have the maximum resolution
    fxycxy = np.array((pixels_per_deg_az / np.pi*180.,
                       pixels_per_deg_el / np.pi*180.,
                       0., 0.), dtype=float)
    if rectification_model == 'LENSMODEL_LATLON':
        # The angular resolution is consistent everywhere, so fx,fy are already
        # set. Let's set cx,cy such that
        # (az0,el0) = unproject(imager center)
        Naz = round(az_fov_deg*pixels_per_deg_az)
        Nel = round(el_fov_deg*pixels_per_deg_el)
        fxycxy[2:] = \
            np.array(((Naz-1.)/2.,(Nel-1.)/2.)) - \
            np.array((az0,el0)) * fxycxy[:2]

    elif rectification_model == 'LENSMODEL_PINHOLE':
        cos_az0 = np.cos(az0)
        cos_el0 = np.cos(el0)
        fxycxy[0] *= cos_az0*cos_az0
        fxycxy[1] *= cos_el0*cos_el0

        # fx,fy are set. Let's set cx,cy. Unlike the LENSMODEL_LATLON case, this
        # is asymmetric, so I explicitly solve for (cx,Naz). cy,Nel work the
        # same way. I want
        #
        #  tan(az0)*fx + cx = (Naz-1)/2
        #
        #  inner( normalized(unproject(x=0)),
        #         normalized(unproject(x=Naz-1)) ) = cos(fov)
        #
        # unproject(x=0    ) = [ (0     - cx)/fx, 0, 1]
        # unproject(x=Naz-1) = [ (Naz-1 - cx)/fx, 0, 1]
        #
        # -> v0 ~ [ -cx/fx,           0, 1]
        # -> v1 ~ [ 2*tanaz0 + cx/fx, 0, 1]
        #
        # Let K = 2*tanaz0 (we have K). Let C = cx/fx (we want to find C)
        # -> v0 ~ [-C,1], v1 ~ [K+C,1]
        # -> cosfov = (1 - K*C - C^2) / sqrt( (1+C^2)*(1+C^2+K^2+2*K*C))
        # -> cos2fov*(1+C^2)*(1+C^2+K^2+2*K*C) - (1 - K*C - C^2)^2 = 0
        # -> 0 =
        #        C^4 * (cos2fov - 1) +
        #        C^3 * 2 K (cos2fov - 1 ) +
        #        C^2 * (cos2fov K^2 + 2 cos2fov - K^2 + 2) +
        #        C   * 2 K ( cos2fov + 1 ) +
        #        cos2fov ( K^2 + 1 ) - 1
        #
        # I can solve this numerically
        def cxy(fxy, tanazel0, fov_deg):
            cosfov = np.cos(fov_deg*np.pi/180.)
            cos2fov = cosfov*cosfov
            K = 2.*tanazel0

            C = np.roots( [ (cos2fov - 1),
                            2.* K * (cos2fov - 1 ),
                            cos2fov * K*K + 2.*cos2fov - K*K + 2,
                            2.* K * (cos2fov + 1 ),
                            cos2fov * (K*K + 1.) - 1 ] )

            # Some numerical fuzz (if fov ~ 90deg) may give me slightly
            # imaginary numbers, so I just look at the real component.
            # Similarly, I allow a bit of numerical fuzz in the logic below
            C = np.real(C)

            # I solve my quartic polynomial numerically. I get 4 solutions,
            # and I need to throw out the invalid ones.
            #
            # fov may be > 90deg, so cos(fov) may be <0. The solution will make
            # sure that cos^2(fov) matches up, but the solution may assume the
            # wrong sign for cos(fov). From above:
            #
            #   cosfov = (1 - K*C - C^2) / sqrt( (1+C^2)*(1+C^2+K^2+2*K*C))
            #
            # So I must have cosfov*(1 - K*C - C^2) > 0
            C = C[cosfov*(1 - K*C - C*C) >= -1e-9]

            # And the implied imager size MUST be positive
            C = C[(tanazel0*fxy + C*fxy)*2. + 1 > 0]

            if len(C) == 0:
                raise Exception("Couldn't compute the rectified pinhole center pixel. Something is wrong.")

            # I should have exactly one solution left. Due to some numerical
            # fuzz, I might have more, and I pick the most positive one in the
            # condition above
            return C[np.argmax(cosfov*(1 - K*C - C*C))] * fxy



        tanaz0 = np.tan(az0)
        tanel0 = np.tan(el0)
        fxycxy[2] = cxy(fxycxy[0], tanaz0, az_fov_deg)
        fxycxy[3] = cxy(fxycxy[1], tanel0, el_fov_deg)

        Naz = round((tanaz0*fxycxy[0] + fxycxy[2])*2.) + 1
        Nel = round((tanel0*fxycxy[1] + fxycxy[3])*2.) + 1

    else:
        raise Exception("Shouldn't get here; This case was checked above")

    if Nel <= 0:
        raise Exception(f"Resulting stereo geometry has Nel={Nel}. This is nonsensical. You should examine the geometry or adjust the elevation bounds or pixels-per-deg")

    ######## The geometry
    Rt_rect0_ref  = mrcal.compose_Rt( Rt_rect0_cam0,
                                      models[0].Rt_cam_ref())
    # rect1 coord system has the same orientation as rect0, but is translated so
    # that its origin is at the origin of cam1
    R_rect1_cam0  = R_rect0_cam0
    R_rect1_cam1  = nps.matmult(R_rect1_cam0, Rt01[:3,:])

    Rt_rect1_cam1 = nps.glue(R_rect1_cam1, np.zeros((3,),), axis=-2)
    Rt_rect1_ref  = mrcal.compose_Rt( Rt_rect1_cam1,
                                      models[1].Rt_cam_ref())

    models_rectified = \
        ( mrcal.cameramodel( intrinsics = (rectification_model, fxycxy),
                             imagersize = (Naz, Nel),
                             Rt_cam_ref = Rt_rect0_ref),

          mrcal.cameramodel( intrinsics = (rectification_model, fxycxy),
                             imagersize = (Naz, Nel),
                             Rt_cam_ref = Rt_rect1_ref) )

    if not return_metadata:
        return models_rectified

    metadata = \
        dict( az_fov_deg        = az_fov_deg,
              el_fov_deg        = el_fov_deg,
              az0_deg           = az0 * 180./np.pi,
              el0_deg           = el0_deg,
              pixels_per_deg_az = pixels_per_deg_az,
              pixels_per_deg_el = pixels_per_deg_el,
              baseline          = baseline )

    return models_rectified, metadata

def _validate_models_rectified(models_rectified):
    r'''Internal function to validate a rectified system

These should have been returned by rectified_system(). Should have two
LENSMODEL_LATLON or LENSMODEL_PINHOLE cameras with identical intrinsics.
extrinsics should be identical too EXCEPT for a baseline translation in the x
rectified direction

    '''

    if len(models_rectified) != 2:
        raise Exception(f"Must have received exactly two models. Got {len(models_rectified)} instead")

    intrinsics = [m.intrinsics() for m in models_rectified]
    Rt01 = mrcal.compose_Rt( models_rectified[0].Rt_cam_ref(),
                             models_rectified[1].Rt_ref_cam())

    if not ( (intrinsics[0][0] == 'LENSMODEL_LATLON'  and intrinsics[1][0] == 'LENSMODEL_LATLON' ) or \
             (intrinsics[0][0] == 'LENSMODEL_PINHOLE' and intrinsics[1][0] == 'LENSMODEL_PINHOLE') ):
        raise Exception(f"Expected two models with the same  'LENSMODEL_LATLON' or 'LENSMODEL_PINHOLE' but got {intrinsics[0][0]} and {intrinsics[1][0]}")

    if nps.norm2(intrinsics[0][1] - intrinsics[1][1]) > 1e-6:
        raise Exception("The two rectified models MUST have the same intrinsics values")

    imagersize_diff = \
        np.array(models_rectified[0].imagersize()) - \
        np.array(models_rectified[1].imagersize())
    if imagersize_diff[0] != 0 or imagersize_diff[1] != 0:
        raise Exceptions("The two rectified models MUST have the same imager size")

    costh = (np.trace(Rt01[:3,:]) - 1.) / 2.
    if costh < 0.999999:
        raise Exception("The two rectified models MUST have the same relative rotation")

    if nps.norm2(Rt01[3,1:]) > 1e-9:
        raise Exception("The two rectified models MUST have a translation ONLY in the +x rectified direction")

def rectification_maps(models,
                       models_rectified):

    r'''Construct image transformation maps to make rectified images

SYNOPSIS

    import sys
    import mrcal
    import cv2
    import numpy as np
    import numpysane as nps

    models = [ mrcal.cameramodel(f) \
               for f in ('left.cameramodel',
                         'right.cameramodel') ]

    images = [ mrcal.load_image(f) \
               for f in ('left.jpg', 'right.jpg') ]

    models_rectified = \
        mrcal.rectified_system(models,
                               az_fov_deg = 120,
                               el_fov_deg = 100)

    rectification_maps = mrcal.rectification_maps(models, models_rectified)

    images_rectified = [ mrcal.transform_image(images[i], rectification_maps[i]) \
                         for i in range(2) ]

    # Find stereo correspondences using OpenCV
    block_size = 3
    max_disp   = 160 # in pixels
    matcher = \
        cv2.StereoSGBM_create(minDisparity      = 0,
                              numDisparities    = max_disp,
                              blockSize         = block_size,
                              P1                = 8 *3*block_size*block_size,
                              P2                = 32*3*block_size*block_size,
                              uniquenessRatio   = 5,

                              disp12MaxDiff     = 1,
                              speckleWindowSize = 50,
                              speckleRange      = 1)
    disparity16 = matcher.compute(*images_rectified) # in pixels*16

    # Point cloud in rectified camera-0 coordinates
    # shape (H,W,3)
    p_rect0 = mrcal.stereo_unproject( disparity16,
                                      models_rectified,
                                      disparity_scale = 16 )

    Rt_cam0_rect0 = mrcal.compose_Rt( models          [0].Rt_cam_ref(),
                                      models_rectified[0].Rt_ref_cam() )

    # Point cloud in camera-0 coordinates
    # shape (H,W,3)
    p_cam0 = mrcal.transform_point_Rt(Rt_cam0_rect0, p_rect0)

After the pair of rectified models has been built by mrcal.rectified_system(),
this function can be called to compute the rectification maps. These can be
passed to mrcal.transform_image() to remap input images into the rectified
space.

The documentation for mrcal.rectified_system() applies here.

This function is implemented in C in the mrcal_rectification_maps() function. An
equivalent Python implementation is available:
mrcal.stereo._rectification_maps_python()

ARGUMENTS

- models: an iterable of two mrcal.cameramodel objects representing the cameras
  in the stereo pair

- models_rectified: the pair of rectified models, corresponding to the input
  images. Usually this is returned by mrcal.rectified_system()

RETURNED VALUES

We return a length-2 tuple of numpy arrays containing transformation maps for
each camera. Each map can be used to mrcal.transform_image() images into
rectified space. Each array contains 32-bit floats (as expected by
mrcal.transform_image() and cv2.remap()). Each array has shape (Nel,Naz,2),
where (Nel,Naz) is the shape of each rectified image. Each shape-(2,) row
contains corresponding pixel coordinates in the input image

    '''

    _validate_models_rectified(models_rectified)

    Naz,Nel = models_rectified[0].imagersize()
    # shape (Ncameras=2, Nel, Naz, Nxy=2)
    rectification_maps = np.zeros((2, Nel, Naz, 2),
                                  dtype = np.float32)
    mrcal._mrcal._rectification_maps(*models[0].intrinsics(),
                                     *models[1].intrinsics(),
                                     *models_rectified[0].intrinsics(),
                                     r_cam0_ref  = models[0].rt_cam_ref()[:3],
                                     r_cam1_ref  = models[1].rt_cam_ref()[:3],
                                     r_rect0_ref = models_rectified[0].rt_cam_ref()[:3],
                                     rectification_maps = rectification_maps)

    return rectification_maps

def _rectification_maps_python(models,
                               models_rectified):
    r'''Reference implementation of mrcal.rectification_maps() in python

The main implementation is written in C in stereo.c:

  mrcal_rectification_maps()

This should be identical to the rectification_maps() function above. This is
checked by the test-rectification-maps.py test.

    '''

    Naz,Nel = models_rectified[0].imagersize()
    fxycxy  = models_rectified[0].intrinsics()[1]

    R_cam_rect = [ nps.matmult(models          [i].Rt_cam_ref()[:3,:],
                               models_rectified[i].Rt_ref_cam  ()[:3,:]) \
                   for i in range(2) ]

    # This is massively inefficient. I should
    #
    # - Not generate any intermediate ARRAYS, but loop through each pixel, and
    #   perform the full transformation on each pixel. All the way through the
    #   project(v0, ...) below
    #
    # - Not compute full sin/cos separately for each pixel, but take advantage
    #   of my even angle steps to compute the sin/cos once, and take
    #   multiplication/addition steps from there

    # shape (Nel,Naz,3)
    if models_rectified[0].intrinsics()[0] == 'LENSMODEL_LATLON':
        unproject = mrcal.unproject_latlon
    else:
        unproject = mrcal.unproject_pinhole

    az, el = \
        np.meshgrid(np.arange(Naz,dtype=float),
                    np.arange(Nel,dtype=float))

    v = unproject( np.ascontiguousarray( \
           nps.mv( nps.cat(az,el),
                   0, -1)),
                   fxycxy)

    v0 = mrcal.rotate_point_R(R_cam_rect[0], v)
    v1 = mrcal.rotate_point_R(R_cam_rect[1], v)

    return                                                                \
        (mrcal.project( v0, *models[0].intrinsics()).astype(np.float32),  \
         mrcal.project( v1, *models[1].intrinsics()).astype(np.float32))

def stereo_range(disparity,
                 models_rectified,
                 *,
                 disparity_scale      = 1,
                 disparity_min        = None,
                 disparity_scaled_min = None,
                 disparity_max        = None,
                 disparity_scaled_max = None,
                 qrect0               = None):

    r'''Compute ranges from observed disparities

SYNOPSIS

    import sys
    import mrcal
    import cv2
    import numpy as np
    import numpysane as nps

    models = [ mrcal.cameramodel(f) \
               for f in ('left.cameramodel',
                         'right.cameramodel') ]

    images = [ mrcal.load_image(f) \
               for f in ('left.jpg', 'right.jpg') ]

    models_rectified = \
        mrcal.rectified_system(models,
                               az_fov_deg = 120,
                               el_fov_deg = 100)

    rectification_maps = mrcal.rectification_maps(models, models_rectified)

    images_rectified = [ mrcal.transform_image(images[i], rectification_maps[i]) \
                         for i in range(2) ]

    # Find stereo correspondences using OpenCV
    block_size = 3
    max_disp   = 160 # in pixels
    matcher = \
        cv2.StereoSGBM_create(minDisparity      = 0,
                              numDisparities    = max_disp,
                              blockSize         = block_size,
                              P1                = 8 *3*block_size*block_size,
                              P2                = 32*3*block_size*block_size,
                              uniquenessRatio   = 5,

                              disp12MaxDiff     = 1,
                              speckleWindowSize = 50,
                              speckleRange      = 1)
    disparity16 = matcher.compute(*images_rectified) # in pixels*16

    # Convert the disparities to range-to-camera0
    ranges = mrcal.stereo_range( disparity16,
                                 models_rectified,
                                 disparity_scale = 16 )

    H,W = disparity16.shape

    # shape (H,W,2)
    q = np.ascontiguousarray( \
           nps.mv( nps.cat( *np.meshgrid(np.arange(W,dtype=float),
                                         np.arange(H,dtype=float))),
                   0, -1))

    # Point cloud in rectified camera-0 coordinates
    # shape (H,W,3)
    p_rect0 = \
        mrcal.unproject_latlon(q, models_rectified[0].intrinsics()[1]) * \
        nps.dummy(ranges, axis=-1)

    Rt_cam0_rect0 = mrcal.compose_Rt( models          [0].Rt_cam_ref(),
                                      models_rectified[0].Rt_ref_cam() )

    # Point cloud in camera-0 coordinates
    # shape (H,W,3)
    p_cam0 = mrcal.transform_point_Rt(Rt_cam0_rect0, p_rect0)

As shown in the example above, we can perform stereo processing by building
rectified models and transformation maps, rectifying our images, and then doing
stereo matching to get pixel disparities. The disparities can be converted to
usable geometry by calling one of two functions:

- stereo_range() to convert pixel disparities to ranges

- stereo_unproject() to convert pixel disparities to a point cloud. This is a
  superset of stereo_range()

In the most common usage of stereo_range() we take a full disparity IMAGE, and
then convert it to a range IMAGE. In this common case we call

    range_image = mrcal.stereo_range(disparity_image, models_rectified)

If we aren't processing the full disparity image, we can pass in an array of
rectified pixel coordinates (in the first rectified camera) in the "qrect0"
argument. These must be broadcastable with the disparity argument. So we
can pass in a scalar for disparity and a single (2,) array for qrect0. Or
we can pass in full arrays for both. Or we can pass in a shape (H,W) image for
disparity, but only a shape (W,2) array for qrect0: this would use the
same qrect0 value for a whole column of disparity, as dictated by the
broadcasting rules. Such identical-az-in-a-column behavior is valid for
LENSMODEL_LATLON stereo, but not for LENSMODEL_PINHOLE stereo. It's the user's
responsibility to know when to omit data like this. When in doubt, pass a
separate qrect0 for each disparity value.

Each epipolar plane looks like this:

camera0
+ . . . .
\ az0
|----------------
|               \--------------------
|                         range      \-----------------------
|                                                            \-------- p
|                                                             a -----/
|                                                         -----/
|                                                   -----/
|baseline                                     -----/
|                                       -----/
|                                 -----/
|                           -----/
|                     -----/
|               -----/
|         -----/
|   -----/
---/ az1
+. . . . .
camera1

The cameras are at the top-left and bottom-left of the figure, looking out to
the right at a point p in space. The observation ray from camera0 makes an angle
az0 with the "forward" direction (here az0 > 0), while the observation ray from
camera1 makes an angle az1 (here az1 < 0). A LENSMODEL_LATLON disparity is a
difference of azimuth angles: disparity ~ az0-az1. A LENSMODEL_PINHOLE disparity
is a scaled difference of tangents: disparity ~ tan(az0)-tan(az1)

The law of sines tells us that

    baseline / sin(a) = range / sin(90 + az1)

Thus

    range = baseline cos(az1) / sin(a) =
          = baseline cos(az1) / sin( 180 - (90-az0 + 90+az1) ) =
          = baseline cos(az1) / sin(az0-az1) =
          = baseline cos(az0 - az0-az1) / sin(az0-az1)

az0-az1 is the angular disparity. If using LENSMODEL_LATLON, this is what we
have, and this is a usable expression. Otherwise we keep going:

    range = baseline cos(az0 - az0-az1) / sin(az0-az1)
          = baseline (cos(az0)cos(az0-az1) + sin(az0)sin(az0-az1)) / sin(az0-az1)
          = baseline cos(az0)/tan(az0-az1) + sin(az0)
          = baseline cos(az0)* (1 + tan(az0)tan(az1))/(tan(az0) - tan(az1)) + sin(az0)
          = baseline cos(az0)*((1 + tan(az0)tan(az1))/(tan(az0) - tan(az1)) + tan(az0))

A scaled tan(az0)-tan(az1) is the disparity when using LENSMODEL_PINHOLE, so
this is the final expression we use.

When using LENSMODEL_LATLON, the azimuth values in the projection ARE the
azimuth values inside each epipolar plane, so there's nothing extra to do. When
using LENSMODEL_PINHOLE however, there's an extra step. We need to convert pixel
disparity values to az0 and az1.

Let's say we're looking two rectified pinhole points on the same epipolar plane,
a "forward" point and a "query" point:

    q0 = [0, qy]    and    q1 = [qx1, qy]

We convert these to normalized coords: tanxy = (q-cxy)/fxy

    t0 = [0, ty]    and    t1 = [tx1, ty]

These unproject to

    v0 = [0, ty, 1]    and    v1 = [tx1, ty, 1]

These lie on an epipolar plane with normal [0, -1, ty]. I define a coordinate
system basis using the normal as one axis. The other two axes are

    b0 = [1, 0,    0  ]
    b1 = [0, ty/L, 1/L]

where L = sqrt(ty^2 + 1)

Projecting my two vectors to (b0,b1) I get

    [0,   ty^2/L + 1/L]
    [tx1, ty^2/L + 1/L]

Thus the the angle this query point makes with the "forward" vector is

    tan(az_in_epipolar_plane) = tx1 / ( (ty^2 + 1)/L ) = tx1 / sqrt(ty^2 + 1)

Thus to get tan(az) expressions we use to compute ranges, we need to scale our
(qx1-cx)/fx values by 1./sqrt(ty^2 + 1). This is one reason to use
LENSMODEL_LATLON for stereo processing instead of LENSMODEL_PINHOLE: the az
angular scale stays constant across different el, which produces better stereo
matches.

ARGUMENTS

- disparity: a numpy array of disparities being processed. If disparity_scale is
  omitted, this array contains floating-point disparity values in PIXELS. Many
  stereo-matching algorithms produce integer disparities, in units of some
  constant number of pixels (the OpenCV StereoSGBM and StereoBM routines use
  16). In this common case, you can pass the integer scaled disparities here,
  with the scale factor in disparity_scale. Any array shape is supported. In the
  common case of a disparity IMAGE, this is an array of shape (Nel, Naz)

- models_rectified: the pair of rectified models, corresponding to the input
  images. Usually this is returned by mrcal.rectified_system()

- disparity_scale: optional scale factor for the "disparity" array. If omitted,
  the "disparity" array is assumed to contain the disparities, in pixels.
  Otherwise it contains data in the units of 1/disparity_scale pixels.

- disparity_min: optional minimum-expected disparity value. If omitted,
  disparity_min = 0 is assumed. disparity below this limit is interpreted as an
  invalid value: range=0 is reported. This has units of "pixels", so we scale by
  disparity_scale before comparing to the dense stereo correlator result

- disparity_max: optional maximum-expected disparity value. If omitted, no
  maximum exists. Works the same as disparity_min.

- disparity_scaled_min
  disparity_scaled_max: optional disparity values with the disparity_scaled
  already applied. These can be compared directly against the scaled disparity
  values. Both the scaled and unscaled flavors of these may NOT be given at the
  same time. In the common case of 16-bit signed disparities AND disparity_max
  is None: disparity_scales_max defaults to 0x7FFF to force disparity<0 to be
  treated as invalid

- qrect0: optional array of rectified camera0 pixel coordinates corresponding to
  the given disparities. By default, a full disparity image is assumed.
  Otherwise we use the given rectified coordinates. The shape of this array must
  be broadcasting-compatible with the disparity array. See the
  description above.

RETURNED VALUES

- An array of ranges of the same dimensionality as the input disparity
  array. Contains floating-point data. Invalid or missing ranges are represented
  as 0.

    '''

    _validate_models_rectified(models_rectified)

    if disparity_min is not None and disparity_scaled_min is not None and \
       disparity_min != disparity_scaled_min:
        raise Exception("disparity_min and disparity_scaled_min may not both be given")
    if disparity_max is not None and disparity_scaled_max is not None and \
       disparity_max != disparity_scaled_max:
        raise Exception("disparity_max and disparity_scaled_max may not both be given")

    if disparity_scaled_max is None and disparity_max is None and \
       (isinstance(disparity, np.ndarray) and disparity.dtype == np.int16):
        disparity_scaled_max = 0x7FFF

    if qrect0 is None:
        if disparity_scaled_min is None:
            if disparity_min is None:
                disparity_scaled_min = 0
            else:
                disparity_scaled_min = disparity_min * disparity_scale
        if disparity_scaled_max is None:
            if disparity_max is None:
                disparity_scaled_max = np.uint16(np.iinfo(np.uint16).max)
            else:
                disparity_scaled_max = disparity_max * disparity_scale
    else:
        if disparity_min is None:
            if disparity_scaled_min is None:
                disparity_min = 0
            else:
                disparity_min = disparity_scaled_min / disparity_scale
        if disparity_max is None:
            if disparity_scaled_max is None:
                disparity_max = np.finfo(float).max
            else:
                disparity_max = disparity_scaled_max / disparity_scale

    # I want to support scalar disparities. If one is given, I convert it into
    # an array of shape (1,), and then pull it out at the end
    is_scalar = False
    try:
        s = disparity.shape
    except:
        is_scalar = True
    if not is_scalar:
        if len(s) == 0:
            is_scalar = True
    if is_scalar:
        disparity = np.array((disparity,),)

    Rt01 = mrcal.compose_Rt( models_rectified[0].Rt_cam_ref(),
                             models_rectified[1].Rt_ref_cam())
    baseline = nps.mag(Rt01[3,:])

    if qrect0 is None:
        W,H = models_rectified[0].imagersize()
        if np.any(disparity.shape - np.array((H,W),dtype=int)):
            raise Exception(f"qrect0 is None, so the given disparity and full rectified images MUST have the same dimensions. I have {disparity.shape=} and {models_rectified[0].imagersize()=}")

        r =                                    \
            mrcal._mrcal_npsp._stereo_range_dense \
                ( disparity_scaled     = disparity.astype(np.uint16),
                  disparity_scale      = np.uint16(disparity_scale),
                  disparity_scaled_min = np.uint16(disparity_scaled_min),
                  disparity_scaled_max = np.uint16(disparity_scaled_max),
                  rectification_model_type = models_rectified[0].intrinsics()[0],
                  fxycxy_rectified     = models_rectified[0].intrinsics()[1].astype(float),
                  baseline             = baseline )

    else:

        r =                                     \
            mrcal._mrcal_npsp._stereo_range_sparse \
                ( disparity            = disparity.astype(float) / disparity_scale,
                  qrect0               = qrect0.astype(float),
                  disparity_min        = float(disparity_min),
                  disparity_max        = float(disparity_max),
                  rectification_model_type = models_rectified[0].intrinsics()[0],
                  fxycxy_rectified     = models_rectified[0].intrinsics()[1].astype(float),
                  baseline             = baseline )

    if is_scalar:
        r = r[0]
    return r


def _stereo_range_python(disparity,
                         models_rectified,
                         *,
                         disparity_scale = 1,
                         disparity_min        = None,
                         disparity_scaled_min = None,
                         disparity_max        = None,
                         disparity_scaled_max = None,
                         qrect0          = None):

    r'''Reference implementation of mrcal.stereo_range() in python

The main implementation is written in C in stereo.c:

  mrcal_stereo_range_sparse() and mrcal_stereo_range_dense()

This should be identical to the stereo_range() function above. This is
checked by the test-stereo-range.py test.

    '''

    _validate_models_rectified(models_rectified)

    if disparity_min is not None and disparity_scaled_min is not None and \
       disparity_min != disparity_scaled_min:
        raise Exception("disparity_min and disparity_scaled_min may not both be given")
    if disparity_max is not None and disparity_scaled_max is not None and \
       disparity_max != disparity_scaled_max:
        raise Exception("disparity_max and disparity_scaled_max may not both be given")

    if qrect0 is None:
        if disparity_scaled_min is None:
            if disparity_min is None:
                disparity_scaled_min = 0
            else:
                disparity_scaled_min = disparity_min * disparity_scale
        if disparity_scaled_max is None:
            if disparity_max is None:
                disparity_scaled_max = np.uint16(np.iinfo(np.uint16).max)
            else:
                disparity_scaled_max = disparity_max * disparity_scale
    else:
        if disparity_min is None:
            if disparity_scaled_min is None:
                disparity_min = 0
            else:
                disparity_min = disparity_scaled_min / disparity_scale
        if disparity_max is None:
            if disparity_scaled_max is None:
                disparity_max = np.finfo(float).max
            else:
                disparity_max = disparity_scaled_max / disparity_scale


    # I want to support scalar disparities. If one is given, I convert it into
    # an array of shape (1,), and then pull it out at the end
    is_scalar = False
    try:
        s = disparity.shape
    except:
        is_scalar = True
    if not is_scalar:
        if len(s) == 0:
            is_scalar = True
    if is_scalar:
        disparity = np.array((disparity,),)

    W,H = models_rectified[0].imagersize()
    if qrect0 is None and np.any(disparity.shape - np.array((H,W),dtype=int)):
        raise Exception(f"qrect0 is None, so the disparity image must have the full dimensions of a rectified image")

    intrinsics = models_rectified[0].intrinsics()

    fx = intrinsics[1][0]
    cx = intrinsics[1][2]

    Rt01 = mrcal.compose_Rt( models_rectified[0].Rt_cam_ref(),
                             models_rectified[1].Rt_ref_cam())
    baseline = nps.mag(Rt01[3,:])

    if qrect0 is None:
        mask_invalid = \
            (disparity < disparity_scaled_min) + \
            (disparity > disparity_scaled_max) + \
            (disparity <= 0)
    else:
        mask_invalid = \
            (disparity / disparity_scale < disparity_min) + \
            (disparity / disparity_scale > disparity_max) + \
            (disparity <= 0)


    if intrinsics[0] == 'LENSMODEL_LATLON':
        if qrect0 is None:
            az0 = (np.arange(W, dtype=float) - cx)/fx
        else:
            az0 = (qrect0[...,0] - cx)/fx

        disparity_rad = disparity.astype(np.float32) / (fx * disparity_scale)

        s = np.sin(disparity_rad)
        s[mask_invalid] = 1 # to prevent division by 0

        r = baseline * np.cos(az0 - disparity_rad) / s

    else:
        # pinhole

        fy = intrinsics[1][1]
        cy = intrinsics[1][3]

        if qrect0 is None:
            # shape (W,)
            tanaz0 = (np.arange(W, dtype=float) - cx)/fx

            # shape (H,1)
            tanel  = (np.arange(H, dtype=float) - cy)/fy
            tanel  = nps.dummy(tanel, -1)
        else:
            tanaz0 = (qrect0[...,0] - cx) / fx
            tanel  = (qrect0[...,1] - cy) / fy
        s_sq_recip = tanel*tanel + 1.


        tanaz0_tanaz1 = disparity.astype(np.float32) / (fx * disparity_scale)

        tanaz0_tanaz1[mask_invalid] = 1 # to prevent division by 0

        tanaz1 = tanaz0 - tanaz0_tanaz1
        r = baseline / \
            np.sqrt(s_sq_recip + tanaz0*tanaz0) * \
            ((s_sq_recip + tanaz0*tanaz1) / tanaz0_tanaz1 + \
             tanaz0)

        mask_invalid += ~np.isfinite(r)

    r[mask_invalid] = 0

    if is_scalar:
        r = r[0]
    return r

def stereo_unproject(disparity,
                     models_rectified,
                     *,
                     ranges          = None,
                     disparity_scale = 1,
                     qrect0          = None):

    r'''Compute a point cloud from observed disparities

SYNOPSIS

    import sys
    import mrcal
    import cv2
    import numpy as np
    import numpysane as nps

    models = [ mrcal.cameramodel(f) \
               for f in ('left.cameramodel',
                         'right.cameramodel') ]

    images = [ mrcal.load_image(f) \
               for f in ('left.jpg', 'right.jpg') ]

    models_rectified = \
        mrcal.rectified_system(models,
                               az_fov_deg = 120,
                               el_fov_deg = 100)

    rectification_maps = mrcal.rectification_maps(models, models_rectified)

    images_rectified = [ mrcal.transform_image(images[i], rectification_maps[i]) \
                         for i in range(2) ]

    # Find stereo correspondences using OpenCV
    block_size = 3
    max_disp   = 160 # in pixels
    matcher = \
        cv2.StereoSGBM_create(minDisparity      = 0,
                              numDisparities    = max_disp,
                              blockSize         = block_size,
                              P1                = 8 *3*block_size*block_size,
                              P2                = 32*3*block_size*block_size,
                              uniquenessRatio   = 5,

                              disp12MaxDiff     = 1,
                              speckleWindowSize = 50,
                              speckleRange      = 1)
    disparity16 = matcher.compute(*images_rectified) # in pixels*16

    # Point cloud in rectified camera-0 coordinates
    # shape (H,W,3)
    p_rect0 = mrcal.stereo_unproject( disparity16,
                                      models_rectified,
                                      disparity_scale = 16 )

    Rt_cam0_rect0 = mrcal.compose_Rt( models          [0].Rt_cam_ref(),
                                      models_rectified[0].Rt_ref_cam() )

    # Point cloud in camera-0 coordinates
    # shape (H,W,3)
    p_cam0 = mrcal.transform_point_Rt(Rt_cam0_rect0, p_rect0)

As shown in the example above, we can perform stereo processing by building
rectified models and transformation maps, rectifying our images, and then doing
stereo matching to get pixel disparities. The disparities can be converted to
usable geometry by calling one of two functions:

- stereo_range() to convert pixel disparities to ranges

- stereo_unproject() to convert pixel disparities to a point cloud, each point in
  the rectified-camera-0 coordinates. This is a superset of stereo_range()

In the most common usage of stereo_unproject() we take a full disparity IMAGE,
and then convert it to a dense point cloud: one point per pixel. In this common
case we call

    p_rect0 = mrcal.stereo_unproject(disparity_image, models_rectified)

If we aren't processing the full disparity image, we can pass in an array of
rectified pixel coordinates (in the first rectified camera) in the "qrect0"
argument. These must be broadcastable with the disparity argument.

The arguments are identical to those accepted by stereo_range(), except an
optional "ranges" argument is accepted. This can be given in lieu of the
"disparity" argument, if we already have ranges returned by stereo_range().
Exactly one of (disparity,ranges) must be given as non-None. "ranges" is a
keyword-only argument, while "disparity" is positional. So if passing ranges,
the disparity must be explicitly given as None:

    p_rect0 = mrcal.stereo_unproject( disparity        = None,
                                      models_rectified = models_rectified,
                                      ranges           = ranges )

ARGUMENTS

- disparity: a numpy array of disparities being processed. If disparity_scale is
  omitted, this array contains floating-point disparity values in PIXELS. Many
  stereo-matching algorithms produce integer disparities, in units of some
  constant number of pixels (the OpenCV StereoSGBM and StereoBM routines use
  16). In this common case, you can pass the integer scaled disparities here,
  with the scale factor in disparity_scale. Any array shape is supported. In the
  common case of a disparity IMAGE, this is an array of shape (Nel, Naz)

- models_rectified: the pair of rectified models, corresponding to the input
  images. Usually this is returned by mrcal.rectified_system()

- ranges: optional numpy array with the ranges returned by stereo_range().
  Exactly one of (disparity,ranges) should be given as non-None.

- disparity_scale: optional scale factor for the "disparity" array. If omitted,
  the "disparity" array is assumed to contain the disparities, in pixels.
  Otherwise it contains data in the units of 1/disparity_scale pixels.

- qrect0: optional array of rectified camera0 pixel coordinates corresponding to
  the given disparities. By default, a full disparity image is assumed.
  Otherwise we use the given rectified coordinates. The shape of this array must
  be broadcasting-compatible with the disparity array. See the
  description above.

RETURNED VALUES

- An array of points of the same dimensionality as the input disparity (or
  ranges) array. Contains floating-point data. Invalid or missing points are
  represented as (0,0,0).

    '''

    if (ranges is     None and disparity is     None) or \
       (ranges is not None and disparity is not None):
        raise Exception("Exactly one of (disparity,ranges) must be non-None")

    if ranges is None:
        ranges = stereo_range(disparity,
                              models_rectified,
                              disparity_scale = disparity_scale,
                              qrect0          = qrect0)

    if qrect0 is None:

        W,H = models_rectified[0].imagersize()

        # shape (H,W,2)
        qrect0 = np.ascontiguousarray( \
               nps.mv( nps.cat( *np.meshgrid(np.arange(W,dtype=float),
                                             np.arange(H,dtype=float))),
                       0, -1))

    # shape (..., 3)
    vrect0 = mrcal.unproject(qrect0, *models_rectified[0].intrinsics(),
                             normalize = True)
    # shape (..., 3)
    p_rect0 = vrect0 * nps.dummy(ranges, axis = -1)

    return p_rect0

def match_feature( image0, image1,
                   q0,
                   *,
                   search_radius1,
                   template_size1,
                   q1_estimate      = None,
                   H10              = None,
                   method           = None,
                   visualize        = False,
                   extratitle       = None,
                   return_plot_args = False,
                   **kwargs):

    r'''Find a pixel correspondence in a pair of images

SYNOPSIS

    # Let's assume that the two cameras are roughly observing the plane defined
    # by z=0 in the ref coordinate system. We also have an estimate of the
    # camera extrinsics and intrinsics, so we can construct the homography that
    # defines the relationship between the pixel observations in the vicinity of
    # the q0 estimate. We use this homography to estimate the corresponding
    # pixel coordinate q1, and we use it to transform the search template
    def xy_from_q(model, q):
        v, dv_dq, _ = mrcal.unproject(q, *model.intrinsics(),
                                      get_gradients = True)
        t_ref_cam = model.Rt_ref_cam()[ 3,:]
        R_ref_cam = model.Rt_ref_cam()[:3,:]
        vref      = mrcal.rotate_point_R(R_ref_cam, v)

        # We're looking at the plane z=0, so z = 0 = t_ref_cam[2] + k*vref[2]
        k = -t_ref_cam[2]/vref[2]
        xy = t_ref_cam[:2] + k*vref[:2]

        H_xy_vref = np.array((( -t_ref_cam[2], 0,             xy[0] - k*vref[0]),
                              (             0, -t_ref_cam[2], xy[1] - k*vref[1]),
                              (             0, 0,             1)))

        H_v_q = nps.glue( dv_dq, nps.transpose(v - nps.inner(dv_dq,q)),
                          axis = -1)
        H_xy_q = nps.matmult(H_xy_vref, R_ref_cam, H_v_q)

        return xy, H_xy_q

    xy, H_xy_q0 = xy_from_q(model0, q0)

    v1 = mrcal.transform_point_Rt(model1.Rt_cam_ref(),
                                  np.array((*xy, 0.)))
    q1 = mrcal.project(v1, *model1.intrinsics())

    _, H_xy_q1 = xy_from_q(model1, q1)

    H10 = np.linalg.solve( H_xy_q1, H_xy_q0)


    q1, diagnostics = \
        mrcal.match_feature( image0, image1,
                             q0,
                             H10            = H10,
                             search_radius1 = 200,
                             template_size1 = 17 )

This function wraps the OpenCV cv2.matchTemplate() function to provide
additional functionality. The big differences are

1. The mrcal.match_feature() interface reports a matching pixel coordinate, NOT
   a matching template. The conversions between templates and pixel coordinates
   at their center are tedious and error-prone, and they're handled by this
   function.

2. mrcal.match_feature() can take into account a homography that is applied to
   the two images to match their appearance. The caller can estimate this from
   the relative geometry of the two cameras and the geometry of the observed
   object. If two pinhole cameras are observing a plane in space, a homography
   exists to perfectly represent the observed images everywhere in view. The
   homography can include a scaling (if the two cameras are looking at the same
   object from different distances) and/or a rotation (if the two cameras are
   oriented differently) and/or a skewing (if the object is being observed from
   different angles)

3. mrcal.match_feature() performs simple sub-pixel interpolation to increase the
   resolution of the reported pixel match

4. Visualization capabilities are included to allow the user to evaluate the
   results

It is usually required to pre-filter the images being matched to get good
results. This function does not do this, and it is the caller's job to apply the
appropriate filters.

All inputs and outputs use the (x,y) convention normally utilized when talking
about images; NOT the (y,x) convention numpy uses to talk about matrices. So
template_size1 is specified as (width,height).

The H10 homography estimate is used in two separate ways:

1. To define the image transformation we apply to the template before matching

2. To compute the initial estimate of q1. This becomes the center of the search
   window. We have

   q1_estimate = mrcal.apply_homography(H10, q0)

A common use case is a translation-only homography. This avoids any image
transformation, but does select a q1_estimate. This special case is supported by
this function accepting a q1_estimate argument instead of H10. Equivalently, a
full translation-only homography may be passed in:

  H10 = np.array((( 1., 0., q1_estimate[0]-q0[0]),
                  ( 0., 1., q1_estimate[1]-q0[1]),
                  ( 0., 0., 1.)))

The top-level logic of this function:

1. q1_estimate = mrcal.apply_homography(H10, q0)

2. Select a region in image1, centered at q1_estimate, with dimensions given in
   template_size1

3. Transform this region to image0, using H10. The resulting transformed image
   patch in image0 is used as the template

4. Select a region in image1, centered at q1_estimate, that fits the template
   search_radius1 pixels off center in each dimension

5. cv2.matchTemplate() to search for the template in this region of image1

If the template being matched is out-of-bounds in either image, this function
raises an exception.

If the search_radius1 pushes the search outside of the search image, the search
bounds are reduced to fit into the given image, and the function works as
expected.

If the match fails in some data-dependent way, we return q1 = None instead of
raising an Exception. This can happen if the optimum cannot be found or if
subpixel interpolation fails.

if visualize: we produce a visualization of the best-fitting match. if not
return_plot_args: we display this visualization; else: we return the plot data,
so that we can create the plot later. The diagnostic plot contains 3 overlaid
images:

- The image being searched
- The homography-transformed template placed at the best-fitting location
- The correlation (or difference) image, placed at the best-fitting location

In an interactive gnuplotlib window, each image can be shown/hidden by clicking
on the relevant legend entry at the top-right of the image. Repeatedly toggling
the visibility of the template image is useful to communicate the fit accuracy.
The correlation image is guaranteed to appear at the end of plot_data_tuples, so
it can be omitted by plotting plot_data_tuples[:-1]. Skipping this image is
often most useful for quick human evaluation.

ARGUMENTS

- image0: the first image to use in the matching. This image is cropped, and
  transformed using the H10 homography to produce the matching template. This is
  interpreted as a grayscale image: 2-dimensional numpy array

- image1: the second image to use in the matching. This image is not
  transformed, but cropped to accomodate the given template size and search
  radius. We use this image as the base to compare the template against. The
  same dimensionality, dtype logic applies as with image0

- q0: a numpy array of shape (2,) representing the pixel coordinate in image0
  for which we seek a correspondence in image1

- search_radius1: integer selecting the search window size, in image1 pixels

- template_size1: an integer width or an iterable (width,height) describing the
  size of the template used for matching. If an integer width is given, we use
  (width,width). This is given in image1 coordinates, even though the template
  itself comes from image0

- q1_estimate: optional numpy array of shape (2,) representing the pixel
  coordinate in image1, which is our rough estimate for the camera1 observation
  of the q0 observation in image0. If omitted, H10 specifies the initial
  estimate. Exactly one of (q1_estimate,H10) must be given

- H10: optional numpy array of shape (3,3) containing the homography mapping q0
  to q1 in the vicinity of the match. If omitted, we assume a translation-only
  homography mapping q0 to q1_estimate. Exactly one of (q1_estimate,H10) must be
  given

- method: optional constant, selecting the correlation function used in the
  template comparison. If omitted or None, we default to normalized
  cross-correlation: cv2.TM_CCORR_NORMED. For a description of available methods
  see:

  https://docs.opencv.org/master/df/dfb/group__imgproc__object.html

- visualize: optional boolean, defaulting to False. If True, we generate a plot
  that describes the matching results. This overlays the search image, the
  template, and the matching-output image, shifted to their optimized positions.
  All 3 images are plotted direclty on top of one another. Clicking on the
  legend in the resulting gnuplot window toggles that image on/off, which allows
  the user to see how well things line up. if visualize and not
  return_plot_args: we generate an interactive plot, and this function blocks
  until the interactive plot is closed. if visualize and return_plot_args: we
  generate the plot data and commands, but instead of creating the plot, we
  return these data and commands, for the caller to post-process

- extratitle: optional string to include in the title of the resulting plot.
  Used to extend the default title string. If kwargs['title'] is given, it is
  used directly, and the extratitle is ignored. Used only if visualize

- return_plot_args: boolean defaulting to False. if return_plot_args: we return
  data_tuples, plot_options objects instead of making the plot. The plot can
  then be made with gp.plot(*data_tuples, **plot_options). Useful if we want to
  include this as a part of a more complex plot. Used only if visualize

- **kwargs: optional arguments passed verbatim as plot options to gnuplotlib.
  Useful to make hardcopies, etc. Used only if visualize

RETURNED VALUES

We return a tuple:

- q1: a numpy array of shape (2,): the pixel coordinate in image1 corresponding
  to the given q0. If the computation fails in some data-dependent way, this
  value is None

- diagnostics: a dict containing diagnostics that describe the match. keys:

  - matchoutput_image: the matchoutput array computed by cv2.matchTemplate()

  - matchoutput_optimum_subpixel_at: the subpixel-refined coordinate of the
    optimum in the matchoutput image

  - matchoutput_optimum_subpixel: the value of the subpixel-refined optimum in the
    matchoutput_image

  - qshift_image1_matchoutput: the shift between matchoutput image coords and
    image1 coords. We have

      q1 = diagnostics['matchoutput_optimum_subpixel_at'] +
           diagnostics['qshift_image1_matchoutput']

If visualize and return_plot_args: we return two more elements in the tuple:
data_tuples, plot_options. The plot can then be made with gp.plot(*data_tuples,
**plot_options).

    '''
    try:
        N = len(template_size1)
    except:
        N = 2
        template_size1 = (template_size1, template_size1)
    if N != 2:
        raise Exception(f"template_size1 must be an interable of length 2 OR a scalar. Got an iterable of length {N}")
    for i in range(2):
        if not (isinstance(template_size1[i], int) and template_size1[i] > 0):
            raise Exception(f"Each element of template_size1 must be an integer > 0. Got {template_size1[i]}")

    template_size1 = np.array(template_size1, dtype=int)

    if image0.ndim != 2:
        raise Exception("match_feature() accepts ONLY grayscale images of shape (H,W)")

    if image1.ndim != 2:
        raise Exception("match_feature() accepts ONLY grayscale images of shape (H,W)")

    if (H10 is      None and q1_estimate is     None) or \
       (H10 is not  None and q1_estimate is not None):
        raise Exception("Exactly one of (q1_estimate,H10) must be given")

    q0 = q0.astype(np.float32)

    if H10 is None:
        H10 = np.array((( 1., 0., q1_estimate[0]-q0[0]),
                        ( 0., 1., q1_estimate[1]-q0[1]),
                        ( 0., 0., 1.)), dtype=np.float32)
        q1_estimate = q1_estimate.astype(np.float32)
    else:
        H10 = H10.astype(np.float32)
        q1_estimate = mrcal.apply_homography(H10, q0)

    # I default to normalized cross-correlation. The method arg defaults to None
    # instead of cv2.TM_CCORR_NORMED so that I don't need to import cv2, unless
    # the user actually calls this function
    import cv2
    if method is None:
        method = cv2.TM_CCORR_NORMED

    ################### BUILD TEMPLATE
    # I construct the template I'm searching for. This is a slice of image0 that
    # is
    # - centered at the given q0
    # - remapped using the homography to correct for the geometric
    #   differences in the two images

    q1_template_min = np.round(q1_estimate - (template_size1-1.)/2.).astype(int)
    q1_template_max = q1_template_min + template_size1 - 1 # last pixel


    def checkdims(image_shape, what, *qall):
        for q in qall:
            if q[0] < 0:
                raise Exception(f"Too close to the left edge in {what}")
            if q[1] < 0:
                raise Exception(f"Too close to the top edge in {what} ")
            if q[0] >= image_shape[1]:
                raise Exception(f"Too close to the right edge in {what} ")
            if q[1] >= image_shape[0]:
                raise Exception(f"Too close to the bottom edge in {what} ")

    checkdims( image1.shape,
               "image1",
               q1_template_min,
               q1_template_max)

    # shape (H,W,2)
    q1 = nps.glue(*[ nps.dummy(arr, -1) for arr in \
                     np.meshgrid( np.arange(q1_template_min[0], q1_template_max[0]+1),
                                  np.arange(q1_template_min[1], q1_template_max[1]+1))],
                  axis=-1).astype(np.float32)

    q0 = mrcal.apply_homography(np.linalg.inv(H10), q1)
    checkdims( image0.shape,
               "image0",
               q0[ 0, 0],
               q0[-1, 0],
               q0[ 0,-1],
               q0[-1,-1] )

    image0_template = mrcal.transform_image(image0, q0)


    ################### MATCH TEMPLATE
    q1_min = q1_template_min - search_radius1
    q1_max = q1_template_min + search_radius1 + template_size1 - 1 # last pixel

    # Adjust the bounds in case the search radius pushes us past the bounds of
    # image1
    if q1_min[0] < 0:                    q1_min[0] = 0
    if q1_min[1] < 0:                    q1_min[1] = 0
    if q1_max[0] > image1.shape[-1] - 1: q1_max[0] = image1.shape[-1] - 1
    if q1_max[1] > image1.shape[-2] - 1: q1_max[1] = image1.shape[-2] - 1

    # q1_min, q1_max are now corners of image1 we should search
    image1_cut = image1[ q1_min[1]:q1_max[1]+1, q1_min[0]:q1_max[0]+1 ]

    template_size1_hw = np.array((template_size1[-1],template_size1[-2]))
    matchoutput = np.zeros( image1_cut.shape - template_size1_hw+1, dtype=np.float32 )

    cv2.matchTemplate(image1_cut,
                      image0_template,
                      method, matchoutput)

    if method == cv2.TM_SQDIFF or method == cv2.TM_SQDIFF_NORMED:
        matchoutput_optimum_flatindex = np.argmin( matchoutput.ravel() )
    else:
        matchoutput_optimum_flatindex = np.argmax( matchoutput.ravel() )
    matchoutput_optimum           = matchoutput.ravel()[matchoutput_optimum_flatindex]
    # optimal, discrete-pixel q1 in image1_cut coords
    q1_cut = \
        np.array( np.unravel_index(matchoutput_optimum_flatindex,
                                   matchoutput.shape) )[(-1,-2),]
    diagnostics = \
        dict(matchoutput_image = matchoutput)

    ###################### SUBPIXEL INTERPOLATION
    # I fit a simple quadratic surface to the 3x3 points around the discrete
    # max, and report the max of that fitted surface
    # c = (c00, c10, c01, c20, c11, c02)
    # z = c00 + c10*x + c01*y + c20*x*x + c11*x*y + c02*y*y
    # z ~ M c
    # dz/dx = c10 + 2 c20 x + c11 y = 0
    # dz/dy = c01 + 2 c02 y + c11 x = 0
    # -> [ 2 c20     c11 ] [x] =  [-c10]
    #    [   c11   2 c02 ] [y] =  [-c01]
    #
    # -> xy = -1/(4 c20 c02 - c11^2) [ 2 c02   -c11 ] [c10]
    #                                [  -c11  2 c20 ] [c01]

    default_plot_args = (None,None) if visualize and return_plot_args else ()

    if q1_cut[0] <= 0                        or \
       q1_cut[1] <= 0                        or \
       q1_cut[0] >= matchoutput.shape[-1]-1  or \
       q1_cut[1] >= matchoutput.shape[-2]-1:
        # discrete matchoutput peak at the edge. Cannot compute subpixel
        # interpolation
        return (None, diagnostics) + default_plot_args

    x,y = np.meshgrid( np.arange(3) - 1, np.arange(3) - 1 )
    x = x.ravel().astype(float)
    y = y.ravel().astype(float)
    M = nps.transpose( nps.cat( np.ones(9,),
                                x, y, x*x, x*y, y*y ))
    z = matchoutput[ q1_cut[1]-1:q1_cut[1]+2,
                     q1_cut[0]-1:q1_cut[0]+2 ].ravel()

    # I try rcond = -1 to work in an old numpy
    lsqsq_result = None
    if True:
        try: lsqsq_result = np.linalg.lstsq( M, z, rcond = None)
        except: pass
    if lsqsq_result is None:
        try: lsqsq_result = np.linalg.lstsq( M, z, rcond = -1)
        except: pass
    if lsqsq_result is None:
        return (None, diagnostics) + default_plot_args

    c = lsqsq_result[0]
    (c00, c10, c01, c20, c11, c02) = c
    det = 4.*c20*c02 - c11*c11
    xy_subpixel = -np.array((2.*c10*c02 - c01*c11,
                             2.*c01*c20 - c10*c11)) / det
    x,y = xy_subpixel
    matchoutput_optimum_subpixel = c00 + c10*x + c01*y + c20*x*x + c11*x*y + c02*y*y
    q1_cut = q1_cut.astype(float) + xy_subpixel

    diagnostics['matchoutput_optimum_subpixel_at'] = q1_cut
    diagnostics['matchoutput_optimum_subpixel']    = matchoutput_optimum_subpixel

    # The translation to pixel coordinates

    # Top-left pixel of the template, in image1 coordinates
    q1_aligned_template_topleft = q1_min + q1_cut

    # Shift for the best-fitting pixel of image1 of the template center
    qshift_image1_matchoutput = \
        q1_min +                \
        q1_estimate -           \
        q1_template_min
    diagnostics['qshift_image1_matchoutput'] = qshift_image1_matchoutput

    # the best-fitting pixel of image1 of the template center

    q1 = q1_cut + qshift_image1_matchoutput

    matchoutput_min = np.min(matchoutput)
    matchoutput_max = np.max(matchoutput)

    if not visualize:
        return q1, diagnostics

    import gnuplotlib as gp

    plot_options = dict(kwargs)

    if 'title' not in plot_options:
        title   = 'Feature-matching results'
        if extratitle is not None:
            title += ": " + extratitle
        plot_options['title'] = title

    gp.add_plot_option(plot_options,
                       _with     = 'image',
                       ascii     = True,
                       overwrite = True)
    gp.add_plot_option(plot_options,
                       square    = True,
                       yinv      = True,
                       _set      = 'palette gray',
                       overwrite = False)

    data_tuples = \
        ( ( image1_cut,
            dict(legend='image',
                 using = f'($1 + {q1_min[0]}):($2 + {q1_min[1]}):3',
                 tuplesize = 3)),
          ( image0_template,
            dict(legend='template',
                 using = \
                 f'($1 + {q1_aligned_template_topleft[0]}):' + \
                 f'($2 + {q1_aligned_template_topleft[1]}):3',
                 tuplesize = 3)),
          ( (matchoutput - matchoutput_min) /
            (matchoutput_max - matchoutput_min) * 255,
            dict(legend='matchoutput',
                 using = \
                 f'($1 + {qshift_image1_matchoutput[0]}):' + \
                 f'($2 + {qshift_image1_matchoutput[1]}):3',
                 tuplesize = 3)) )

    if return_plot_args:
        return q1, diagnostics, data_tuples, plot_options

    gp.plot( *data_tuples, **plot_options, wait=True)
    return q1, diagnostics
