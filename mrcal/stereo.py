#!/usr/bin/python3

'''Routines for stereo processing: rectification and ranging

All functions are exported into the mrcal module. So you can call these via
mrcal.stereo.fff() or mrcal.fff(). The latter is preferred.
'''

import sys
import numpy as np
import numpysane as nps
import mrcal

def rectified_system(models,
                     az_fov_deg,
                     el_fov_deg,
                     az0_deg           = None,
                     el0_deg           = 0,
                     pixels_per_deg_az = None,
                     pixels_per_deg_el = None):

    r'''Build rectified models for stereo rectification

SYNOPSIS

    import sys
    import mrcal
    import cv2
    import numpy as np

    models = [ mrcal.cameramodel(f) \
               for f in ('left.cameramodel',
                         'right.cameramodel') ]

    images = [ cv2.imread(f) \
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
    r = mrcal.stereo_range( disparity16.astype(np.float32) / 16.,
                            models_rectified )

This function computes the geometry of a rectified system from the two
cameramodels describing the stereo pair. The output is a pair of "rectified"
models. Each of these is a normal mrcal.cameramodel object describing a camera
somewhere in space, with some particular projection behavior. The pair of models
returned here have the desired property that each row of pixels represents a
plane in space AND each corresponding row in the pair of rectified images
represents the SAME plane. The we can use the rectified models returned by this
function to transform the input images into "rectified" images, and then we can
perform stereo matching effiently, by treating each row independently.

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

The lensmodel used in the cameramodels we return is LENSMODEL_LATLON (described
in detail here: http://mrcal.secretsauce.net/lensmodels.html#lensmodel-latlon ).
This lens model has even angular spacing between pixels and has the desired
property of mapping rows of pixels to planes in space.

ARGUMENTS

- models: an iterable of two mrcal.cameramodel objects representing the cameras
  in the stereo pair. Any sane combination of lens models and resolutions and
  geometries is valid

- az_fov_deg: required value for the azimuth (along-the-baseline) field-of-view
  of the desired rectified system, in degrees

- el_fov_deg: required value for the elevation (across-the-baseline)
  field-of-view of the desired rectified system, in degrees

- az0_deg: optional value for the azimuth center of the rectified images. This
  is especially significant in a camera system with some forward/backward shift.
  That causes the baseline to no longer be perpendicular with the view axis of
  the cameras, and thus the azimuth=0 vector no longer points "forward". If
  omitted, we compute az0_deg to aligns the center of the rectified system with
  the center of the two cameras' views

- el0_deg: optional value for the elevation center of the rectified system.
  Defaults to 0.

- pixels_per_deg_az: optional value for the azimuth resolution of the rectified
  image. If omitted (or None), we use the resolution of the input image at the
  center of the rectified system. If a resolution of <0 is requested, we use
  this as a scale factor on the resolution of the input image. For instance, to
  downsample by a factor of 2, pass pixels_per_deg_az = -0.5

- pixels_per_deg_el: same as pixels_per_deg_az but in the elevation direction

RETURNED VALUES

We return a tuple of mrcal.cameramodels describing the two rectified cameras.
These two models are identical, except for a baseline translation in the +x
direction in rectified coordinates.

    '''

    if len(models) != 2:
        raise Exception("I need exactly 2 camera models")

    ######## Compute the geometry of the rectified stereo system. This is a
    ######## rotation, centered at camera0. More or less we have axes:
    ########
    ######## x: from camera0 to camera1
    ######## y: completes the system from x,z
    ######## z: component of the cameras' viewing direction
    ########    normal to the baseline
    Rt01 = mrcal.compose_Rt( models[0].extrinsics_Rt_fromref(),
                             models[1].extrinsics_Rt_toref())

    # Rotation relating camera0 coords to the rectified camera coords. I fill in
    # each row separately
    R_rect0_cam0 = np.zeros((3,3), dtype=float)

    # Axes of the rectified system, in the cam0 coord system
    right       = R_rect0_cam0[0,:]
    down        = R_rect0_cam0[1,:]
    forward     = R_rect0_cam0[2,:]

    # "right" of the rectified coord system: towards the origin of camera1 from
    # camera0, in camera0 coords
    right += Rt01[3,:]
    baseline = nps.mag(right)
    right   /= baseline

    # "forward" for each of the two cameras, in the cam0 coord system
    forward0 = np.array((0,0,1.))
    forward1 = Rt01[:3,2]

    # "forward" of the rectified coord system, in camera0 coords. The mean
    # optical-axis direction of the two cameras: component orthogonal to "right"
    forward01 = forward0 + forward1
    forward01_proj_right = nps.inner(forward01,right)
    forward += forward01 - forward01_proj_right*right
    forward /= nps.mag(forward)

    # "down" of the rectified coord system, in camera0 coords. Completes the
    # right,down,forward coordinate system
    down[:] = np.cross(forward,right)

    # All components of R_rect0_cam0 are now filled in

    R_cam0_rect0 = nps.transpose(R_rect0_cam0)

    # rect1 coord system has the same orientation as rect0, but is translated so
    # that its origin is at the origin of cam1
    R_rect1_cam0  = R_rect0_cam0
    R_rect1_cam1  = nps.matmult(R_rect1_cam0, Rt01[:3,:])

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

    el0 = el0_deg * np.pi/180.

    ####### Rectified image resolution
    if pixels_per_deg_az is None or pixels_per_deg_az < 0 or \
       pixels_per_deg_el is None or pixels_per_deg_el < 0:
        # I need to compute the resolution of the rectified images. I try to
        # match the resolution of the cameras. I just look at camera0. If your
        # two cameras are different, pass in the pixels_per_deg yourself
        #
        # I look at the center of the stereo field of view. There I have q =
        # project(v) where v is a unit projection vector. I compute dq/dth where
        # th is an angular perturbation applied to v.
        v,dv_dazel = mrcal.unproject_latlon( np.array((az0,el0)),
                                             get_gradients = True )
        v0         = mrcal.rotate_point_R(R_cam0_rect0, v)
        dv0_dazel  = nps.matmult(R_cam0_rect0, dv_dazel)

        _,dq_dv0,_ = mrcal.project(v0, *models[0].intrinsics(), get_gradients = True)

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
        #     def rotation_any_v_to_z(v):
        #         r'''Return any rotation matrix that maps the given unit vector v to [0,0,1]'''
        #         z = v
        #         if np.abs(v[0]) < .9:
        #             x = np.array((1,0,0))
        #         else:
        #             x = np.array((0,1,0))
        #         x -= nps.inner(x,v)*v
        #         x /= nps.mag(x)
        #         y = np.cross(z,x)
        #         return nps.cat(x,y,z)
        #     Ruv = rotation_any_v_to_z(v0)
        #     M = nps.matmult(dq_dv0, nps.transpose(Ruv[:2,:]))
        #     # I pick the densest direction: highest |dq/dth|
        #     pixels_per_rad = mrcal.worst_direction_stdev( nps.matmult( nps.transpose(M),M) )

        dq_dazel = nps.matmult(dq_dv0, dv0_dazel)

        if pixels_per_deg_az is None or pixels_per_deg_az < 0:
            pixels_per_deg_az_have = nps.mag(dq_dazel[:,0])*np.pi/180.

            if pixels_per_deg_az is not None:
                # negative pixels_per_deg_az requested means I use the requested
                # value as a scaling
                pixels_per_deg_az = -pixels_per_deg_az * pixels_per_deg_az_have
            else:
                pixels_per_deg_az = pixels_per_deg_az_have

        if pixels_per_deg_el is None or pixels_per_deg_el < 0:
            pixels_per_deg_el_have = nps.mag(dq_dazel[:,1])*np.pi/180.

            if pixels_per_deg_el is not None:
                # negative pixels_per_deg_el requested means I use the requested
                # value as a scaling
                pixels_per_deg_el = -pixels_per_deg_el * pixels_per_deg_el_have
            else:
                pixels_per_deg_el = pixels_per_deg_el_have

    Naz = round(az_fov_deg*pixels_per_deg_az)
    Nel = round(el_fov_deg*pixels_per_deg_el)

    # Adjust resolution to be consistent with my just-computed integer pixel
    # counts
    pixels_per_deg_az = Naz/az_fov_deg
    pixels_per_deg_el = Nel/el_fov_deg


    ######## Construct rectified models. These can be used for all further
    ######## stereo processing
    Rt_rect0_cam0 = nps.glue(R_rect0_cam0, np.zeros((3,),), axis=-2)
    Rt_rect0_ref  = mrcal.compose_Rt( Rt_rect0_cam0,
                                      models[0].extrinsics_Rt_fromref())
    # rect1 coord system has the same orientation as rect0, but is translated so
    # that its origin is at the origin of cam1
    Rt_rect1_cam1 = nps.glue(R_rect1_cam1, np.zeros((3,),), axis=-2)
    Rt_rect1_ref  = mrcal.compose_Rt( Rt_rect1_cam1,
                                      models[1].extrinsics_Rt_fromref())

    fxycxy = np.zeros((4,), dtype=float)
    fxycxy[:2] = np.array((pixels_per_deg_az, pixels_per_deg_el)) / np.pi*180.
    # (az0,el0) = unproject(imager center)
    v = mrcal.unproject_latlon( np.array((az0,el0)) )
    fxycxy[2:] = \
        np.array(((Naz-1.)/2.,(Nel-1.)/2.)) - \
        mrcal.project_latlon( v, *fxycxy[:2] )

    models_rectified = \
        ( mrcal.cameramodel( intrinsics = ('LENSMODEL_LATLON', fxycxy),
                             imagersize = (Naz, Nel),
                             extrinsics_Rt_fromref = Rt_rect0_ref),

          mrcal.cameramodel( intrinsics = ('LENSMODEL_LATLON', fxycxy),
                             imagersize = (Naz, Nel),
                             extrinsics_Rt_fromref = Rt_rect1_ref) )

    return models_rectified


def _validate_models_rectified(models_rectified):
    r'''Internal function to validate a rectified system

These should have been returned by rectified_system(). Should have two
LENSMODEL_LATLON cameras with identical intrinsics. extrinsics should be
identical too EXCEPT for a baseline translation in the x rectified direction

    '''

    if len(models_rectified) != 2:
        raise Exception(f"Must have received exactly two models. Got {len(models_rectified)} instead")

    intrinsics = [m.intrinsics() for m in models_rectified]
    Rt01 = mrcal.compose_Rt( models_rectified[0].extrinsics_Rt_fromref(),
                             models_rectified[1].extrinsics_Rt_toref())

    if intrinsics[0][0] != 'LENSMODEL_LATLON' or \
       intrinsics[1][0] != 'LENSMODEL_LATLON':
        raise Exception(f"Expected lensmodel 'LENSMODEL_LATLON' but got {intrinsics[0][0]} and {intrinsics[1][0]}")

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

    models = [ mrcal.cameramodel(f) \
               for f in ('left.cameramodel',
                         'right.cameramodel') ]

    images = [ cv2.imread(f) \
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
    r = mrcal.stereo_range( disparity16.astype(np.float32) / 16.,
                            models_rectified )

After the pair of rectified models has been built by mrcal.rectified_system(),
this function can be called to compute the rectification maps. These can be
passed to mrcal.transform_image() to remap input images into the rectified
space.

The documentation for mrcal.rectified_system() applies here.

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
    fxycxy  = models_rectified[0].intrinsics()[1]

    R_cam_rect = [ nps.matmult(models          [i].extrinsics_Rt_fromref()[:3,:],
                               models_rectified[i].extrinsics_Rt_toref  ()[:3,:]) \
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
    v = mrcal.unproject_latlon( np.ascontiguousarray( \
           nps.mv( nps.cat( *np.meshgrid(np.arange(Naz,dtype=float),
                                         np.arange(Nel,dtype=float))),
                   0, -1)),
                                 *fxycxy)

    v0 = mrcal.rotate_point_R(R_cam_rect[0], v)
    v1 = mrcal.rotate_point_R(R_cam_rect[1], v)

    return                                                                \
        (mrcal.project( v0, *models[0].intrinsics()).astype(np.float32),  \
         mrcal.project( v1, *models[1].intrinsics()).astype(np.float32))


def stereo_range(disparity_pixels,
                 models_rectified,
                 az_deg = None):

    r'''Compute ranges from observed disparities

SYNOPSIS

    import sys
    import mrcal
    import cv2
    import numpy as np

    models = [ mrcal.cameramodel(f) \
               for f in ('left.cameramodel',
                         'right.cameramodel') ]

    images = [ cv2.imread(f) \
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
    r = mrcal.stereo_range( disparity16.astype(np.float32) / 16.,
                            models_rectified )

As shown in the example above, we can perform stereo processing by building
rectified models and transformation maps, rectifying our images, and then doing
stereo matching to get pixel disparities. This function performs the last step:
converting pixel disparities to ranges.

In the most common usage we take a full disparity IMAGE, and then convert it to
a range IMAGE. In this common case we call

    range_image = mrcal.stereo_range(disparity_image, models_rectified)

If we aren't processing the full disparity image, we can pass in an array of
azimuths in the "az_deg" argument. These must be broadcastable with the
disparity_pixels argument. So we can pass in a scalar for disparity_pixels and
az_deg. or we can pass in full shape (H,W) images for both. Or we can pass in a
shape (H,W) image for disparity_pixels, but only a shape (W,) array for az_deg:
this would use the same az value for a whole column of disparity_pixels, as
dictated by the broadcasting rules. Such identical-az-in-a-column behavior is
valid in our current rectification system.

A top view onto an epipolar plane:

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
camera1 makes an angle az1 (here az1 < 0). A disparity is a difference of
azimuths: disparity ~ az0-az1.

The law of sines tells us that

    baseline / sin(a) = range / sin(90 + az1)

Thus

    range = baseline cos(az1) / sin(a) =
          = baseline cos(az1) / sin( 180 - (90-az0 + 90+az1) ) =
          = baseline cos(az1) / sin(az0 - az1) =
          = baseline cos(az1) / sin(disparity)
          = baseline cos(az0 - disparity) / sin(disparity)

ARGUMENTS

- disparity_pixels: a numpy array of disparities being processed. This array
  contains disparity in PIXELS. If the stereo-matching algorithm you're using
  reports a scaled value, you need to convert it to pixels. Usually this is a
  floating-point array. Any array shape is supported. In the common case of a
  disparity IMAGE, this is an array of shape (Nel, Naz)

- models_rectified: the pair of rectified models, corresponding to the input
  images. Usually this is returned by mrcal.rectified_system()

- az_deg: optional array of azimuths corresponding to the given disparities. By
  default, a full disparity image is assumed. Otherwise we use the given
  azimuths. The shape of this array must be broadcasting-compatible with the the
  disparity_pixels array. See the description above.

RETURNED VALUES

- An array of ranges of the same dimensionality as the input disparity_pixels
  array. Contains floating-point data. Invalid or missing ranges are represented
  as 0.

    '''

    _validate_models_rectified(models_rectified)

    intrinsics = models_rectified[0].intrinsics()

    fx  = intrinsics[1][0]
    cx  = intrinsics[1][2]

    W,H = models_rectified[0].imagersize()
    if az_deg is None:
        if disparity_pixels.shape != (H,W):
            raise Exception(f"az_deg is None, so the disparity image must have the full dimensions of a rectified image")

        az = (np.arange(W, dtype=float) - cx)/fx

    else:
        az = az_deg * np.pi/180.

    pixels_per_rad_az = fx
    disparity_rad = disparity_pixels / pixels_per_rad_az

    mask_invalid = (disparity_pixels <= 0)

    s = np.sin(disparity_rad)
    s[mask_invalid] = 1 # to prevent division by 0

    Rt01 = mrcal.compose_Rt( models_rectified[0].extrinsics_Rt_fromref(),
                             models_rectified[1].extrinsics_Rt_toref())
    baseline = nps.mag(Rt01[3,:])

    r = baseline * np.cos(az - disparity_rad) / s
    r[mask_invalid] = 0

    return r


def match_feature( image0, image1,
                   template_size, # (width,height)
                   method,
                   search_radius,
                   q0,
                   H10,
                   visualize = False):

    r'''Find a pixel correspondence in a pair of images

SYNOPSIS

    q1_matched, diagnostics = \
        mrcal.match_feature( image0, image1,
                             template_size = (17,17),
                             method        = cv2.TM_CCORR_NORMED,
                             search_radius = 20,
                             q0,  # pixel coordinate in image0
                             H10, # homography mapping q0 to q1
                           )

    import gnuplotlib as gp
    gp.plot( * diagnostics['plot_data_tuples'],
             **diagnostics['plot_options'],
             wait=True)

This function wraps cv2.matchTemplate() to provide additional functionality. The
big differences are

1. The mrcal.match_feature() interface reports a matching pixel coordinate, NOT
   a matching template. The conversions between templates and pixel coordinates
   at their center are tedious and error-prone, and they're handled by this
   function.

2. mrcal.match_feature() can take into account a transformation that must be
   applied to the two images to match their appearance. The caller can estimate
   this from the relative geometry of the two cameras and the geometry of the
   observed object. This transformation can include a scaling (if the two
   cameras are looking at the same object from different distances) and/or a
   rotation (if the two cameras are oriented differently) or a skewing (if the
   object is being observed from different angles)

3. mrcal.match_feature() performs simple sub-pixel interpolation to increase the
   resolution of the reported pixel match

4. Some visualization capabilities are included to allow the user to evaluate
   the results

All inputs and outputs use the (x,y) convention normally utilized when talking
about images; NOT the (y,x) convention numpy uses to talk about matrices. So
template_size is (width,height).

The H10 homography estimate is used in two separate ways:

1. To define the image transformation we apply to the template before matching

2. To compute the initial estimate of q1. This becomes the center of the search
   window. We have

   q1_estimate = mrcal.apply_homography(H10, q0)

A common use case is to pass a translation-only homography. This avoids any
image transformation, but does select a q1_estimate. So if we want to find the
corresponding q1 for a given q0, with a q1_estimate, and without an image
transformation, pass in

  H10 = np.array((( 1., 0., q1_estimate[0]-q0[0]),
                  ( 0., 1., q1_estimate[1]-q0[1]),
                  ( 0., 0., 1.)))

If the template being matched is out-of-bounds in either image, this function
raises an exception.

If the search_radius pushes the search outside of the search image, the search
bounds are reduced to fit into the given image, and the function works as
expected.

If the match fails in some data-dependent way (i.e. not caused by a likely bug),
we return q1 = None. This can happen if the optimum cannot be found or if
subpixel interpolation fails.

ARGUMENTS

- image0: the first image to use in the matching. This image is cropped, and
  transformed using the H10 homography to produce the matching template. This
  MUST be a 2-dimensional numpy array: only grayscale images are accepted.

- image1: the second image to use in the matching. This image is not transformed
  or cropped. We use this image as the base to compare the template agianst. As
  with image 0, this MUST be a 2-dimensional numpy array: only grayscale images
  are accepted.

- template_size: the (width,height) iterable describing the size of the template
  used for the matching

- method: one of the methods accepted by cv2.matchTemplate() See

  https://docs.opencv.org/master/df/dfb/group__imgproc__object.html

  The most common method is normalized cross-correlation, selected by

  method = cv2.TM_CCORR_NORMED

- search_radius: integer selecting how far we should search for the match

- q0: a numpy array of shape (2,) representing the pixel coordinate in image0
  for which we see a correspondence in image1

- H10: the homography mapping q0 to q1 in the vicinity of the match

- visualize: optional boolean, defaulting to False. If True, an interactive plot
  pops up, describing the results. This overlays the search image, the template,
  and the matching-output image, shifted to their optimized positions. All 3
  images are plotted direclty on top of one another. Clicking on the legend in
  the resulting gnuplot window toggles that image on/off, which allows the user
  to see how well things line up. Even if not visualize: the plot components are
  still returned in the diagnostics, and the plot can be created later (such as
  when we find out that something failed and needs investigating)

RETURNED VALUES

We return a tuple:

- q1: a numpy array of shape (2,) representing the pixel coordinate in image1
  corresponding to the given image0. If the computation fails in some data-dependent way (not caused by a likely bug), this value is None

- diagnostics: a dict containing diagnostics that can be used to interpret the
  match. Keys:

  - matchoutput_image: the matchoutput array computed by cv2.matchTemplate()

  - matchoutput_optimum_at: the integer pixel of the optimum in the
    matchoutput_image

  - matchoutput_optimum: the value of the optimum in the matchoutput image

  - matchoutput_optimum_subpixel_at: the subpixel-refined coordinate of the optimum
    in the matchoutput image

  - matchoutput_optimum_subpixel: the value of the subpixel-refined optimum in the
    matchoutput_image

  - qshift_image1_matchoutput: the shift between matchoutput image coords and
    image1 coords. We have

      q1 = diagnostics['matchoutput_optimum_subpixel_at'] +
           diagnostics['qshift_image1_matchoutput']

  - plot_data_tuples, plot_options: gnuplotlib structures that can be used to
    create the diagnostic plot like this:

    import gnuplotlib as gp
    gp.plot( * diagnostics['plot_data_tuples'],
             **diagnostics['plot_options'],
             wait=True)

    The diagnostic plot contains 3 overlaid images:

    - The image being searched
    - The homography-transformed template placed at the best-fitting location
    - The correlation (or difference) image, placed at the best-fitting location

    In an interactive gnuplotlib window, each image can be shown/hidden by
    clicking on the relevant legend entry at the top-right of the image.
    Repeatedly toggling the visibility of the template image is useful to
    communicate the fit accuracy. The correlation image is guaranteed to appear
    at the end of the diagnostics['plot_data_tuples'] tuple, so it can be
    omitted by plotting diagnostics['plot_data_tuples'][:-1]. Skipping this
    image is often most useful for quick human evaluation.

    '''

    if image0.ndim != 2 or image1.ndim != 2:
        raise Exception("match_feature() works with grayscale images only: both images MUST have exactly 2 dimensions")

    import cv2

    H10           = H10.astype(np.float32)
    q0            = q0 .astype(np.float32)
    template_size = np.array(template_size)

    ################### BUILD TEMPLATE
    # I construct the template I'm searching for. This is a slice of image0 that
    # is
    # - centered at the given q0
    # - remapped using the homography to correct for the geometric
    #   differences in the two images

    q1_estimate = mrcal.apply_homography(H10, q0)

    q1_template_min = np.round(q1_estimate - (template_size-1.)/2.).astype(int)
    q1_template_max = q1_template_min + template_size

    # (W,H)
    image1_dim = np.array((image1.shape[-1],image1.shape[-2]))

    def checkdims(w, h, what, *qall):
        for q in qall:
            if q[0] < 0:
                raise Exception(f"Too close to the left edge in {what}")
            if q[1] < 0:
                raise Exception(f"Too close to the top edge in {what} ")
            if q[0] >= w:
                raise Exception(f"Too close to the right edge in {what} ")
            if q[1] >= h:
                raise Exception(f"Too close to the bottom edge in {what} ")

    checkdims( *image1_dim, "image1",
               q1_template_min,
               q1_template_max-1)

    # shape (H,W,2)
    q1 = nps.glue(*[ nps.dummy(arr, -1) for arr in \
                     np.meshgrid( np.arange(q1_template_min[0], q1_template_max[0]),
                                  np.arange(q1_template_min[1], q1_template_max[1]))],
                  axis=-1).astype(np.float32)

    q0 = mrcal.apply_homography(np.linalg.inv(H10), q1)
    checkdims( image0.shape[-1], image0.shape[-2],
               "image0",
               q1[ 0, 0],
               q1[-1, 0],
               q1[ 0,-1],
               q1[-1,-1] )

    image0_template = mrcal.transform_image(image0, q0)


    ################### MATCH TEMPLATE
    q1_min = q1_template_min - search_radius
    q1_max = q1_template_min + search_radius + template_size # one past the edge

    # the margins are needed in case we get past the edges
    q1_min_margin = -np.clip(q1_min,
                             a_min = None,
                             a_max = 0)
    q1_min       += q1_min_margin

    q1_max_margin = -np.clip(image1_dim - q1_max,
                             a_min = None,
                             a_max = 0)
    q1_max       -= q1_max_margin

    # q1_min, q1_max are now corners of the template. There are in-bounds of
    # image1. The margins are the sizes of the out-of-bounds region
    image1_cut = image1[ q1_min[1]:q1_max[1], q1_min[0]:q1_max[0] ]

    template_size_hw = np.array((template_size[-1],template_size[-2]))
    matchoutput = np.zeros( image1_cut.shape - template_size_hw+1, dtype=np.float32 )
    cv2.matchTemplate(image1_cut,
                      image0_template,
                      method, matchoutput)

    # the best-fitting pixel of q1_cut of the template origin
    if method == cv2.TM_SQDIFF or method == cv2.TM_SQDIFF_NORMED:
        matchoutput_optimum_flatindex = np.argmin( matchoutput.ravel() )
    else:
        matchoutput_optimum_flatindex = np.argmax( matchoutput.ravel() )
    matchoutput_optimum           = matchoutput.ravel()[matchoutput_optimum_flatindex]
    q1_in_corr = \
        np.array( np.unravel_index(matchoutput_optimum_flatindex,
                                   matchoutput.shape) )[(-1,-2),]
    diagnostics = \
        dict(matchoutput_image      = matchoutput,
             matchoutput_optimum_at = q1_in_corr.copy(),
             matchoutput_optimum    = matchoutput_optimum)

    ###################### SUBPIXEL INTERPOLATION
    # I fit a simple quadratic surface to the 3x3 points around the discrete
    # max, and report the max of that fitted surface
    # c = (c00, c10, c01, c20, c11, c02)
    # z = c00 + c10*x + c01*y + c20*x*x + c11*x*y + c02*y*y
    # z ~ M c
    # dz/dx = c10 + 2 c20 x + c11 y = 0
    # dz/dy = c01 + 2 c02 y + c11 x = 0
    # -> [ 2 c20    c11 ] [x] =  [-c10]
    #    [  c11   2 c02 ] [y] =  [-c01]
    #
    # -> xy = -1/(4 c20 c02 - c11^2) [ 2 c02   -c11 ] [c10]
    #                                [  -c11  2 c20 ] [c01]

    if q1_in_corr[0] <= 0                        or \
       q1_in_corr[1] <= 0                        or \
       q1_in_corr[0] >= matchoutput.shape[-1]-1  or \
       q1_in_corr[1] >= matchoutput.shape[-2]-1:
        # discrete matchoutput peak at the edge. Cannot compute subpixel
        # interpolation
        return None, diagnostics

    x,y = np.meshgrid( np.arange(3) - 1, np.arange(3) - 1 )
    x = x.ravel().astype(float)
    y = y.ravel().astype(float)
    M = nps.transpose( nps.cat( np.ones(9,),
                                x, y, x*x, x*y, y*y ))
    z = matchoutput[ q1_in_corr[1]-1:q1_in_corr[1]+2,
                     q1_in_corr[0]-1:q1_in_corr[0]+2 ].ravel()
    try:
        lsqsq_result = np.linalg.lstsq( M, z, rcond = None)
    except:
        return None, diagnostics

    c = lsqsq_result[0]
    (c00, c10, c01, c20, c11, c02) = c
    det = 4.*c20*c02 - c11*c11
    xy_subpixel = -np.array((2.*c10*c02 - c01*c11,
                             2.*c01*c20 - c10*c11)) / det
    x,y = xy_subpixel
    matchoutput_optimum_subpixel = c00 + c10*x + c01*y + c20*x*x + c11*x*y + c02*y*y
    q1_in_corr = q1_in_corr + xy_subpixel

    diagnostics['matchoutput_optimum_subpixel_at'] = q1_in_corr
    diagnostics['matchoutput_optimum_subpixel']    = matchoutput_optimum_subpixel

    # The translation to pixel coordinates
    #
    # Shift for the best-fitting pixel of image1 of the template origin
    qshift_image1_matchoutput = q1_min

    # Top-left pixel of the template, in image1 coordinates
    q1_aligned_template_topleft = q1_in_corr + q1_min

    # Shift for the best-fitting pixel of image1 of the template center
    qshift_image1_matchoutput =     \
        qshift_image1_matchoutput + \
        q1_estimate -               \
        q1_template_min
    diagnostics['qshift_image1_matchoutput'] = qshift_image1_matchoutput

    # the best-fitting pixel of image1 of the template center
    q1 = q1_in_corr + qshift_image1_matchoutput

    matchoutput_min = np.min(matchoutput)
    matchoutput_max = np.max(matchoutput)

    plot_options = dict( _with     = 'image',
                         square    = True,
                         yinv      = True,
                         _set      = 'palette gray',
                         tuplesize = 3,
                         ascii     = True)
    data_tuples = \
        ( ( image1_cut,
            dict(legend='image',
                 using = f'($1 + {q1_min[0]}):($2 + {q1_min[1]}):3')),
          ( image0_template,
            dict(legend='template',
                 using = \
                 f'($1 + {q1_aligned_template_topleft[0]}):' + \
                 f'($2 + {q1_aligned_template_topleft[1]}):3')),
          ( (matchoutput - matchoutput_min) /
            (matchoutput_max - matchoutput_min) * 255,
            dict(legend='matchoutput',
                 using = \
                 f'($1 + {qshift_image1_matchoutput[0]}):' + \
                 f'($2 + {qshift_image1_matchoutput[1]}):3')) )

    if visualize:
        import gnuplotlib as gp
        gp.plot( *data_tuples, **plot_options, wait=True)

    diagnostics['plot_data_tuples'] = data_tuples
    diagnostics['plot_options'    ] = plot_options

    return q1, diagnostics
