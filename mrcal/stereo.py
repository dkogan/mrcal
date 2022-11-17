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
                     *,
                     az_fov_deg,
                     el_fov_deg,
                     az0_deg             = None,
                     el0_deg             = 0,
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

    Rt_cam0_rect0 = mrcal.compose_Rt( models          [0].extrinsics_Rt_fromref(),
                                      models_rectified[0].extrinsics_Rt_toref() )

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
  http://mrcal.secretsauce.net/lensmodels.html#lensmodel-latlon

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

- pixels_per_deg_az: optional value for the azimuth resolution of the rectified
  image. If a resultion of >0 is requested, the value is used as is. If a
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

    cos_az0 = np.cos(az0)
    cos_el0 = np.cos(el0)

    ####### Rectified image resolution
    if pixels_per_deg_az < 0 or \
       pixels_per_deg_el < 0:
        # I need to compute the resolution of the rectified images. I try to
        # match the resolution of the cameras. I just look at camera0. If your
        # two cameras are different, pass in the pixels_per_deg yourself
        #
        # I look at the center of the stereo field of view. There I have q =
        # project(v) where v is a unit projection vector. I compute dq/dth where
        # th is an angular perturbation applied to v.

        if rectification_model == 'LENSMODEL_LATLON':
            q0_normalized = np.array((az0,el0))
            v,dv_dazel = \
                mrcal.unproject_latlon( q0_normalized,
                                        get_gradients = True )
        else:
            q0_normalized = np.array((np.tan(az0),np.tan(el0)))
            v,dv_dq0normalized = \
                mrcal.unproject_pinhole( q0_normalized,
                                         get_gradients = True )
            # dq/dth = dtanth/dth = 1/cos^2(th)
            dv_dazel = dv_dq0normalized
            dv_dazel[:,0] /= cos_az0*cos_az0
            dv_dazel[:,1] /= cos_el0*cos_el0

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

        if pixels_per_deg_az < 0:
            pixels_per_deg_az_have = nps.mag(dq_dazel[:,0])*np.pi/180.
            pixels_per_deg_az = -pixels_per_deg_az * pixels_per_deg_az_have

        if pixels_per_deg_el < 0:
            pixels_per_deg_el_have = nps.mag(dq_dazel[:,1])*np.pi/180.
            pixels_per_deg_el = -pixels_per_deg_el * pixels_per_deg_el_have

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
        v = mrcal.unproject_latlon( np.array((az0,el0)) )
        fxycxy[2:] = \
            np.array(((Naz-1.)/2.,(Nel-1.)/2.)) - \
            mrcal.project_latlon( v, fxycxy )

    elif rectification_model == 'LENSMODEL_PINHOLE':
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

            # I solve my quadratic polynomial numerically. I get 4 solutions,
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

            # I should have exactly one solution let. Due to some numerical
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
    Rt_rect0_cam0 = nps.glue(R_rect0_cam0, np.zeros((3,),), axis=-2)
    Rt_rect0_ref  = mrcal.compose_Rt( Rt_rect0_cam0,
                                      models[0].extrinsics_Rt_fromref())
    # rect1 coord system has the same orientation as rect0, but is translated so
    # that its origin is at the origin of cam1
    Rt_rect1_cam1 = nps.glue(R_rect1_cam1, np.zeros((3,),), axis=-2)
    Rt_rect1_ref  = mrcal.compose_Rt( Rt_rect1_cam1,
                                      models[1].extrinsics_Rt_fromref())

    models_rectified = \
        ( mrcal.cameramodel( intrinsics = (rectification_model, fxycxy),
                             imagersize = (Naz, Nel),
                             extrinsics_Rt_fromref = Rt_rect0_ref),

          mrcal.cameramodel( intrinsics = (rectification_model, fxycxy),
                             imagersize = (Naz, Nel),
                             extrinsics_Rt_fromref = Rt_rect1_ref) )

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
    Rt01 = mrcal.compose_Rt( models_rectified[0].extrinsics_Rt_fromref(),
                             models_rectified[1].extrinsics_Rt_toref())

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

    Rt_cam0_rect0 = mrcal.compose_Rt( models          [0].extrinsics_Rt_fromref(),
                                      models_rectified[0].extrinsics_Rt_toref() )

    # Point cloud in camera-0 coordinates
    # shape (H,W,3)
    p_cam0 = mrcal.transform_point_Rt(Rt_cam0_rect0, p_rect0)

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

    Naz,Nel     = models_rectified[0].imagersize()
    fxycxy      = models_rectified[0].intrinsics()[1]
    fx,fy,cx,cy = fxycxy

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
    if models_rectified[0].intrinsics()[0] == 'LENSMODEL_LATLON':
        unproject = mrcal.unproject_latlon
    else:
        unproject = mrcal.unproject_pinhole










    qx = np.arange(Naz,dtype=float)
    qy = np.arange(Nel,dtype=float)
    # shape (Nel,Naz,3)
    vrect_nominal = \
        unproject( np.ascontiguousarray( \
                     nps.mv( nps.cat(*np.meshgrid(qx, qy)),
                             0, -1)),
                   fxycxy )



    if 1:

        r'''* Adaptive rectification

Normally when we're looking far out

- Our range accuracy is low. We're very sensitive to all errors, including
  disparity search errors
- Point cloud coverage is sparse

In this case we want to keep the full resolution of the input images since all
data is precious in this case.

Conversely, when looking close in

- The range accuracy is very high. We're robust to all sorts of stuff, including
  disparity search errors
- Point cloud coverage is dense

Here, we're often doing far better than we need. In fact, we can throw out lots
of data, and still have decent-enough point cloud coverage and range errors.

This is interesting, but it's only actionable if we know where stuff is close
and where it is far. Usually we find that out AFTER we already finished all the
stereo processing, so the usual stereo rectification routines use a constant
resolution everywhere (mrcal.rectified_system(pixels_per_deg_az,
pixels_per_deg_el).

If we DID have a range estimate a-priori, however, we could vary the resolution
of the rectified image. In the closer-in areas we could reduce pixels_per_deg to
generate fewer pixels, which would then greatly speed up the dense stereo
correlation. That step is potentially very slow, so this is desirable.

If we have a ground vehicle with cameras looking out along the ground, then we
DO have an a-priori range estimate: at the bottom of the image we're looking at
the ground next to us, and as we move up in the image, we're looking further and
further out.

Here I'm ingesting an arbitrary representation of a plane. The adaptive
rectification routine then computes the predicted range sensitivity to
correlation errors, and picks a resolution that doesn't give us more range
robustness than necessary. The current method is general: a different resolution
is computed for each pixel, not even for each row.
'''




        # initially azel is nominal

        # shape (Nel,Naz,2)
        azel = np.zeros((Nel,Naz,2), dtype=float)

        # az is nominal
        azel[..., 0] = (np.arange(Naz, dtype=float) - cx) / fx

        # el is nominal
        azel[..., 1] = nps.dummy( (np.arange(Nel, dtype=float) - cy) / fy,
                                  axis = -1)


        n0 = np.array((0.02652536, 0.84598057, 0.53255355))
        d  = 0.1565642688765455
        Rt_rect0_cam0 = \
            mrcal.compose_Rt( models_rectified[0].extrinsics_Rt_fromref(),
                              models          [0].extrinsics_Rt_toref() )
        nrect0 = mrcal.rotate_point_R(Rt_rect0_cam0[:3,:], n0)

        # shape (Nel,Naz,3)
        vrect0 = vrect_nominal
        # shape (Nel,Naz)
        k = d / nps.inner(vrect0, nrect0)
        # shape (Nel,Naz,3)
        prect0 = nps.dummy(k,-1) * vrect0

        # This should be just p1 = p0 - b*xhat
        # shape (Nel,Naz,3)
        prect1 = \
            mrcal.transform_point_Rt( mrcal.compose_Rt( models_rectified[1].extrinsics_Rt_fromref(),
                                                        models_rectified[0].extrinsics_Rt_toref()),
                                      prect0 )

        # shape (Nel,Naz)
        az0 = mrcal.project_latlon(prect0)[...,0]
        az1 = mrcal.project_latlon(prect1)[...,0]

        # points at infinity have nominal az
        mask_infinity = ~np.isfinite(k) + (k<=0)
        az0[mask_infinity] = azel[mask_infinity, 0]
        az1[mask_infinity] = azel[mask_infinity, 0]


        baseline = \
            nps.mag(mrcal.compose_Rt( models_rectified[1].extrinsics_Rt_fromref(),
                                      models_rectified[0].extrinsics_Rt_toref())[3,:])


        # Alright. I'm assuming LENSMODEL_LATLON, so I'm inside the epipolar plane (this
        # is different for pinhole rectification). Law of sines for the figure in
        # http://mrcal.secretsauce.net/stereo.html
        #
        #   b/sin(180-(90-az0)-(90+az1)) = r/sin(90+az1)
        #   b/sin(az0-az1) = r/cos(az1)
        #
        #   -> r =  b * cos(az1)/sin(az0-az1)
        #   ->   = -b * cos(az1)/sin(az1-az0)
        #   ->   = -b * cos(az1-az0 + az0)/sin(az1-az0)
        #   ->   = -b * (cos(az1-az0)cos(az0) - sin(az1-az0)sin(az0))/sin(az1-az0)
        #        = -b * (cos(az0)/tan(az1-az0) - sin(az0))
        #
        # drange/daz1 = d( -b * cos(az0)/tan(az1-az0) )/daz1 =
        #             = -b cos(az0) d( 1/tan(az1-az0) )/daz1 =
        #             = -b cos(az0) d( cos()/sin() )/daz1 =
        #             = -b cos(az0) (-sin()^2 - cos()^2)/sin()^2 =
        #             = b cos(az0)/sin(az1-az0)^2

        # shape (Nel,Naz)
        ## _range            = -baseline * (np.cos(az0)/np.tan(az1-az0) - np.sin(az0))
        s                          = np.sin(az1-az0)
        # take care to not /0. An ugly warning on the console results
        mask_infinity              = (s==0)
        s[mask_infinity]           = 1.
        drange_daz1                = baseline * np.cos(az0)/(s*s)
        drange_daz1[mask_infinity] = 1e6
        drange_ddisparity          = drange_daz1 / fx

        # I now have the nominal drange/daz1. For simplicity let's use the same
        # function for the two cameras. So I have the nominal drange_daz. I have
        # drange = drange_daz daz/dqx dqx_expected
        dqx_expected = 0.3
        rangeerr_min = 0.002

        # shape (Nel,Naz)
        daz_dqx = rangeerr_min / drange_daz1 / dqx_expected

        # I want to work with dqx/daz since that's the fx in the nominal model
        dqx_daz = 1. / daz_dqx

        # I now have the daz/dqx that give me the desired accuracy. I never want
        # to exceed the nominal pixel resolution. And I want to do the nominal
        # thing above the horizon
        mask_nominal = (dqx_daz > fx) + (k <= 0)
        dqx_daz[mask_nominal] = fx


        if 0:
            # Let's ask for those resolutions at az0. This isn't "right", but
            # probably it's close-enough: the errors should lie within our final
            # margin. This is nice because it makes all rows of the az array
            # identical, which makes interpolation easy
            az = az0

        else:
            # Let's ask for those resolutions at a mean az. az0 is our linear
            # sample. az1 is the corresponding az on the other camera when observing
            # the plane. az1 is NOT linear. Maybe in a pinhole projection
            az = (az0 + az1)/2

        # I now have dqx_daz(az) that don't give me too much accuracy. I capped
        # it at the nominal fx, so this can only squeeze the rectified image,
        # not expand it
        #
        # I want to resample the qx-vs-az curve, so I integrate
        daz          = np.diff(az, axis=-1)
        az_midpoint  = (az[...,:-1] + az[...,1:])/2
        qx           = np.cumsum( (dqx_daz[..., :-1] + dqx_daz[..., 1:])/2 * daz, axis=-1)

        # I now have some irregular qx(az). I can add an arbitrary constant term
        # to each integration. Let me line up the center of view. This is the
        # usual case where "disparity=0" means "range=infinity". I can also set
        # it up such that "disparity=0" means "we're looking at the plane"
        #
        # Usual case has qx = fx az + cx. So at the center of view I have
        #   (Naz-1)/2 = fx * az_center + cx ->
        #   az_center = ((Naz-1)/2 - cx)/fx
        #
        # I set the center of each row to hit az_center
        az_center = ((Naz-1)/2 - cx)/fx

        import scipy.interpolate

        # Python loop. Yuck!
        for i in range(qx.shape[0]):

            qx_az_interpolator = \
                scipy.interpolate.interp1d( az_midpoint[i],
                                            qx[i],
                                            bounds_error  = False,
                                            fill_value    = 0.,
                                            assume_sorted = True,
                                            copy          = False)

            qx[i] += (qx.shape[-1] - 1)/2 - qx_az_interpolator(az_center)


        for i in range(qx.shape[0]):

            az_qx_interpolator = \
                scipy.interpolate.interp1d( qx[i],
                                            az_midpoint[i],
                                            bounds_error  = False,
                                            fill_value    = 'extrapolate',
                                            assume_sorted = True,
                                            copy          = False)

            azel[i,:,0] = az_qx_interpolator(np.arange(Naz))

        # shape (Nel,Naz,3)
        v = unproject(azel)

        if np.min(np.diff(azel[...,0], axis=-1)) <= 0:
            raise Exception("az-vs-qx MUST be monotonically increasing. This is important for finding the edges")

        # Fit stuff. Not yet
        if 0:
            x = np.arange(Naz)[~mask_nominal[i]]
            y = dqx_daz[i][~mask_nominal[i]]

            xmean = np.mean(x)
            xscale = (np.max(x) - np.min(x))/2
            x = (x-xmean)/xscale


        v0 = mrcal.rotate_point_R(R_cam_rect[0], v)
        v1 = mrcal.rotate_point_R(R_cam_rect[1], v)

        mapxy0 = mrcal.project( v0, *models[0].intrinsics()).astype(np.float32)
        mapxy1 = mrcal.project( v1, *models[1].intrinsics()).astype(np.float32)

        # In the code above I set the center of the rectified image to sit at the az
        # center. This is in-bounds. I search in both directions from this center to
        # find the edge of the image. This is needed because the projection
        # functions are not well-defined outside of the nominal FOV, and can wrap
        # around
        def valid_projection_boundary(mapxy, model):
            W,H = model.imagersize()
            jmid = mapxy.shape[1] // 2

            # first in-bounds pixel on the left
            icol0 = \
                jmid - \
                np.argmax( (mapxy[:,:jmid,0][..., ::-1] < 0)  +
                           (mapxy[:,:jmid,1][..., ::-1] < 0)  +
                           (mapxy[:,:jmid,0][..., ::-1] > W-1)+
                           (mapxy[:,:jmid,1][..., ::-1] > H-1),
                           axis=-1 )
            # first out-of-bounds pixel on the right
            icol1 = \
                jmid + \
                np.argmax( (mapxy[:,jmid:,0] < 0)  +
                           (mapxy[:,jmid:,1] < 0)  +
                           (mapxy[:,jmid:,0] > W-1)+
                           (mapxy[:,jmid:,1] > H-1),
                           axis=-1 )

            # In each row (icol0,icol1) now form a python-style range describing the
            # in-bounds pixels
            return nps.transpose(nps.cat(icol0, icol1))


        # shape (Nel,2)
        qx01_0 = valid_projection_boundary(mapxy0, models[0])
        qx01_1 = valid_projection_boundary(mapxy1, models[1])

        # I have the bounds for the two images. I intersect them
        qx01 = np.zeros(qx01_0.shape, qx01_0.dtype)
        qx01[:,0] = np.max(nps.cat(qx01_0[:,0], qx01_1[:,0]), axis=0)
        qx01[:,1] = np.min(nps.cat(qx01_0[:,1], qx01_1[:,1]), axis=0)

        # import IPython
        # IPython.embed()
        # sys.exit()

        return mapxy0, mapxy1




    v = vrect_nominal

    v0 = mrcal.rotate_point_R(R_cam_rect[0], v)
    v1 = mrcal.rotate_point_R(R_cam_rect[1], v)

    return                                                                \
        (mrcal.project( v0, *models[0].intrinsics()).astype(np.float32),  \
         mrcal.project( v1, *models[1].intrinsics()).astype(np.float32))


def stereo_range(disparity,
                 models_rectified,
                 *,
                 disparity_scale = 1,
                 qrect0          = None):

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

    Rt_cam0_rect0 = mrcal.compose_Rt( models          [0].extrinsics_Rt_fromref(),
                                      models_rectified[0].extrinsics_Rt_toref() )

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

    Rt01 = mrcal.compose_Rt( models_rectified[0].extrinsics_Rt_fromref(),
                             models_rectified[1].extrinsics_Rt_toref())
    baseline = nps.mag(Rt01[3,:])

    if intrinsics[0] == 'LENSMODEL_LATLON':
        if qrect0 is None:
            az0 = (np.arange(W, dtype=float) - cx)/fx
        else:
            az0 = (qrect0[...,0] - cx)/fx

        disparity_rad = disparity.astype(np.float32) / (fx * disparity_scale)

        mask_invalid = (disparity <= 0)

        s = np.sin(disparity_rad)
        s[mask_invalid] = 1 # to prevent division by 0

        r = baseline * np.cos(az0 - disparity_rad) / s

    else:
        # pinhole

        fy = intrinsics[1][1]
        cy = intrinsics[1][3]

        if qrect0 is None:
            tanaz0 = (np.arange(W, dtype=float) - cx)/fx
            tanel  = (np.arange(H, dtype=float) - cy)/fy
            tanel  = nps.dummy(tanel, -1)
        else:
            tanaz0 = (qrect0[...,0] - cx) / fx
            tanel  = (qrect0[...,1] - cy) / fy
        s_sq_recip = tanel*tanel + 1.


        tanaz0_tanaz1 = disparity.astype(np.float32) / (fx * disparity_scale)

        mask_invalid  = (disparity <= 0)
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

    Rt_cam0_rect0 = mrcal.compose_Rt( models          [0].extrinsics_Rt_fromref(),
                                      models_rectified[0].extrinsics_Rt_toref() )

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
        t_ref_cam = model.extrinsics_Rt_toref()[ 3,:]
        R_ref_cam = model.extrinsics_Rt_toref()[:3,:]
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

    v1 = mrcal.transform_point_Rt(model1.extrinsics_Rt_fromref(),
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

4. cv2.matchTemplate() to search for the template in this region of image1

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
    try:
        lsqsq_result = np.linalg.lstsq( M, z, rcond = None)
    except:
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
