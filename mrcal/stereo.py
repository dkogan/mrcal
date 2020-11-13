#!/usr/bin/python3

import sys
import numpy as np
import numpysane as nps
import mrcal

def stereo_rectify_prepare(models,
                           az_fov_deg,
                           el_fov_deg,
                           az0_deg           = None,
                           el0_deg           = 0,
                           pixels_per_deg_az = None,
                           pixels_per_deg_el = None):

    r'''Precompute everything needed for stereo rectification and matching

SYNOPSIS

    import sys
    import mrcal
    import cv2
    import numpy as np

    # Read commandline arguments: model0 model1 image0 image1
    models = [ mrcal.cameramodel(sys.argv[1]),
               mrcal.cameramodel(sys.argv[2]), ]

    images = [ cv2.imread(sys.argv[i]) \
               for i in (3,4) ]

    # Prepare the stereo system
    rectification_maps,cookie = \
        mrcal.stereo_rectify_prepare(models,
                                     az_fov_deg = 120,
                                     el_fov_deg = 100)

    # Visualize the geometry of the two cameras and of the rotated stereo
    # coordinate system
    Rt_cam0_ref    = models[0].extrinsics_Rt_fromref()
    Rt_cam0_stereo = cookie['Rt_cam0_stereo']
    Rt_stereo_ref  = mrcal.compose_Rt( mrcal.invert_Rt(Rt_cam0_stereo),
                                       Rt_cam0_ref )
    rt_stereo_ref  = mrcal.rt_from_Rt(Rt_stereo_ref)

    mrcal.show_geometry( models + [ rt_stereo_ref ],
                         ( "camera0", "camera1", "stereo" ),
                         show_calobjects = False,
                         wait            = True )

    # Rectify the images
    images_rectified = \
      [ mrcal.transform_image(images[i], rectification_maps[i]) \
        for i in range(2) ]

    cv2.imwrite('/tmp/rectified0.jpg', images_rectified[0])
    cv2.imwrite('/tmp/rectified1.jpg', images_rectified[1])

    # Find stereo correspondences using OpenCV
    block_size = 3
    max_disp   = 160 # in pixels
    stereo = \
        cv2.StereoSGBM_create(minDisparity      = 0,
                              numDisparities    = max_disp,
                              blockSize         = block_size,
                              P1                = 8 *3*block_size*block_size,
                              P2                = 32*3*block_size*block_size,
                              uniquenessRatio   = 5,

                              disp12MaxDiff     = 1,
                              speckleWindowSize = 50,
                              speckleRange      = 1)
    disparity16 = stereo.compute(*images_rectified) # in pixels*16

    cv2.imwrite('/tmp/disparity.png',
                mrcal.apply_color_map(disparity16,
                                      0, max_disp*16.))

    # Convert the disparities to range to camera0
    r = mrcal.stereo_range( disparity16.astype(np.float32) / 16.,
                            **cookie )

    cv2.imwrite('/tmp/range.png', mrcal.apply_color_map(r, 5, 1000))

This function does the initial computation required to perform stereo matching,
and to get ranges from a stereo pair. It computes

- the pose of the rectified stereo coordinate system

- the azimuth/elevation grid used in the rectified images

- the rectification maps used to transform images into the rectified space

Using the results of one call to this function we can compute the stereo
disparities of many pairs of synchronized images.

This function is generic: the two cameras may have any lens models, any
resolution and any geometry. They don't even have to match. As long as there's
some non-zero baseline and some overlapping views, we can set up stereo matching
using this function. The input images are tranformed into a "rectified" space.
Geometrically, the rectified coordinate system sits at the origin of camera0,
with a rotation. The axes of the rectified coordinate system:

- x: from the origin of camera0 to the origin of camera1 (the baseline direction)

- y: completes the system from x,z

- z: the "forward" direction of the two cameras, with the component parallel to
     the baseline subtracted off

In a nominal geometry (the two cameras are square with each other, camera1
strictly to the right of camera0), the rectified coordinate system exactly
matches the coordinate system of camera0. The above formulation supports any
geometry, however, including vertical and/or forward/backward shifts. Vertical
stereo is supported.

Rectified images represent 3D planes intersecting the origins of the two
cameras. The tilt of each plane is the "elevation". While the left/right
direction inside each plane is the "azimuth". We generate rectified images where
each pixel coordinate represents (x = azimuth, y = elevation). Thus each row
scans the azimuths in a particular elevation, and thus each row in the two
rectified images represents the same plane in 3D, and matching features in each
row can produce a stereo disparity and a range.

In the rectified system, elevation is a rotation along the x axis, while azimuth
is a rotation normal to the resulting tilted plane.

We produce rectified images whose pixel coordinates are linear with azimuths and
elevations. This means that the azimuth angular resolution is constant
everywhere, even at the edges of a wide-angle image.

We return a set of transformation maps and a cookie. The maps can be used to
generate rectified images. These rectified images can be processed by any
stereo-matching routine to generate a disparity image. To interpret the
disparity image, call stereo_unproject() or stereo_range() with the cookie
returned here.

The cookie is a Python dict that describes the rectified space. It is guaranteed
to have the following keys:

- Rt_cam0_stereo: an Rt transformation to map a representation of points in the
  rectified coordinate system to a representation in the camera0 coordinate system

- baseline: the distance between the two cameras

- az_row: an array of shape (Naz,) describing the azimuths in each row of the
  disparity image

- el_col: an array of shape (Nel,1) describing the elevations in each column of
  the disparity image

ARGUMENTS

- models: an iterable of two mrcal.cameramodel objects representing the cameras
  in the stereo system. Any sane combination of lens models and resolutions and
  geometries is valid

- az_fov_deg: required value for the azimuth (along-the-baseline) field-of-view
  of the desired rectified view, in pixels

- el_fov_deg: required value for the elevation (across-the-baseline)
  field-of-view of the desired rectified view, in pixels

- az0_deg: optional value for the azimuth center of the rectified images. This
  is especially significant in a camera system with some forward/backward shift.
  That causes the baseline to no longer be perpendicular with the view axis of
  the cameras, and thus azimuth = 0 is no longer at the center of the input
  images. If omitted, we compute the center azimuth that aligns with the center
  of the cameras' view

- el0_deg: optional value for the elevation center of the rectified system.
  Defaults to 0.

- pixels_per_deg_az: optional value for the azimuth resolution of the rectified
  image. If omitted (or None), we use the resolution of the input image at
  (azimuth, elevation) = 0. If a resolution of <0 is requested, we use this as a
  scale factor on the resolution of the input image. For instance, to downsample
  by a factor of 2, pass pixels_per_deg_az = -0.5

- pixels_per_deg_el: same as pixels_per_deg_az but in the elevation direction

RETURNED VALUES

We return a tuple

- transformation maps: a tuple of length-2 containing transformation maps for
  each camera. Each map can be used to mrcal.transform_image() images to the
  rectified space

- cookie: a dict describing the rectified space. Intended as input to
  stereo_unproject() and stereo_range(). See the description above for more
  detail

    '''

    if len(models) != 2:
        raise Exception("I need exactly 2 camera models")

    def normalize(v):
        v /= nps.mag(v)
        return v

    def remove_projection(a, proj_base):
        r'''Returns the normalized component of a orthogonal to proj_base

        proj_base assumed normalized'''
        v = a - nps.inner(a,proj_base)*proj_base
        return normalize(v)

    ######## Compute the geometry of the rectified stereo system. This is a
    ######## rotation, centered at camera0. More or less we have axes:
    ########
    ######## x: from camera0 to camera1
    ######## y: completes the system from x,z
    ######## z: component of the cameras' viewing direction
    ########    normal to the baseline
    Rt_cam0_ref = models[0].extrinsics_Rt_fromref()
    Rt01 = mrcal.compose_Rt( Rt_cam0_ref,
                             models[1].extrinsics_Rt_toref())
    Rt10 = mrcal.invert_Rt(Rt01)

    # Rotation relating camera0 coords to the rectified camera coords. I fill in
    # each row separately
    R_stereo_cam0 = np.zeros((3,3), dtype=float)
    right         = R_stereo_cam0[0,:]
    down          = R_stereo_cam0[1,:]
    forward       = R_stereo_cam0[2,:]

    # "right" of the rectified coord system: towards the origin of camera1 from
    # camera0, in camera0 coords
    right[:] = Rt01[3,:]
    baseline = nps.mag(right)
    right   /= baseline

    # "forward" for each of the two cameras, in the cam0 coord system
    forward0 = np.array((0,0,1.))
    forward1 = Rt01[:3,2]

    # "forward" of the rectified coord system, in camera0 coords. The mean of
    # the two non-right "forward" directions
    forward[:] = normalize( ( remove_projection(forward0,right) +
                              remove_projection(forward1,right) ) / 2. )

    # "down" of the rectified coord system, in camera0 coords. Completes the
    # right,down,forward coordinate system
    down[:] = np.cross(forward,right)

    R_cam0_stereo = nps.transpose(R_stereo_cam0)



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
        v0 = nps.matmult( forward0, R_cam0_stereo ).ravel()
        v1 = nps.matmult( forward1, R_cam0_stereo ).ravel()
        v0[1] = 0.0
        v1[1] = 0.0
        normalize(v0)
        normalize(v1)
        v = v0 + v1
        az0 = np.arctan2(v[0],v[2])


    el0 = el0_deg * np.pi/180.

    ####### Rectified image resolution
    if pixels_per_deg_az is None or pixels_per_deg_az < 0 or \
       pixels_per_deg_el is None or pixels_per_deg_el < 0:
        # I need to compute the resolution of the rectified images. I try to
        # match the resolution of the cameras. I just look at camera0. If you
        # have different cameras, pass in pixels_per_deg yourself :)
        #
        # I look at the center of the stereo field of view. There I have q =
        # project(v) where v is a unit projection vector. I compute dq/dth where
        # th is an angular perturbation applied to v.
        def rotation_any_v_to_z(v):
            r'''Return any rotation matrix that maps the given unit vector v to [0,0,1]'''
            z = v
            if np.abs(v[0]) < .9:
                x = np.array((1,0,0))
            else:
                x = np.array((0,1,0))
            x -= nps.inner(x,v)*v
            x /= nps.mag(x)
            y = np.cross(z,x)

            return nps.cat(x,y,z)


        v, dv_dazel = stereo_unproject(az0, el0, get_gradients = True)
        v0          = mrcal.rotate_point_R(R_cam0_stereo, v)
        dv0_dazel   = nps.matmult(R_cam0_stereo, dv_dazel)

        _,dq_dv0,_ = mrcal.project(v0, *models[0].intrinsics(), get_gradients = True)

        # I rotate my v to a coordinate system where u = rotate(v) is [0,0,1].
        # Then u = [a,b,0] are all orthogonal to v. So du/dth = [cos, sin, 0].
        # I then have dq/dth = dq/dv dv/du [cos, sin, 0]t
        # ---> dq/dth = dq/dv dv/du[:,:2] [cos, sin]t = M [cos,sin]t
        #
        # norm2(dq/dth) = [cos,sin] MtM [cos,sin]t is then an ellipse with the
        # eigenvalues of MtM giving me the best and worst sensitivities. I can
        # use mrcal.worst_direction_stdev() to find the densest direction. But I
        # actually know the directions I care about, so I evaluate them
        # independently for the az and el directions

        # Ruv = rotation_any_v_to_z(v0)
        # M = nps.matmult(dq_dv0, nps.transpose(Ruv[:2,:]))
        # # I pick the densest direction: highest |dq/dth|
        # pixels_per_rad = mrcal.worst_direction_stdev( nps.matmult( nps.transpose(M),M) )

        if pixels_per_deg_az is None or pixels_per_deg_az < 0:
            dq_daz = nps.inner( dq_dv0, dv0_dazel[:,0] )
            pixels_per_rad_az_have = nps.mag(dq_daz)

            if pixels_per_deg_az is not None:
                # negative pixels_per_deg_az requested means I use the requested
                # value as a scaling
                pixels_per_deg_az = -pixels_per_deg_az * pixels_per_rad_az_have*np.pi/180.
            else:
                pixels_per_deg_az = pixels_per_rad_az_have*np.pi/180.

        if pixels_per_deg_el is None or pixels_per_deg_el < 0:
            dq_del = nps.inner( dq_dv0, dv0_dazel[:,1] )
            pixels_per_rad_el_have = nps.mag(dq_del)

            if pixels_per_deg_el is not None:
                # negative pixels_per_deg_el requested means I use the requested
                # value as a scaling
                pixels_per_deg_el = -pixels_per_deg_el * pixels_per_rad_el_have*np.pi/180.
            else:
                pixels_per_deg_el = pixels_per_rad_el_have*np.pi/180.



    Naz = round(az_fov_deg*pixels_per_deg_az)
    Nel = round(el_fov_deg*pixels_per_deg_el)

    # Adjust the fov to keep the requested resolution and pixel counts
    az_fov_radius_deg = Naz / (2.*pixels_per_deg_az)
    el_fov_radius_deg = Nel / (2.*pixels_per_deg_el)

    # shape (Naz,)
    az = np.linspace(az0 - az_fov_radius_deg*np.pi/180.,
                     az0 + az_fov_radius_deg*np.pi/180.,
                     Naz)
    # shape (Nel,1)
    el = nps.dummy( np.linspace(el0 - el_fov_radius_deg*np.pi/180.,
                                el0 + el_fov_radius_deg*np.pi/180.,
                                Nel),
                    -1 )

    # v has shape (Nel,Naz,3)
    v = stereo_unproject(az, el)

    v0 = nps.matmult( nps.dummy(v,  -2), R_stereo_cam0 )[...,0,:]
    v1 = nps.matmult( nps.dummy(v0, -2), Rt01[:3,:]    )[...,0,:]

    cookie = \
        dict( Rt_cam0_stereo    = nps.glue(R_cam0_stereo, np.zeros((3,)), axis=-2),
              baseline          = baseline,
              az_row            = az,
              el_col            = el,

              # The caller should NOT assume these are available in the cookie:
              # some other rectification scheme may not produce linear az/el
              # maps
              pixels_per_deg_az = pixels_per_deg_az,
              pixels_per_deg_el = pixels_per_deg_el,
            )

    return                                                                \
        (mrcal.project( v0, *models[0].intrinsics()).astype(np.float32),  \
         mrcal.project( v1, *models[1].intrinsics()).astype(np.float32)), \
        cookie


def stereo_unproject(az                = None,
                     el                = None,
                     disparity_pixels  = None,
                     baseline          = None,
                     pixels_per_deg_az = None,
                     get_gradients     = False,
                     az_row            = None,
                     el_col            = None,

                     # to capture the remainder I don't need
                     **kwargs):
    r'''Unprojection in the rectified stereo system

SYNOPSIS

    ...

    rectification_maps,cookie = \
        mrcal.stereo_rectify_prepare(models, ...)

    ...

    disparity16 = stereo.compute(*images_rectified) # in pixels*16

    disparity_pixels = disparity16.astype(np.float32) / 16.

    p_stereo = \
        mrcal.stereo_unproject( disparity_pixels = disparity_pixels,
                                **cookie )

    p_cam0 = mrcal.transform_point_Rt( cookie['Rt_cam0_stereo'],
                                       p_stereo )

See the docstring stereo_rectify_prepare() for a complete example, and for a
description of the rectified space and the meaning of "azimuth" and "elevations"
in this context.

This function supports several kinds of rectified-space unprojections:

- normalized (range-less) unprojections

This is the simplest mode. Two arguments are accepted: "az" and "el", and an
array of unit vectors is returned. The default value of disparity_pixels = None
selects this mode. "az" and "el" may have identical dimensions, or may be
broadcasting compatible. A very common case is evaluating a rectified image
where each row represents an identical sequence azimuths and every column
represents an identical sequence of elevations. We can use broadcasting to pass
in "az" of shape (Naz,) and "el" of shape (Nel,1) to get an output array of
vectors of shape (Nel,Naz,3)


- normalized (range-less) unprojections with gradients

This is identical as the previous mode, but we return the vectors AND their
gradients in respect to az,el. Pass in az, el as before, but ALSO pass in
get_gradients = True. Same dimensionality logic as before applies. We return (v,
dv_dazel) where v.shape is (..., 3) and dv_dazel.shape is (..., 3,2)


- ranged unprojections without gradients

Here we use a disparity image to compute ranges, and use them to scale the
unprojected direction vectors. This is very similar to stereo_range(), but we
report a 3D point, not just ranges.

The rectification maps from stereo_rectify_prepare() can be used to produce
rectified images, which can then be processed by a stereo-matching routine to
get a disparity image. The next step is converting the disparities to 3D points,
and is implemented by this function. If instead of 3D points we just want ranges,
call stereo_range() instead.

A common usage is to produce a disparity IMAGE, and then convert it to a point
cloud. In this common case we can call

    r = mrcal.stereo_unproject(disparity_pixels = disparity_pixels,
                               **cookie)

Where the cookie was returned to us by mrcal.stereo_rectify_prepare(). It is a
dict that contains all the information needed to interpret the disparities.

If we aren't processing the full disparity IMAGE, we can still use this
function. Pass in the disparities of any shape. And the corresponding azimuths
in the "az" argument and elevations in "el". These 3 arrays describe
corresponding points. They may have identical dimensions. If there's any
redundancy (if every row of the disparity array has identical azimuths for
instance), then being broadcasting-compatible is sufficient. A common case of
this is disparity images. In this case the disparity array has shape (Nel,Naz),
with each row containing an identical sequences of azimuths and each column
containing identical sequences of elevations. We can then pass an azimuth array
of shape (Naz,) and en elevation array of shape (Nel,1)

The azimuths, elevations may also be passed in the "az_row", "el_col" arguments.
This is usually not done by the user directly, but via the **cookie from
stereo_rectify_prepare(). At least one of "az","az_row" and "el","el_col" must
be given. If both are given, we use "az" and "el".

ARGUMENTS

- az: optional array of azimuths corresponding to the given disparities. If
  processing normal disparity images, this can be omitted, and we'll use az_row
  from **cookie, with the cookie returned by stereo_rectify_prepare(). If
  processing anything other than a full disparity image, pass the azimuths here.
  The az array must be broadcasting-compatible with disparity_pixels. The
  dimensions of the two arrays may be identical, but brodcasting can be used to
  exploit any redundancy. For instance, disparity images have shape (Nel, Naz)
  with each row containing identical azimuths. We can thus work with az of shape
  (Naz,). At least one of az and az_row must be non-None. If both are given, we
  use az.

- el: optional array of elevations corresponding to the given disparities.
  Works identically to the "az" argument, but represents elevations.

- disparity_pixels: optional array of disparities, defaulting to None. If None,
  we report unit vectors. Otherwise we use the disparities to compute ranges to
  scale the vectors. This array contains disparity in PIXELS. If the
  stereo-matching algorithm you're using reports a scaled value, you need to
  convert it to pixels, possibly converting to floating-point in the process.
  Any array shape is supported. In the common case of a disparity IMAGE, this is
  an array of shape (Nel, Naz)

- baseline: optional value, representing the distance between the two stereo
  cameras. This is required only if disparity_pixels is not None. Usually it
  comes from **cookie, with the cookie returned by stereo_rectify_prepare()

- pixels_per_deg_az: optional value, representing the angular resolution of the
  disparity image. This is required only if disparity_pixels is not None.
  Usually it comes from **cookie, with the cookie returned by
  stereo_rectify_prepare(). Don't pass this yourself. This argument may go away
  in the future

- get_gradients: optional boolean, defaulting to False. If True, we return the
  unprojected vectors AND their gradients in respect to (az,el). Gradients are
  only available when performing rangeless unprojections (disparity_pixels is
  None).

- az_row: optional array of azimuths in the disparity_pixels array. Works
  exactly like the "az" array. At least one MUST be given. If both are given, we
  use "az". Usually az_row is not given by the user directly, but via a **cookie
  from stereo_rectify_prepare()

- el_col: optional array of elevations in the disparity_pixels array. Works just
  like az_row, but for elevations

RETURNED VALUES

if get_gradients is False:

- An array of unprojected points of shape (....,3). Invalid or missing ranges
  are represented as vectors of length 0.

else:

A tuple:

- The array of unprojected points as before.
- The array of gradients dv_dazel of shape (...,3,2)

    '''

    if get_gradients and disparity_pixels is not None:
        raise Exception("stereo_unproject(get_gradients = True) only supported with disparity is None (not reporting the ranges)")

    if disparity_pixels is not None and baseline is None:
        raise Exception("stereo_unproject(get_gradients = True) requires a given baseline")

    if az is None:
        if az_row is None:
            raise Exception("At least one of (az, az_row) must be non-None.")
        az = az_row
    if el is None:
        if el_col is None:
            raise Exception("At least one of (el, el_col) must be non-None.")
        el = el_row


    caz = np.cos(az)
    saz = np.sin(az)
    cel = np.cos(el)
    sel = np.sin(el)

    try:    az1 = np.ones(az.shape, dtype=az.dtype)
    except: az1 = np.array(1.)
    try:    el1 = np.ones(el.shape, dtype=el.dtype)
    except: el1 = np.array(1.)

    # v(az=0, el=0) = (0,0,1)
    # v(az=0): el sweeps (forward,down)  in a circle
    # v(el=0): az sweeps (forward,right) in a circle
    # v(az=90) = (1,0,0)

    # shape (...,3)
    v = \
        nps.glue( nps.dummy(saz*el1, -1), # right
                  nps.dummy(caz*sel, -1), # down
                  nps.dummy(caz*cel, -1), # forward
                  axis = -1 )

    if disparity_pixels is not None:
        # shape (...)
        r = stereo_range(disparity_pixels, baseline, pixels_per_deg_az,
                         az = az)

        # shape (...,3)
        return v * nps.dummy(r,-1)


    if not get_gradients:
        return v

    # shape (...,3,2)
    dv_dazel = \
        nps.glue( nps.glue(nps.dummy( caz*el1, -1,-1), np.zeros((az1*el1).shape + (1,1,)), axis=-1),
                  nps.glue(nps.dummy(-saz*sel, -1,-1), nps.dummy( caz*cel, -1,-1),         axis=-1),
                  nps.glue(nps.dummy(-saz*cel, -1,-1), nps.dummy(-caz*sel, -1,-1),         axis=-1),
                  axis = -2 )
    return v, dv_dazel


def stereo_range(disparity_pixels,

                 baseline,
                 pixels_per_deg_az,

                 az     = None,
                 az_row = None,


                 # to capture the remainder I don't need
                 **kwargs):

    r'''Compute ranges from observed disparities

SYNOPSIS

    ...

    rectification_maps,cookie = \
        mrcal.stereo_rectify_prepare(models, ...)

    ...

    disparity16 = stereo.compute(*images_rectified) # in pixels*16

    r = mrcal.stereo_range( disparity16.astype(np.float32) / 16.,
                            **cookie )


See the docstring stereo_rectify_prepare() for a complete example, and for a
description of the rectified space and the meaning of "azimuth" and "elevations"
in this context.

The rectification maps from stereo_rectify_prepare() can be used to produce
rectified images, which can then be processed by a stereo-matching routine to
get a disparity image. The next step is converting the disparities to ranges,
and is implemented by this function. If instead of ranges we want 3D points,
call stereo_unproject() instead.

In the most common usage we produce a disparity IMAGE, and then convert it to a
range IMAGE. In this common case we can simply call

    r = mrcal.stereo_range(disparity, **cookie)

Where the cookie was returned to us by mrcal.stereo_rectify_prepare(). It is a
dict that contains all the information needed to interpret the disparities.

If we aren't processing the full disparity IMAGE, we can still use this
function. Pass in the disparities of any shape. And the corresponding azimuths
in the "az" argument. The "az" and "disparity_pixels" arrays describe
corresponding points. They may have identical dimensions. If there's any
redundancy (if every row of the disparity array has identical azimuths for
instance), then being broadcasting-compatible is sufficient. A very common case
of this is disparity images. In this case the disparity array has shape
(Nel,Naz), with each row containing an identical sequences of azimuths. We can
then pass an azimuth array of shape (Naz,).

The azimuths may also be passed in the "az_row" argument. This is usually not
done by the user directly, but via the **cookie from stereo_rectify_prepare().
At least one of "az", "az_row" must be given. If both are given, we use "az".

ARGUMENTS

- disparity_pixels: the numpy array of disparities being processed. This
  array contains disparity in PIXELS. If the stereo-matching algorithm you're
  using reports a scaled value, you need to convert it to pixels, possibly
  converting to floating-point in the process. Any array shape is supported. In
  the common case of a disparity IMAGE, this is an array of shape (Nel, Naz)

- baseline: the distance between the two stereo cameras. This is required, and
  usually it comes from **cookie, with the cookie returned by
  stereo_rectify_prepare()

- pixels_per_deg_az: the angular resolution of the disparity image. This is
  required, and usually it comes from **cookie, with the cookie returned by
  stereo_rectify_prepare(). Don't pass this yourself. This argument may go away
  in the future

- az: optional array of azimuths corresponding to the given disparities. If
  processing normal disparity images, this can be omitted, and we'll use az_row
  from **cookie, with the cookie returned by stereo_rectify_prepare(). If
  processing anything other than a full disparity image, pass the azimuths here.
  The az array must be broadcasting-compatible with disparity_pixels. The
  dimensions of the two arrays may be identical, but brodcasting can be used to
  exploit any redundancy. For instance, disparity images have shape (Nel, Naz)
  with each row containing identical azimuths. We can thus work with az of shape
  (Naz,). At least one of az and az_row must be non-None. If both are given, we
  use az.

- az_row: optional array of azimuths in the disparity_pixels array. Works
  exactly like the "az" array. At least one MUST be given. If both are given, we
  use "az". Usually az_row is not given by the user directly, but via a **cookie
  from stereo_rectify_prepare()

RETURNED VALUES

- An array of ranges of the same dimensionality as the input disparity_pixels
  array. Contains floating-point data. Invalid or missing ranges are represented
  as 0.

    '''

    # +. . . . .
    # |\__az0_________________________
    # |                 range         \___________
    # |                                       a  -
    # |                                      ----/
    # |                                 ----/
    # | baseline                   ----/
    # |                      -----/
    # |                 ----/
    # |            ----/
    # |       ----/
    # |  ----/ az1
    # +-/. . . . . . . . . .

    # A disparity image is indexed by (azimuth,elevation). Each row of a
    # disparity represents a plane tilted by a contant elevation angle. Inside
    # each row, pixels linearly span a space of azimuths. A disparity is a
    # difference of azimuths: disparity ~ az0-az1.
    #
    # I measure asimuth from the forward direction. In the above az0 > 0 and az1
    # < 0

    #     baseline / sin(a) = range / sin(90 + az1)

    # -->

    #     range = baseline cos(az1) / sin(a) =
    #           = baseline cos(az1) / sin( 180 - (90-az0 + 90+az1) ) =
    #           = baseline cos(az1) / sin(az0 - az1) =
    #           = baseline cos(az1) / sin(disparity)
    #           = baseline cos(az0 - disparity) / sin(disparity)

    if az is None:
        if az_row is None:
            raise Exception("At least one of (az, az_row) must be non-None.")
        az = az_row

    disparity_rad = disparity_pixels / pixels_per_deg_az / 180.*np.pi

    mask_invalid = (disparity_pixels <= 0)

    s = np.sin(disparity_rad)
    s[mask_invalid] = 1 # to prevent division by 0

    r = baseline * np.cos(az - disparity_rad) / s
    r[mask_invalid] = 0

    return r

