#!/usr/bin/python3
import sys
import mrcal
import cv2
import numpy as np

# Read the models and images from the commandline arguments
try:
    models = [ mrcal.cameramodel(sys.argv[1]) if sys.argv[1] != '-' else None,
               mrcal.cameramodel(sys.argv[2]) if sys.argv[2] != '-' else None, ]
    images = [ cv2.imread(sys.argv[i]) \
               for i in (3,4) ]
    kind = sys.argv[5]
except:
    print(f"Usage: {sys.argv[0]} model0 model1 image0 image1 kind", file=sys.stderr)
    sys.exit(1)

if models[0] is None or models[1] is None:
    images_rectified = images
else:

    # Annotate the image with its valid-intrinsics region. This will end up in the
    # rectified images, and make it clear where successful matching shouldn't be
    # expected
    for i in range(2):
        try:
            mrcal.annotate_image__valid_intrinsics_region(images[i], models[i])
        except:
            pass

    # Generate the rectification maps
    if kind != "narrow":
        azel_kwargs = dict(az_fov_deg = 145.,
                           el_fov_deg = 135.,
                           el0_deg    = 5 )
    else:
        azel_kwargs = dict(az_fov_deg = 80.,
                           el_fov_deg = 80.,
                           el0_deg    = 0 )
    rectification_maps, cookie = \
        mrcal.stereo_rectify_prepare(models, **azel_kwargs)

    # Display the geometry of the two cameras in the stereo pair, and of the
    # rectified system
    Rt_cam0_stereo = cookie['Rt_cam0_stereo']
    Rt_cam0_ref    = models[0].extrinsics_Rt_fromref()
    Rt_stereo_ref  = mrcal.compose_Rt( mrcal.invert_Rt(Rt_cam0_stereo),
                                      Rt_cam0_ref )
    rt_stereo_ref  = mrcal.rt_from_Rt(Rt_stereo_ref)
    mrcal.show_geometry( models + [ rt_stereo_ref ],
                         ( "camera0", "camera1", "stereo" ),
                         show_calobjects = False,
                         _set            = ('xyplane at -0.5',
                                            'view 60,30,1.7'),
                         hardcopy        = f'/tmp/stereo-geometry-{kind}.svg')
    sys.exit()

    # Generate the rectified images, and write to disk
    images_rectified = [mrcal.transform_image(images[i], rectification_maps[i]) for i in range(2)]
    cv2.imwrite(f'/tmp/rectified0-{kind}.jpg', images_rectified[0])
    cv2.imwrite(f'/tmp/rectified1-{kind}.jpg', images_rectified[1])

# Perform stereo-matching with OpenCV to produce a disparity map, which we write
# to disk
block_size = 5
max_disp   = 400
max_disp = 16*round(max_disp/16) # round to nearest multiple of 16
stereo = \
    cv2.StereoSGBM_create(
                          minDisparity      = 0,
                          numDisparities    = max_disp,
                          blockSize         = block_size,
                          P1                = 8 *3*block_size*block_size,
                          P2                = 32*3*block_size*block_size,
                          uniquenessRatio   = 5,
                          disp12MaxDiff     = 1,
                          speckleWindowSize = 200,
                          speckleRange      = 2 )
disparity = stereo.compute(*images_rectified)
cv2.imwrite(f'/tmp/disparity-{kind}.png',
            mrcal.apply_color_map(disparity,
                                  0, max_disp*16.))

if models[0] is not None and models[1] is not None:
    # Convert the disparity image to ranges, and write to disk
    r = mrcal.stereo_range( disparity_pixels = disparity.astype(np.float32) / 16.,
                            **cookie )
    cv2.imwrite(f'/tmp/range-{kind}.png', mrcal.apply_color_map(r, 5, 1000))
