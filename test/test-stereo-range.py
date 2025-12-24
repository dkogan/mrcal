#!/usr/bin/env python3

r'''Checks the C and Python implementations of stereo_range()

This test needs external data, so it isn't included in the enabled-by-default set
'''

import sys
import numpy as np
import numpysane as nps
import os

testdir = os.path.dirname(os.path.realpath(__file__))

# I import the LOCAL mrcal since that's what I'm testing
sys.path[:0] = f"{testdir}/..",
import mrcal
import testutils
import cv2



# This test needs data external to this repository. Can be downloaded like this:
if False:
    import subprocess
    subprocess.check_output("wget -O /tmp/0.cameramodel https://mrcal.secretsauce.net/external/2022-11-05--dtla-overpass--samyang--alpha7/stereo/0.cameramodel && " +
                            "wget -O /tmp/1.cameramodel https://mrcal.secretsauce.net/external/2022-11-05--dtla-overpass--samyang--alpha7/stereo/1.cameramodel && "
                            "wget -O /tmp/0.jpg         https://mrcal.secretsauce.net/external/2022-11-05--dtla-overpass--samyang--alpha7/stereo/0.jpg && "
                            "wget -O /tmp/1.jpg         https://mrcal.secretsauce.net/external/2022-11-05--dtla-overpass--samyang--alpha7/stereo/1.jpg",
                            shell = True)

model_filenames=("/tmp/0.cameramodel",
                 "/tmp/1.cameramodel")
image_filenames=("/tmp/0.jpg",
                 "/tmp/1.jpg")

pixels_per_deg_az      = -.25
pixels_per_deg_el      = -.25
az_fov_deg             = dict(LENSMODEL_LATLON  = 160,
                              LENSMODEL_PINHOLE = 120)
el_fov_deg             = 60
az0_deg                = 0
el0_deg                = 0



try:
    models = [mrcal.cameramodel(f) \
              for f in model_filenames]
except FileNotFoundError:
    print("Data not found. Commands to download appear at the top of this script")
    sys.exit(1)

images = [mrcal.load_image(f, bits_per_pixel = 8, channels = 1) \
          for f in image_filenames]

clahe = cv2.createCLAHE()
clahe.setClipLimit(8)
images = [ clahe.apply(image) for image in images ]

# This is a hard-coded property of the OpenCV StereoSGBM implementation
disparity_scale = 16

q = np.array(((30,   100),
              (300,  200),
              (1000, 400),
              (500,  300),
              (80,   500),
              (400,  350),
              (900,  350),
              (1100, 350),
              (1400, 350),
              (400,  450),
              (900,  450),
              (1100, 450),
              (1400, 450),
              (1300, 450)))



for rectification_model in ('LENSMODEL_LATLON',
                            'LENSMODEL_PINHOLE',
                            ):

    models_rectified = \
        mrcal.rectified_system(models,
                               az_fov_deg          = az_fov_deg[rectification_model],
                               el_fov_deg          = el_fov_deg,
                               az0_deg             = az0_deg,
                               el0_deg             = el0_deg,
                               pixels_per_deg_az   = pixels_per_deg_az,
                               pixels_per_deg_el   = pixels_per_deg_el,
                               rectification_model = rectification_model)

    for disparity_min in (0,10,):
        disparity_max = 128

        # round to nearest multiple of disparity_scale. The OpenCV StereoSGBM
        # implementation requires this
        num_disparities = disparity_max - disparity_min
        num_disparities = disparity_scale*round(num_disparities/disparity_scale)

        stereo_sgbm = \
            cv2.StereoSGBM_create(minDisparity      = disparity_min,
                                  numDisparities    = num_disparities,
                                  P1                = 600,
                                  P2                = 2400,
                                  disp12MaxDiff     = 1,
                                  uniquenessRatio   = 5,
                                  speckleWindowSize = 100,
                                  speckleRange      = 2,
                                  mode              = cv2.StereoSGBM_MODE_SGBM)

        rectification_maps = mrcal.rectification_maps(models, models_rectified)

        images_rectified = [mrcal.transform_image(images[i],
                                                  rectification_maps[i]) \
                            for i in range(2)]

        disparity = stereo_sgbm.compute(*images_rectified)
        mask_valid = \
            (disparity > 0) * \
            (disparity >= disparity_min*disparity_scale) * \
            (disparity <= disparity_max*disparity_scale)

        ranges_dense = \
            mrcal.stereo_range( disparity,
                                models_rectified,
                                disparity_scale      = disparity_scale,
                                disparity_min        = disparity_min)
        mask_valid_ranges_dense = ranges_dense > 0

        ranges_dense_python = \
            mrcal.stereo._stereo_range_python( disparity,
                                               models_rectified,
                                               disparity_scale      = disparity_scale,
                                               disparity_min        = disparity_min)
        mask_valid_ranges_dense_pythion = ranges_dense_python > 0

        ranges_sparse = \
            mrcal.stereo_range( disparity[q[:,1], q[:,0]],
                                models_rectified,
                                qrect0               = q,
                                disparity_scale      = disparity_scale,
                                disparity_min        = disparity_min)

        ranges_sparse_python = \
            mrcal.stereo._stereo_range_python( disparity[q[:,1], q[:,0]],
                                               models_rectified,
                                               qrect0               = q,
                                               disparity_scale      = disparity_scale,
                                               disparity_min        = disparity_min)


        testutils.confirm_equal( ranges_dense,
                                 ranges_dense_python,
                                 worstcase = True,
                                 eps = 1e-3,
                                 msg=f'Dense stereo_range() matches in C and Python: rectification_model={rectification_model} disparity_min={disparity_min}')

        testutils.confirm_equal( ranges_sparse,
                                 ranges_sparse_python,
                                 worstcase = True,
                                 eps = 1e-3,
                                 msg=f'Sparse stereo_range() matches in C and Python: rectification_model={rectification_model} disparity_min={disparity_min}')

        testutils.confirm_equal( ranges_sparse,
                                 ranges_dense[q[:,1], q[:,0]],
                                 worstcase = True,
                                 eps = 1e-3,
                                 msg=f'Sparse and dense stereo_range() match: rectification_model={rectification_model} disparity_min={disparity_min}')

        testutils.confirm_equal( mask_valid_ranges_dense,
                                 mask_valid,
                                 msg=f'Dense stereo_range() invalid values handled correctly: rectification_model={rectification_model} disparity_min={disparity_min}')
        testutils.confirm_equal( mask_valid_ranges_dense_pythion,
                                 mask_valid,
                                 msg=f'Dense stereo_range() in Python: invalid values handled correctly: rectification_model={rectification_model} disparity_min={disparity_min}')
        testutils.confirm( np.all(ranges_dense[~mask_valid] == 0.),
                           msg=f'Dense stereo_range() values are all 0 in invalid areas: rectification_model={rectification_model} disparity_min={disparity_min}')



        if False:
            range_image_limits = (1,1000)
            disparity_colored = mrcal.apply_color_map(disparity,
                                                      a_min = disparity_min*disparity_scale,
                                                      a_max = disparity_max*disparity_scale)
            ranges_colored = mrcal.apply_color_map(ranges_dense,
                                                   a_min = range_image_limits[0],
                                                   a_max = range_image_limits[1])
            mrcal.save_image("/tmp/disparity.png", disparity_colored)
            mrcal.save_image("/tmp/range.png",     ranges_colored)
            print("Wrote /tmp/disparity.png and /tmp/range.png")

testutils.finish()
