#!/usr/bin/python3
import sys
import mrcal
import cv2
import numpy as np

# Read the model and image from the commandline arguments
try:
    model   = mrcal.cameramodel(sys.argv[1])
    image   = cv2.imread(sys.argv[2])
    yaw_deg = float(sys.argv[3])
    what    = sys.argv[4]

except:
    print(f"Usage: {sys.argv[0]} model image yaw_deg what", file=sys.stderr)
    sys.exit(1)

# I want a pinhole model to cover the middle 1/3rd of my pixels
W,H = model.imagersize()
fit_points = \
    np.array((( W/3.,    H/3.),
              ( W*2./3., H/3.),
              ( W/3.,    H*2./3.),
              ( W*2./3., H*2./3.)))

model_pinhole = \
    mrcal.pinhole_model_for_reprojection(model,
                                         fit         = fit_points,
                                         scale_image = 0.5)

# yaw transformation: pure rotation around the y axis
rt_yaw = np.array((0., yaw_deg*np.pi/180., 0,  0,0,0))

# apply the extra yaw transformation to my extrinsics
model_pinhole.extrinsics_rt_toref( \
    mrcal.compose_rt(model_pinhole.extrinsics_rt_toref(),
                     rt_yaw) )

mapxy = mrcal.image_transformation_map(model, model_pinhole,
                                       use_rotation = True)

image_transformed = mrcal.transform_image(image, mapxy)

cv2.imwrite(f'/tmp/narrow-{what}.jpg', image_transformed)
model_pinhole.write(f'/tmp/pinhole-narrow-yawed-{what}.cameramodel')
