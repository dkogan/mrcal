#!/usr/bin/python3

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

image = mrcal.load_image(f'{testdir}/data/figueroa-overpass-looking-S.0.downsampled.jpg',
                         bits_per_pixel = 8,
                         channels       = 1)

# some made-up homography with scaling, rotating, skewing and translating
H01 = np.array((( 0.7, 0.3, 1.),
                ( -0.4, 0.95, 2.),
                ( 3e-5, 4e-6, 1.),),
               dtype=np.float32)

H10 = np.linalg.inv(H01)

# The feature I'm going to test with. This is the corner of one of the towers
q0 = np.array((294,159), dtype=np.float32)

# The transformed image. The matcher should end-up reversing this
# transformation, since it will be given the homography.
#
# shape (H,W,2)
image1 = \
    mrcal.transform_image(
        image,
        mrcal.apply_homography(
            H01,
            nps.glue(*[ nps.dummy(arr, -1) for arr in \
                        np.meshgrid( np.arange(500),
                                     np.arange(600))],
                     axis=-1).astype(np.float32) ))

# I have the source images and the "right" homography and the "right" matching
# pixels coords. Run the matcher, and compare

templatesize  = (30,20)
search_radius = 50

H10_shifted = H10.copy()
H10_shifted[0,2] += 10.2
H10_shifted[1,2] -= 20.4

q1_matched, diagnostics = \
    mrcal.match_feature( image, image1,
                         q0,
                         H10            = H10_shifted,
                         search_radius1 = 50,
                         template_size1 = templatesize,
                         method         = cv2.TM_CCOEFF_NORMED)
testutils.confirm_equal( q1_matched,
                         mrcal.apply_homography(H10, q0),
                         worstcase = True,
                         eps = 0.1,
                         msg=f'match_feature(method=TM_CCOEFF_NORMED) reports the correct pixel coordinate')

q1_matched, diagnostics = \
    mrcal.match_feature( image, image1,
                         q0,
                         H10            = H10_shifted,
                         search_radius1 = 50,
                         template_size1 = templatesize,
                         method         = cv2.TM_SQDIFF_NORMED)
testutils.confirm_equal( q1_matched,
                         mrcal.apply_homography(H10, q0),
                         worstcase = True,
                         eps = 0.1,
                         msg=f'match_feature(method=TM_SQDIFF_NORMED) reports the correct pixel coordinate')

q1_matched, diagnostics = \
    mrcal.match_feature( image, image1,
                         q0,
                         H10            = H10_shifted,
                         search_radius1 = 1000,
                         template_size1 = templatesize,
                         method         = cv2.TM_CCOEFF_NORMED)
testutils.confirm_equal( q1_matched,
                         mrcal.apply_homography(H10, q0),
                         worstcase = True,
                         eps = 0.1,
                         msg=f'out-of-bounds search_radius works ok')
templatesize_hw = np.array((templatesize[-1],templatesize[-2]))
testutils.confirm_equal( diagnostics['matchoutput_image'].shape,
                         image1.shape - templatesize_hw + 1,
                         msg = 'out-of-bounds search radius looks at the whole image')

q1_matched, diagnostics = \
    mrcal.match_feature( image*0, image1,
                         q0,
                         H10            = H10_shifted,
                         search_radius1 = 50,
                         template_size1 = templatesize,
                         method         = cv2.TM_CCOEFF_NORMED)
testutils.confirm_equal( q1_matched, None,
                         msg = 'failing correlation returns None')

try:
    mrcal.match_feature( image*0, image1,
                         q0,
                         H10            = H10_shifted,
                         search_radius1 = 50,
                         template_size1 = (5000, 5000),
                         method         = cv2.TM_CCOEFF_NORMED)
except:
    testutils.confirm(True, msg='Too-big template size throws an exception')
else:
    testutils.confirm(False, msg='Too-big template size throws an exception')

testutils.finish()
