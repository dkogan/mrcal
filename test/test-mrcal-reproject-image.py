#!/usr/bin/env python3

r'''Tests mrcal-reproject-image tool

THIS IS NOT ENABLED IN THE MAIN SET OF TESTS
The data is not checked-in. I don't remember if this works at all
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
import subprocess
import cv2

def unlink_no_error(f):
    try:    os.unlink(f)
    except: pass

def _check(cmd, filenames_output_ref):
    for o,r in filenames_output_ref:
        unlink_no_error(o)

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='ascii')
        out,err = proc.communicate()
    except:
        errcode = -1
    else:
        errcode = proc.returncode
    if errcode != 0:
        testutils.confirm(False, msg=f"Command failed: {cmd}. stderr: {err}")
        return

    for filename_output, filename_ref in filenames_output_ref:
        try:
            image_output = mrcal.load_image(filename_output)
        except:
            testutils.confirm(False, msg=f"Reading output file failed: {image_output}")
            continue

        try:
            image_ref = mrcal.load_image(filename_ref)
            kernel = np.ones((11,11), dtype = np.uint8)
            d = np.abs(image_ref - image_output, dtype=np.int8).astype(np.uint8)
            worst_err = np.max(cv2.erode(d, kernel))

        except:
            testutils.confirm(False, msg=f"Couldn't compare to reference {filename_ref}")
            continue

        testutils.confirm_equal(worst_err, 0, msg=f"Image discrepancy for {filename_ref}", eps=5)
    return True

def check(cmd, filenames_output_ref):
    if not _check(cmd, filenames_output_ref):
        for o,r in filenames_output_ref:
            unlink_no_error(o)
    # If the test failed I leave the output there, so that the human can look at
    # it





check( ( f"{testdir}/../mrcal-reproject-image",
         "--to-pinhole", "--scale-image", "0.25",
         f"{testdir}/data/reprojection-test/cam0.cameramodel",
         f"{testdir}/data/reprojection-test/cam0.jpg"),
       ( ( f"{testdir}/data/reprojection-test/cam0-reprojected.jpg",
           f"{testdir}/data/reprojection-test/ref-cam0-undistorted.jpg"), ) )

check( ( f"{testdir}/../mrcal-reproject-image",
         "--to-pinhole", "--scale-image", "0.25",
         "--fit", "centers-horizontal",
         f"{testdir}/data/reprojection-test/cam0.cameramodel",
         f"{testdir}/data/reprojection-test/cam0.jpg"),
       ( ( f"{testdir}/data/reprojection-test/cam0-reprojected.jpg",
           f"{testdir}/data/reprojection-test/ref-cam0-undistorted-fit-centers-horizontal.jpg"), ) )

check( ( f"{testdir}/../mrcal-reproject-image",
         "--to-pinhole", "--scale-image", "0.25",
         f"{testdir}/data/reprojection-test/cam0.cameramodel",
         f"{testdir}/data/reprojection-test/cam1.cameramodel",
         f"{testdir}/data/reprojection-test/cam0.jpg",
         f"{testdir}/data/reprojection-test/cam1.jpg"),
       ( ( f"{testdir}/data/reprojection-test/cam0-pinhole-remapped.jpg",
           f"{testdir}/data/reprojection-test/ref-cam0-reprojected-to-cam1-pinhole-with-rotation.jpg"  ),
         ( f"{testdir}/data/reprojection-test/cam1-pinhole.jpg",
           f"{testdir}/data/reprojection-test/ref-cam1-reprojected-to-cam1-pinhole-with-rotation.jpg"  ), ) )

check( ( f"{testdir}/../mrcal-reproject-image",
         "--to-pinhole", "--scale-image", "0.25",
         "--intrinsics-only",
         f"{testdir}/data/reprojection-test/cam0.cameramodel",
         f"{testdir}/data/reprojection-test/cam1.cameramodel",
         f"{testdir}/data/reprojection-test/cam0.jpg",
         f"{testdir}/data/reprojection-test/cam1.jpg"),
       ( ( f"{testdir}/data/reprojection-test/cam0-pinhole-remapped.jpg",
           f"{testdir}/data/reprojection-test/ref-cam0-reprojected-to-cam1-pinhole-without-rotation.jpg"  ),
         ( f"{testdir}/data/reprojection-test/cam1-pinhole.jpg",
           f"{testdir}/data/reprojection-test/ref-cam1-reprojected-to-cam1-pinhole-without-rotation.jpg"  ), ) )

# non-pinhole reprojection. SLOW
check( ( f"{testdir}/../mrcal-reproject-image",
         f"{testdir}/data/reprojection-test/cam0.cameramodel",
         f"{testdir}/data/reprojection-test/cam1.cameramodel",
         f"{testdir}/data/reprojection-test/cam0.jpg"),
       ( ( f"{testdir}/data/reprojection-test/cam0-reprojected.jpg",
           f"{testdir}/data/reprojection-test/ref-cam0-reprojected-to-cam1-with-rotation.jpg" ), ) )

check( ( f"{testdir}/../mrcal-reproject-image",
         "--to-pinhole", "--scale-image", "0.25",
         "--plane-n", "-0.011", "0.990", "0.142", "--plane-d", "3.5",
         f"{testdir}/data/reprojection-test/cam0.cameramodel",
         f"{testdir}/data/reprojection-test/cam1.cameramodel",
         f"{testdir}/data/reprojection-test/cam0.jpg",
         f"{testdir}/data/reprojection-test/cam1.jpg"),
       ( ( f"{testdir}/data/reprojection-test/cam0-pinhole-remapped.jpg",
           f"{testdir}/data/reprojection-test/ref-cam0-reprojected-to-cam1-pinhole-plane.jpg"  ),
         ( f"{testdir}/data/reprojection-test/cam1-pinhole.jpg",
           f"{testdir}/data/reprojection-test/ref-cam1-reprojected-to-cam1-pinhole-plane.jpg"  ), ) )

testutils.finish()
