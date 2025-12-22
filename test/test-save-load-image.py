#!/usr/bin/env python3

r'''Test of the save_image() and load_image() functions
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



import tempfile
import atexit
import shutil
workdir = tempfile.mkdtemp()
def cleanup():
    global workdir
    try:
        shutil.rmtree(workdir)
        workdir = None
    except:
        pass
atexit.register(cleanup)


W,H = 133,211

images = dict()
images[8] = np.arange(W*H, dtype=np.uint8).reshape(H,W)
images[8] *= images[8]

images[16] = np.arange(W*H, dtype=np.uint16).reshape(H,W)
images[16] *= images[16]

images[24] = np.arange(W*H*3, dtype=np.uint8).reshape(H,W,3)
images[24] *= images[24]

filename = f"{workdir}/tst.png"


def _check_load(filename, image_ref, what,
                should_fail_load = False,
                compare_value    = True,
                **kwargs):

    if should_fail_load:
        testutils.confirm_raises(lambda: mrcal.load_image(filename, **kwargs),
                                 msg = f"Loading {what} image should fail")
        return


    try:
        image_check = mrcal.load_image(filename, **kwargs)
    except Exception as e:
        testutils.confirm(False,
                          msg = f"Error loading {what} image: {e}")

        return
    testutils.confirm(True,
                      msg = f"Success loading {what} image")

    # print(image_ref.shape)
    # print(image_check.shape)
    # print(image_ref.dtype)
    # print(image_check.dtype)

    bpp_load      = kwargs.get('bits_per_pixel')
    channels_load = kwargs.get('channels')
    if bpp_load is not None:
        H,W = image_ref.shape[:2]
        if bpp_load == 8:
            testutils.confirm_equal(image_check.shape, (H,W),
                                    msg = "Shapes match. Loading 8bpp-1channel image")
            testutils.confirm_equal(image_check.ndim, 2,
                                    msg = "channels match. Loading 8bpp-1channel image")
            testutils.confirm_equal(image_check.dtype, np.uint8,
                                    msg = "dtype match. Loading 8bpp-1channel image")
        elif bpp_load == 16:
            testutils.confirm_equal(image_check.shape, (H,W),
                                    msg = "Shapes match. Loading 16bpp-1channel image")
            testutils.confirm_equal(image_check.ndim, 2,
                                    msg = "channels match. Loading 16bpp-1channel image")
            testutils.confirm_equal(image_check.dtype, np.uint16,
                                    msg = "dtype match. Loading 16bpp-1channel image")
        elif bpp_load == 24:
            testutils.confirm_equal(image_check.shape, (H,W,3),
                                    msg = "Shapes match. Loading 24bpp-3channel image")
            testutils.confirm_equal(image_check.ndim, 3,
                                    msg = "channels match. Loading 24bpp-3channel image")
            testutils.confirm_equal(image_check.dtype, np.uint8,
                                    msg = "dtype match. Loading 24bpp-3channel image")

    if compare_value:
        if bpp_load == 24:
            image_ref = nps.glue( nps.dummy(image_ref,-1),
                                  nps.dummy(image_ref,-1),
                                  nps.dummy(image_ref,-1),
                                  axis = -1)
        testutils.confirm_equal(image_ref, image_check,
                                worstcase=True,
                                msg = f"load/save match for {what}")



for bpp,what in ( (8,  "8bpp grayscale"),
                 (16, "16bpp grayscale"),
                 (24, "24bpp bgr")):

    image = images[bpp]

    try:
        mrcal.save_image(filename, image)
    except:
        testutils.confirm(False,
                          msg = f"Error saving {what} image")
        continue
    testutils.confirm(True,
                      msg = f"Success saving {what} image")

    def check_load(**kwargs):
        if 'bits_per_pixel' not in kwargs:
            _check_load(filename, image,
                        what,
                        **kwargs)
        else:
            _check_load(filename, image,
                        f"{what} (loading as {kwargs['bits_per_pixel']}bpp)",
                        **kwargs)

    check_load()

    if bpp == 8:
        check_load(bits_per_pixel = 16,
                   channels       = 1,
                   should_fail_load = True)
        check_load(bits_per_pixel = 24,
                   channels       = 3)

    if bpp == 16:
        check_load(bits_per_pixel = 8,
                   channels       = 1,
                   # values won't match: this path equalizes
                   compare_value  = False)
        check_load(bits_per_pixel = 24,
                   channels       = 3,
                   should_fail_load = True)

    if bpp == 24:
        check_load(bits_per_pixel = 8,
                   channels       = 1,
                   # values won't match: colors are collapsed
                   compare_value  = False)
        check_load(bits_per_pixel = 16,
                   channels       = 1,
                   should_fail_load = True)

testutils.finish()
