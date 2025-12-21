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


def check_load(filename, image, what):
    try:
        image_check = mrcal.load_image(filename)
    except:
        testutils.confirm(False,
                          msg = f"Error loading {what} image")
        return
    testutils.confirm(True,
                      msg = f"Success loading {what} image")

    # print(image.shape)
    # print(image_check.shape)
    # print(image.dtype)
    # print(image_check.dtype)

    testutils.confirm_equal(image, image_check,
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

    check_load(filename,image,what)

testutils.finish()
