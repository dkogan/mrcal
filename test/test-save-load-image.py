#!/usr/bin/python3

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

im8 = np.arange(W*H, dtype=np.uint8).reshape(H,W)
im8 *= im8

im16 = np.arange(W*H, dtype=np.uint16).reshape(H,W)
im16 *= im16

im24 = np.arange(W*H*3, dtype=np.uint8).reshape(H,W,3)
im24 *= im24

filename = f"{workdir}/tst.png"

for im,what in ( (im8,  "8bpp grayscale"),
                 (im16, "16bpp grayscale"),
                 (im24, "24bpp bgr")):

    try:
        mrcal.save_image(filename, im)
    except:
        testutils.confirm(False,
                          msg = f"Error saving {what} image")
        continue
    testutils.confirm(True,
                      msg = f"Success saving {what} image")

    try:
        im_check = mrcal.load_image(filename)
    except:
        testutils.confirm(False,
                          msg = f"Error loading {what} image")
        continue
    testutils.confirm(True,
                      msg = f"Success loading {what} image")

    # print(im.shape)
    # print(im_check.shape)
    # print(im.dtype)
    # print(im_check.dtype)

    testutils.confirm_equal(im, im_check,
                            worstcase=True,
                            msg = f"load/save match for {what}")

testutils.finish()
