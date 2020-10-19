#!/usr/bin/python3

r'''Test of the mrcal-convert-lensmodel tool

'''





# add test for mrcal-convert-lensmodel. Should test
# 1. reoptimization
# 2. sampled at various distances with/without uncertainties
#
# splined -> opencv8 is probably the interesting direction. I can imagine
# that the no-geometry sampled solves would fail here because opencv8 just
# wouldn't fit in that case

# splined model: fix the core









import sys
import numpy as np
import numpysane as nps
import os
import subprocess

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


filename_splined = f"{testdir}/data/cam0.splined.cameramodel"
with open(filename_splined, "r") as f: text_splined = f.read()
model_splined = mrcal.cameramodel(filename_splined)

# These settings are semi-arbitrary. I could test that higher radii fit more
# stuff until we go too high, and it doesn't fit at all anymore. Need --sampled
# because my models don't have optimization_inputs. For a basic test this is
# fine
text_out_cahvor = \
    subprocess.check_output( (f"{testdir}/../mrcal-convert-lensmodel",
                              "--radius", "800",
                              "--intrinsics-only",
                              "--sampled",
                              "--distance", "3",
                              "LENSMODEL_CAHVOR",
                              "-",),
                             encoding = 'ascii',
                             input    = text_splined,
                             stderr   = subprocess.DEVNULL)

filename_out_cahvor = f"{workdir}/cam0.out.cahvor.cameramodel"
with open(filename_out_cahvor, "w") as f:
    print(text_out_cahvor, file=f)

model_out_cahvor = mrcal.cameramodel(filename_out_cahvor)

difflen, diff, q0, implied_Rt10 = \
    mrcal.projection_diff( (model_out_cahvor, model_splined),
                           use_uncertainties = False,
                           distance          = 3)
icenter = np.array(difflen.shape) // 2

testutils.confirm_equal( 0, difflen[icenter[0],icenter[1]],
                         eps = 0.1,
                         msg = "Low-enough diff at the center")

testutils.finish()
