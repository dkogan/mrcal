#!/usr/bin/python3

r'''Test of the mrcal-convert-lensmodel tool

'''

import sys
import numpy as np
import numpysane as nps
import os
import subprocess
import re

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


filename_splined          = f"{testdir}/data/cam0.splined.cameramodel"
filename_format_converted = workdir + "/cam0.splined-{model}.cameramodel"
model_splined             = mrcal.cameramodel(filename_splined)


def check(lensmodel, args, *,
          must_warn_of_aphysical_translation,
          what):
    process = \
        subprocess.Popen( (f"{testdir}/../mrcal-convert-lensmodel",
                           *args,
                           # need multiple attempts to hit the accuracy targets below
                           "--num-trials", "4",
                           "-f",
                           "--outdir", workdir,
                           lensmodel,
                           filename_splined,),
                          encoding = 'ascii',
                          stdout   = subprocess.PIPE,
                          stderr   = subprocess.PIPE)
    stdout,stderr = process.communicate()

    if process.returncode != 0:
        testutils.confirm( False, msg = f"{what}: convert failed: {stderr}" )
    else:
        testutils.confirm( True, msg = f"{what}: convert succeeded" )

        have_warn_of_aphysical_translation = \
            re.search("WARNING: fitted camera moved by.*aphysically high", stderr)
        if must_warn_of_aphysical_translation:
            testutils.confirm( have_warn_of_aphysical_translation,
                               msg = "Expected a warning about an aphysical translation")
        else:
            testutils.confirm( not have_warn_of_aphysical_translation,
                               msg = "Expected no warning about an aphysical translation")

        filename_converted = filename_format_converted.format(model = lensmodel)
        model_converted    = mrcal.cameramodel(filename_converted)

        difflen, diff, q0, implied_Rt10 = \
            mrcal.projection_diff( (model_converted, model_splined),
                                   use_uncertainties = False,
                                   distance          = 3)
        icenter = np.array(difflen.shape) // 2
        ithird  = np.array(difflen.shape) // 3

        testutils.confirm_equal( 0, difflen[icenter[0],icenter[1]],
                                 eps = 0.6,
                                 msg = f"{what}: low-enough diff at the center")

        testutils.confirm_equal( 0, difflen[ithird[0],ithird[1]],
                                 eps = 0.6,
                                 msg = f"{what}: low-enough diff at 1/3 from top-left")


check( "LENSMODEL_CAHVOR",
       ("--radius", "800",
        "--intrinsics-only",
        "--sampled"),
       must_warn_of_aphysical_translation = False,
       what = 'CAHVOR, sampled, intrinsics-only',)
check( "LENSMODEL_CAHVOR",
       ("--radius", "800",
        "--sampled",
        "--distance", "3"),
       must_warn_of_aphysical_translation = True,
       what = 'CAHVOR, sampled at 3m',)
check( "LENSMODEL_CAHVOR",
       ("--radius", "800",
        "--sampled",
        "--distance", "3000"),
       must_warn_of_aphysical_translation = True,
       what = 'CAHVOR, sampled at 3000m',)
check( "LENSMODEL_CAHVOR",
       ("--radius", "800",
        "--sampled",
        "--distance", "3000,3"),
       must_warn_of_aphysical_translation = False,
       what = 'CAHVOR, sampled at 3000m,3m',)

testutils.finish()
