#!/usr/bin/env python3

r'''Test of the mrcal-convert-lensmodel tool

'''

import sys
import numpy as np
import numpysane as nps
import os
import subprocess
import re
import io

testdir = os.path.dirname(os.path.realpath(__file__))

# I import the LOCAL mrcal since that's what I'm testing
sys.path[:0] = f"{testdir}/..",
import mrcal
import testutils


def check(filename_from, model_from,
          lensmodel_to,
          args, *,
          must_warn_of_aphysical_translation,
          what,
          distance = 3):
    process = \
        subprocess.Popen( (f"{testdir}/../mrcal-convert-lensmodel",
                           *args,
                           lensmodel_to),
                          encoding = 'ascii',
                          stdin    = subprocess.PIPE,
                          stdout   = subprocess.PIPE,
                          stderr   = subprocess.PIPE)

    with open(filename_from, "r") as f:
        text_from = f.read()
    stdout,stderr = process.communicate(input = text_from)

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

        with io.StringIO(stdout) as f:
            model_converted = mrcal.cameramodel(f)

        difflen, diff, q0, implied_Rt10 = \
            mrcal.projection_diff( (model_converted, model_from),
                                   use_uncertainties = False,
                                   distance          = distance,
                                   focus_radius      = 800)
        icenter = np.array(difflen.shape) // 2
        ithird  = np.array(difflen.shape) // 3

        testutils.confirm_equal( 0, difflen[icenter[0],icenter[1]],
                                 eps = 2.0,
                                 msg = f"{what}: low-enough diff at the center")

        testutils.confirm_equal( 0, difflen[ithird[0],ithird[1]],
                                 eps = 3.0,
                                 msg = f"{what}: low-enough diff at 1/3 from top-left")

filename_from = f"{testdir}/data/cam0.splined.cameramodel"
model_from    = mrcal.cameramodel(filename_from)

check( filename_from, model_from,
       "LENSMODEL_CAHVOR",
       ("--radius", "1000",
        "--intrinsics-only",
        "--sampled",
        # need multiple attempts to hit the accuracy targets below
        "--num-trials", "8",),
       must_warn_of_aphysical_translation = False,
       what = 'CAHVOR, sampled, intrinsics-only',)

check( filename_from, model_from,
       "LENSMODEL_CAHVOR",
       ("--radius", "1000",
        "--sampled",
        "--distance", "3",
        # need multiple attempts to hit the accuracy targets below
        "--num-trials", "8",),
       must_warn_of_aphysical_translation = True,
       what = 'CAHVOR, sampled at 3m',)
check( filename_from, model_from,
       "LENSMODEL_CAHVOR",
       ("--radius", "1000",
        "--sampled",
        "--distance", "3000",
        # need multiple attempts to hit the accuracy targets below
        "--num-trials", "8",),
       must_warn_of_aphysical_translation = True,
       what = 'CAHVOR, sampled at 3000m',)
check( filename_from, model_from,
       "LENSMODEL_CAHVOR",
       ("--radius", "1000",
        "--sampled",
        "--distance", "3000,3",
        # need multiple attempts to hit the accuracy targets below
        "--num-trials", "8",),
       must_warn_of_aphysical_translation = False,
       what = 'CAHVOR, sampled at 3000m,3m',)

# Need a model with optimization_inputs to do non-sampled fits
if not os.path.isdir(f"{testdir}/../../mrcal-doc-external"):
    testutils.print_blue(f"../mrcal-doc-external isn't on this disk. Skipping non-sampled tests")
else:

    filename_from = f"{testdir}/../../mrcal-doc-external/2022-11-05--dtla-overpass--samyang--alpha7/2-f22-infinity/splined.cameramodel"
    model_from = mrcal.cameramodel(filename_from)

    check( filename_from, model_from,
           "LENSMODEL_OPENCV8",
           ("--radius", "-1000"),
           must_warn_of_aphysical_translation = False,
           what = 'non-sampled fit to opencv8',)

testutils.finish()
