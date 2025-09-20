#!/usr/bin/env python3

r'''Test of the mrcal-graft-models tool

The basic usage of the tool is simple, but it also supports a non-trivial mode
where it applies an implied transformation. I make sure to test that here

'''

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



rt_0r = np.array((1.,  2.,    3.,   4., 5.,  6.))
rt_1r = np.array((0.8, -0.01, -0.2, 2., -4., 2.))

imagersize = np.array((2000, 1000), dtype=int)

cxy_center  = (imagersize-1.)/2.
pitch_y     = 100.
cxy_pitched = cxy_center + np.array((0., pitch_y))

# very long lenses. I want rotation to look very much like panning
fxycxy0  = nps.glue( np.array(( 50000., 50000.)), cxy_pitched, axis=-1)
fxycxy1  = nps.glue( np.array(( 50000., 50000.)), cxy_center,  axis=-1)

model0 = mrcal.cameramodel( intrinsics            = ('LENSMODEL_PINHOLE', fxycxy0),
                            imagersize            = imagersize,
                            rt_cam_ref = rt_0r )

model1 = mrcal.cameramodel( intrinsics            = ('LENSMODEL_PINHOLE', fxycxy1),
                            imagersize            = imagersize,
                            rt_cam_ref = rt_1r )

filename0 = f"{workdir}/model0.cameramodel"
filename1 = f"{workdir}/model1.cameramodel"

model0.write(filename0)
model1.write(filename1)

# Basic test. Combine intrinsics and extrinsics without fitting any extra
# transform
out = subprocess.check_output( (f"{testdir}/../mrcal-graft-models",
                                filename0, filename1),
                               encoding = 'ascii',
                               stderr   =  subprocess.DEVNULL)

filename01 = f"{workdir}/model01.cameramodel"
with open(filename01, "w") as f:
    print(out, file=f)

model01 = mrcal.cameramodel(filename01)

testutils.confirm_equal(model01.intrinsics()[1], model0.intrinsics()[1],
                        msg = f"Basic grafted intrinsics match",
                        eps = 1.0e-6)
testutils.confirm_equal(model01.rt_cam_ref(), model1.rt_cam_ref(),
                        msg = f"Basic grafted extrinsics match",
                        eps = 1.0e-6)


# More complicated test. I want to compensate for the different intrinsics with
# modified extrinsics such that the old-intrinsics and new-intrinsics project
# world points to the same place
out = subprocess.check_output( (f"{testdir}/../mrcal-graft-models",
                                '--radius', '-1',
                                filename0, filename1),
                               encoding = 'ascii',
                               stderr   =  subprocess.DEVNULL)

filename01_compensated = f"{workdir}/model01_compensated.cameramodel"
with open(filename01_compensated, "w") as f:
    print(out, file=f)

model01_compensated = mrcal.cameramodel(filename01_compensated)

p1 = np.array((11., 17., 10000.))
pref = mrcal.transform_point_rt( model1.rt_ref_cam(),
                                 p1)

q = mrcal.project( mrcal.transform_point_rt( model1.rt_cam_ref(),
                                             pref ),
                   *model1.intrinsics())
q_compensated = \
    mrcal.project( mrcal.transform_point_rt( model01_compensated.rt_cam_ref(),
                                             pref),
                   *model01_compensated.intrinsics())

testutils.confirm_equal(q_compensated, q,
                        msg = f"Compensated projection ended up in the same place",
                        eps = 0.1)

testutils.finish()
