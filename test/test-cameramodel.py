#!/usr/bin/python3

r'''Tests for cameramodel reading/writing'''

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


# I do this:
#   load file
#   compare with hardcoded
#   save
#   load again
#   compare with hardcoded
#
#   modify with setter
#   call getter and compare

m = mrcal.cameramodel(f"{testdir}/data/opencv8.cameramodel")

testutils.confirm_equal( m.extrinsics_rt_fromref(), [ 2e-2, -3e-1, -1e-2,  1., 2, -3., ] )
testutils.confirm_equal( m.intrinsics()[0], 'DISTORTION_OPENCV8' )
testutils.confirm_equal( m.intrinsics()[1], [ 1761.181055, 1761.250444, 1965.706996, 1087.518797, -0.01266096516, 0.03590794372, -0.0002547045941, 0.0005275929652, 0.01968883397, 0.01482863541, -0.0562239888, 0.0500223357,] )

m.write(f'{workdir}/out.cameramodel')
m.write(f'{workdir}/out.cahvor')

m1 = mrcal.cameramodel(f'{workdir}/out.cameramodel')
testutils.confirm_equal( m1.extrinsics_rt_fromref(), [ 2e-2, -3e-1, -1e-2,  1., 2, -3., ] )
testutils.confirm_equal( m1.intrinsics()[0], 'DISTORTION_OPENCV8' )
testutils.confirm_equal( m1.intrinsics()[1], [ 1761.181055, 1761.250444, 1965.706996, 1087.518797, -0.01266096516, 0.03590794372, -0.0002547045941, 0.0005275929652, 0.01968883397, 0.01482863541, -0.0562239888, 0.0500223357,] )

m2 = mrcal.cameramodel(f'{workdir}/out.cahvor')
testutils.confirm_equal( m2.extrinsics_rt_fromref(), [ 2e-2, -3e-1, -1e-2,  1., 2, -3., ] )
testutils.confirm_equal( m2.intrinsics()[0], 'DISTORTION_OPENCV8' )
testutils.confirm_equal( m2.intrinsics()[1], [ 1761.181055, 1761.250444, 1965.706996, 1087.518797, -0.01266096516, 0.03590794372, -0.0002547045941, 0.0005275929652, 0.01968883397, 0.01482863541, -0.0562239888, 0.0500223357,] )


rt_0r = np.array([ 4e-1, -1e-2, 1e-3,  -2., 3, -5., ])
Rt_r0 = mrcal.invert_Rt( mrcal.Rt_from_rt( rt_0r ))
m.extrinsics_Rt_toref( Rt_r0 )
testutils.confirm_equal( m.extrinsics_rt_fromref(), rt_0r )

testutils.finish()
