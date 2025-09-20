#!/usr/bin/env python3

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

m = mrcal.cameramodel(f"{testdir}/data/cam0.opencv8.cameramodel")

testutils.confirm_equal( m.rt_cam_ref(), [ 2e-2, -3e-1, -1e-2,  1., 2, -3., ] )
testutils.confirm_equal( m.intrinsics()[0], 'LENSMODEL_OPENCV8' )
testutils.confirm_equal( m.intrinsics()[1], [ 1761.181055, 1761.250444, 1965.706996, 1087.518797, -0.01266096516, 0.03590794372, -0.0002547045941, 0.0005275929652, 0.01968883397, 0.01482863541, -0.0562239888, 0.0500223357,] )

m.write(f'{workdir}/out.cameramodel')
m.write(f'{workdir}/out.cahvor')

m1 = mrcal.cameramodel(f'{workdir}/out.cameramodel')
testutils.confirm_equal( m1.rt_cam_ref(), [ 2e-2, -3e-1, -1e-2,  1., 2, -3., ] )
testutils.confirm_equal( m1.intrinsics()[0], 'LENSMODEL_OPENCV8' )
testutils.confirm_equal( m1.intrinsics()[1], [ 1761.181055, 1761.250444, 1965.706996, 1087.518797, -0.01266096516, 0.03590794372, -0.0002547045941, 0.0005275929652, 0.01968883397, 0.01482863541, -0.0562239888, 0.0500223357,] )

m2 = mrcal.cameramodel(f'{workdir}/out.cahvor')
testutils.confirm_equal( m2.rt_cam_ref(), [ 2e-2, -3e-1, -1e-2,  1., 2, -3., ] )
testutils.confirm_equal( m2.intrinsics()[0], 'LENSMODEL_OPENCV8' )
testutils.confirm_equal( m2.intrinsics()[1], [ 1761.181055, 1761.250444, 1965.706996, 1087.518797, -0.01266096516, 0.03590794372, -0.0002547045941, 0.0005275929652, 0.01968883397, 0.01482863541, -0.0562239888, 0.0500223357,] )

rt_0r = np.array([ 4e-1, -1e-2, 1e-3,  -2., 3, -5., ])
Rt_r0 = mrcal.invert_Rt( mrcal.Rt_from_rt( rt_0r ))
m.Rt_ref_cam( Rt_r0 )
testutils.confirm_equal( m.rt_cam_ref(), rt_0r )

# Let's make sure I can read and write empty and non-empty valid-intrinsics
# regions
m = mrcal.cameramodel(f"{testdir}/data/cam0.opencv8.cameramodel")
testutils.confirm_equal( m.valid_intrinsics_region(), None,
                         msg = "read 'valid_intrinsics_region is None' properly")

r_open = np.array(((0,  0),
                   (0, 10),
                   (10,10),
                   (10, 0)))
r_closed = np.array(((0,  0),
                     (0, 10),
                     (10,10),
                     (10, 0),
                     (0,  0)))
r_empty = np.zeros((0,2))

m.valid_intrinsics_region(r_open)
testutils.confirm_equal( m.valid_intrinsics_region(), r_closed,
                         msg="was able to set an open valid_intrinsics_region and to see it be closed")
m.write(f'{workdir}/out.cameramodel')
m1 = mrcal.cameramodel(f'{workdir}/out.cameramodel')
testutils.confirm_equal( m1.valid_intrinsics_region(), r_closed,
                         msg="read 'valid_intrinsics_region' properly")

m.valid_intrinsics_region(r_empty)
testutils.confirm_equal( m.valid_intrinsics_region(), r_empty,
                         msg="was able to set an empty valid_intrinsics_region")
m.write(f'{workdir}/out.cameramodel')
m1 = mrcal.cameramodel(f'{workdir}/out.cameramodel')
testutils.confirm_equal( m1.valid_intrinsics_region(), r_empty,
                         msg="read empty valid_intrinsics_region properly")


# Make sure we can read model data with extra spacing
string = r'''
{
    'lens_model':  'LENSMODEL_OPENCV8',

    # intrinsics are fx,fy,cx,cy,distortion0,distortion1,....
    'intrinsics': [ 1761.181055, 1761.250444,
                    1965.706996, 1087.518797,

  -0.01266096516, 0.03590794372, -0.0002547045941,
                    0.0005275929652, 0.01968883397, 0.01482863541, -0.0562239888, 0.0500223357,],

    # extrinsics are rt_fromref
    'extrinsics': [ 2e-2, -3e-1, -1e-2,  1., 2, -3., ],

    'imagersize': [ 4000, 2200 ]
}
'''

import io
with io.StringIO(string) as f:
    m = mrcal.cameramodel(f)

    testutils.confirm_equal( m.intrinsics()[1], [ 1761.181055, 1761.250444, 1965.706996, 1087.518797, -0.01266096516, 0.03590794372, -0.0002547045941, 0.0005275929652, 0.01968883397, 0.01482863541, -0.0562239888, 0.0500223357,],
                             msg="extra spaces don't confuse the parser")





testutils.finish()
