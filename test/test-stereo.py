#!/usr/bin/python3

r'''Tests the stereo routines
'''

import sys
import numpy as np
import numpysane as nps
import os

testdir = os.path.dirname(os.path.realpath(__file__))

# I import the LOCAL mrcal since that's what I'm testing
sys.path[:0] = f"{testdir}/..",
import mrcal
import scipy.interpolate
import testutils


model0 = mrcal.cameramodel(f"{testdir}/data/cam0.opencv8.cameramodel")
model1 = mrcal.cameramodel(model0)

# I create geometries to test. First off, a vanilla geometry for left-right stereo
rt01 = np.array((0,0,0,  3.0, 0, 0))
model1.extrinsics_rt_toref( mrcal.compose_rt(model0.extrinsics_rt_toref(),
                                             rt01))

az_fov_deg = 90
el_fov_deg = 50
models_rectified = \
    mrcal.rectified_system( (model0, model1),
                            az_fov_deg = az_fov_deg,
                            el_fov_deg = el_fov_deg,
                            pixels_per_deg_az = -1./8.,
                            pixels_per_deg_el = -1./4.)
try:
    mrcal.stereo._validate_models_rectified(models_rectified)
    testutils.confirm(True,
                      msg='Generated models pass validation')
except:
    testutils.confirm(False,
                      msg='Generated models pass validation')

Rt_cam0_stereo = mrcal.compose_Rt( model0.extrinsics_Rt_fromref(),
                                   models_rectified[0].extrinsics_Rt_toref())
Rt01_rectified = mrcal.compose_Rt( models_rectified[0].extrinsics_Rt_fromref(),
                                   models_rectified[1].extrinsics_Rt_toref())
fxycxy = models_rectified[0].intrinsics()[1]

testutils.confirm_equal(Rt_cam0_stereo, mrcal.identity_Rt(),
                        msg='vanilla stereo has a vanilla geometry')

testutils.confirm_equal( Rt01_rectified[3,0],
                         nps.mag(rt01[3:]),
                         msg='vanilla stereo: baseline')

q0,q0x,q0y = mrcal.project( np.array(((0,      0,1.),
                                      (1e-6,   0,1.),
                                      (0,   1e-6, 1.))), *model0.intrinsics() )

testutils.confirm_equal(fxycxy[0] * np.pi/180. * 8.,
                        (q0x-q0)[0] / 1e-6 * np.pi/180.,
                        msg='vanilla stereo: correct az pixel density',
                        eps = 0.05)

testutils.confirm_equal(fxycxy[1] * np.pi/180. * 4.,
                        (q0y-q0)[1] / 1e-6 * np.pi/180.,
                        msg='vanilla stereo: correct el pixel density',
                        eps = 0.05)


# Weirder geometry. Left-right stereo, with sizeable rotation and position fuzz.
# I especially make sure there's a forward/back shift
rt01 = np.array((0.1, 0.2, 0.05,  3.0, 0.2, 1.0))
model1.extrinsics_rt_toref( mrcal.compose_rt(model0.extrinsics_rt_toref(),
                                             rt01))
models_rectified = \
    mrcal.rectified_system( (model0, model1),
                            az_fov_deg = az_fov_deg,
                            el_fov_deg = el_fov_deg,
                            pixels_per_deg_az = -1./8.,
                            pixels_per_deg_el = -1./4.)
try:
    mrcal.stereo._validate_models_rectified(models_rectified)
    testutils.confirm(True,
                      msg='Generated models pass validation')
except:
    testutils.confirm(False,
                      msg='Generated models pass validation')

Rt_cam0_stereo = mrcal.compose_Rt( model0.extrinsics_Rt_fromref(),
                                   models_rectified[0].extrinsics_Rt_toref())
Rt01_rectified = mrcal.compose_Rt( models_rectified[0].extrinsics_Rt_fromref(),
                                   models_rectified[1].extrinsics_Rt_toref())
fxycxy = models_rectified[0].intrinsics()[1]

# I visualized the geometry, and confirmed that it is correct. The below array
# is the correct-looking geometry
#
# Rt_cam0_ref    = model0.extrinsics_Rt_fromref()
# Rt_stereo_ref  = mrcal.compose_Rt( mrcal.invert_Rt(Rt_cam0_stereo),
#                                    Rt_cam0_ref )
# rt_stereo_ref  = mrcal.rt_from_Rt(Rt_stereo_ref)
# mrcal.show_geometry( [ model0, model1, rt_stereo_ref ],
#                      ( "camera0", "camera1", "stereo" ),
#                      show_calobjects = False,
#                      wait            = True )
# print(repr(Rt_cam0_stereo))

testutils.confirm_equal(Rt_cam0_stereo,
                        np.array([[ 0.9467916 , -0.08500675, -0.31041828],
                                  [ 0.06311944,  0.99480206, -0.07990489],
                                  [ 0.3155972 ,  0.05605985,  0.94723582],
                                  [ 0.        , -0.        , -0.        ]]),
                        msg='funny stereo geometry')

testutils.confirm_equal( Rt01_rectified[3,0],
                         nps.mag(rt01[3:]),
                         msg='funny stereo: baseline')

# I examine points somewhere in space. I make sure the rectification maps
# transform it properly. And I compute what its az,el and disparity would have
# been, and I check the geometric functions
pcam0 = np.array(((  1., 2., 10.),
                   (-4., 3., 10.)))

qcam0 = mrcal.project( pcam0, *model0.intrinsics() )

pcam1 = mrcal.transform_point_rt(mrcal.invert_rt(rt01), pcam0)
qcam1 = mrcal.project( pcam1, *model1.intrinsics() )


pstereo0 = mrcal.transform_point_Rt( mrcal.invert_Rt(Rt_cam0_stereo), pcam0)
el0 = np.arctan2(pstereo0[:,1], pstereo0[:,2])
az0 = np.arctan2(pstereo0[:,0], pstereo0[:,2] / np.cos(el0))

pstereo1 = pstereo0 - Rt01_rectified[3,:]
el1 = np.arctan2(pstereo1[:,1], pstereo1[:,2])
az1 = np.arctan2(pstereo1[:,0], pstereo1[:,2] / np.cos(el1))

Naz,Nel = models_rectified[0].imagersize()
az_row = (np.arange(Naz, dtype=float) - fxycxy[2]) / fxycxy[0]
el_col = (np.arange(Nel, dtype=float) - fxycxy[3]) / fxycxy[1]

rectification_maps = mrcal.rectification_maps((model0,model1),
                                              models_rectified)

interp_rectification_map0x = \
    scipy.interpolate.RectBivariateSpline(az_row, el_col,
                                          nps.transpose(rectification_maps[0][...,0]))
interp_rectification_map0y = \
    scipy.interpolate.RectBivariateSpline(az_row, el_col,
                                          nps.transpose(rectification_maps[0][...,1]))
interp_rectification_map1x = \
    scipy.interpolate.RectBivariateSpline(az_row, el_col,
                                          nps.transpose(rectification_maps[1][...,0]))
interp_rectification_map1y = \
    scipy.interpolate.RectBivariateSpline(az_row, el_col,
                                          nps.transpose(rectification_maps[1][...,1]))

qcam0_from_map = \
    nps.transpose( nps.cat( interp_rectification_map0x(az0,el0, grid=False),
                            interp_rectification_map0y(az0,el0, grid=False) ) )
qcam1_from_map = \
    nps.transpose( nps.cat( interp_rectification_map1x(az1,el1, grid=False),
                            interp_rectification_map1y(az1,el1, grid=False) ) )

testutils.confirm_equal( qcam0_from_map, qcam0,
                         eps=1e-1,
                         msg='rectification map for camera 0 points')
testutils.confirm_equal( qcam1_from_map, qcam1,
                         eps=1e-1,
                         msg='rectification map for camera 1 points')

# same point, so we should have the same el
testutils.confirm_equal( el0, el1,
                         msg='elevations of the same observed point match')

disparity = az0 - az1
r = mrcal.stereo_range( disparity * fxycxy[0],
                        models_rectified,
                        az_deg = az0 * 180./np.pi )

testutils.confirm_equal( r, nps.mag(pcam0),
                         msg=f'stereo_range reports the right thing')

testutils.finish()
