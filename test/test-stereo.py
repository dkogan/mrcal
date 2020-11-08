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
import testutils


model0 = mrcal.cameramodel(f"{testdir}/data/cam0.opencv8.cameramodel")
model1 = mrcal.cameramodel(model0)

# I create geometries to test. First off, a vanilla geometry for left-right stereo
rt01 = np.array((0,0,0,  3.0, 0, 0))
model1.extrinsics_rt_toref( mrcal.compose_rt(model0.extrinsics_rt_toref(),
                                             rt01))

az_fov_deg = 90
el_fov_deg = 50
rectification_maps,cookie = \
    mrcal.stereo_rectify_prepare( (model0, model1),
                                  az_fov_deg = az_fov_deg,
                                  el_fov_deg = el_fov_deg,
                                  pixels_per_deg_az = -1./8.,
                                  pixels_per_deg_el = -1./4.)
Rt_cam0_stereo = cookie['Rt_cam0_stereo']

testutils.confirm_equal(Rt_cam0_stereo, mrcal.identity_Rt(),
                        msg='vanilla stereo has a vanilla geometry')

testutils.confirm_equal( cookie['baseline'],
                         nps.mag(rt01[3:]),
                         msg='vanilla stereo: baseline')

q0,q0x,q0y = mrcal.project( np.array(((0,      0,1.),
                                      (1e-6,   0,1.),
                                      (0,   1e-6, 1.))), *model0.intrinsics() )

testutils.confirm_equal(cookie['pixels_per_deg_az'] * 8.,
                        (q0x-q0)[0] / 1e-6 * np.pi/180.,
                        msg='vanilla stereo: correct az pixel density')

testutils.confirm_equal(cookie['pixels_per_deg_el'] * 4.,
                        (q0y-q0)[1] / 1e-6 * np.pi/180.,
                        msg='vanilla stereo: correct el pixel density')

testutils.confirm_equal(cookie['az_row'].ndim, 1,
                        msg='correct az shape')
Naz = cookie['az_row'].shape[0]

testutils.confirm_equal(cookie['el_col'].ndim, 2,
                        msg='correct el shape')
testutils.confirm_equal(cookie['el_col'].shape[-1], 1,
                        msg='correct el shape')
Nel = cookie['el_col'].shape[0]

testutils.confirm_equal( cookie['az_row'][-1] - cookie['az_row'][0],
                         az_fov_deg * np.pi/180.,
                         relative = True,
                         eps = 0.01,
                         msg='az_fov_deg')

testutils.confirm_equal( cookie['el_col'][-1,0] - cookie['el_col'][0,0],
                         el_fov_deg * np.pi/180.,
                         relative = True,
                         eps = 0.01,
                         msg='el_fov_deg')

testutils.confirm_equal( cookie['az_row'][1] - cookie['az_row'][0],
                         np.pi/180./cookie['pixels_per_deg_az'],
                         relative = True,
                         eps = 0.01,
                         msg='az spacing')

testutils.confirm_equal( cookie['el_col'][1] - cookie['el_col'][0],
                         np.pi/180./cookie['pixels_per_deg_el'],
                         relative = True,
                         eps = 0.01,
                         msg='el spacing')


# Weirder geometry. Left-right stereo, with sizeable rotation and position fuzz.
# I especially make sure there's a forward/back shift
rt01 = np.array((0.1, 0.2, 0.05,  3.0, 0.2, 1.0))
model1.extrinsics_rt_toref( mrcal.compose_rt(model0.extrinsics_rt_toref(),
                                             rt01))
rectification_maps,cookie = \
    mrcal.stereo_rectify_prepare( (model0, model1),
                                  az_fov_deg = az_fov_deg,
                                  el_fov_deg = el_fov_deg,
                                  pixels_per_deg_az = -1./8.,
                                  pixels_per_deg_el = -1./4.)
Rt_cam0_stereo = cookie['Rt_cam0_stereo']


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
                        np.array([[ 0.9467916 , -0.08583181, -0.31019116],
                                  [ 0.06311944,  0.99458609, -0.08254964],
                                  [ 0.3155972 ,  0.05857821,  0.94708342],
                                  [ 0.        ,  0.        ,  0.        ]]),
                        msg='funny stereo geometry')

testutils.confirm_equal( cookie['baseline'],
                         nps.mag(rt01[3:]),
                         msg='funny stereo: baseline')


# I examine points somewhere in space. I make sure the rectification maps
# transform it properly. And I compute what its az,el and disparity would have
# been, and I check the geometric functions
pcam0 = np.array(((  1., 2., 10.),
                  (-5.,  3., 10.)))
qcam0 = mrcal.project( pcam0, *model0.intrinsics() )

pcam1 = mrcal.transform_point_rt(mrcal.invert_rt(rt01), pcam0)
qcam1 = mrcal.project( pcam1, *model1.intrinsics() )


pstereo0 = mrcal.transform_point_Rt( mrcal.invert_Rt(Rt_cam0_stereo), pcam0)
el0 = np.arctan2(pstereo0[:,1], pstereo0[:,2])
az0 = np.arctan2(pstereo0[:,0], pstereo0[:,2] / np.cos(el0))

pstereo1 = pstereo0 - np.array((cookie['baseline'], 0, 0))
el1 = np.arctan2(pstereo1[:,1], pstereo1[:,2])
az1 = np.arctan2(pstereo1[:,0], pstereo1[:,2] / np.cos(el1))

disparity = az0 - az1

Naz = cookie['az_row'].shape[0]
Nel = cookie['el_col'].shape[0]
iaz0 = np.round(np.interp(az0, cookie['az_row'].ravel(), np.arange(Naz))).astype(int)
iel0 = np.round(np.interp(el0, cookie['el_col'].ravel(), np.arange(Nel))).astype(int)
iaz1 = np.round(np.interp(az1, cookie['az_row'].ravel(), np.arange(Naz))).astype(int)
iel1 = np.round(np.interp(el1, cookie['el_col'].ravel(), np.arange(Nel))).astype(int)

for i in range(len(pcam0)):
    qrect0 = rectification_maps[0][ iel0[i], iaz0[i] ]
    testutils.confirm_equal( qrect0, qcam0[i],
                             eps=8., # inexact because I round() above
                             msg=f'rectification map for camera 0 point {i}')

    qrect1 = rectification_maps[1][ iel1[i], iaz1[i] ]
    testutils.confirm_equal( qrect1, qcam1[i],
                             eps=8., # inexact because I round() above
                             msg=f'rectification map for camera 1 point {i}')

    # same point, so we should have the same el
    testutils.confirm_equal( el0[i], el1[i],
                             msg=f'elevations of the same observed point match')


r = mrcal.stereo_range( disparity * 180./np.pi * cookie['pixels_per_deg_az'],
                        az = az0,
                        **cookie )

testutils.confirm_equal( r, nps.mag(pcam0),
                         msg=f'stereo_range reports the right thing')

v0 = mrcal.stereo_unproject(az0, el0)
testutils.confirm_equal( v0, pstereo0/nps.dummy(nps.mag(pstereo0), -1),
                         msg=f'stereo_unproject reports the right vector')

v0,dv_dazel = mrcal.stereo_unproject(az0, el0, get_gradients = True)
testutils.confirm_equal( v0, pstereo0/nps.dummy(nps.mag(pstereo0), -1),
                         msg=f'stereo_unproject reports the right vector with get_gradients=True')

v0az = mrcal.stereo_unproject(az0 + 1e-6, el0)
v0el = mrcal.stereo_unproject(az0,        el0 + 1e-6)
testutils.confirm_equal( dv_dazel[...,0], (v0az - v0) / 1e-6,
                         msg=f'stereo_unproject az gradient')
testutils.confirm_equal( dv_dazel[...,1], (v0el - v0) / 1e-6,
                         msg=f'stereo_unproject el gradient')

punproj = mrcal.stereo_unproject(az0, el0,
                                 disparity_pixels = disparity * 180./np.pi * cookie['pixels_per_deg_az'],
                                 **cookie)
testutils.confirm_equal( punproj, pstereo0,
                         msg=f'stereo_unproject can parse the disparity')

testutils.finish()
