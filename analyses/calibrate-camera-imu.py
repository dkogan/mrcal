#!/usr/bin/python3

import sys
import numpy as np
import numpysane as nps
import gnuplotlib as gp
import mrcal
import glob
import vnlog
import os
import re
import scipy


# I have several measurements of stationary chessboards and the gravity vector
# from a stationary camera+IMU rig

# Unknowns:
# - rt_cam_cam0[i]   for each camera i (6*(Ncameras-1) DOF)
# - rt_cam0_board[j] at each time j    (6*Nsnapshots   DOF)
# - r_imu_cam0                         (3 DOF)
# - g_board                            (2 DOF)

# Measurements:
# - q_board[i,j]
# - g_imu[j]

# Cost function for each i,j:
# - Reprojection error q_board[i,j] - project(rt_cam_cam0[i] rt_cam0_board[j] p_board)
# - r_imu_cam0 r_cam0_board[j] g_board - g_imu[j]



# Write the results here
Dout = '/tmp'
Din  = '/tmp'

cameras = ('cam0', 'cam1', 'cam2',)
models  = [mrcal.cameramodel(f"{cam}.cameramodel") \
          for cam in cameras]

# The data is in vnl tables with the given columns. Each table is in a separate
# file, with the date/time ("YYYY-MM-DD-HH-MM-SS") in the filename identifying
# the instant in time when that data was captured. Sensor readings with the same
# tag can be assumed to have been gathered at the same instant in time. The INS
# data ends in "-gravity", the chessboard observation files have the camera name
# at the end of the filename instead.
dtype_gravity = np.dtype( [('acc0 acc1 acc2', float,(3,))])
dtype_corners = np.dtype( [('x y',            float,(2,))])
tag_regex = '(20[0-9][0-9]-[0-9][0-9]-[0-9][0-9]-[0-9][0-9]-[0-9][0-9]-[0-9][0-9])'
glob_files_gravity = f'{Din}/*-gravity.vnl'
glob_files_corners = [f'{Din}/*-{cam}.vnl' for cam in cameras]

gridn                          = 14
object_spacing                 = 0.0588
imu_err_scale__pixels_per_m_s2 = 0.3/0.05



files_gravity         = sorted(glob.glob(glob_files_gravity))
files_corners_allcams = [sorted(glob.glob(g)) for g in glob_files_corners]


def tag_from_file(f):
    m = re.search(tag_regex,f)
    return m.group(1)

_isnapshot_from_tag = dict()
for i,f in enumerate(files_gravity):
    _isnapshot_from_tag[ tag_from_file(f) ] = i

def isnapshot_from_file(f):
    return _isnapshot_from_tag[tag_from_file(f)]


Nsnapshots = len(files_gravity)
g_imu = np.zeros( (Nsnapshots,3), dtype=float)
for f in files_gravity:
    g = vnlog.slurp(f, dtype=dtype_gravity)['acc0 acc1 acc2']
    g_imu[isnapshot_from_file(f)] = g
g_imu_mag = nps.mag(g_imu)
gunit_imu = g_imu / nps.dummy( g_imu_mag, -1)

Ncameras = len(cameras)
icam_from_cam = dict()
for i,c in enumerate(cameras):
    icam_from_cam[c] = i


Nobservations = sum(len(files_corners) for files_corners in files_corners_allcams)

# qx,qy will be filled in later; weight=1 is hard-coded here now
observations_qxqyw   = np.ones ( (Nobservations,gridn,gridn,3), dtype=float)
indices_frame_camera = np.zeros( (Nobservations,2), dtype=np.int32 )
i = 0
for icam in range(Ncameras):
    for f in files_corners_allcams[icam]:
        isnapshot = isnapshot_from_file(f)
        observations_qxqyw[i,...,:2] = vnlog.slurp(f, dtype=dtype_corners)['x y'].reshape((gridn,gridn,2))
        indices_frame_camera[i,0] = isnapshot
        indices_frame_camera[i,1] = icam
        i += 1


######## Seed
# I need to estimate each of the state:
# - rt_cam_cam0[i]   for each camera i (6*(Ncameras-1) DOF)
# - rt_cam0_board[j] at each time j    (6*Nsnapshots   DOF)
# - r_imu_cam0                         (3 DOF)
# - g_board                            (2 DOF)

# For each observation, estimate the cam-board transform. This assumes a nearby
# pinhole model, so it's a rough estimate. I take arbitrary subsets of these to
# get rt_cam_cam0 and rt_cam0_board
Rt_cam_board_all = \
    mrcal.estimate_monocular_calobject_poses_Rt_tocam( indices_frame_camera,
                                                       observations_qxqyw,
                                                       object_spacing,
                                                       models)

rt_cam_cam0   = np.zeros( (Ncameras-1, 6), dtype=float)
rt_cam0_board = np.zeros( (Nsnapshots,6),  dtype=float)
Nrt_cam_cam0_have   = 0
Nrt_cam0_board_have = 0

# I go through to accumulate my seed.
#### WARNING: THIS DOES NOT WORK IN GENERAL
#### HERE I'M ASSUMING THAT CAMERA0 IS IN EVERY SNAPSHOT
for iobservation in range(Nobservations):
    Rt_cam_board   = Rt_cam_board_all[iobservation]
    isnapshot,icam = indices_frame_camera[iobservation]
    if icam==0 and not np.any(rt_cam0_board[isnapshot]):
        rt_cam0_board[isnapshot] = mrcal.rt_from_Rt(Rt_cam_board)
        Nrt_cam0_board_have += 1
        if Nrt_cam0_board_have == len(rt_cam0_board):
            break
if Nrt_cam0_board_have != len(rt_cam0_board):
    raise Exception("ERROR: did not init all of rt_cam0_board")

for iobservation in range(Nobservations):
    Rt_cam_board   = Rt_cam_board_all[iobservation]
    isnapshot,icam = indices_frame_camera[iobservation]
    if icam!=0 and not np.any(rt_cam_cam0[icam-1]):
        rt_cam_cam0[icam-1] = \
            mrcal.compose_rt(mrcal.rt_from_Rt(Rt_cam_board),
                             rt_cam0_board[isnapshot],
                             inverted1 = True)
        Nrt_cam_cam0_have += 1
        if Nrt_cam_cam0_have == len(rt_cam_cam0):
            break
if Nrt_cam_cam0_have != len(rt_cam_cam0):
    raise Exception("ERROR: did not init all of rt_cam_cam0")

######## Solve

def unpack_state(b, have_imu = False):
    Nstate_cameras = (Ncameras-1 + Nsnapshots) * 6
    bcameras = b[:Nstate_cameras].reshape( (Ncameras-1 + Nsnapshots, 6) )
    rt_cam_cam0   = bcameras[:Ncameras-1,:]
    rt_cam0_board = bcameras[Ncameras-1:,:]

    if not have_imu:
        return rt_cam_cam0,rt_cam0_board

    bimu = b[Nstate_cameras:]
    r_imu_cam0 = bimu[:3]

    a,b = bimu[3:]
    gunit_board = np.array(( np.sin(a),
                             np.cos(a)*np.cos(b),
                             np.cos(a)*np.sin(b),))
    return rt_cam_cam0,rt_cam0_board,r_imu_cam0,gunit_board

def pack_state(rt_cam_cam0, rt_cam0_board,
               r_imu_cam0 = None,
               gunit_board = None):
    if r_imu_cam0 is None:
        return \
            nps.cat( *rt_cam_cam0, *rt_cam0_board ).ravel()

    # have_imu
    a = np.arcsin( gunit_board[0])
    c = np.cos(a)
    if np.abs(c) < 1e-8:
        b = 0
    else:
        b = np.arctan2(gunit_board[2]/c,
                       gunit_board[1]/c,)
    return \
        nps.glue( rt_cam_cam0.ravel(),
                  rt_cam0_board.ravel(),
                  r_imu_cam0.ravel(),
                  a,
                  b,
                  axis = -1)

def optimizer_callback_camera(b, have_imu = False, debug = False):
    if not have_imu:
        rt_cam_cam0,rt_cam0_board = unpack_state(b, have_imu = have_imu)
    else:
        rt_cam_cam0,rt_cam0_board,r_imu_cam0,gunit_board = unpack_state(b, have_imu = have_imu)

    x_corners = np.zeros((Nobservations,gridn,gridn,2), dtype=float)

    for iobservation in range(Nobservations):
        isnapshot,icam = indices_frame_camera[iobservation]
        q_observed = observations_qxqyw[iobservation, ..., :2]

        pcam = mrcal.transform_point_rt( rt_cam0_board[isnapshot],
                                         calobject )
        if icam > 0:
            pcam = mrcal.transform_point_rt( rt_cam_cam0[icam-1],
                                             pcam )
        q = mrcal.project(pcam, *models[icam].intrinsics())
        x_corners[iobservation,...] = q - q_observed

    if debug:
        print(f"Camera solve callback RMS error: {np.sqrt(np.mean(nps.norm2(x_corners).ravel()))}")

    if not have_imu:
        return x_corners.ravel()

    gunit_cam0      = mrcal.rotate_point_r(rt_cam0_board[:,:3], gunit_board)
    gunit_imu_solve = mrcal.rotate_point_r(r_imu_cam0, gunit_cam0)
    g_imu_solve = gunit_imu_solve * 9.8
    x_imu = g_imu_solve - g_imu
    if debug:
        print(f"IMU vector error: {x_imu}m/s^2")

    return nps.glue(x_corners.ravel(),
                    x_imu.ravel() * imu_err_scale__pixels_per_m_s2,
                    axis = -1)


###### I solve the camera stuff by itself initially, since that should fit very
###### well
calobject = mrcal.ref_calibration_object(gridn, gridn,
                                         object_spacing)

# state vector at the seed
b0 = pack_state(rt_cam_cam0, rt_cam0_board)
result = scipy.optimize.least_squares(optimizer_callback_camera, b0,
                                      kwargs=dict(have_imu = False))

# if True:
#     print("At the solution:")
#     optimizer_callback_camera(result['x'], debug = True)
#     mrcal.show_geometry( nps.glue( mrcal.identity_rt(),
#                                    rt_cam_cam0,
#                                    axis = -2),
#                          wait = True)


###### Now do a bigger optimization, including the IMU stuff
rt_cam_cam0,rt_cam0_board = unpack_state(result['x'])

####### For now I assume the board is sitting roughly vertically, and seed off
####### that. THIS IS NOT TRUE IN GENERAL
gunit_board = np.array((0., -1., 0.))
gunit_cam0 = mrcal.rotate_point_r(rt_cam0_board[:,:3], gunit_board)
R_imu_cam0 = mrcal.align_procrustes_vectors_R01(gunit_imu, gunit_cam0)
r_imu_cam0 = mrcal.r_from_R(R_imu_cam0)
gunit_imu_seed = mrcal.rotate_point_R(R_imu_cam0, gunit_cam0)
th_err_imu_seed = np.arccos(nps.inner(gunit_imu_seed, gunit_imu))
if False:
    # should be "small"
    print(th_err_imu_seed)

b0 = pack_state(rt_cam_cam0, rt_cam0_board, r_imu_cam0, gunit_board)
result = scipy.optimize.least_squares(optimizer_callback_camera, b0,
                                      kwargs=dict(have_imu = True))
rt_cam_cam0,rt_cam0_board,r_imu_cam0,gunit_board = unpack_state(result['x'], have_imu = True)

rt_cam_cam0_mounted = \
    nps.glue( mrcal.identity_rt(),
              rt_cam_cam0,
              axis = -2)
rt_imu_cam0 = nps.glue( r_imu_cam0, np.zeros((3,)),
                        axis = -1)

for icam in range(Ncameras):
    cam = cameras[icam]
    rt_cam_cam0_this = rt_cam_cam0_mounted[icam]

    filename = f"{Dout}/{cam}-final.cameramodel"
    m = mrcal.cameramodel(models[icam])
    m.rt_cam_ref( rt_cam_cam0_this )
    m.write(filename)
    print(f"Wrote '{filename}'")
# And a dummy imu model; for visualization only
filename = f"{Dout}/imu-final.cameramodel"
m = mrcal.cameramodel(rt_cam_ref = rt_imu_cam0,
                      # dummy
                      intrinsics = ('LENSMODEL_PINHOLE', np.array((1,1,0,0))),
                      imagersize = (1,1) )
m.write(filename)
print(f"Wrote '{filename}'")

if True:
    print("At the solution:")
    optimizer_callback_camera(result['x'],
                              have_imu = True,
                              debug = True)

    gunit_cam0  = mrcal.rotate_point_r(r_imu_cam0,          gunit_imu,  inverted=True)
    gunit_board = mrcal.rotate_point_r(rt_cam0_board[:,:3], gunit_cam0, inverted=True)
    print(f"guinit_board=\n{gunit_board}")

    mrcal.show_geometry( nps.glue( rt_cam_cam0_mounted,
                                   rt_imu_cam0,
                                   axis = -2),
                         cameranames = cameras + ['imu'],
                         wait = True)

