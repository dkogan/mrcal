#!/usr/bin/env python3

r'''Tests the parsing of corners.vnl data
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
import io


corners_all_or_none_noweight = r'''# filename x y
frame100-cam1.jpg 0 0
frame100-cam1.jpg 1 0
frame100-cam1.jpg 0 1
frame100-cam1.jpg 1 1
frame100-cam1.jpg 0 2
frame100-cam1.jpg 1 2
frame100-cam2.jpg 10 10
frame100-cam2.jpg 11 10
frame100-cam2.jpg 10 11
frame100-cam2.jpg 11 11
frame100-cam2.jpg 10 12
frame100-cam2.jpg 11 12
frame101-cam1.jpg 20 20
frame101-cam1.jpg 21 20
frame101-cam1.jpg 20 21
frame101-cam1.jpg 21 21
frame101-cam1.jpg 20 22
frame101-cam1.jpg 21 22
frame101-cam2.jpg 30 30
frame101-cam2.jpg 31 30
frame101-cam2.jpg 30 31
frame101-cam2.jpg 31 31
frame101-cam2.jpg 30 32
frame101-cam2.jpg 31 32
frame102-cam1.jpg - -
frame102-cam2.jpg 40 40
frame102-cam2.jpg 41 40
frame102-cam2.jpg 40 41
frame102-cam2.jpg 41 41
frame102-cam2.jpg 40 42
frame102-cam2.jpg 41 42
'''

observations_ref = np.empty((5,3,2,3), dtype=float)
observations_ref_x = observations_ref[:,:,:,0]
observations_ref_y = nps.transpose(observations_ref[:,:,:,1])
observations_ref_x[:] = np.arange(2)
observations_ref_y[:] = np.arange(3)

observations_ref_w = nps.clump(observations_ref[:,:,:,2], n=3)
observations_ref_w[:] = 1.0 # default weight

observations_ref_frame = nps.mv(observations_ref[..., :2], 0, -1)
observations_ref_frame += np.arange(5)*10

indices_frame_camera_ref = np.array(((0,0),
                                     (0,1),
                                     (1,0),
                                     (1,1),
                                     (2,1),
                                     ), dtype=np.int32)

paths_ref = ("frame100-cam1.jpg",
             "frame100-cam2.jpg",
             "frame101-cam1.jpg",
             "frame101-cam2.jpg",
             "frame102-cam2.jpg")

try:
    observations, indices_frame_camera, paths = \
        mrcal.compute_chessboard_corners(W                 = 2,
                                         H                 = 3,
                                         globs_per_camera  = ('frame*-cam1.jpg','frame*-cam2.jpg'),
                                         corners_cache_vnl = io.StringIO(corners_all_or_none_noweight),
                                         Npoint_observations_per_board_min = 3)
except Exception as e:
    observations         = f"Error: {e}"
    indices_frame_camera = f"Error: {e}"


testutils.confirm_equal( observations,
                         observations_ref,
                         msg = "observations: all-or-none-no-weight")
testutils.confirm_equal( indices_frame_camera,
                         indices_frame_camera_ref,
                         msg = "indices_frame_camera: all-or-none-no-weight")
testutils.confirm_equal( paths,
                         paths_ref,
                         msg = "paths: all-or-none-no-weight")

###############################################################

corners_all_or_none = r'''# filename x y weight
frame100-cam1.jpg 0 0   0.01
frame100-cam1.jpg 1 0   0.02
frame100-cam1.jpg 0 1   0.03
frame100-cam1.jpg 1 1   0.04
frame100-cam1.jpg 0 2   0.05
frame100-cam1.jpg 1 2   0.06
frame100-cam2.jpg 10 10 0.07
frame100-cam2.jpg 11 10 0.08
frame100-cam2.jpg 10 11 0.09
frame100-cam2.jpg 11 11 0.10
frame100-cam2.jpg 10 12 0.11
frame100-cam2.jpg 11 12 0.12
frame101-cam1.jpg 20 20 0.13
frame101-cam1.jpg 21 20 0.14
frame101-cam1.jpg 20 21 0.15
frame101-cam1.jpg 21 21 0.16
frame101-cam1.jpg 20 22 0.17
frame101-cam1.jpg 21 22 0.18
frame101-cam2.jpg 30 30 0.19
frame101-cam2.jpg 31 30 0.20
frame101-cam2.jpg 30 31 0.21
frame101-cam2.jpg 31 31 0.22
frame101-cam2.jpg 30 32 0.23
frame101-cam2.jpg 31 32 0.24
frame102-cam1.jpg - - -
frame102-cam2.jpg 40 40 0.25
frame102-cam2.jpg 41 40 0.26
frame102-cam2.jpg 40 41 -2
frame102-cam2.jpg 41 41 0.28
frame102-cam2.jpg 40 42 -
frame102-cam2.jpg 41 42 0.30
'''

observations_ref = np.empty((5,3,2,3), dtype=float)
observations_ref_x = observations_ref[:,:,:,0]
observations_ref_y = nps.transpose(observations_ref[:,:,:,1])
observations_ref_x[:] = np.arange(2)
observations_ref_y[:] = np.arange(3)

observations_ref_w = nps.clump(observations_ref[:,:,:,2], n=3)
observations_ref_w[:] = (np.arange(30) + 1) / 100
observations_ref[4,1,0,2] = -1.
observations_ref[4,2,0,2] = -1.

observations_ref_frame = nps.mv(observations_ref[..., :2], 0, -1)
observations_ref_frame += np.arange(5)*10

indices_frame_camera_ref = np.array(((0,0),
                                     (0,1),
                                     (1,0),
                                     (1,1),
                                     (2,1),
                                     ), dtype=np.int32)

paths_ref = ("frame100-cam1.jpg",
             "frame100-cam2.jpg",
             "frame101-cam1.jpg",
             "frame101-cam2.jpg",
             "frame102-cam2.jpg")

try:
    observations, indices_frame_camera, paths = \
        mrcal.compute_chessboard_corners(W                  = 2,
                                         H                  = 3,
                                         globs_per_camera   = ('frame*-cam1.jpg','frame*-cam2.jpg'),
                                         corners_cache_vnl  = io.StringIO(corners_all_or_none),
                                         weight_column_kind = 'weight',
                                         Npoint_observations_per_board_min = 3)

except Exception as e:
    observations         = f"Error: {e}"
    indices_frame_camera = f"Error: {e}"

testutils.confirm_equal( observations,
                         observations_ref,
                         msg = "observations: all-or-none")
testutils.confirm_equal( indices_frame_camera,
                         indices_frame_camera_ref,
                         msg = "indices_frame_camera: all-or-none")
testutils.confirm_equal( paths,
                         paths_ref,
                         msg = "paths: all-or-none")


##### same thing again, but with the default Npoint_observations_per_board_min.
##### Should cut out one of the observations
try:
    observations, indices_frame_camera, paths = \
        mrcal.compute_chessboard_corners(W                  = 2,
                                         H                  = 3,
                                         globs_per_camera   = ('frame*-cam1.jpg','frame*-cam2.jpg'),
                                         corners_cache_vnl  = io.StringIO(corners_all_or_none),
                                         weight_column_kind = 'weight')
except Exception as e:
    observations         = f"Error: {e}"
    indices_frame_camera = f"Error: {e}"
testutils.confirm( len(observations) < len(observations_ref),
                   msg = "observations: default Npoint_observations_per_board_min culls some observations")


###############################################################

corners_all_or_none_level = r'''# filename x y level
frame100-cam1.jpg 0 0   0
frame100-cam1.jpg 1 0   1
frame100-cam1.jpg 0 1   2
frame100-cam1.jpg 1 1   3
frame100-cam1.jpg 0 2   4
frame100-cam1.jpg 1 2   5
frame100-cam2.jpg 10 10 0
frame100-cam2.jpg 11 10 1
frame100-cam2.jpg 10 11 2
frame100-cam2.jpg 11 11 3
frame100-cam2.jpg 10 12 4
frame100-cam2.jpg 11 12 5
frame101-cam1.jpg 20 20 0
frame101-cam1.jpg 21 20 1
frame101-cam1.jpg 20 21 2
frame101-cam1.jpg 21 21 3
frame101-cam1.jpg 20 22 4
frame101-cam1.jpg 21 22 5
frame101-cam2.jpg 30 30 0
frame101-cam2.jpg 31 30 1
frame101-cam2.jpg 30 31 2
frame101-cam2.jpg 31 31 3
frame101-cam2.jpg 30 32 4
frame101-cam2.jpg 31 32 5
frame102-cam1.jpg - - -
frame102-cam2.jpg 40 40 0
frame102-cam2.jpg 41 40 1
frame102-cam2.jpg 40 41 -2
frame102-cam2.jpg 41 41 3
frame102-cam2.jpg 40 42 -
frame102-cam2.jpg 41 42 5
'''

observations_ref = np.empty((5,3,2,3), dtype=float)
observations_ref_x = observations_ref[:,:,:,0]
observations_ref_y = nps.transpose(observations_ref[:,:,:,1])
observations_ref_x[:] = np.arange(2)
observations_ref_y[:] = np.arange(3)

observations_ref_w = nps.clump(observations_ref[:,:,:,2], n=-2)
observations_ref_w[:] = np.power(2., -np.arange(6))
observations_ref[4,1,0,2] = -1.
observations_ref[4,2,0,2] = -1.

observations_ref_frame = nps.mv(observations_ref[..., :2], 0, -1)
observations_ref_frame += np.arange(5)*10

indices_frame_camera_ref = np.array(((0,0),
                                     (0,1),
                                     (1,0),
                                     (1,1),
                                     (2,1),
                                     ), dtype=np.int32)

paths_ref = ("frame100-cam1.jpg",
             "frame100-cam2.jpg",
             "frame101-cam1.jpg",
             "frame101-cam2.jpg",
             "frame102-cam2.jpg")

try:
    observations, indices_frame_camera, paths = \
        mrcal.compute_chessboard_corners(W                  = 2,
                                         H                  = 3,
                                         globs_per_camera   = ('frame*-cam1.jpg','frame*-cam2.jpg'),
                                         corners_cache_vnl  = io.StringIO(corners_all_or_none_level),
                                         weight_column_kind = 'level',
                                         Npoint_observations_per_board_min = 3)

except Exception as e:
    observations         = f"Error: {e}"
    indices_frame_camera = f"Error: {e}"

testutils.confirm_equal( observations,
                         observations_ref,
                         msg = "observations: all-or-none-level")
testutils.confirm_equal( indices_frame_camera,
                         indices_frame_camera_ref,
                         msg = "indices_frame_camera: all-or-none-level")
testutils.confirm_equal( paths,
                         paths_ref,
                         msg = "paths: all-or-none-level")

###############################################################

corners_complicated = r'''# filename x y weight
frame100-cam1.jpg 0 0   0.01
frame100-cam1.jpg 1 0   0.02
frame100-cam1.jpg 0 1   0.03
frame100-cam1.jpg 1 1   0.04
frame100-cam1.jpg 0 2   0.05
frame100-cam1.jpg 1 2   0.06
frame100-cam2.jpg 10 10 0.07
frame100-cam2.jpg 11 10 0.08
frame100-cam2.jpg 10 11 0.09
frame100-cam2.jpg 11 11 0.10
  # comments and whitespace shouldn't break stuff
frame100-cam2.jpg 10 12 0.11
frame100-cam2.jpg 11 12   0.12
frame101-cam1.jpg 20	20 0.13
frame101-cam1.jpg 21 20 0.14
# missing weight should default to 1.0
frame101-cam1.jpg 20 21
# weight of 0 means "ignore point"
frame101-cam1.jpg 21 21 0
frame101-cam1.jpg 20 22 0.17
frame101-cam1.jpg 21 22 0.18
frame101-cam2.jpg 30 30 0.19
frame101-cam2.jpg 31 30 0.20
frame101-cam2.jpg 30 31 0.21
frame101-cam2.jpg 31 31 0.22
frame101-cam2.jpg 30 32 0.23
frame101-cam2.jpg 31 32 0.24
frame102-cam1.jpg - - -
frame102-cam2.jpg 40 40 0.25
frame102-cam2.jpg 41 40 0.26
frame102-cam2.jpg - - -2
frame102-cam2.jpg 41 41 0.28
frame102-cam2.jpg - - -
frame102-cam2.jpg 41 42 0.30
'''

observations_ref = np.empty((5,3,2,3), dtype=float)
observations_ref_x = observations_ref[:,:,:,0]
observations_ref_y = nps.transpose(observations_ref[:,:,:,1])
observations_ref_x[:] = np.arange(2)
observations_ref_y[:] = np.arange(3)

observations_ref_frame = nps.mv(observations_ref[..., :2], 0, -1)
observations_ref_frame += np.arange(5)*10

observations_ref_w = nps.clump(observations_ref[:,:,:,2], n=3)
observations_ref_w[:] = (np.arange(30) + 1) / 100
observations_ref[4,1,0,:] = -1.
observations_ref[4,2,0,:] = -1.
observations_ref[2,1,0,2] = 1.0 # missing weight in the datafile: use default
observations_ref[2,1,1,2] = -1.0

indices_frame_camera_ref = np.array(((0,0),
                                     (0,1),
                                     (1,0),
                                     (1,1),
                                     (2,1),
                                     ), dtype=np.int32)

paths_ref = ("frame100-cam1.jpg",
             "frame100-cam2.jpg",
             "frame101-cam1.jpg",
             "frame101-cam2.jpg",
             "frame102-cam2.jpg")

try:
    observations, indices_frame_camera, paths = \
        mrcal.compute_chessboard_corners(W                  = 2,
                                         H                  = 3,
                                         globs_per_camera   = ('frame*-cam1.jpg','frame*-cam2.jpg'),
                                         corners_cache_vnl  = io.StringIO(corners_complicated),
                                         weight_column_kind = 'weight',
                                         Npoint_observations_per_board_min = 3)
except Exception as e:
    observations         = f"Error: {e}"
    indices_frame_camera = f"Error: {e}"

testutils.confirm_equal( observations,
                         observations_ref,
                         msg = "observations: complicated")
testutils.confirm_equal( indices_frame_camera,
                         indices_frame_camera_ref,
                         msg = "indices_frame_camera: complicated")
testutils.confirm_equal( paths,
                         paths_ref,
                         msg = "paths: complicated")

testutils.finish()
