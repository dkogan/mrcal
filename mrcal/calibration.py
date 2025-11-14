#!/usr/bin/python3

# Copyright (c) 2017-2023 California Institute of Technology ("Caltech"). U.S.
# Government sponsorship acknowledged. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

'''Helper routines for seeding and solving camera calibration problems

All functions are exported into the mrcal module. So you can call these via
mrcal.calibration.fff() or mrcal.fff(). The latter is preferred.

'''

import numpy as np
import numpysane as nps
import sys
import re
import mrcal

def compute_chessboard_corners(W, H,
                               *,
                               globs_per_camera  = ('*',),
                               corners_cache_vnl = None,
                               image_path_prefix = None,
                               image_directory   = None,

                               jobs              = 1,
                               exclude_images    = set(),
                               weight_column_kind= 'level',
                               Npoint_observations_per_board_min = None):
    r'''Compute the chessboard observations and returns them in a usable form

SYNOPSIS

    observations, indices_frame_camera, paths = \
        mrcal.compute_chessboard_corners(10, 10,
                                         globs_per_camera  = ('frame*-cam0.jpg','frame*-cam1.jpg'),
                                         corners_cache_vnl = "corners.vnl")

The input to a calibration problem is a set of images of a calibration object
from different angles and positions. This function ingests these images, and
outputs the detected chessboard corner coordinates in a form usable by the mrcal
optimization routines.

The "corners_cache_vnl" argument specifies a file containing cached results of
the chessboard detector. If this file already exists, we don't run the detector,
but just use the contents of the file. Otherwise, we run the detector, and store
the results here.

The "corner cache" file is a vnlog 3 or 4 columns. Each row describes a
chessboard corner. The first 3 columns are

    # filename x y

If the 4th column is given, it usually is a 'level' or a 'weight'. It encodes
the confidence we have in that corner, and the exact interpretation is dependent
on the value of the 'weight_column_kind' argument. The output of this function
is an array with a weight for each point, so the logic serves to convert the
extra column to a weight.

if weight_column_kind == 'level': the 4th column is a decimation level of the
  detected corner. If we needed to cut down the image resolution to detect a
  corner, its coordinates are known less precisely, and we use that information
  to weight the errors appropriately later. We set the output weight =
  1/2^level. If the 4th column is '-' or <0, the given point was not detected,
  and should be ignored: we set weight = -1

elif weight_column_kind == 'weight': the 4th column is already represented as a
  weight, so I just copy it to the output. If the 4th column is '-' or <=0, the
  given point was not detected, and should be ignored: we set weight = -1

elif weight_column_kind is None: I hard-code the output weight to 1.0. Any data
  in the extra column is ignored

ARGUMENTS

- W, H: the width and height of the point grid in the calibration object we're
  using

- globs_per_camera: a list of strings, one per camera, containing
  globs_per_camera matching the image filenames for that camera. The filenames
  are expected to encode the instantaneous frame numbers, with identical frame
  numbers implying synchronized images. A common scheme is to name an image
  taken by frame C at time T "frameT-camC.jpg". Then images frame10-cam0.jpg and
  frame10-cam1.jpg are assumed to have been captured at the same moment in time
  by cameras 0 and 1. With this scheme, if you wanted to calibrate these two
  cameras together, you'd pass ('frame*-cam0.jpg','frame*-cam1.jpg') in the
  "globs_per_camera" argument.

  The "globs_per_camera" argument may be omitted. In this case all images are
  mapped to the same camera.

- corners_cache_vnl: the name of a file to use to read/write the detected
  corners; or a python file object to read data from. If the given file exists
  or a python file object is given, we read the detections from it, and do not
  run the detector. If the given file does NOT exist (which is what happens the
  first time), mrgingham will be invoked to compute the corners from the images,
  and the results will be written to that file. So the same function call can be
  used to both compute the corners initially, and to reuse the pre-computed
  corners with subsequent calls. This exists to save time where re-analyzing the
  same data multiple times.

  When the corners_cache_vnl exists, and refers images that have existed on
  disk, some logic is needed to adjust the paths in corners_cache_vnl. Because
  those paths don't need to exist anymore and because the corners_cache_vnl file
  and/or the images may have moved since they were detected. The logic is:

    if image_path_prefix is given: we use this as a prefix to the paths in
      corners_cache_vnl
    elif image_directory is given: we strip out the directory from the paths in
      corners_cache_vnl, and use image_directory instead
    elif we're reading from a corners_cache_vnl file on disk and
         the path in corners_cache_vnl is relative and
         this path does not exist relative to the current directory:
      use the directory of the corners_cache_vnl as the prefix
    else:
      use the path in corners_cache_vnl as is

  The above logic is complex to cover the various possible case. The canonical
  case is to put the corners.vnl file into the same directory as the images, to
  put the filenames only into the corners.vnl, and to not have any images in the
  current directory. If you want to be extra sure about what it's doing, pass
  image_path_prefix or image_directory

- image_path_prefix: optional argument, defaulting to None, exclusive with
  "image_directory". Used to locate the images when reading from a given
  corners_cache_vnl. If given, the image paths in the "paths" argument are
  prefixed with the given string. See docs for corners_cache_vnl above

- image_directory: optional argument, defaulting to None, exclusive with
  "image_path_prefix". Used to locate the images when reading from a given
  corners_cache_vnl. If given, we extract the filename from the image path in
  the "paths" argument, and look for the images in the directory given here
  instead. See docs for corners_cache_vnl above

- jobs: a GNU-Make style parallelization flag. Indicates how many parallel
  processes should be invoked when computing the corners. If given, a numerical
  argument is required. If jobs<0: the corners_cache_vnl MUST already contain
  valid data; if it doesn't, an exception is thrown instead of the corners being
  recomputed.

- exclude_images: a set of filenames to exclude from reported results

- weight_column_kind: an optional argument, defaulting to 'level'. Selects the
  interpretation of the 4th column describing each corner. Valid options are:

  - 'level': the 4th column is a decimation level. Level-0 means
    'full-resolution', level-1 means 'half-resolution' and so on. I set output
    weight = 1/2^level. If the 4th column is '-' or <0, the given point was not
    detected, and should be ignored: we set output weight = -1
  - 'weight': the 4th column is already a weight; I copy it to the output. If
    the 4th column is '-' or <0, the given point was not detected, and should be
    ignored: we set output weight = -1
  - None: the 4th column should be ignored, and I set the output weight to 1.0

- Npoint_observations_per_board_min: optional integer, defaulting to 2*min(W,H).
  We throw out any images where a chessboard detection has fewer than this many
  points.

RETURNED VALUES

This function returns a tuple (observations, indices_frame_camera, files_sorted)

- observations: an ordered (N,object_height_n,object_width_n,3) array describing
  N board observations where the board has dimensions
  (object_height_n,object_width_n) and each point is an (x,y,weight) pixel
  observation. A weight<0 means "ignore this point". Incomplete chessboard
  observations can be specified in this way.

- indices_frame_camera is an (N,2) array of contiguous, sorted integers where
  each observation is (index_frame,index_camera)

- files_sorted is a list of paths of images corresponding to the observations

Note that this assumes we're solving a calibration problem (stationary cameras)
observing a moving object, so this returns indices_frame_camera. It is the
caller's job to convert this into indices_frame_camintrinsics_camextrinsics,
which mrcal.optimize() expects

    '''

    if not (weight_column_kind == 'level' or
            weight_column_kind == 'weight' or
            weight_column_kind is None):
        raise Exception(f"weight_column_kind must be one of ('level','weight',None); got '{weight_column_kind}'")

    if Npoint_observations_per_board_min is None:
        Npoint_observations_per_board_min = 2*min(W,H)


    import os
    import fnmatch
    import subprocess
    import shutil
    from tempfile import mkstemp
    import io
    import copy

    def get_corner_observations(W, H, globs_per_camera, corners_cache_vnl, exclude_images=set()):
        r'''Return dot observations, from a cache or from mrgingham

        Returns a dict mapping from filename to a numpy array with a full grid
        of dot observations. If no grid was observed in a particular image, the
        relevant dict entry is empty

        The corners_cache_vnl argument is for caching corner-finder results.
        This can be None if we want to ignore this. Otherwise, this is treated
        as a path to a file on disk or a python file object. If this file
        exists:

            The corner coordinates are read from this file instead of being
            computed. We don't need to actually have the images stored on disk.
            Any image filenames mentioned in this cache file are matched against
            the globs_per_camera to decide which camera the image belongs to. If
            it matches none of the globs_per_camera, that image filename is
            silently ignored

        If this file does not exist:

            We process the images to compute the corner coordinates. Before we
            compute the calibration off these coordinates, we create the cache
            file and store this data there. Thus a subsequent identical
            invocation of mrcal-calibrate-cameras will see this file as
            existing, and will automatically use the data it contains instead of
            recomputing the corner coordinates

        '''

        # Expand any ./ and // etc
        globs_per_camera = [os.path.normpath(g) for g in globs_per_camera]

        Ncameras = len(globs_per_camera)
        files_per_camera = []
        for i in range(Ncameras):
            files_per_camera.append([])

        corners_dir = None

        fd_already_opened = isinstance(corners_cache_vnl, io.IOBase)

        pipe_corners_write_fd          = None
        pipe_corners_write_tmpfilename = None

        if corners_cache_vnl is not None and not fd_already_opened:
            if os.path.isdir(corners_cache_vnl):
                raise Exception(f"Given cache path '{corners_cache_vnl}' is a directory. Must be a file or must not exist")
            corners_dir = os.path.dirname( corners_cache_vnl )

        if corners_cache_vnl is not None and \
           ( fd_already_opened or \
             os.path.exists(corners_cache_vnl) ):

            # Have an existing cache file. Just read it
            if fd_already_opened:
                pipe_corners_read = corners_cache_vnl
            else:
                pipe_corners_read = open(corners_cache_vnl, 'r', encoding='ascii')
            corners_output    = None

        else:

            if jobs < 0:
                raise Exception("I was asked to use an existing cache file, but it couldn't be read. jobs<0, so I do not recompute")

            # Need to compute the dot coords. And maybe need to save them into a
            # cache file too
            if W != H:
                if corners_cache_vnl is None:
                    raise Exception(f"No corners cache file given, so we need to run mrgingham to compute them. mrgingham currently accepts ONLY square grids, but we were asked for a {W}x{H} grid")
                else:
                    raise Exception(f"Requested corners cache file '{corners_cache_vnl}' doesn't exist, so we need to run mrgingham to compute them. mrgingham currently accepts ONLY square grids, but we were asked for a {W}x{H} grid")

            if weight_column_kind == 'weight':
                raise Exception("Need to run mrgingham, so I will get a column of decimation levels, but weight_column_kind == 'weight'")

            args_mrgingham = ['mrgingham', '--jobs', str(jobs),
                              '--gridn', str(W)]
            args_mrgingham.extend(globs_per_camera)

            sys.stderr.write("Computing chessboard corners by running:\n   {}\n". \
                             format(' '.join(mrcal.shellquote(s) for s in args_mrgingham)))
            if corners_cache_vnl is not None:
                # need to save the corners into a cache. I want to do this
                # atomically: if the dot-finding is interrupted I don't want to
                # be writing incomplete results, so I write to a temporary file
                # and then rename when done
                pipe_corners_write_fd,pipe_corners_write_tmpfilename = mkstemp('.vnl')
                sys.stderr.write(f"Will save corners to '{corners_cache_vnl}'\n")

            corners_output = subprocess.Popen(args_mrgingham, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                              encoding='ascii')
            pipe_corners_read = corners_output.stdout


        mapping = {}
        context0 = dict(f            = '',
                        igrid        = 0,
                        Nvalidpoints = 0)

        # Init the array to -1.0: everything is invalid
        context0['grid'] = -np.ones( (H*W,3), dtype=float)

        context = copy.deepcopy(context0)

        # relative-path globs_per_camera: add explicit "*/" to the start
        globs_per_camera = [g if g[0]=='/' else '*/'+g for g in globs_per_camera]

        def finish_chessboard_observation():
            nonlocal context

            def accum_files(f):
                for icam in range(Ncameras):
                    if fnmatch.fnmatch(os.path.abspath(f), globs_per_camera[icam]):
                        files_per_camera[icam].append(f)
                        return True
                return False

            if context['igrid'] > 1:
                if W*H != context['igrid']:
                    raise Exception("File '{}' expected to have {} points, but got {}". \
                                    format(context['f'], W*H, context['igrid']))
                if context['f'] not in exclude_images and \
                   context['Nvalidpoints'] > Npoint_observations_per_board_min:

                    if image_path_prefix is not None:
                        filename_canonical = f"{image_path_prefix}/{context['f']}"
                    elif image_directory is not None:
                        filename_canonical = f"{image_directory}/{os.path.basename(context['f'])}"
                    elif corners_dir is not None      and \
                         context['f'][0] != '/'       and \
                         not os.path.exists(context['f']):
                        filename_canonical = f"{corners_dir}/{context['f']}"
                    else:
                        filename_canonical = os.path.normpath(context['f'])

                    if accum_files(filename_canonical):
                        mapping[filename_canonical] = context['grid'].reshape(H,W,3)

            context = copy.deepcopy(context0)

        for line in pipe_corners_read:
            if pipe_corners_write_fd is not None:
                os.write(pipe_corners_write_fd, line.encode())

            if re.match(r'\s*#', line):
                continue
            m = re.match(r'\s*(\S+)\s+(.*?)$', line)
            if m is None:
                raise Exception(f"Unexpected line in the corners output: '{line}'")
            if context['f'] != m.group(1):
                # Got data for the next image. Finish out this one
                finish_chessboard_observation()
                context['f'] = m.group(1)

            # The row may have 2 or 3 values: if 3, it contains a decimation
            # level of the corner observation (used for the weight). If 2, a
            # weight of 1.0 is assumed. The weight array is pre-filled with 1.0.
            # A decimation level of - will be used to set weight <0 which means
            # "ignore this point"
            fields = m.group(2).split()
            if not (len(fields) == 2 or len(fields) == 3):
                raise Exception(f"'corners.vnl' data rows must contain a filename and 2 or 3 values. Instead got line '{line}'")


            def parse_point(out, fields):
                r'''Parse the given split-string fields

                Write results into out[:]. Return True if the given point is
                valid. The initial value of out[:] is (-1,-1,-1): "invalid"

                '''
                if fields[0] == '-' or fields[1] == '-':
                    # out[:] already is invalid
                    return False

                try:
                    out[0] = float(fields[0])
                    out[1] = float(fields[1])
                except:
                    raise Exception(f"'corners.vnl' data rows must lead with 'filename x y' with x and y being numerical or '-'. Instead got line '{line}'")

                if out[0] < 0 or out[1] < 0:
                    # out[:] already is invalid
                    return False

                if len(fields) == 2 or weight_column_kind is None:
                    # default weight
                    out[2] = 1.0
                    return True

                if fields[2] == '-':
                    # out[:] already is invalid
                    return False

                try:
                    w = float(fields[2])
                except:
                    raise Exception(f"'corners.vnl' data rows expected as 'filename x y {weight_column_kind}' with {weight_column_kind} being numerical or '-'. Instead got line '{line}'")

                if weight_column_kind == 'weight':
                    if w <= 0.0:
                        # out[:] already is invalid
                        return False
                    out[2] = w
                else:
                    if w < 0.0:
                        # out[:] already is invalid
                        return False
                    # convert decimation level to weight. The weight is
                    # 2^(-level). I.e. level-0 -> weight=1, level-1 ->
                    # weight=0.5, etc
                    out[2] = 1. / (1 << int(w))

                return True


            if parse_point(context['grid'][context['igrid'],:],
                              fields):
                context['Nvalidpoints'] += 1

            context['igrid'] += 1

        finish_chessboard_observation()

        if corners_output is not None:
            sys.stderr.write("Done computing chessboard corners\n")

            if corners_output.wait() != 0:
                err = corners_output.stderr.read()
                raise Exception(f"mrgingham failed: {err}")
            if pipe_corners_write_fd is not None:
                os.close(pipe_corners_write_fd)
                shutil.move(pipe_corners_write_tmpfilename, corners_cache_vnl)
        elif not fd_already_opened:
            pipe_corners_read.close()

        # If I have multiple cameras, I use the filenames to figure out what
        # indexes the frame and what indexes the camera, so I need at least
        # two images for each camera to figure that out. Example:
        #
        #   I have two cameras, with one image each:
        #   - frame2-cam0.jpg
        #   - frame3-cam1.jpg
        #
        # If this is all I had, it'd be impossible for me to tell whether
        # the images correspond to the same frame or not. But if cam0 also
        # had "frame4-cam0.jpg" then I could look at the same-camera cam0
        # filenames, find the common prefixes,suffixes, and conclude that
        # the frame indices are 2 and 4.
        #
        # If I only have one camera, however, then the details of the
        # filenames don't matter, and I just make sure I have at least one
        # image to look at
        min_num_images = 2 if len(files_per_camera) > 1 else 1
        for icam in range(len(files_per_camera)):
            N = len(files_per_camera[icam])

            if N < min_num_images:
                raise Exception("Found too few ({}; need at least {}) images containing a calibration pattern in camera {}; glob '{}'". \
                                format(N, min_num_images, icam, globs_per_camera[icam]))

        return mapping,files_per_camera


    indices_frame_camera = np.array((), dtype=np.int32)
    observations         = np.array((), dtype=float)

    # basic logic is this:
    #   for frames:
    #       for cameras:
    #           if have observation:
    #               push observations
    #               push indices_frame_camera
    mapping_file_corners,files_per_camera = get_corner_observations(W, H, globs_per_camera, corners_cache_vnl, exclude_images)
    file_framenocameraindex               = mrcal.mapping_file_framenocameraindex(*files_per_camera)

    # I create a file list sorted by frame and then camera. So my for(frames)
    # {for(cameras) {}} loop will just end up looking at these files in order
    files_sorted = sorted(mapping_file_corners.keys(), key=lambda f: file_framenocameraindex[f][1])
    files_sorted = sorted(files_sorted,                key=lambda f: file_framenocameraindex[f][0])

    i_observation = 0

    iframe_last = None
    index_frame  = -1
    for f in files_sorted:
        # The frame indices I return are consecutive starting from 0, NOT the
        # original frame numbers
        iframe,icam = file_framenocameraindex[f]
        if iframe_last == None or iframe_last != iframe:
            index_frame += 1
            iframe_last = iframe

        indices_frame_camera = nps.glue(indices_frame_camera,
                                        np.array((index_frame, icam), dtype=np.int32),
                                        axis=-2)
        observations = nps.glue(observations,
                                mapping_file_corners[f],
                                axis=-4)

    return observations, indices_frame_camera, files_sorted


def _estimate_camera_pose_from_fixed_point_observations(lensmodel,
                                                        intrinsics_data,
                                                        observation_qxqyw,
                                                        points_ref,
                                                        what):
    r'''Wrapper around solvePnP that tries out different focal lengths

This is an ugly hack. The opencv solvePnP() function I'm using here assumes a
pinhole model. But this function accepts stereographic data, so observations
could be really wide; behind the camera even. I can't do much behind the camera,
but I can accept wide observations by using a much smaller pinhole focal length
than the stereographic one the user passed. This really shouldn't be hard-coded,
and I should only adjust if observations would be thrown away. And REALLY I
should be using a flavor of solvePnP that uses observation vectors instead of
pinhole pixel observations. Like opengv

    '''

    import cv2

    class SolvePnPerror_negz(Exception):
        def __init__(self, err): self.err = err
        def __str__(self):       return self.err
    class SolvePnPerror_toofew(Exception):
        def __init__(self, err): self.err = err
        def __str__(self):       return self.err

    def solvepnp__try_focal_scale(scale):

        fx,fy,cx,cy = intrinsics_data[:4]
        fxy         = intrinsics_data[0:2]
        cxy         = intrinsics_data[2:4]
        camera_matrix = \
            np.array(((fx*scale, 0,        cx),
                      ( 0,       fy*scale, cy),
                      ( 0,       0,        1)),
                     dtype=float)

        v = mrcal.unproject((observation_qxqyw[..., :2] - cxy) / scale + cxy,
                            lensmodel,
                            intrinsics_data)
        observation_qxqy_pinhole = \
            mrcal.project(v,
                          'LENSMODEL_PINHOLE',
                          intrinsics_data[:4])
        # observation_qxqy_pinhole = (observation_qxqy_pinhole - cxy)*scale + cxy
        observation_qxqy_pinhole *= scale
        observation_qxqy_pinhole += cxy*(1. - scale)

        # I pick off those rows where the point observation is valid
        i = \
            (observation_qxqyw[..., 2] > 0.0)        * \
            (np.isfinite(v                       [..., 0])) * \
            (np.isfinite(v                       [..., 1])) * \
            (np.isfinite(v                       [..., 2])) * \
            (np.isfinite(observation_qxqy_pinhole[..., 0])) * \
            (np.isfinite(observation_qxqy_pinhole[..., 1]))

        if np.count_nonzero(i) < 6:
            raise SolvePnPerror_toofew(f"Insufficient observations; need at least 6; got {np.count_nonzero(i)} instead. Cannot estimate initial extrinsics for {what}")

        observations_local = observation_qxqy_pinhole[i]
        ref_object         = points_ref[i]
        result,rvec,tvec   = cv2.solvePnP(ref_object,
                                          observations_local,
                                          camera_matrix,
                                          None)
        if not result:
            raise Exception(f"solvePnP() failed! Cannot estimate initial extrinsics for {what}")
        if tvec[2] <= 0:

            # The object ended up behind the camera. I flip it, and try to solve
            # again
            result,rvec,tvec = cv2.solvePnP(np.array(ref_object),
                                            np.array(observations_local),
                                            camera_matrix,
                                            None,
                                            rvec, -tvec,
                                            useExtrinsicGuess = True)
            if not result:
                raise Exception(f"Retried solvePnP() failed! Cannot estimate initial extrinsics for {what}")
            if tvec[2] <= 0:
                raise SolvePnPerror_negz(f"Retried solvePnP() insists that tvec.z <= 0 (i.e. the chessboard is behind us). Cannot estimate initial extrinsics for {what}")

        Rt_cam_points = mrcal.Rt_from_rt(nps.glue(rvec.ravel(), tvec.ravel(), axis=-1))

        # visualize the fit
        # x_cam    = nps.matmult(Rt_cam_points[:3,:],ref_object)[..., 0] + Rt_cam_points[3,:]
        # x_imager = x_cam[...,:2]/x_cam[...,(2,)] * focal + (imagersize-1)/2
        # import gnuplotlib as gp
        # gp.plot( (x_imager[:,0],x_imager[:,1], dict(legend='solved')),
        #          (observations_local[:,0,0],observations_local[:,1,0], dict(legend='observed')),
        #          square=1,xrange=(0,4000),yrange=(4000,0),
        #          wait=1)
        # import IPython
        # IPython.embed()
        # sys.exit()

        return Rt_cam_points


    # if z<0, try again with bigger f
    # if too few points: try again with smaller f
    try:
        return solvepnp__try_focal_scale(1.)
    except SolvePnPerror_negz as e:
        return solvepnp__try_focal_scale(1.5)
    except SolvePnPerror_toofew as e:
        return solvepnp__try_focal_scale(0.7)

    # unreachable
    raise Exception("This should be unreachable. This is a bug")


def estimate_monocular_calobject_poses_Rt_tocam( indices_frame_camera,
                                                 observations,
                                                 object_spacing,
                                                 models_or_intrinsics,
                                                 *,
                                                 # for diagnostics only
                                                 paths = None):
    r"""Estimate camera-referenced poses of the calibration object from monocular views

SYNOPSIS

    print( indices_frame_camera.shape )
    ===>
    (123, 2)

    print( observations.shape )
    ===>
    (123, 10,9, 3)

    models = [mrcal.cameramodel(f) for f in ("cam0.cameramodel",
                                             "cam1.cameramodel")]

    # Estimated poses of the calibration object from monocular observations
    Rt_camera_frame = \
        mrcal.estimate_monocular_calobject_poses_Rt_tocam( indices_frame_camera,
                                                           observations,
                                                           object_spacing,
                                                           models)

    print( Rt_camera_frame.shape )
    ===>
    (123, 4, 3)

    i_observation = 10
    icam = indices_frame_camera[i_observation,1]

    # The calibration object in its reference coordinate system
    calobject = mrcal.ref_calibration_object(object_width_n,
                                             object_height_n,
                                             object_spacing)

    # The estimated calibration object points in the observing camera coordinate
    # system
    pcam = mrcal.transform_point_Rt( Rt_camera_frame[i_observation],
                                     calobject )

    # The pixel observations we would see if the calibration object pose was
    # where it was estimated to be
    q = mrcal.project(pcam, *models[icam].intrinsics())

    # The reprojection error, comparing these hypothesis pixel observations from
    # what we actually observed. We estimated the calibration object pose from
    # the observations, so this should be small
    err = q - observations[i_observation][:2]

    print( np.linalg.norm(err) )
    ===>
    [something small]

mrcal solves camera calibration problems by iteratively optimizing a nonlinear
least squares problem to bring the pixel observation predictions in line with
actual pixel observations. This requires an initial "seed", an estimate of the
solution. This function is a part of that computation. Since this is just an
initial estimate that will be refined, the results of this function do not need
to be exact.

We have pixel observations of a known calibration object, and we want to
estimate the pose of this object in the coordinate system of the camera that
produced these observations. This function ingests a number of such
observations, and solves this "PnP problem" separately for each one. The
observations may come from any lens model; everything is reprojected to a
pinhole model first to work with OpenCV. This function is a wrapper around the
solvePnP() openCV call, which does all the work.

ARGUMENTS

- indices_frame_camera: an array of shape (Nobservations,2) and dtype
  numpy.int32. Each row (iframe,icam) represents an observation of a
  calibration object by camera icam. iframe is not used by this function

- observations: an array of shape
  (Nobservations,object_height_n,object_width_n,3). Each observation corresponds
  to a row in indices_frame_camera, and contains a row of shape (3,) for each
  point in the calibration object. Each row is (x,y,weight) where x,y are the
  observed pixel coordinates. Any point where x<0 or y<0 or weight<0 is ignored.
  This is the only use of the weight in this function.

- object_spacing: the distance between adjacent points in the calibration
  object. A square object is assumed, so the vertical and horizontal distances
  are assumed to be identical. Usually we need the object dimensions in the
  object_height_n,object_width_n arguments, but here we get those from the shape
  of the observations array

- paths: optional iterable of strings, with the image filename corresponding to
  each slice of observations. Used for diagnostics only

- models_or_intrinsics: either

  - a list of mrcal.cameramodel objects from which we use the intrinsics
  - a list of (lensmodel,intrinsics_data) tuples

  These are indexed by icam from indices_frame_camera

RETURNED VALUE

An array of shape (Nobservations,4,3). Each slice is an Rt transformation TO the
camera coordinate system FROM the calibration object coordinate system.

    """

    # I'm given models. I remove the distortion so that I can pass the data
    # on to solvePnP()
    Ncameras      = len(models_or_intrinsics)

    lensmodels_intrinsics_data = [ m.intrinsics() if isinstance(m,mrcal.cameramodel) else m for m in models_or_intrinsics ]
    lensmodels                 = [di[0] for di in lensmodels_intrinsics_data]
    intrinsics_data            = [di[1] for di in lensmodels_intrinsics_data]

    if not all([mrcal.lensmodel_metadata_and_config(m)['has_core'] for m in lensmodels]):
        raise Exception("this currently works only with models that have an fxfycxcy core. It might not be required. Take a look at the following code if you want to add support")

    # We're looking at chessboards. indices_frame_camera is
    # indices_frame_camintrinsics_camextrinsics[:,:2]
    object_height_n,object_width_n = observations.shape[-3:-1]

    # No calobject_warp. Good-enough for the seeding
    # shape (Npoints,3)
    points_ref = \
        nps.clump(mrcal.ref_calibration_object(object_width_n, object_height_n, object_spacing),
                  n = 2)
    # observations has shape (Nobservations,Nh,Nw,3). I reshape it into
    # shape (Nobservations,Nh*Nw,3)
    observations = nps.mv(nps.clump(nps.mv(observations, -1,0),
                                    n = -2),
                          0, -1)

    Nobservations = len(observations)

    # this wastes memory, but makes it easier to keep track of which data goes
    # with what
    Rt_cam_points_all = np.zeros( (Nobservations, 4, 3), dtype=float)


    for i_observation in range(Nobservations):

        icam_intrinsics = indices_frame_camera[i_observation,1]

        path_annotation = '' if paths is None else f'; "{paths[i_observation]}"'
        kwargs = dict( lensmodel         = lensmodels     [icam_intrinsics],
                       intrinsics_data   = intrinsics_data[icam_intrinsics],
                       observation_qxqyw = observations[i_observation],
                       points_ref        = points_ref,
                       what              = f"observation {i_observation} (camera {icam_intrinsics}{path_annotation})" )

        Rt_cam_points = _estimate_camera_pose_from_fixed_point_observations(**kwargs)
        Rt_cam_points_all[i_observation, :, :] = Rt_cam_points


    return Rt_cam_points_all


def _estimate_camera_pose_from_point_observations( indices_point_camintrinsics_camextrinsics,
                                                   observations_point,
                                                   models_or_intrinsics,
                                                   points,
                                                   icam_intrinsics):
    r"""Estimate camera poses from monocular views of fixed points

SYNOPSIS

    # I have a monocular solve observing fixed points

    optimization_inputs = model.optimization_inputs()

    lensmodel                                 = optimization_inputs['lensmodel']
    intrinsics_data                           = optimization_inputs['intrinsics']
    indices_point_camintrinsics_camextrinsics = optimization_inputs['indices_point_camintrinsics_camextrinsics']
    observations_point                        = optimization_inputs['observations_point']
    points                                    = optimization_inputs['points']

    icam_intrinsics = 0

    Rt_camera_ref_estimate = \
        mrcal.calibration._estimate_camera_pose_from_point_observations( \
            indices_point_camintrinsics_camextrinsics,
            observations_point,
            ( (lensmodel, intrinsics_data[0]), ),
            points,
            icam_intrinsics = icam_intrinsics)

    print( Rt_camera_ref_estimate.shape )
    ===>
    (1, 4, 3)

    i = indices_point_camintrinsics_camextrinsics[:,1] == icam_intrinsics

    # The estimated calibration object points in the observing camera coordinate
    # system
    pcam = mrcal.transform_point_Rt( Rt_camera_ref_estimate[0], points[i] )

    # The pixel observations we would see if the calibration object pose was
    # where it was estimated to be
    q = mrcal.project(pcam, lensmodel, intrinsics_data)

    # The reprojection error, comparing these hypothesis pixel observations from
    # what we actually observed. We estimated the calibration object pose from
    # the observations, so this should be small
    err = q - observations_point[i][...,:2]

    print( np.linalg.norm(err) )
    ===>
    [something small]

mrcal solves camera calibration problems by iteratively optimizing a nonlinear
least squares problem to bring the pixel observation predictions in line with
actual pixel observations. This requires an initial "seed", an estimate of the
solution. This function is a part of that computation. Since this is just an
initial estimate that will be refined, the results of this function do not need
to be exact.

We have pixel observations of a known points in space, and we want to estimate
the pose of the camera in the coordinate system of these points. This function
ingests a number of point observations, and solves this "PnP problem". The
observations may come from any lens model; everything is reprojected to a
pinhole model first to work with OpenCV. This function is a wrapper around the
solvePnP() openCV call, which does all the work.

THIS FUNCTION IS A TEMPORARY PLACEHOLDER, AND IS SUBJECT TO CHANGE. IT CURRENTLY
WORKS WITH ONLY ONE PHYSICAL CAMERA AT A TIME, AND THIS CAMERA IS ASSUMED
STATIONARY

ARGUMENTS

- indices_point_camintrinsics_camextrinsics: an array of shape
  (Npoint_observations,3) and dtype numpy.int32. Each row
  (ipoint,icam_intrinsics,icam_extrinsics) represents a point observation. This
  is optimization_inputs['indices_point_camintrinsics_camextrinsics']

- observations_point: an array of shape (Npoint_observations,3). Each
  observation corresponds to a row in indices_point_camintrinsics_camextrinsics,
  and contains a row of shape (3,) for each point being observed. Each row is
  (x,y,weight) where x,y are the observed pixel coordinates. Any point where x<0
  or y<0 or weight<0 is ignored. This is the only use of the weight in this
  function. This is optimization_inputs['observations_point']

- models_or_intrinsics: either

  - a list of mrcal.cameramodel objects from which we use the intrinsics
  - a list of (lensmodel,intrinsics_data) tuples

  These are indexed by indices_point_camintrinsics_camextrinsics[...,1]

- points: an array of shape (Npoints,3). This is indexed by
  indices_point_camintrinsics_camextrinsics[...,0]. This is
  optimization_inputs['points']

- icam_intrinsics: the integer representing the physical camera being evaluated.
  This is compared against indices_point_camintrinsics_camextrinsics[...,1]

RETURNED VALUE

An array of shape (1,4,3). Each slice is an Rt transformation TO the
camera coordinate system FROM the points coordinate system.

    """

    Nobservations = len(indices_point_camintrinsics_camextrinsics)
    indices_point = indices_point_camintrinsics_camextrinsics[:,0]
    indices_cami  = indices_point_camintrinsics_camextrinsics[:,1]
    indices_came  = indices_point_camintrinsics_camextrinsics[:,2]

    if np.any(indices_came != indices_came[0]):
        raise Exception("For now this function only works with stationary monocular cameras")

    i = (indices_cami == icam_intrinsics)
    if np.count_nonzero(i) < 4:
        raise Exception(f"Requested icam_intrinsics{icam_intrinsics} matched too few observations: {np.count_nonzero(i)}")

    # I'm given models. I remove the distortion so that I can pass the data
    # on to solvePnP()
    Ncameras = len(models_or_intrinsics)
    if Ncameras != 1:
        raise Exception(f"For now this function only works with stationary monocular cameras, but given Ncameras={Ncameras}")

    lensmodels_intrinsics_data = [ m.intrinsics() if isinstance(m,mrcal.cameramodel) else m for m in models_or_intrinsics ]
    lensmodels                 = [di[0] for di in lensmodels_intrinsics_data]
    intrinsics_data            = np.array([di[1] for di in lensmodels_intrinsics_data])

    if not all([mrcal.lensmodel_metadata_and_config(m)['has_core'] for m in lensmodels]):
        raise Exception("this currently works only with models that have an fxfycxcy core. It might not be required. Take a look at the following code if you want to add support")

    Rt_cam_points_all = np.zeros( (1, 4, 3), dtype=float)

    kwargs = dict( lensmodel         = lensmodels     [icam_intrinsics],
                   intrinsics_data   = intrinsics_data[icam_intrinsics],
                   observation_qxqyw = observations_point[i],
                   points_ref        = points[indices_point[i]],
                   what              = f"point observations (camera {icam_intrinsics})" )

    Rt_cam_points_all[0,...] = _estimate_camera_pose_from_fixed_point_observations(**kwargs)
    return Rt_cam_points_all


def _estimate_camera_poses( # shape (Nobservations,4,3)
                            calobject_poses_local_Rt_cf,
                            # shape (Nobservations,2)
                            indices_frame_camera, \
                            object_width_n, object_height_n,
                            object_spacing):
    r'''Estimate camera poses in respect to each other

    We are given poses of the calibration object in respect to each observing
    camera. We also have multiple cameras observing the same calibration object
    at the same time, and we have local poses for each. We can thus compute the
    relative camera pose from these observations.

    We have many frames that have different observations from the same set of
    fixed-relative-pose cameras, so we compute the relative camera pose to
    optimize the observations

    Note that this assumes we're solving a calibration problem (stationary
    cameras) observing a moving object, so uses indices_frame_camera, not
    indices_frame_camintrinsics_camextrinsics, which mrcal.optimize() expects
    '''

    Ncameras = np.max(indices_frame_camera[:,1]) + 1

    # I need to compute an estimate of the pose of each camera in the coordinate
    # system of camera0. This is only possible if there're enough overlapping
    # observations. For instance if camera1 has overlapping observations with
    # camera2, but neither overlap with camera0, then I can't relate camera1,2 to
    # camera0. However if camera0 has overlap with camera1, then I can compute
    # the relative pose of camera2-camera0 from its overlapping observations with
    # camera1.
    #
    # I solve a shortest-path problem using Dijkstra's algorithm to find a set
    # of pair overlaps between cameras that leads to camera0. I favor edges with
    # large numbers of shared observed frames

    # list of camera-i to camera-0 transforms. I keep doing stuff until this
    # list is full of valid data
    Rt_0c = [None] * (Ncameras-1)

    def compute_pairwise_Rt(icam_to, icam_from):

        # I want to assume that icam_from > icam_to. If it's not true, compute the
        # opposite transform, and invert
        if icam_to > icam_from:
            Rt = compute_pairwise_Rt(icam_from, icam_to)
            return mrcal.invert_Rt(Rt)

        if icam_to == icam_from:
            raise Exception(f"Got icam_to == icam_from ( = {icam_to} ). This was probably a mistake")

        # Now I KNOW that icam_from > icam_to


        Nobservations = indices_frame_camera.shape[0]

        # This is a hack. I look at the correspondence of camera0 to camera i for i
        # in 1:N-1. I ignore all correspondences between cameras i,j if i!=0 and
        # j!=0. Good enough for now

        # No calobject_warp. Good-enough for the seeding
        ref_object = \
            nps.clump(mrcal.ref_calibration_object(object_width_n,object_height_n,
                                                   object_spacing),
                      n = 2)

        A = np.array(())
        B = np.array(())

        # I traverse my observation list, and pick out observations from frames
        # that had data from both my cameras
        iframe_last   = -1
        d0            = None
        d1            = None
        Rt_cam0_frame = None
        Rt_cam1_frame = None
        for i_observation in range(Nobservations):
            iframe_this,icam_this = indices_frame_camera[i_observation, ...]
            if iframe_this != iframe_last:
                d0            = None
                d1            = None
                Rt_cam0_frame = None
                Rt_cam1_frame = None
                iframe_last   = iframe_this

            # The cameras appear in order. And above I made sure that icam_from >
            # icam_to, so I take advantage of that here
            if icam_this == icam_to:
                if Rt_cam0_frame is not None:
                    raise Exception(f"Saw multiple camera{icam_this} observations in frame {iframe_this}")
                Rt_cam0_frame = calobject_poses_local_Rt_cf[i_observation, ...]

            elif icam_this == icam_from:
                if Rt_cam0_frame is None: # have camera1 observation, but not camera0
                    continue

                if Rt_cam1_frame is not None:
                    raise Exception(f"Saw multiple camera{icam_this} observations in frame {iframe_this}")
                Rt_cam1_frame = calobject_poses_local_Rt_cf[i_observation, ...]

                A = nps.glue(A, mrcal.transform_point_Rt(Rt_cam0_frame,
                                                         ref_object),
                             axis = -2)
                B = nps.glue(B, mrcal.transform_point_Rt(Rt_cam1_frame,
                                                         ref_object),
                             axis = -2)

        return mrcal.align_procrustes_points_Rt01(A, B)

    def compute_connectivity_matrix():
        r'''Returns a connectivity matrix of camera observations

        Returns a symmetric (Ncamera,Ncamera) matrix of integers, where each
        entry contains the number of frames containing overlapping observations
        for that pair of cameras

        '''

        camera_connectivity = np.zeros( (Ncameras,Ncameras), dtype=np.uint16 )
        def finish_frame(i0, i1):
            for ic0 in range(i0, i1):
                for ic1 in range(ic0+1, i1+1):
                    camera_connectivity[indices_frame_camera[ic0,1], indices_frame_camera[ic1,1]] += 1
                    camera_connectivity[indices_frame_camera[ic1,1], indices_frame_camera[ic0,1]] += 1

        f_current       = -1
        i_start_current = -1

        for i in range(len(indices_frame_camera)):
            f,c = indices_frame_camera[i]
            if f < f_current:
                raise Exception("I'm assuming the frame indices are increasing monotonically")
            if f > f_current:
                # first camera in this observation
                f_current = f
                if i_start_current >= 0:
                    finish_frame(i_start_current, i-1)
                i_start_current = i
        finish_frame(i_start_current, len(indices_frame_camera)-1)
        return camera_connectivity

    def found_best_path_to_node(camera_idx, from_idx):
        '''A shortest path was found'''
        if camera_idx == 0:
            # This is the reference camera. Nothing to do
            return

        Rt_fc = compute_pairwise_Rt(from_idx, camera_idx)

        if from_idx == 0:
            Rt_0c[camera_idx-1] = Rt_fc
            return

        Rt_0f = Rt_0c[from_idx-1]
        Rt_0c[camera_idx-1] = mrcal.compose_Rt( Rt_0f, Rt_fc)

    # shape (Ncamera,Ncameras)
    # shape (Ncameras,Ncameras); each element is the number of shared
    # observations
    shared_frames = compute_connectivity_matrix()

    # We need at least 2 shared observations to be useful; otherwise
    # align_point_clouds() complains about insufficient overlap. I thus set to 0
    # and too-sparse links, to make the graph traversal ignore those
    shared_frames[shared_frames<2] = 0

    mrcal.traverse_sensor_links( connectivity_matrix  = shared_frames,
                                 callback_sensor_link = found_best_path_to_node )

    if any([x is None for x in Rt_0c]):
        raise Exception("ERROR: Don't have complete camera observations overlap!\n" +
                        f"Past-camera-0 Rt:\n{Rt_0c}\n"                             +
                        f"Shared observations matrix:\n{shared_frames}\n")


    return np.ascontiguousarray(nps.cat(*Rt_0c))


def _traverse_sensor_links_python( Nsensors,
                                   callback__neighbors,
                                   callback__cost_edge,
                                   callback__sensor_link ):
    '''Traverses a connectivity graph of sensors

    Starts from the root sensor (defined to have idx==0), and visits each one in
    order of total distance from the root. Useful to evaluate the whole set of
    sensors using pairwise metrics, building the network up from the
    best-connected, to the worst-connected. Any sensor not connected to the root
    at all will NOT be visited. The caller should check for any unvisited
    sensors.

    We have Nsensors sensors. Each one is identified by an integer in
    [0,Nsensors). The root is defined to be sensor 0.

    Three callbacks must be passed in:

    - callback__neighbors (i)
      An iterable returning each neigbor of sensor i.

    - callback__cost_edge(i, i_parent)
      The cost between two adjacent nodes

    - callback__sensor_link(i, i_parent)
      Called when the best path to node i is found. This path runs through
      i_parent as the previous sensor

    '''

    import heapq

    class Node:
        def __init__(self, idx):
            self.idx        = idx
            self.idx_parent = -1
            self.cost       = None
            self.done       = False

        def __lt__(self, other):
            return self.cost < other.cost

        def visit(self):
            if self.idx != 0:
                # We're not at the start; propagate the connection
                callback__sensor_link(self.idx,
                                      self.idx_parent)
            self.done = True

            for neighbor_idx in callback__neighbors(self.idx):
                neighbor = nodes[neighbor_idx]

                if neighbor.done:
                    continue

                cost_to_neighbor_via_node = \
                    self.cost + \
                    callback__cost_edge(neighbor_idx,self.idx)

                if neighbor.cost is None:
                    # Haven't seen this node yet
                    neighbor.cost = cost_to_neighbor_via_node
                    neighbor.idx_parent     = self.idx
                    heapq.heappush(heap, neighbor)
                else:
                    # This node is already in the heap, ready to be processed.
                    # If this new path to this node is better, use it
                    if cost_to_neighbor_via_node < neighbor.cost:
                        neighbor.cost = cost_to_neighbor_via_node
                        neighbor.idx_parent     = self.idx
                        heapq.heapify(heap) # is this the most efficient "update" call?


    nodes = [Node(i) for i in range(Nsensors)]
    nodes[0].cost = 0
    heap = []

    nodes[0].visit()
    while heap:
        node_top = heapq.heappop(heap)
        node_top.visit()


def estimate_joint_frame_poses(calobject_Rt_camera_frame,
                               Rt_cam_ref,
                               indices_frame_camera,
                               object_width_n, object_height_n,
                               object_spacing):

    r'''Estimate world-referenced poses of the calibration object

SYNOPSIS

    print( calobject_Rt_camera_frame.shape )
    ===>
    (123, 4,3)

    print( Rt_cam_ref.shape )
    ===>
    (2, 4,3)
    # We have 3 cameras. The first one is at the reference coordinate system,
    # the pose estimates of the other two are in this array

    print( indices_frame_camera.shape )
    ===>
    (123, 2)

    rt_ref_frame = \
        mrcal.estimate_joint_frame_poses(calobject_Rt_camera_frame,
                                         Rt_cam_ref,
                                         indices_frame_camera,
                                         object_width_n, object_height_n,
                                         object_spacing)

    print( rt_ref_frame.shape )
    ===>
    (87, 6)

    # We have 123 observations of the calibration object by ANY camera. 87
    # instances of time when the object was observed. Most of the time it was
    # observed by multiple cameras simultaneously, hence 123 > 87

    i_observation = 10
    iframe,icam = indices_frame_camera[i_observation, :]

    # The calibration object in its reference coordinate system
    calobject = mrcal.ref_calibration_object(object_width_n,
                                             object_height_n,
                                             object_spacing)

    # The estimated calibration object points in the reference coordinate
    # system, for this one observation
    pref = mrcal.transform_point_rt( rt_ref_frame[iframe],
                                     calobject )

    # The estimated calibration object points in the camera coord system. Camera
    # 0 is at the reference
    if icam >= 1:
        pcam = mrcal.transform_point_Rt( Rt_cam_ref[icam-1],
                                         pref )
    else:
        pcam = pref

    # The pixel observations we would see if the pose estimates were correct
    q = mrcal.project(pcam, *models[icam].intrinsics())

    # The reprojection error, comparing these hypothesis pixel observations from
    # what we actually observed. This should be small
    err = q - observations[i_observation][:2]

    print( np.linalg.norm(err) )
    ===>
    [something small]

mrcal solves camera calibration problems by iteratively optimizing a nonlinear
least squares problem to bring the pixel observation predictions in line with
actual pixel observations. This requires an initial "seed", an initial estimate
of the solution. This function is a part of that computation. Since this is just
an initial estimate that will be refined, the results of this function do not
need to be exact.

This function ingests an estimate of the camera poses in respect to each other,
and the estimate of the calibration objects in respect to the observing camera.
Most of the time we have simultaneous calibration object observations from
multiple cameras, so this function consolidates all this information to produce
poses of the calibration object in the reference coordinate system, NOT the
observing-camera coordinate system poses we already have.

By convention, we have a "reference" coordinate system that ties the poses of
all the frames (calibration objects) and the cameras together. And by
convention, this "reference" coordinate system is the coordinate system of
camera 0. Thus the array of camera poses Rt_cam_ref holds Ncameras-1
transformations: the first camera has an identity transformation, by definition.

This function assumes we're observing a moving object from stationary cameras
(i.e. a vanilla camera calibration problem). The mrcal solver is more general,
and supports moving cameras, hence it uses a more general
indices_frame_camintrinsics_camextrinsics array instead of the
indices_frame_camera array used here.

ARGUMENTS

- calobject_Rt_camera_frame: an array of shape (Nobservations,4,3). Each slice
  is an Rt transformation TO the observing camera coordinate system FROM the
  calibration object coordinate system. This is returned by
  estimate_monocular_calobject_poses_Rt_tocam()

- Rt_cam_ref: an array of shape (Ncameras-1,4,3). Each slice is an Rt
  transformation TO the camera coordinate system FROM the reference coordinate
  system. By convention camera 0 defines the reference coordinate system, so
  that camera's extrinsics are the identity, by definition, and we don't store
  that data in this array

- indices_frame_camera: an array of shape (Nobservations,2) and dtype
  numpy.int32. Each row (iframe,icam) represents an observation at time
  instant iframe of a calibration object by camera icam

- object_width_n: number of horizontal points in the calibration object grid

- object_height_n: number of vertical points in the calibration object grid

- object_spacing: the distance between adjacent points in the calibration
  object. A square object is assumed, so the vertical and horizontal distances
  are assumed to be identical

RETURNED VALUE

An array of shape (Nframes,6). Each slice represents the pose of the calibration
object at one instant in time: an rt transformation TO the reference coordinate
system FROM the calibration object coordinate system.

    '''

    Rt_ref_cam = mrcal.invert_Rt( Rt_cam_ref )


    def Rt_ref_frame(i_observation0, i_observation1):
        R'''Given a range of observations corresponding to the same frame, estimate the
        pose of that frame

        '''

        def Rt_ref_frame__single_observation(i_observation):
            r'''Transform from the board coords to the reference coords'''
            iframe,icam = indices_frame_camera[i_observation, ...]

            Rt_cam_frame = calobject_Rt_camera_frame[i_observation, :,:]
            if icam == 0:
                return Rt_cam_frame

            return mrcal.compose_Rt( Rt_ref_cam[icam-1, ...], Rt_cam_frame)


        # frame poses should map FROM the frame coord system TO the ref coord
        # system (camera 0).

        # special case: if there's a single observation, I just use it
        if i_observation1 - i_observation0 == 1:
            return Rt_ref_frame__single_observation(i_observation0)

        # Multiple cameras have observed the object for this frame. I have an
        # estimate of these for each camera. I merge them in a lame way: I
        # average out the positions of each point, and fit the calibration
        # object into the mean point cloud
        #
        # No calobject_warp. Good-enough for the seeding
        obj = mrcal.ref_calibration_object(object_width_n, object_height_n,
                                           object_spacing)

        sum_obj_unproj = obj*0
        for i_observation in range(i_observation0, i_observation1):
            Rt = Rt_ref_frame__single_observation(i_observation)
            sum_obj_unproj += mrcal.transform_point_Rt(Rt, obj)

        mean_obj_ref = sum_obj_unproj / (i_observation1 - i_observation0)

        # Got my point cloud. fit

        # transform both to shape = (N*N, 3)
        obj          = nps.clump(obj,  n=2)
        mean_obj_ref = nps.clump(mean_obj_ref, n=2)
        return mrcal.align_procrustes_points_Rt01( mean_obj_ref, obj )




    rt_ref_frame = np.array(())

    iframe_current          = -1
    i_observation_framestart = -1;

    for i_observation in range(indices_frame_camera.shape[0]):
        iframe,icam = indices_frame_camera[i_observation, ...]

        if iframe != iframe_current:
            if i_observation_framestart >= 0:
                Rt = Rt_ref_frame(i_observation_framestart,
                                  i_observation)
                rt_ref_frame = nps.glue(rt_ref_frame,
                                        mrcal.rt_from_Rt(Rt),
                                        axis=-2)

            i_observation_framestart = i_observation
            iframe_current          = iframe

    if i_observation_framestart >= 0:
        Rt = Rt_ref_frame(i_observation_framestart,
                          indices_frame_camera.shape[0])
        rt_ref_frame = nps.glue(rt_ref_frame,
                                mrcal.rt_from_Rt(Rt),
                                axis=-2)

    return rt_ref_frame


def seed_stereographic( imagersizes,
                        focal_estimate,
                        indices_frame_camera,
                        observations,
                        object_spacing,
                        *,
                        # for diagnostics only
                        paths = None):
    r'''Compute an optimization seed for a camera calibration

SYNOPSIS

    print( imagersizes.shape )
    ===>
    (4, 2)

    print( indices_frame_camera.shape )
    ===>
    (123, 2)

    print( observations.shape )
    ===>
    (123, 3)

    intrinsics_data,       \
    rt_cam_ref, \
    rt_ref_frame =      \
        mrcal.seed_stereographic(imagersizes          = imagersizes,
                                 focal_estimate       = 1500,
                                 indices_frame_camera = indices_frame_camera,
                                 observations         = observations,
                                 object_spacing       = object_spacing)

    ....

    mrcal.optimize(intrinsics            = intrinsics_data,
                   rt_cam_ref            = rt_cam_ref,
                   rt_ref_frame          = rt_ref_frame,
                   lensmodel             = 'LENSMODEL_STEREOGRAPHIC',
                   ...)

mrcal solves camera calibration problems by iteratively optimizing a nonlinear
least squares problem to bring the pixel observation predictions in line with
actual pixel observations. This requires an initial "seed", an initial estimate
of the solution. This function computes a usable seed, and its results can be
fed to mrcal.optimize(). The output of this function is just an initial estimate
that will be refined, so the results of this function do not need to be exact.

This function assumes we have stereographic lenses, and the returned intrinsics
apply to LENSMODEL_STEREOGRAPHIC. This is usually good-enough to serve as a seed
for both long lenses and wide lenses, every ultra-wide fisheyes. The returned
intrinsics can be expanded to whatever lens model we actually want to use prior
to invoking the optimizer.

By convention, we have a "reference" coordinate system that ties the poses of
all the frames (calibration objects) and the cameras together. And by
convention, this "reference" coordinate system is the coordinate system of
camera 0. Thus the array of camera poses rt_cam_ref holds Ncameras-1
transformations: the first camera has an identity transformation, by definition.

This function assumes we're observing a moving object from stationary cameras
(i.e. a vanilla camera calibration problem). The mrcal solver is more general,
and supports moving cameras, hence it uses a more general
indices_frame_camintrinsics_camextrinsics array instead of the
indices_frame_camera array used here.

See test/test-basic-calibration.py and mrcal-calibrate-cameras for usage
examples.

ARGUMENTS

- imagersizes: an iterable of (imager_width,imager_height) iterables. Defines
  the imager dimensions for each camera we're calibrating. May be an array of
  shape (Ncameras,2) or a tuple of tuples or a mix of the two

- focal_estimate: an initial estimate of the focal length of the cameras, in
  pixels. For the purposes of the initial estimate we assume the same focal
  length value for both the x and y focal length. This argument can be

  - a scalar: this is applied to all the cameras
  - an iterable of length 1: this value is applied to all the cameras
  - an iterable of length Ncameras: each camera uses a different value

- indices_frame_camera: an array of shape (Nobservations,2) and dtype
  numpy.int32. Each row (iframe,icam) represents an observation of a
  calibration object by camera icam. iframe is not used by this function

- observations: an array of shape
  (Nobservations,object_height_n,object_width_n,3). Each observation corresponds
  to a row in indices_frame_camera, and contains a row of shape (3,) for each
  point in the calibration object. Each row is (x,y,weight) where x,y are the
  observed pixel coordinates. Any point where x<0 or y<0 or weight<0 is ignored.
  This is the only use of the weight in this function.

- object_spacing: the distance between adjacent points in the calibration
  object. A square object is assumed, so the vertical and horizontal distances
  are assumed to be identical. Usually we need the object dimensions in the
  object_height_n,object_width_n arguments, but here we get those from the shape
  of the observations array

- paths: optional iterable of strings, with the image filename corresponding to
  each slice of observations. Used for diagnostics only

RETURNED VALUES

We return a tuple:

- intrinsics_data: an array of shape (Ncameras,4). Each slice contains the
  stereographic intrinsics for the given camera. These intrinsics are
  (focal_x,focal_y,centerpixel_x,centerpixel_y), and define
  LENSMODEL_STEREOGRAPHIC model. mrcal refers to these 4 values as the
  "intrinsics core". For models that have such a core (currently, ALL supported
  models), the core is the first 4 parameters of the intrinsics vector. So to
  calibrate some cameras, call seed_stereographic(), append to intrinsics_data
  the proper number of parameters to match whatever lens model we're using, and
  then invoke the optimizer.

- rt_cam_ref: an array of shape (Ncameras-1,6). Each slice is an rt
  transformation TO the camera coordinate system FROM the reference coordinate
  system. By convention camera 0 defines the reference coordinate system, so
  that camera's extrinsics are the identity, by definition, and we don't store
  that data in this array

- rt_ref_frame: an array of shape (Nframes,6). Each slice represents the pose
  of the calibration object at one instant in time: an rt transformation TO the
  reference coordinate system FROM the calibration object coordinate system.

    '''

    Ncameras = len(imagersizes)

    try:
        Nfocal = len(focal_estimate)
    except:
        # scalar
        Nfocal = 1
        focal_estimate = (focal_estimate,)

    if Nfocal == 1:
        focal_estimate = focal_estimate * Ncameras
    elif Nfocal == Ncameras:
        # already good
        pass
    else:
        raise Exception(f"Ncameras mismatch: len(imagersizes) = {Ncameras} but len(focal_estimate) = {len(focal_estimate)}")


    # Done. focal_estimate is now an iterable length Ncameras

    # I compute an estimate of the poses of the calibration object in the local
    # coord system of each camera for each frame. This is done for each frame
    # and for each camera separately. This isn't meant to be precise, and is
    # only used for seeding.
    #
    # I get rotation, translation in a (4,3) array, such that R*calobject + t
    # produces the calibration object points in the coord system of the camera.
    # The result has dimensions (N,4,3)
    intrinsics = [('LENSMODEL_STEREOGRAPHIC',
                   np.array((focal_estimate[icam],
                             focal_estimate[icam],
                             (imagersizes[icam][0]-1.)/2,
                             (imagersizes[icam][1]-1.)/2,))) \
                  for icam in range(Ncameras)]

    calobject_poses_local_Rt_cf = \
        mrcal.estimate_monocular_calobject_poses_Rt_tocam( indices_frame_camera,
                                                           observations,
                                                           object_spacing,
                                                           intrinsics,
                                                           paths = paths)
    # these map FROM the coord system of the calibration object TO the coord
    # system of this camera

    object_height_n,object_width_n = observations.shape[-3:-1]

    # I now have a rough estimate of calobject poses in the coord system of each
    # camera. One can think of these as two sets of point clouds, each attached
    # to their camera. I can move around the two sets of point clouds to try to
    # match them up, and this will give me an estimate of the relative pose of
    # the two cameras in respect to each other. I need to set up the
    # correspondences, and mrcal.align_procrustes_points_Rt01() does the rest
    #
    # I get transformations that map points in camera-cami coord system to 0th
    # camera coord system. Rt have dimensions (N-1,4,3)
    camera_poses_Rt_0_cami = \
        _estimate_camera_poses( calobject_poses_local_Rt_cf,
                                indices_frame_camera,
                                object_width_n, object_height_n,
                                object_spacing)

    if len(camera_poses_Rt_0_cami):
        # extrinsics should map FROM the ref coord system TO the coord system of the
        # camera in question. This is backwards from what I have
        Rt_cam_ref = \
            nps.atleast_dims( mrcal.invert_Rt(camera_poses_Rt_0_cami),
                              -3 )
    else:
        Rt_cam_ref = np.zeros((0,4,3))

    rt_ref_frame = \
        mrcal.estimate_joint_frame_poses(
            calobject_poses_local_Rt_cf,
            Rt_cam_ref,
            indices_frame_camera,
            object_width_n, object_height_n,
            object_spacing)

    return \
        nps.cat(*[i[1] for i in intrinsics]), \
        nps.atleast_dims(mrcal.rt_from_Rt(Rt_cam_ref), -2), \
        rt_ref_frame


def _compute_valid_intrinsics_region(model,
                                     threshold_uncertainty,
                                     threshold_mean,
                                     threshold_stdev,
                                     threshold_count,
                                     distance,
                                     gridn_width  = 30,
                                     gridn_height = None):
    r'''Returns the valid-intrinsics region for the camera in the model

Internal function use by the mrcal-calibrate-cameras utility.

The model is expected to contain the optimization_inputs, which are used for all
the work. The thresholds come from the --valid-intrinsics-region-parameters
argument to mrcal-calibrate-cameras

This is a closed contour, in an (N,2) numpy array. None means "no
valid-intrinsics region computed". An empty array of shape (0,2) means "the
region was computed and it is empty"

The imager of a camera is subdivided into regions (controlled by the
gridn_width, gridn_height arguments). The residual statistics are then computed
for each bin separately. We can then clearly see areas of insufficient data
(observation counts will be low). And we can clearly see lens-model-induced
biases (non-zero mean) and we can see heteroscedasticity (uneven standard
deviation). The mrcal-calibrate-cameras tool uses these metrics to construct a
valid-intrinsics region for the models it computes. This serves as a quick/dirty
method of modeling projection reliability, which can be used even if projection
uncertainty cannot be computed.

    '''

    import cv2

    W,H = model.imagersize()
    if gridn_height is None:
        gridn_height = int(round(H/W*gridn_width))

    # Each has shape (Nheight,Nwidth)
    mean,stdev,count,using = \
        _report_regional_statistics(model,
                                    gridn_width  = gridn_width,
                                    gridn_height = gridn_height)

    q    = mrcal.sample_imager( gridn_width, gridn_height, *model.imagersize() )
    vcam = mrcal.unproject(q, *model.intrinsics(),
                           normalize = True)

    if distance <= 0:
        atinfinity = True
        pcam       = vcam
    else:
        atinfinity = False
        pcam = vcam * distance

    uncertainty = mrcal.projection_uncertainty(pcam,
                                               model      = model,
                                               atinfinity = atinfinity,
                                               what       = 'worstdirection-stdev' )

    # Uncertainty of nan or inf is invalid, so I mark it as very high. This
    # serves to silence a warning in the next statement where we're comparing
    # something with nan
    uncertainty.ravel() [~np.isfinite(uncertainty.ravel())] = 1e9

    # shape (Nheight,Nwidth).
    mask = \
        (uncertainty < threshold_uncertainty)

    if not re.match('LENSMODEL_SPLINED_', model.intrinsics()[0]):
        mask *= \
            (mean        < threshold_mean) * \
            (stdev       < threshold_stdev) * \
            (count       > threshold_count)

    # I compute the contour. OpenCV can't process binary images, so I need
    # to convert to a different image type first. AND findContours() reports
    # the coordinates in the opposite order as how they're given (image is
    # given as y,x; returned coords are x,y). This is what I want, but feels
    # conterintuitive

    # This is a hoaky mess. I ignore all the topological corner cases, and just
    # grab the contour with the biggest area
    contours = \
        cv2.findContours(mask.astype(np.uint8),
                         cv2.RETR_EXTERNAL,
                         cv2.CHAIN_APPROX_SIMPLE)[-2]

    areas = np.array([ cv2.contourArea(c) for c in contours ])
    if areas.size == 0:
        return np.zeros((0,2)) # empty valid-intrinsics region

    contour = contours[np.argmax(areas)][:,0,:].astype(float)

    contour = mrcal.close_contour(contour)
    if contour.ndim != 2 or contour.shape[0] < 4:
        # I have a closed contour, so the only way for it to not be
        # degenerate is to include at least 4 points
        return np.zeros((0,2)) # empty valid-intrinsics region

    # I convert the contours back to the full-res image coordinate. The grid
    # mapping is based on the corner pixels
    W,H = model.imagersize()
    contour[:,0] *= float(W-1)/(gridn_width -1)
    contour[:,1] *= float(H-1)/(gridn_height-1)

    return contour.round().astype(np.int32)


def _report_regional_statistics( model,
                                 gridn_width  = 20,
                                 gridn_height = None):

    r'''Reports fit statistics for regions across the imager

SYNOPSIS

    mean, stdev, count, using = \
        mrcal._report_regional_statistics(model,
                                          gridn_width = 30)

    import gnuplotlib as gp
    W,H = imagersize
    gp.plot( np.abs(mean),
             tuplesize = 3,
             _with     = 'image',
             ascii     = True,
             square    = True,
             using     = using)

This is an internal function used by mrcal._compute_valid_intrinsics_region()
and mrcal.show_residuals_regional(). The mrcal solver optimizes reprojection
errors for ALL the observations in ALL cameras at the same time. It is useful to
evaluate the optimal solution by examining reprojection errors in subregions of
the imager, which is accomplished by this function. All the observations and
reprojection errors and subregion gridding are given. The mean and standard
derivation of the reprojection errors and a point count are returned for each
subregion cell. A "using" expression for plotting is reported as well.

After a problem-free solve, the error distributions in each area of the imager
should be similar, and should match the error distribution of the pixel
observations. If the lens model doesn't fit the data, the statistics will not be
consistent across the region: the residuals would be heteroscedastic.

The imager of a camera is subdivided into regions (controlled by the
gridn_width, gridn_height arguments). The residual statistics are then computed
for each bin separately. We can then clearly see areas of insufficient data
(observation counts will be low). And we can clearly see lens-model-induced
biases (non-zero mean) and we can see heteroscedasticity (uneven standard
deviation). The mrcal-calibrate-cameras tool uses these metrics to construct a
valid-intrinsics region for the models it computes. This serves as a quick/dirty
method of modeling projection reliability, which can be used even if projection
uncertainty cannot be computed.

ARGUMENTS

- model: the model of the camera we're looking at. This model must contain the
  optimization_inputs.

- gridn_width: how many points along the horizontal gridding dimension

- gridn_height: how many points along the vertical gridding dimension. If None,
  we compute an integer gridn_height to maintain a square-ish grid:
  gridn_height/gridn_width ~ imager_height/imager_width

RETURNED VALUES

This function returns a tuple

- mean: an array of shape (gridn_height,gridn_width). Contains the mean of
  the residuals in the corresponding cell

- stdev: an array of shape (gridn_height,gridn_width). Contains the standard
  deviation of the residuals in the corresponding cell

- count: an array of shape (gridn_height,gridn_width). Contains the count of
  observations in the corresponding cell

- using: is a "using" keyword for plotting the output matrices with gnuplotlib.
  See the docstring for imagergrid_using() for details

    '''

    W,H=model.imagersize()

    if gridn_height is None:
        gridn_height = int(round(H/W*gridn_width))

    # shape: (Nheight,Nwidth,2). Contains (x,y) rows
    q_cell_center = mrcal.sample_imager(gridn_width, gridn_height, W, H)

    wcell = float(W-1) / (gridn_width -1)
    hcell = float(H-1) / (gridn_height-1)
    rcell = np.array((wcell,hcell), dtype=float) / 2.

    @nps.broadcast_define( (('N',2), ('N',2), (2,)),
                           (3,) )
    def stats(q, err, q_cell_center):
        r'''Compute the residual statistics in a single cell

        '''

        # boolean (x,y separately) map of observations that are within a cell
        idx = np.abs(q - q_cell_center) < rcell

        # join x,y: both the x,y must be within a cell for the observation to be
        # within a cell
        idx = idx[:,0] * idx[:,1]

        err = err[idx, ...].ravel()
        if len(err) <= 5:
            # we have too little data in this cell
            return np.array((0.,0.,len(err)))

        mean   = np.mean(err)
        stdev  = np.std(err)
        return np.array((mean,stdev,len(err)))



    optimization_inputs = model.optimization_inputs()
    icam                = model.icam_intrinsics()

    # shape (Nobservations, object_height_n, object_width_n, 3)
    observations         = optimization_inputs['observations_board']
    indices_frame_camera = optimization_inputs['indices_frame_camintrinsics_camextrinsics'][...,:2]

    residuals_shape = observations.shape[:-1] + (2,)

    # shape (Nobservations,object_height_n,object_width_n,2)
    x = \
        mrcal.optimizer_callback(**optimization_inputs,
                                 no_jacobian      = True,
                                 no_factorization = True)[1][:np.prod(residuals_shape)]. \
        reshape(*residuals_shape)

    # shape (Nobservations, object_height_n, object_width_n)
    idx = np.ones( observations.shape[:-1], dtype=bool)

    # select residuals from THIS camera
    idx[indices_frame_camera[:,1] != icam, ...] = False
    # select non-outliers
    idx[ observations[...,2] <= 0.0 ] = False

    # shape (N,2)
    err = x           [idx, ...    ]
    obs = observations[idx, ..., :2]

    # Each has shape (Nheight,Nwidth)
    mean,stdev,count = nps.mv( stats(obs, err, q_cell_center),
                               -1, 0)
    return     \
        mean,  \
        stdev, \
        count, \
        mrcal.imagergrid_using(model.imagersize(), gridn_width, gridn_height)
