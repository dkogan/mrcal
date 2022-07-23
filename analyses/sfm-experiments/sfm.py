#!/usr/bin/python3

r'''
'''

import sys
import numpy as np
import numpysane as nps
import gnuplotlib as gp
import cv2
import glob
import re
import os
import sqlite3


sys.path[:0] = '/home/dima/projects/mrcal-2022-06--triangulated-solve/',
import mrcal


def imread(filename, decimation):
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    return image, image[::decimation, ::decimation]

def plot_flow(image, flow, decimation,
              **kwargs):
    H,W = flow.shape[:2]

    # each has shape (H,W)
    xx,yy = np.meshgrid(np.arange(0,W,decimation),
                        np.arange(0,H,decimation))

    # shape (H,W,4)
    vectors = \
        nps.glue( nps.dummy(xx,-1),
                  nps.dummy(yy,-1),
                  flow[::decimation, ::decimation, :],
                  axis = -1);
    vectors = nps.clump(vectors, n=2)

    gp.plot( (image, dict(_with='image',
                          tuplesize = 3)),
             (vectors,
              dict(_with='vectors lc "red"',
                   tuplesize=-4)),

             _set  = 'palette gray',
             unset = 'colorbox',
             square = True,
             _xrange = (0,W),
             _yrange = (H,0),
             **kwargs)

def match_looks_valid(q, match, flow):

    flow_observed = q[1] - q[0]
    flow_err_sq   = nps.norm2(flow_observed - flow)

    return \
        match.distance < 30 and \
        flow_err_sq    < 2*2

def seed_rt10_pair(q0, q1):
    lensmodel,intrinsics_data = model.intrinsics()
    if not re.match("LENSMODEL_(OPENCV|PINHOLE)", lensmodel):
        raise Exception("This assumes a pinhole or opencv model. You have something else, and you should reproject to pinhole")
    fx,fy,cx,cy = intrinsics_data[:4]
    distortions = intrinsics_data[4:]
    camera_matrix = np.array((( fx,  0, cx),
                              ( 0,  fy, cy),
                              ( 0,   0, 1.)))

    # arbitrary scale
    p0 = mrcal.unproject(q0, lensmodel,intrinsics_data)

    result,rvec,tvec = \
        cv2.solvePnP( p0,
                      np.ascontiguousarray(q1),
                      camera_matrix,
                      distortions,
                      rvec = (0,0,0),
                      tvec = (0,0,0),
                      useExtrinsicGuess = True)
    if not result:
        raise Exception("solvePnP failed!")

    return nps.glue(rvec.ravel(), tvec.ravel(), axis=-1)

def feature_matching__colmap(colmap_database_filename,
                             Nimages = None # None means "all the images"
                             ):

    r'''Read matches from a COLMAP database This isn't trivial to make, and I'm
not 100% sure what I did. I did "colmap gui" then "new project", then some
feature stuff. The output is only available in the opaque database with BLOBs
that this function tries to parse. I mostly reverse-engineered the format, but
there's documentation here:

  https://colmap.github.io/database.html

I can also do a similar thing using alicevision:

av=$HOME/debianstuff/AliceVision/build/Linux-x86_64
$av/aliceVision_cameraInit --defaultFieldOfView 80 --imageFolder ~/data/xxxxx/delta -o xxxxx.sfm
$av/aliceVision_featureExtraction -i xxxxx.sfm -o xxxxx-features
$av/aliceVision_featureMatching -i xxxxx.sfm -f xxxxx-features -o xxxxx-matches
$av/aliceVision_incrementalSfM  -i xxxxx.sfm -f xxxxx-features -m xxxxx-matches -o xxxxx-sfm-output/

    '''

    db = sqlite3.connect(colmap_database_filename)

    def parse_row(image_id, rows, cols, data):
        return np.frombuffer(data, dtype=np.float32).reshape(rows,cols)

    rows_keypoints = db.execute("SELECT * from keypoints")

    keypoints = [ parse_row(*row) for row in rows_keypoints ]

    rows_matches = db.execute("SELECT * from matches")

    def get_correspondences_from_one_image_pair(row):
        r'''Reports all the corresponding pixels in ONE pair of images

If we have N images, and all of them have some overlapping views, the we'll have
to make N*(N-1)/2 get_correspondences_from_one_image_pair() calls to get all the data

        '''

        pair_id, rows, cols, data = row

        def pair_id_to_image_ids(pair_id):
            r'''function from the docs

  https://colmap.github.io/database.html

It reports 1-based indices. It also looks wrong: dividing by 0x7FFFFFFF is
WEIRD. But I guess that's what they did...

            '''
            image_id2 = pair_id % 2147483647
            image_id1 = (pair_id - image_id2) // 2147483647
            return image_id1, image_id2
        def pair_id_to_image_ids__old(pair_id):
            r'''function I reverse-engineered It reports 0-based indices. Agrees
            with the one above for a few small numbers.

            '''
            i0 = (pair_id >> 31) - 1
            i1 = (pair_id & 0x7FFFFFFF) + i0
            return i0,i1

        i0,i1 = pair_id_to_image_ids(pair_id)
        # move their image indices to be 0-based
        i0 -= 1
        i1 -= 1

        # shape (Ncorrespondences, 2)
        # feature indices
        f01 = np.frombuffer(data, dtype=np.uint32).reshape(rows,cols)

        # shape (Nimages=2, Npoints, Nxy=2)
        q =  nps.cat(keypoints[i0][f01[:,0],:2],
                     keypoints[i1][f01[:,1],:2])
        # shape (Npoints, Nimages=2, Nxy=2)
        q = nps.xchg(q, 0,1)

        # Convert colmap pixels to mrcal pixels. Colmap has the image origin at
        # the top-left corner of the image, NOT at the center of the top-left
        # pixel:
        #
        # https://colmap.github.io/database.html?highlight=convention#keypoints-and-descriptors
        q -= 0.5

        return (i0,
                i1,
                f01,
                q.astype(float))

    point_indices__from_image = dict()
    ipoint_next               = 0

    def retrieve_cache(i,f):
        nonlocal point_indices__from_image

        # Return the point-index cache for image i. This indexes on feature
        # indices f. It's possible that f contains out-of-bounds indices. In
        # that case I grow my cache.
        if i not in point_indices__from_image:
            # New never-before-seen image. I initialize the cache, with each
            # value being <0: I've never seen any of these point. I start out
            # with an arbitrary number of features: 100
            point_indices__from_image[i] = -np.ones((100,), dtype=int)

        idx = point_indices__from_image[i]

        maxf = np.max(f)
        if maxf >= idx.size:
            # I grow to double what I need now. I waste memory in order to
            # reallocate less often
            idx_new = -np.ones((maxf*2,), dtype=int)
            idx_new[:idx.size] = idx
            point_indices__from_image[i] = idx_new
            idx = point_indices__from_image[i]

        # I now have a big-enough cache. The caller can use it
        return idx

    def look_up_point_index(i0, i1, f01):
        # i0, i1 are integer scalars: which images we're looking at
        #
        # f01 has shape (Npoints, Nimages=2). Integers identifying the point IN
        # EACH IMAGE
        f0 = f01[:,0]
        f1 = f01[:,1]

        # Each ipt_idx is a 1D numpy array. It's indexed by the f0,f1 feature
        # indices. The value is a point index, or <0 if it hasn't been seen yet
        ipt_idx0 = retrieve_cache(i0, f0)
        ipt_idx1 = retrieve_cache(i1, f1)

        # The point indices
        idx0 = ipt_idx0[f0]
        idx1 = ipt_idx1[f1]

        # - If an index is found in only one cache, I add it to the other cache.
        #
        # - If an index is found in BOTH caches, it should refer to the same
        #   point. If it doesn't I need to rethink this
        #
        # - If an index is not found in either cache, I make a new point index,
        #   and add it to both caches
        idx_found0 = idx0 >= 0
        idx_found1 = idx1 >= 0

        if np.any(idx_found0 * idx_found1):
            raise Exception("I encountered an observation, and I've seen both of these points already. I think this shouldn't happen? It did happen, though. So I should make sure that both of these refer to the same point. I'm not implementing that yet because I don't think this will ever actually happen")

        # copy the found indices
        idx1[idx_found0] = idx0[idx_found0]
        idx0[idx_found1] = idx1[idx_found1]

        # Now I make new indices for newly-observed points
        nonlocal ipoint_next
        idx_not_found = ~idx_found0 * ~idx_found1
        Npoints_new = np.count_nonzero(idx_not_found)
        idx0[idx_not_found] = np.arange(Npoints_new) + ipoint_next
        idx1[idx_not_found] = np.arange(Npoints_new) + ipoint_next
        ipoint_next += Npoints_new

        # Everything is cached and done. I can look up the point indices for
        # this call
        if np.any(idx0 - idx1):
            raise Exception("Point index mismatch. This is a bug")
        return idx0



    Nobservations_max = 1000000
    indices_point_camintrinsics_camextrinsics_pool = \
        np.zeros((Nobservations_max,3),
                 dtype = np.int32)
    observations_pool = \
        np.ones((Nobservations_max,3),
                dtype = float)
    Nobservations = 0

    for row in rows_matches:
        # i0, i1 are scalars
        # q has shape (Npoints, Nimages=2, Nxy=2)
        # f01 has shape (Npoints, Nimages=2)
        i0,i1,f01,q = get_correspondences_from_one_image_pair(row)

        if Nimages is not None and \
           (i0 >= Nimages or \
            i1 >= Nimages):
            continue

        # shape (Npoints,)
        ipt = look_up_point_index(i0,i1,f01)

        Nobservations_here = q.shape[-3] * 2
        if Nobservations + Nobservations_here > Nobservations_max:
            raise Exception("Exceeded Nobservations_max")

        # pixels observed from the first camera
        iobservation0 = range(Nobservations,
                              Nobservations+Nobservations_here//2)
        # pixels observed from the second camera
        iobservation1 = range(Nobservations+Nobservations_here//2,
                              Nobservations+Nobservations_here)

        indices_point_camintrinsics_camextrinsics_pool[iobservation0, 0] = ipt
        indices_point_camintrinsics_camextrinsics_pool[iobservation1, 0] = ipt

        # [iobservation, 1] is already 0: I'm moving around a single camera

        # -1 because camera0 defines my coord system, and is not present in the
        # -extrinsics vector
        indices_point_camintrinsics_camextrinsics_pool[iobservation0, 2] = i0-1
        indices_point_camintrinsics_camextrinsics_pool[iobservation1, 2] = i1-1

        observations_pool[iobservation0, :2] = q[:,0,:]
        observations_pool[iobservation1, :2] = q[:,1,:]
        # [iobservation1,2] already has weight=1.0

        Nobservations += Nobservations_here

    indices_point_camintrinsics_camextrinsics = \
        indices_point_camintrinsics_camextrinsics_pool[:Nobservations]
    observations = \
        observations_pool[:Nobservations]

    # I resort the observations to cluster them by points, as (currently; for
    # now?) required by mrcal. I want a stable sort to preserve the camera
    # sorting order within each point. This isn't strictly required, but makes
    # it easier to think about
    iobservations = \
        np.argsort(indices_point_camintrinsics_camextrinsics[:,0],
                   kind = 'stable')

    indices_point_camintrinsics_camextrinsics[:] = \
        indices_point_camintrinsics_camextrinsics[iobservations,...]
    observations[:] = \
        observations[iobservations,...]

    return \
        indices_point_camintrinsics_camextrinsics, \
        observations

def feature_matching__opencv(image0_decimated, image1_decimated):
    # shape (Hdecimated,Wdecimated,2)
    flow_decimated = \
        cv2.calcOpticalFlowFarneback(image0_decimated, image1_decimated,
                                     flow       = None, # for in-place output
                                     pyr_scale  = 0.5,
                                     levels     = 3,
                                     winsize    = 15,
                                     iterations = 3,
                                     poly_n     = 5,
                                     poly_sigma = 1.2,
                                     flags      = 0# cv2.OPTFLOW_USE_INITIAL_FLOW
                                     )

    if 0:
        plot_flow(image0_decimated, flow_decimated,
                  decimation_extra_plot,
                  hardcopy = f"{outdir}/flow{i_image:03d}.png")

    keypoints0, descriptors0 = feature_finder.detectAndCompute(image0_decimated, None)
    keypoints1, descriptors1 = feature_finder.detectAndCompute(image1_decimated, None)
    matches = matcher.match(descriptors0, descriptors1)

    # shape (Nmatches, Npair=2, Nxy=2)
    qall_decimated = nps.cat(*[np.array((keypoints0[m.queryIdx].pt,
                                         keypoints1[m.trainIdx].pt)) \
                               for m in matches])

    i_match_valid = \
        np.array([i for i in range(len(matches)) \
                  if match_looks_valid(qall_decimated[i],
                                       matches[i],
                                       flow_decimated[int(round(qall_decimated[i][0,1])),
                                                      int(round(qall_decimated[i][0,0]))]
                                       )])

    if len(i_match_valid) < 10:
        raise Exception(f"Too few valid features found: N = {len(i_match_valid)}")

    return \
        decimation * qall_decimated[i_match_valid]

def show_matched_features(image0_decimated, image1_decimated, q):
    ##### feature-matching visualizations
    if 0:
        # Plot two sets of points: one for each image in the pair
        gp.plot( # shape (Npair=2, Nmatches, Nxy=2)
                 nps.xchg(q,0,1),
                 legend = np.arange(2),
                 _with='points',
                 tuplesize=-2,
                 square=1,
                 _xrange = (0,W),
                 _yrange = (H,0),
                 hardcopy='/tmp/tst.gp')
        sys.exit()
    elif 0:
        # one plot, with connected lines: vertical stacking
        gp.plot( (nps.glue(image0_decimated,image1_decimated,
                           axis=-2),
                  dict( _with     = 'image', \
                        tuplesize = 3 )),

                 (q/decimation + np.array(((0,0), (0,image0_decimated.shape[-2])),),
                  dict( _with     = 'lines',
                        tuplesize = -2)),

                 _set = ('xrange noextend',
                         'yrange noextend reverse',
                         'palette gray'),
                 square=1,
                 hardcopy='/tmp/tst.gp')
        sys.exit()
    elif 0:
        # one plot, with connected lines: horizontal stacking
        gp.plot( (nps.glue(image0_decimated,image1_decimated,
                           axis=-1),
                  dict( _with     = 'image', \
                        tuplesize = 3 )),

                 (q/decimation + np.array(((0,0), (image0_decimated.shape[-1],0)),),
                  dict( _with     = 'lines',
                        tuplesize = -2)),

                 _set = ('xrange noextend',
                         'yrange noextend reverse',
                         'palette gray'),
                 square=1,
                 hardcopy='/tmp/tst.gp')
        sys.exit()
    elif 0:
        # two plots. Discrete points
        images_decimated = (image0_decimated,
                            image1_decimated)

        g = [None] * 2
        pids = set()
        for i in range(2):
            g[i] = gp.gnuplotlib(_set = ('xrange noextend',
                                         'yrange noextend reverse',
                                         'palette gray'),
                                 square=1)
            g[i].plot( (images_decimated[i],
                        dict( _with     = 'image', \
                              tuplesize = 3 )),
                       (q[:,(i,),:]/decimation,
                        dict( _with     = 'points',
                              tuplesize = -2,
                              legend    = i_match_valid)))

            pid = os.fork()
            if pid == 0:
                # child
                g[i].wait()
                sys.exit(0)

            pids.add(pid)

        # wait until all the children finish
        while len(pids):
            pid,status = os.wait()
            if pid in pids:
                pids.remove(pid)

def get_observation_pair(i0, i1,
                         indices_point_camintrinsics_camextrinsics,
                         observations):

    # i0, i1 are indexed from -1: these are the camextrinsics indices
    if i0+1 != i1:
        raise Exception("get_observation_pair() currently only works for consecutive indices")

    # The data is clumped by points. I'm looking at
    # same-point-consecutive-camera observations, so they're guaranteed to
    # appear consecutively
    mask_cam0 = indices_point_camintrinsics_camextrinsics[:,2] == i0
    mask_cam0[-1] = False # I'm going to be looking at the "next" row, so I
                          # ignore the last row, since there's no "next" one
                          # after it

    idx_cam0 = np.nonzero(mask_cam0)[0]
    row_cam0 = \
        indices_point_camintrinsics_camextrinsics[idx_cam0]
    row_next = \
        indices_point_camintrinsics_camextrinsics[idx_cam0+1]

    # I care about corresponding rows that represent the same point and my two
    # cameras
    idx_cam0_selected = \
        idx_cam0[(row_cam0[:,0] == row_next[:,0]) * (row_next[:,2] == i1)]

    q0 = observations[idx_cam0_selected,   :2]
    q1 = observations[idx_cam0_selected+1, :2]

    return q0,q1

def solve(indices_point_camintrinsics_camextrinsics,
          observations,
          Nimages):

    # hard-coded known at-infinity point seen in the first two cameras
    q0_infinity = np.array((1853,1037), dtype=float)
    q1_infinity = np.array((1920,1039), dtype=float)
    observations_fixed = nps.glue( q0_infinity,
                                   q1_infinity,
                                   axis = -2 )
    p_infinity = \
        10000 *  \
        mrcal.unproject(q0_infinity, *model.intrinsics(),
                        normalize = True)
    indices_point_camintrinsics_camextrinsics_fixed = \
        np.array((( 0, 0, -1),
                  ( 0, 0,  0),),
                 dtype = np.int32)
    # weights
    observations_fixed = nps.glue( observations_fixed,
                                   np.ones((2,1), dtype=np.int32),
                                   axis = -1)

    # i0 is indexed from -1: it's a camextrinsics index
    #
    # This is a relative array:
    # [ rt10 ]
    # [ rt21 ]
    # [ rt32 ]
    # [ .... ]
    rt_cam_camprev = \
        np.array([seed_rt10_pair( *get_observation_pair(i0, i0+1,
                                                        indices_point_camintrinsics_camextrinsics,
                                                        observations) ) \
                  for i0 in range(-1,Nimages-2)])

    # Make an absolute extrinsics array:
    # [ rt10 ]
    # [ rt20 ]
    # [ rt30 ]
    # [ .... ]
    rt_cam_ref = np.zeros(rt_cam_camprev.shape, dtype=float)
    rt_cam_ref[0] = rt_cam_camprev[0]
    for i in range(1,len(rt_cam_camprev)):
        rt_cam_ref[i] = mrcal.compose_rt( rt_cam_camprev[i],
                                          rt_cam_ref[i-1] )

    optimization_inputs = \
        dict( intrinsics            = nps.atleast_dims(model.intrinsics()[1], -2),
              extrinsics_rt_fromref = nps.atleast_dims(rt_cam_ref, -2),

              observations_point_triangulated                        = observations,
              indices_point_triangulated_camintrinsics_camextrinsics = indices_point_camintrinsics_camextrinsics,

              points                                    = nps.atleast_dims(p_infinity, -2),
              indices_point_camintrinsics_camextrinsics = indices_point_camintrinsics_camextrinsics_fixed,
              observations_point                        = observations_fixed,
              Npoints_fixed                             = 1,
              point_min_range                           = 1.,
              point_max_range                           = 20000.,



              lensmodel                           = model.intrinsics()[0],
              imagersizes                         = nps.atleast_dims(model.imagersize(), -2),
              do_optimize_intrinsics_core         = False,
              do_optimize_intrinsics_distortions  = False,
              do_optimize_extrinsics              = True,
              do_optimize_frames                  = True,
              do_apply_outlier_rejection          = True,
              do_apply_regularization             = True,
              do_apply_regularization_unity_cam01 = True,
              verbose                             = True)

    filename = '/tmp/geometry-seed.gp'
    mrcal.show_geometry( nps.glue(mrcal.identity_rt(),
                                  optimization_inputs['extrinsics_rt_fromref'],
                                  axis = -2),
                         cameranames = np.arange(Nimages),
                         hardcopy    = filename)
    print(f"Wrote '{filename}'")

    stats = mrcal.optimize(**optimization_inputs)

    filename = '/tmp/geometry-solve.gp'
    mrcal.show_geometry( nps.glue(mrcal.identity_rt(),
                                  optimization_inputs['extrinsics_rt_fromref'],
                                  axis = -2),
                         cameranames = np.arange(Nimages),
                         hardcopy    = filename)
    print(f"Wrote '{filename}'")

    x = stats['x']
    return x, optimization_inputs

def show_solution(optimization_inputs, Nimages):

    ply_header = rb'''ply
format binary_little_endian 1.0
element vertex NNNNNNN
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar alpha
end_header
'''

    Npoints_pointcloud = 0

    filename_point_cloud = "/tmp/points.ply"
    with open(filename_point_cloud, 'wb') as f:

        f.write(ply_header)

        # Here I only look at consecutive image pairs, even though the
        # optimization looked at ALL the pairs
        for i0 in range(-1, Nimages-2):
            i1 = i0+1

            q0, q1 = \
                get_observation_pair(i0, i1,
                                     optimization_inputs['indices_point_triangulated_camintrinsics_camextrinsics'],
                                     optimization_inputs['observations_point_triangulated'])

            if i0 < 0:
                rt_0r = mrcal.identity_rt()
                rt_01 = mrcal.invert_rt(optimization_inputs['extrinsics_rt_fromref'][i1])
            else:
                rt_0r = optimization_inputs['extrinsics_rt_fromref'][i0]
                rt_1r = optimization_inputs['extrinsics_rt_fromref'][i1]

                rt_01 = mrcal.compose_rt( rt_0r,
                                          mrcal.invert_rt(rt_1r) )

            v0 = mrcal.unproject(q0, *model.intrinsics())
            v1 = mrcal.unproject(q1, *model.intrinsics())

            plocal0 = \
                mrcal.triangulate_leecivera_mid2(v0, v1,
                                                 v_are_local = True,
                                                 Rt01        = mrcal.Rt_from_rt(rt_01))

            r = nps.mag(plocal0)
            index_good_triangulation = r > 0

            q0_r_recip = nps.glue(q0[index_good_triangulation],
                                  nps.transpose(1./r[index_good_triangulation]),
                                  axis = -1)

            filename_overlaid_points = f'/tmp/overlaid-points-{i0+1}.pdf'
            gp.plot(q0_r_recip,
                    tuplesize = -3,
                    _with     = 'points pt 7 ps 0.5 palette',
                    square    = 1,
                    _xrange   = (0,W),
                    _yrange   = (H,0),
                    rgbimage  = image_filename[i0+1],
                    hardcopy  = filename_overlaid_points,
                    cbmax     = 0.1)
            print(f"Wrote '{filename_overlaid_points}'")


            ######### point cloud
            #### THIS IS WRONG: I report a separate point in each consecutive
            #### triangulation, so if I tracked a feature over N frames, instead
            #### of reporting one point for that feature, I'll report N-1 of
            #### them

            # I'm including the alpha byte to align each row to 16 bytes.
            # Otherwise I have unaligned 32-bit floats. I don't know for a fact
            # that this breaks anything, but it feels like it would maybe.
            N = np.count_nonzero(index_good_triangulation)
            binary_ply = np.empty( (N,),
                                   dtype = np.dtype([ ('xyz',np.float32,3), ('rgba', np.uint8, 4) ]))
            binary_ply['xyz'] = mrcal.transform_point_rt(mrcal.invert_rt(rt_0r),
                                                         plocal0[index_good_triangulation])

            image = cv2.imread(image_filename[i0+1])
            if not (len(image.shape) == 3 and image.shape[-1] == 3):
                raise Exception("I only support color RGB images. If you need it, YOU implement the grayscale ones")

            i = (q0[index_good_triangulation] + 0.5).astype(int)
            bgr = image[i[:,1], i[:,0]]

            binary_ply['rgba'][:,0] = bgr[:,2]
            binary_ply['rgba'][:,1] = bgr[:,1]
            binary_ply['rgba'][:,2] = bgr[:,0]
            binary_ply['rgba'][:,3] = 255

            binary_ply.tofile(f)

            Npoints_pointcloud += N


    # I wrote the point cloud file with an unknown number of points. Now that I
    # have the count, I go back, and fill it in.
    import mmap
    with open(filename_point_cloud, 'r+b') as f:
        m = mmap.mmap(f.fileno(), 0)

        i_placeholder_start = ply_header.find(b'NNN')
        placeholder_width   = ply_header[i_placeholder_start:].find(b'\n')
        i_placeholder_end   = i_placeholder_start + placeholder_width

        m[i_placeholder_start:i_placeholder_end] = \
            '{:{width}d}'.format(Npoints_pointcloud, width=placeholder_width).encode()

        m.close()

    print(f"Wrote '{filename_point_cloud}'")

def write_model(filename, model):
    print(f"Writing '{filename}'")
    model.write(filename)





image_directory          = "/home/dima/data/xxxxx/delta/*.jpg"
outdir                   = "/tmp"
decimation               = 20
decimation_extra_plot    = 5
model_filename           = "/home/dima/xxxxx-sfm/cam.cameramodel"
colmap_database_filename = '/tmp/xxxxx.db'



model = mrcal.cameramodel(model_filename)
W,H   = model.imagersize()

feature_finder = cv2.ORB_create()
matcher        = cv2.BFMatcher(cv2.NORM_HAMMING,
                               crossCheck = True)

image_filename = sorted(glob.glob(image_directory))

image0,image0_decimated = imread(image_filename[0], decimation)






Nimages = 3


# q.shape = (Npoints, Nimages=2, Nxy=2)
if 0:
    image1,image1_decimated = imread(f, decimation)
    q = feature_matching__opencv(image0_decimated, image1_decimated)
    show_matched_features(image0_decimated, image1_decimated, q)
else:
    indices_point_camintrinsics_camextrinsics, \
    observations = \
        feature_matching__colmap(colmap_database_filename,
                                 Nimages)

x,optimization_inputs = solve(indices_point_camintrinsics_camextrinsics,
                              observations,
                              Nimages)

model0 = mrcal.cameramodel(model)
model0.extrinsics_rt_fromref(np.zeros((6,), dtype=float))
write_model("/tmp/xxxxx-cam0.cameramodel", model0)

for i in range(1,Nimages):
    rt_cam_ref = optimization_inputs['extrinsics_rt_fromref'][i-1]

    model1 = mrcal.cameramodel(model)
    model1.extrinsics_rt_fromref(rt_cam_ref)

    write_model(f"/tmp/xxxxx-cam{i}.cameramodel", model1)

show_solution(optimization_inputs, Nimages)

import IPython
IPython.embed()
sys.exit()




try:
    image0_decimated = image1_decimated
except:
    # if I'm looking at cached features, I never read any actual images
    pass

