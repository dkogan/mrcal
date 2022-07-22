#!/usr/bin/python3

r'''
'''

import sys
import numpy as np
import numpysane as nps
import gnuplotlib as gp
import cv2
import glob
import mrcal
import re
import os
import sqlite3


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

def seed_pose(intrinsics, q):
    r'''
    q.shape is (Npairs, Npair=2, Nxy=2)'''
    lensmodel,intrinsics_data = intrinsics
    if not re.match("LENSMODEL_(OPENCV|PINHOLE)", lensmodel):
        raise Exception("This assumes a pinhole or opencv model. You have something else, and you should reproject to pinhole")
    fx,fy,cx,cy = intrinsics_data[:4]
    distortions = intrinsics_data[4:]
    camera_matrix = np.array((( fx,  0, cx),
                              ( 0,  fy, cy),
                              ( 0,   0, 1.)))

    # arbitrary scale
    p0 = mrcal.unproject(q[:,0,:], *intrinsics)

    result,rvec,tvec = \
        cv2.solvePnP( p0,
                      np.ascontiguousarray(q[:,1,:]),
                      camera_matrix,
                      distortions,
                      rvec = (0,0,0),
                      tvec = (0,0,0),
                      useExtrinsicGuess = True)
    if not result:
        raise Exception("solvePnP failed!")

    rt10 = nps.glue(rvec.ravel(), tvec.ravel(), axis=-1)
    return rt10

def feature_matching__colmap(colmap_database_filename):

    r'''Read matches from a COLMAP database This isn't trivial to make, and I'm
not 100% sure what I did. I did "colmap gui" then "new project", then some
feature stuff. The output is only available in the opaque database with BLOBs
that this function tries to parse

    '''
    # 2324.100870044619,2323.6136731231927,2028.6718475216383,1535.3190950118333,0.5587497543608404,-0.11460295129080221,0.0001822218980685378,4.018761940244227e-05,0.06818259596025401,0.6494564849581099,-0.1800063119555238,0.05230106331813401
    db = sqlite3.connect(colmap_database_filename)

    def parse_row(image_id, rows, cols, data):
        return np.frombuffer(data, dtype=np.float32).reshape(rows,cols)

    rows_keypoints = db.execute("SELECT * from keypoints")

    keypoints = [ parse_row(*row) for row in rows_keypoints ]

    rows_matches = db.execute("SELECT * from matches")

    # for row in rows_matches:
    #     pair_id, rows, cols, data = row
    #     print(pair_id)
    #     # matches = np.frombuffer(data, dtype=np.uint32).reshape(rows,cols)
    # sys.exit()



    def get_matches(row):
        pair_id, rows, cols, data = row

        def pair_id_to_image_ids(pair_id):
            r'''function from the docs
            https://colmap.github.io/database.html
            It reports 1-based indices
'''
            image_id2 = pair_id % 2147483647
            image_id1 = (pair_id - image_id2) // 2147483647
            return image_id1, image_id2
        def pair_id_to_image_ids__old(pair_id):
            r'''function I reverse-engineered It reports 0-based indices. Agrees
            with the one above, apparently

            '''
            i0 = (pair_id >> 31) - 1
            i1 = (pair_id & 0x7FFFFFFF) + i0
            return i0,i1

        i0,i1 = pair_id_to_image_ids(pair_id)

        # move their image indices to be 0-based
        i0 -= 1
        i1 -= 1


        matches = np.frombuffer(data, dtype=np.uint32).reshape(rows,cols)
        f0,f1 = nps.transpose(matches)

        # shape (Nimages=2, Npoints, Nxy=2)
        q =  nps.cat(keypoints[i0][f0,:2],
                     keypoints[i1][f1,:2])
        # shape (Npoints, Nimages=2, Nxy=2)
        q = nps.xchg(q, 0,1)

        # Convert colmap pixels to mrcal pixels. Colmap has the image origin at
        # the top-left corner of the image, NOT at the center of the top-left
        # pixel:
        #
        # https://colmap.github.io/database.html?highlight=convention#keypoints-and-descriptors
        q -= 0.5

        return q.astype(float)

        if 0:
            image0 = cv2.imread(l[i0], cv2.IMREAD_GRAYSCALE)
            image1 = cv2.imread(l[i1], cv2.IMREAD_GRAYSCALE)
            image01 = nps.glue(image0,image1, axis=-1)
            cv2.imwrite(f'/tmp/tst-{i0}-{i1}.jpg', image01)
            image_shape = image0.shape
        else:
            image_shape = (3000,4096)

        gp.plot( (q + np.array(((0,0), (image_shape[-1],0)),),
                  dict( _with     = 'linespoints pt 2',
                        legend    = np.arange(q.shape[0]),
                        tuplesize = -2)),

                 _set = ('xrange noextend',
                         'yrange noextend reverse',
                         'palette gray'),
                 square=1,
                 rgbimage = f'/tmp/tst-{i0}-{i1}.jpg',
                 hardcopy=f'/tmp/tst-{i0}-{i1}.gp')
        print(f"wrote /tmp/tst-{i0}-{i1}.gp")

        gp.plot( (q[:,(0,),:],
                  dict( _with     = 'linespoints pt 2',
                        legend    = np.arange(q.shape[0]),
                        tuplesize = -2)),

                 _set = ('xrange noextend',
                         'yrange noextend reverse',
                         'palette gray'),
                 square=1,
                 rgbimage = l[i0],
                 hardcopy=f'/tmp/tst-{i0}-{i1}--0.gp')
        print(f"wrote /tmp/tst-{i0}-{i1}--0.gp")
        gp.plot( (q[:,(1,),:],
                  dict( _with     = 'linespoints pt 2',
                        legend    = np.arange(q.shape[0]),
                        tuplesize = -2)),

                 _set = ('xrange noextend',
                         'yrange noextend reverse',
                         'palette gray'),
                 square=1,
                 rgbimage = l[i1],
                 hardcopy=f'/tmp/tst-{i0}-{i1}--1.gp')
        print(f"wrote /tmp/tst-{i0}-{i1}--1.gp")


    if 1:
        for row in rows_matches:
            # images 0,1
            return get_matches(row)
    else:
        row = next(rows_matches)
        row = next(rows_matches)
        row = next(rows_matches)
        for row in rows_matches:
            # images 0,4
            return get_matches(row)

    return keypoints, matches

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

def solve(q):
    Npoints = len(q)
    Ncameras = 2 # We're looking at sequential pairs

    q0_infinity = np.array((1853,1037), dtype=float)
    q1_infinity = np.array((1920,1039), dtype=float)
    q4_infinity = np.array((2129,1025), dtype=float)
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




    rt10 = seed_pose(model.intrinsics(), q)

    # shape (Npoints*Ncameras,2)
    observations = nps.clump(q, n=2)

    # shape (Npoints,Ncameras,3)
    indices_point_camintrinsics_camextrinsics = np.zeros((Npoints, Ncameras, 3), dtype=np.int32)
    indices_point_camintrinsics_camextrinsics[:,0,0] = np.arange(Npoints)
    indices_point_camintrinsics_camextrinsics[:,1,0] = np.arange(Npoints)
    indices_point_camintrinsics_camextrinsics[:,0,1] = 0
    indices_point_camintrinsics_camextrinsics[:,1,1] = 0
    indices_point_camintrinsics_camextrinsics[:,0,2] = -1
    indices_point_camintrinsics_camextrinsics[:,1,2] = 0
    # shape (Npoints*Ncameras,3)
    indices_point_camintrinsics_camextrinsics = nps.clump(indices_point_camintrinsics_camextrinsics, n=2)


    # Add weight column. All weights are 1.0
    observations_triangulated = nps.glue(observations,
                                         np.ones(observations.shape[:-1] + (1,),
                                                 dtype = np.float32),
                                         axis = -1)

    optimization_inputs = \
        dict( intrinsics            = nps.atleast_dims(model.intrinsics()[1], -2),
              extrinsics_rt_fromref = nps.atleast_dims(rt10, -2),

              observations_point_triangulated                        = observations_triangulated,
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

    stats = mrcal.optimize(**optimization_inputs)
    x = stats['x']
    return x, optimization_inputs





directory             = "/home/dima/data/xxxxx/delta/*.jpg"
outdir                = "/tmp"
decimation            = 20
decimation_extra_plot = 5
Nimages               = 3 #None # all of them
model_filename        = "/home/dima/xxxxx-sfm/cam.cameramodel"
colmap_database_filename = '/tmp/xxxxx.db'



model = mrcal.cameramodel(model_filename)
W,H   = model.imagersize()

feature_finder = cv2.ORB_create()
matcher        = cv2.BFMatcher(cv2.NORM_HAMMING,
                               crossCheck = True)

l = sorted(glob.glob(directory))

image0,image0_decimated = imread(l[0], decimation)

i_image=0
for f in l[1:Nimages]:

    image1,image1_decimated = imread(f, decimation)

    # q.shape = (Npoints, Nimages=2, Nxy=2)
    if 0: q = feature_matching__opencv(image0_decimated, image1_decimated)
    else: q = feature_matching__colmap(colmap_database_filename)


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



    x,optimization_inputs = solve(q)

    # p,x,j,f = mrcal.optimizer_callback(**optimization_inputs)
    # gp.plot(np.abs(x), _with='points')

    # import IPython
    # IPython.embed()
    # sys.exit()

    rt10 = optimization_inputs['extrinsics_rt_fromref'][0]

    model0 = mrcal.cameramodel(model)
    model0.extrinsics_rt_fromref(np.zeros((6,), dtype=float))

    model1 = mrcal.cameramodel(model)
    model1.extrinsics_rt_fromref(rt10)

    model0.write("/tmp/xxxxx-cam0.cameramodel")
    model1.write("/tmp/xxxxx-cam1.cameramodel")


    p = optimization_inputs['points']
    r = nps.mag(p)
    # p[ r>100] *= 0
    mrcal.show_geometry(optimization_inputs['extrinsics_rt_fromref'],
                        # points      = p,
                        cameranames = np.arange(2),
                        axis_scale  = 0.1,
                        _set        = 'view 180,90,2.8',
                        hardcopy    = '/tmp/geometry.gp')

    if 0:
        # discrete points
        gp.plot( (image0_decimated / 255. * np.max(r),
                  dict( _with     = 'image', \
                        tuplesize = 3 )),
                 (q[:,0,0]/decimation,
                  q[:,0,1]/decimation,
                  r,
                  dict( _with     = 'points pt 7 ps 2 palette',
                        tuplesize = 3)),
                 _set = ('xrange noextend',
                         'yrange noextend reverse'),
                 square=1,
                 wait = 1)


    # triangulated points
    if 1:
        q0 = q[:,0,:]
        q1 = q[:,1,:]

        v0 = mrcal.unproject(q0, *model.intrinsics())
        v1 = mrcal.unproject(q1, *model.intrinsics())

        p0 = \
            mrcal.triangulate_leecivera_mid2(v0, v1,
                                             v_are_local = True,
                                             Rt01        = mrcal.invert_Rt(mrcal.Rt_from_rt(rt10)))

        # gp.plot(p0,
        #         tuplesize = -3,
        #         _3d       = 1,
        #         square    = 1,
        #         _with     = 'dots')

        r = nps.mag(p0)
        index_good_triangulation = r > 0

        q0_r_recip = nps.glue(q0[index_good_triangulation],
                              nps.transpose(1./r[index_good_triangulation]),
                              axis = -1)

        gp.plot(q0_r_recip,
                tuplesize = -3,
                _with     = 'points pt 7 palette',
                square    = 1,
                _xrange   = (0,W),
                _yrange   = (H,0),
                rgbimage  = l[0],
                hardcopy  = '/tmp/tst.gp',
                cbmax     = 0.1)








    import IPython
    IPython.embed()
    sys.exit()




    image0_decimated = image1_decimated
    i_image += 1
