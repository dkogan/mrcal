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


def imread(filename, decimation):
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, image[::decimation, ::decimation]

def plot_flow(filename,
              image, flow, decimation):
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
             hardcopy = filename)

    print(f"Wrote {filename}")


def match_looks_valid(q, match, flow):

    flow_observed = q[1] - q[0]
    flow_err_sq   = nps.norm2(flow_observed - flow)

    return \
        match.distance < 500 and \
        flow_err_sq    < 10*10


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

    result,rvec,tvec = \
        cv2.solvePnP( mrcal.unproject(q[:,0,:], *intrinsics),
                      np.ascontiguousarray(q[:,1,:]),
                      camera_matrix,
                      distortions,
                      useExtrinsicGuess = False)
    if not result:
        raise Exception("solvePnP failed!")

    return nps.glue(rvec.ravel(), tvec.ravel(), axis=-1)



directory             = "/mnt/fatty/home/dima/data/xxxxx/delta/*.jpg"
outdir                = "/tmp"
decimation            = 20
decimation_extra_plot = 5
Nimages               = 10 #None # all of them
model_filename        = "/mnt/fatty/home/dima/xxxxx-sfm/cam.cameramodel"



model = mrcal.cameramodel(model_filename)

feature_finder = cv2.ORB_create()
matcher        = cv2.BFMatcher()

l = sorted(glob.glob(directory))


image0,image0_decimated = imread(l[0], decimation)

i=0
for f in l[1:Nimages]:
    image1,image1_decimated = imread(f, decimation)

    # shape (Hdecimated,Wdecimated,2)
    flow = -cv2.calcOpticalFlowFarneback(image0_decimated, image1_decimated,
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
        plot_flow(f"{outdir}/flow{i:03d}.png",
                  image0_decimated, flow,
                  decimation_extra_plot)

    keypoints0, descriptors0 = feature_finder.detectAndCompute(image0_decimated, None)
    keypoints1, descriptors1 = feature_finder.detectAndCompute(image1_decimated, None)

    matches = matcher.match(descriptors0, descriptors1)

    # shape (Nmatches, Npair=2, Nxy=2)
    qall_decimated = nps.cat(*[np.array((keypoints0[m.queryIdx].pt,
                                         keypoints1[m.trainIdx].pt)) \
                               for m in matches])

    q = \
        decimation * \
        qall_decimated[([i for i in range(len(matches)) \
               if match_looks_valid(qall_decimated[i],
                                    matches[i],
                                    flow[int(round(qall_decimated[i][0,1])),
                                         int(round(qall_decimated[i][0,0]))]
                                    )])]

    if 0:
        gp.plot(nps.xchg(q,0,1), _with='points', tuplesize=-2, square=1)
    if 0:
        gp.plot( (nps.glue(image0_decimated,image1_decimated,axis=-1),
                  dict( _with     = 'image', \
                        tuplesize = 3 )),

                 (q*decimation + np.array(((0,0), (image0_decimated.shape[-1],0)),),
                  dict( _with     = 'lines',
                        tuplesize = -2)),

                 _set = ('xrange noextend',
                         'yrange noextend reverse',
                         'palette gray'),
                 square=1,
                 wait = 1)


    rt10 = seed_pose(model.intrinsics(), q)

    import IPython
    IPython.embed()
    sys.exit()



    model0 = model
    model.extrinsics_rt_fromref(np.zeros((6,), dtype=float))

    model1 = mrcal.cameramodel(model)
    model1.extrinsics_rt_fromref(rt10)

    p0 = mrcal.triangulate( q, (model0,model1) )
    p1 = mrcal.transform_point_rt(rt10, p0)

    # shape (Npoints*2,3). Each row is (qx, qy, weight). All weights are 1.0
    observations_point = np.ones( (len(q)*2, 3), dtype=float)
    observations_point[:,:2] = nps.clump(q, n=2)

    # shape (Npoints,2,3)
    indices_point_camintrinsics_camextrinsics = np.zeros((len(q),2, 3), dtype=np.int32)
    indices_point_camintrinsics_camextrinsics[:,0,0] = np.arange(len(q))
    indices_point_camintrinsics_camextrinsics[:,1,0] = np.arange(len(q))
    indices_point_camintrinsics_camextrinsics[:,0,2] = -1
    indices_point_camintrinsics_camextrinsics[:,0,1] = 0
    # shape (Npoints*2,3)
    indices_point_camintrinsics_camextrinsics = nps.clump(indices_point_camintrinsics_camextrinsics, n=2)

    optimization_inputs = \
        dict( lensmodel                                 = model.intrinsics()[0],
              intrinsics                                = nps.atleast_dims(model.intrinsics()[1], -2),
              extrinsics_rt_fromref                     = nps.atleast_dims(rt10, -2),
              frames_rt_toref                           = None,
              points                                    = p0,
              point_min_range                           = 0.5,
              point_max_range                           = 1000.0,
              observations_board                        = None,
              indices_frame_camintrinsics_camextrinsics = None,
              observations_point                        = observations_point,
              indices_point_camintrinsics_camextrinsics = indices_point_camintrinsics_camextrinsics,
              Npoints_fixed                             = 1, # one fixed point to establish scale
              imagersizes                               = nps.atleast_dims(model.imagersize(), -2),
              verbose                                   = True,
              do_optimize_extrinsics                    = True,
              do_optimize_frames                        = True,
              do_optimize_intrinsics_core               = False,
              do_optimize_intrinsics_distortions        = False,
              do_apply_regularization                   = False,
              do_apply_outlier_rejection                = False)

    mrcal.optimize(**optimization_inputs)

    p = optimization_inputs['points']
    # p[ nps.mag(p)>100] *= 0
    mrcal.show_geometry(optimization_inputs['extrinsics_rt_fromref'],
                        points      = p,
                        cameranames = np.arange(2),
                        axis_scale  = 1.0,
                        _set        = 'view 180,90,2.8')

    import IPython
    IPython.embed()
    sys.exit()




    image0_decimated = image1_decimated
    i += 1
