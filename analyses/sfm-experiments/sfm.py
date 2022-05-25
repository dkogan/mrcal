#!/usr/bin/python3

r'''
s=5;
mkdir -p scaled$s;
parallel convert '{}' -scale $s'%' scaled$s/'{/}' ::: delta/*.jpg
'''

import sys
import numpy as np
import numpysane as nps
import gnuplotlib as gp
import cv2
import glob

directory = "scaled5/*.jpg"
D = 5
do_sparse_features = True

l = sorted(glob.glob(directory))

frame1 = cv2.imread(l[0])
prvs   = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

H,W = frame1.shape[:2]




if do_sparse_features:
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15, 15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    p0 = cv2.goodFeaturesToTrack(prvs, mask = None, **feature_params)
    #mask = np.zeros_like(old_frame)



i=0

for f in l[1:]:
    frame2 = cv2.imread(f)

    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    filename = f"/tmp/flow{i:03d}.png"

    if not do_sparse_features:
        # shape (H,W,2)
        flow = -cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # each has shape (H/D,W/D)
        xx,yy = np.meshgrid(np.arange(0,W,D),
                            np.arange(0,H,D))

        # shape (H,W,4)
        vectors = \
            nps.glue( nps.dummy(xx,-1),
                      nps.dummy(yy,-1),
                      flow[0:H:D, 1:W:D, :],
                      axis = -1);
        vectors = nps.clump(vectors, n=2)

    else:
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(prvs, next, p0, None, **lk_params)
        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]


        d = good_new - good_old

        vectors = nps.glue(good_new,d, axis=-1)
        p0 = good_new.reshape(-1, 1, 2)


    gp.plot( (next, dict(_with='image',
                         tuplesize = 3)),
             (vectors,
              dict(tuplesize=-4, _with='vectors lc "red"',)),

             _set  = 'palette gray',
             unset = 'colorbox',
             square = True,
             _xrange = (0,W),
             _yrange = (H,0),
             hardcopy = filename)



    print(f"Wrote {filename}")
    prvs = next
    i += 1
