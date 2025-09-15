#!/usr/bin/python3

r'''
'''

import sys
import numpy as np
import numpysane as nps
import gnuplotlib as gp
import glob
import re
import os
import sqlite3
import pyopengv


sys.path[:0] = '/home/dima/projects/mrcal-2022-06--triangulated-solve/',
import mrcal
from mrcal.utils import _sorted_eig


np.set_printoptions(linewidth = 800000)


def imread(filename, decimation):
    image = mrcal.load_image(filename, bits_per_pixel = 8, channels = 1)
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

def decompose_essential_matrix(E):
    '''Returns an R,t defined by an essential matrix

E = R * skew_symmetric(t) = R * T

I know that cross(t,t) = T * t = 0, so I can get t as an eigenvector of E
corresponding to an eigenvalue of 0. If I have T then I need to solve E = R * T.
This is the procrustes problem that I can solve with
mrcal.align_procrustes_vectors_R01()

    '''
    l,t = np.linalg.eig(E)
    i = np.argmin(np.abs(l))
    if np.abs(l[i]) > 1e-10:
        raise Exception("E doesn't have a 0 eigenvalue")
    if nps.norm2(t[:,i].imag) > 1e-10:
        raise Exception("null eigenvector of E has non-0 imaginary components")
    t = t[:,i].real

    # The "true" t is k*t for some unknown constant k
    # And the "true" T is k*T for some unknown constant k
    T = mrcal.skew_symmetric(t)

    # E = k*R*T -> EtE = k^2 TtT
    ksq = nps.matmult(nps.transpose(E),E) / nps.matmult(nps.transpose(T),T)

    mean_ksq = np.mean(ksq)

    if nps.norm2( ksq.ravel() - mean_ksq ) > 1e-10:
        raise Exception("t doesn't have a consistent scale")

    k = np.sqrt(mean_ksq)
    t *= k
    T *= k

    # I now have t and T with the right scale, BUT STILL WITH AN UNKNOWN SIGN. I
    # report both versions
    Rt = np.empty((2,4,3), dtype=float)

    Rt[0,:3,:] = \
        mrcal.align_procrustes_vectors_R01(nps.transpose(E),
                                           nps.transpose(T))
    Rt[0,3,:] = t
    if nps.norm2((nps.matmult(Rt[0,:3,:],T) - E).ravel()) > 1e-10:
        raise Exception("Non-fitting rotation")

    t *= -1.
    T *= -1.

    Rt[1,:3,:] = \
        mrcal.align_procrustes_vectors_R01(nps.transpose(E),
                                           nps.transpose(T))
    Rt[1,3,:] = t
    if nps.norm2((nps.matmult(Rt[1,:3,:],T) - E).ravel()) > 1e-10:
        raise Exception("Non-fitting rotation")

    return Rt

def seed_rt10_pair_from_far_subset(q0, q1, mask_far):
    r'''Estimates a transform between two cameras

This method ingests two sets of corresponding features, with a subset of these
features deemed to be "far". It then

- Computes a Procrustes fit on the "far" features to get an estimate for the
  rotation. This is valid because observations at infinity are not affected by
  the relatively tiny translations, and I only need to rotate the vectors
  properly.

- Assumes this rotation is correct, and uses all the features to estimate the
  translation. This part is more involved, so I write it up here

I use the geometric triangulation expression derived here:

  https://github.com/dkogan/mrcal/blob/8be76fc28278f8396c0d3b07dcaada2928f1aae0/triangulation.cc#L112

I assume that I'm triangulating normalized v0,v1 both expressed in cam-0
coordinates. And I have a t01 translation that I call "t" from here on. This is
unique only up-to-scale, so I assume that norm2(t) = 1. The geometric
triangulation from the above link says that

  [k0] = 1/(v0.v0 v1.v1 -(v0.v1)**2) [ v1.v1   v0.v1][ t01.v0]
  [k1]                               [ v0.v1   v0.v0][-t01.v1]

  The midpoint p is

  p = (k0 v0 + t01 + k1 v1)/2

I assume that v are normalized and I represent k as a vector. I also define

  c = inner(v0,v1)

So

  k = 1/(1 - c^2) [1 c] [ v0t] t
                  [c 1] [-v1t]

I define

  A = 1/(1 - c^2) [1 c]     This is a 2x2 array
                  [c 1]

  B = [ v0t]                This is a 2x3 array
      [-v1t]

  V = [v0 v1]               This is a 3x2 array

Note that none of A,B,V depend on t.

So

  k = A B t

Then

  p = (k0 v0 + t01 + k1 v1)/2
    = (V k + t)/2
    = (V A B t + t)/2
    = (I + V A B) t/2

Each triangulated error is

  e = mag(p - k0 v0)

I split A into its rows

  A = [ a0t ]
      [ a1t ]

Then

  e = p - k0 v0
    = (I + V A B) t/2 - v0 a0t B t
    = I t/2 + V A B t/2 - v0 a0t B t
    = I t/2 + v0 a0t B t/2 + v1 a1t B t/2 - v0 a0t B t
    = I t/2 - v0 a0t B t/2 + v1 a1t B t/2
    = ((I - v0 a0t B + v1 a1t B) t) / 2
    = ((I + (- v0 a0t + v1 a1t) B) t) / 2
    = ((I - Bt A B) t) / 2

I define a joint error function I'm optimizing as the sum of all the individual
triangulation errors:

  E = sum(norm2(e_i))

Each component is

  norm2(e) = 1/4 tt (I - Bt A B)t (I - Bt A B) t
           = 1/4 tt (I - 2 Bt A B + Bt A B Bt A B ) t

  B Bt = [1  -c]
         [-c  1]

  B Bt A = 1/(1 - c^2) [1  -c] [1 c]
                       [-c  1] [c 1]
         = 1/(1 - c^2) [1-c^2  0     ]
                       [0      1-c^2 ]
         = I

-> norm2(e) = 1/4 tt (I - 2 Bt A B + Bt A B) t
            = 1/4 tt (I - Bt A B) t
            = 1/4 - 1/4 tt Bt A B t

So

    E = N/4 - 1/4 tt sum(Bt A B) t

I let

    M  = sum(Bt A B)
    M  = sum(Mi)
    Mi = Bt A B

So

    E = N/4 - 1/4 tt M t
      = N/4 - 1/4 lambda

So to minimize E I find t that is the eigenvector of M that corresponds to its
largest eigenvalue lambda. Furthermore, lambda depends on the rotation. If I
couldn't estimate the rotation from far-away features I can solve the
eigenvalue-optimization problem to maximize lambda.

More simplification:

    Mi = Bt A B = [ v0  -v1 ] A [ v0t]
                                [-v1t]
       = 1/(1-c^2) [ v0 - v1 c    v0 c - v1] [ v0t]
                                             [-v1t]
       = 1/(1-c^2) ((v0 - v1 c) v0t - (v0 c - v1) v1t)

    c  = v0t v1 ->
    F0 = v0 v0t
    F1 = v1 v1t

    -> Mi = 1/(1-c^2) (v0 v0t - v1 v1t v0 v0t + v1 v1t - v0 v0t v1 v1t)
          = 1/(1-c^2) (F0 + F1 - (F1 F0 + F0 F1))
          = (F0 - F1)^2 / (1 - c^2)
          = ((F0 - F1)/s)^2

    where s = mag(cross(v0,v1))


    tt M t = sum( norm2((F0i - F1i)/si t) )

    Let Di = (F0i - F1i)/si

    I want to maximize sum( norm2(Di t) )


    (F0 - F1)/s = (v0 v0t - v1 v1t) / mag(cross(v0,v1))
                ~ (v0 v0t - R v1 v1t Rt) / mag(cross(v0,R v1))

experiments:

          = 1/(1-c^2) (F0 + F1 - (F1 F0 + F0 F1))
          = 1/(1-c^2) (v0 v0t + v1 v1t - (c v0 v1t + c v1 v0t))


    '''

    # shape (N,3)
    # These are in their LOCAL coord system
    v0 = mrcal.unproject(q0, *model.intrinsics(),
                         normalize = True)
    v1 = mrcal.unproject(q1, *model.intrinsics(),
                         normalize = True)

    R01 = mrcal.align_procrustes_vectors_R01(v0[mask_far], v1[mask_far])



    # can try to do outlier rejection here:
    #   co = nps.inner(v0[mask_far], mrcal.rotate_point_R(R01, v1[mask_far]))
    #   gp.plot(co)



    # Keep all all non-far points initially
    mask_keep_near = ~mask_far


    # I only use the near features to compute t01. The far features don't affect
    # t very much, and they'll have c ~ 1 and A ~ infinity

    # shape (N,3)
    v0_cam0coords = v0
    v1_cam0coords = mrcal.rotate_point_R(R01, v1)
    # shape (N,)
    c = nps.inner(v0_cam0coords[mask_keep_near],
                  v1_cam0coords[mask_keep_near])

    # Any near features that have parallel rays is disqualified
    mask_parallel = np.abs(1. - c) < 1e-6
    mask_keep_near[np.nonzero(mask_keep_near)[0][mask_parallel]] = False



    # Can try to do outlier rejection here. At t=0 all points should be
    # convergent or all should be divergent. Any non-consensus points are
    # outliers
    #   p = mrcal.triangulate_geometric(v0[~mask_far],
    #                                   mrcal.rotate_point_R(R01, v1[~mask_far]),
    #                                   np.zeros((3,)))
    #   mask_divergent = nps.norm2(p) == 0
    def compute_t(v0_cam0coords, v1_cam0coords):
        # shape (N,)
        c = nps.inner(v0_cam0coords,
                      v1_cam0coords)

        N = len(c)

        # shape (N,2,2)
        A = np.ones((N,2,2), dtype=float)
        A[:,0,1] = c
        A[:,1,0] = c
        A /= nps.mv(1. - c*c, -1,-3)

        # shape (N,2,3)
        B = np.empty((N,2,3), dtype=float)
        B[:,0,:] =  v0_cam0coords
        B[:,1,:] = -v1_cam0coords

        # shape (3,3)
        M = np.sum( nps.matmult(nps.transpose(B), A, B), axis = 0 )

        l,v = _sorted_eig(M)

        # The answer is the eigenvector corresponding to the biggest eigenvalue
        t01 = v[:,2]

        # Almost done. I want either t or -t. The wrong one will produce
        # mostly triangulations behind me
        k = nps.matmult(A,B, nps.transpose(t01))[..., 0]

        mask_divergent_t    = (k[:,0] <= 0) + (k[:,1] <= 0)
        mask_divergent_negt = (k[:,0] >= 0) + (k[:,1] >= 0)
        N_divergent_t    = np.count_nonzero( mask_divergent_t )
        N_divergent_negt = np.count_nonzero( mask_divergent_negt )

        if N_divergent_t == 0 or N_divergent_negt == 0:

            # from before: norm2(e) = 1/4 - 1/4 tt Bt A B t
            # shape (N,2,1)
            Bt = nps.dummy(nps.inner(B, t01),
                           -1)

            # shape (N,1,1)
            tBtABt = nps.matmult(nps.transpose(Bt), A, Bt)

            # shape (N,)
            tBtABt = tBtABt[:,0,0]

            norm2e = 1/4 * (1 - tBtABt)

        else:
            norm2e = None

        if N_divergent_t < N_divergent_negt:
            return  t01, mask_divergent_t, N_divergent_t, norm2e
        else:
            return -t01, mask_divergent_negt, N_divergent_negt, norm2e


    i_iteration = 0
    while True:

        print(f"seed_rt10_pair_from_far_subset() iteration {i_iteration}")

        t01, mask_divergent, Ndivergent, norm2e = \
            compute_t(v0_cam0coords[mask_keep_near],
                      v1_cam0coords[mask_keep_near])
        if Ndivergent == 0:

            # No divergences, and we have norm2e available. I look through
            # norm2e, and throw away outliers there. If there aren't any of
            # those either, I'm done.
            mask_convergent_outlier = norm2e > 0.04
            if not np.any(mask_convergent_outlier):
                # no outliers. I'm done!
                break

            print(f"No divergences, but have {np.count_nonzero(mask_convergent_outlier)} outliers")
            mask_outlier = mask_convergent_outlier
        else:
            # I have divergences. Mark these as outliers, and move on
            print(f"saw {Ndivergent} divergences. Total len(v) = {len(v0)}")
            mask_outlier = mask_divergent

        mask_keep_near[np.nonzero(mask_keep_near)[0][mask_outlier]] = False
        i_iteration += 1

    Rt01 = nps.glue(R01, t01, axis=-2)
    Rt10 = mrcal.invert_Rt(Rt01)
    rt10 = mrcal.rt_from_Rt(Rt10)
    return \
        rt10, \
        (mask_keep_near + mask_far)

def seed_rt10_pair_kneip_eigensolver(q0, q1):
    r'''Estimates a transform between two cameras

opengv does all the work
    '''


    # shape (N,3)
    # These are in their LOCAL coord system
    v0 = mrcal.unproject(q0, *model.intrinsics(),
                         normalize = True)
    v1 = mrcal.unproject(q1, *model.intrinsics(),
                         normalize = True)

    # Keep all all non-far points initially
    mask_inliers = np.ones( (q0.shape[0],), dtype=bool )


    def compute(v0, v1):

        Rt01 = np.empty((4,3), dtype=float)
        Rt01[:3,:] = pyopengv.relative_pose_eigensolver(v0, v1,
                                                        # seed
                                                        mrcal.identity_R())

        # opengv should do this too, but its Python bindings are lacking. I
        # recompute the t myself for now

        # shape (N,3)
        c = np.cross(v0, mrcal.rotate_point_R(Rt01[:3,:], v1))
        l,v = _sorted_eig(np.sum(nps.outer(c,c), axis=0))
        # t is the eigenvector corresponding to the smallest eigenvalue
        Rt01[3,:] = v[:,0]

        # Almost done. I want either t or -t. The wrong one will produce
        # mostly triangulations behind me
        p_t = mrcal.triangulate_geometric(v0, v1,
                                          v_are_local = True,
                                          Rt01        = Rt01 )
        mask_divergent_t = (nps.norm2(p_t) == 0)
        N_divergent_t    = np.count_nonzero( mask_divergent_t )

        Rt01_negt = Rt01 * nps.transpose(np.array((1,1,1,-1),))
        p_negt = mrcal.triangulate_leecivera_mid2(v0, v1,
                                                  v_are_local = True,
                                                  Rt01        = Rt01_negt )
        mask_divergent_negt = (nps.norm2(p_negt) == 0)
        N_divergent_negt    = np.count_nonzero( mask_divergent_negt )

        if N_divergent_t != 0 and N_divergent_negt != 0:
            # We definitely have divergences. Mark them as outliers, and move on
            if N_divergent_t < N_divergent_negt: return Rt01,      mask_divergent_t,    N_divergent_t
            else:                                return Rt01_negt, mask_divergent_negt, N_divergent_negt


        # Nothing is divergent. I look for outliers
        if N_divergent_t == 0:
            p              = p_t
            mask_divergent = mask_divergent_t
            N_divergent    = N_divergent_t
        else:
            p              = p_negt
            mask_divergent = mask_divergent_negt
            N_divergent    = N_divergent_negt
            Rt01           = Rt01_negt

        costh = nps.inner(p, v0) / nps.mag(p)

        costh_threshold = np.cos(1.0 * np.pi/180.)

        mask_convergent_outliers = costh < costh_threshold
        if not np.any(mask_convergent_outliers):
            # no outliers. I'm done!
            return Rt01, mask_divergent, N_divergent

        Nmask_convergent_outliers = np.count_nonzero(mask_convergent_outliers)
        print(f"No divergences, but have {Nmask_convergent_outliers} outliers")
        return Rt01, mask_convergent_outliers, Nmask_convergent_outliers



    i_iteration = 0
    while True:

        print(f"seed_rt10_pair_kneip_eigensolver() iteration {i_iteration}")

        Rt01, mask_outlier, Noutliers = compute(v0[mask_inliers], v1[mask_inliers])
        print(f"saw {Noutliers} outliers. Total len(v) = {len(v0)}")
        if Noutliers == 0:
            break

        mask_inliers[np.nonzero(mask_inliers)[0][mask_outlier]] = False
        i_iteration += 1

    Rt10 = mrcal.invert_Rt(Rt01)
    rt10 = mrcal.rt_from_Rt(Rt10)
    return \
        rt10, \
        mask_inliers



def seed_rt10_pair(i0, q0, q1):

    if False:
        # driving forward
        return np.array((-0.0001,0.0001,0.00001, 0.2,0.2,-.95), dtype=float), \
            None
    elif True:
        # My method using known-far features
        rt10, mask_inliers = \
            seed_rt10_pair_from_far_subset(q0,q1,
                                           (q0[:,1] < 800) * (q1[:,1] < 800))

        return rt10, mask_inliers

    elif True:
        # The method in opengv. Similar to my method, but better!
        rt10, mask_inliers = \
            seed_rt10_pair_kneip_eigensolver(q0,q1)
        return rt10, mask_inliers

    else:
        # Reading Shehryar's poses
        if i0 < 0:
            return mrcal.identity_rt(), \
                None
        return rt_cam_camprev__from_data_file[i0], \
            None


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
$av/aliceVision_featureExtraction -p low -i xxxxx.sfm -o xxxxx-features
$av/aliceVision_featureMatching -i xxxxx.sfm -f xxxxx-features -o xxxxx-matches
$av/aliceVision_incrementalSfM  -i xxxxx.sfm -f xxxxx-features -m xxxxx-matches -o xxxxx-sfm-output.ply
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


feature_finder = None
matcher        = None
def feature_matching__opencv(i_image, image0_decimated, image1_decimated):

    import cv2

    global feature_finder, matcher

    if feature_finder is None:
        feature_finder = cv2.ORB_create()
        matcher        = cv2.BFMatcher(cv2.NORM_HAMMING,
                                       crossCheck = True)

    if 0:
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
    else:

        import IPython
        IPython.embed()
        sys.exit()



        # shape (Hdecimated,Wdecimated,2)
        flow_decimated = \
            cv2.optflow.calcOpticalFlowSF(image0_decimated, image1_decimated,
                                          flow       = None, # for in-place output
                                          pyr_scale  = 0.5,
                                          levels     = 3,
                                          winsize    = 15,
                                          iterations = 3,
                                          poly_n     = 5,
                                          poly_sigma = 1.2,
                                          flags      = 0# cv2.OPTFLOW_USE_INITIAL_FLOW
                                          )

        # 'calcOpticalFlowDenseRLOF',
        # 'calcOpticalFlowSF',
        # 'calcOpticalFlowSparseRLOF',
        # 'calcOpticalFlowSparseToDense',
        # 'createOptFlow_DeepFlow',
        # 'createOptFlow_DenseRLOF',
        # 'createOptFlow_DualTVL1',
        # 'createOptFlow_Farneback',
        # 'createOptFlow_PCAFlow',
        # 'createOptFlow_SimpleFlow',
        # 'createOptFlow_SparseRLOF',
        # 'createOptFlow_SparseToDense'





    if 1:
        plot_flow(image0_decimated, flow_decimated,
                  decimation_extra_plot,
                  hardcopy = f"{outdir}/flow{i_image:03d}.png")

    import IPython
    IPython.embed()
    sys.exit()


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


def mark_outliers_from_seed(i0, i1,
                            optimization_inputs,
                            mask_correspondence_outliers):

    # Very similar to get_observation_pair(). Please consolidate

    indices_point_camintrinsics_camextrinsics = optimization_inputs['indices_point_triangulated_camintrinsics_camextrinsics']
    observations                              = optimization_inputs['observations_point_triangulated']



    # i0, i1 are indexed from -1: these are the camextrinsics indices
    if i0+1 != i1:
        raise Exception("mark_outliers_from_seed() currently only works for consecutive indices")

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

    print(f"i0 = {i0}. marking {np.count_nonzero(mask_correspondence_outliers)} correspondences as outliers. 2x observations: {2*np.count_nonzero(mask_correspondence_outliers)}")

    iobservation0 = (idx_cam0_selected  )[mask_correspondence_outliers]
    iobservation1 = (idx_cam0_selected+1)[mask_correspondence_outliers]
    observations[iobservation0, 2] = -1.
    observations[iobservation1, 2] = -1.


def mark_outliers(indices_point_camintrinsics_camextrinsics, observations,
                  rt_cam_ref):
    # I consider consecutive observations only. So if a single point was
    # observed by in-order frames 0,3,4,6 then here I will consider (0,3), (3,4)
    # and (4,6)
    ipoint_current = -1
    iobservation0  = -1

    vlocal = mrcal.unproject(observations[:,:2], *model.intrinsics(),
                             normalize = True)

    cos_half_theta_threshold = np.cos(1. * np.pi/180. / 2.)

    for iobservation1 in range(len(indices_point_camintrinsics_camextrinsics)):
        ipoint,icami,icame = indices_point_camintrinsics_camextrinsics[iobservation1]
        weight             = observations[iobservation1,  2]

        if weight <= 0:
            # This observation is an outlier. There's nothing at all to do
            continue

        if ipoint == ipoint_current and \
           iobservation0 >= 0:
            # I'm observing the same point as the previous observation. Compare
            # them
            ipoint_prev,icami_prev,icame_prev = \
                indices_point_camintrinsics_camextrinsics[iobservation0]

            if icame_prev >= 0: rt0r = rt_cam_ref[icame_prev]
            else:               rt0r = mrcal.identity_rt()
            if icame      >= 0: rt01 = mrcal.compose_rt(rt0r, mrcal.invert_rt(rt_cam_ref[icame]))
            else:               rt01 = rt0r

            v0 = vlocal[iobservation0]
            v1 = mrcal.rotate_point_r(rt01[:3], vlocal[iobservation1])
            t01 = rt01[3:]

            p = mrcal.triangulate_leecivera_mid2(v0,v1,t01)
            if nps.norm2(p) == 0 or \
               nps.inner(p, v0) / nps.mag(p) < cos_half_theta_threshold:

                # outlier
                observations[iobservation0,2] = -1.
                observations[iobservation1,2] = -1.

                iobservation0 = -1
                continue
                # I KEEP GOING, BUT THIS IS A BUG. What about previous
                # observations of this point that were deemed not an outlier?
                # Are those still good? I should also set iobservation0 to
                # the previous observation in this point that wasn't an outlier.
                # The logic is complicated so I just ignore it for the time
                # being

        else:
            # This is a different point from the previous. Or the previous point
            # was an outlier. The next iteration will compare against this one
            ipoint_current= ipoint

        iobservation0 = iobservation1


def solve(indices_point_camintrinsics_camextrinsics,
          observations,
          Nimages):

    observations_fixed                              = None
    indices_point_camintrinsics_camextrinsics_fixed = None
    points_fixed                                    = None
    Npoints_fixed                                   = 0

    if 0:
        if 1:
            # delta. desert

            # hard-coded known at-infinity point seen in the first two cameras
            q0_infinity = np.array((1853,1037), dtype=float)
            q1_infinity = np.array((1920,1039), dtype=float)
        else:
            # xxxxx ranch
            q0_infinity = np.array((2122, 501), dtype=float)
            q1_infinity = np.array((1379, 548), dtype=float)


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

        points_fixed = nps.atleast_dims(p_infinity, -2)

        Npoints_fixed = 1




    # i0 is indexed from -1: it's a camextrinsics index
    #
    # This is a relative array:
    # [ rt10 ]
    # [ rt21 ]
    # [ rt32 ]
    # [ .... ]
    seed_rt10_and_mask_correspondences_inliers = \
        [seed_rt10_pair( i0+1,
                         *get_observation_pair(i0, i0+1,
                                               indices_point_camintrinsics_camextrinsics,
                                               observations) ) \
                  for i0 in range(-1,Nimages-2)]
    rt_cam_camprev = \
        np.array([rm[0] for rm in seed_rt10_and_mask_correspondences_inliers])

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

              points                                    = points_fixed,
              indices_point_camintrinsics_camextrinsics = indices_point_camintrinsics_camextrinsics_fixed,
              observations_point                        = observations_fixed,
              Npoints_fixed                             = Npoints_fixed,
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

    # I injest whatever outliers I got from the seeding algorithm
    for i0 in range(-1,Nimages-2):
        mask_correspondence_inliers = seed_rt10_and_mask_correspondences_inliers[i0+1][1]
        if mask_correspondence_inliers is None:
            continue
        mark_outliers_from_seed(i0, i0+1,
                                optimization_inputs,
                                ~mask_correspondence_inliers)

    # And now another pass of pre-solve outlier rejection. As currently
    # implemented, the seeding only considers consecutive-frame observations.
    # Any observations of a point in frames 0,2 that are NOT observed in frame 1
    # are not considered by the seeding algorithm, and any outliers will not
    # appear. Here I go through each observation

    mark_outliers(indices_point_camintrinsics_camextrinsics, observations,
                  rt_cam_ref)


    print(f"Seed rt_cam_ref = {rt_cam_ref}")
    write_models("/tmp/xxxxx-seed-cam{}.cameramodel",
                 model,
                 optimization_inputs['extrinsics_rt_fromref'])

    stats = mrcal.optimize(**optimization_inputs)

    write_models("/tmp/xxxxx-solve-cam{}.cameramodel",
                 model,
                 optimization_inputs['extrinsics_rt_fromref'])

    print(f"Solved rt_cam_ref = {optimization_inputs['extrinsics_rt_fromref']}")

    x = stats['x']
    return x, optimization_inputs

def show_solution(optimization_inputs, Nimages, binary = True):

    def write_points(f, p, bgr):

        N = len(p)

        if binary:
            binary_ply = np.empty( (N,),
                                   dtype = dtype)
            binary_ply['xyz'] = p

            binary_ply['rgba'][:,0] = bgr[:,2]
            binary_ply['rgba'][:,1] = bgr[:,1]
            binary_ply['rgba'][:,2] = bgr[:,0]
            binary_ply['rgba'][:,3] = 255

            binary_ply.tofile(f)
        else:
            fp = nps.glue(p,
                          bgr[:,(2,)],
                          bgr[:,(1,)],
                          bgr[:,(0,)],
                          255*np.ones((N,1)),
                          axis = -1)
            np.savetxt(f,
                       fp,
                       fmt      = ('%.1f','%.1f','%.1f','%d','%d','%d','%d'),
                       comments = '',
                       header   = '')
        return N



    if binary:
        ply_type = 'binary_little_endian'
    else:
        ply_type = 'ascii'

    placeholder = 'NNNNNNN'
    ply_header = f'''ply
format {ply_type} 1.0
element vertex {placeholder}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar alpha
end_header
'''.encode()

    Npoints_pointcloud = 0

    filename_point_cloud = "/tmp/points.ply"
    with open(filename_point_cloud, 'wb') as f:

        dtype = np.dtype([ ('xyz',np.float32,3), ('rgba', np.uint8, 4) ])
        f.write(ply_header)

        # Camera positions in red
        rt_cam_ref = nps.glue( np.zeros((6,)),
                               optimization_inputs['extrinsics_rt_fromref'],
                               axis = -2 )
        t_ref_cam = mrcal.invert_rt(rt_cam_ref)[:,3:]
        write_points(f,
                     t_ref_cam,
                     np.zeros((Nimages,3),  dtype=np.uint8) +
                     np.array((0,0,255),    dtype=np.uint8))
        Npoints_pointcloud += Nimages

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

            # filename_overlaid_points = f'/tmp/overlaid-points-{i0+1}.pdf'
            # gp.plot(q0_r_recip,
            #         tuplesize = -3,
            #         _with     = 'points pt 7 ps 0.5 palette',
            #         square    = 1,
            #         _xrange   = (0,W),
            #         _yrange   = (H,0),
            #         rgbimage  = image_filename[i0+1],
            #         hardcopy  = filename_overlaid_points,
            #         cbmax     = 0.1)
            # print(f"Wrote '{filename_overlaid_points}'")

            image = mrcal.load_image(image_filename[i0+1], bits_per_pixel = 24, channels = 3)

            ######### point cloud
            #### THIS IS WRONG: I report a separate point in each consecutive
            #### triangulation, so if I tracked a feature over N frames, instead
            #### of reporting one point for that feature, I'll report N-1 of
            #### them

            # I'm including the alpha byte to align each row to 16 bytes.
            # Otherwise I have unaligned 32-bit floats. I don't know for a fact
            # that this breaks anything, but it feels like it would maybe.
            i = (q0[index_good_triangulation] + 0.5).astype(int)
            bgr = image[i[:,1], i[:,0]]
            Npoints_pointcloud += \
                write_points(f,
                             mrcal.transform_point_rt(mrcal.invert_rt(rt_0r),
                                                      plocal0[index_good_triangulation]),
                             bgr)


    # I wrote the point cloud file with an unknown number of points. Now that I
    # have the count, I go back, and fill it in.
    import mmap
    with open(filename_point_cloud, 'r+b') as f:
        m = mmap.mmap(f.fileno(), 0)

        i_placeholder_start = ply_header.find(placeholder.encode())
        placeholder_width   = ply_header[i_placeholder_start:].find(b'\n')
        i_placeholder_end   = i_placeholder_start + placeholder_width

        m[i_placeholder_start:i_placeholder_end] = \
            '{:{width}d}'.format(Npoints_pointcloud, width=placeholder_width).encode()

        m.close()

    print(f"Wrote '{filename_point_cloud}'")

def write_model(filename, model):
    print(f"Writing '{filename}'")
    model.write(filename)

def write_models(filename_format,
                 model_baseline, rt_cam_ref):

    model0 = mrcal.cameramodel(model_baseline)
    model0.extrinsics_rt_fromref(np.zeros((6,), dtype=float))
    write_model(filename_format.format(0), model0)
    for i in range(1,len(rt_cam_ref)+1):
        model1 = mrcal.cameramodel(model_baseline)
        model1.extrinsics_rt_fromref(rt_cam_ref[i-1])
        write_model(filename_format.format(i), model1)


if 1:
    # delta. desert
    image_glob               = "/home/dima/data/xxxxx/delta/*.jpg"
    outdir                   = "/tmp"
    decimation               = 20
    decimation_extra_plot    = 5
    model_filename           = "/home/dima/xxxxx-sfm/cam.cameramodel"
    colmap_database_filename = '/tmp/xxxxx.exhaustive.db'

    image_filename = sorted(glob.glob(image_glob))

else:
    # xxxxx ranch

    # t1_t2_p_qxyzw = np.loadtxt("/mnt/nvm/xxxxx-xxxxx-ranch/time_stamp_xyz_xyzw.vnl",
    #                            dtype = [ ('time',      np.uint64),
    #                                      ('timestamp', np.uint64),
    #                                      ('p',         float, (3,)),
    #                                      ('quat_xyzw', float, (4,)),])
    # t_filename    = np.loadtxt("/mnt/nvm/xxxxx-xxxxx-ranch/time_filename.vnl",
    #                            dtype = [ ('timestamp', np.uint64),
    #                                      ('filename', 'S50') ])

    # quat_xyzw      = t1_t2_p_qxyzw['quat_xyzw']
    # quat           = quat_xyzw[...,(3,0,1,2)]
    # r              = mrcal.r_from_R( mrcal.R_from_quat(quat) )
    # rt_ref_veh_all = nps.glue(r,
    #                           t1_t2_p_qxyzw['p'],
    #                           axis = -1)

    t_dt_p_qxyzw = np.loadtxt("/mnt/nvm/xxxxx-xxxxx-ranch/relative-poses.vnl",
                               dtype = float)
    t_filename    = np.loadtxt("/mnt/nvm/xxxxx-xxxxx-ranch/time_filename.vnl",
                               dtype = [ ('timestamp', np.uint64),
                                         ('filename', 'S250') ])
    quat_xyzw        = t_dt_p_qxyzw[:,5:]
    quat             = quat_xyzw[...,(3,0,1,2)]
    r                = mrcal.r_from_R( mrcal.R_from_quat(quat) )
    rt_cam0_cam1_all = nps.glue(r,
                                t_dt_p_qxyzw[:,2:5],
                                axis = -1)



    # Row i in the pose file has
    #   t[i]-t[i-1] == dt[i]
    # So I presume it has rt_camprev_cam
    #
    # I also checked, and the timestamps in t_filename match those in
    # t_dt_p_qxyzw exactly. No "interpolation" is needed, but I'll ask for it
    # anyway
    import scipy.interpolate

    f = \
        scipy.interpolate.interp1d(t_dt_p_qxyzw[:,0],
                                   rt_cam0_cam1_all,
                                   axis = -2,
                                   bounds_error  = True,
                                   assume_sorted = True)

    # I want the last 7 images
    image_filename = t_filename['filename' ][-7:]
    rt_camprev_cam__from_data_file = f(t_filename['timestamp'][-7:].astype(float) / 1e9)


    # The first image doesn't have a camprev. Throw it away
    rt_camprev_cam__from_data_file = rt_camprev_cam__from_data_file[1:]

    rt_cam_camprev__from_data_file = mrcal.invert_rt(rt_camprev_cam__from_data_file)

    image_dir               = "/mnt/nvm/xxxxx-xxxxx-ranch/images-last10"
    outdir                   = "/tmp"
    decimation               = 20
    decimation_extra_plot    = 5
    model_filename           = "/mnt/nvm/xxxxx-xxxxx-ranch/oryx.cameramodel"
    colmap_database_filename = '/mnt/nvm/xxxxx-xxxxx-ranch/xxxxx.db'

    image_filename = [ f"{image_dir}/{os.path.basename(f.decode())}" for f in image_filename]





model = mrcal.cameramodel(model_filename)
W,H   = model.imagersize()


Nimages = 3

# q.shape = (Npoints, Nimages=2, Nxy=2)
if 1:
    image0,image0_decimated = imread(image_filename[0], decimation)
    image1,image1_decimated = imread(image_filename[1], decimation)
    q = feature_matching__opencv(0, image0_decimated, image1_decimated)
    show_matched_features(image0_decimated, image1_decimated, q)
else:
    indices_point_camintrinsics_camextrinsics, \
    observations = \
        feature_matching__colmap(colmap_database_filename,
                                 Nimages)

x,optimization_inputs = solve(indices_point_camintrinsics_camextrinsics,
                              observations,
                              Nimages)











show_solution(optimization_inputs, Nimages)

import IPython
IPython.embed()
sys.exit()




try:
    image0_decimated = image1_decimated
except:
    # if I'm looking at cached features, I never read any actual images
    pass


r'''
'''
