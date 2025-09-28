#!/usr/bin/env python3

r'''Tests various pose transformations'''

import sys
import numpy as np
import numpysane as nps
import os

testdir = os.path.dirname(os.path.realpath(__file__))

# I import the LOCAL mrcal since that's what I'm testing
sys.path[:0] = f"{testdir}/..",
import mrcal
import testutils


p = np.array([[1.79294696, 3.84665205, 8.88059626],
              [1.94285588, 5.5994178,  9.08980913],
              [1.16926135, 7.11203249, 6.49111756],
              [4.97016243, 3.36377894, 9.91309303],
              [7.0578383,  6.96290783, 2.74822211],
              [2.09286004, 8.32613405, 4.29534757],
              [5.390182,   6.97763688, 8.75498418],
              [6.54142328, 3.77331963, 3.29282044],
              [5.84571708, 4.87274621, 4.32639624],
              [7.00320821, 0.70546965, 5.67190652]])

Rt = np.array([[ 0.37796712, -0.5149426 ,  0.7693991 ],
               [ 0.52322138,  0.80441527,  0.28134581],
               [-0.76379333,  0.29622659,  0.5734715 ],
               [ 0.9447233,   6.8439095,   9.6958398 ]])
R = Rt[:3, :]
t = Rt[ 3, :]

noise = np.array([[0.00035356, 0.00043613, 0.00006606],
                  [0.00043968, 0.00043783, 0.00060678],
                  [0.00063803, 0.00024423, 0.00010871],
                  [0.00004966, 0.00053377, 0.00018905],
                  [0.00007708, 0.00023529, 0.0002229 ],
                  [0.00090558, 0.00072379, 0.00004062],
                  [0.00072059, 0.00074467, 0.00044128],
                  [0.00024228, 0.00058201, 0.00041458],
                  [0.00018121, 0.00078172, 0.00016128],
                  [0.00019021, 0.00001371, 0.00096808]])

Tp = nps.matmult( p, nps.transpose(R) ) + t
Rt_fit = \
    mrcal.align_procrustes_points_Rt01(Tp + noise,
                                       p)
R_fit = Rt_fit[:3, :]
t_fit = Rt_fit[ 3, :]
testutils.confirm_equal( R_fit, R, eps=1e-2, msg='Procrustes fit R' )
testutils.confirm_equal( t_fit, t, eps=1e-2, msg='Procrustes fit t' )

R_fit_vectors = \
    mrcal.align_procrustes_vectors_R01(nps.matmult( p, nps.transpose(R) ) + noise,
                                       p)
testutils.confirm_equal( R_fit_vectors, R, eps=1e-2, msg='Procrustes fit R (vectors)' )


testutils.confirm_equal( mrcal.invert_Rt(mrcal.Rt_from_rt(mrcal.invert_rt(mrcal.rt_from_Rt(Rt)))), Rt,
                         msg = 'Rt/rt and invert')

testutils.confirm_equal( mrcal.compose_Rt( Rt, mrcal.invert_Rt(Rt)),
                         nps.glue(np.eye(3), np.zeros((3,)), axis=-2),
                         msg = 'compose_Rt')

testutils.confirm_equal( mrcal.compose_rt( mrcal.rt_from_Rt(Rt), mrcal.invert_rt(mrcal.rt_from_Rt(Rt))),
                         np.zeros((6,)),
                         msg = 'compose_rt')

testutils.confirm_equal( mrcal.compose_r( mrcal.r_from_R(R),
                                          -mrcal.r_from_R(R)),
                         np.zeros((3,)),
                         msg = 'compose_r')

testutils.confirm_equal( mrcal.identity_Rt(),
                         nps.glue(np.eye(3), np.zeros((3,)), axis=-2),
                         msg = 'identity_Rt')

testutils.confirm_equal( mrcal.identity_rt(),
                         np.zeros((6,)),
                         msg = 'identity_rt')

testutils.confirm_equal( mrcal.transform_point_Rt( Rt, p ),
                         Tp,
                         msg = 'transform_point_Rt')

testutils.confirm_equal( mrcal.transform_point_rt( mrcal.rt_from_Rt(Rt), p ),
                         Tp,
                         msg = 'transform_point_rt')

testutils.confirm_equal( mrcal.transform_point_Rt( mrcal.invert_Rt(Rt), Tp ),
                         p,
                         msg = 'transform_point_Rt inverse')

testutils.confirm_equal( mrcal.transform_point_rt( mrcal.invert_rt(mrcal.rt_from_Rt(Rt)), Tp ),
                         p,
                         msg = 'transform_point_rt inverse')

######### quaternion business
testutils.confirm_equal( mrcal.R_from_quat( mrcal.quat_from_R(R) ),
                         R,
                         msg = 'R <-> quaternion transforms are inverses of one another')

# shape (2,3,3)
RR = nps.cat( R, nps.matmult(R,R) )
# shape (2,4,3)
RtRt = nps.cat( Rt, mrcal.compose_Rt(Rt,Rt) )
testutils.confirm_equal( mrcal.R_from_quat( mrcal.quat_from_R(RR) ),
                         RR,
                         msg = 'R <-> quaternion transforms are inverses of one another. Broadcasted')

# I'm concerned about quat_from_R() broadcasting properly. I check
testutils.confirm_equal( mrcal.quat_from_R(RR.reshape(2,1,3,3)).shape,
                         (2,1,4),
                         msg = 'quat_from_R() shape')
testutils.confirm_equal( mrcal.quat_from_R(RR.reshape(2,3,3)).shape,
                         (2,4),
                         msg = 'quat_from_R() shape')
testutils.confirm_equal( mrcal.quat_from_R(R).shape,
                         (4,),
                         msg = 'quat_from_R() shape')

# in-place output
R2  = np.zeros(R. shape, dtype=float)
RR2 = np.zeros(RR.shape, dtype=float)
r2  = np.zeros(R. shape[:-2] + (3,), dtype=float)
rr2 = np.zeros(RR.shape[:-2] + (3,), dtype=float)
q2  = np.zeros(R. shape[:-2] + (4,), dtype=float)
qq2 = np.zeros(RR.shape[:-2] + (4,), dtype=float)
q  = mrcal.quat_from_R(R)
qq = mrcal.quat_from_R(RR)
mrcal.quat_from_R(R, out=q2)
testutils.confirm_equal( q2, q,
                         msg = 'quat_from_R() in-place')
mrcal.quat_from_R(RR, out=qq2)
testutils.confirm_equal( qq2, qq,
                         msg = 'quat_from_R() in-place')
mrcal.R_from_quat(q, out=R2)
testutils.confirm_equal( R2,
                         mrcal.R_from_quat(q),
                         msg = 'R_from_quat() in-place')
mrcal.R_from_quat(qq, out=RR2)
testutils.confirm_equal( RR2,
                         mrcal.R_from_quat(qq),
                         msg = 'R_from_quat() in-place')

Rt2   = np.zeros(Rt.  shape, dtype=float)
RtRt2 = np.zeros(RtRt.shape, dtype=float)
rt2   = np.zeros(Rt.  shape[:-2] + (6,), dtype=float)
rtrt2 = np.zeros(RtRt.shape[:-2] + (6,), dtype=float)
qt2   = np.zeros(Rt.  shape[:-2] + (7,), dtype=float)
qtqt2 = np.zeros(RtRt.shape[:-2] + (7,), dtype=float)
qt    = mrcal.qt_from_Rt(Rt)
qtqt  = mrcal.qt_from_Rt(RtRt)
mrcal.qt_from_Rt(Rt, out=qt2)
testutils.confirm_equal( qt2, qt,
                         msg = 'qt_from_Rt() in-place')
mrcal.qt_from_Rt(RtRt, out=qtqt2)
testutils.confirm_equal( qtqt2, qtqt,
                         msg = 'qt_from_Rt() in-place')
mrcal.Rt_from_qt(qt, out=Rt2)
testutils.confirm_equal( Rt2,
                         mrcal.Rt_from_qt(qt),
                         msg = 'Rt_from_qt() in-place')
mrcal.Rt_from_qt(qtqt, out=RtRt2)
testutils.confirm_equal( RtRt2,
                         mrcal.Rt_from_qt(qtqt),
                         msg = 'Rt_from_qt() in-place')


####### small-angle R_from_r()
for sign in (1, -1):
    for th0 in (1e-12, 1e-10, 1e-8, 1e-6,1e-4,1e-2):

        for extra90 in range(-4,4):
            th = np.pi/2. * extra90 + sign*th0
            c,s = np.cos(th), np.sin(th)

            Rtiny = ( np.array((( 1, 0,  0),
                                ( 0, c,  -s),
                                ( 0, s,  c))),

                      np.array((( c, 0,  s),
                                ( 0, 1,  0),
                                (-s, 0,  c))),

                      np.array((( c, -s, 0),
                                ( s, c,  0),
                                ( 0, 0,  1))) )

            for iaxis in range(3):
                rtiny = np.array((0,0,0.))
                rtiny[iaxis] = th

                # xxx
                ########### should check gradients

                testutils.confirm_equal( mrcal.R_from_r(rtiny),
                                         Rtiny[iaxis],
                                         worstcase = True,
                                         eps = 1e-13,
                                         msg = f'R_from_r() for tiny rotations around axis {iaxis}: th = {sign*th0} + {extra90}*90deg')

                testutils.confirm_equal( mrcal.r_from_R(Rtiny[iaxis]),
                                         rtiny,
                                         worstcase = True,
                                         eps = 1e-13,
                                         r   = True,
                                         msg = f'r_from_R() for tiny rotations around axis {iaxis}: th = {sign*th0} + {extra90}*90deg')


###### skew_symmetric()
a = np.array((1.,5.,7.))
b = np.array((3.,-.1,-10.))

A = mrcal.skew_symmetric(a)
testutils.confirm_equal( nps.inner(A,b),
                         np.cross(a,b),
                         eps = 1e-13,
                         msg = f'skew_symmetric()')


# Checks to make sure the Python and C flavors of the procrustes solves do the
# same thing
#
# shape (Nsamples, Npair=2,Npoints,N3d=3)
p = \
    np.array([[[[0.14704423, 0.39775277, 0.30555471],
                [0.92170835, 0.46068765, 0.46597563],
                [0.85681395, 0.14640106, 0.72339265],
                [0.64150377, 0.9689525 , 0.18343949],
                [0.37721179, 0.61115926, 0.72395643],
                [0.90988371, 0.54785929, 0.00305871]],
               [[0.17938669, 0.870795  , 0.80838611],
                [0.76591916, 0.41732501, 0.29741046],
                [0.43456858, 0.80027015, 0.40425126],
                [0.51486614, 0.75565368, 0.19814464],
                [0.13175592, 0.63443547, 0.26494142],
                [0.36305645, 0.93237726, 0.56876193]]],
              [[[0.91457097, 0.06700906, 0.74084108],
                [0.34972644, 0.19535236, 0.93707003],
                [0.05978952, 0.68233545, 0.01814865],
                [0.46846867, 0.53959814, 0.18141228],
                [0.78532912, 0.29627199, 0.58184341],
                [0.93356584, 0.14646568, 0.77161276]],
               [[0.70057041, 0.33016758, 0.94419219],
                [0.62895361, 0.34410519, 0.83516703],
                [0.8054023 , 0.37424879, 0.19989159],
                [0.81918909, 0.18177506, 0.72753215],
                [0.2796512 , 0.24979332, 0.04228793],
                [0.08684694, 0.72923008, 0.86589431]]],
              [[[0.5174404 , 0.89101492, 0.75315541],
                [0.96703036, 0.35592374, 0.26891381],
                [0.11491883, 0.3553755 , 0.07406605],
                [0.87666567, 0.72322093, 0.78270873],
                [0.73797503, 0.48847173, 0.43649112],
                [0.40196225, 0.36487594, 0.23143557]],
               [[0.28714974, 0.84705018, 0.62632826],
                [0.2525358 , 0.05370839, 0.69798499],
                [0.41555077, 0.48653363, 0.20137377],
                [0.50153789, 0.86894349, 0.39839871],
                [0.69501319, 0.927363  , 0.78524037],
                [0.20165683, 0.74075575, 0.93348345]]],
              [[[0.1670269 , 0.68982035, 0.25166721],
                [0.6252659 , 0.97867961, 0.31571348],
                [0.99385309, 0.55140928, 0.86169534],
                [0.37788756, 0.06677707, 0.25035251],
                [0.45994065, 0.42615147, 0.93190971],
                [0.19488194, 0.70151131, 0.49352064]],
               [[0.62938171, 0.10537443, 0.04530917],
                [0.97526369, 0.31124128, 0.21464597],
                [0.18855158, 0.44590893, 0.24078581],
                [0.31546736, 0.33583147, 0.55277668],
                [0.37431502, 0.72881629, 0.5461762 ],
                [0.85790425, 0.76488894, 0.17539372]]],
              [[[0.7934717 , 0.98374073, 0.51727301],
                [0.86370405, 0.28009948, 0.6730014 ],
                [0.356218  , 0.31673491, 0.66938557],
                [0.66419795, 0.23694415, 0.74916447],
                [0.49975421, 0.94466077, 0.30366929],
                [0.4893559 , 0.33181135, 0.44038687]],
               [[0.95820648, 0.86295328, 0.08792959],
                [0.78256322, 0.94953821, 0.51813174],
                [0.4836827 , 0.30111458, 0.52616329],
                [0.06204762, 0.78452476, 0.20771378],
                [0.37545885, 0.78866451, 0.82157995],
                [0.81211514, 0.59884354, 0.9740134 ]]]])
w = np.array([0.78040546, 0.59692462, 0.81189354, 0.18877835, 0.53251149, 0.79716902])

# shape (Nsamples, Npair=2,Npoints,N3d=3)
for p0,p1 in p:
    # each of p0,p1 has shape (Npoints,3)

    Rt01        = mrcal.align_procrustes_points_Rt01(              p0,p1,weights=w)
    Rt01_python = mrcal.utils._align_procrustes_points_Rt01_python(p0,p1,weights=w)

    testutils.confirm_equal( Rt01, Rt01_python,
                             worstcase = True,
                             eps = 1e-10,
                             msg = f'align_procrustes_points_Rt01() does the same thing in C and Python')

    v0 = p0 / nps.dummy(nps.mag(p0),axis=-1)
    v1 = p1 / nps.dummy(nps.mag(p1),axis=-1)
    R01        = mrcal.align_procrustes_vectors_R01(              v0,v1,weights=w)
    R01_python = mrcal.utils._align_procrustes_vectors_R01_python(v0,v1,weights=w)

    testutils.confirm_equal( R01, R01_python,
                             worstcase = True,
                             eps = 1e-10,
                             msg = f'align_procrustes_vectors_R01() does the same thing in C and Python')


# And one more: make sure the error handling works. Trying to align degenerate
# data should fail
#
# If I have 1 points, both flavors should fail
# should fail
testutils.confirm(not np.any(mrcal.align_procrustes_points_Rt01(p[0,0,:1,:],
                                                                p[0,1,:1,:])),
                  msg = "align_procrustes_points_Rt01() should fail with one point")
testutils.confirm(not np.any(mrcal.align_procrustes_vectors_R01(p[0,0,:1,:],
                                                                p[0,1,:1,:])),
                  msg = "align_procrustes_vectors_R01() should fail with one point")

# If I have 2 points, vector alignment should succeed, while point alignment
# should fail
testutils.confirm(not np.any(mrcal.align_procrustes_points_Rt01(p[0,0,:2,:],
                                                                p[0,1,:2,:])),
                  msg = "align_procrustes_points_Rt01() should fail with two points")
testutils.confirm(np.any(mrcal.align_procrustes_vectors_R01(p[0,0,:2,:],
                                                            p[0,1,:2,:])),
                  msg = "align_procrustes_vectors_R01() should succeed with two points")


# 2 samples
p0 = p[:2,0,...]
p1 = p[:2,1,...]
# The first set of points in p1 all lie along a line
p1[0,...] = np.array((1.,0.1, 5.)) + \
    nps.dummy(np.arange(p0.shape[1]), axis=-1) * np.array((-1.,-2., 0.5))

Rt01        = mrcal.align_procrustes_points_Rt01(              p0,p1,weights=w)
Rt01_python = mrcal.utils._align_procrustes_points_Rt01_python(p0,p1,weights=w)

testutils.confirm_equal( Rt01[0,...], 0,
                         worstcase = True,
                         eps = 1e-10,
                         msg = f'align_procrustes_points_Rt01() reports errors with degenerate data')
testutils.confirm_equal( Rt01, Rt01_python,
                         worstcase = True,
                         eps = 1e-10,
                         msg = f'align_procrustes_points_Rt01() reports errors with degenerate data')


v        = np.array( (0.1, 6.3, -2.0) )
R        = mrcal.R_aligned_to_vector(v)
R_python = mrcal.utils._R_aligned_to_vector_python(v)

testutils.confirm_equal( R, R_python,
                         worstcase = True,
                         eps = 1e-10,
                         msg = f'R_aligned_to_vector() does the same thing in C and Python')

testutils.finish()
