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
for th in (1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6):
    c,s = np.cos(th), np.sin(th)
    rtiny = np.array((0, th, 0))
    Rtiny = np.array((( c, 0, s),
                      ( 0, 1, 0),
                      (-s, 0, c)))

    testutils.confirm_equal( mrcal.R_from_r(rtiny),
                             Rtiny,
                             worstcase = True,
                             eps = 1e-13,
                             msg = f'R_from_r() for tiny rotations: th = {th}')
    testutils.confirm_equal( mrcal.r_from_R(Rtiny),
                             rtiny,
                             worstcase = True,
                             eps = 1e-13,
                             msg = f'r_from_R() for tiny rotations: th = {th}')


###### skew_symmetric()
a = np.array((1.,5.,7.))
b = np.array((3.,-.1,-10.))

A = mrcal.skew_symmetric(a)
testutils.confirm_equal( nps.inner(A,b),
                         np.cross(a,b),
                         eps = 1e-13,
                         msg = f'skew_symmetric()')


testutils.finish()
