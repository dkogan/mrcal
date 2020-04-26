#!/usr/bin/python3

r'''Tests gradients reported by the python code

This is conceptually similar to test-gradients.py, but here I validate the
python code path

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


intrinsics = \
    ( ('LENSMODEL_PINHOLE',
       np.array((1512., 1112, 500., 333.))),
      ('LENSMODEL_OPENCV4',
       np.array((1512., 1112, 500., 333.,
                 -0.012, 0.035, -0.001, 0.002))),
      ('LENSMODEL_CAHVOR',
       np.array((4842.918,4842.771,1970.528,1085.302,
                 -0.001, 0.002, -0.637, -0.002, 0.016))),
      ('LENSMODEL_SPLINED_STEREOGRAPHIC_3_11_8_200_1499.5_999.5',
       np.array([ 2017.284705,1242.204557,2053.514381,1214.368063,2037.9067,1212.609628,
                  2033.278227,1183.689487,2040.018023,1188.554431,2069.146825,1196.304649,
                  2085.708658,1186.478238,2065.787617,1163.377825,2086.372192,1138.856716,
                  2131.609155,1125.678279,2128.812604,1120.525061,2008.41491,1218.64154,
                  2024.522768,1239.588759,2034.947935,1198.14079,2065.474055,1198.97294,
                  2044.562395,1200.557321,2087.714092,1160.440038,2086.478691,1151.822407,
                  2112.862582,1147.567288,2101.575718,1146.312256,2100.56469,1157.015327,
                  2113.488262,1111.679758,2019.837901,1244.168216,2025.847768,1215.633807,
                  2041.980956,1205.751212,2075.077056,1199.787561,2070.877831,1203.261678,
                  2067.244278,1184.705736,2082.225077,1185.558149,2091.519961,1175.01817,
                  2120.258866,1137.775228,2120.020747,1152.409316,2121.870228,1113.069319,
                  2043.650555,1247.757041,2019.661062,1230.723629,2067.917203,1209.753396,
                  2035.034141,1219.514335,2045.350268,1178.474255,2046.346049,1169.372592,
                  2097.839998,1194.836758,2112.724938,1172.186377,2110.996386,1154.899043,
                  2128.456883,1133.228404,2122.513384,1131.717886,2044.279196,1233.288366,
                  2023.197297,1230.118703,2067.07694,1199.998862,2044.147271,1191.607451,
                  2058.590053,1167.7808,2081.593501,1182.074581,2086.63053,1159.156329,
                  2084.329086,1157.727374,2073.666528,1151.261965,2114.290905,1144.710519,
                  2138.600912,1119.405248,2016.299528,1206.147494,2029.434175,1211.507857,
                  2057.936091,1198.01196,2035.691392,1174.035359,2084.718618,1203.604729,
                  2085.910021,1158.385222,2080.800068,1150.199852,2087.991586,1162.019581,
                  2094.754507,1151.061493,2115.144642,1154.299799,2107.014195,1127.608146,
                  2005.632475,1238.607328,2020.33157,1202.101384,2061.021703,1214.868271,
                  2043.015135,1211.903685,2052.91186,1188.092787,2094.86724,1179.277314,
                  2078.230124,1186.273023,2077.743945,1148.028845,2081.634186,1131.207467,
                  2112.936851,1126.412871,2113.220553,1114.991063,2017.901873,1244.588667,
                  2051.238803,1201.855728,2043.256406,1216.674722,2035.286046,1178.380907,
                  2080.28318,1178.783085,2051.214271,1173.560417,2059.298121,1182.414688,
                  2094.607679,1177.960959,2086.998287,1147.371259,2120.29442,1138.197348,
                  2138.994213, 1114.846113,])))

# a few points, some wide, some not. None behind the camera
p = np.array(((1.0, 2.0, 10.0),
              (-1.1, 0.3, 1.0),
              (-0.9, -1.5, 1.0)))

delta = 1e-6

for i in intrinsics:

    q,dq_dp,dq_di = mrcal.project(p, *i, get_gradients=True)

    Nintrinsics = mrcal.getNlensParams(i[0])
    testutils.confirm_equal(dq_di.shape[-1], Nintrinsics,
                            msg=f"{i[0]}: Nintrinsics match for {i[0]}")
    if Nintrinsics != dq_di.shape[-1]:
        continue

    for ivar in range(dq_dp.shape[-1]):

        # center differences
        p1 = p.copy()
        p1[..., ivar] = p[..., ivar] - delta/2
        q1 = mrcal.project(p1, *i, get_gradients=False)
        p1[..., ivar] += delta
        q2 = mrcal.project(p1, *i, get_gradients=False)

        dq_dp_observed = (q2 - q1) / delta
        dq_dp_reported = dq_dp[..., ivar]

        testutils.confirm_equal(dq_dp_reported, dq_dp_observed,
                                eps=1e-5,
                                msg=f"{i[0]}: dq_dp matches for var {ivar}")

    for ivar in range(dq_di.shape[-1]):

        # center differences
        i1 = i[1].copy()
        i1[..., ivar] = i[1][..., ivar] - delta/2
        q1 = mrcal.project(p, i[0], i1, get_gradients=False)
        i1[..., ivar] += delta
        q2 = mrcal.project(p, i[0], i1, get_gradients=False)

        dq_di_observed = (q2 - q1) / delta
        dq_di_reported = dq_di[..., ivar]

        testutils.confirm_equal(dq_di_reported, dq_di_observed,
                                eps=1e-5,
                                msg=f"{i[0]}: dq_di matches for var {ivar}")
testutils.finish()
