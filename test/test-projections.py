#!/usr/bin/python3

r'''Regression tests project()

Here I make sure the projection functions return the correct values. This is a
regression test, so the "right" values were recorded at some point, and any
deviation is flagged.

This test confirms the correct values, and test-gradients.py confirms that these
values are consistent with the reported gradients. So together these two tests
validate the projection functionality

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



def check(intrinsics, p_ref, q_ref, unproject = True):
    q_projected = mrcal.project(p_ref, *intrinsics)
    testutils.confirm_equal(q_projected,
                            q_ref,
                            msg = f"Projecting {intrinsics[0]}",
                            eps = 1e-2)
    if not unproject:
        return

    p_unprojected = mrcal.unproject(q_projected, *intrinsics)
    cos = nps.inner(p_unprojected, p_ref) / (nps.mag(p_ref)*nps.mag(p_unprojected))
    cos = np.clip(cos, -1, 1)
    testutils.confirm_equal( np.arccos(cos),
                             np.zeros((p_ref.shape[0],), dtype=float),
                             msg = f"Unprojecting {intrinsics[0]}",
                             eps = 1e-6)



# a few points, some wide, some not. None behind the camera
p = np.array(((1.0, 2.0, 10.0),
              (-1.1, 0.3, 1.0),
              (-0.9, -1.5, 1.0)))

check( ('LENSMODEL_PINHOLE', np.array((1512., 1112, 500., 333.))),
       p,
       np.array([[  651.2,   555.4],
                 [-1163.2,   666.6],
                 [ -860.8, -1335. ]]))

check( ('LENSMODEL_OPENCV4', np.array((1512., 1112, 500., 333.,
                                       -0.012, 0.035, -0.001, 0.002))),
       p,
       np.array([[  651.27371  ,   555.23042  ],
                 [-1223.38516  ,   678.01468  ],
                 [-1246.7310448, -1822.799928 ]]))

check( ('LENSMODEL_OPENCV5', np.array((1512., 1112, 500., 333.,
                                       -0.012, 0.035, -0.001, 0.002, 0.019))),
       p,
       np.array([[  651.2740691 ,   555.2309482 ],
                 [-1292.8121176 ,   691.9401448 ],
                 [-1987.550162  , -2730.85863427]]))

check( ('LENSMODEL_OPENCV8', np.array((1512., 1112, 500., 333.,
                                       -0.012, 0.035, -0.001, 0.002, 0.019, 0.014, -0.056, 0.050))),
       p,
       np.array([[  651.1885442 ,   555.10514968],
                 [-1234.45480366,   680.23499814],
                 [ -770.03274263, -1238.4871943 ]]))

check( ('LENSMODEL_CAHVOR', np.array((4842.918, 4842.771, 1970.528, 1085.302,
                                      -0.001, 0.002, -0.637, -0.002, 0.016))),
       p,
       np.array([[ 2143.17840406,  1442.93419919],
                 [  -92.63813066,  1653.09646897],
                 [ -249.83199315, -2606.46477164]]))

check( ('LENSMODEL_CAHVORE', np.array((4842.918, 4842.771, 1970.528, 1085.302,
                                       -0.001, 0.002, -0.637, -0.002, 0.016, 1e-8, 2e-8, 3e-8, 0.0))),
       p,
       np.array([[2140.34076919, 1437.37148001],
                 [ 496.63465931, 1493.31670636],
                 [ 970.11788123, -568.30114806]]))

check( ('LENSMODEL_CAHVORE', np.array((4842.918, 4842.771, 1970.528, 1085.302,
                                       -0.001, 0.002, -0.637, -0.002, 0.016, 1e-2, 2e-2, 3e-2, 0.0))),
       p,
       np.array([[2140.35607966, 1437.40149368],
                 [ 489.05797783, 1495.37110356],
                 [ 954.60918375, -594.21144463]]))

check( ('LENSMODEL_CAHVORE', np.array((4842.918, 4842.771, 1970.528, 1085.302,
                                       -0.001, 0.002, -0.637, -0.002, 0.016, 1e-2, 2e-2, 3e-2, 0.4))),
       p,
       np.array([[2140.80289923, 1438.2774104 ],
                 [ 423.27156274, 1513.20891648],
                 [ 872.53696336, -731.32905711]]))


# I don't have unprojections implemented yet for splined models, so I only test
# the forward direction for now. This is a semi-random sampling from
# analyses/splines/verify-interpolated-surface.py. Note that some of the
# projected points are behind the camera (z<0), which is possible with these
# models. Also note that some of the projected points are off the imager (x<0).
# This is aphysical, but it just means that the model was made up; which it was.
# The math still works normally, and this is just fine as a test
check( ('LENSMODEL_SPLINED_STEREOGRAPHIC_3_11_8_200_1499.5_999.5',
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
                   2138.994213, 1114.846113,],)),

       # some points behind the camera!
       np.array([[-0.8479983,  -0.52999894, -0.34690877],
                 [-0.93984618,  0.34159794, -0.16119387],
                 [-0.97738792,  0.21145412,  5.49068928]]),
       np.array([[-3333.74107926,  -826.41922033],
                 [-2999.02162307,  1970.09116005],
                 [ 1135.2353517 ,  1044.39227558]]))


testutils.finish()
