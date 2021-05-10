#!/usr/bin/python3

r'''Tests for project() and unproject()

Here I make sure the projection functions return the correct values. This is a
regression test, so the "right" project() results were recorded at some point,
and any deviation is flagged.

This also test gradients, normalization and in-place output. The reference
values for those are computed on the fly, rather than being hard-coded a-priori

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
from test_calibration_helpers import grad


def check(intrinsics, p_ref, q_ref):
    ########## project
    q_projected = mrcal.project(p_ref, *intrinsics)
    testutils.confirm_equal(q_projected,
                            q_ref,
                            msg = f"Projecting {intrinsics[0]}",
                            eps = 1e-2)

    q_projected *= 0
    mrcal.project(p_ref, *intrinsics,
                  out = q_projected)
    testutils.confirm_equal(q_projected,
                            q_ref,
                            msg = f"Projecting {intrinsics[0]} in-place",
                            eps = 1e-2)

    meta = mrcal.lensmodel_metadata_and_config(intrinsics[0])
    if meta['has_gradients']:
        @nps.broadcast_define( ((3,),('N',)) )
        def grad_broadcasted(p_ref, i_ref):
            return grad(lambda pi: mrcal.project(pi[:3], intrinsics[0], pi[3:]),
                        nps.glue(p_ref,i_ref, axis=-1))

        dq_dpi_ref = grad_broadcasted(p_ref,intrinsics[1])

        q_projected,dq_dp,dq_di = mrcal.project(p_ref, *intrinsics, get_gradients=True)
        testutils.confirm_equal(q_projected,
                                q_ref,
                                msg = f"Projecting {intrinsics[0]} with grad",
                                eps = 1e-2)
        testutils.confirm_equal(dq_dp,
                                dq_dpi_ref[...,:3],
                                msg = f"dq_dp {intrinsics[0]}",
                                eps = 1e-2)
        testutils.confirm_equal(dq_di,
                                dq_dpi_ref[...,3:],
                                msg = f"dq_di {intrinsics[0]}",
                                eps = 1e-2)

        out=[q_projected,dq_dp,dq_di]
        out[0] *= 0
        out[1] *= 0
        out[2] *= 0
        mrcal.project(p_ref, *intrinsics, get_gradients=True, out=out)

        testutils.confirm_equal(q_projected,
                                q_ref,
                                msg = f"Projecting {intrinsics[0]} with grad in-place",
                                eps = 1e-2)
        testutils.confirm_equal(dq_dp,
                                dq_dpi_ref[...,:3],
                                msg = f"dq_dp in-place",
                                eps = 1e-2)
        testutils.confirm_equal(dq_di,
                                dq_dpi_ref[...,3:],
                                msg = f"dq_di in-place",
                                eps = 1e-2)


    ########## unproject

    v_unprojected = mrcal.unproject(q_projected, *intrinsics,
                                    normalize = True)

    testutils.confirm_equal( nps.norm2(v_unprojected),
                             1,
                             msg = f"Unprojected v are normalized",
                             eps = 1e-6)
    cos = nps.inner(v_unprojected, p_ref) / nps.mag(p_ref)
    cos = np.clip(cos, -1, 1)
    testutils.confirm_equal( np.arccos(cos),
                             np.zeros((p_ref.shape[0],), dtype=float),
                             msg = f"Unprojecting {intrinsics[0]}",
                             eps = 1e-6)

    if not meta['has_gradients']:
        # no in-place output for the no-gradients unproject() path
        return

    v_unprojected *= 0
    mrcal.unproject(q_projected, *intrinsics,
                    normalize = True,
                    out = v_unprojected)
    testutils.confirm_equal( nps.norm2(v_unprojected),
                             1,
                             msg = f"Unprojected in-place v are normalized",
                             eps = 1e-6)
    cos = nps.inner(v_unprojected, p_ref) / nps.mag(p_ref)
    cos = np.clip(cos, -1, 1)
    testutils.confirm_equal( np.arccos(cos),
                             np.zeros((p_ref.shape[0],), dtype=float),
                             msg = f"Unprojecting in-place {intrinsics[0]}",
                             eps = 1e-6)

    ### unproject gradients
    v_unprojected,dv_dq,dv_di = mrcal.unproject(q_projected,
                                                *intrinsics, get_gradients=True)

    # # v - inner(v,axis)axis
    # dv_dq -= nps.xchg(nps.dummy(nps.inner( nps.xchg(dv_dq,-1,-2),
    #                                        nps.dummy(v_unprojected, -2) ),
    #                             -1) * nps.dummy(v_unprojected, -2),
    #                   -1, -2)
    # dv_di -= nps.xchg(nps.dummy(nps.inner( nps.xchg(dv_di,-1,-2),
    #                                        nps.dummy(v_unprojected, -2) ),
    #                             -1) * nps.dummy(v_unprojected, -2),
    #                   -1, -2)

    @nps.broadcast_define( ((2,),('N',)) )
    def grad_broadcasted(q_ref, i_ref):
        return grad(lambda qi: \
                    mrcal.unproject_stereographic( \
                    mrcal.project_stereographic(
                        mrcal.unproject(qi[:2], intrinsics[0], qi[2:]))),
                    nps.glue(q_ref,i_ref, axis=-1))

    dv_dqi_ref = grad_broadcasted(q_projected,intrinsics[1])

    testutils.confirm_equal(mrcal.project(v_unprojected, *intrinsics),
                            q_projected,
                            msg = f"Unprojecting {intrinsics[0]} with grad",
                            eps = 1e-2)
    testutils.confirm_equal(dv_dq,
                            dv_dqi_ref[...,:2],
                            msg = f"dv_dq: {intrinsics[0]}",
                            worstcase = True,
                            relative  = True,
                            eps = 0.01)
    testutils.confirm_equal(dv_di,
                            dv_dqi_ref[...,2:],
                            msg = f"dv_di {intrinsics[0]}",
                            worstcase = True,
                            relative  = True,
                            eps = 0.01)

    # Normalized unprojected gradients
    v_unprojected,dv_dq,dv_di = mrcal.unproject(q_projected,
                                                *intrinsics,
                                                normalize     = True,
                                                get_gradients = True)
    testutils.confirm_equal( nps.norm2(v_unprojected),
                             1,
                             msg = f"Unprojected v (with gradients) are normalized",
                             eps = 1e-6)
    cos = nps.inner(v_unprojected, p_ref) / nps.mag(p_ref)
    cos = np.clip(cos, -1, 1)
    testutils.confirm_equal( np.arccos(cos),
                             np.zeros((p_ref.shape[0],), dtype=float),
                             msg = f"Unprojecting (normalized, with gradients) {intrinsics[0]}",
                             eps = 1e-6)

    @nps.broadcast_define( ((2,),('N',)) )
    def grad_normalized_broadcasted(q_ref, i_ref):
        return grad(lambda qi: \
                    mrcal.unproject(qi[:2], intrinsics[0], qi[2:], normalize=True),
                    nps.glue(q_ref,i_ref, axis=-1))

    dv_dqi_ref = grad_normalized_broadcasted(q_projected,intrinsics[1])

    testutils.confirm_equal(dv_dq,
                            dv_dqi_ref[...,:2],
                            msg = f"dv_dq (normalized v): {intrinsics[0]}",
                            worstcase = True,
                            relative  = True,
                            eps = 0.01)
    testutils.confirm_equal(dv_di,
                            dv_dqi_ref[...,2:],
                            msg = f"dv_di (normalized v): {intrinsics[0]}",
                            worstcase = True,
                            relative  = True,
                            eps = 0.01)

    if 0:
        # In-place reporting of unproject() gradients isn't supported yet, so I
        # don't test it
        out=[v_unprojected,dv_dq,dv_di]
        out[0] *= 0
        out[1] *= 0
        out[2] *= 0

        mrcal.unproject(q_projected,
                        *intrinsics,
                        normalize     = True,
                        get_gradients = True,
                        out           = out)
        testutils.confirm_equal( nps.norm2(v_unprojected),
                                 1,
                                 msg = f"Unprojected v (with gradients, in-place) are normalized",
                                 eps = 1e-6)
        cos = nps.inner(v_unprojected, p_ref) / nps.mag(p_ref)
        cos = np.clip(cos, -1, 1)
        testutils.confirm_equal( np.arccos(cos),
                                 np.zeros((p_ref.shape[0],), dtype=float),
                                 msg = f"Unprojecting (normalized, with gradients, in-place) {intrinsics[0]}",
                                 eps = 1e-6)

        testutils.confirm_equal(dv_dq,
                                dv_dqi_ref[...,:2],
                                msg = f"dv_dq (normalized v, in-place): {intrinsics[0]}",
                                worstcase = True,
                                relative  = True,
                                eps = 0.01)
        testutils.confirm_equal(dv_di,
                                dv_dqi_ref[...,2:],
                                msg = f"dv_di (normalized v, in-place): {intrinsics[0]}",
                                worstcase = True,
                                relative  = True,
                                eps = 0.01)


# a few points, some wide, some not. None behind the camera
p = np.array(((1.0, 2.0, 10.0),
              (-1.1, 0.3, 1.0),
              (-0.9, -1.5, 1.0)))

check( ('LENSMODEL_PINHOLE', np.array(((1512., 1112, 500., 333.),
                                       (1512., 1112, 500., 433.),
                                       (1512., 1112, 500., 533.)))),
       p,
       np.array([[  651.2,   555.4],
                 [-1163.2,   766.6],
                 [ -860.8, -1135. ]]))

check( ('LENSMODEL_STEREOGRAPHIC', np.array((1512., 1112, 500., 333.))),
       p,
       np.array([[ 649.35582325,  552.6874014],
                 [-821.79644263,  598.1222302],
                 [-402.7032835,  -773.48815174]]))

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

check( ('LENSMODEL_CAHVORE_linearity=0.00', np.array((4842.918, 4842.771, 1970.528, 1085.302,
                                       -0.001, 0.002, -0.637, -0.002, 0.016, 1e-8, 2e-8, 3e-8))),
       p,
       np.array([[2140.34076919, 1437.37148001],
                 [ 496.63465931, 1493.31670636],
                 [ 970.11788123, -568.30114806]]))

check( ('LENSMODEL_CAHVORE_linearity=0.00', np.array((4842.918, 4842.771, 1970.528, 1085.302,
                                       -0.001, 0.002, -0.637, -0.002, 0.016, 1e-2, 2e-2, 3e-2))),
       p,
       np.array([[2140.35607966, 1437.40149368],
                 [ 489.05797783, 1495.37110356],
                 [ 954.60918375, -594.21144463]]))

check( ('LENSMODEL_CAHVORE_linearity=0.40', np.array((4842.918, 4842.771, 1970.528, 1085.302,
                                       -0.001, 0.002, -0.637, -0.002, 0.016, 1e-2, 2e-2, 3e-2))),
       p,
       np.array([[2140.80289923, 1438.2774104 ],
                 [ 423.27156274, 1513.20891648],
                 [ 872.53696336, -731.32905711]]))


# Note that some of the projected points are behind the camera (z<0), which is
# possible with these models. Also note that some of the projected points are
# off the imager (x<0). This is aphysical, but it just means that the model was
# made up; which it was. The math still works normally, and this is just fine as
# a test
check( ('LENSMODEL_SPLINED_STEREOGRAPHIC_order=3_Nx=11_Ny=8_fov_x_deg=200',
        np.array([ 1500.0, 1800.0, 1499.5,999.5,
                   2.017284705,1.242204557,2.053514381,1.214368063,2.0379067,1.212609628,
                   2.033278227,1.183689487,2.040018023,1.188554431,2.069146825,1.196304649,
                   2.085708658,1.186478238,2.065787617,1.163377825,2.086372192,1.138856716,
                   2.131609155,1.125678279,2.128812604,1.120525061,2.00841491,1.21864154,
                   2.024522768,1.239588759,2.034947935,1.19814079,2.065474055,1.19897294,
                   2.044562395,1.200557321,2.087714092,1.160440038,2.086478691,1.151822407,
                   2.112862582,1.147567288,2.101575718,1.146312256,2.10056469,1.157015327,
                   2.113488262,1.111679758,2.019837901,1.244168216,2.025847768,1.215633807,
                   2.041980956,1.205751212,2.075077056,1.199787561,2.070877831,1.203261678,
                   2.067244278,1.184705736,2.082225077,1.185558149,2.091519961,1.17501817,
                   2.120258866,1.137775228,2.120020747,1.152409316,2.121870228,1.113069319,
                   2.043650555,1.247757041,2.019661062,1.230723629,2.067917203,1.209753396,
                   2.035034141,1.219514335,2.045350268,1.178474255,2.046346049,1.169372592,
                   2.097839998,1.194836758,2.112724938,1.172186377,2.110996386,1.154899043,
                   2.128456883,1.133228404,2.122513384,1.131717886,2.044279196,1.233288366,
                   2.023197297,1.230118703,2.06707694,1.199998862,2.044147271,1.191607451,
                   2.058590053,1.1677808,2.081593501,1.182074581,2.08663053,1.159156329,
                   2.084329086,1.157727374,2.073666528,1.151261965,2.114290905,1.144710519,
                   2.138600912,1.119405248,2.016299528,1.206147494,2.029434175,1.211507857,
                   2.057936091,1.19801196,2.035691392,1.174035359,2.084718618,1.203604729,
                   2.085910021,1.158385222,2.080800068,1.150199852,2.087991586,1.162019581,
                   2.094754507,1.151061493,2.115144642,1.154299799,2.107014195,1.127608146,
                   2.005632475,1.238607328,2.02033157,1.202101384,2.061021703,1.214868271,
                   2.043015135,1.211903685,2.05291186,1.188092787,2.09486724,1.179277314,
                   2.078230124,1.186273023,2.077743945,1.148028845,2.081634186,1.131207467,
                   2.112936851,1.126412871,2.113220553,1.114991063,2.017901873,1.244588667,
                   2.051238803,1.201855728,2.043256406,1.216674722,2.035286046,1.178380907,
                   2.08028318,1.178783085,2.051214271,1.173560417,2.059298121,1.182414688,
                   2.094607679,1.177960959,2.086998287,1.147371259,2.12029442,1.138197348,
                   2.138994213, 1.114846113,],)),

       # some points behind the camera!
       np.array([[-0.8479983,  -0.52999894, -0.34690877],
                 [-0.93984618,  0.34159794, -0.16119387],
                 [-0.97738792,  0.21145412,  5.49068928]]),
       np.array([[ 965.9173441 ,  524.31894367],
                 [1246.58668369, 4621.35427783],
                 [4329.41598149, 3183.75121559]]))

check( ('LENSMODEL_SPLINED_STEREOGRAPHIC_order=2_Nx=11_Ny=8_fov_x_deg=200',
        np.array([ 1500.0, 1800.0, 1499.5,999.5,
                   2.017284705,1.242204557,2.053514381,1.214368063,2.0379067,1.212609628,
                   2.033278227,1.183689487,2.040018023,1.188554431,2.069146825,1.196304649,
                   2.085708658,1.186478238,2.065787617,1.163377825,2.086372192,1.138856716,
                   2.131609155,1.125678279,2.128812604,1.120525061,2.00841491,1.21864154,
                   2.024522768,1.239588759,2.034947935,1.19814079,2.065474055,1.19897294,
                   2.044562395,1.200557321,2.087714092,1.160440038,2.086478691,1.151822407,
                   2.112862582,1.147567288,2.101575718,1.146312256,2.10056469,1.157015327,
                   2.113488262,1.111679758,2.019837901,1.244168216,2.025847768,1.215633807,
                   2.041980956,1.205751212,2.075077056,1.199787561,2.070877831,1.203261678,
                   2.067244278,1.184705736,2.082225077,1.185558149,2.091519961,1.17501817,
                   2.120258866,1.137775228,2.120020747,1.152409316,2.121870228,1.113069319,
                   2.043650555,1.247757041,2.019661062,1.230723629,2.067917203,1.209753396,
                   2.035034141,1.219514335,2.045350268,1.178474255,2.046346049,1.169372592,
                   2.097839998,1.194836758,2.112724938,1.172186377,2.110996386,1.154899043,
                   2.128456883,1.133228404,2.122513384,1.131717886,2.044279196,1.233288366,
                   2.023197297,1.230118703,2.06707694,1.199998862,2.044147271,1.191607451,
                   2.058590053,1.1677808,2.081593501,1.182074581,2.08663053,1.159156329,
                   2.084329086,1.157727374,2.073666528,1.151261965,2.114290905,1.144710519,
                   2.138600912,1.119405248,2.016299528,1.206147494,2.029434175,1.211507857,
                   2.057936091,1.19801196,2.035691392,1.174035359,2.084718618,1.203604729,
                   2.085910021,1.158385222,2.080800068,1.150199852,2.087991586,1.162019581,
                   2.094754507,1.151061493,2.115144642,1.154299799,2.107014195,1.127608146,
                   2.005632475,1.238607328,2.02033157,1.202101384,2.061021703,1.214868271,
                   2.043015135,1.211903685,2.05291186,1.188092787,2.09486724,1.179277314,
                   2.078230124,1.186273023,2.077743945,1.148028845,2.081634186,1.131207467,
                   2.112936851,1.126412871,2.113220553,1.114991063,2.017901873,1.244588667,
                   2.051238803,1.201855728,2.043256406,1.216674722,2.035286046,1.178380907,
                   2.08028318,1.178783085,2.051214271,1.173560417,2.059298121,1.182414688,
                   2.094607679,1.177960959,2.086998287,1.147371259,2.12029442,1.138197348,
                   2.138994213, 1.114846113,],)),

       # some points behind the camera!
       np.array([[-0.8479983,  -0.52999894, -0.34690877],
                 [-0.93984618,  0.34159794, -0.16119387],
                 [-0.97738792,  0.21145412,  5.49068928]]),
       np.array([[ 958.48347896,  529.99410342],
                 [1229.87308989, 4625.05434521],
                 [4327.8166836 , 3183.44237796]]))


testutils.finish()
