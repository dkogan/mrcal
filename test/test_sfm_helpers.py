import sys
import numpy as np
import numpysane as nps
import os

testdir = os.path.dirname(os.path.realpath(__file__))

# I import the LOCAL mrcal since that's what I'm testing
sys.path[:0] = f"{testdir}/..",
import mrcal


############# Set up my world, and compute all the perfect positions, pixel
############# observations of everything
def generate_world(Npoints_fixed = None):
    model = mrcal.cameramodel(f"{testdir}/data/cam0.opencv8.cameramodel")
    imagersize = model.imagersize()
    lensmodel,intrinsics_data = model.intrinsics()

    pref_true = np.array((( 10.,  20., 100.),
                          ( 25.,  30.,  90.),
                          (  5.,  10.,  94.),
                          (-45., -20.,  95.),
                          (-35.,  14.,  77.),
                          (  5.,  -0., 110.),
                          (  1.,  50.,  50.)))

    # The points are all somewhere at +z. So the camera rotations are all ~ identity
    rt_cam_ref_true = np.array(((-0.1, -0.07, 0.01,  10.0, 4.0, -7.0),
                                (-0.01, 0.05,-0.02,  30.0,-8.0, -8.0),
                                (-0.1,  0.03,-0.03,  10.0,-9.0, 20.0),
                                ( 0.04,-0.04, 0.03, -20.0, 2.0,-11.0),
                                ( 0.01, 0.05,-0.05, -10.0, 3.0,  9.0)))

    # shape (Ncamposes, Npoints, 3)
    pcam_true = mrcal.transform_point_rt(nps.mv(rt_cam_ref_true, -2,-3),
                                    pref_true)

    # shape (Ncamposes, Npoints, 2)
    qcam_true = mrcal.project(pcam_true, lensmodel, intrinsics_data)

    #print(repr(np.random.randn(*qcam_true.shape) * 1.0))
    qcam_noise = np.array([[[-0.3008229 , -0.06386847],
                            [ 0.15690675,  0.35829428],
                            [-1.15637874, -0.68073402],
                            [-0.2271372 ,  0.6229026 ],
                            [ 0.59155325,  0.68842454],
                            [-2.24109699, -0.25995747],
                            [-0.70677139,  0.13430803]],
                           [[-1.07811457, -0.1816532 ],
                            [-1.08579286, -1.07642887],
                            [ 0.44522045,  2.37729233],
                            [ 0.53204839,  0.17330028],
                            [-1.19966246,  0.95206635],
                            [ 0.84925049,  1.44290438],
                            [-0.26285984,  0.62597288]],
                           [[-0.57862724,  0.73605612],
                            [-0.23827258, -0.55540618],
                            [-0.29209469,  0.31539725],
                            [-0.7524089 ,  1.83662404],
                            [ 0.49367672,  0.65044992],
                            [ 1.04093529,  0.61213177],
                            [-0.61648765,  0.49728066]],
                           [[-1.06310686,  0.56850477],
                            [-0.0448958 , -0.26478884],
                            [-0.85378168, -1.01622018],
                            [ 0.91716149,  1.35368339],
                            [-0.73417391,  0.51115139],
                            [ 0.61964307, -1.17082615],
                            [-1.32632285,  0.29379208]],
                           [[ 0.62453505, -1.76356842],
                            [-0.21081138,  1.05730862],
                            [-0.63014551, -0.4755803 ],
                            [-0.38298468,  1.31122154],
                            [ 0.9190388 , -2.19933583],
                            [ 2.22796702, -0.83784105],
                            [ 0.79775114, -1.9516097 ]]]) / 10.

    qcam_noisy = qcam_true + qcam_noise

    # Observations are incomplete. Not all points are observed from everywhere
    indices_point_camera = \
        np.array(((0, 1),
                  (0, 2),
                  (0, 4),
                  (1, 0),
                  (1, 1),
                  (1, 4),
                  (2, 0),
                  (2, 1),
                  (2, 2),
                  (3, 1),
                  (3, 2),
                  (3, 3),
                  (3, 4),
                  (4, 0),
                  (4, 3),
                  (4, 4),
                  (5, 0),
                  (5, 1),
                  (5, 2),
                  (5, 3),
                  (5, 4),
                  (6, 2),
                  (6, 3),
                  (6, 4)),
                 dtype = np.int32)


    ########### add noise

    # The seed points array is the true array, but corrupted by noise. All the
    # points are observed at some point
    #print(repr((np.random.random(points.shape)-0.5)/3))
    points_noise = np.array([[-0.16415198,  0.10697666,  0.07137079],
                             [-0.02353459,  0.07269802,  0.05804911],
                             [-0.05218085, -0.09302461, -0.16626839],
                             [ 0.03649283, -0.04345566, -0.1589429 ],
                             [-0.05530528,  0.03942736, -0.02755858],
                             [-0.16252387,  0.07792151, -0.12200266],
                             [-0.02611094, -0.13695699,  0.06799326]])

    if Npoints_fixed is not None:
        # The fixed points are perfect
        points_noise[-Npoints_fixed:, :] = 0

    pref_noisy = pref_true * (1. + points_noise)

    Ncamposes,Npoints = pcam_true.shape[:2]
    ipoints   = indices_point_camera[:,0]
    icamposes = indices_point_camera[:,1]
    qcam_indexed_true  = nps.clump(qcam_true,  n=2)[icamposes*Npoints+ipoints,:]
    qcam_indexed_noisy = nps.clump(qcam_noisy, n=2)[icamposes*Npoints+ipoints,:]

    observations_true = \
        nps.glue(qcam_indexed_true,
                 nps.transpose(np.ones((qcam_indexed_true.shape[0],))),
                 axis = -1)

    observations_noisy = \
        nps.glue(qcam_indexed_noisy,
                 nps.transpose(np.ones((qcam_indexed_noisy.shape[0],))),
                 axis = -1)

    #print(repr((np.random.random(rt_cam_ref_true.shape)-0.5)/10))
    rt_cam_ref_noise = \
        np.array([[-0.00781127, -0.04067386, -0.01039731,  0.02057068, -0.0461704 ,  0.02112582],
                  [-0.02466267, -0.01445134, -0.01290107, -0.01956848,  0.04604318,  0.0439563 ],
                  [-0.02335697,  0.03171099, -0.00900416, -0.0346394 , -0.0392821 ,  0.03892269],
                  [ 0.00229462, -0.01716853,  0.01336239, -0.0228473 , -0.03919978,  0.02671576],
                  [ 0.03782446, -0.016981  ,  0.03949906, -0.03256744,  0.02496247,  0.02924358]])
    rt_cam_ref_noisy = rt_cam_ref_true * (1.0 + rt_cam_ref_noise)


    return                    \
        model,                \
        imagersize,           \
        lensmodel,            \
        intrinsics_data,      \
        indices_point_camera, \
                              \
        pref_true,            \
        rt_cam_ref_true,      \
        qcam_true,            \
        observations_true,    \
                              \
        pref_noisy,           \
        rt_cam_ref_noisy,     \
        qcam_noisy,           \
        observations_noisy

