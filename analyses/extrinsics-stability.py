#!/usr/bin/python3

import sys
import numpy as np
import numpysane as nps
import gnuplotlib as gp
import mrcal


def compute_Rt_implied_01(*models):
    lensmodels      = [model.intrinsics()[0] for model in models]
    intrinsics_data = [model.intrinsics()[1] for model in models]

    # v  shape (...,Ncameras,Nheight,Nwidth,...)
    # q0 shape (...,         Nheight,Nwidth,...)
    v,q0 = \
        mrcal.sample_imager_unproject(60, None,
                                      ,*models[0].imagersize(),
                                      lensmodels, intrinsics_data,
                                      normalize = True)



    distance     = (1.0, 100.)
    focus_center = None
    focus_radius = 500



    if distance is None:
        atinfinity = True
        distance   = np.ones((1,), dtype=float)
    else:
        atinfinity = False
        distance   = nps.atleast_dims(np.array(distance), -1)
    distance   = nps.mv(distance.ravel(), -1,-4)

    if focus_center is None:
        focus_center = (models[0].imagersize() - 1.)/2.

    implied_Rt10 = \
        mrcal.implied_Rt10__from_unprojections(q0,
                                               # shape (len(distance),Nheight,Nwidth,3)
                                               v[0,...] * distance,
                                               v[1,...],
                                               atinfinity   = atinfinity,
                                               focus_center = focus_center,
                                               focus_radius = focus_radius)

    return mrcal.invert_Rt(implied_Rt10)



p = sys.argv[1:5]


pairs = ( ( mrcal.cameramodel(p[0]), mrcal.cameramodel(p[1])),
          ( mrcal.cameramodel(p[2]), mrcal.cameramodel(p[3])) )

# The "before" extrinsic transformation
m0,m1 = pairs[0]
Rt01 = mrcal.compose_Rt( m0.extrinsics_Rt_fromref(),
                         m1.extrinsics_Rt_toref())

# The "after" extrinsics transformation. I remap both cameras into the "before"
# space, so that we're looking at the extrinsics transformation in the "before"
# coord system. This will allow us to compare before and after
#
#   Rt_implied__0before_0after Rt_0after_1after Rt_implied__1after_1before
Rt_implied__0before_0after = compute_Rt_implied_01(pairs[0][0], pairs[1][0])
Rt_implied__1after_1before = compute_Rt_implied_01(pairs[1][1], pairs[0][1])

m0,m1 = pairs[1]
Rt_0after_1after = \
    mrcal.compose_Rt( m0.extrinsics_Rt_fromref(),
                      m1.extrinsics_Rt_toref())

Rt01_after_extrinsicsbefore = \
    mrcal.compose_Rt( Rt_implied__0before_0after,
                      Rt_0after_1after,
                      Rt_implied__1after_1before )

# I have the two relative transforms. If camera0 is fixed, how much am I moving
# camera1?
Rt_1before_1after = mrcal.compose_Rt(mrcal.invert_Rt(Rt01), Rt01_after_extrinsicsbefore)

r_1before_1after = mrcal.r_from_R(Rt_1before_1after[:3,:])
t_1before_1after = Rt_1before_1after[3,:]
magnitude        = nps.mag(t_1before_1after)
direction        = t_1before_1after/magnitude
angle            = nps.mag(r_1before_1after)
axis             = r_1before_1after/angle
angle_deg        = angle*180./np.pi

np.set_printoptions(precision=2)
print(f"translation: {magnitude*1000:.2f}mm in the direction {direction}")
print(f"rotation:    {angle_deg:.2f}deg around the axis {axis}")
