#!/usr/bin/env python3

r'''Look at the extrinsics drift of a camera pair over time

Described here:

https://mrcal.secretsauce.net/docs-2.5/differencing.html#extrinsics-diff

'''

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
                                      *models[0].imagersize(),
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



models_filenames = sys.argv[1:5]
models           = [mrcal.cameramodel(f) for f in models_filenames]

pairs = ( ( models[0],models[1]),
          ( models[2],models[3]) )

# The "before" extrinsic transformation
m0,m1 = pairs[0]
Rt01 = mrcal.compose_Rt( m0.Rt_cam_ref(),
                         m1.Rt_ref_cam())

# The "after" extrinsics transformation. I remap both cameras into the "before"
# space, so that we're looking at the extrinsics transformation in the "before"
# coord system. This will allow us to compare before and after
#
#   Rt_implied__0before_0after Rt_0after_1after Rt_implied__1after_1before
Rt_implied__0before_0after = compute_Rt_implied_01(pairs[0][0], pairs[1][0])
Rt_implied__1after_1before = compute_Rt_implied_01(pairs[1][1], pairs[0][1])

m0,m1 = pairs[1]
Rt_0after_1after = \
    mrcal.compose_Rt( m0.Rt_cam_ref(),
                      m1.Rt_ref_cam())

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
print(f"rotation:    {angle_deg:.3f}deg around the axis {axis}")


for i in range(len(models)):
    m = models[i]

    qcenter,dq_dv,_ = mrcal.project(np.array((0,0,1.)),
                                    *m.intrinsics(),
                                    get_gradients=True)

    # I now do a simple thing. I have v=[0,0,1] so dq_dv[:,2]=0. A small pitch
    # gives me dv = (0,sinth,costh) ~ (0,th,1). So dq = dq_dv[:,1]*th +
    # dq_dv[:,2] = dq_dv[:,1]*th so for a pitch: mag(dq/dth) = mag(dq_dv[:,1]).
    # Similarly for a yaw I have mag(dq_dv[:,0]). I find the worst one, and call
    # it good. I can do that because dq_dv will be diagonally dominant, and the
    # diagonal elements will be very similar. mrcal.rectified_resolution() does
    # this
    resolution__pix_per_rad = np.max(nps.transpose(nps.mag(dq_dv[:,:2])))
    resolution__pix_per_deg = resolution__pix_per_rad * np.pi/180.

    if 0:
        # More complicated, but probably not better. And not completed
        #
        # As the camera rotates, v shifts: rotate(v,r) ~ v + dv/dr dr, so the
        # projection shifts to q + dq/dv dv = q + dq/dv dv/dr dr
        #
        # Rodrigues rotation formula. th = mag(r), axis = normalize(r) = r/th
        #
        #   rotate(r,v) = v cos(th) + cross(axis, v) sin(th) + axis axist v (1 - cos(th))
        #
        # If th is small:
        #
        #   rotate(r,v) = v + cross(axis, v) th
        #               = v + [ axis1*v2-axis2*v1  axis2*v0-axis0*v2  axis0*v1-axis1*v0] th
        #
        # v = [0,0,1] so
        #
        #   rotate(r,v) = v + [ axis1  axis0  0] th
        #               = v + [ r1 r0 0 ]
        #
        # So
        dv_dr = np.array(((0,1,0),
                          (1,0,0),
                          (0,0,0)))

        dq_dr = nps.matmult(dq_dv,dv_dr)
        # I have dq_dr2 = 0, so lets ignore it
        dq_dr01 = dq_dr[:,:2]

        # Can finish this now

    print(f"Camera {i} has a resolution of {1./resolution__pix_per_deg:.3f} degrees per pixel at the center")
