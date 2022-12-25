#!/usr/bin/python3

import sys
import mrcal
import numpy as np
import numpysane as nps
import pickle



def project_adaptive_rectification(p,
                                   *, # cookie
                                   qx,
                                   az_domain,
                                   fy, cy):

    # Python loop. Yuck!
    @nps.broadcast_define( ((),),
                           ())
    def interp_one(az,qy):
        return \
            np.interp(az,
                      az_domain[qy], qx[qy])



    azel  = mrcal.project_latlon(p)
    az,el = nps.mv(azel, -1, 0)

    q = np.zeros(azel.shape,
                 dtype=float)

    q[...,1] = el * fy + cy

    # sy is in [0,1] between [floor(qy),floor(qy)+1]
    sy = q[...,1] % 1

    q[...,0] = \
        interp_one(az,np.floor(q[...,1]).astype(int)  ) * (1-sy) + \
        interp_one(az,np.floor(q[...,1]).astype(int)+1) * sy

    return q


def unproject_adaptive_rectification(q,
                                     *, # cookie
                                     qx,
                                     az_domain,
                                     fy, cy):

    # Python loop. Yuck!
    @nps.broadcast_define( ((),),
                           ())
    def interp_one(qx_here,qy):
        return \
            np.interp(qx_here,
                      qx[qy], az_domain[qy])


    # sy is in [0,1] between [floor(qy),floor(qy)+1]
    sy = q[...,1] % 1


    azel = np.zeros(q.shape,
                    dtype=float)
    azel[...,1] = (q[...,1] - cy) / fy
    azel[...,0] = \
        interp_one(q[...,0],np.floor(q[...,1]).astype(int)  ) * (1-sy) + \
        interp_one(q[...,0],np.floor(q[...,1]).astype(int)+1) * sy

    return mrcal.unproject_latlon(azel)


if __name__ == '__main__':


    filename = "/tmp/cookie"
    with open(filename, "rb") as f:
        (qx,
         az_domain,
         fy,cy) = \
             pickle.load(f)

    # q_nominal = ((1574.738,855.411))
    p = np.array([0.57652792, 0.0982791 , 0.81114535])
    # q_adaptive = ((1096.8,855.7))

    q = \
        project_adaptive_rectification(p,
                                       qx          = qx,
                                       az_domain   = az_domain,
                                       fy          = fy,
                                       cy          = cy)
    pp = \
        unproject_adaptive_rectification(q,
                                         qx          = qx,
                                         az_domain   = az_domain,
                                         fy          = fy,
                                         cy          = cy)

    print(p)
    print(pp)
    print(q)
