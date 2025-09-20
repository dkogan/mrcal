#!/usr/bin/env python3

r'''Tests mrcal_rectification_maps() and mrcal.rectification_maps()
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


# I want to generate a stereo pair. I tweak the intrinsics a bit
model0 = mrcal.cameramodel(f"{testdir}/data/cam0.opencv8.cameramodel")
model1 = mrcal.cameramodel(model0)

intrinsics_data = model1.intrinsics()[1]
intrinsics_data[ :2] *= np.array((1.01, -0.98))
intrinsics_data[2:4] += np.array((50, 80.))
model1.intrinsics( intrinsics = (model1.intrinsics()[0],
                                 intrinsics_data) )

# Left-right stereo, with sizeable rotation and position fuzz.
# I especially make sure there's a forward/back shift
rt01 = np.array((0.1, 0.2, 0.05,  3.0, 0.2, 1.0))
model1.rt_ref_cam( mrcal.compose_rt(model0.rt_ref_cam(),
                                    rt01))

for rectification in ('LENSMODEL_LATLON', 'LENSMODEL_PINHOLE'):
    for zoom in (0.6, 1., 10.):
        def apply_zoom(model, zoom):
            intrinsics_data_zoomed = np.array(model.intrinsics()[1])
            intrinsics_data_zoomed[:2] *= zoom
            return (model.intrinsics()[0],
                    intrinsics_data_zoomed)

        model0_zoom = mrcal.cameramodel(model0)
        model1_zoom = mrcal.cameramodel(model1)
        for m in (model0_zoom, model1_zoom):
            m.intrinsics( intrinsics = apply_zoom(m,zoom) )

        # I use the canonical rectified-system function here to make sure that this
        # test checks only the rectification_maps function
        az_fov_deg = 90
        el_fov_deg = 50
        models_rectified = \
            mrcal.stereo._rectified_system_python( (model0_zoom, model1_zoom),
                                                   az_fov_deg = az_fov_deg/zoom,
                                                   el_fov_deg = el_fov_deg/zoom,
                                                   pixels_per_deg_az = -1./8.,
                                                   pixels_per_deg_el = -1./4.,
                                                   rectification_model = rectification)

        rectification_maps_ref = \
            mrcal.stereo._rectification_maps_python((model0,model1),
                                                    models_rectified)
        rectification_maps_ref = np.array(rectification_maps_ref)

        rectification_maps = \
            mrcal.rectification_maps((model0,model1),
                                     models_rectified)

        # some pinhole maps have crazy behavior on the edges (WAAAAAAY out of
        # view), and I ignore it
        rectification_maps    [rectification_maps     > 1e6] = 0
        rectification_maps_ref[rectification_maps_ref > 1e6] = 0

        testutils.confirm_equal(rectification_maps,
                                rectification_maps_ref,
                                msg=f'Pixel error with ({rectification}). Zoom = {zoom}',
                                worstcase = True,
                                eps = 1e-6)

testutils.finish()
