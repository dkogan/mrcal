#!/usr/bin/python3

r'''Visualize noncentrality of a model

Given a model and query pixel this function

- Computes a ray from the camera center to the unprojection at infinity
- Samples different distances along this ray
- Projects them, and visualizes the difference projection from the sample point

As we get closer to the camera, stronger and stronger noncentral effects are
observed

'''

import sys
import numpy as np
import numpysane as nps
import gnuplotlib as gp
import re
import mrcal

model_filename = sys.argv[1]
qref = np.array((100,100), dtype=float)


model = mrcal.cameramodel(model_filename)

lensmodel,intrinsics_data = model.intrinsics()

if not mrcal.lensmodel_metadata_and_config(lensmodel)['noncentral']:
    print("The given model isn't noncentral. Nothing to do", file=sys.stderr)
    sys.exit(1)

if not re.match('^LENSMODEL_CAHVORE_', lensmodel):
    print("This is only implemented for CAHVORE today", file=sys.stderr)
    sys.exit(1)

# Special-case to centralize CAHVORE
intrinsics_data_centralized = intrinsics_data.copy()
intrinsics_data_centralized[-3:] = 0

v_at_infinity = \
    mrcal.unproject(qref,
                    lensmodel, intrinsics_data_centralized,
                    normalize = True)

Ndistances = 100
d = np.linspace(0.01, 10., Ndistances)

# shape (Ndistances, 3)
p = nps.dummy(d, -1) * v_at_infinity

# shape (Ndistances, 2)
q = mrcal.project(p, *model.intrinsics())

# shape (Ndistances,)
qshift = nps.mag(q - qref)

gp.plot( d, qshift,
         _with = 'linespoints',
         xlabel = 'Distance (m)',
         ylabel = 'Pixel shift (pixels)',
         wait = True )
