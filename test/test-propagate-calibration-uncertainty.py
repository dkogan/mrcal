#!/usr/bin/env python3

# Copyright (c) 2017-2023 California Institute of Technology ("Caltech"). U.S.
# Government sponsorship acknowledged. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0


import sys
import os
import numpy as np
import numpysane as nps

testdir = os.path.dirname(os.path.realpath(__file__))

# I import the LOCAL mrcal since that's what I'm testing
sys.path[:0] = f"{testdir}/..",
import mrcal

import testutils

m = mrcal.cameramodel(f"{testdir}/../doc/data/figueroa-overpass-looking-S/opencv8-0.cameramodel")
optimization_inputs = m.optimization_inputs()
Nstate = mrcal.num_states(**optimization_inputs)




@nps.broadcast_define( (('N','N',),),
                       () )
def worstdirection_stdev_ref(V):
    if V.size == 1:
        return np.sqrt(nps.atleast_dims(V,-1).ravel()[0])
    return np.sqrt( np.sort( np.linalg.eigvalsh(V) ) [-1] )

@nps.broadcast_define( (('N','N',),),
                       () )
def rms_stdev_ref(V):
    if V.size == 1:
        return np.sqrt(nps.atleast_dims(V,-1).ravel()[0])
    l = np.linalg.eigvalsh(V)
    return np.sqrt(np.mean(l))

for shape in ((2,3,5),
              (3,5),
              (5,),
              (3,2),
              (2,),
              (1,),
              ()):

    dF_dbpacked = np.random.random(shape + (Nstate,))

    if len(shape) > 0:
        Var_shape_ref = shape + (shape[-1],)
    else:
        Var_shape_ref = ()


    # THIS test doesn't know the true uncertainty;
    # test-projection-uncertainty.py does that. So here I assume that
    # the covariances are reported correctly, and I make sure that
    # everything else is self-consistent
    Var = \
        mrcal.model_analysis. \
        _propagate_calibration_uncertainty( \
            what                = 'covariance',
            dF_dbpacked         = dF_dbpacked,
            optimization_inputs = optimization_inputs)

    testutils.confirm_equal( Var.shape,
                             Var_shape_ref,
                             msg = f"Checking covariance.shape for shape {shape}" )

    worstdirection_stdev = \
        mrcal.model_analysis. \
        _propagate_calibration_uncertainty( \
            what                = 'worstdirection-stdev',
            dF_dbpacked         = dF_dbpacked,
            optimization_inputs = optimization_inputs)
    testutils.confirm_equal( worstdirection_stdev.shape,
                             worstdirection_stdev_ref(Var).shape,
                             msg = f"Checking worstdirection_stdev.shape for shape {shape}" )
    testutils.confirm_equal( worstdirection_stdev,
                             worstdirection_stdev_ref(Var),
                             msg = f"Checking worstdirection_stdev for shape {shape}" )

    rms_stdev = \
        mrcal.model_analysis. \
        _propagate_calibration_uncertainty( \
            what                = 'rms-stdev',
            dF_dbpacked         = dF_dbpacked,
            optimization_inputs = optimization_inputs)
    testutils.confirm_equal( rms_stdev.shape,
                             rms_stdev_ref(Var).shape,
                             msg = f"Checking rms_stdev.shape for shape {shape}" )
    testutils.confirm_equal( rms_stdev,
                             rms_stdev_ref(Var),
                             msg = f"Checking rms_stdev for shape {shape}" )


testutils.finish()
