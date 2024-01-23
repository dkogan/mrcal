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


np.random.seed(0)


@nps.broadcast_define( (('N','N',),),
                       () )
def worstdirection_stdev_ref(V):
    if V.size == 1:
        return np.sqrt(nps.atleast_dims(V,-1).ravel()[0])
    return np.sqrt( np.sort( np.linalg.eigvalsh(V) ) [-1] )

def random_positive_definite(shape):
    if len(shape) == 0:
        return np.abs(np.random.random(1)[0])

    if len(shape) == 1:
        if shape[0] != 1:
            raise Exception("Variance matrices should be square")
        return np.abs(np.random.random(1))

    if shape[-1] != shape[-2]:
        raise Exception("Variance matrices should be square")

    N = shape[-1]

    M = np.random.random(shape)
    return nps.matmult(nps.transpose(M), M)


for shape in ((2,3,5,5),
              (5,5),
              (3,2,2),
              (2,2),
              (1,1),
              (1,),
              ()):
    V = random_positive_definite(shape)

    testutils.confirm_equal( mrcal.worst_direction_stdev(V),
                             worstdirection_stdev_ref(V),
                             msg = f"Checking shape {shape}" )

testutils.finish()
