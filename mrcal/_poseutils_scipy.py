#!/usr/bin/python3

import sys
import numpy as np
import numpysane as nps


def quat_from_R(R):
    r"""Convert a rotation defined as a rotation matrix to a unit quaternion

SYNOPSIS

    print(R.shape)
    ===>
    (3,3)

    quat = mrcal.quat_from_R(R)

    print(quat.shape)
    ===>
    (4,)

    c = quat[0]
    s = nps.mag(quat[1:])

    rotation_magnitude = 2. * np.arctan2(s,c)

    rotation_axis = quat[1:] / s

This is mostly for compatibility with some old stuff. mrcal doesn't use
quaternions anywhere. Test this thoroughly before using.

This function supports broadcasting fully.

ARGUMENTS

- R: array of shape (3,3,). The rotation matrix that defines the rotation.

RETURNED VALUE

We return an array of unit quaternions. Each broadcasted slice has shape (4,).
The values in the array are (u,i,j,k)

LICENSE AND COPYRIGHT

The implementation comes directly from the scipy project, the from_dcm()
function in

  https://github.com/scipy/scipy/blob/master/scipy/spatial/transform/rotation.py

Commit: 1169d27ad47a29abafa8a3d2cb5b67ff0df80a8f

License:

Copyright (c) 2001-2002 Enthought, Inc.  2003-2019, SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    """

    # extra broadcasted shape
    extra_dims = R.shape[:-2]
    # R.shape = (..., 3,3) with some non-empty ...
    R = nps.atleast_dims(R, -3)

    # R.shape = (N,3,3)
    R = nps.clump(R, n=R.ndim-2)

    num_rotations = R.shape[0]

    decision_matrix = np.empty((num_rotations, 4))
    decision_matrix[:, :3] = R.diagonal(axis1=1, axis2=2)
    decision_matrix[:, -1] = decision_matrix[:, :3].sum(axis=1)
    choices = decision_matrix.argmax(axis=1)

    quat = np.empty((num_rotations, 4))

    ind = np.nonzero(choices != 3)[0]
    i = choices[ind]
    j = (i + 1) % 3
    k = (j + 1) % 3

    quat[ind, i+1] = 1 - decision_matrix[ind, -1] + 2 * R[ind, i, i]
    quat[ind, j+1] = R[ind, j, i] + R[ind, i, j]
    quat[ind, k+1] = R[ind, k, i] + R[ind, i, k]
    quat[ind, 0  ] = R[ind, k, j] - R[ind, j, k]

    ind = np.nonzero(choices == 3)[0]
    quat[ind, 1] = R[ind, 2, 1] - R[ind, 1, 2]
    quat[ind, 2] = R[ind, 0, 2] - R[ind, 2, 0]
    quat[ind, 3] = R[ind, 1, 0] - R[ind, 0, 1]
    quat[ind, 0] = 1 + decision_matrix[ind, -1]

    quat /= np.linalg.norm(quat, axis=1)[:, None]

    return quat.reshape(extra_dims + (4,))
