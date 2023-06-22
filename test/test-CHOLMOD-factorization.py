#!/usr/bin/env python3

r'''Tests the CHOLMOD_factorization python class'''

import sys
import numpy as np
import numpysane as nps
import os

testdir = os.path.dirname(os.path.realpath(__file__))

# I import the LOCAL mrcal since that's what I'm testing
sys.path[:0] = f"{testdir}/..",
import mrcal
import testutils

from scipy.sparse import csr_matrix

indptr  = np.array([0, 2, 3, 6, 8])
indices = np.array([0, 2, 2, 0, 1, 2, 1, 2])
data    = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)

Jsparse = csr_matrix((data, indices, indptr))
Jdense  = Jsparse.toarray()
Jdense_ref = \
    np.array(((1, 0, 2),
              (0, 0, 3),
              (4, 5, 6),
              (0, 7, 8)), dtype=float)

testutils.confirm_equal(Jdense,
                        Jdense_ref,
                        relative  = True,
                        worstcase = True,
                        eps       = 1e-6,
                        msg       = "csr_matrix representation works as expected")

bt  = np.array(((1., 5., 3.), (2., -2., -8)))

F  = mrcal.CHOLMOD_factorization(Jsparse)
xt = F.solve_xt_JtJ_bt(bt)

JtJ    = nps.matmult(nps.transpose(Jdense), Jdense)
xt_ref = nps.transpose(np.linalg.solve(JtJ, nps.transpose(bt)))

testutils.confirm_equal(xt, xt_ref,
                        relative  = True,
                        worstcase = True,
                        eps       = 1e-6,
                        msg       = "solve_xt_JtJ_bt produces the correct result")

testutils.finish()
