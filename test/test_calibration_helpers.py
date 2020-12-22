#!/usr/bin/python3

import sys
import numpy as np
import numpysane as nps
import copy
import os

# I import the LOCAL mrcal since that's what I'm testing
testdir = os.path.dirname(os.path.realpath(__file__))
sys.path[:0] = f"{testdir}/..",
import mrcal

def sample_dqref(observations,
                 pixel_uncertainty_stdev,
                 make_outliers = False):

    # Outliers have weight < 0. The code will adjust the outlier observations
    # also. But that shouldn't matter: they're outliers so those observations
    # should be ignored
    weight  = observations[...,-1]
    q_noise = np.random.randn(*observations.shape[:-1], 2) * pixel_uncertainty_stdev / nps.mv(nps.cat(weight,weight),0,-1)

    if make_outliers:
        if not hasattr(sample_dqref, 'idx_outliers_ref_flat'):
            NobservedPoints = observations.size // 3
            sample_dqref.idx_outliers_ref_flat = \
                np.random.choice( NobservedPoints,
                                  (NobservedPoints//100,), # 1% outliers
                                  replace = False )
        nps.clump(q_noise, n=3)[sample_dqref.idx_outliers_ref_flat, :] *= 20

    observations_perturbed = observations.copy()
    observations_perturbed[...,:2] += q_noise
    return q_noise, observations_perturbed


def sorted_eig(C):
    'like eig(), but the results are sorted by eigenvalue'
    l,v = np.linalg.eig(C)
    i = np.argsort(l)
    return l[i], v[:,i]


def grad(f, x, step=1e-6):
    r'''Computes df/dx at x

    f is a function of one argument. If the input has shape Si and the output
    has shape So, the returned gradient has shape So+Si. This applies central
    differences

    '''

    d     = x*0
    dflat = d.ravel()

    def df_dxi(i, d,dflat):

        dflat[i] = step
        fplus  = f(x+d)
        fminus = f(x-d)
        j = (fplus-fminus)/(2.*step)
        dflat[i] = 0
        return j

    # grad variable is in first dim
    Jflat = nps.cat(*[df_dxi(i, d,dflat) for i in range(len(dflat))])
    # grad variable is in last dim
    Jflat = nps.mv(Jflat, 0, -1)
    return Jflat.reshape( Jflat.shape[:-1] + d.shape )
