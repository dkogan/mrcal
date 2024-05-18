#!/usr/bin/env python3

import sys
import numpy as np
import numpysane as nps
import os
testdir = os.path.dirname(os.path.realpath(__file__))

# I import the LOCAL mrcal since that's what I'm testing
sys.path[:0] = f"{testdir}/..",
import mrcal

from testutils import *
from test_calibration_helpers import grad

from test_poseutils_helpers import \
    R_from_r,                      \
    r_from_R,                      \
    compose_r

def wrap_r_unconditional(r, dr_dX = None):
    '''Unwrap a Rodrigues vector r

    returns r'     if dr_dX is None
    returns dr'_dX if dr_dX is not None

    I define rotations with a Rodrigues vector r = th v. Where v is the unit
    vector representing the rotation axis; and th is a scalar representing the
    magnitude or this rotation, in radians. Thus rotations th v and (th + 2pi*n)
    v are identical for all integer n. Furthermore the rotation (pi+th) v is
    equivalent to a rotation (pi-th) (-v). So usually we have th = mag(r) in
    [0,pi].

    For various analyses I want to convert a rotation r to its equivalent
    rotation 2*pi rad away:

      r' = v (th - 2pi)
         = r/magr (magr - 2pi)
         = r - r/magr 2pi
         = r (1 - 2pi/magr)

    Let

      k = 1 - 2pi/magr

    So

      r' = r k

    As noted above, usually magr <= pi, so k < 0. Thus

      magr' = -magr k

    Let's apply this wrapping a second time:

      r'' = r' (1 - 2pi/magr')
          = r k (1 + 2pi/magr/k)
          = r k (1 + (1-k) /k)
          = r (k + 1 - k)
          = r

    So this wrapping operation switches back/forth between two modes

    '''


    if dr_dX is None:
        return r - r/nps.mag(r) * 2.*np.pi

    # drw_dX = d( r - r/nps.mag(r) * 2.*np.pi )/dX
    #        = dr_dX - 2pi (dr_dX/nps.mag(r) + r d/dx (1/magr))
    #        = dr_dX - 2pi (dr_dX/nps.mag(r) - r /magr^2 /2magr 2rt dr_dX )
    #        = dr_dX - 2pi (dr_dX - r rt /norm2(r) dr_dX ) /nps.mag(r)
    #        = dr_dX + 2pi /nps.mag(r) (r rt /norm2(r) - I) dr_dX

    # dr_dX has shape (3,...) so r rt dr_dX also has shape (3,...) I clump
    # the trailing shapes so that dr_dX.shape is (3,N), and then reshape it
    # at the end
    s = dr_dX.shape[1:]
    # shape (3,N)
    dr_dX = nps.clump(dr_dX, n = -(dr_dX.ndim-1))

    drw_dX =    \
        dr_dX + \
        2.*np.pi / nps.mag(r) * nps.matmult(nps.outer(r,r)/nps.norm2(r) - np.eye(r.size),
                                            dr_dX)
    return drw_dX.reshape((3,) + s)

def wrap_r(r,
           *,
           r_match_direction = None,
           dr_dX             = None):

    '''Unwrap a Rodrigues vector r

    returns r'     if dr_dX is None
    returns dr'_dX if dr_dX is not None

    This function only wraps the argument if it needs to: if mag(r) > np.pi or
    if we're pointing opposite r_match_direction

    '''

    if r_match_direction is not None:
        if nps.inner(r, r_match_direction) > 0:
            return r if dr_dX is None else dr_dX
        return wrap_r_unconditional(r, dr_dX = dr_dX)
    if nps.mag(r) <= np.pi:
        return r if dr_dX is None else dr_dX
    return wrap_r_unconditional(r, dr_dX = dr_dX)


################### Check the behavior around the th=0, th=180 singularities.
################### Gradients and values should be correct
axes = \
    np.array(((1.,  2.,  0.1),
              (1,   0,   0),
              (0,   1,   0),
              (0,   0,   1),
              (-1,  0,   0),
              (0,   -1,  0),
              (0,   0,   -1)))
axes /= nps.dummy(nps.mag(axes), -1) # normalize
for axis in axes:
    for th0 in (-np.pi, 0, np.pi):
        for dth in (-1e-4, -1e-7, -1e-10, 0, 1e-10, 1e-7, 1e-4):

            r = (th0 + dth) * axis

            ######### r_from_R
            if False:

                R_ref      = R_from_r(r)
                R,dR_dr    = mrcal.R_from_r(r, get_gradients=True)
                dR_dr__ref = grad(R_from_r,r)

                confirm_equal( R,
                               R_ref,
                               msg=f'R_from_r result near a singularity. axis={axis}, th0={th0:.2f}, dth={dth}')
                confirm_equal( dR_dr,
                               dR_dr__ref,
                               msg=f'R_from_r J near a singularity. axis={axis}, th0={th0:.2f}, dth={dth}')

                r_ref             = r_from_R(R_ref)
                r_roundtrip,dr_dR = mrcal.r_from_R(R_ref, get_gradients=True)
                # need smaller step than dth
                dr_dR__ref        = grad(r_from_R,R_ref,
                                         switch = wrap_r_unconditional,
                                         step   = 1e-11)

                confirm_equal( wrap_r(r),
                               wrap_r(r_ref),
                               msg=f'r_from_R result near a singularity. axis={axis}, th0={th0:.2f}, dth={dth}')

                # I need to compare two different dr/dR. This isn't
                # well-defined: not all sets of 9 numbers are valid R. Only the
                # subspace spanned by the columns of dRflat_dr should be
                # evaluated when comparing dr_dR. So what I should be comparing
                # is
                #
                #   dr_dRflat * proj_into_subspace
                #
                # I compute proj_into_subspace. If I have an orthogonal subspace
                # then I add up the projections onto each basis vector:
                #
                #   p_projected = proj_into_subspace p
                #               = sum_i( bi inner(bi,p) )
                #               = sum_i( outer(bi,bi) ) p
                #
                # So proj_into_subspace = sum_i( outer(bi,bi) )
                #
                # I orthogonalize the valid subspace using the QR factorization
                # and I compute the projection matrix

                # shape (9,3)
                dRflat_dr = nps.clump(dR_dr, n=2)
                dR_basis = np.linalg.qr(dRflat_dr)[0]
                proj_into_dR = nps.matmult(dR_basis,dR_basis.T)

                dr_dRflat__ref = nps.matmult(nps.clump(wrap_r(r_ref,
                                                              dr_dX = dr_dR__ref),
                                                       n=-2),
                                             proj_into_dR)
                dr_dRflat      = nps.matmult(nps.clump(wrap_r(r_roundtrip,
                                                              dr_dX             = dr_dR,
                                                              r_match_direction = r_ref),
                                                       n=-2),
                                             proj_into_dR)

                confirm_equal( dr_dRflat,
                               dr_dRflat__ref,
                               relative    = True,
                               worstcase   = True,
                               eps         = 2e-2,
                               reldiff_eps = 1e-5,
                               msg         = f'r_from_R J near a singularity. axis={axis}, th0={th0:.2f}, dth={dth}')


                import IPython
                IPython.embed()
                sys.exit()



                nps.matmult(nps.clump(dr_dR__ref,n=-2),proj_into_dR)
                nps.matmult(nps.clump(dr_dR,     n=-2),proj_into_dR)



                # Test code to manually compute a forward difference
                delta = 1e-11
                R0 = R_ref
                R1 = np.array(R_ref)
                Delta = np.zeros(R0.shape, dtype=float)
                Delta[0,0] = delta
                R1 += Delta
                Deltaflat = Delta.ravel()
                print((r_from_R(R1) - r_from_R(R0)) / delta)

                # I applied a tweak to R. What if I evaluate its effect in the
                # valid-R subspace?
                import gnuplotlib as gp
                gp.plot(nps.cat(Deltaflat,
                                nps.inner(proj_into_dR, Deltaflat)),
                        legend=np.array(('orig',
                                         'projected-into-valid-subspace')))


                continue


            ######### compose_r
            if True:
                r0 = r
                r1 = np.array((-0.02, -1.2, 0.4),)


                ###### r01
                r01, dr01_dr0, dr01_dr1 = mrcal.compose_r(r0,r1, get_gradients = True)
                r01_ref                 = compose_r(r0,r1)

                dr01_dr0__ref = grad(lambda r0: compose_r(r0,r1),
                                    r0)
                dr01_dr1__ref = grad(lambda r1: compose_r(r0,r1),
                                    r1)
                confirm_equal( wrap_r(r01),
                               wrap_r(r01_ref),
                               msg=f'compose_r(r0,r1) r0 near a singularity. axis={axis}, th0={th0:.2f}, dth={dth}')
                confirm_equal( wrap_r(r01,     dr_dX = dr01_dr0),
                               wrap_r(r01_ref, dr_dX = dr01_dr0__ref),
                               msg=f'compose_r(r0,r1) dr01_dr0 near a singularity. axis={axis}, th0={th0:.2f}, dth={dth}')
                confirm_equal( wrap_r(r01,     dr_dX = dr01_dr1),
                               wrap_r(r01_ref, dr_dX = dr01_dr1__ref),
                               msg=f'compose_r(r0,r1) dr01_dr1 near a singularity. axis={axis}, th0={th0:.2f}, dth={dth}')

                ###### r10
                r10, dr10_dr1, dr10_dr0 = mrcal.compose_r(r1,r0, get_gradients = True)
                r10_ref = compose_r(r1,r0)
                dr10_dr0__ref = grad(lambda r0: compose_r(r1,r0),
                                     r0)
                dr10_dr1__ref = grad(lambda r1: compose_r(r1,r0),
                                     r1)
                confirm_equal( wrap_r(r10),
                               wrap_r(r10_ref),
                               msg=f'compose_r(r1,r0) r0 near a singularity. axis={axis}, th0={th0:.2f}, dth={dth}')
                confirm_equal( wrap_r(r10,     dr_dX = dr10_dr0),
                               wrap_r(r10_ref, dr_dX = dr10_dr0__ref),
                               msg=f'compose_r(r1,r0) dr10_dr0 near a singularity. axis={axis}, th0={th0:.2f}, dth={dth}')
                confirm_equal( wrap_r(r10,     dr_dX = dr10_dr1),
                               wrap_r(r10_ref, dr_dX = dr10_dr1__ref),
                               msg=f'compose_r(r1,r0) dr10_dr1 near a singularity. axis={axis}, th0={th0:.2f}, dth={dth}')

            ######### rotate_point_r
            if True:

                # Simple reference implementation. Should move this to
                # test_poseutils_helpers.py. And test_poseutils_lib.py should
                # use it
                def rotate_point_r(r,p):
                    return nps.inner(p, R_from_r(r))

                p = np.array((3., -0.2, -0.9),)

                pt, dpt_dr, dpt_dp = mrcal.rotate_point_r(r,p, get_gradients = True)
                pt_ref             = rotate_point_r(r,p)

                dpt_dr__ref = grad(lambda r: rotate_point_r(r,p),
                                   r)
                dpt_dp__ref = grad(lambda p: rotate_point_r(r,p),
                                   p)
                confirm_equal( pt,
                               pt_ref,
                               msg=f'rotate_point_r(r,p) r near a singularity. axis={axis}, th0={th0:.2f}, dth={dth}')
                confirm_equal(dpt_dr,
                              dpt_dr__ref,
                              msg=f'rotate_point_r(r,p) dpt_dr near a singularity. axis={axis}, th0={th0:.2f}, dth={dth}')
                confirm_equal(dpt_dp,
                              dpt_dp__ref,
                              msg=f'rotate_point_r(r,p) dpt_dp near a singularity. axis={axis}, th0={th0:.2f}, dth={dth}')

finish()
