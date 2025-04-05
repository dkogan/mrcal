#!/usr/bin/python3

# Copyright (c) 2017-2023 California Institute of Technology ("Caltech"). U.S.
# Government sponsorship acknowledged. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

'''Routines for analysis of camera projection

This is largely dealing with uncertainty and projection diff operations.

All functions are exported into the mrcal module. So you can call these via
mrcal.model_analysis.fff() or mrcal.fff(). The latter is preferred.

'''

import numpy as np
import numpysane as nps
import sys
import re
import mrcal


def implied_Rt10__from_unprojections(q0, p0, v1,
                                     *,
                                     weights      = None,
                                     atinfinity   = True,
                                     focus_center = np.zeros((2,), dtype=float),
                                     focus_radius = 1.0e8):

    r'''Compute the implied-by-the-intrinsics transformation to fit two cameras' projections

SYNOPSIS

    models = ( mrcal.cameramodel('cam0-dance0.cameramodel'),
               mrcal.cameramodel('cam0-dance1.cameramodel') )

    lensmodels      = [model.intrinsics()[0] for model in models]
    intrinsics_data = [model.intrinsics()[1] for model in models]

    # v  shape (...,Ncameras,Nheight,Nwidth,...)
    # q0 shape (...,         Nheight,Nwidth,...)
    v,q0 = \
        mrcal.sample_imager_unproject(60, None,
                                      *models[0].imagersize(),
                                      lensmodels, intrinsics_data,
                                      normalize = True)
    implied_Rt10 = \
        mrcal.implied_Rt10__from_unprojections(q0, v[0,...], v[1,...])

    q1 = mrcal.project( mrcal.transform_point_Rt(implied_Rt10, v[0,...]),
                        *models[1].intrinsics())

    projection_diff = q1 - q0

When comparing projections from two lens models, it is usually necessary to
align the geometry of the two cameras, to cancel out any transformations implied
by the intrinsics of the lenses. This transformation is computed by this
function, used primarily by mrcal.show_projection_diff() and the
mrcal-show-projection-diff tool.

What are we comparing? We project the same world point into the two cameras, and
report the difference in projection. Usually, the lens intrinsics differ a bit,
and the implied origin of the camera coordinate systems and their orientation
differ also. These geometric uncertainties are baked into the intrinsics. So
when we project "the same world point" we must apply a geometric transformation
to compensate for the difference in the geometry of the two cameras. This
transformation is unknown, but we can estimate it by fitting projections across
the imager: the "right" transformation would result in apparent low projection
diffs in a wide area.

The primary inputs are unprojected gridded samples of the two imagers, obtained
with something like mrcal.sample_imager_unproject(). We grid the two imagers,
and produce normalized observation vectors for each grid point. We pass the
pixel grid from camera0 in q0, and the two unprojections in p0, v1. This
function then tries to find a transformation to minimize

  norm2( project(camera1, transform(p0)) - q1 )

We return an Rt transformation to map points in the camera0 coordinate system to
the camera1 coordinate system. Some details about this general formulation are
significant:

- The subset of points we use for the optimization
- What kind of transformation we use

In most practical usages, we would not expect a good fit everywhere in the
imager: areas where no chessboards were observed will not fit well, for
instance. From the point of view of the fit we perform, those ill-fitting areas
should be treated as outliers, and they should NOT be a part of the solve. How
do we specify the well-fitting area? The best way is to use the model
uncertainties to pass the weights in the "weights" argument (see
show_projection_diff() for an implementation). If uncertainties aren't
available, or if we want a faster solve, the focus region can be passed in the
focus_center, focus_radius arguments. By default, these are set to encompass the
whole imager, since the uncertainties would take care of everything, but without
uncertainties (weights = None), these should be set more discriminately. It is
possible to pass both a focus region and weights, but it's probably not very
useful.

Unlike the projection operation, the diff operation is NOT invariant under
geometric scaling: if we look at the projection difference for two points at
different locations along a single observation ray, there will be a variation in
the observed diff. This is due to the geometric difference in the two cameras.
If the models differed only in their intrinsics parameters, then this would not
happen. Thus this function needs to know how far from the camera it should look.
By default (atinfinity = True) we look out to infinity. In this case, p0 is
expected to contain unit vectors. To use any other distance, pass atinfinity =
False, and pass POINTS in p0 instead of just observation directions. v1 should
always be normalized. Generally the most confident distance will be where the
chessboards were observed at calibration time.

Practically, it is very easy for the unprojection operation to produce nan or
inf values. And the weights could potentially have some invalid values also.
This function explicitly checks for such illegal data in p0, v1 and weights, and
ignores those points.

ARGUMENTS

- q0: an array of shape (Nh,Nw,2). Gridded pixel coordinates covering the imager
  of both cameras

- p0: an array of shape (...,Nh,Nw,3). An unprojection of q0 from camera 0. If
  atinfinity, this should contain unit vectors, else it should contain points in
  space at the desired distance from the camera. This array may have leading
  dimensions that are all used in the fit. These leading dimensions correspond
  to those in the "weights" array

- v1: an array of shape (Nh,Nw,3). An unprojection of q0 from camera 1. This
  should always contain unit vectors, regardless of the value of atinfinity

- weights: optional array of shape (...,Nh,Nw); None by default. If given, these
  are used to weigh each fitted point differently. Usually we use the projection
  uncertainties to apply a stronger weight to more confident points. If omitted
  or None, we weigh each point equally. This array may have leading dimensions
  that are all used in the fit. These leading dimensions correspond to those in
  the "p0" array

- atinfinity: optional boolean; True by default. If True, we're looking out to
  infinity, and I compute a rotation-only fit; a full Rt transformation is still
  returned, but Rt[3,:] is 0; p0 should contain unit vectors. If False, I'm
  looking out to a finite distance, and p0 should contain 3D points specifying
  the positions of interest.

- focus_center: optional array of shape (2,); (0,0) by default. Used to indicate
  that we're interested only in a subset of pixels q0, a distance focus_radius
  from focus_center. By default focus_radius is LARGE, so we use all the points.
  This is intended to be used if no uncertainties are available, and we need to
  manually select the focus region.

- focus_radius: optional value; LARGE by default. Used to indicate that we're
  interested only in a subset of pixels q0, a distance focus_radius from
  focus_center. By default focus_radius is LARGE, so we use all the points. This
  is intended to be used if no uncertainties are available, and we need to
  manually select the focus region.

RETURNED VALUE

An array of shape (4,3), representing an Rt transformation from camera0 to
camera1. If atinfinity then we're computing a rotation-fit only, but we still
report a full Rt transformation with the t component set to 0

    '''

    # This is very similar in spirit to what compute_Rcorrected_dq_dintrinsics() did
    # (removed in commit 4240260), but that function worked analytically, while this
    # one explicitly computes the rotation by matching up known vectors.

    import scipy.optimize

    ### flatten all the input arrays
    # shape (N,2)
    q0 = nps.clump(q0, n=q0.ndim-1)
    # shape (M,N,3)
    p0 = nps.transpose(nps.clump(nps.mv( nps.atleast_dims(p0, -3),
                                         -1,-3),
                                 n=-2))
    # shape (N,3)
    v1 = nps.clump(v1, n=v1.ndim-1)

    if weights is None:
        weights = np.ones(p0.shape[:-1], dtype=float)
    else:
        # shape (..., Nh,Nw) -> (M,N,) where N = Nh*Nw
        weights = nps.clump( nps.clump(weights, n=-2),
                             n = weights.ndim-2)

        # Any inf/nan weight or vector are set to 0
        weights = weights.copy()
        weights[ ~np.isfinite(weights) ] = 0.0

    p0 = p0.copy()
    v1 = v1.copy()

    # p0 had shape (N,3). Collapse all the leading dimensions into one
    # And do the same for weights

    i_nan_p0 = ~np.isfinite(p0)
    p0[i_nan_p0] = 0.
    weights[i_nan_p0[...,0]] = 0.0
    weights[i_nan_p0[...,1]] = 0.0
    weights[i_nan_p0[...,2]] = 0.0

    i_nan_v1 = ~np.isfinite(v1)
    v1[i_nan_v1] = 0.
    weights[..., i_nan_v1[...,0]] = 0.0
    weights[..., i_nan_v1[...,1]] = 0.0
    weights[..., i_nan_v1[...,2]] = 0.0

    # We try to match the geometry in a particular region
    q_off_center = q0 - focus_center
    i = nps.norm2(q_off_center) < focus_radius*focus_radius
    if np.count_nonzero(i)<3:
        raise Exception("Focus region contained too few points")

    p0_cut  = p0     [..., i, :]
    v1_cut  = v1     [     i, :]
    weights = weights[..., i   ]

    def residual_jacobian_rt(rt):

        # rtp0      has shape (M,N,3)
        # drtp0_drt has shape (M,N,3,6)
        rtp0, drtp0_drt, _ = \
            mrcal.transform_point_rt(rt, p0_cut,
                                     get_gradients = True)

        # inner(a,b)/(mag(a)*mag(b)) = cos(x) ~ 1 - x^2/2
        # Each of these has shape (M,N,)
        mag_rtp0 = nps.mag(rtp0)
        inner    = nps.inner(rtp0, v1_cut)
        th2      = 2.* (1.0 - inner / mag_rtp0)
        x        = th2 * weights

        # shape (M,N,6)
        dmag_rtp0_drt = nps.matmult( nps.dummy(rtp0, -2),   # shape (M,N,1,3)
                                     drtp0_drt              # shape (M,N,3,6)
                                     # matmult has shape (M,N,1,6)
                                   )[...,0,:] / \
                                   nps.dummy(mag_rtp0, -1)  # shape (M,N,1)
        # shape (M,N,6)
        dinner_drt    = nps.matmult( nps.dummy(v1_cut, -2), # shape (M,N,1,3)
                                     drtp0_drt              # shape (M,N,3,6)
                                     # matmult has shape (M,N,1,6)
                                   )[...,0,:]

        # dth2 = 2 (inner dmag_rtp0 - dinner mag_rtp0)/ mag_rtp0^2
        # shape (M,N,6)
        J = 2. * \
            (nps.dummy(inner,    -1) * dmag_rtp0_drt - \
             nps.dummy(mag_rtp0, -1) * dinner_drt) / \
             nps.dummy(mag_rtp0*mag_rtp0, -1) * \
             nps.dummy(weights,-1)
        return x.ravel(), nps.clump(J, n=J.ndim-1)


    def residual_jacobian_r(r):

        # rp0     has shape (M,N,3)
        # drp0_dr has shape (M,N,3,3)
        rp0, drp0_dr, _ = \
            mrcal.rotate_point_r(r, p0_cut,
                                 get_gradients = True)

        # inner(a,b)/(mag(a)*mag(b)) ~ cos(x) ~ 1 - x^2/2
        # Each of these has shape (M,N)
        inner = nps.inner(rp0, v1_cut)
        th2   = 2.* (1.0 - inner)
        x     = th2 * weights

        # shape (M,N,3)
        dinner_dr = nps.matmult( nps.dummy(v1_cut, -2), # shape (M,N,1,3)
                                 drp0_dr                # shape (M,N,3,3)
                                 # matmult has shape (M,N,1,3)
                               )[...,0,:]

        J = -2. * dinner_dr * nps.dummy(weights,-1)
        return x.ravel(), nps.clump(J, n=J.ndim-1)


    cache = {'rt': None}
    def residual(rt, f):
        if cache['rt'] is None or not np.array_equal(rt,cache['rt']):
            cache['rt'] = rt
            cache['x'],cache['J'] = f(rt)
        return cache['x']
    def jacobian(rt, f):
        if cache['rt'] is None or not np.array_equal(rt,cache['rt']):
            cache['rt'] = rt
            cache['x'],cache['J'] = f(rt)
        return cache['J']


    # # gradient check
    # import gnuplotlib as gp
    # rt0 = np.random.random(6)*1e-3
    # x0,J0 = residual_jacobian_rt(rt0)
    # drt = np.random.random(6)*1e-7
    # rt1 = rt0+drt
    # x1,J1 = residual_jacobian_rt(rt1)
    # dx_theory = nps.matmult(J0, nps.transpose(drt)).ravel()
    # dx_got    = x1-x0
    # relerr = (dx_theory-dx_got) / ( (np.abs(dx_theory)+np.abs(dx_got))/2. )
    # gp.plot(relerr, wait=1, title='rt')
    # r0 = np.random.random(3)*1e-3
    # x0,J0 = residual_jacobian_r(r0)
    # dr = np.random.random(3)*1e-7
    # r1 = r0+dr
    # x1,J1 = residual_jacobian_r(r1)
    # dx_theory = nps.matmult(J0, nps.transpose(dr)).ravel()
    # dx_got    = x1-x0
    # relerr = (dx_theory-dx_got) / ( (np.abs(dx_theory)+np.abs(dx_got))/2. )
    # gp.plot(relerr, wait=1, title='r')
    # sys.exit()


    # I was using loss='soft_l1', but it behaved strangely. For large
    # f_scale_deg it should be equivalent to loss='linear', but I was seeing
    # large diffs when comparing a model to itself:
    #
    #   ./mrcal-show-projection-diff --gridn 50 28 test/data/cam0.splined.cameramodel{,} --distance 3
    #
    # f_scale_deg needs to be > 0.1 to make test-projection-diff.py pass, so
    # there was an uncomfortably-small usable gap for f_scale_deg. loss='huber'
    # should work similar-ish to 'soft_l1', and it works even for high
    # f_scale_deg
    f_scale_deg = 5
    loss        = 'huber'

    if atinfinity:


        # This is similar to a basic procrustes fit, but here we're using an L1
        # cost function
        r = np.random.random(3) * 1e-5

        res = scipy.optimize.least_squares(residual,
                                           r,
                                           jac=jacobian,
                                           method='dogbox',

                                           loss=loss,
                                           f_scale = (f_scale_deg * np.pi/180.)**2.,
                                           # max_nfev=1,
                                           args=(residual_jacobian_r,),

                                           # Without this, the optimization was
                                           # ending too quickly, and I was
                                           # seeing not-quite-optimal solutions.
                                           # Especially for
                                           # very-nearly-identical rotations.
                                           # This is tested by diffing the same
                                           # model in test-projection-diff.py.
                                           # I'd like to set this to None to
                                           # disable the comparison entirely,
                                           # but that requires scipy >= 1.3.0.
                                           # So instead I set the threshold so
                                           # low that it's effectively disabled
                                           gtol = np.finfo(float).eps,
                                           verbose=0)
        Rt = np.zeros((4,3), dtype=float)
        Rt[:3,:] = mrcal.R_from_r(res.x)
        return Rt

    else:

        rt = np.random.random(6) * 1e-5

        res = scipy.optimize.least_squares(residual,
                                           rt,
                                           jac=jacobian,
                                           method='dogbox',

                                           loss=loss,
                                           f_scale = (f_scale_deg * np.pi/180.)**2.,
                                           # max_nfev=1,
                                           args=(residual_jacobian_rt,),

                                           # Without this, the optimization was
                                           # ending too quickly, and I was
                                           # seeing not-quite-optimal solutions.
                                           # Especially for
                                           # very-nearly-identical rotations.
                                           # This is tested by diffing the same
                                           # model in test-projection-diff.py.
                                           # I'd like to set this to None to
                                           # disable the comparison entirely,
                                           # but that requires scipy >= 1.3.0.
                                           # So instead I set the threshold so
                                           # low that it's effectively disabled
                                           gtol = np.finfo(float).eps )
        return mrcal.Rt_from_rt(res.x)


def worst_direction_stdev(cov):
    r'''Compute the worst-direction standard deviation from a NxN covariance matrix

SYNOPSIS

    # A covariance matrix
    print(cov)
    ===>
    [[ 1.  -0.4]
     [-0.4  0.5]]

    # Sample 1000 0-mean points using this covariance
    x = np.random.multivariate_normal(mean = np.array((0,0)),
                                      cov  = cov,
                                      size = (1000,))

    # Compute the worst-direction standard deviation of the sampled data
    print(np.sqrt(np.max(np.linalg.eig(np.mean(nps.outer(x,x),axis=0))[0])))
    ===>
    1.1102510878087053

    # The predicted worst-direction standard deviation
    print(mrcal.worst_direction_stdev(cov))
    ===> 1.105304960905736

The covariance of a (N,) random vector can be described by a (N,N)
positive-definite symmetric matrix. The 1-sigma contour of this random variable
is described by an ellipse with its axes aligned with the eigenvectors of the
covariance, and the semi-major and semi-minor axis lengths specified as the sqrt
of the corresponding eigenvalues. This function returns the worst-case standard
deviation of the given covariance: the sqrt of the largest eigenvalue.

Given the common case of a 2x2 covariance this function computes the result
directly. Otherwise it uses the numpy functions to compute the biggest
eigenvalue.

This function supports broadcasting fully.

DERIVATION

I solve this directly for the 2x2 case.

Let cov = (a b). If l is an eigenvalue of the covariance then
          (b c)

    (a-l)*(c-l) - b^2 = 0 --> l^2 - (a+c) l + ac-b^2 = 0

    --> l = (a+c +- sqrt( a^2 + 2ac + c^2 - 4ac + 4b^2)) / 2 =
          = (a+c +- sqrt( a^2 - 2ac + c^2 + 4b^2)) / 2 =
          = (a+c)/2 +- sqrt( (a-c)^2/4 + b^2)

So the worst-direction standard deviation is

    sqrt((a+c)/2 + sqrt( (a-c)^2/4 + b^2))

ARGUMENTS

- cov: the covariance matrices given as a (..., 2,2) array. Valid covariances
  are positive-semi-definite (symmetric with eigenvalues >= 0), but this is not
  checked

RETURNED VALUES

The worst-direction standard deviation. This is a scalar or an array, if we're
broadcasting

    '''

    cov = nps.atleast_dims(cov,-2)

    if cov.shape[-2:] == (1,1):
        return np.sqrt(cov[...,0,0])

    if cov.shape[-2:] == (2,2):
        a = cov[..., 0,0]
        b = cov[..., 1,0]
        c = cov[..., 1,1]
        return np.sqrt((a+c)/2 + np.sqrt( (a-c)*(a-c)/4 + b*b))

    if cov.shape[-1] != cov.shape[-2]:
        raise Exception(f"covariance matrices must be square. Got cov.shape = {cov.shape}")

    import scipy.sparse.linalg
    @nps.broadcast_define( (('N','N',),),
                           () )
    def largest_eigenvalue(V):
        return \
            scipy.sparse.linalg.eigsh(V, 1,
                                      which               = 'LM',
                                      return_eigenvectors = False)[0]
    return np.sqrt(largest_eigenvalue(cov))


def _observed_pixel_uncertainty_from_inputs(optimization_inputs,
                                            x = None):

    if x is None:
        x = mrcal.optimizer_callback(**optimization_inputs,
                                     no_jacobian      = True,
                                     no_factorization = True)[1]

    sum_of_squares_measurements = 0
    Nobservations               = 0

    # shape (Nobservations*2)
    measurements = mrcal.measurements_board(optimization_inputs, x = x).ravel()
    if measurements.size:
        sum_of_squares_measurements += np.var(measurements) * measurements.size
        Nobservations += measurements.size

    measurements = mrcal.measurements_point(optimization_inputs, x = x).ravel()
    if measurements.size:
        sum_of_squares_measurements += np.var(measurements) * measurements.size
        Nobservations += measurements.size

    if Nobservations == 0:
        raise Exception("observed_pixel_uncertainty cannot be computed because we don't have any board or point observations")
    observed_pixel_uncertainty = np.sqrt(sum_of_squares_measurements / Nobservations)

    return observed_pixel_uncertainty



def _covariance_processed(what, Var_dF, observed_pixel_uncertainty,
                          *,
                          scalar):
    if what == 'covariance':
        if scalar:
            Var_dF = Var_dF[0,0]
        return Var_dF * observed_pixel_uncertainty*observed_pixel_uncertainty
    if what == 'worstdirection-stdev':
        return worst_direction_stdev(Var_dF) * observed_pixel_uncertainty
    if what == 'rms-stdev':
        # Compute the RMS of the standard deviations in each direction
        # RMS(stdev) =
        # = sqrt( mean(stdev^2) )
        # = sqrt( mean(var) )
        # = sqrt( sum(var)/N )
        # = sqrt( trace/N )
        return np.sqrt(nps.trace(Var_dF)/Var_dF.shape[-1]) * observed_pixel_uncertainty
    else:
        raise Exception("Shouldn't have gotten here. There's a bug")

def _propagate_calibration_uncertainty( what,
                                        *,

                                        # One of these must be given. If it's
                                        # dF_dbunpacked, then
                                        # optimization_inputs must be given too
                                        dF_dbpacked                        = None,
                                        dF_dbunpacked                      = None,
                                        # These are partly optional. I need
                                        # everything except optimization_inputs.
                                        # If any of the non-optimization_inputs
                                        # arguments are missing I need
                                        # optimization_inputs to compute them.
                                        x                                  = None,
                                        factorization                      = None,
                                        Jpacked                            = None,
                                        Nmeasurements_observations_leading = 0,
                                        observed_pixel_uncertainty         = None,
                                        # can compute each of the above
                                        optimization_inputs                = None):
    r'''Helper for uncertainty propagation functions

Propagates the calibration-time uncertainty to compute Var(F) for some arbitrary
vector F. The user specifies the gradient dF/db: the sensitivity of F to noise
in the calibration state. The vector F can have any length: this is inferred
from the dimensions of the given dF/db gradient.

The given factorization uses the packed, unitless state: b*.

The given Jpacked uses the packed, unitless state: b*. Jpacked applies to all
observations. The leading Nmeasurements_observations_leading rows apply to the
observations of the calibration object, and we use just those for the input
noise propagation. if Nmeasurements_observations_leading==0: assume that ALL the
measurements come from the calibration object observations; a simplifed
expression can be used in this case. This is the default: this simplified
expression (ignoring the regularization; see below) evaluates much faster

The given dF_dbpacked uses the packed, unitless state b*, so it already includes
the multiplication by D in the expressions below. It's usually sparse, but
stored densely.

The uncertainty computation in
https://mrcal.secretsauce.net/uncertainty.html concludes that

  Var(b) = observed_pixel_uncertainty^2 inv(JtJ) J[observations]t J[observations] inv(JtJ)

I actually operate with b* and J*: the UNITLESS state and the jacobian
respectively. I have

  b = D b*
  J = J* inv(D)

so

  Var(b*) = observed_pixel_uncertainty^2 inv(J*tJ*) J*[observations]t J*[observations] inv(J*tJ*)

In the special case where all the measurements come from observations, this
simplifies to

  Var(b*) = observed_pixel_uncertainty^2 inv(J*tJ*)

My factorization is of packed (scaled, unitless) flavors of J (J*). So

  Var(b) = D Var(b*) D

I want Var(F) = dF/db Var(b) dF/dbt

So

  Var(F) = dF/db D Var(b*) D dF/dbt

In the regularized case I have

  Var(F) = dF/db D inv(J*tJ*) J*[observations]t J*[observations] inv(J*tJ*) D dF/dbt observed_pixel_uncertainty^2

It is far more efficient to compute inv(J*tJ*) D dF/dbt than
inv(J*tJ*) J*[observations]t: there's far less to compute, and the matrices
are far smaller. Thus I don't compute the covariances directly.

In the non-regularized case:

  Var(F) = dF/db D inv(J*tJ*) D dF/dbt

  1. solve( J*tJ*, D dF/dbt)
     The result has shape (Nstate,len(F))

  2. pre-multiply by dF/db D

  3. multiply by observed_pixel_uncertainty^2

In the regularized case:

  Var(F) = dF/db D inv(J*tJ*) J*[observations]t J*[observations] inv(J*tJ*) D dF/dbt

  1. solve( J*tJ*, D dF/dbt)
     The result has shape (Nstate,len(F))

  2. Pre-multiply by J*[observations]
     The result has shape (Nmeasurements_observations_leading,2)

  3. Compute the sum of the outer products of each row

  4. multiply by observed_pixel_uncertainty^2

    '''

    what_known = set(('covariance', 'worstdirection-stdev', 'rms-stdev', '_covariance-raw'))
    if not what in what_known:
        raise Exception(f"'what' kwarg must be in {what_known}, but got '{what}'")

    if dF_dbpacked   is None and \
       dF_dbunpacked is None:
        raise Exception("Exactly one of dF_dbpacked,dF_dbunpacked must be given")
    if dF_dbpacked   is not None and \
       dF_dbunpacked is not None:
        raise Exception("Exactly one of dF_dbpacked,dF_dbunpacked must be given")

    if dF_dbunpacked is not None:
        if optimization_inputs is None:
            raise Exception('dF_dbunpacked is given but optimization_inputs is not. Either pass dF_dbpacked or pass optimization_inputs in as well')

        # Make dF_db use the packed state. I call "unpack_state" because the
        # state is in the denominator
        dF_dbpacked = np.array(dF_dbunpacked) # make a copy
        mrcal.unpack_state(dF_dbpacked, **optimization_inputs)

    if \
       x                                  is None or \
       factorization                      is None or \
       Jpacked                            is None or \
       Nmeasurements_observations_leading is None or \
       observed_pixel_uncertainty         is None:
        if optimization_inputs is None:
            raise Exception("At least one of (factorization,Jpacked,Nmeasurements_observations_leading,observed_pixel_uncertainty) are None, so optimization_inputs MUST have been given to compute them")

    if factorization is None or Jpacked is None or x is None:
        _,x,Jpacked,factorization = mrcal.optimizer_callback(**optimization_inputs)
        if factorization is None:
            raise Exception("Cannot compute the uncertainty: factorization computation failed")


    if Nmeasurements_observations_leading is None:
        Nmeasurements_boards         = mrcal.num_measurements_boards(**optimization_inputs)
        Nmeasurements_points         = mrcal.num_measurements_points(**optimization_inputs)
        Nmeasurements_regularization = mrcal.num_measurements_regularization(**optimization_inputs)
        Nmeasurements_all            = mrcal.num_measurements(**optimization_inputs)
        imeas_regularization         = mrcal.measurement_index_regularization(**optimization_inputs)
        if Nmeasurements_boards + \
           Nmeasurements_points + \
           Nmeasurements_regularization != \
               Nmeasurements_all:
            raise Exception("Some measurements other than boards, points and regularization are present. Don't know what to do")
        if imeas_regularization is not None and \
           imeas_regularization + Nmeasurements_regularization != Nmeasurements_all:
            raise Exception("Regularization measurements are NOT at the end. Don't know what to do")

        if Nmeasurements_regularization == 0:
            # Note the special-case where I'm using all the observations. No other
            # measurements are present other than the chessboard observations
            Nmeasurements_observations_leading = 0
        else:
            Nmeasurements_observations_leading = \
                Nmeasurements_all - Nmeasurements_regularization
            if Nmeasurements_observations_leading == 0:
                raise Exception("No non-regularization measurements. Don't know what to do")

    if observed_pixel_uncertainty is None:
        observed_pixel_uncertainty = _observed_pixel_uncertainty_from_inputs(optimization_inputs,
                                                                             x = x)

    scalar = (dF_dbpacked.ndim == 1)

    if Nmeasurements_observations_leading > 0:
        # I have regularization. Use the more complicated expression

        # I can probably adapt this path to use the faster solve_xt_JtJ_bt()
        # expressions below, but I don't bother. I will be using the fast path
        # 99% of the time: ignoring regularization
        # (Nmeasurements_observations_leading=0)

        # shape (N,Nstate) where N=2 usually
        A = factorization.solve_xt_JtJ_bt( dF_dbpacked )

        # I see no python way to do matrix multiplication with sparse matrices,
        # so I have my own routine in C. AND the C routine does the outer
        # product, so there's no big temporary expression. It's much faster
        if len(A.shape) == 2 and A.shape[0] == 2:
            f = mrcal._mrcal_npsp._A_Jt_J_At__2
        else:
            f = mrcal._mrcal_npsp._A_Jt_J_At
        Var_dF = f(A, Jpacked.indptr, Jpacked.indices, Jpacked.data,
                   Nleading_rows_J = Nmeasurements_observations_leading)
    else:
        # No regularization. Use the simplified expression

        # The expression I had earlier. Works properly, but is slow:
        # # time ./mrcal-show-projection-uncertainty --gridn 120 90 --hardcopy /tmp/tst.gp **/*.cameramodel(OL[1])
        # # 21.48s user 0.27s system 99% cpu 21.750 total
        #
        # # shape (N,Nstate) where N=2 usually
        # A = factorization.solve_xt_JtJ_bt( dF_dbpacked )
        # Var_dF = nps.matmult(dF_dbpacked, nps.transpose(A))
        #
        # Instead, I do something smarter. I need to compute
        #
        #   sigma^2 dF/db D inv(J*tJ*) D dF/dbt
        #
        # I just computed a factorization, so I have
        #
        #   J*tJ* = L Lt
        #
        # so I can equivalently compute
        #
        #   sigma^2 norm2( inv(L) D dF/dbt )
        #
        # when cholmod_solve2() tries to solve inv(JtJ) x = b, it solves two
        # linear systems in series: one with L and then again with Lt. This new
        # expression runs only one solve, which speeds things up dramatically.

        # I tried several ways to compute this. All produce the same result, but
        # have different speeds. By default cholmod gives me an LDLt
        # factorization, not an LLt factorization (this is a different D), so I
        # need to handle this extra D in some way.

        # Slowest; two different solves.
        # time ./mrcal-show-projection-uncertainty --gridn 120 90 --hardcopy /tmp/tst.gp **/*.cameramodel(OL[1])
        # 20.64s user 1.30s system 99% cpu 21.938 total
        #
        # A1 = factorization.solve_xt_JtJ_bt( dF_dbpacked, sys='P' )
        # A2 = factorization.solve_xt_JtJ_bt( A1,          sys='L' )
        # A3 = factorization.solve_xt_JtJ_bt( A1,          sys='LD' )
        # Var_dF = nps.matmult(A2, nps.transpose(A3))
        #

        # Fastest, but works only if I have an LLt factorization. Can be
        # requested with this patch:
        #
        #   diff --git a/mrcal-pywrap.c b/mrcal-pywrap.c
        #   index b0d45fcc..b7c87090 100644
        #   --- a/mrcal-pywrap.c
        #   +++ b/mrcal-pywrap.c
        #   @@ -290,3 +290,5 @@
        #            self->common.supernodal = 0;
        #
        #   +        // self->common.final_ll = 1;
        #   +
        #            // I want all output to go to STDERR, not STDOUT
        #
        # I don't know if there are downsides to this, so I don't do this.
        #
        # time ./mrcal-show-projection-uncertainty --gridn 120 90 --hardcopy /tmp/tst.gp **/*.cameramodel(OL[1])
        # 11.80s user 0.38s system 99% cpu 12.187 total
        #
        # A1 = factorization.solve_xt_JtJ_bt( dF_dbpacked, sys='P' )
        # A2 = factorization.solve_xt_JtJ_bt( A1,          sys='L' )
        # Var_dF = nps.matmult(A2, nps.transpose(A2))

        # A bit slower, uses more memory, but works with LDLt. This is what I
        # use
        #
        # time ./mrcal-show-projection-uncertainty --gridn 120 90 --hardcopy /tmp/tst.gp **/*.cameramodel(OL[1])
        # 12.29s user 0.96s system 99% cpu 13.253 total
        A1 = factorization.solve_xt_JtJ_bt( dF_dbpacked, sys='P' )
        del dF_dbpacked
        A2 = factorization.solve_xt_JtJ_bt( A1,          sys='L' )
        del A1
        A3 = factorization.solve_xt_JtJ_bt( A2,          sys='D' )
        Var_dF = nps.matmult(A2, nps.transpose(A3))

    if what == '_covariance-raw':
        return Var_dF,observed_pixel_uncertainty

    return _covariance_processed(what, Var_dF,observed_pixel_uncertainty,
                                 scalar = scalar)


def _propagate_calibration_uncertainty_bestq( dq_db,
                                              observed_pixel_uncertainty,
                                              optimization_inputs):
    r'''Simplified flavor of _propagate_calibration_uncertainty()

Same logic as that function, but with some specific arguments assumed. Used for
the bestq uncertainty method. More memory-efficient for that application.

THIS FUNCTION OVERWRITES dq_db

    '''

    # OVERWRITE dq_db. It now has packed state
    mrcal.unpack_state(dq_db, **optimization_inputs)

    _,x,_,factorization = mrcal.optimizer_callback(**optimization_inputs)
    if factorization is None:
        raise Exception("Cannot compute the uncertainty: factorization computation failed")

    if observed_pixel_uncertainty is None:
        observed_pixel_uncertainty = _observed_pixel_uncertainty_from_inputs(optimization_inputs,
                                                                             x = x)

    @nps.broadcast_define( (('Ngeometry',2,'Nstate'), ),
                           (2,2) )
    def process_slice(dq_db):
        # Here "dq_db" is actually "dq_dbpacked"
        A1 = factorization.solve_xt_JtJ_bt( dq_db, sys='P' )
        A2 = factorization.solve_xt_JtJ_bt( A1,    sys='L' )
        del A1
        A3 = factorization.solve_xt_JtJ_bt( A2,    sys='D' )

        # shape (Ngeometry,2,2)
        Var_dq = nps.matmult(A2, nps.transpose(A3))
        del A2
        del A3
        # scalar
        i = np.argmin( np.trace(Var_dq, axis1=-1, axis2=-2) )
        return Var_dq[i]

    # shape (...,2,2)
    Var_dq = process_slice(dq_db)

    return Var_dq,observed_pixel_uncertainty


def _dq_db__Kunpacked_rrp(## write output here
                          # shape (..., 2,Nstate)
                          dq_db,
                          ## inputs
                          # shape (..., Ncameras_extrinsics, 3)
                          p_ref,
                          # shape (..., 2,3)
                          dq_dpcam,
                          # shape (..., Ncameras_extrinsics,3,3)
                          dpcam_dpref,
                          # shape (6,Nstate)
                          Kunpacked_rrp,
                          *,
                          atinfinity):

    # atinfinity = rotation-only
    # K is drt_ref_refperturbed/db


    # From https://mrcal.secretsauce.net/docs-3.0/uncertainty-cross-reprojection.html:
    #   q*    = project( pcam*, intrinsics* )
    #   pcam* = Tc*r* pref*
    #   pref* = Tr*r  pref
    #
    # So q* depends on Tr*r, Tc*,r*, intrinsics*. The extrinsics and intrinsics
    # dependence was handled by the caller. Here I ingest only the dependence on
    # Tr*r
    Ncameras_extrinsics = p_ref.shape[-2]
    if Ncameras_extrinsics != 1:
        raise Exception("I only handle stationary cameras for now")
    # shape (..., 3)
    p_ref = p_ref[...,0,:]
    # shape (..., 3,3)
    dpcam_dpref = dpcam_dpref[...,0,:,:]

    # So
    #   dq/db[frame_all,calobject_warp] =
    #     dq_dpcam dpcam__dpref dpref*__drt_ref_ref* Kunpacked_rrp
    if dpcam_dpref is not None:
        dq_dpref = nps.matmult(dq_dpcam, dpcam_dpref)
    else:
        dq_dpref = dq_dpcam

    # pref = transform(rt_ref_ref*, pref*)
    #   where rt_ref_ref* is tiny
    # -> pref ~ pref* + cross(r, pref) + t
    #         = pref* - cross(pref, r) + t
    # -> 0 = dpref*/dr - skew(pref)
    #    0 = dpref*/dt + I
    # -> dpref*/dr = skew(pref)
    #    dpref*/dt = -I
    dprefp__dr_ref_refp = mrcal.skew_symmetric(p_ref)

    # I don't explicitly store dpref*/dt. I multiply by I implicitly

    dq__dr_ref_refp = nps.matmult(dq_dpref, dprefp__dr_ref_refp)

    # I apply this to the whole dq_db array. Kunpacked_rrp has 0 rows for
    # the unaffected state, so the columns that should be untouched will
    # be untouched
    dq_db += nps.matmult(dq__dr_ref_refp, Kunpacked_rrp[:3,:])

    if not atinfinity:
        dq_db -= nps.matmult(dq_dpref,        Kunpacked_rrp[3:,:])


def _dq_db__projection_uncertainty( # shape (...,3)
                                    p_cam,
                                    lensmodel, intrinsics_data,
                                    # all the camera extrinsics where this
                                    # specific camera observed the board. This
                                    # may be all or none of the extrinsics in
                                    # the state, or anything in-between
                                    extrinsics_rt_fromref,
                                    # all frame poses from the optimizaiton
                                    frames_rt_toref,
                                    Nstate,
                                    # in the state vector
                                    istate_intrinsics0,
                                    # in the camera; generally: 0 if do_optimize_intrinsics_core else 0
                                    istate_intrinsics0_onecam,
                                    Nstates_intrinsics,
                                    istate_extrinsics0, istate_frames0,
                                    *,
                                    # shape (6,Nstate)
                                    Kunpacked_rrp, # used iff method ~ "cross-reprojection..."
                                    method,
                                    atinfinity):
    r'''Helper for projection_uncertainty()

    See docs for _propagate_calibration_uncertainty() and
    projection_uncertainty()

    This function does all the work when observing points with a finite range

    The underlying math is documented here:

    - https://mrcal.secretsauce.net/docs-3.0/uncertainty-mean-pcam.html
    - https://mrcal.secretsauce.net/docs-3.0/uncertainty-cross-reprojection.html

    The end result for method == "mean-pcam":

      q* - q

      ~   dq_dpcam (pcam* - pcam)
        + dq_dintrinsics db[intrinsics_this]

      ~   dq_dpcam/Ncam_frame sum(dpcam__drt_camj_ref db[extrinsics_j] +
                                  dpcam__dpref_i (pref_i* - pref_i) )
        + dq_dintrinsics db[intrinsics_this]

      ~   dq_dpcam/Ncam_frame sum(dpcam__drt_camj_ref db[extrinsics_j] +
                                  dpcam__dpref_i ( dpref__drt_ref_framei db[frame_i] +
                                                   dpref__dpframe_i d(pframe_i) )
        + dq_dintrinsics db[intrinsics_this]

      Here I'm assuming fixed pframe, so d(pframe_) = 0:

      dq
      ~   dq_dpcam/Ncam_frame sum(dpcam__drt_camj_ref db[extrinsics_j] +
                                  dpcam__dpref_i dpref__drt_ref_framei db[frame_i] )
        + dq_dintrinsics db[intrinsics_this]

      --->

      dq/db[extrinsics_j]    = dq_dpcam/Ncam_frame sum(dpcam__drt_camj_ref)
      dq/db[frames_i]        = dq_dpcam/Ncam_frame sum(dpcam__dpref_i dpref__drt_ref_framei )
      dq/db[intrinsics_this] = dq_dintrinsics

    The end result for method == "bestq":

      For bestq I report a separate result for each camera/board geometry. So
      the expressions are the same as for pcam, but there's no /Ncam_frame and
      no sum(): I instead report an array for each slice

    '''

    # extrinsics_rt_fromref and frames_rt_toref contain poses. These are
    # available here, whether they're being optimized or not. istate_... are the
    # state variables. These may be None if the quantity in question is fixed.

    # shape (Ncameras_extrinsics,6) where Ncameras_extrinsics may be 1
    extrinsics_rt_fromref = nps.atleast_dims(extrinsics_rt_fromref, -2)
    # shape (Nframes,6)             where Nframes may be 1
    frames_rt_toref       = nps.atleast_dims(frames_rt_toref,       -2)

    Ncameras_extrinsics = extrinsics_rt_fromref.shape[0]
    Nframes             = frames_rt_toref      .shape[0]

    ### The output array. This function fills this in, and returns it
    # shape (..., 2,Nstate)
    if method != 'bestq':
        dq_db = np.zeros(p_cam.shape[:-1] + (2,Nstate), dtype=float)
    else:
        if Ncameras_extrinsics > 1 and Nframes > 1:
            raise Exception("method=='bestq' works only if either the camera or board are stationary")
        if Ncameras_extrinsics == 1 and Nframes == 1:
            raise Exception("method=='bestq' works only if either the camera or board are moving")
        Ngeometry = max(Ncameras_extrinsics,Nframes)
        dq_db = np.zeros(p_cam.shape[:-1] + (Ngeometry, 2,Nstate), dtype=float)


    # shape (..., Ncameras_extrinsics, 3)
    if not atinfinity:
        p_ref = \
            mrcal.transform_point_rt( mrcal.invert_rt(extrinsics_rt_fromref),
                                      nps.dummy(p_cam,-2) )
    else:
        p_ref = \
            mrcal.rotate_point_r( -extrinsics_rt_fromref[...,:3],
                                  nps.dummy(p_cam,-2) )

    # shape (..., 2,3)
    # shape (..., 2,Nintrinsics)
    _, dq_dpcam, dq_dintrinsics = \
        mrcal.project( p_cam, lensmodel, intrinsics_data,
                       get_gradients = True)

    if istate_intrinsics0 is not None:
        if method != 'bestq':
            dq_db[         ...,
                           istate_intrinsics0:
                           istate_intrinsics0+Nstates_intrinsics] = \
            dq_dintrinsics[...,
                           istate_intrinsics0_onecam:
                           istate_intrinsics0_onecam+Nstates_intrinsics]
        else:
            dq_db[         ...,
                           istate_intrinsics0:
                           istate_intrinsics0+Nstates_intrinsics] += \
            nps.dummy(dq_dintrinsics[...,
                                     istate_intrinsics0_onecam:
                                     istate_intrinsics0_onecam+Nstates_intrinsics],
                      -3)

    if not atinfinity:
        # shape (..., Ncameras_extrinsics,3,6)
        # shape (..., Ncameras_extrinsics,3,3)
        _, dpcam_drt, dpcam_dpref = \
            mrcal.transform_point_rt(extrinsics_rt_fromref, p_ref,
                                     get_gradients = True)
    else:
        # shape (..., Ncameras_extrinsics,3,3)
        # shape (..., Ncameras_extrinsics,3,3)
        _, dpcam_dr, dpcam_dpref = \
            mrcal.rotate_point_r(extrinsics_rt_fromref[...,:3], p_ref,
                                 get_gradients = True)

    if istate_extrinsics0 is not None:
        # I want dq/db[extrinsics_j] = dq_dpcam/Ncam_frame sum(dpcam__drt_camj_ref)
        #
        # Here I'm computing the mean over all camera,frame combinations. The
        # quantity is constant for each frame, so I use the mean over the
        # extrinsics. The gradients are distributed across the state vector, but
        # the mean comes through as the /Ncameras_extrinsics

        if method != 'bestq':

            # shape (..., 2, Ncameras_extrinsics,6)
            dq_db_slice_extrinsics = \
                dq_db[...,
                      istate_extrinsics0:
                      istate_extrinsics0 + Ncameras_extrinsics*6]. \
                      reshape(dq_db.shape[:-1] + (Ncameras_extrinsics,6) )

            if not atinfinity:
                # shape (..., 2, Ncameras_extrinsics,6)
                dq_db_slice_extrinsics[...] = \
                    nps.xchg( nps.matmult(# shape (..., Ncameras_extrinsics=1,2,3)
                                          nps.dummy(dq_dpcam,-3),
                                          # shape (..., Ncameras_extrinsics,  3,6)
                                          dpcam_drt),
                              -2, -3 ) / Ncameras_extrinsics
            else:
                # shape (..., 2, Ncameras_extrinsics,3)
                dq_db_slice_extrinsics[...,:3] = \
                    nps.xchg( nps.matmult(# shape (..., Ncameras_extrinsics=1,2,3)
                                          nps.dummy(dq_dpcam,-3),
                                          # shape (..., Ncameras_extrinsics,  3,3)
                                          dpcam_dr),
                              -2, -3 ) / Ncameras_extrinsics
        else:

            # bestq
            if Ncameras_extrinsics == 1:
                # The camera is stationary; the extra dimension is for the
                # moving board. So the camera pose applies to each slice.

                # shape (..., 2,6)
                dq_db_slice_extrinsics = \
                    dq_db[...,
                          istate_extrinsics0:
                          istate_extrinsics0 + 6]

                if not atinfinity:
                    # shape (..., 2,6)
                    dq_db_slice_extrinsics[...] = \
                        nps.matmult(# shape (..., 2,3)
                            dq_dpcam,
                            # shape (..., 3,6)
                            dpcam_drt[...,0,:,:])
                else:
                    # shape (..., 2,3)
                    dq_db_slice_extrinsics[...,:3] = \
                        nps.matmult(# shape (..., 2,3)
                            dq_dpcam,
                            # shape (..., 3,3)
                            dpcam_dr[...,0,:,:])
            else:
                # Moving cameras. Each camera pose gets its own slice in
                # dimension -3 of dq_db. So I don't have a nice rectangular
                # slice, and I need to loop
                for icamera_extrinsics in range(Ncameras_extrinsics):

                    # shape (..., 2,6)
                    dq_db_slice_extrinsics = \
                        dq_db[...,
                              icamera_extrinsics,
                              :,
                              istate_extrinsics0 + icamera_extrinsics*6:
                              istate_extrinsics0 + icamera_extrinsics*6 + 6]

                    if not atinfinity:
                        # shape (..., 2,6)
                        dq_db_slice_extrinsics[...] = \
                            nps.matmult(# shape (..., 2,3)
                                dq_dpcam,
                                # shape (..., 3,6)
                                dpcam_drt[...,icamera_extrinsics,:,:])
                    else:
                        # shape (..., 2,3)
                        dq_db_slice_extrinsics[...,:3] = \
                            nps.matmult(# shape (..., 2,3)
                                dq_dpcam,
                                # shape (..., 3,3)
                                dpcam_dr[...,icamera_extrinsics,:,:])


    if method == 'mean-pcam' or method == 'bestq':
        if istate_frames0 is not None:

            # shape (..., Ncameras_extrinsics, 2, 3)
            dq_dpref = \
                nps.matmult(# shape (...,          Ncameras_extrinsics=1, 2, 3)
                            nps.dummy(dq_dpcam,    -3),
                            # shape (...,          Ncameras_extrinsics,   3, 3)
                            dpcam_dpref)

            if not atinfinity:
                # shape (Nframes,Ncameras_extrinsics,3)
                p_frames = mrcal.transform_point_rt( # shape (Nframes,Ncameras_extrinsics=1,6)
                                                     nps.dummy(mrcal.invert_rt(frames_rt_toref), -2),
                                                     # shape (..., Nframes=1, Ncameras_extrinsics, 3)
                                                     nps.dummy(p_ref,-3) )

                # shape (...,Nframes,Ncameras_extrinsics,3,6)
                _, \
                dpref_dframes, \
                _ = mrcal.transform_point_rt( # shape (Nframes,Ncameras_extrinsics=1,6)
                                              nps.dummy(frames_rt_toref, -2),
                                              p_frames,
                                              get_gradients = True)

            else:

                # shape (Nframes,Ncameras_extrinsics,3)
                p_frames = mrcal.rotate_point_r( # shape (Nframes,Ncameras_extrinsics=1,6)
                                                 nps.dummy(-frames_rt_toref[...,:3], -2),
                                                 # shape (..., Nframes=1, Ncameras_extrinsics, 3)
                                                 nps.dummy(p_ref,-3) )

                # shape (...,Nframes,Ncameras_extrinsics,3,6)
                _, \
                dpref_dframes, \
                _ = mrcal.rotate_point_r( # shape (Nframes,Ncameras_extrinsics=1,3)
                                          nps.dummy(frames_rt_toref[...,:3], -2),
                                          p_frames,
                                          get_gradients = True)

            # shape is either of
            #    (..., Nframes, Ncameras_extrinsics, 2, 6)
            #    (..., Nframes, Ncameras_extrinsics, 2, 3)
            # depending on atinfinity
            dq_dframes = \
                nps.matmult(# shape (...,          Nframes=1, Ncameras_extrinsics, 2, 3)
                            nps.dummy(dq_dpref,    -4),
                            # shape (...,          Nframes,   Ncameras_extrinsics, 3, 6)
                            dpref_dframes)

            if method != 'bestq':

                # shape (..., 2, Nframes,6)
                dq_db_slice_frames = \
                    dq_db[...,
                          istate_frames0:
                          istate_frames0 + Nframes*6]. \
                          reshape(dq_db.shape[:-1] + (Nframes,6) )
                if not atinfinity:
                    # shape (..., 2, Nframes,6)
                    dq_db_slice_frames[...] = \
                        nps.xchg( np.mean(dq_dframes, axis=-3),
                                  -2, -3 ) / Nframes
                else:
                    # shape (..., 2, Nframes,3)
                    dq_db_slice_frames[...,:3] = \
                        nps.xchg( np.mean(dq_dframes, axis=-3),
                                  -2, -3 ) / Nframes
            else:
                # bestq
                if Nframes == 1:
                    # The board is stationary; the extra dimension is for the
                    # moving camera. So the board pose applies to each slice.

                    # shape (..., 2,6)
                    dq_db_slice_frames = \
                        dq_db[...,
                              istate_frames0:
                              istate_frames0 + 6]
                    if not atinfinity:
                        # shape (..., 2,6)
                        dq_db_slice_frames[...] = \
                            dq_dframes[...,0,:,:,:]
                    else:
                        # shape (..., 2,3)
                        dq_db_slice_frames[...,:3] = \
                            dq_dframes[...,0,:,:,:]
                else:
                    # Moving board. Each board pose gets its own slice in
                    # dimension -3 of dq_db. So I don't have a nice rectangular
                    # slice, and I need to loop
                    for iframe in range(Nframes):

                        # shape (..., 2,6)
                        dq_db_slice_frames = \
                            dq_db[...,
                                  iframe,
                                  :,
                                  istate_frames0 + iframe*6:
                                  istate_frames0 + iframe*6 + 6]
                        if not atinfinity:
                            # shape (..., 2,6)
                            dq_db_slice_frames[...] = \
                                dq_dframes[...,iframe,0,:,:]
                        else:
                            # shape (..., 2,3)
                            dq_db_slice_frames[...,:3] = \
                                dq_dframes[...,iframe,0,:,:]

    elif method == 'cross-reprojection-rrp-Jfp':
        _dq_db__Kunpacked_rrp(## write output here
                              # shape (..., 2,Nstate)
                              dq_db,
                              ## inputs
                              # shape (..., Ncameras_extrinsics, 3)
                              p_ref,
                              # shape (..., 2,3)
                              dq_dpcam,
                              # shape (..., Ncameras_extrinsics,3,3)
                              dpcam_dpref,
                              # shape (6,Nstate)
                              Kunpacked_rrp,
                              atinfinity = atinfinity)
    else:
        raise Exception(f"Unknown {method=}")


    return dq_db



def projection_uncertainty( p_cam, model,
                            *,
                            method     = 'mean-pcam',
                            atinfinity = False,

                            # what we're reporting
                            what = 'covariance',
                            observed_pixel_uncertainty = None):
    r'''Compute the projection uncertainty of a camera-referenced point

This is the interface to the uncertainty computations described in
https://mrcal.secretsauce.net/uncertainty.html

SYNOPSIS

    model = mrcal.cameramodel("xxx.cameramodel")

    q        = np.array((123., 443.))
    distance = 10.0

    pcam = distance * mrcal.unproject(q, *model.intrinsics(), normalize=True)

    print(mrcal.projection_uncertainty(pcam,
                                       model = model,
                                       what  = 'worstdirection-stdev'))
    ===> 0.5

    # So if we have observed a world point at pixel coordinates q, and we know
    # it's 10m out, then we know that the standard deviation of the noise of the
    # pixel observation is 0.5 pixels, in the worst direction

After a camera model is computed via a calibration process, the model is
ultimately used in projection/unprojection operations to map between 3D
camera-referenced coordinates and projected pixel coordinates. We never know the
parameters of the model perfectly, and it is VERY useful to have the resulting
uncertainty of projection. This can be used, among other things, to

- propagate the projection noise down to whatever is using the observed pixels
  to do stuff

- evaluate the quality of calibrations, to know whether a given calibration
  should be accepted, or rejected

- evaluate the stability of a computed model

I quantify uncertainty by propagating the expected pixel observation noise
through the optimization problem. This gives us the uncertainty of the solved
optimization parameters. And then we propagate this parameter noise through
projection to produce the projected pixel uncertainty.

As noted in the docs (https://mrcal.secretsauce.net/uncertainty.html), this
measures the SAMPLING error, which is a direct function of the quality of the
gathered calibration data. It does NOT measure model errors, which arise from
inappropriate lens models, for instance.

The uncertainties can be visualized with the mrcal-show-projection-uncertainty
tool.

ARGUMENTS

This function accepts an array of camera-referenced points p_cam, a model and a
few meta-parameters that describe details of the behavior. This function
broadcasts on p_cam only. We accept

- p_cam: a numpy array of shape (..., 3). This is the set of camera-coordinate
  points where we're querying uncertainty. if not atinfinity: then the full 3D
  coordinates of p_cam are significant, even distance to the camera. if
  atinfinity: the distance to the camera is ignored.

- model: a mrcal.cameramodel object that contains optimization_inputs, which are
  used to propagate the uncertainty

- atinfinity: optional boolean, defaults to False. If True, we want to know the
  projection uncertainty, looking at a point infinitely-far away. We propagate
  all the uncertainties, ignoring the translation components of the poses

- what: optional string, defaults to 'covariance'. This chooses what kind of
  output we want. Known options are:

  - 'covariance':           return a full (2,2) covariance matrix Var(q) for
                            each p_cam
  - 'worstdirection-stdev': return the worst-direction standard deviation for
                            each p_cam

  - 'rms-stdev':            return the RMS of the standard deviations in each
                            direction

- observed_pixel_uncertainty: optional value, defaulting to None. The
  uncertainty of the pixel observations being propagated through the solve and
  through projection. If omitted or None, this input uncertainty is inferred
  from the residuals at the optimum. Most people should omit this

RETURN VALUE

A numpy array of uncertainties. If p_cam has shape (..., 3) then:

if what == 'covariance': we return an array of shape (..., 2,2)
else:                    we return an array of shape (...)

    '''

    # The math implemented here is documented in
    #
    #   https://mrcal.secretsauce.net/uncertainty.html

    # Non-None if this exists, isn't None, and has non-zero elements
    def get_input(what):
        x = optimization_inputs.get(what)
        if x is not None and x.size > 0: return x
        else:                            return None



    known_methods = set(('mean-pcam',
                         'bestq',
                         'cross-reprojection-rrp-Jfp'),)
    if method not in known_methods:
        raise Exception(f"Unknown uncertainty method: '{method}'. I know about {known_methods}")


    # which calibration-time camera we're looking at
    icam_intrinsics = model.icam_intrinsics()

    optimization_inputs = model.optimization_inputs()
    if optimization_inputs is None:
        raise Exception("optimization_inputs are unavailable in this model. Uncertainty cannot be computed")

    # Stuff may or may not be optimized: I get the geometry arrays regardless.
    # The istate_... variables are None if the particular quantity isn't up for
    # optimization (it is fixed)
    frames_rt_toref = get_input('frames_rt_toref')
    istate_frames0  = mrcal.state_index_frames(0, **optimization_inputs)

    if method == 'cross-reprojection-rrp-Jfp':
        Kunpacked_rrp = mrcal.drt_ref_refperturbed__dbpacked(**optimization_inputs)
        # The value was packed in the denominator. So I call pack() to unpack it
        mrcal.pack_state(Kunpacked_rrp, **optimization_inputs)
    else:
        Kunpacked_rrp = None
        if get_input('observations_point')              is not None or \
           get_input('observations_point_triangulated') is not None:
            raise Exception("We have point observations; only cross-reprojection uncertainty can work here")

    if frames_rt_toref is None:
        raise Exception("Some frames_rt_toref must exist for the uncertainty computation, but we don't have any")

    # Now the extrinsics. I look at all the ones that correspond with the
    # specific camera I care about. If the camera is stationary, this will
    # produce exactly one set of extrinsics. If the camera is moving, we may get
    # more than one. At this time I limit to myself to a consecutive block of
    # extrinsics vectors. Once this all works I can relax that requirement
    ifcice = optimization_inputs['indices_frame_camintrinsics_camextrinsics']
    icam_extrinsics = np.unique( ifcice[ifcice[:,1] == icam_intrinsics, 2] ) # sorted
    if icam_extrinsics.size == 0:
        raise Exception(f"No extrinsics corresponding to {icam_intrinsics=}. I don't know what to do")
    if icam_extrinsics.size > 1:
        d = np.unique(np.diff(icam_extrinsics))
        if not (d.size == 1 and d[0] == 1):
            raise Exception("At this point I'm only supporting consecutive block of extrinsics for a given icam_intrinsics")
    if icam_extrinsics[0] < 0:
        if icam_extrinsics.size == 1:
            # Stationary camera, at the reference
            extrinsics_rt_fromref = mrcal.identity_rt()
            istate_extrinsics0    = None
        else:
            # Moving camera. One of the poses is at the reference. This requires
            # more typing. I'll do this later
            raise Exception("Have moving camera, some poses are at the reference. This isn't supported yet")
    else:
        # I will now be guaranteed to get extrinsics_rt_fromref with the right
        # number of extrinsics (all the ones that correspond to this
        # icam_intrinsics). And I know they're a contiguous block in my
        # optimization vector starting with istate_extrinsics0
        extrinsics_rt_fromref = get_input('extrinsics_rt_fromref')[icam_extrinsics,:]
        istate_extrinsics0 = mrcal.state_index_extrinsics(icam_extrinsics[0],
                                                          **optimization_inputs)

    # The intrinsics,extrinsics,frames MUST come from the solve when evaluating
    # the uncertainties. The user is allowed to update the extrinsics in the
    # model after the solve, as long as I use the solve-time ones for the
    # uncertainty computation. Updating the intrinsics invalidates the
    # uncertainty stuff so I COULD grab those from the model. But for good
    # hygiene I get them from the solve as well

    lensmodel       = optimization_inputs['lensmodel']
    intrinsics_data = optimization_inputs['intrinsics'][icam_intrinsics]
    istate_intrinsics0        = mrcal.state_index_intrinsics(icam_intrinsics, **optimization_inputs)
    Nstates_intrinsics        = \
        mrcal.num_intrinsics_optimization_params(**optimization_inputs)
    if not optimization_inputs.get('do_optimize_intrinsics_core') and \
       mrcal.lensmodel_metadata_and_config(lensmodel)['has_core']:
        istate_intrinsics0_onecam = 4
    else:
        istate_intrinsics0_onecam = 0

    Nstate = mrcal.num_states(**optimization_inputs)

    # if method == 'bestq', this has shape (..., Ngeometry, 2, Nstate)
    # else:                                (...,            2, Nstate)
    dq_db = \
        _dq_db__projection_uncertainty( p_cam,
                                        lensmodel, intrinsics_data,
                                        extrinsics_rt_fromref, frames_rt_toref,
                                        Nstate,
                                        istate_intrinsics0,
                                        istate_intrinsics0_onecam,
                                        Nstates_intrinsics,
                                        istate_extrinsics0, istate_frames0,
                                        atinfinity = atinfinity,
                                        method     = method,
                                        Kunpacked_rrp = Kunpacked_rrp)

    # In case of bestq I compute the uncertainty Ngeometry times, and report
    # the best one. To keep things simple I use the trace metric: "best" means
    # the lowest trace(Var_dq)
    if method == 'bestq':
        # shape (..., 2,2)
        V,observed_pixel_uncertainty = \
            _propagate_calibration_uncertainty_bestq( dq_db,
                                                      observed_pixel_uncertainty = observed_pixel_uncertainty,
                                                      optimization_inputs        = optimization_inputs)

        return _covariance_processed(what, V, observed_pixel_uncertainty,
                                     scalar = False)

    else:
        return _propagate_calibration_uncertainty(what,
                                                  dF_dbunpacked              = dq_db,
                                                  observed_pixel_uncertainty = observed_pixel_uncertainty,
                                                  optimization_inputs        = optimization_inputs)


def projection_diff(models,
                    *,
                    implied_Rt10 = None,
                    gridn_width  = 60,
                    gridn_height = None,

                    intrinsics_only = False,
                    distance        = None,

                    use_uncertainties     = True,
                    focus_center          = None,
                    focus_radius          = -1.):
    r'''Compute the difference in projection between N models

SYNOPSIS

    models = ( mrcal.cameramodel('cam0-dance0.cameramodel'),
               mrcal.cameramodel('cam0-dance1.cameramodel') )

    difference,_,q0,_ = mrcal.projection_diff(models)

    print(q0.shape)
    ==> (40,60)

    print(difference.shape)
    ==> (40,60)

    # The differences are computed across a grid. 'q0' is the pixel centers of
    # each grid cell. 'difference' is the projection variation between the two
    # models at each cell

The operation of this tool is documented at
https://mrcal.secretsauce.net/differencing.html

It is often useful to compare the projection behavior of two camera models. For
instance, one may want to validate a calibration by comparing the results of two
different chessboard dances. Or one may want to evaluate the stability of the
intrinsics in response to mechanical or thermal stresses. This function makes
these comparisons, and returns the results. A visualization wrapper is
available: mrcal.show_projection_diff() and the mrcal-show-projection-diff tool.

In the most common case we're given exactly 2 models to compare, and we compute
the differences in projection of each point. If we're given more than 2 models,
we instead compute the standard deviation of the differences between models 1..N
and model0.

The top-level operation of this function:

- Grid the imager
- Unproject each point in the grid using one camera model
- Apply a transformation to map this point from one camera's coord system to the
  other. How we obtain this transformation is described below
- Project the transformed points to the other camera
- Look at the resulting pixel difference in the reprojection

If implied_Rt10 is given, we simply use that as the transformation (this is
currently supported ONLY for diffing exactly 2 cameras). If implied_Rt10 is not
given, we estimate it. Several variables control this. Top-level logic:

  if intrinsics_only:
      Rt10 = identity_Rt()
  else:
      if focus_radius == 0:
          Rt10 = relative_extrinsics(models)
      else:
          Rt10 = implied_Rt10__from_unprojections()

Sometimes we want to look at the intrinsics differences in isolation (if
intrinsics_only), and sometimes we want to use the known geometry in the given
models (not intrinsics_only and focus_radius == 0). If neither of these apply,
we estimate the transformation: this is needed if we're comparing different
calibration results from the same lens.

Given different camera models, we have a different set of intrinsics for each.
Extrinsics differ also, even if we're looking at different calibration of the
same stationary lens: the position and orientation of the camera coordinate
system in respect to the physical camera housing shift with each calibration.
This geometric variation is baked into the intrinsics. So when we project "the
same world point" into both cameras (as is desired when comparing repeated
calibrations), we must apply a geometric transformation because we want to be
comparing projections of world points (relative to the camera housing), not
projections relative to the (floating) camera coordinate systems. This
transformation is unknown, but we can estimate it by fitting projections across
the imager: the "right" transformation would result in apparent low projection
differences in a wide area.

This transformation is computed by implied_Rt10__from_unprojections(), and some
details of its operation are significant:

- The imager area we use for the fit
- Which world points we're looking at

In most practical usages, we would not expect a good fit everywhere in the
imager: areas where no chessboards were observed will not fit well, for
instance. From the point of view of the fit we perform, those ill-fitting areas
should be treated as outliers, and they should NOT be a part of the solve. How
do we specify the well-fitting area? The best way is to use the model
uncertainties: these can be used to emphasize the confident regions of the
imager. This behavior is selected with use_uncertainties=True, which is the
default. If uncertainties aren't available, or if we want a faster solve, pass
use_uncertainties=False. The well-fitting region can then be passed using the
focus_center,focus_radius arguments to indicate the circle in the imager we care
about.

If use_uncertainties then the defaults for focus_center,focus_radius are set to
utilize all the data in the imager. If not use_uncertainties, then the defaults
are to use a more reasonable circle of radius min(width,height)/6 at the center
of the imager. Usually this is sufficiently correct, and we don't need to mess
with it. If we aren't guided to the correct focus region, the
implied-by-the-intrinsics solve will try to fit lots of outliers, which would
result in an incorrect transformation, which in turn would produce overly-high
reported diffs. A common case when this happens is if the chessboard
observations used in the calibration were concentrated to the side of the image
(off-center), no uncertainties were used, and the focus_center was not pointed
to that area.

Unlike the projection operation, the diff operation is NOT invariant under
geometric scaling: if we look at the projection difference for two points at
different locations along a single observation ray, there will be a variation in
the observed diff. This is due to the geometric difference in the two cameras.
If the models differed only in their intrinsics parameters, then this variation
would not appear. Thus we need to know how far from the camera to look, and this
is specified by the "distance" argument. By default (distance = None) we look
out to infinity. If we care about the projection difference at some other
distance, pass that here. Multiple distances can be passed in an iterable. We'll
then fit the implied-by-the-intrinsics transformation using all the distances,
and we'll return the differences for each given distance. Generally the most
confident distance will be where the chessboards were observed at calibration
time.

ARGUMENTS

- models: iterable of mrcal.cameramodel objects we're comparing. Usually there
  will be 2 of these, but more than 2 is allowed. The intrinsics are always
  used; the extrinsics are used only if not intrinsics_only and focus_radius==0

- implied_Rt10: optional transformation to use to line up the camera coordinate
  systems. Most of the time we want to estimate this transformation, so this
  should be omitted or None. Currently this is supported only if exactly two
  models are being compared.

- gridn_width: optional value, defaulting to 60. How many points along the
  horizontal gridding dimension

- gridn_height: how many points along the vertical gridding dimension. If None,
  we compute an integer gridn_height to maintain a square-ish grid:
  gridn_height/gridn_width ~ imager_height/imager_width

- intrinsics_only: optional boolean, defaulting to False. If True: we evaluate
  the intrinsics of each lens in isolation by assuming that the coordinate
  systems of each camera line up exactly

- distance: optional value, defaulting to None. Has an effect only if not
  intrinsics_only. The projection difference varies depending on the range to
  the observed world points, with the queried range set in this 'distance'
  argument. If None (the default) we look out to infinity. We can compute the
  implied-by-the-intrinsics transformation off multiple distances if they're
  given here as an iterable. This is especially useful if we have uncertainties,
  since then we'll emphasize the best-fitting distances.

- use_uncertainties: optional boolean, defaulting to True. Used only if not
  intrinsics_only and focus_radius!=0. If True we use the whole imager to fit
  the implied-by-the-intrinsics transformation, using the uncertainties to
  emphasize the confident regions. If False, it is important to select the
  confident region using the focus_center and focus_radius arguments. If
  use_uncertainties is True, but that data isn't available, we report a warning,
  and try to proceed without.

- focus_center: optional array of shape (2,); the imager center by default. Used
  only if not intrinsics_only and focus_radius!=0. Used to indicate that the
  implied-by-the-intrinsics transformation should use only those pixels a
  distance focus_radius from focus_center. This is intended to be used if no
  uncertainties are available, and we need to manually select the focus region.

- focus_radius: optional value. If use_uncertainties then the default is LARGE,
  to use the whole imager. Else the default is min(width,height)/6. Used to
  indicate that the implied-by-the-intrinsics transformation should use only
  those pixels a distance focus_radius from focus_center. This is intended to be
  used if no uncertainties are available, and we need to manually select the
  focus region. To avoid computing the transformation, either pass
  focus_radius=0 (to use the extrinsics in the given models) or pass
  intrinsics_only=True (to use the identity transform).

RETURNED VALUE

A tuple

- difflen: a numpy array of shape (...,gridn_height,gridn_width) containing the
  magnitude of differences at each cell, or the standard deviation of the
  differences between models 1..N and model0 if len(models)>2. if
  len(models)==2: this is nps.mag(diff). If the given 'distance' argument was an
  iterable, the shape is (len(distance),...). Otherwise the shape is (...)

- diff: a numpy array of shape (...,gridn_height,gridn_width,2) containing the
  vector of differences at each cell. If len(models)>2 this isn't defined, so
  None is returned. If the given 'distance' argument was an iterable, the shape
  is (len(distance),...). Otherwise the shape is (...)

- q0: a numpy array of shape (gridn_height,gridn_width,2) containing the
  pixel coordinates of each grid cell

- Rt10: the geometric Rt transformation in an array of shape (...,4,3). This is
  the relative transformation we ended up using, which is computed using the
  logic above (using intrinsics_only and focus_radius). if len(models)>2: this
  is an array of shape (len(models)-1,4,3), with slice i representing the
  transformation between camera 0 and camera i+1.

    '''

    if len(models) < 2:
        raise Exception("At least 2 models are required to compute the diff")
    if len(models) > 2 and implied_Rt10 is not None:
        raise Exception("A given implied_Rt10 is currently supported ONLY if exactly 2 models are being compared")

    # If the distance is iterable, the shape of the output is (len(distance),
    # ...). Otherwise it is just (...). In the intermediate computations, the
    # quantities ALWAYS have shape (len(distance), ...). I select the desired
    # shape at the end
    distance_is_iterable = True
    try:    len(distance)
    except: distance_is_iterable = False

    if distance is None:
        atinfinity = True
        distance   = np.ones((1,), dtype=float)
    else:
        atinfinity = False
        distance   = nps.atleast_dims(np.array(distance), -1)
    distance   = nps.mv(distance.ravel(), -1,-4)

    imagersizes = np.array([model.imagersize() for model in models])
    if np.linalg.norm(np.std(imagersizes, axis=-2)) != 0:
        raise Exception("The diff function needs all the imager dimensions to match. Instead got {}". \
                        format(imagersizes))
    W,H=imagersizes[0]

    lensmodels      = [model.intrinsics()[0] for model in models]
    intrinsics_data = [model.intrinsics()[1] for model in models]

    for i in range(len(models)):
        import re
        if mrcal.lensmodel_metadata_and_config(lensmodels[i])['noncentral'] and not \
           (re.match("LENSMODEL_CAHVORE_",models[i].intrinsics()[0]) and nps.norm2(models[i].intrinsics()[1][-3:]) < 1e-12):
            if not atinfinity:
                raise Exception(f"Model {i} is noncentral, so I can only evaluate the diff at infinity")
            if re.match("LENSMODEL_CAHVORE_",lensmodels[i]):
                if use_uncertainties:
                    raise Exception("I have a noncentral model. No usable uncertainties for those yet")
                # Special-case to centralize CAHVORE. This path will need to be
                # redone when I do noncentral models "properly", but this will
                # do in the meantime
                intrinsics_data[i][-3:] = 0
            else:
                raise Exception("I have a non-CAHVORE noncentral model. This isn't supported yet")

    # v  shape (Ncameras,Nheight,Nwidth,3)
    # q0 shape (         Nheight,Nwidth,2)
    v,q0 = mrcal.sample_imager_unproject(gridn_width, gridn_height,
                                         W, H,
                                         lensmodels, intrinsics_data,
                                         normalize = True)

    uncertainties = None
    if use_uncertainties and \
       not intrinsics_only and focus_radius != 0 and \
       implied_Rt10 is None:
        try:
            # len(uncertainties) = Ncameras. Each has shape (len(distance),Nh,Nw)
            uncertainties = \
                [ mrcal.projection_uncertainty(# shape (len(distance),Nheight,Nwidth,3)
                                               v[i] * distance,
                                               models[i],
                                               atinfinity = atinfinity,
                                               what       = 'worstdirection-stdev') \
                  for i in range(len(models)) ]
        except Exception as e:
            print(f"WARNING: projection_diff() was asked to use uncertainties, but they aren't available/couldn't be computed. Falling back on the region-based-only logic. Caught exception: {e}",
                  file = sys.stderr)

    if focus_center is None:
        focus_center = ((W-1.)/2., (H-1.)/2.)

    if focus_radius < 0:
        if uncertainties is not None:
            focus_radius = max(W,H) * 100 # whole imager
        else:
            focus_radius = min(W,H)/6.

    if len(models) == 2:
        # Two models. Take the difference and call it good
        if implied_Rt10 is not None:
            Rt10 = implied_Rt10

        else:
            if intrinsics_only:
                Rt10 = mrcal.identity_Rt()
            else:
                if focus_radius == 0:
                    Rt10 = mrcal.compose_Rt(models[1].extrinsics_Rt_fromref(),
                                            models[0].extrinsics_Rt_toref())
                else:
                    # weights has shape (len(distance),Nh,Nw))
                    if uncertainties is not None:
                        weights = 1.0 / (uncertainties[0]*uncertainties[1])

                        # It appears to work better if I discount the uncertain regions
                        # even more. This isn't a principled decision, and is supported
                        # only by a little bit of data. The differencing.org I'm writing
                        # now will contain a weighted diff of culled and not-culled
                        # splined model data. That diff computation requires this.
                        weights *= weights
                    else:
                        weights = None

                    # weight may be inf or nan. implied_Rt10__from_unprojections() will
                    # clean those up, as well as any inf/nan in v (from failed
                    # unprojections)
                    Rt10 = \
                        implied_Rt10__from_unprojections(q0,
                                                         # shape (len(distance),Nheight,Nwidth,3)
                                                         v[0,...] * distance,
                                                         v[1,...],
                                                         weights      = weights,
                                                         atinfinity   = atinfinity,
                                                         focus_center = focus_center,
                                                         focus_radius = focus_radius)


        q1 = mrcal.project( mrcal.transform_point_Rt(Rt10,
                                                     # shape (len(distance),Nheight,Nwidth,3)
                                                     v[0,...] * distance),
                            lensmodels[1], intrinsics_data[1])
        # shape (len(distance),Nheight,Nwidth,2)
        q1 = nps.atleast_dims(q1, -4)

        diff    = q1 - q0
        difflen = nps.mag(diff)

    else:

        # Many models. Look at the stdev
        def get_Rt10( i0, i1 ):

            if intrinsics_only:
                return mrcal.identity_Rt()

            if focus_radius == 0:
                return mrcal.compose_Rt(models[i1].extrinsics_Rt_fromref(),
                                        models[i0].extrinsics_Rt_toref())

            # weights has shape (len(distance),Nh,Nw))
            if uncertainties is not None:
                weights = 1.0 / (uncertainties[i0]*uncertainties[i1])

                # It appears to work better if I discount the uncertain regions
                # even more. This isn't a principled decision, and is supported
                # only by a little bit of data. The differencing.org I'm writing
                # now will contain a weighted diff of culled and not-culled
                # splined model data. That diff computation requires this.
                weights *= weights
            else:
                weights = None

            v0 = v[i0,...]
            v1 = v[i1,...]

            return \
                implied_Rt10__from_unprojections(q0,
                                                 # shape (len(distance),Nheight,Nwidth,3)
                                                 v0*distance,
                                                 v1,
                                                 weights      = weights,
                                                 atinfinity   = atinfinity,
                                                 focus_center = focus_center,
                                                 focus_radius = focus_radius)
        def get_reprojections(q0, Rt10,
                              lensmodel, intrinsics_data):
            q1 = mrcal.project(mrcal.transform_point_Rt(Rt10,
                                                        # shape (len(distance),Nheight,Nwidth,3)
                                                        v[0,...]*distance),
                               lensmodel, intrinsics_data)
            # returning shape (len(distance),Nheight,Nwidth,2)
            return nps.atleast_dims(q1, -4)

        Ncameras = len(v)
        # shape (Ncameras-1, 4,3)
        Rt10 = nps.cat(*[ get_Rt10(0,i) for i in range(1,Ncameras)])

        # shape (Ncameras-1,len(distance),Nheight,Nwidth,2)
        grids = nps.cat(*[get_reprojections(q0, Rt10[i-1],
                                            lensmodels[i], intrinsics_data[i]) \
                          for i in range(1,Ncameras)])

        diff    = None
        # shape (len(distance),Nheight,Nwidth)
        difflen = np.sqrt(np.mean(nps.norm2(grids-q0),axis=0))

    # difflen, diff, q0 currently all have shape (len(distance), ...). If the
    # given distance was NOT an iterable, I strip out that leading dimension
    if not distance_is_iterable:
        if difflen.shape[0] != 1:
            raise Exception(f"distance_is_iterable is False, but leading shape of difflen is not 1 (difflen.shape = {difflen.shape}). This is a bug. Giving up")
        difflen = difflen[0,...]

        if diff is not None:
            if diff.shape[0] != 1:
                raise Exception(f"distance_is_iterable is False, but leading shape of diff is not 1 (diff.shape = {diff.shape}). This is a bug. Giving up")
            diff = diff[0,...]

    return difflen, diff, q0, Rt10


def stereo_pair_diff(model_pairs,
                     *,
                     gridn_width  = 60,
                     gridn_height = None,
                     distance     = None):
    r'''Compute the difference in projection between N model_pairs

SYNOPSIS

    model_pairs = ( mrcal.cameramodel('cam0-dance0.cameramodel'),
               mrcal.cameramodel('cam0-dance1.cameramodel') )

    difference,_,q0,_ = mrcal.stereo_pair_diff(model_pairs)

    print(q0.shape)
    ==> (40,60)

    print(difference.shape)
    ==> (40,60)

    # The differences are computed across a grid. 'q0' is the pixel centers of
    # each grid cell. 'difference' is the projection variation between the two
    # model_pairs at each cell

The operation of this tool is documented at
https://mrcal.secretsauce.net/differencing.html

It is often useful to compare the projection behavior of two camera model_pairs. For
instance, one may want to validate a calibration by comparing the results of two
different chessboard dances. Or one may want to evaluate the stability of the
intrinsics in response to mechanical or thermal stresses. This function makes
these comparisons, and returns the results. A visualization wrapper is
available: mrcal.show_projection_diff() and the mrcal-show-projection-diff tool.

In the most common case we're given exactly 2 model_pairs to compare, and we compute
the differences in projection of each point. If we're given more than 2 model_pairs,
we instead compute the standard deviation of the differences between model_pairs 1..N
and model0.

The top-level operation of this function:

- Grid the imager
- Unproject each point in the grid using one camera model
- Apply a transformation to map this point from one camera's coord system to the
  other. How we obtain this transformation is described below
- Project the transformed points to the other camera
- Look at the resulting pixel difference in the reprojection

Sometimes we want to look at the intrinsics differences in isolation (if
intrinsics_only), and sometimes we want to use the known geometry in the given
model_pairs (not intrinsics_only and focus_radius == 0). If neither of these apply,
we estimate the transformation: this is needed if we're comparing different
calibration results from the same lens.

Given different camera model_pairs, we have a different set of intrinsics for each.
Extrinsics differ also, even if we're looking at different calibration of the
same stationary lens: the position and orientation of the camera coordinate
system in respect to the physical camera housing shift with each calibration.
This geometric variation is baked into the intrinsics. So when we project "the
same world point" into both cameras (as is desired when comparing repeated
calibrations), we must apply a geometric transformation because we want to be
comparing projections of world points (relative to the camera housing), not
projections relative to the (floating) camera coordinate systems. This
transformation is unknown, but we can estimate it by fitting projections across
the imager: the "right" transformation would result in apparent low projection
differences in a wide area.

This transformation is computed by implied_Rt10__from_unprojections(), and some
details of its operation are significant:

- The imager area we use for the fit
- Which world points we're looking at

In most practical usages, we would not expect a good fit everywhere in the
imager: areas where no chessboards were observed will not fit well, for
instance. From the point of view of the fit we perform, those ill-fitting areas
should be treated as outliers, and they should NOT be a part of the solve. How
do we specify the well-fitting area? The best way is to use the model
uncertainties: these can be used to emphasize the confident regions of the
imager. This behavior is selected with use_uncertainties=True, which is the
default. If uncertainties aren't available, or if we want a faster solve, pass
use_uncertainties=False. The well-fitting region can then be passed using the
focus_center,focus_radius arguments to indicate the circle in the imager we care
about.

If use_uncertainties then the defaults for focus_center,focus_radius are set to
utilize all the data in the imager. If not use_uncertainties, then the defaults
are to use a more reasonable circle of radius min(width,height)/6 at the center
of the imager. Usually this is sufficiently correct, and we don't need to mess
with it. If we aren't guided to the correct focus region, the
implied-by-the-intrinsics solve will try to fit lots of outliers, which would
result in an incorrect transformation, which in turn would produce overly-high
reported diffs. A common case when this happens is if the chessboard
observations used in the calibration were concentrated to the side of the image
(off-center), no uncertainties were used, and the focus_center was not pointed
to that area.

Unlike the projection operation, the diff operation is NOT invariant under
geometric scaling: if we look at the projection difference for two points at
different locations along a single observation ray, there will be a variation in
the observed diff. This is due to the geometric difference in the two cameras.
If the model_pairs differed only in their intrinsics parameters, then this
variation would not appear. Thus we need to know how far from the camera to
look, and this is specified by the "distance" argument. By default (distance =
None) we look out to infinity. If we care about the projection difference at
some other distance, pass that here. Generally the most confident distance will
be where the chessboards were observed at calibration time.

ARGUMENTS

- model_pairs: iterable of mrcal.cameramodel objects we're comparing. Exactly
  two pairs are expected. The intrinsics are always used; the extrinsics are
  used only if not intrinsics_only and focus_radius==0

- implied_Rt10: optional transformation to use to line up the camera coordinate
  systems. Most of the time we want to estimate this transformation, so this
  should be omitted or None. Currently this is supported only if exactly two
  model_pairs are being compared.

- gridn_width: optional value, defaulting to 60. How many points along the
  horizontal gridding dimension

- gridn_height: how many points along the vertical gridding dimension. If None,
  we compute an integer gridn_height to maintain a square-ish grid:
  gridn_height/gridn_width ~ imager_height/imager_width

- intrinsics_only: optional boolean, defaulting to False. If True: we evaluate
  the intrinsics of each lens in isolation by assuming that the coordinate
  systems of each camera line up exactly

- distance: optional value, defaulting to None. Has an effect only if not
  intrinsics_only. The projection difference varies depending on the range to
  the observed world points, with the queried range set in this 'distance'
  argument. If None (the default) we look out to infinity.

- use_uncertainties: optional boolean, defaulting to True. Used only if not
  intrinsics_only and focus_radius!=0. If True we use the whole imager to fit
  the implied-by-the-intrinsics transformation, using the uncertainties to
  emphasize the confident regions. If False, it is important to select the
  confident region using the focus_center and focus_radius arguments. If
  use_uncertainties is True, but that data isn't available, we report a warning,
  and try to proceed without.

- focus_center: optional array of shape (2,); the imager center by default. Used
  only if not intrinsics_only and focus_radius!=0. Used to indicate that the
  implied-by-the-intrinsics transformation should use only those pixels a
  distance focus_radius from focus_center. This is intended to be used if no
  uncertainties are available, and we need to manually select the focus region.

- focus_radius: optional value. If use_uncertainties then the default is LARGE,
  to use the whole imager. Else the default is min(width,height)/6. Used to
  indicate that the implied-by-the-intrinsics transformation should use only
  those pixels a distance focus_radius from focus_center. This is intended to be
  used if no uncertainties are available, and we need to manually select the
  focus region. To avoid computing the transformation, either pass
  focus_radius=0 (to use the extrinsics in the given model_pairs) or pass
  intrinsics_only=True (to use the identity transform).

RETURNED VALUE

A tuple

- difflen: a numpy array of shape (...,gridn_height,gridn_width) containing the
  magnitude of differences at each cell

- diff: a numpy array of shape (...,gridn_height,gridn_width,2) containing the
  vector of differences at each cell

- q0: a numpy array of shape (gridn_height,gridn_width,2) containing the
  pixel coordinates of each grid cell

- Rt10: the geometric Rt transformation in an array of shape (...,4,3). This is
  the relative transformation we ended up using, which is computed using the
  logic above (using intrinsics_only and focus_radius)

    '''

    if len(model_pairs) != 2:
        raise Exception("Exactly 2 model_pairs are expected")



    ### radius does nothing
    ### "distance is None" does something useful?

    if distance is None:
        atinfinity = True
        distance   = np.ones((1,), dtype=float)
    else:
        atinfinity = False
        distance   = nps.atleast_dims(np.array(distance), -1)
    distance   = nps.mv(distance.ravel(), -1,-4)

    Rt10_pairs = [ mrcal.compose_Rt( model_pair[1].extrinsics_Rt_fromref(),
                                     model_pair[0].extrinsics_Rt_toref() ) \
                   for model_pair in model_pairs ]

    if atinfinity:
        for Rt10 in Rt10_pairs:
            Rt10[3,:] = 0

    for model_pair in model_pairs:
        for model in model_pair:
            import re
            if mrcal.lensmodel_metadata_and_config(model.intrinsics()[0])['noncentral'] and not \
               (re.match("LENSMODEL_CAHVORE_",model.intrinsics()[0]) and nps.norm2(model.intrinsics()[1][-3:]) < 1e-12):
                if not atinfinity:
                    raise Exception(f"Model {model.intrinsics()[0]} is noncentral, so I can only evaluate the diff at infinity")
                if re.match("LENSMODEL_CAHVORE_",model.intrinsics()[0]):
                    if use_uncertainties:
                        raise Exception("I have a noncentral model. No usable uncertainties for those yet")
                    # Special-case to centralize CAHVORE. This path will need to be
                    # redone when I do noncentral models "properly", but this will
                    # do in the meantime
                    intrinsics_data = model.intrinsics()[1]
                    intrinsics_data[-3:] = 0
                    model.intrinsics( intrinsics = (model.intrinsics()[0],
                                                    intrinsics_data) )
                else:
                    raise Exception("I have a non-CAHVORE noncentral model. This isn't supported yet")



    imagersizes0 = np.array([model_pair[0].imagersize() for model_pair in model_pairs])
    if np.linalg.norm(np.std(imagersizes0, axis=-2)) != 0:
        raise Exception("The diff function needs all the imager dimensions of the first camera in each pair to match. Instead got {}". \
                        format(imagersizes0))


    q0 = mrcal.sample_imager(gridn_width, gridn_height,
                             *imagersizes0[0])

    # q1 shape (Npairs, Nheight,Nwidth,2)
    q1 = [ mrcal.project( \
              mrcal.transform_point_Rt(Rt10_pairs[ipair],
                                       distance *
                                       mrcal.unproject(q0,
                                                       *model_pairs[ipair][0].intrinsics(),
                                                       normalize = True)),
                          *model_pairs[ipair][1].intrinsics() ) \
           for ipair in range(len(model_pairs)) ]

    diff    = q1[1] - q1[0]
    difflen = nps.mag(diff)

    # difflen, diff, q0 currently all have shape (len(distance), ...). If the
    # given distance was NOT an iterable, I strip out that leading dimension
    if difflen.shape[0] != 1:
        raise Exception(f"The leading shape of difflen is not 1 (difflen.shape = {difflen.shape}). This is a bug. Giving up")
    difflen = difflen[0,...]

    if diff.shape[0] != 1:
        raise Exception(f"The leading shape of diff is not 1 (diff.shape = {diff.shape}). This is a bug. Giving up")
    diff = diff[0,...]

    return difflen, diff, q0


def is_within_valid_intrinsics_region(q, model):
    r'''Which of the pixel coordinates fall within the valid-intrinsics region?

SYNOPSIS

    mask = mrcal.is_within_valid_intrinsics_region(q, model)
    q_trustworthy = q[mask]

mrcal camera models may have an estimate of the region of the imager where the
intrinsics are trustworthy (originally computed with a low-enough error and
uncertainty). When using a model, we may want to process points that fall
outside of this region differently from points that fall within this region.
This function returns a mask that indicates whether each point is within the
region or not.

If no valid-intrinsics region is defined in the model, returns None.

ARGUMENTS

- q: an array of shape (..., 2) of pixel coordinates

- model: the model we're interrogating

RETURNED VALUE

The mask that indicates whether each point is within the region

    '''

    r = model.valid_intrinsics_region()
    if r is None:
        return None

    from shapely.geometry import Polygon,Point

    r = Polygon(r)

    mask = np.zeros(q.shape[:-1], dtype=bool)
    mask_flat = mask.ravel()
    q_flat = q.reshape(q.size//2, 2)
    for i in range(q.size // 2):
        if r.contains(Point(q_flat[i])):
            mask_flat[i] = True
    return mask

