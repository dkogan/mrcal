#!/usr/bin/python3

'''Routines for analysis of camera projection

This is largely dealing with uncertainty and projection diff operations.

All functions are exported into the mrcal module. So you can call these via
mrcal.model_analysis.fff() or mrcal.fff(). The latter is preferred.

'''

import numpy as np
import numpysane as nps
import sys
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
        r = np.random.random(3) * 1e-3

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

        rt = np.random.random(6) * 1e-3

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
    r'''Compute the worst-direction standard deviation from a 2x2 covariance matrix

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

The covariance of a (2,) random variable can be described by a (2,2)
positive-definite symmetric matrix. The 1-sigma contour of this random variable
is described by an ellipse with its axes aligned with the eigenvectors of the
covariance, and the semi-major and semi-minor axis lengths specified as the sqrt
of the corresponding eigenvalues. This function returns the worst-case standard
deviation of the given covariance: the sqrt of the larger of the two
eigenvalues.

This function supports broadcasting fully.

DERIVATION

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

    a = cov[..., 0,0]
    b = cov[..., 1,0]
    c = cov[..., 1,1]
    return np.sqrt((a+c)/2 + np.sqrt( (a-c)*(a-c)/4 + b*b))


def _propagate_calibration_uncertainty( dF_dbpacked,
                                        factorization, Jpacked,
                                        Nmeasurements_observations,
                                        observed_pixel_uncertainty, what ):
    r'''Helper for uncertainty propagation functions

Propagates the calibration-time uncertainty to compute Var(F) for some arbitrary
vector F. The user specifies the gradient dF/db: the sensitivity of F to noise
in the calibration state. The vector F can have any length: this is inferred
from the dimensions of the given dF/db gradient.

The given factorization uses the packed, unitless state: b*.

The given Jpacked uses the packed, unitless state: b*. Jpacked applies to all
observations. The leading Nmeasurements_observations rows apply to the
observations of the calibration object, and we use just those for the input
noise propagation. if Nmeasurements_observations is None: assume that ALL the
measurements come from the calibration object observations; a simplifed
expression can be used in this case

The given dF_dbpacked uses the packed, unitless state b*, so it already includes
the multiplication by D in the expressions below. It's usually sparse, but
stored densely.

The uncertainty computation in
http://mrcal.secretsauce.net/uncertainty.html concludes that

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
     The result has shape (Nstate,2)

  2. pre-multiply by dF/db D

  3. multiply by observed_pixel_uncertainty^2

In the regularized case:

  Var(F) = dF/db D inv(J*tJ*) J*[observations]t J*[observations] inv(J*tJ*) D dF/dbt

  1. solve( J*tJ*, D dF/dbt)
     The result has shape (Nstate,2)

  2. Pre-multiply by J*[observations]
     The result has shape (Nmeasurements_observations,2)

  3. Compute the sum of the outer products of each row

  4. multiply by observed_pixel_uncertainty^2

    '''

    # shape (2,Nstate)
    A = factorization.solve_xt_JtJ_bt( dF_dbpacked )
    if Nmeasurements_observations is not None:
        # I have regularization. Use the more complicated expression

        # I see no python way to do matrix multiplication with sparse matrices,
        # so I have my own routine in C. AND the C routine does the outer
        # product, so there's no big temporary expression. It's much faster
        Var_dF = mrcal._mrcal_npsp._A_Jt_J_At(A, Jpacked.indptr, Jpacked.indices, Jpacked.data,
                                              Nleading_rows_J = Nmeasurements_observations)
    else:
        # No regularization. Use the simplified expression
        Var_dF = nps.matmult(dF_dbpacked, nps.transpose(A))

    if what == 'covariance':           return Var_dF * observed_pixel_uncertainty*observed_pixel_uncertainty
    if what == 'worstdirection-stdev': return worst_direction_stdev(Var_dF) * observed_pixel_uncertainty
    if what == 'rms-stdev':            return np.sqrt(nps.trace(Var_dF)/2.) * observed_pixel_uncertainty
    else: raise Exception("Shouldn't have gotten here. There's a bug")


def _projection_uncertainty( p_cam,
                             lensmodel, intrinsics_data,
                             extrinsics_rt_fromref, frames_rt_toref,
                             factorization, Jpacked, optimization_inputs,
                             istate_intrinsics, istate_extrinsics, istate_frames,
                             slice_optimized_intrinsics,
                             Nmeasurements_observations,
                             observed_pixel_uncertainty,
                             what):
    r'''Helper for projection_uncertainty()

    See docs for _propagate_calibration_uncertainty() and
    projection_uncertainty()

    This function does all the work when observing points with a finite range

    '''

    Nstate = Jpacked.shape[-1]
    dq_dbief = np.zeros(p_cam.shape[:-1] + (2,Nstate), dtype=float)

    if frames_rt_toref is not None:
        Nframes = len(frames_rt_toref)

    if extrinsics_rt_fromref is not None:
        p_ref = \
            mrcal.transform_point_rt( mrcal.invert_rt(extrinsics_rt_fromref),
                                      p_cam )
    else:
        p_ref = p_cam

    if frames_rt_toref is not None:
        # The point in the coord system of all the frames. I index the frames on
        # axis -2
        # shape (..., Nframes, 3)
        p_frames = mrcal.transform_point_rt( # shape (Nframes,6)
                                             mrcal.invert_rt(frames_rt_toref),
                                             # shape (...,1,3)
                                             nps.dummy(p_ref,-2) )

        # I now have the observed point represented in the coordinate system of
        # the frames. This is independent of any intrinsics-implied rotation, or
        # anything of the sort. I project this point back to pixels, through
        # noisy estimates of the frames, extrinsics and intrinsics.
        #
        # I transform each frame-represented point back to the reference coordinate
        # system, and I average out each estimate to get the one p_ref I will use. I
        # already have p_ref, so I don't actually need to compute the value; I just
        # need the gradients

        # dprefallframes_dframes has shape (..., Nframes,3,6)
        _, \
        dprefallframes_dframes, \
        _ = mrcal.transform_point_rt( frames_rt_toref, p_frames,
                                      get_gradients = True)

        # shape (..., 3,6*Nframes)
        # /Nframes because I compute the mean over all the frames
        dpref_dframes = nps.clump( nps.mv(dprefallframes_dframes, -3, -2),
                                   n = -2 ) / Nframes

    _, dq_dpcam, dq_dintrinsics = \
        mrcal.project( p_cam, lensmodel, intrinsics_data,
                       get_gradients = True)

    if istate_intrinsics is not None:
        dq_dintrinsics_optimized = dq_dintrinsics[..., slice_optimized_intrinsics]
        Nintrinsics = dq_dintrinsics_optimized.shape[-1]
        dq_dbief[..., istate_intrinsics:istate_intrinsics+Nintrinsics] = \
            dq_dintrinsics_optimized

    if extrinsics_rt_fromref is not None:
        _, dpcam_drt, dpcam_dpref = \
            mrcal.transform_point_rt(extrinsics_rt_fromref, p_ref,
                                     get_gradients = True)

        dq_dbief[..., istate_extrinsics:istate_extrinsics+6] = \
            nps.matmult(dq_dpcam, dpcam_drt)

        if frames_rt_toref is not None:
            dq_dbief[..., istate_frames:istate_frames+Nframes*6] = \
                nps.matmult(dq_dpcam, dpcam_dpref, dpref_dframes)
    else:
        if frames_rt_toref is not None:
            dq_dbief[..., istate_frames:istate_frames+Nframes*6] = \
                nps.matmult(dq_dpcam, dpref_dframes)

    # Make dq_dbief use the packed state. I call "unpack_state" because the
    # state is in the denominator
    mrcal.unpack_state(dq_dbief, **optimization_inputs)
    return \
        _propagate_calibration_uncertainty( dq_dbief,
                                            factorization, Jpacked,
                                            Nmeasurements_observations,
                                            observed_pixel_uncertainty,
                                            what)


def _projection_uncertainty_rotationonly( p_cam,
                                          lensmodel, intrinsics_data,
                                          extrinsics_rt_fromref, frames_rt_toref,
                                          factorization, Jpacked, optimization_inputs,
                                          istate_intrinsics, istate_extrinsics, istate_frames,
                                          slice_optimized_intrinsics,
                                          Nmeasurements_observations,
                                          observed_pixel_uncertainty,
                                          what):
    r'''Helper for projection_uncertainty()

    See docs for _propagate_calibration_uncertainty() and
    projection_uncertainty()

    This function does all the work when observing points at infinity

    '''

    Nstate = Jpacked.shape[-1]
    dq_dbief = np.zeros(p_cam.shape[:-1] + (2,Nstate), dtype=float)

    if frames_rt_toref is not None:
        Nframes = len(frames_rt_toref)

    if extrinsics_rt_fromref is not None:
        p_ref = \
            mrcal.rotate_point_r( -extrinsics_rt_fromref[..., :3], p_cam )
    else:
        p_ref = p_cam

    if frames_rt_toref is not None:
        # The point in the coord system of all the frames. I index the frames on
        # axis -2
        # shape (..., Nframes, 3)
        p_frames = mrcal.rotate_point_r( # shape (Nframes,3)
                                         -frames_rt_toref[...,:3],
                                         # shape (...,1,3)
                                         nps.dummy(p_ref,-2) )

        # I now have the observed point represented in the coordinate system of
        # the frames. This is independent of any intrinsics-implied rotation, or
        # anything of the sort. I project this point back to pixels, through
        # noisy estimates of the frames, extrinsics and intrinsics.
        #
        # I transform each frame-represented point back to the reference coordinate
        # system, and I average out each estimate to get the one p_ref I will use. I
        # already have p_ref, so I don't actually need to compute the value; I just
        # need the gradients

        # dprefallframes_dframesr has shape (..., Nframes,3,3)
        _, \
        dprefallframes_dframesr, \
        _ = mrcal.rotate_point_r( frames_rt_toref[...,:3], p_frames,
                                  get_gradients = True)

    _, dq_dpcam, dq_dintrinsics = \
        mrcal.project( p_cam, lensmodel, intrinsics_data,
                       get_gradients = True)

    if istate_intrinsics is not None:
        dq_dintrinsics_optimized = dq_dintrinsics[..., slice_optimized_intrinsics]
        Nintrinsics = dq_dintrinsics_optimized.shape[-1]
        dq_dbief[..., istate_intrinsics:istate_intrinsics+Nintrinsics] = \
            dq_dintrinsics_optimized

    if extrinsics_rt_fromref is not None:
        _, dpcam_dr, dpcam_dpref = \
            mrcal.rotate_point_r(extrinsics_rt_fromref[...,:3], p_ref,
                                 get_gradients = True)
        dq_dbief[..., istate_extrinsics:istate_extrinsics+3] = \
            nps.matmult(dq_dpcam, dpcam_dr)

        if frames_rt_toref is not None:

            dq_dpref = nps.matmult(dq_dpcam, dpcam_dpref)

            # dprefallframes_dframesr has shape (..., Nframes,3,3)
            for i in range(Nframes):
                dq_dbief[..., istate_frames+6*i:istate_frames+6*i+3] = \
                    nps.matmult(dq_dpref, dprefallframes_dframesr[...,i,:,:]) / Nframes
    else:
        if frames_rt_toref is not None:
            # dprefallframes_dframesr has shape (..., Nframes,3,3)
            for i in range(Nframes):
                dq_dbief[..., istate_frames+6*i:istate_frames+6*i+3] = \
                    nps.matmult(dq_dpcam, dprefallframes_dframesr[...,i,:,:]) / Nframes

    # Make dq_dbief use the packed state. I call "unpack_state" because the
    # state is in the denominator
    mrcal.unpack_state(dq_dbief, **optimization_inputs)
    return \
        _propagate_calibration_uncertainty( dq_dbief,
                                            factorization, Jpacked,
                                            Nmeasurements_observations,
                                            observed_pixel_uncertainty,
                                            what)


def projection_uncertainty( p_cam, model,
                            *,
                            atinfinity = False,

                            # what we're reporting
                            what = 'covariance',
                            observed_pixel_uncertainty = None):
    r'''Compute the projection uncertainty of a camera-referenced point

This is the interface to the uncertainty computations described in
http://mrcal.secretsauce.net/uncertainty.html

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
    # pixel obsevation is 0.5 pixels, in the worst direction

After a camera model is computed via a calibration process, the model is
ultimately used in projection/unprojection operations to map between world
coordinates and projected pixel coordinates. We never know the parameters of the
model perfectly, and it is VERY useful to know the resulting uncertainty of
projection. This can be used, among other things, to

- propagate the projection noise down to whatever is using the observed pixels
  to do stuff

- evaluate the quality of calibrations, to know whether a given calibration
  should be accepted, or rejected

- evaluate the stability of a computed model

I quantify uncertainty by propagating expected noise on observed chessboard
corners through the optimization problem we're solving during calibration time
to the solved parameters. And then propagating the noise on the parameters
through projection.

The uncertainties can be visualized with the mrcal-show-projection-uncertainty
tool.

ARGUMENTS

This function accepts an array of camera-referenced points p_cam, a
mrcal.cameramodel object and a few meta-parameters that describe details of the
behavior. This function broadcasts on p_cam only. We accept

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

  - 'rms-stdev':            return the RMS of the worst and best direction
                            standard deviations

- observed_pixel_uncertainty: optional value, defaulting to None. The
  uncertainty of the observed chessboard corners being propagated through the
  solve and projection. If omitted or None, this input uncertainty is inferred
  from the residuals at the optimum. Most people should omit this

RETURN VALUE

A numpy array of uncertainties. If p_cam has shape (..., 3) then:

if what == 'covariance': we return an array of shape (..., 2,2)
else:                    we return an array of shape (...)

    '''


    # I computed Var(b) earlier, which contains the variance of ALL the optimization
    # parameters together. The noise on the chessboard poses is coupled to the noise
    # on the extrinsics and to the noise on the intrinsics. And we can apply all these
    # together to propagate the uncertainty.

    # Let's define some variables:

    # - b_i: the intrinsics of a camera
    # - b_e: the extrinsics of that camera (T_cr)
    # - b_f: ALL the chessboard poses (T_fr)
    # - b_ief: the concatenation of b_i, b_e and b_f

    # I have

    #     dq = q0 + dq/db_ief db_ief

    #     Var(q) = dq/db_ief Var(b_ief) (dq/db_ief)t

    #     Var(b_ief) is a subset of Var(b), computed above.

    #     dq/db_ief = [dq/db_i dq/db_e dq/db_f]

    #     dq/db_e = dq/dpcam dpcam/db_e

    #     dq/db_f = dq/dpcam dpcam/dpref dpref/db_f / Nframes

    # dq/db_i and all the constituent expressions comes directly from the project()
    # and transform calls above. Depending on the details of the optimization problem,
    # some of these may not exist. For instance, if we're looking at a camera that is
    # sitting at the reference coordinate system, then there is no b_e, and Var_ief is
    # smaller: it's just Var_if. If we somehow know the poses of the frames, then
    # there's no Var_f. If we want to know the uncertainty at distance=infinity, then
    # we ignore all the translation components of b_e and b_f.



    # Alright, so we have Var(q). We could claim victory at that point. But it'd be
    # nice to convert Var(q) into a single number that describes my projection
    # uncertainty at q. Empirically I see that Var(dq) often describes an eccentric
    # ellipse, so I want to look at the length of the major axis of the 1-sigma
    # ellipse:

    #     eig (a b) --> (a-l)*(c-l)-b^2 = 0 --> l^2 - (a+c) l + ac-b^2 = 0
    #         (b c)

    #     --> l = (a+c +- sqrt( a^2+2ac+c^2 - 4ac + 4b^2)) / 2 =
    #           = (a+c +- sqrt( a^2-2ac+c^2 + 4b^2)) / 2 =
    #           = (a+c)/2 +- sqrt( (a-c)^2/4 + b^2)

    # So the worst-case stdev(q) is

    #     sqrt((a+c)/2 + sqrt( (a-c)^2/4 + b^2))






    what_known = set(('covariance', 'worstdirection-stdev', 'rms-stdev'))
    if not what in what_known:
        raise Exception(f"'what' kwarg must be in {what_known}, but got '{what}'")


    lensmodel = model.intrinsics()[0]

    optimization_inputs = model.optimization_inputs()
    if optimization_inputs is None:
        raise Exception("optimization_inputs are unavailable in this model. Uncertainty cannot be computed")

    if not optimization_inputs.get('do_optimize_extrinsics'):
        raise Exception("Computing uncertainty if !do_optimize_extrinsics not supported currently. This is possible, but not implemented. _projection_uncertainty...() would need a path for fixed extrinsics like they already do for fixed frames")

    bpacked,x,Jpacked,factorization = \
        mrcal.optimizer_callback( **optimization_inputs )

    if factorization is None:
        raise Exception("Cannot compute the uncertainty: factorization computation failed")

    # The intrinsics,extrinsics,frames MUST come from the solve when
    # evaluating the uncertainties. The user is allowed to update the
    # extrinsics in the model after the solve, as long as I use the
    # solve-time ones for the uncertainty computation. Updating the
    # intrinsics invalidates the uncertainty stuff so I COULD grab those
    # from the model. But for good hygiene I get them from the solve as
    # well

    # which calibration-time camera we're looking at
    icam_intrinsics = model.icam_intrinsics()
    icam_extrinsics = mrcal.corresponding_icam_extrinsics(icam_intrinsics, **optimization_inputs)

    intrinsics_data   = optimization_inputs['intrinsics'][icam_intrinsics]

    if not optimization_inputs.get('do_optimize_intrinsics_core') and \
       not optimization_inputs.get('do_optimize_intrinsics_distortions'):
        istate_intrinsics          = None
        slice_optimized_intrinsics = None
    else:
        istate_intrinsics = mrcal.state_index_intrinsics(icam_intrinsics, **optimization_inputs)

        i0,i1 = None,None # everything by default

        has_core     = mrcal.lensmodel_metadata_and_config(lensmodel)['has_core']
        Ncore        = 4 if has_core else 0
        Ndistortions = mrcal.lensmodel_num_params(lensmodel) - Ncore

        if not optimization_inputs.get('do_optimize_intrinsics_core'):
            i0 = Ncore
        if not optimization_inputs.get('do_optimize_intrinsics_distortions'):
            i1 = -Ndistortions

        slice_optimized_intrinsics  = slice(i0,i1)

    istate_frames = mrcal.state_index_frames(0, **optimization_inputs)

    if icam_extrinsics < 0:
        extrinsics_rt_fromref = None
        istate_extrinsics     = None
    else:
        extrinsics_rt_fromref = optimization_inputs['extrinsics_rt_fromref'][icam_extrinsics]
        istate_extrinsics     = mrcal.state_index_extrinsics (icam_extrinsics, **optimization_inputs)

    frames_rt_toref = None
    if optimization_inputs.get('do_optimize_frames'):
        frames_rt_toref = optimization_inputs.get('frames_rt_toref')


    Nmeasurements_observations = mrcal.num_measurements_boards(**optimization_inputs)
    if Nmeasurements_observations == mrcal.num_measurements(**optimization_inputs):
        # Note the special-case where I'm using all the observations
        Nmeasurements_observations = None

    if observed_pixel_uncertainty is None:
        observed_pixel_uncertainty = \
            np.std(mrcal.residuals_chessboard(optimization_inputs,
                                              residuals = x).ravel())

    # Two distinct paths here that are very similar, but different-enough to not
    # share any code. If atinfinity, I ignore all translations
    if not atinfinity:
        return \
            _projection_uncertainty(p_cam,
                                    lensmodel, intrinsics_data,
                                    extrinsics_rt_fromref, frames_rt_toref,
                                    factorization, Jpacked, optimization_inputs,
                                    istate_intrinsics, istate_extrinsics, istate_frames,
                                    slice_optimized_intrinsics,
                                    Nmeasurements_observations,
                                    observed_pixel_uncertainty,
                                    what)
    else:
        return \
            _projection_uncertainty_rotationonly(p_cam,
                                                 lensmodel, intrinsics_data,
                                                 extrinsics_rt_fromref, frames_rt_toref,
                                                 factorization, Jpacked, optimization_inputs,
                                                 istate_intrinsics, istate_extrinsics, istate_frames,
                                                 slice_optimized_intrinsics,
                                                 Nmeasurements_observations,
                                                 observed_pixel_uncertainty,
                                                 what)


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
http://mrcal.secretsauce.net/differencing.html

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
        if mrcal.lensmodel_metadata_and_config(lensmodels[i])['noncentral']:
            if not atinfinity:
                raise Exception(f"Model {i} are noncentral, so I can only evaluate the diff at infinity")
            import re
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
            print(f"WARNING: projection_diff() was asked to use uncertainties, but they aren't available/couldn't be computed. Falling back on the region-based-only logic\nException: {e}",
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

