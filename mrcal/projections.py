#!/usr/bin/python3

'''Routines to (un)project points using any camera model

Most of these are Python wrappers around the written-in-C Python extension
module mrcal._mrcal_npsp. Most of the time you want to use this module
instead of touching mrcal._mrcal_npsp directly.

All functions are exported into the mrcal module. So you can call these via
mrcal.projections.fff() or mrcal.fff(). The latter is preferred.

'''

import numpy as np
import numpysane as nps
import sys

import mrcal


def project(v, lensmodel, intrinsics_data,
            *,
            get_gradients = False,
            out           = None):
    r'''Projects a set of 3D camera-frame points to the imager

SYNOPSIS

    # v is a (...,3) array of 3D points we're projecting
    points = mrcal.project( v,
                            lensmodel, intrinsics_data )

    ### OR ###

    m = mrcal.cameramodel(...)
    points = mrcal.project( v, *m.intrinsics() )

    # points is a (...,2) array of pixel coordinates

Given a shape-(...,3) array of points in the camera frame (x,y aligned with the
imager coords, z 'forward') and an intrinsic model, this function computes the
projection, optionally with gradients.

Projecting out-of-bounds points (beyond the field of view) returns undefined
values. Generally things remain continuous even as we move off the imager
domain. Pinhole-like projections will work normally if projecting a point behind
the camera. Splined projections clamp to the nearest spline segment: the
projection will fly off to infinity quickly since we're extrapolating a
polynomial, but the function will remain continuous.

Broadcasting is fully supported across v and intrinsics_data

ARGUMENTS

- points: array of dims (...,3); the points we're projecting

- lensmodel: a string such as

  LENSMODEL_PINHOLE
  LENSMODEL_OPENCV4
  LENSMODEL_CAHVOR
  LENSMODEL_SPLINED_STEREOGRAPHIC_order=3_Nx=16_Ny=12_fov_x_deg=100

- intrinsics: array of dims (Nintrinsics):

    (focal_x, focal_y, center_pixel_x, center_pixel_y, distortion0, distortion1,
    ...)

  The focal lengths are given in pixels.

- get_gradients: optional boolean that defaults to False. Whether we should
  compute and report the gradients. This affects what we return

- out: optional argument specifying the destination. By default, new numpy
  array(s) are created and returned. To write the results into existing arrays,
  specify them with the 'out' kwarg. If not get_gradients: 'out' is the one
  numpy array we will write into. Else: 'out' is a tuple of all the output numpy
  arrays. If 'out' is given, we return the same arrays passed in. This is the
  standard behavior provided by numpysane_pywrap.

RETURNED VALUE

if not get_gradients:

  we return an (...,2) array of projected pixel coordinates

if get_gradients: we return a tuple:

  - (...,2) array of projected pixel coordinates
  - (...,2,3) array of gradients of the pixel coordinates in respect to the
    input 3D point positions
  - (...,2,Nintrinsics) array of the gradients of the pixel coordinates in
    respect to the intrinsics

    '''

    # Internal function must have a different argument order so
    # that all the broadcasting stuff is in the leading arguments
    if not get_gradients:
        return mrcal._mrcal_npsp._project(v, intrinsics_data, lensmodel=lensmodel, out=out)
    return mrcal._mrcal_npsp._project_withgrad(v, intrinsics_data, lensmodel=lensmodel, out=out)


def unproject(q, lensmodel, intrinsics_data,
              *,
              normalize     = False,
              get_gradients = False,
              out           = None):
    r'''Unprojects pixel coordinates to observation vectors

SYNOPSIS

    # q is a (...,2) array of pixel observations
    v = mrcal.unproject( q,
                         lensmodel, intrinsics_data )

    ### OR ###

    m = mrcal.cameramodel(...)
    v = mrcal.unproject( q, *m.intrinsics() )

Maps a set of 2D imager points q to a set of 3D vectors in camera coordinates
that produced these pixel observations. Each 3D vector is unique only
up-to-length, and the returned vectors aren't normalized by default. The default
length of the returned vector is arbitrary, and selected for the convenience of
the implementation. Pass normalize=True to always return unit vectors.

This is the "reverse" direction, so an iterative nonlinear optimization is
performed internally to compute this result. This is much slower than
mrcal_project. For OpenCV distortions specifically, OpenCV has
cvUndistortPoints() (and cv2.undistortPoints()), but these are inaccurate and we
do not use them: https://github.com/opencv/opencv/issues/8811

Gradients are available by passing get_gradients=True. Since unproject() is
implemented as an iterative solve around project(), the unproject() gradients
are computed by manipulating the gradients reported by project() at the
solution. The reported gradients are relative to whatever unproject() is
reporting; the unprojection is unique only up-to-length, and the magnitude isn't
fixed. So the gradients may include a component in the direction of the returned
observation vector: this follows the arbitrary scaling used by unproject(). It
is possible to pass normalize=True; we then return NORMALIZED observation
vectors and the gradients of those NORMALIZED vectors. In that case, those
gradients are guaranteed to be orthogonal to the observation vector. The vector
normalization involves a bit more computation, so it isn't the default.

NOTE: THE MAGNITUDE OF THE RETURNED VECTOR CHANGES IF get_gradients CHANGES. The
reported gradients are correct relative to the output returned with
get_gradients=True. Passing normalize=True can be used to smooth this out:

    unproject(..., normalize=True)

returns the same vectors as

    unproject(..., normalize=True, get_gradients=True)[0]

Broadcasting is fully supported across q and intrinsics_data.

Models that have no gradients available cannot use mrcal_unproject() in C, but
CAN still use this mrcal.unproject() Python routine: a slower routine is
employed that uses numerical differences instead of analytical gradients.

ARGUMENTS

- q: array of dims (...,2); the pixel coordinates we're unprojecting

- lensmodel: a string such as

  LENSMODEL_PINHOLE
  LENSMODEL_OPENCV4
  LENSMODEL_CAHVOR
  LENSMODEL_SPLINED_STEREOGRAPHIC_order=3_Nx=16_Ny=12_fov_x_deg=100

- intrinsics_data: array of dims (Nintrinsics):

    (focal_x, focal_y, center_pixel_x, center_pixel_y, distortion0, distortion1,
    ...)

  The focal lengths are given in pixels.

- normalize: optional boolean defaults to False. If True: normalize the output
  vectors

- get_gradients: optional boolean that defaults to False. Whether we should
  compute and report the gradients. This affects what we return (see below). If
  not normalize, the magnitude of the reported vectors changes if get_gradients
  is turned on/off (see above)

- out: optional argument specifying the destination. By default, new numpy
  array(s) are created and returned. To write the results into existing arrays,
  specify them with the 'out' kwarg. If not get_gradients: 'out' is the one
  numpy array we will write into. Else: 'out' is a tuple of all the output numpy
  arrays. If 'out' is given, we return the same arrays passed in. This is the
  standard behavior provided by numpysane_pywrap.

RETURNED VALUE

if not get_gradients:

  we return an (...,3) array of unprojected observation vectors. Not normalized
  by default; see description above

if get_gradients: we return a tuple:

  - (...,3) array of unprojected observation vectors
  - (...,3,2) array of gradients of unprojected observation vectors in respect
    to pixel coordinates
  - (...,3,Nintrinsics) array of gradients of unprojected observation vectors in
    respect to the intrinsics

    '''

    def apply_normalization_to_output_with_gradients(v,dv_dq,dv_di):
        # vn = v/mag(v)
        # dvn = dv (1/mag(v)) + v d(1/mag(v))
        #     = dv( 1/mag(v) - v vt / mag^3(v) )
        #     = dv( 1/mag(v) - vn vnt / mag(v) )
        #     = dv/mag(v) ( 1 - vn vnt )

        # v has shape (...,3)
        # dv_dq has shape (...,3,2)
        # dv_di has shape (...,3,N)

        # shape (...,1)
        magv_recip = 1. / nps.dummy(nps.mag(v), -1)
        v *= magv_recip

        # shape (...,1,1)
        magv_recip = nps.dummy(magv_recip,-1)
        dv_dq *= magv_recip

        dv_dq -= nps.xchg( nps.matmult( nps.dummy(nps.xchg(dv_dq, -1,-2), -2),
                                        nps.dummy(nps.outer(v,v),-3) )[...,0,:],
                           -1, -2)

        dv_di *= magv_recip

        dv_di -= nps.xchg( nps.matmult( nps.dummy(nps.xchg(dv_di, -1,-2), -2),
                                        nps.dummy(nps.outer(v,v),-3) )[...,0,:],
                           -1, -2)


    # First, handle some trivial cases. I don't want to run the
    # optimization-based unproject() if I don't have to
    if lensmodel == 'LENSMODEL_PINHOLE' or \
       lensmodel == 'LENSMODEL_LONLAT'  or \
       lensmodel == 'LENSMODEL_LATLON'  or \
       lensmodel == 'LENSMODEL_STEREOGRAPHIC':

        if   lensmodel == 'LENSMODEL_PINHOLE':
            func = mrcal.unproject_pinhole
            always_normalized = False
        elif lensmodel == 'LENSMODEL_LONLAT':
            func = mrcal.unproject_lonlat
            always_normalized = True
        elif lensmodel == 'LENSMODEL_LATLON':
            func = mrcal.unproject_latlon
            always_normalized = True
        elif lensmodel == 'LENSMODEL_STEREOGRAPHIC':
            func = mrcal.unproject_stereographic
            always_normalized = False


        if not get_gradients:

            v = func(q, intrinsics_data, out = out)
            if normalize and not always_normalized:
                v /= nps.dummy(nps.mag(v), axis=-1)
            return v


        # shapes (...,2)
        fxy = intrinsics_data[..., :2]
        cxy = intrinsics_data[..., 2:]

        # shapes (...,3) and (...,3,2)
        v, dv_dq = \
            func(q, intrinsics_data,
                 get_gradients = True,
                 out = None if out is None else (out[0],out[1]))

        # q = f l(v) + c
        # l(v) = (q-c)/f
        #
        # dl/dv dv/df = (c-q) / f^2
        # dl/dv dv/dq = 1/f
        # -> dl/dv = 1 / ( f dv/dq )
        # -> dv/df =  (c-q) / (f^2 dl/dv) = (c-q) dv/dq / f
        #
        # dl/dv dv/dc = -1/f
        # -> dv/dc =  -1 / (f dl/dv) = -1 / (f /( f dv/dq )) = -dv/dq
        dv_di_shape = dv_dq.shape[:-1] + (4,)
        if out is None:
            dv_di = np.zeros( dv_di_shape, dtype=float)
        else:
            if not (out[2].shape[-len(dv_di_shape):] == dv_di_shape and \
                    not any(np.array(out[2].shape[:-len(dv_di_shape)]) - 1)):
                raise Exception(f"Shape of out[2] doesn't match broadcasted shape for dv_di. Wanted {dv_di_shape}, but got {out[2].shape}")
            dv_di = out[2]
            dv_di *= 0

        # dv/df
        dv_di[..., :2] += nps.dummy((cxy - q)/fxy, -2) * dv_dq
        # dv/dc
        dv_di[..., 2:] -= dv_dq

        if normalize and not always_normalized:
            apply_normalization_to_output_with_gradients(v,dv_dq,dv_di)

        return v,dv_dq,dv_di



    try:
        meta = mrcal.lensmodel_metadata_and_config(lensmodel)
    except:
        raise Exception(f"Invalid lens model '{lensmodel}': couldn't get the metadata")
    if meta['has_gradients']:

        # Main path. We have gradients.
        #
        # Internal function must have a different argument order so
        # that all the broadcasting stuff is in the leading arguments
        if not get_gradients:
            v = mrcal._mrcal_npsp._unproject(q, intrinsics_data, lensmodel=lensmodel,
                                             out=out)
            if normalize:

                # Explicitly handle nan and inf to set their normalized values
                # to 0. Otherwise I get a scary-looking warning from numpy
                i_vgood = \
                    np.isfinite(v[...,0]) * \
                    np.isfinite(v[...,1]) * \
                    np.isfinite(v[...,2])
                v[~i_vgood] = np.array((0.,0.,1.))
                v /= nps.dummy(nps.mag(v), -1)
                v[~i_vgood] = np.array((0.,0.,0.))
            return v

        # We need to report gradients
        vs = mrcal._mrcal_npsp._unproject(q, intrinsics_data, lensmodel=lensmodel)

        # I have no gradients available for unproject(), and I need to invert a
        # non-square matrix to use the gradients from project(). I deal with this
        # with a stereographic mapping
        #
        # With a simple unprojection I have    q -> v
        # Instead I now do                     q -> vs -> u -> v

        # I reproject vs, to produce a scaled v = k*vs. I'm assuming all
        # projections are central, so vs represents q just as well as v does. u
        # is a 2-vector, so dq_du is (2x2), and I can invert it
        u = mrcal.project_stereographic(vs)
        dv_du = np.zeros( vs.shape + (2,), dtype=float)
        v, dv_du = \
            mrcal.unproject_stereographic(u,
                                          get_gradients = True,
                                          out = (vs if out is None else out[0],
                                                 dv_du))

        _,dq_dv,dq_di = mrcal.project(v,
                                      lensmodel, intrinsics_data,
                                      get_gradients = True)

        # shape (..., 2,2). Square. Invertible!
        dq_du = nps.matmult( dq_dv, dv_du )

        # dv/dq = dv/du du/dq =
        #       = dv/du inv(dq/du)
        #       = transpose(inv(transpose(dq/du)) transpose(dv/du))
        dv_dq = nps.transpose(np.linalg.solve( nps.transpose(dq_du),
                                               nps.transpose(dv_du) ))
        if out is not None:
            out[1] *= 0.
            out[1] += dv_dq
            dv_dq = out[1]


        # dv/di is a bit different. I have (q,i) -> v. I want to find out
        # how moving i affects v while keeping q constant. Taylor expansion
        # of projection: q = q0 + dq/dv dv + dq/di di. q is constant so
        # dq/dv dv + dq/di di = 0 -> dv/di = - dv/dq dq/di
        dv_di = nps.matmult(dv_dq, dq_di,
                            out = None if out is None else out[2])
        dv_di *= -1.

        if normalize:
            apply_normalization_to_output_with_gradients(v,dv_dq,dv_di)

        return v, dv_dq, dv_di




    # No projection gradients implemented in C. We should get here approximately
    # never. At this time, the only projection function that has no gradients
    # implemented is LENSMODEL_CAHVORE, which nobody is really expected to be
    # using. If these see use, real gradients should be implemented
    #
    # We compute the gradients numerically. This is a reimplementation of the C
    # code. It's barely maintained, and here for legacy compatibility only

    if get_gradients:
        raise Exception(f"unproject(..., get_gradients=True) is unsupported for models with no gradients, such as '{lensmodel}'")

    if q is None: return q
    if q.size == 0:
        s = q.shape
        return np.zeros(s[:-1] + (3,))

    if out is not None:
        raise Exception(f"unproject(..., out) is unsupported if out is not None and we're using a model with no gradients, such as '{lensmodel}'")

    fxy = intrinsics_data[...,  :2]
    cxy = intrinsics_data[..., 2:4]

    # undistort the q, by running an optimizer

    import scipy.optimize

    # I optimize each point separately because the internal optimization
    # algorithm doesn't know that each point is independent, so if I optimized
    # it all together, it would solve a dense linear system whose size is linear
    # in Npoints. The computation time thus would be much slower than
    # linear(Npoints)
    @nps.broadcast_define( ((2,),), )
    def undistort_this(q0):

        def cost_no_gradients(vxy, *args, **kwargs):
            '''Optimization functions'''
            return \
                mrcal.project(np.array((vxy[0],vxy[1],1.)), lensmodel, intrinsics_data) - \
                q0

        # seed assuming distortions aren't there
        vxy_seed = (q0 - cxy) / fxy

        # no gradients available
        result = scipy.optimize.least_squares(cost_no_gradients, vxy_seed,
                                              '3-point')

        vxy = result.x

        # This needs to be precise; if it isn't, I barf. Shouldn't happen
        # very often
        if np.sqrt(result.cost/2.) > 1e-3:
            if not unproject.__dict__.get('already_complained'):
                sys.stderr.write("WARNING: unproject() wasn't able to precisely compute some points. Returning nan for those. Will complain just once\n")
                unproject.already_complained = True
            return np.array((np.nan,np.nan))
        return vxy

    vxy = undistort_this(q)

    # I append a 1. shape = (..., 3)
    v = nps.glue(vxy, np.ones( vxy.shape[:-1] + (1,) ), axis=-1)
    if normalize:
        v /= nps.dummy(nps.mag(v), -1)
    return v


def project_pinhole(points,
                    fxycxy = np.array((1.0, 1.0, 0.0, 0.0), dtype=float),
                    *,
                    get_gradients = False,
                    out           = None):
    r'''Projects 3D camera-frame points using a pinhole projection

SYNOPSIS

    # points is a (N,3) array of camera-coordinate-system points
    q = mrcal.project_pinhole( points, fxycxy )

    # q is now a (N,2) array of pinhole coordinates

This is a special case of mrcal.project(). Useful to represent a very simple,
very perfect lens. Wide lenses do not follow this model. Long lenses usually
more-or-less DO follow this model. See the lensmodel documentation for details:

http://mrcal.secretsauce.net/lensmodels.html#lensmodel-pinhole

Given a (N,3) array of points in the camera frame (x,y aligned with the imager
coords, z 'forward') and the parameters fxycxy, this function computes the
projection, optionally with gradients.

ARGUMENTS

- points: array of dims (...,3); the points we're projecting. This supports
  broadcasting fully, and any leading dimensions are allowed, including none

- fxycxy: optional intrinsics core. This is a shape (4,) array (fx,fy,cx,cy),
  with all elements given in units of pixels. fx and fy are the horizontal and
  vertical focal lengths, respectively. (cx,cy) are pixel coordinates
  corresponding to the projection of p = [0,0,1]. If omitted, default values are
  used: fx=fy=1.0 and cx=cy=0.0.

- get_gradients: optional boolean, defaults to False. This affects what we
  return (see below)

- out: optional argument specifying the destination. By default, new numpy
  array(s) are created and returned. To write the results into existing arrays,
  specify them with the 'out' kwarg. If not get_gradients: 'out' is the one
  numpy array we will write into. Else: 'out' is a tuple of all the output numpy
  arrays. If 'out' is given, we return the same arrays passed in. This is the
  standard behavior provided by numpysane_pywrap.

RETURNED VALUE

if not get_gradients: we return an (...,2) array of projected transverse
equirectangular coordinates

if get_gradients: we return a tuple:

  - (...,2) array of projected pinhole coordinates
  - (...,2,3) array of the gradients of the transverse equirectangular
    coordinates in respect to the input 3D point positions

    '''

    # Internal function must have a different argument order so
    # that all the broadcasting stuff is in the leading arguments
    if not get_gradients:
        return mrcal._mrcal_npsp._project_pinhole(points, fxycxy,
                                                  out=out)
    return mrcal._mrcal_npsp._project_pinhole_withgrad(points, fxycxy,
                                                       out=out)


def unproject_pinhole(points,
                      fxycxy = np.array((1.0, 1.0, 0.0, 0.0), dtype=float),
                      *,
                      get_gradients = False,
                      out           = None):
    r'''Unprojects 2D pixel coordinates using a pinhole projection

SYNOPSIS

    # points is a (N,2) array of imager points
    v = mrcal.unproject_pinhole( points,
                                 fxycxy )

    # v is now a (N,3) array of observation directions in the camera coordinate
    # system. v are NOT normalized

This is a special case of mrcal.unproject(). Useful to represent a very simple,
very perfect lens. Wide lenses do not follow this model. Long lenses usually
more-or-less DO follow this model. See the lensmodel documentation for details:

http://mrcal.secretsauce.net/lensmodels.html#lensmodel-pinhole

Given a (N,2) array of pinhole coordinates and the parameters fxycxy, this
function computes the inverse projection, optionally with gradients.

The vectors returned by this function are NOT normalized.

ARGUMENTS

- points: array of dims (...,2); the pinhole coordinates
  we're unprojecting. This supports broadcasting fully, and any leading
  dimensions are allowed, including none

- fxycxy: optional intrinsics core. This is a shape (4,) array (fx,fy,cx,cy),
  with all elements given in units of pixels. fx and fy are the horizontal and
  vertical focal lengths, respectively. (cx,cy) are pixel coordinates
  corresponding to the projection of p = [0,0,1]. If omitted, default values are
  used: fx=fy=1.0 and cx=cy=0.0.

- get_gradients: optional boolean, defaults to False. This affects what we
  return (see below)

- out: optional argument specifying the destination. By default, new numpy
  array(s) are created and returned. To write the results into existing arrays,
  specify them with the 'out' kwarg. If not get_gradients: 'out' is the one
  numpy array we will write into. Else: 'out' is a tuple of all the output numpy
  arrays. If 'out' is given, we return the same arrays passed in. This is the
  standard behavior provided by numpysane_pywrap.

RETURNED VALUE

if not get_gradients: we return an (...,3) array of unprojected observation
vectors. These are NOT normalized.

if get_gradients: we return a tuple:

  - (...,3) array of unprojected observation vectors. These are NOT normalized.
  - (...,3,2) array of the gradients of the observation vectors in respect to
    the input 2D pinhole coordinates

    '''
    if not get_gradients:
        return mrcal._mrcal_npsp._unproject_pinhole(points, fxycxy,
                                                    out=out)
    return mrcal._mrcal_npsp._unproject_pinhole_withgrad(points, fxycxy,
                                                         out=out)


def project_stereographic(points,
                          fxycxy = np.array((1.0, 1.0, 0.0, 0.0), dtype=float),
                          *,
                          get_gradients = False,
                          out           = None):
    r'''Projects a set of 3D camera-frame points using a stereographic model

SYNOPSIS

    # points is a (N,3) array of camera-coordinate-system points
    q = mrcal.project_stereographic( points )

    # q is now a (N,2) array of normalized stereographic coordinates

This is a special case of mrcal.project(). No actual lens ever follows this
model exactly, but this is useful as a baseline for other models. See the
lensmodel documentation for details:

http://mrcal.secretsauce.net/lensmodels.html#lensmodel-stereographic

Given a (N,3) array of points in the camera frame (x,y aligned with the imager
coords, z 'forward') and parameters of a perfect stereographic camera, this
function computes the projection, optionally with gradients.

The user can pass in focal length and center-pixel values. Or they can be
omitted to compute a "normalized" stereographic projection (fx = fy = 1, cx = cy
= 0).

The stereographic projection is able to represent points behind the camera, and
has only one singular observation direction: directly behind the camera, along
the optical axis.

This projection acts radially. If the observation vector v makes an angle theta
with the optical axis, then the projected point q is 2 tan(theta/2) f from the
image center.

ARGUMENTS

- points: array of dims (...,3); the points we're projecting. This supports
  broadcasting fully, and any leading dimensions are allowed, including none

- fxycxy: optional intrinsics core. This is a shape (4,) array (fx,fy,cx,cy),
  with all elements given in units of pixels. fx and fy are the horizontal and
  vertical focal lengths, respectively. (cx,cy) are pixel coordinates
  corresponding to the projection of p = [0,0,1]. If omitted, default values are
  used to specify a normalized stereographic projection : fx=fy=1.0 and
  cx=cy=0.0.

- get_gradients: optional boolean, defaults to False. This affects what we
  return (see below)

- out: optional argument specifying the destination. By default, new numpy
  array(s) are created and returned. To write the results into existing arrays,
  specify them with the 'out' kwarg. If not get_gradients: 'out' is the one
  numpy array we will write into. Else: 'out' is a tuple of all the output numpy
  arrays. If 'out' is given, we return the same arrays passed in. This is the
  standard behavior provided by numpysane_pywrap.

RETURNED VALUE

if not get_gradients: we return an (...,2) array of projected stereographic
coordinates

if get_gradients: we return a tuple:

  - (...,2) array of projected stereographic coordinates
  - (...,2,3) array of the gradients of the stereographic coordinates in respect
    to the input 3D point positions

    '''
    if not get_gradients:
        return mrcal._mrcal_npsp._project_stereographic(points, fxycxy,
                                                        out=out)
    return mrcal._mrcal_npsp._project_stereographic_withgrad(points, fxycxy,
                                                             out=out)


def unproject_stereographic(points,
                            fxycxy = np.array((1.0, 1.0, 0.0, 0.0), dtype=float),
                            *,
                            get_gradients = False,
                            out           = None):
    r'''Unprojects a set of 2D pixel coordinates using a stereographic model

SYNOPSIS

    # points is a (N,2) array of pixel coordinates
    v = mrcal.unproject_stereographic( points, fxycxy)

    # v is now a (N,3) array of observation directions in the camera coordinate
    # system. v are NOT normalized

This is a special case of mrcal.unproject(). No actual lens ever follows this
model exactly, but this is useful as a baseline for other models. See the
lensmodel documentation for details:

http://mrcal.secretsauce.net/lensmodels.html#lensmodel-stereographic

Given a (N,2) array of stereographic coordinates and parameters of a perfect
stereographic camera, this function computes the inverse projection, optionally
with gradients.

The user can pass in focal length and center-pixel values. Or they can be
omitted to compute a "normalized" stereographic projection (fx = fy = 1, cx = cy
= 0).

The stereographic projection is able to represent points behind the camera, and
has only one singular observation direction: directly behind the camera, along
the optical axis.

This projection acts radially. If the observation vector v makes an angle theta
with the optical axis, then the projected point q is 2 tan(theta/2) f from the
image center.

ARGUMENTS

- points: array of dims (...,2); the stereographic coordinates we're
  unprojecting. This supports broadcasting fully, and any leading dimensions are
  allowed, including none

- fxycxy: optional intrinsics core. This is a shape (4,) array (fx,fy,cx,cy),
  with all elements given in units of pixels. fx and fy are the horizontal and
  vertical focal lengths, respectively. (cx,cy) are pixel coordinates
  corresponding to the projection of p = [0,0,1]. If omitted, default values are
  used to specify a normalized stereographic projection : fx=fy=1.0 and
  cx=cy=0.0.

- get_gradients: optional boolean, defaults to False. This affects what we
  return (see below)

- out: optional argument specifying the destination. By default, new numpy
  array(s) are created and returned. To write the results into existing arrays,
  specify them with the 'out' kwarg. If not get_gradients: 'out' is the one
  numpy array we will write into. Else: 'out' is a tuple of all the output numpy
  arrays. If 'out' is given, we return the same arrays passed in. This is the
  standard behavior provided by numpysane_pywrap.

RETURNED VALUE

if not get_gradients: we return an (...,3) array of unprojected observation
vectors. These are NOT normalized.

if get_gradients: we return a tuple:

  - (...,3) array of unprojected observation vectors. These are NOT normalized.
  - (...,3,2) array of the gradients of the observation vectors in respect to
    the input 2D stereographic coordinates

    '''
    if not get_gradients:
        return mrcal._mrcal_npsp._unproject_stereographic(points, fxycxy,
                                                          out=out)
    return mrcal._mrcal_npsp._unproject_stereographic_withgrad(points, fxycxy,
                                                               out=out)


def project_lonlat(points,
                   fxycxy = np.array((1.0, 1.0, 0.0, 0.0), dtype=float),
                   *,
                   get_gradients = False,
                   out           = None):
    r'''Projects a set of 3D camera-frame points using an equirectangular projection

SYNOPSIS

    # points is a (N,3) array of camera-coordinate-system points
    q = mrcal.project_lonlat( points, fxycxy )

    # q is now a (N,2) array of equirectangular coordinates

This is a special case of mrcal.project(). Useful not for
representing lenses, but for describing the projection function of wide
panoramic images. Lenses do not follow this model. See the lensmodel
documentation for details:

http://mrcal.secretsauce.net/lensmodels.html#lensmodel-lonlat

Given a (N,3) array of points in the camera frame (x,y aligned with the imager
coords, z 'forward') and the parameters fxycxy, this function computes the
projection, optionally with gradients.

ARGUMENTS

- points: array of dims (...,3); the points we're projecting. This supports
  broadcasting fully, and any leading dimensions are allowed, including none

- fxycxy: optional intrinsics core. This is a shape (4,) array (fx,fy,cx,cy). fx
  and fy are the "focal lengths": they specify the angular resolution of the
  image, in pixels/radian. (cx,cy) are pixel coordinates corresponding to the
  projection of p = [0,0,1]. If omitted, default values are used to specify a
  normalized equirectangular projection : fx=fy=1.0 and cx=cy=0.0. This produces
  q = (lon,lat)

- get_gradients: optional boolean, defaults to False. This affects what we
  return (see below)

- out: optional argument specifying the destination. By default, new numpy
  array(s) are created and returned. To write the results into existing arrays,
  specify them with the 'out' kwarg. If not get_gradients: 'out' is the one
  numpy array we will write into. Else: 'out' is a tuple of all the output numpy
  arrays. If 'out' is given, we return the same arrays passed in. This is the
  standard behavior provided by numpysane_pywrap.

RETURNED VALUE

if not get_gradients: we return an (...,2) array of projected equirectangular
coordinates

if get_gradients: we return a tuple:

  - (...,2) array of projected equirectangular coordinates
  - (...,2,3) array of the gradients of the equirectangular coordinates in respect
    to the input 3D point positions

    '''

    # Internal function must have a different argument order so
    # that all the broadcasting stuff is in the leading arguments
    if not get_gradients:
        return mrcal._mrcal_npsp._project_lonlat(points, fxycxy,
                                                 out=out)
    return mrcal._mrcal_npsp._project_lonlat_withgrad(points, fxycxy,
                                                      out=out)


def unproject_lonlat(points,
                     fxycxy = np.array((1.0, 1.0, 0.0, 0.0), dtype=float),
                     *,
                     get_gradients = False,
                     out           = None):
    r'''Unprojects a set of 2D pixel coordinates using an equirectangular projection

SYNOPSIS

    # points is a (N,2) array of imager points
    v = mrcal.unproject_lonlat( points, fxycxy )

    # v is now a (N,3) array of observation directions in the camera coordinate
    # system. v are normalized

This is a special case of mrcal.unproject(). Useful not for
representing lenses, but for describing the projection function of wide
panoramic images. Lenses do not follow this model. See the lensmodel
documentation for details:

http://mrcal.secretsauce.net/lensmodels.html#lensmodel-lonlat

Given a (N,2) array of equirectangular coordinates and the parameters fxycxy,
this function computes the inverse projection, optionally with gradients.

The vectors returned by this function are normalized.

ARGUMENTS

- points: array of dims (...,2); the equirectangular coordinates we're
  unprojecting. This supports broadcasting fully, and any leading dimensions are
  allowed, including none

- fxycxy: optional intrinsics core. This is a shape (4,) array (fx,fy,cx,cy). fx
  and fy are the "focal lengths": they specify the angular resolution of the
  image, in pixels/radian. (cx,cy) are pixel coordinates corresponding to the
  projection of p = [0,0,1]. If omitted, default values are used to specify a
  normalized equirectangular projection : fx=fy=1.0 and cx=cy=0.0. This produces
  q = (lon,lat)

- get_gradients: optional boolean, defaults to False. This affects what we
  return (see below)

- out: optional argument specifying the destination. By default, new numpy
  array(s) are created and returned. To write the results into existing arrays,
  specify them with the 'out' kwarg. If not get_gradients: 'out' is the one
  numpy array we will write into. Else: 'out' is a tuple of all the output numpy
  arrays. If 'out' is given, we return the same arrays passed in. This is the
  standard behavior provided by numpysane_pywrap.

RETURNED VALUE

if not get_gradients: we return an (...,3) array of unprojected observation
vectors. These are normalized.

if get_gradients: we return a tuple:

  - (...,3) array of unprojected observation vectors. These are normalized.
  - (...,3,2) array of the gradients of the observation vectors in respect to
    the input 2D equirectangular coordinates

    '''
    if not get_gradients:
        return mrcal._mrcal_npsp._unproject_lonlat(points, fxycxy,
                                                   out=out)
    return mrcal._mrcal_npsp._unproject_lonlat_withgrad(points, fxycxy,
                                                        out=out)


def project_latlon(points,
                   fxycxy = np.array((1.0, 1.0, 0.0, 0.0), dtype=float),
                   *,
                   get_gradients = False,
                   out           = None):
    r'''Projects 3D camera-frame points using a transverse equirectangular projection

SYNOPSIS

    # points is a (N,3) array of camera-coordinate-system points
    q = mrcal.project_latlon( points, fxycxy )

    # q is now a (N,2) array of transverse equirectangular coordinates

This is a special case of mrcal.project(). Useful not for representing lenses,
but for performing stereo rectification. Lenses do not follow this model. See
the lensmodel documentation for details:

http://mrcal.secretsauce.net/lensmodels.html#lensmodel-latlon

Given a (N,3) array of points in the camera frame (x,y aligned with the imager
coords, z 'forward') and the parameters fxycxy, this function computes the
projection, optionally with gradients.

ARGUMENTS

- points: array of dims (...,3); the points we're projecting. This supports
  broadcasting fully, and any leading dimensions are allowed, including none

- fxycxy: optional intrinsics core. This is a shape (4,) array (fx,fy,cx,cy). fx
  and fy are the "focal lengths": they specify the angular resolution of the
  image, in pixels/radian. (cx,cy) are pixel coordinates corresponding to the
  projection of p = [0,0,1]. If omitted, default values are used to specify a
  normalized transverse equirectangular projection : fx=fy=1.0 and cx=cy=0.0.
  This produces q = (lat,lon)

- get_gradients: optional boolean, defaults to False. This affects what we
  return (see below)

- out: optional argument specifying the destination. By default, new numpy
  array(s) are created and returned. To write the results into existing arrays,
  specify them with the 'out' kwarg. If not get_gradients: 'out' is the one
  numpy array we will write into. Else: 'out' is a tuple of all the output numpy
  arrays. If 'out' is given, we return the same arrays passed in. This is the
  standard behavior provided by numpysane_pywrap.

RETURNED VALUE

if not get_gradients: we return an (...,2) array of projected transverse
equirectangular coordinates

if get_gradients: we return a tuple:

  - (...,2) array of projected transverse equirectangular coordinates
  - (...,2,3) array of the gradients of the transverse equirectangular
    coordinates in respect to the input 3D point positions

    '''

    # Internal function must have a different argument order so
    # that all the broadcasting stuff is in the leading arguments
    if not get_gradients:
        return mrcal._mrcal_npsp._project_latlon(points, fxycxy,
                                                 out=out)
    return mrcal._mrcal_npsp._project_latlon_withgrad(points, fxycxy,
                                                      out=out)


def unproject_latlon(points,
                     fxycxy = np.array((1.0, 1.0, 0.0, 0.0), dtype=float),
                     *,
                     get_gradients = False,
                     out           = None):
    r'''Unprojects 2D pixel coordinates using a transverse equirectangular projection

SYNOPSIS

    # points is a (N,2) array of imager points
    v = mrcal.unproject_latlon( points, fxycxy )

    # v is now a (N,3) array of observation directions in the camera coordinate
    # system. v are normalized

This is a special case of mrcal.unproject(). Useful not for representing lenses,
but for performing stereo rectification. Lenses do not follow this model. See
the lensmodel documentation for details:

http://mrcal.secretsauce.net/lensmodels.html#lensmodel-latlon

Given a (N,2) array of transverse equirectangular coordinates and the parameters
fxycxy, this function computes the inverse projection, optionally with
gradients.

The vectors returned by this function are normalized.

ARGUMENTS

- points: array of dims (...,2); the transverse equirectangular coordinates
  we're unprojecting. This supports broadcasting fully, and any leading
  dimensions are allowed, including none

- fxycxy: optional intrinsics core. This is a shape (4,) array (fx,fy,cx,cy),
  with all elements given in units of pixels. fx and fy are the horizontal and
  vertical focal lengths, respectively. (cx,cy) are pixel coordinates
  corresponding to the projection of p = [0,0,1]. If omitted, default values are
  used to specify a normalized transverse equirectangular projection : fx=fy=1.0
  and cx=cy=0.0. This produces q = (lat,lon)

- get_gradients: optional boolean, defaults to False. This affects what we
  return (see below)

- out: optional argument specifying the destination. By default, new numpy
  array(s) are created and returned. To write the results into existing arrays,
  specify them with the 'out' kwarg. If not get_gradients: 'out' is the one
  numpy array we will write into. Else: 'out' is a tuple of all the output numpy
  arrays. If 'out' is given, we return the same arrays passed in. This is the
  standard behavior provided by numpysane_pywrap.

RETURNED VALUE

if not get_gradients: we return an (...,3) array of unprojected observation
vectors. These are normalized.

if get_gradients: we return a tuple:

  - (...,3) array of unprojected observation vectors. These are normalized.
  - (...,3,2) array of the gradients of the observation vectors in respect to
    the input 2D transverse equirectangular coordinates

    '''
    if not get_gradients:
        return mrcal._mrcal_npsp._unproject_latlon(points, fxycxy,
                                                   out=out)
    return mrcal._mrcal_npsp._unproject_latlon_withgrad(points, fxycxy,
                                                        out=out)
