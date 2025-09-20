#!/usr/bin/python3

# Copyright (c) 2017-2023 California Institute of Technology ("Caltech"). U.S.
# Government sponsorship acknowledged. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

'''General utility functions used throughout mrcal

All functions are exported into the mrcal module. So you can call these via
mrcal.utils.fff() or mrcal.fff(). The latter is preferred.

'''

import numpy as np
import numpysane as nps
import sys
import re

import mrcal


def _align_procrustes_points_Rt01_python(p0, p1, weights=None):
    r'''Reference implementation of align_procrustes_points_Rt01() in python

The main implementation is written in C in poseutils.c:

  mrcal_align_procrustes_points_Rt01()

The two implementations are identical, with a test to verify this

    '''
    if weights is None:
        weights = np.ones(p0.shape[:-1], dtype=float)
    return _align_procrustes_points_Rt01(p0,p1,weights)


@nps.broadcast_define( (('N',3,), ('N',3,), ('N',)),
                       (4,3), )
def _align_procrustes_points_Rt01(p0, p1, weights):

    p0 = nps.transpose(p0)
    p1 = nps.transpose(p1)

    # I process Mt instead of M to not need to transpose anything later, and to
    # end up with contiguous-memory results
    Mt = nps.matmult(              (p0 - np.mean(p0, axis=-1)[..., np.newaxis])*weights,
                      nps.transpose(p1 - np.mean(p1, axis=-1)[..., np.newaxis]))
    V,S,Ut = np.linalg.svd(Mt)

    # I look at the second-lowest singular value. One 0 singular value is OK;
    # two isn't
    if S[-2] < 1e-12:
        # Poorly-defined problem. Return error
        return np.zeros((4,3), dtype=float)

    R = nps.matmult(V, Ut)

    # det(R) is now +1 or -1. If it's -1, then this contains a mirror, and thus
    # is not a physical rotation. I compensate by negating the least-important
    # pair of singular vectors
    if np.linalg.det(R) < 0:
        V[:,2] *= -1
        R = nps.matmult(V, Ut)

    # Now that I have my optimal R, I compute the optimal t. From before:
    #
    #   t = mean(a) - R mean(b)
    t = np.mean(p0, axis=-1)[..., np.newaxis] - nps.matmult( R, np.mean(p1, axis=-1)[..., np.newaxis] )

    return nps.glue( R, t.ravel(), axis=-2)


def _align_procrustes_vectors_R01_python(v0, v1, weights=None):
    r'''Reference implementation of align_procrustes_vectors_R01() in python

The main implementation is written in C in poseutils.c:

  mrcal_align_procrustes_vectors_R01()

The two implementations are identical, with a test to verify this

    '''
    if weights is None:
        weights = np.ones(v0.shape[:-1], dtype=float)
    return _align_procrustes_vectors_R01(v0,v1,weights)

@nps.broadcast_define( (('N',3,), ('N',3,), ('N',)),
                       (3,3), )
def _align_procrustes_vectors_R01(v0, v1, weights):

    v0 = nps.transpose(v0)
    v1 = nps.transpose(v1)

    # I process Mt instead of M to not need to transpose anything later, and to
    # end up with contiguous-memory results
    Mt = nps.matmult( v0*weights, nps.transpose(v1) )
    V,S,Ut = np.linalg.svd(Mt)

    # I look at the second-lowest singular value. One 0 singular value is OK;
    # two isn't
    if S[-2] < 1e-12:
        # Poorly-defined problem. Return error
        return np.zeros((3,3), dtype=float)

    R = nps.matmult(V, Ut)

    # det(R) is now +1 or -1. If it's -1, then this contains a mirror, and thus
    # is not a physical rotation. I compensate by negating the least-important
    # pair of singular vectors
    if np.linalg.det(R) < 0:
        V[:,2] *= -1
        R = nps.matmult(V, Ut)

    return R


def align_procrustes_points_Rt01(p0, p1, weights=None):
    r"""Compute a rigid transformation to align two point clouds

SYNOPSIS

    print(points0.shape)
    ===>
    (100,3)

    print(points1.shape)
    ===>
    (100,3)

    Rt01 = mrcal.align_procrustes_points_Rt01(points0, points1)

    print( np.sum(nps.norm2(mrcal.transform_point_Rt(Rt01, points1) -
                            points0)) )
    ===>
    [The fit error from applying the optimal transformation. If the two point
     clouds match up, this will be small]

Given two sets of 3D points in numpy arrays of shape (N,3), we find the optimal
rotation, translation to align these sets of points. This is done with a
well-known direct method. See:

- https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
- https://en.wikipedia.org/wiki/Kabsch_algorithm

We return a transformation that minimizes the sum 2-norm of the misalignment:

    cost = sum( norm2( w[i] (a[i] - transform(b[i])) ))

We return an Rt transformation to map points in set 1 to points in set 0.

At least 3 sets of points must be passed-in to produce a non-singular result. If
a set of inputs produces a singular result, a (4,3) array of 0 is returned. This
is not a valid Rt transform, and is used to signal an error.

A similar computation can be performed to instead align a set of UNIT VECTORS to
compute an optimal rotation matrix R by calling align_procrustes_vectors_R01().

ARGUMENTS

- p0: an array of shape (..., N, 3). Each row is a point in the coordinate
  system we're transforming TO

- p1: an array of shape (..., N, 3). Each row is a point in the coordinate
  system we're transforming FROM

- weights: optional array of shape (..., N). Specifies the relative weight of
  each point. If omitted, all the given points are weighted equally

RETURNED VALUES

The Rt transformation in an array of shape (4,3). We return the optimal
transformation to align the given point clouds. The transformation maps points
TO coord system 0 FROM coord system 1.

If the inputs were singular or insufficient, a (4,3) array of 0 is returned.
This is not a valid Rt transform, and is used to signal an error.

    """
    # Use the python implementation when the data isn't contiguous
    if not (p0.flags['C_CONTIGUOUS'] and \
            p1.flags['C_CONTIGUOUS']):
        return _align_procrustes_points_Rt01_python(p0, p1, weights=weights)

    if weights is None:
        return mrcal._poseutils_npsp._align_procrustes_points_Rt01_noweights(p0, p1)
    else:
        return mrcal._poseutils_npsp._align_procrustes_points_Rt01_weights(p0, p1, weights)

def align_procrustes_vectors_R01(v0, v1, weights=None):
    r"""Compute a rotation to align two sets of direction vectors

SYNOPSIS

    print(vectors0.shape)
    ===>
    (100,3)

    print(vectors1.shape)
    ===>
    (100,3)

    R01 = mrcal.align_procrustes_vectors_R01(vectors0, vectors1)

    print( np.mean(1. - nps.inner(mrcal.rotate_point_R(R01, vectors1),
                                  vectors0)) )
    ===>
    [The fit error from applying the optimal rotation. If the two sets of
     vectors match up, this will be small]

Given two sets of normalized direction vectors in 3D (stored in numpy arrays of
shape (N,3)), we find the optimal rotation to align them. This is done with a
well-known direct method. See:

- https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
- https://en.wikipedia.org/wiki/Kabsch_algorithm

We return a rotation that minimizes the weighted sum of the cosine of the
misalignment:

    cost = -sum( w[i] inner(a[i], rotate(b[i])) )

We return a rotation to map vectors in set 1 to vectors in set 0.

At least 2 sets of vectors must be passed-in to produce a non-singular result.
If a set of inputs produces a singular result, a (3,3) array of 0 is returned.
This is not a valid rotation, and is used to signal an error.

A similar computation can be performed to instead align a set of POINTS to
compute an optimal transformation Rt by calling align_procrustes_points_Rt01().

ARGUMENTS

- v0: an array of shape (..., N, 3). Each row is a vector in the coordinate
  system we're transforming TO. These are assumed normalized.

- v1: an array of shape (..., N, 3). Each row is a vector in the coordinate
  system we're transforming FROM. These are assumed normalized.

- weights: optional array of shape (..., N). Specifies the relative weight of
  each vector. If omitted, everything is weighted equally

RETURNED VALUES

The rotation matrix in an array of shape (3,3). We return the optimal rotation
to align the given vector sets. The rotation maps vectors TO coord system 0 FROM
coord system 1.

If the inputs were singular or insufficient, a (3,3) array of 0 is returned.
This is not a valid rotation, and is used to signal an error.

    """

    # Use the python implementation when the data isn't contiguous
    if not (v0.flags['C_CONTIGUOUS'] and \
            v1.flags['C_CONTIGUOUS']):
        return _align_procrustes_vectors_R01_python(v0, v1, weights=weights)

    if weights is None:
        return mrcal._poseutils_npsp._align_procrustes_vectors_R01_noweights(v0, v1)
    else:
        return mrcal._poseutils_npsp._align_procrustes_vectors_R01_weights(v0, v1, weights)


def sample_imager(gridn_width, gridn_height, imager_width, imager_height):
    r'''Returns regularly-sampled, gridded pixels coordinates across the imager

SYNOPSIS

    q = sample_imager( 60, 40, *model.imagersize() )

    print(q.shape)
    ===>
    (40,60,2)

Note that the arguments are given in width,height order, as is customary when
generally talking about images and indexing. However, the output is in
height,width order, as is customary when talking about matrices and numpy
arrays.

If we ask for gridding dimensions (gridn_width, gridn_height), the output has
shape (gridn_height,gridn_width,2) where each row is an (x,y) pixel coordinate.

The top-left corner is at [0,0,:]:

    sample_imager(...)[0,0] = [0,0]

The the bottom-right corner is at [-1,-1,:]:

     sample_imager(...)[            -1,           -1,:] =
     sample_imager(...)[gridn_height-1,gridn_width-1,:] =
     (imager_width-1,imager_height-1)

When making plots you probably want to call mrcal.imagergrid_using(). See the
that docstring for details.

ARGUMENTS

- gridn_width: how many points along the horizontal gridding dimension

- gridn_height: how many points along the vertical gridding dimension. If None,
  we compute an integer gridn_height to maintain a square-ish grid:
  gridn_height/gridn_width ~ imager_height/imager_width

- imager_width,imager_height: the width, height of the imager. With a
  mrcal.cameramodel object this is *model.imagersize()

RETURNED VALUES

We return an array of shape (gridn_height,gridn_width,2). Each row is an (x,y)
pixel coordinate.

    '''

    if gridn_height is None:
        gridn_height = int(round(imager_height/imager_width*gridn_width))

    w = np.linspace(0,imager_width -1,gridn_width)
    h = np.linspace(0,imager_height-1,gridn_height)
    return np.ascontiguousarray(nps.mv(nps.cat(*np.meshgrid(w,h)),
                                       0,-1))


def sample_imager_unproject(gridn_width,  gridn_height,
                            imager_width, imager_height,
                            lensmodel, intrinsics_data,
                            normalize = False):
    r'''Reports 3D observation vectors that regularly sample the imager

SYNOPSIS

    import gnuplotlib as gp
    import mrcal

    ...

    Nwidth  = 60
    Nheight = 40

    # shape (Nheight,Nwidth,3)
    v,q = \
        mrcal.sample_imager_unproject(Nw, Nh,
                                      *model.imagersize(),
                                      *model.intrinsics())

    # shape (Nheight,Nwidth)
    f = interesting_quantity(v)

    gp.plot(f,
            tuplesize = 3,
            ascii     = True,
            using     = mrcal.imagergrid_using(model.imagersize, Nw, Nh),
            square    = True,
            _with     = 'image')

This is a utility function used by functions that evalute some interesting
quantity for various locations across the imager. Grid dimensions and lens
parameters are passed in, and the grid points and corresponding unprojected
vectors are returned. The unprojected vectors are unique only up-to-length, and
the returned vectors aren't normalized by default. If we want them to be
normalized, pass normalize=True.

This function has two modes of operation:

- One camera. lensmodel is a string, and intrinsics_data is a 1-dimensions numpy
  array. With a mrcal.cameramodel object together these are *model.intrinsics().
  We return (v,q) where v is a shape (Nheight,Nwidth,3) array of observation
  vectors, and q is a (Nheight,Nwidth,2) array of corresponding pixel
  coordinates (the grid returned by sample_imager())

- Multiple cameras. lensmodel is a list or tuple of strings; intrinsics_data is
  an iterable of 1-dimensional numpy arrays (a list/tuple or a 2D array). We
  return the same q as before (only one camera is gridded), but the unprojected
  array v has shape (Ncameras,Nheight,Nwidth,3) where Ncameras is the leading
  dimension of lensmodel. The gridded imager appears in camera0: v[0,...] =
  unproject(q)

ARGUMENTS

- gridn_width: how many points along the horizontal gridding dimension

- gridn_height: how many points along the vertical gridding dimension. If None,
  we compute an integer gridn_height to maintain a square-ish grid:
  gridn_height/gridn_width ~ imager_height/imager_width

- imager_width,imager_height: the width, height of the imager. With a
  mrcal.cameramodel object this is *model.imagersize()

- lensmodel, intrinsics_data: the lens parameters. With a single camera,
  lensmodel is a string, and intrinsics_data is a 1-dimensions numpy array; with
  a mrcal.cameramodel object together these are *model.intrinsics(). With
  multiple cameras, lensmodel is a list/tuple of strings. And intrinsics_data is
  an iterable of 1-dimensional numpy arrays (a list/tuple or a 2D array).

- normalize: optional boolean defaults to False. If True: normalize the output
  vectors

RETURNED VALUES

We return a tuple:

- v: the unprojected vectors. If we have a single camera this has shape
  (Nheight,Nwidth,3). With multiple cameras this has shape
  (Ncameras,Nheight,Nwidth,3). These are NOT normalized by default. To get
  normalized vectors, pass normalize=True

- q: the imager-sampling grid. This has shape (Nheight,Nwidth,2) regardless of
  how many cameras were given (we always sample just one camera). This is what
  sample_imager() returns

    '''

    def is_list_or_tuple(l):
        return isinstance(l,list) or isinstance(l,tuple)


    # shape: (Nheight,Nwidth,2). Contains (x,y) rows
    grid = sample_imager(gridn_width, gridn_height, imager_width, imager_height)

    if is_list_or_tuple(lensmodel):
        # shape: Ncameras,Nwidth,Nheight,3
        return np.array([mrcal.unproject(np.ascontiguousarray(grid),
                                         lensmodel[i],
                                         intrinsics_data[i],
                                         normalize = normalize) \
                         for i in range(len(lensmodel))]), \
               grid
    else:
        # shape: Nheight,Nwidth,3
        return \
            mrcal.unproject(np.ascontiguousarray(grid),
                            lensmodel, intrinsics_data,
                            normalize = normalize), \
            grid


def hypothesis_board_corner_positions(icam_intrinsics = None,
                                      idx_inliers     = None,
                                      **optimization_inputs):
    r'''Reports the 3D chessboard points observed by a camera at calibration time

SYNOPSIS

    model = mrcal.cameramodel("xxx.cameramodel")

    optimization_inputs = model.optimization_inputs()

    # shape (Nobservations, Nheight, Nwidth, 3)
    pcam = mrcal.hypothesis_board_corner_positions(**optimization_inputs)[0]

    i_intrinsics = \
      optimization_inputs['indices_frame_camintrinsics_camextrinsics'][:,1]

    # shape (Nobservations,1,1,Nintrinsics)
    intrinsics = nps.mv(optimization_inputs['intrinsics'][i_intrinsics],-2,-4)

    optimization_inputs['observations_board'][...,:2] = \
        mrcal.project( pcam,
                       optimization_inputs['lensmodel'],
                       intrinsics )

    # optimization_inputs now contains perfect, noiseless board observations

    x = mrcal.optimizer_callback(**optimization_inputs)[1]
    print(nps.norm2(x[:mrcal.num_measurements_boards(**optimization_inputs)]))
    ==>
    0

The optimization routine generates hypothetical observations from a set of
parameters being evaluated, trying to match these hypothetical observations to
real observations. To facilitate analysis, this routine returns these
hypothetical coordinates of the chessboard corners being observed. This routine
reports the 3D points in the coordinate system of the observing camera.

The hypothetical points are constructed from

- The calibration object geometry
- The calibration object-reference transformation in
  optimization_inputs['rt_ref_frame']
- The camera extrinsics (reference-camera transformation) in
  optimization_inputs['rt_cam_ref']
- The table selecting the camera and calibration object frame for each
  observation in
  optimization_inputs['indices_frame_camintrinsics_camextrinsics']

ARGUMENTS

- icam_intrinsics: optional integer specifying which intrinsic camera in the
  optimization_inputs we're looking at. If omitted (or None), we report
  camera-coordinate points for all the cameras

- idx_inliers: optional numpy array of booleans of shape
  (Nobservations,object_height,object_width) to select the outliers manually. If
  omitted (or None), the outliers are selected automatically: idx_inliers =
  observations_board[...,2] > 0. This argument is available to pick common
  inliers from two separate solves.

- **optimization_inputs: a dict() of arguments passable to mrcal.optimize() and
  mrcal.optimizer_callback(). We use the geometric data. This dict is obtainable
  from a cameramodel object by calling cameramodel.optimization_inputs()

RETURNED VALUE

- An array of shape (Nobservations, Nheight, Nwidth, 3) containing the
  coordinates (in the coordinate system of each camera) of the chessboard
  corners, for ALL the cameras. These correspond to the observations in
  optimization_inputs['observations_board'], which also have shape
  (Nobservations, Nheight, Nwidth, 3)

- An array of shape (Nobservations_thiscamera, Nheight, Nwidth, 3) containing
  the coordinates (in the camera coordinate system) of the chessboard corners,
  for the particular camera requested in icam_intrinsics. If icam_intrinsics is
  None: this is the same array as the previous returned value

- an (N,3) array containing camera-frame 3D points observed at calibration time,
  and accepted by the solver as inliers. This is a subset of the 2nd returned
  array.

- an (N,3) array containing camera-frame 3D points observed at calibration time,
  but rejected by the solver as outliers. This is a subset of the 2nd returned
  array.

    '''

    observations_board = optimization_inputs.get('observations_board')
    if observations_board is None:
        raise Exception("No board observations available")

    indices_frame_camintrinsics_camextrinsics = \
        optimization_inputs['indices_frame_camintrinsics_camextrinsics']

    object_width_n      = observations_board.shape[-2]
    object_height_n     = observations_board.shape[-3]
    object_spacing      = optimization_inputs['calibration_object_spacing']
    calobject_warp      = optimization_inputs.get('calobject_warp')
    # shape (Nh,Nw,3)
    full_object         = mrcal.ref_calibration_object(object_width_n,
                                                       object_height_n,
                                                       object_spacing,
                                                       calobject_warp = calobject_warp)
    Rt_ref_frame = \
        mrcal.Rt_from_rt( optimization_inputs['rt_ref_frame'] )\
        [ indices_frame_camintrinsics_camextrinsics[:,0] ]
    Rt_cam_ref = \
        nps.glue( mrcal.identity_Rt(),
                  mrcal.Rt_from_rt(optimization_inputs['rt_cam_ref']),
                  axis = -3 ) \
        [ indices_frame_camintrinsics_camextrinsics[:,2]+1 ]

    Rt_cam_frame = mrcal.compose_Rt( Rt_cam_ref,
                                     Rt_ref_frame )

    p_cam_calobjects = \
        mrcal.transform_point_Rt(nps.mv(Rt_cam_frame,-3,-5), full_object)

    # shape (Nobservations,Nheight,Nwidth)
    if idx_inliers is None:
        idx_inliers = observations_board[...,2] > 0
    idx_outliers = ~idx_inliers

    if icam_intrinsics is None:
        return                                       \
            p_cam_calobjects,                        \
            p_cam_calobjects,                        \
            p_cam_calobjects[idx_inliers,      ...], \
            p_cam_calobjects[idx_outliers,     ...]


    # The user asked for a specific camera. Separate out its data

    # shape (Nobservations,)
    idx_observations = indices_frame_camintrinsics_camextrinsics[:,1]==icam_intrinsics

    idx_inliers [~idx_observations] = False
    idx_outliers[~idx_observations] = False

    return                                       \
        p_cam_calobjects,                        \
        p_cam_calobjects[idx_observations, ...], \
        p_cam_calobjects[idx_inliers,      ...], \
        p_cam_calobjects[idx_outliers,     ...]


def _splined_stereographic_domain(lensmodel):
    r'''Return the stereographic domain for splined-stereographic lens models

SYNOPSIS

    model = mrcal.cameramodel(model_filename)

    lensmodel = model.intrinsics()[0]

    domain_contour = mrcal._splined_stereographic_domain(lensmodel)

Splined stereographic models are defined by a splined surface. This surface is
indexed by normalized stereographic-projected points. This surface is defined in
some finite area, and this function reports a piecewise linear contour reporting
this region.

This function only makes sense for splined stereographic models.

RETURNED VALUE

An array of shape (N,2) containing a contour representing the projection domain.

    '''

    if not re.match('LENSMODEL_SPLINED_STEREOGRAPHIC', lensmodel):
        raise Exception(f"This only makes sense with splined models. Input uses {lensmodel}")

    ux,uy = mrcal.knots_for_splined_models(lensmodel)
    # shape (Ny,Nx,2)
    u = np.ascontiguousarray(nps.mv(nps.cat(*np.meshgrid(ux,uy)), 0, -1))

    meta = mrcal.lensmodel_metadata_and_config(lensmodel)
    if meta['order'] == 2:
        # spline order is 3. The valid region is 1/2 segments inwards from the
        # outer contour
        return \
            nps.glue( (u[0,1:-2] + u[1,1:-2]) / 2.,
                      (u[0,-2] + u[1,-2] + u[0,-1] + u[1,-1]) / 4.,

                      (u[1:-2, -2] + u[1:-2, -1]) / 2.,
                      (u[-2,-2] + u[-1,-2] + u[-2,-1] + u[-1,-1]) / 4.,

                      (u[-2, -2:1:-1] + u[-1, -2:1:-1]) / 2.,
                      (u[-2, 1] + u[-1, 1] + u[-2, 0] + u[-1, 0]) / 4.,

                      (u[-2:0:-1, 0] +u[-2:0:-1, 1]) / 2.,
                      (u[0, 0] +u[0, 1] + u[1, 0] +u[1, 1]) / 4.,

                      (u[0,1] + u[1,1]) / 2.,
                      axis = -2 )

    elif meta['order'] == 3:
        # spline order is 3. The valid region is the outer contour, leaving one
        # knot out
        return \
            nps.glue( u[1,1:-2], u[1:-2, -2], u[-2, -2:1:-1], u[-2:0:-1, 1],
                      axis=-2 )
    else:
        raise Exception("I only support cubic (order==3) and quadratic (order==2) models")


def polygon_difference(positive, negative):
    r'''Return the difference of two closed polygons

SYNOPSIS

    import numpy as np
    import numpysane as nps
    import gnuplotlib as gp

    A = np.array(((-1,-1),( 1,-1),( 1, 1),(-1, 1),(-1,-1)))
    B = np.array(((-.1,-1.1),( .1,-1.1),( .1, 1.1),(-.1, 1.1),(-.1,-1.1)))

    diff = mrcal.polygon_difference(A, B)

    gp.plot( (A, dict(legend = 'A', _with = 'lines')),
             (B, dict(legend = 'B', _with = 'lines')),
             *[ ( r, dict( _with     = 'filledcurves closed fillcolor "red"',
                           legend    = 'difference'))
                for r in diff],
             tuplesize = -2,
             square    = True,
             wait      = True)

Given two polygons specified as a point sequence in arrays of shape (N,2) this
function computes the topological difference: all the regions contained in the
positive polygon, but missing in the negative polygon. The result could be
empty, or it could contain any number of disconnected polygons, so a list of
polygons is returned. Each of the constituent resulting polygons is guaranteed
to not have holes. If any holes are found when computing the difference, we cut
apart the resulting shape until no holes remain.

ARGUMENTS

- positive: a polygon specified by a sequence of points in an array of shape
  (N,2). The resulting difference describes regions contained inside the
  positive polygon

- negative: a polygon specified by a sequence of points in an array of shape
  (N,2). The resulting difference describes regions outside the negative polygon

RETURNED VALUE

A list of arrays of shape (N,2). Each array in the list describes a hole-free
polygon as a sequence of points. The difference is a union of all these
constituent polygons. This list could have 0 elements (empty difference) or N
element (difference consists of N separate polygons)

    '''

    from shapely.geometry import Polygon,MultiPolygon,GeometryCollection,LineString
    import shapely.ops


    diff = Polygon(positive).difference(Polygon(negative))
    if isinstance(diff, (MultiPolygon,GeometryCollection)):
        diff = list(diff.geoms)
    elif isinstance(diff, Polygon):
        diff = [diff]
    else:
        raise Exception(f"I only know how to deal with MultiPolygon or Polygon, but instead got type '{type(diff)}")

    def split_polygon_to_remove_holes(p):
        if not isinstance(p, Polygon):
            raise Exception(f"Expected a 'Polygon' type, but got {type(p)}")

        if not (p.interiors and len(p.interiors)):
            # No hole. Return the coords, if they exist
            try:
                coords = p.exterior.coords
                if len(coords) == 0:
                    return []
                return [np.array(coords)]
            except:
                return []

        # There's a hole! We need to split this polygon. I cut the polygon by a
        # line between the centroid and some vertex. Which one doesn't matter; I
        # keep trying until some cut works
        hole = p.interiors[0]
        for i in range(0,len(hole.coords)):

            l0 = np.array((hole.centroid))
            l1 = np.array((hole.coords[i]))
            l0,l1 = (l1 + 100*(l0-l1)),(l0 + 100*(l1-l0))
            line = LineString( (l0,l1) )

            s = shapely.ops.split(p, line)
            if len(s) > 1:
                # success. split into multiple pieces. I process each one
                # recursively, and I'm done. I return a flattened list
                return [subpiece for piece in s for subpiece in split_polygon_to_remove_holes(piece)]
            # Split didn't work. Try the next vertex

        print("WARNING: Couldn't split the region. Ignoring",
              file = sys.stderr)
        return []

    return \
        [subpiece for p in diff for subpiece in split_polygon_to_remove_holes(p)]


def _densify_polyline(p, spacing):
    r'''Returns the input polyline, but resampled more densely
    The input and output polylines are a numpy array of shape (N,2). The output
    is resampled such that each input point is hit, but each linear segment is
    also sampled with at least the given spacing

    '''

    if p is None or p.size == 0:
        return p

    p1 = np.array(p[0,:], dtype=p.dtype)

    for i in range(1,len(p)):
        a = p[i-1,:]
        b = p[i,  :]
        d = b-a
        l = nps.mag(d)

        # A hacky method of rounding up
        N = int(l/spacing - 1e-6 + 1.)

        for j in range(N):
            p1 = nps.glue(p1,
                          float(j+1) / N * d + a,
                          axis=-2)
    return p1


# mrcal.shellquote is either pipes.quote or shlex.quote, depending on
# python2/python3
try:
    import pipes
    shellquote = pipes.quote
except:
    # python3 puts this into a different module
    import shlex
    shellquote = shlex.quote

def mapping_file_framenocameraindex(*files_per_camera):
    r'''Parse image filenames to get the frame numbers

SYNOPSIS

    mapping = \
      mapping_file_framenocameraindex( ('img5-cam2.jpg', 'img6-cam2.jpg'),
                                       ('img6-cam3.jpg', 'img7-cam3.jpg'),)

    print(mapping)
    ===>
    { 'frame5-cam2.jpg': (5, 0),
      'frame6-cam2.jpg': (6, 0),
      'frame6-cam3.jpg': (6, 1),
      'frame7-cam3.jpg': (7, 1) }


Prior to this call we already applied a glob to some images, so we already know
which images belong to which camera. This function further classifies the images
to find the frame number of each image. This is done by looking at the filenames
of images in each camera, removing common prefixes and suffixes, and using the
central varying filename component as the frame number. This varying component
should be numeric. If it isn't and we have multiple cameras, then we barf. If it
isn't, but we only have one camera, we fallback on sequential frame numbers.

If we have just one image for a camera, I can't tell what is constant in the
filenames, so I return framenumber=0.

ARGUMENTS

- *files_per_camera: one argument per camera. Each argument is a list of strings
   of filenames of images observed by that camera

RETURNED VALUES

We return a dict from filenames to (framenumber, cameraindex) tuples. The
"cameraindex" is a sequential index counting up from 0. cameraindex==0
corresponds to files_per_camera[0] and so on.

The "framenumber" may not be sequential OR starting from 0: this comes directly
from the filename.

    '''

    i_empty = [i for i in range(len(files_per_camera)) if len(files_per_camera[i]) == 0]
    if len(i_empty) > 0:
        raise Exception("These camera globs matched no files: {}".format(i_empty))


    def get_longest_leading_trailing_substrings(strings):
        r'''Given a list of strings, returns the length of the longest leading and
        trailing substring common to all the strings

        Main use case is to take in strings such as

          a/b/c/frame001.png
          a/b/c/frame002.png
          a/b/c/frame003.png

        and return ("a/b/c/frame00", ".png")

        '''

        # These feel inefficient, especially being written in python. There's
        # probably some built-in primitive I'm not seeing
        def longest_leading_substring(a,b):
            for i in range(len(a)):
                if i >= len(b) or a[i] != b[i]:
                    return a[:i]
            return a
        def longest_trailing_substring(a,b):
            for i in range(len(a)):
                if i >= len(b) or a[-i-1] != b[-i-1]:
                    if i == 0:
                        return ''
                    return a[-i:]
            return a

        if not strings:
            return (None,None)

        leading  = strings[0]
        trailing = strings[0]

        for s in strings[1:]:
            leading  = longest_leading_substring (leading,s)
            trailing = longest_trailing_substring(trailing,s)
        return leading,trailing

    def pull_framenumbers(files):

        if len(files) == 1:
            # special case where only one file is given. In this case I can't
            # tell where the frame number is, but I don't really care. I just
            # say that the frame number is 0
            return [0]

        leading,trailing = get_longest_leading_trailing_substrings(files)
        Nleading  = len(leading)
        Ntrailing = len(trailing)

        # I now have leading and trailing substrings. I make sure that all the stuff
        # between the leading and trailing strings is numeric

        # needed because I want s[i:-0] to mean s[i:], but that doesn't work, but
        # s[i:None] does
        Itrailing = -Ntrailing if Ntrailing > 0 else None
        for f in files:
            if not re.match("^[0-9]+$", f[Nleading:Itrailing]):
                raise Exception(("Image filenames MUST be of the form 'something..number..something'\n" +   \
                                 "where the somethings are common to all the filenames. File '{}'\n" + \
                                 "has a non-numeric middle: '{}'. The somethings are: '{}' and '{}'\n" + \
                                 "Did you forget to pass globs for each camera separately?"). \
                                format(f, f[Nleading:Itrailing],
                                       leading, trailing))

        # Alrighty. The centers are all numeric. I gather all the digits around the
        # centers, and I'm done
        m = re.match("^(.*?)([0-9]*)$", leading)
        if m:
            pre_numeric = m.group(2)
        else:
            pre_numeric = ''

        m = re.match("^([0-9]*)(.*?)$", trailing)
        if m:
            post_numeric = m.group(1)
        else:
            post_numeric = ''

        return [int(pre_numeric + f[Nleading:Itrailing] + post_numeric) for f in files]




    Ncameras = len(files_per_camera)
    mapping = {}
    for icamera in range(Ncameras):
        try:
            framenumbers = pull_framenumbers(files_per_camera[icamera])
        except:
            # If we couldn't parse out the frame numbers, but there's only one
            # camera, then I just use a sequential list of integers. Since it
            # doesn't matter
            if Ncameras == 1:
                framenumbers = range(len(files_per_camera[icamera]))
            else:
                raise
        if framenumbers is not None:
            mapping.update(zip(files_per_camera[icamera], [(iframe,icamera) for iframe in framenumbers]))
    return mapping


def close_contour(c):
    r'''Close a polyline, if it isn't already closed

SYNOPSIS

    print( a.shape )
    ===>
    (5, 2)

    print( a[0] )
    ===>
    [844 204]

    print( a[-1] )
    ===>
    [886 198]

    b = mrcal.close_contour(a)

    print( b.shape )
    ===>
    (6, 2)

    print( b[0] )
    ===>
    [844 204]

    print( b[-2:] )
    ===>
    [[886 198]
     [844 204]]

This function works with polylines represented as arrays of shape (N,2). The
polygon represented by such a polyline is "closed" if its first and last points
sit at the same location. This function ingests a polyline, and returns the
corresponding, closed polygon. If the first and last points of the input match,
the input is returned. Otherwise, the first point is appended to the end, and
this extended polyline is returned.

None is accepted as an empty polygon: we return None.

ARGUMENTS

- c: an array of shape (N,2) representing the polyline to be closed. None and
  arrays of shape (0,2) are accepted as special cases ("unknown" and "empty"
  regions, respectively)

RETURNED VALUE

An array of shape (N,2) representing the closed polygon. The input is returned
if the input was None or has shape (0,2)

    '''
    if c is None or c.size == 0: return c

    if np.linalg.norm( c[0,:] - c[-1,:]) < 1e-6:
        return c
    return nps.glue(c, c[0,:], axis=-2)



def _plot_options_state_meas_boundary(f,optimization_inputs,arg0):
    if arg0 is not None: i0 = f(arg0,**optimization_inputs)
    else:                i0 = f(     **optimization_inputs)
    if i0 is None:
        return

    what = f.__name__.replace('state_index_','').replace('measurement_index_','')
    yield from \
        (f'arrow from {i0}, graph 0 to {i0}, graph 1 nohead',
         f'label "{what}" at {i0},graph 0 left front offset 0,character 2 boxed')

def plotoptions_state_boundaries(**optimization_inputs):
    r'''Return the 'set' plot options for gnuplotlib to show the state boundaries

SYNOPSIS

    import numpy as np
    import gnuplotlib as gp
    import mrcal

    model               = mrcal.cameramodel('xxx.cameramodel')
    optimization_inputs = model.optimization_inputs()

    J = mrcal.optimizer_callback(**optimization_inputs)[2]

    gp.plot( np.sum(np.abs(J.toarray()), axis=-2),
             _set = mrcal.plotoptions_state_boundaries(**optimization_inputs) )

    # a plot pops up showing the magnitude of the effects of each element of the
    # packed state (as seen by the optimizer), with boundaries between the
    # different state variables denoted

When plotting the state vector (or anything relating to it, such as rows of the
Jacobian), it is usually very useful to infer at a glance the meaning of each
part of the plot. This function returns a list of 'set' directives passable to
gnuplotlib that show the boundaries inside the state vector.

ARGUMENTS

**optimization_inputs: a dict() of arguments passable to mrcal.optimize() and
mrcal.optimizer_callback(). These define the full optimization problem, and can
be obtained from the optimization_inputs() method of mrcal.cameramodel

RETURNED VALUE

A list of 'set' directives passable as plot options to gnuplotlib

    '''
    return \
        list(_plot_options_state_meas_boundary(mrcal.state_index_intrinsics,     optimization_inputs, 0)) + \
        list(_plot_options_state_meas_boundary(mrcal.state_index_extrinsics,     optimization_inputs, 0)) + \
        list(_plot_options_state_meas_boundary(mrcal.state_index_frames,         optimization_inputs, 0)) + \
        list(_plot_options_state_meas_boundary(mrcal.state_index_points,         optimization_inputs, 0)) + \
        list(_plot_options_state_meas_boundary(mrcal.state_index_calobject_warp, optimization_inputs, None))


def plotoptions_measurement_boundaries(**optimization_inputs):
    r'''Return the 'set' plot options for gnuplotlib to show the measurement boundaries

SYNOPSIS

    import numpy as np
    import gnuplotlib as gp
    import mrcal

    model               = mrcal.cameramodel('xxx.cameramodel')
    optimization_inputs = model.optimization_inputs()

    x = mrcal.optimizer_callback(**optimization_inputs)[1]

    gp.plot( np.abs(x),
             _set = mrcal.plotoptions_measurement_boundaries(**optimization_inputs) )

    # a plot pops up showing the magnitude of each measurement, with boundaries
    # between the different measurements denoted

When plotting the measurement vector (or anything relating to it, such as
columns of the Jacobian), it is usually very useful to infer at a glance the
meaning of each part of the plot. This function returns a list of 'set'
directives passable to gnuplotlib that show the boundaries inside the
measurement vector.

ARGUMENTS

**optimization_inputs: a dict() of arguments passable to mrcal.optimize() and
mrcal.optimizer_callback(). These define the full optimization problem, and can
be obtained from the optimization_inputs() method of mrcal.cameramodel

RETURNED VALUE

A list of 'set' directives passable as plot options to gnuplotlib

    '''
    return \
        list(_plot_options_state_meas_boundary(mrcal.measurement_index_boards,         optimization_inputs, 0)) + \
        list(_plot_options_state_meas_boundary(mrcal.measurement_index_points,         optimization_inputs, 0)) + \
        list(_plot_options_state_meas_boundary(mrcal.measurement_index_regularization, optimization_inputs, None))


def ingest_packed_state(b_packed,
                        **optimization_inputs):
    r'''Read a given packed state into optimization_inputs

SYNOPSIS

    # A simple gradient check

    model               = mrcal.cameramodel('xxx.cameramodel')
    optimization_inputs = model.optimization_inputs()

    b0,x0,J = mrcal.optimizer_callback(no_factorization = True,
                                       **optimization_inputs)[:3]

    db = np.random.randn(len(b0)) * 1e-9

    mrcal.ingest_packed_state(b0 + db,
                              **optimization_inputs)

    x1 = mrcal.optimizer_callback(no_factorization = True,
                                  no_jacobian      = True,
                                  **optimization_inputs)[1]

    dx_observed  = x1 - x0
    dx_predicted = nps.inner(J, db_packed)

This is the converse of mrcal.optimizer_callback(). One thing
mrcal.optimizer_callback() does is to convert the expanded (intrinsics,
extrinsics, ...) arrays into a 1-dimensional scaled optimization vector
b_packed. mrcal.ingest_packed_state() allows updates to b_packed to be absorbed
back into the (intrinsics, extrinsics, ...) arrays for further evaluation with
mrcal.optimizer_callback() and others.

ARGUMENTS

- b_packed: a numpy array of shape (Nstate,) containing the input packed state

- **optimization_inputs: a dict() of arguments passable to mrcal.optimize() and
  mrcal.optimizer_callback(). The arrays in this dict are updated


RETURNED VALUE

None

    '''

    intrinsics     = optimization_inputs.get("intrinsics")
    extrinsics     = optimization_inputs.get("rt_cam_ref")
    frames         = optimization_inputs.get("rt_ref_frame")
    points         = optimization_inputs.get("points")
    calobject_warp = optimization_inputs.get("calobject_warp")

    Npoints_fixed  = optimization_inputs.get('Npoints_fixed', 0)

    Nvars_intrinsics     = mrcal.num_states_intrinsics    (**optimization_inputs)
    Nvars_extrinsics     = mrcal.num_states_extrinsics    (**optimization_inputs)
    Nvars_frames         = mrcal.num_states_frames        (**optimization_inputs)
    Nvars_points         = mrcal.num_states_points        (**optimization_inputs)
    Nvars_calobject_warp = mrcal.num_states_calobject_warp(**optimization_inputs)

    Nvars_expected = \
        Nvars_intrinsics + \
        Nvars_extrinsics + \
        Nvars_frames     + \
        Nvars_points     + \
        Nvars_calobject_warp

    # Defaults MUST match those in OPTIMIZER_ARGUMENTS_OPTIONAL in
    # mrcal-pywrap.c. Or better yet, this whole function should
    # come from the C code instead of being reimplemented here in Python
    do_optimize_intrinsics_core        = optimization_inputs.get('do_optimize_intrinsics_core',        True)
    do_optimize_intrinsics_distortions = optimization_inputs.get('do_optimize_intrinsics_distortions', True)
    do_optimize_extrinsics             = optimization_inputs.get('do_optimize_extrinsics',             True)
    do_optimize_frames                 = optimization_inputs.get('do_optimize_frames',                 True)
    do_optimize_calobject_warp         = optimization_inputs.get('do_optimize_calobject_warp',         True)


    if b_packed.ravel().size != Nvars_expected:
        raise Exception(f"Mismatched array size: b_packed.size={b_packed.ravel().size} while the optimization problem expects {Nvars_expected}")

    b = b_packed.copy()
    mrcal.unpack_state(b, **optimization_inputs)

    if do_optimize_intrinsics_core or \
       do_optimize_intrinsics_distortions:

        ivar0 = mrcal.state_index_intrinsics(0, **optimization_inputs)
        if ivar0 is not None:
            iunpacked0,iunpacked1 = None,None # everything by default

            lensmodel    = optimization_inputs['lensmodel']
            has_core     = mrcal.lensmodel_metadata_and_config(lensmodel)['has_core']
            Ncore        = 4 if has_core else 0
            Ndistortions = mrcal.lensmodel_num_params(lensmodel) - Ncore

            if not do_optimize_intrinsics_core:
                iunpacked0 = Ncore
            if not do_optimize_intrinsics_distortions:
                iunpacked1 = -Ndistortions

            intrinsics[:, iunpacked0:iunpacked1].ravel()[:] = \
                b[ ivar0:Nvars_intrinsics ]

    if do_optimize_extrinsics:
        ivar0 = mrcal.state_index_extrinsics(0, **optimization_inputs)
        if ivar0 is not None:
            extrinsics.ravel()[:] = b[ivar0:ivar0+Nvars_extrinsics]

    if do_optimize_frames:
        ivar0 = mrcal.state_index_frames(0, **optimization_inputs)
        if ivar0 is not None:
            frames.ravel()[:] = b[ivar0:ivar0+Nvars_frames]

    if do_optimize_frames:
        ivar0 = mrcal.state_index_points(0, **optimization_inputs)
        if ivar0 is not None:
            points.ravel()[:-Npoints_fixed*3] = b[ivar0:ivar0+Nvars_points]

    if do_optimize_calobject_warp:
        ivar0 = mrcal.state_index_calobject_warp(**optimization_inputs)
        if ivar0 is not None:
            calobject_warp.ravel()[:] = b[ivar0:ivar0+Nvars_calobject_warp]


def sorted_eig(M):
    r'''Compute eigenvalues, eigenvectors; sorted results returned

SYNOPSIS

    # p is an array of shape (N,3): a point cloud. I find a normal to a fitted
    # plane
    p -= np.mean(p, axis=-2)

    l,v = mrcal.sorted_eig(nps.matmult(nps.transpose(p),p))

    n = v[:,0]

    # n is the eigenvector corresponding to the smallest eigenvalue

This function calls numpy.linalg.eig(), and sorts the output by eigenvalue, in
ascending order

ARGUMENTS

- M: the matrix to eigendecompose

RETURNED VALUE

Same as that of numpy.linalg.eig(). A tuple:

- Eigenvalues, sorted in ascending order

- Eigenvectors, sorted by the corresponding eigenvalue. Each eigenvector is
  stored in a COLUMN of this array

    '''
    l,v = np.linalg.eig(M)
    i = np.argsort(l)

    # for single matrices, I can simply do
    #   return l[i], v[:,i]
    # But to support broadcasting I need these more complex expressions
    return \
        ( np.take_along_axis(l, i, -1), \
          np.take_along_axis(v, nps.dummy(i,-2), -1) )


def _plot_arg_covariance_ellipse(q_mean, Var, what):

    # if the variance is 0, the ellipse is infinitely small, and I don't even
    # try to plot it. Gnuplot has the arguably-buggy behavior where drawing an
    # ellipse with major_diam = minor_diam = 0 plots a nominally-sized ellipse.
    if np.max(np.abs(Var)) == 0:
        return None

    l,v   = mrcal.sorted_eig(Var)
    l0,l1 = l
    v0,v1 = nps.transpose(v)

    major = np.sqrt(l0)
    minor = np.sqrt(l1)

    return \
      (q_mean[0], q_mean[1], 2*major, 2*minor, 180./np.pi*np.arctan2(v0[1],v0[0]),
       dict(_with='ellipses', tuplesize=5, legend=what))


def _plot_args_points_and_covariance_ellipse(q, what):
    q_mean  = np.mean(q,axis=-2)
    q_mean0 = q - q_mean
    Var     = np.mean( nps.outer(q_mean0,q_mean0), axis=0 )
    # Some functions assume that we're plotting _with = "dots". Look for the
    # callers if changing this
    return ( _plot_arg_covariance_ellipse(q_mean,Var, what),
             ( q, dict(_with = 'dots',
                         tuplesize = -2)) )

def measurements_board(optimization_inputs,
                       *,
                       icam_intrinsics     = None,
                       x                   = None,
                       return_observations = False,

                       # for backwards-compatibility only; same as
                       # icam_intrinsics
                       i_cam               = None):
    r'''Compute and return the chessboard measurements

SYNOPSIS

    model = mrcal.cameramodel(model_filename)

    gp.plot( mrcal.measurements_board(
               optimization_inputs = model.optimization_inputs(),
               icam_intrinsics     = icam_intrinsics ).ravel(),
             histogram = True,
             binwidth  = 0.02 )

    ... A plot pops up showing the empirical distribution of chessboard fit
    ... errors in this solve. For the given camera only

Given a calibration solve, returns the measurements of chessboard observations,
throwing out outliers and, optionally, selecting the measurements from a
specific camera. Measurements are the WEIGHTED reprojection errors in the
measurement vector.

If no chessboards are present in the solve I return arrays of appropriate shape
with N = 0

ARGUMENTS

- optimization_inputs: the optimization inputs dict passed into and returned
  from mrcal.optimize(). This describes the solved optimization problem

- icam_intrinsics: optional integer to select the camera whose measurements
  we're interested in. If omitted or None, we report the measurements for ALL
  the cameras together.

- x: optional numpy array of shape (Nmeasurements,) containing the optimization
  measurements. If omitted or None, this will be recomputed. To use a cached
  value, pass the result of mrcal.optimize(**optimization_inputs)['x'] or
  mrcal.optimizer_callback(**optimization_inputs)[1]

- return_observations: optional boolean, defaulting to False. if
  return_observations: we return a tuple (x,observations) instead of just x

RETURNED VALUES

if not return_observations:

  we return a numpy array of shape (N,2) of all the measurements. N is the number
  of pixel observations remaining after outliers and other cameras are thrown
  out

else:

  we return a tuple:

  - The same measurements array as before

  - The corresponding observation points in a numpy array of shape (N,2). These
    are a slice of observations_board[] corresponding to each measurement

    '''

    # shape (Nobservations, object_height_n, object_width_n, 3)
    observations_board = optimization_inputs.get('observations_board')
    if observations_board is None or observations_board.size == 0:
        # no board observations
        if not return_observations:
            # shape (N,2)
            return \
                np.zeros((0,2), dtype=float)
        else:
            # shape (N,2), (N,2)
            return \
                np.zeros((0,2), dtype=float), \
                np.zeros((0,2), dtype=float)



    # for backwards compatibility
    if i_cam is not None:
        icam_intrinsics = i_cam

    if x is None:
        # Flattened measurements. This is ALL the measurements: chessboard,
        # point, regularization...
        x = \
            mrcal.optimizer_callback(**optimization_inputs,
                                     no_jacobian      = True,
                                     no_factorization = True)[1]

    measurements_shape = observations_board.shape[:-1] + (2,)

    # shape (Nobservations, object_height_n, object_width_n, 2)
    imeas0  = mrcal.measurement_index_boards(0, **optimization_inputs)
    Nmeas   = mrcal.num_measurements_boards(**optimization_inputs)
    x_board = x[imeas0:imeas0+Nmeas].reshape(*measurements_shape)

    # shape (Nobservations, object_height_n, object_width_n, 3)
    indices_frame_camera = optimization_inputs['indices_frame_camintrinsics_camextrinsics'][...,:2]

    # shape (Nobservations, object_height_n, object_width_n)
    idx = np.ones( observations_board.shape[:-1], dtype=bool)

    if icam_intrinsics is not None:
        # select measurements from THIS camera
        idx[indices_frame_camera[:,1] != icam_intrinsics, ...] = False

    # select non-outliers
    idx[ observations_board[...,2] <= 0.0 ] = False

    if not return_observations:
        # shape (N,2)
        return \
            x_board[idx, ...    ]
    else:
        # shape (N,2), (N,2)
        return \
            x_board[idx, ...    ], \
            observations_board[idx, ..., :2]

def residuals_board(*args,
                    residuals = None,
                    **kwargs):
    r'''Backwards-compatibility flavor of measurements_board()

Use measurements_board() and the "x" argument. The new version uses clearer
nomenclature. We are talking about MEASUREMENTS: the optimization vector x that
contains the WEIGHTED reprojection errors

    '''
    return measurements_board(*args,
                              x = residuals,
                              **kwargs)

# For backwards compatibility
residuals_chessboard = residuals_board

def measurements_point(optimization_inputs,
                       *,
                       icam_intrinsics     = None,
                       x                   = None,
                       return_observations = False):
    r'''Compute and return the observed-point measurements

SYNOPSIS

    model = mrcal.cameramodel(model_filename)

    gp.plot( mrcal.measurements_point(
               optimization_inputs = model.optimization_inputs(),
               icam_intrinsics     = icam_intrinsics ).ravel(),
             histogram = True,
             binwidth  = 0.02 )

    ... A plot pops up showing the empirical distribution of observed-point fit
    ... errors in this solve. For the given camera only

Given a calibration solve, returns the measurements of point observations,
throwing out outliers and, optionally, selecting the measurements from a
specific camera. Measurements are the WEIGHTED reprojection errors in the
measurement vector.

If no points are present in the solve I return arrays of appropriate shape with
N = 0

ARGUMENTS

- optimization_inputs: the optimization inputs dict passed into and returned
  from mrcal.optimize(). This describes the solved optimization problem

- icam_intrinsics: optional integer to select the camera whose measurements
  we're interested in. If omitted or None, we report the measurements for ALL
  the cameras together.

- x: optional numpy array of shape (Nmeasurements,) containing the optimization
  measurements. If omitted or None, this will be recomputed. To use a cached
  value, pass the result of mrcal.optimize(**optimization_inputs)['x'] or
  mrcal.optimizer_callback(**optimization_inputs)[1]

- return_observations: optional boolean, defaulting to False. if
  return_observations: we return a tuple (x,observations) instead of just x

RETURNED VALUES

if not return_observations:

  we return a numpy array of shape (N,2) of all the measurements. N is the number
  of pixel observations remaining after outliers and other cameras are thrown
  out

else:

  we return a tuple:

  - The same measurements array as before

  - The corresponding observation points in a numpy array of shape (N,2). These
    are a slice of observations_point[] corresponding to each measurement

    '''

    # shape (Nobservations, 3)
    observations_point = optimization_inputs.get('observations_point')
    if observations_point is None or observations_point.size == 0:
        # no point observations
        if not return_observations:
            # shape (N,2)
            return \
                np.zeros((0,2), dtype=float)
        else:
            # shape (N,2), (N,2)
            return \
                np.zeros((0,2), dtype=float), \
                np.zeros((0,2), dtype=float)

    if x is None:
        # Flattened measurements. This is ALL the measurements: chessboard,
        # point, regularization...
        x = \
            mrcal.optimizer_callback(**optimization_inputs,
                                     no_jacobian      = True,
                                     no_factorization = True)[1]

    Nobservations = len(observations_point)

    imeas0 = mrcal.measurement_index_points(0, **optimization_inputs)
    Nmeas  = mrcal.num_measurements_points(**optimization_inputs)

    # shape (Nobservations, 2); each observation row is (err_x, err_y)
    x_point = x[imeas0:imeas0+Nmeas].reshape(Nobservations,2)

    indices_point_camera = optimization_inputs['indices_point_camintrinsics_camextrinsics'][...,:2]

    # shape (Nobservations,)
    idx = np.ones( (Nobservations,), dtype=bool)

    if icam_intrinsics is not None:
        # select measurements from THIS camera
        idx[indices_point_camera[:,1] != icam_intrinsics, ...] = False

    # select non-outliers
    idx[ observations_point[...,2] <= 0.0 ] = False

    if not return_observations:
        # shape (N,2)
        return \
            x_point[idx, ...    ]
    else:
        # shape (N,2), (N,2)
        return \
            x_point[idx, ...    ], \
            observations_point[idx, ..., :2]

def residuals_point(*args,
                    residuals = None,
                    **kwargs):
    r'''Backwards-compatibility flavor of measurements_point()

Use measurements_point() and the "x" argument. The new version uses clearer
nomenclature. We are talking about MEASUREMENTS: the optimization vector x that
contains the WEIGHTED reprojection errors

    '''
    return measurements_point(*args,
                              x = residuals,
                              **kwargs)

def _R_aligned_to_vector_python(v):
    r'''Reference implementation of R_aligned_to_vector in python

The main implementation is written in C in poseutils.c:

  mrcal_R_aligned_to_vector()

The two implementations are identical, with a test to verify this
    '''
    z = v/nps.mag(v)
    if np.abs(z[0]) < .9:
        x = np.array((1,0,0.))
    else:
        x = np.array((0,1,0.))
    x -= nps.inner(x,z)*z
    x /= nps.mag(x)
    y = np.cross(z,x)
    return nps.cat(x,y,z)

def write_point_cloud_as_ply(filename,
                             points,
                             *,
                             color     = None,
                             binary    = True):

    r'''Write a point cloud to a .ply file

SYNOPSIS

    print( points.shape )
    ===>
    (1000, 3)

    write_point_cloud_as_ply( "points.ply", points )

The point cloud to visualize may contain colors for each point: either as a BGR
triplet or as a single intensity value. The colors may be given either in the
trailing columns of the "points" array or in a separate "color" array.

Both binary and ascii .ply formats are supported, with the binary format the
default.

ARGUMENTS

- filename: the filename of the .ply file being written

- points: the numpy array of points being visualized. Required. One of 3 shapes
  is allowed:

  - shape (N,3) if no color information is given here; may be given in the
    separate "color" array
  - shape (N,4) if a grayscale intensity is given in the last column
  - shape (N,6) if a BGR color is given in the last 3 columns

  If color is given here, the separate "color" array must be None

- color: the numpy array of colors corresponding to each point in "points".
  Optional. If the colors are given here, they must NOT be given in "points":
  points.shape MUST be (N,3). If not None, 2 shapes are allowed:

  - shape (N,)  if we have a grayscale intensity
  - shape (N,3) if we have BGR colors

- binary: optional boolean, defaulting to True. Selects the binary or ascii .ply
  format.


RETURNED VALUE

None
'''


    if points.ndim != 2:
        raise Exception(f"points.shape must be (N,3) or (N,4) or (N,6), but points.ndim={points.ndim}")
    N,Ntrailing = points.shape

    if not (Ntrailing == 3 or \
            Ntrailing == 4 or \
            Ntrailing == 6):
        raise Exception(f"points.shape[-1] must be one of (3,4,6) to indicate no-color or grayscale or bgr colors respectively. Instead got points.shape[-1] = {points.shape[-1]}")

    if color is not None:

        if Ntrailing != 3:
            raise Exception("Both 'points' and 'color' are specifying color information. At most one of those should provide it")

        if color.ndim > 2:
            raise Exception(f"color.shape must be (N,) or (N,3), but color.ndim={color.ndim}")
        if N != color.shape[0]:
            raise Exception(f"Inconsistent point counts: len(points)={len(points)} but len(color)={len(color)}")

        if not (color.shape == (N,) or \
                color.shape == (N,3) ):
            raise Exception(f"color.shape must be (N,) or (N,3), but color.shape={color.shape}")

    else:
        # No "color" specified
        if Ntrailing == 4:
            # But it exists as grayscale in the points. Separate it into the
            # "color" array
            color  = points[:,3]
            points = points[:,:3]
        elif Ntrailing == 6:
            # But it exists as bgr in the points. Separate it into the
            # "color" array
            color  = points[:,3:]
            points = points[:,:3]



    # I now have points.shape = (N,3) and color is None or (N,) or (N,3)



    if binary:
        ply_type = 'binary_little_endian'
    else:
        ply_type = 'ascii'

    if color is not None and color.ndim == 1:
        # grayscale
        dtype = np.dtype([ ('xyz',np.float32,3), ('i', np.uint8) ])
        header_color = '''property uchar intensity
'''
    elif color is not None and color.ndim == 2:
        # bgr
        dtype = np.dtype([ ('xyz',np.float32,3), ('rgb', np.uint8, 3) ])
        header_color = '''property uchar red
property uchar green
property uchar blue
'''
    else:
        dtype = np.dtype([ ('xyz',np.float32,3), ])
        header_color = ''

    ply_header = f'''ply
format {ply_type} 1.0
element vertex {N}
property float x
property float y
property float z
{header_color}end_header
'''.encode()


    with open(filename, 'wb') as f:
        f.write(ply_header)

        if binary:
            binary_ply = np.empty( (N,),
                                   dtype = dtype)
            binary_ply['xyz'] = points.astype(np.float32)
            if color is not None and color.ndim == 1:
                # grayscale
                binary_ply['i'  ] = color .astype(np.uint8)
            elif color is not None and color.ndim == 2:
                # bgr
                binary_ply['rgb'][:,0] = color[:,2]
                binary_ply['rgb'][:,1] = color[:,1]
                binary_ply['rgb'][:,2] = color[:,0]

            binary_ply.tofile(f)
        else:
            if color is not None and color.ndim == 1:
                # grayscale
                pbgr = nps.glue(points,
                                nps.dummy(color, -1),
                                axis = -1)
                np.savetxt(f,
                           pbgr,
                           fmt      = ('%.1f','%.1f','%.1f','%d'),
                           comments = '',
                           header   = '')
            elif color is not None and color.ndim == 2:
                # bgr
                pbgr = nps.glue(points,
                                color[:,(2,)],
                                color[:,(1,)],
                                color[:,(0,)],
                                255*np.ones((N,1)),
                                axis = -1)
                np.savetxt(f,
                           pbgr,
                           fmt      = ('%.1f','%.1f','%.1f','%d','%d','%d','%d'),
                           comments = '',
                           header   = '')
            else:
                np.savetxt(f, points,
                           fmt      = ('%.1f','%.1f','%.1f',),
                           comments = '',
                           header   = '')
