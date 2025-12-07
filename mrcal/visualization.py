#!/usr/bin/python3

# Copyright (c) 2017-2023 California Institute of Technology ("Caltech"). U.S.
# Government sponsorship acknowledged. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

'''Visualization routines

All functions are exported into the mrcal module. So you can call these via
mrcal.visualization.fff() or mrcal.fff(). The latter is preferred.

'''

import numpy as np
import numpysane as nps
import sys
import re
import os
import mrcal

def show_geometry(models_or_rt_cam_ref,
                  *,
                  cameranames                 = None,
                  cameras_Rt_plot_ref         = None,
                  rt_ref_frame                = None,
                  points                      = None,

                  show_calobjects    = 'all',
                  show_points        = 'all',
                  axis_scale         = None,
                  object_width_n     = None,
                  object_height_n    = None,
                  object_spacing     = None,
                  calobject_warp     = None,
                  point_labels       = None,
                  extratitle         = None,
                  return_plot_args   = False,
                  **kwargs):

    r'''Visualize the world resulting from a calibration run

SYNOPSIS

    # Visualize the geometry from some models on disk
    models = [mrcal.cameramodel(m) for m in model_filenames]
    plot1 = mrcal.show_geometry(models)

    # Solve a calibration problem. Visualize the resulting geometry AND the
    # observed calibration objects and points
    ...
    mrcal.optimize(intrinsics            = intrinsics,
                   rt_cam_ref            = rt_cam_ref,
                   rt_ref_frame          = rt_ref_frame,
                   points                = points,
                   ...)
    plot2 = \
      mrcal.show_geometry(rt_cam_ref,
                          rt_ref_frame    = rt_ref_frame,
                          points          = points,
                          xlabel          = 'Northing (m)',
                          ylabel          = 'Easting (m)',
                          zlabel          = 'Down (m)')

This function visualizes the world described by a set of camera models. It shows

- The geometry of the cameras themselves. Each one is represented by the axes of
  its coordinate system

- The geometry of the observed objects used to compute these models (calibration
  boards and/or points).

The geometry is shown only if requested and available

- Requested: we show the calibration boards if show_calobjects, and the points
  if show_points. If the data comes from model.optimization_inputs(), then we
  can have a bit more control. if show_calobjects == 'all': we show ALL the
  calibration objects, observed by ANY camera. elif show_calobjects ==
  'thiscamera': we only show the calibration objects that were observed by the
  given camera at calibration time. Similarly with show_points.

- Available: The data comes either from the rt_ref_frame and/or points
  arguments or from the first model.optimization_inputs() that is given. If we
  have both, we use the rt_ref_frame/points

This function can also be used to visualize the output (or input) of
mrcal.optimize(); the relevant parameters are all identical to those
mrcal.optimize() takes.

This function is the core of the mrcal-show-geometry tool.

All arguments except models_or_rt_cam_ref are optional.

Extra **kwargs are passed directly to gnuplotlib to control the plot.

ARGUMENTS

- models_or_rt_cam_ref: an iterable of mrcal.cameramodel objects or
  (6,) rt arrays. A array of shape (N,6) works to represent N cameras. If
  mrcal.cameramodel objects are given here and rt_ref_frame (or points) are
  omitted, we get the rt_ref_frame (or points) from the first model that
  provides optimization_inputs().

- cameranames: optional array of strings of labels for the cameras. If omitted,
  we use generic labels. If given, the array must have the same length as
  models_or_rt_cam_ref

- cameras_Rt_plot_ref: optional transformation(s). If omitted or None, we plot
  everything in the reference coordinate system. If given, we use a "plot"
  coordinate system with the transformation TO plot coordinates FROM the
  reference coordinates given in this argument. This argument can be given as
  single Rt transformation to apply to everything; or an iterable of Rt
  transformations to use a different one for each camera (the number of
  transforms must match the number of cameras exactly). If a separate transform
  per camera is given, we must not be plotting points or frames

- rt_ref_frame: optional array of shape (N,6). If given, each row of shape
  (6,) is an rt transformation representing the transformation TO the reference
  coordinate system FROM the calibration object coordinate system. The
  calibration object then MUST be defined by passing in valid object_width_n,
  object_height_n, object_spacing parameters. If rt_ref_frame is omitted or
  None, we look for this data in the given camera models. I look at the given
  models in order, and grab the frames from the first model that has them. If
  none of the models have this data and rt_ref_frame is omitted or NULL, then
  I don't plot any frames at all

- object_width_n: the number of horizontal points in the calibration object
  grid. Required only if rt_ref_frame is not None

- object_height_n: the number of vertical points in the calibration object grid.
  Required only if rt_ref_frame is not None

- object_spacing: the distance between adjacent points in the calibration
  object. A square object is assumed, so the vertical and horizontal distances
  are assumed to be identical. Required only if rt_ref_frame is not None

- calobject_warp: optional (2,) array describing the calibration board warping.
  None means "no warping": the object is flat. Used only if rt_ref_frame is
  not None. See the docs for mrcal.ref_calibration_object() for a description.

- points: optional array of shape (N,3). If omitted, we don't plot the observed
  points. If given, each row of shape (3,) is a point in the reference
  coordinate system.

- point_labels: optional dict from a point index to a string describing it.
  Points in this dict are plotted with this legend; all other points are plotted
  under a generic "points" legend. As many or as few of the points may be
  labelled in this way. If omitted, none of the points will be labelled
  specially. This is used only if points is not None

- show_calobjects: optional value defaults to 'all'. if show_calobjects: we
  render the observed calibration objects (if they are available in
  rt_ref_frame or model.optimization_inputs()['rt_ref_frame']; we look at
  the FIRST model that provides this data). If we have optimization_inputs and
  show_calobjects == 'all': we display the objects observed by ANY camera. elif
  show_calobjects == 'thiscamera': we only show those observed by THIS camera.
  The only allowed values are 'all', 'thiscamera' or anything evaluating to
  False (False, 0, None, ...)

- show_points: same as show_calobjects, but applying to discrete points, not
  chessboard poses

- axis_scale: optional scale factor for the size of the axes used to represent
  the cameras. Can be omitted to use some reasonable default size, but tweaking
  it might be necessary to make the plot look right. If 1 or fewer cameras are
  given, this defaults to 1.0

- extratitle: optional string to include in the title of the resulting plot.
  Used to extend the default title string. If kwargs['title'] is given, it is
  used directly, and the extratitle is ignored

- return_plot_args: boolean defaulting to False. if return_plot_args: we return
  a (data_tuples, plot_options) tuple instead of making the plot. The plot can
  then be made with gp.plot(*data_tuples, **plot_options). Useful if we want to
  include this as a part of a more complex plot

- **kwargs: optional arguments passed verbatim as plot options to gnuplotlib.
  Useful to make hardcopies, set the plot title, etc.

RETURNED VALUES

if not return_plot_args (the usual path): we return the gnuplotlib plot object.
The plot disappears when this object is destroyed (by the garbage collection,
for instance), so save this returned plot object into a variable, even if you're
not going to be doing anything with this object.

if return_plot_args: we return a (data_tuples, plot_options) tuple instead of
making the plot. The plot can then be made with gp.plot(*data_tuples,
**plot_options). Useful if we want to include this as a part of a more complex
plot

    '''

    import gnuplotlib as gp

    # First one with optimization_inputs. If we're not given frames and/or
    # points, we get them from the optimization inputs in this model
    i_model_with_optimization_inputs = \
        next((i for i in range(len(models_or_rt_cam_ref)) \
              if isinstance(models_or_rt_cam_ref[i], mrcal.cameramodel) and \
                  models_or_rt_cam_ref[i].optimization_inputs() is not None),
             None)

    Ncameras = len(models_or_rt_cam_ref)

    if cameras_Rt_plot_ref is not None:
        # have cameras_Rt_plot_ref. It can be
        # - a list of transforms
        # - a single transform
        # If we're given rt_ref_frame or points, then a different transform
        # per camera doesn't make sense, and I barf
        if not isinstance(cameras_Rt_plot_ref, np.ndarray):
            # cameras_Rt_plot_ref must be an iterable with each slice being an
            # array of shape (4,3)
            try:
                cameras_Rt_plot_ref = np.array(cameras_Rt_plot_ref, dtype=float)
            except:
                raise Exception("cameras_Rt_plot_ref must be a transform or a list of transforms")

        # cameras_Rt_plot_ref is now an array. Does it have the right shape?
        if cameras_Rt_plot_ref.shape[-2:] != (4,3):
            raise Exception("cameras_Rt_plot_ref must be a transform or a list of Rt transforms. Got an array that looks like neither")
        # shape is (....,4,3)
        if cameras_Rt_plot_ref.ndim == 2:
            # shape is (4,3). I extend it to
            # shape (N,4,3)
            cameras_Rt_plot_ref = \
                np.repeat(nps.atleast_dims(cameras_Rt_plot_ref,-3),
                          Ncameras,
                          axis=-3)
        elif cameras_Rt_plot_ref.ndim == 3:
            # shape is (N,4,3).
            if len(cameras_Rt_plot_ref) != Ncameras:
                raise Exception("cameras_Rt_plot_ref must be a transform or a list of transforms, exactly one per camera. Got mismatched number of transforms")
            # have an array of transforms with the right shape
            if rt_ref_frame is not None or points is not None:
                raise Exception("Got multiple transforms in cameras_Rt_plot_ref AND rt_ref_frame or points. This makes no sense: I can't tell which transform to use")
            pass
        else:
            raise Exception("cameras_Rt_plot_ref must be a transform or a list of transforms, exactly one per camera. Got mismatched number of transforms")
    # cameras_Rt_plot_ref is now either None or an array of shape (N,4,3)

    if rt_ref_frame is not None:
        if object_width_n  is None or \
           object_height_n is None or \
           object_spacing  is None:
            raise Exception("rt_ref_frame is given, so object_width_n and object_height_n and object_spacing must be given as well")


    def get_Rt_ref_cam_one(m):
        if isinstance(m, mrcal.cameramodel):
            return m.Rt_ref_cam()
        else:
            return mrcal.invert_Rt(mrcal.Rt_from_rt(m))

    Rt_ref_cam_all = \
        nps.cat(*[get_Rt_ref_cam_one(m) \
                  for m in models_or_rt_cam_ref])

    # reshape Rt_ref_cam_all to exactly (N,4,3)
    Rt_ref_cam_all = nps.atleast_dims(Rt_ref_cam_all, -3)
    Rt_ref_cam_all = nps.clump( Rt_ref_cam_all,
                                n = Rt_ref_cam_all.ndim-3 )

    if axis_scale is None:
        # This is intended to work with the behavior in the mrcal-stereo
        # tool. That tool sets the fov-indicating hair lengths to
        # baseline/4. Here I default to a bit more: baseline/3

        # I want this routine to work with any N cameras. I need to compute a
        # "characteristic distance" for them to serve as a "baseline". I do this
        # by first computing a "geometric median" that minimizes the mean
        # of the distances from each point to this median. Then the
        # characteristic radius is this constant-ish distance.

        # shape (N,3); the position of each camera in the ref coord system
        p = Rt_ref_cam_all[:,3,:]

        if len(p) <= 1:
            axis_scale = 1.0
        else:
            import scipy.optimize
            res = scipy.optimize.least_squares(# cost function
                                               lambda c: np.sum(nps.mag(p-c)),

                                               # seed
                                               np.mean(p,axis=-2),
                                               method='trf',
                                               verbose=False)
            center = res.x
            baseline = np.mean(nps.mag(p-center)) * 2.
            axis_scale = baseline/3.

            if axis_scale == 0.:
                # All the cameras are at exactly the same spot
                axis_scale = 1.0


    def get_boards_to_plot():
        r'''get the chessboard poses to plot'''

        if not show_calobjects:
            # Not requested
            return None

        if rt_ref_frame is not None:
            # We have explicit poses given. Use them
            return \
                nps.atleast_dims(rt_ref_frame, -2), \
                object_width_n,     \
                object_height_n,    \
                object_spacing,     \
                calobject_warp

        # Poses aren't given. Grab them from the model
        if not (show_calobjects is True  or
                show_calobjects == 'all' or
                show_calobjects == 'thiscamera'):
            raise Exception("show_calobjects must be 'all' or 'thiscamera' or True or False")

        if i_model_with_optimization_inputs is None:
            return None

        m = models_or_rt_cam_ref[i_model_with_optimization_inputs]
        optimization_inputs = m.optimization_inputs()

        if not ('observations_board' in optimization_inputs and \
                optimization_inputs['observations_board'] is not None and \
                optimization_inputs['observations_board'].size ):
            return None;

        _rt_ref_frame = optimization_inputs['rt_ref_frame']
        _object_spacing  = optimization_inputs['calibration_object_spacing']
        _object_width_n  = optimization_inputs['observations_board'].shape[-2]
        _object_height_n = optimization_inputs['observations_board'].shape[-3]
        _calobject_warp  = optimization_inputs['calobject_warp']

        icam_intrinsics = m.icam_intrinsics()

        if show_calobjects == 'thiscamera':
            indices_frame_camintrinsics_camextrinsics = \
                optimization_inputs['indices_frame_camintrinsics_camextrinsics']
            mask_observations = \
                indices_frame_camintrinsics_camextrinsics[:,1] == icam_intrinsics
            idx_frames = indices_frame_camintrinsics_camextrinsics[mask_observations,0]
            _rt_ref_frame = _rt_ref_frame[idx_frames]

        # The current rt_ref_frame uses the calibration-time ref, NOT
        # the current ref. I transform. rt_ref_frame = T_rcal_f
        # I want T_rnow_rcal T_rcal_f
        icam_extrinsics = \
            mrcal.corresponding_icam_extrinsics(icam_intrinsics,
                                                **optimization_inputs)
        if icam_extrinsics >= 0:
            _rt_ref_frame = \
                mrcal.compose_rt( mrcal.rt_from_Rt(Rt_ref_cam_all[i_model_with_optimization_inputs]),
                                  optimization_inputs['rt_cam_ref'][icam_extrinsics],
                                  _rt_ref_frame )
        else:
            _rt_ref_frame = \
                mrcal.compose_rt( mrcal.rt_from_Rt(Rt_ref_cam_all[i_model_with_optimization_inputs]),
                                  _rt_ref_frame )

        # All good. Done
        return \
            nps.atleast_dims(_rt_ref_frame, -2), \
            _object_width_n,     \
            _object_height_n,    \
            _object_spacing,     \
            _calobject_warp

    def get_points_to_plot():
        r'''get the discrete points to plot'''

        if not show_points:
            # Not requested
            return None

        if points is not None:
            # We have explicit points given. Use them
            return \
                nps.atleast_dims(points, -2)

        # Points aren't given. Grab them from the model
        if not (show_points is True  or
                show_points == 'all' or
                show_points == 'thiscamera'):
            raise Exception("show_points must be 'all' or 'thiscamera' or True or False")

        if i_model_with_optimization_inputs is None:
            return None

        m = models_or_rt_cam_ref[i_model_with_optimization_inputs]
        optimization_inputs = m.optimization_inputs()

        if not ('points' in optimization_inputs and \
                optimization_inputs['points'] is not None and \
                optimization_inputs['points'].size ):
            return None;

        _points = optimization_inputs['points']

        icam_intrinsics = m.icam_intrinsics()

        if show_points == 'thiscamera':
            indices_point_camintrinsics_camextrinsics = \
                optimization_inputs['indices_point_camintrinsics_camextrinsics']
            mask_observations = \
                indices_point_camintrinsics_camextrinsics[:,1] == icam_intrinsics
            idx_points = indices_point_camintrinsics_camextrinsics[mask_observations,0]
            _points = _points[idx_points]

        # The current points uses the calibration-time ref, NOT
        # the current ref. I transform.
        icam_extrinsics = \
            mrcal.corresponding_icam_extrinsics(icam_intrinsics,
                                                **optimization_inputs)

        if icam_extrinsics >= 0:
            _points = \
                mrcal.transform_point_rt( optimization_inputs['rt_cam_ref'][icam_extrinsics],
                                          _points )
            _points = \
                mrcal.transform_point_Rt( Rt_ref_cam_all[i_model_with_optimization_inputs],
                                          _points )
        else:
            _points = \
                mrcal.transform_point_Rt( Rt_ref_cam_all[i_model_with_optimization_inputs],
                                          _points )
        # All good. Done
        return \
            nps.atleast_dims(_points, -2)


    boards_to_plot_list = get_boards_to_plot()
    if boards_to_plot_list is not None:
        rt_ref_frame, \
        object_width_n,  \
        object_height_n, \
        object_spacing,  \
        calobject_warp = boards_to_plot_list
    else:
        rt_ref_frame = None
        # the others will not be referenced if rt_ref_frame is None

    points = get_points_to_plot()

    def extend_axes_for_plotting(axes):
        r'''Input is a 4x3 axes array: center, center+x, center+y, center+z. I transform
        this into a 3x6 array that can be gnuplotted "with vectors"

        '''

        # first, copy the center 3 times
        out = nps.cat( axes[0,:],
                       axes[0,:],
                       axes[0,:] )

        # then append just the deviations to each row containing the center
        out = nps.glue( out, axes[1:,:] - axes[0,:], axis=-1)
        return out


    def gen_plot_axes(transforms, legend, scale = 1.0):
        r'''Given a list of transforms (applied to the reference set of axes in reverse
        order) and a legend, return a list of plotting directives gnuplotlib
        understands. Each transform is an Rt (4,3) matrix

        Transforms are in reverse order so a point x being transformed as A*B*C*x
        can be represented as a transforms list (A,B,C).
        '''
        axes = np.array( ((0,0,0),
                          (1,0,0),
                          (0,1,0),
                          (0,0,2),), dtype=float ) * scale

        transform = mrcal.identity_Rt()

        for x in transforms:
            transform = mrcal.compose_Rt(transform, x)

        return \
            (extend_axes_for_plotting(mrcal.transform_point_Rt(transform, axes)),
             dict(_with     = 'vectors',
                  tuplesize = -6,
                  legend    = legend), )

    def gen_plot_axes_labels(transforms, scale = 1.0):
        r'''Given a list of transforms (applied to the reference set of axes in reverse
        order), return a list of plotting directives gnuplotlib
        understands. Each transform is an Rt (4,3) matrix

        Transforms are in reverse order so a point x being transformed as A*B*C*x
        can be represented as a transforms list (A,B,C).
        '''
        axes = np.array( ((0,0,0),
                          (1,0,0),
                          (0,1,0),
                          (0,0,2),), dtype=float ) * scale

        transform = mrcal.identity_Rt()

        for x in transforms:
            transform = mrcal.compose_Rt(transform, x)

        return tuple(nps.transpose(mrcal.transform_point_Rt(transform,
                                                    axes[1:,:]*1.01))) + \
                                   (np.array(('x', 'y', 'z')),
                                    dict(_with     = 'labels',
                                         tuplesize = 4,
                                         legend    = None),)




    # I need to plot 3 things:
    #
    # - Cameras
    # - Calibration object poses
    # - Observed points
    def gen_curves_cameras():

        def camera_Rt_toplotcoords(i):
            Rt_ref_cam = Rt_ref_cam_all[i]
            if cameras_Rt_plot_ref is None:
                return Rt_ref_cam
            return mrcal.compose_Rt(cameras_Rt_plot_ref[i], Rt_ref_cam)

        def camera_name(i):
            try:
                return cameranames[i]
            except:
                return f'cam{i}'

        cam_axes  = \
            [gen_plot_axes( ( camera_Rt_toplotcoords(i), ),
                            legend = camera_name(i),
                            scale=axis_scale) for i in range(0,Ncameras)]
        cam_axes_labels = \
            [gen_plot_axes_labels( ( camera_Rt_toplotcoords(i), ),
                                   scale=axis_scale) for i in range(0,Ncameras)]

        # I collapse all the labels into one gnuplotlib dataset. Thus I'll be
        # able to turn them all on/off together
        return cam_axes + [(np.ravel(nps.cat(*[l[0] for l in cam_axes_labels])),
                            np.ravel(nps.cat(*[l[1] for l in cam_axes_labels])),
                            np.ravel(nps.cat(*[l[2] for l in cam_axes_labels])),
                            np.tile(cam_axes_labels[0][3], Ncameras)) + \
                            cam_axes_labels[0][4:]]


    def gen_curves_calobjects():

        if rt_ref_frame is None or len(rt_ref_frame) == 0:
            return []

        if object_spacing <= 0     or \
           object_width_n  is None or \
           object_height_n is None:
            raise Exception("We're observing calibration boards, so object_spacing and object_width_n and object_height_n must be valid")

        # if observations_board              is None or \
        #    indices_frame_camera_board      is None or \
        #    len(observations_board)         == 0    or \
        #    len(indices_frame_camera_board) == 0:
        #     return []
        # Nobservations = len(indices_frame_camera_board)

        # if icam_highlight is not None:
        #     i_observations_frames = [(i_observation,indices_frame_camera_board[i_observation,0]) \
        #                              for i_observation in range(Nobservations) \
        #                              if indices_frame_camera_board[i_observation,1] == icam_highlight]

        #     i_observations, iframes = nps.transpose(np.array(i_observations_frames))
        #     rt_ref_frame = rt_ref_frame[iframes, ...]


        calobject_ref = mrcal.ref_calibration_object(object_width_n, object_height_n,
                                                     object_spacing,
                                                     calobject_warp = calobject_warp)

        # object in the ref coord system.
        # shape (Nframes, object_height_n, object_width_n, 3)
        calobject_ref = mrcal.transform_point_rt(nps.mv(rt_ref_frame, -2, -4),
                                                 calobject_ref)

        if cameras_Rt_plot_ref is not None:
            # I checked earlier that if separate rt_ref_frame or points are
            # given, then only a single cameras_Rt_plot_ref may be given.
            # They're all the same, so I use 0 here arbitrarily
            calobject_ref = \
                mrcal.transform_point_Rt(cameras_Rt_plot_ref[i_model_with_optimization_inputs if i_model_with_optimization_inputs is not None else 0],
                                         calobject_ref)


        # if icam_highlight is not None:
        #     # shape=(Nobservations, object_height_n, object_width_n, 2)
        #     calobject_cam = nps.transform_point_Rt( models[icam_highlight].Rt_cam_ref(), calobject_ref )

        #     print("double-check this. I don't broadcast over the intrinsics anymore")
        #     err = observations[i_observations, ...] - mrcal.project(calobject_cam, *models[icam_highlight].intrinsics())
        #     err = nps.clump(err, n=-3)
        #     rms = np.mag(err) / (object_height_n*object_width_n))
        #     # igood = rms <  0.4
        #     # ibad  = rms >= 0.4
        #     # rms[igood] = 0
        #     # rms[ibad] = 1
        #     calobject_ref = nps.glue( calobject_ref,
        #                               nps.dummy( nps.mv(rms, -1, -3) * np.ones((object_height_n,object_width_n)),
        #                                          -1 ),
        #                               axis = -1)

        # calobject_ref shape: (3, Nframes, object_height_n*object_width_n).
        # This will broadcast nicely
        calobject_ref = nps.clump( nps.mv(calobject_ref, -1, -4), n=-2)

        # if icam_highlight is not None:
        #     calobject_curveopts = {'with':'lines palette', 'tuplesize': 4}
        # else:
        calobject_curveopts = {'with':'lines', 'tuplesize': 3}

        return [tuple(list(calobject_ref) + [calobject_curveopts,])]


    def gen_curves_points(points):
        if points is None or len(points) == 0:
            return []

        if cameras_Rt_plot_ref is not None:
            # I checked earlier that if separate rt_ref_frame or points are
            # given, then only a single cameras_Rt_plot_ref may be given.
            # They're all the same, so I use 0 here arbitrarily
            points = \
                mrcal.transform_point_Rt(cameras_Rt_plot_ref[i_model_with_optimization_inputs if i_model_with_optimization_inputs is not None else 0],
                                         points)

        if point_labels is not None:

            # all the non-fixed point indices
            ipoint_not_labeled = np.ones( (len(points),), dtype=bool)
            ipoint_not_labeled[np.array(list(point_labels.keys()))] = False

            return \
                [ (points[ipoint_not_labeled],
                   dict(tuplesize = -3,
                        _with = 'points',
                        legend = 'points')) ] + \
                [ (points[ipoint],
                   dict(tuplesize = -3,
                        _with = 'points',
                        legend = point_labels[ipoint])) \
                  for ipoint in point_labels.keys() ]

        else:
            return [ (points, dict(tuplesize = -3,
                                   _with = 'points',
                                   legend = 'points')) ]

    curves_cameras    = gen_curves_cameras()
    curves_calobjects = gen_curves_calobjects()
    curves_points     = gen_curves_points(points)

    kwargs = dict(kwargs)
    gp.add_plot_option(kwargs,
                       xlabel = 'x',
                       ylabel = 'y',
                       zlabel = 'z',
                       overwrite = False)

    plot_options = \
        dict(_3d=1,
             square=1,
             **kwargs)

    if 'title' not in plot_options:
        title = 'Camera geometry'
        if extratitle is not None:
            title += ": " + extratitle
        plot_options['title'] = title

    data_tuples = curves_points + curves_cameras + curves_calobjects

    if not return_plot_args:
        plot = gp.gnuplotlib(**plot_options)
        plot.plot(*data_tuples)
        return plot
    return (data_tuples, plot_options)


def _options_heatmap_with_contours( plotoptions, # we update this on output

                                    *,
                                    contour_min           = 0,
                                    contour_max,
                                    contour_increment     = None,
                                    imagersize,
                                    gridn_width,
                                    gridn_height,
                                    do_contours           = True,
                                    # None means "no labels"
                                    contour_labels_styles = 'boxed',
                                    contour_labels_font   = None):
    r'''Update plotoptions, return curveoptions for a contoured heat map'''

    import gnuplotlib as gp

    gp.add_plot_option(plotoptions,
                       'set',
                       ('view equal xy',
                        'view map'))

    if do_contours:
        if contour_increment is None:
            # Compute a "nice" contour increment. I pick a round number that gives
            # me a reasonable number of contours

            Nwant = 10
            increment = (contour_max - contour_min)/Nwant

            # I find the nearest 1eX or 2eX or 5eX
            base10_floor = np.power(10., np.floor(np.log10(increment)))

            # Look through the options, and pick the best one
            m   = np.array((1., 2., 5., 10.))
            err = np.abs(m * base10_floor - increment)
            contour_increment = -m[ np.argmin(err) ] * base10_floor

        gp.add_plot_option(plotoptions,
                           'set',
                           ('style textbox opaque',
                            'contour base',
                            f'cntrparam levels incremental {contour_max},{contour_increment},{contour_min}'))

        if contour_labels_font is not None:
            gp.add_plot_option(plotoptions,
                               'set',
                               f'cntrlabel font "{contour_labels_font}"' )

        plotoptions['cbrange'] = [contour_min, contour_max]

        # I plot 3 times:
        # - to make the heat map
        # - to make the contours
        # - to make the contour labels
        if contour_labels_styles is not None:
            _with = np.array(('image',
                              'lines nosurface',
                              f'labels {contour_labels_styles} nosurface'))
        else:
            _with = np.array(('image',
                              'lines nosurface'))
    else:
        _with = 'image'

    gp.add_plot_option(plotoptions, 'unset', 'key')
    plotoptions['_3d']     = True
    plotoptions['_xrange'] = [0,             imagersize[0]]
    plotoptions['_yrange'] = [imagersize[1], 0]
    plotoptions['ascii']   = True # needed for imagergrid_using to work

    gp.add_plot_option(plotoptions, 'unset', 'grid')

    return \
        dict( tuplesize=3,
              legend = "", # needed to force contour labels
              using = imagergrid_using(imagersize, gridn_width, gridn_height),
              _with=_with)


def fitted_gaussian_equation(*,
                             binwidth,
                             x     = None,
                             mean  = None,
                             sigma = None,
                             N     = None,
                             legend = None):
    r'''Get an 'equation' gnuplotlib expression for a gaussian curve fitting some data

SYNOPSIS

    import gnuplotlib as gp
    import numpy as np

    ...

    # x is a one-dimensional array with samples that we want to compare to a
    # normal distribution. For instance:
    #   x = np.random.randn(10000) * 0.1 + 10

    binwidth = 0.01

    equation = \
        mrcal.fitted_gaussian_equation(x        = x,
                                       binwidth = binwidth)

    gp.plot(x,
            histogram       = True,
            binwidth        = binwidth,
            equation_above  = equation)

    # A plot pops ups displaying a histogram of the array x overlaid by an ideal
    # gaussian probability density function. This PDF corresponds to the mean
    # and standard deviation of the data, and takes into account the histogram
    # parameters to overlay nicely on top of it

Overlaying a ideal PDF on top of an empirical histogram requires a bit of math
to figure out the proper vertical scaling of the plotted PDF that would line up
with the histogram. This is evaluated by this function.

This function can be called in one of two ways:

- Passing the data in the 'x' argument. The statistics are computed from the
  data, and there's no reason to pass 'mean', 'sigma', 'N'

- Passing the statistics 'mean', 'sigma', 'N' instead of the data 'x'. This is
  useful to compare an empirical histogram with idealized distributions that are
  expected to match the data, but do not. For instance, we may want to plot the
  expected standard deviation that differs from the observed standard deviation

ARGUMENTS

- binwidth: the width of each bin in the histogram we will plot. This is the
  only required argument

- x: one-dimensional numpy array containing the data that will be used to
  construct the histogram. If given, (mean, sigma, N) must all NOT be given:
  they will be computed from x. If omitted, then all those must be given instead

- mean: mean the gaussian to plot. Must be given if and only if x is not given

- sigma: standard deviation of the gaussian to plot. Must be given if and only
  if x is not given

- N: the number of values in the dataset in the histogram. Must be given if and
  only if x is not given

- legend: string containing the legend of the curve in the plot. May be omitted
  or None to leave the curve unlabelled

RETURNED VALUES

String passable to gnuplotlib in the 'equation' or 'equation_above' plot option

    '''
    # I want to plot a PDF of a normal distribution together with the
    # histogram to get a visual comparison. This requires a scaling on
    # either the PDF or the histogram. I plot a scaled pdf:
    #
    #   f = k*pdf = k * exp(-x^2 / (2 s^2)) / sqrt(2*pi*s^2)
    #
    # I match up the size of the central bin of the histogram (-binwidth/2,
    # binwidth/2):
    #
    #   bin(0) ~ k*pdf(0) ~ pdf(0) * N * binwidth
    #
    # So k = N*binwdith should work. I can do this more precisely:
    #
    #   bin(0) ~ k*pdf(0) ~
    #     = N * integral( pdf(x) dx,                                -binwidth/2, binwidth/2)
    #     = N * integral( exp(-x^2 / (2 s^2)) / sqrt( 2*pi*s^2) dx, -binwidth/2, binwidth/2)
    # ->k = N * integral( exp(-x^2 / (2 s^2)) / sqrt( 2*pi*s^2) dx, -binwidth/2, binwidth/2) / pdf(0)
    #     = N * integral( exp(-x^2 / (2 s^2)) dx,                   -binwidth/2, binwidth/2)
    #     = N * integral( exp(-(x/(sqrt(2) s))^2) dx )
    #
    # Let u  = x/(sqrt(2) s)
    #     du = dx/(sqrt(2) s)
    #     u(x = binwidth/2) = binwidth/(s 2sqrt(2)) ->
    #
    #   k = N * sqrt(2) s * integral( exp(-u^2) du )
    #     = N*sqrt(2pi) s * erf(binwidth / (s 2*sqrt(2)))
    #
    # for low x erf(x) ~ 2x/sqrt(pi). So if binwidth << sigma
    # k = N*sqrt(2pi) s * erf(binwidth / (s 2*sqrt(2)))
    #   ~ N*sqrt(2pi) s * (binwidth/(s 2*sqrt(2))) *2 / sqrt(pi)
    #   ~ N binwidth

    from scipy.special import erf

    if x is not None:
        if mean is not None or sigma is not None or N is not None:
            raise Exception("x was given. So none of (mean,sigma,N) should have been given")
        sigma = np.std(x)
        mean  = np.mean(x)
        N     = len(x)
    else:
        if mean is None or sigma is None or N is None:
            raise Exception("x was not given. So all of (mean,sigma,N) should have been given")

    var = sigma*sigma
    k   = N * np.sqrt(2.*np.pi) * sigma * erf(binwidth/(2.*np.sqrt(2)*sigma))

    if legend is None:
        title = 'notitle'
    else:
        title = f'title "{legend}"'
    return \
        f'{k}*exp(-(x-{mean})*(x-{mean})/(2.*{var})) / sqrt(2.*pi*{var}) {title} with lines lw 2'




def _append_observation_visualizations(plot_data_args,
                                       *,
                                       model,
                                       legend_prefix = '',
                                       pointtype,
                                       _2d,
                                       # for splined models
                                       reproject_to_stereographic = False):
    optimization_inputs = model.optimization_inputs()
    if optimization_inputs is None:
        raise Exception("mrcal.show_...(observations=True) requires optimization_inputs to be available")

    q_cam_boards_inliers  = None
    q_cam_boards_outliers = None
    observations_board = optimization_inputs.get('observations_board')
    if observations_board is not None:
        ifcice = optimization_inputs['indices_frame_camintrinsics_camextrinsics']
        mask_observation = (ifcice[:,1] == model.icam_intrinsics())
        # shape (N,3)
        observations_board = nps.clump(observations_board[mask_observation], n=3)
        mask_inliers  = observations_board[...,2] > 0

        q_cam_boards_inliers  = observations_board[ mask_inliers,:2]
        q_cam_boards_outliers = observations_board[~mask_inliers,:2]

    q_cam_points_inliers  = None
    q_cam_points_outliers = None
    observations_point = optimization_inputs.get('observations_point')
    if observations_point is not None:
        ipcice = optimization_inputs['indices_point_camintrinsics_camextrinsics']
        mask_observation = (ipcice[:,1] == model.icam_intrinsics())
        # shape (N,3)
        observations_point = observations_point[mask_observation]
        mask_inliers       = observations_point[...,2] > 0

        q_cam_points_inliers  = observations_point[ mask_inliers,:2]
        q_cam_points_outliers = observations_point[~mask_inliers,:2]

    # Disabled for now. I see a legend entry for each broadcasted slice,
    # which isn't what I want
    #
    # if len(q_cam_calobjects):
    #     plot_data_args.append( ( nps.clump(q_cam_calobjects[...,0], n=-2),
    #                              nps.clump(q_cam_calobjects[...,1], n=-2) ) +
    #                            ( () if _2d else
    #                              (np.zeros((q_cam_calobjects.shape[-2]*
    #                                         q_cam_calobjects.shape[-3],)),)) +
    #                            ( dict( tuplesize = 2 if _2d else 3,
    #                                    _with     = f'lines lc "black"' + ("" if _2d else ' nocontour'),
    #                                    legend    = f"{legend_prefix} board sequences"),))

    for q,color,what in ( (q_cam_boards_inliers,  "black", "board inliers"),
                          (q_cam_points_inliers,  "black", "point inliers"),
                          (q_cam_boards_outliers, "red",   "board outliers"),
                          (q_cam_points_outliers, "red",   "point outliers"), ):
        if q is not None and len(q) > 0:

            if reproject_to_stereographic:
                q = mrcal.project_stereographic( \
                        mrcal.unproject(q,
                                        *model.intrinsics()))

            if pointtype < 0:
                _with = f'dots lc "{color}"'
            else:
                _with = f'points lc "{color}" pt {pointtype}'
            if not _2d:
                _with += ' nocontour'
            plot_data_args.append( ( q[...,0],
                                     q[...,1] ) +
                                   ( () if _2d else ( np.zeros(q.shape[:-1]), )) +
                                   ( dict( tuplesize = 2 if _2d else 3,
                                           _with     = _with,
                                           legend    = f'{legend_prefix}{what}'), ))


def show_projection_diff(models,
                         *,
                         implied_Rt10 = None,
                         gridn_width  = 60,
                         gridn_height = None,

                         observations            = False,
                         valid_intrinsics_region = False,
                         intrinsics_only         = False,
                         distance                = None,

                         use_uncertainties= True,
                         focus_center     = None,
                         focus_radius     = -1.,

                         vectorfield      = False,
                         vectorscale      = 1.0,
                         directions       = False,
                         cbmax            = 4,
                         contour_increment     = None,
                         contour_labels_styles = 'boxed',
                         contour_labels_font   = None,
                         extratitle       = None,
                         return_plot_args = False,
                         **kwargs):
    r'''Visualize the difference in projection between N models

SYNOPSIS

    models = ( mrcal.cameramodel('cam0-dance0.cameramodel'),
               mrcal.cameramodel('cam0-dance1.cameramodel') )

    mrcal.show_projection_diff(models)

    # A plot pops up displaying the projection difference between the two models

The operation of this tool is documented at
https://mrcal.secretsauce.net/differencing.html

This function visualizes the results of mrcal.projection_diff()

It is often useful to compare the projection behavior of two camera models. For
instance, one may want to validate a calibration by comparing the results of two
different chessboard dances. Or one may want to evaluate the stability of the
intrinsics in response to mechanical or thermal stresses.

In the most common case we're given exactly 2 models to compare. We then display
the projection difference as either a vector field or a heat map. If we're given
more than 2 models, then a vector field isn't possible and we instead display as
a heatmap the standard deviation of the differences between models 1..N and
model0.

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

The details of how the comparison is computed, and the meaning of the arguments
controlling this, are in the docstring of mrcal.projection_diff().

ARGUMENTS

- models: iterable of mrcal.cameramodel objects we're comparing. Usually there
  will be 2 of these, but more than 2 is possible. The intrinsics are used; the
  extrinsics are NOT.

- implied_Rt10: optional transformation to use to line up the camera coordinate
  systems. Most of the time we want to estimate this transformation, so this
  should be omitted or None. Currently this is supported only if exactly two
  models are being compared.

- gridn_width: optional value, defaulting to 60. How many points along the
  horizontal gridding dimension

- gridn_height: how many points along the vertical gridding dimension. If None,
  we compute an integer gridn_height to maintain a square-ish grid:
  gridn_height/gridn_width ~ imager_height/imager_width

- observations: optional value, defaulting to False. If observations: we overlay
  calibration-time observations on top of the difference plot. We should then
  see that more data produces more consistent results. If a special value of
  'dots' is passed, the observations are plotted as dots instead of points

- valid_intrinsics_region: optional boolean, defaulting to False. If True, we
  overlay the valid-intrinsics regions onto the plot. If the valid-intrinsics
  regions aren't available, we will silently omit them

- intrinsics_only: optional boolean, defaulting to False. If True: we evaluate
  the intrinsics of each lens in isolation by assuming that the coordinate
  systems of each camera line up exactly

- distance: optional value, defaulting to None. Has an effect only if not
  intrinsics_only. The projection difference varies depending on the range to
  the observed world points, with the queried range set in this 'distance'
  argument. If None (the default) we look out to infinity. We can compute the
  implied-by-the-intrinsics transformation off multiple distances if they're
  given here as an iterable. This is especially useful if we have uncertainties,
  since then we'll emphasize the best-fitting distances. If multiple distances
  are given, the generated plot displays the difference using the FIRST distance
  in the list

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

- vectorfield: optional boolean, defaulting to False. By default we produce a
  heat map of the projection differences. If vectorfield: we produce a vector
  field instead. This is more busy, and is often less clear at first glance, but
  unlike a heat map, this shows the directions of the differences in addition to
  the magnitude. This is only valid if we're given exactly two models to compare

- vectorscale: optional value, defaulting to 1.0. Applicable only if
  vectorfield. The magnitude of the errors displayed in the vector field is
  often very small, and impossible to make out when looking at the whole imager.
  This argument can be used to scale all the displayed vectors to improve
  legibility.

- directions: optional boolean, defaulting to False. By default the plot is
  color-coded by the magnitude of the difference vectors. If directions: we
  color-code by the direction instead. This is especially useful if we're
  plotting a vector field. This is only valid if we're given exactly two models
  to compare

- cbmax: optional value, defaulting to 4.0. Sets the maximum range of the color
  map
- contour_increment: optional value, defaulting to None. If given, this will be
  used as the distance between adjacent contours. If omitted of None, a
  reasonable value will be estimated

- contour_labels_styles: optional string, defaulting to 'boxed'. The style of
  the contour labels. This will be passed to gnuplot as f"with labels
  {contour_labels_styles} nosurface". Can be used to box/unbox the label, set
  the color, etc. To change the font use contour_labels_font. Set to None to
  omit the labels entirely

- contour_labels_font: optional string, defaulting to None, If given, this is
  the font string for the contour labels. Will be passed to gnuplot as f'set
  cntrlabel font "{contour_labels_font}"'

- extratitle: optional string to include in the title of the resulting plot.
  Used to extend the default title string. If kwargs['title'] is given, it is
  used directly, and the extratitle is ignored

- return_plot_args: boolean defaulting to False. if return_plot_args: we return
  a (data_tuples, plot_options) tuple instead of making the plot. The plot can
  then be made with gp.plot(*data_tuples, **plot_options). Useful if we want to
  include this as a part of a more complex plot

- **kwargs: optional arguments passed verbatim as plot options to gnuplotlib.
  Useful to make hardcopies, etc

RETURNED VALUE

A tuple:

- if not return_plot_args (the usual path): the gnuplotlib plot object. The plot
  disappears when this object is destroyed (by the garbage collection, for
  instance), so save this returned plot object into a variable, even if you're
  not going to be doing anything with this object.

  if return_plot_args: a (data_tuples, plot_options) tuple. The plot can then be
  made with gp.plot(*data_tuples, **plot_options). Useful if we want to include
  this as a part of a more complex plot

- Rt10: the geometric Rt transformation in an array of shape (...,4,3). This is
  the relative transformation we ended up using, which is computed using the
  logic above (using intrinsics_only and focus_radius). if len(models)>2: this
  is an array of shape (len(models)-1,4,3), with slice i representing the
  transformation between camera 0 and camera i+1.

    '''

    if len(models) < 2:
        raise Exception("At least 2 models are required to compute the diff")


    import gnuplotlib as gp

    if 'title' not in kwargs:
        if intrinsics_only:
            title_note = "using an identity extrinsics transform"
        elif focus_radius == 0:
            title_note = "using given extrinsics transform"
        else:
            distance_string = "infinity" if distance is None else f"distance={distance}"

            using_uncertainties_string = f"{'' if use_uncertainties else 'not '}using uncertainties"
            title_note = f"computing the extrinsics transform {using_uncertainties_string} from data at {distance_string}"

        # should say something about the focus too, but it's already too long
        # elif focus_radius > 2*(W+H):
        #     where = "extrinsics transform fitted everywhere"
        # else:
        #     where = "extrinsics transform fit looking at {} with radius {}". \
        #         format('the imager center' if focus_center is None else focus_center,
        #                focus_radius)

        title = f"Diff looking at {len(models)} models, {title_note}"
        if extratitle is not None:
            title += ": " + extratitle
        kwargs['title'] = title

    if vectorfield:
        if len(models) > 2:
            raise Exception("I can only plot a vectorfield when looking at exactly 2 models. Instead I have {}". \
                            format(len(models)))

    if directions and len(models) > 2:
        raise Exception("I can only color-code by directions when looking at exactly 2 models. Instead I have {}". \
                        format(len(models)))

    # Now do all the actual work
    difflen,diff,q0,Rt10 = mrcal.projection_diff(models,
                                                 implied_Rt10      = implied_Rt10,
                                                 gridn_width       = gridn_width,
                                                 gridn_height      = gridn_height,
                                                 intrinsics_only   = intrinsics_only,
                                                 distance          = distance,
                                                 use_uncertainties = use_uncertainties,
                                                 focus_center      = focus_center,
                                                 focus_radius      = focus_radius)
    # difflen,diff have shape (len(distance), ...) if multiple distances are
    # given. In this case I display the difference using the FIRST distance in
    # the list
    # shape (Nheight, Nwidth)
    if difflen is not None and difflen.ndim > 2:
        difflen = nps.clump(difflen, n=difflen.ndim-2)[0]
    # shape (Nheight, Nwidth,2)
    if diff is not None and diff.ndim > 3:
        diff = nps.clump(diff, n=diff.ndim-3)[0]



    plot_options = kwargs

    if vectorfield:
        # Not plotting a matrix image. I collapse (Nheight, Nwidth, ...) to (Nheight*Nwidth, ...)
        if q0      is not None: q0      = nps.clump(q0,      n=2)
        if difflen is not None: difflen = nps.clump(difflen, n=2)
        if diff    is not None: diff    = nps.clump(diff,    n=2)

    if directions:
        gp.add_plot_option(plot_options,
                           cbrange = [-180.,180.],
                           _set = 'palette defined ( 0 "#00ffff", 0.5 "#80ffff", 1 "#ffffff") model HSV')
        color = 180./np.pi * np.arctan2(diff[...,1], diff[...,0])
    else:
        gp.add_plot_option(plot_options,
                           cbrange = [0,cbmax])
        color = difflen

    # Any invalid values (nan or inf) are set to an effectively infinite
    # difference
    color[~np.isfinite(color)] = 1e6

    if vectorfield:
        # The mrcal.projection_diff() call made sure they're the same for all
        # the models
        W,H=models[0].imagersize()

        gp.add_plot_option(plot_options,
                           square   = 1,
                            _xrange = [0,W],
                            _yrange = [H,0])

        curve_options = dict(_with='vectors filled palette',
                             tuplesize=5)
        plot_data_args = \
            [ (q0  [:,0], q0  [:,1],
               diff[:,0] * vectorscale, diff[:,1] * vectorscale,
               color,
               curve_options) ]
    else:
        curve_options = \
            _options_heatmap_with_contours(
                # update these plot options
                kwargs,

                contour_max           = cbmax,
                contour_increment     = contour_increment,
                imagersize            = models[0].imagersize(),
                gridn_width           = gridn_width,
                gridn_height          = gridn_height,
                contour_labels_styles = contour_labels_styles,
                contour_labels_font   = contour_labels_font,
                do_contours           = not directions)

        plot_data_args = [ (color, curve_options) ]

    if valid_intrinsics_region:
        valid_region0 = models[0].valid_intrinsics_region()
        if valid_region0 is not None:
            if vectorfield:
                # 2d plot
                plot_data_args.append( (valid_region0[:,0], valid_region0[:,1],
                                        dict(_with = 'lines lw 4 lc "green"',
                                             legend = "valid region of 1st camera")) )
            else:
                # 3d plot
                plot_data_args.append( (valid_region0[:,0], valid_region0[:,1], valid_region0[:,0]*0,
                                        dict(_with = 'lines lw 4 lc "green" nocontour',
                                             legend = "valid region of 1st camera")) )

        valid_region1 = models[1].valid_intrinsics_region()
    else:
        valid_region0 = None
        valid_region1 = None

    if len(models) == 2 and valid_region1 is not None:
        # The second camera has a valid region, and I should plot it. This has
        # more complexity: each point on the contour of the valid region of the
        # second camera needs to be transformed to the coordinate system of the
        # first camera to make sense. The transformation is complex, and
        # straight lines will not remain straight. I thus resample the polyline
        # more densely.
        if not intrinsics_only:

            v1 = mrcal.unproject(mrcal.utils._densify_polyline(valid_region1, spacing = 50),
                                 *models[1].intrinsics(),
                                 normalize = True)

            if distance is not None:
                try:    v1 *= distance
                except: v1 *= distance[0]

            valid_region1 = mrcal.project( mrcal.transform_point_Rt( mrcal.invert_Rt(Rt10),
                                                                     v1 ),
                                           *models[0].intrinsics() )

        if vectorfield:
            # 2d plot
            plot_data_args.append( (valid_region1[:,0], valid_region1[:,1],
                                    dict(_with = 'lines lw 3 lc "gray80"',
                                         legend = "valid region of 2nd camera")) )
        else:
            # 3d plot
            plot_data_args.append( (valid_region1[:,0], valid_region1[:,1], valid_region1[:,0]*0,
                                    dict(_with = 'lines lw 3 lc "gray80" nocontour',
                                         legend = "valid region of 2nd camera")) )

    if observations:
        for i in range(len(models)):
            _append_observation_visualizations(plot_data_args,
                                               model         = models[i],
                                               legend_prefix = f"Camera {i} ",
                                               pointtype     = -1 if observations == 'dots' else (1+i),
                                               _2d           = bool(vectorfield))

    data_tuples = plot_data_args

    if not return_plot_args:
        plot = gp.gnuplotlib(**plot_options)
        plot.plot(*data_tuples)
        return plot, Rt10
    return (data_tuples, plot_options), Rt10


def show_stereo_pair_diff(model_pairs,
                          *,
                          gridn_width  = 60,
                          gridn_height = None,

                          observations            = False,
                          valid_intrinsics_region = False,
                          distance                = None,

                          vectorfield      = False,
                          vectorscale      = 1.0,
                          cbmax            = 4,
                          extratitle       = None,
                          return_plot_args = False,
                          **kwargs):
    r'''Visualize the difference in projection between N model pairs

SYNOPSIS

    models = ( mrcal.cameramodel('cam0-dance0.cameramodel'),
               mrcal.cameramodel('cam0-dance1.cameramodel') )

    mrcal.show_stereo_pair_diff(models)

    # A plot pops up displaying the projection difference between the two models

The operation of this tool is documented at
https://mrcal.secretsauce.net/differencing.html

This function visualizes the results of mrcal.stereo_pair_diff()

It is often useful to compare the projection behavior of two camera models. For
instance, one may want to validate a calibration by comparing the results of two
different chessboard dances. Or one may want to evaluate the stability of the
intrinsics in response to mechanical or thermal stresses.

In the most common case we're given exactly 2 models to compare. We then display
the projection difference as either a vector field or a heat map. If we're given
more than 2 models, then a vector field isn't possible and we instead display as
a heatmap the standard deviation of the differences between models 1..N and
model0.

The top-level operation of this function:

- Grid the imager
- Unproject each point in the grid using one camera model
- Apply a transformation to map this point from one camera's coord system to the
  other. How we obtain this transformation is described below
- Project the transformed points to the other camera
- Look at the resulting pixel difference in the reprojection

The details of how the comparison is computed, and the meaning of the arguments
controlling this, are in the docstring of mrcal.stereo_pair_diff().

ARGUMENTS

- models: iterable of mrcal.cameramodel objects we're comparing. Usually there
  will be 2 of these, but more than 2 is possible. The intrinsics are used; the
  extrinsics are NOT.

- gridn_width: optional value, defaulting to 60. How many points along the
  horizontal gridding dimension

- gridn_height: how many points along the vertical gridding dimension. If None,
  we compute an integer gridn_height to maintain a square-ish grid:
  gridn_height/gridn_width ~ imager_height/imager_width

- observations: optional value, defaulting to False. If observations: we overlay
  calibration-time observations on top of the difference plot. We should then
  see that more data produces more consistent results. If a special value of
  'dots' is passed, the observations are plotted as dots instead of points

- valid_intrinsics_region: optional boolean, defaulting to False. If True, we
  overlay the valid-intrinsics regions onto the plot. If the valid-intrinsics
  regions aren't available, we will silently omit them

- distance: optional value, defaulting to None. Has an effect only if not
  intrinsics_only. The projection difference varies depending on the range to
  the observed world points, with the queried range set in this 'distance'
  argument. If None (the default) we look out to infinity

- vectorfield: optional boolean, defaulting to False. By default we produce a
  heat map of the projection differences. If vectorfield: we produce a vector
  field instead. This is more busy, and is often less clear at first glance, but
  unlike a heat map, this shows the directions of the differences in addition to
  the magnitude. This is only valid if we're given exactly two models to compare

- vectorscale: optional value, defaulting to 1.0. Applicable only if
  vectorfield. The magnitude of the errors displayed in the vector field is
  often very small, and impossible to make out when looking at the whole imager.
  This argument can be used to scale all the displayed vectors to improve
  legibility.

- cbmax: optional value, defaulting to 4.0. Sets the maximum range of the color
  map

- extratitle: optional string to include in the title of the resulting plot.
  Used to extend the default title string. If kwargs['title'] is given, it is
  used directly, and the extratitle is ignored

- return_plot_args: boolean defaulting to False. if return_plot_args: we return
  a (data_tuples, plot_options) tuple instead of making the plot. The plot can
  then be made with gp.plot(*data_tuples, **plot_options). Useful if we want to
  include this as a part of a more complex plot

- **kwargs: optional arguments passed verbatim as plot options to gnuplotlib.
  Useful to make hardcopies, etc

RETURNED VALUE

A tuple:

- if not return_plot_args (the usual path): the gnuplotlib plot object. The plot
  disappears when this object is destroyed (by the garbage collection, for
  instance), so save this returned plot object into a variable, even if you're
  not going to be doing anything with this object.

  if return_plot_args: a (data_tuples, plot_options) tuple. The plot can then be
  made with gp.plot(*data_tuples, **plot_options). Useful if we want to include
  this as a part of a more complex plot

- Rt10: the geometric Rt transformation in an array of shape (...,4,3). This is
  the relative transformation we ended up using, which is computed using the
  logic above (using intrinsics_only and focus_radius). if len(models)>2: this
  is an array of shape (len(models)-1,4,3), with slice i representing the
  transformation between camera 0 and camera i+1.

    '''

    if len(model_pairs) < 2:
        raise Exception("At least 2 model_pairs are required to compute the diff")


    import gnuplotlib as gp

    if 'title' not in kwargs:
        title = f"Diff looking at {len(model_pairs)} model_pairs"
        if extratitle is not None:
            title += ": " + extratitle
        kwargs['title'] = title

    if vectorfield:
        if len(model_pairs) > 2:
            raise Exception("I can only plot a vectorfield when looking at exactly 2 model_pairs. Instead I have {}". \
                            format(len(model_pairs)))

    # Now do all the actual work
    difflen,diff,q0 = mrcal.stereo_pair_diff(model_pairs,
                                             gridn_width       = gridn_width,
                                             gridn_height      = gridn_height,
                                             distance          = distance)
    # shape (Nheight, Nwidth)
    if difflen is not None and difflen.ndim > 2:
        difflen = nps.clump(difflen, n=difflen.ndim-2)[0]
    # shape (Nheight, Nwidth,2)
    if diff is not None and diff.ndim > 3:
        diff = nps.clump(diff, n=diff.ndim-3)[0]



    plot_options = kwargs

    if vectorfield:
        # Not plotting a matrix image. I collapse (Nheight, Nwidth, ...) to (Nheight*Nwidth, ...)
        if q0      is not None: q0      = nps.clump(q0,      n=2)
        if difflen is not None: difflen = nps.clump(difflen, n=2)
        if diff    is not None: diff    = nps.clump(diff,    n=2)

    gp.add_plot_option(plot_options,
                       cbrange = [0,cbmax])
    color = difflen

    # Any invalid values (nan or inf) are set to an effectively infinite
    # difference
    color[~np.isfinite(color)] = 1e6

    if vectorfield:
        # The mrcal.stereo_pair_diff() call made sure they're the same for all
        # the model_pairs
        W,H=model_pairs[0][0].imagersize()

        gp.add_plot_option(plot_options,
                           square   = 1,
                            _xrange = [0,W],
                            _yrange = [H,0])

        curve_options = dict(_with='vectors filled palette',
                             tuplesize=5)
        plot_data_args = \
            [ (q0  [:,0], q0  [:,1],
               diff[:,0] * vectorscale, diff[:,1] * vectorscale,
               color,
               curve_options) ]
    else:
        curve_options = \
            _options_heatmap_with_contours(
                # update these plot options
                kwargs,

                contour_max  = cbmax,
                imagersize   = model_pairs[0][0].imagersize(),
                gridn_width  = gridn_width,
                gridn_height = gridn_height,
                do_contours  = True)

        plot_data_args = [ (color, curve_options) ]

    if valid_intrinsics_region:
        raise Exception("finish this")
        valid_region0 = model_pairs[0].valid_intrinsics_region()
        if valid_region0 is not None:
            if vectorfield:
                # 2d plot
                plot_data_args.append( (valid_region0[:,0], valid_region0[:,1],
                                        dict(_with = 'lines lw 4 lc "green"',
                                             legend = "valid region of 1st camera")) )
            else:
                # 3d plot
                plot_data_args.append( (valid_region0[:,0], valid_region0[:,1], valid_region0[:,0]*0,
                                        dict(_with = 'lines lw 4 lc "green" nocontour',
                                             legend = "valid region of 1st camera")) )

        valid_region1 = model_pairs[1].valid_intrinsics_region()
    else:
        valid_region0 = None
        valid_region1 = None

    if len(model_pairs) == 2 and valid_region1 is not None:
        # The second camera has a valid region, and I should plot it. This has
        # more complexity: each point on the contour of the valid region of the
        # second camera needs to be transformed to the coordinate system of the
        # first camera to make sense. The transformation is complex, and
        # straight lines will not remain straight. I thus resample the polyline
        # more densely.
        v1 = mrcal.unproject(mrcal.utils._densify_polyline(valid_region1, spacing = 50),
                             *model_pairs[1].intrinsics(),
                             normalize = True)

        if distance is not None:
            v1 *= distance

        valid_region1 = mrcal.project( mrcal.transform_point_Rt( mrcal.invert_Rt(Rt10),
                                                                 v1 ),
                                       *model_pairs[0].intrinsics() )

        if vectorfield:
            # 2d plot
            plot_data_args.append( (valid_region1[:,0], valid_region1[:,1],
                                    dict(_with = 'lines lw 3 lc "gray80"',
                                         legend = "valid region of 2nd camera")) )
        else:
            # 3d plot
            plot_data_args.append( (valid_region1[:,0], valid_region1[:,1], valid_region1[:,0]*0,
                                    dict(_with = 'lines lw 3 lc "gray80" nocontour',
                                         legend = "valid region of 2nd camera")) )

    if observations:
        raise Exception("finish this")
        for i in range(len(model_pairs)):
            _append_observation_visualizations(plot_data_args,
                                               model         = model_pairs[i],
                                               legend_prefix = f"Camera {i} ",
                                               pointtype     = -1 if observations == 'dots' else (1+i),
                                               _2d           = bool(vectorfield))

    data_tuples = plot_data_args

    if not return_plot_args:
        plot = gp.gnuplotlib(**plot_options)
        plot.plot(*data_tuples)
        return plot
    return (data_tuples, plot_options)


def show_projection_uncertainty(model,
                                *,
                                gridn_width             = 60,
                                gridn_height            = None,

                                observed_pixel_uncertainty = None,

                                observations               = False,
                                valid_intrinsics_region    = False,
                                distance                   = None,
                                isotropic                  = False,
                                method                     = 'mean-pcam',
                                cbmax                      = 3,
                                contour_increment          = None,
                                contour_labels_styles      = 'boxed',
                                contour_labels_font        = None,
                                extratitle                 = None,
                                return_plot_args           = False,
                                **kwargs):
    r'''Visualize the uncertainty in camera projection

SYNOPSIS

    model = mrcal.cameramodel('xxx.cameramodel')

    mrcal.show_projection_uncertainty(model)

    ... A plot pops up displaying the expected projection uncertainty across the
    ... imager

This function uses the expected noise of the calibration-time observations to
estimate the uncertainty of projection of the final model. At calibration time
we estimate

- The intrinsics (lens paramaters) of a number of cameras
- The extrinsics (geometry) of a number of cameras in respect to some reference
  coordinate system
- The poses of observed chessboards and/or the coordinates of the discrete
  points, also in respect to some reference coordinate system

All the coordinate systems move around, and all 3 of these sets of data have
some uncertainty. This tool takes into account all the uncertainties to report
an estimated uncertainty metric. See
https://mrcal.secretsauce.net/uncertainty.html for a detailed description of
the computation.

This function grids the imager, and reports an uncertainty for each point on the
grid. The resulting plot contains a heatmap of the uncertainties for each cell
in the grid, and corresponding contours.

Since the projection uncertainty is based on calibration-time observation
uncertainty, it is sometimes useful to see where the calibration-time
observations were. Pass observations=True to do that.

Since the projection uncertainty is based partly on the uncertainty of the
camera pose, points at different distance from the camera will have different
reported uncertainties EVEN IF THEY PROJECT TO THE SAME PIXEL. The queried
distance is passed in the distance argument. If distance is None (the default)
then we look out to infinity.

For each cell we compute the covariance matrix of the projected (x,y) coords,
and by default we report the worst-direction standard deviation. If isotropic:
we report the RMS standard deviation instead.

ARGUMENTS

- model: the mrcal.cameramodel object being evaluated

- gridn_width: optional value, defaulting to 60. How many points along the
  horizontal gridding dimension

- gridn_height: how many points along the vertical gridding dimension. If None,
  we compute an integer gridn_height to maintain a square-ish grid:
  gridn_height/gridn_width ~ imager_height/imager_width

- observed_pixel_uncertainty: optional value, defaulting to None. The
  uncertainty of the pixel observations being propagated through the solve and
  projection. If omitted or None, this input uncertainty is inferred from the
  residuals at the optimum. Most people should omit this

- observations: optional value, defaulting to False. If observatoins:, we
  overlay calibration-time observations on top of the uncertainty plot. We
  should then see that more data produces more confident results. If a special
  value of 'dots' is passed, the observations are plotted as dots instead of
  points

- valid_intrinsics_region: optional boolean, defaulting to False. If True, we
  overlay the valid-intrinsics region onto the plot. If the valid-intrinsics
  region isn't available, we will silently omit it

- distance: optional value, defaulting to None. The projection uncertainty
  varies depending on the range to the observed point, with the queried range
  set in this 'distance' argument. If None (the default) we look out to
  infinity.

- isotropic: optional boolean, defaulting to False. We compute the full 2x2
  covariance matrix of the projection. The 1-sigma contour implied by this
  matrix is an ellipse, and we use the worst-case direction by default. If we
  want the RMS size of the ellipse instead of the worst-direction size, pass
  isotropic=True.

- method: optional string, defaulting to 'mean-pcam'. Multiple uncertainty
  quantification methods are available. One of ('mean-pcam', 'bestq',
  'cross-reprojection-rrp-Jfp') is selected by this option

- cbmax: optional value, defaulting to 3.0. Sets the maximum range of the color
  map

- contour_increment: optional value, defaulting to None. If given, this will be
  used as the distance between adjacent contours. If omitted of None, a
  reasonable value will be estimated

- contour_labels_styles: optional string, defaulting to 'boxed'. The style of
  the contour labels. This will be passed to gnuplot as f"with labels
  {contour_labels_styles} nosurface". Can be used to box/unbox the label, set
  the color, etc. To change the font use contour_labels_font. Set to None to
  omit the labels entirely

- contour_labels_font: optional string, defaulting to None, If given, this is
  the font string for the contour labels. Will be passed to gnuplot as f'set
  cntrlabel font "{contour_labels_font}"'

- extratitle: optional string to include in the title of the resulting plot.
  Used to extend the default title string. If kwargs['title'] is given, it is
  used directly, and the extratitle is ignored

- return_plot_args: boolean defaulting to False. if return_plot_args: we return
  a (data_tuples, plot_options) tuple instead of making the plot. The plot can
  then be made with gp.plot(*data_tuples, **plot_options). Useful if we want to
  include this as a part of a more complex plot

- **kwargs: optional arguments passed verbatim as plot options to gnuplotlib.
  Useful to make hardcopies, etc

RETURNED VALUE

if not return_plot_args (the usual path): we return the gnuplotlib plot object.
The plot disappears when this object is destroyed (by the garbage collection,
for instance), so save this returned plot object into a variable, even if you're
not going to be doing anything with this object.

if return_plot_args: we return a (data_tuples, plot_options) tuple instead of
making the plot. The plot can then be made with gp.plot(*data_tuples,
**plot_options). Useful if we want to include this as a part of a more complex
plot

    '''

    import gnuplotlib as gp

    known_methods = set(('mean-pcam', 'bestq',
                         'cross-reprojection-rrp-Jfp'),)
    if method not in known_methods:
        raise Exception(f"Unknown uncertainty method: '{method}'. I know about {known_methods}")

    W,H=model.imagersize()
    if gridn_height is None:
        gridn_height = int(round(H/W*gridn_width))

    q    = mrcal.sample_imager( gridn_width, gridn_height, W,H )
    pcam = mrcal.unproject(q, *model.intrinsics(),
                           normalize = True)

    err = mrcal.projection_uncertainty(pcam * (distance if distance is not None else 1.0),
                                       model           = model,
                                       method          = method,
                                       atinfinity      = distance is None,
                                       what            = 'rms-stdev' if isotropic else 'worstdirection-stdev',
                                       observed_pixel_uncertainty = observed_pixel_uncertainty)

    # Any nan or inf uncertainty is set to a very high value. This usually
    # happens if the unproject() call failed, resulting in pcam == 0
    err[~np.isfinite(err)] = 1e6

    if 'title' not in kwargs:
        if distance is None:
            distance_description = ". Looking out to infinity"
        else:
            distance_description = f". Looking out to {distance}m"

        if not isotropic:
            what_description = "Projection"
        else:
            what_description = "Isotropic projection"

        title = f"{what_description} uncertainty (in pixels) based on calibration input noise{distance_description}"
        if extratitle is not None:
            title += ": " + extratitle
        kwargs['title'] = title

    curveoptions = \
        _options_heatmap_with_contours(
            # update these plot options
            kwargs,

            contour_max           = cbmax,
            contour_increment     = contour_increment,
            imagersize            = model.imagersize(),
            gridn_width           = gridn_width,
            gridn_height          = gridn_height,
            contour_labels_styles = contour_labels_styles,
            contour_labels_font   = contour_labels_font)

    plot_data_args = [(err, curveoptions)]

    if valid_intrinsics_region:
        valid_intrinsics_region = model.valid_intrinsics_region()
    else:
        valid_intrinsics_region = None
    if valid_intrinsics_region is not None:
        plot_data_args.append( (valid_intrinsics_region[:,0],
                                valid_intrinsics_region[:,1],
                                np.zeros(valid_intrinsics_region.shape[-2]),
                                dict(_with  = 'lines lw 4 lc "green" nocontour',
                                     legend = "Valid-intrinsics region")) )

    if observations:
        _append_observation_visualizations(plot_data_args,
                                           model         = model,
                                           pointtype     = -1 if observations == 'dots' else 1,
                                           _2d           = False)

    plot_options = kwargs
    data_tuples  = plot_data_args

    if not return_plot_args:
        plot = gp.gnuplotlib(**plot_options)
        plot.plot(*data_tuples)
        return plot
    return (data_tuples, plot_options)



def _observed_hypothesis_points_and_boards_at_calibration_time(model):

    optimization_inputs = model.optimization_inputs()

    p_cam_observed_at_calibration_time = np.zeros((0,3), dtype=float)

    try:
        # [-2] means "inliers"
        p_cam_observed_at_calibration_time = \
            mrcal.hypothesis_board_corner_positions(model.icam_intrinsics(),
                                                    **optimization_inputs)[-2]
    except:
        # no chessboards perhaps
        pass

    points = optimization_inputs.get('points')
    if points is not None:
        indices_point_camintrinsics_camextrinsics = \
            optimization_inputs['indices_point_camintrinsics_camextrinsics']
        mask_thiscam = \
            indices_point_camintrinsics_camextrinsics[...,1] == model.icam_intrinsics()

        ipoint             = indices_point_camintrinsics_camextrinsics[mask_thiscam,0]
        icame              = indices_point_camintrinsics_camextrinsics[mask_thiscam,2]
        observations_point = optimization_inputs['observations_point'][mask_thiscam]

        mask_inliers       = observations_point[...,2] > 0
        ipoint             = ipoint[mask_inliers]
        icame              = icame [mask_inliers]

        icame[icame<0] = -1
        rt_cam_ref = \
            nps.glue( mrcal.identity_rt(),
                      optimization_inputs['rt_cam_ref'],
                      axis = -2 ) \
                      [ icame+1 ]

        rt_cam_ref = optimization_inputs['rt_cam_ref'][icame]

        p_ref_points = points[ipoint]
        p_cam_points = mrcal.transform_point_rt(rt_cam_ref, p_ref_points)

        p_cam_observed_at_calibration_time = \
            nps.glue(p_cam_observed_at_calibration_time,
                     p_cam_points,
                     axis = -2)

    return p_cam_observed_at_calibration_time


def show_projection_uncertainty_vs_distance(model,
                                            *,
                                            where                      = "centroid",
                                            observed_pixel_uncertainty = None,
                                            isotropic                  = False,
                                            method                     = 'mean-pcam',
                                            distance_min               = None,
                                            distance_max               = None,
                                            extratitle                 = None,
                                            return_plot_args           = False,
                                            **kwargs):
    r'''Visualize the uncertainty in camera projection along one observation ray

SYNOPSIS

    model = mrcal.cameramodel('xxx.cameramodel')

    mrcal.show_projection_uncertainty_vs_distance(model)

    ... A plot pops up displaying the expected projection uncertainty along an
    ... observation ray at different distances from the camera

This function is similar to show_projection_uncertainty(). That function
displays the uncertainty at different locations along the imager, for one
observation distance. Conversely, THIS function displays it in one location on
the imager, but at different distances.

This function uses the expected noise of the calibration-time observations to
estimate the uncertainty of projection of the final model. At calibration time
we estimate

- The intrinsics (lens paramaters) of a number of cameras
- The extrinsics (geometry) of a number of cameras in respect to some reference
  coordinate system
- The poses of observed chessboards and/or the coordinates of the discrete
  points, also in respect to some reference coordinate system

All the coordinate systems move around, and all 3 of these sets of data have
some uncertainty. This tool takes into account all the uncertainties to report
an estimated uncertainty metric. See
https://mrcal.secretsauce.net/uncertainty.html for a detailed description of
the computation.

The curve produced by this function has a characteristic shape:

- At low ranges, the camera translation dominates, and the uncertainty increases
  to infinity, as the distance to the camera goes to 0

- As we move away from the camera, the uncertainty drops to a minimum, at around
  the distance where the chessboards were observed

- Past the minimum, the uncertainty climbs to asymptotically approach the
  uncertainty at infinity

ARGUMENTS

- model: the mrcal.cameramodel object being evaluated

- where: optional value, defaulting to "centroid". Indicates the point on the
  imager we're examining. May be one of

  - "center": the center of the imager
  - "centroid": the midpoint of all the points observed at calibration time
  - A numpy array (x,y) indicating the pixel

- observed_pixel_uncertainty: optional value, defaulting to None. The
  uncertainty of the pixel observations being propagated through the solve and
  projection. If omitted or None, this input uncertainty is inferred from the
  residuals at the optimum. Most people should omit this

- isotropic: optional boolean, defaulting to False. We compute the full 2x2
  covariance matrix of the projection. The 1-sigma contour implied by this
  matrix is an ellipse, and we use the worst-case direction by default. If we
  want the RMS size of the ellipse instead of the worst-direction size, pass
  isotropic=True.

- method: optional string, defaulting to 'mean-pcam'. Multiple uncertainty
  quantification methods are available. One of ('mean-pcam', 'bestq',
  'cross-reprojection-rrp-Jfp') is selected by this option

- extratitle: optional string to include in the title of the resulting plot.
  Used to extend the default title string. If kwargs['title'] is given, it is
  used directly, and the extratitle is ignored

- return_plot_args: boolean defaulting to False. if return_plot_args: we return
  a (data_tuples, plot_options) tuple instead of making the plot. The plot can
  then be made with gp.plot(*data_tuples, **plot_options). Useful if we want to
  include this as a part of a more complex plot

- **kwargs: optional arguments passed verbatim as plot options to gnuplotlib.
  Useful to make hardcopies, etc

RETURNED VALUE

if not return_plot_args (the usual path): we return the gnuplotlib plot object.
The plot disappears when this object is destroyed (by the garbage collection,
for instance), so save this returned plot object into a variable, even if you're
not going to be doing anything with this object.

if return_plot_args: we return a (data_tuples, plot_options) tuple instead of
making the plot. The plot can then be made with gp.plot(*data_tuples,
**plot_options). Useful if we want to include this as a part of a more complex
plot

    '''

    import gnuplotlib as gp

    known_methods = set(('mean-pcam',
                         'bestq',
                         'cross-reprojection-rrp-Jfp'),)
    if method not in known_methods:
        raise Exception(f"Unknown uncertainty method: '{method}'. I know about {known_methods}")

    p_cam_observed_at_calibration_time = \
        _observed_hypothesis_points_and_boards_at_calibration_time(model)

    if p_cam_observed_at_calibration_time.size == 0:
        raise Exception("No inlier chessboards or points observed at calibration time")

    if isinstance(where, str):
        if   where == 'center':
            q = (model.imagersize() - 1.) / 2.

            vcam = mrcal.unproject(q, *model.intrinsics(),
                                   normalize = True)

        elif where == 'centroid':
            p    = np.mean(p_cam_observed_at_calibration_time, axis=-2)
            vcam = p / nps.mag(p)

        else:
            raise Exception("'where' should be 'center' or an array specifying a pixel")

    elif isinstance(where, np.ndarray):
        q    = where
        vcam = mrcal.unproject(q, *model.intrinsics(),
                               normalize = True)
    else:
        raise Exception("'where' should be 'center' or an array specifying a pixel")

    # shape (Ndistances)
    if distance_min is None or distance_max is None:
        distance_observed_at_calibration_time = \
            nps.mag(p_cam_observed_at_calibration_time)
        if distance_min is None:
            # / 5 for a bit of extra margin
            distance_min = np.min(distance_observed_at_calibration_time) / 5.
        if distance_max is None:
            # * 10 for a bit of extra margin
            distance_max = np.max(distance_observed_at_calibration_time) * 10.

    distances = np.logspace( np.log10(distance_min),
                             np.log10(distance_max),
                             80 )

    # shape (Ndistances, 3)
    pcam = vcam * nps.dummy(distances, -1)

    # shape (Ndistances)
    uncertainty = \
        mrcal.projection_uncertainty( pcam,
                                      model = model,
                                      method= method,
                                      what  = 'rms-stdev' if isotropic else 'worstdirection-stdev',
                                      observed_pixel_uncertainty = observed_pixel_uncertainty)
    if 'title' not in kwargs:
        if not isotropic:
            what_description = "Projection"
        else:
            what_description = "Isotropic projection"

        title = f"{what_description} uncertainty (in pixels) based on calibration input noise at q = {where}"
        if extratitle is not None:
            title += ": " + extratitle
        kwargs['title'] = title

    plot_options = \
        dict( xlabel   = 'Observation distance',
              ylabel   = 'Projection uncertainty (pixels)',
              _with    = 'lines',
              **kwargs )

    data_tuples = ( distances, uncertainty )

    if not return_plot_args:
        plot = gp.gnuplotlib(**plot_options)
        plot.plot(*data_tuples)
        return plot
    return (data_tuples, plot_options)


def show_distortion_off_pinhole_radial(model,
                                       *,
                                       show_fisheye_projections = False,
                                       extratitle               = None,
                                       return_plot_args         = False,
                                       **kwargs):

    r'''Visualize a lens's deviation from a pinhole projection

SYNOPSIS

    model = mrcal.cameramodel('xxx.cameramodel')

    mrcal.show_distortion_off_pinhole_radial(model)

    ... A plot pops up displaying how much this model deviates from a pinhole
    ... model across the imager in the radial direction

This function treats a pinhole projection as a baseline, and visualizes
deviations from this baseline. So wide lenses will have a lot of reported
"distortion".

This function looks at radial distortion only. Plots a curve showing the
magnitude of the radial distortion as a function of the distance to the center

ARGUMENTS

- model: the mrcal.cameramodel object being evaluated

- show_fisheye_projections: optional boolean defaulting to False. If
  show_fisheye_projections: the radial plots include the behavior of common
  fisheye projections, in addition to the behavior of THIS lens

- extratitle: optional string to include in the title of the resulting plot.
  Used to extend the default title string. If kwargs['title'] is given, it is
  used directly, and the extratitle is ignored

- return_plot_args: boolean defaulting to False. if return_plot_args: we return
  a (data_tuples, plot_options) tuple instead of making the plot. The plot can
  then be made with gp.plot(*data_tuples, **plot_options). Useful if we want to
  include this as a part of a more complex plot

- **kwargs: optional arguments passed verbatim as plot options to gnuplotlib.
  Useful to make hardcopies, etc

RETURNED VALUE

if not return_plot_args (the usual path): we return the gnuplotlib plot object.
The plot disappears when this object is destroyed (by the garbage collection,
for instance), so save this returned plot object into a variable, even if you're
not going to be doing anything with this object.

if return_plot_args: we return a (data_tuples, plot_options) tuple instead of
making the plot. The plot can then be made with gp.plot(*data_tuples,
**plot_options). Useful if we want to include this as a part of a more complex
plot

    '''

    import gnuplotlib as gp

    lensmodel, intrinsics_data = model.intrinsics()

    # plot the radial distortion. For now I only deal with opencv here
    m = re.search("OPENCV([0-9]+)", lensmodel)
    if not m:
        raise Exception("Radial distortion visualization implemented only for OpenCV distortions; for now")
    N = int(m.group(1))

    # OpenCV does this:
    #
    # This is the opencv distortion code in cvProjectPoints2 in
    # calibration.cpp Here x,y are x/z and y/z. OpenCV applies distortion to
    # x/z, y/z and THEN does the ...*f + c thing. The distortion factor is
    # based on r, which is ~ x/z ~ tan(th) where th is the deviation off
    # center
    #
    #         z = z ? 1./z : 1;
    #         x *= z; y *= z;
    #         r2 = x*x + y*y;
    #         r4 = r2*r2;
    #         r6 = r4*r2;
    #         a1 = 2*x*y;
    #         a2 = r2 + 2*x*x;
    #         a3 = r2 + 2*y*y;
    #         cdist = 1 + k[0]*r2 + k[1]*r4 + k[4]*r6;
    #         icdist2 = 1./(1 + k[5]*r2 + k[6]*r4 + k[7]*r6);
    #         xd = x*cdist*icdist2 + k[2]*a1 + k[3]*a2 + k[8]*r2+k[9]*r4;
    #         yd = y*cdist*icdist2 + k[2]*a3 + k[3]*a1 + k[10]*r2+k[11]*r4;
    #         ...
    #         m[i].x = xd*fx + cx;
    #         m[i].y = yd*fy + cy;
    cxy         = intrinsics_data[2:4]
    distortions = intrinsics_data[4:]
    k2 = distortions[0]
    k4 = distortions[1]
    k6 = 0
    if N >= 5:
        k6 = distortions[4]
    numerator = '1. + r*r * ({} + r*r * ({} + r*r * {}))'.format(k2,k4,k6)
    numerator = numerator.replace('r', 'tan(x*pi/180.)')

    if N >= 8:
        denominator = '1. + r*r * ({} + r*r * ({} + r*r * {}))'.format(*distortions[5:8])
        denominator = denominator.replace('r', 'tan(x*pi/180.)')
        scale = '({})/({})'.format(numerator,denominator)
    else:
        scale = numerator

    W,H = model.imagersize()
    x0,x1 = 0, W-1
    y0,y1 = 0, H-1

    q_corners = np.array(((x0,y0),
                          (x0,y1),
                          (x1,y0),
                          (x1,y1)), dtype=float)
    q_centersx  = np.array(((cxy[0],y0),
                            (cxy[0],y1)), dtype=float)
    q_centersy  = np.array(((x0,cxy[1]),
                            (x1,cxy[1])), dtype=float)

    def th_from_q(q, *intrinsics):
        v = mrcal.unproject( q, *intrinsics)
        # some unprojections may be nan (we're looking beyond where the projection
        # is valid), so I explicitly ignore those
        v = v [np.all(np.isfinite(v),  axis=-1)]
        return 180./np.pi * np.arctan2(nps.mag(v [...,:2]), v [...,2])

    th_corners  = th_from_q(q_corners,  lensmodel, intrinsics_data)
    th_centersx = th_from_q(q_centersx, lensmodel, intrinsics_data)
    th_centersy = th_from_q(q_centersy, lensmodel, intrinsics_data)

    # Now the equations. The 'x' value here is "pinhole pixels off center",
    # which is f*tan(th). I plot this model's radial relationship, and that
    # from other common fisheye projections (formulas mostly from
    # https://en.wikipedia.org/wiki/Fisheye_lens)
    equations = [f'180./pi*atan(tan(x*pi/180.) * ({scale})) with lines lw 2 title "THIS model"',
                 'x title "pinhole"']
    if show_fisheye_projections:
        equations += [f'180./pi*atan(2. * tan( x*pi/180. / 2.)) title "stereographic"',
                      f'180./pi*atan(x*pi/180.) title "equidistant"',
                      f'180./pi*atan(2. * sin( x*pi/180. / 2.)) title "equisolid angle"',
                      f'180./pi*atan( sin( x*pi/180. )) title "orthogonal"']

    gp.add_plot_option(kwargs, 'set',
                       ['arrow from {th}, graph 0 to {th}, graph 1 nohead lc "red"'  . \
                        format(th=th) for th in th_centersy] + \
                       ['arrow from {th}, graph 0 to {th}, graph 1 nohead lc "green"'. \
                        format(th=th) for th in th_centersx] + \
                       ['arrow from {th}, graph 0 to {th}, graph 1 nohead lc "blue"' . \
                        format(th=th) for th in th_corners ])

    if N >= 8:
        equations.extend( [numerator   + ' axis x1y2 title "numerator (y2)"',
                           denominator + ' axis x1y2 title "denominator (y2)"',
                           '0 axis x1y2 notitle with lines lw 2' ] )
        gp.add_plot_option(kwargs, 'set', 'y2tics')
        kwargs['y2label'] = 'Rational correction numerator, denominator'

    if 'title' not in kwargs:
        title = 'Radial distortion. Red: x edges. Green: y edges. Blue: corners'
        if extratitle is not None:
            title += ": " + extratitle
        kwargs['title'] = title

    plot_options = \
        dict(equation = equations,
             # any of the unprojections could be nan, so I do the best I can
             _xrange = [0,np.max(np.nan_to_num(nps.glue(th_corners,
                                                        th_centersx,
                                                        th_centersy,
                                                        axis=-1)))
                        * 1.01],
             xlabel = 'Angle off the projection center (deg)',
             ylabel = 'Distorted angle off the projection center',
             **kwargs)
    data_tuples = ()
    if not return_plot_args:
        plot = gp.gnuplotlib(**plot_options)
        plot.plot(*data_tuples)
        return plot
    return (data_tuples, plot_options)


def show_distortion_off_pinhole(model,
                                *,
                                vectorfield              = False,
                                vectorscale              = 1.0,
                                cbmax                    = 25.0,
                                gridn_width              = 60,
                                gridn_height             = None,
                                extratitle               = None,
                                return_plot_args         = False,
                                **kwargs):

    r'''Visualize a lens's deviation from a pinhole projection

SYNOPSIS

    model = mrcal.cameramodel('xxx.cameramodel')

    mrcal.show_distortion_off_pinhole( model )

    ... A plot pops up displaying how much this model deviates from a pinhole
    ... model across the imager

This function treats a pinhole projection as a baseline, and visualizes
deviations from this baseline. So wide lenses will have a lot of reported
"distortion".

ARGUMENTS

- model: the mrcal.cameramodel object being evaluated

- vectorfield: optional boolean, defaulting to False. By default we produce a
  heat map of the differences. If vectorfield: we produce a vector field
  instead

- vectorscale: optional value, defaulting to 1.0. Applicable only if
  vectorfield. The magnitude of the errors displayed in the vector field could
  be small, and difficult to see. This argument can be used to scale all the
  displayed vectors to improve legibility.

- cbmax: optional value, defaulting to 25.0. Sets the maximum range of the color
  map and of the contours if plotting a heat map

- gridn_width: how many points along the horizontal gridding dimension

- gridn_height: how many points along the vertical gridding dimension. If None,
  we compute an integer gridn_height to maintain a square-ish grid:
  gridn_height/gridn_width ~ imager_height/imager_width

- extratitle: optional string to include in the title of the resulting plot.
  Used to extend the default title string. If kwargs['title'] is given, it is
  used directly, and the extratitle is ignored

- return_plot_args: boolean defaulting to False. if return_plot_args: we return
  a (data_tuples, plot_options) tuple instead of making the plot. The plot can
  then be made with gp.plot(*data_tuples, **plot_options). Useful if we want to
  include this as a part of a more complex plot

- **kwargs: optional arguments passed verbatim as plot options to gnuplotlib.
  Useful to make hardcopies, etc

RETURNED VALUE

if not return_plot_args (the usual path): we return the gnuplotlib plot object.
The plot disappears when this object is destroyed (by the garbage collection,
for instance), so save this returned plot object into a variable, even if you're
not going to be doing anything with this object.

if return_plot_args: we return a (data_tuples, plot_options) tuple instead of
making the plot. The plot can then be made with gp.plot(*data_tuples,
**plot_options). Useful if we want to include this as a part of a more complex
plot

    '''

    import gnuplotlib as gp

    lensmodel, intrinsics_data = model.intrinsics()
    imagersize                 = model.imagersize()

    if 'title' not in kwargs:
        title = "Off-pinhole effects of {}".format(lensmodel)
        if extratitle is not None:
            title += ": " + extratitle
        kwargs['title'] = title

    W,H = imagersize
    if gridn_height is None:
        gridn_height = int(round(H/W*gridn_width))

    if not mrcal.lensmodel_metadata_and_config(lensmodel)['has_core']:
        raise Exception("This currently works only with models that have an fxfycxcy core. It might not be required. Take a look at the following code if you want to add support")
    fxy = intrinsics_data[ :2]
    cxy = intrinsics_data[2:4]

    # shape: (Nheight,Nwidth,2). Contains (x,y) rows
    grid  = np.ascontiguousarray(nps.mv(nps.cat(*np.meshgrid(np.linspace(0,W-1,gridn_width),
                                                             np.linspace(0,H-1,gridn_height))),
                                        0,-1),
                                 dtype = float)

    dgrid =  mrcal.project( nps.glue( (grid-cxy)/fxy,
                                    np.ones(grid.shape[:-1] + (1,), dtype=float),
                                    axis = -1 ),
                          lensmodel, intrinsics_data )

    if not vectorfield:
        curveoptions = \
            _options_heatmap_with_contours(
                # update these plot options
                kwargs,

                contour_max  = cbmax,
                imagersize   = imagersize,
                gridn_width  = gridn_width,
                gridn_height = gridn_height)
        delta = dgrid-grid

        # shape: gridn_height,gridn_width. Because numpy (and thus gnuplotlib) want it that
        # way
        distortion = nps.mag(delta)

        data_tuples = ((distortion, curveoptions), )

    else:
        # vectorfield

        # shape: gridn_height*gridn_width,2
        grid  = nps.clump(grid,  n=2)
        dgrid = nps.clump(dgrid, n=2)

        delta = dgrid-grid
        delta *= vectorscale

        kwargs['_xrange']=(-50,W+50)
        kwargs['_yrange']=(H+50, -50)
        kwargs['_set'   ]=['object 1 rectangle from 0,0 to {},{} fillstyle empty'.format(W,H)]
        kwargs['square' ]=True

        if '_set' in kwargs:
            if type(kwargs['_set']) is list: kwargs['_set'].extend(kwargs['_set'])
            else:                            kwargs['_set'].append(kwargs['_set'])
            del kwargs['_set']

        data_tuples = \
            ( (grid[:,0], grid[:,1], delta[:,0], delta[:,1],
               {'with': 'vectors filled',
                'tuplesize': 4,
               }),
              (grid[:,0], grid[:,1],
               {'with': 'points',
                'tuplesize': 2,
               }))

    plot_options = kwargs

    if not return_plot_args:
        plot = gp.gnuplotlib(**plot_options)
        plot.plot(*data_tuples)
        return plot
    return (data_tuples, plot_options)


def show_valid_intrinsics_region(models,
                                 *,
                                 cameranames      = None,
                                 image            = None,
                                 points           = None,
                                 extratitle       = None,
                                 return_plot_args = False,
                                 **kwargs):
    r'''Visualize a model's valid-intrinsics region

SYNOPSIS

    filenames = ('cam0-dance0.cameramodel',
                 'cam0-dance1.cameramodel')

    models = [ mrcal.cameramodel(f) for f in filenames ]

    mrcal.show_valid_intrinsics_region( models,
                                        cameranames = filenames,
                                        image       = 'image.jpg' )

This function displays the valid-intrinsics region in the given camera models.
Multiple models can be passed-in to see their valid-intrinsics regions together.
This is useful to evaluate different calibrations of the same lens. A captured
image can be passed-in to see the regions overlaid on an actual image produced
by the camera.

All given models MUST have a valid-intrinsics region defined. A model may have
an empty region. This cannot be plotted (there's no contour to plot), but the
plot legend will still contain an entry for this model, with a note indicating
its emptiness

This tool produces a gnuplotlib plot. To annotate an image array, call
annotate_image__valid_intrinsics_region() instead

ARGUMENTS

- models: an iterable of mrcal.cameramodel objects we're visualizing. If we're
  looking at just a single model, it can be passed directly in this argument,
  instead of wrapping it into a list.

- cameranames: optional an iterable of labels, one for each model. These will
  appear as the legend in the plot. If omitted, we will simply enumerate the
  models.

- image: optional image to annotate. May be given as an image filename or an
  array of image data. If omitted, we plot the valid-intrinsics region only.

- points: optional array of shape (N,2) of pixel coordinates to plot. If given,
  we show these arbitrary points in our plot. Useful to visualize the feature
  points used in a vision algorithm to see how reliable they are expected to be

- extratitle: optional string to include in the title of the resulting plot.
  Used to extend the default title string. If kwargs['title'] is given, it is
  used directly, and the extratitle is ignored

- return_plot_args: boolean defaulting to False. if return_plot_args: we return
  a (data_tuples, plot_options) tuple instead of making the plot. The plot can
  then be made with gp.plot(*data_tuples, **plot_options). Useful if we want to
  include this as a part of a more complex plot

- **kwargs: optional arguments passed verbatim as plot options to gnuplotlib.
  Useful to make hardcopies, etc

RETURNED VALUE

A tuple:

- if not return_plot_args (the usual path): the gnuplotlib plot object. The plot
  disappears when this object is destroyed (by the garbage collection, for
  instance), so save this returned plot object into a variable, even if you're
  not going to be doing anything with this object.

  if return_plot_args: a (data_tuples, plot_options) tuple. The plot can then be
  made with gp.plot(*data_tuples, **plot_options). Useful if we want to include
  this as a part of a more complex plot

    '''
    if isinstance(models, mrcal.cameramodel):
        models = (models,)

    if cameranames is None:
        cameranames = ['Model {}'.format(i) for i in range(len(models))]
    else:
        if len(models) != len(cameranames):
            raise Exception("Must get the same number of models and cameranames")

    W,H = models[0].imagersize()
    for m in models[1:]:
        WH1 = m.imagersize()
        if W != WH1[0] or H != WH1[1]:
            raise Exception("All given models MUST have the same imagersize. Got {} and {}".format((W,H), WH1))

    valid_regions = [ m.valid_intrinsics_region() for m in models ]
    if any(r is None for r in valid_regions):
        raise Exception("Some given models have no valid-intrinsics region defined")

    # I want to render empty regions, to at least indicate this in the legend
    for i in range(len(models)):
        if valid_regions[i].size == 0:
            valid_regions[i] = np.zeros((1,2))
            cameranames[i] += ": empty"

    import gnuplotlib as gp

    gp.add_plot_option(kwargs, 'set', 'key opaque')

    plot_data_args = []

    if image is not None:
        if isinstance(image, np.ndarray):
            plot_data_args.append( (image, dict(_with='image',
                                                tuplesize = 3)))
        else:
            kwargs['rgbimage'] = image

    plot_data_args.extend( (r[:,0], r[:,1],
                            dict(_with = 'lines lw 4 lc "green"',
                                 legend = cameranames[i])) \
                           for i,r in enumerate(valid_regions) )

    if points is not None:
        plot_data_args.append( (points, dict(tuplesize = -2,
                                             _with = 'points pt 7 ps 1')))

    plot_options = dict(square=1,
                        _xrange=[0,W],
                        _yrange=[H,0],
                        **kwargs)

    if 'title' not in plot_options:
        title = 'Valid-intrinsics region'
        if extratitle is not None:
            title += ": " + extratitle
        plot_options['title'] = title

    data_tuples = plot_data_args

    if not return_plot_args:
        plot = gp.gnuplotlib(**plot_options)
        plot.plot(*data_tuples)
        return plot
    return (data_tuples, plot_options)


def show_splined_model_correction(model,
                                  *,
                                  vectorfield             = False,
                                  xy                      = None,
                                  imager_domain           = False,
                                  vectorscale             = 1.0,
                                  valid_intrinsics_region = True,
                                  observations            = False,
                                  gridn_width             = 60,
                                  gridn_height            = None,
                                  extratitle              = None,
                                  return_plot_args        = False,
                                  **kwargs):

    r'''Visualize the projection corrections defined by a splined model

SYNOPSIS

    model = mrcal.cameramodel(model_filename)

    mrcal.show_splined_model_correction(model)

    # A plot pops up displaying the spline knots, the magnitude of the
    # corrections defined by the spline surfaces, the spline-in-bounds
    # regions and the valid-intrinsics region

Splined models are parametrized by flexible surfaces that define the projection
corrections (off some baseline model), and visualizing these corrections is
useful for understanding the projection behavior. Details of these models are
described in the documentation:

  https://mrcal.secretsauce.net/lensmodels.html#splined-stereographic-lens-model

At this time LENSMODEL_SPLINED_STEREOGRAPHIC is the only splined model mrcal
has, so the baseline model is always LENSMODEL_STEREOGRAPHIC. In spots, the
below documentation assumes a stereographic baseline model.

This function can produce a plot in the domain either of the input or the output
of the spline functions.

if not imager_domain:
    The default. The plot is presented based on the spline index. With
    LENSMODEL_SPLINED_STEREOGRAPHIC, this is the stereographic projection u.
    This is the "forward" direction, what the projection operation actually
    computes. In this view the knots form a regular grid, and the edge of the
    imager forms a (possibly very irregular) curve

if imager_domain:
    The plot is presented based on the pixels in the imager. This is the
    backward direction: the domain is the OUTPUT of the splined functions. In
    this view the knot layout is (possibly highly) irregular. The edge of the
    imager is a perfect rectangle.

Separate from the domain, the data can be presented in 3 different ways:

- Magnitude heatmap. This is the default. Selected by "not vectorfield and xy is
  None". We plot mag(deltauxy). This displays the deviation from the baseline
  model as a heat map.

- Individual heatmap. Selected by "not vectorfield and xy is not None". We plot
  deltaux or deltauy, depending on the value of xy. This displays the value of
  one of the two splined surfaces individually, as a heat map.

- Vector field. Selected by "bool(vectorfield) is True". Displays the correction
  (deltaux, deltauy) as a vector field.

The splined surfaces are defined by control points we call "knots". These knots
are arranged in a fixed grid (defined by the model configuration) with the value
at each knot set in the intrinsics vector.

The configuration selects the control point density and the expected field of
view of the lens. If the fov_x_deg configuration value is too big, many of the
knots will lie well outside the visible area, and will not be used. This is
wasteful. If fov_x_deg is too small, then some parts of the imager will lie
outside of the spline-in-bounds region, resulting in less-flexible projection
behavior at the edges of the imager. So the field of view should roughly match
the actual lens+camera we're using, and we can evaluate that with this function.
This function displays the spline-in-bounds region together with the usable
projection region (either the valid-intrinsics region or the imager bounds).
Ideally, the spline-in-bounds region is slightly bigger than the usable
projection region.

The usable projection region visualized by this function is controlled by the
valid_intrinsics_region argument. If True (the default), we display the
valid-intrinsics region. This is recommended, but keep in mind that this region
is smaller than the full imager, so a fov_x_deg that aligns well for one
calibration may be too small in a subsequent calibration of the same lens. If
the subsequent calibration has better coverage, and thus a bigger
valid-intrinsics region. If not valid_intrinsics_region: we use the imager
bounds instead. The issue here is that the projection near the edges of the
imager is usually poorly-defined because usually there isn't a lot of
calibration data there. This makes the projection behavior at the imager edges
unknowable. Consequently, plotting the projection at the imager edges is usually
too alarming or not alarming enough. Passing valid_intrinsics_region=False is
thus recommended only if we have very good calibration coverage at the edge of
the imager.

ARGUMENTS

- model: the mrcal.cameramodel object being evaluated

- vectorfield: optional boolean defaults to False. if vectorfield: we plot the
  stereographic correction deltau as vectors. if not vectorfield (the default):
  we plot either deltaux or deltauy or mag(deltauxy) as a heat map. if
  vectorfield: xy must be None

- xy: optional string. Must be either 'x' or 'y' or None. Selects the surface
  we're looking at. We have a separate surface for the x and y coordinates, with
  the two sharing the knot positions. We can display one of the surfaces
  individually, or if xy is None: we display the magnitude of the (deltaux,
  deltauy) vector. if xy is not None: vectorfield MUST be false

- imager_domain: optional boolean defaults to False. If False: we plot
  everything against normalized stereographic coordinates; in this
  representation the knots form a regular grid, and the surface domain is a
  rectangle, but the imager boundary is curved. If True: we plot everything
  against the rendered pixel coordinates; the imager boundary is a rectangle,
  while the knots and domain become curved

- vectorscale: optional value defaulting to 1.0. if vectorfield: this is a scale
  factor on the length of the vectors. If we have small deltau, longer vectors
  increase legibility of the plot.

- valid_intrinsics_region: optional boolean defaults to True. If True: we
  communicate the usable projection region to the user by displaying the
  valid-intrinsics region. This isn't available in all models. To fall back on
  the boundary of the full imager, pass False here. In the usual case of
  incomplete calibration-time coverage at the edges, this results in a very
  unrealistic representation of reality. Passing True here is strongly
  recommended

- observations: optional value, defaulting to False. If observations: we plot
  the calibration-time point observations on top of the surface and the knots.
  These make it more clear if the unprojectable regions in the model really are
  a problem. If a special value of 'dots' is passed, the observations are
  plotted as dots instead of points

- gridn_width: optional value, defaulting to 60. How many points along the
  horizontal gridding dimension

- gridn_height: how many points along the vertical gridding dimension. If None,
  we compute an integer gridn_height to maintain a square-ish grid:
  gridn_height/gridn_width ~ imager_height/imager_width

- extratitle: optional string to include in the title of the resulting plot.
  Used to extend the default title string. If kwargs['title'] is given, it is
  used directly, and the extratitle is ignored

- return_plot_args: boolean defaulting to False. if return_plot_args: we return
  a (data_tuples, plot_options) tuple instead of making the plot. The plot can
  then be made with gp.plot(*data_tuples, **plot_options). Useful if we want to
  include this as a part of a more complex plot

- **kwargs: optional arguments passed verbatim as plot options to gnuplotlib.
  Useful to make hardcopies, etc

RETURNED VALUES

If not return_plot_args (the usual path): we return the gnuplotlib plot object.
The plot disappears when this object is destroyed (by the garbage collection,
for instance), so save this returned plot object into a variable, even if you're
not going to be doing anything with this object.

if return_plot_args: we return a (data_tuples, plot_options) tuple instead of
making the plot. The plot can then be made with gp.plot(*data_tuples,
**plot_options). Useful if we want to include this as a part of a more complex
plot

    '''
    if xy is not None:
        if vectorfield:
            raise Exception("Plotting a vectorfield, so xy should be None")

        if not (xy == 'x' or xy == 'y'):
            raise Exception("If given, xy should be either 'x' or 'y'")

    lensmodel,intrinsics_data = model.intrinsics()
    W,H                       = model.imagersize()

    if not re.match('LENSMODEL_SPLINED_STEREOGRAPHIC', lensmodel):
        raise Exception(f"This only makes sense with splined models. Input uses {lensmodel}")

    if gridn_height is None:
        gridn_height = int(round(H/W*gridn_width))


    import gnuplotlib as gp

    if 'title' not in kwargs:
        title = f"Correction for {lensmodel}"
        if xy is not None:
            title += ". Looking at deltau{xy}"
        if extratitle is not None:
            title += ": " + extratitle
        kwargs['title'] = title

    ux_knots,uy_knots = mrcal.knots_for_splined_models(lensmodel)

    if imager_domain:
        # Shape (Ny,Nx,2); contains (x,y) rows
        q = \
            nps.mv( nps.cat(*np.meshgrid( np.linspace(0, W-1, gridn_width),
                                          np.linspace(0, H-1, gridn_height) )),
                    0,-1)
        v = mrcal.unproject(np.ascontiguousarray(q), lensmodel, intrinsics_data)
        u = mrcal.project_stereographic(v)
    else:
        # Shape (gridn_height,gridn_width,2); contains (x,y) rows
        u = \
            nps.mv( nps.cat(*np.meshgrid( np.linspace(ux_knots[0], ux_knots[-1],gridn_width),
                                          np.linspace(uy_knots[0], uy_knots[-1],gridn_height) )),
                    0,-1)

        # My projection is q = (u + deltau) * fxy + cxy. deltau is queried from the
        # spline surface
        v = mrcal.unproject_stereographic(np.ascontiguousarray(u))
        q = mrcal.project(v, lensmodel, intrinsics_data)

    fxy = intrinsics_data[0:2]
    cxy = intrinsics_data[2:4]
    deltau = (q - cxy) / fxy - u

    if valid_intrinsics_region:
        imager_boundary_sparse = model.valid_intrinsics_region()
        if imager_boundary_sparse is None:
            raise Exception("No valid-intrinsics region is available in this model. Pass valid_intrinsics_region=False")
    else:
        # the imager boundary
        imager_boundary_sparse = \
            np.array(((0,   0),
                      (W-1, 0),
                      (W-1, H-1),
                      (0,   H-1),
                      (0,   0)), dtype=float)

    if imager_domain:
        imager_boundary = imager_boundary_sparse
    else:
        imager_boundary = \
            mrcal.project_stereographic(
                mrcal.unproject(
                    mrcal.utils._densify_polyline(imager_boundary_sparse,
                                                  spacing = 50),
                    lensmodel, intrinsics_data ))

    plot_options = dict(kwargs,
                        square  = True,
                        yinv    = True,
                        ascii   = True)

    if imager_domain:
        plot_options['xlabel'] = 'X pixel coord'
        plot_options['ylabel'] = 'Y pixel coord'
    else:
        plot_options['xlabel'] = 'Stereographic ux'
        plot_options['ylabel'] = 'Stereographic uy'

    if not vectorfield:
        gp.add_plot_option(plot_options,
                           _set =  'cblabel "u correction (unitless)"')

        surface_curveoptions = dict( _with     = 'image',
                                     tuplesize = 3 )
        if imager_domain:
            surface_curveoptions['using'] = \
                f'($1/({deltau.shape[1]-1})*({W-1})):' + \
                f'($2/({deltau.shape[0]-1})*({H-1})):' + \
                '3'
        else:
            surface_curveoptions['using'] = \
                f'({(ux_knots[0])}+$1/({deltau.shape[1]-1})*({(ux_knots[-1])-(ux_knots[0])})):' + \
                f'({(uy_knots[0])}+$2/({deltau.shape[0]-1})*({(uy_knots[-1])-(uy_knots[0])})):' + \
                '3'

        if xy is not None:
            plot_data_tuples_surface = \
                ( ( deltau[..., 0 if xy == 'x' else 1],
                    surface_curveoptions ), )
        else:
            plot_data_tuples_surface = \
                ( ( nps.mag(deltau),
                    surface_curveoptions ), )

    else:
        if imager_domain:

            # Vector field in the imager domain. I have q = f (u+du) + cx. So I
            # render the vectors dq = f du
            plot_data_tuples_surface = \
                ( ( *(x.ravel() for x in (q[...,0],
                                          q[...,1],
                                          vectorscale * fxy[0] * deltau[..., 0],
                                          vectorscale * fxy[1] * deltau[..., 1])),
                    dict( _with     = 'vectors filled',
                          tuplesize = 4) ), )
        else:
            plot_data_tuples_surface = \
                ( ( *(x.ravel() for x in (u[...,0],
                                          u[...,1],
                                          vectorscale * deltau[..., 0],
                                          vectorscale * deltau[..., 1])),
                    dict( _with     = 'vectors filled',
                          tuplesize = 4) ), )

    domain_contour_u = mrcal.utils._splined_stereographic_domain(lensmodel)
    knots_u = nps.clump(nps.mv(nps.cat(*np.meshgrid(ux_knots,uy_knots)),
                               0, -1),
                        n = 2)
    if imager_domain:
        domain_contour = \
            mrcal.project(
                mrcal.unproject_stereographic( domain_contour_u),
                lensmodel, intrinsics_data)
        knots = \
            mrcal.project(
                mrcal.unproject_stereographic( np.ascontiguousarray(knots_u)),
                lensmodel, intrinsics_data)
    else:
        domain_contour = domain_contour_u
        knots = knots_u

    plot_data_tuples_boundaries = \
        ( ( imager_boundary,
            dict(_with     = 'lines lw 2 lc "green"',
                 tuplesize = -2,
                 legend    = 'Valid-intrinsics region' if valid_intrinsics_region else 'Imager boundary')),
          ( domain_contour,
            dict(_with     = 'lines lw 1',
                 tuplesize = -2,
                 legend    = 'Spline-in-bounds')), )

    plot_data_tuples_knots = \
        ( ( knots,
            dict(_with     = 'points pt 2 ps 2 lc "green"',
                 tuplesize = -2,
                 legend    = 'knots')), )

    plot_data_tuples_observations = []

    if observations:
        _append_observation_visualizations(plot_data_tuples_observations,
                                           model         = model,
                                           pointtype     = -1 if observations == 'dots' else 1,
                                           _2d           = True,
                                           reproject_to_stereographic = not imager_domain)

    # Anything outside the valid region contour but inside the imager is an
    # invalid area: the field-of-view of the camera needs to be increased. I
    # plot this area
    imager_boundary_nonan = \
        imager_boundary[ np.isfinite(imager_boundary[:,0]) *
                         np.isfinite(imager_boundary[:,1]),:]

    try:
        invalid_regions = mrcal.polygon_difference(imager_boundary_nonan,
                                                   domain_contour)
    except Exception as e:
        # sometimes the domain_contour self-intersects, and this makes us
        # barf
        # print(f"WARNING: Couldn't compute invalid projection region. Exception: {e}")
        invalid_regions = []

    if len(invalid_regions) > 0:
        print("WARNING: some parts of the imager cannot be projected from a region covered by the spline surface! You should increase the field-of-view of the model")

        plot_data_tuples_invalid_regions = \
            tuple( ( r,
                     dict( tuplesize = -2,
                           _with     = 'filledcurves fill transparent pattern 1 lc "royalblue"',
                           legend    = 'Visible spline-out-of-bounds region'))
                       for r in invalid_regions )

    else:
        plot_data_tuples_invalid_regions = ()

    data_tuples = \
        plot_data_tuples_surface         + \
        plot_data_tuples_boundaries      + \
        plot_data_tuples_invalid_regions + \
        tuple(plot_data_tuples_observations) + \
        plot_data_tuples_knots

    if not return_plot_args:
        plot = gp.gnuplotlib(**plot_options)
        plot.plot(*data_tuples)
        return plot
    return (data_tuples, plot_options)


def annotate_image__valid_intrinsics_region(image, model, *, color=(0,255,0)):
    r'''Annotate an image with a model's valid-intrinsics region

SYNOPSIS

    model = mrcal.cameramodel('cam0.cameramodel')

    image = mrcal.load_image('image.jpg')

    mrcal.annotate_image__valid_intrinsics_region(image, model)

    mrcal.save_image('image-annotated.jpg', image)

This function reads a valid-intrinsics region from a given camera model, and
draws it on top of a given image. This is useful to see what parts of a captured
image have reliable intrinsics.

This function modifies the input image.

If the given model has no valid-intrinsics region defined, an exception is
thrown. If the valid-intrinsics region is empty, a solid circle is drawn at the
center.

If we want an interactive plot instead of an annotated image, call
mrcal.show_valid_intrinsics_region() instead.

ARGUMENTS

- model: the mrcal.cameramodel object that contains the valid-intrinsics region
  contour

- image: the numpy array containing the image we're annotating. This is both an
  input and an output

- color: optional tuple of length 3 indicating the BGR color of the annotation.
  Green by default

RETURNED VALUES

None. The input image array is modified

    '''
    valid_intrinsics_region = model.valid_intrinsics_region()

    if valid_intrinsics_region is None:
        raise Exception("The given model has no valid-intrinsics region defined")

    import cv2

    if valid_intrinsics_region is None:
        cv2.circle( image, tuple((model.imagersize() - 1)//2), 10, color, -1)
        print("WARNING: annotate_image__valid_intrinsics_region(): valid-intrinsics region is undefined. Drawing a circle")
    elif valid_intrinsics_region.size == 0:
        cv2.circle( image, tuple((model.imagersize() - 1)//2), 10, color, -1)
        print("WARNING: annotate_image__valid_intrinsics_region(): valid-intrinsics region is empty. Drawing a circle")
    else:
        cv2.polylines(image, [valid_intrinsics_region.astype(np.int32)], True, color, 3)


def imagergrid_using(imagersize, gridn_width, gridn_height = None):
    r'''Get a 'using' gnuplotlib expression for imager colormap plots

SYNOPSIS

    import gnuplotlib as gp
    import numpy as np
    import mrcal

    ...

    Nwidth  = 60
    Nheight = 40

    # shape (Nheight,Nwidth,3)
    v,_ = \
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

We often want to plot some quantity at every location on the imager (intrinsics
uncertainties for instance). This is done by gridding the imager, computing the
quantity at each grid point, and sending this to gnuplot. This involves a few
non-obvious plotting idioms, with the full usage summarized in the above
example.

Due to peculiarities of gnuplot, the 'using' expression produced by this
function can only be used in plots using ascii data commands (i.e. pass
'ascii=True' to gnuplotlib).

ARGUMENTS

- imagersize: a (width,height) tuple for the size of the imager. With a
  mrcal.cameramodel object this is model.imagersize()

- gridn_width: how many points along the horizontal gridding dimension

- gridn_height: how many points along the vertical gridding dimension. If
  omitted or None, we compute an integer gridn_height to maintain a square-ish
  grid: gridn_height/gridn_width ~ imager_height/imager_width

RETURNED VALUE

The 'using' string.

    '''

    W,H = imagersize
    if gridn_height is None:
        gridn_height = int(round(H/W*gridn_width))
    return '($1*{}):($2*{}):3'.format(float(W-1)/(gridn_width-1), float(H-1)/(gridn_height-1))


def show_residuals_board_observation(optimization_inputs,
                                     i_observation,
                                     *,
                                     from_worst                       = False,
                                     i_observations_sorted_from_worst = None,
                                     x                                = None,
                                     # backwards-compatibility synonym of "x"
                                     residuals                        = None,
                                     paths                            = None,
                                     image_path_prefix                = None,
                                     image_directory                  = None,
                                     circlescale                      = 1.0,
                                     vectorscale                      = 1.0,
                                     cbmax                            = None,
                                     showimage                        = True,
                                     extratitle                       = None,
                                     return_plot_args                 = False,
                                     **kwargs):
    r'''Visualize calibration residuals for a single observation

SYNOPSIS

    model = mrcal.cameramodel(model_filename)

    mrcal.show_residuals_board_observation( model.optimization_inputs(),
                                            0,
                                            from_worst = True )

    ... A plot pops up showing the worst-fitting chessboard observation from
    ... the calibration run that produced the model in model_filename

Given a calibration solve, visualizes the fit for a single observation. Plots
the chessboard image overlaid with its residuals. Each residual is plotted as a
circle and a vector. The circles are color-coded by the residual error. The size
of the circle indicates the weight. Bigger means higher weight. The vector shows
the weighted residual from the observation to the prediction.

ARGUMENTS

- optimization_inputs: the optimization inputs dict passed into and returned
  from mrcal.optimize(). This describes the solved optimization problem that
  we're visualizing

- i_observation: integer that selects the chessboard observation. If not
  from_worst (the default), this indexes sequential observations, in the order
  in which they appear in the optimization problem. If from_worst: the
  observations are indexed in order from the worst-fitting to the best-fitting:
  i_observation=0 refers to the worst-fitting observation. This is very useful
  to investigate issues in the calibration

- from_worst: optional boolean, defaulting to False. If not from_worst (the
  default), i_observation indexes sequential observations, in the order in which
  they appear in the optimization problem. If from_worst: the observations are
  indexed in order from the worst-fitting to the best-fitting: i_observation=0
  refers to the worst-fitting observation. This is very useful to investigate
  issues in the calibration

- i_observations_sorted_from_worst: optional iterable of integers used to
  convert sorted-from-worst observation indices to as-specified observation
  indices. If omitted or None, this will be recomputed. To use a cached value,
  pass in a precomputed value. See the sources for an example of how to compute
  it

- x: optional numpy array of shape (Nmeasurements,) containing the optimization
  measurements (the residual weighted reprojection errors). If omitted or None,
  this will be recomputed. To use a cached value, pass the result of
  mrcal.optimize(**optimization_inputs)['x'] or
  mrcal.optimizer_callback(**optimization_inputs)[1]

- residuals: backwards-compatibility synonym for x. At most one of these may be
  non-None

- paths: optional iterable of strings, containing image filenames corresponding
  to each observation. If omitted or None or if the image couldn't be found, the
  residuals will be plotted without the source image. The path we search is
  controlled by the image_path_prefix and image_directory options

- image_path_prefix: optional argument, defaulting to None, exclusive with
  "image_directory". If given, the image paths in the "paths" argument are
  prefixed with the given string.

- image_directory: optional argument, defaulting to None, exclusive with
  "image_path_prefix". If given, we extract the filename from the image path in
  the "paths" argument, and look for the images in the directory given here
  instead

- circlescale: optional scale factor to adjust the size of the plotted circles.
  If omitted, a unit scale (1.0) is used. This exists to improve the legibility
  of the generated plot

- vectorscale: optional scale factor to adjust the length of the plotted
  vectors. If omitted, a unit scale (1.0) is used: this results in the vectors
  representing pixel errors directly. This exists to improve the legibility of
  the generated plot

- cbmax: optional value, defaulting to None. If given, sets the maximum range of
  the color map

- showimage: optional boolean, defaulting to True. If False, we do NOT plot the
  image beneath the residuals.

- extratitle: optional string to include in the title of the resulting plot.
  Used to extend the default title string. If kwargs['title'] is given, it is
  used directly, and the extratitle is ignored

- return_plot_args: boolean defaulting to False. if return_plot_args: we return
  a (data_tuples, plot_options) tuple instead of making the plot. The plot can
  then be made with gp.plot(*data_tuples, **plot_options). Useful if we want to
  include this as a part of a more complex plot

- **kwargs: optional arguments passed verbatim as plot options to gnuplotlib.
  Useful to make hardcopies, etc

RETURNED VALUES

if not return_plot_args (the usual path): we return the gnuplotlib plot object.
The plot disappears when this object is destroyed (by the garbage collection,
for instance), so save this returned plot object into a variable, even if you're
not going to be doing anything with this object.

If return_plot_args: we return a (data_tuples, plot_options) tuple instead of
making the plot. The plot can then be made with gp.plot(*data_tuples,
**plot_options). Useful if we want to include this as a part of a more complex
plot

    '''

    if residuals is not None:
        if x is not None:
            raise Exception("residuals and x are mutually exclusive")
        x = residuals

    if image_path_prefix is not None and \
       image_directory   is not None:
        raise Exception("image_path_prefix and image_directory are mutually exclusive")

    import gnuplotlib as gp


    if x is None:
        # Flattened residuals. The board measurements are at the start of the
        # array
        x = \
            mrcal.optimizer_callback(**optimization_inputs,
                                     no_jacobian      = True,
                                     no_factorization = True)[1]

    # shape (Nobservations, object_height_n, object_width_n, 3)
    observations = optimization_inputs['observations_board']
    x_shape = observations.shape[:-1] + (2,)

    # shape (Nobservations, object_height_n, object_width_n, 2)
    x = x[:np.prod(x_shape)].reshape(*x_shape)

    if from_worst:
        if i_observations_sorted_from_worst is None:
            # shape (Nobservations,)
            err_per_observation = nps.norm2(nps.clump(x, n=-3))
            i_observations_sorted_from_worst = \
                list(reversed(np.argsort(err_per_observation)))

        i_observation_from_worst = i_observation
        i_observation = i_observations_sorted_from_worst[i_observation_from_worst]

    # shape (Nh*Nw,2)
    x         = nps.clump(x   [i_observation         ], n=2)
    # shape (Nh*Nw,2)
    obs       = nps.clump(observations[i_observation, ..., :2], n=2)
    # shape (Nh*Nw)
    weight    = nps.clump(observations[i_observation, ...,  2], n=2)

    # take non-outliers
    i_inliers = weight > 0.
    x         = x[i_inliers] # shape (Ninliers,2)
    weight    = weight   [i_inliers] # shape (Ninliers,)
    obs       = obs      [i_inliers] # shape (Ninliers,2)

    plot_options = dict(kwargs)

    if 'title' not in plot_options:
        title = \
            '{}: i_observation={}{}, iframe={}, icam={}, {}RMS_error={:.2f}'. \
            format( optimization_inputs['lensmodel'],
                    i_observation,
                    f'({i_observation_from_worst} from worst)' if from_worst else '',
                    *optimization_inputs['indices_frame_camintrinsics_camextrinsics'][i_observation, :2],
                    "" if paths is None else f"path={paths[i_observation]}, ",
                    np.sqrt(np.mean(nps.norm2(x))))
        if extratitle is not None:
            title += ": " + extratitle
        plot_options['title'] = title


    gp.add_plot_option(plot_options,
                       square = True,
                       cbmin  = 0,

                       overwrite = False)

    if paths is not None and showimage:
        imagepath = paths[i_observation]

        if image_path_prefix is not None:
            imagepath = f"{image_path_prefix}/{imagepath}"
        elif image_directory is not None:
            imagepath = f"{image_directory}/{os.path.basename(imagepath)}"

        if not os.path.exists(imagepath):
            print(f"WARNING: Couldn't read image at '{imagepath}'", file=sys.stderr)
            imagepath = None
    else:
        imagepath = None

    if imagepath is not None:

        # only plot an image overlay if the image exists
        gp.add_plot_option(plot_options,
                           rgbimage = imagepath,
                           overwrite = True)
        gp.add_plot_option(plot_options,
                           'set',
                           'autoscale noextend')
    else:
        icam = optimization_inputs['indices_frame_camintrinsics_camextrinsics'][i_observation, 1]
        W,H=optimization_inputs['imagersizes'][icam]
        gp.add_plot_option(plot_options,
                           xrange = [0,W-1],
                           yrange = [H-1,0],
                           overwrite = False)


    gp.add_plot_option(plot_options, 'unset', 'key')

    if cbmax is not None:
        gp.add_plot_option(plot_options,
                           cbmax = cbmax)

    data_tuples = \
        (
            # Points. Color indicates error. Size indicates level. Bigger =
            # more confident = higher weight
            (obs[:,0], obs[:,1],
             3. * weight * circlescale, # size
             nps.mag(x),        # color
             dict(_with     = 'points pt 7 ps variable palette',
                  tuplesize = 4)),

            # Vectors. From observation to prediction. Scaled by the weight.
            # Vector points AT the prediction only if weight = 1
            (obs[:,0],
             obs[:,1],
             vectorscale*x[:,0],
             vectorscale*x[:,1],
             dict(_with     = 'vectors filled lw 2',
                  tuplesize = 4)) )

    if not return_plot_args:
        plot = gp.gnuplotlib(**plot_options)
        plot.plot(*data_tuples)
        return plot
    return (data_tuples, plot_options)


def show_residuals_histogram(optimization_inputs,
                             icam_intrinsics  = None,
                             x                = None,
                             # backwards-compatibility synonym of "x"
                             residuals        = None,
                             *,
                             binwidth         = 0.02,
                             extratitle       = None,
                             return_plot_args = False,
                             **kwargs):

    r'''Visualize the distribution of the optimized residuals

SYNOPSIS

    model = mrcal.cameramodel(model_filename)

    mrcal.show_residuals_histogram( model.optimization_inputs() )

    ... A plot pops up showing the empirical distribution of fit errors
    ... in this solve. For ALL the cameras

Given a calibration solve, visualizes the distribution of errors at the optimal
solution. We display a histogram of residuals and overlay it with an idealized
gaussian distribution.

ARGUMENTS

- optimization_inputs: the optimization inputs dict passed into and returned
  from mrcal.optimize(). This describes the solved optimization problem that
  we're visualizing

- icam_intrinsics: optional integer to select the camera whose residuals we're
  visualizing If omitted or None, we display the residuals for ALL the cameras
  together.

- x: optional numpy array of shape (Nmeasurements,) containing the optimization
  measurements (the residual weighted reprojection errors). If omitted or None,
  this will be recomputed. To use a cached value, pass the result of
  mrcal.optimize(**optimization_inputs)['x'] or
  mrcal.optimizer_callback(**optimization_inputs)[1]

- residuals: backwards-compatibility synonym for x. At most one of these may be
  non-None

- binwidth: optional floating-point value selecting the width of each bin in the
  computed histogram. A default of 0.02 pixels is used if this value is omitted.

- extratitle: optional string to include in the title of the resulting plot.
  Used to extend the default title string. If kwargs['title'] is given, it is
  used directly, and the extratitle is ignored

- return_plot_args: boolean defaulting to False. if return_plot_args: we return
  a (data_tuples, plot_options) tuple instead of making the plot. The plot can
  then be made with gp.plot(*data_tuples, **plot_options). Useful if we want to
  include this as a part of a more complex plot

- **kwargs: optional arguments passed verbatim as plot options to gnuplotlib.
  Useful to make hardcopies, etc

RETURNED VALUES

if not return_plot_args (the usual path): we return the gnuplotlib plot object.
The plot disappears when this object is destroyed (by the garbage collection,
for instance), so save this returned plot object into a variable, even if you're
not going to be doing anything with this object.

If return_plot_args: we return a (data_tuples, plot_options) tuple instead of
making the plot. The plot can then be made with gp.plot(*data_tuples,
**plot_options). Useful if we want to include this as a part of a more complex
plot

    '''

    if residuals is not None:
        if x is not None:
            raise Exception("residuals and x are mutually exclusive")
        x = residuals

    import gnuplotlib as gp

    if 'observations_board' in optimization_inputs and \
       optimization_inputs['observations_board'] is not None:
        x_chessboard = \
            mrcal.measurements_board(optimization_inputs = optimization_inputs,
                                     icam_intrinsics     = icam_intrinsics,
                                     x                   = x).ravel()
    else:
        x_chessboard = np.array(())

    if 'observations_point' in optimization_inputs and \
       optimization_inputs['observations_point'] is not None:
        x_point = \
            mrcal.measurements_point(optimization_inputs = optimization_inputs,
                                     icam_intrinsics     = icam_intrinsics,
                                     x                   = x).ravel()
    else:
        x_point = np.array(())

    # I just pool all the observations together for now. I could display them
    # separately...
    x = nps.glue(x_chessboard,
                 x_point,
                 axis=-1)

    if x.size == 0:
        raise Exception("No board or point observations in this solve!")

    sigma_observed = np.std(x)

    equation = fitted_gaussian_equation(sigma    = sigma_observed,
                                        mean     = np.mean(x),
                                        N        = len(x),
                                        binwidth = binwidth,
                                        legend   = f'Normal distribution of measurements with observed stdev: {sigma_observed:.02g} pixels')

    if icam_intrinsics is None:
        what = 'all the cameras'
    else:
        what = f"camera {icam_intrinsics}"

    plot_options = dict(kwargs)

    if 'title' not in plot_options:
        title = f'Distribution of fitted measurements and a gaussian fit for {what}'
        if extratitle is not None:
            title += ": " + extratitle
        plot_options['title'] = title

    gp.add_plot_option(plot_options,
                       equation_above = equation,
                       overwrite = True)
    gp.add_plot_option(plot_options,
                       xlabel = 'Measurements (pixels). x and y components of error are counted separately',
                       ylabel = 'Observed frequency',
                       overwrite = False)
    data_tuples = [ (x, dict(histogram = True,
                             binwidth  = binwidth)) ]
    if not return_plot_args:
        plot = gp.gnuplotlib(**plot_options)
        plot.plot(*data_tuples)
        return plot
    return (data_tuples, plot_options)


def _get_show_residuals_data_onecam(model,
                                    # shape (Nobservations,object_height_n,object_width_n,2)
                                    x,
                                    valid_intrinsics_region):
    r'''Return the data used by the various show_residuals_...() functions

    icam is the camera in question, or None for ALL the cameras'''

    optimization_inputs = model.optimization_inputs()
    icam_intrinsics     = model.icam_intrinsics()






    if 'observations_board' in optimization_inputs and \
       optimization_inputs['observations_board'] is not None:
        # shape (N,2), (N,2)
        err_chessboard,obs_chessboard = \
            mrcal.measurements_board(optimization_inputs = optimization_inputs,
                                     icam_intrinsics     = icam_intrinsics,
                                     x                   = x,
                                     return_observations = True)
    else:
        err_chessboard,obs_chessboard = \
            np.array(()),np.array(())

    if 'observations_point' in optimization_inputs and \
       optimization_inputs['observations_point'] is not None:
        # shape (N,2), (N,2)
        err_point,obs_point = \
            mrcal.measurements_point(optimization_inputs = optimization_inputs,
                                     icam_intrinsics     = icam_intrinsics,
                                     x                   = x,
                                     return_observations = True)
    else:
        err_point,obs_point = \
            np.array(()),np.array(())

    # I just pool all the observations together for now. I could display them
    # separately...
    err = nps.glue(err_chessboard,
                   err_point,
                   axis=-1)
    obs = nps.glue(obs_chessboard,
                   obs_point,
                   axis=-1)

    if valid_intrinsics_region and icam_intrinsics is not None:
        legend = "Valid-intrinsics region"
        valid_region = model.valid_intrinsics_region()

        if valid_region.size is None:
            valid_region = np.zeros((1,2))
            legend += ": undefined"
        elif valid_region == 0:
            valid_region = np.zeros((1,2))
            legend += ": empty"

        valid_intrinsics_region_plotarg_3d = \
            (valid_region[:,0],
             valid_region[:,1],
             np.zeros(valid_region.shape[-2]),
             dict(_with  = 'lines lw 4 lc "green"',
                  legend = legend))
        valid_intrinsics_region_plotarg_2d = \
            (valid_region[:,0],
             valid_region[:,1],
             dict(_with  = 'lines lw 4 lc "green"',
                  legend = legend))
    else:
        valid_intrinsics_region_plotarg_2d = None
        valid_intrinsics_region_plotarg_3d = None

    return                                  \
        err,                                \
        obs,                                \
        valid_intrinsics_region_plotarg_2d, \
        valid_intrinsics_region_plotarg_3d


def show_residuals_vectorfield(model,
                               x                       = None,
                               # backwards-compatibility synonym of "x"
                               residuals               = None,
                               *,
                               vectorscale             = 1.0,
                               cbmax                   = None,
                               valid_intrinsics_region = True,
                               extratitle              = None,
                               return_plot_args        = False,
                               **kwargs):

    r'''Visualize the optimized residuals as a vector field

SYNOPSIS

    model = mrcal.cameramodel(model_filename)

    mrcal.show_residuals_vectorfield( model )

    ... A plot pops up showing each observation from this camera used to
    ... compute this calibration as a vector field. Each vector shows the
    ... observed and predicted location of each chessboard corner

Given a calibration solve, visualizes the errors at the optimal solution as a
vector field. Each vector runs from the observed chessboard corner to its
prediction at the optimal solution.

ARGUMENTS

- model: the mrcal.cameramodel object representing the camera model we're
  investigating. This cameramodel MUST contain the optimization_inputs data

- x: optional numpy array of shape (Nmeasurements,) containing the optimization
  measurements (the residual weighted reprojection errors). If omitted or None,
  this will be recomputed. To use a cached value, pass the result of
  mrcal.optimize(**optimization_inputs)['x'] or
  mrcal.optimizer_callback(**optimization_inputs)[1]

- residuals: backwards-compatibility synonym for x. At most one of these may be
  non-None

- vectorscale: optional scale factor to adjust the length of the plotted
  vectors. If omitted, a unit scale (1.0) is used. Any other scale factor makes
  the tip of each vector run past (or short) of the predicted corner position.
  This exists to improve the legibility of the generated plot

- cbmax: optional value, defaulting to None. If given, sets the maximum range of
  the color map

- valid_intrinsics_region: optional boolean, defaulting to True. If
  valid_intrinsics_region: the valid-intrinsics region present in the model is
  shown in the plot. This is usually interesting to compare to the set of
  observations plotted by the rest of this function

- extratitle: optional string to include in the title of the resulting plot.
  Used to extend the default title string. If kwargs['title'] is given, it is
  used directly, and the extratitle is ignored

- return_plot_args: boolean defaulting to False. if return_plot_args: we return
  a (data_tuples, plot_options) tuple instead of making the plot. The plot can
  then be made with gp.plot(*data_tuples, **plot_options). Useful if we want to
  include this as a part of a more complex plot

- **kwargs: optional arguments passed verbatim as plot options to gnuplotlib.
  Useful to make hardcopies, etc

RETURNED VALUES

if not return_plot_args (the usual path): we return the gnuplotlib plot object.
The plot disappears when this object is destroyed (by the garbage collection,
for instance), so save this returned plot object into a variable, even if you're
not going to be doing anything with this object.

If return_plot_args: we return a (data_tuples, plot_options) tuple instead of
making the plot. The plot can then be made with gp.plot(*data_tuples,
**plot_options). Useful if we want to include this as a part of a more complex
plot

    '''

    if residuals is not None:
        if x is not None:
            raise Exception("residuals and x are mutually exclusive")
        x = residuals

    import gnuplotlib as gp

    err,obs, \
    valid_intrinsics_region_plotarg_2d, \
    valid_intrinsics_region_plotarg_3d = \
        _get_show_residuals_data_onecam(model, x, valid_intrinsics_region)

    W,H = model.imagersize()
    plot_options = dict(kwargs)

    if 'title' not in plot_options:
        title   = 'Fitted measurements. Errors shown as vectors and colors'
        if extratitle is not None:
            title += ": " + extratitle
        plot_options['title'] = title

    gp.add_plot_option(plot_options,

                       square   = True,
                       _xrange = [0,W], yrange=[H,0],
                       xlabel  = 'Imager x',
                       ylabel  = 'Imager y',

                       overwrite = False)

    if cbmax is not None:
        gp.add_plot_option(plot_options,
                           cbrange = [0,cbmax])

    data_tuples = [(obs[:,0], obs[:,1],
                    vectorscale*err[:,0], vectorscale*err[:,1],
                    np.sqrt(nps.norm2(err)),
                    dict(_with='vectors filled palette',
                         tuplesize=5))]
    if valid_intrinsics_region_plotarg_2d is not None:
        data_tuples.append(valid_intrinsics_region_plotarg_2d)

    if not return_plot_args:
        plot = gp.gnuplotlib(**plot_options)
        plot.plot(*data_tuples)
        return plot
    return (data_tuples, plot_options)


def show_residuals_magnitudes(model,
                              x                       = None,
                              # backwards-compatibility synonym of "x"
                              residuals               = None,
                              *,
                              cbmax                   = None,
                              valid_intrinsics_region = True,
                              extratitle              = None,
                              return_plot_args        = False,
                              **kwargs):

    r'''Visualize the optimized residual magnitudes as color-coded points

SYNOPSIS

    model = mrcal.cameramodel(model_filename)

    mrcal.show_residuals_magnitudes( model )

    ... A plot pops up showing each observation from this camera used to
    ... compute this calibration. Each displayed point represents an
    ... observation and its fit error coded as a color

Given a calibration solve, visualizes the errors at the optimal solution. Each
point sits at the observed chessboard corner, with its color representing how
well the solved model fits the observation

ARGUMENTS

- model: the mrcal.cameramodel object representing the camera model we're
  investigating. This cameramodel MUST contain the optimization_inputs data

- x: optional numpy array of shape (Nmeasurements,) containing the optimization
  measurements (the residual weighted reprojection errors). If omitted or None,
  this will be recomputed. To use a cached value, pass the result of
  mrcal.optimize(**optimization_inputs)['x'] or
  mrcal.optimizer_callback(**optimization_inputs)[1]

- residuals: backwards-compatibility synonym for x. At most one of these may be
  non-None

- cbmax: optional value, defaulting to None. If given, sets the maximum range of
  the color map

- valid_intrinsics_region: optional boolean, defaulting to True. If
  valid_intrinsics_region: the valid-intrinsics region present in the model is
  shown in the plot. This is usually interesting to compare to the set of
  observations plotted by the rest of this function

- extratitle: optional string to include in the title of the resulting plot.
  Used to extend the default title string. If kwargs['title'] is given, it is
  used directly, and the extratitle is ignored

- return_plot_args: boolean defaulting to False. if return_plot_args: we return
  a (data_tuples, plot_options) tuple instead of making the plot. The plot can
  then be made with gp.plot(*data_tuples, **plot_options). Useful if we want to
  include this as a part of a more complex plot

- **kwargs: optional arguments passed verbatim as plot options to gnuplotlib.
  Useful to make hardcopies, etc

RETURNED VALUES

if not return_plot_args (the usual path): we return the gnuplotlib plot object.
The plot disappears when this object is destroyed (by the garbage collection,
for instance), so save this returned plot object into a variable, even if you're
not going to be doing anything with this object.

If return_plot_args: we return a (data_tuples, plot_options) tuple instead of
making the plot. The plot can then be made with gp.plot(*data_tuples,
**plot_options). Useful if we want to include this as a part of a more complex
plot

    '''

    if residuals is not None:
        if x is not None:
            raise Exception("residuals and x are mutually exclusive")
        x = residuals

    import gnuplotlib as gp

    err,obs, \
    valid_intrinsics_region_plotarg_2d, \
    valid_intrinsics_region_plotarg_3d = \
        _get_show_residuals_data_onecam(model, x, valid_intrinsics_region)

    W,H = model.imagersize()
    plot_options = dict(kwargs)

    if 'title' not in plot_options:
        title   = 'Fitted measurements. Errors shown as colors'
        if extratitle is not None:
            title += ": " + extratitle
        plot_options['title'] = title

    gp.add_plot_option(plot_options,

                       square   = True,
                       _xrange = [0,W], yrange=[H,0],
                       xlabel  = 'Imager x',
                       ylabel  = 'Imager y',

                       overwrite = False)

    if cbmax is not None:
        gp.add_plot_option(plot_options,
                           cbrange = [0,cbmax])

    data_tuples = [( obs[:,0], obs[:,1], np.sqrt(nps.norm2(err)),
                     dict(_with='points pt 7 palette',
                          tuplesize=3))]
    if valid_intrinsics_region_plotarg_2d is not None:
        data_tuples.append(valid_intrinsics_region_plotarg_2d)

    if not return_plot_args:
        plot = gp.gnuplotlib(**plot_options)
        plot.plot(*data_tuples)
        return plot
    return (data_tuples, plot_options)


def show_residuals_directions(model,
                              x                       = None,
                              # backwards-compatibility synonym of "x"
                              residuals               = None,
                              *,
                              valid_intrinsics_region = True,
                              extratitle              = None,
                              return_plot_args        = False,
                              **kwargs):

    r'''Visualize the optimized residual directions as color-coded points

SYNOPSIS

    model = mrcal.cameramodel(model_filename)

    mrcal.show_residuals_directions( model )

    ... A plot pops up showing each observation from this camera used to
    ... compute this calibration. Each displayed point represents an
    ... observation and the direction of its fit error coded as a color

Given a calibration solve, visualizes the errors at the optimal solution. Each
point sits at the observed chessboard corner, with its color representing the
direction of the fit error. Magnitudes are ignored: large errors and small
errors are displayed identically as long as they're off in the same direction.
This is very useful to detect systematic errors in a solve due to an
insufficiently-flexible camera model.

ARGUMENTS

- model: the mrcal.cameramodel object representing the camera model we're
  investigating. This cameramodel MUST contain the optimization_inputs data

- x: optional numpy array of shape (Nmeasurements,) containing the optimization
  measurements (the residual weighted reprojection errors). If omitted or None,
  this will be recomputed. To use a cached value, pass the result of
  mrcal.optimize(**optimization_inputs)['x'] or
  mrcal.optimizer_callback(**optimization_inputs)[1]

- residuals: backwards-compatibility synonym for x. At most one of these may be
  non-None

- valid_intrinsics_region: optional boolean, defaulting to True. If
  valid_intrinsics_region: the valid-intrinsics region present in the model is
  shown in the plot. This is usually interesting to compare to the set of
  observations plotted by the rest of this function

- extratitle: optional string to include in the title of the resulting plot.
  Used to extend the default title string. If kwargs['title'] is given, it is
  used directly, and the extratitle is ignored

- return_plot_args: boolean defaulting to False. if return_plot_args: we return
  a (data_tuples, plot_options) tuple instead of making the plot. The plot can
  then be made with gp.plot(*data_tuples, **plot_options). Useful if we want to
  include this as a part of a more complex plot

- **kwargs: optional arguments passed verbatim as plot options to gnuplotlib.
  Useful to make hardcopies, etc

RETURNED VALUES

if not return_plot_args (the usual path): we return the gnuplotlib plot object.
The plot disappears when this object is destroyed (by the garbage collection,
for instance), so save this returned plot object into a variable, even if you're
not going to be doing anything with this object.

If return_plot_args: we return a (data_tuples, plot_options) tuple instead of
making the plot. The plot can then be made with gp.plot(*data_tuples,
**plot_options). Useful if we want to include this as a part of a more complex
plot

    '''

    if residuals is not None:
        if x is not None:
            raise Exception("residuals and x are mutually exclusive")
        x = residuals

    import gnuplotlib as gp

    err,obs, \
    valid_intrinsics_region_plotarg_2d, \
    valid_intrinsics_region_plotarg_3d = \
        _get_show_residuals_data_onecam(model, x, valid_intrinsics_region)

    W,H = model.imagersize()
    plot_options = dict(kwargs)

    if 'title' not in plot_options:
        title   = 'Fitted measurements. Directions shown as colors. Magnitudes ignored'
        if extratitle is not None:
            title += ": " + extratitle
        plot_options['title'] = title

    # Use an maximum-saturation, maximum-value HSV
    # palette where the hue encodes the error direction.
    # The direction is periodic, as is the hue
    gp.add_plot_option(plot_options,
                       'set',
                       'palette defined ( 0 "#00ffff", 0.5 "#80ffff", 1 "#ffffff") model HSV')

    gp.add_plot_option(plot_options,

                       square   = True,
                       _xrange = [0,W], yrange=[H,0],
                       xlabel  = 'Imager x',
                       ylabel  = 'Imager y',
                       cbrange = [-180., 180.],

                       overwrite = False)

    data_tuples = [( obs[:,0], obs[:,1], 180./np.pi * np.arctan2(err[...,1], err[...,0]),
                     dict(_with='points pt 7 palette',
                          tuplesize=3))]
    if valid_intrinsics_region_plotarg_2d is not None:
        data_tuples.append(valid_intrinsics_region_plotarg_2d)

    if not return_plot_args:
        plot = gp.gnuplotlib(**plot_options)
        plot.plot(*data_tuples)
        return plot
    return (data_tuples, plot_options)


def show_residuals_regional(model,
                            x                       = None,
                            # backwards-compatibility synonym of "x"
                            residuals               = None,
                            *,
                            gridn_width             = 20,
                            gridn_height            = None,
                            valid_intrinsics_region = True,
                            extratitle              = None,
                            return_plot_args        = False,
                            **kwargs):

    r'''Visualize the optimized residuals, broken up by region

SYNOPSIS

    model = mrcal.cameramodel(model_filename)

    mrcal.show_residuals_regional( model )

    ... Three plots pop up, showing the mean, standard deviation and the count
    ... of residuals in each region in the imager

This serves as a simple method of estimating calibration reliability, without
computing the projection uncertainty.

The imager of a camera is subdivided into bins (controlled by the gridn_width,
gridn_height arguments). The residual statistics are then computed for each bin
separately. We can then clearly see areas of insufficient data (observation
counts will be low). And we can clearly see lens-model-induced biases (non-zero
mean) and we can see heteroscedasticity (uneven standard deviation). The
mrcal-calibrate-cameras tool uses these metrics to construct a valid-intrinsics
region for the models it computes. This serves as a quick/dirty method of
modeling projection reliability, which can be used even if projection
uncertainty cannot be computed.

ARGUMENTS

- model: the mrcal.cameramodel object representing the camera model we're
  investigating. This cameramodel MUST contain the optimization_inputs data

- gridn_width: optional value, defaulting to 20. How many bins along the
  horizontal gridding dimension

- gridn_height: how many bins along the vertical gridding dimension. If None, we
  compute an integer gridn_height to maintain a square-ish grid:
  gridn_height/gridn_width ~ imager_height/imager_width

- x: optional numpy array of shape (Nmeasurements,) containing the optimization
  measurements (the residual weighted reprojection errors). If omitted or None,
  this will be recomputed. To use a cached value, pass the result of
  mrcal.optimize(**optimization_inputs)['x'] or
  mrcal.optimizer_callback(**optimization_inputs)[1]

- residuals: backwards-compatibility synonym for x. At most one of these may be
  non-None

- valid_intrinsics_region: optional boolean, defaulting to True. If
  valid_intrinsics_region: the valid-intrinsics region present in the model is
  shown in the plot. This is usually interesting to compare to the set of
  observations plotted by the rest of this function

- extratitle: optional string to include in the title of the resulting plot.
  Used to extend the default title string. If kwargs['title'] is given, it is
  used directly, and the extratitle is ignored

- return_plot_args: boolean defaulting to False. if return_plot_args: we return
  a (data_tuples, plot_options) tuple instead of making the plot. The plot can
  then be made with gp.plot(*data_tuples, **plot_options). Useful if we want to
  include this as a part of a more complex plot

- **kwargs: optional arguments passed verbatim as plot options to gnuplotlib.
  Useful to make hardcopies, etc. A "hardcopy" here is a base name:
  kwargs['hardcopy']="/a/b/c/d.pdf" will produce plots in "/a/b/c/d.XXX.pdf"
  where XXX is the type of plot being made

RETURNED VALUES

if not return_plot_args (the usual path): we return the gnuplotlib plot object.
The plot disappears when this object is destroyed (by the garbage collection,
for instance), so save this returned plot object into a variable, even if you're
not going to be doing anything with this object.

If return_plot_args: we return a (data_tuples, plot_options) tuple instead of
making the plot. The plot can then be made with gp.plot(*data_tuples,
**plot_options). Useful if we want to include this as a part of a more complex
plot

    '''

    if residuals is not None:
        if x is not None:
            raise Exception("residuals and x are mutually exclusive")
        x = residuals

    import gnuplotlib as gp

    err,obs, \
    valid_intrinsics_region_plotarg_2d, \
    valid_intrinsics_region_plotarg_3d = \
        _get_show_residuals_data_onecam(model, x, valid_intrinsics_region)

    # Each has shape (Nheight,Nwidth)
    mean,stdev,count,using = \
        mrcal.calibration._report_regional_statistics(model)
    def mkplot(x, title, **plot_options):
        plot_options.update(kwargs)
        if 'hardcopy' in plot_options and plot_options['hardcopy'] is not None:
            # hardcopy "/a/b/c/d.pdf" -> "/a/b/c/d.stdev.pdf" where "stdev" is
            # the "what" without special characters
            what = re.sub('[^a-zA-Z0-9_-]+', '_', title)
            plot_options['hardcopy'] = re.sub(r'(\.[^\.]+$)', '.' + what + r'\1', plot_options['hardcopy'])

            print(f"Writing '{plot_options['hardcopy']}'")

        if 'title' not in plot_options:
            if extratitle is not None:
                title += ": " + extratitle
            plot_options['title'] = title

        gp.add_plot_option(plot_options,
                           'set',
                           ('xrange [:] noextend',
                            'yrange [:] noextend reverse',
                            'view equal xy',
                            'view map'))
        gp.add_plot_option(plot_options,
                           'unset',
                           'grid')

        gp.add_plot_option(plot_options,
                           _3d   = True,
                           ascii = True,
                           overwrite = True)

        W,H = model.imagersize()
        gp.add_plot_option(plot_options,
                           _xrange = [0,W],
                           _yrange = [H,0],
                           overwrite = False)

        data_tuples = [( x,
                         dict(tuplesize=3,
                              _with='image',
                              using=using))]
        if valid_intrinsics_region_plotarg_3d is not None:
            data_tuples.append(valid_intrinsics_region_plotarg_3d)

        if not return_plot_args:
            plot = gp.gnuplotlib(**plot_options)
            plot.plot(*data_tuples)
            return plot
        return (data_tuples, plot_options)

    return \
        [ mkplot(np.abs(mean), 'abs(mean)'),
          mkplot(stdev,        'stdev'),
          mkplot(count,        'count', cbrange = (0, 20)) ]
