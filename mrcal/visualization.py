#!/usr/bin/python3

'''Visualization routines

All functions are exported into the mrcal module. So you can call these via
mrcal.visualization.fff() or mrcal.fff(). The latter is preferred.

'''

import numpy as np
import numpysane as nps
import sys
import re
import cv2
import mrcal

def show_geometry(models_or_extrinsics_rt_fromref,
                  cameranames                 = None,
                  cameras_Rt_plot_ref         = None,
                  frames_rt_toref             = None,
                  points                      = None,

                  show_calobjects    = True,
                  axis_scale         = 1.0,
                  object_width_n     = None,
                  object_height_n    = None,
                  object_spacing     = 0,
                  calobject_warp     = None,
                  point_labels       = None,
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
    mrcal.optimize(intrinsics,
                   extrinsics_rt_fromref,
                   frames_rt_toref,
                   points,
                   ...)
    plot2 = \
      mrcal.show_geometry(extrinsics_rt_fromref,
                          frames_rt_toref = frames_rt_toref,
                          points          = points)

This function visualizes the world described by a set of camera models. It shows
the geometry of the cameras themselves (each one is represented by the axes of
its coordinate system). If available (via a frames_rt_toref argument or from
model.optimization_inputs() in the given models), the geometry of the
calibration objects used to compute these models is shown also. We use
frames_rt_toref if this is given. If not, we use the optimization_inputs() from
the FIRST model that provides them.

This function can also be used to visualize the output (or input) of
mrcal.optimize(); the relevant parameters are all identical to those
mrcal.optimize() takes.

This function is the core of the mrcal-show-geometry tool.

All arguments except models_or_extrinsics_rt_fromref are optional.

ARGUMENTS

- models_or_extrinsics_rt_fromref: an iterable of mrcal.cameramodel objects or
  (6,) rt arrays. A array of shape (N,6) works to represent N cameras. If
  mrcal.cameramodel objects are given here and frames_rt_toref is omitted, we
  get the frames_rt_toref from the first model that provides
  optimization_inputs().

- cameranames: optional array of strings of labels for the cameras. If omitted,
  we use generic labels. If given, the array must have the same length as
  models_or_extrinsics_rt_fromref

- cameras_Rt_plot_ref: optional transformation(s). If omitted, we plot
  everything in the camera reference coordinate system. If given, we use a
  "plot" coordinate system with the transformation TO plot coordinates FROM the
  reference coordinates given in this argument. This argument can be given as an
  iterable of Rt transformations to use a different one for each camera (None
  means "identity"). Or a single Rt transformation can be given to use that one
  for ALL the cameras

- frames_rt_toref: optional array of shape (N,6). If given, each row of shape
  (6,) is an rt transformation representing the transformation TO the reference
  coordinate system FROM the calibration object coordinate system. The
  calibration object then MUST be defined by passing in valid object_width_n,
  object_height_n, object_spacing parameters. If frames_rt_toref is omitted or
  None, we look for this data in the given camera models. I look at the given
  models in order, and grab the frames from the first model that has them. If
  none of the models have this data and frames_rt_toref is omitted or NULL, then
  I don't plot any frames at all


- object_width_n: the number of horizontal points in the calibration object
  grid. Required only if frames_rt_toref is not None

- object_height_n: the number of vertical points in the calibration object grid.
  Required only if frames_rt_toref is not None

- object_spacing: the distance between adjacent points in the calibration
  object. A square object is assumed, so the vertical and horizontal distances
  are assumed to be identical. Required only if frames_rt_toref is not None

- calobject_warp: optional (2,) array describing the calibration board warping.
  None means "no warping": the object is flat. Used only if frames_rt_toref is
  not None. See the docs for mrcal.ref_calibration_object() for a description.

- points: optional array of shape (N,3). If omitted, we don't plot the observed
  points. If given, each row of shape (3,) is a point in the reference
  coordinate system.

- point_labels: optional dict from a point index to a string describing it.
  Points in this dict are plotted with this legend; all other points are plotted
  under a generic "points" legend. As many or as few of the points may be
  labelled in this way. If omitted, none of the points will be labelled
  specially. This is used only if points is not None

- show_calobjects: optional boolean defaults to True. if show_calobjects: we
  render the observed calibration objects (if they are available in
  frames_rt_toref of model.optimization_inputs())

- axis_scale: optional scale factor for the size of the axes used to represent
  the cameras. Can be omitted to use some reasonable default size, but for very
  large or very small problems, this may be required to make the plot look right

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

    def get_extrinsics_Rt_toref_one(m):
        if isinstance(m, mrcal.cameramodel):
            return m.extrinsics_Rt_toref()
        else:
            return mrcal.invert_Rt(mrcal.Rt_from_rt(m))

    extrinsics_Rt_toref = \
        nps.cat(*[get_extrinsics_Rt_toref_one(m) \
                  for m in models_or_extrinsics_rt_fromref])
    extrinsics_Rt_toref = nps.atleast_dims(extrinsics_Rt_toref, -3)

    if not show_calobjects:
        frames_rt_toref = None
    elif frames_rt_toref is None:
        # No frames were given. I grab them from the first .cameramodel that has
        # them. If none of the models have this data, I don't plot any frames at
        # all
        for m in models_or_extrinsics_rt_fromref:
            _frames_rt_toref = None
            _object_spacing  = None
            _object_width_n  = None
            _object_height_n = None
            _calobject_warp  = None
            if not isinstance(m, mrcal.cameramodel):
                continue
            try:
                optimization_inputs = m.optimization_inputs()
                _frames_rt_toref = optimization_inputs['frames_rt_toref']
                _object_spacing  = optimization_inputs['calibration_object_spacing']
                _object_width_n  = optimization_inputs['observations_board'].shape[-2]
                _object_height_n = optimization_inputs['observations_board'].shape[-3]
                _calobject_warp  = optimization_inputs['calobject_warp']
            except:
                _frames_rt_toref = None
                _object_spacing  = None
                _object_width_n  = None
                _object_height_n = None
                _calobject_warp  = None
                continue
            break

        # Use the data from the model if everything I need was valid
        if _frames_rt_toref is not None:
            frames_rt_toref = _frames_rt_toref
            object_width_n  = _object_width_n
            object_height_n = _object_height_n
            object_spacing  = _object_spacing
            calobject_warp  = _calobject_warp

    if frames_rt_toref is not None:
        frames_rt_toref = nps.atleast_dims(frames_rt_toref, -2)
    if points          is not None:
        points          = nps.atleast_dims(points,          -2)

    try:
        if cameras_Rt_plot_ref.shape == (4,3):
            cameras_Rt_plot_ref = \
                np.repeat(nps.atleast_dims(cameras_Rt_plot_ref,-3),
                          len(extrinsics_Rt_toref),
                          axis=-3)
    except:
        pass

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

        if legend is not None: plot ONLY the axis vectors. If legend is None: plot
        ONLY the axis labels

        '''
        axes = np.array( ((0,0,0),
                          (1,0,0),
                          (0,1,0),
                          (0,0,2),), dtype=float ) * scale

        transform = mrcal.identity_Rt()

        for x in transforms:
            transform = mrcal.compose_Rt(transform, x)

        if legend is not None:
            return \
                (extend_axes_for_plotting(mrcal.transform_point_Rt(transform, axes)),
                 dict(_with     = 'vectors',
                      tuplesize = -6,
                      legend    = legend), )
        return tuple(nps.transpose(mrcal.transform_point_Rt(transform,
                                                    axes[1:,:]*1.01))) + \
                                   (np.array(('x', 'y', 'z')),
                                    dict(_with     = 'labels',
                                         tuplesize = 4,
                                         legend    = 'labels'),)




    # I need to plot 3 things:
    #
    # - Cameras
    # - Calibration object poses
    # - Observed points
    def gen_curves_cameras():

        def camera_Rt_toplotcoords(i):
            Rt_ref_cam = extrinsics_Rt_toref[i]
            try:
                Rt_plot_ref = cameras_Rt_plot_ref[i]
                return mrcal.compose_Rt(Rt_plot_ref,
                                        Rt_ref_cam)
            except:
                return Rt_ref_cam

        def camera_name(i):
            try:
                return cameranames[i]
            except:
                return 'cam{}'.format(i)

        cam_axes  = \
            [gen_plot_axes( ( camera_Rt_toplotcoords(i), ),
                            legend = camera_name(i),
                            scale=axis_scale) for i in range(0,len(extrinsics_Rt_toref))]
        cam_axes_labels = \
            [gen_plot_axes( ( camera_Rt_toplotcoords(i), ),
                            legend = None,
                            scale=axis_scale) for i in range(0,len(extrinsics_Rt_toref))]

        # I collapse all the labels into one gnuplotlib dataset. Thus I'll be
        # able to turn them all on/off together
        return cam_axes + [(np.ravel(nps.cat(*[l[0] for l in cam_axes_labels])),
                            np.ravel(nps.cat(*[l[1] for l in cam_axes_labels])),
                            np.ravel(nps.cat(*[l[2] for l in cam_axes_labels])),
                            np.tile(cam_axes_labels[0][3], len(extrinsics_Rt_toref))) + \
                            cam_axes_labels[0][4:]]


    def gen_curves_calobjects():

        if frames_rt_toref is None or len(frames_rt_toref) == 0:
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
        #     frames_rt_toref = frames_rt_toref[iframes, ...]


        calobject_ref = mrcal.ref_calibration_object(object_width_n, object_height_n,
                                                     object_spacing, calobject_warp)

        Rf = mrcal.R_from_r(frames_rt_toref[..., :3])
        Rf = nps.mv(Rf,                       0, -4)
        tf = nps.mv(frames_rt_toref[..., 3:], 0, -4)

        # object in the cam0 coord system. shape=(Nframes, object_height_n, object_width_n, 3)
        calobject_cam0 = nps.matmult( calobject_ref, nps.transpose(Rf)) + tf

        # if icam_highlight is not None:
        #     # shape=(Nobservations, object_height_n, object_width_n, 2)
        #     calobject_cam = nps.transform_point_Rt( models[icam_highlight].extrinsics_Rt_fromref(), calobject_cam0 )

        #     print("double-check this. I don't broadcast over the intrinsics anymore")
        #     err = observations[i_observations, ...] - mrcal.project(calobject_cam, *models[icam_highlight].intrinsics())
        #     err = nps.clump(err, n=-3)
        #     rms = np.mag(err) / (object_height_n*object_width_n))
        #     # igood = rms <  0.4
        #     # ibad  = rms >= 0.4
        #     # rms[igood] = 0
        #     # rms[ibad] = 1
        #     calobject_cam0 = nps.glue( calobject_cam0,
        #                             nps.dummy( nps.mv(rms, -1, -3) * np.ones((object_height_n,object_width_n)),
        #                                        -1 ),
        #                             axis = -1)

        # calobject_cam0 shape: (3, Nframes, object_height_n*object_width_n).
        # This will broadcast nicely
        calobject_cam0 = nps.clump( nps.mv(calobject_cam0, -1, -4), n=-2)

        # if icam_highlight is not None:
        #     calobject_curveopts = {'with':'lines palette', 'tuplesize': 4}
        # else:
        calobject_curveopts = {'with':'lines', 'tuplesize': 3}

        return [tuple(list(calobject_cam0) + [calobject_curveopts,])]


    def gen_curves_points():
        if points is None or len(points) == 0:
            return []

        if point_labels is not None:

            # all the non-fixed point indices
            ipoint_not = np.ones( (len(points),), dtype=bool)
            ipoint_not[np.array(list(point_labels.keys()))] = False

            return \
                [ (points[ipoint_not],
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
    curves_points     = gen_curves_points()

    plot_options = \
        dict(_3d=1,
             square=1,
             xlabel='x',
             ylabel='y',
             zlabel='z',
             **kwargs)

    data_tuples = curves_points + curves_cameras + curves_calobjects

    if not return_plot_args:
        plot = gp.gnuplotlib(**plot_options)
        plot.plot(*data_tuples)
        return plot
    return (data_tuples, plot_options)


def _options_heatmap_with_contours( # update these
                                    plotoptions,

                                    contour_max, contour_increment,
                                    imagersize, gridn_width, gridn_height):
    r'''Update plotoptions, return curveoptions for a contoured heat map'''


    if '_set' not in plotoptions:
        plotoptions['_set'] = []
    elif not isinstance(plotoptions['_set'], list):
        plotoptions['_set'] = [plotoptions['_set']]
    if 'unset' not in plotoptions:
        plotoptions['unset'] = []
    elif not isinstance(plotoptions['unset'], list):
        plotoptions['unset'] = [plotoptions['unset']]

    if contour_increment is None:
        # Compute a "nice" contour increment. I pick a round number that gives
        # me a reasonable number of contours

        Nwant = 10
        increment = contour_max/Nwant

        # I find the nearest 1eX or 2eX or 5eX
        base10_floor = np.power(10., np.floor(np.log10(increment)))

        # Look through the options, and pick the best one
        m   = np.array((1., 2., 5., 10.))
        err = np.abs(m * base10_floor - increment)
        contour_increment = -m[ np.argmin(err) ] * base10_floor

    plotoptions['_set'].extend( ['view equal xy',
                                 'view map',
                                 'contour base',
                                 'key box opaque',
                                 'style textbox opaque',
                                 f'cntrparam levels incremental {contour_max},{contour_increment},0'] )

    plotoptions['_3d']     = True
    plotoptions['_xrange'] = [0,             imagersize[0]]
    plotoptions['_yrange'] = [imagersize[1], 0]
    plotoptions['cbrange'] = [0,             contour_max]
    plotoptions['ascii']   = True # needed for imagergrid_using to work

    plotoptions['unset'].extend(['grid'])

    return \
        dict( tuplesize=3,
              legend = "", # needed to force contour labels
              using = imagergrid_using(imagersize, gridn_width, gridn_height),

              # I plot 3 times:
              # - to make the heat map
              # - to make the contours
              # - to make the contour labels
              _with=np.array(('image',
                              'lines nosurface',
                              'labels boxed nosurface')))


def show_projection_diff(models,
                         gridn_width  = 60,
                         gridn_height = None,

                         observations = False,
                         distance     = None,

                         use_uncertainties= True,
                         focus_center     = None,
                         focus_radius     = -1.,
                         implied_Rt10     = None,

                         vectorfield      = False,
                         vectorscale      = 1.0,
                         extratitle       = None,
                         cbmax            = 4,
                         return_plot_args = False,
                         **kwargs):
    r'''Visualize the difference in projection between N models

SYNOPSIS

    models = ( mrcal.cameramodel('cam0-dance0.cameramodel'),
               mrcal.cameramodel('cam0-dance1.cameramodel') )

    mrcal.show_projection_diff(models)

    # A plot pops up displaying the projection difference between the two models

It is often useful to compare the projection behavior of two camera models. For
instance, one may want to evaluate the quality of a calibration by comparing the
results of two different chessboard dances. Or one may want to evaluate the
stability of the intrinsics in response to mechanical or thermal stresses. This
function makes these comparisons, and produces a visualization of the results.
mrcal.projection_diff() computes the differences, and returns the results
WITHOUT making plots.

In the most common case we're given exactly 2 models to compare. We then show
the projection DIFFERENCE as either a vector field or a heat map. If we're given
more than 2 models, then a vector field isn't possible and we show a heat map of
the STANDARD DEVIATION of all the differences.

The details of how the comparison is computed, and the meaning of the arguments
controlling this, are in the docstring of mrcal.projection_diff().

ARGUMENTS

- models: iterable of mrcal.cameramodel objects we're comparing. Usually there
  will be 2 of these, but more than 2 is possible. The intrinsics are used; the
  extrinsics are NOT.

- gridn_width: optional value, defaulting to 60. How many points along the
  horizontal gridding dimension

- gridn_height: how many points along the vertical gridding dimension. If None,
  we compute an integer gridn_height to maintain a square-ish grid:
  gridn_height/gridn_width ~ imager_height/imager_width

- observations: optional boolean, defaulting to False. If True, we overlay
  calibration-time observations on top of the difference plot. We should then
  see that more data produces more consistent results.

- distance: optional value, defaulting to None. The projection difference varies
  depending on the range to the observed world points, with the queried range
  set in this 'distance' argument. If None (the default) we look out to
  infinity. We can compute the implied-by-the-intrinsics transformation off
  multiple distances if they're given here as an iterable. This is especially
  useful if we have uncertainties, since then we'll emphasize the best-fitting
  distances.

- use_uncertainties: optional boolean, defaulting to True. If True we use the
  whole imager to fit the implied-by-the-intrinsics transformation, using the
  uncertainties to emphasize the confident regions. If False, it is important to
  select the confident region using the focus_center and focus_radius arguments.
  If use_uncertainties is True, but that data isn't available, we report a
  warning, and try to proceed without.

- focus_center: optional array of shape (2,); the imager center by default. Used
  to indicate that the implied-by-the-intrinsics transformation should use only
  those pixels a distance focus_radius from focus_center. This is intended to be
  used if no uncertainties are available, and we need to manually select the
  focus region.

- focus_radius: optional value. If use_uncertainties then the default is LARGE,
  to use the whole imager. Else the default is min(width,height)/6. Used to
  indicate that the implied-by-the-intrinsics transformation should use only
  those pixels a distance focus_radius from focus_center. This is intended to be
  used if no uncertainties are available, and we need to manually select the
  focus region. Pass focus_radius=0 to avoid computing the transformation, and
  to use the identity. This would mean there're no geometric differences, and
  we're comparing the intrinsics only

- implied_Rt10: optional Rt transformation (numpy array of shape (4,3)). If
  given, I use the given value for the implied-by-the-intrinsics transformation
  instead of fitting it. If omitted, I compute the transformation. Exclusive
  with focus_center, focus_radius. Valid only if exactly two models are given.

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

- extratitle: optional string to include in the title of the resulting plot

- cbmax: optional value, defaulting to 4.0. Sets the maximum range of the color
  map

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

- implied_Rt10: the geometric Rt transformation in an array of shape (...,4,3).
  This is either whatever was passed into this function (if anything was), or
  the identity if focus_radius==0 or the fitted results. if len(models)>1: this
  is an array of shape (len(models)-1,4,3), with slice i representing the
  transformation between camera 0 and camera i+1.

    '''

    import gnuplotlib as gp

    if 'title' not in kwargs:
        if implied_Rt10 is not None:
            title_note = "using given extrinsics transform"
        elif focus_radius == 0:
            title_note = "using an identity extrinsics transform"
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
        if not (distance is None or \
                isinstance(distance,float) or \
                len(distance) == 1):
            raise Exception("I don't know how to plot multiple-distance diff with vectorfields")

    # Now do all the actual work
    difflen,diff,q0,implied_Rt10 = mrcal.projection_diff(models,
                                                         gridn_width, gridn_height,
                                                         distance,
                                                         use_uncertainties,
                                                         focus_center,focus_radius,
                                                         implied_Rt10)
    if vectorfield:

        # The mrcal.projection_diff() call made sure they're the same for all
        # the models
        W,H=models[0].imagersize()

        plot_options = dict(square=1,
                            _xrange=[0,W],
                            _yrange=[H,0],
                            cbrange=[0,cbmax],
                            **kwargs)

        q0      = nps.clump(q0,      n=len(q0     .shape)-1)
        diff    = nps.clump(diff,    n=len(diff   .shape)-1)
        difflen = nps.clump(difflen, n=len(difflen.shape)  )

        plot_data_args = \
            [ (q0  [:,0], q0  [:,1],
               diff[:,0] * vectorscale, diff[:,1] * vectorscale,
               difflen,
               dict(_with='vectors size screen 0.01,20 fixed filled palette',
                    tuplesize=5)) ]

    else:
        curveoptions = \
            _options_heatmap_with_contours( # update these plot options
                kwargs,

                cbmax, None,
                models[0].imagersize(),
                gridn_width, gridn_height)
        plot_options = kwargs
        plot_data_args = [ (difflen, curveoptions) ]


    valid_region0 = models[0].valid_intrinsics_region()
    if valid_region0 is not None:
        if vectorfield:
            # 2d plot
            plot_data_args.append( (valid_region0[:,0], valid_region0[:,1],
                                    dict(_with = 'lines lw 3',
                                         legend = "valid region of 1st camera")) )
        else:
            # 3d plot
            plot_data_args.append( (valid_region0[:,0], valid_region0[:,1], valid_region0[:,0]*0,
                                    dict(_with = 'lines lw 3 nocontour',
                                         legend = "valid region of 1st camera")) )

    valid_region1 = models[1].valid_intrinsics_region()
    if len(models) == 2 and valid_region1 is not None:
        # The second camera has a valid region, and I should plot it. This has
        # more complexity: each point on the contour of the valid region of the
        # second camera needs to be transformed to the coordinate system of the
        # first camera to make sense. The transformation is complex, and
        # straight lines will not remain straight. I thus resample the polyline
        # more densely.
        v1 = mrcal.unproject(mrcal.utils._densify_polyline(valid_region1, spacing = 50),
                             *models[1].intrinsics())
        valid_region1 = mrcal.project( mrcal.transform_point_Rt( mrcal.invert_Rt(implied_Rt10),
                                                                 v1 ),
                                       *models[0].intrinsics() )
        if vectorfield:
            # 2d plot
            plot_data_args.append( (valid_region1[:,0], valid_region1[:,1],
                                    dict(_with = 'lines lw 3',
                                         legend = "valid region of 2nd camera")) )
        else:
            # 3d plot
            plot_data_args.append( (valid_region1[:,0], valid_region1[:,1], valid_region1[:,0]*0,
                                    dict(_with = 'lines lw 3 nocontour',
                                         legend = "valid region of 2nd camera")) )

    if observations:

        # "nocontour" only for 3d plots
        _2d = bool(vectorfield)
        if _2d:
            _with     = 'points'
            tuplesize = 2
        else:
            _with     = 'points nocontour'
            tuplesize = 3

        for i in range(len(models)):

            m = models[i]

            p_cam_calobjects_inliers, p_cam_calobjects_outliers = \
                mrcal.utils.hypothesis_corner_positions(m.icam_intrinsics(),
                                                        **m.optimization_inputs())[1:]
            q_cam_calobjects_inliers = \
                mrcal.project( p_cam_calobjects_inliers, *m.intrinsics() )
            q_cam_calobjects_outliers = \
                mrcal.project( p_cam_calobjects_outliers, *m.intrinsics() )

            if len(q_cam_calobjects_inliers):
                plot_data_args.append( ( q_cam_calobjects_inliers[...,0],
                                         q_cam_calobjects_inliers[...,1] ) +
                                       ( () if _2d else ( np.zeros(q_cam_calobjects_inliers.shape[:-1]), )) +
                                       ( dict( tuplesize = tuplesize,
                                               _with     = _with,
                                               legend    = f'Camera {i} inliers'), ))
            if len(q_cam_calobjects_outliers):
                plot_data_args.append( ( q_cam_calobjects_outliers[...,0],
                                         q_cam_calobjects_outliers[...,1] ) +
                                       ( () if _2d else ( np.zeros(q_cam_calobjects_outliers.shape[:-1]), )) +
                                       ( dict( tuplesize = tuplesize,
                                               _with     = _with,
                                               legend    = f'Camera {i} outliers'), ))

    data_tuples = plot_data_args

    if not return_plot_args:
        plot = gp.gnuplotlib(**plot_options)
        plot.plot(*data_tuples)
        return plot, implied_Rt10
    return (data_tuples, plot_options), implied_Rt10


def show_projection_uncertainty(model,
                                gridn_width  = 60,
                                gridn_height = None,

                                observations = False,
                                distance     = None,
                                isotropic    = False,
                                extratitle   = None,
                                cbmax        = 3,
                                return_plot_args = False,
                                **kwargs):
    r'''Visualize the uncertainty in camera projection

SYNOPSIS

    model = mrcal.cameramodel('xxx.cameramodel')

    mrcal.show_projection_uncertainty(model)

    ... A plot pops up displaying the expected projection uncertainty across the
    ... imager

This function uses the expected noise of the calibration-time observations to
estimate the uncertainty of projection of the final model. At calibration time we estimate

- The intrinsics (lens paramaters) of a number of cameras
- The extrinsics (geometry) of a number of cameras in respect to some reference
  coordinate system
- The poses of observed chessboards, also in respect to some reference
  coordinate system

All the coordinate systems move around, and all 3 of these sets of data have
some uncertainty. This tool takes into account all the uncertainties to report
an estimated uncertainty metric. See the docstring for
mrcal.projection_uncertainty() for a detailed description of the computation.

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

To see a 3D view showing the calibration-time observations AND uncertainties for
a set of distances at the same time, call show_projection_uncertainty_xydist()
instead.

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

- observations: optional boolean, defaulting to False. If True, we overlay
  calibration-time observations on top of the uncertainty plot. We should then
  see that more data produces more confident results.

- distance: optional value, defaulting to None. The projection uncertainty
  varies depending on the range to the observed point, with the queried range
  set in this 'distance' argument. If None (the default) we look out to
  infinity.

- isotropic: optional boolean, defaulting to False. We compute the full 2x2
  covariance matrix of the projection. The 1-sigma contour implied by this
  matrix is an ellipse, and we use the worst-case direction by default. If we
  want the RMS size of the ellipse instead of the worst-direction size, pass
  isotropic=True.

- extratitle: optional string to include in the title of the resulting plot

- cbmax: optional value, defaulting to 3.0. Sets the maximum range of the color
  map

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
    W,H=model.imagersize()
    if gridn_height is None:
        gridn_height = int(round(H/W*gridn_width))

    lensmodel, intrinsics_data = model.intrinsics()

    q    = mrcal.sample_imager( gridn_width, gridn_height, *model.imagersize() )
    pcam = mrcal.unproject(q, *model.intrinsics(),
                           normalize = True)

    err = mrcal.projection_uncertainty(pcam * (distance if distance is not None else 1.0),
                                       model           = model,
                                       atinfinity      = distance is None,
                                       what            = 'rms-stdev' if isotropic else 'worstdirection-stdev')
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
        _options_heatmap_with_contours( # update these plot options
            kwargs,

            cbmax, None,
            model.imagersize(),
            gridn_width, gridn_height)

    plot_data_args = [(err, curveoptions)]

    valid_intrinsics_region = model.valid_intrinsics_region()
    if valid_intrinsics_region is not None:
        plot_data_args.append( (valid_intrinsics_region[:,0],
                                valid_intrinsics_region[:,1],
                                np.zeros(valid_intrinsics_region.shape[-2]),
                                dict(_with  = 'lines lw 3 nocontour',
                                     legend = "Valid-intrinsics region")) )

    if observations:
        p_cam_calobjects_inliers, p_cam_calobjects_outliers = \
            mrcal.utils.hypothesis_corner_positions(model.icam_intrinsics(),
                                                    **model.optimization_inputs())[1:]
        q_cam_calobjects_inliers = \
            mrcal.project( p_cam_calobjects_inliers, *model.intrinsics() )
        q_cam_calobjects_outliers = \
            mrcal.project( p_cam_calobjects_outliers, *model.intrinsics() )

        if len(q_cam_calobjects_inliers):
            plot_data_args.append( ( q_cam_calobjects_inliers[...,0],
                                     q_cam_calobjects_inliers[...,1],
                                     np.zeros(q_cam_calobjects_inliers.shape[:-1]),
                                     dict( tuplesize = 3,
                                           _with  = 'points nocontour',
                                           legend = 'inliers')) )
        if len(q_cam_calobjects_outliers):
            plot_data_args.append( ( q_cam_calobjects_outliers[...,0],
                                     q_cam_calobjects_outliers[...,1],
                                     np.zeros(q_cam_calobjects_outliers.shape[:-1]),
                                     dict( tuplesize = 3,
                                           _with  = 'points nocontour',
                                           legend = 'outliers')) )

    plot_options = kwargs
    data_tuples  = plot_data_args

    if not return_plot_args:
        plot = gp.gnuplotlib(**plot_options)
        plot.plot(*data_tuples)
        return plot
    return (data_tuples, plot_options)


def show_projection_uncertainty_xydist(model,
                                       gridn_width  = 15,
                                       gridn_height = None,

                                       extratitle   = None,
                                       cbmax        = 3,
                                       return_plot_args = False,
                                       **kwargs):
    r'''Visualize in 3D the uncertainty in camera projection

SYNOPSIS

    model = mrcal.cameramodel('xxx.cameramodel')

    mrcal.show_projection_uncertainty_xydist(model)

    ... A plot pops up displaying the calibration-time observations and the
    ... expected projection uncertainty for various ranges and for various
    ... locations on the imager

This function is similar to show_projection_uncertainty(), but it visualizes the
uncertainty at multiple ranges at the same time.

This function uses the expected noise of the calibration-time observations to
estimate the uncertainty of projection of the final model. At calibration time
we estimate

- The intrinsics (lens paramaters) of a number of cameras
- The extrinsics (geometry) of a number of cameras in respect to some reference
  coordinate system
- The poses of observed chessboards, also in respect to some reference
  coordinate system

All the coordinate systems move around, and all 3 of these sets of data have
some uncertainty. This tool takes into account all the uncertainties to report
an estimated uncertainty metric. See the docstring for
mrcal.projection_uncertainty() for a detailed description of the computation.

This function grids the imager and a set of observation distances, and reports
an uncertainty for each point on the grid. It also plots the calibration-time
observations. The expectation is that the uncertainty will be low in the areas
where we had observations at calibration time.

ARGUMENTS

- model: the mrcal.cameramodel object being evaluated

- gridn_width: optional value, defaulting to 15. How many points along the
  horizontal gridding dimension

- gridn_height: how many points along the vertical gridding dimension. If None,
  we compute an integer gridn_height to maintain a square-ish grid:
  gridn_height/gridn_width ~ imager_height/imager_width

- extratitle: optional string to include in the title of the resulting plot

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
    W,H=model.imagersize()
    if gridn_height is None:
        gridn_height = int(round(H/W*gridn_width))


    p_cam_calobjects_inliers, p_cam_calobjects_outliers = \
        mrcal.utils.hypothesis_corner_positions(model.icam_intrinsics(),
                                                **model.optimization_inputs())[1:]
    q_cam_calobjects_inliers = \
        mrcal.project( p_cam_calobjects_inliers, *model.intrinsics() )
    q_cam_calobjects_outliers = \
        mrcal.project( p_cam_calobjects_outliers, *model.intrinsics() )
    dist_inliers     = nps.mag(p_cam_calobjects_inliers)
    dist_outliers    = nps.mag(p_cam_calobjects_outliers)
    xydist_inliers   = nps.glue(q_cam_calobjects_inliers,  nps.dummy(dist_inliers, -1), axis=-1)
    xydist_outliers  = nps.glue(q_cam_calobjects_outliers, nps.dummy(dist_outliers,-1), axis=-1)

    # shape (gridn_height,gridn_width,2)
    qxy = mrcal.sample_imager(gridn_width, gridn_height, *model.imagersize())

    ranges = np.linspace( np.min( nps.glue( xydist_inliers [...,2].ravel(),
                                            xydist_outliers[...,2].ravel(),
                                            axis=-1) )/3.0,
                          np.max( nps.glue( xydist_inliers [...,2].ravel(),
                                            xydist_outliers[...,2].ravel(),
                                            axis=-1) )*2.0,
                          10)

    # shape (gridn_height,gridn_width,Nranges,3)
    pcam = \
        nps.dummy(mrcal.unproject( qxy, *model.intrinsics()), -2) * \
        nps.dummy(ranges, -1)

    # shape (gridn_height, gridn_width, Nranges)
    worst_direction_stdev_grid = \
        mrcal.projection_uncertainty( pcam,
                                      model = model,
                                      what = 'worstdirection-stdev')

    grid__x_y_ranges = \
        np.meshgrid(qxy[0,:,0],
                    qxy[:,0,1],
                    ranges,
                    indexing = 'ij')

    if 'title' not in kwargs:
        title = f"Projection uncertainty (in pixels) based on calibration input noise"
        if extratitle is not None:
            title += ": " + extratitle
        kwargs['title'] = title

    plotargs = [ (grid__x_y_ranges[0].ravel(), grid__x_y_ranges[1].ravel(), grid__x_y_ranges[2].ravel(),
                  nps.xchg(worst_direction_stdev_grid,0,1).ravel().clip(max=3),
                  nps.xchg(worst_direction_stdev_grid,0,1).ravel(),
                  dict(tuplesize = 5,
                       _with = 'points pt 7 ps variable palette',))]
    if len(xydist_inliers):
        plotargs.append( ( xydist_inliers,
                           dict(tuplesize = -3,
                                _with = 'points',
                                legend = 'inliers')) )
    if len(xydist_outliers):
        plotargs.append( ( xydist_outliers,
                           dict(tuplesize = -3,
                                _with = 'points',
                                legend = 'outliers')) )

    plot_options = \
        dict( _3d      = True,
              squarexy = True,
              xlabel   = 'Pixel x',
              ylabel   = 'Pixel y',
              zlabel   = 'Range',
              **kwargs )
    data_tuples = plotargs

    if not return_plot_args:
        plot = gp.gnuplotlib(**plot_options)
        plot.plot(*data_tuples)
        return plot
    return (data_tuples, plot_options)


def show_projection_uncertainty_vs_distance(model,

                                            where        = "centroid",
                                            isotropic    = False,
                                            extratitle   = None,
                                            return_plot_args = False,
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
- The poses of observed chessboards, also in respect to some reference
  coordinate system

All the coordinate systems move around, and all 3 of these sets of data have
some uncertainty. This tool takes into account all the uncertainties to report
an estimated uncertainty metric. See the docstring for
mrcal.projection_uncertainty() for a detailed description of the computation.

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
  - "centroid": the midpoint of all the chessboard corners observed at
    calibration time
  - A numpy array (x,y) indicating the pixel

- isotropic: optional boolean, defaulting to False. We compute the full 2x2
  covariance matrix of the projection. The 1-sigma contour implied by this
  matrix is an ellipse, and we use the worst-case direction by default. If we
  want the RMS size of the ellipse instead of the worst-direction size, pass
  isotropic=True.

- extratitle: optional string to include in the title of the resulting plot

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

    p_cam_observed_at_calibration_time = \
        mrcal.utils.hypothesis_corner_positions(model.icam_intrinsics(),
                                                **model.optimization_inputs())[1]

    if   where == 'center':
        q = (model.imagersize() - 1.) / 2.

        vcam = mrcal.unproject(q, *model.intrinsics(),
                               normalize = True)

    elif where == 'centroid':
        p    = np.mean(p_cam_observed_at_calibration_time, axis=-2)
        vcam = p / nps.mag(p)

    elif isinstance(where, np.ndarray):
        q    = where
        vcam = mrcal.unproject(q, *model.intrinsics(),
                               normalize = True)
    else:
        raise Exception("'where' should be 'center' or an array specifying a pixel")

    # shape (Ndistances)
    distance_observed_at_calibration_time = \
        nps.mag(p_cam_observed_at_calibration_time)
    distance_min = np.min(distance_observed_at_calibration_time)
    distance_max = np.max(distance_observed_at_calibration_time)

    distances = np.logspace( np.log10(distance_min/5.),
                             np.log10(distance_max*10.),
                             80 )

    # shape (Ndistances, 3)
    pcam = vcam * nps.dummy(distances, -1)

    # shape (Ndistances)
    uncertainty = \
        mrcal.projection_uncertainty( pcam,
                                      model = model,
                                      what  = 'rms-stdev' if isotropic else 'worstdirection-stdev')
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


def show_distortion_off_pinhole(model,
                                mode,
                                scale        = 1.,
                                cbmax        = 25.0,
                                gridn_width  = 60,
                                gridn_height = None,
                                extratitle   = None,
                                return_plot_args = False,
                                **kwargs):

    r'''Visualize a lens's deviation from a pinhole projection

SYNOPSIS

    model = mrcal.cameramodel('xxx.cameramodel')

    mrcal.show_distortion_off_pinhole( model, 'heatmap' )

    ... A plot pops up displaying how much this model deviates from a pinhole
    ... model across the imager

This function treats a pinhole projection as a baseline, and visualizes
deviations from this baseline. So wide lenses will have a lot of reported
"distortion".

This function has 3 modes of operation, specified as a string in the 'mode'
argument.

ARGUMENTS

- model: the mrcal.cameramodel object being evaluated

- mode: this function can produce several kinds of visualizations, with the
  specific mode selected as a string in this argument. Known values:

  - 'heatmap': the imager is gridded, as specified by the
    gridn_width,gridn_height arguments. For each point in the grid, we evaluate
    the difference in projection between the given model, and a pinhole model
    with the same core intrinsics (focal lengths, center pixel coords). This
    difference is color-coded and a heat map is displayed.

  - 'vectorfield': this is the same as 'heatmap', except we display a vector for
    each point, intead of a color-coded cell. If legibility requires the vectors
    being larger or smaller, pass an appropriate value in the 'scale' argument

  - 'radial': Looks at radial distortion only. Plots a curve showing the
    magnitude of the radial distortion as a function of the distance to the
    center

- scale: optional value, defaulting to 1.0. Used to scale the rendered vectors
  if mode=='vectorfield'

- cbmax: optional value, defaulting to 25.0. Sets the maximum range of the color
  map and of the contours if mode=='heatmap'

- gridn_width: how many points along the horizontal gridding dimension. Used if
  mode=='vectorfield' or mode=='heatmap'

- gridn_height: how many points along the vertical gridding dimension. If None,
  we compute an integer gridn_height to maintain a square-ish grid:
  gridn_height/gridn_width ~ imager_height/imager_width. Used if
  mode=='vectorfield' or mode=='heatmap'

- extratitle: optional string to include in the title of the resulting plot

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
    imagersize                  = model.imagersize()

    if 'title' not in kwargs:

        title = "Effects of {}".format(lensmodel)
        if extratitle is not None:
            title += ": " + extratitle
        kwargs['title'] = title

    W,H = imagersize
    if gridn_height is None:
        gridn_height = int(round(H/W*gridn_width))

    if not mrcal.lensmodel_metadata(lensmodel)['has_core']:
        raise Exception("This currently works only with models that have an fxfycxcy core. It might not be required. Take a look at the following code if you want to add support")
    fxy = intrinsics_data[ :2]
    cxy = intrinsics_data[2:4]

    if mode == 'radial':

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

        v_corners  = mrcal.unproject( q_corners,  lensmodel, intrinsics_data)
        v_centersx = mrcal.unproject( q_centersx, lensmodel, intrinsics_data)
        v_centersy = mrcal.unproject( q_centersy, lensmodel, intrinsics_data)

        th_corners  = 180./np.pi * np.arctan2(nps.mag(v_corners [..., :2]), v_corners [..., 2])
        th_centersx = 180./np.pi * np.arctan2(nps.mag(v_centersx[..., :2]), v_centersx[..., 2])
        th_centersy = 180./np.pi * np.arctan2(nps.mag(v_centersy[..., :2]), v_centersy[..., 2])

        # Now the equations. The 'x' value here is "pinhole pixels off center",
        # which is f*tan(th). I plot this model's radial relationship, and that
        # from other common fisheye projections (formulas mostly from
        # https://en.wikipedia.org/wiki/Fisheye_lens)
        equations = [f'180./pi*atan(tan(x*pi/180.) * ({scale})) with lines lw 2 title "THIS model"',
                     'x title "pinhole"',
                     f'180./pi*atan(2. * tan( x*pi/180. / 2.)) title "stereographic"',
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
                               '0 axis x1y2 with lines lw 2' ] )
            gp.add_plot_option(kwargs, 'set', 'y2tics')
            kwargs['y2label'] = 'Rational correction numerator, denominator'
        kwargs['title'] += ': radial distortion. Red: x edges. Green: y edges. Blue: corners'
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


    if not ( mode == 'heatmap' or mode == 'vectorfield' ):
        raise Exception("Unknown mode '{}'. I only know about 'heatmap','vectorfield','radial'".format(mode))


    # shape: (Nheight,Nwidth,2). Contains (x,y) rows
    grid  = np.ascontiguousarray(nps.mv(nps.cat(*np.meshgrid(np.linspace(0,W-1,gridn_width),
                                                             np.linspace(0,H-1,gridn_height))),
                                        0,-1),
                                 dtype = float)

    dgrid =  mrcal.project( nps.glue( (grid-cxy)/fxy,
                                    np.ones(grid.shape[:-1] + (1,), dtype=float),
                                    axis = -1 ),
                          lensmodel, intrinsics_data )

    if mode == 'heatmap':

        curveoptions = \
            _options_heatmap_with_contours( # update these plot options
                kwargs,

                cbmax, None,
                imagersize,
                gridn_width, gridn_height)
        delta = dgrid-grid

        # shape: gridn_height,gridn_width. Because numpy (and thus gnuplotlib) want it that
        # way
        distortion = nps.mag(delta)

        plot_options = kwargs
        data_tuples = ((distortion, curveoptions), )
        if not return_plot_args:
            plot = gp.gnuplotlib(**plot_options)
            plot.plot(*data_tuples)
            return plot
        return (data_tuples, plot_options)

    else:
        # vectorfield

        # shape: gridn_height*gridn_width,2
        grid  = nps.clump(grid,  n=2)
        dgrid = nps.clump(dgrid, n=2)

        delta = dgrid-grid
        delta *= scale

        kwargs['_xrange']=(-50,W+50)
        kwargs['_yrange']=(H+50, -50)
        kwargs['_set'   ]=['object 1 rectangle from 0,0 to {},{} fillstyle empty'.format(W,H)]
        kwargs['square' ]=True

        if '_set' in kwargs:
            if type(kwargs['_set']) is list: kwargs['_set'].extend(kwargs['_set'])
            else:                            kwargs['_set'].append(kwargs['_set'])
            del kwargs['_set']

        plot_options = kwargs
        data_tuples = \
            ( (grid[:,0], grid[:,1], delta[:,0], delta[:,1],
               {'with': 'vectors size screen 0.01,20 fixed filled',
                'tuplesize': 4,
               }),
              (grid[:,0], grid[:,1],
               {'with': 'points',
                'tuplesize': 2,
               }))

        if not return_plot_args:
            plot = gp.gnuplotlib(**plot_options)
            plot.plot(*data_tuples)
            return plot
        return (data_tuples, plot_options)


def show_valid_intrinsics_region(models,
                                 cameranames      = None,
                                 image            = None,
                                 points           = None,
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
                            dict(_with = 'lines lw 3',
                                 legend = cameranames[i])) \
                           for i,r in enumerate(valid_regions) )

    if points is not None:
        plot_data_args.append( (points, dict(tuplesize = -2,
                                             _with = 'points pt 7 ps 1')))

    plot_options = dict(square=1,
                        _xrange=[0,W],
                        _yrange=[H,0],
                        **kwargs)
    data_tuples = plot_data_args

    if not return_plot_args:
        plot = gp.gnuplotlib(**plot_options)
        plot.plot(*data_tuples)
        return plot
    return (data_tuples, plot_options)


def show_splined_model_surface(model, xy,
                               imager_domain = True,
                               extratitle    = None,
                               return_plot_args = False,
                               **kwargs):

    r'''Visualize the surface represented by a splined model

SYNOPSIS

    model = mrcal.cameramodel(model_filename)

    mrcal.show_splined_model_surface( model, 'x' )

    ... A plot pops up displaying the spline knots, the spline surface (for the
    ... "x" coordinate), the spline domain and the imager boundary

Splined models are built with a splined surface that we index to compute the
projection. The meaning of what indexes the surface and the values of the
surface varies by model, but in all cases, visualizing the surface is useful.

The surface is defined by control points. The value of the control points is
set in the intrinsics vector, but their locations (called "knots") are
fixed, as defined by the model configuration. The configuration selects the
control point density AND the expected field of view of the lens. This field
of view should roughly match the actual lens+camera we're using, and this
fit can be visualized with this tool.

If the field of view is too small, some parts of the imager will lie outside
of the region that the splined surface covers. This tool throws a warning in
that case, and displays the offending regions.

This function can produce a plot in the imager domain or in the spline index
domain. Both are useful, and this is controlled by the imager_domain argument
(the default is True). The spline is defined in the stereographic projection
domain, so in the imager domain the knot grid and the domain boundary become
skewed. At this time the spline representation can cross itself, which is
visible as a kink in the domain boundary.

One use for this function is to check that the field-of-view we're using for
this model is reasonable. We'd like the field of view to be wide-enough to cover
the whole imager, but not much wider, since representing invisible areas isn't
useful. Ideally the surface domain boundary (that this tool displays) is just
wider than the imager edges (which this tool also displays).

ARGUMENTS

- model: the mrcal.cameramodel object being evaluated

- xy: 'x' or 'y': selects the surface we're looking at. We have a separate
  surface for the x and y coordinates, with the two sharing the knot positions

- imager_domain: optional boolean defaults to True. If False: we plot everything
  against normalized stereographic coordinates; in this representation the knots
  form a regular grid, and the surface domain is a rectangle, but the imager
  boundary is curved. If True: we plot everything against the rendered pixel
  coordinates; the imager boundary is a rectangle, while the knots and domain
  become curved

- extratitle: optional string to include in the title of the resulting plot

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

if return_plot_args: we return a (data_tuples, plot_options) tuple instead of
making the plot. The plot can then be made with gp.plot(*data_tuples,
**plot_options). Useful if we want to include this as a part of a more complex
plot

    '''

    if   xy == 'x': ixy = 0
    elif xy == 'y': ixy = 1
    else:
        raise Exception("xy should be either 'x' or 'y'")

    lensmodel,intrinsics_data = model.intrinsics()
    W,H                       = model.imagersize()

    if not re.match('LENSMODEL_SPLINED_STEREOGRAPHIC', lensmodel):
        raise Exception(f"This only makes sense with splined models. Input uses {lensmodel}")


    import gnuplotlib as gp

    if 'title' not in kwargs:

        title = f"Surface for {lensmodel}. Looking at deltau{xy}"
        if extratitle is not None:
            title += ": " + extratitle
        kwargs['title'] = title

    ux_knots,uy_knots = mrcal.knots_for_splined_models(lensmodel)
    meta = mrcal.lensmodel_metadata(lensmodel)
    Nx = meta['Nx']
    Ny = meta['Ny']

    if imager_domain:
        # Shape (Ny,Nx,2); contains (x,y) rows
        q = \
            nps.mv( nps.cat(*np.meshgrid( np.linspace(0, W-1, 60),
                                          np.linspace(0, H-1, 40) )),
                    0,-1)
        v = mrcal.unproject(np.ascontiguousarray(q), lensmodel, intrinsics_data)
        u = mrcal.project_stereographic(v)
    else:

        # In the splined_stereographic models, the spline is indexed by u. So u is
        # linear with the knots. I can thus get u at the edges, and linearly
        # interpolate between

        # Shape (Ny,Nx,2); contains (x,y) rows
        u = \
            nps.mv( nps.cat(*np.meshgrid( np.linspace(ux_knots[0], ux_knots[-1],Nx*5),
                                          np.linspace(uy_knots[0], uy_knots[-1],Ny*5) )),
                    0,-1)

        # My projection is q = (u + deltau) * fxy + cxy. deltau is queried from the
        # spline surface
        v = mrcal.unproject_stereographic(np.ascontiguousarray(u))
        q = mrcal.project(v, lensmodel, intrinsics_data)

    fxy = intrinsics_data[0:2]
    cxy = intrinsics_data[2:4]
    deltau = (q - cxy) / fxy - u

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
                       zlabel   = f"Deltau{xy} (unitless)")
    surface_curveoptions = dict()
    if imager_domain:
        plot_options['xlabel'] = 'X pixel coord'
        plot_options['ylabel'] = 'Y pixel coord'
        surface_curveoptions['using'] = \
            f'($1/({deltau.shape[1]-1})*({W-1})):' + \
            f'($2/({deltau.shape[0]-1})*({H-1})):' + \
            '3'
    else:
        plot_options['xlabel'] = 'Stereographic ux'
        plot_options['ylabel'] = 'Stereographic uy'
        surface_curveoptions['using'] = \
            f'({ux_knots[0]}+$1/({deltau.shape[1]-1})*({ux_knots[-1]-ux_knots[0]})):' + \
            f'({uy_knots[0]}+$2/({deltau.shape[0]-1})*({uy_knots[-1]-uy_knots[0]})):' + \
            '3'

    plot_options['square']   = True
    plot_options['yinv']     = True
    plot_options['ascii']    = True
    surface_curveoptions['_with']     = 'image'
    surface_curveoptions['tuplesize'] = 3

    data = [ ( deltau[..., ixy],
               surface_curveoptions ) ]

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

    data.extend( [ ( imager_boundary,
                     dict(_with     = 'lines lw 2',
                          tuplesize = -2,
                          legend    = 'imager boundary')),
                   ( domain_contour,
                     dict(_with     = 'lines lw 1',
                          tuplesize = -2,
                          legend    = 'Valid projection region')),
                   ( knots,
                     dict(_with     = 'points pt 2 ps 2',
                          tuplesize = -2,
                          legend    = 'knots'))] )


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

        data.extend( [ ( r,
                         dict( tuplesize = -2,
                               _with     = 'filledcurves closed fillcolor "red"',
                               legend    = 'Invalid regions'))
                       for r in invalid_regions] )

    data_tuples  = data
    if not return_plot_args:
        plot = gp.gnuplotlib(**plot_options)
        plot.plot(*data_tuples)
        return plot
    return (data_tuples, plot_options)


def annotate_image__valid_intrinsics_region(image, model, color=(0,0,255)):
    r'''Annotate an image with a model's valid-intrinsics region

SYNOPSIS

    model = mrcal.cameramodel('cam0.cameramodel')

    image = cv2.imread('image.jpg')

    mrcal.annotate_image__valid_intrinsics_region(image, model)

    cv2.imwrite('image-annotated.jpg', image)

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
  Red by default

RETURNED VALUES

None. The input image array is modified

    '''
    valid_intrinsics_region = model.valid_intrinsics_region()

    if valid_intrinsics_region is None:
        raise Exception("The given model has no valid-intrinsics region defined")

    if valid_intrinsics_region.size == 0:
        cv2.circle( image, tuple((model.imagersize() - 1)//2), 10, color, -1)
        print("WARNING: annotate_image__valid_intrinsics_region(): valid-intrinsics region is empty. Drawing a circle")
    else:
        cv2.polylines(image, [valid_intrinsics_region], True, color, 3)


def imagergrid_using(imagersize, gridn_width, gridn_height = None):
    '''Get a 'using' expression for imager colormap plots

SYNOPSIS

    import gnuplotlib as gp
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
