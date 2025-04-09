#!/usr/bin/python3

# Copyright (c) 2017-2023 California Institute of Technology ("Caltech"). U.S.
# Government sponsorship acknowledged. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

'''A class to read/write/manipulate camera models

SYNOPSIS

    model_for_intrinsics = mrcal.cameramodel('model0.cameramodel')
    model_for_extrinsics = mrcal.cameramodel('model1.cameramodel')

    model_joint = mrcal.cameramodel( model_for_intrinsics )

    extrinsics = model_for_extrinsics.extrinsics_rt_fromref()
    model_joint.extrinsics_rt_fromref(extrinsics)

    # model_joint now has intrinsics from 'model0.cameramodel' and extrinsics
    # from 'model1.cameramodel'. I write it to disk
    model_joint.write('model-joint.cameramodel')

All functions are exported into the mrcal module. So you can call these via
mrcal.cameramodel.fff() or mrcal.fff(). The latter is preferred.

'''

import sys
import numpy as np
import numpysane as nps
import numbers
import ast
import re
import warnings
import io
import base64

import mrcal

def _validateExtrinsics(e):
    r'''Raises an exception if the given extrinsics are invalid'''

    # Internal extrinsic representation is a 6-long array
    try:
        N = len(e)
    except:
        raise Exception("Valid extrinsics are an iterable of len 6")


    if N != 6:
        raise Exception("Valid extrinsics are an iterable of len 6")

    for x in e:
        if not isinstance(x, numbers.Number):
            raise Exception("All extrinsics elements should be numeric, but '{}' isn't".format(x))


def _validateIntrinsics(imagersize,
                        i,
                        optimization_inputs = None,
                        icam_intrinsics     = None):
    r'''Raises an exception if given components of the intrinsics is invalid'''

    # need two integers in the imager size
    try:
        N = len(imagersize)
        if N != 2:
            raise Exception("The imagersize must be an iterable of two positive integers")
        if imagersize[0] <= 0 or imagersize[1] <= 0:
            raise Exception("The imagersize must be an iterable of two positive integers")
        if imagersize[0] != int(imagersize[0]) or imagersize[1] != int(imagersize[1]):
            raise Exception("The imagersize must be an iterable of two positive integers")
    except:
        raise Exception("The imagersize must be an iterable of two positive integers")

    try:
        N = len(i)
    except:
        raise Exception("Valid intrinsics are an iterable of len 2")


    if N != 2:
        raise Exception("Valid intrinsics are an iterable of len 2")

    lensmodel  = i[0]
    intrinsics = i[1]

    # If this fails, I keep the exception and let it fall through
    Nintrinsics_want = mrcal.lensmodel_num_params(lensmodel)

    try:
        Nintrinsics_have = len(intrinsics)
    except:
        raise Exception("Valid intrinsics are (lensmodel, intrinsics) where 'intrinsics' is an iterable with a length")

    if Nintrinsics_want != Nintrinsics_have:
        raise Exception("Mismatched Nintrinsics. Got {}, but model {} must have {}".format(Nintrinsics_have,lensmodel,Nintrinsics_want))

    for x in intrinsics:
        if not isinstance(x, numbers.Number):
            raise Exception("All intrinsics elements should be numeric, but '{}' isn't".format(x))

    if optimization_inputs is not None:
        # Currently this is only checked when we set the optimization_inputs by
        # calling intrinsics(). We do NOT check this if reading a file from
        # disk. This is done as an optimization: we store the unprocessed
        # _optimization_inputs_string bytes and only decode them as needed.
        # Perhaps I should expand this
        if not isinstance(optimization_inputs, dict):
            raise Exception(f"'optimization_inputs' must be a dict. Instead got type {type(optimization_inputs)}")

        if icam_intrinsics is None:
            raise Exception(f"optimization_inputs is given, so icam_intrinsics MUST be given too")
        if not isinstance(icam_intrinsics, int):
            raise Exception(f"icam_intrinsics not an int. This must be an int >= 0")
        if icam_intrinsics < 0:
            raise Exception(f"icam_intrinsics < 0. This must be an int >= 0")


def _validateValidIntrinsicsRegion(valid_intrinsics_region):
    r'''Raises an exception if the given valid_intrinsics_region is illegal'''

    if valid_intrinsics_region is None:
        return

    try:
        # valid intrinsics region is a closed contour, so I need at least 4
        # points to be valid. Or as a special case, a (0,2) array is legal, and
        # means "intrinsics are valid nowhere"
        if not (valid_intrinsics_region.ndim == 2     and \
                valid_intrinsics_region.shape[1] == 2 and \
                (valid_intrinsics_region.shape[0] >= 4 or \
                 valid_intrinsics_region.shape[0] == 0)):
            raise Exception("The valid extrinsics region must be a numpy array of shape (N,2) with N >= 4 or N == 0")
    except:
        raise Exception("The valid extrinsics region must be a numpy array of shape (N,2) with N >= 4. Instead got type {} of shape {}". \
                        format(type(valid_intrinsics_region), valid_intrinsics_region.shape if type(valid_intrinsics_region) is np.ndarray else None))

    if valid_intrinsics_region.size > 0 and \
       nps.norm2(valid_intrinsics_region[0] - valid_intrinsics_region[-1]) > 1e-6:
        raise Exception("The valid extrinsics region must be a closed contour: the first and last points must be identical")


class CameramodelParseException(Exception):
    r'''Raised if a .cameramodel file couldn't be parsed successfully

    This is just a normal "Exception" with a different name, so I can handle
    this specifically

    '''
    pass


def _serialize_optimization_inputs(optimization_inputs):
    r'''Convert a optimization_inputs dict to an ascii string

This is an internal function.

I store the optimization inputs as an opaque string in the .cameramodel
file. This is a potentially large data blob that has little value in
being readable to the user. To serialize I do this:

- normalize the data in the dict. The numpy.save...() functions have
  quirks, so I must work around them:

  - All non-numpy-array values are read in as scalar numpy arrays, so I
    must extract them at reading time

  - Any non-trivial objects are pickled, but pickling is not safe for
    untrusted data, so I disable pickling, which means that non-trivial
    objects are not serializable. This includes None. So I store None as
    ''. This means I can't store '', which I guess is OK

- np.savez_compressed() to make a compressed binary data stream

- base64.b85encode() to convert to printable ascii

This works, but the normalization is ugly, and this thing is
inefficient. b85encode() is written in Python for instance. I can
replace some of this with tarfile, which doesn't really solve the
problems, but it could be a starting point for something better in the
future


    def write():
        d = dict( x  = np.arange(1000) + 5,
                  y  = np.arange(300, dtype=float) / 2,
                  z  = None,
                  s  = 'abc',
                  bt = True,
                  bf = False)

        f = io.BytesIO()
        tar = \
            tarfile.open(name    = None,
                         mode    = 'w|gz',
                         fileobj = f)

        for k in d.keys():
            v = d[k]
            if v is None: v = ''

            d_bytes = io.BytesIO()
            np.save(d_bytes, v, allow_pickle = False)

            tarinfo = tarfile.TarInfo(name=k)
            tarinfo.size = d_bytes.tell()
            d_bytes.seek(0)
            tar.addfile( tarinfo,
                         fileobj = d_bytes )
        tar.close()
        sys.stdout.buffer.write(f.getvalue())

    def read():
        with open("/tmp/tst.tar.gz", "rb") as f:
            tar = \
                tarfile.open(name    = None,
                             mode    = 'r|gz',
                             fileobj = f)

            for m in tar:
                b = tar.extractfile(m).read()
                arr = np.load(io.BytesIO(b), allow_pickle=False, encoding='bytes')
                if arr.shape == ():
                    arr = arr.item()
                if type(arr) is str and arr == '':
                    arr = None
                print(arr)
            tar.close()
    '''

    data_bytes = io.BytesIO()

    optimization_inputs_normalized = dict()
    for k in optimization_inputs.keys():
        v = optimization_inputs[k]
        if v is None: v = ''

        # Skip the default do_apply_regularization_unity_cam01 value when
        # writing to the model on disk. This isn't required, since it is
        # default. And its presense will make the older mrcal.cameramodel()
        # parsers complain
        if k == 'do_apply_regularization_unity_cam01' and \
           not v:
            continue

        optimization_inputs_normalized[k] = v

    np.savez_compressed(data_bytes, **optimization_inputs_normalized)
    return \
        base64.b85encode(data_bytes.getvalue())


def _deserialize_optimization_inputs(data_bytes):
    r'''Convert an ascii string for the optimization-input to a full dict

This is an internal function.


This is the inverse of _serialize_optimization_inputs(). See the docstring of
that function for details

    '''

    optimization_inputs_bytes = io.BytesIO(base64.b85decode(data_bytes))

    _optimization_inputs = np.load(optimization_inputs_bytes, allow_pickle = False)

    # Now I need to post-process my output array. Numpy converts everything
    # to numpy arrays for some reason, even things that aren't numpy arrays.
    # So I find everything that's an array of shape (), and convert it to
    # the actual thing contained in the array
    optimization_inputs = dict()
    for k in _optimization_inputs.keys():
        arr = _optimization_inputs[k]
        if arr.shape == ():
            arr = arr.item()
        if type(arr) is str and arr == '':
            arr = None
        optimization_inputs[k] = arr

    # for legacy compatibility
    def renamed(s0, s1, d):
        if s0 in d and not s1 in d:
            d[s1] = d[s0]
            del d[s0]
    renamed('do_optimize_intrinsic_core',
            'do_optimize_intrinsics_core',
            optimization_inputs)
    renamed('do_optimize_intrinsic_distortions',
            'do_optimize_intrinsics_distortions',
            optimization_inputs)
    # renamed('icam_intrinsics_optimization_inputs',
    #         'icam_intrinsics',
    #         optimization_inputs)

    if 'calibration_object_width_n'  in optimization_inputs:
        del optimization_inputs['calibration_object_width_n' ]
    if 'calibration_object_height_n' in optimization_inputs:
        del optimization_inputs['calibration_object_height_n']

    return optimization_inputs


class cameramodel(object):
    r'''A class that describes the lens parameters and geometry of a single camera

SYNOPSIS

    model = mrcal.cameramodel('xxx.cameramodel')

    extrinsics_Rt_toref = model.extrinsics_Rt_toref()

    extrinsics_Rt_toref[3,2] += 10.0

    extrinsics_Rt_toref = model.extrinsics_Rt_toref(extrinsics_Rt_toref)

    model.write('moved.cameramodel')

    # we read a model from disk, moved it 10 units along the z axis, and wrote
    # the results back to disk

This class represents

- The intrinsics: parameters that describe the lens. These do not change as the
  camera moves around in space. These are represented by a tuple
  (lensmodel,intrinsics_data)

  - lensmodel: a string "LENSMODEL_...". The full list of supported models is
    returned by mrcal.supported_lensmodels()

  - intrinsics_data: a numpy array of shape
    (mrcal.lensmodel_num_params(lensmodel),). For lensmodels that have an
    "intrinsics core" (all of them, currently) the first 4 elements are

    - fx: the focal-length along the x axis, in pixels
    - fy: the focal-length along the y axis, in pixels
    - cx: the projection center along the x axis, in pixels
    - cy: the projection center along the y axis, in pixels

    The remaining elements (both their number and their meaning) are dependent
    on the specific lensmodel being used. Some models (LENSMODEL_PINHOLE for
    example) do not have any elements other than the core.

- The imager size: the dimensions of the imager, in a (width,height) list

- The extrinsics: the pose of this camera in respect to some reference
  coordinate system. The meaning of this "reference" coordinate system is
  user-defined, and means nothing to mrcal.cameramodel. If we have multiple
  cameramodels, they generally share this "reference" coordinate system, and it
  is used as an anchor to position the cameras in respect to one another.
  Internally this is stored as an rt transformation converting points
  represented in the reference coordinate TO a representation in the camera
  coordinate system. These are gettable/settable by these methods:

  - extrinsics_rt_toref()
  - extrinsics_rt_fromref()
  - extrinsics_Rt_toref()
  - extrinsics_Rt_fromref()

  These exist for convenience, and handle the necessary conversion internally.
  These make it simple to use the desired Rt/rt transformation and the desired
  to/from reference direction.

- The valid-intrinsics region: a contour in the imager where the projection
  behavior is "reliable". The meaning of "reliable" is user-defined. mrcal is
  able to report the projection uncertainty anywhere in space, and this is a
  more fine-grained way to measure "reliability", but sometimes it is convenient
  to define an uncertainty limit, and to compute a region where this limit is
  met. This region can then be stored in the mrcal.cameramodel object. A missing
  valid-intrinsics region means "unknown". An empty valid-intrinsics region
  (array of length 0) means "intrinsics are valid nowhere". Storing this is
  optional.

- The optimization inputs: this is a dict containing all the data that was used
  to compute the contents of this model. These are optional. These are the
  kwargs passable to mrcal.optimize() and mrcal.optimizer_callback() that
  describe the optimization problem at its final optimum. Storing these is
  optional, but they are very useful for diagnostics, since everything in the
  model can be re-generated from this data. Some things (most notably the
  projection uncertainties) are also computed off the optimization_inputs(),
  making these extra-useful. The optimization_inputs dict can be queried by the
  optimization_inputs() method. Setting this can be done only together with the
  intrinsics(), using the intrinsics() method. For the purposes of computing the
  projection uncertainty it is allowed to move the camera (change the
  extrinsics), so the extrinsics_...() methods may be called without
  invalidating the optimization_inputs.

This class provides facilities to read/write models on disk, and to get/set the
various components.

The format of a .cameramodel file is a python dictionary that we (safely) eval.
A sample valid .cameramodel file:

    # generated with ...
    { 'lensmodel': 'LENSMODEL_OPENCV8',

      # intrinsics are fx,fy,cx,cy,distortion0,distortion1,....
      'intrinsics': [1766.0712405930,
                     1765.8925266865,
                     1944.0664501036,
                     1064.5231421210,
                     2.1648025156,
                     -1.1851581377,
                     -0.0000931342,
                     0.0007782462,
                     -0.2351910903,
                     2.4460295029,
                     -0.6697132481,
                     -0.6284355415],

      # extrinsics are rt_fromref
      'extrinsics': [0,0,0,0,0,0],
      'imagersize': [3840,2160]
    }

    '''

    def _write(self, f, note=None):
        r'''Writes out this camera model to an open file'''

        if note is not None:
            for l in note.splitlines():
                f.write('# ' + l + '\n')

        # I write this out manually instead of using repr for the whole thing
        # because I want to preserve key ordering
        f.write("{\n")
        f.write("    'lensmodel':  '{}',\n".format(self._intrinsics[0]))
        f.write("\n")

        N = len(self._intrinsics[1])
        if(mrcal.lensmodel_metadata_and_config(self._intrinsics[0])['has_core']):
            f.write("    # intrinsics are fx,fy,cx,cy,distortion0,distortion1,....\n")
        f.write(("    'intrinsics': [" + (" {:.10g}," * N) + "],\n").format(*self._intrinsics[1]))
        f.write("\n")

        if self._valid_intrinsics_region is not None:
            f.write("    'valid_intrinsics_region': [\n")
            for row in self._valid_intrinsics_region:
                f.write(("    [ {:.10g}, {:.10g} ],\n").format(*row))
            f.write("],\n\n")

        N = len(self._extrinsics)
        f.write("    # extrinsics are rt_fromref\n")
        f.write(("    'extrinsics': [" + (" {:.10g}," * N) + "],\n").format(*self._extrinsics))
        f.write("\n")

        N = 2
        f.write(("    'imagersize': [" + (" {:d}," * N) + "],\n").format(*(int(x) for x in self._imagersize)))
        f.write("\n")

        if self._icam_intrinsics is not None:
            f.write(("    'icam_intrinsics': {:d},\n").format(self._icam_intrinsics))
        f.write("\n")

        if self._optimization_inputs_string is not None:
            f.write(r"""    # The optimization inputs contain all the data used to compute this model.
    # This contains ALL the observations for ALL the cameras in the solve. The uses of
    # this are to be able to compute projection uncertainties, to visualize the
    # calibration-time geometry and to re-run the original optimization for
    # diagnostics. This is a big chunk of data that isn't useful for humans to
    # interpret, so it's stored in binary, compressed, and encoded to ascii in
    # base-85. Modifying the intrinsics of the model invalidates the optimization
    # inputs: the optimum implied by the inputs would no longer match the stored
    # parameters. Modifying the extrinsics is OK, however: if we move the camera
    # elsewhere, the original solve can still be used to represent the camera-relative
    # projection uncertainties
""")
            f.write(f"    'optimization_inputs': {self._optimization_inputs_string},\n\n")

        f.write("}\n")


    def _read_into_self(self, f):
        r'''Reads in a model from an open file, or the model given as a string

        Note that the string is NOT a filename, it's the model data. This
        function reads the native .cameramodel format only

        '''

        # workaround for python3 idiocy
        try:
            filetype = file
        except:
            filetype = io.IOBase
        if isinstance(f, filetype):
            s    = f.read()
            try:
                name = f.name
            except:
                name = None
        else:
            s    = f
            name = None

        try:
            model = ast.literal_eval(s)
        except:
            if name is None:
                raise CameramodelParseException("Failed to parse cameramodel!")
            else:
                raise CameramodelParseException(f"Failed to parse cameramodel '{name}'")

        # for legacy compatibility
        def renamed(s0, s1, d):
            if s0 in d and not s1 in d:
                d[s1] = d[s0]
                del d[s0]
        renamed('distortion_model',
                'lensmodel',
                model)
        renamed('lens_model',
                'lensmodel',
                model)
        renamed('icam_intrinsics_optimization_inputs',
                'icam_intrinsics',
                model)
        model['lensmodel'] = model['lensmodel'].replace('DISTORTION', 'LENSMODEL')



        keys_required = set(('lensmodel',
                             'intrinsics',
                             'extrinsics',
                             'imagersize'))
        keys_received = set(model.keys())
        if keys_received < keys_required:
            raise Exception("Model must have at least these keys: '{}'. Instead I got '{}'". \
                            format(keys_required, keys_received))

        valid_intrinsics_region = None
        if 'valid_intrinsics_region' in model:
            if len(model['valid_intrinsics_region']) > 0:
                valid_intrinsics_region = np.array(model['valid_intrinsics_region'])
            else:
                valid_intrinsics_region = np.zeros((0,2))

        intrinsics = (model['lensmodel'], np.array(model['intrinsics'], dtype=float))

        _validateIntrinsics(model['imagersize'],
                            intrinsics)
        try:
            _validateValidIntrinsicsRegion(valid_intrinsics_region)
        except Exception as e:
            warnings.warn("Invalid valid_intrinsics region; skipping: '{}'".format(e))
            valid_intrinsics_region = None

        _validateExtrinsics(model['extrinsics'])

        self._intrinsics                 = intrinsics
        self._valid_intrinsics_region    = mrcal.close_contour(valid_intrinsics_region)
        self._extrinsics                 = np.array(model['extrinsics'], dtype=float)
        self._imagersize                 = np.array(model['imagersize'], dtype=np.int32)

        if 'optimization_inputs' in model:
            if not isinstance(model['optimization_inputs'], bytes):
                raise CameramodelParseException("'optimization_inputs' is given, but it's not a byte string. type(optimization_inputs)={}". \
                                                format(type(model['optimization_inputs'])))
            self._optimization_inputs_string           = model['optimization_inputs']

            if 'icam_intrinsics' not in model:
                raise CameramodelParseException("'optimization_inputs' is given, but icam_intrinsics NOT given")
            if not isinstance(model['icam_intrinsics'], int):
                raise CameramodelParseException("'icam_intrinsics' is given, but it's not an int")
            if model['icam_intrinsics'] < 0:
                raise CameramodelParseException("'icam_intrinsics' is given, but it's <0. Must be >= 0")
            self._icam_intrinsics = model['icam_intrinsics']
        else:
            self._optimization_inputs_string          = None
            self._icam_intrinsics = None

    def __init__(self,

                 file_or_model           = None,
                 *,
                 intrinsics              = None,
                 imagersize              = None,
                 extrinsics_Rt_toref     = None,
                 extrinsics_Rt_fromref   = None,
                 extrinsics_rt_toref     = None,
                 extrinsics_rt_fromref   = None,

                 optimization_inputs     = None,
                 icam_intrinsics         = None,
                 icam_extrinsics         = None,

                 valid_intrinsics_region = None ):
        r'''Initialize a new camera-model object

SYNOPSIS

    # reading from a file on disk
    model0 = mrcal.cameramodel('xxx.cameramodel')

    # using discrete arguments
    model1 = mrcal.cameramodel( intrinsics = ('LENSMODEL_PINHOLE',
                                              np.array((fx,fy,cx,cy))),
                                imagersize = (640,480) )

    # using a optimization_inputs dict
    model2 = mrcal.cameramodel( optimization_inputs = optimization_inputs,
                                icam_intrinsics     = 0 )

We can initialize using one of several methods, depending on which arguments are
given. The arguments for the methods we're not using MUST all be None. Methods:

- Read a file on disk. The filename should be given in the 'file_or_model'
  argument (possibly as a positional argument)

- Read a python 'file' object. Similarly, the opened file should be given in the
  'file_or_model' argument (possibly as a poitional argument)

- Copy an existing cameramodel object. Pass the object in the 'file_or_model'
  argument (possibly as a poitional argument)

- Read discrete arguments. The components of this model (intrinsics, extrinsics,
  etc) can be passed-in separetely via separate keyword arguments

- optimization_inputs. If we have an optimization_inputs dict to store, this
  alone may be passed-in, and all the model components can be read from it.

ARGUMENTS

- file_or_model: we read the camera model from a filename or a pre-opened file
  object or from an existing cameramodel object to copy. Both .cameramodel and
  the legacy .cahvor formats are supported. This may be given as a positional
  argument. Everything else may be given only as keyword arguments.

- intrinsics: a tuple (lensmodel, intrinsics_data). If given, 'imagersize' is
  also required. This may be given only as a keyword argument.

- imagersize: tuple (width,height) for the size of the imager. If given,
  'intrinsics' is also required. This may be given only as a keyword argument.

- extrinsics_Rt_toref: numpy array of shape (4,3) describing the Rt
  transformation FROM the camera coordinate system TO the reference coordinate
  system. Exclusive with the other 'extrinsics_...' arguments. If given,
  'intrinsics' and 'imagersize' are both required. If no 'extrinsics_...'
  arguments are given, an identity transformation is set. This may be given only
  as a keyword argument.

- extrinsics_Rt_fromref: numpy array of shape (4,3) describing the Rt
  transformation FROM the reference coordinate system TO the camera coordinate
  system. Exclusive with the other 'extrinsics_...' arguments. If given,
  'intrinsics' and 'imagersize' are both required. If no 'extrinsics_...'
  arguments are given, an identity transformation is set. This may be given only
  as a keyword argument.

- extrinsics_rt_toref: numpy array of shape (6,) describing the rt
  transformation FROM the camera coordinate system TO the reference coordinate
  system. Exclusive with the other 'extrinsics_...' arguments. If given,
  'intrinsics' and 'imagersize' are both required. If no 'extrinsics_...'
  arguments are given, an identity transformation is set. This may be given only
  as a keyword argument.

- extrinsics_rt_fromref: numpy array of shape (6,) describing the rt
  transformation FROM the reference coordinate system TO the camera coordinate
  system. Exclusive with the other 'extrinsics_...' arguments. If given,
  'intrinsics' and 'imagersize' are both required. If no 'extrinsics_...'
  arguments are given, an identity transformation is set. This may be given only
  as a keyword argument.

- optimization_inputs: a dict of arguments to mrcal.optimize() at the optimum.
  These contain all the information needed to populate the camera model (and
  more!). If given, 'icam_intrinsics' is also required. This may be given only
  as a keyword argument.

- icam_intrinsics: integer identifying this camera in the solve defined by
  'optimization_inputs'. If given, 'optimization_inputs' is required. This may
  be given only as a keyword argument.

- icam_extrinsics: optional integer identifying this camera in the solve defined
  by 'optimization_inputs'. If given, 'optimization_inputs' is required. This
  may be given only as a keyword argument. If icam_extrinsics<0 we set the
  extrinsics to the identify transformation. If icam_extrinsics>=0 we look up
  the extrinsics in the optimization_inputs. If omitted, we call
  corresponding_icam_extrinsics() to find the unique pose of this camera; this
  works ONLY if we have a stationary-camera scenario. If we DO have such a
  scenario, omitting icam_extrinsics is recommended

- valid_intrinsics_region': numpy array of shape (N,2). Defines a closed contour
  in the imager pixel space. Points inside this contour are assumed to have
  'valid' intrinsics, with the meaning of 'valid' defined by the user. An array
  of shape (0,2) menas "valid nowhere". This may be given only as a keyword
  argument.

        '''

        Nargs = dict(file_or_model       = 0,
                     discrete            = 0,
                     extrinsics          = 0,
                     optimization_inputs = 0)

        if file_or_model is not None: Nargs['file_or_model'] += 1
        if intrinsics    is not None: Nargs['discrete']      += 1
        if imagersize    is not None: Nargs['discrete']      += 1

        if extrinsics_Rt_toref is not None:
            Nargs['discrete']   += 1
            Nargs['extrinsics'] += 1
        if extrinsics_Rt_fromref is not None:
            Nargs['discrete']   += 1
            Nargs['extrinsics'] += 1
        if extrinsics_rt_toref is not None:
            Nargs['discrete']   += 1
            Nargs['extrinsics'] += 1
        if extrinsics_rt_fromref is not None:
            Nargs['discrete']   += 1
            Nargs['extrinsics'] += 1

        if optimization_inputs is not None:
            Nargs['optimization_inputs'] += 1
        if icam_intrinsics is not None:
            Nargs['optimization_inputs'] += 1
        if icam_extrinsics is not None:
            Nargs['optimization_inputs'] += 1


        if Nargs['file_or_model']:
            if Nargs['discrete'] + \
               Nargs['optimization_inputs']:
                raise Exception("'file_or_model' specified, so none of the other inputs should be")

            if isinstance(file_or_model, cameramodel):
                self._imagersize = np.array(file_or_model._imagersize, dtype=np.int32)
                self._extrinsics = np.array(file_or_model._extrinsics, dtype=float)
                self._intrinsics = (str(file_or_model._intrinsics[0]),
                                    np.array(file_or_model._intrinsics[1], dtype=float))
                if file_or_model._valid_intrinsics_region is not None:
                    self._valid_intrinsics_region = np.array(mrcal.close_contour(file_or_model._valid_intrinsics_region),
                                                             dtype=float)
                else:
                    self._valid_intrinsics_region = None

                if file_or_model._optimization_inputs_string is not None:
                    self._optimization_inputs_string = str(file_or_model._optimization_inputs_string)
                else:
                    self._optimization_inputs_string = None

                if file_or_model._icam_intrinsics is not None:
                    self._icam_intrinsics = int(file_or_model._icam_intrinsics)
                else:
                    self._icam_intrinsics = None
                return




            def parse_as_opencv_or_ros(modelstring):
                r'''Try to parse model as an opencv/ros string

Supports yaml, json. Supports opencv and ros formats. And the output of
"rostopic echo" for "sensor_msgs/CameraInfo" messages. This functions tries to
be general, and accept everything.

This is documented here: https://wiki.ros.org/camera_calibration_parsers

A sample file:

  image_width: 2448
  image_height: 2050
  camera_name: prosilica
  camera_matrix:
    rows: 3
    cols: 3
    data: [4827.94, 0, 1223.5, 0, 4835.62, 1024.5, 0, 0, 1]
  distortion_model: plumb_bob
  distortion_coefficients:
    rows: 1
    cols: 5
    data: [-0.41527, 0.31874, -0.00197, 0.00071, 0]
  rectification_matrix:
    rows: 3
    cols: 3
    data: [1, 0, 0, 0, 1, 0, 0, 0, 1]
  projection_matrix:
    rows: 3
    cols: 4
    data: [4827.94, 0, 1223.5, 0, 0, 4835.62, 1024.5, 0, 0, 0, 1, 0]

A sample sensor_msgs/CameraInfo message:

  $ rostopic echo -n1 -b tst.bag /camera/camera_info

  ....
  height: 600
  width: 960
  distortion_model: "rational_polynomial"
  D: [1.5, 0.4, 0.1, -9.2e-05, 0.1, 1.9, 0.9, 0.2]
  K: [420.1, 0.1, 479.1, 0.1, 420.1, 295.1, 0.1, 0.1, 1.1]
  R: [0.9998926520347595, 0.014629560522735119, -0.0007753203972242773, -0.014624223113059998, 0.9998719692230225, 0.006493249908089638, 0.0008702144841663539, -0.006481214426457882, 0.9999786019325256]
  P: [600.0, 0.0, 480.0, -20.3, 0.0, 600.0, 300.0, 0.0, 0.0, 0.0, 1.0, 0.0]
  ....

These are apparently trying to include rectification in the model, which is
silly: I should be able to have a single camera with extrinsics. And there're no
clear extrinsics stored here either, and I must figure out what is really
intended here.

Here the only "extrinsics" relate the camera to its rectified version.
Rectification may rotate a camera, but may NOT translate it.

- From previous experience, the rotation in R is R_leftrect_cam

- P[:,3] are scaled translations: t*fx as described here:

    https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#stereorectify

- From previous experience, the translation in P[:,3]/fx is
  t_rightrect_leftrect.

So for the purposes of extrinsics, the reference is the "left-rectified" camera

                '''

                def load():
                    try:
                        import yaml
                        return yaml.safe_load(modelstring)
                    except Exception as e:
                        e1 = e
                        pass

                    try:
                        import json
                        return json.loads(modelstring)
                    except Exception as e:
                        e2 = e
                        pass

                    raise CameramodelParseException(f"Couldn't parse yaml (Exception '{e1}') or json (Exception '{e2}')")


                # output
                model = dict()

                # This will raise the exception, as needed
                model_in = load()

                def find_array( key_sequences,
                                dtype, shape,
                                model):
                    r'''Search a model dictionary for a given array

There are multiple ros/opencv data formats that aren't identical, so this
general function is used to find the data. The args are:


- key_sequences: a list of key sequences to look for. A key sequence such as
  ('camera_matrix','data') will find
  model[...][...][...].....['camera_matrix']['data']. At most one such matching
  sequence is allowed. If we find more than one, this function will throw an
  error

- dtype: the requested dtype of the array we look for. If the data can't be
  interpreted in this way, I barf

- shape: the shape of array we look for. This function will try to reshape into
  this shape, and accept any array that may be reshaped in this way. Pass () to
  interpret this value as a scalar. Pass None to not reshape, and pass the resulting array as is

- model: the input dict

                    '''


                    def find_sequence(s,d, at=''):
                        for k in d.keys():
                            if isinstance(k,str) and k == s[0]:
                                if len(s) == 1:
                                    if isinstance(d[k],dict):
                                        return None,None
                                    return d[k], f"{at}['{k}']"
                                if not isinstance(d[k], dict):
                                    return None,None
                                return \
                                    find_sequence(s[1:], d[k], at=f"{at}['{k}']")
                        return None,None

                    matches = [find_sequence(s,model) for s in key_sequences]

                    matches = [(m,at) for m,at in matches if m is not None]
                    if len(matches) == 0:
                        raise CameramodelParseException(f"None of required key sequences '{key_sequences}' found. Must have exactly one")
                    if len(matches) > 1:
                        raise CameramodelParseException(f"More than one of required key sequences '{key_sequences}' found. Must have exactly one")

                    m,at = matches[0]
                    try:
                        m = np.array(m, dtype=dtype)
                    except:
                        raise CameramodelParseException(f"Could not parse model{at} as dtype = {dtype}")

                    if shape is not None:
                        try:
                            m = m.reshape(shape)
                        except:
                            raise CameramodelParseException(f"Could not parse model{at} with shape {shape}. Input has shape {m.shape}")

                        if len(shape) == 0: m = m.item() # extract the value if we have a scalar

                    return m,at


                M,M_at = \
                    find_array( ( ('camera_matrix','data'),
                                  ('camera_matrix',),
                                  ('K',) ),
                                dtype = float,
                                shape = (3,3),
                                model = model_in )
                if M[0,1] != 0 or \
                   M[1,0] != 0 or \
                   M[2,0] != 0 or \
                   M[2,1] != 0 or \
                   M[2,2] != 1:
                    raise CameramodelParseException(f"model {M_at} should have [fx 0 cx; 0 fy cy; 0 0 1] structure")

                P,P_at = \
                    find_array( ( ('projection_matrix','data'),
                                  ('projection_matrix',),
                                  ('P',) ),
                                dtype = float,
                                shape = (3,4),
                                model = model_in )
                if P[1,3] != 0 or \
                   P[2,3] != 0:
                    raise CameramodelParseException(f"model {P_at} expected to have last column of [x*fx,0,0], but instead have: {P[:,3]}")

                try:
                    R,R_at = \
                        find_array( ( ('rotation','data'),
                                      ('rotation',),
                                      ('R',) ),
                                    dtype = float,
                                    shape = (3,3),
                                    model = model_in )
                except:
                    R = mrcal.identity_R()
                    R_at = 'default'

                # Special-case P=0 or R=0. Sometimes I see this. Set everything to identity
                if nps.norm2(P.ravel()) == 0:
                    P[:,:3] = np.eye(3)
                if nps.norm2(R.ravel()) == 0:
                    R       = np.eye(3)

                lensmodel,lensmodel_at = \
                    find_array( ( ('distortion_model',),),
                                dtype = str,
                                shape = (),
                                model = model_in )

                map_lensmodel = \
                    dict(plumb_bob           = 'LENSMODEL_OPENCV5',
                         rational_polynomial = 'LENSMODEL_OPENCV8')

                try:
                    model['lensmodel'] = map_lensmodel[lensmodel]
                except:
                    if lensmodel == 'equidistant':
                        raise CameramodelParseException('"equidistant" OpenCV model not supported yet')
                    else:
                        raise CameramodelParseException(f"Unknown OpenCV model \"{lensmodel}\". I only about: {list(map_lensmodel.keys())}")

                distortion,distortion_at = \
                    find_array( ( ('distortion_coefficients','data'),
                                  ('distortion_coefficients',),
                                  ('D',)),
                                dtype = float,
                                shape = None, # Any shape. Do not reshape
                                model = model_in )

                try:
                    model['intrinsics'] = \
                          [M[0,0],M[1,1],M[0,2],M[1,2]] + \
                          list(distortion)
                except Exception as e:
                    raise CameramodelParseException(f"No model {distortion_at}")
                # not checking len(distortion_coefficients);
                #_read_into_self() will do that

                image_width,image_width_at = \
                    find_array( ( ('image_width',),
                                  ('width',)),
                                dtype = int,
                                shape = (),
                                model = model_in )
                image_height,image_height_at = \
                    find_array( ( ('image_height',),
                                  ('height',)),
                                dtype = int,
                                shape = (),
                                model = model_in )

                model['imagersize'] = [image_width,image_height]

                Rt_ref_cam = np.zeros((4,3), dtype=float)
                Rt_ref_cam[:3,:] = R
                if nps.norm2((nps.matmult(R, R.T) - np.eye(3)).ravel()) > 1e-12:
                    raise CameramodelParseException(f"R must be a valid rotation. Instead it is {R}")


                # In rectified coords ("ref" coords here) I want the camera to
                # sit at -P[:,3] / P[0,0]
                Rt_ref_cam[ 3,:] = -P[:,3] / P[0,0]

                # extrinsics are rt_fromref
                model['extrinsics'] = list(mrcal.rt_from_Rt(mrcal.invert_Rt(Rt_ref_cam)))

                return repr(model)


            # Some readable file. Read it!
            def tryread(f, what):
                r'''Try all the formats I support'''

                modelstring = f.read()

                errors = dict()
                try:
                    self._read_into_self(modelstring)
                    return
                except CameramodelParseException as e:
                    errors['cameramodel'] = e
                    pass

                try:
                    self._read_into_self(parse_as_opencv_or_ros(modelstring))
                    return
                except CameramodelParseException as e:
                    errors['yaml_or_json'] = e
                    pass


                # Couldn't read the file in any known method. Last try: does a
                # .cahvor work?

                # This is more complicated than it looks. I want to read the
                # .cahvor file into self, but the current cahvor interface
                # wants to generate a new model object. So I do that, write
                # it as a .cameramodel-formatted string, and then read that
                # back into self. Inefficient, but this is far from a hot
                # path
                try:
                    from . import cahvor
                    model = cahvor.read_from_string(modelstring)
                    modelfile = io.StringIO()
                    model.write(modelfile)
                    self._read_into_self(modelfile.getvalue())
                    return
                except Exception as e:
                    errors['cahvor'] = e

                raise Exception(f"Couldn't parse {what}. Errors for each attempt: {errors}")

            if isinstance(file_or_model, str):

                if file_or_model == '-':
                    if sys.stdin.isatty():
                        # This isn't an error per-se. But most likely the user
                        # ran something like "mrcal-to-cahvor" without
                        # redirecting any data into it. Without this check the
                        # program will sit there, waiting for input. Which will
                        # look strange to an unsuspecting user
                        raise Exception("Trying to read a model from standard input, but no file is being redirected into it")
                    tryread(sys.stdin, "STDIN")
                else:
                    with open(file_or_model, 'r') as openedfile:
                        tryread(openedfile, f"file '{file_or_model}'")
            else:
                # I assume this is a readable file
                tryread(file_or_model, "file object")

            return




        if Nargs['discrete']:

            if Nargs['file_or_model'] + \
               Nargs['optimization_inputs']:
                raise Exception("discrete values specified, so none of the other inputs should be")

            if Nargs['discrete']-Nargs['extrinsics'] != 2:
                raise Exception("Discrete values given. Must have gotten 'intrinsics' AND 'imagersize' AND optionally ONE of the extrinsics_...")

            if Nargs['extrinsics'] == 0:
                # No extrinsics. Use the identity
                self.extrinsics_rt_fromref(np.zeros((6,),dtype=float))
            elif Nargs['extrinsics'] == 1:
                if   extrinsics_Rt_toref   is not None: self.extrinsics_Rt_toref  (extrinsics_Rt_toref)
                elif extrinsics_Rt_fromref is not None: self.extrinsics_Rt_fromref(extrinsics_Rt_fromref)
                elif extrinsics_rt_toref   is not None: self.extrinsics_rt_toref  (extrinsics_rt_toref)
                elif extrinsics_rt_fromref is not None: self.extrinsics_rt_fromref(extrinsics_rt_fromref)
            else:
                raise Exception("At most one of the extrinsics_... arguments may be given")

            self.intrinsics(intrinsics, imagersize=imagersize)

        elif Nargs['optimization_inputs']:
            if Nargs['file_or_model'] + \
               Nargs['discrete']:
                raise Exception("optimization_inputs specified, so none of the other inputs should be")

            # Not looking at Nargs['optimization_inputs'] here because
            # icam_extrinsics is optional. so ==2 may or may not be valid
            if optimization_inputs is None or \
               icam_intrinsics is None:
                raise Exception("optimization_input given. Must have gotten 'optimization_input' AND 'icam_intrinsics'")

            self.intrinsics( ( optimization_inputs['lensmodel'],
                               optimization_inputs['intrinsics'][icam_intrinsics] ),
                             imagersize = optimization_inputs['imagersizes'][icam_intrinsics],
                             optimization_inputs = optimization_inputs,
                             icam_intrinsics     = icam_intrinsics)

            if icam_extrinsics is None:
                icam_extrinsics = mrcal.corresponding_icam_extrinsics(icam_intrinsics,
                                                                      **optimization_inputs)
            if icam_extrinsics < 0:
                self.extrinsics_rt_fromref(mrcal.identity_rt())
            else:
                self.extrinsics_rt_fromref(optimization_inputs['extrinsics_rt_fromref'][icam_extrinsics])

        else:
            raise Exception("At least one source of initialization data must have been given. Need a filename or a cameramodel object or discrete arrays or optimization_inputs")


        self._valid_intrinsics_region = None # default
        if valid_intrinsics_region is not None:
            try:
                self.valid_intrinsics_region(valid_intrinsics_region)
            except Exception as e:
                warnings.warn("Invalid valid_intrinsics region; skipping: '{}'".format(e))


    def __str__(self):
        '''Stringification

        Return what would be written to a .cameramodel file'''

        f = io.StringIO()
        self._write(f)
        return f.getvalue()

    def __repr__(self):
        '''Representation

        Return a string of a constructor function call'''

        funcs = (self.imagersize,
                 self.intrinsics,
                 self.extrinsics_rt_fromref,
                 self.valid_intrinsics_region)

        return 'mrcal.cameramodel(' + \
            ', '.join( f.__func__.__code__.co_name + '=' + repr(f()) for f in funcs ) + \
            ')'


    def write(self, f,
              *,
              note    = None,
              cahvor  = False,
              _opencv = False):
        r'''Write out this camera model to disk

SYNOPSIS

    model.write('left.cameramodel')

We write the contents of the given mrcal.cameramodel object to the given
filename or a given pre-opened file. If the filename is 'xxx.cahv' or
'xxx.cahvor' or 'xxx.cahvore' or if cahvor: we use the legacy cahvor file format
for output

ARGUMENTS

- f: a string for the filename or an opened Python 'file' object to use

- note: an optional string, defaulting to None. This is a comment that will be
  written to the top of the output file. This should describe how this model was
  generated

- cahvor: an optional boolean, defaulting to False. If True: we write out the
  data using the legacy .cahvor file format

RETURNED VALUES

None

        '''

        # opencv models may be written by setting _opencv=True. But this has no
        # clear way to specify extrinsics, and requires specifying stereo
        # rectification, so this is undocumented for now
        known_format_options = ('cahvor','_opencv')

        NformatOptions = 0
        for o in known_format_options:
            if locals()[o]:
                NformatOptions += 1
        if NformatOptions > 1:
            raise Exception(f"At most 1 of {known_format_options} may be given.")


        _validateIntrinsics(self._imagersize,
                            self._intrinsics)
        _validateValidIntrinsicsRegion(self._valid_intrinsics_region)
        _validateExtrinsics(self._extrinsics)



        def write_cahvor(f):
            from . import cahvor
            cahvor.write(f, self, note)

        def write_opencv(f):
            r'''Write out an opencv-format file

This has no clear way to specify extrinsics, and requires specifying stereo
rectification, so this is undone and undocumented for now. This function always
sets an identity extrinsics transform and completely made-up projection and
rectification matrices


This is documented here: https://wiki.ros.org/camera_calibration_parsers

A sample file:

  image_width: 2448
  image_height: 2050
  camera_name: prosilica
  camera_matrix:
    rows: 3
    cols: 3
    data: [4827.94, 0, 1223.5, 0, 4835.62, 1024.5, 0, 0, 1]
  distortion_model: plumb_bob
  distortion_coefficients:
    rows: 1
    cols: 5
    data: [-0.41527, 0.31874, -0.00197, 0.00071, 0]
  rectification_matrix:
    rows: 3
    cols: 3
    data: [1, 0, 0, 0, 1, 0, 0, 0, 1]
  projection_matrix:
    rows: 3
    cols: 4
    data: [4827.94, 0, 1223.5, 0, 0, 4835.62, 1024.5, 0, 0, 0, 1, 0]

            '''

            if   self._intrinsics[0] == 'LENSMODEL_OPENCV5':
                distortion_model = 'plumb_bob'
                distortions      = self._intrinsics[1][4:]
            elif self._intrinsics[0] == 'LENSMODEL_OPENCV4':
                distortion_model = 'plumb_bob'
                distortions      = nps.glue(self._intrinsics[1][4:], 0,
                                            axis = -1)
            if   self._intrinsics[0] == 'LENSMODEL_PINHOLE':
                distortion_model = 'plumb_bob'
                distortions      = nps.glue(self._intrinsics[1][4:],
                                            np.zeros((5,), dtype=float),
                                            axis = -1)
            elif self._intrinsics[0] == 'LENSMODEL_OPENCV8':
                distortion_model = 'rational_polynomial'
                distortions      = self._intrinsics[1][4:]
            # I don't support "equidistant" yet
            else:
                raise Exception(f"OpenCV yaml can't store the \"{self._intrinsics[0]}\" model")
            fxycxy = self._intrinsics[1][:4]


            M = np.array(((fxycxy[0], 0,         fxycxy[2]),
                          (        0, fxycxy[1], fxycxy[3]),
                          (        0,         0,         1)),
                         dtype=float)

            f.write(f"image_width: {self._imagersize[0]}\nimage_height: {self._imagersize[1]}\ncamera_name: mrcalmodel\n")

            f.write(f"camera_matrix:\n  rows: 3\n  cols: 3\n  data: [{fxycxy[0]}, 0, {fxycxy[2]}, 0, {fxycxy[1]}, {fxycxy[3]}, 0, 0, 1]\n")
            f.write(f"distortion_model: {distortion_model}\n")
            f.write(f"distortion_coefficients:\n  rows: 1\n  cols: {len(distortions)}\n  data: [")
            np.savetxt(f, nps.atleast_dims(distortions,-2),
                       delimiter=', ',
                       newline  ='',
                       fmt='%.12g')
            f.write("]\n")
            f.write('''rectification_matrix:
  rows: 3
  cols: 3
  data: [1, 0, 0, 0, 1, 0, 0, 0, 1]
projection_matrix:
  rows: 3
  cols: 4
  data: [1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0]
''')


        write_function = None
        if   cahvor:  write_function = write_cahvor
        elif _opencv: write_function = write_opencv

        if isinstance(f, str):
            with open(f, 'w') as openedfile:
                if write_function is not None:
                    write_function(openedfile)
                elif re.match(r".*\.cahv(or(e)?)?$", f):
                    write_cahvor(openedfile)
                elif re.match(r".*\.ya?ml$", f):
                    write_opencv(openedfile)
                else:
                    self._write( openedfile, note )
        else:
            if write_function is not None:
                write_function(f)
            else:
                self._write( f, note )


    def intrinsics(self,
                   intrinsics          = None,
                   *,
                   imagersize          = None,
                   optimization_inputs = None,
                   icam_intrinsics     = None):
        r'''Get or set the intrinsics in this model

SYNOPSIS

    # getter
    lensmodel,intrinsics_data = model.intrinsics()

    # setter
    model.intrinsics( intrinsics          = ('LENSMODEL_PINHOLE',
                                             fx_fy_cx_cy),
                      imagersize          = (640,480),
                      optimization_inputs = optimization_inputs,
                      icam_intrinsics     = 0)

This function has two modes of operation: a getter and a setter, depending on
the arguments.

- If no arguments are given: this is a getter. The getter returns the
  (lensmodel, intrinsics_data) tuple only.

  - lensmodel: a string "LENSMODEL_...". The full list of supported models is
    returned by mrcal.supported_lensmodels()

  - intrinsics_data: a numpy array of shape
    (mrcal.lensmodel_num_params(lensmodel),). For lensmodels that have an
    "intrinsics core" (all of them, currently) the first 4 elements are

    - fx: the focal-length along the x axis, in pixels
    - fy: the focal-length along the y axis, in pixels
    - cx: the projection center along the x axis, in pixels
    - cy: the projection center along the y axis, in pixels

    The remaining elements (both their number and their meaning) are dependent
    on the specific lensmodel being used. Some models (LENSMODEL_PINHOLE for
    example) do not have any elements other than the core.

- If any arguments are given, this is a setter. The setter takes in

  - the (lensmodel, intrinsics_data) tuple
  - (optionally) the imagersize
  - (optionally) optimization_inputs
  - (optionally) icam_intrinsics

  Changing any of these 4 parameters automatically invalidates the others, and
  it only makes sense to set them in unison.

The getters return a copy of the data, and the setters make a copy of the input:
so it's impossible for the caller or callee to modify each other's data.

ARGUMENTS

- intrinsics: a (lensmodel, intrinsics_data) if we're setting; None if we're
  getting

- imagersize: optional iterable of length 2. The (width,height) for the size of
  the imager. If omitted, I use the imagersize already stored in this object.
  This is useful if a valid cameramodel object already exists, and I want to
  update it with new lens parameters

- optimization_inputs: optional dict of arguments to mrcal.optimize() at the
  optimum. These contain all the information needed to populate the camera model
  (and more!). If given, 'icam_intrinsics' is also required. If omitted, no
  optimization_inputs are stored; re-solving and computing of uncertainties is
  impossible.

- icam_intrinsics: optional integer identifying this camera in the solve defined
  by 'optimization_inputs'. May be omitted if 'optimization_inputs' is omitted

RETURNED VALUE

If this is a getter (no arguments given), returns a (lensmodel, intrinsics_data)
tuple where

- lensmodel is a string "LENSMODEL_..."

- intrinsics_data is a numpy array of shape
  (mrcal.lensmodel_num_params(lensmodel),)

        '''

        # This is a getter
        if \
           imagersize          is None and \
           intrinsics          is None and \
           optimization_inputs is None and \
           icam_intrinsics     is None:
            return (str(self._intrinsics[0]),
                    np.array(self._intrinsics[1], dtype=float))


        # This is a setter. The rest of this function does all that work
        if imagersize is None: imagersize = self._imagersize
        _validateIntrinsics(imagersize,
                            intrinsics,
                            optimization_inputs,
                            icam_intrinsics)

        self._imagersize = np.array(imagersize, dtype=np.int32)
        self._intrinsics = (str(intrinsics[0]),
                            np.array(intrinsics[1], dtype=float))

        if optimization_inputs is not None:
            self._optimization_inputs_string = \
                _serialize_optimization_inputs(optimization_inputs)
            self._icam_intrinsics = icam_intrinsics
        else:
            self._optimization_inputs_string = None
            self._icam_intrinsics            = None


    def _extrinsics_rt(self, toref, rt=None):
        r'''Get or set the extrinsics in this model

This function represents the pose as a 6-long numpy array that contains
a 3-long Rodrigues rotation followed by a 3-long translation in the last
row:

  r = rt[:3]
  t = rt[3:]
  R = mrcal.R_from_r(r)

The transformation is b <-- R*a + t:

  import numpysane as nps
  b = nps.matmult(a, nps.transpose(R)) + t

if rt is None: this is a getter; otherwise a setter.

toref is a boolean. if toref: then rt maps points in the coord system of
THIS camera to the reference coord system. Otherwise in the opposite
direction

        '''

        # The internal representation is rt_fromref

        if rt is None:
            # getter
            if not toref:
                return np.array(self._extrinsics)
            return mrcal.invert_rt(self._extrinsics)


        # setter
        if not toref:
            self._extrinsics = np.array(rt, dtype=float)
            return True

        self._extrinsics = mrcal.invert_rt(rt.astype(float))
        return True


    def extrinsics_rt_toref(self, rt=None):
        r'''Get or set the extrinsics in this model

SYNOPSIS

    # getter
    rt_rc = model.extrinsics_rt_toref()

    # setter
    model.extrinsics_rt_toref( rt_rc )

This function gets/sets rt_toref: a numpy array of shape (6,) describing the rt
transformation FROM the camera coordinate system TO the reference coordinate
system.

if rt is None: this is a getter; otherwise a setter.

The getters return a copy of the data, and the setters make a copy of the input:
so it's impossible for the caller or callee to modify each other's data.

ARGUMENTS

- rt: if we're setting, a numpy array of shape (6,). The rt transformation TO
  the reference coordinate system. If we're getting, None

RETURNED VALUE

If this is a getter (no arguments given), returns a a numpy array of shape (6,).
The rt transformation TO the reference coordinate system.

        '''
        return self._extrinsics_rt(True, rt)


    def extrinsics_rt_fromref(self, rt=None):
        r'''Get or set the extrinsics in this model

SYNOPSIS

    # getter
    rt_cr = model.extrinsics_rt_fromref()

    # setter
    model.extrinsics_rt_fromref( rt_cr )

This function gets/sets rt_fromref: a numpy array of shape (6,) describing the
rt transformation FROM the reference coordinate system TO the camera coordinate
system.

if rt is None: this is a getter; otherwise a setter.

The getters return a copy of the data, and the setters make a copy of the input:
so it's impossible for the caller or callee to modify each other's data.

ARGUMENTS

- rt: if we're setting, a numpy array of shape (6,). The rt transformation FROM
  the reference coordinate system. If we're getting, None

RETURNED VALUE

If this is a getter (no arguments given), returns a a numpy array of shape (6,).
The rt transformation FROM the reference coordinate system.

        '''
        return self._extrinsics_rt(False, rt)


    def _extrinsics_Rt(self, toref, Rt=None):
        r'''Get or set the extrinsics in this model

        This function represents the pose as a shape (4,3) numpy array that
        contains a (3,3) rotation matrix, followed by a (1,3) translation in the
        last row:

          R = Rt[:3,:]
          t = Rt[ 3,:]

        The transformation is b <-- R*a + t:

          import numpysane as nps
          b = nps.matmult(a, nps.transpose(R)) + t

        if Rt is None: this is a getter; otherwise a setter.

        toref is a boolean. if toref: then Rt maps points in the coord system of
        THIS camera to the reference coord system. Otherwise in the opposite
        direction

        '''

        # The internal representation is rt_fromref

        if Rt is None:
            # getter
            rt_fromref = self._extrinsics
            Rt_fromref = mrcal.Rt_from_rt(rt_fromref)
            if not toref:
                return Rt_fromref
            return mrcal.invert_Rt(Rt_fromref)

        # setter
        if toref:
            Rt_fromref = mrcal.invert_Rt(Rt.astype(float))
            self._extrinsics = mrcal.rt_from_Rt(Rt_fromref)
            return True

        self._extrinsics = mrcal.rt_from_Rt(Rt.astype(float))
        return True


    def extrinsics_Rt_toref(self, Rt=None):
        r'''Get or set the extrinsics in this model

SYNOPSIS

    # getter
    Rt_rc = model.extrinsics_Rt_toref()

    # setter
    model.extrinsics_Rt_toref( Rt_rc )

This function gets/sets Rt_toref: a numpy array of shape (4,3) describing the Rt
transformation FROM the camera coordinate system TO the reference coordinate
system.

if Rt is None: this is a getter; otherwise a setter.

The getters return a copy of the data, and the setters make a copy of the input:
so it's impossible for the caller or callee to modify each other's data.

ARGUMENTS

- Rt: if we're setting, a numpy array of shape (4,3). The Rt transformation TO
  the reference coordinate system. If we're getting, None

RETURNED VALUE

If this is a getter (no arguments given), returns a a numpy array of shape
(4,3). The Rt transformation TO the reference coordinate system.

        '''
        return self._extrinsics_Rt(True, Rt)


    def extrinsics_Rt_fromref(self, Rt=None):
        r'''Get or set the extrinsics in this model

SYNOPSIS

    # getter
    Rt_cr = model.extrinsics_Rt_fromref()

    # setter
    model.extrinsics_Rt_fromref( Rt_cr )

This function gets/sets Rt_fromref: a numpy array of shape (4,3) describing the
Rt transformation FROM the reference coordinate system TO the camera coordinate
system.

if Rt is None: this is a getter; otherwise a setter.

The getters return a copy of the data, and the setters make a copy of the input:
so it's impossible for the caller or callee to modify each other's data.

ARGUMENTS

- Rt: if we're setting, a numpy array of shape (4,3). The Rt transformation FROM
  the reference coordinate system. If we're getting, None

RETURNED VALUE

If this is a getter (no arguments given), returns a a numpy array of shape
(4,3). The Rt transformation FROM the reference coordinate system.

        '''
        return self._extrinsics_Rt(False, Rt)


    def imagersize(self, *args, **kwargs):
        r'''Get the imagersize in this model

SYNOPSIS

    width,height = model.imagersize()

This function retrieves the dimensions of the imager described by this camera
model. This function is NOT a setter; use intrinsics() to set all the intrinsics
together.

ARGUMENTS

None

RETURNED VALUE

A length-2 tuple (width,height)

        '''
        if len(args) or len(kwargs):
            raise Exception("imagersize() is NOT a setter. Please use intrinsics() to set them all together")

        return np.array(self._imagersize, dtype=np.int32)


    def valid_intrinsics_region(self, valid_intrinsics_region=None):
        r'''Get or set the valid-intrinsics region

SYNOPSIS

    # getter
    region = model.valid_intrinsics_region()

    # setter
    model.valid_intrinsics_region( region )

The valid-intrinsics region is a closed contour in imager space representated as
a numpy array of shape (N,2). This is a region where the intrinsics are
"reliable", with a user-defined meaning of that term. A region of shape (0,2)
means "intrinsics are valid nowhere".

if valid_intrinsics_region is None: this is a getter; otherwise a setter.

The getters return a copy of the data, and the setters make a copy of the input:
so it's impossible for the caller or callee to modify each other's data.

ARGUMENTS

- valid_intrinsics_region: if we're setting, a numpy array of shape (N,2). If
  we're getting, None

RETURNED VALUE

If this is a getter (no arguments given), returns a numpy array of shape
(N,2) or None, if no valid-intrinsics region is defined in this model

        '''
        if valid_intrinsics_region is None:
            # getter
            if self._valid_intrinsics_region is None:
                return None
            return np.array(self._valid_intrinsics_region, dtype=float)

        # setter
        if valid_intrinsics_region is None:
            self._valid_intrinsics_region = None
            return True

        valid_intrinsics_region = mrcal.close_contour(valid_intrinsics_region)

        # raises exception on error
        _validateValidIntrinsicsRegion(valid_intrinsics_region)
        self._valid_intrinsics_region = np.array(valid_intrinsics_region, dtype=float)
        return True


    def optimization_inputs(self):
        r'''Get the original optimization inputs

SYNOPSIS

    b,x,j = mrcal.optimizer_callback(**model.optimization_inputs())[:3]

This function retrieves the optimization inputs: a dict containing all the data
that was used to compute the contents of this model. These are the kwargs
passable to mrcal.optimize() and mrcal.optimizer_callback(), that describe the
optimization problem at its final optimum. A cameramodel object may not contain
this data, in which case we return None.

This function is NOT a setter; use intrinsics() to set all the intrinsics
together. The optimization inputs aren't a part of the intrinsics per se, but
modifying any part of the intrinsics invalidates the optimization inputs, so it
makes sense to set them all together

ARGUMENTS

None

RETURNED VALUE

The optimization_inputs dict, or None if one isn't stored in this model.
        '''

        if self._optimization_inputs_string is None:
            return None
        x = _deserialize_optimization_inputs(self._optimization_inputs_string)
        if x['extrinsics_rt_fromref'] is None:
            x['extrinsics_rt_fromref'] = np.zeros((0,6), dtype=float)
        x['verbose'] = False
        return x


    def optimization_inputs_reset(self):
        r'''Reset the original optimization inputs

SYNOPSIS

    model.optimization_inputs_reset()

    print(model.optimization_inputs() is None)
    ----> True

Clears out the optimization inputs stored in the model

ARGUMENTS

None

RETURNED VALUE

None
        '''
        self._optimization_inputs_string = None


    def _optimization_inputs_match(self, other_model):
        if self.       _optimization_inputs_string is None:
            if other_model._optimization_inputs_string is None:
                return True
            else:
                return False

        if other_model._optimization_inputs_string is None:
            return False;

        # Both non-None
        return \
            self.       _optimization_inputs_string == \
            other_model._optimization_inputs_string


    def _extrinsics_moved_since_calibration(self):
        optimization_inputs = self.optimization_inputs()
        icam_extrinsics = \
            mrcal.corresponding_icam_extrinsics(self.icam_intrinsics(),
                                                **optimization_inputs)

        rt_cam_ref = self.extrinsics_rt_fromref()

        if icam_extrinsics < 0:
            # extrinsics WERE at the reference. So I should have an identity
            # transform
            return np.max(np.abs(rt_cam_ref)) > 0.0

        d = rt_cam_ref - \
            optimization_inputs['extrinsics_rt_fromref'][icam_extrinsics]
        return np.max(np.abs(d)) > 1e-6


    def icam_intrinsics(self):
        r'''Get the camera index indentifying this camera at optimization time

SYNOPSIS

    m = mrcal.cameramodel('xxx.cameramodel')

    optimization_inputs = m.optimization_inputs()

    icam_intrinsics = m.icam_intrinsics()

    icam_extrinsics = \
        mrcal.corresponding_icam_extrinsics(icam_intrinsics,
                                            **optimization_inputs)

    if icam_extrinsics >= 0:
        extrinsics_rt_fromref_at_calibration_time = \
            optimization_inputs['extrinsics_rt_fromref'][icam_extrinsics]
    else:
        extrinsics_rt_fromref_at_calibration_time = \
            mrcal.identity_rt()

This function retrieves the integer identifying this camera in the solve defined
by 'optimization_inputs'. When the optimization happened, we may have been
calibrating multiple cameras at the same time, and only one of those cameras is
described by this 'cameramodel' object. The 'icam_intrinsics' index returned by
this function specifies which camera this is.

This function is NOT a setter; use intrinsics() to set all the intrinsics
together. The optimization inputs and icam_intrinsics aren't a part of the
intrinsics per se, but modifying any part of the intrinsics invalidates the
optimization inputs, so it makes sense to set them all together

ARGUMENTS

None

RETURNED VALUE

The icam_intrinsics integer, or None if one isn't stored in this model.

        '''

        if self._icam_intrinsics is None:
            return None
        return self._icam_intrinsics
