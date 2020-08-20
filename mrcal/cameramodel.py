#!/usr/bin/python3

'''A 'cameramodel' class to read/write/manipulate camera models'''


import sys
import numpy as np
import numpysane as nps
import numbers
import ast
import re
import warnings
import io
import copy
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

    return True


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

    return True


def _validateValidIntrinsicsRegion(valid_intrinsics_region):

    if valid_intrinsics_region is None:
        return True

    try:
        # valid intrinsics region is a closed contour, so I need at least 4 points to be valid
        if valid_intrinsics_region.ndim != 2     or \
           valid_intrinsics_region.shape[1] != 2 or \
           valid_intrinsics_region.shape[0] < 4:
            raise Exception("The valid extrinsics region must be a numpy array of shape (N,2) with N >= 4")
    except:
        raise Exception("The valid extrinsics region must be a numpy array of shape (N,2) with N >= 4. Instead got type {} of shape {}". \
                        format(type(valid_intrinsics_region), valid_intrinsics_region.shape if type(valid_intrinsics_region) is np.ndarray else None))
    return True


class CameramodelParseException(Exception):
    pass


def _serialize_optimization_inputs(optimization_inputs):
    r'''Convert a optimization-input dict to an ascii string

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
        optimization_inputs_normalized[k] = v

    np.savez_compressed(data_bytes, **optimization_inputs_normalized)
    return \
        base64.b85encode(data_bytes.getvalue())


def _deserialize_optimization_inputs(data_bytes):
    r'''Convert an ascii string for the optimization-input to a full dict

    This is the inverse of _serialize_optimization_inputs(). See the
    docstring of that function for details
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
    if 'calibration_object_width_n'  in optimization_inputs:
        del optimization_inputs['calibration_object_width_n' ]
    if 'calibration_object_height_n' in optimization_inputs:
        del optimization_inputs['calibration_object_height_n']

    return optimization_inputs



class cameramodel(object):
    r'''A class that describes the lens parameters and geometry of a single camera

    This class represents

    - The intrinsics: parameters internal to this camera. These do not change as
      the camera moves around in space. Usually these contain

      - The 4 pinhole-camera parameters: focal lengths, coordinates of the
        center pixel

      - Some representation of the camera projection behavior. This is dependent
        on the lens model being used. The full list of supported models is
        returned by mrcal.supported_lensmodels()

    - The extrinsics: the pose of this camera in respect to some reference
      coordinate system. The meaning of this coordinate system is user-defined,
      and carries no special meaning to the mrcal.cameramodel class

    - The optimization inputs: this is a big blob containing all the data that
      was used to compute the intrinsics and extrinsics in this model. This can
      be used to compute the projection uncertainties, visualize the
      calibration-time frames, and re-solve the calibration problem for
      diagnostics

    This class provides facilities to read/write models, and to get/set the
    various parameters.

    The format of a .cameramodel file is a python dictionary that we eval. A
    sample valid .cameramodel:

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

        _validateIntrinsics(self._imagersize,
                            self._intrinsics)
        _validateValidIntrinsicsRegion(self._valid_intrinsics_region)
        _validateExtrinsics(self._extrinsics)

        # I write this out manually instead of using repr for the whole thing
        # because I want to preserve key ordering
        f.write("{\n")
        f.write("    'lensmodel':  '{}',\n".format(self._intrinsics[0]))
        f.write("\n")

        N = len(self._intrinsics[1])
        if(mrcal.lensmodel_meta(self._intrinsics[0])['has_core']):
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

        Note that the string is NOT a filename, it's the model data'''

        # workaround for python3 idiocy
        try:
            filetype = file
        except:
            filetype = io.IOBase
        if isinstance(f, filetype):
            s    = f.read()
            name = f.name
        else:
            s    = f
            name = None

        try:
            model = ast.literal_eval(s)
        except:
            if name is None:
                raise CameramodelParseException("Failed to parse cameramodel!\n")
            else:
                raise CameramodelParseException("Failed to parse cameramodel '{}'\n".format(name))

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
            valid_intrinsics_region = np.array(model['valid_intrinsics_region'])

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

                 intrinsics              = None,
                 imagersize              = None,
                 extrinsics_Rt_toref     = None,
                 extrinsics_Rt_fromref   = None,
                 extrinsics_rt_toref     = None,
                 extrinsics_rt_fromref   = None,

                 optimization_inputs     = None,
                 icam_intrinsics         = None,

                 valid_intrinsics_region = None ):
        r'''Initializes a new camera-model object

        We read the input in one of several ways. The arguments for the OTHER
        methods must all be None

        - file_or_model: we read the camera model from a filename or a
          pre-opened file object or from an existing cameramodel object to copy.
          If reading a filename, and the filename is xxx.cahvor, then we assume
          the legacy cahvor file format instead of the usual .cameramodel. If
          the filename is '-' we read standard input (both .cahvor and
          .cameramodel supported)

        - discrete arguments. Each thing we want to store is passed in a
          separate argument. Exactly ONE or ZERO of the extrinsics_... arguments
          must be given; if omitted we use an identity transformation. The known
          arguments:

          - intrinsics (a tuple (lensmodel, parameters))
          - imagersize ((width,height) of the imager)
          - extrinsics_Rt_toref
          - extrinsics_Rt_fromref
          - extrinsics_rt_toref
          - extrinsics_rt_fromref

        - optimization_inputs. These are a dict of arguments to mrcal.optimize()
          at the optimum, and contain all the information needed for the camera
          model (and more!) 'icam_intrinsics' is also
          required in this case. This indicates the identity of this camera at
          calibration time

        There's also a 'valid_intrinsics_region' argument, which is optional,
        and works with the discrete arguments or the optimization_inputs.


        file_or_model is first because I want to be able to say
        mrcal.cameramodel(file)

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



        if Nargs['file_or_model']:
            if Nargs['discrete'] + \
               Nargs['optimization_inputs']:
                raise Exception("'file_or_model' specified, so none of the other inputs should be")

            if isinstance(file_or_model, cameramodel):
                self._imagersize                 = copy.deepcopy(file_or_model._imagersize)
                self._extrinsics                 = copy.deepcopy(file_or_model._extrinsics)
                self._intrinsics                 = copy.deepcopy(file_or_model._intrinsics)
                self._valid_intrinsics_region    = copy.deepcopy(mrcal.close_contour(file_or_model._valid_intrinsics_region))
                self._optimization_inputs_string = copy.deepcopy(file_or_model._optimization_inputs_string)
                self._icam_intrinsics            = copy.deepcopy(file_or_model._icam_intrinsics)
                return

            if type(file_or_model) is str:
                if re.match(".*\.cahvor$", file_or_model):
                    # Read a .cahvor. This is more complicated than it looks. I
                    # want to read the .cahvor file into self, but the current
                    # cahvor interface wants to generate a new model object. So
                    # I do that, write it as a .cameramodel-formatted string,
                    # and then read that back into self. Inefficient, but this
                    # is far from a hot path
                    from . import cahvor
                    model = cahvor.read(file_or_model)
                    modelfile = io.StringIO()
                    model.write(modelfile)
                    self._read_into_self(modelfile.getvalue())
                    return

                # Some readable file. Read it!
                def tryread(f):
                    modelstring = f.read()
                    try:
                        self._read_into_self(modelstring)
                    except CameramodelParseException:
                        from . import cahvor
                        model = cahvor.read_from_string(modelstring)
                        modelfile = io.StringIO()
                        model.write(modelfile)
                        self._read_into_self(modelfile.getvalue())
                if file_or_model == '-':
                    tryread(sys.stdin)
                else:
                    with open(file_or_model, 'r') as openedfile:
                        tryread(openedfile)
                return

            self._read_into_self(file_or_model)
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

            self.intrinsics(intrinsics, imagersize)

        elif Nargs['optimization_inputs']:
            if Nargs['file_or_model'] + \
               Nargs['discrete']:
                raise Exception("optimization_inputs specified, so none of the other inputs should be")

            if Nargs['optimization_inputs'] != 2:
                raise Exception("optimization_input given. Must have gotten 'optimization_input' AND 'icam_intrinsics'")


            icam_extrinsics = mrcal.corresponding_icam_extrinsics(icam_intrinsics,
                                                                  **optimization_inputs)

            self.intrinsics( ( optimization_inputs['lensmodel'],
                               optimization_inputs['intrinsics'][icam_intrinsics] ),
                            optimization_inputs['imagersizes'][icam_intrinsics],
                            optimization_inputs,
                            icam_intrinsics)

            icam_extrinsics = mrcal.corresponding_icam_extrinsics(icam_intrinsics,
                                                                  **optimization_inputs)
            if icam_extrinsics < 0:
                self.extrinsics_rt_fromref(np.zeros((6,), dtype=float))
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

        I return what would be written to a .cameramodel file'''

        f = io.StringIO()
        self._write(f)
        return f.getvalue()

    def __repr__(self):
        '''Representation

        I return a string of a constructor function call'''

        funcs = (self.imagersize,
                 self.intrinsics,
                 self.extrinsics_rt_fromref,
                 self.valid_intrinsics_region)

        return 'mrcal.cameramodel(' + \
            ', '.join( f.__func__.__code__.co_name + '=' + repr(f()) for f in funcs ) + \
            ')'


    def write(self, f, note=None, cahvor=False):
        r'''Writes out this camera model

        We write to the given filename or a given pre-opened file. If the
        filename is xxx.cahvor or if the 'cahvor' parameter is True, we use the
        legacy cahvor file format

        '''

        if cahvor:
            from . import cahvor
            cahvor.write(f, self, note)
            return

        if type(f) is str:
            if re.match(".*\.cahvor$", f):
                from . import cahvor
                cahvor.write(f, self, note)

            else:
                with open(f, 'w') as openedfile:
                    self._write( openedfile, note )

        else:
            self._write( f, note )


    def intrinsics(self,
                   intrinsics          = None,
                   imagersize          = None,
                   optimization_inputs = None,
                   icam_intrinsics     = None):
        r'''Get or set the intrinsics in this model

        if no arguments are given: this is a getter of the INTRINSICS parameters
        only; otherwise this is a setter. As a setter, everything related to the
        lens is set together (dimensions, lens parameters, uncertainty, etc).

        "intrinsics" is a tuple (lensmodel, parameters):

        - lensmodel is a string for the specific lens model we're
          using. mrcal.supported_lensmodels() returns a list
          of supported models.

        - intrinsics is a numpy array of lens parameters. The number and meaning
          of these parameters depends on the lens model we're using. For those
          models that have a core the first 4 values are the pinhole-camera
          parameters: (fx,fy,cx,cy), with the following values representing the
          lens distortion.

        This "intrinsics" tuple is required. The other arguments are optional.
        If the imagersize is omitted, the current model's imagersize will be
        used. If anything else is omitted, it will be unset in the model.

        '''

        # This is a getter
        if \
           imagersize          is None and \
           intrinsics          is None and \
           optimization_inputs is None and \
           icam_intrinsics     is None:
            return copy.deepcopy(self._intrinsics)


        # This is a setter. The rest of this function does all that work
        if imagersize is None: imagersize = self._imagersize
        _validateIntrinsics(imagersize,
                            intrinsics,
                            optimization_inputs,
                            icam_intrinsics)

        self._imagersize = copy.deepcopy(imagersize)
        self._intrinsics = copy.deepcopy(intrinsics)

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
                return copy.deepcopy(self._extrinsics)
            return mrcal.invert_rt(self._extrinsics)


        # setter
        if not toref:
            self._extrinsics = copy.deepcopy(rt)
            return True

        self._extrinsics = mrcal.invert_rt(rt)
        return True


    def extrinsics_rt_toref(self, rt=None):
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

        In this function rt maps points in the coord system of THIS camera to
        the reference coord system

        '''
        return self._extrinsics_rt(True, rt)


    def extrinsics_rt_fromref(self, rt=None):
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

        In this function Rt maps points in the REFERENCE coord system to the
        coordinate system of THIS camera

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
            Rt_fromref = mrcal.invert_Rt(Rt)
            self._extrinsics = mrcal.rt_from_Rt(Rt_fromref)
            return True

        self._extrinsics = mrcal.rt_from_Rt(Rt)
        return True


    def extrinsics_Rt_toref(self, Rt=None):
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

        In this function Rt maps points in the coord system of THIS camera to
        the reference coord system

        '''
        return self._extrinsics_Rt(True, Rt)


    def extrinsics_Rt_fromref(self, Rt=None):
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

        In this function Rt maps points in the REFERENCE coord system to the
        coordinate system of THIS camera

        '''
        return self._extrinsics_Rt(False, Rt)


    def imagersize(self, *args, **kwargs):
        r'''Get the imager imagersize in this model

        This function is NOT a setter. Use intrinsics() to set all the
        intrinsics together
        '''
        if len(args) or len(kwargs):
            raise Exception("imagersize() is NOT a setter. Please use intrinsics() to set them all together")

        return copy.deepcopy(self._imagersize)


    def valid_intrinsics_region(self, valid_intrinsics_region=None):
        r'''Get or set the valid-intrinsics region

        The contour is a numpy array of shape (N,2). These are a sequence of
        pixel coordinates describing the shape of the valid region. The first
        and last points will be the same: this is a closed contour

        if valid_intrinsics_region is None: this is a getter; otherwise a
        setter.

        '''
        if valid_intrinsics_region is None:
            # getter
            return copy.deepcopy(self._valid_intrinsics_region)

        # setter
        valid_intrinsics_region = mrcal.close_contour(valid_intrinsics_region)

        # raises exception on error
        _validateValidIntrinsicsRegion(valid_intrinsics_region)
        self._valid_intrinsics_region = copy.deepcopy(valid_intrinsics_region)
        return True


    def optimization_inputs(self):
        r'''Get the original optimization inputs AT THE OPTIMUM

        This function is NOT a setter. Use intrinsics() to set this and all the
        intrinsics together. The optimization inputs aren't a part of the
        intrinsics per se, but modifying any part of the intrinsics invalidates
        the optimization inputs, so it only makes sense to set them all together

        '''

        if self._optimization_inputs_string is None:
            return None
        return _deserialize_optimization_inputs(self._optimization_inputs_string)

    def icam_intrinsics(self):
        r'''Get the camera index at optimization time

        This function is NOT a setter. Use intrinsics() to set this and all the
        intrinsics together. The optimization inputs aren't a part of the
        intrinsics per se, but modifying any part of the intrinsics invalidates
        the optimization inputs, so it only makes sense to set them all together

        '''

        if self._icam_intrinsics is None:
            return None
        return self._icam_intrinsics






