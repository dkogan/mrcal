#!/usr/bin/python3

r'''A module that provides a 'cameramodel' class to read/write/manipulate camera
models.

'''

from __future__ import print_function

import sys
import numpy as np
import numpysane as nps
import numbers
import ast
import re
import warnings
import io

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

class BadCovariance_Exception(Exception):
    pass
class BadValidIntrinsicsRegion_Exception(Exception):
    pass
def _validateIntrinsics(imagersize,
                        i,
                        observed_pixel_uncertainty,
                        covariance_intrinsics_full,
                        covariance_intrinsics,
                        valid_intrinsics_region):
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

    distortion_model = i[0]
    intrinsics       = i[1]

    # If this fails, I keep the exception and let it fall through
    Ndistortions_want = mrcal.getNdistortionParams(distortion_model)

    try:
        Ndistortions_have = len(intrinsics) - 4
    except:
        raise Exception("Valid intrinsics are (distortion_model, intrinsics) where 'intrinsics' is an iterable with a length")

    if Ndistortions_want != Ndistortions_have:
        raise Exception("Mismatched Ndistortions. Got {}, but model {} must have {}".format(Ndistortions_have,distortion_model,Ndistortions_want))

    for x in intrinsics:
        if not isinstance(x, numbers.Number):
            raise Exception("All intrinsics elements should be numeric, but '{}' isn't".format(x))

    def _check_covariance(covariance):
        if covariance is None:
            return

        Nintrinsics = len(intrinsics)

        try:
            s = covariance.shape
            s0 = s[0]
            s1 = s[1]
        except:
            raise BadCovariance_Exception("A valid covariance is an (Nintrinsics,Nintrinsics) positive-semi-definite matrix")

        if not (len(s) == 2 and s0 == Nintrinsics and s1 == Nintrinsics):
            raise BadCovariance_Exception("A valid covariance is an (Nintrinsics,Nintrinsics) positive-semi-definite matrix")

        if not np.allclose(covariance, covariance.transpose()):
            raise BadCovariance_Exception("A valid covariance is an (Nintrinsics,Nintrinsics) positive-semi-definite matrix; this one isn't even symmetric")

        # surely computing ALL the eigenvalues is overkill for just figuring out if
        # this thing is positive-semi-definite or not?
        try:
            eigenvalue_smallest = np.linalg.eigvalsh(covariance)[0]
        except:
            raise BadCovariance_Exception("A valid covariance is an (Nintrinsics,Nintrinsics) positive-semi-definite matrix; couldn't compute eigenvalues")

        if eigenvalue_smallest < -1e-9:
            raise BadCovariance_Exception("A valid covariance is an (Nintrinsics,Nintrinsics) positive-semi-definite matrix; this one isn't positive-semi-definite; smallest eigenvalue: {}".format(eigenvalue_smallest))


    _check_covariance(covariance_intrinsics_full)
    _check_covariance(covariance_intrinsics)

    if observed_pixel_uncertainty is not None:
        try:
            not_positive = observed_pixel_uncertainty <= 0
        except:
            not_positive = True
        if not_positive:
            raise BadCovariance_Exception("observed_pixel_uncertainty must be a positive number")
    if valid_intrinsics_region is not None:
        try:
            # valid intrinsics region is a closed contour, so I need at least 4 points to be valid
            if valid_intrinsics_region.ndim != 2     or \
               valid_intrinsics_region.shape[1] != 2 or \
               valid_intrinsics_region.shape[0] < 4:
                raise BadValidIntrinsicsRegion_Exception("The valid extrinsics region must be a numpy array of shape (N,2) with N >= 4")
        except:
            raise BadValidIntrinsicsRegion_Exception("The valid extrinsics region must be a numpy array of shape (N,2) with N >= 4. Instead got type {} of shape {}". \
                            format(type(valid_intrinsics_region), valid_intrinsics_region.shape if type(valid_intrinsics_region) is np.ndarray else None))

class cameramodel(object):
    r'''A class that encapsulates an extrinsic,intrinsic model of a single camera

    For ONE camera this class represents

    - The intrinsics: parameters internal to this camera. These do not change as
      the camera moves around in space. These include

      - The 4 pinhole-camera parameters: focal lengths, coordinates of the
        center pixel

      - Some representation of the camera distortion. Multiple distortion models
        are supported. mrcal.getSupportedDistortionModels() returns a
        list of supported models.

    - The extrinsics: the pose of this camera in respect to SOME reference
      coordinate system. The meaning of this coordinate system is defined by the
      user of this class: this class itself does not care

    - Optionally, some covariances that represent this camera model as a
      probabilistic quantity: the parameters are all gaussian, with mean at the
      given values, and with some distribution defined by covariance_intrinsics.
      These can be missing or None, in which case they will be assumed to be
      unknown

    This class provides facilities to read/write models, and to get/set the
    various parameters.

    The format of a .cameramodel file is a python dictionary that we eval. A
    sample valid .cameramodel:

        # generated with ...
        { 'distortion_model': 'DISTORTION_OPENCV8',

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
            f.write('# ' + note + '\n')

        _validateIntrinsics(self._imagersize,
                            self._intrinsics,
                            self._observed_pixel_uncertainty,
                            self._covariance_intrinsics_full,
                            self._covariance_intrinsics,
                            self._valid_intrinsics_region)

        _validateExtrinsics(self._extrinsics)

        # I write this out manually instead of using repr for the whole thing
        # because I want to preserve key ordering
        f.write("{\n")
        f.write("    'distortion_model':  '{}',\n".format(self._intrinsics[0]))
        f.write("\n")

        N = len(self._intrinsics[1])
        f.write("    # intrinsics are fx,fy,cx,cy,distortion0,distortion1,....\n")
        f.write(("    'intrinsics': [" + (" {:.10g}," * N) + "],\n").format(*self._intrinsics[1]))
        f.write("\n")

        if self._observed_pixel_uncertainty is not None:
            f.write("    'observed_pixel_uncertainty': {},\n\n".format(self._observed_pixel_uncertainty))
        if self._covariance_intrinsics_full is not None:
            distortion_model = self._intrinsics[0]
            Ndistortions_want = mrcal.getNdistortionParams(distortion_model)
            Nintrinsics = Ndistortions_want+4
            f.write( r'''    # The FULL intrinsics covariance of this model
    #
    # You probably want covariance_intrinsics. These are used primarily for the
    # outlierness-based uncertainty evaluation. See the docstring of
    # mrcal.utils.compute_intrinsics_uncertainty() for a full description of what this
    # is.
    #
    # The intrinsics are represented as a probabilistic quantity: the parameters are
    # all gaussian, with mean at the given values, and with some covariance. The
    # flavor of the covariance returned by this function comes from JtJ in the
    # optimization: this is the block of inv(JtJ) corresponding to the intrinsics of
    # this camera.
''')
            f.write("    'covariance_intrinsics_full': [\n")
            for row in self._covariance_intrinsics_full:
                f.write(("    [" + (" {:.10g}," * Nintrinsics) + "],\n").format(*row))
            f.write("],\n\n")
        if self._covariance_intrinsics is not None:
            distortion_model = self._intrinsics[0]
            Ndistortions_want = mrcal.getNdistortionParams(distortion_model)
            Nintrinsics = Ndistortions_want+4
            f.write( r'''    # The intrinsics covariance of this model
    #
    # This is the covariance of the intrinsics vector that comes from the measurement
    # noise in the calibration process that generated this model. See the docstring of
    # mrcal.utils.compute_intrinsics_uncertainty() for a full description of what this
    # is.
    #
    # The intrinsics are represented as a probabilistic quantity: the parameters are
    # all gaussian, with mean at the given values, and with some inv(JtJ). The flavor
    # of inv(JtJ) returned by this function comes from JtJ in the optimization: this
    # is the block of
    #
    # inv(JtJ) * transpose(Jobservations) Jobservations inv(JtJ)
    #
    # corresponding to the intrinsics of this camera. Jobservations is the
    # rows of J corresponding only to the pixel measurements being perturbed
    # by noise: regularization terms are NOT a part of Jobservations
''')
            f.write("    'covariance_intrinsics': [\n")
            for row in self._covariance_intrinsics:
                f.write(("    [" + (" {:.10g}," * Nintrinsics) + "],\n").format(*row))
            f.write("],\n\n")
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
        f.write(("    'imagersize': [" + (" {:d}," * N) + "]\n").format(*(int(x) for x in self._imagersize)))
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
                raise Exception("Failed to parse cameramodel!\n")
            else:
                raise Exception("Failed to parse cameramodel '{}'\n".format(name))

        keys_required = set(('distortion_model',
                             'intrinsics',
                             'extrinsics',
                             'imagersize'))
        keys_received = set(model.keys())
        if keys_received < keys_required:
            raise Exception("Model must have at least these keys: '{}'. Instead I got '{}'". \
                            format(keys_required, keys_received))

        observed_pixel_uncertainty = model.get('observed_pixel_uncertainty')

        covariance_intrinsics_full = None
        covariance_intrinsics      = None
        if 'covariance_intrinsics_full' in model:
            covariance_intrinsics_full = np.array(model['covariance_intrinsics_full'], dtype=float)
        elif 'invJtJ_intrinsics_full' in model and observed_pixel_uncertainty is not None:
            # compatibility layer to be able to load old-style .cameramodel files
            covariance_intrinsics_full = np.array(model['invJtJ_intrinsics_full'], dtype=float) * \
                observed_pixel_uncertainty*observed_pixel_uncertainty
        if 'covariance_intrinsics' in model:
            covariance_intrinsics = np.array(model['covariance_intrinsics'], dtype=float)
        elif 'invJtJ_intrinsics_observations_only' in model and observed_pixel_uncertainty is not None:
            # compatibility layer to be able to load old-style .cameramodel files
            covariance_intrinsics = np.array(model['invJtJ_intrinsics_observations_only'], dtype=float) * \
                observed_pixel_uncertainty*observed_pixel_uncertainty

        valid_intrinsics_region = None
        if 'valid_intrinsics_region' in model:
            valid_intrinsics_region = np.array(model['valid_intrinsics_region'])

        intrinsics = (model['distortion_model'], np.array(model['intrinsics'], dtype=float))

        try:
            _validateIntrinsics(model['imagersize'],
                                intrinsics,
                                observed_pixel_uncertainty,
                                covariance_intrinsics_full,
                                covariance_intrinsics,
                                valid_intrinsics_region)
        except BadCovariance_Exception as e:
            warnings.warn("Invalid covariance; skipping covariance_intrinsics and valid_intrinsics_region: '{}'".format(e))
            covariance_intrinsics_full = None
            covariance_intrinsics      = None
            valid_intrinsics_region    = None
        except BadValidIntrinsicsRegion_Exception as e:
            warnings.warn("Invalid valid_intrinsics region; skipping: '{}'".format(e))

        _validateExtrinsics(model['extrinsics'])

        self._intrinsics                 = intrinsics
        self._observed_pixel_uncertainty = observed_pixel_uncertainty
        self._covariance_intrinsics_full = covariance_intrinsics_full
        self._covariance_intrinsics      = covariance_intrinsics
        self._valid_intrinsics_region    = mrcal.close_contour(valid_intrinsics_region)
        self._extrinsics                 = np.array(model['extrinsics'], dtype=float)
        self._imagersize                 = np.array(model['imagersize'], dtype=np.int32)


    def __init__(self, file_or_model=None, **kwargs):
        r'''Initializes a new camera-model object

        If file_or_model is not None: we read the camera model from a filename,
        a pre-opened file or from another camera model (copy constructor). In
        this case kwargs MUST be None. If reading a filename, and the filename
        is xxx.cahvor, then we assume a legacy cahvor file format instead of the
        usual one. If the filename is '-' we read standard input

        if file_or_model is None, then the input comes from kwargs, and they
        must NOT be None. The following keys are expected

        - 'intrinsics': REQUIRED tuple (distortion_model, parameters)
        - Exactly ONE or ZERO of the following for the extrinsics (if omitted we
          use an identity transformation):
          - 'extrinsics_Rt_toref'
          - 'extrinsics_Rt_fromref'
          - 'extrinsics_rt_toref'
          - 'extrinsics_rt_fromref'
        - 'imagersize': REQUIRED iterable for the (width,height) of the imager
        - 'observed_pixel_uncertainty': OPTIONAL
        - 'covariance_intrinsics_full': OPTIONAL
        - 'covariance_intrinsics'     : OPTIONAL
        - 'valid_intrinsics_region'   : OPTIONAL

        '''

        if len(kwargs) == 0:
            if file_or_model is None:
                raise Exception("We have neither an existing model to read nor a set of parameters")

            elif type(file_or_model) is cameramodel:
                import copy
                self._imagersize                 = copy.deepcopy(file_or_model._imagersize)
                self._extrinsics                 = copy.deepcopy(file_or_model._extrinsics)
                self._intrinsics                 = copy.deepcopy(file_or_model._intrinsics)
                self._observed_pixel_uncertainty = copy.deepcopy(file_or_model._observed_pixel_uncertainty)
                self._covariance_intrinsics_full = copy.deepcopy(file_or_model._covariance_intrinsics_full)
                self._covariance_intrinsics      = copy.deepcopy(file_or_model._covariance_intrinsics)
                self._valid_intrinsics_region    = copy.deepcopy(mrcal.close_contour(file_or_model._valid_intrinsics_region))


            elif type(file_or_model) is str:

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
                    except:
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
            else:
                self._read_into_self(file_or_model)

        else:
            if file_or_model is not None:
                raise Exception("We have kwargs AND file_or_model. Should have gotten exactly one of these")

            keys_remaining = set( kwargs.keys() )

            if 'intrinsics' not in kwargs:
                raise Exception("No file_or_model was given, so we MUST have gotten an 'intrinsics' kwarg")
            if 'imagersize' not in kwargs:
                raise Exception("No file_or_model was given, so we MUST have gotten a 'imagersize' kwarg")
            keys_remaining -= set(('intrinsics', 'imagersize'))

            extrinsics_keys = set(('extrinsics_Rt_toref',
                                   'extrinsics_Rt_fromref',
                                   'extrinsics_rt_toref',
                                   'extrinsics_rt_fromref'))
            extrinsics_got = keys_remaining.intersection(extrinsics_keys)
            if len(extrinsics_got) == 0:
                # No extrinsics. Use the identity
                self.extrinsics_rt_fromref(np.zeros((6,),dtype=float))
            elif len(extrinsics_got) == 1:
                if 'extrinsics_Rt_toref'   in kwargs: self.extrinsics_Rt_toref  (kwargs['extrinsics_Rt_toref'  ])
                if 'extrinsics_Rt_fromref' in kwargs: self.extrinsics_Rt_fromref(kwargs['extrinsics_Rt_fromref'])
                if 'extrinsics_rt_toref'   in kwargs: self.extrinsics_rt_toref  (kwargs['extrinsics_rt_toref'  ])
                if 'extrinsics_rt_fromref' in kwargs: self.extrinsics_rt_fromref(kwargs['extrinsics_rt_fromref'])
            else:
                raise Exception("No file_or_model was given, so we can take ONE of {}. Instead we got '{}". \
                                format(extrinsics_keys, extrinsics_got))
            keys_remaining -= extrinsics_keys

            keys_remaining -= set(('observed_pixel_uncertainty',
                                   'covariance_intrinsics_full',
                                   'covariance_intrinsics',
                                   'valid_intrinsics_region'),)
            if keys_remaining:
                raise Exception("We were given some unknown parameters: {}".format(keys_remaining))

            self.intrinsics(kwargs['imagersize'],
                            kwargs['intrinsics'],
                            kwargs.get('observed_pixel_uncertainty'),
                            kwargs.get('covariance_intrinsics_full'),
                            kwargs.get('covariance_intrinsics'),
                            kwargs.get('valid_intrinsics_region'))


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
                 self.observed_pixel_uncertainty,
                 self.covariance_intrinsics_full,
                 self.covariance_intrinsics,
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
                   imagersize                 = None,
                   intrinsics                 = None,
                   observed_pixel_uncertainty = None,
                   covariance_intrinsics_full = None,
                   covariance_intrinsics      = None,
                   valid_intrinsics_region    = None):
        r'''Get or set the intrinsics in this model

        if no arguments are given: this is a getter of the INTRINSICS parameters
        only; otherwise this is a setter. As a setter, everything related to the
        lens is set together (dimensions, distortion parameters, uncertainty,
        etc)

        intrinsics is a tuple (distortion_model, parameters):

        - distortion_model is a string for the specific distortion model we're
          using. mrcal.getSupportedDistortionModels() returns a list
          of supported models.

        - parameters is a numpy array of distortion parameters. The first 4
          values are the pinhole-camera parameters: (fx,fy,cx,cy). The following
          values represent the lens distortion. The number and meaning of these
          parameters depends on the distortion model we're using

        '''

        if \
           imagersize                 is None and \
           intrinsics                 is None and \
           observed_pixel_uncertainty is None and \
           covariance_intrinsics_full is None and \
           covariance_intrinsics      is None and \
           valid_intrinsics_region    is None:
            return self._intrinsics

        try:
            _validateIntrinsics(imagersize,
                                intrinsics,
                                observed_pixel_uncertainty,
                                covariance_intrinsics_full,
                                covariance_intrinsics,
                                valid_intrinsics_region)
        except BadCovariance_Exception as e:
            warnings.warn("Invalid covariance_intrinsics; skipping covariance_intrinsics and valid_intrinsics_region: '{}'".format(e))
            covariance_intrinsics_full = None
            covariance_intrinsics      = None
            valid_intrinsics_region    = None
        except BadValidIntrinsicsRegion_Exception as e:
            warnings.warn("Invalid valid_intrinsics region; skipping: '{}'".format(e))

        self._imagersize                 = imagersize
        self._intrinsics                 = intrinsics
        self._observed_pixel_uncertainty = observed_pixel_uncertainty
        self._covariance_intrinsics_full = covariance_intrinsics_full
        self._covariance_intrinsics      = covariance_intrinsics
        self._valid_intrinsics_region    = mrcal.close_contour(valid_intrinsics_region)


    def _extrinsics_rt(self, toref, rt=None):
        r'''Get or set the extrinsics in this model

        This function represents the pose as a 6-long numpy array that contains
        a 3-long Rodrigues rotation followed by a 3-long translation in the last
        row:

          r = rt[:3]
          t = rt[3:]
          R = cv2.Rodrigues(r)[0]

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
                return self._extrinsics
            return mrcal.invert_rt(self._extrinsics)


        # setter
        if not toref:
            self._extrinsics = rt
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
          R = cv2.Rodrigues(r)[0]

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
          R = cv2.Rodrigues(r)[0]

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

        return self._imagersize

    def observed_pixel_uncertainty(self, *args, **kwargs):
        r'''Get the observed pixel uncertainty in this model

        This function is NOT a setter. Use intrinsics() to set all the
        intrinsics together

        '''

        if len(args) or len(kwargs):
            raise Exception("observed_pixel_uncertainty() is NOT a setter. Please use intrinsics() to set them all together")
        return self._observed_pixel_uncertainty

    def covariance_intrinsics_full(self, *args, **kwargs):
        r'''Get the FULL intrinsics covariance for this model

        You probably want covariance_intrinsics(). These are used primarily for
        the outlierness-based uncertainty evaluation. See the docstring of
        mrcal.utils.compute_intrinsics_uncertainty() for a full description of
        what this is.

        The intrinsics are represented as a probabilistic quantity: the
        parameters are all gaussian, with mean at the given values, and with
        some covariance. The flavor of the covariance returned by this function
        comes from JtJ in the optimization: this is the block of inv(JtJ)
        corresponding to the intrinsics of this camera.

        This function is NOT a setter. Use intrinsics() to set all the
        intrinsics together

        '''

        if len(args) or len(kwargs):
            raise Exception("covariance_intrinsics_full() is NOT a setter. Please use intrinsics() to set them all together")
        return self._covariance_intrinsics_full

    def covariance_intrinsics(self, *args, **kwargs):
        r'''Get the intrinsics covariance for this model

        This is the covariance of the intrinsics vector that comes from the
        measurement noise in the calibration process that generated this model.
        See the docstring of mrcal.utils.compute_intrinsics_uncertainty() for a
        full description of what this is.

        The intrinsics are represented as a probabilistic quantity: the
        parameters are all gaussian, with mean at the given values, and with
        some inv(JtJ). The flavor of inv(JtJ) returned by this function comes
        from JtJ in the optimization: this is the block of

            inv(JtJ) * transpose(Jobservations) Jobservations inv(JtJ)

        corresponding to the intrinsics of this camera. Jobservations is the
        rows of J corresponding only to the pixel measurements being perturbed
        by noise: regularization terms are NOT a part of Jobservations

        This function is NOT a setter. Use intrinsics() to set all the
        intrinsics together

        '''

        if len(args) or len(kwargs):
            raise Exception("covariance_intrinsics() is NOT a setter. Please use intrinsics() to set them all together")
        return self._covariance_intrinsics

    def valid_intrinsics_region(self, *args, **kwargs):
        r'''Get the valid-intrinsics region

        This function is NOT a setter. Use intrinsics() to set all the
        intrinsics together.

        The contour is a numpy array of shape (N,2). These are a sequence of
        pixel coordinates describing the shape of the valid region. The first
        and last points will be the same: this is a closed contour

        '''

        if len(args) or len(kwargs):
            raise Exception("valid_intrinsics_region() is NOT a setter. Please use intrinsics() to set them all together")
        return self._valid_intrinsics_region

    def set_cookie(self, cookie):
        r'''Store some arbitrary cookie for somebody to use later

        This data means nothing to me, but the caller may want it'''
        self._cookie = cookie


    def get_cookie(self):
        r'''Retrive some arbitrary cookie from an earlier set_cookie

        This data means nothing to me, but the caller may want it'''
        try:
            return self._cookie
        except:
            return None
