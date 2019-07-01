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
                        observed_pixel_uncertainty,
                        invJtJ_intrinsics_full,
                        invJtJ_intrinsics_observations_only,
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

    def check_invJtJ(invJtJ):
        if invJtJ is None:
            return

        Nintrinsics = len(intrinsics)

        try:
            s = invJtJ.shape
            s0 = s[0]
            s1 = s[1]
        except:
            raise Exception("A valid invJtJ is an (Nintrinsics,Nintrinsics) positive-semi-definite matrix")

        if not (len(s) == 2 and s0 == Nintrinsics and s1 == Nintrinsics):
            raise Exception("A valid invJtJ is an (Nintrinsics,Nintrinsics) positive-semi-definite matrix")

        if not np.allclose(invJtJ, invJtJ.transpose()):
            raise Exception("A valid invJtJ is an (Nintrinsics,Nintrinsics) positive-semi-definite matrix; this one isn't even symmetric")

        # surely computing ALL the eigenvalues is overkill for just figuring out if
        # this thing is positive-semi-definite or not?
        try:
            eigenvalue_smallest = np.linalg.eigvalsh(invJtJ)[0]
        except:
            raise Exception("A valid invJtJ is an (Nintrinsics,Nintrinsics) positive-semi-definite matrix; couldn't compute eigenvalues")

        if eigenvalue_smallest < -1e-9:
            raise Exception("A valid invJtJ is an (Nintrinsics,Nintrinsics) positive-semi-definite matrix; this one isn't positive-semi-definite; smallest eigenvalue: {}".format(eigenvalue_smallest))


    check_invJtJ(invJtJ_intrinsics_full)
    check_invJtJ(invJtJ_intrinsics_observations_only)

    if observed_pixel_uncertainty is not None:
        try:
            not_positive = observed_pixel_uncertainty <= 0
        except:
            not_positive = True
        if not_positive:
            raise Exception("observed_pixel_uncertainty must be a positive number")
    else:
        if invJtJ_intrinsics_full              is not None or \
           invJtJ_intrinsics_observations_only is not None:
            raise Exception("If any invJtJ are given, the observed pixel uncertainty must be given too")

    if valid_intrinsics_region is not None:
        try:
            # valid intrinsics region is a closed contour, so I need at least 4 points to be valid
            if valid_intrinsics_region.ndim != 2     or \
               valid_intrinsics_region.shape[1] != 2 or \
               valid_intrinsics_region.shape[0] < 4:
                raise Exception("The valid extrinsics region must be a numpy array of shape (N,2) with N >= 4")
        except:
            raise Exception("The valid extrinsics region must be a numpy array of shape (N,2) with N >= 4. Instead got type {} of shape {}". \
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

    - Optionally, some invJtJ and observed_pixel_uncertainty that represent this
      camera model as a probabilistic quantity: the parameters are all gaussian,
      with mean at the given values, and with some distribution defined by
      invJtJ and observed_pixel_uncertainty. These can be missing or None, in
      which case they will be assumed to be 0: no uncertainty exists.

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
                            self._invJtJ_intrinsics_full,
                            self._invJtJ_intrinsics_observations_only,
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
        if self._invJtJ_intrinsics_full is not None:
            distortion_model = self._intrinsics[0]
            Ndistortions_want = mrcal.getNdistortionParams(distortion_model)
            Nintrinsics = Ndistortions_want+4
            f.write( '    # An inv(JtJ) used for the outlierness-based uncertainty computations\n')
            f.write("    'invJtJ_intrinsics_full': [\n")
            for row in self._invJtJ_intrinsics_full:
                f.write(("    [" + (" {:.10g}," * Nintrinsics) + "],\n").format(*row))
            f.write("],\n\n")
        if self._invJtJ_intrinsics_observations_only is not None:
            distortion_model = self._intrinsics[0]
            Ndistortions_want = mrcal.getNdistortionParams(distortion_model)
            Nintrinsics = Ndistortions_want+4
            f.write( r'''    # The intrinsics are represented as a probabilistic quantity: the parameters
    # are all gaussian, with mean at the given values, and with some inv(JtJ).
    # This inv(JtJ) comes from the uncertainty of the pixel observations in
    # the calibration process
''')
            f.write("    'invJtJ_intrinsics_observations_only': [\n")
            for row in self._invJtJ_intrinsics_observations_only:
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
        r'''Reads in a model from an open file'''

        s = ''.join(f)

        try:
            model = ast.literal_eval(s)
        except:
            name = f.name
            if name is None:
                sys.stderr.write("Failed to parse cameramodel!\n")
            else:
                sys.stderr.write("Failed to parse cameramodel '{}'\n".format(name))
            raise

        keys_required = set(('distortion_model',
                             'intrinsics',
                             'extrinsics',
                             'imagersize'))
        keys_received = set(model.keys())
        if keys_received < keys_required:
            raise Exception("Model must have at least these keys: '{}'. Instead I got '{}'". \
                            format(keys_required, keys_received))

        observed_pixel_uncertainty = model.get('observed_pixel_uncertainty')

        invJtJ_intrinsics_full              = None
        invJtJ_intrinsics_observations_only = None
        if 'invJtJ_intrinsics_full' in model:
            invJtJ_intrinsics_full = np.array(model['invJtJ_intrinsics_full'], dtype=float)
        if 'invJtJ_intrinsics_observations_only' in model:
            invJtJ_intrinsics_observations_only = np.array(model['invJtJ_intrinsics_observations_only'], dtype=float)

        valid_intrinsics_region = None
        if 'valid_intrinsics_region' in model:
            valid_intrinsics_region = np.array(model['valid_intrinsics_region'])

        intrinsics = (model['distortion_model'], np.array(model['intrinsics'], dtype=float))

        _validateIntrinsics(model['imagersize'],
                            intrinsics,
                            observed_pixel_uncertainty,
                            invJtJ_intrinsics_full,
                            invJtJ_intrinsics_observations_only,
                            valid_intrinsics_region)
        _validateExtrinsics(model['extrinsics'])

        self._intrinsics                          = intrinsics
        self._observed_pixel_uncertainty          = observed_pixel_uncertainty
        self._invJtJ_intrinsics_full              = invJtJ_intrinsics_full
        self._invJtJ_intrinsics_observations_only = invJtJ_intrinsics_observations_only
        self._valid_intrinsics_region             = mrcal.close_contour(valid_intrinsics_region)
        self._extrinsics                          = np.array(model['extrinsics'], dtype=float)
        self._imagersize                          = np.array(model['imagersize'], dtype=np.int32)


    def __init__(self, file_or_model=None, **kwargs):
        r'''Initializes a new camera-model object

        If file_or_model is not None: we read the camera model from a filename,
        a pre-opened file or from another camera model (copy constructor). If
        reading a filename, and the filename is xxx.cahvor, then we assume a
        legacy cahvor file format instead of the usual one

        If f is None and kwargs is empty: we init the model with invalid
        intrinsics (None initially; will need to be set later), and identity
        extrinsics

        if f is none and we have kwargs:

        - 'intrinsics'
        - Exactly ONE of the following for theextrinsics:
          - 'extrinsics_Rt_toref'
          - 'extrinsics_Rt_fromref'
          - 'extrinsics_rt_toref'
          - 'extrinsics_rt_fromref'
        - 'imagersize'
        - 'observed_pixel_uncertainty',          optionally
        - 'invJtJ_intrinsics_full',              optionally
        - 'invJtJ_intrinsics_observations_only', optionally
        - 'valid_intrinsics_region',     optionally

        '''

        # special-case cahvor logic. This is here purely for legacy
        # compatibility
        if len(kwargs) == 0           and \
           type(file_or_model) is str and \
           re.match(".*\.cahvor$", file_or_model):
            from . import cahvor
            file_or_model = cahvor.read(file_or_model)
            # now follow this usual path. This becomes a copy constructor.





        if len(kwargs) == 0:
            if file_or_model is None:
                self._extrinsics = np.zeros(6)
                self._intrinsics = None

            elif type(file_or_model) is cameramodel:
                import copy
                self._imagersize                          = copy.deepcopy(file_or_model._imagersize)
                self._extrinsics                          = copy.deepcopy(file_or_model._extrinsics)
                self._intrinsics                          = copy.deepcopy(file_or_model._intrinsics)
                self._observed_pixel_uncertainty          = copy.deepcopy(file_or_model._observed_pixel_uncertainty)
                self._invJtJ_intrinsics_full              = copy.deepcopy(file_or_model._invJtJ_intrinsics_full)
                self._invJtJ_intrinsics_observations_only = copy.deepcopy(file_or_model._invJtJ_intrinsics_observations_only)
                self._valid_intrinsics_region             = copy.deepcopy(mrcal.close_contour(file_or_model._valid_intrinsics_region))


            elif type(file_or_model) is str:
                with open(file_or_model, 'r') as openedfile:
                    self._read_into_self(openedfile)

            else:
                self._read_into_self(file_or_model)

        else:
            if file_or_model is not None:
                raise Exception("We have kwargs AND file_or_model. These are supposed to be mutually exclusive")

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
            if len(extrinsics_got) != 1:
                raise Exception("No file_or_model was given, so we MUST have gotten one of {}".format(extrinsics_keys))
            keys_remaining -= extrinsics_keys
            if keys_remaining - set(('observed_pixel_uncertainty',
                                     'invJtJ_intrinsics_full',
                                     'invJtJ_intrinsics_observations_only',
                                     'valid_intrinsics_region'),):
                raise Exception("No file_or_model was given, so we MUST have gotten 'intrinsics', 'extrinsics_...', 'imagersize' and MAYBE 'invJtJ_intrinsics_full' and/or 'invJtJ_intrinsics_observations_only' and/or 'valid_intrinsics_region' and/or 'observed_pixel_uncertainty'. Questionable keys: '{}'".format(keys_remaining))

            self.intrinsics(kwargs['imagersize'],
                            kwargs['intrinsics'],
                            kwargs.get('observed_pixel_uncertainty'),
                            kwargs.get('invJtJ_intrinsics_full'),
                            kwargs.get('invJtJ_intrinsics_observations_only'),
                            kwargs.get('valid_intrinsics_region'))

            if 'extrinsics_Rt_toref'   in kwargs: self.extrinsics_Rt_toref  (kwargs['extrinsics_Rt_toref'  ])
            if 'extrinsics_Rt_fromref' in kwargs: self.extrinsics_Rt_fromref(kwargs['extrinsics_Rt_fromref'])
            if 'extrinsics_rt_toref'   in kwargs: self.extrinsics_rt_toref  (kwargs['extrinsics_rt_toref'  ])
            if 'extrinsics_rt_fromref' in kwargs: self.extrinsics_rt_fromref(kwargs['extrinsics_rt_fromref'])



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
                   imagersize                          = None,
                   intrinsics                          = None,
                   observed_pixel_uncertainty          = None,
                   invJtJ_intrinsics_full              = None,
                   invJtJ_intrinsics_observations_only = None,
                   valid_intrinsics_region     = None):
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
           imagersize                          is None and \
           intrinsics                          is None and \
           observed_pixel_uncertainty          is None and \
           invJtJ_intrinsics_full              is None and \
           invJtJ_intrinsics_observations_only is None and \
           valid_intrinsics_region     is None:
            return self._intrinsics

        _validateIntrinsics(imagersize,
                            intrinsics,
                            observed_pixel_uncertainty,
                            invJtJ_intrinsics_full,
                            invJtJ_intrinsics_observations_only,
                            valid_intrinsics_region)

        self._imagersize                          = imagersize
        self._intrinsics                          = intrinsics
        self._observed_pixel_uncertainty          = observed_pixel_uncertainty
        self._invJtJ_intrinsics_full              = invJtJ_intrinsics_full
        self._invJtJ_intrinsics_observations_only = invJtJ_intrinsics_observations_only
        self._valid_intrinsics_region             = mrcal.close_contour(valid_intrinsics_region)


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

    def invJtJ_intrinsics_full(self, *args, **kwargs):
        r'''Get an intrinsics invJtJ in this model

        This function looks at the FULL invJtJ. This is used for the
        outlierness-based uncertainty computations

        This function is NOT a setter. Use intrinsics() to set all the
        intrinsics together

        '''

        if len(args) or len(kwargs):
            raise Exception("invJtJ_intrinsics_full() is NOT a setter. Please use intrinsics() to set them all together")
        return self._invJtJ_intrinsics_full

    def invJtJ_intrinsics_observations_only(self, *args, **kwargs):
        r'''Get an intrinsics invJtJ in this model

        This function looks at the observations-only invJtJ. This is used
        for the uncertainty based on the noise of input observations to the
        calibration routine

        This function is NOT a setter. Use intrinsics() to set all the
        intrinsics together

        '''

        if len(args) or len(kwargs):
            raise Exception("invJtJ_intrinsics_observations_only() is NOT a setter. Please use intrinsics() to set them all together")
        return self._invJtJ_intrinsics_observations_only

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
