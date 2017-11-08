#!/usr/bin/python2

import cv2
import re

import numpy     as np
import numpysane as nps
import numbers

import mrcal
import poseutils


r'''A module that provides a 'cameramodel' class to read/write/manipulate camera
models'''


def _validateExtrinsics(e):
    r'''Raises an exception if the given extrinsics are invalid'''

    # Internal extrinsic representation is a 6-long array
    try:
        N = len(e)
    except:
        raise Exception("Valid extrinsics are an iterable of len 6")


    if N != 6:
        raise Exception("Valid extrinsics are an iterable of len 6")

    for x in extrinsics:
        if not isinstance(x, numbers.Number):
            raise Exception("All extrinsics elements should be numeric, but '{}' isn't".format(x))

    return True


def _validateIntrinsics(i):
    r'''Raises an exception if the given intrinsics are invalid'''

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

    return True

class cameramodel:
    r'''A class that encapsulates an extrinsic,intrinsic model of a single camera

    For ONE camera this class represents

    - The intrinsics: parameters internal to this camera. These do not change as
      the camera moves around in space. These include

      - The 4 pinhole-camera parameters: focal lengths, coordinates of the
        center pixel

      - Some representation of the camera distortion. Multiple distortion models
        are supported. mrcal.getSupportedDistortionModels() returns a list of
        supported models.

    - The extrinsics: the pose of this camera in respect to SOME reference
      coordinate system. The meaning of this coordinate system is defined by the
      user of this class: this class itself does not care

    This class provides facilities to read/write models, and to get/set the
    various parameters

    '''


    def _write(self, f):
        r'''Writes out this camera model to an open file'''

        _validateIntrinsics(self._intrinsics)
        _validateExtrinsics(self._intrinsics)

        f.write("distortion_model = {}\n".format(self._intrinsics[0]))
        f.write("\n")

        N = len(self._intrinsics[1])
        f.write(("intrinsics =" + (" {:15.10f}" * N) + "\n").format(*self._intrinsics[1]))
        f.write("\n")

        N = len(self._extrinsics)
        f.write("# extrinsics are rt_fromref\n")
        f.write(("extrinsics =" + (" {:15.10f}" * N) + "\n").format(*self._extrinsics))
        f.write("\n")


    def _read_and_parse(self, f):
        r'''Reads in a model from an open file'''

        re_f = '[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
        re_u = '\d+'
        re_d = '[-+]?\d+'
        re_s = '.+'

        distortion_model  = None
        intrinsics_values = None
        extrinsics        = None

        for l in f:
            # skip comments and blank lines
            if re.match('^\s*#|^\s*$', l):
                continue

            # all valid lines look like 'key = value'. I allow lots of
            # whitespace, and (value) works too.
            m = re.match('\s*(\w+)\s*=\s*\(?\s*(.+?)\s*\)?\s*\n?$', l)
            if not m:
                raise Exception("All valid non-comment, non-empty lines should be parseable as key=value, but got '{}'".format(l))

            key   = m.group(1)
            value = m.group(2)
            if key == 'distortion_model':
                if distortion_model is not None:
                    raise Exception("Duplicate value for '{}' had '{}' and got new '{}'". \
                                    format('distortion_model', distortion_model, value))
                distortion_model = value
            elif key == 'intrinsics':
                if intrinsics_values is not None:
                    raise Exception("Duplicate value for '{}' had '{}' and got new '{}'". \
                                    format('intrinsics', intrinsics_values, value))
                intrinsics_values = np.array([float(x) for x in value.split(', \t')])
            elif key == 'extrinsics':
                if extrinsics is not None:
                    raise Exception("Duplicate value for '{}' had '{}' and got new '{}'". \
                                    format('extrinsics', extrinsics, value))
                extrinsics = np.array([float(x) for x in value.split(', \t')])
            else:
                raise Exception("Unknown key '{}'. I only know about 'distortion_model', 'intrinsics', 'extrinsics'".format(key))

        if distortion_model is None:
            raise Exception("Unspecified distortion_model")
        if intrinsics_values is None:
            raise Exception("Unspecified intrinsics")
        if extrinsics is None:
            raise Exception("Unspecified extrinsics")

        intrinsics = (distortion_model, intrinsics_values)
        _validateIntrinsics(intrinsics_values)
        _validateIntrinsics(extrinsics)

        self._intrinsics = intrinsics
        self._extrinsics = extrinsics


    def __init__(self, f=None, **kwargs):
        r'''Initializes a new camera-model object

        If f is not None: we read the camera model from a filename, a pre-opened
        file or from another camera model

        If f is None and kwargs is empty: we init the model with invalid
        intrinsics (None initially; will need to be set later), and identity
        extrinsics

        if f is none and we have kwargs: required 'intrinsics' kwarg has the
        intrinsic, and we have extrinsics in one of

        - extrinsics_Rt_toref
        - extrinsics_Rt_fromref
        - extrinsics_rt_toref
        - extrinsics_rt_fromref

        '''

        if len(kwargs) == 0:
            if f is None:
                self._extrinsics = np.zeros(6)
                self._intrinsics = None

            elif type(f) is cameramodel:
                self = f

            elif type(f) is str:
                with open(f, 'r') as openedfile:
                    self._read_and_parse(openedfile)

            else:
                self._read_and_parse(f)

        else:
            if f is not None:
                raise Exception("We have kwargs AND f. These are supposed to be mutually exclusive")

            if 'intrinsics' not in kwargs:
                raise Exception("No f was given, so we MUST have gotten an 'intrinsics' kwarg")
            N=0
            extrinsics_keys=('extrinsics_Rt_toref','extrinsics_Rt_fromref','extrinsics_rt_toref','extrinsics_rt_fromref')
            for k in extrinsics_keys:
                if k in kwargs:
                    N += 1
            if N != 1 or len(kwargs) != 2:
                raise Exception("No f was given, so we MUST have gotten an 'intrinsics' kwarg and one of {}".format(extrinsics_keys))

            self.intrinsics(kwargs['intrinsics'])
            if 'extrinsics_Rt_toref'   in kwargs: self.extrinsics_Rt(True,  kwargs['extrinsics_Rt_toref'  ])
            if 'extrinsics_Rt_fromref' in kwargs: self.extrinsics_Rt(False, kwargs['extrinsics_Rt_fromref'])
            if 'extrinsics_rt_toref'   in kwargs: self.extrinsics_rt(True,  kwargs['extrinsics_rt_toref'  ])
            if 'extrinsics_rt_fromref' in kwargs: self.extrinsics_rt(False, kwargs['extrinsics_rt_fromref'])



    def write(self, f):
        r'''Writes out this camera model

        We write to the given filename or a given pre-opened file.

        '''

        if type(f) is str:
            with open(f, 'w') as openedfile:
                self._write( openedfile )

        else:
            self._write( f )


    def intrinsics(self, i=None):
        r'''Get or set the intrinsics in this model

        if i is None: this is a getter; otherwise a setter.

        i is a tuple (distortion_model, parameters):

        - distortion_model is a string for the specific distortion model we're
          using. mrcal.getSupportedDistortionModels() returns a list of
          supported models.

        - parameters is a numpy array of distortion parameters. The first 4
          values are the pinhole-camera parameters: (fx,fy,cx,cy). The following
          values represent the lens distortion. The number and meaning of these
          parameters depends on the distortion model we're using

        '''

        if i is None:
            return self._intrinsics

        _validateIntrinsics(i)
        self._intrinsics = i


    def extrinsics_rt(self, toref, rt=None):
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
            return poseutils.invert_rt(self._extrinsics)


        # setter
        if not toref:
            self._extrinsics = rt
            return True

        self._extrinsics = poseutils.invert_rt(rt)
        return True


    def extrinsics_Rt(self, toref, Rt=None):
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
            Rt_fromref = poseutils.Rt_from_rt(rt_fromref)
            if not toref:
                return Rt_fromref
            return poseutils.invert_Rt(Rt_fromref)


        # setter
        if toref:
            Rt_fromref = poseutils.invert_Rt(Rt)
            self._extrinsics = poseutils.rt_from_Rt(Rt_fromref)
            return True

        self._extrinsics = poseutils.rt_from_Rt(Rt)
        return True


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
