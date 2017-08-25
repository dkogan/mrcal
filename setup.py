#!/usr/bin/python2

from setuptools import setup
from distutils.core import Extension

setup(name         = 'mrcal',
      version      = '0.1',
      author       = 'Dima Kogan',
      author_email = 'Dmitriy.Kogan@jpl.nasa.gov',
      ext_modules  = [Extension('mrcal',
                                sources = ['mrcal_pywrap.c'])])
