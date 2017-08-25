#!/usr/bin/python2

from setuptools import setup
from distutils.core import Extension
import os

makelevel = os.environ.get('MAKELEVEL')
if makelevel is None or int(makelevel) < 1:
    raise Exception("Please do not run setup.py directly. It needs the Makefile to handle dependencies")

setup(name         = 'mrcal',
      version      = '0.1',
      author       = 'Dima Kogan',
      author_email = 'Dmitriy.Kogan@jpl.nasa.gov',
      ext_modules  = [Extension('mrcal',
                                sources = ['mrcal_pywrap.c'])])
