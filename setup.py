#!/usr/bin/python2

from setuptools import setup
from distutils.core import Extension
import os

makelevel = os.environ.get('MAKELEVEL')
if makelevel is None or int(makelevel) < 1:
    raise Exception("Please do not run setup.py directly, use the Makefile instead. The Makefile handles the dependencies")

setup(name         = '_mrcal',
      version      = '0.1',
      author       = 'Dima Kogan',
      author_email = 'Dmitriy.Kogan@jpl.nasa.gov',
      ext_modules  = [Extension('_mrcal',
                                sources = ['mrcal_pywrap.c'],
                                extra_compile_args = ['--std=gnu99'],
                                extra_link_args = ['-L{cwd}'         .format(cwd=os.getcwd()),
                                                   '-Wl,-rpath={cwd}'.format(cwd=os.getcwd())],
                                libraries=['mrcal'])])
