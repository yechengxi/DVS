#!/usr/bin/python3

from distutils.core import setup, Extension
import numpy

# define the extension module
libdvs = Extension('pydvs', sources=['pydvs.c'],
                   include_dirs=[numpy.get_include()])

# run the setup
setup(ext_modules=[libdvs])
