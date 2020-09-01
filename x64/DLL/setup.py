# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 17:30:14 2020

@author: willr
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext_modules = [
    Extension('PhaseCorrelation',
              ['phaseCorr.pyx'],
              # Note here that the C++ language was specified
              # The default language is C
              language="c++",  
              libraries=['PhaseCorrelation'],
              library_dirs=['.'])
    ]

setup(
    name = 'PhaseCorrelation',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
    include_dirs=[np.get_include()]  # This gets all the required Numpy core files
)