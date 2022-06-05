# -*- coding: utf-8 -*-
#******************************************************************#
#
#                     Filename: setup.py
#
#                       Author: kewuaa
#                      Created: 2022-05-23 19:07:11
#                last modified: 2022-06-01 15:32:54
#******************************************************************#
from distutils.core import Extension
from distutils.core import setup

from Cython.Build import cythonize
import numpy as np


ext = Extension('image', sources=['image.pyx'], extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'], include_dirs=['.', np.get_include()])
setup(ext_modules=cythonize(ext))

