from distutils.core import setup
from Cython.Build import cythonize
from Cython.Distutils import build_ext

import numpy

setup(
    ext_modules = cythonize(["bcfind/fast_threshold.pyx", "bcfind/local_entropy.pyx"]),
    include_dirs=[numpy.get_include()]
)


# python setup.py build_ext --inplace
