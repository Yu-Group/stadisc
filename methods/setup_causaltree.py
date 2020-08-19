from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy

ext = Extension("causaltree", sources=["causaltree.pyx"])


setup(
    ext_modules=cythonize("causaltree.pyx"),
    include_dirs=[numpy.get_include()]
)   

"""
setup(ext_modules=[ext, include_dirs=[numpy.get_include()]],
      cmdclass={'build_ext': build_ext})

"""