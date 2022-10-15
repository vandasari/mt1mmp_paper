from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy


ext = Extension("mt1mmp", sources=["mt1mmp.pyx"])

setup(
    ext_modules=cythonize([ext]),
    include_dirs=[numpy.get_include()]
)