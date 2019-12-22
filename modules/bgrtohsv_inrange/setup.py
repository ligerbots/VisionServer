import os.path
import numpy

from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='cbgrtohsv_inrange',
    version='1.0',
    author='Paul Rensing',
    author_email='prensing@ligerbots.org',
    ext_modules=cythonize('cbgrtohsv_inrange.pyx'),
    include_dirs=[numpy.get_include()]
)
