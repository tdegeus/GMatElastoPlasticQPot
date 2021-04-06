desc = '''
Elasto-plastic material model that usage a manifold of quadratic potentials.
'''

from setuptools import setup, Extension

import re
import os
import pybind11
import pyxtensor
from setuptools_scm import get_version

version = get_version()

include_dirs = [
    os.path.abspath('include/'),
    pyxtensor.find_pyxtensor(),
    pyxtensor.find_pybind11(),
    pyxtensor.find_xtensor(),
    pyxtensor.find_xtl()]

build = pyxtensor.BuildExt

xsimd = pyxtensor.find_xsimd()

if xsimd:
    if len(xsimd) > 0:
        include_dirs += [xsimd]
        build.c_opts['unix'] += ['-march=native', '-DXTENSOR_USE_XSIMD']
        build.c_opts['msvc'] += ['/DXTENSOR_USE_XSIMD']

build.c_opts['unix'] += ['-DGMATELASTOPLASTICQPOT_VERSION="{0:s}"'.format(version)]
build.c_opts['msvc'] += ['/DGMATELASTOPLASTICQPOT_VERSION="{0:s}"'.format(version)]

ext_modules = [Extension(
    'GMatElastoPlasticQPot',
    ['python/main.cpp'],
    include_dirs = include_dirs,
    language = 'c++')]

setup(
    name = 'GMatElastoPlasticQPot',
    description = 'Elasto-plastic material model.',
    long_description = desc,
    keywords = 'Material model; FEM; FFT',
    version = version,
    license = 'MIT',
    author = 'Tom de Geus',
    author_email = 'tom@geus.me',
    url = 'https://github.com/tdegeus/GMatElastoPlasticQPot',
    ext_modules = ext_modules,
    setup_requires = ['pybind11', 'pyxtensor'],
    cmdclass = {'build_ext': build},
    zip_safe = False)
