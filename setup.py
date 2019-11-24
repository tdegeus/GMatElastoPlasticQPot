desc = '''
Elasto-plastic material model that usage a manifold of quadratic potentials.
'''

from setuptools import setup, Extension

import re
import os
import pybind11
import pyxtensor

header = open('include/GMatElastoPlasticQPot/config.h','r').read()
major = re.split(r'(.*)(\#define GMATELASTOPLASTICQPOT_VERSION_MAJOR\ )([0-9]+)(.*)',header)[3]
minor = re.split(r'(.*)(\#define GMATELASTOPLASTICQPOT_VERSION_MINOR\ )([0-9]+)(.*)',header)[3]
patch = re.split(r'(.*)(\#define GMATELASTOPLASTICQPOT_VERSION_PATCH\ )([0-9]+)(.*)',header)[3]

__version__ = '.'.join([world,major,minor])

ext_modules = [
  Extension(
    'GMatElastoPlasticQPot',
    ['python/main.cpp'],
    include_dirs=[
      os.path.abspath('include/'),
      pyxtensor.find_pyxtensor(),
      pyxtensor.find_pybind11(),
      pyxtensor.find_xtensor(),
      pyxtensor.find_xtl()],
    language='c++'
  ),
]

setup(
  name = 'GMatElastoPlasticQPot',
  description = 'Elasto-plastic material model',
  long_description = desc,
  version = __version__,
  license = 'MIT',
  author = 'Tom de Geus',
  author_email = 'tom@geus.me',
  url = 'https://github.com/tdegeus/GMatElastoPlasticQPot',
  ext_modules = ext_modules,
  install_requires = ['pybind11>=2.2.0', 'pyxtensor>=0.1.1'],
  cmdclass = {'build_ext': pyxtensor.BuildExt},
  zip_safe = False,
)
