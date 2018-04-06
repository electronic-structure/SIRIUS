from distutils.core import setup
from setuptools import setup, Extension
import glob

ext_modules = [
  Extension(
    "cppmodule",
    glob.glob('src/*.cpp'),
    include_dirs       = ['lib/include', 'lib/pybind11/'],
    language           = 'c++',
    extra_compile_args = ['-std=c++17'],
    define_macros      = [('DOCTEST_CONFIG_DISABLE',None)]
  )


setup(name='py_sirius',
      version='0.1',
      py_modules=['sirius', 'bands', '__init__'],
      ext_modules = ext_modules
      )
