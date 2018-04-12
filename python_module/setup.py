from distutils.core import setup
from setuptools import find_packages

setup(
    name = 'sirius',
    version = '0.0.1',
    description = 'Python binding for the SIRIUS library - a domain specific library for electronic structure calculations',
    packages = find_packages('sirius'),

)
