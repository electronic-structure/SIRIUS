from setuptools import setup, find_packages

setup(
    name = 'sirius',
    version = '0.0.3',
    url = 'https://github.com/electronic-structure/SIRIUS/tree/develop/python_module',
    description = 'Python binding for the SIRIUS library - a domain specific library for electronic structure calculations',
    packages = ['sirius'],
    package_data={'sirius': ['includes/path_dict.json']},
    include_package_data = True,
    zip_safe = False,

)
