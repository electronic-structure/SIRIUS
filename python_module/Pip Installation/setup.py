import os
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
from setuptools.extension import Extension

this_dir = os.path.dirname(os.path.abspath(__file__))


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)



import1 = os.path.join(this_dir, 'src')
import2 = os.path.join(this_dir, 'libs/gsl_2.4')
my_module = Extension(
    name='sirius_module',
    language='c++',
    include_dirs=[
        get_pybind_include(),
        get_pybind_include(user=True),
        os.path.join(this_dir, 'src'),
        os.path.join(this_dir, 'src', 'SDDK'),
        os.path.join(this_dir, 'libs', 'gsl_2.4'),
        #os.path.join(this_dir, 'libs', 'gsl', 'gsl_sf_bessel.h'),
        os.path.join(this_dir, 'libs', 'gsl', '/fftw-3.3.7/api'),
        os.path.join(this_dir, 'libs', 'libxc-3.0.0/src'),
        os.path.join(this_dir, 'libs', 'libxc-3.0.0'),
        os.path.join(this_dir, 'libs', 'hdf5-1.10.1/src'),
        os.path.join(this_dir, 'libs', 'spglib-1.9.9/src'),
    ],
    extra_compile_args=["-fopenmp", "-std=c++14"],
    extra_link_args=["-fopenmp", "-shared", "-undefined dynamic_lookup", "`python3.6 -m pybind11 --includes`", "-I" + import1, "-I" + import2],

    extra_objects = [
    os.path.join(this_dir, 'src', 'libsirius.a'),
    os.path.join(this_dir, 'libs', 'gsl_2.4', '.libs', 'libgsl.a')],
    sources=[
        os.path.join(this_dir, 'python_module', 'py_sirius.cpp'),
        #os.path.join(this_dir, 'src', 'sirius.cpp'),
    ],
    library_dirs = [
        os.path.join(this_dir, 'src'),
        os.path.join(this_dir, 'libs', 'gsl_2.4', '.libs')

    ],
    libraries = ['libgsl.a', 'libsirius.a'],
)

def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True

def cpp_flag(compiler):

    if has_flag(compiler, '-std=c++14'): return '-std=c++14'
    else: raise RuntimeError('Unsupported compiler -- at least C++14 support is needed!')

class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        os.environ["CC"] = "mpic++"

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_args = []

        if ct == 'unix':
            opts.append("-llapack -lblas -lz -lgfortran -lstdc++ -lc++")
            opts.append("-fopenmp")
            link_args.append('-fopenmp')
            opts.append("-shared")
            opts.append("-undefined dynamic_lookup")
            #opts.append("`python3.6 -m pybind11 --includes`
            #opts.append("-I" + import1 + " -I" + import2)
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            #opts.append(cpp_flag(self.compiler))
            #opts.append()
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')

        for ext in self.extensions:
            ext.extra_compile_args += opts
            ext.extra_link_args += link_args

        build_ext.build_extensions(self)

setup(
    name = 'sirius',
    version = '0.0.1',
    description = 'Python binding for the SIRIUS library - a domain specific library for electronic structure calculations',
    packages = find_packages('sirius'),
    ext_modules = [my_module],
    zip_safe = False,
    install_requires = ['pybind11>=2.1.0'],
    setup_requires = ['pybind11>=2.1.0'],
    cmdclass={'build_ext': BuildExt},
    include_package_data = True,
)
