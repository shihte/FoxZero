from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import subprocess
import setuptools

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11 until it is actually
    installed, so that the ``get_include()`` method can be invoked. """

    def __str__(self):
        import pybind11
        return pybind11.get_include()

extra_includes = [
    get_pybind_include(),
    '7Game/include'
]

if sys.platform == 'darwin':
    try:
        sdk_path = subprocess.check_output(['xcrun', '--show-sdk-path']).decode('utf-8').strip()
        extra_includes.append(os.path.join(sdk_path, 'usr/include/c++/v1'))
        extra_includes.append(os.path.join(sdk_path, 'usr/include'))
    except Exception:
        pass

ext_modules = [
    Extension(
        'sevens_core',
        [
            'foxzero/sevens_python.cpp',
            '7Game/src/Card.cpp',
            '7Game/src/Hand.cpp',
            '7Game/src/Deck.cpp',
            '7Game/src/PlacedSuit.cpp',
            '7Game/src/SevensGame.cpp',
        ],
        include_dirs=extra_includes,
        language='c++'
    ),
]

def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler."""
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True

class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }
    l_opts = {
        'msvc': [],
        'unix': [],
    }

    if sys.platform == 'darwin':
        pass

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append('-std=c++17') # SevensGame uses C++17 possibly (optional, optional)
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
            
        build_ext.build_extensions(self)

setup(
    name='sevens_core',
    version='0.0.1',
    author='Shih-Te Hsiao',
    description='Sevens Card Game C++ Core Extension',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.5.0'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
