from setuptools import setup, Extension, find_packages
import os
import platform
import minimap2
import mmappy

from Cython.Build import cythonize

compiler_directives = dict()

if platform.system() == 'Linux':
    compiler_directives['language_level'] = '3'

DEBUG = False

if DEBUG:
    extra_flags = []
else:
    extra_flags = []

extensions = [
    Extension('pykraken._kraken',
              sources=['pykraken/_kraken.pyx',
                       'kraken_src/krakendb.cpp',
                       'kraken_src/quickfile.cpp',
                       'kraken_src/krakenutil.cpp',
                       ],
              extra_compile_args=["-std=c++11",
                  ] + extra_flags,
              extra_link_args=[] + extra_flags,
              include_dirs=['./kraken_src/', "./",
                            os.path.dirname(minimap2.__file__),
                            os.path.dirname(mmappy.__file__),
                            ],
              library_dirs=['./kraken_src'],
              language='c++',
              ),
]

setup(
    name='taxanet',
    packages=find_packages(include=['taxanet.*']),
    ext_modules=cythonize(extensions, gdb_debug=DEBUG,
                          compiler_directives=compiler_directives,
                          ),
    install_requires=[
        'torch>=1.4',
        'logomaker',
        'pandas',
        'matplotlib',
        'seaborn',
        'numpy',
        'scikit-learn',
    ]
)
