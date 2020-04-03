from setuptools import setup, Extension, find_packages

from Cython.Build import cythonize

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
              extra_compile_args=["-std=c++11", "-Xpreprocessor",
                                  # "-fopenmp",
                                  # "-lomp",
                                  ] + extra_flags,
              extra_link_args=[] + extra_flags,
              include_dirs=['./kraken_src/', "./"],
              library_dirs=['./kraken_src'],
              language='c++',
              ),
    # Extension(
    #     "queue",
    #     language="c",
    #     sources=["q/queue.pyx",
    #              "c-algorithms/src/queue.c",
    #              ],
    #     include_dirs=["c-algorithms/src/", "./" "q/"],
    #     # cmd_class={'build_ext': build_ext},
    # )
]

setup(
    packages=find_packages(include=['taxanet.*']),
    ext_modules=cythonize(extensions, gdb_debug=DEBUG),
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

# setup(
#     ext_modules=cythonize([Extension("queue", ["q/queue.pyx"])])
# )
