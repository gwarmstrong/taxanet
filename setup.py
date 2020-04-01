from setuptools import setup, Extension

from Cython.Build import cythonize

DEBUG = True

if DEBUG:
    extra_flags = []
else:
    extra_flags = []

extensions = [
    Extension('kraken',
              sources=['kraken_api/kraken.pyx',
                       'src/krakendb.cpp',
                       'src/quickfile.cpp',
                       'src/krakenutil.cpp',
                       ],
              extra_compile_args=["-std=c++11", "-Xpreprocessor",
                                  # "-fopenmp",
                                  # "-lomp",
                                  ] + extra_flags,
              extra_link_args=[] + extra_flags,
              include_dirs=['./src/', "./"],
              library_dirs=['./src'],
              language='c++',
              ),
    Extension(
        "queue",
        language="c",
        sources=["q/queue.pyx",
                 "c-algorithms/src/queue.c",
                 ],
        include_dirs=["c-algorithms/src/", "./" "q/"],
        # cmd_class={'build_ext': build_ext},
    )
]

setup(
    ext_modules=cythonize(extensions, gdb_debug=True),
        # extensions, force=True, language='c++'),
)

# setup(
#     ext_modules=cythonize([Extension("queue", ["q/queue.pyx"])])
# )
