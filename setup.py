import setuptools
from distutils.extension import Extension
import os
import re


# pull version from package __init__
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "fasttext_keras", "__init__.py"), "r") as rf:
    initfile = rf.read()
vmatch = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", initfile, re.M)
if vmatch:
    version = vmatch.group(1)
else:
    raise RuntimeError("could not find version")


# rebuild from raw pyx if Cython/numpy are available,
# else use the included C++ converted source file
try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
    import numpy as np
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True

cmd_class = {}
if USE_CYTHON:
    print("using Cython to rebuild from .pyx source")
    ext_modules = [
        Extension("fasttext_keras.dictionary.dictionary",
                  ["fasttext_keras/dictionary/dictionary.pyx"],
                  include_dirs=[np.get_include()])
    ]
    ext_modules = cythonize(ext_modules)
    cmd_class.update({'build_ext': build_ext})
else:
    print("building from included C++ extension files")
    ext_modules = [
        Extension("fasttext_keras.dictionary.dictionary",
                  ["fasttext_keras/dictionary/dictionary.cpp"])
    ]


# main setup
setuptools.setup(
    name='fasttext-keras',
    author='John Walk',
    author_email='jwalk@wayfair.com',
    url='https://github.com/wayfair/fasttext-keras',
    ext_modules=ext_modules,
    packages=setuptools.find_packages(),
    cmdclass=cmd_class,
    long_description=open("README.md").read(),
    install_requires=[
        "numpy>=1.15",
        "keras>=2.2.4",
        "tensorflow>=1.12.1"
    ],
    version=version
)
