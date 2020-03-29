from setuptools import setup, Extension

import numpy

USE_CYTHON = True
VERSION = '0.0.1'


if USE_CYTHON:
    try:
        from Cython.Build import cythonize
    except ImportError:
        USE_CYTHON = False
        cythonize = None
else:
    cythonize = None

ext = '.pyx' if USE_CYTHON else '.c'

include_dirs = []
extensions = [Extension('gauss_map', [f'giwaxs_simulation/gauss_map{ext}'],
                        include_dirs=[numpy.get_include()])]

if USE_CYTHON:
    extensions = cythonize(extensions)
    include_dirs = [numpy.get_include()]


setup(
    name='giwaxs_simulation',
    packages=['giwaxs_simulation'],
    version=VERSION,
    ext_modules=extensions,
    include_dirs=include_dirs,
    author='Vladimir Starostin',
    author_email='vladimir.starostin@uni-tuebingen.de',
    description='GIWAXS-like data simulation.',
    license='MIT',
    include_package_data=True,
    python_requires='>=3.6.*',
    install_requires=['numpy', 'h5py', 'tqdm', 'Cython']
)
