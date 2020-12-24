"""Setup file for GewitterGefahr."""

from setuptools import setup

PACKAGE_NAMES = [
    'gewittergefahr', 'gewittergefahr.gg_io', 'gewittergefahr.gg_utils',
    'gewittergefahr.deep_learning', 'gewittergefahr.plotting',
    'gewittergefahr.scripts', 'gewittergefahr.feature_selection_example',
    'gewittergefahr.nature2019'
]

KEYWORDS = [
    'machine learning', 'deep learning', 'artificial intelligence',
    'data mining', 'weather', 'meteorology', 'thunderstorm', 'wind', 'tornado'
]

SHORT_DESCRIPTION = (
    'End-to-end machine-learning library for predicting thunderstorm hazards.')

LONG_DESCRIPTION = (
    'GewitterGefahr is an end-to-end machine-learning library for predicting '
    'thunderstorm hazards, primarily tornadoes and damaging straight-line wind.'
)

CLASSIFIERS = [
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3'
]

# You also need to install the following packages, which are not available in
# pip.  They can both be installed by "git clone" and "python setup.py install",
# the normal way one installs a GitHub package.
#
# https://github.com/matplotlib/basemap
# https://github.com/sharppy/SHARPpy
# https://github.com/tkrajina/srtm.py

PACKAGE_REQUIREMENTS = [
    'numpy',
    'scipy',
    'roipoly',
    'tensorflow',
    'keras',
    'scikit-learn',
    'scikit-image',
    'netCDF4',
    'pyproj==3.0.0',
    'opencv-python',
    'matplotlib',
    'pandas',
    'shapely',
    'descartes',
    'geopy',
    'metpy',
    'python-srtm'
]

if __name__ == '__main__':
    setup(name='GewitterGefahr',
          version='0.1',
          description=SHORT_DESCRIPTION,
          long_description=LONG_DESCRIPTION,
          license='MIT',
          author='Ryan Lagerquist',
          author_email='ryan.lagerquist@ou.edu',
          url='https://github.com/thunderhoser/GewitterGefahr',
          packages=PACKAGE_NAMES,
          scripts=[],
          keywords=KEYWORDS,
          classifiers=CLASSIFIERS,
          include_package_data=True,
          zip_safe=False,
          install_requires=PACKAGE_REQUIREMENTS)
