"""Setup file for GewitterGefahr."""

from setuptools import setup

PACKAGE_NAMES = [
    'gewittergefahr', 'gewittergefahr.gg_io', 'gewittergefahr.gg_utils',
    'gewittergefahr.deep_learning', 'gewittergefahr.plotting',
    'gewittergefahr.scripts', 'gewittergefahr.feature_selection_example'
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
    'Programming Language :: Python :: 2.7'
]

# These packages are probably best to install via pip.
PIP_REQUIREMENTS = [
    'numpy',
    'scipy',
    'tensorflow',
    'keras',
    'scikit-learn',
    'scikit-image',
    'netCDF4',
    'pyproj',
    'opencv-python',
    'matplotlib',
    'basemap',
    'pandas',
    'shapely',
]

# These packages are probably best to install from the GitHub repository.
GITHUB_REQUIREMENTS = [
    'ambhas',
    'descartes',
    'geopy',
    'metpy'
]

PACKAGE_REQUIREMENTS = PIP_REQUIREMENTS + GITHUB_REQUIREMENTS

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
