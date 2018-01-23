"""Setup file for GewitterGefahr."""

from setuptools import setup

PACKAGE_NAMES = [
    'gewittergefahr', 'gewittergefahr.gg_io', 'gewittergefahr.gg_utils',
    'gewittergefahr.plotting', 'gewittergefahr.scripts',
    'gewittergefahr.feature_selection_example']
KEYWORDS = [
    'machine learning', 'artificial intelligence', 'data mining', 'weather',
    'meteorology', 'thunderstorm', 'straight-line wind']
SHORT_DESCRIPTION = (
    'Object-based machine learning to predict thunderstorm hazards')
LONG_DESCRIPTION = (
    'GewitterGefahr is a Python package for object-based machine learning '
    'to predict thunderstorm hazards, primarily straight-line wind.')

CLASSIFIERS = [
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 2.7']

PACKAGE_REQUIREMENTS = [
    'descartes', 'geopy', 'netCDF4', 'pyproj', 'scipy', 'sharppy', 'skewt',
    'scikit-learn', 'matplotlib', 'numpy', 'pandas', 'shapely', 'scikit-image']

# PACKAGE_REQUIREMENTS = [
#     'descartes', 'geopy==1.11.2', 'netCDF4==1.2.4', 'pyproj==1.9.5.1',
#     'scipy==0.19.0', 'sharppy==1.3.0', 'skewt==0.1.4r2',
#     'scikit-learn==0.18.1', 'opencv==3.1.0', 'matplotlib==2.0.2',
#     'numpy==1.11.3', 'pandas==0.21.0', 'shapely==1.5.16',
#     'scikit-image==0.13.0']

if __name__ == '__main__':
    setup(name='GewitterGefahr', version='0.1', description=SHORT_DESCRIPTION,
          author='Ryan Lagerquist', author_email='ryan.lagerquist@ou.edu',
          long_description=LONG_DESCRIPTION, license='MIT',
          url='https://github.com/thunderhoser/GewitterGefahr',
          packages=PACKAGE_NAMES, scripts=[], keywords=KEYWORDS,
          classifiers=CLASSIFIERS, include_package_data=True, zip_safe=False,
          install_requires=PACKAGE_REQUIREMENTS)
