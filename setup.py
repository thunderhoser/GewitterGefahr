"""Setup file for GewitterGefahr."""

from setuptools import setup

PACKAGE_NAMES = [
    'gewittergefahr', 'gewittergefahr.gg_io', 'gewittergefahr.gg_utils',
    'gewittergefahr.linkage', 'gewittergefahr.plotting',
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

if __name__ == '__main__':
    setup(name='GewitterGefahr', version='0.1', description=SHORT_DESCRIPTION,
          author='Ryan Lagerquist', author_email='ryan.lagerquist@ou.edu',
          long_description=LONG_DESCRIPTION, license='MIT',
          url='https://github.com/thunderhoser/GewitterGefahr',
          packages=PACKAGE_NAMES, scripts=[], keywords=KEYWORDS,
          classifiers=CLASSIFIERS, include_package_data=True, zip_safe=False)
