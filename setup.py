"""Setup file for GewitterGefahr."""

from setuptools import setup

if __name__ == '__main__':
    short_description = (
        'Object-based machine learning to predict thunderstorm hazards')
    long_description = (
        'GewitterGefahr is a Python package for object-based machine learning '
        'to predict thunderstorm hazards, primarily straight-line wind.')

    classifiers = ['Development Status :: 2 - Pre-Alpha',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: MIT License',
                   'Programming Language :: Python :: 2.7']

    setup(name='GewitterGefahr',
          version='0.1',
          description=short_description,
          author='Ryan Lagerquist',
          author_email='ryan.lagerquist@ou.edu',
          long_description=long_description,
          license='MIT',
          url='https://github.com/thunderhoser/GewitterGefahr',
          packages=['gewittergefahr', 'gewittergefahr.gg_io',
                    'gewittergefahr.gg_utils'],
          scripts=[],
          keywords=['machine learning', 'artificial intelligence',
                    'data mining', 'weather', 'meteorology', 'thunderstorm',
                    'straight-line wind'],
          classifiers=classifiers,
          include_package_data=True,
          zip_safe=False)
