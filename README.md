# GewitterGefahr

GewitterGefahr is an end-to-end machine-learning library for predicting thunderstorm hazards, primarily tornadoes and damaging straight-line wind.  The machine-learning methods are storm-centered, which means that each case is one storm object (one storm cell at one time step).  "End-to-end" means that this library includes code for data acquisition and pre-processing; training, validation, and testing of machine-learning models; and post-processing of machine-learning output.

External documentation is still a work in progress (this README is currently the only external documentation).  I plan to add external documentation in the coming weeks, and I hope to have it finished by early **November 2018**.  However, keep in mind that this library changes a lot (I use it for most of my Ph.D. work), so at any given time some portion of it will be probably be undocumented.  I will try to be clear about what that portion is.

Despite the lack of external documentation, there are three types of internal documentation.  First, there is a docstring at the top of each method, explaining the inputs and outputs along with their formats (*e.g.,* number, string, list, numpy array, etc.).  Second, the variable and method names are verbose and include units where applicable (*e.g.,* `DRY_AIR_GAS_CONSTANT_J_KG01_K01`, `specific_humidities_kg_kg01`), so the code is self-documenting to some extent.  Third, most modules (Python files) are accompanied by unit tests.  For example, the unit tests for moisture_conversions.py are in moisture_conversions_test.py.  With that said, the unit tests are not exhaustive and there are no integration tests, so I make no guarantee that the code is bug-free.

# Requirements

* numpy
* scipy
* tensorflow
* keras
* scikit-image
* netCDF4
* pyproj
* scikit-learn
* opencv
* matplotlib
* basemap
* pandas
* shapely
* ambhas
* descartes
* geopy
* metpy