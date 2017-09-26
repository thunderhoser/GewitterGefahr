"""Unit tests for grib_io.py."""

import copy
import unittest
import numpy
from gewittergefahr.gg_io import grib_io

TOLERANCE = 1e-6
SENTINEL_VALUE = 9.999e20

DATA_MATRIX_WITH_SENTINELS = (
    numpy.array([[999.25, 1000., 1001., 1001.5, SENTINEL_VALUE],
                 [999.5, 1000.75, 1002., 1002.2, SENTINEL_VALUE],
                 [999.9, 1001.1, 1001.7, 1002.3, 1003.],
                 [SENTINEL_VALUE, 1001.5, 1001.6, 1002.1, 1002.5],
                 [SENTINEL_VALUE, SENTINEL_VALUE, 1002., 1002., 1003.3]]))

EXPECTED_DATA_MATRIX_NO_SENTINELS = (
    numpy.array([[999.25, 1000., 1001., 1001.5, numpy.nan],
                 [999.5, 1000.75, 1002., 1002.2, numpy.nan],
                 [999.9, 1001.1, 1001.7, 1002.3, 1003.],
                 [numpy.nan, 1001.5, 1001.6, 1002.1, 1002.5],
                 [numpy.nan, numpy.nan, 1002., 1002., 1003.3]]))


class GribIoTests(unittest.TestCase):
    """Each method is a unit test for grib_io.py."""

    def test_replace_sentinels_with_nan_sentinel_value_defined(self):
        """Ensures correct output from _replace_sentinels_with_nan.

        In this case the sentinel value is defined (not None).
        """

        this_data_matrix = grib_io._replace_sentinels_with_nan(
            copy.deepcopy(DATA_MATRIX_WITH_SENTINELS), SENTINEL_VALUE)

        self.assertTrue(
            numpy.allclose(this_data_matrix, EXPECTED_DATA_MATRIX_NO_SENTINELS,
                           atol=TOLERANCE, equal_nan=True))

    def test_replace_sentinels_with_nan_sentinel_value_none(self):
        """Ensures correct output from _replace_sentinels_with_nan.

        In this case the sentinel value is None.
        """

        this_data_matrix = grib_io._replace_sentinels_with_nan(
            copy.deepcopy(DATA_MATRIX_WITH_SENTINELS), None)

        self.assertTrue(
            numpy.allclose(this_data_matrix, DATA_MATRIX_WITH_SENTINELS,
                           atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
