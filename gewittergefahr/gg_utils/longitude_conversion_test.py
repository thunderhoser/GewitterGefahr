"""Unit tests for longitude_conversion.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion

TOLERANCE = 1e-6

LONGITUDES_POSITIVE_IN_WEST_DEG = numpy.array([
    [numpy.nan, 45., 90.],
    [45., 90., 135.],
    [90., 135., 180.],
    [135., 180., 225.],
    [180., 225., 270.],
    [225., 270., 315.],
    [270., 315., 0.],
    [315., 0., numpy.nan],
    [0., numpy.nan, numpy.nan]])

LONGITUDES_NEGATIVE_IN_WEST_DEG = numpy.array([
    [numpy.nan, 45., 90.],
    [45., 90., 135.],
    [90., 135., 180.],
    [135., 180., -135.],
    [180., -135., -90.],
    [-135., -90., -45.],
    [-90., -45., 0.],
    [-45., 0., numpy.nan],
    [0., numpy.nan, numpy.nan]])

LONGITUDES_MIXED_SIGNS_DEG = numpy.array([
    [numpy.nan, 45., 90.],
    [45., 90., 135.],
    [90., 135., 180.],
    [135., 180., 225.],
    [180., -135., 270.],
    [-135., -90., 315.],
    [270., -45., 0.],
    [315., 0., numpy.nan],
    [0., numpy.nan, numpy.nan]])


class LongitudeConversionTests(unittest.TestCase):
    """Each method is a unit test for longitude_conversion.py."""

    def test_convert_lng_negative_in_west_inputs_mixed(self):
        """Ensures correct output from convert_lng_negative_in_west.

        In this case, inputs have mixed signs.
        """

        these_longitudes_deg = lng_conversion.convert_lng_negative_in_west(
            LONGITUDES_MIXED_SIGNS_DEG)
        self.assertTrue(
            numpy.allclose(these_longitudes_deg,
                           LONGITUDES_NEGATIVE_IN_WEST_DEG, atol=TOLERANCE,
                           equal_nan=True))

    def test_convert_lng_negative_in_west_inputs_negative(self):
        """Ensures correct output from convert_lng_negative_in_west.

        In this case, inputs are already negative in western hemisphere.
        """

        these_longitudes_deg = lng_conversion.convert_lng_negative_in_west(
            LONGITUDES_NEGATIVE_IN_WEST_DEG)
        self.assertTrue(
            numpy.allclose(these_longitudes_deg,
                           LONGITUDES_NEGATIVE_IN_WEST_DEG, atol=TOLERANCE,
                           equal_nan=True))

    def test_convert_lng_negative_in_west_inputs_positive(self):
        """Ensures correct output from convert_lng_negative_in_west.

        In this case, inputs are positive in western hemisphere.
        """

        these_longitudes_deg = lng_conversion.convert_lng_negative_in_west(
            LONGITUDES_POSITIVE_IN_WEST_DEG)
        self.assertTrue(
            numpy.allclose(these_longitudes_deg,
                           LONGITUDES_NEGATIVE_IN_WEST_DEG, atol=TOLERANCE,
                           equal_nan=True))

    def test_convert_lng_positive_in_west_inputs_mixed(self):
        """Ensures correct output from convert_lng_positive_in_west.

        In this case, inputs have mixed signs.
        """

        these_longitudes_deg = lng_conversion.convert_lng_positive_in_west(
            LONGITUDES_MIXED_SIGNS_DEG)
        self.assertTrue(
            numpy.allclose(these_longitudes_deg,
                           LONGITUDES_POSITIVE_IN_WEST_DEG, atol=TOLERANCE,
                           equal_nan=True))

    def test_convert_lng_positive_in_west_inputs_negative(self):
        """Ensures correct output from convert_lng_positive_in_west.

        In this case, inputs are negative in western hemisphere.
        """

        these_longitudes_deg = lng_conversion.convert_lng_positive_in_west(
            LONGITUDES_NEGATIVE_IN_WEST_DEG)
        self.assertTrue(
            numpy.allclose(these_longitudes_deg,
                           LONGITUDES_POSITIVE_IN_WEST_DEG, atol=TOLERANCE,
                           equal_nan=True))

    def test_convert_lng_positive_in_west_inputs_positive(self):
        """Ensures correct output from convert_lng_positive_in_west.

        In this case, inputs are already positive in western hemisphere.
        """

        these_longitudes_deg = lng_conversion.convert_lng_positive_in_west(
            LONGITUDES_POSITIVE_IN_WEST_DEG)
        self.assertTrue(
            numpy.allclose(these_longitudes_deg,
                           LONGITUDES_POSITIVE_IN_WEST_DEG, atol=TOLERANCE,
                           equal_nan=True))


if __name__ == '__main__':
    unittest.main()
