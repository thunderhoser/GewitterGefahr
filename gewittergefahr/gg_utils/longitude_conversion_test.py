"""Unit tests for longitude_conversion.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion

TOLERANCE = 1e-6

LONGITUDES_MIXED_SIGNS_DEG = numpy.array(
    [-160., 240., -80., 320., 0., 40., 80., 120., 160., 180.])
LONGITUDES_POSITIVE_IN_WEST_DEG = numpy.array(
    [200., 240., 280., 320., 0., 40., 80., 120., 160., 180.])
LONGITUDES_NEGATIVE_IN_WEST_DEG = numpy.array(
    [-160., -120., -80., -40., 0., 40., 80., 120., 160., 180.])


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
                           LONGITUDES_NEGATIVE_IN_WEST_DEG, atol=TOLERANCE))

    def test_convert_lng_negative_in_west_inputs_negative(self):
        """Ensures correct output from convert_lng_negative_in_west.

        In this case, inputs are already negative in western hemisphere.
        """

        these_longitudes_deg = lng_conversion.convert_lng_negative_in_west(
            LONGITUDES_NEGATIVE_IN_WEST_DEG)
        self.assertTrue(
            numpy.allclose(these_longitudes_deg,
                           LONGITUDES_NEGATIVE_IN_WEST_DEG, atol=TOLERANCE))

    def test_convert_lng_negative_in_west_inputs_positive(self):
        """Ensures correct output from convert_lng_negative_in_west.

        In this case, inputs are positive in western hemisphere.
        """

        these_longitudes_deg = lng_conversion.convert_lng_negative_in_west(
            LONGITUDES_POSITIVE_IN_WEST_DEG)
        self.assertTrue(
            numpy.allclose(these_longitudes_deg,
                           LONGITUDES_NEGATIVE_IN_WEST_DEG, atol=TOLERANCE))

    def test_convert_lng_positive_in_west_inputs_mixed(self):
        """Ensures correct output from convert_lng_positive_in_west.

        In this case, inputs have mixed signs.
        """

        these_longitudes_deg = lng_conversion.convert_lng_positive_in_west(
            LONGITUDES_MIXED_SIGNS_DEG)
        self.assertTrue(
            numpy.allclose(these_longitudes_deg,
                           LONGITUDES_POSITIVE_IN_WEST_DEG, atol=TOLERANCE))

    def test_convert_lng_positive_in_west_inputs_negative(self):
        """Ensures correct output from convert_lng_positive_in_west.

        In this case, inputs are negative in western hemisphere.
        """

        these_longitudes_deg = lng_conversion.convert_lng_positive_in_west(
            LONGITUDES_NEGATIVE_IN_WEST_DEG)
        self.assertTrue(
            numpy.allclose(these_longitudes_deg,
                           LONGITUDES_POSITIVE_IN_WEST_DEG, atol=TOLERANCE))

    def test_convert_lng_positive_in_west_inputs_positive(self):
        """Ensures correct output from convert_lng_positive_in_west.

        In this case, inputs are already positive in western hemisphere.
        """

        these_longitudes_deg = lng_conversion.convert_lng_positive_in_west(
            LONGITUDES_POSITIVE_IN_WEST_DEG)
        self.assertTrue(
            numpy.allclose(these_longitudes_deg,
                           LONGITUDES_POSITIVE_IN_WEST_DEG, atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
