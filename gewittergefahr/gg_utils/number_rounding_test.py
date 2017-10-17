"""Unit tests for number_rounding.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import number_rounding

ROUNDING_BASE = 0.05
TOLERANCE = 1e-6

INPUT_VALUE_SCALAR = 1.66
EXPECTED_NEAREST_SCALAR = 1.65
EXPECTED_FLOOR_OF_SCALAR = 1.65
EXPECTED_CEILING_OF_SCALAR = 1.7
EXPECTED_HALF_INTEGER_FOR_SCALAR = 1.5

INPUT_VALUES = numpy.array([-0.875, -0.72, -0.05, 0., 0.51, 0.69])
EXPECTED_NEAREST_VALUES = numpy.array([-0.9, -0.7, -0.05, 0., 0.5, 0.7])
EXPECTED_FLOOR_VALUES = numpy.array([-0.9, -0.75, -0.05, 0., 0.5, 0.65])
EXPECTED_CEILING_VALUES = numpy.array([-0.85, -0.7, -0.05, 0., 0.55, 0.7])
EXPECTED_HALF_INTEGERS = numpy.array([-0.5, -0.5, -0.5, -0.5, 0.5, 0.5])


class NumberRoundingTests(unittest.TestCase):
    """Each method is a unit test for number_rounding.py."""

    def test_round_to_nearest_scalar(self):
        """Ensures correct output from round_to_nearest with scalar input."""

        nearest_scalar = number_rounding.round_to_nearest(INPUT_VALUE_SCALAR,
                                                          ROUNDING_BASE)
        self.assertTrue(numpy.isclose(nearest_scalar, EXPECTED_NEAREST_SCALAR,
                                      atol=TOLERANCE))

    def test_floor_to_nearest_scalar(self):
        """Ensures correct output from floor_to_nearest with scalar input."""

        floor_of_scalar = number_rounding.floor_to_nearest(INPUT_VALUE_SCALAR,
                                                           ROUNDING_BASE)
        self.assertTrue(numpy.isclose(floor_of_scalar, EXPECTED_FLOOR_OF_SCALAR,
                                      atol=TOLERANCE))

    def test_ceiling_to_nearest_scalar(self):
        """Ensures correct output from ceiling_to_nearest with scalar input."""

        ceiling_of_scalar = number_rounding.ceiling_to_nearest(
            INPUT_VALUE_SCALAR, ROUNDING_BASE)
        self.assertTrue(
            numpy.isclose(ceiling_of_scalar, EXPECTED_CEILING_OF_SCALAR,
                          atol=TOLERANCE))

    def test_round_to_half_integer_scalar(self):
        """Ensures correct output from round_to_half_integer w/ scalar input."""

        half_integer_for_scalar = number_rounding.round_to_half_integer(
            INPUT_VALUE_SCALAR)
        self.assertTrue(numpy.isclose(
            half_integer_for_scalar, EXPECTED_HALF_INTEGER_FOR_SCALAR,
            atol=TOLERANCE))

    def test_round_to_nearest_array(self):
        """Ensures correct output from round_to_nearest with array input."""

        nearest_values = number_rounding.round_to_nearest(INPUT_VALUES,
                                                          ROUNDING_BASE)
        self.assertTrue(numpy.allclose(nearest_values, EXPECTED_NEAREST_VALUES,
                                       atol=TOLERANCE))

    def test_floor_to_nearest_array(self):
        """Ensures correct output from floor_to_nearest with array input."""

        floor_values = number_rounding.floor_to_nearest(INPUT_VALUES,
                                                        ROUNDING_BASE)
        self.assertTrue(numpy.allclose(floor_values, EXPECTED_FLOOR_VALUES,
                                       atol=TOLERANCE))

    def test_ceiling_to_nearest_array(self):
        """Ensures correct output from ceiling_to_nearest with array input."""

        ceiling_values = number_rounding.ceiling_to_nearest(INPUT_VALUES,
                                                            ROUNDING_BASE)
        self.assertTrue(numpy.allclose(ceiling_values, EXPECTED_CEILING_VALUES,
                                       atol=TOLERANCE))

    def test_round_to_half_integer_array(self):
        """Ensures correct output from round_to_half_integer w/ array input."""

        half_integers = number_rounding.round_to_half_integer(INPUT_VALUES)
        self.assertTrue(numpy.allclose(
            half_integers, EXPECTED_HALF_INTEGERS, atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
