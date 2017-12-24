"""Unit tests for dilation.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import dilation

TOLERANCE = 1e-6
SMALL_PERCENTILE = 12.5
LARGE_PERCENTILE = 87.5
DILATION_HALF_WIDTH_IN_PIXELS = 1

INPUT_MATRIX = numpy.array(
    [[-20., -15., -10., -5., 0.],
     [-10., -5., 0., 5., 10.],
     [0., 5., 10., numpy.nan, numpy.nan],
     [10., 15., 20., numpy.nan, numpy.nan]])

OUTPUT_MATRIX_SMALL_PERCENTILE = numpy.array(
    [[-15., -15., -10., -5., numpy.nan],
     [-15., -15., -10., -5., numpy.nan],
     [-5., -5., numpy.nan, numpy.nan, numpy.nan],
     [numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan]])

OUTPUT_MATRIX_LARGE_PERCENTILE = numpy.array(
    [[numpy.nan, numpy.nan, numpy.nan, 5., 5.],
     [numpy.nan, 5., 5., 10., 5.],
     [10., 15., 15., 10., 5.],
     [10., 15., 15., 10., numpy.nan]])

OUTPUT_MATRIX_LARGEST_ABS_VALUE = numpy.array(
    [[-15., -15., -10., 5., 5.],
     [-15., -15., -10., 10., 5.],
     [10., 15., 15., 10., 5.],
     [10., 15., 15., 10., numpy.nan]])


class DilationTests(unittest.TestCase):
    """Each method is a unit test for dilation.py."""

    def test_dilate_2d_matrix_small_percentile(self):
        """Ensures correct output from dilate_2d_matrix with small prctile."""

        this_output_matrix = dilation.dilate_2d_matrix(
            INPUT_MATRIX, percentile_level=SMALL_PERCENTILE,
            half_width_in_pixels=DILATION_HALF_WIDTH_IN_PIXELS)

        self.assertTrue(numpy.allclose(
            this_output_matrix, OUTPUT_MATRIX_SMALL_PERCENTILE, atol=TOLERANCE,
            equal_nan=True))

    def test_dilate_2d_matrix_large_percentile(self):
        """Ensures correct output from dilate_2d_matrix with large prctile."""

        this_output_matrix = dilation.dilate_2d_matrix(
            INPUT_MATRIX, percentile_level=LARGE_PERCENTILE,
            half_width_in_pixels=DILATION_HALF_WIDTH_IN_PIXELS)

        self.assertTrue(numpy.allclose(
            this_output_matrix, OUTPUT_MATRIX_LARGE_PERCENTILE, atol=TOLERANCE,
            equal_nan=True))

    def test_dilate_2d_matrix_take_largest_abs_value(self):
        """Ensures correct output from dilate_2d_matrix.

        In this case, take_largest_absolute_value = True.
        """

        this_output_matrix = dilation.dilate_2d_matrix(
            INPUT_MATRIX, percentile_level=LARGE_PERCENTILE,
            half_width_in_pixels=DILATION_HALF_WIDTH_IN_PIXELS,
            take_largest_absolute_value=True)

        self.assertTrue(numpy.allclose(
            this_output_matrix, OUTPUT_MATRIX_LARGEST_ABS_VALUE, atol=TOLERANCE,
            equal_nan=True))


if __name__ == '__main__':
    unittest.main()
