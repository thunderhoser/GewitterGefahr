"""Unit tests for saliency_plotting.py."""

import unittest
import numpy
import matplotlib.pyplot as pyplot
from gewittergefahr.plotting import saliency_plotting

TOLERANCE = 1e-6

SALIENCY_MATRIX = numpy.array([
    [-0.4, -0.9, -1.8, 2.7],
    [-0.6, 1.4, 2.4, 1.7],
    [-1.5, -0.7, 0.2, -1.5]
])

MAX_ABSOLUTE_COLOUR_VALUE = 2.
MIN_FONT_SIZE = 8.
MAX_FONT_SIZE = 20.

FONT_SIZE_MATRIX = numpy.array([
    [10.4, 13.4, 18.8, 20],
    [11.6, 16.4, 20, 18.2],
    [17, 12.2, 9.2, 17]
])


class SaliencyPlottingTests(unittest.TestCase):
    """Each method is a unit test for saliency_plotting.py."""

    def test_saliency_to_colour_and_size(self):
        """Ensures correct output from _saliency_to_colour_and_size."""

        this_rgb_matrix, this_font_size_matrix = (
            saliency_plotting._saliency_to_colour_and_size(
                saliency_matrix=SALIENCY_MATRIX,
                colour_map_object=pyplot.cm.gist_yarg,
                max_absolute_colour_value=MAX_ABSOLUTE_COLOUR_VALUE,
                min_font_size=MIN_FONT_SIZE,
                max_font_size=MAX_FONT_SIZE)
        )

        expected_dimensions = SALIENCY_MATRIX.shape + (3,)
        self.assertTrue(expected_dimensions == this_rgb_matrix.shape)

        self.assertTrue(numpy.allclose(
            this_font_size_matrix, FONT_SIZE_MATRIX, atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
