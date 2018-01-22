"""Unit tests for histograms.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import histograms

INPUT_VALUES = numpy.array(
    [-numpy.inf, -100., 0., 0.1, 0.2001, 0.3, 0.4001, 0.5, 0.6001, 0.7, 0.8001,
     0.9, 1., 5000., numpy.inf])
NUM_BINS = 5
MIN_VALUE_FOR_HISTOGRAM = 0.
MAX_VALUE_FOR_HISTOGRAM = 1.

BIN_INDICES = numpy.array([0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 4],
                          dtype=int)
NUM_EXAMPLES_BY_BIN = numpy.array([4, 2, 2, 2, 5], dtype=int)


class HistogramsTests(unittest.TestCase):
    """Each method is a unit test for histograms.py."""

    def test_create_histogram(self):
        """Ensures correct output from create_histogram."""

        these_bin_indices, this_num_examples_by_bin = (
            histograms.create_histogram(
                input_values=INPUT_VALUES, num_bins=NUM_BINS,
                min_value=MIN_VALUE_FOR_HISTOGRAM,
                max_value=MAX_VALUE_FOR_HISTOGRAM))

        self.assertTrue(numpy.array_equal(these_bin_indices, BIN_INDICES))
        self.assertTrue(numpy.array_equal(
            this_num_examples_by_bin, NUM_EXAMPLES_BY_BIN))


if __name__ == '__main__':
    unittest.main()
