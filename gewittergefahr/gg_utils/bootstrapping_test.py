"""Unit tests for bootstrapping.py."""

import copy
import unittest
import numpy
from gewittergefahr.gg_utils import bootstrapping
from gewittergefahr.gg_utils import error_checking

TOLERANCE = 1e-6

INPUT_VECTOR = numpy.array(
    [10., 30., 50., 70., 75., 70., 75., 60., 0., numpy.nan])
NUM_EXAMPLES_SMALL = 5
NUM_EXAMPLES_LARGE = len(INPUT_VECTOR)

STAT_VALUES = copy.deepcopy(INPUT_VECTOR)
CONFIDENCE_LEVEL = 0.95
CONFIDENCE_INTERVAL_MIN = 2.
CONFIDENCE_INTERVAL_MAX = 75.


class BootstrappingTests(unittest.TestCase):
    """Each method is a unit test for bootstrapping.py."""

    def test_draw_sample_small(self):
        """Ensures correct output from draw_sample.

        In this case, number of examples to draw < number of original examples.
        """

        this_sample_vector, these_sample_indices = bootstrapping.draw_sample(
            INPUT_VECTOR, NUM_EXAMPLES_SMALL)
        self.assertTrue(len(this_sample_vector) == NUM_EXAMPLES_SMALL)
        self.assertTrue(len(these_sample_indices) == NUM_EXAMPLES_SMALL)
        error_checking.assert_is_integer_numpy_array(these_sample_indices)

    def test_draw_sample_large(self):
        """Ensures correct output from draw_sample.

        In this case, number of examples to draw = number of original examples.
        """

        this_sample_vector, these_sample_indices = bootstrapping.draw_sample(
            INPUT_VECTOR, NUM_EXAMPLES_LARGE)
        self.assertTrue(len(this_sample_vector) == NUM_EXAMPLES_LARGE)
        self.assertTrue(len(these_sample_indices) == NUM_EXAMPLES_LARGE)
        error_checking.assert_is_integer_numpy_array(these_sample_indices)

    def test_get_confidence_interval(self):
        """Ensures correct output from get_confidence_interval."""

        this_conf_interval_min, this_conf_interval_max = (
            bootstrapping.get_confidence_interval(
                STAT_VALUES, CONFIDENCE_LEVEL))

        self.assertTrue(numpy.isclose(
            this_conf_interval_min, CONFIDENCE_INTERVAL_MIN, atol=TOLERANCE))
        self.assertTrue(numpy.isclose(
            this_conf_interval_max, CONFIDENCE_INTERVAL_MAX, atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
