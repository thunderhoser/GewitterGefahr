"""Unit tests for find_climo_averages_of_gridrad_training.py."""

import unittest
import numpy
from gewittergefahr.scripts import \
    find_climo_averages_of_gridrad_training as find_climo_avgs

TOLERANCE = 1e-6

PRELIM_AVERAGES = numpy.array([15, 10], dtype=float)
PRELIM_NUM_CASES = numpy.array([200, 20], dtype=int)
FINAL_AVERAGE = 3200. / 220


class FindClimoAveragesTests(unittest.TestCase):
    """Unit tests for find_climo_averages_of_gridrad_training.py."""

    def test_get_weighted_average(self):
        """Ensures correct output from _get_weighted_average."""

        this_average = find_climo_avgs._get_weighted_average(
            input_values=PRELIM_AVERAGES, input_weights=PRELIM_NUM_CASES)
        self.assertTrue(numpy.isclose(
            this_average, FINAL_AVERAGE, atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
