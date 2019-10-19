"""Unit tests for correlation.py."""

import unittest
import numpy
from gewittergefahr.deep_learning import correlation

TOLERANCE = 1e-6

# The following constants are used to test _linear_idx_to_matrix_channel_idxs.
NUM_PREDICTORS_BY_MATRIX = numpy.array([4, 2, 5], dtype=int)
LINEAR_INDICES = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=int)
MATRIX_INDICES = numpy.array([0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2], dtype=int)
CHANNEL_INDICES = numpy.array([0, 1, 2, 3, 0, 1, 0, 1, 2, 3, 4], dtype=int)

# The following constants are used to test _take_spatial_mean.
SPATIAL_DATA_MATRIX = numpy.array([
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11]
], dtype=float)

SPATIAL_MEANS = numpy.array([1.5, 5.5, 9.5])


class CorrelationTests(unittest.TestCase):
    """Each method is a unit test for correlation.py."""

    def test_linear_idx_to_matrix_channel_idxs(self):
        """Ensures correct output from _linear_idx_to_matrix_channel_idxs."""

        for (i, j, k) in zip(MATRIX_INDICES, CHANNEL_INDICES, LINEAR_INDICES):
            this_i, this_j = correlation._linear_idx_to_matrix_channel_idxs(
                linear_index=k,
                num_predictors_by_matrix=NUM_PREDICTORS_BY_MATRIX)

            self.assertTrue(this_i == i)
            self.assertTrue(this_j == j)

    def test_take_spatial_mean(self):
        """Ensures correct output from _take_spatial_mean."""

        these_means = correlation._take_spatial_mean(SPATIAL_DATA_MATRIX)

        self.assertTrue(numpy.allclose(
            these_means, SPATIAL_MEANS, atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
