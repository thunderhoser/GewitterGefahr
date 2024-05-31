"""Unit tests for upconvnet.py."""

import unittest
import numpy
from gewittergefahr.deep_learning import upconvnet

TOLERANCE = 1e-6

# The following constants are used to test create_smoothing_filter.
SMOOTHING_RADIUS_PX = 1
NUM_HALF_FILTER_ROWS = 2
NUM_HALF_FILTER_COLUMNS = 2
NUM_CHANNELS = 10

# Expected weight matrix taken from: https://stackoverflow.com/a/35832345
SMALL_WEIGHT_MATRIX = numpy.array([
    [0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902],
    [0.01330621, 0.0596343, 0.09832033, 0.0596343, 0.01330621],
    [0.02193823, 0.09832033, 0.16210282, 0.09832033, 0.02193823],
    [0.01330621, 0.0596343, 0.09832033, 0.0596343, 0.01330621],
    [0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902]
])

WEIGHT_MATRIX = numpy.zeros((
    2 * NUM_HALF_FILTER_ROWS + 1, 2 * NUM_HALF_FILTER_COLUMNS + 1,
    NUM_CHANNELS, NUM_CHANNELS
))

numpy.repeat()

for k in range(NUM_CHANNELS):
    WEIGHT_MATRIX[..., k, k] = SMALL_WEIGHT_MATRIX


class UpconvnetTests(unittest.TestCase):
    """Each method is a unit test for upconvnet.py."""

    def test_create_smoothing_filter(self):
        """Ensures correct output from create_smoothing_filter."""

        this_weight_matrix = upconvnet.create_smoothing_filter(
            smoothing_radius_px=SMOOTHING_RADIUS_PX,
            num_half_filter_rows=NUM_HALF_FILTER_ROWS,
            num_half_filter_columns=NUM_HALF_FILTER_COLUMNS,
            num_channels=NUM_CHANNELS)

        self.assertTrue(numpy.allclose(
            this_weight_matrix, WEIGHT_MATRIX, atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
