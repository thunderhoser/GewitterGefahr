"""Unit tests for cnn.py."""

import unittest
import numpy
from gewittergefahr.deep_learning import cnn

TOLERANCE = 1e-6

BINARY_PROBABILITIES_1D = numpy.array(
    [1, 0.8, 0.9, 0.7, 0.3, 1, 0.8, 0.7, 0.4, 0.7])

BINARY_PROBABILITIES_2D = numpy.array([[0, 1],
                                       [0.2, 0.8],
                                       [0.1, 0.9],
                                       [0.3, 0.7],
                                       [0.7, 0.3],
                                       [0, 1],
                                       [0.2, 0.8],
                                       [0.3, 0.7],
                                       [0.6, 0.4],
                                       [0.3, 0.7]])


class CnnTests(unittest.TestCase):
    """Each method is a unit test for cnn.py."""

    def test_binary_probabilities_to_matrix_1d(self):
        """Ensures correct output from _binary_probabilities_to_matrix.

        In this case the input array is 1-D.
        """

        these_probabilities = cnn._binary_probabilities_to_matrix(
            BINARY_PROBABILITIES_1D + 0.)

        self.assertTrue(numpy.allclose(
            these_probabilities, BINARY_PROBABILITIES_2D, atol=TOLERANCE
        ))

    def test_binary_probabilities_to_matrix_2d(self):
        """Ensures correct output from _binary_probabilities_to_matrix.

        In this case the input array is 2-D.
        """

        these_probabilities = cnn._binary_probabilities_to_matrix(
            BINARY_PROBABILITIES_2D + 0.)

        self.assertTrue(numpy.allclose(
            these_probabilities, BINARY_PROBABILITIES_2D, atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
