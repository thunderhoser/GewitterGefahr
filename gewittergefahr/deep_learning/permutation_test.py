"""Unit tests for permutation.py."""

import unittest
import numpy
from sklearn.metrics import log_loss as sklearn_cross_entropy
from gewittergefahr.deep_learning import permutation

XENTROPY_TOLERANCE = 1e-4

# The following constants are used to test cross_entropy_function.
BINARY_TARGET_VALUES = numpy.array([0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0], dtype=int)
BINARY_PROBABILITY_MATRIX = numpy.array([[0, 1],
                                         [1, 0],
                                         [0, 1],
                                         [1, 0],
                                         [0.5, 0.5],
                                         [0.5, 0.5],
                                         [0, 1],
                                         [1, 0],
                                         [0, 1],
                                         [1, 0],
                                         [0.5, 0.5]])

BINARY_CROSS_ENTROPY = sklearn_cross_entropy(
    BINARY_TARGET_VALUES, BINARY_PROBABILITY_MATRIX[:, 1])

TERNARY_TARGET_VALUES = numpy.array(
    [2, 0, 1, 1, 2, 1, 2, 0, 2, 0, 0], dtype=int)

TERNARY_PROBABILITY_MATRIX = numpy.array([[0, 0, 1],
                                          [1, 0, 0],
                                          [0, 1, 0],
                                          [1. / 3, 1. / 3, 1. / 3],
                                          [0.5, 0.25, 0.25],
                                          [0.25, 0.5, 0.25],
                                          [1. / 3, 0, 2. / 3],
                                          [1, 0, 0],
                                          [0, 0.5, 0.5],
                                          [0.75, 0, 0.25],
                                          [0.6, 0.2, 0.2]])

TERNARY_CROSS_ENTROPY = sklearn_cross_entropy(
    TERNARY_TARGET_VALUES, TERNARY_PROBABILITY_MATRIX)


class PermutationTests(unittest.TestCase):
    """Each method is a unit test for permutation.py."""

    def test_cross_entropy_function_binary(self):
        """Ensures correct output from cross_entropy_function.

        In this case there are 2 target classes (binary).
        """

        this_cross_entropy = permutation.cross_entropy_function(
            target_values=BINARY_TARGET_VALUES,
            class_probability_matrix=BINARY_PROBABILITY_MATRIX, test_mode=True)

        self.assertTrue(numpy.isclose(
            this_cross_entropy, BINARY_CROSS_ENTROPY, atol=XENTROPY_TOLERANCE))

    def test_cross_entropy_function_ternary(self):
        """Ensures correct output from cross_entropy_function.

        In this case there are 3 target classes (ternary).
        """

        this_cross_entropy = permutation.cross_entropy_function(
            target_values=TERNARY_TARGET_VALUES,
            class_probability_matrix=TERNARY_PROBABILITY_MATRIX, test_mode=True)

        self.assertTrue(numpy.isclose(
            this_cross_entropy, TERNARY_CROSS_ENTROPY, atol=XENTROPY_TOLERANCE))


if __name__ == '__main__':
    unittest.main()
