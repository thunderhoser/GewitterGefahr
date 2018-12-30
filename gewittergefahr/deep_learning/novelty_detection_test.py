"""Unit tests for novelty_detection.py."""

import unittest
import numpy
from gewittergefahr.deep_learning import novelty_detection

TOLERANCE = 1e-6

# The following constants are used to test _normalize_features.
FEATURE_MATRIX_DENORM = numpy.array([[1, 2, 3, 6, 5],
                                     [5, 4, 2, 2, 1],
                                     [3, 4, 0, 4, -1],
                                     [-1, -2, -1, 0, -1]], dtype=float)

FEATURE_MEANS = numpy.array([2, 2, 1, 3, 1], dtype=float)
FEATURE_STANDARD_DEVIATIONS = numpy.array([20, 24, 10, 20, 24], dtype=float)
FEATURE_STANDARD_DEVIATIONS = numpy.sqrt(FEATURE_STANDARD_DEVIATIONS / 3)

FEATURE_MATRIX_NORM = FEATURE_MATRIX_DENORM + 0.

for k in range(len(FEATURE_MEANS)):
    FEATURE_MATRIX_NORM[:, k] = (
        (FEATURE_MATRIX_DENORM[:, k] - FEATURE_MEANS[k]) /
        FEATURE_STANDARD_DEVIATIONS[k]
    )

# The following constants are used to test _fit_svd and _apply_svd.
BASELINE_FEATURE_MATRIX = numpy.array([[0, -1, 0, -0],
                                       [-2, 4, -0, 3],
                                       [-1, 1, 2, -0],
                                       [-3, -1, -3, -1],
                                       [-0, -2, 1, -2],
                                       [2, -0, 3, -0]], dtype=float)

TEST_FEATURE_MATRIX = numpy.array([[2, -2, 0, 5],
                                   [-4, 1, -4, 1],
                                   [-1, -2, 3, -1],
                                   [8, 2, 2, -2],
                                   [3, 4, 3, -5],
                                   [6, 5, -2, 3]], dtype=float)

NUM_MODES_TO_KEEP = TEST_FEATURE_MATRIX.shape[1]


class NoveltyDetectionTests(unittest.TestCase):
    """Each method is a unit test for novelty_detection.py."""

    def test_normalize_features(self):
        """Ensures correct output from _normalize_features."""

        this_matrix, these_means, these_standard_deviations = (
            novelty_detection._normalize_features(
                feature_matrix=FEATURE_MATRIX_DENORM + 0.)
        )

        self.assertTrue(numpy.allclose(
            this_matrix, FEATURE_MATRIX_NORM, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_means, FEATURE_MEANS, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_standard_deviations, FEATURE_STANDARD_DEVIATIONS,
            atol=TOLERANCE
        ))

    def test_fit_and_apply_svd(self):
        """Ensures correct output from _fit_svd and _apply_svd."""

        this_svd_dictionary = novelty_detection._fit_svd(
            baseline_feature_matrix=BASELINE_FEATURE_MATRIX + 0.,
            test_feature_matrix=TEST_FEATURE_MATRIX + 0.,
            percent_variance_to_keep=100.)

        this_test_feature_matrix = numpy.full(
            TEST_FEATURE_MATRIX.shape, numpy.nan)
        num_test_examples = TEST_FEATURE_MATRIX.shape[0]

        for i in range(num_test_examples):
            this_test_feature_matrix[i, :] = novelty_detection._apply_svd(
                feature_vector=TEST_FEATURE_MATRIX[i, ...],
                svd_dictionary=this_svd_dictionary)

        self.assertTrue(numpy.allclose(
            this_test_feature_matrix, TEST_FEATURE_MATRIX, atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
