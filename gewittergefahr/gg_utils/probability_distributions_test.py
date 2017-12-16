"""Unit tests for probability_distributions.py."""

import copy
import unittest
import numpy
from gewittergefahr.gg_utils import probability_distributions as prob_dist
from gewittergefahr.gg_utils import number_rounding as rounder

TOLERANCE = 1e-6

# The following constants are used to test _transform_each_marginal_to_uniform
# and _transform_each_marginal_to_normal.
FEATURE_MATRIX = numpy.array(
    [[1., 0., 10.],
     [2., 0., 10.],
     [numpy.nan, 1., 9.],
     [4., numpy.nan, 8.],
     [5., numpy.nan, 3.]])

FEATURE_MATRIX_UNIFORM_MARGINALS = numpy.array(
    [[1. / 5, 1.5 / 4, 4.5 / 6],
     [2. / 5, 1.5 / 4, 4.5 / 6],
     [numpy.nan, 3. / 4, 3. / 6],
     [3. / 5, numpy.nan, 2. / 6],
     [4. / 5, numpy.nan, 1. / 6]])

# The following constants are used to test _normalize_class_probabilities.
ORIGINAL_PROB_MATRIX = numpy.array(
    [[2., 3.],
     [18., 16.],
     [10., 10.],
     [6., 0.],
     [11., 20.],
     [30., numpy.nan]])

NORMALIZED_PROB_MATRIX = numpy.array(
    [[0.1 / 0.25, 0.15 / 0.25],
     [0.9 / 1.7, 0.8 / 1.7],
     [0.5, 0.5],
     [1., 0.],
     [0.55 / 1.55, 1. / 1.55],
     [numpy.nan, numpy.nan]])

# The following constants are used to test _get_feature_means and
# _get_covariance_matrix.
FEATURE_MEANS = numpy.array([3., 0.333333, 8.])
DIAG_COVARIANCE_MATRIX = numpy.array(
    [[10., 0., 0.], [0., 1., 0.], [0., 0., 25.5]]) / 3
NON_DIAG_COVARIANCE_MATRIX = numpy.array(
    [[10., 3., -16.], [3., 1., -1.], [-16., -1., 25.5]]) / 3

# The following constants are used to test fit_multivariate_normal and
# fit_mvn_for_each_class.
FEATURE_MATRIX_TOO_MANY_NAN = numpy.array(
    [[1., 0., 10.],
     [2., numpy.nan, 10.],
     [numpy.nan, numpy.nan, 9.],
     [4., numpy.nan, 8.],
     [5., numpy.nan, 3.]])

BINARY_CLASS_LABELS = numpy.array([0, 1, 0, 0, 1], dtype=int)


class ProbabilityDistributionsTests(unittest.TestCase):
    """Each method is a unit test for probability_distributions.py."""

    def test_transform_each_marginal_to_uniform(self):
        """Ensures correct output from _transform_each_marginal_to_uniform."""

        this_feature_matrix = prob_dist._transform_each_marginal_to_uniform(
            FEATURE_MATRIX)

        self.assertTrue(numpy.allclose(
            this_feature_matrix, FEATURE_MATRIX_UNIFORM_MARGINALS,
            atol=TOLERANCE, equal_nan=True))

    def test_transform_each_marginal_to_normal(self):
        """Ensures correct output from _transform_each_marginal_to_normal.

        Rather than testing exact values (which have many decimal points and are
        hard to calculate), this method ensures that each column has the desired
        rank order.

        This is technically an integration test, not a unit test, since it calls
        _transform_each_marginal_to_uniform.
        """

        this_feature_matrix_normal_marginals = (
            prob_dist._transform_each_marginal_to_normal(FEATURE_MATRIX))
        this_feature_matrix_normal_marginals = rounder.round_to_nearest(
            this_feature_matrix_normal_marginals, TOLERANCE)

        this_feature_matrix_uniform_marginals = (
            prob_dist._transform_each_marginal_to_uniform(
                this_feature_matrix_normal_marginals))

        self.assertTrue(numpy.allclose(
            this_feature_matrix_uniform_marginals,
            FEATURE_MATRIX_UNIFORM_MARGINALS, atol=TOLERANCE, equal_nan=True))

    def test_normalize_class_probabilities(self):
        """Ensures correct output from _normalize_class_probabilities."""

        this_prob_matrix = prob_dist._normalize_class_probabilities(
            ORIGINAL_PROB_MATRIX)

        self.assertTrue(numpy.allclose(
            this_prob_matrix, NORMALIZED_PROB_MATRIX, atol=TOLERANCE,
            equal_nan=True))

    def test_get_feature_means(self):
        """Ensures correct output from _get_feature_means."""

        these_feature_means = prob_dist._get_feature_means(FEATURE_MATRIX)
        self.assertTrue(numpy.allclose(
            these_feature_means, FEATURE_MEANS, atol=TOLERANCE))

    def test_get_covariance_matrix_diagonal(self):
        """Ensures correct output from _get_covariance_matrix.

        In this case, covariance matrix is constrained to be diagonal.
        """

        this_covariance_matrix, _ = prob_dist._get_covariance_matrix(
            FEATURE_MATRIX, assume_diagonal=True)
        self.assertTrue(numpy.allclose(
            this_covariance_matrix, DIAG_COVARIANCE_MATRIX, atol=TOLERANCE))

    def test_get_covariance_matrix_non_diagonal(self):
        """Ensures correct output from _get_covariance_matrix.

        In this case, covariance matrix is *not* constrained to be diagonal.
        """

        this_covariance_matrix, _ = prob_dist._get_covariance_matrix(
            FEATURE_MATRIX, assume_diagonal=False)
        self.assertTrue(numpy.allclose(
            this_covariance_matrix, NON_DIAG_COVARIANCE_MATRIX, atol=TOLERANCE))

    def test_fit_multivariate_normal_too_many_nans(self):
        """Ensures that fit_multivariate_normal throws too-many-NaN error."""

        with self.assertRaises(ValueError):
            prob_dist.fit_multivariate_normal(FEATURE_MATRIX_TOO_MANY_NAN)

    def test_fit_mvn_for_each_class_bad_labels(self):
        """Ensures that fit_mvn_for_each_class throws error*.

        * Because there are no example for the 3rd class.
        """

        this_feature_matrix = copy.deepcopy(FEATURE_MATRIX)
        with self.assertRaises(ValueError):
            prob_dist.fit_mvn_for_each_class(
                this_feature_matrix, class_labels=BINARY_CLASS_LABELS, num_classes=3)


if __name__ == '__main__':
    unittest.main()
