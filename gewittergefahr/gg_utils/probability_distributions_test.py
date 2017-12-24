"""Unit tests for probability_distributions.py."""

import copy
import unittest
import numpy
import pandas
from gewittergefahr.gg_utils import probability_distributions as prob_dist

TOLERANCE = 1e-6

# The following constants are used to test _transform_each_marginal_to_uniform
# and _transform_each_marginal_to_normal.
NEW_VALUES_FEATURE_A = numpy.array([1., 2., numpy.nan, 4., 5.])
NEW_VALUES_FEATURE_B = numpy.array([0., -1., 1., numpy.nan, numpy.nan])
NEW_VALUES_FEATURE_C = numpy.array([12., 10., 8., 6., 4.])

NEW_FEATURE_DICT = {'a': NEW_VALUES_FEATURE_A, 'b': NEW_VALUES_FEATURE_B,
                    'c': NEW_VALUES_FEATURE_C}
NEW_FEATURE_TABLE = pandas.DataFrame.from_dict(NEW_FEATURE_DICT)

ORIG_VALUES_FEATURE_A = numpy.array([2., 4., 6., 8.])
ORIG_VALUES_FEATURE_B = numpy.array([-3., 0., numpy.nan, 0.])
ORIG_VALUES_FEATURE_C = numpy.array([5., numpy.nan, 10., 15.])

ORIG_FEATURE_DICT = {'a': ORIG_VALUES_FEATURE_A, 'b': ORIG_VALUES_FEATURE_B,
                     'c': ORIG_VALUES_FEATURE_C}
ORIG_FEATURE_TABLE = pandas.DataFrame.from_dict(ORIG_FEATURE_DICT)

THESE_VALUES_FEATURE_A = numpy.array([0.25, 0.5, 0.5, 0.75, 1.])
THESE_VALUES_FEATURE_B = numpy.array([0.666667, 0.333333, 1., 0.5, 0.5])
THESE_VALUES_FEATURE_C = numpy.array([1., 0.8, 0.6, 0.4, 0.2])

THIS_FEATURE_DICT = {'a': THESE_VALUES_FEATURE_A, 'b': THESE_VALUES_FEATURE_B,
                     'c': THESE_VALUES_FEATURE_C}
FEATURE_TABLE_UNIF_MARGINALS_NEW_TO_NEW = pandas.DataFrame.from_dict(
    THIS_FEATURE_DICT)

THESE_VALUES_FEATURE_A = numpy.array([0., 0.25, 0.5, 0.5, 0.5])
THESE_VALUES_FEATURE_B = numpy.array([1., 0.333333, 1., 0.5, 0.5])
THESE_VALUES_FEATURE_C = numpy.array(
    [0.666667, 0.666667, 0.333333, 0.333333, 0.])

THIS_FEATURE_DICT = {'a': THESE_VALUES_FEATURE_A, 'b': THESE_VALUES_FEATURE_B,
                     'c': THESE_VALUES_FEATURE_C}
FEATURE_TABLE_UNIF_MARGINALS_NEW_TO_ORIG = pandas.DataFrame.from_dict(
    THIS_FEATURE_DICT)

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
FEATURE_MATRIX = numpy.array(
    [[1., 0., 12.],
     [2., -1, 10.],
     [numpy.nan, 1., 8.],
     [4., numpy.nan, 6.],
     [5., numpy.nan, 4.]])

FEATURE_MEANS = numpy.array([3., 0., 8.])
DIAG_COVARIANCE_MATRIX = numpy.array(
    [[10., 0., 0.], [0., 3., 0.], [0., 0., 30.]]) / 3
NON_DIAG_COVARIANCE_MATRIX = numpy.array(
    [[10., 3., -20.], [3., 3., -3.], [-20., -3., 30.]]) / 3

# The following constants are used to test fit_multivariate_normal.
THESE_VALUES_FEATURE_A = numpy.array([1., 2., numpy.nan, 4., 5.])
THESE_VALUES_FEATURE_B = numpy.array(
    [0., numpy.nan, numpy.nan, numpy.nan, numpy.nan])
THESE_VALUES_FEATURE_C = numpy.array([12., 10., 8., 6., 4.])

THIS_FEATURE_DICT = {'a': THESE_VALUES_FEATURE_A, 'b': THESE_VALUES_FEATURE_B,
                     'c': THESE_VALUES_FEATURE_C}
FEATURE_TABLE_TOO_MANY_NAN = pandas.DataFrame.from_dict(THIS_FEATURE_DICT)

# The following constants are used to test fit_mvn_for_each_class.
BINARY_CLASS_LABELS = numpy.array([0, 1, 0, 0, 1], dtype=int)

# The following constants are used to test apply_mvn_for_each_class.
THIS_MVN_DICTIONARY = {prob_dist.FEATURE_NAMES_KEY: ['b', 'c', 'a']}
LIST_OF_MVN_DICTIONARIES = [THIS_MVN_DICTIONARY, THIS_MVN_DICTIONARY]


class ProbabilityDistributionsTests(unittest.TestCase):
    """Each method is a unit test for probability_distributions.py."""

    def test_transform_each_marginal_to_uniform_no_orig_table(self):
        """Ensures correct output from _transform_each_marginal_to_uniform.

        In this case the argument `orig_feature_table` is None.
        """

        this_feature_table = prob_dist._transform_each_marginal_to_uniform(
            NEW_FEATURE_TABLE, orig_feature_table=None)

        self.assertTrue(numpy.allclose(
            this_feature_table.as_matrix(),
            FEATURE_TABLE_UNIF_MARGINALS_NEW_TO_NEW.as_matrix(), atol=TOLERANCE,
            equal_nan=True))

    def test_transform_each_marginal_to_uniform_columns_in_same_order(self):
        """Ensures correct output from _transform_each_marginal_to_uniform.

        In this case, columns of `new_feature_table` and `orig_feature_table`
        are in the same order.
        """

        this_feature_table = prob_dist._transform_each_marginal_to_uniform(
            new_feature_table=NEW_FEATURE_TABLE,
            orig_feature_table=ORIG_FEATURE_TABLE)

        self.assertTrue(numpy.allclose(
            this_feature_table.as_matrix(),
            FEATURE_TABLE_UNIF_MARGINALS_NEW_TO_ORIG.as_matrix(),
            atol=TOLERANCE, equal_nan=True))

    def test_transform_each_marginal_to_uniform_columns_in_diff_order(self):
        """Ensures correct output from _transform_each_marginal_to_uniform.

        In this case, columns of `new_feature_table` and `orig_feature_table`
        are in different orders.
        """

        this_feature_table = prob_dist._transform_each_marginal_to_uniform(
            new_feature_table=NEW_FEATURE_TABLE,
            orig_feature_table=ORIG_FEATURE_TABLE[['b', 'c', 'a']])

        self.assertTrue(numpy.allclose(
            this_feature_table.as_matrix(),
            FEATURE_TABLE_UNIF_MARGINALS_NEW_TO_ORIG.as_matrix(),
            atol=TOLERANCE, equal_nan=True))

    def test_transform_each_marginal_to_uniform_extra_columns(self):
        """Ensures correct output from _transform_each_marginal_to_uniform.

        In this case, `orig_feature_table` contains all columns of
        `new_feature_table` plus extra columns.
        """

        this_orig_feature_table = copy.deepcopy(ORIG_FEATURE_TABLE)
        this_orig_feature_table['d'] = this_orig_feature_table['b']
        this_feature_table = prob_dist._transform_each_marginal_to_uniform(
            new_feature_table=NEW_FEATURE_TABLE,
            orig_feature_table=this_orig_feature_table)

        self.assertTrue(numpy.allclose(
            this_feature_table.as_matrix(),
            FEATURE_TABLE_UNIF_MARGINALS_NEW_TO_ORIG.as_matrix(),
            atol=TOLERANCE, equal_nan=True))

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
            prob_dist.fit_multivariate_normal(FEATURE_TABLE_TOO_MANY_NAN)

    def test_fit_mvn_for_each_class_missing_class(self):
        """Ensures that fit_mvn_for_each_class throws error*.

        * Because there are no examples for the 3rd class.
        """

        with self.assertRaises(ValueError):
            prob_dist.fit_mvn_for_each_class(
                NEW_FEATURE_TABLE, class_labels=BINARY_CLASS_LABELS,
                num_classes=3)

    def test_apply_mvn_for_each_class_features_in_diff_order(self):
        """Ensures that apply_mvn_for_each_class throws error*.

        Because features in each MVN (multivariate normal) dictionary are in
        different order than in `feature_table`.
        """

        with self.assertRaises(ValueError):
            prob_dist.apply_mvn_for_each_class(
                feature_table=NEW_FEATURE_TABLE,
                list_of_mvn_dictionaries=LIST_OF_MVN_DICTIONARIES)


if __name__ == '__main__':
    unittest.main()
