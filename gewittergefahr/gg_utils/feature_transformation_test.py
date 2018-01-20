"""Unit tests for feature_transformation.py."""

import copy
import unittest
import numpy
import pandas
from gewittergefahr.gg_utils import feature_transformation as feature_trans

TOLERANCE = 1e-6

FEATURE_NAMES = ['a', 'b', 'c']
FEATURE_MATRIX = numpy.array(
    [[1., 0., 10.],
     [2., 1., 9.],
     [3., 2., numpy.nan],
     [4., numpy.nan, 8.],
     [5., numpy.nan, -3.]])

FEATURE_MATRIX_TOO_MANY_NAN = numpy.array(
    [[1., 0., 10.],
     [2., numpy.nan, 10.],
     [3., numpy.nan, numpy.nan],
     [4., numpy.nan, 5.],
     [5., numpy.nan, 0.]])

THIS_FEATURE_DICT = {}
for j in range(len(FEATURE_NAMES)):
    THIS_FEATURE_DICT.update({FEATURE_NAMES[j]: FEATURE_MATRIX[:, j]})
FEATURE_TABLE = pandas.DataFrame.from_dict(THIS_FEATURE_DICT)

THIS_FEATURE_DICT = {}
for j in range(len(FEATURE_NAMES)):
    THIS_FEATURE_DICT.update(
        {FEATURE_NAMES[j]: FEATURE_MATRIX_TOO_MANY_NAN[:, j]})
FEATURE_TABLE_TOO_MANY_NAN = pandas.DataFrame.from_dict(THIS_FEATURE_DICT)

# The following constants are used to test replace_missing_values.
FEATURE_MEANS = numpy.array([3., 1., 6.])
REPLACEMENT_DICT_MEAN = {
    feature_trans.FEATURE_NAMES_KEY: FEATURE_NAMES,
    feature_trans.ORIGINAL_MEANS_KEY: FEATURE_MEANS
}
FEATURE_MATRIX_MISSING_TO_MEAN = numpy.array(
    [[1., 0., 10.],
     [2., 1., 9.],
     [3., 2., 6.],
     [4., 1., 8.],
     [5., 1., -3.]])

FEATURE_MEDIANS = numpy.array([3., 1., 8.5])
REPLACEMENT_DICT_MEDIAN = {
    feature_trans.FEATURE_NAMES_KEY: FEATURE_NAMES,
    feature_trans.ORIGINAL_MEDIANS_KEY: FEATURE_MEDIANS
}
FEATURE_MATRIX_MISSING_TO_MEDIAN = numpy.array(
    [[1., 0., 10.],
     [2., 1., 9.],
     [3., 2., 8.5],
     [4., 1., 8.],
     [5., 1., -3.]])

# The following constants are used to test standardize_features.
FEATURE_STANDARD_DEVIATIONS = numpy.sqrt(numpy.array([2.5, 1., 110. / 3]))
STANDARDIZATION_DICT = {
    feature_trans.FEATURE_NAMES_KEY: FEATURE_NAMES,
    feature_trans.ORIGINAL_MEANS_KEY: FEATURE_MEANS,
    feature_trans.ORIGINAL_STDEVS_KEY: FEATURE_STANDARD_DEVIATIONS
}

# The following constants are used to test
# _reorder_standardization_or_replacement_dict.
PERMUTED_FEATURE_NAMES = ['b', 'c', 'a']
PERMUTED_STANDARDIZATION_DICT = {
    feature_trans.FEATURE_NAMES_KEY: PERMUTED_FEATURE_NAMES,
    feature_trans.ORIGINAL_MEANS_KEY: numpy.array([1., 6., 3.]),
    feature_trans.ORIGINAL_STDEVS_KEY:
        numpy.sqrt(numpy.array([1., 110. / 3, 2.5]))
}

# The following constants are used to test filter_svd_by_explained_variance.
THIS_PC_MATRIX = numpy.array(
    [[0., 0.1, 0.2, 0.3, 0.4],
     [0.5, 0.3, 0.2, 0.22, 0.11],
     [1.2, -0.6, -0.1, 0.17, 0.4],
     [-1.9, -1.2, 0.2, 0.1, 0.1],
     [-2.6, -1., -0.3, 0.01, 0.005]])
THIS_EIGENVALUE_MATRIX = numpy.array(
    [[50., 0., 0.],
     [0., 2.7, 0.],
     [0., 0.00123, 0.]])
THIS_EOF_MATRIX = numpy.array(
    [[0.72, 0.87, -0.03],
     [-0.43, -0.14, 0.025],
     [0., 0.17, 1.03]])

SVD_DICTIONARY = {
    feature_trans.PC_MATRIX_KEY: THIS_PC_MATRIX,
    feature_trans.EIGENVALUE_MATRIX_KEY: THIS_EIGENVALUE_MATRIX,
    feature_trans.EOF_MATRIX_KEY: THIS_EOF_MATRIX
}
SVD_DICTIONARY_90PCT_VARIANCE = {
    feature_trans.PC_MATRIX_KEY: THIS_PC_MATRIX[:, :1],
    feature_trans.EIGENVALUE_MATRIX_KEY: THIS_EIGENVALUE_MATRIX[:1, :1],
    feature_trans.EOF_MATRIX_KEY: THIS_EOF_MATRIX[:, :1]
}
SVD_DICTIONARY_95PCT_VARIANCE = {
    feature_trans.PC_MATRIX_KEY: THIS_PC_MATRIX[:, :2],
    feature_trans.EIGENVALUE_MATRIX_KEY: THIS_EIGENVALUE_MATRIX[:2, :2],
    feature_trans.EOF_MATRIX_KEY: THIS_EOF_MATRIX[:, :2]
}


class FeatureTransformationTests(unittest.TestCase):
    """Each method is a unit test for feature_transformation.py."""

    def test_reorder_standardization_or_replacement_dict(self):
        """Ensures crctness of _reorder_standardization_or_replacement_dict."""

        this_standardization_dict = copy.deepcopy(STANDARDIZATION_DICT)
        this_standardization_dict = (
            feature_trans._reorder_standardization_or_replacement_dict(
                this_standardization_dict, PERMUTED_FEATURE_NAMES))

        self.assertTrue(set(this_standardization_dict.keys()) ==
                        set(PERMUTED_STANDARDIZATION_DICT.keys()))
        self.assertTrue(numpy.allclose(
            this_standardization_dict[feature_trans.ORIGINAL_MEANS_KEY],
            PERMUTED_STANDARDIZATION_DICT[feature_trans.ORIGINAL_MEANS_KEY],
            atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            this_standardization_dict[feature_trans.ORIGINAL_STDEVS_KEY],
            PERMUTED_STANDARDIZATION_DICT[feature_trans.ORIGINAL_STDEVS_KEY],
            atol=TOLERANCE))

    def test_reorder_standardization_or_replacement_dict_mismatch(self):
        """Ensures _reorder_standardization_or_replacement_dict throws error*.

        * Because standardization_dict does not contain all desired features.
        """

        these_feature_names = FEATURE_NAMES + ['foo']
        with self.assertRaises(ValueError):
            feature_trans._reorder_standardization_or_replacement_dict(
                STANDARDIZATION_DICT, these_feature_names)

    def test_replace_missing_values_with_mean(self):
        """Ensures correct output from replace_missing_values.

        In this case, missing values of feature F are replaced with the mean
        F-value.
        """

        this_feature_table, this_replacement_dict = (
            feature_trans.replace_missing_values(
                FEATURE_TABLE,
                replacement_method=feature_trans.MEAN_VALUE_REPLACEMENT_METHOD))

        self.assertTrue(numpy.allclose(
            FEATURE_MATRIX_MISSING_TO_MEAN, this_feature_table.as_matrix(),
            atol=TOLERANCE))

        self.assertTrue(set(this_replacement_dict.keys()) ==
                        set(REPLACEMENT_DICT_MEAN.keys()))
        self.assertTrue(numpy.allclose(
            this_replacement_dict[feature_trans.ORIGINAL_MEANS_KEY],
            REPLACEMENT_DICT_MEAN[feature_trans.ORIGINAL_MEANS_KEY],
            atol=TOLERANCE))

    def test_replace_missing_values_with_median(self):
        """Ensures correct output from replace_missing_values.

        In this case, missing values of feature F are replaced with the median
        F-value.
        """

        this_feature_table, this_replacement_dict = (
            feature_trans.replace_missing_values(
                FEATURE_TABLE, replacement_method=
                feature_trans.MEDIAN_VALUE_REPLACEMENT_METHOD))

        self.assertTrue(numpy.allclose(
            FEATURE_MATRIX_MISSING_TO_MEDIAN, this_feature_table.as_matrix(),
            atol=TOLERANCE))

        self.assertTrue(set(this_replacement_dict.keys()) ==
                        set(REPLACEMENT_DICT_MEDIAN.keys()))
        self.assertTrue(numpy.allclose(
            this_replacement_dict[feature_trans.ORIGINAL_MEDIANS_KEY],
            REPLACEMENT_DICT_MEDIAN[feature_trans.ORIGINAL_MEDIANS_KEY],
            atol=TOLERANCE))

    def test_standardize_features(self):
        """Ensures correct output from standardize_features."""

        _, this_standardization_dict = feature_trans.standardize_features(
            FEATURE_TABLE)

        self.assertTrue(set(this_standardization_dict.keys()) ==
                        set(STANDARDIZATION_DICT.keys()))
        self.assertTrue(numpy.allclose(
            this_standardization_dict[feature_trans.ORIGINAL_MEANS_KEY],
            STANDARDIZATION_DICT[feature_trans.ORIGINAL_MEANS_KEY],
            atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            this_standardization_dict[feature_trans.ORIGINAL_STDEVS_KEY],
            STANDARDIZATION_DICT[feature_trans.ORIGINAL_STDEVS_KEY],
            atol=TOLERANCE))

    def test_standardize_features_too_many_nans(self):
        """Ensures that standardize_features throws too-many-NaN error."""

        with self.assertRaises(ValueError):
            feature_trans.standardize_features(FEATURE_TABLE_TOO_MANY_NAN)

    def test_perform_svd_no_crash(self):
        """Ensures that perform_svd does not crash.

        I do not test for expected values, because these would be very tedious
        to calculate.
        """

        feature_trans.perform_svd(FEATURE_TABLE)

    def test_filter_svd_by_explained_variance_90pct(self):
        """Ensures correct output from filter_svd_by_explained_variance.

        In this case, fraction of variance to keep is 90%.
        """

        this_svd_dictionary = copy.deepcopy(SVD_DICTIONARY)
        this_svd_dictionary = feature_trans.filter_svd_by_explained_variance(
            this_svd_dictionary, fraction_of_variance_to_keep=0.9)

        for this_key in this_svd_dictionary.keys():
            self.assertTrue(numpy.allclose(
                this_svd_dictionary[this_key],
                SVD_DICTIONARY_90PCT_VARIANCE[this_key], atol=TOLERANCE))

    def test_filter_svd_by_explained_variance_95pct(self):
        """Ensures correct output from filter_svd_by_explained_variance.

        In this case, fraction of variance to keep is 95%.
        """

        this_svd_dictionary = copy.deepcopy(SVD_DICTIONARY)
        this_svd_dictionary = feature_trans.filter_svd_by_explained_variance(
            this_svd_dictionary, fraction_of_variance_to_keep=0.95)

        for this_key in this_svd_dictionary.keys():
            self.assertTrue(numpy.allclose(
                this_svd_dictionary[this_key],
                SVD_DICTIONARY_95PCT_VARIANCE[this_key], atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
