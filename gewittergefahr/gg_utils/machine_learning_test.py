"""Unit tests for machine_learning.py."""

import copy
import unittest
import numpy
import pandas
from gewittergefahr.gg_utils import machine_learning

THESE_FEATURE_VALUES0 = numpy.array([1., 4.])
THESE_FEATURE_VALUES1 = numpy.array([2., 5.])
THESE_FEATURE_VALUES2 = numpy.array([3., 6.])
THIS_FEATURE_MATRIX = numpy.hstack((
    numpy.reshape(THESE_FEATURE_VALUES0, (2, 1)),
    numpy.reshape(THESE_FEATURE_VALUES1, (2, 1)),
    numpy.reshape(THESE_FEATURE_VALUES2, (2, 1))))

SVD_TRANSFORMED_FEATURE_TABLE_DEFAULT_NAMES = pandas.DataFrame(
    THIS_FEATURE_MATRIX)

THIS_FEATURE_DICT = {
    'svd_transformed_feature0000': THESE_FEATURE_VALUES0,
    'svd_transformed_feature0001': THESE_FEATURE_VALUES1,
    'svd_transformed_feature0002': THESE_FEATURE_VALUES2
}
SVD_TRANSFORMED_FEATURE_TABLE_CUSTOM_NAMES = pandas.DataFrame.from_dict(
    THIS_FEATURE_DICT)


class MachineLearningTests(unittest.TestCase):
    """Each method is a unit test for machine_learning.py."""

    def test_rename_svd_transformed_features(self):
        """Ensures correct output from _rename_svd_transformed_features."""

        this_feature_table = copy.deepcopy(
            SVD_TRANSFORMED_FEATURE_TABLE_DEFAULT_NAMES)
        this_feature_table = machine_learning._rename_svd_transformed_features(
            this_feature_table)

        self.assertTrue(this_feature_table.equals(
            SVD_TRANSFORMED_FEATURE_TABLE_CUSTOM_NAMES))


if __name__ == '__main__':
    unittest.main()
