"""Unit tests for general_utils.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import general_utils

TOLERANCE = 1e-6

# The following constants are used to test _find_nearest_value.
SORTED_ARRAY = numpy.array([-5., -3., -1., 0., 1.5, 4., 9.])
TEST_VALUES = numpy.array([-6., -5., -4., -3., 5., 8., 9., 10.])
NEAREST_INDICES_FOR_TEST_VALUES = numpy.array([0., 0., 1., 1., 5., 6., 6., 6.])

# The following constants are used to test split_array_by_nan.
ARRAY_WITHOUT_NAN = numpy.array([30, 50])
ARRAY_WITHOUT_NAN_AS_LIST = [numpy.array([30, 50])]

ARRAY_WITH_ONE_NAN = numpy.array([30, 50, numpy.nan, 40])
ARRAY_WITH_ONE_NAN_AS_LIST = [numpy.array([30, 50]), numpy.array([40])]

ARRAY_WITH_MANY_NANS = numpy.array(
    [numpy.nan, 30, 50, numpy.nan, 40, numpy.nan, numpy.nan, 70, 40, 10,
     numpy.nan, numpy.nan])
ARRAY_WITH_MANY_NANS_AS_LIST = [
    numpy.array([30, 50]), numpy.array([40]), numpy.array([70, 40, 10])]


class GeneralUtilsTests(unittest.TestCase):
    """Each method is a unit test for general_utils.py."""

    def test_find_nearest_value(self):
        """Ensures correct output from _find_nearest_value."""

        these_nearest_indices = numpy.full(len(TEST_VALUES), -1)
        for i in range(len(TEST_VALUES)):
            _, these_nearest_indices[i] = general_utils.find_nearest_value(
                sorted_input_values=SORTED_ARRAY, test_value=TEST_VALUES[i])

        self.assertTrue(numpy.array_equal(
            these_nearest_indices, NEAREST_INDICES_FOR_TEST_VALUES))

    def test_split_array_by_nan_0nans(self):
        """Ensures correct output from split_array_by_nan.

        In this case, input array has zero NaN's.
        """

        this_list_of_arrays = general_utils.split_array_by_nan(
            ARRAY_WITHOUT_NAN)
        self.assertTrue(len(this_list_of_arrays) ==
                        len(ARRAY_WITHOUT_NAN_AS_LIST))

        for i in range(len(this_list_of_arrays)):
            self.assertTrue(numpy.allclose(
                this_list_of_arrays[i], ARRAY_WITHOUT_NAN_AS_LIST[i]))

    def test_split_array_by_nan_1nan(self):
        """Ensures correct output from split_array_by_nan.

        In this case, input array has one NaN.
        """

        this_list_of_arrays = general_utils.split_array_by_nan(
            ARRAY_WITH_ONE_NAN)
        self.assertTrue(len(this_list_of_arrays) ==
                        len(ARRAY_WITH_ONE_NAN_AS_LIST))

        for i in range(len(this_list_of_arrays)):
            self.assertTrue(numpy.allclose(
                this_list_of_arrays[i], ARRAY_WITH_ONE_NAN_AS_LIST[i]))

    def test_split_array_by_nan_many_nans(self):
        """Ensures correct output from split_array_by_nan.

        In this case, input array has many NaN's.
        """

        this_list_of_arrays = general_utils.split_array_by_nan(
            ARRAY_WITH_MANY_NANS)
        self.assertTrue(len(this_list_of_arrays) ==
                        len(ARRAY_WITH_MANY_NANS_AS_LIST))

        for i in range(len(this_list_of_arrays)):
            self.assertTrue(numpy.allclose(
                this_list_of_arrays[i], ARRAY_WITH_MANY_NANS_AS_LIST[i]))


if __name__ == '__main__':
    unittest.main()
