"""Unit tests for find_gridrad_normalization_params.py."""

import copy
import unittest
import numpy
import pandas
from gewittergefahr.scripts import \
    find_gridrad_normalization_params as find_norm_params

TOLERANCE = 1e-6

# The following constants are used to test _update_normalization_params.
ORIG_PARAMETER_DICT = {
    find_norm_params.NUM_VALUES_KEY: 20,
    find_norm_params.MEAN_KEY: 5.,
    find_norm_params.MEAN_OF_SQUARES_KEY: 10.
}

NEW_DATA_MATRIX = numpy.array([[0, 1, 2, 3, 4],
                               [1, 2, 4, 2, 1]], dtype=float)

NEW_PARAMETER_DICT = {
    find_norm_params.NUM_VALUES_KEY: 30,
    find_norm_params.MEAN_KEY: 4.,
    find_norm_params.MEAN_OF_SQUARES_KEY: 8.533333
}

# The following constants are used to test _get_standard_deviation.
STDEV_INPUT_MATRIX = numpy.array([[1, 2, 3, 4, 5],
                                  [6, 7, 8, 9, 10]], dtype=float)
STANDARD_DEVIATION = numpy.std(STDEV_INPUT_MATRIX, ddof=1)

# The following constants are used to test _convert_normalization_params with
# single-indexing.
FIRST_KEY_NO_HEIGHT = 'reflectivity_dbz'
SECOND_KEY_NO_HEIGHT = 'reflectivity_column_max_dbz'
THIRD_KEY_NO_HEIGHT = 'low_level_shear_s01'

FIRST_PARAMETER_DICT = {
    find_norm_params.NUM_VALUES_KEY: 100,
    find_norm_params.MEAN_KEY: 15.,
    find_norm_params.MEAN_OF_SQUARES_KEY: 300.
}
SECOND_PARAMETER_DICT = {
    find_norm_params.NUM_VALUES_KEY: 100,
    find_norm_params.MEAN_KEY: 25.,
    find_norm_params.MEAN_OF_SQUARES_KEY: 1000.
}
THIRD_PARAMETER_DICT = {
    find_norm_params.NUM_VALUES_KEY: 400,
    find_norm_params.MEAN_KEY: 8e-3,
    find_norm_params.MEAN_OF_SQUARES_KEY: 1e-4
}

PARAMETER_DICT_DICT_NO_HEIGHT = {
    FIRST_KEY_NO_HEIGHT: FIRST_PARAMETER_DICT,
    SECOND_KEY_NO_HEIGHT: SECOND_PARAMETER_DICT,
    THIRD_KEY_NO_HEIGHT: THIRD_PARAMETER_DICT
}

FIRST_NORM_VALUES = numpy.array([
    FIRST_PARAMETER_DICT[find_norm_params.MEAN_KEY],
    find_norm_params._get_standard_deviation(FIRST_PARAMETER_DICT)
])
SECOND_NORM_VALUES = numpy.array([
    SECOND_PARAMETER_DICT[find_norm_params.MEAN_KEY],
    find_norm_params._get_standard_deviation(SECOND_PARAMETER_DICT)
])
THIRD_NORM_VALUES = numpy.array([
    THIRD_PARAMETER_DICT[find_norm_params.MEAN_KEY],
    find_norm_params._get_standard_deviation(THIRD_PARAMETER_DICT)
])

NORMALIZATION_DICT_NO_HEIGHT = {
    FIRST_KEY_NO_HEIGHT: FIRST_NORM_VALUES,
    SECOND_KEY_NO_HEIGHT: SECOND_NORM_VALUES,
    THIRD_KEY_NO_HEIGHT: THIRD_NORM_VALUES
}
NORMALIZATION_TABLE_NO_HEIGHT = pandas.DataFrame.from_dict(
    NORMALIZATION_DICT_NO_HEIGHT, orient='index')

COLUMN_DICT_OLD_TO_NEW = {
    0: find_norm_params.MEAN_KEY, 1: find_norm_params.STANDARD_DEVIATION_KEY
}
NORMALIZATION_TABLE_NO_HEIGHT.rename(
    columns=COLUMN_DICT_OLD_TO_NEW, inplace=True)

# The following constants are used to test _convert_normalization_params with
# double-indexing.
FIRST_KEY_WITH_HEIGHT = ('reflectivity_dbz', 1000)
SECOND_KEY_WITH_HEIGHT = ('reflectivity_column_max_dbz', 250)
THIRD_KEY_WITH_HEIGHT = ('low_level_shear_s01', 250)

PARAMETER_DICT_DICT_WITH_HEIGHT = {
    FIRST_KEY_WITH_HEIGHT: FIRST_PARAMETER_DICT,
    SECOND_KEY_WITH_HEIGHT: SECOND_PARAMETER_DICT,
    THIRD_KEY_WITH_HEIGHT: THIRD_PARAMETER_DICT
}

NORMALIZATION_DICT_WITH_HEIGHT = {
    FIRST_KEY_WITH_HEIGHT: FIRST_NORM_VALUES,
    SECOND_KEY_WITH_HEIGHT: SECOND_NORM_VALUES,
    THIRD_KEY_WITH_HEIGHT: THIRD_NORM_VALUES
}
NORMALIZATION_TABLE_WITH_HEIGHT = pandas.DataFrame.from_dict(
    NORMALIZATION_DICT_WITH_HEIGHT, orient='index')
NORMALIZATION_TABLE_WITH_HEIGHT.rename(
    columns=COLUMN_DICT_OLD_TO_NEW, inplace=True)


def _compare_parameter_dicts(first_parameter_dict, second_parameter_dict):
    """Compares two dictionaries with normalization params.

    :param first_parameter_dict: First dictionary.
    :param second_parameter_dict: Second dictionary.
    :return: are_dicts_equal: Boolean flag.
    """

    first_keys = first_parameter_dict.keys()
    second_keys = second_parameter_dict.keys()
    if set(first_keys) != set(second_keys):
        return False

    for this_key in first_keys:
        if this_key == find_norm_params.MEAN_KEY:
            if (first_parameter_dict[this_key] !=
                    second_parameter_dict[this_key]):
                return False
        else:
            if not numpy.isclose(first_parameter_dict[this_key],
                                 second_parameter_dict[this_key],
                                 atol=TOLERANCE):
                return False

    return True


def _compare_normalization_tables(first_norm_table, second_norm_table):
    """Compares two pandas DataFrame with normalization params.

    :param first_norm_table: First table.
    :param second_norm_table: Second table.
    :return: are_tables_equal: Boolean flag.
    """

    first_indices = first_norm_table.index
    second_indices = second_norm_table.index
    if len(first_indices) != len(second_indices):
        return False

    first_column_names = list(first_norm_table)
    second_column_names = list(second_norm_table)
    if set(first_column_names) != set(second_column_names):
        return False

    for this_index in first_indices:
        for this_column_name in first_column_names:
            if not numpy.isclose(
                    first_norm_table[this_column_name].loc[[this_index]],
                    second_norm_table[this_column_name].loc[[this_index]],
                    atol=TOLERANCE):
                return False

    return True


class FindGridadNormParamsTests(unittest.TestCase):
    """Each method is a unit test for find_gridrad_normalization_params.py."""

    def test_update_normalization_params(self):
        """Ensures correct output from _update_normalization_params."""

        this_new_parameter_dict = find_norm_params._update_normalization_params(
            parameter_dict=copy.deepcopy(ORIG_PARAMETER_DICT),
            new_data_matrix=NEW_DATA_MATRIX)

        self.assertTrue(_compare_parameter_dicts(
            this_new_parameter_dict, NEW_PARAMETER_DICT))

    def test_get_standard_deviation(self):
        """Ensures correct output from _get_standard_deviation."""

        this_parameter_dict = {
            find_norm_params.NUM_VALUES_KEY: STDEV_INPUT_MATRIX.size,
            find_norm_params.MEAN_KEY: numpy.mean(STDEV_INPUT_MATRIX),
            find_norm_params.MEAN_OF_SQUARES_KEY:
                numpy.mean(STDEV_INPUT_MATRIX ** 2)
        }

        this_standard_deviation = find_norm_params._get_standard_deviation(
            this_parameter_dict)
        self.assertTrue(numpy.isclose(
            this_standard_deviation, STANDARD_DEVIATION, atol=TOLERANCE))

    def test_convert_normalization_params_no_height(self):
        """Ensures correct output from _convert_normalization_params.

        In this case, each key is a field name, rather than field/height pair.
        Thus, the table should be single-indexed, not double-indexed.
        """

        this_normalization_table = (
            find_norm_params._convert_normalization_params(
                PARAMETER_DICT_DICT_NO_HEIGHT)
        )

        self.assertTrue(_compare_normalization_tables(
            this_normalization_table, NORMALIZATION_TABLE_NO_HEIGHT))

    def test_convert_normalization_params_with_height(self):
        """Ensures correct output from _convert_normalization_params.

        In this case, each key is a field/height pair, rather than one field
        name.  Thus, the table should be double-indexed, not single-indexed.
        """

        this_normalization_table = (
            find_norm_params._convert_normalization_params(
                PARAMETER_DICT_DICT_WITH_HEIGHT)
        )

        self.assertTrue(_compare_normalization_tables(
            this_normalization_table, NORMALIZATION_TABLE_WITH_HEIGHT))


if __name__ == '__main__':
    unittest.main()
