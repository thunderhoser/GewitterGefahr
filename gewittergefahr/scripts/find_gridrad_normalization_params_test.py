"""Unit tests for find_gridrad_normalization_params.py."""

import copy
import unittest
import numpy
import pandas
from gewittergefahr.scripts import \
    find_gridrad_normalization_params as find_norm_params

TOLERANCE = 1e-6

# The following constants are used to test _update_z_score_params.
ORIGINAL_Z_SCORE_DICT = {
    find_norm_params.NUM_VALUES_KEY: 20,
    find_norm_params.MEAN_VALUE_KEY: 5.,
    find_norm_params.MEAN_OF_SQUARES_KEY: 10.
}

NEW_MATRIX_FOR_Z_SCORES = numpy.array([[0, 1, 2, 3, 4],
                                       [1, 2, 4, 2, 1]], dtype=float)

NEW_Z_SCORE_DICT = {
    find_norm_params.NUM_VALUES_KEY: 30,
    find_norm_params.MEAN_VALUE_KEY: 4.,
    find_norm_params.MEAN_OF_SQUARES_KEY: 8.533333
}

# The following constants are used to test _update_frequency_dict.
ROUNDING_BASE = 0.001
MAIN_FREQUENCY_DICT = {
    0.001: 5, 0.002: 3, 0.004: 7, 0.006: 2
}

NEW_MATRIX_FOR_FREQUENCIES = numpy.array([[0.003, 0.007, 0.002, 0.004],
                                          [0.006, 0.005, 0.002, -0.001],
                                          [0.006, 0.001, 0.008, 0.007]])

NEW_FREQUENCY_DICT = {
    -0.001: 1, 0.001: 6, 0.002: 5, 0.003: 1, 0.004: 8, 0.005: 1, 0.006: 4,
    0.007: 2, 0.008: 1
}

# The following constants are used to test _get_standard_deviation.
STDEV_INPUT_MATRIX = numpy.array([[1, 2, 3, 4, 5],
                                  [6, 7, 8, 9, 10]], dtype=float)
STANDARD_DEVIATION = numpy.std(STDEV_INPUT_MATRIX, ddof=1)

# The following constants are used to test _get_percentile.
SMALL_PERCENTILE_LEVEL = 3.125
MEDIUM_PERCENTILE_LEVEL = 43.75
LARGE_PERCENTILE_LEVEL = 93.75

SMALL_PERCENTILE = 0.001
MEDIUM_PERCENTILE = 0.002
LARGE_PERCENTILE = 0.005

# The following constants are used to test _convert_normalization_params with
# single-indexing and no percentiles.
FIRST_KEY_NO_HEIGHT = 'reflectivity_dbz'
SECOND_KEY_NO_HEIGHT = 'reflectivity_column_max_dbz'
THIRD_KEY_NO_HEIGHT = 'low_level_shear_s01'

FIRST_Z_SCORE_DICT = {
    find_norm_params.NUM_VALUES_KEY: 100,
    find_norm_params.MEAN_VALUE_KEY: 15.,
    find_norm_params.MEAN_OF_SQUARES_KEY: 300.
}
SECOND_Z_SCORE_DICT = {
    find_norm_params.NUM_VALUES_KEY: 100,
    find_norm_params.MEAN_VALUE_KEY: 25.,
    find_norm_params.MEAN_OF_SQUARES_KEY: 1000.
}
THIRD_Z_SCORE_DICT = {
    find_norm_params.NUM_VALUES_KEY: 400,
    find_norm_params.MEAN_VALUE_KEY: 8e-3,
    find_norm_params.MEAN_OF_SQUARES_KEY: 1e-4
}
Z_SCORE_DICT_DICT_NO_HEIGHT = {
    FIRST_KEY_NO_HEIGHT: FIRST_Z_SCORE_DICT,
    SECOND_KEY_NO_HEIGHT: SECOND_Z_SCORE_DICT,
    THIRD_KEY_NO_HEIGHT: THIRD_Z_SCORE_DICT
}

FIRST_Z_SCORE_PARAMS = numpy.array([
    FIRST_Z_SCORE_DICT[find_norm_params.MEAN_VALUE_KEY],
    find_norm_params._get_standard_deviation(FIRST_Z_SCORE_DICT)
])
SECOND_Z_SCORE_PARAMS = numpy.array([
    SECOND_Z_SCORE_DICT[find_norm_params.MEAN_VALUE_KEY],
    find_norm_params._get_standard_deviation(SECOND_Z_SCORE_DICT)
])
THIRD_Z_SCORE_PARAMS = numpy.array([
    THIRD_Z_SCORE_DICT[find_norm_params.MEAN_VALUE_KEY],
    find_norm_params._get_standard_deviation(THIRD_Z_SCORE_DICT)
])

NORMALIZATION_DICT_NO_PRCTILE = {
    FIRST_KEY_NO_HEIGHT: FIRST_Z_SCORE_PARAMS,
    SECOND_KEY_NO_HEIGHT: SECOND_Z_SCORE_PARAMS,
    THIRD_KEY_NO_HEIGHT: THIRD_Z_SCORE_PARAMS
}
NORMALIZATION_TABLE_NO_PRCTILE = pandas.DataFrame.from_dict(
    NORMALIZATION_DICT_NO_PRCTILE, orient='index')

COLUMN_DICT_OLD_TO_NEW = {
    0: find_norm_params.MEAN_VALUE_KEY,
    1: find_norm_params.STANDARD_DEVIATION_KEY
}
NORMALIZATION_TABLE_NO_PRCTILE.rename(
    columns=COLUMN_DICT_OLD_TO_NEW, inplace=True)

# The following constants are used to test _convert_normalization_params with
# single-indexing and percentiles.
FIRST_FREQUENCY_DICT = {
    0: 5, 5: 10, 10: 20, 15: 30, 20: 20, 25: 10, 30: 5
}
SECOND_FREQUENCY_DICT = {
    0: 0, 5: 5, 10: 5, 15: 10, 20: 15, 25: 30, 30: 15, 35: 10, 40: 5, 45: 5
}
THIRD_FREQUENCY_DICT = {
    -4e-3: 10, -1e-3: 15, 2e-3: 25, 5e-3: 50, 8e-3: 200, 11e-3: 50, 14e-3: 25,
    17e-3: 15, 20e-3: 10
}
FREQUENCY_DICT_DICT = {
    FIRST_KEY_NO_HEIGHT: FIRST_FREQUENCY_DICT,
    SECOND_KEY_NO_HEIGHT: SECOND_FREQUENCY_DICT,
    THIRD_KEY_NO_HEIGHT: THIRD_FREQUENCY_DICT
}

MIN_PERCENTILE_LEVEL = 1.
MAX_PERCENTILE_LEVEL = 99.

FIRST_PERCENTILES = numpy.array([
    find_norm_params._get_percentile(
        FIRST_FREQUENCY_DICT, MIN_PERCENTILE_LEVEL),
    find_norm_params._get_percentile(FIRST_FREQUENCY_DICT, MAX_PERCENTILE_LEVEL)
])
SECOND_PERCENTILES = numpy.array([
    find_norm_params._get_percentile(
        SECOND_FREQUENCY_DICT, MIN_PERCENTILE_LEVEL),
    find_norm_params._get_percentile(
        SECOND_FREQUENCY_DICT, MAX_PERCENTILE_LEVEL)
])
THIRD_PERCENTILES = numpy.array([
    find_norm_params._get_percentile(
        THIRD_FREQUENCY_DICT, MIN_PERCENTILE_LEVEL),
    find_norm_params._get_percentile(THIRD_FREQUENCY_DICT, MAX_PERCENTILE_LEVEL)
])

NORMALIZATION_DICT_WITH_PRCTILE = {
    FIRST_KEY_NO_HEIGHT: numpy.concatenate((
        FIRST_Z_SCORE_PARAMS, FIRST_PERCENTILES)),
    SECOND_KEY_NO_HEIGHT: numpy.concatenate((
        SECOND_Z_SCORE_PARAMS, SECOND_PERCENTILES)),
    THIRD_KEY_NO_HEIGHT: numpy.concatenate((
        THIRD_Z_SCORE_PARAMS, THIRD_PERCENTILES))
}
NORMALIZATION_TABLE_WITH_PRCTILE = pandas.DataFrame.from_dict(
    NORMALIZATION_DICT_WITH_PRCTILE, orient='index')

COLUMN_DICT_OLD_TO_NEW.update({
    2: find_norm_params.MIN_VALUE_KEY,
    3: find_norm_params.MAX_VALUE_KEY
})
NORMALIZATION_TABLE_WITH_PRCTILE.rename(
    columns=COLUMN_DICT_OLD_TO_NEW, inplace=True)

# The following constants are used to test _convert_normalization_params with
# double-indexing.
FIRST_KEY_WITH_HEIGHT = ('reflectivity_dbz', 1000)
SECOND_KEY_WITH_HEIGHT = ('reflectivity_column_max_dbz', 250)
THIRD_KEY_WITH_HEIGHT = ('low_level_shear_s01', 250)

Z_SCORE_DICT_DICT_WITH_HEIGHT = {
    FIRST_KEY_WITH_HEIGHT: FIRST_Z_SCORE_DICT,
    SECOND_KEY_WITH_HEIGHT: SECOND_Z_SCORE_DICT,
    THIRD_KEY_WITH_HEIGHT: THIRD_Z_SCORE_DICT
}

NORMALIZATION_DICT_WITH_HEIGHT = {
    FIRST_KEY_WITH_HEIGHT: FIRST_Z_SCORE_PARAMS,
    SECOND_KEY_WITH_HEIGHT: SECOND_Z_SCORE_PARAMS,
    THIRD_KEY_WITH_HEIGHT: THIRD_Z_SCORE_PARAMS
}
NORMALIZATION_TABLE_WITH_HEIGHT = pandas.DataFrame.from_dict(
    NORMALIZATION_DICT_WITH_HEIGHT, orient='index')
NORMALIZATION_TABLE_WITH_HEIGHT.rename(
    columns=COLUMN_DICT_OLD_TO_NEW, inplace=True)


def _compare_z_score_dicts(first_z_score_dict, second_z_score_dict):
    """Compares two dictionaries with z-score parameters.

    :param first_z_score_dict: First dictionary.
    :param second_z_score_dict: Second dictionary.
    :return: are_dicts_equal: Boolean flag.
    """

    first_keys = first_z_score_dict.keys()
    second_keys = second_z_score_dict.keys()
    if set(first_keys) != set(second_keys):
        return False

    for this_key in first_keys:
        if this_key == find_norm_params.MEAN_VALUE_KEY:
            if first_z_score_dict[this_key] != second_z_score_dict[this_key]:
                return False
        else:
            if not numpy.isclose(first_z_score_dict[this_key],
                                 second_z_score_dict[this_key], atol=TOLERANCE):
                return False

    return True


def _compare_frequency_dicts(first_frequency_dict, second_frequency_dict):
    """Compares two dictionaries with measurement frequencies.

    :param first_frequency_dict: First dictionary.
    :param second_frequency_dict: Second dictionary.
    :return: are_dicts_equal: Boolean flag.
    """

    first_keys, first_values = zip(*first_frequency_dict.iteritems())
    first_keys = numpy.array(first_keys)
    first_values = numpy.array(first_values, dtype=int)

    sort_indices = numpy.argsort(first_keys)
    first_keys = first_keys[sort_indices]
    first_values = first_values[sort_indices]

    second_keys, second_values = zip(*second_frequency_dict.iteritems())
    second_keys = numpy.array(second_keys)
    second_values = numpy.array(second_values, dtype=int)

    sort_indices = numpy.argsort(second_keys)
    second_keys = second_keys[sort_indices]
    second_values = second_values[sort_indices]

    if not numpy.array_equal(first_keys, second_keys):
        return False
    if not numpy.array_equal(first_values, second_values):
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

    def test_update_z_score_params(self):
        """Ensures correct output from _update_z_score_params."""

        this_new_param_dict = find_norm_params._update_z_score_params(
            z_score_param_dict=copy.deepcopy(ORIGINAL_Z_SCORE_DICT),
            new_data_matrix=NEW_MATRIX_FOR_Z_SCORES)

        self.assertTrue(_compare_z_score_dicts(
            this_new_param_dict, NEW_Z_SCORE_DICT))

    def test_update_frequency_dict(self):
        """Ensures correct output from _update_frequency_dict."""

        this_new_frequency_dict = find_norm_params._update_frequency_dict(
            frequency_dict=copy.deepcopy(MAIN_FREQUENCY_DICT),
            new_data_matrix=NEW_MATRIX_FOR_FREQUENCIES,
            rounding_base=ROUNDING_BASE)

        self.assertTrue(_compare_frequency_dicts(
            this_new_frequency_dict, NEW_FREQUENCY_DICT))

    def test_get_standard_deviation(self):
        """Ensures correct output from _get_standard_deviation."""

        this_z_score_param_dict = {
            find_norm_params.NUM_VALUES_KEY: STDEV_INPUT_MATRIX.size,
            find_norm_params.MEAN_VALUE_KEY: numpy.mean(STDEV_INPUT_MATRIX),
            find_norm_params.MEAN_OF_SQUARES_KEY:
                numpy.mean(STDEV_INPUT_MATRIX ** 2)
        }

        this_standard_deviation = find_norm_params._get_standard_deviation(
            this_z_score_param_dict)
        self.assertTrue(numpy.isclose(
            this_standard_deviation, STANDARD_DEVIATION, atol=TOLERANCE))

    def test_get_percentile_small(self):
        """Ensures correct output from _get_percentile.

        In this case, percentile level is small.
        """

        this_percentile = find_norm_params._get_percentile(
            frequency_dict=MAIN_FREQUENCY_DICT,
            percentile_level=SMALL_PERCENTILE_LEVEL)
        self.assertTrue(numpy.isclose(
            this_percentile, SMALL_PERCENTILE, atol=TOLERANCE))

    def test_get_percentile_medium(self):
        """Ensures correct output from _get_percentile.

        In this case, percentile level is medium.
        """

        this_percentile = find_norm_params._get_percentile(
            frequency_dict=MAIN_FREQUENCY_DICT,
            percentile_level=MEDIUM_PERCENTILE_LEVEL)
        self.assertTrue(numpy.isclose(
            this_percentile, MEDIUM_PERCENTILE, atol=TOLERANCE))

    def test_get_percentile_large(self):
        """Ensures correct output from _get_percentile.

        In this case, percentile level is large.
        """

        this_percentile = find_norm_params._get_percentile(
            frequency_dict=MAIN_FREQUENCY_DICT,
            percentile_level=LARGE_PERCENTILE_LEVEL)
        self.assertTrue(numpy.isclose(
            this_percentile, LARGE_PERCENTILE, atol=TOLERANCE))

    def test_convert_normalization_params_no_prctile(self):
        """Ensures correct output from _convert_normalization_params.

        In this case, the table should be single-indexed (field name only) and
        should *not* contain percentiles.
        """

        this_norm_table = find_norm_params._convert_normalization_params(
            z_score_dict_dict=Z_SCORE_DICT_DICT_NO_HEIGHT)

        self.assertTrue(_compare_normalization_tables(
            this_norm_table, NORMALIZATION_TABLE_NO_PRCTILE))

    def test_convert_normalization_params_with_prctile(self):
        """Ensures correct output from _convert_normalization_params.

        In this case, the table should be single-indexed (field name only) and
        contain percentiles.
        """

        this_norm_table = find_norm_params._convert_normalization_params(
            z_score_dict_dict=Z_SCORE_DICT_DICT_NO_HEIGHT,
            frequency_dict_dict=FREQUENCY_DICT_DICT,
            min_percentile_level=MIN_PERCENTILE_LEVEL,
            max_percentile_level=MAX_PERCENTILE_LEVEL)

        self.assertTrue(_compare_normalization_tables(
            this_norm_table, NORMALIZATION_TABLE_WITH_PRCTILE))

    def test_convert_normalization_params_with_height(self):
        """Ensures correct output from _convert_normalization_params.

        In this case, the table should be double-indexed (field name and
        height).
        """

        this_norm_table = find_norm_params._convert_normalization_params(
            z_score_dict_dict=Z_SCORE_DICT_DICT_WITH_HEIGHT)

        self.assertTrue(_compare_normalization_tables(
            this_norm_table, NORMALIZATION_TABLE_WITH_HEIGHT))


if __name__ == '__main__':
    unittest.main()
