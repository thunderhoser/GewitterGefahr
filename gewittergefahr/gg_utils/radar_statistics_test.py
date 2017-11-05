"""Unit tests for radar_statistics.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import radar_statistics as radar_stats

TOLERANCE = 1e-6
FAKE_STATISTIC_NAME = 'foo'
FAKE_PERCENTILE_LEVEL = -9999.

# The following constants are used to test
# _radar_field_and_statistic_to_column_name and
# _column_name_to_statistic_params.
REFLECTIVITY_FIELD_NAME = 'reflectivity_dbz'
NON_REFLECTIVITY_FIELD_NAME = 'vil_mm'
REFLECTIVITY_HEIGHT_M_AGL = 2000
STATISTIC_NAME = 'kurtosis'
COLUMN_NAME_FOR_REFLECTIVITY_STAT = 'reflectivity_dbz_2000m_kurtosis'
COLUMN_NAME_FOR_NON_REFLECTIVITY_STAT = 'vil_mm_kurtosis'

# The following constants are used to test
# _radar_field_and_percentile_to_column_name and
# _column_name_to_statistic_params.
PERCENTILE_LEVEL_ORIG = 75.12
PERCENTILE_LEVEL = 75.1
COLUMN_NAME_FOR_REFLECTIVITY_PRCTILE = 'reflectivity_dbz_2000m_percentile075.1'
COLUMN_NAME_FOR_NON_REFLECTIVITY_PRCTILE = 'vil_mm_percentile075.1'
INVALID_COLUMN_NAME = 'foo'

# The following constants are used to test extract_radar_grid_points.
RADAR_FIELD_MATRIX = numpy.array(
    [[numpy.nan, numpy.nan, 10., 20., 30., 40.],
     [numpy.nan, 5., 15., 25., 35., 50.],
     [5., 10., 25., 40., 55., 70.],
     [10., 30., 50., 70., 75., numpy.nan]])
ROW_INDICES_FOR_1D_ARRAY = numpy.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=int)
COLUMN_INDICES_FOR_1D_ARRAY = numpy.array([0, 5, 1, 4, 2, 3, 0, 5], dtype=int)
RADAR_FIELD_1D_ARRAY = numpy.array(
    [numpy.nan, 40., 5., 35., 25., 40., 10., numpy.nan])

# The following constants are used to test get_spatial_statistics.
RADAR_FIELD_FOR_STATS = numpy.array([[numpy.nan, 0., 20.],
                                     [20., 50., 60.]])
STATISTIC_NAMES = [
    radar_stats.AVERAGE_NAME, radar_stats.STANDARD_DEVIATION_NAME,
    radar_stats.SKEWNESS_NAME, radar_stats.KURTOSIS_NAME]
STATISTIC_VALUES = numpy.array([30., 24.494897, 0.170103, -1.75])
PERCENTILE_LEVELS = numpy.array([0., 5., 25., 50., 75., 95., 100.])
PERCENTILE_VALUES = numpy.array([0., 4., 20., 20., 50., 58., 60.])


class RadarStatisticsTests(unittest.TestCase):
    """Each method is a unit test for radar_statistics.py."""

    def test_radar_field_and_statistic_to_column_name_reflectivity(self):
        """Ensures correctness of _radar_field_and_statistic_to_column_name.

        In this case, radar field is reflectivity.
        """

        this_column_name = (
            radar_stats._radar_field_and_statistic_to_column_name(
                radar_field_name=REFLECTIVITY_FIELD_NAME,
                radar_height_m_agl=REFLECTIVITY_HEIGHT_M_AGL,
                statistic_name=STATISTIC_NAME))
        self.assertTrue(this_column_name == COLUMN_NAME_FOR_REFLECTIVITY_STAT)

    def test_radar_field_and_statistic_to_column_name_non_reflectivity(self):
        """Ensures correctness of _radar_field_and_statistic_to_column_name.

        In this case, radar field is not reflectivity.
        """

        this_column_name = (
            radar_stats._radar_field_and_statistic_to_column_name(
                radar_field_name=NON_REFLECTIVITY_FIELD_NAME,
                statistic_name=STATISTIC_NAME))
        self.assertTrue(
            this_column_name == COLUMN_NAME_FOR_NON_REFLECTIVITY_STAT)

    def test_radar_field_and_percentile_to_column_name_reflectivity(self):
        """Ensures correctness of _radar_field_and_percentile_to_column_name.

        In this case, radar field is reflectivity.
        """

        this_column_name = (
            radar_stats._radar_field_and_percentile_to_column_name(
                radar_field_name=REFLECTIVITY_FIELD_NAME,
                radar_height_m_agl=REFLECTIVITY_HEIGHT_M_AGL,
                percentile_level=PERCENTILE_LEVEL_ORIG))
        self.assertTrue(
            this_column_name == COLUMN_NAME_FOR_REFLECTIVITY_PRCTILE)

    def test_radar_field_and_percentile_to_column_name_non_reflectivity(self):
        """Ensures correctness of _radar_field_and_percentile_to_column_name.

        In this case, radar field is not reflectivity.
        """

        this_column_name = (
            radar_stats._radar_field_and_percentile_to_column_name(
                radar_field_name=NON_REFLECTIVITY_FIELD_NAME,
                percentile_level=PERCENTILE_LEVEL_ORIG))
        self.assertTrue(
            this_column_name == COLUMN_NAME_FOR_NON_REFLECTIVITY_PRCTILE)

    def test_column_name_to_statistic_params_prctile_no_refl_yes(self):
        """Ensures correct output from _column_name_to_statistic_params.

        In this case, statistic is non-percentile and radar field is
        reflectivity.
        """

        this_parameter_dict = radar_stats._column_name_to_statistic_params(
            COLUMN_NAME_FOR_REFLECTIVITY_STAT)

        self.assertTrue(this_parameter_dict[radar_stats.RADAR_FIELD_NAME_KEY] ==
                        REFLECTIVITY_FIELD_NAME)
        self.assertTrue(this_parameter_dict[radar_stats.RADAR_HEIGHT_KEY] ==
                        REFLECTIVITY_HEIGHT_M_AGL)
        self.assertTrue(this_parameter_dict[radar_stats.STATISTIC_NAME_KEY] ==
                        STATISTIC_NAME)
        self.assertTrue(
            this_parameter_dict[radar_stats.PERCENTILE_LEVEL_KEY] is None)

    def test_column_name_to_statistic_params_prctile_no_refl_no(self):
        """Ensures correct output from _column_name_to_statistic_params.

        In this case, statistic is non-percentile and radar field is
        non-reflectivity.
        """

        this_parameter_dict = radar_stats._column_name_to_statistic_params(
            COLUMN_NAME_FOR_NON_REFLECTIVITY_STAT)

        self.assertTrue(this_parameter_dict[radar_stats.RADAR_FIELD_NAME_KEY] ==
                        NON_REFLECTIVITY_FIELD_NAME)
        self.assertTrue(
            this_parameter_dict[radar_stats.RADAR_HEIGHT_KEY] is None)
        self.assertTrue(this_parameter_dict[radar_stats.STATISTIC_NAME_KEY] ==
                        STATISTIC_NAME)
        self.assertTrue(
            this_parameter_dict[radar_stats.PERCENTILE_LEVEL_KEY] is None)

    def test_column_name_to_statistic_params_prctile_yes_refl_yes(self):
        """Ensures correct output from _column_name_to_statistic_params.

        In this case, statistic is percentile and radar field is reflectivity.
        """

        this_parameter_dict = radar_stats._column_name_to_statistic_params(
            COLUMN_NAME_FOR_REFLECTIVITY_PRCTILE)

        self.assertTrue(this_parameter_dict[radar_stats.RADAR_FIELD_NAME_KEY] ==
                        REFLECTIVITY_FIELD_NAME)
        self.assertTrue(this_parameter_dict[radar_stats.RADAR_HEIGHT_KEY] ==
                        REFLECTIVITY_HEIGHT_M_AGL)
        self.assertTrue(
            this_parameter_dict[radar_stats.STATISTIC_NAME_KEY] is None)
        self.assertTrue(this_parameter_dict[radar_stats.PERCENTILE_LEVEL_KEY] ==
                        PERCENTILE_LEVEL)

    def test_column_name_to_statistic_params_prctile_yes_refl_no(self):
        """Ensures correct output from _column_name_to_statistic_params.

        In this case, statistic is percentile and radar field is non-
        reflectivity.
        """

        this_parameter_dict = radar_stats._column_name_to_statistic_params(
            COLUMN_NAME_FOR_NON_REFLECTIVITY_PRCTILE)

        self.assertTrue(this_parameter_dict[radar_stats.RADAR_FIELD_NAME_KEY] ==
                        NON_REFLECTIVITY_FIELD_NAME)
        self.assertTrue(
            this_parameter_dict[radar_stats.RADAR_HEIGHT_KEY] is None)
        self.assertTrue(
            this_parameter_dict[radar_stats.STATISTIC_NAME_KEY] is None)
        self.assertTrue(this_parameter_dict[radar_stats.PERCENTILE_LEVEL_KEY] ==
                        PERCENTILE_LEVEL)

    def test_column_name_to_statistic_params_invalid(self):
        """Ensures correct output from _column_name_to_statistic_params.

        In this case, column name is invalid (does not correspond to a radar
        statistic).
        """

        this_parameter_dict = radar_stats._column_name_to_statistic_params(
            INVALID_COLUMN_NAME)
        self.assertTrue(this_parameter_dict is None)

    def test_check_statistic_params_all_good(self):
        """Ensures correct output from _check_statistic_params.

        In this case, all inputs are valid.
        """

        radar_stats._check_statistic_params(
            radar_stats.STATISTIC_NAMES, radar_stats.DEFAULT_PERCENTILE_LEVELS)

    def test_check_statistic_params_bad_string(self):
        """Ensures correct output from _check_statistic_params.

        In this case, one statistic name is invalid.
        """

        with self.assertRaises(ValueError):
            radar_stats._check_statistic_params(
                radar_stats.STATISTIC_NAMES + [FAKE_STATISTIC_NAME],
                radar_stats.DEFAULT_PERCENTILE_LEVELS)

    def test_check_statistic_params_bad_percentile(self):
        """Ensures correct output from _check_statistic_params.

        In this case, one percentile level is invalid.
        """

        these_percentile_levels = numpy.concatenate((
            radar_stats.DEFAULT_PERCENTILE_LEVELS,
            numpy.array([FAKE_PERCENTILE_LEVEL])))

        with self.assertRaises(ValueError):
            radar_stats._check_statistic_params(
                radar_stats.STATISTIC_NAMES, these_percentile_levels)

    def test_extract_radar_grid_points(self):
        """Ensures correct output from extract_radar_grid_points."""

        this_field_1d_array = radar_stats.extract_radar_grid_points(
            RADAR_FIELD_MATRIX, row_indices=ROW_INDICES_FOR_1D_ARRAY,
            column_indices=COLUMN_INDICES_FOR_1D_ARRAY)

        self.assertTrue(numpy.allclose(
            this_field_1d_array, RADAR_FIELD_1D_ARRAY, equal_nan=True,
            atol=TOLERANCE))

    def test_get_spatial_statistics(self):
        """Ensures correct output from get_spatial_statistics."""

        these_statistic_values, these_percentile_values = (
            radar_stats.get_spatial_statistics(
                RADAR_FIELD_FOR_STATS, statistic_names=STATISTIC_NAMES,
                percentile_levels=PERCENTILE_LEVELS))

        self.assertTrue(numpy.allclose(
            these_statistic_values, STATISTIC_VALUES, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_percentile_values, PERCENTILE_VALUES, atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
