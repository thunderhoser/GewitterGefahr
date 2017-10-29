"""Unit tests for radar_statistics.py."""

import unittest
import numpy
from gewittergefahr.gg_io import radar_io
from gewittergefahr.gg_utils import radar_statistics as radar_stats

TOLERANCE = 1e-6
FAKE_STATISTIC_NAME = 'foo'
FAKE_PERCENTILE_LEVEL = -9999.

# The following constants are used to test
# _radar_field_and_statistic_to_column_name.
REFLECTIVITY_FIELD_NAME = 'reflectivity_dbz'
NON_REFLECTIVITY_FIELD_NAME = 'vil_mm'
REFLECTIVITY_HEIGHT_M_AGL = 2000
STATISTIC_NAME = 'kurtosis'
COLUMN_NAME_FOR_REFLECTIVITY_STAT = 'reflectivity_dbz_2000m_kurtosis'
COLUMN_NAME_FOR_NON_REFLECTIVITY_STAT = 'vil_mm_kurtosis'

PERCENTILE_LEVEL = 75.12
COLUMN_NAME_FOR_REFLECTIVITY_PRCTILE = 'reflectivity_dbz_2000m_percentile075.1'
COLUMN_NAME_FOR_NON_REFLECTIVITY_PRCTILE = 'vil_mm_percentile075.1'

# The following constants are used to test _center_points_latlng_to_rowcol.
NW_GRID_POINT_LAT_DEG = 55.
NW_GRID_POINT_LNG_DEG = 230.
LAT_SPACING_DEG = 0.01
LNG_SPACING_DEG = 0.01
NUM_GRID_ROWS = 3501
NUM_GRID_COLUMNS = 7001

CENTER_LATITUDES_DEG = numpy.array([55., 54.995, 54.99, 54.985, 54.98])
CENTER_LONGITUDES_DEG = numpy.array([230., 230.005, 230.01, 230.015, 230.02])
CENTER_ROW_INDICES = numpy.array([-0.5, 0.5, 1.5, 1.5, 1.5])
CENTER_COLUMN_INDICES = numpy.array([-0.5, 0.5, 1.5, 1.5, 1.5])

# The following constants are used to test _get_rowcol_indices_for_subgrid.
CENTER_ROW_INDEX_TOP_LEFT = 10.5
CENTER_ROW_INDEX_MIDDLE = 1000.5
CENTER_ROW_INDEX_BOTTOM_RIGHT = 3490.5
CENTER_COLUMN_INDEX_TOP_LEFT = 10.5
CENTER_COLUMN_INDEX_MIDDLE = 1000.5
CENTER_COLUMN_INDEX_BOTTOM_RIGHT = 6990.5
NUM_ROWS_IN_SUBGRID = 32
NUM_COLUMNS_IN_SUBGRID = 64

SUBGRID_DICT_TOP_LEFT = {
    radar_stats.MIN_ROW_IN_SUBGRID_COLUMN: 0,
    radar_stats.MAX_ROW_IN_SUBGRID_COLUMN: 26,
    radar_stats.MIN_COLUMN_IN_SUBGRID_COLUMN: 0,
    radar_stats.MAX_COLUMN_IN_SUBGRID_COLUMN: 42,
    radar_stats.NUM_PADDED_ROWS_AT_START_COLUMN: 5,
    radar_stats.NUM_PADDED_ROWS_AT_END_COLUMN: 0,
    radar_stats.NUM_PADDED_COLUMNS_AT_START_COLUMN: 21,
    radar_stats.NUM_PADDED_COLUMNS_AT_END_COLUMN: 0
}

SUBGRID_DICT_MIDDLE = {
    radar_stats.MIN_ROW_IN_SUBGRID_COLUMN: 985,
    radar_stats.MAX_ROW_IN_SUBGRID_COLUMN: 1016,
    radar_stats.MIN_COLUMN_IN_SUBGRID_COLUMN: 969,
    radar_stats.MAX_COLUMN_IN_SUBGRID_COLUMN: 1032,
    radar_stats.NUM_PADDED_ROWS_AT_START_COLUMN: 0,
    radar_stats.NUM_PADDED_ROWS_AT_END_COLUMN: 0,
    radar_stats.NUM_PADDED_COLUMNS_AT_START_COLUMN: 0,
    radar_stats.NUM_PADDED_COLUMNS_AT_END_COLUMN: 0
}

SUBGRID_DICT_BOTTOM_RIGHT = {
    radar_stats.MIN_ROW_IN_SUBGRID_COLUMN: 3475,
    radar_stats.MAX_ROW_IN_SUBGRID_COLUMN: 3500,
    radar_stats.MIN_COLUMN_IN_SUBGRID_COLUMN: 6959,
    radar_stats.MAX_COLUMN_IN_SUBGRID_COLUMN: 7000,
    radar_stats.NUM_PADDED_ROWS_AT_START_COLUMN: 0,
    radar_stats.NUM_PADDED_ROWS_AT_END_COLUMN: 6,
    radar_stats.NUM_PADDED_COLUMNS_AT_START_COLUMN: 0,
    radar_stats.NUM_PADDED_COLUMNS_AT_END_COLUMN: 22
}

# The following constants are used to test _are_grids_equal.
GRID_METADATA_DICT_MYRORSS = {
    radar_io.NW_GRID_POINT_LAT_COLUMN: 55.,
    radar_io.NW_GRID_POINT_LNG_COLUMN: 230.,
    radar_io.LAT_SPACING_COLUMN: 0.01, radar_io.LNG_SPACING_COLUMN: 0.01,
    radar_io.NUM_LAT_COLUMN: 3501, radar_io.NUM_LNG_COLUMN: 7001
}
GRID_METADATA_DICT_MRMS_SHEAR = {
    radar_io.NW_GRID_POINT_LAT_COLUMN: 51.,
    radar_io.NW_GRID_POINT_LNG_COLUMN: 233.,
    radar_io.LAT_SPACING_COLUMN: 0.005, radar_io.LNG_SPACING_COLUMN: 0.005,
    radar_io.NUM_LAT_COLUMN: 6000, radar_io.NUM_LNG_COLUMN: 12400
}

# The following constants are used to test extract_points_as_1d_array.
RADAR_FIELD_MATRIX = numpy.array(
    [[numpy.nan, numpy.nan, 10., 20., 30., 40.],
     [numpy.nan, 5., 15., 25., 35., 50.],
     [5., 10., 25., 40., 55., 70.],
     [10., 30., 50., 70., 75., numpy.nan]])
ROW_INDICES_FOR_1D_ARRAY = numpy.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=int)
COLUMN_INDICES_FOR_1D_ARRAY = numpy.array([0, 5, 1, 4, 2, 3, 0, 5], dtype=int)
RADAR_FIELD_1D_ARRAY = numpy.array(
    [numpy.nan, 40., 5., 35., 25., 40., 10., numpy.nan])

# The following constants are used to test extract_points_as_2d_array.
CENTER_ROW_INDEX_TO_EXTRACT = 1.5
CENTER_COLUMN_INDEX_TO_EXTRACT = 2.5
NUM_ROWS_TO_EXTRACT = 2
NUM_COLUMNS_TO_EXTRACT = 4
RADAR_FIELD_SUBMATRIX = numpy.array([[5., 15., 25., 35.],
                                     [10., 25., 40., 55.]])

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
                percentile_level=PERCENTILE_LEVEL))
        self.assertTrue(
            this_column_name == COLUMN_NAME_FOR_REFLECTIVITY_PRCTILE)

    def test_radar_field_and_percentile_to_column_name_non_reflectivity(self):
        """Ensures correctness of _radar_field_and_percentile_to_column_name.

        In this case, radar field is not reflectivity.
        """

        this_column_name = (
            radar_stats._radar_field_and_percentile_to_column_name(
                radar_field_name=NON_REFLECTIVITY_FIELD_NAME,
                percentile_level=PERCENTILE_LEVEL))
        self.assertTrue(
            this_column_name == COLUMN_NAME_FOR_NON_REFLECTIVITY_PRCTILE)

    def test_center_points_latlng_to_rowcol(self):
        """Ensures correct output from _center_points_latlng_to_rowcol."""

        these_center_row_indices, these_center_column_indices = (
            radar_stats._center_points_latlng_to_rowcol(
                CENTER_LATITUDES_DEG, CENTER_LONGITUDES_DEG,
                nw_grid_point_lat_deg=NW_GRID_POINT_LAT_DEG,
                nw_grid_point_lng_deg=NW_GRID_POINT_LNG_DEG,
                lat_spacing_deg=LAT_SPACING_DEG,
                lng_spacing_deg=LNG_SPACING_DEG))

        self.assertTrue(numpy.array_equal(
            these_center_row_indices, CENTER_ROW_INDICES))
        self.assertTrue(numpy.array_equal(
            these_center_column_indices, CENTER_COLUMN_INDICES))

    def test_get_rowcol_indices_for_subgrid_top_left(self):
        """Ensures correct output from _get_rowcol_indices_for_subgrid.

        In this case, center point is near the top left of the full grid.
        """

        this_subgrid_dict = radar_stats._get_rowcol_indices_for_subgrid(
            num_rows_in_full_grid=NUM_GRID_ROWS,
            num_columns_in_full_grid=NUM_GRID_COLUMNS,
            center_row_index=CENTER_ROW_INDEX_TOP_LEFT,
            center_column_index=CENTER_COLUMN_INDEX_TOP_LEFT,
            num_rows_in_subgrid=NUM_ROWS_IN_SUBGRID,
            num_columns_in_subgrid=NUM_COLUMNS_IN_SUBGRID)

        self.assertTrue(this_subgrid_dict == SUBGRID_DICT_TOP_LEFT)

    def test_get_rowcol_indices_for_subgrid_middle(self):
        """Ensures correct output from _get_rowcol_indices_for_subgrid.

        In this case, center point is in the middle of the full grid.
        """

        this_subgrid_dict = radar_stats._get_rowcol_indices_for_subgrid(
            num_rows_in_full_grid=NUM_GRID_ROWS,
            num_columns_in_full_grid=NUM_GRID_COLUMNS,
            center_row_index=CENTER_ROW_INDEX_MIDDLE,
            center_column_index=CENTER_COLUMN_INDEX_MIDDLE,
            num_rows_in_subgrid=NUM_ROWS_IN_SUBGRID,
            num_columns_in_subgrid=NUM_COLUMNS_IN_SUBGRID)

        self.assertTrue(this_subgrid_dict == SUBGRID_DICT_MIDDLE)

    def test_get_rowcol_indices_for_subgrid_bottom_right(self):
        """Ensures correct output from _get_rowcol_indices_for_subgrid.

        In this case, center point is near the bottom right of the full grid.
        """

        this_subgrid_dict = radar_stats._get_rowcol_indices_for_subgrid(
            num_rows_in_full_grid=NUM_GRID_ROWS,
            num_columns_in_full_grid=NUM_GRID_COLUMNS,
            center_row_index=CENTER_ROW_INDEX_BOTTOM_RIGHT,
            center_column_index=CENTER_COLUMN_INDEX_BOTTOM_RIGHT,
            num_rows_in_subgrid=NUM_ROWS_IN_SUBGRID,
            num_columns_in_subgrid=NUM_COLUMNS_IN_SUBGRID)

        self.assertTrue(this_subgrid_dict == SUBGRID_DICT_BOTTOM_RIGHT)

    def test_are_grids_equal_true(self):
        """Ensures correct output from _are_grids_equal when answer is yes."""

        self.assertTrue(radar_stats._are_grids_equal(
            GRID_METADATA_DICT_MYRORSS, GRID_METADATA_DICT_MYRORSS))

    def test_are_grids_equal_false(self):
        """Ensures correct output from _are_grids_equal when answer is no."""

        self.assertFalse(radar_stats._are_grids_equal(
            GRID_METADATA_DICT_MYRORSS, GRID_METADATA_DICT_MRMS_SHEAR))

    def test_check_statistic_names_all_good(self):
        """Ensures correct output from _check_statistic_names.

        In this case, all inputs are valid.
        """

        radar_stats._check_statistic_names(
            radar_stats.STATISTIC_NAMES, radar_stats.DEFAULT_PERCENTILE_LEVELS)

    def test_check_statistic_names_bad_string(self):
        """Ensures correct output from _check_statistic_names.

        In this case, one statistic name is invalid.
        """

        with self.assertRaises(ValueError):
            radar_stats._check_statistic_names(
                radar_stats.STATISTIC_NAMES + [FAKE_STATISTIC_NAME],
                radar_stats.DEFAULT_PERCENTILE_LEVELS)

    def test_check_statistic_names_bad_percentile(self):
        """Ensures correct output from _check_statistic_names.

        In this case, one percentile level is invalid.
        """

        these_percentile_levels = numpy.concatenate((
            radar_stats.DEFAULT_PERCENTILE_LEVELS,
            numpy.array([FAKE_PERCENTILE_LEVEL])))

        with self.assertRaises(ValueError):
            radar_stats._check_statistic_names(
                radar_stats.STATISTIC_NAMES, these_percentile_levels)

    def test_extract_points_as_1d_array(self):
        """Ensures correct output from extract_points_as_1d_array."""

        this_field_1d_array = radar_stats.extract_points_as_1d_array(
            RADAR_FIELD_MATRIX, row_indices=ROW_INDICES_FOR_1D_ARRAY,
            column_indices=COLUMN_INDICES_FOR_1D_ARRAY)

        self.assertTrue(numpy.allclose(
            this_field_1d_array, RADAR_FIELD_1D_ARRAY, equal_nan=True,
            atol=TOLERANCE))

    def test_extract_points_as_2d_array(self):
        """Ensures correct output from extract_points_as_2d_array."""

        this_submatrix = radar_stats.extract_points_as_2d_array(
            RADAR_FIELD_MATRIX, center_row_index=CENTER_ROW_INDEX_TO_EXTRACT,
            center_column_index=CENTER_COLUMN_INDEX_TO_EXTRACT,
            num_rows_in_subgrid=NUM_ROWS_TO_EXTRACT,
            num_columns_in_subgrid=NUM_COLUMNS_TO_EXTRACT)

        self.assertTrue(numpy.allclose(
            this_submatrix, RADAR_FIELD_SUBMATRIX, atol=TOLERANCE))

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
