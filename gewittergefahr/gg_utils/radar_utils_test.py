"""Unit tests for radar_utils.py."""

import unittest
import numpy
from gewittergefahr.gg_io import radar_io
from gewittergefahr.gg_utils import radar_utils

TOLERANCE = 1e-6

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
    radar_utils.MIN_ROW_IN_SUBGRID_COLUMN: 0,
    radar_utils.MAX_ROW_IN_SUBGRID_COLUMN: 26,
    radar_utils.MIN_COLUMN_IN_SUBGRID_COLUMN: 0,
    radar_utils.MAX_COLUMN_IN_SUBGRID_COLUMN: 42,
    radar_utils.NUM_PADDED_ROWS_AT_START_COLUMN: 5,
    radar_utils.NUM_PADDED_ROWS_AT_END_COLUMN: 0,
    radar_utils.NUM_PADDED_COLUMNS_AT_START_COLUMN: 21,
    radar_utils.NUM_PADDED_COLUMNS_AT_END_COLUMN: 0
}

SUBGRID_DICT_MIDDLE = {
    radar_utils.MIN_ROW_IN_SUBGRID_COLUMN: 985,
    radar_utils.MAX_ROW_IN_SUBGRID_COLUMN: 1016,
    radar_utils.MIN_COLUMN_IN_SUBGRID_COLUMN: 969,
    radar_utils.MAX_COLUMN_IN_SUBGRID_COLUMN: 1032,
    radar_utils.NUM_PADDED_ROWS_AT_START_COLUMN: 0,
    radar_utils.NUM_PADDED_ROWS_AT_END_COLUMN: 0,
    radar_utils.NUM_PADDED_COLUMNS_AT_START_COLUMN: 0,
    radar_utils.NUM_PADDED_COLUMNS_AT_END_COLUMN: 0
}

SUBGRID_DICT_BOTTOM_RIGHT = {
    radar_utils.MIN_ROW_IN_SUBGRID_COLUMN: 3475,
    radar_utils.MAX_ROW_IN_SUBGRID_COLUMN: 3500,
    radar_utils.MIN_COLUMN_IN_SUBGRID_COLUMN: 6959,
    radar_utils.MAX_COLUMN_IN_SUBGRID_COLUMN: 7000,
    radar_utils.NUM_PADDED_ROWS_AT_START_COLUMN: 0,
    radar_utils.NUM_PADDED_ROWS_AT_END_COLUMN: 6,
    radar_utils.NUM_PADDED_COLUMNS_AT_START_COLUMN: 0,
    radar_utils.NUM_PADDED_COLUMNS_AT_END_COLUMN: 22
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
STRING_STATS = [radar_utils.MEAN_STRING, radar_utils.STDEV_STRING,
                radar_utils.SKEWNESS_STRING, radar_utils.KURTOSIS_STRING]
STRING_STAT_VALUES = numpy.array([30., 24.494897, 0.170103, -1.75])
PERCENTILE_LEVELS = numpy.array([0., 5., 25., 50., 75., 95., 100.])
PERCENTILE_VALUES = numpy.array([0., 4., 20., 20., 50., 58., 60.])


class RadarUtilsTests(unittest.TestCase):
    """Each method is a unit test for radar_utils.py."""

    def test_center_points_latlng_to_rowcol(self):
        """Ensures correct output from _center_points_latlng_to_rowcol."""

        these_center_row_indices, these_center_column_indices = (
            radar_utils._center_points_latlng_to_rowcol(
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

        this_subgrid_dict = radar_utils._get_rowcol_indices_for_subgrid(
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

        this_subgrid_dict = radar_utils._get_rowcol_indices_for_subgrid(
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

        this_subgrid_dict = radar_utils._get_rowcol_indices_for_subgrid(
            num_rows_in_full_grid=NUM_GRID_ROWS,
            num_columns_in_full_grid=NUM_GRID_COLUMNS,
            center_row_index=CENTER_ROW_INDEX_BOTTOM_RIGHT,
            center_column_index=CENTER_COLUMN_INDEX_BOTTOM_RIGHT,
            num_rows_in_subgrid=NUM_ROWS_IN_SUBGRID,
            num_columns_in_subgrid=NUM_COLUMNS_IN_SUBGRID)

        self.assertTrue(this_subgrid_dict == SUBGRID_DICT_BOTTOM_RIGHT)

    def test_are_grids_equal_true(self):
        """Ensures correct output from _are_grids_equal when answer is yes."""

        self.assertTrue(radar_utils._are_grids_equal(
            GRID_METADATA_DICT_MYRORSS, GRID_METADATA_DICT_MYRORSS))

    def test_are_grids_equal_false(self):
        """Ensures correct output from _are_grids_equal when answer is no."""

        self.assertFalse(radar_utils._are_grids_equal(
            GRID_METADATA_DICT_MYRORSS, GRID_METADATA_DICT_MRMS_SHEAR))

    def test_check_stats_to_compute_all_good(self):
        """Ensures correct output from _check_stats_to_compute.

        In this case, inputs are valid.
        """

        radar_utils._check_stats_to_compute(
            radar_utils.VALID_STRING_STATS,
            radar_utils.DEFAULT_PERCENTILE_LEVELS)

    def test_check_stats_to_compute_bad_string(self):
        """Ensures correct output from _check_stats_to_compute.

        In this case, one string statistic is invalid.
        """

        these_string_stats = radar_utils.VALID_STRING_STATS + ['foo']

        with self.assertRaises(ValueError):
            radar_utils._check_stats_to_compute(
                these_string_stats, radar_utils.DEFAULT_PERCENTILE_LEVELS)

    def test_check_stats_to_compute_bad_pct_level(self):
        """Ensures correct output from _check_stats_to_compute.

        In this case, one percentile level is invalid.
        """

        these_percentile_levels = numpy.concatenate((
            radar_utils.DEFAULT_PERCENTILE_LEVELS, numpy.array([-9999.])))

        with self.assertRaises(ValueError):
            radar_utils._check_stats_to_compute(
                radar_utils.VALID_STRING_STATS, these_percentile_levels)

    def test_extract_points_as_1d_array(self):
        """Ensures correct output from extract_points_as_1d_array."""

        this_field_1d_array = radar_utils.extract_points_as_1d_array(
            RADAR_FIELD_MATRIX, row_indices=ROW_INDICES_FOR_1D_ARRAY,
            column_indices=COLUMN_INDICES_FOR_1D_ARRAY)

        self.assertTrue(numpy.allclose(
            this_field_1d_array, RADAR_FIELD_1D_ARRAY, equal_nan=True,
            atol=TOLERANCE))

    def test_extract_points_as_2d_array(self):
        """Ensures correct output from extract_points_as_2d_array."""

        this_submatrix = radar_utils.extract_points_as_2d_array(
            RADAR_FIELD_MATRIX, center_row_index=CENTER_ROW_INDEX_TO_EXTRACT,
            center_column_index=CENTER_COLUMN_INDEX_TO_EXTRACT,
            num_rows_in_subgrid=NUM_ROWS_TO_EXTRACT,
            num_columns_in_subgrid=NUM_COLUMNS_TO_EXTRACT)

        self.assertTrue(numpy.allclose(
            this_submatrix, RADAR_FIELD_SUBMATRIX, atol=TOLERANCE))

    def test_get_spatial_statistics(self):
        """Ensures correct output from get_spatial_statistics."""

        these_string_stat_values, these_percentile_values = (
            radar_utils.get_spatial_statistics(
                RADAR_FIELD_FOR_STATS, string_statistics=STRING_STATS,
                percentile_levels=PERCENTILE_LEVELS))

        self.assertTrue(numpy.allclose(
            these_string_stat_values, STRING_STAT_VALUES, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_percentile_values, PERCENTILE_VALUES, atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
