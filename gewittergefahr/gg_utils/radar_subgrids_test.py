"""Unit tests for radar_subgrids.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import radar_subgrids

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
    radar_subgrids.MIN_ROW_IN_SUBGRID_COLUMN: 0,
    radar_subgrids.MAX_ROW_IN_SUBGRID_COLUMN: 26,
    radar_subgrids.MIN_COLUMN_IN_SUBGRID_COLUMN: 0,
    radar_subgrids.MAX_COLUMN_IN_SUBGRID_COLUMN: 42,
    radar_subgrids.NUM_PADDED_ROWS_AT_START_COLUMN: 5,
    radar_subgrids.NUM_PADDED_ROWS_AT_END_COLUMN: 0,
    radar_subgrids.NUM_PADDED_COLUMNS_AT_START_COLUMN: 21,
    radar_subgrids.NUM_PADDED_COLUMNS_AT_END_COLUMN: 0
}

SUBGRID_DICT_MIDDLE = {
    radar_subgrids.MIN_ROW_IN_SUBGRID_COLUMN: 985,
    radar_subgrids.MAX_ROW_IN_SUBGRID_COLUMN: 1016,
    radar_subgrids.MIN_COLUMN_IN_SUBGRID_COLUMN: 969,
    radar_subgrids.MAX_COLUMN_IN_SUBGRID_COLUMN: 1032,
    radar_subgrids.NUM_PADDED_ROWS_AT_START_COLUMN: 0,
    radar_subgrids.NUM_PADDED_ROWS_AT_END_COLUMN: 0,
    radar_subgrids.NUM_PADDED_COLUMNS_AT_START_COLUMN: 0,
    radar_subgrids.NUM_PADDED_COLUMNS_AT_END_COLUMN: 0
}

SUBGRID_DICT_BOTTOM_RIGHT = {
    radar_subgrids.MIN_ROW_IN_SUBGRID_COLUMN: 3475,
    radar_subgrids.MAX_ROW_IN_SUBGRID_COLUMN: 3500,
    radar_subgrids.MIN_COLUMN_IN_SUBGRID_COLUMN: 6959,
    radar_subgrids.MAX_COLUMN_IN_SUBGRID_COLUMN: 7000,
    radar_subgrids.NUM_PADDED_ROWS_AT_START_COLUMN: 0,
    radar_subgrids.NUM_PADDED_ROWS_AT_END_COLUMN: 6,
    radar_subgrids.NUM_PADDED_COLUMNS_AT_START_COLUMN: 0,
    radar_subgrids.NUM_PADDED_COLUMNS_AT_END_COLUMN: 22
}

# The following constants are used to test extract_radar_subgrid.
RADAR_FIELD_MATRIX = numpy.array(
    [[numpy.nan, numpy.nan, 10., 20., 30., 40.],
     [numpy.nan, 5., 15., 25., 35., 50.],
     [5., 10., 25., 40., 55., 70.],
     [10., 30., 50., 70., 75., numpy.nan]])

CENTER_ROW_INDEX_TO_EXTRACT = 1.5
CENTER_COLUMN_INDEX_TO_EXTRACT = 2.5
NUM_ROWS_TO_EXTRACT = 2
NUM_COLUMNS_TO_EXTRACT = 4
RADAR_FIELD_SUBMATRIX = numpy.array([[5., 15., 25., 35.],
                                     [10., 25., 40., 55.]])

# The following constants are used to test find_storm_image_file.
TOP_STORM_IMAGE_DIR_NAME = 'storm_images'
VALID_TIME_UNIX_SEC = 1516749825
SPC_DATE_STRING = '20180123'
RADAR_FIELD_NAME = 'echo_top_40dbz_km'
RADAR_HEIGHT_M_ASL = 250
STORM_IMAGE_FILE_NAME = (
    'storm_images/2018/20180123/EchoTop_40/00.25/storm_images_'
    '2018-01-23-232345.p')


class RadarSubgridsTests(unittest.TestCase):
    """Each method is a unit test for radar_subgrids.py."""

    def test_center_points_latlng_to_rowcol(self):
        """Ensures correct output from _center_points_latlng_to_rowcol."""

        these_center_row_indices, these_center_column_indices = (
            radar_subgrids._center_points_latlng_to_rowcol(
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

        this_subgrid_dict = radar_subgrids._get_rowcol_indices_for_subgrid(
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

        this_subgrid_dict = radar_subgrids._get_rowcol_indices_for_subgrid(
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

        this_subgrid_dict = radar_subgrids._get_rowcol_indices_for_subgrid(
            num_rows_in_full_grid=NUM_GRID_ROWS,
            num_columns_in_full_grid=NUM_GRID_COLUMNS,
            center_row_index=CENTER_ROW_INDEX_BOTTOM_RIGHT,
            center_column_index=CENTER_COLUMN_INDEX_BOTTOM_RIGHT,
            num_rows_in_subgrid=NUM_ROWS_IN_SUBGRID,
            num_columns_in_subgrid=NUM_COLUMNS_IN_SUBGRID)

        self.assertTrue(this_subgrid_dict == SUBGRID_DICT_BOTTOM_RIGHT)

    def test_extract_points_as_2d_array(self):
        """Ensures correct output from extract_points_as_2d_array."""

        this_submatrix = radar_subgrids.extract_radar_subgrid(
            RADAR_FIELD_MATRIX, center_row_index=CENTER_ROW_INDEX_TO_EXTRACT,
            center_column_index=CENTER_COLUMN_INDEX_TO_EXTRACT,
            num_rows_in_subgrid=NUM_ROWS_TO_EXTRACT,
            num_columns_in_subgrid=NUM_COLUMNS_TO_EXTRACT)

        self.assertTrue(numpy.allclose(
            this_submatrix, RADAR_FIELD_SUBMATRIX, atol=TOLERANCE))

    def test_find_storm_image_file(self):
        """Ensures correct output from find_storm_image_file."""

        this_file_name = radar_subgrids.find_storm_image_file(
            top_directory_name=TOP_STORM_IMAGE_DIR_NAME,
            unix_time_sec=VALID_TIME_UNIX_SEC, spc_date_string=SPC_DATE_STRING,
            radar_field_name=RADAR_FIELD_NAME,
            radar_height_m_asl=RADAR_HEIGHT_M_ASL, raise_error_if_missing=False)

        self.assertTrue(this_file_name == STORM_IMAGE_FILE_NAME)


if __name__ == '__main__':
    unittest.main()
