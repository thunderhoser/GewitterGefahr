"""Unit tests for myrorss_io.py."""

import copy
import unittest
import numpy
import pandas
from gewittergefahr.gg_io import myrorss_io

TOLERANCE = 1e-6

RADAR_VAR_NAME_ORIG = 'MergedReflectivityQCComposite'
RADAR_VAR_NAME = 'reflectivity_column_max_dbz'

GRID_ROWS = numpy.linspace(0, 10, num=11, dtype=int)
GRID_COLUMNS = numpy.linspace(0, 10, num=11, dtype=int)
GRID_CELL_COUNTS = numpy.linspace(0, 10, num=11, dtype=int)
RADAR_VALUES = numpy.linspace(0, 10, num=11)

SENTINEL_VALUES = numpy.array([-99000., -99001.])
SENTINEL_INDICES = numpy.array([7, 9], dtype=int)
RADAR_VALUES[SENTINEL_INDICES[0]] = SENTINEL_VALUES[0]
RADAR_VALUES[SENTINEL_INDICES[1]] = SENTINEL_VALUES[1]

SPARSE_GRID_DICT_WITH_SENTINELS = {myrorss_io.GRID_ROW_COLUMN: GRID_ROWS,
                                   myrorss_io.GRID_COLUMN_COLUMN: GRID_COLUMNS,
                                   myrorss_io.NUM_GRID_CELL_COLUMN:
                                       GRID_CELL_COUNTS,
                                   RADAR_VAR_NAME: RADAR_VALUES}
SPARSE_GRID_TABLE_WITH_SENTINELS = pandas.DataFrame.from_dict(
    SPARSE_GRID_DICT_WITH_SENTINELS)

EXPECTED_SPARSE_GRID_TABLE_NO_SENTINELS = copy.deepcopy(
    SPARSE_GRID_TABLE_WITH_SENTINELS)
EXPECTED_SPARSE_GRID_TABLE_NO_SENTINELS.drop(
    EXPECTED_SPARSE_GRID_TABLE_NO_SENTINELS.index[SENTINEL_INDICES], axis=0,
    inplace=True)

LONGITUDES_DEG = numpy.array(
    [0., 45., 90., 135., 180., 210., -120., 270., -60., 330.])
EXPECTED_LNG_NEGATIVE_IN_WEST_DEG = numpy.array(
    [0., 45., 90., 135., 180., -150., -120., -90., -60., -30.])
EXPECTED_LNG_POSITIVE_IN_WEST_DEG = numpy.array(
    [0., 45., 90., 135., 180., 210., 240., 270., 300., 330.])

NW_GRID_POINT_LAT_DEG = 55.
NW_GRID_POINT_LNG_DEG = 230.
LAT_SPACING_DEG = 0.01
LNG_SPACING_DEG = 0.01
EXPECTED_GRID_LAT_DEG = numpy.array(
    [55., 54.99, 54.98, 54.97, 54.96, 54.95, 54.94, 54.93, 54.92, 54.91, 54.9])
EXPECTED_GRID_LNG_DEG = numpy.array(
    [230., 230.01, 230.02, 230.03, 230.04, 230.05, 230.06, 230.07, 230.08,
     230.09, 230.1])

NUM_LAT_IN_GRID = 3501
NUM_LNG_IN_GRID = 7001
CENTER_LAT_IN_GRID_DEG = 37.5
CENTER_LNG_IN_GRID_DEG = 265.


class MyrorssIoTests(unittest.TestCase):
    """Each method is a unit test for myrorss_io.py."""

    def test_convert_var_name(self):
        """Ensures correct output from _convert_var_name."""

        this_radar_var_name = myrorss_io._convert_var_name(RADAR_VAR_NAME_ORIG)
        self.assertTrue(this_radar_var_name == RADAR_VAR_NAME)

    def test_remove_sentinels(self):
        """Ensures correct output from _remove_sentinels."""

        sparse_grid_table_no_sentinels = myrorss_io._remove_sentinels(
            SPARSE_GRID_TABLE_WITH_SENTINELS, RADAR_VAR_NAME, SENTINEL_VALUES)
        self.assertTrue(sparse_grid_table_no_sentinels.equals(
            EXPECTED_SPARSE_GRID_TABLE_NO_SENTINELS))

    def test_convert_lng_negative_in_west(self):
        """Ensures correct output from convert_lng_negative_in_west."""

        lng_negative_in_west_deg = myrorss_io.convert_lng_negative_in_west(
            LONGITUDES_DEG)
        self.assertTrue(numpy.allclose(lng_negative_in_west_deg,
                                       EXPECTED_LNG_NEGATIVE_IN_WEST_DEG,
                                       atol=TOLERANCE))

    def test_convert_lng_positive_in_west(self):
        """Ensures correct output from convert_lng_positive_in_west."""

        lng_positive_in_west_deg = myrorss_io.convert_lng_positive_in_west(
            LONGITUDES_DEG)
        self.assertTrue(numpy.allclose(lng_positive_in_west_deg,
                                       EXPECTED_LNG_POSITIVE_IN_WEST_DEG,
                                       atol=TOLERANCE))

    def test_rowcol_to_latlng(self):
        """Ensures correct output from rowcol_to_latlng."""

        (grid_latitudes_deg, grid_longitudes_deg) = myrorss_io.rowcol_to_latlng(
            GRID_ROWS, GRID_COLUMNS,
            nw_grid_point_lat_deg=NW_GRID_POINT_LAT_DEG,
            nw_grid_point_lng_deg=NW_GRID_POINT_LNG_DEG,
            lat_spacing_deg=LAT_SPACING_DEG, lng_spacing_deg=LNG_SPACING_DEG)

        self.assertTrue(
            numpy.allclose(grid_latitudes_deg, EXPECTED_GRID_LAT_DEG,
                           atol=TOLERANCE))
        self.assertTrue(
            numpy.allclose(grid_longitudes_deg, EXPECTED_GRID_LNG_DEG,
                           atol=TOLERANCE))

    def test_latlng_to_rowcol(self):
        """Ensures correct output from latlng_to_rowcol."""

        (these_grid_rows, these_grid_columns) = myrorss_io.latlng_to_rowcol(
            EXPECTED_GRID_LAT_DEG, EXPECTED_GRID_LNG_DEG,
            nw_grid_point_lat_deg=NW_GRID_POINT_LAT_DEG,
            nw_grid_point_lng_deg=NW_GRID_POINT_LNG_DEG,
            lat_spacing_deg=LAT_SPACING_DEG, lng_spacing_deg=LNG_SPACING_DEG)

        self.assertTrue(
            numpy.allclose(these_grid_rows, GRID_ROWS, atol=TOLERANCE))
        self.assertTrue(
            numpy.allclose(these_grid_columns, GRID_COLUMNS, atol=TOLERANCE))

    def test_get_center_of_grid(self):
        """Ensures correct output from get_center_of_grid."""

        (this_center_lat_deg,
         this_center_lng_deg) = myrorss_io.get_center_of_grid(
            nw_grid_point_lat_deg=NW_GRID_POINT_LAT_DEG,
            nw_grid_point_lng_deg=NW_GRID_POINT_LNG_DEG,
            lat_spacing_deg=LAT_SPACING_DEG, lng_spacing_deg=LNG_SPACING_DEG,
            num_lat_in_grid=NUM_LAT_IN_GRID, num_lng_in_grid=NUM_LNG_IN_GRID)

        self.assertTrue(numpy.allclose(numpy.array([this_center_lat_deg]),
                                       numpy.array([CENTER_LAT_IN_GRID_DEG]),
                                       atol=TOLERANCE))
        self.assertTrue(numpy.allclose(numpy.array([this_center_lng_deg]),
                                       numpy.array([CENTER_LNG_IN_GRID_DEG]),
                                       atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
