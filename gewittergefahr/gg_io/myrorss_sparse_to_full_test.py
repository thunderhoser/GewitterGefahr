"""Unit tests for myrorss_sparse_to_full.py."""

import copy
import unittest
import numpy
import pandas
from gewittergefahr.gg_io import myrorss_sparse_to_full as s2f
from gewittergefahr.gg_io import myrorss_io

TOLERANCE = 1e-6
RADAR_VAR_NAME = 'reflectivity_column_max_dbz'

NW_GRID_POINT_LAT_DEG = 40.
NW_GRID_POINT_LNG_DEG = 260.
LAT_SPACING_DEG = 0.1
LNG_SPACING_DEG = 0.2
NUM_LAT_IN_GRID = 4
NUM_LNG_IN_GRID = 6

EXPECTED_MIN_CENTER_LAT_DEG = 39.7
EXPECTED_MAX_CENTER_LAT_DEG = copy.deepcopy(NW_GRID_POINT_LAT_DEG)
EXPECTED_MIN_CENTER_LNG_DEG = copy.deepcopy(NW_GRID_POINT_LNG_DEG)
EXPECTED_MAX_CENTER_LNG_DEG = 261.

EXPECTED_BOUNDING_BOX_DICT = {
    s2f.MIN_CENTER_LAT_COLUMN: EXPECTED_MIN_CENTER_LAT_DEG,
    s2f.MAX_CENTER_LAT_COLUMN: EXPECTED_MAX_CENTER_LAT_DEG,
    s2f.MIN_CENTER_LNG_COLUMN: EXPECTED_MIN_CENTER_LNG_DEG,
    s2f.MAX_CENTER_LNG_COLUMN: EXPECTED_MAX_CENTER_LNG_DEG}

EXPECTED_CENTER_LATITUDES_DEG = numpy.array([39.7, 39.8, 39.9, 40.])
EXPECTED_CENTER_LONGITUDES_DEG = numpy.array(
    [260., 260.2, 260.4, 260.6, 260.8, 261.])

EXPECTED_EDGE_LATITUDES_DEG = numpy.array([39.65, 39.75, 39.85, 39.95, 40.05])
EXPECTED_EDGE_LONGITUDES_DEG = numpy.array(
    [259.9, 260.1, 260.3, 260.5, 260.7, 260.9, 261.1])

EXPECTED_CENTER_LAT_MATRIX_DEG = numpy.array(
    [[39.7, 39.7, 39.7, 39.7, 39.7, 39.7],
     [39.8, 39.8, 39.8, 39.8, 39.8, 39.8],
     [39.9, 39.9, 39.9, 39.9, 39.9, 39.9],
     [40., 40., 40., 40., 40., 40.]])
EXPECTED_CENTER_LNG_MATRIX_DEG = numpy.array(
    [[260., 260.2, 260.4, 260.6, 260.8, 261.],
     [260., 260.2, 260.4, 260.6, 260.8, 261.],
     [260., 260.2, 260.4, 260.6, 260.8, 261.],
     [260., 260.2, 260.4, 260.6, 260.8, 261.]])

START_ROWS = numpy.array([0, 1, 2, 3], dtype=int)
START_COLUMNS = numpy.array([3, 2, 1, 0], dtype=int)
GRID_CELL_COUNTS = numpy.array([1, 2, 3, 4], dtype=int)
RADAR_VALUES = numpy.array([35., 50., 70., 65.])

SPARSE_GRID_DICT = {myrorss_io.GRID_ROW_COLUMN: START_ROWS,
                    myrorss_io.GRID_COLUMN_COLUMN: START_COLUMNS,
                    myrorss_io.NUM_GRID_CELL_COLUMN: GRID_CELL_COUNTS,
                    RADAR_VAR_NAME: RADAR_VALUES}
SPARSE_GRID_TABLE = pandas.DataFrame.from_dict(SPARSE_GRID_DICT)

EXPECTED_DATA_MATRIX = numpy.array(
    [[numpy.nan, numpy.nan, numpy.nan, 35., numpy.nan, numpy.nan],
     [numpy.nan, numpy.nan, 50., 50., numpy.nan, numpy.nan],
     [numpy.nan, 70., 70., 70., numpy.nan, numpy.nan],
     [65., 65., 65., 65., numpy.nan, numpy.nan]])


class MyrorssIoTests(unittest.TestCase):
    """Each method is a unit test for myrorss_sparse_to_full.py."""

    def test_get_bounding_box(self):
        """Ensures correct output from _get_bounding_box_of_grid_points."""

        bounding_box_dict = s2f._get_bounding_box_of_grid_points(
            nw_grid_point_lat_deg=NW_GRID_POINT_LAT_DEG,
            nw_grid_point_lng_deg=NW_GRID_POINT_LNG_DEG,
            lat_spacing_deg=LAT_SPACING_DEG, lng_spacing_deg=LNG_SPACING_DEG,
            num_lat_in_grid=NUM_LAT_IN_GRID, num_lng_in_grid=NUM_LNG_IN_GRID)

        self.assertTrue(bounding_box_dict == EXPECTED_BOUNDING_BOX_DICT)

    def test_generate_grid_points(self):
        """Ensures correct output from _generate_grid_points."""

        (center_latitudes_deg,
         center_longitudes_deg) = s2f._generate_grid_points(
            EXPECTED_BOUNDING_BOX_DICT, num_lat_in_grid=NUM_LAT_IN_GRID,
            num_lng_in_grid=NUM_LNG_IN_GRID)

        self.assertTrue(
            numpy.allclose(center_latitudes_deg, EXPECTED_CENTER_LATITUDES_DEG,
                           atol=TOLERANCE))
        self.assertTrue(numpy.allclose(center_longitudes_deg,
                                       EXPECTED_CENTER_LONGITUDES_DEG,
                                       atol=TOLERANCE))

    def test_generate_grid_cell_edges(self):
        """Ensures correct output from _generate_grid_cell_edges."""

        (edge_latitudes_deg,
         edge_longitudes_deg) = s2f._generate_grid_cell_edges(
            EXPECTED_BOUNDING_BOX_DICT, lat_spacing_deg=LAT_SPACING_DEG,
            lng_spacing_deg=LNG_SPACING_DEG, num_lat_in_grid=NUM_LAT_IN_GRID,
            num_lng_in_grid=NUM_LNG_IN_GRID)

        self.assertTrue(
            numpy.allclose(edge_latitudes_deg, EXPECTED_EDGE_LATITUDES_DEG,
                           atol=TOLERANCE))
        self.assertTrue(
            numpy.allclose(edge_longitudes_deg, EXPECTED_EDGE_LONGITUDES_DEG,
                           atol=TOLERANCE))

    def test_grid_vectors_to_matrices(self):
        """Ensures correct output from grid_vectors_to_matrices."""

        (center_lat_matrix_deg,
         center_lng_matrix_deg) = s2f.grid_vectors_to_matrices(
            EXPECTED_CENTER_LATITUDES_DEG, EXPECTED_CENTER_LONGITUDES_DEG)

        self.assertTrue(numpy.allclose(center_lat_matrix_deg,
                                       EXPECTED_CENTER_LAT_MATRIX_DEG,
                                       atol=TOLERANCE))
        self.assertTrue(numpy.allclose(center_lng_matrix_deg,
                                       EXPECTED_CENTER_LNG_MATRIX_DEG,
                                       atol=TOLERANCE))

    def test_sparse_to_full_grid(self):
        """Ensures correct output from sparse_to_full_grid."""

        data_matrix = s2f.sparse_to_full_grid(SPARSE_GRID_TABLE,
                                              var_name=RADAR_VAR_NAME,
                                              num_lat_in_grid=NUM_LAT_IN_GRID,
                                              num_lng_in_grid=NUM_LNG_IN_GRID)

        self.assertTrue(
            numpy.allclose(data_matrix, EXPECTED_DATA_MATRIX, atol=TOLERANCE,
                           equal_nan=True))


if __name__ == '__main__':
    unittest.main()
