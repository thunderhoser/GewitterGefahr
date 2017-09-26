"""Unit tests for grids.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import grids

TOLERANCE = 1e-6

X_MIN_METRES = 0.
Y_MIN_METRES = 50000.
X_SPACING_METRES = 10000.
Y_SPACING_METRES = 20000.
NUM_XY_ROWS = 5
NUM_XY_COLUMNS = 6

EXPECTED_GRID_POINT_X_METRES = numpy.array(
    [0., 10000., 20000., 30000., 40000., 50000.])
EXPECTED_GRID_POINT_Y_METRES = numpy.array(
    [50000., 70000., 90000., 110000., 130000.])
EXPECTED_GRID_CELL_EDGE_X_METRES = numpy.array(
    [-5000., 5000., 15000., 25000., 35000., 45000., 55000.])
EXPECTED_GRID_CELL_EDGE_Y_METRES = numpy.array(
    [40000., 60000., 80000., 100000., 120000., 140000.])

EXPECTED_GRID_POINT_X_MATRIX_METRES = (
    numpy.array([[0., 10000., 20000., 30000., 40000., 50000.],
                 [0., 10000., 20000., 30000., 40000., 50000.],
                 [0., 10000., 20000., 30000., 40000., 50000.],
                 [0., 10000., 20000., 30000., 40000., 50000.],
                 [0., 10000., 20000., 30000., 40000., 50000.]]))

EXPECTED_GRID_POINT_Y_MATRIX_METRES = (
    numpy.array([[50000., 50000., 50000., 50000., 50000., 50000.],
                 [70000., 70000., 70000., 70000., 70000., 70000.],
                 [90000., 90000., 90000., 90000., 90000., 90000.],
                 [110000., 110000., 110000., 110000., 110000., 110000.],
                 [130000., 130000., 130000., 130000., 130000., 130000.]]))

MIN_LATITUDE_DEG = 50.
MIN_LONGITUDE_DEG = 240.
LAT_SPACING_DEG = 0.5
LNG_SPACING_DEG = 1.
NUM_LATLNG_ROWS = 4
NUM_LATLNG_COLUMNS = 8

EXPECTED_GRID_POINT_LATITUDES_DEG = numpy.array([50., 50.5, 51., 51.5])
EXPECTED_GRID_POINT_LONGITUDES_DEG = numpy.array(
    [240., 241., 242., 243., 244., 245., 246., 247.])
EXPECTED_GRID_CELL_EDGE_LATITUDES_DEG = numpy.array(
    [49.75, 50.25, 50.75, 51.25, 51.75])
EXPECTED_GRID_CELL_EDGE_LONGITUDES_DEG = numpy.array(
    [239.5, 240.5, 241.5, 242.5, 243.5, 244.5, 245.5, 246.5, 247.5])

EXPECTED_GRID_POINT_LAT_MATRIX_DEG = (
    numpy.array([[50., 50., 50., 50., 50., 50., 50., 50.],
                 [50.5, 50.5, 50.5, 50.5, 50.5, 50.5, 50.5, 50.5],
                 [51., 51., 51., 51., 51., 51., 51., 51.],
                 [51.5, 51.5, 51.5, 51.5, 51.5, 51.5, 51.5, 51.5]]))

EXPECTED_GRID_POINT_LNG_MATRIX_DEG = (
    numpy.array([[240., 241., 242., 243., 244., 245., 246., 247.],
                 [240., 241., 242., 243., 244., 245., 246., 247.],
                 [240., 241., 242., 243., 244., 245., 246., 247.],
                 [240., 241., 242., 243., 244., 245., 246., 247.]]))


class GridsTests(unittest.TestCase):
    """Each method is a unit test for grids.py."""

    def test_get_xy_grid_points(self):
        """Ensures correct output from get_xy_grid_points."""

        (grid_point_x_metres, grid_point_y_metres) = grids.get_xy_grid_points(
            x_min_metres=X_MIN_METRES, y_min_metres=Y_MIN_METRES,
            x_spacing_metres=X_SPACING_METRES,
            y_spacing_metres=Y_SPACING_METRES, num_rows=NUM_XY_ROWS,
            num_columns=NUM_XY_COLUMNS)

        self.assertTrue(
            numpy.allclose(grid_point_x_metres, EXPECTED_GRID_POINT_X_METRES,
                           atol=TOLERANCE))
        self.assertTrue(
            numpy.allclose(grid_point_y_metres, EXPECTED_GRID_POINT_Y_METRES,
                           atol=TOLERANCE))

    def test_get_xy_grid_cell_edges(self):
        """Ensures correct output from get_xy_grid_cell_edges."""

        (grid_cell_edge_x_metres,
         grid_cell_edge_y_metres) = grids.get_xy_grid_cell_edges(
             x_min_metres=X_MIN_METRES, y_min_metres=Y_MIN_METRES,
             x_spacing_metres=X_SPACING_METRES,
             y_spacing_metres=Y_SPACING_METRES, num_rows=NUM_XY_ROWS,
             num_columns=NUM_XY_COLUMNS)

        self.assertTrue(
            numpy.allclose(grid_cell_edge_x_metres,
                           EXPECTED_GRID_CELL_EDGE_X_METRES,
                           atol=TOLERANCE))
        self.assertTrue(
            numpy.allclose(grid_cell_edge_y_metres,
                           EXPECTED_GRID_CELL_EDGE_Y_METRES,
                           atol=TOLERANCE))

    def test_get_latlng_grid_points(self):
        """Ensures correct output from get_latlng_grid_points."""

        (grid_point_latitudes_deg,
         grid_point_longitudes_deg) = grids.get_latlng_grid_points(
             min_latitude_deg=MIN_LATITUDE_DEG,
             min_longitude_deg=MIN_LONGITUDE_DEG,
             lat_spacing_deg=LAT_SPACING_DEG, lng_spacing_deg=LNG_SPACING_DEG,
             num_rows=NUM_LATLNG_ROWS, num_columns=NUM_LATLNG_COLUMNS)

        self.assertTrue(
            numpy.allclose(grid_point_latitudes_deg,
                           EXPECTED_GRID_POINT_LATITUDES_DEG,
                           atol=TOLERANCE))
        self.assertTrue(
            numpy.allclose(grid_point_longitudes_deg,
                           EXPECTED_GRID_POINT_LONGITUDES_DEG,
                           atol=TOLERANCE))

    def test_get_latlng_grid_cell_edges(self):
        """Ensures correct output from get_latlng_grid_cell_edges."""

        (grid_cell_edge_latitudes_deg,
         grid_cell_edge_longitudes_deg) = grids.get_latlng_grid_cell_edges(
             min_latitude_deg=MIN_LATITUDE_DEG,
             min_longitude_deg=MIN_LONGITUDE_DEG,
             lat_spacing_deg=LAT_SPACING_DEG, lng_spacing_deg=LNG_SPACING_DEG,
             num_rows=NUM_LATLNG_ROWS, num_columns=NUM_LATLNG_COLUMNS)

        self.assertTrue(
            numpy.allclose(grid_cell_edge_latitudes_deg,
                           EXPECTED_GRID_CELL_EDGE_LATITUDES_DEG,
                           atol=TOLERANCE))
        self.assertTrue(
            numpy.allclose(grid_cell_edge_longitudes_deg,
                           EXPECTED_GRID_CELL_EDGE_LONGITUDES_DEG,
                           atol=TOLERANCE))

    def test_xy_vectors_to_matrices(self):
        """Ensures correct output from xy_vectors_to_matrices."""

        (grid_point_x_matrix_metres,
         grid_point_y_matrix_metres) = grids.xy_vectors_to_matrices(
             EXPECTED_GRID_POINT_X_METRES, EXPECTED_GRID_POINT_Y_METRES)

        self.assertTrue(numpy.allclose(grid_point_x_matrix_metres,
                                       EXPECTED_GRID_POINT_X_MATRIX_METRES,
                                       atol=TOLERANCE))
        self.assertTrue(numpy.allclose(grid_point_y_matrix_metres,
                                       EXPECTED_GRID_POINT_Y_MATRIX_METRES,
                                       atol=TOLERANCE))

    def test_latlng_vectors_to_matrices(self):
        """Ensures correct output from latlng_vectors_to_matrices."""

        (grid_point_lat_matrix_deg,
         grid_point_lng_matrix_deg) = grids.latlng_vectors_to_matrices(
             EXPECTED_GRID_POINT_LATITUDES_DEG,
             EXPECTED_GRID_POINT_LONGITUDES_DEG)

        self.assertTrue(numpy.allclose(grid_point_lat_matrix_deg,
                                       EXPECTED_GRID_POINT_LAT_MATRIX_DEG,
                                       atol=TOLERANCE))
        self.assertTrue(numpy.allclose(grid_point_lng_matrix_deg,
                                       EXPECTED_GRID_POINT_LNG_MATRIX_DEG,
                                       atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
