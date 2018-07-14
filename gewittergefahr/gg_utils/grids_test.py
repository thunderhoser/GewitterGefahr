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

FIELD_MATRIX_AT_GRID_POINTS = numpy.array([
    [0., 1., 2., 3., 3., 2., 1., 0.],
    [-2., 2., 4., 6., 6., 4., 2., -2.],
    [5., 10., 15., 20., 20., 15., 10., 5.],
    [0., 0., 0., 0., 0., 0., 0., 0.]])
EXPECTED_FIELD_MATRIX_AT_EDGES = numpy.array([
    [0., 1., 2., 3., 3., 2., 1., 0., numpy.nan],
    [-2., 2., 4., 6., 6., 4., 2., -2., numpy.nan],
    [5., 10., 15., 20., 20., 15., 10., 5., numpy.nan],
    [0., 0., 0., 0., 0., 0., 0., 0., numpy.nan],
    [numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
     numpy.nan, numpy.nan, numpy.nan]])

# The following constants are used to test extract_latlng_subgrid.
FULL_DATA_MATRIX = numpy.array([[9., 1., 2., 2., 7., 8., 8., 9.],
                                [10., 3., numpy.nan, 5., 1., 8., 1., 7.],
                                [2., 6., 10., 10., 9., numpy.nan, 3., 4.],
                                [10., 10., 5., 8., 10., 7., 1., 10.],
                                [7., 10., 9., 10., 7., 2., 1., 1.]])

FULL_GRID_POINT_LATITUDES_DEG = numpy.array([53.7, 53.6, 53.5, 53.4, 53.3])
FULL_GRID_POINT_LONGITUDES_DEG = numpy.array(
    [246.2, 246.4, 246.6, 246.8, 247., 247.2, 247.4, 247.6])
CENTER_LATITUDE_DEG = 53.5
CENTER_LONGITUDE_DEG = 247.3
MAX_DISTANCE_FROM_CENTER_METRES = 20000.

EXPECTED_SUBGRID_DATA_MATRIX = numpy.array([[numpy.nan, 8., 1., numpy.nan],
                                            [9., numpy.nan, 3., 4.],
                                            [numpy.nan, 7., 1., numpy.nan]])
EXPECTED_ROW_OFFSET = 1
EXPECTED_COLUMN_OFFSET = 4

# The following constants are used to test count_events_on_equidistant_grid.
GRID_POINTS_X_FOR_COUNTING_METRES = numpy.array([0, 1, 2, 3, 4, 5], dtype=float)
GRID_POINTS_Y_FOR_COUNTING_METRES = numpy.array([10, 20, 30, 40], dtype=float)
EVENT_X_COORDS_METRES = numpy.array(
    [0.3, 1.5, 3.4, 5.2, 5, 0, 1.9, 2.9, 4.6, 0.2, 1, 4.4, 1.7, 2.5, 4.5])
EVENT_Y_COORDS_METRES = numpy.array(
    [36, 37, 36.5, 42, 36, 28, 30, 25, 25, 19, 22, 23, 15, 10, 7.5])
NUM_EVENTS_MATRIX_EQUIDISTANT = numpy.array([[0, 0, 0, 1, 0, 1],
                                             [1, 1, 1, 0, 1, 0],
                                             [1, 0, 1, 1, 0, 1],
                                             [1, 0, 1, 1, 0, 2]], dtype=int)

# The following constants are used to test count_events_on_non_equidistant_grid.
GRID_POINT_X_MATRIX_FOR_COUNTING_METRES = numpy.array(
    [[0, 1, 2, 3, 4, 5],
     [0, 1, 2, 3, 4, 5],
     [0, 1, 2, 3, 4, 5],
     [0, 1, 2, 3, 4, 5]], dtype=float)
GRID_POINT_Y_MATRIX_FOR_COUNTING_METRES = numpy.array(
    [[10, 10, 10, 10, 10, 10],
     [20, 20, 20, 20, 20, 20],
     [30, 30, 30, 30, 30, 30],
     [40, 40, 40, 40, 40, 40]], dtype=float)

COUNTING_RADIUS_METRES = 5.
NUM_EVENTS_MATRIX_NON_EQUIDISTANT = numpy.array([[1, 2, 2, 2, 2, 2],
                                                 [2, 3, 3, 3, 3, 3],
                                                 [2, 2, 2, 2, 2, 1],
                                                 [3, 4, 5, 5, 4, 4]], dtype=int)

# The following constants are used to test get_latlng_grid_points_in_radius.
LATITUDES_FOR_RADIUS_TEST_DEG = numpy.array([51, 52, 53, 54], dtype=float)
LONGITUDES_FOR_RADIUS_TEST_DEG = numpy.array([245.8, 246.2, 246.6, 247, 247.4])

CALGARY_LATITUDE_DEG = 51.1
CALGARY_LONGITUDE_DEG = 246.
EDMONTON_LATITUDE_DEG = 53.6
EDMONTON_LONGITUDE_DEG = 246.5
EFFECTIVE_RADIUS_METRES = 105000.

ROWS_IN_CALGARY_RADIUS_LAT_INCREASING = numpy.array(
    [0, 0, 0, 0, 0, 1, 1], dtype=int)
COLUMNS_IN_CALGARY_RADIUS_LAT_INCREASING = numpy.array(
    [0, 1, 2, 3, 4, 0, 1], dtype=int)
ROWS_IN_CALGARY_RADIUS_LAT_DECREASING = numpy.array(
    [2, 2, 3, 3, 3, 3, 3], dtype=int)
COLUMNS_IN_CALGARY_RADIUS_LAT_DECREASING = numpy.array(
    [0, 1, 0, 1, 2, 3, 4], dtype=int)

ROWS_IN_EDMONTON_RADIUS_LAT_INCREASING = numpy.array(
    [2, 2, 2, 2, 2, 3, 3, 3, 3, 3], dtype=int)
COLUMNS_IN_EDMONTON_RADIUS_LAT_INCREASING = numpy.array(
    [0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=int)
ROWS_IN_EDMONTON_RADIUS_LAT_DECREASING = numpy.array(
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=int)
COLUMNS_IN_EDMONTON_RADIUS_LAT_DECREASING = numpy.array(
    [0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=int)


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

    def test_latlng_field_grid_points_to_edges(self):
        """Ensures correct output from latlng_field_grid_points_to_edges."""

        (this_field_matrix_at_edges,
         these_edge_latitudes_deg,
         these_edge_longitudes_deg) = grids.latlng_field_grid_points_to_edges(
             field_matrix=FIELD_MATRIX_AT_GRID_POINTS,
             min_latitude_deg=MIN_LATITUDE_DEG,
             min_longitude_deg=MIN_LONGITUDE_DEG,
             lat_spacing_deg=LAT_SPACING_DEG, lng_spacing_deg=LNG_SPACING_DEG)

        self.assertTrue(numpy.allclose(
            these_edge_latitudes_deg, EXPECTED_GRID_CELL_EDGE_LATITUDES_DEG,
            atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_edge_longitudes_deg, EXPECTED_GRID_CELL_EDGE_LONGITUDES_DEG,
            atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            this_field_matrix_at_edges, EXPECTED_FIELD_MATRIX_AT_EDGES,
            equal_nan=True, atol=TOLERANCE))

    def test_extract_latlng_subgrid(self):
        """Ensures correct output from extract_latlng_subgrid."""

        this_subgrid_data_matrix, this_row_offset, this_column_offset = (
            grids.extract_latlng_subgrid(
                data_matrix=FULL_DATA_MATRIX,
                grid_point_latitudes_deg=FULL_GRID_POINT_LATITUDES_DEG,
                grid_point_longitudes_deg=FULL_GRID_POINT_LONGITUDES_DEG,
                center_latitude_deg=CENTER_LATITUDE_DEG,
                center_longitude_deg=CENTER_LONGITUDE_DEG,
                max_distance_from_center_metres=
                MAX_DISTANCE_FROM_CENTER_METRES))

        self.assertTrue(this_row_offset == EXPECTED_ROW_OFFSET)
        self.assertTrue(this_column_offset == EXPECTED_COLUMN_OFFSET)
        self.assertTrue(numpy.allclose(
            this_subgrid_data_matrix, EXPECTED_SUBGRID_DATA_MATRIX,
            equal_nan=True, atol=TOLERANCE))

    def test_count_events_on_equidistant_grid(self):
        """Ensures correct output from count_events_on_equidistant_grid."""

        this_num_events_matrix = grids.count_events_on_equidistant_grid(
            event_x_coords_metres=EVENT_X_COORDS_METRES,
            event_y_coords_metres=EVENT_Y_COORDS_METRES,
            grid_point_x_coords_metres=GRID_POINTS_X_FOR_COUNTING_METRES,
            grid_point_y_coords_metres=GRID_POINTS_Y_FOR_COUNTING_METRES)

        self.assertTrue(numpy.array_equal(
            this_num_events_matrix, NUM_EVENTS_MATRIX_EQUIDISTANT))

    def test_count_events_on_non_equidistant_grid(self):
        """Ensures correct output from count_events_on_non_equidistant_grid."""

        this_num_events_matrix = grids.count_events_on_non_equidistant_grid(
            event_x_coords_metres=EVENT_X_COORDS_METRES,
            event_y_coords_metres=EVENT_Y_COORDS_METRES,
            grid_point_x_matrix_metres=GRID_POINT_X_MATRIX_FOR_COUNTING_METRES,
            grid_point_y_matrix_metres=GRID_POINT_Y_MATRIX_FOR_COUNTING_METRES,
            effective_radius_metres=COUNTING_RADIUS_METRES)

        self.assertTrue(numpy.array_equal(
            this_num_events_matrix, NUM_EVENTS_MATRIX_NON_EQUIDISTANT))

    def test_get_latlng_grid_points_in_radius_calgary_lat_incr(self):
        """Ensures correct output from get_latlng_grid_points_in_radius.

        In this case, the test point is Calgary and latitude increases with row
        number.
        """

        these_rows, these_columns, _ = grids.get_latlng_grid_points_in_radius(
            grid_point_latitudes_deg=LATITUDES_FOR_RADIUS_TEST_DEG,
            grid_point_longitudes_deg=LONGITUDES_FOR_RADIUS_TEST_DEG,
            test_latitude_deg=CALGARY_LATITUDE_DEG,
            test_longitude_deg=CALGARY_LONGITUDE_DEG,
            effective_radius_metres=EFFECTIVE_RADIUS_METRES)

        self.assertTrue(numpy.array_equal(
            these_rows, ROWS_IN_CALGARY_RADIUS_LAT_INCREASING))
        self.assertTrue(numpy.array_equal(
            these_columns, COLUMNS_IN_CALGARY_RADIUS_LAT_INCREASING))

    def test_get_latlng_grid_points_in_radius_calgary_lat_decr(self):
        """Ensures correct output from get_latlng_grid_points_in_radius.

        In this case, the test point is Calgary and latitude decreases with row
        number.
        """

        these_rows, these_columns, _ = grids.get_latlng_grid_points_in_radius(
            grid_point_latitudes_deg=LATITUDES_FOR_RADIUS_TEST_DEG[::-1],
            grid_point_longitudes_deg=LONGITUDES_FOR_RADIUS_TEST_DEG,
            test_latitude_deg=CALGARY_LATITUDE_DEG,
            test_longitude_deg=CALGARY_LONGITUDE_DEG,
            effective_radius_metres=EFFECTIVE_RADIUS_METRES)

        self.assertTrue(numpy.array_equal(
            these_rows, ROWS_IN_CALGARY_RADIUS_LAT_DECREASING))
        self.assertTrue(numpy.array_equal(
            these_columns, COLUMNS_IN_CALGARY_RADIUS_LAT_DECREASING))

    def test_get_latlng_grid_points_in_radius_edmonton_lat_incr(self):
        """Ensures correct output from get_latlng_grid_points_in_radius.

        In this case, the test point is Edmonton and latitude increases with row
        number.
        """

        these_rows, these_columns, _ = grids.get_latlng_grid_points_in_radius(
            grid_point_latitudes_deg=LATITUDES_FOR_RADIUS_TEST_DEG,
            grid_point_longitudes_deg=LONGITUDES_FOR_RADIUS_TEST_DEG,
            test_latitude_deg=EDMONTON_LATITUDE_DEG,
            test_longitude_deg=EDMONTON_LONGITUDE_DEG,
            effective_radius_metres=EFFECTIVE_RADIUS_METRES)

        self.assertTrue(numpy.array_equal(
            these_rows, ROWS_IN_EDMONTON_RADIUS_LAT_INCREASING))
        self.assertTrue(numpy.array_equal(
            these_columns, COLUMNS_IN_EDMONTON_RADIUS_LAT_INCREASING))

    def test_get_latlng_grid_points_in_radius_edmonton_lat_decr(self):
        """Ensures correct output from get_latlng_grid_points_in_radius.

        In this case, the test point is Edmonton and latitude decreases with row
        number.
        """

        these_rows, these_columns, _ = grids.get_latlng_grid_points_in_radius(
            grid_point_latitudes_deg=LATITUDES_FOR_RADIUS_TEST_DEG[::-1],
            grid_point_longitudes_deg=LONGITUDES_FOR_RADIUS_TEST_DEG,
            test_latitude_deg=EDMONTON_LATITUDE_DEG,
            test_longitude_deg=EDMONTON_LONGITUDE_DEG,
            effective_radius_metres=EFFECTIVE_RADIUS_METRES)

        self.assertTrue(numpy.array_equal(
            these_rows, ROWS_IN_EDMONTON_RADIUS_LAT_DECREASING))
        self.assertTrue(numpy.array_equal(
            these_columns, COLUMNS_IN_EDMONTON_RADIUS_LAT_DECREASING))


if __name__ == '__main__':
    unittest.main()
