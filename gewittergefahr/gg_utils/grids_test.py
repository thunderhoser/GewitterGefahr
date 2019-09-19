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

POINT_X_COORDS_METRES = numpy.array(
    [0, 10000, 20000, 30000, 40000, 50000], dtype=float
)
POINT_Y_COORDS_METRES = numpy.array(
    [50000, 70000, 90000, 110000, 130000], dtype=float
)
EDGE_X_COORDS_METRES = numpy.array(
    [-5000, 5000, 15000, 25000, 35000, 45000, 55000], dtype=float
)
EDGE_Y_COORDS_METRES = numpy.array(
    [40000, 60000, 80000, 100000, 120000, 140000], dtype=float
)

POINT_X_MATRIX_METRES = numpy.array([
    [0, 10000, 20000, 30000, 40000, 50000],
    [0, 10000, 20000, 30000, 40000, 50000],
    [0, 10000, 20000, 30000, 40000, 50000],
    [0, 10000, 20000, 30000, 40000, 50000],
    [0, 10000, 20000, 30000, 40000, 50000]
], dtype=float)

POINT_Y_MATRIX_METRES = numpy.array([
    [50000, 50000, 50000, 50000, 50000, 50000],
    [70000, 70000, 70000, 70000, 70000, 70000],
    [90000, 90000, 90000, 90000, 90000, 90000],
    [110000, 110000, 110000, 110000, 110000, 110000],
    [130000, 130000, 130000, 130000, 130000, 130000]
], dtype=float)

MIN_LATITUDE_DEG = 50.
MIN_LONGITUDE_DEG = 240.
LATITUDE_SPACING_DEG = 0.5
LONGITUDE_SPACING_DEG = 1.
NUM_LATLNG_ROWS = 4
NUM_LATLNG_COLUMNS = 8

POINT_LATITUDES_DEG = numpy.array([50, 50.5, 51, 51.5])
POINT_LONGITUDES_DEG = numpy.array(
    [240, 241, 242, 243, 244, 245, 246, 247], dtype=float
)
EDGE_LATITUDES_DEG = numpy.array([
    49.75, 50.25, 50.75, 51.25, 51.75
])
EDGE_LONGITUDES_DEG = numpy.array([
    239.5, 240.5, 241.5, 242.5, 243.5, 244.5, 245.5, 246.5, 247.5
])

POINT_LATITUDE_MATRIX_DEG = numpy.array([
    [50, 50, 50, 50, 50, 50, 50, 50],
    [50.5, 50.5, 50.5, 50.5, 50.5, 50.5, 50.5, 50.5],
    [51, 51, 51, 51, 51, 51, 51, 51],
    [51.5, 51.5, 51.5, 51.5, 51.5, 51.5, 51.5, 51.5]
])

POINT_LONGITUDE_MATRIX_DEG = numpy.array([
    [240, 241, 242, 243, 244, 245, 246, 247],
    [240, 241, 242, 243, 244, 245, 246, 247],
    [240, 241, 242, 243, 244, 245, 246, 247],
    [240, 241, 242, 243, 244, 245, 246, 247]
], dtype=float)

FIELD_MATRIX_AT_POINTS = numpy.array([
    [0, 1, 2, 3, 3, 2, 1, 0],
    [-2, 2, 4, 6, 6, 4, 2, -2],
    [5, 10, 15, 20, 20, 15, 10, 5],
    [0, 0, 0, 0, 0, 0, 0, 0]
], dtype=float)

FIELD_MATRIX_AT_EDGES = numpy.array([
    [0, 1, 2, 3, 3, 2, 1, 0, numpy.nan],
    [-2, 2, 4, 6, 6, 4, 2, -2, numpy.nan],
    [5, 10, 15, 20, 20, 15, 10, 5, numpy.nan],
    [0, 0, 0, 0, 0, 0, 0, 0, numpy.nan],
    [numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan,
     numpy.nan, numpy.nan, numpy.nan]
])

# The following constants are used to test extract_latlng_subgrid.
FULL_GRID_DATA_MATRIX = numpy.array([
    [9, 1, 2, 2, 7, 8, 8, 9],
    [10, 3, numpy.nan, 5, 1, 8, 1, 7],
    [2, 6, 10, 10, 9, numpy.nan, 3, 4],
    [10, 10, 5, 8, 10, 7, 1, 10],
    [7, 10, 9, 10, 7, 2, 1, 1]
])

FULL_GRID_LATITUDES_DEG = numpy.array([53.7, 53.6, 53.5, 53.4, 53.3])
FULL_GRID_LONGITUDES_DEG = numpy.array([
    246.2, 246.4, 246.6, 246.8, 247, 247.2, 247.4, 247.6
])

SUBGRID_CENTER_LATITUDE_DEG = 53.5
SUBGRID_CENTER_LONGITUDE_DEG = 247.3
SUBGRID_MAX_DISTANCE_METRES = 20000.

SUBGRID_DATA_MATRIX = numpy.array([
    [numpy.nan, 8, 1, numpy.nan],
    [9, numpy.nan, 3, 4],
    [numpy.nan, 7, 1, numpy.nan]
])

FULL_TO_SUBGRID_ROWS = numpy.array([1, 2, 3], dtype=int)
FULL_TO_SUBGRID_COLUMNS = numpy.array([4, 5, 6, 7], dtype=int)

# The following constants are used to test count_events_on_equidistant_grid.
EVENT_GRID_POINTS_X_METRES = numpy.array([0, 1, 2, 3, 4, 5], dtype=float)
EVENT_GRID_POINTS_Y_METRES = numpy.array([10, 20, 30, 40], dtype=float)
EVENT_X_COORDS_METRES = numpy.array([
    0.3, 1.5, 3.4, 5.2, 5, 0, 1.9, 2.9, 4.6, 0.2, 1, 4.4, 1.7, 2.5, 4.5
])
EVENT_Y_COORDS_METRES = numpy.array([
    36, 37, 36.5, 42, 36, 28, 30, 25, 25, 19, 22, 23, 15, 10, 7.5
])
INTEGER_EVENT_IDS = numpy.array(
    [0, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], dtype=int
)

NUM_EVENTS_MATRIX = numpy.array([
    [0, 0, 0, 1, 0, 1],
    [1, 1, 1, 0, 1, 0],
    [1, 0, 1, 1, 0, 1],
    [1, 0, 1, 1, 0, 2]
], dtype=int)

NUM_UNIQUE_EVENTS_MATRIX = numpy.array([
    [0, 0, 0, 1, 0, 1],
    [1, 1, 1, 0, 1, 0],
    [1, 0, 1, 1, 0, 1],
    [1, 0, 1, 1, 0, 1]
], dtype=int)

# The following constants are used to test get_latlng_grid_points_in_radius.
RADIUS_TEST_LATITUDES_DEG = numpy.array([51, 52, 53, 54], dtype=float)
RADIUS_TEST_LONGITUDES_DEG = numpy.array([245.8, 246.2, 246.6, 247, 247.4])

CALGARY_LATITUDE_DEG = 51.1
CALGARY_LONGITUDE_DEG = 246.
EDMONTON_LATITUDE_DEG = 53.6
EDMONTON_LONGITUDE_DEG = 246.5
EFFECTIVE_RADIUS_METRES = 105000.

CALGARY_ROWS_INCREASING_LAT = numpy.array([0, 0, 0, 0, 0, 1, 1], dtype=int)
CALGARY_COLUMNS_INCREASING_LAT = numpy.array([0, 1, 2, 3, 4, 0, 1], dtype=int)
CALGARY_ROWS_DECREASING_LAT = numpy.array([2, 2, 3, 3, 3, 3, 3], dtype=int)
CALGARY_COLUMNS_DECREASING_LAT = numpy.array([0, 1, 0, 1, 2, 3, 4], dtype=int)

EDMONTON_ROWS_INCREASING_LAT = numpy.array(
    [2, 2, 2, 2, 2, 3, 3, 3, 3, 3], dtype=int
)
EDMONTON_COLUMNS_INCREASING_LAT = numpy.array(
    [0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=int
)
EDMONTON_ROWS_DECREASING_LAT = numpy.array(
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=int
)
EDMONTON_COLUMNS_DECREASING_LAT = numpy.array(
    [0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=int
)

# The following columns are used to test find_events_in_grid_cell.
EVENT_GRID_EDGES_X_METRES = numpy.array([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
EVENT_GRID_EDGES_Y_METRES = numpy.array([5, 15, 25, 35, 45], dtype=float)

GRID_CELL_TO_EVENT_DICT = dict()
GRID_CELL_TO_EVENT_DICT[0, 3] = numpy.array([13], dtype=int)
GRID_CELL_TO_EVENT_DICT[0, 5] = numpy.array([14], dtype=int)
GRID_CELL_TO_EVENT_DICT[1, 0] = numpy.array([9], dtype=int)
GRID_CELL_TO_EVENT_DICT[1, 1] = numpy.array([10], dtype=int)
GRID_CELL_TO_EVENT_DICT[1, 2] = numpy.array([12], dtype=int)
GRID_CELL_TO_EVENT_DICT[1, 4] = numpy.array([11], dtype=int)
GRID_CELL_TO_EVENT_DICT[2, 0] = numpy.array([5], dtype=int)
GRID_CELL_TO_EVENT_DICT[2, 2] = numpy.array([6], dtype=int)
GRID_CELL_TO_EVENT_DICT[2, 3] = numpy.array([7], dtype=int)
GRID_CELL_TO_EVENT_DICT[2, 5] = numpy.array([8], dtype=int)
GRID_CELL_TO_EVENT_DICT[3, 0] = numpy.array([0], dtype=int)
GRID_CELL_TO_EVENT_DICT[3, 2] = numpy.array([1], dtype=int)
GRID_CELL_TO_EVENT_DICT[3, 3] = numpy.array([2], dtype=int)
GRID_CELL_TO_EVENT_DICT[3, 5] = numpy.array([3, 4], dtype=int)

THIS_NUM_ROWS = len(EVENT_GRID_POINTS_Y_METRES)
THIS_NUM_COLUMNS = len(EVENT_GRID_POINTS_X_METRES)

for k in range(THIS_NUM_ROWS):
    for m in range(THIS_NUM_COLUMNS):
        if (k, m) in GRID_CELL_TO_EVENT_DICT:
            continue

        GRID_CELL_TO_EVENT_DICT[k, m] = numpy.array([], dtype=int)


class GridsTests(unittest.TestCase):
    """Each method is a unit test for grids.py."""

    def test_get_xy_grid_points(self):
        """Ensures correct output from get_xy_grid_points."""

        these_x_coords_metres, these_y_coords_metres = grids.get_xy_grid_points(
            x_min_metres=X_MIN_METRES, y_min_metres=Y_MIN_METRES,
            x_spacing_metres=X_SPACING_METRES,
            y_spacing_metres=Y_SPACING_METRES, num_rows=NUM_XY_ROWS,
            num_columns=NUM_XY_COLUMNS)

        self.assertTrue(numpy.allclose(
            these_x_coords_metres, POINT_X_COORDS_METRES, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            these_y_coords_metres, POINT_Y_COORDS_METRES, atol=TOLERANCE
        ))

    def test_get_xy_grid_cell_edges(self):
        """Ensures correct output from get_xy_grid_cell_edges."""

        these_x_coords_metres, these_y_coords_metres = (
            grids.get_xy_grid_cell_edges(
                x_min_metres=X_MIN_METRES, y_min_metres=Y_MIN_METRES,
                x_spacing_metres=X_SPACING_METRES,
                y_spacing_metres=Y_SPACING_METRES, num_rows=NUM_XY_ROWS,
                num_columns=NUM_XY_COLUMNS)
        )

        self.assertTrue(numpy.allclose(
            these_x_coords_metres, EDGE_X_COORDS_METRES, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            these_y_coords_metres, EDGE_Y_COORDS_METRES, atol=TOLERANCE
        ))

    def test_get_latlng_grid_points(self):
        """Ensures correct output from get_latlng_grid_points."""

        these_latitudes_deg, these_longitudes_deg = (
            grids.get_latlng_grid_points(
                min_latitude_deg=MIN_LATITUDE_DEG,
                min_longitude_deg=MIN_LONGITUDE_DEG,
                lat_spacing_deg=LATITUDE_SPACING_DEG,
                lng_spacing_deg=LONGITUDE_SPACING_DEG,
                num_rows=NUM_LATLNG_ROWS, num_columns=NUM_LATLNG_COLUMNS)
        )

        self.assertTrue(numpy.allclose(
            these_latitudes_deg, POINT_LATITUDES_DEG, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            these_longitudes_deg, POINT_LONGITUDES_DEG, atol=TOLERANCE
        ))

    def test_get_latlng_grid_cell_edges(self):
        """Ensures correct output from get_latlng_grid_cell_edges."""

        these_latitudes_deg, these_longitudes_deg = (
            grids.get_latlng_grid_cell_edges(
                min_latitude_deg=MIN_LATITUDE_DEG,
                min_longitude_deg=MIN_LONGITUDE_DEG,
                lat_spacing_deg=LATITUDE_SPACING_DEG,
                lng_spacing_deg=LONGITUDE_SPACING_DEG,
                num_rows=NUM_LATLNG_ROWS, num_columns=NUM_LATLNG_COLUMNS)
        )

        self.assertTrue(numpy.allclose(
            these_latitudes_deg, EDGE_LATITUDES_DEG, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            these_longitudes_deg, EDGE_LONGITUDES_DEG, atol=TOLERANCE
        ))

    def test_xy_vectors_to_matrices(self):
        """Ensures correct output from xy_vectors_to_matrices."""

        this_x_matrix_metres, this_y_matrix_metres = (
            grids.xy_vectors_to_matrices(
                x_unique_metres=POINT_X_COORDS_METRES,
                y_unique_metres=POINT_Y_COORDS_METRES)
        )

        self.assertTrue(numpy.allclose(
            this_x_matrix_metres, POINT_X_MATRIX_METRES, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            this_y_matrix_metres, POINT_Y_MATRIX_METRES, atol=TOLERANCE
        ))

    def test_latlng_vectors_to_matrices(self):
        """Ensures correct output from latlng_vectors_to_matrices."""

        this_latitude_matrix_deg, this_longitude_matrix_deg = (
            grids.latlng_vectors_to_matrices(
                unique_latitudes_deg=POINT_LATITUDES_DEG,
                unique_longitudes_deg=POINT_LONGITUDES_DEG)
        )

        self.assertTrue(numpy.allclose(
            this_latitude_matrix_deg, POINT_LATITUDE_MATRIX_DEG, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            this_longitude_matrix_deg, POINT_LONGITUDE_MATRIX_DEG,
            atol=TOLERANCE
        ))

    def test_latlng_field_grid_points_to_edges(self):
        """Ensures correct output from latlng_field_grid_points_to_edges."""

        this_field_matrix, these_latitudes_deg, these_longitudes_deg = (
            grids.latlng_field_grid_points_to_edges(
                field_matrix=FIELD_MATRIX_AT_POINTS,
                min_latitude_deg=MIN_LATITUDE_DEG,
                min_longitude_deg=MIN_LONGITUDE_DEG,
                lat_spacing_deg=LATITUDE_SPACING_DEG,
                lng_spacing_deg=LONGITUDE_SPACING_DEG)
        )

        self.assertTrue(numpy.allclose(
            these_latitudes_deg, EDGE_LATITUDES_DEG, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            these_longitudes_deg, EDGE_LONGITUDES_DEG, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            this_field_matrix, FIELD_MATRIX_AT_EDGES, equal_nan=True,
            atol=TOLERANCE
        ))

    def test_extract_latlng_subgrid(self):
        """Ensures correct output from extract_latlng_subgrid."""

        this_data_matrix, these_rows, these_columns = (
            grids.extract_latlng_subgrid(
                data_matrix=FULL_GRID_DATA_MATRIX,
                grid_point_latitudes_deg=FULL_GRID_LATITUDES_DEG,
                grid_point_longitudes_deg=FULL_GRID_LONGITUDES_DEG,
                center_latitude_deg=SUBGRID_CENTER_LATITUDE_DEG,
                center_longitude_deg=SUBGRID_CENTER_LONGITUDE_DEG,
                max_distance_from_center_metres=SUBGRID_MAX_DISTANCE_METRES)
        )

        self.assertTrue(numpy.allclose(
            this_data_matrix, SUBGRID_DATA_MATRIX, equal_nan=True,
            atol=TOLERANCE
        ))

        self.assertTrue(numpy.array_equal(these_rows, FULL_TO_SUBGRID_ROWS))
        self.assertTrue(numpy.array_equal(
            these_columns, FULL_TO_SUBGRID_COLUMNS
        ))

    def test_count_events_on_equidistant_grid_no_ids(self):
        """Ensures correct output from count_events_on_equidistant_grid.

        In this case event IDs are *not* included as input.
        """

        this_num_events_matrix, _ = grids.count_events_on_equidistant_grid(
            event_x_coords_metres=EVENT_X_COORDS_METRES,
            event_y_coords_metres=EVENT_Y_COORDS_METRES, integer_event_ids=None,
            grid_point_x_coords_metres=EVENT_GRID_POINTS_X_METRES,
            grid_point_y_coords_metres=EVENT_GRID_POINTS_Y_METRES)

        self.assertTrue(numpy.array_equal(
            this_num_events_matrix, NUM_EVENTS_MATRIX
        ))

    def test_count_events_on_equidistant_grid_with_ids(self):
        """Ensures correct output from count_events_on_equidistant_grid.

        In this case event IDs are included as input.
        """

        this_num_events_matrix, _ = grids.count_events_on_equidistant_grid(
            event_x_coords_metres=EVENT_X_COORDS_METRES,
            event_y_coords_metres=EVENT_Y_COORDS_METRES,
            integer_event_ids=INTEGER_EVENT_IDS,
            grid_point_x_coords_metres=EVENT_GRID_POINTS_X_METRES,
            grid_point_y_coords_metres=EVENT_GRID_POINTS_Y_METRES)

        self.assertTrue(numpy.array_equal(
            this_num_events_matrix, NUM_UNIQUE_EVENTS_MATRIX
        ))

    def test_radius_calgary_increasing_lat(self):
        """Ensures correct output from get_latlng_grid_points_in_radius.

        In this case, the test point is Calgary and latitude increases with row
        number.
        """

        these_rows, these_columns, _ = grids.get_latlng_grid_points_in_radius(
            grid_point_latitudes_deg=RADIUS_TEST_LATITUDES_DEG,
            grid_point_longitudes_deg=RADIUS_TEST_LONGITUDES_DEG,
            test_latitude_deg=CALGARY_LATITUDE_DEG,
            test_longitude_deg=CALGARY_LONGITUDE_DEG,
            effective_radius_metres=EFFECTIVE_RADIUS_METRES)

        self.assertTrue(numpy.array_equal(
            these_rows, CALGARY_ROWS_INCREASING_LAT
        ))
        self.assertTrue(numpy.array_equal(
            these_columns, CALGARY_COLUMNS_INCREASING_LAT
        ))

    def test_radius_calgary_decreasing_lat(self):
        """Ensures correct output from get_latlng_grid_points_in_radius.

        In this case, the test point is Calgary and latitude decreases with row
        number.
        """

        these_rows, these_columns, _ = grids.get_latlng_grid_points_in_radius(
            grid_point_latitudes_deg=RADIUS_TEST_LATITUDES_DEG[::-1],
            grid_point_longitudes_deg=RADIUS_TEST_LONGITUDES_DEG,
            test_latitude_deg=CALGARY_LATITUDE_DEG,
            test_longitude_deg=CALGARY_LONGITUDE_DEG,
            effective_radius_metres=EFFECTIVE_RADIUS_METRES)

        self.assertTrue(numpy.array_equal(
            these_rows, CALGARY_ROWS_DECREASING_LAT
        ))
        self.assertTrue(numpy.array_equal(
            these_columns, CALGARY_COLUMNS_DECREASING_LAT
        ))

    def test_radius_edmonton_increasing_lat(self):
        """Ensures correct output from get_latlng_grid_points_in_radius.

        In this case, the test point is Edmonton and latitude increases with row
        number.
        """

        these_rows, these_columns, _ = grids.get_latlng_grid_points_in_radius(
            grid_point_latitudes_deg=RADIUS_TEST_LATITUDES_DEG,
            grid_point_longitudes_deg=RADIUS_TEST_LONGITUDES_DEG,
            test_latitude_deg=EDMONTON_LATITUDE_DEG,
            test_longitude_deg=EDMONTON_LONGITUDE_DEG,
            effective_radius_metres=EFFECTIVE_RADIUS_METRES)

        self.assertTrue(numpy.array_equal(
            these_rows, EDMONTON_ROWS_INCREASING_LAT
        ))
        self.assertTrue(numpy.array_equal(
            these_columns, EDMONTON_COLUMNS_INCREASING_LAT
        ))

    def test_radius_edmonton_decreasing_lat(self):
        """Ensures correct output from get_latlng_grid_points_in_radius.

        In this case, the test point is Edmonton and latitude decreases with row
        number.
        """

        these_rows, these_columns, _ = grids.get_latlng_grid_points_in_radius(
            grid_point_latitudes_deg=RADIUS_TEST_LATITUDES_DEG[::-1],
            grid_point_longitudes_deg=RADIUS_TEST_LONGITUDES_DEG,
            test_latitude_deg=EDMONTON_LATITUDE_DEG,
            test_longitude_deg=EDMONTON_LONGITUDE_DEG,
            effective_radius_metres=EFFECTIVE_RADIUS_METRES)

        self.assertTrue(numpy.array_equal(
            these_rows, EDMONTON_ROWS_DECREASING_LAT
        ))
        self.assertTrue(numpy.array_equal(
            these_columns, EDMONTON_COLUMNS_DECREASING_LAT
        ))

    def test_find_events_in_grid_cell(self):
        """Ensures correct output from find_events_in_grid_cell."""

        num_grid_rows = len(EVENT_GRID_POINTS_Y_METRES)
        num_grid_columns = len(EVENT_GRID_POINTS_X_METRES)

        for i in range(num_grid_rows):
            for j in range(num_grid_columns):
                these_indices = grids.find_events_in_grid_cell(
                    event_x_coords_metres=EVENT_X_COORDS_METRES,
                    event_y_coords_metres=EVENT_Y_COORDS_METRES,
                    grid_edge_x_coords_metres=EVENT_GRID_EDGES_X_METRES,
                    grid_edge_y_coords_metres=EVENT_GRID_EDGES_Y_METRES,
                    row_index=i, column_index=j, verbose=False)

                self.assertTrue(numpy.array_equal(
                    these_indices, GRID_CELL_TO_EVENT_DICT[i, j]
                ))


if __name__ == '__main__':
    unittest.main()
