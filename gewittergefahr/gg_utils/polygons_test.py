"""Unit tests for polygons.py."""

import unittest
import numpy
import shapely.geometry
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import projections

TOLERANCE = 1e-6
TOLERANCE_DECIMAL_PLACE = 6

# The following constants are used to test _get_longest_inner_list.
SHORT_LIST = []
MEDIUM_LIST = [0, 1, 2, 3]
LONG_LIST = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
LIST_OF_LISTS = [SHORT_LIST, MEDIUM_LIST, LONG_LIST]

# The following constants are used to test _get_longest_simple_polygon.
VERTEX_X_METRES_SHORT = numpy.array([0., 4., 2., 0.])
VERTEX_Y_METRES_SHORT = numpy.array([0., 0., 4., 0.])
VERTEX_X_METRES_MEDIUM = numpy.array([0., 2., 4., 4., 2., 0., 0.])
VERTEX_Y_METRES_MEDIUM = numpy.array([0., -1., 0., 2., 3., 2., 0.])
VERTEX_X_METRES_LONG = numpy.array([0., 2., 2., 4., 4., 1., 1., 0., 0.])
VERTEX_Y_METRES_LONG = numpy.array([0., 0., -1., -1., 4., 4., 2., 2., 0.])

NAN_ARRAY = numpy.array([numpy.nan])
VERTEX_X_METRES_COMPLEX = numpy.concatenate((
    VERTEX_X_METRES_LONG, NAN_ARRAY, VERTEX_X_METRES_MEDIUM, NAN_ARRAY,
    VERTEX_X_METRES_SHORT))
VERTEX_Y_METRES_COMPLEX = numpy.concatenate((
    VERTEX_Y_METRES_LONG, NAN_ARRAY, VERTEX_Y_METRES_MEDIUM, NAN_ARRAY,
    VERTEX_Y_METRES_SHORT))

# The following constants are used to test _vertex_arrays_to_list.
VERTEX_LIST_LONG_METRES = [
    (0., 0.), (2., 0.), (2., -1.), (4., -1.), (4., 4.), (1., 4.), (1., 2.),
    (0., 2.), (0., 0.)]

# The following constants are used to test _get_edge_direction.
FIRST_VERTEX_ROW = 5
FIRST_VERTEX_COLUMN = 5
SECOND_VERTEX_ROW_UP = 4
SECOND_VERTEX_COLUMN_UP = 5
SECOND_VERTEX_ROW_DOWN = 6
SECOND_VERTEX_COLUMN_DOWN = 5
SECOND_VERTEX_ROW_RIGHT = 5
SECOND_VERTEX_COLUMN_RIGHT = 6
SECOND_VERTEX_ROW_LEFT = 5
SECOND_VERTEX_COLUMN_LEFT = 4
SECOND_VERTEX_ROW_UP_RIGHT = 4
SECOND_VERTEX_COLUMN_UP_RIGHT = 6
SECOND_VERTEX_ROW_UP_LEFT = 4
SECOND_VERTEX_COLUMN_UP_LEFT = 4
SECOND_VERTEX_ROW_DOWN_RIGHT = 6
SECOND_VERTEX_COLUMN_DOWN_RIGHT = 6
SECOND_VERTEX_ROW_DOWN_LEFT = 6
SECOND_VERTEX_COLUMN_DOWN_LEFT = 4

# The following constants are used to test _vertices_from_grid_points_to_edges.
VERTEX_ROWS_ONLY_ONE_ORIG = numpy.array([5])
VERTEX_COLUMNS_ONLY_ONE_ORIG = numpy.array([3])
VERTEX_ROWS_ONE_UNIQUE_ORIG = numpy.array([5, 5, 5])
VERTEX_COLUMNS_ONE_UNIQUE_ORIG = numpy.array([3, 3, 3])
VERTEX_ROWS_ONLY_ONE_NEW = numpy.array([5.5, 5.5, 4.5, 4.5, 5.5])
VERTEX_COLUMNS_ONLY_ONE_NEW = numpy.array([2.5, 3.5, 3.5, 2.5, 2.5])

# The following constants are used to test _remove_redundant_vertices,
# _vertices_from_grid_points_to_edges, and fix_probsevere_vertices.
VERTEX_ROWS_GRID_POINTS = numpy.array(
    [101, 102, 103, 103, 104, 102, 102, 101, 101])
VERTEX_ROWS_GRID_POINTS_COMPLEX = numpy.array(
    [101, 102, 103, 103, 104, 102, 102, 101, 101, numpy.nan, 0, 1, 1, 0, 0])
VERTEX_COLUMNS_GRID_POINTS = numpy.array(
    [501, 501, 502, 503, 504, 504, 503, 502, 501])
VERTEX_COLUMNS_GRID_POINTS_COMPLEX = numpy.array(
    [501, 501, 502, 503, 504, 504, 503, 502, 501, numpy.nan, 0, 0, 1, 1, 0])

VERTEX_ROWS_GRID_CELL_EDGES_REDUNDANT = numpy.array(
    [100.5, 102.5, 102.5, 103.5, 103.5, 103.5, 103.5, 103.5, 104.5, 104.5,
     102.5, 103.5, 104.5, 101.5, 101.5, 100.5, 100.5])
VERTEX_COLUMNS_GRID_CELL_EDGES_REDUNDANT = numpy.array(
    [500.5, 500.5, 501.5, 501.5, 503.5, 502.5, 501.5, 503.5, 503.5, 504.5,
     504.5, 504.5, 504.5, 504.5, 502.5, 502.5, 500.5])
VERTEX_ROWS_GRID_CELL_EDGES_NON_REDUNDANT = numpy.array(
    [100.5, 102.5, 102.5, 103.5, 103.5, 104.5, 104.5, 101.5, 101.5, 100.5,
     100.5])
VERTEX_COLUMNS_GRID_CELL_EDGES_NON_REDUNDANT = numpy.array(
    [500.5, 500.5, 501.5, 501.5, 503.5, 503.5, 504.5, 504.5, 502.5, 502.5,
     500.5])

# The following constants are used to test
# _patch_diag_connections_in_binary_matrix.
BINARY_MATRIX_2DIAG_CONNECTIONS = numpy.array([[0, 1, 1, 0, 0, 0, 0, 0],
                                               [0, 0, 1, 1, 0, 0, 0, 0],
                                               [0, 1, 1, 1, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 1, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 1, 1, 0],
                                               [0, 0, 0, 0, 1, 1, 1, 1]],
                                              dtype=bool)
BINARY_MATRIX_1DIAG_CONNECTION = numpy.array([[0, 1, 1, 0, 0, 0, 0, 0],
                                              [0, 0, 1, 1, 0, 0, 0, 0],
                                              [0, 1, 1, 1, 0, 0, 0, 0],
                                              [0, 0, 0, 1, 1, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 1, 1, 0],
                                              [0, 0, 0, 0, 1, 1, 1, 1]],
                                             dtype=bool)
BINARY_MATRIX_NO_DIAG_CONNECTIONS = numpy.array([[0, 1, 1, 0, 0, 0, 0, 0],
                                                 [0, 0, 1, 1, 0, 0, 0, 0],
                                                 [0, 1, 1, 1, 0, 0, 0, 0],
                                                 [0, 0, 0, 1, 1, 0, 0, 0],
                                                 [0, 0, 0, 0, 1, 1, 1, 0],
                                                 [0, 0, 0, 0, 1, 1, 1, 1]],
                                                dtype=bool)

# The following constants are used to test grid_points_in_poly_to_binary_matrix,
# grid_points_in_poly_to_vertices, and simple_polygon_to_grid_points.
ROW_INDICES_IN_POLYGON = numpy.array(
    [101, 101, 102, 102, 102, 102, 103, 103, 103, 104])
COLUMN_INDICES_IN_POLYGON = numpy.array(
    [501, 502, 501, 502, 503, 504, 502, 503, 504, 504])

POINT_IN_OR_ON_POLYGON_MATRIX = numpy.array([[0, 0, 0, 0, 0, 0],
                                             [0, 1, 1, 0, 0, 0],
                                             [0, 1, 1, 1, 1, 0],
                                             [0, 0, 1, 1, 1, 0],
                                             [0, 0, 0, 0, 1, 0],
                                             [0, 0, 0, 0, 0, 0]]).astype(bool)
FIRST_ROW_INDEX = 100
FIRST_COLUMN_INDEX = 500

# The following constants are used to test project_latlng_to_xy and
# project_xy_to_latlng.
EXTERIOR_VERTEX_LATITUDES_DEG = numpy.array([49., 49., 60., 60., 53.8, 49.])
EXTERIOR_VERTEX_LONGITUDES_DEG = numpy.array(
    [246., 250., 250., 240., 240., 246.])
HOLE1_VERTEX_LATITUDES_DEG = numpy.array(
    [51.1, 52.2, 52.2, 53.3, 53.3, 51.1, 51.1])
HOLE1_VERTEX_LONGITUDES_DEG = numpy.array(
    [246., 246., 246.1, 246.1, 246.4, 246.4, 246.])

POLYGON_OBJECT_LATLNG = polygons.vertex_arrays_to_polygon_object(
    EXTERIOR_VERTEX_LONGITUDES_DEG, EXTERIOR_VERTEX_LATITUDES_DEG,
    hole_x_coords_list=[HOLE1_VERTEX_LONGITUDES_DEG],
    hole_y_coords_list=[HOLE1_VERTEX_LATITUDES_DEG])
PROJECTION_OBJECT = projections.init_azimuthal_equidistant_projection(
    central_latitude_deg=55., central_longitude_deg=245.)

# The following constants are used to test vertex_arrays_to_polygon_object and
# polygon_object_to_vertex_arrays.
EXTERIOR_VERTEX_X_METRES = numpy.array([0., 0., 10., 10., 0.])
EXTERIOR_VERTEX_Y_METRES = numpy.array([0., 10., 10., 0., 0.])
EXTERIOR_VERTEX_METRES_LIST = [
    (0., 0.), (0., 10.), (10., 10.), (10., 0.), (0., 0.)]

HOLE1_VERTEX_X_METRES = numpy.array([2., 2., 4., 4., 2.])
HOLE1_VERTEX_Y_METRES = numpy.array([2., 4., 4., 2., 2.])
HOLE1_VERTEX_METRES_LIST = [(2., 2.), (2., 4.), (4., 4.), (4., 2.), (2., 2.)]

HOLE2_VERTEX_X_METRES = numpy.array([6., 6., 8., 8., 6.])
HOLE2_VERTEX_Y_METRES = numpy.array([6., 8., 8., 6., 6.])
HOLE2_VERTEX_METRES_LIST = [(6., 6.), (6., 8.), (8., 8.), (8., 6.), (6., 6.)]

POLYGON_OBJECT_2HOLES_XY_METRES = shapely.geometry.Polygon(
    shell=EXTERIOR_VERTEX_METRES_LIST,
    holes=(HOLE1_VERTEX_METRES_LIST, HOLE2_VERTEX_METRES_LIST))

# The following constants are used to test simple_polygon_to_grid_points.
VERTEX_ROWS_SIMPLE = numpy.array(
    [3.5, 3.5, 4.5, 4.5, -0.5, -0.5, 1.5, 1.5, 3.5])
VERTEX_COLUMNS_SIMPLE = numpy.array(
    [-0.5, 1.5, 1.5, 3.5, 3.5, 0.5, 0.5, -0.5, -0.5])
GRID_POINT_ROWS_IN_SIMPLE_POLY = numpy.array(
    [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4])
GRID_POINT_COLUMNS_IN_SIMPLE_POLY = numpy.array(
    [1, 2, 3, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 2, 3])

# The following constants are used to test point_in_or_on_polygon and
# buffer_simple_polygon.
SMALL_BUFFER_DIST_METRES = 2.5
LARGE_BUFFER_DIST_METRES = 5.

SMALL_BUFFER_VERTEX_X_METRES = numpy.array([-2.5, -2.5, 12.5, 12.5, -2.5])
SMALL_BUFFER_VERTEX_Y_METRES = numpy.array([-2.5, 12.5, 12.5, -2.5, -2.5])
LARGE_BUFFER_VERTEX_X_METRES = numpy.array([-5., -5., 15., 15., -5.])
LARGE_BUFFER_VERTEX_Y_METRES = numpy.array([-5., 15., 15., -5., -5.])

EXCLUSIVE_BUFFER_EXTERIOR_X_METRES = numpy.array([-5., -5., 15., 15., -5.])
EXCLUSIVE_BUFFER_EXTERIOR_Y_METRES = numpy.array([-5., 15., 15., -5., -5.])
EXCLUSIVE_BUFFER_HOLE_X_METRES = numpy.array([-2.5, -2.5, 12.5, 12.5, -2.5])
EXCLUSIVE_BUFFER_HOLE_Y_METRES = numpy.array([-2.5, 12.5, 12.5, -2.5, -2.5])

POLYGON_OBJECT_EXCL_BUFFER_XY_METRES = polygons.vertex_arrays_to_polygon_object(
    EXCLUSIVE_BUFFER_EXTERIOR_X_METRES, EXCLUSIVE_BUFFER_EXTERIOR_Y_METRES,
    hole_x_coords_list=[EXCLUSIVE_BUFFER_HOLE_X_METRES],
    hole_y_coords_list=[EXCLUSIVE_BUFFER_HOLE_Y_METRES])

X_IN_NESTED_BUFFER = -4.
X_ON_NESTED_BUFFER = -2.5
X_OUTSIDE_NESTED_BUFFER = 3.
Y_IN_NESTED_BUFFER = 5.
Y_ON_NESTED_BUFFER = 5.
Y_OUTSIDE_NESTED_BUFFER = 5.


class PolygonsTests(unittest.TestCase):
    """Each method is a unit test for polygons.py."""

    def test_get_longest_inner_list(self):
        """Ensures correct output from _get_longest_inner_list."""

        this_longest_list = polygons._get_longest_inner_list(LIST_OF_LISTS)
        self.assertTrue(this_longest_list == LONG_LIST)

    def test_get_longest_simple_polygon_complex(self):
        """Ensures correct output from _get_longest_simple_polygon.

        In this case the input is a complex polygon.
        """

        these_vertex_x_metres, these_vertex_y_metres = (
            polygons._get_longest_simple_polygon(
                VERTEX_X_METRES_COMPLEX, VERTEX_Y_METRES_COMPLEX))

        self.assertTrue(numpy.allclose(
            these_vertex_x_metres, VERTEX_X_METRES_LONG, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_vertex_y_metres, VERTEX_Y_METRES_LONG, atol=TOLERANCE))

    def test_get_longest_simple_polygon_simple(self):
        """Ensures correct output from _get_longest_simple_polygon.

        In this case the input is a complex polygon.
        """

        these_vertex_x_metres, these_vertex_y_metres = (
            polygons._get_longest_simple_polygon(
                VERTEX_X_METRES_LONG, VERTEX_Y_METRES_LONG))

        self.assertTrue(numpy.allclose(
            these_vertex_x_metres, VERTEX_X_METRES_LONG, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_vertex_y_metres, VERTEX_Y_METRES_LONG, atol=TOLERANCE))

    def test_vertex_arrays_to_list(self):
        """Ensures correct output from _vertex_arrays_to_list."""

        this_vertex_list_metres = polygons._vertex_arrays_to_list(
            VERTEX_X_METRES_LONG, VERTEX_Y_METRES_LONG)
        self.assertTrue(
            len(this_vertex_list_metres) == len(VERTEX_LIST_LONG_METRES))

        for i in range(len(VERTEX_LIST_LONG_METRES)):
            self.assertTrue(numpy.allclose(
                numpy.asarray(this_vertex_list_metres[i]),
                numpy.asarray(VERTEX_LIST_LONG_METRES[i]), atol=TOLERANCE,
                equal_nan=True))

    def test_get_edge_direction_up(self):
        """Ensures correct output from _get_edge_direction.

        In this case, direction is up.
        """

        this_direction_name = polygons._get_edge_direction(
            first_row=FIRST_VERTEX_ROW, second_row=SECOND_VERTEX_ROW_UP,
            first_column=FIRST_VERTEX_COLUMN,
            second_column=SECOND_VERTEX_COLUMN_UP)
        self.assertTrue(this_direction_name == polygons.UP_DIRECTION_NAME)

    def test_get_edge_direction_down(self):
        """Ensures correct output from _get_edge_direction.

        In this case, direction is down.
        """

        this_direction_name = polygons._get_edge_direction(
            first_row=FIRST_VERTEX_ROW, second_row=SECOND_VERTEX_ROW_DOWN,
            first_column=FIRST_VERTEX_COLUMN,
            second_column=SECOND_VERTEX_COLUMN_DOWN)
        self.assertTrue(this_direction_name == polygons.DOWN_DIRECTION_NAME)

    def test_get_edge_direction_left(self):
        """Ensures correct output from _get_edge_direction.

        In this case, direction is left.
        """

        this_direction_name = polygons._get_edge_direction(
            first_row=FIRST_VERTEX_ROW, second_row=SECOND_VERTEX_ROW_LEFT,
            first_column=FIRST_VERTEX_COLUMN,
            second_column=SECOND_VERTEX_COLUMN_LEFT)
        self.assertTrue(this_direction_name == polygons.LEFT_DIRECTION_NAME)

    def test_get_edge_direction_right(self):
        """Ensures correct output from _get_edge_direction.

        In this case, direction is right.
        """

        this_direction_name = polygons._get_edge_direction(
            first_row=FIRST_VERTEX_ROW, second_row=SECOND_VERTEX_ROW_RIGHT,
            first_column=FIRST_VERTEX_COLUMN,
            second_column=SECOND_VERTEX_COLUMN_RIGHT)
        self.assertTrue(this_direction_name == polygons.RIGHT_DIRECTION_NAME)

    def test_get_edge_direction_up_left(self):
        """Ensures correct output from _get_edge_direction.

        In this case, direction is up-left.
        """

        this_direction_name = polygons._get_edge_direction(
            first_row=FIRST_VERTEX_ROW, second_row=SECOND_VERTEX_ROW_UP_LEFT,
            first_column=FIRST_VERTEX_COLUMN,
            second_column=SECOND_VERTEX_COLUMN_UP_LEFT)
        self.assertTrue(this_direction_name == polygons.UP_LEFT_DIRECTION_NAME)

    def test_get_edge_direction_up_right(self):
        """Ensures correct output from _get_edge_direction.

        In this case, direction is up-right.
        """

        this_direction_name = polygons._get_edge_direction(
            first_row=FIRST_VERTEX_ROW, second_row=SECOND_VERTEX_ROW_UP_RIGHT,
            first_column=FIRST_VERTEX_COLUMN,
            second_column=SECOND_VERTEX_COLUMN_UP_RIGHT)
        self.assertTrue(this_direction_name == polygons.UP_RIGHT_DIRECTION_NAME)

    def test_get_edge_direction_down_left(self):
        """Ensures correct output from _get_edge_direction.

        In this case, direction is down-left.
        """

        this_direction_name = polygons._get_edge_direction(
            first_row=FIRST_VERTEX_ROW, second_row=SECOND_VERTEX_ROW_DOWN_LEFT,
            first_column=FIRST_VERTEX_COLUMN,
            second_column=SECOND_VERTEX_COLUMN_DOWN_LEFT)
        self.assertTrue(
            this_direction_name == polygons.DOWN_LEFT_DIRECTION_NAME)

    def test_get_edge_direction_down_right(self):
        """Ensures correct output from _get_edge_direction.

        In this case, direction is down-right.
        """

        this_direction_name = polygons._get_edge_direction(
            first_row=FIRST_VERTEX_ROW, second_row=SECOND_VERTEX_ROW_DOWN_RIGHT,
            first_column=FIRST_VERTEX_COLUMN,
            second_column=SECOND_VERTEX_COLUMN_DOWN_RIGHT)
        self.assertTrue(
            this_direction_name == polygons.DOWN_RIGHT_DIRECTION_NAME)

    def test_remove_redundant_vertices(self):
        """Ensures correct output from _remove_redundant_vertices."""

        these_vertex_rows, these_vertex_columns = (
            polygons._remove_redundant_vertices(
                VERTEX_ROWS_GRID_CELL_EDGES_REDUNDANT,
                VERTEX_COLUMNS_GRID_CELL_EDGES_REDUNDANT))

        self.assertTrue(numpy.array_equal(
            these_vertex_rows, VERTEX_ROWS_GRID_CELL_EDGES_NON_REDUNDANT))
        self.assertTrue(numpy.array_equal(
            these_vertex_columns, VERTEX_COLUMNS_GRID_CELL_EDGES_NON_REDUNDANT))

    def test_patch_diag_connections_in_binary_matrix_2(self):
        """Ensures correct output from _patch_diag_connections_in_binary_matrix.

        In this case there are 2 diagonal connections to patch.
        """

        this_binary_matrix = polygons._patch_diag_connections_in_binary_matrix(
            BINARY_MATRIX_2DIAG_CONNECTIONS)
        self.assertTrue(numpy.array_equal(
            this_binary_matrix, BINARY_MATRIX_NO_DIAG_CONNECTIONS))

    def test_patch_diag_connections_in_binary_matrix_1(self):
        """Ensures correct output from _patch_diag_connections_in_binary_matrix.

        In this case there is one diagonal connection to patch.
        """

        this_binary_matrix = polygons._patch_diag_connections_in_binary_matrix(
            BINARY_MATRIX_1DIAG_CONNECTION)
        self.assertTrue(numpy.array_equal(
            this_binary_matrix, BINARY_MATRIX_NO_DIAG_CONNECTIONS))

    def test_patch_diag_connections_in_binary_matrix_0(self):
        """Ensures correct output from _patch_diag_connections_in_binary_matrix.

        In this case there are no diagonal connections to patch.
        """

        this_binary_matrix = polygons._patch_diag_connections_in_binary_matrix(
            BINARY_MATRIX_NO_DIAG_CONNECTIONS)
        self.assertTrue(numpy.array_equal(
            this_binary_matrix, BINARY_MATRIX_NO_DIAG_CONNECTIONS))

    def test_vertices_from_grid_points_to_edges_many_inputs(self):
        """Ensures correct output from _vertices_from_grid_points_to_edges.

        In this case there are many, unique input vertices.

        This is an integration test, because it also depends on
        _remove_redundant_vertices.  This makes more sense than testing
        _vertices_from_grid_points_to_edges alone, because the expected output
        is hard to define (involves redundant vertices).
        """

        these_vertex_rows_redundant, these_vertex_columns_redundant = (
            polygons._vertices_from_grid_points_to_edges(
                VERTEX_ROWS_GRID_POINTS, VERTEX_COLUMNS_GRID_POINTS))

        these_vertex_rows_non_redundant, these_vertex_columns_non_redundant = (
            polygons._remove_redundant_vertices(
                these_vertex_rows_redundant, these_vertex_columns_redundant))

        self.assertTrue(numpy.array_equal(
            these_vertex_rows_non_redundant,
            VERTEX_ROWS_GRID_CELL_EDGES_NON_REDUNDANT))
        self.assertTrue(numpy.array_equal(
            these_vertex_columns_non_redundant,
            VERTEX_COLUMNS_GRID_CELL_EDGES_NON_REDUNDANT))

    def test_vertices_from_grid_points_to_edges_one_input(self):
        """Ensures correct output from _vertices_from_grid_points_to_edges.

        In this case there is only one input vertex.
        """

        these_vertex_rows, these_vertex_columns = (
            polygons._vertices_from_grid_points_to_edges(
                VERTEX_ROWS_ONLY_ONE_ORIG,
                VERTEX_COLUMNS_ONLY_ONE_ORIG))

        self.assertTrue(numpy.array_equal(
            these_vertex_rows, VERTEX_ROWS_ONLY_ONE_NEW))
        self.assertTrue(numpy.array_equal(
            these_vertex_columns, VERTEX_COLUMNS_ONLY_ONE_NEW))

    def test_vertices_from_grid_points_to_edges_one_unique_input(self):
        """Ensures correct output from _vertices_from_grid_points_to_edges.

        In this case, although it looks like there are many input vertices,
        there is only one unique input vertex.
        """

        these_vertex_rows, these_vertex_columns = (
            polygons._vertices_from_grid_points_to_edges(
                VERTEX_ROWS_ONE_UNIQUE_ORIG,
                VERTEX_COLUMNS_ONE_UNIQUE_ORIG))

        self.assertTrue(numpy.array_equal(
            these_vertex_rows, VERTEX_ROWS_ONLY_ONE_NEW))
        self.assertTrue(numpy.array_equal(
            these_vertex_columns, VERTEX_COLUMNS_ONLY_ONE_NEW))

    def test_project_latlng_to_xy(self):
        """Ensures correct output from project_latlng_to_xy.

        This is an integration test, because it also depends on
        project_xy_to_latlng.  This makes more sense than testing
        project_latlng_to_xy alone, because the expected output is hard to
        define (involves complicated projection equations).
        """

        this_polygon_object_xy, this_projection_object = (
            polygons.project_latlng_to_xy(POLYGON_OBJECT_LATLNG))
        this_polygon_object_latlng = polygons.project_xy_to_latlng(
            this_polygon_object_xy, this_projection_object)
        self.assertTrue(this_polygon_object_latlng.almost_equals(
            POLYGON_OBJECT_LATLNG, decimal=TOLERANCE_DECIMAL_PLACE))

    def test_project_xy_to_latlng(self):
        """Ensures correct output from project_xy_to_latlng.

        This is an integration test, because it also depends on
        project_latlng_to_xy.  This makes more sense than testing
        project_xy_to_latlng alone, because the expected output is hard to
        define (involves complicated projection equations).
        """

        this_polygon_object_latlng = polygons.project_xy_to_latlng(
            POLYGON_OBJECT_2HOLES_XY_METRES, PROJECTION_OBJECT)
        this_polygon_object_xy, _ = polygons.project_latlng_to_xy(
            this_polygon_object_latlng, PROJECTION_OBJECT)
        self.assertTrue(this_polygon_object_xy.almost_equals(
            POLYGON_OBJECT_2HOLES_XY_METRES, decimal=TOLERANCE_DECIMAL_PLACE))

    def test_grid_points_in_poly_to_binary_matrix(self):
        """Ensures correct output from grid_points_in_poly_to_binary_matrix."""

        this_binary_matrix, this_first_row_index, this_first_column_index = (
            polygons.grid_points_in_poly_to_binary_matrix(
                ROW_INDICES_IN_POLYGON, COLUMN_INDICES_IN_POLYGON))

        self.assertTrue(numpy.array_equal(
            this_binary_matrix, POINT_IN_OR_ON_POLYGON_MATRIX))
        self.assertTrue(this_first_row_index == FIRST_ROW_INDEX)
        self.assertTrue(this_first_column_index == FIRST_COLUMN_INDEX)

    def test_sort_counterclockwise_already_ccw(self):
        """Ensures correct output from sort_counterclockwise.

        In this case, vertices are already sorted counterclockwise.
        """

        these_vertex_x_metres, these_vertex_y_metres = (
            polygons.sort_counterclockwise(
                VERTEX_X_METRES_LONG, VERTEX_Y_METRES_LONG))

        self.assertTrue(numpy.allclose(
            these_vertex_x_metres, VERTEX_X_METRES_LONG, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_vertex_y_metres, VERTEX_Y_METRES_LONG, atol=TOLERANCE))

    def test_sort_counterclockwise_not_ccw(self):
        """Ensures correct output from sort_counterclockwise.

        In this case, input vertices are sorted clockwise.
        """

        these_vertex_x_metres, these_vertex_y_metres = (
            polygons.sort_counterclockwise(
                VERTEX_X_METRES_LONG[::-1], VERTEX_Y_METRES_LONG[::-1]))

        self.assertTrue(numpy.allclose(
            these_vertex_x_metres, VERTEX_X_METRES_LONG, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_vertex_y_metres, VERTEX_Y_METRES_LONG, atol=TOLERANCE))

    def test_vertex_arrays_to_polygon_object(self):
        """Ensures correct output from vertex_arrays_to_polygon_object."""

        this_polygon_object = polygons.vertex_arrays_to_polygon_object(
            EXTERIOR_VERTEX_X_METRES, EXTERIOR_VERTEX_Y_METRES,
            hole_x_coords_list=[HOLE1_VERTEX_X_METRES, HOLE2_VERTEX_X_METRES],
            hole_y_coords_list=[HOLE1_VERTEX_Y_METRES, HOLE2_VERTEX_Y_METRES])

        self.assertTrue(this_polygon_object.almost_equals(
            POLYGON_OBJECT_2HOLES_XY_METRES, decimal=TOLERANCE_DECIMAL_PLACE))

    def test_polygon_object_to_vertex_arrays(self):
        """Ensures correct output from polygon_object_to_vertex_arrays."""

        this_vertex_dict = polygons.polygon_object_to_vertex_arrays(
            POLYGON_OBJECT_2HOLES_XY_METRES)

        self.assertTrue(numpy.allclose(
            this_vertex_dict[polygons.EXTERIOR_X_COLUMN],
            EXTERIOR_VERTEX_X_METRES, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            this_vertex_dict[polygons.EXTERIOR_Y_COLUMN],
            EXTERIOR_VERTEX_Y_METRES, atol=TOLERANCE))

        self.assertTrue(
            len(this_vertex_dict[polygons.HOLE_X_COLUMN]) == 2)
        self.assertTrue(numpy.allclose(
            this_vertex_dict[polygons.HOLE_X_COLUMN][0], HOLE1_VERTEX_X_METRES,
            atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            this_vertex_dict[polygons.HOLE_Y_COLUMN][0], HOLE1_VERTEX_Y_METRES,
            atol=TOLERANCE))

        self.assertTrue(numpy.allclose(
            this_vertex_dict[polygons.HOLE_X_COLUMN][1], HOLE2_VERTEX_X_METRES,
            atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            this_vertex_dict[polygons.HOLE_Y_COLUMN][1], HOLE2_VERTEX_Y_METRES,
            atol=TOLERANCE))

    def test_grid_points_in_poly_to_vertices(self):
        """Ensures correct output from grid_points_in_poly_to_vertices."""

        these_vertex_rows, these_vertex_columns = (
            polygons.grid_points_in_poly_to_vertices(
                ROW_INDICES_IN_POLYGON, COLUMN_INDICES_IN_POLYGON))

        self.assertTrue(numpy.array_equal(
            these_vertex_rows, VERTEX_ROWS_GRID_CELL_EDGES_NON_REDUNDANT))
        self.assertTrue(numpy.array_equal(
            these_vertex_columns, VERTEX_COLUMNS_GRID_CELL_EDGES_NON_REDUNDANT))

    def test_simple_polygon_to_grid_points(self):
        """Ensures correct output from simple_polygon_to_grid_points."""

        these_grid_point_rows, these_grid_point_columns = (
            polygons.simple_polygon_to_grid_points(
                VERTEX_ROWS_SIMPLE, VERTEX_COLUMNS_SIMPLE))

        self.assertTrue(numpy.array_equal(
            these_grid_point_rows, GRID_POINT_ROWS_IN_SIMPLE_POLY))
        self.assertTrue(numpy.array_equal(
            these_grid_point_columns, GRID_POINT_COLUMNS_IN_SIMPLE_POLY))

    def test_fix_probsevere_vertices_simple(self):
        """Ensures correct output from fix_probsevere_vertices.

        In this case the input polygon is simple.
        """

        these_vertex_rows, these_vertex_columns = (
            polygons.fix_probsevere_vertices(
                VERTEX_ROWS_GRID_POINTS, VERTEX_COLUMNS_GRID_POINTS))

        self.assertTrue(numpy.allclose(
            these_vertex_rows, VERTEX_ROWS_GRID_CELL_EDGES_NON_REDUNDANT,
            atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_vertex_columns, VERTEX_COLUMNS_GRID_CELL_EDGES_NON_REDUNDANT,
            atol=TOLERANCE))

    def test_fix_probsevere_vertices_complex(self):
        """Ensures correct output from fix_probsevere_vertices.

        In this case the input polygon is complex.
        """

        these_vertex_rows, these_vertex_columns = (
            polygons.fix_probsevere_vertices(
                VERTEX_ROWS_GRID_POINTS_COMPLEX,
                VERTEX_COLUMNS_GRID_POINTS_COMPLEX))

        self.assertTrue(numpy.allclose(
            these_vertex_rows, VERTEX_ROWS_GRID_CELL_EDGES_NON_REDUNDANT,
            atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_vertex_columns, VERTEX_COLUMNS_GRID_CELL_EDGES_NON_REDUNDANT,
            atol=TOLERANCE))

    def test_point_in_or_on_polygon_false(self):
        """Ensures correct output from point_in_or_on_polygon.

        In this case, answer = False.
        """

        this_flag = polygons.point_in_or_on_polygon(
            POLYGON_OBJECT_EXCL_BUFFER_XY_METRES,
            query_x_coordinate=X_OUTSIDE_NESTED_BUFFER,
            query_y_coordinate=Y_OUTSIDE_NESTED_BUFFER)
        self.assertFalse(this_flag)

    def test_point_in_or_on_polygon_touching(self):
        """Ensures correct output from point_in_or_on_polygon.

        In this case, answer = True (point touches polygon).
        """

        this_flag = polygons.point_in_or_on_polygon(
            POLYGON_OBJECT_EXCL_BUFFER_XY_METRES,
            query_x_coordinate=X_ON_NESTED_BUFFER,
            query_y_coordinate=Y_ON_NESTED_BUFFER)
        self.assertTrue(this_flag)

    def test_point_in_or_on_polygon_inside(self):
        """Ensures correct output from point_in_or_on_polygon.

        In this case, answer = True (point inside polygon).
        """

        this_flag = polygons.point_in_or_on_polygon(
            POLYGON_OBJECT_EXCL_BUFFER_XY_METRES,
            query_x_coordinate=X_IN_NESTED_BUFFER,
            query_y_coordinate=Y_IN_NESTED_BUFFER)
        self.assertTrue(this_flag)

    def test_buffer_simple_polygon_small_inclusive(self):
        """Ensures correct output from buffer_simple_polygon.

        In this case the buffer is small and inclusive (includes original
        polygon).
        """

        this_buffered_polygon_object = polygons.buffer_simple_polygon(
            vertex_x_metres=EXTERIOR_VERTEX_X_METRES,
            vertex_y_metres=EXTERIOR_VERTEX_Y_METRES,
            max_buffer_dist_metres=SMALL_BUFFER_DIST_METRES,
            preserve_angles=True)

        this_buffer_vertex_dict = polygons.polygon_object_to_vertex_arrays(
            this_buffered_polygon_object)

        self.assertTrue(numpy.allclose(
            this_buffer_vertex_dict[polygons.EXTERIOR_X_COLUMN],
            SMALL_BUFFER_VERTEX_X_METRES, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            this_buffer_vertex_dict[polygons.EXTERIOR_Y_COLUMN],
            SMALL_BUFFER_VERTEX_Y_METRES, atol=TOLERANCE))

        self.assertTrue(
            len(this_buffer_vertex_dict[polygons.HOLE_X_COLUMN]) == 0)
        self.assertTrue(
            len(this_buffer_vertex_dict[polygons.HOLE_Y_COLUMN]) == 0)

    def test_buffer_simple_polygon_large_inclusive(self):
        """Ensures correct output from buffer_simple_polygon.

        In this case the buffer is large and inclusive (includes original
        polygon).
        """

        this_buffered_polygon_object = polygons.buffer_simple_polygon(
            vertex_x_metres=EXTERIOR_VERTEX_X_METRES,
            vertex_y_metres=EXTERIOR_VERTEX_Y_METRES,
            max_buffer_dist_metres=LARGE_BUFFER_DIST_METRES,
            preserve_angles=True)

        this_buffer_vertex_dict = polygons.polygon_object_to_vertex_arrays(
            this_buffered_polygon_object)

        self.assertTrue(numpy.allclose(
            this_buffer_vertex_dict[polygons.EXTERIOR_X_COLUMN],
            LARGE_BUFFER_VERTEX_X_METRES, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            this_buffer_vertex_dict[polygons.EXTERIOR_Y_COLUMN],
            LARGE_BUFFER_VERTEX_Y_METRES, atol=TOLERANCE))

        self.assertTrue(
            len(this_buffer_vertex_dict[polygons.HOLE_X_COLUMN]) == 0)
        self.assertTrue(
            len(this_buffer_vertex_dict[polygons.HOLE_Y_COLUMN]) == 0)

    def test_buffer_simple_polygon_exclusive(self):
        """Ensures correct output from buffer_simple_polygon.

        In this case the buffer is exclusive (does not include original
        polygon).
        """

        this_buffer_polygon_object = polygons.buffer_simple_polygon(
            EXTERIOR_VERTEX_X_METRES, EXTERIOR_VERTEX_Y_METRES,
            min_buffer_dist_metres=SMALL_BUFFER_DIST_METRES,
            max_buffer_dist_metres=LARGE_BUFFER_DIST_METRES,
            preserve_angles=True)

        this_buffer_vertex_dict = polygons.polygon_object_to_vertex_arrays(
            this_buffer_polygon_object)

        self.assertTrue(numpy.allclose(
            this_buffer_vertex_dict[polygons.EXTERIOR_X_COLUMN],
            EXCLUSIVE_BUFFER_EXTERIOR_X_METRES, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            this_buffer_vertex_dict[polygons.EXTERIOR_Y_COLUMN],
            EXCLUSIVE_BUFFER_EXTERIOR_Y_METRES, atol=TOLERANCE))

        self.assertTrue(
            len(this_buffer_vertex_dict[polygons.HOLE_X_COLUMN]) == 1)
        self.assertTrue(
            len(this_buffer_vertex_dict[polygons.HOLE_Y_COLUMN]) == 1)
        self.assertTrue(numpy.allclose(
            this_buffer_vertex_dict[polygons.HOLE_X_COLUMN][0],
            EXCLUSIVE_BUFFER_HOLE_X_METRES, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            this_buffer_vertex_dict[polygons.HOLE_Y_COLUMN][0],
            EXCLUSIVE_BUFFER_HOLE_Y_METRES, atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
