"""Unit tests for human_polygons.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import human_polygons

TOLERANCE = 1e-6
TOLERANCE_NUM_DECIMAL_PLACES = 6

# The following constants are used to test _vertex_list_to_polygon_list and
# _polygon_list_to_vertex_list.
TOY_VERTEX_ROWS = numpy.array([
    0, 5, 5, 0, 0, numpy.nan, 0, 0, 10, 10, 0, numpy.nan, -10, 20, 20, -10, -10
])
TOY_VERTEX_COLUMNS = numpy.array([
    0, 0, 10, 10, 0, numpy.nan, 0, 5, 5, 0, 0, numpy.nan, 40, 40, 60, 60, 40
])

THIS_FIRST_POLYGON_OBJECT = polygons.vertex_arrays_to_polygon_object(
    exterior_x_coords=TOY_VERTEX_COLUMNS[:5],
    exterior_y_coords=TOY_VERTEX_ROWS[:5]
)

THIS_SECOND_POLYGON_OBJECT = polygons.vertex_arrays_to_polygon_object(
    exterior_x_coords=TOY_VERTEX_COLUMNS[6:11],
    exterior_y_coords=TOY_VERTEX_ROWS[6:11]
)

THIS_THIRD_POLYGON_OBJECT = polygons.vertex_arrays_to_polygon_object(
    exterior_x_coords=TOY_VERTEX_COLUMNS[12:],
    exterior_y_coords=TOY_VERTEX_ROWS[12:]
)

TOY_POLYGON_OBJECTS = [
    THIS_FIRST_POLYGON_OBJECT, THIS_SECOND_POLYGON_OBJECT,
    THIS_THIRD_POLYGON_OBJECT
]

TOY_VERTEX_TO_POLY_INDICES = numpy.array(
    [0, 0, 0, 0, 0, -1, 1, 1, 1, 1, 1, -1, 2, 2, 2, 2, 2], dtype=int
)

TOY_POLY_TO_FIRST_VERTEX_INDICES = numpy.array([0, 6, 12], dtype=int)

# The following constants are used to test polygons_from_pixel_to_grid_coords.
NUM_GRID_ROWS = 24
NUM_GRID_COLUMNS = 32
NUM_PANEL_ROWS = 3
NUM_PANEL_COLUMNS = 2
NUM_PIXEL_ROWS = 3000
NUM_PIXEL_COLUMNS = 3000

THESE_X_COORDS = numpy.array([-0.5, 149.5, 299.5, 149.5, -0.5])
THESE_Y_COORDS = numpy.array([99.5, -0.5, 99.5, 199.5, 99.5])
FIRST_POLYGON_OBJECT_XY = polygons.vertex_arrays_to_polygon_object(
    exterior_x_coords=THESE_X_COORDS, exterior_y_coords=THESE_Y_COORDS)

SECOND_POLYGON_OBJECT_XY = polygons.vertex_arrays_to_polygon_object(
    exterior_x_coords=THESE_X_COORDS + 1500, exterior_y_coords=THESE_Y_COORDS)

THIRD_POLYGON_OBJECT_XY = polygons.vertex_arrays_to_polygon_object(
    exterior_x_coords=THESE_X_COORDS, exterior_y_coords=THESE_Y_COORDS + 1000)

FOURTH_POLYGON_OBJECT_XY = polygons.vertex_arrays_to_polygon_object(
    exterior_x_coords=THESE_X_COORDS + 1500,
    exterior_y_coords=THESE_Y_COORDS + 2000)

THESE_X_COORDS = numpy.array([
    1299.5, 1452.625, 1452.625, 1452.625, 1299.5, 1299.5
])
THESE_Y_COORDS = numpy.array([
    957 + 5. / 6, 957 + 5. / 6, 799.5, 599.5, 599.5, 957 + 5. / 6
])

FIFTH_POLYGON_OBJECT_XY = polygons.vertex_arrays_to_polygon_object(
    exterior_x_coords=THESE_X_COORDS, exterior_y_coords=THESE_Y_COORDS)

SIXTH_POLYGON_OBJECT_XY = polygons.vertex_arrays_to_polygon_object(
    exterior_x_coords=THESE_X_COORDS + 1500, exterior_y_coords=THESE_Y_COORDS)

SEVENTH_POLYGON_OBJECT_XY = polygons.vertex_arrays_to_polygon_object(
    exterior_x_coords=THESE_X_COORDS, exterior_y_coords=THESE_Y_COORDS + 1000)

EIGHTH_POLYGON_OBJECT_XY = polygons.vertex_arrays_to_polygon_object(
    exterior_x_coords=THESE_X_COORDS + 1500,
    exterior_y_coords=THESE_Y_COORDS + 2000)

POLYGON_OBJECTS_PIXEL_COORDS = [
    FIRST_POLYGON_OBJECT_XY, SECOND_POLYGON_OBJECT_XY, THIRD_POLYGON_OBJECT_XY,
    FOURTH_POLYGON_OBJECT_XY, FIFTH_POLYGON_OBJECT_XY, SIXTH_POLYGON_OBJECT_XY,
    SEVENTH_POLYGON_OBJECT_XY, EIGHTH_POLYGON_OBJECT_XY
]

THESE_ROWS = numpy.array([21.1, 23.5, 21.1, 18.7, 21.1])
THESE_COLUMNS = numpy.array([-0.5, 2.7, 5.9, 2.7, -0.5])
THIS_FIRST_OBJECT = polygons.vertex_arrays_to_polygon_object(
    exterior_x_coords=THESE_COLUMNS, exterior_y_coords=THESE_ROWS)

THESE_ROWS = numpy.array([0.5, 0.5, 4.3, 9.1, 9.1, 0.5])
THESE_COLUMNS = numpy.array([
    27.2333333, 30.5, 30.5, 30.5, 27.2333333, 27.2333333
])
THIS_SECOND_OBJECT = polygons.vertex_arrays_to_polygon_object(
    exterior_x_coords=THESE_COLUMNS, exterior_y_coords=THESE_ROWS)

POLYGON_OBJECTS_GRID_COORDS = (
    [THIS_FIRST_OBJECT] * 4 + [THIS_SECOND_OBJECT] * 4
)
PANEL_ROW_BY_POLYGON = numpy.array([0, 0, 1, 2, 0, 0, 1, 2], dtype=int)
PANEL_COLUMN_BY_POLYGON = numpy.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=int)

# The following constants are used to test polygons_to_mask.
ROWS_IN_FIRST_4POLYGONS = numpy.array([
    19,
    20, 20, 20, 20,
    21, 21, 21, 21, 21, 21,
    22, 22, 22, 22,
    23
], dtype=int)

COLUMNS_IN_FIRST_4POLYGONS = numpy.array([
    3,
    1, 2, 3, 4,
    0, 1, 2, 3, 4, 5,
    1, 2, 3, 4,
    3
], dtype=int)

MASK_MATRIX = numpy.full(
    (NUM_PANEL_ROWS, NUM_PANEL_COLUMNS, NUM_GRID_ROWS, NUM_GRID_COLUMNS),
    False, dtype=bool
)

MASK_MATRIX[0, 0, ROWS_IN_FIRST_4POLYGONS, COLUMNS_IN_FIRST_4POLYGONS] = True
MASK_MATRIX[0, 1, ROWS_IN_FIRST_4POLYGONS, COLUMNS_IN_FIRST_4POLYGONS] = True
MASK_MATRIX[1, 0, ROWS_IN_FIRST_4POLYGONS, COLUMNS_IN_FIRST_4POLYGONS] = True
MASK_MATRIX[2, 1, ROWS_IN_FIRST_4POLYGONS, COLUMNS_IN_FIRST_4POLYGONS] = True

ROWS_IN_LAST_4POLYGONS = numpy.array([
    1, 1, 1,
    2, 2, 2,
    3, 3, 3,
    4, 4, 4,
    5, 5, 5,
    6, 6, 6,
    7, 7, 7,
    8, 8, 8,
    9, 9, 9
], dtype=int)

COLUMNS_IN_LAST_4POLYGONS = numpy.array([
    28, 29, 30,
    28, 29, 30,
    28, 29, 30,
    28, 29, 30,
    28, 29, 30,
    28, 29, 30,
    28, 29, 30,
    28, 29, 30,
    28, 29, 30
], dtype=int)

MASK_MATRIX[0, 0, ROWS_IN_LAST_4POLYGONS, COLUMNS_IN_LAST_4POLYGONS] = True
MASK_MATRIX[0, 1, ROWS_IN_LAST_4POLYGONS, COLUMNS_IN_LAST_4POLYGONS] = True
MASK_MATRIX[1, 0, ROWS_IN_LAST_4POLYGONS, COLUMNS_IN_LAST_4POLYGONS] = True
MASK_MATRIX[2, 1, ROWS_IN_LAST_4POLYGONS, COLUMNS_IN_LAST_4POLYGONS] = True


class HumanPolygonsTests(unittest.TestCase):
    """Each method is a unit test for human_polygons.py."""

    def test_vertex_list_to_polygon_list(self):
        """Ensures correct output from _vertex_list_to_polygon_list."""

        these_polygon_objects, these_poly_to_first_vertex_indices = (
            human_polygons._vertex_list_to_polygon_list(
                vertex_rows=TOY_VERTEX_ROWS, vertex_columns=TOY_VERTEX_COLUMNS)
        )

        self.assertTrue(numpy.array_equal(
            these_poly_to_first_vertex_indices, TOY_POLY_TO_FIRST_VERTEX_INDICES
        ))

        for k in range(len(these_polygon_objects)):
            self.assertTrue(
                these_polygon_objects[k].almost_equals(
                    TOY_POLYGON_OBJECTS[k], decimal=TOLERANCE_NUM_DECIMAL_PLACES
                )
            )

    def test_polygon_list_to_vertex_list(self):
        """Ensures correct output from _polygon_list_to_vertex_list."""

        these_rows, these_columns, these_vertex_to_polygon_indices = (
            human_polygons._polygon_list_to_vertex_list(TOY_POLYGON_OBJECTS)
        )

        self.assertTrue(numpy.allclose(
            these_rows, TOY_VERTEX_ROWS, atol=TOLERANCE, equal_nan=True
        ))

        self.assertTrue(numpy.allclose(
            these_columns, TOY_VERTEX_COLUMNS, atol=TOLERANCE, equal_nan=True
        ))

        self.assertTrue(numpy.array_equal(
            these_vertex_to_polygon_indices, TOY_VERTEX_TO_POLY_INDICES
        ))

    def test_polygons_from_pixel_to_grid_coords(self):
        """Ensures correct output from polygons_from_pixel_to_grid_coords."""

        these_polygon_objects, these_panel_rows, these_panel_columns = (
            human_polygons.polygons_from_pixel_to_grid_coords(
                polygon_objects_pixel_coords=POLYGON_OBJECTS_PIXEL_COORDS,
                num_grid_rows=NUM_GRID_ROWS, num_grid_columns=NUM_GRID_COLUMNS,
                num_pixel_rows=NUM_PIXEL_ROWS,
                num_pixel_columns=NUM_PIXEL_COLUMNS,
                num_panel_rows=NUM_PANEL_ROWS,
                num_panel_columns=NUM_PANEL_COLUMNS)
        )

        self.assertTrue(numpy.array_equal(
            these_panel_rows, PANEL_ROW_BY_POLYGON
        ))

        self.assertTrue(numpy.array_equal(
            these_panel_columns, PANEL_COLUMN_BY_POLYGON
        ))

        actual_num_polygons = len(these_polygon_objects)
        expected_num_polygons = len(POLYGON_OBJECTS_GRID_COORDS)
        self.assertTrue(actual_num_polygons == expected_num_polygons)

        for k in range(expected_num_polygons):
            self.assertTrue(
                these_polygon_objects[k].almost_equals(
                    POLYGON_OBJECTS_GRID_COORDS[k],
                    decimal=TOLERANCE_NUM_DECIMAL_PLACES
                )
            )

    def test_polygons_to_mask(self):
        """Ensures correct output from polygons_to_mask."""

        this_mask_matrix = human_polygons.polygons_to_mask(
            polygon_objects_grid_coords=POLYGON_OBJECTS_GRID_COORDS,
            num_grid_rows=NUM_GRID_ROWS, num_grid_columns=NUM_GRID_COLUMNS,
            num_panel_rows=NUM_PANEL_ROWS, num_panel_columns=NUM_PANEL_COLUMNS,
            panel_row_by_polygon=PANEL_ROW_BY_POLYGON,
            panel_column_by_polygon=PANEL_COLUMN_BY_POLYGON)

        self.assertTrue(numpy.array_equal(this_mask_matrix, MASK_MATRIX))


if __name__ == '__main__':
    unittest.main()
