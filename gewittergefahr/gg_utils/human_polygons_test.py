"""Unit tests for human_polygons.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import human_polygons

TOLERANCE_NUM_DECIMAL_PLACES = 6

NUM_GRID_ROWS = 24
NUM_GRID_COLUMNS = 32
NUM_PIXEL_ROWS = 1000
NUM_PIXEL_COLUMNS = 1500

THESE_X_COORDS = numpy.array([-0.5, 149.5, 299.5, 149.5, -0.5])
THESE_Y_COORDS = numpy.array([99.5, -0.5, 99.5, 199.5, 99.5])
FIRST_POLYGON_OBJECT_XY = polygons.vertex_arrays_to_polygon_object(
    exterior_x_coords=THESE_X_COORDS, exterior_y_coords=THESE_Y_COORDS)

THESE_X_COORDS = numpy.array([1299.5, 1499.5, 1499.5, 1499.5, 1299.5, 1299.5])
THESE_Y_COORDS = numpy.array([999.5, 999.5, 799.5, 599.5, 599.5, 999.5])
SECOND_POLYGON_OBJECT_XY = polygons.vertex_arrays_to_polygon_object(
    exterior_x_coords=THESE_X_COORDS, exterior_y_coords=THESE_Y_COORDS)

LIST_OF_POLYGON_OBJECTS_XY = [FIRST_POLYGON_OBJECT_XY, SECOND_POLYGON_OBJECT_XY]

THESE_ROWS = numpy.array([21.1, 23.5, 21.1, 18.7, 21.1])
THESE_COLUMNS = numpy.array([-0.5, 2.7, 5.9, 2.7, -0.5])
FIRST_POLYGON_OBJECT_ROWCOL = polygons.vertex_arrays_to_polygon_object(
    exterior_x_coords=THESE_COLUMNS, exterior_y_coords=THESE_ROWS)

THESE_ROWS = numpy.array([-0.5, -0.5, 4.3, 9.1, 9.1, -0.5])
THESE_COLUMNS = numpy.array(
    [27.2333333, 31.5, 31.5, 31.5, 27.2333333, 27.2333333], dtype=float
)
SECOND_POLYGON_OBJECT_ROWCOL = polygons.vertex_arrays_to_polygon_object(
    exterior_x_coords=THESE_COLUMNS, exterior_y_coords=THESE_ROWS)

LIST_OF_POLYGON_OBJECTS_ROWCOL = [
    FIRST_POLYGON_OBJECT_ROWCOL, SECOND_POLYGON_OBJECT_ROWCOL
]

ROWS_IN_FIRST_POLYGON = numpy.array([
    19,
    20, 20, 20, 20,
    21, 21, 21, 21, 21, 21,
    22, 22, 22, 22,
    23
], dtype=int)

COLUMNS_IN_FIRST_POLYGON = numpy.array([
    3,
    1, 2, 3, 4,
    0, 1, 2, 3, 4, 5,
    1, 2, 3, 4,
    3
], dtype=int)

MASK_MATRIX = numpy.full((NUM_GRID_ROWS, NUM_GRID_COLUMNS), False, dtype=bool)
MASK_MATRIX[ROWS_IN_FIRST_POLYGON, COLUMNS_IN_FIRST_POLYGON] = True

ROWS_IN_SECOND_POLYGON = numpy.array([
    0, 0, 0, 0,
    1, 1, 1, 1,
    2, 2, 2, 2,
    3, 3, 3, 3,
    4, 4, 4, 4,
    5, 5, 5, 5,
    6, 6, 6, 6,
    7, 7, 7, 7,
    8, 8, 8, 8,
    9, 9, 9, 9
], dtype=int)

COLUMNS_IN_SECOND_POLYGON = numpy.array([
    28, 29, 30, 31,
    28, 29, 30, 31,
    28, 29, 30, 31,
    28, 29, 30, 31,
    28, 29, 30, 31,
    28, 29, 30, 31,
    28, 29, 30, 31,
    28, 29, 30, 31,
    28, 29, 30, 31,
    28, 29, 30, 31
], dtype=int)

MASK_MATRIX[ROWS_IN_SECOND_POLYGON, COLUMNS_IN_SECOND_POLYGON] = True


class HumanPolygonsTests(unittest.TestCase):
    """Each method is a unit test for human_polygons.py."""

    def test_polygons_from_pixel_to_grid_coords(self):
        """Ensures correct output from polygons_from_pixel_to_grid_coords."""

        these_polygon_objects = (
            human_polygons.polygons_from_pixel_to_grid_coords(
                list_of_polygon_objects_xy=LIST_OF_POLYGON_OBJECTS_XY,
                num_grid_rows=NUM_GRID_ROWS, num_grid_columns=NUM_GRID_COLUMNS,
                num_pixel_rows=NUM_PIXEL_ROWS,
                num_pixel_columns=NUM_PIXEL_COLUMNS)
        )

        actual_num_polygons = len(these_polygon_objects)
        expected_num_polygons = len(LIST_OF_POLYGON_OBJECTS_ROWCOL)
        self.assertTrue(actual_num_polygons == expected_num_polygons)

        for k in range(expected_num_polygons):
            self.assertTrue(
                these_polygon_objects[k].almost_equals(
                    LIST_OF_POLYGON_OBJECTS_ROWCOL[k],
                    decimal=TOLERANCE_NUM_DECIMAL_PLACES
                )
            )

    def test_polygons_to_mask(self):
        """Ensures correct output from polygons_to_mask."""

        this_mask_matrix = human_polygons.polygons_to_mask(
            list_of_polygon_objects_rowcol=LIST_OF_POLYGON_OBJECTS_ROWCOL,
            num_grid_rows=NUM_GRID_ROWS, num_grid_columns=NUM_GRID_COLUMNS)

        self.assertTrue(numpy.array_equal(this_mask_matrix, MASK_MATRIX))


if __name__ == '__main__':
    unittest.main()
