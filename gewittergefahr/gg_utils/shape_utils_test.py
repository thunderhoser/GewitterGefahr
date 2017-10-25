"""Unit tests for shape_utils.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import shape_utils
from gewittergefahr.gg_utils import polygons

TOLERANCE = 1e-6

POLYLINE_X_COORDS = numpy.array(
    [3., 3., 0., 0., 3., 3., 5., 5., 8., 8., 5., 5.])
POLYLINE_Y_COORDS = numpy.array(
    [6., 3., 3., 1., 1., 0., 0., 1., 1., 3., 3., 6.])

POLYGON_X_COORDS = numpy.concatenate((
    POLYLINE_X_COORDS, numpy.array([POLYLINE_X_COORDS[0]])))
POLYGON_Y_COORDS = numpy.concatenate((
    POLYLINE_Y_COORDS, numpy.array([POLYLINE_Y_COORDS[0]])))
POLYGON_OBJECT = polygons.vertex_arrays_to_polygon_object(
    POLYGON_X_COORDS, POLYGON_Y_COORDS)

POLYLINE_X_COORDS_PADDED1 = numpy.array(
    [3., 3., 3., 0., 0., 3., 3., 5., 5., 8., 8., 5., 5., 5.])
POLYLINE_Y_COORDS_PADDED1 = numpy.array(
    [9., 6., 3., 3., 1., 1., 0., 0., 1., 1., 3., 3., 6., 9.])

POLYLINE_X_COORDS_PADDED3 = numpy.array(
    [3., 3., 3., 3., 3., 0., 0., 3., 3., 5., 5., 8., 8., 5., 5., 5., 5., 5.])
POLYLINE_Y_COORDS_PADDED3 = numpy.array([
    15., 12., 9., 6., 3., 3., 1., 1., 0., 0., 1., 1., 3., 3., 6., 9., 12., 15.])

POLYGON_X_COORDS_PADDED1 = numpy.array(
    [5., 3., 3., 0., 0., 3., 3., 5., 5., 8., 8., 5., 5., 3.])
POLYGON_Y_COORDS_PADDED1 = numpy.array(
    [6., 6., 3., 3., 1., 1., 0., 0., 1., 1., 3., 3., 6., 6.])

POLYGON_X_COORDS_PADDED3 = numpy.array(
    [8., 5., 5., 3., 3., 0., 0., 3., 3., 5., 5., 8., 8., 5., 5., 3., 3., 0.])
POLYGON_Y_COORDS_PADDED3 = numpy.array(
    [3., 3., 6., 6., 3., 3., 1., 1., 0., 0., 1., 1., 3., 3., 6., 6., 3., 3.])


class ShapeUtilsTests(unittest.TestCase):
    """Each method is a unit test for shape_utils.py."""

    def test_pad_polyline_1vertex(self):
        """Ensures correct output from pad_polyline with one padding vertex."""

        vertex_x_coords_padded, vertex_y_coords_padded = (
            shape_utils.pad_polyline(
                POLYLINE_X_COORDS, POLYLINE_Y_COORDS, num_padding_vertices=1))

        self.assertTrue(numpy.allclose(
            vertex_x_coords_padded, POLYLINE_X_COORDS_PADDED1, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            vertex_y_coords_padded, POLYLINE_Y_COORDS_PADDED1, atol=TOLERANCE))

    def test_pad_polyline_3vertices(self):
        """Ensures correct output from pad_polyline with 3 padding vertices."""

        vertex_x_coords_padded, vertex_y_coords_padded = (
            shape_utils.pad_polyline(
                POLYLINE_X_COORDS, POLYLINE_Y_COORDS, num_padding_vertices=3))

        self.assertTrue(numpy.allclose(
            vertex_x_coords_padded, POLYLINE_X_COORDS_PADDED3, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            vertex_y_coords_padded, POLYLINE_Y_COORDS_PADDED3, atol=TOLERANCE))

    def test_pad_closed_polygon_1vertex(self):
        """Ensures correct output from pad_closed_polygon; 1 padding vertex."""

        vertex_x_coords_padded, vertex_y_coords_padded = (
            shape_utils.pad_closed_polygon(
                POLYGON_OBJECT, num_padding_vertices=1))

        self.assertTrue(numpy.allclose(
            vertex_x_coords_padded, POLYGON_X_COORDS_PADDED1, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            vertex_y_coords_padded, POLYGON_Y_COORDS_PADDED1, atol=TOLERANCE))

    def test_pad_closed_polygon_3vertices(self):
        """Ensures correct output from pad_closed_polygon; 3 pad vertices."""

        vertex_x_coords_padded, vertex_y_coords_padded = (
            shape_utils.pad_closed_polygon(
                POLYGON_OBJECT, num_padding_vertices=3))

        self.assertTrue(numpy.allclose(
            vertex_x_coords_padded, POLYGON_X_COORDS_PADDED3, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            vertex_y_coords_padded, POLYGON_Y_COORDS_PADDED3, atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
