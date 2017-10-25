"""Unit tests for smoothing_via_iterative_averaging.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import smoothing_via_iterative_averaging as sia

TOLERANCE = 1e-6

VERTEX_X_COORDS = numpy.array([3., 3., 0., 0., 3., 3., 5., 5., 8., 8., 5., 5.])
VERTEX_Y_COORDS = numpy.array([6., 3., 3., 1., 1., 0., 0., 1., 1., 3., 3., 6.])

VERTEX_X_COORDS_POLYLINE_PADDED1 = numpy.array(
    [3., 3., 3., 0., 0., 3., 3., 5., 5., 8., 8., 5., 5., 5.])
VERTEX_Y_COORDS_POLYLINE_PADDED1 = numpy.array(
    [9., 6., 3., 3., 1., 1., 0., 0., 1., 1., 3., 3., 6., 9.])

VERTEX_X_COORDS_POLYLINE_PADDED3 = numpy.array(
    [3., 3., 3., 3., 3., 0., 0., 3., 3., 5., 5., 8., 8., 5., 5., 5., 5., 5.])
VERTEX_Y_COORDS_POLYLINE_PADDED3 = numpy.array([
    15., 12., 9., 6., 3., 3., 1., 1., 0., 0., 1., 1., 3., 3., 6., 9., 12., 15.])

VERTEX_X_COORDS_POLYGON_PADDED1 = numpy.array(
    [5., 3., 3., 0., 0., 3., 3., 5., 5., 8., 8., 5., 5., 3.])
VERTEX_Y_COORDS_POLYGON_PADDED1 = numpy.array(
    [6., 6., 3., 3., 1., 1., 0., 0., 1., 1., 3., 3., 6., 6.])

VERTEX_X_COORDS_POLYGON_PADDED3 = numpy.array(
    [8., 5., 5., 3., 3., 0., 0., 3., 3., 5., 5., 8., 8., 5., 5., 3., 3., 0.])
VERTEX_Y_COORDS_POLYGON_PADDED3 = numpy.array(
    [3., 3., 6., 6., 3., 3., 1., 1., 0., 0., 1., 1., 3., 3., 6., 6., 3., 3.])

NUM_VERTICES_IN_HALF_WINDOW_FOR_SIA = 1
VERTEX_X_COORDS_POLYLINE_SMOOTHED = numpy.array(
    [3., 2., 1., 1., 2., 3.666667, 4.333333, 6., 7., 7., 6., 5.])
VERTEX_Y_COORDS_POLYLINE_SMOOTHED = numpy.array(
    [6., 4., 2.333333, 1.666667, 0.666667, 0.333333, 0.333333, 0.666667,
     1.666667, 2.333333, 4., 6.])

VERTEX_X_COORDS_POLYGON_SMOOTHED = numpy.array(
    [3.666667, 2., 1., 1., 2., 3.666667, 4.333333, 6., 7., 7., 6., 4.333333])
VERTEX_Y_COORDS_POLYGON_SMOOTHED = numpy.array(
    [5., 4., 2.333333, 1.666667, 0.666667, 0.333333, 0.333333, 0.666667,
     1.666667, 2.333333, 4., 5.])


class SmoothingViaIterativeAveragingTests(unittest.TestCase):
    """Each method is a unit test for smoothing_via_iterative_averaging.py."""

    def test_pad_polyline_for_sia_1vertex(self):
        """Ensures correct output from _pad_polyline_for_sia.

        In this case, smoothing half-window is one vertex.
        """

        vertex_x_coords_padded, vertex_y_coords_padded = (
            sia._pad_polyline_for_sia(VERTEX_X_COORDS, VERTEX_Y_COORDS, 1))

        self.assertTrue(numpy.allclose(
            vertex_x_coords_padded, VERTEX_X_COORDS_POLYLINE_PADDED1,
            atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            vertex_y_coords_padded, VERTEX_Y_COORDS_POLYLINE_PADDED1,
            atol=TOLERANCE))

    def test_pad_polyline_for_sia_3vertices(self):
        """Ensures correct output from _pad_polyline_for_sia.

        In this case, smoothing half-window is 3 vertices.
        """

        vertex_x_coords_padded, vertex_y_coords_padded = (
            sia._pad_polyline_for_sia(VERTEX_X_COORDS, VERTEX_Y_COORDS, 3))

        self.assertTrue(numpy.allclose(
            vertex_x_coords_padded, VERTEX_X_COORDS_POLYLINE_PADDED3,
            atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            vertex_y_coords_padded, VERTEX_Y_COORDS_POLYLINE_PADDED3,
            atol=TOLERANCE))

    def test_pad_closed_polygon_for_sia_1vertex(self):
        """Ensures correct output from _pad_closed_polygon_for_sia.

        In this case, smoothing half-window is one vertex.
        """

        vertex_x_coords_padded, vertex_y_coords_padded = (
            sia._pad_closed_polygon_for_sia(
                VERTEX_X_COORDS, VERTEX_Y_COORDS, 1))

        self.assertTrue(numpy.allclose(
            vertex_x_coords_padded, VERTEX_X_COORDS_POLYGON_PADDED1,
            atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            vertex_y_coords_padded, VERTEX_Y_COORDS_POLYGON_PADDED1,
            atol=TOLERANCE))

    def test_pad_closed_polygon_for_sia_3vertices(self):
        """Ensures correct output from _pad_closed_polygon_for_sia.

        In this case, smoothing half-window is 3 vertices.
        """

        vertex_x_coords_padded, vertex_y_coords_padded = (
            sia._pad_closed_polygon_for_sia(
                VERTEX_X_COORDS, VERTEX_Y_COORDS, 3))

        self.assertTrue(numpy.allclose(
            vertex_x_coords_padded, VERTEX_X_COORDS_POLYGON_PADDED3,
            atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            vertex_y_coords_padded, VERTEX_Y_COORDS_POLYGON_PADDED3,
            atol=TOLERANCE))

    def test_sia_one_iteration_polyline(self):
        """Ensures correct output from _sia_one_iteration for polyline."""

        vertex_x_coords_smoothed, vertex_y_coords_smoothed = (
            sia._sia_one_iteration(VERTEX_X_COORDS_POLYLINE_PADDED1,
                                   VERTEX_Y_COORDS_POLYLINE_PADDED1,
                                   NUM_VERTICES_IN_HALF_WINDOW_FOR_SIA))

        self.assertTrue(numpy.allclose(
            vertex_x_coords_smoothed, VERTEX_X_COORDS_POLYLINE_SMOOTHED,
            atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            vertex_y_coords_smoothed, VERTEX_Y_COORDS_POLYLINE_SMOOTHED,
            atol=TOLERANCE))

    def test_sia_one_iteration_closed_polygon(self):
        """Ensures correct output from _sia_one_iteration for closed polygon."""

        vertex_x_coords_smoothed, vertex_y_coords_smoothed = (
            sia._sia_one_iteration(VERTEX_X_COORDS_POLYGON_PADDED1,
                                   VERTEX_Y_COORDS_POLYGON_PADDED1,
                                   NUM_VERTICES_IN_HALF_WINDOW_FOR_SIA))

        self.assertTrue(numpy.allclose(
            vertex_x_coords_smoothed, VERTEX_X_COORDS_POLYGON_SMOOTHED,
            atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            vertex_y_coords_smoothed, VERTEX_Y_COORDS_POLYGON_SMOOTHED,
            atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
