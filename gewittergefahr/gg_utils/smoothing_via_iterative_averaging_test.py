"""Unit tests for smoothing_via_iterative_averaging.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import smoothing_via_iterative_averaging as sia
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import shape_utils

TOLERANCE = 1e-6
NUM_VERTICES_IN_SMOOTHING_HALF_WINDOW = 1

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

POLYLINE_X_COORDS_SMOOTHED = numpy.array(
    [3., 2., 1., 1., 2., 3.666667, 4.333333, 6., 7., 7., 6., 5.])
POLYLINE_Y_COORDS_SMOOTHED = numpy.array(
    [6., 4., 2.333333, 1.666667, 0.666667, 0.333333, 0.333333, 0.666667,
     1.666667, 2.333333, 4., 6.])

POLYGON_X_COORDS_SMOOTHED = numpy.array(
    [3.666667, 2., 1., 1., 2., 3.666667, 4.333333, 6., 7., 7., 6., 4.333333])
POLYGON_Y_COORDS_SMOOTHED = numpy.array(
    [5., 4., 2.333333, 1.666667, 0.666667, 0.333333, 0.333333, 0.666667,
     1.666667, 2.333333, 4., 5.])


class SmoothingViaIterativeAveragingTests(unittest.TestCase):
    """Each method is a unit test for smoothing_via_iterative_averaging.py."""

    def test_sia_one_iteration_polyline(self):
        """Ensures correct output from _sia_one_iteration for polyline."""

        vertex_x_coords_padded, vertex_y_coords_padded = (
            shape_utils.pad_polyline(
                POLYLINE_X_COORDS, POLYLINE_Y_COORDS,
                num_padding_vertices=NUM_VERTICES_IN_SMOOTHING_HALF_WINDOW))

        vertex_x_coords_smoothed, vertex_y_coords_smoothed = (
            sia._sia_one_iteration(
                vertex_x_coords_padded, vertex_y_coords_padded,
                NUM_VERTICES_IN_SMOOTHING_HALF_WINDOW))

        self.assertTrue(numpy.allclose(
            vertex_x_coords_smoothed, POLYLINE_X_COORDS_SMOOTHED,
            atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            vertex_y_coords_smoothed, POLYLINE_Y_COORDS_SMOOTHED,
            atol=TOLERANCE))

    def test_sia_one_iteration_closed_polygon(self):
        """Ensures correct output from _sia_one_iteration for closed polygon."""

        vertex_x_coords_padded, vertex_y_coords_padded = (
            shape_utils.pad_closed_polygon(
                POLYGON_OBJECT, num_padding_vertices=1))

        vertex_x_coords_smoothed, vertex_y_coords_smoothed = (
            sia._sia_one_iteration(
                vertex_x_coords_padded, vertex_y_coords_padded,
                NUM_VERTICES_IN_SMOOTHING_HALF_WINDOW))

        self.assertTrue(numpy.allclose(
            vertex_x_coords_smoothed, POLYGON_X_COORDS_SMOOTHED,
            atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            vertex_y_coords_smoothed, POLYGON_Y_COORDS_SMOOTHED,
            atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
