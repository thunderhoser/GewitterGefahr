"""Unit tests for grid_smoothing_2d.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import grid_smoothing_2d

TOLERANCE = 1e-6

# The following constants are used to test _get_distances_from_center_point.
GRID_SPACING_X_METRES = 5.
GRID_SPACING_Y_METRES = 10.
CUTOFF_RADIUS_METRES = 25.

RELATIVE_X_MATRIX_METRES = numpy.array(
    [[-25., -20., -15., -10., -5., 0., 5., 10., 15., 20., 25.],
     [-25., -20., -15., -10., -5., 0., 5., 10., 15., 20., 25.],
     [-25., -20., -15., -10., -5., 0., 5., 10., 15., 20., 25.],
     [-25., -20., -15., -10., -5., 0., 5., 10., 15., 20., 25.],
     [-25., -20., -15., -10., -5., 0., 5., 10., 15., 20., 25.]])

RELATIVE_Y_MATRIX_METRES = numpy.array(
    [[-20., -20., -20., -20., -20., -20., -20., -20., -20., -20., -20.],
     [-10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10.],
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
     [10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10.],
     [20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20.]])

DISTANCE_FROM_CENTER_MATRIX = numpy.sqrt(
    RELATIVE_X_MATRIX_METRES ** 2 + RELATIVE_Y_MATRIX_METRES ** 2)

# The following constants are used to test _apply_smoother_at_one_point.
WEIGHT_VECTOR = numpy.array([0., 1., 0., 1., 2., 1., 0., 1., 0.])
INPUT_VECTOR = numpy.array([1., 2., 3., 4., 5., 6., 7., 8., 9.])
SMOOTHED_VALUE = 30.

# The following constants are used to test _apply_smoother_at_all_points.
INPUT_MATRIX = numpy.array(
    [[1., 2., 3., 4., 5.],
     [2., 3., 4., 5., 6.],
     [0., numpy.nan, 2., 3., 4.],
     [-2., -1., numpy.nan, 2., 4.],
     [-6., -3., -3., numpy.nan, 2.]])

WEIGHT_MATRIX = numpy.array(
    [[0., 1., 0.],
     [1., 2., 1.],
     [0., 1., 0.]])

SMOOTHED_MATRIX = numpy.array(
    [[6., 11., 16., 21., 20.],
     [8., 14., 21., 27., 26.],
     [numpy.nan, 4., 11., 19., 21.],
     [-11., -7., numpy.nan, 11., 16.],
     [-17., -16., -9., 1., 8.]])


class GridSmoothing2dTests(unittest.TestCase):
    """Each method is a unit test for grid_smoothing_2d.py."""

    def test_get_distances_from_center_point(self):
        """Ensures correct output from _get_distances_from_center_point."""

        this_distance_matrix = (
            grid_smoothing_2d._get_distances_from_center_point(
                grid_spacing_x=GRID_SPACING_X_METRES,
                grid_spacing_y=GRID_SPACING_Y_METRES,
                cutoff_radius=CUTOFF_RADIUS_METRES))

        self.assertTrue(numpy.allclose(
            this_distance_matrix, DISTANCE_FROM_CENTER_MATRIX, atol=TOLERANCE))

    def test_apply_smoother_at_one_point(self):
        """Ensures correct output from _apply_smoother_at_one_point."""

        this_smoothed_value = grid_smoothing_2d._apply_smoother_at_one_point(
            INPUT_VECTOR, WEIGHT_VECTOR)
        self.assertTrue(numpy.isclose(
            this_smoothed_value, SMOOTHED_VALUE, atol=TOLERANCE))

    def test_apply_smoother_at_all_points(self):
        """Ensures correct output from _apply_smoother_at_all_points."""

        this_smoothed_matrix = grid_smoothing_2d._apply_smoother_at_all_points(
            INPUT_MATRIX, WEIGHT_MATRIX)
        self.assertTrue(numpy.allclose(
            this_smoothed_matrix, SMOOTHED_MATRIX, atol=TOLERANCE,
            equal_nan=True))


if __name__ == '__main__':
    unittest.main()
