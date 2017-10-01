"""Unit tests for interp.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import interp

TOLERANCE = 1e-6

INPUT_MATRIX_TIME0 = numpy.array([[0., 2., 5., 10.],
                                  [-2., 1., 3., 6.],
                                  [-3.5, -2.5, 3., 8.]])
INPUT_MATRIX_TIME4 = numpy.array([[2., 5., 7., 15.],
                                  [0., 2., 5., 8.],
                                  [-1.5, -2.5, 0., 4.]])

INPUT_TIMES_UNIX_SEC = numpy.array([0, 4])
TRUE_INTERP_TIMES_UNIX_SEC = numpy.array([1, 2, 3])
EXTRAP_TIMES_UNIX_SEC = numpy.array([8])
TEMPORAL_INTERP_METHOD = 'linear'
INPUT_MATRIX_FOR_TEMPORAL_INTERP = numpy.stack(
    (INPUT_MATRIX_TIME0, INPUT_MATRIX_TIME4), axis=-1)

EXPECTED_QUERY_MATRIX_TIME1 = numpy.array([[0.5, 2.75, 5.5, 11.25],
                                           [-1.5, 1.25, 3.5, 6.5],
                                           [-3., -2.5, 2.25, 7.]])
EXPECTED_QUERY_MATRIX_TIME2 = numpy.array([[1., 3.5, 6., 12.5],
                                           [-1., 1.5, 4., 7.],
                                           [-2.5, -2.5, 1.5, 6.]])
EXPECTED_QUERY_MATRIX_TIME3 = numpy.array([[1.5, 4.25, 6.5, 13.75],
                                           [-0.5, 1.75, 4.5, 7.5],
                                           [-2., -2.5, 0.75, 5.]])
EXPECTED_QUERY_MATRIX_TIME8 = numpy.array([[4., 8., 9., 20.],
                                           [2., 3., 7., 10.],
                                           [0.5, -2.5, -3., 0.]])

EXPECTED_QUERY_MATRIX_TRUE_INTERP = numpy.stack(
    (EXPECTED_QUERY_MATRIX_TIME1, EXPECTED_QUERY_MATRIX_TIME2,
     EXPECTED_QUERY_MATRIX_TIME3), axis=-1)
EXPECTED_QUERY_MATRIX_EXTRAP = numpy.expand_dims(EXPECTED_QUERY_MATRIX_TIME8,
                                                 axis=-1)

INPUT_MATRIX_FOR_SPATIAL_INTERP = numpy.array([[17., 24., 1., 8.],
                                               [23., 5., 7., 14.],
                                               [4., 6., 13., 20.],
                                               [10., 12., 19., 21.],
                                               [11., 18., 25., 2.]])

POLYNOMIAL_DEGREE_FOR_SPATIAL_INTERP = 1
GRID_POINT_X_METRES = numpy.array([0., 1., 2., 3.])
GRID_POINT_Y_METRES = numpy.array([0., 2., 4., 6., 8.])
QUERY_X_METRES = numpy.array([0., 0.5, 1., 1.5, 2., 2.5, 3.])
QUERY_Y_METRES = numpy.array([0., 2., 2.5, 3., 5., 6., 7.5])

EXPECTED_QUERY_VALUES_FOR_SPATIAL_INTERP = numpy.array(
    [17., 14., 5.25, 7.75, 16., 20., 6.75])


class InterpTests(unittest.TestCase):
    """Each method is a unit test for interp.py."""

    def test_interp_in_time_true_interp(self):
        """Ensures correct output from interp_in_time.

        In this case, doing true interpolation (no extrapolation).
        """

        this_query_matrix = interp.interp_in_time(
            INPUT_MATRIX_FOR_TEMPORAL_INTERP,
            sorted_input_times_unix_sec=INPUT_TIMES_UNIX_SEC,
            query_times_unix_sec=TRUE_INTERP_TIMES_UNIX_SEC,
            method_string=TEMPORAL_INTERP_METHOD)

        self.assertTrue(numpy.allclose(
            this_query_matrix, EXPECTED_QUERY_MATRIX_TRUE_INTERP,
            atol=TOLERANCE))

    def test_interp_in_time_extrap(self):
        """Ensures correct output from interp_in_time.

        In this case, doing extrapolation.
        """

        this_query_matrix = interp.interp_in_time(
            INPUT_MATRIX_FOR_TEMPORAL_INTERP,
            sorted_input_times_unix_sec=INPUT_TIMES_UNIX_SEC,
            query_times_unix_sec=EXTRAP_TIMES_UNIX_SEC,
            method_string=TEMPORAL_INTERP_METHOD, allow_extrap=True)
        self.assertTrue(numpy.allclose(
            this_query_matrix, EXPECTED_QUERY_MATRIX_EXTRAP, atol=TOLERANCE))

    def test_interp_from_xy_grid_to_points(self):
        """Ensures correct output from interp_from_xy_grid_to_points."""

        these_query_values = interp.interp_from_xy_grid_to_points(
            INPUT_MATRIX_FOR_SPATIAL_INTERP,
            sorted_grid_point_x_metres=GRID_POINT_X_METRES,
            sorted_grid_point_y_metres=GRID_POINT_Y_METRES,
            query_x_metres=QUERY_X_METRES, query_y_metres=QUERY_Y_METRES,
            polynomial_degree=POLYNOMIAL_DEGREE_FOR_SPATIAL_INTERP)

        self.assertTrue(numpy.allclose(these_query_values,
                                       EXPECTED_QUERY_VALUES_FOR_SPATIAL_INTERP,
                                       atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
