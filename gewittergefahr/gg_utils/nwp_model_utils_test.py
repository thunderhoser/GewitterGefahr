"""Unit tests for nwp_model_utils.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import nwp_model_utils

TOLERANCE = 1e-6

MIN_QUERY_TIME_UNIX_SEC = 1507766400  # 0000 UTC 12 Oct 2017
MODEL_TIME_STEP_SEC = 10800  # 3 hours
FAKE_INTERP_METHOD = 'foo'

SUBLINEAR_INTERP_METHOD = 'nearest'
LINEAR_INTERP_METHOD = 'linear'
MODEL_TIMES_FOR_LINEAR_AND_SUBLINEAR_INTERP_UNIX_SEC = numpy.array(
    [1507766400, 1507777200])

QUADRATIC_INTERP_METHOD = 'quadratic'
CUBIC_INTERP_METHOD = 'cubic'
MODEL_TIMES_FOR_SUPERLINEAR_INTERP_UNIX_SEC = numpy.array(
    [1507755600, 1507766400, 1507777200, 1507788000])

HALF_ROOT3 = numpy.sqrt(3) / 2
U_WINDS_GRID_RELATIVE_M_S01 = numpy.array(
    [[0., 5., 10.],
     [0., -5., -10.]])
V_WINDS_GRID_RELATIVE_M_S01 = numpy.array(
    [[10., 15., 20.],
     [-10., -15., -20.]])
ROTATION_ANGLE_COSINES = numpy.array(
    [[1., 0.5, -0.5],
     [-1., -0.5, 0.5]])
ROTATION_ANGLE_SINES = numpy.array(
    [[0., HALF_ROOT3, HALF_ROOT3],
     [0., -HALF_ROOT3, -HALF_ROOT3]])

U_WINDS_EARTH_RELATIVE_M_S01 = numpy.array(
    [[0., 2.5 + 15 * HALF_ROOT3, -5. + 20 * HALF_ROOT3],
     [0., 2.5 + 15 * HALF_ROOT3, -5. + 20 * HALF_ROOT3]])
V_WINDS_EARTH_RELATIVE_M_S01 = numpy.array(
    [[10., 7.5 - 5 * HALF_ROOT3, -10. - 10 * HALF_ROOT3],
     [10., 7.5 - 5 * HALF_ROOT3, -10. - 10 * HALF_ROOT3]])


class NwpModelUtilsTests(unittest.TestCase):
    """Each method is a unit test for nwp_model_utils.py."""

    def test_get_times_needed_for_interp_bad_interp_method(self):
        """Ensures correct output from get_times_needed_for_interp.

        In this case, interpolation method does not exist.
        """

        with self.assertRaises(ValueError):
            nwp_model_utils.get_times_needed_for_interp(
                min_query_time_unix_sec=MIN_QUERY_TIME_UNIX_SEC,
                model_time_step_sec=MODEL_TIME_STEP_SEC,
                method_string=FAKE_INTERP_METHOD)

    def test_get_times_needed_for_interp_bad_query_time(self):
        """Ensures correct output from get_times_needed_for_interp.

        In this case, minimum query time is not multiple of model time step.
        """

        with self.assertRaises(ValueError):
            nwp_model_utils.get_times_needed_for_interp(
                min_query_time_unix_sec=MIN_QUERY_TIME_UNIX_SEC + 1,
                model_time_step_sec=MODEL_TIME_STEP_SEC,
                method_string=LINEAR_INTERP_METHOD)

    def test_get_times_needed_for_interp_sublinear(self):
        """Ensures correct output from get_times_needed_for_interp.

        In this case, interpolation method is sublinear.
        """

        these_model_times_unix_sec = (
            nwp_model_utils.get_times_needed_for_interp(
                min_query_time_unix_sec=MIN_QUERY_TIME_UNIX_SEC,
                model_time_step_sec=MODEL_TIME_STEP_SEC,
                method_string=SUBLINEAR_INTERP_METHOD))
        self.assertTrue(numpy.array_equal(
            these_model_times_unix_sec,
            MODEL_TIMES_FOR_LINEAR_AND_SUBLINEAR_INTERP_UNIX_SEC))

    def test_get_times_needed_for_interp_linear(self):
        """Ensures correct output from get_times_needed_for_interp.

        In this case, interpolation method is linear.
        """

        these_model_times_unix_sec = (
            nwp_model_utils.get_times_needed_for_interp(
                min_query_time_unix_sec=MIN_QUERY_TIME_UNIX_SEC,
                model_time_step_sec=MODEL_TIME_STEP_SEC,
                method_string=LINEAR_INTERP_METHOD))
        self.assertTrue(numpy.array_equal(
            these_model_times_unix_sec,
            MODEL_TIMES_FOR_LINEAR_AND_SUBLINEAR_INTERP_UNIX_SEC))

    def test_get_times_needed_for_interp_quadratic(self):
        """Ensures correct output from get_times_needed_for_interp.

        In this case, interpolation method is quadratic spline.
        """

        these_model_times_unix_sec = (
            nwp_model_utils.get_times_needed_for_interp(
                min_query_time_unix_sec=MIN_QUERY_TIME_UNIX_SEC,
                model_time_step_sec=MODEL_TIME_STEP_SEC,
                method_string=QUADRATIC_INTERP_METHOD))
        self.assertTrue(numpy.array_equal(
            these_model_times_unix_sec,
            MODEL_TIMES_FOR_SUPERLINEAR_INTERP_UNIX_SEC))

    def test_get_times_needed_for_interp_cubic(self):
        """Ensures correct output from get_times_needed_for_interp.

        In this case, interpolation method is cubic spline.
        """

        these_model_times_unix_sec = (
            nwp_model_utils.get_times_needed_for_interp(
                min_query_time_unix_sec=MIN_QUERY_TIME_UNIX_SEC,
                model_time_step_sec=MODEL_TIME_STEP_SEC,
                method_string=CUBIC_INTERP_METHOD))
        self.assertTrue(numpy.array_equal(
            these_model_times_unix_sec,
            MODEL_TIMES_FOR_SUPERLINEAR_INTERP_UNIX_SEC))

    def test_rotate_winds(self):
        """Ensures correct output from rotate_winds."""

        (these_u_winds_earth_relative_m_s01,
         these_v_winds_earth_relative_m_s01) = nwp_model_utils.rotate_winds(
             u_winds_grid_relative_m_s01=U_WINDS_GRID_RELATIVE_M_S01,
             v_winds_grid_relative_m_s01=V_WINDS_GRID_RELATIVE_M_S01,
             rotation_angle_cosines=ROTATION_ANGLE_COSINES,
             rotation_angle_sines=ROTATION_ANGLE_SINES)

        self.assertTrue(numpy.allclose(
            these_u_winds_earth_relative_m_s01, U_WINDS_EARTH_RELATIVE_M_S01,
            atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_v_winds_earth_relative_m_s01, V_WINDS_EARTH_RELATIVE_M_S01,
            atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
