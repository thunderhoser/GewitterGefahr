"""Unit tests for nwp_model_utils.py."""

import unittest
import numpy
import pandas
from gewittergefahr.gg_utils import nwp_model_utils

TOLERANCE = 1e-6

MODEL_TIME_STEP_HOURS = 3
FAKE_INTERP_METHOD = 'foo'
SUBLINEAR_INTERP_METHOD = 'nearest'
LINEAR_INTERP_METHOD = 'linear'
QUADRATIC_INTERP_METHOD = 'quadratic'
CUBIC_INTERP_METHOD = 'cubic'

# 2200 UTC 11 Oct, 2300 UTC 11 Oct, ..., 0400 UTC 12 Oct 2017
QUERY_TIMES_UNIX_SEC = numpy.array(
    [1507759200, 1507762800, 1507766400, 1507770000, 1507773600, 1507777200,
     1507780600])

# 2100 UTC 11 Oct, 0000 UTC 12 Oct, 0300 UTC 12 Oct 2017
MIN_QUERY_TIMES_UNIX_SEC = numpy.array([1507755600, 1507766400, 1507777200])

# 0000 UTC 12 Oct, 0300 UTC 12 Oct, 0600 UTC 12 Oct 2017
MAX_QUERY_TIMES_UNIX_SEC = numpy.array([1507766400, 1507777200, 1507788000])

# 2100 UTC 11 Oct, 0000 UTC 12 Oct, 0300 UTC 12 Oct 2017
MODEL_TIMES_PREV_INTERP_UNIX_SEC = numpy.array(
    [1507755600, 1507766400, 1507777200])

# 0000 UTC 12 Oct, 0300 UTC 12 Oct, 0600 UTC 12 Oct 2017
MODEL_TIMES_NEXT_INTERP_UNIX_SEC = numpy.array(
    [1507766400, 1507777200, 1507788000])

# 2100 UTC 11 Oct, 0000 UTC 12 Oct, 0300 UTC 12 Oct, 0600 UTC 12 Oct 2017
MODEL_TIMES_SUB_AND_LINEAR_INTERP_UNIX_SEC = numpy.array(
    [1507755600, 1507766400, 1507777200, 1507788000])

# 1800 UTC 11 Oct, 2100 UTC 11 Oct, 0000 UTC 12 Oct, 0300 UTC 12 Oct,
# 0600 UTC 12 Oct, 0900 UTC 12 Oct 2017
MODEL_TIMES_SUPERLINEAR_INTERP_UNIX_SEC = numpy.array(
    [1507744800, 1507755600, 1507766400, 1507777200, 1507788000, 1507798800])

THIS_DICT = {nwp_model_utils.MIN_QUERY_TIME_COLUMN: MIN_QUERY_TIMES_UNIX_SEC,
             nwp_model_utils.MAX_QUERY_TIME_COLUMN: MAX_QUERY_TIMES_UNIX_SEC}
QUERY_TO_MODEL_TIMES_TABLE_PREV_INTERP = pandas.DataFrame.from_dict(THIS_DICT)
QUERY_TO_MODEL_TIMES_TABLE_NEXT_INTERP = pandas.DataFrame.from_dict(THIS_DICT)
QUERY_TO_MODEL_TIMES_TABLE_SUB_AND_LINEAR_INTERP = pandas.DataFrame.from_dict(
    THIS_DICT)
QUERY_TO_MODEL_TIMES_TABLE_SUPERLINEAR_INTERP = pandas.DataFrame.from_dict(
    THIS_DICT)

THIS_NESTED_ARRAY = QUERY_TO_MODEL_TIMES_TABLE_PREV_INTERP[[
    nwp_model_utils.MIN_QUERY_TIME_COLUMN,
    nwp_model_utils.MIN_QUERY_TIME_COLUMN]].values.tolist()
THIS_ARGUMENT_DICT = {
    nwp_model_utils.MODEL_TIMES_COLUMN: THIS_NESTED_ARRAY,
    nwp_model_utils.MODEL_TIMES_NEEDED_COLUMN: THIS_NESTED_ARRAY}

QUERY_TO_MODEL_TIMES_TABLE_PREV_INTERP = (
    QUERY_TO_MODEL_TIMES_TABLE_PREV_INTERP.assign(**THIS_ARGUMENT_DICT))
QUERY_TO_MODEL_TIMES_TABLE_NEXT_INTERP = (
    QUERY_TO_MODEL_TIMES_TABLE_NEXT_INTERP.assign(**THIS_ARGUMENT_DICT))
QUERY_TO_MODEL_TIMES_TABLE_SUB_AND_LINEAR_INTERP = (
    QUERY_TO_MODEL_TIMES_TABLE_SUB_AND_LINEAR_INTERP.assign(
        **THIS_ARGUMENT_DICT))
QUERY_TO_MODEL_TIMES_TABLE_SUPERLINEAR_INTERP = (
    QUERY_TO_MODEL_TIMES_TABLE_SUPERLINEAR_INTERP.assign(**THIS_ARGUMENT_DICT))

QUERY_TO_MODEL_TIMES_TABLE_PREV_INTERP[
    nwp_model_utils.MODEL_TIMES_COLUMN].values[0] = numpy.array([1507755600])
QUERY_TO_MODEL_TIMES_TABLE_PREV_INTERP[
    nwp_model_utils.MODEL_TIMES_COLUMN].values[1] = numpy.array([1507766400])
QUERY_TO_MODEL_TIMES_TABLE_PREV_INTERP[
    nwp_model_utils.MODEL_TIMES_COLUMN].values[2] = numpy.array([1507777200])

QUERY_TO_MODEL_TIMES_TABLE_PREV_INTERP[
    nwp_model_utils.MODEL_TIMES_NEEDED_COLUMN].values[0] = numpy.array(
        [True, False, False], dtype=bool)
QUERY_TO_MODEL_TIMES_TABLE_PREV_INTERP[
    nwp_model_utils.MODEL_TIMES_NEEDED_COLUMN].values[1] = numpy.array(
        [False, True, False], dtype=bool)
QUERY_TO_MODEL_TIMES_TABLE_PREV_INTERP[
    nwp_model_utils.MODEL_TIMES_NEEDED_COLUMN].values[2] = numpy.array(
        [False, False, True], dtype=bool)

QUERY_TO_MODEL_TIMES_TABLE_NEXT_INTERP[
    nwp_model_utils.MODEL_TIMES_COLUMN].values[0] = numpy.array([1507766400])
QUERY_TO_MODEL_TIMES_TABLE_NEXT_INTERP[
    nwp_model_utils.MODEL_TIMES_COLUMN].values[1] = numpy.array([1507777200])
QUERY_TO_MODEL_TIMES_TABLE_NEXT_INTERP[
    nwp_model_utils.MODEL_TIMES_COLUMN].values[2] = numpy.array([1507788000])

QUERY_TO_MODEL_TIMES_TABLE_NEXT_INTERP[
    nwp_model_utils.MODEL_TIMES_NEEDED_COLUMN].values[0] = numpy.array(
        [True, False, False], dtype=bool)
QUERY_TO_MODEL_TIMES_TABLE_NEXT_INTERP[
    nwp_model_utils.MODEL_TIMES_NEEDED_COLUMN].values[1] = numpy.array(
        [False, True, False], dtype=bool)
QUERY_TO_MODEL_TIMES_TABLE_NEXT_INTERP[
    nwp_model_utils.MODEL_TIMES_NEEDED_COLUMN].values[2] = numpy.array(
        [False, False, True], dtype=bool)

QUERY_TO_MODEL_TIMES_TABLE_SUB_AND_LINEAR_INTERP[
    nwp_model_utils.MODEL_TIMES_COLUMN].values[0] = numpy.array(
        [1507755600, 1507766400])
QUERY_TO_MODEL_TIMES_TABLE_SUB_AND_LINEAR_INTERP[
    nwp_model_utils.MODEL_TIMES_COLUMN].values[1] = numpy.array(
        [1507766400, 1507777200])
QUERY_TO_MODEL_TIMES_TABLE_SUB_AND_LINEAR_INTERP[
    nwp_model_utils.MODEL_TIMES_COLUMN].values[2] = numpy.array(
        [1507777200, 1507788000])

QUERY_TO_MODEL_TIMES_TABLE_SUB_AND_LINEAR_INTERP[
    nwp_model_utils.MODEL_TIMES_NEEDED_COLUMN].values[0] = numpy.array(
        [True, True, False, False], dtype=bool)
QUERY_TO_MODEL_TIMES_TABLE_SUB_AND_LINEAR_INTERP[
    nwp_model_utils.MODEL_TIMES_NEEDED_COLUMN].values[1] = numpy.array(
        [False, True, True, False], dtype=bool)
QUERY_TO_MODEL_TIMES_TABLE_SUB_AND_LINEAR_INTERP[
    nwp_model_utils.MODEL_TIMES_NEEDED_COLUMN].values[2] = numpy.array(
        [False, False, True, True], dtype=bool)

QUERY_TO_MODEL_TIMES_TABLE_SUPERLINEAR_INTERP[
    nwp_model_utils.MODEL_TIMES_COLUMN].values[0] = numpy.array(
        [1507744800, 1507755600, 1507766400, 1507777200])
QUERY_TO_MODEL_TIMES_TABLE_SUPERLINEAR_INTERP[
    nwp_model_utils.MODEL_TIMES_COLUMN].values[1] = numpy.array(
        [1507755600, 1507766400, 1507777200, 1507788000])
QUERY_TO_MODEL_TIMES_TABLE_SUPERLINEAR_INTERP[
    nwp_model_utils.MODEL_TIMES_COLUMN].values[2] = numpy.array(
        [1507766400, 1507777200, 1507788000, 1507798800])

QUERY_TO_MODEL_TIMES_TABLE_SUPERLINEAR_INTERP[
    nwp_model_utils.MODEL_TIMES_NEEDED_COLUMN].values[0] = numpy.array(
        [True, True, True, True, False, False], dtype=bool)
QUERY_TO_MODEL_TIMES_TABLE_SUPERLINEAR_INTERP[
    nwp_model_utils.MODEL_TIMES_NEEDED_COLUMN].values[1] = numpy.array(
        [False, True, True, True, True, False], dtype=bool)
QUERY_TO_MODEL_TIMES_TABLE_SUPERLINEAR_INTERP[
    nwp_model_utils.MODEL_TIMES_NEEDED_COLUMN].values[2] = numpy.array(
        [False, False, True, True, True, True], dtype=bool)

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
                query_times_unix_sec=QUERY_TIMES_UNIX_SEC,
                model_time_step_hours=MODEL_TIME_STEP_HOURS,
                method_string=FAKE_INTERP_METHOD)

    def test_get_times_needed_for_interp_previous(self):
        """Ensures correct output from get_times_needed_for_interp.

        In this case, interpolation method is previous-neighbour.
        """

        these_model_times_unix_sec, this_query_to_model_times_table = (
            nwp_model_utils.get_times_needed_for_interp(
                query_times_unix_sec=QUERY_TIMES_UNIX_SEC,
                model_time_step_hours=MODEL_TIME_STEP_HOURS,
                method_string=nwp_model_utils.PREVIOUS_INTERP_METHOD))

        self.assertTrue(numpy.array_equal(
            these_model_times_unix_sec, MODEL_TIMES_PREV_INTERP_UNIX_SEC))
        self.assertTrue(
            set(list(this_query_to_model_times_table)) ==
            set(list(QUERY_TO_MODEL_TIMES_TABLE_PREV_INTERP)))

        this_num_rows = len(this_query_to_model_times_table.index)
        for this_column in list(this_query_to_model_times_table):
            for i in range(this_num_rows):
                self.assertTrue(numpy.array_equal(
                    this_query_to_model_times_table[this_column].values[i],
                    QUERY_TO_MODEL_TIMES_TABLE_PREV_INTERP[
                        this_column].values[i]))

    def test_get_times_needed_for_interp_next(self):
        """Ensures correct output from get_times_needed_for_interp.

        In this case, interpolation method is next-neighbour.
        """

        these_model_times_unix_sec, this_query_to_model_times_table = (
            nwp_model_utils.get_times_needed_for_interp(
                query_times_unix_sec=QUERY_TIMES_UNIX_SEC,
                model_time_step_hours=MODEL_TIME_STEP_HOURS,
                method_string=nwp_model_utils.NEXT_INTERP_METHOD))

        self.assertTrue(numpy.array_equal(
            these_model_times_unix_sec, MODEL_TIMES_NEXT_INTERP_UNIX_SEC))
        self.assertTrue(
            set(list(this_query_to_model_times_table)) ==
            set(list(QUERY_TO_MODEL_TIMES_TABLE_NEXT_INTERP)))

        this_num_rows = len(this_query_to_model_times_table.index)
        for this_column in list(this_query_to_model_times_table):
            for i in range(this_num_rows):
                self.assertTrue(numpy.array_equal(
                    this_query_to_model_times_table[this_column].values[i],
                    QUERY_TO_MODEL_TIMES_TABLE_NEXT_INTERP[
                        this_column].values[i]))

    def test_get_times_needed_for_interp_sublinear(self):
        """Ensures correct output from get_times_needed_for_interp.

        In this case, interpolation method is sublinear.
        """

        these_model_times_unix_sec, this_query_to_model_times_table = (
            nwp_model_utils.get_times_needed_for_interp(
                query_times_unix_sec=QUERY_TIMES_UNIX_SEC,
                model_time_step_hours=MODEL_TIME_STEP_HOURS,
                method_string=SUBLINEAR_INTERP_METHOD))

        self.assertTrue(numpy.array_equal(
            these_model_times_unix_sec,
            MODEL_TIMES_SUB_AND_LINEAR_INTERP_UNIX_SEC))

        self.assertTrue(
            set(list(this_query_to_model_times_table)) ==
            set(list(QUERY_TO_MODEL_TIMES_TABLE_SUB_AND_LINEAR_INTERP)))

        this_num_rows = len(this_query_to_model_times_table.index)
        for this_column in list(this_query_to_model_times_table):
            for i in range(this_num_rows):
                self.assertTrue(numpy.array_equal(
                    this_query_to_model_times_table[this_column].values[i],
                    QUERY_TO_MODEL_TIMES_TABLE_SUB_AND_LINEAR_INTERP[
                        this_column].values[i]))

    def test_get_times_needed_for_interp_linear(self):
        """Ensures correct output from get_times_needed_for_interp.

        In this case, interpolation method is linear.
        """

        these_model_times_unix_sec, this_query_to_model_times_table = (
            nwp_model_utils.get_times_needed_for_interp(
                query_times_unix_sec=QUERY_TIMES_UNIX_SEC,
                model_time_step_hours=MODEL_TIME_STEP_HOURS,
                method_string=LINEAR_INTERP_METHOD))

        self.assertTrue(numpy.array_equal(
            these_model_times_unix_sec,
            MODEL_TIMES_SUB_AND_LINEAR_INTERP_UNIX_SEC))

        self.assertTrue(
            set(list(this_query_to_model_times_table)) ==
            set(list(QUERY_TO_MODEL_TIMES_TABLE_SUB_AND_LINEAR_INTERP)))

        this_num_rows = len(this_query_to_model_times_table.index)
        for this_column in list(this_query_to_model_times_table):
            for i in range(this_num_rows):
                self.assertTrue(numpy.array_equal(
                    this_query_to_model_times_table[this_column].values[i],
                    QUERY_TO_MODEL_TIMES_TABLE_SUB_AND_LINEAR_INTERP[
                        this_column].values[i]))

    def test_get_times_needed_for_interp_quadratic(self):
        """Ensures correct output from get_times_needed_for_interp.

        In this case, interpolation method is quadratic.
        """

        these_model_times_unix_sec, this_query_to_model_times_table = (
            nwp_model_utils.get_times_needed_for_interp(
                query_times_unix_sec=QUERY_TIMES_UNIX_SEC,
                model_time_step_hours=MODEL_TIME_STEP_HOURS,
                method_string=QUADRATIC_INTERP_METHOD))

        self.assertTrue(numpy.array_equal(
            these_model_times_unix_sec,
            MODEL_TIMES_SUPERLINEAR_INTERP_UNIX_SEC))

        self.assertTrue(
            set(list(this_query_to_model_times_table)) ==
            set(list(QUERY_TO_MODEL_TIMES_TABLE_SUPERLINEAR_INTERP)))

        this_num_rows = len(this_query_to_model_times_table.index)
        for this_column in list(this_query_to_model_times_table):
            for i in range(this_num_rows):
                self.assertTrue(numpy.array_equal(
                    this_query_to_model_times_table[this_column].values[i],
                    QUERY_TO_MODEL_TIMES_TABLE_SUPERLINEAR_INTERP[
                        this_column].values[i]))

    def test_get_times_needed_for_interp_cubic(self):
        """Ensures correct output from get_times_needed_for_interp.

        In this case, interpolation method is cubic.
        """

        these_model_times_unix_sec, this_query_to_model_times_table = (
            nwp_model_utils.get_times_needed_for_interp(
                query_times_unix_sec=QUERY_TIMES_UNIX_SEC,
                model_time_step_hours=MODEL_TIME_STEP_HOURS,
                method_string=CUBIC_INTERP_METHOD))

        self.assertTrue(numpy.array_equal(
            these_model_times_unix_sec,
            MODEL_TIMES_SUPERLINEAR_INTERP_UNIX_SEC))

        self.assertTrue(
            set(list(this_query_to_model_times_table)) ==
            set(list(QUERY_TO_MODEL_TIMES_TABLE_SUPERLINEAR_INTERP)))

        this_num_rows = len(this_query_to_model_times_table.index)
        for this_column in list(this_query_to_model_times_table):
            for i in range(this_num_rows):
                self.assertTrue(numpy.array_equal(
                    this_query_to_model_times_table[this_column].values[i],
                    QUERY_TO_MODEL_TIMES_TABLE_SUPERLINEAR_INTERP[
                        this_column].values[i]))

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
