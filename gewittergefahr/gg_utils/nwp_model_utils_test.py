"""Unit tests for nwp_model_utils.py."""

import unittest
import numpy
import pandas
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import interp

TOLERANCE = 1e-6

FAKE_MODEL_NAME = 'foo'
FAKE_GRID_DIMENSIONS = numpy.array([1, 2])

MAX_MEAN_DISTANCE_ERROR_RAP_METRES = 100.
MAX_MAX_DISTANCE_ERROR_RAP_METRES = 500.
GRID130_LATLNG_FILE_NAME = 'grid_point_latlng_grid130.data'
GRID252_LATLNG_FILE_NAME = 'grid_point_latlng_grid252.data'

MAX_MEAN_DISTANCE_ERROR_RUC_METRES = 100.
MAX_MAX_DISTANCE_ERROR_RUC_METRES = 200.
GRID236_LATLNG_FILE_NAME = 'grid_point_latlng_grid236.data'

MAX_MEAN_DISTANCE_ERROR_NARR_METRES = 250.
MAX_MAX_DISTANCE_ERROR_NARR_METRES = 1000.
NARR_LATLNG_FILE_NAME = 'grid_point_latlng_grid221.data'

MAX_MEAN_SIN_OR_COS_ERROR = 1e-5
MAX_MAX_SIN_OR_COS_ERROR = 1e-4
GRID130_WIND_ROTATION_FILE_NAME = 'wind_rotation_angles_grid130.data'
GRID252_WIND_ROTATION_FILE_NAME = 'wind_rotation_angles_grid252.data'
GRID236_WIND_ROTATION_FILE_NAME = 'wind_rotation_angles_grid236.data'

# The following constants are used to test get_times_needed_for_interp.
MODEL_TIME_STEP_HOURS = 3
QUERY_TIMES_UNIX_SEC = numpy.array(
    [14400, 18000, 21600, 25200, 28800, 32400, 36000])
MIN_QUERY_TIMES_UNIX_SEC = numpy.array([10800, 21600, 32400])
MAX_QUERY_TIMES_UNIX_SEC = numpy.array([21600, 32400, 43200])
MODEL_TIMES_PREV_INTERP_UNIX_SEC = numpy.array([10800, 21600, 32400])
MODEL_TIMES_NEXT_INTERP_UNIX_SEC = numpy.array([21600, 32400, 43200])
MODEL_TIMES_SUB_AND_LINEAR_INTERP_UNIX_SEC = numpy.array(
    [10800, 21600, 32400, 43200])
MODEL_TIMES_SUPERLINEAR_INTERP_UNIX_SEC = numpy.array(
    [0, 10800, 21600, 32400, 43200, 54000])

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
    nwp_model_utils.MODEL_TIMES_COLUMN].values[0] = numpy.array([10800])
QUERY_TO_MODEL_TIMES_TABLE_PREV_INTERP[
    nwp_model_utils.MODEL_TIMES_COLUMN].values[1] = numpy.array([21600])
QUERY_TO_MODEL_TIMES_TABLE_PREV_INTERP[
    nwp_model_utils.MODEL_TIMES_COLUMN].values[2] = numpy.array([32400])

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
    nwp_model_utils.MODEL_TIMES_COLUMN].values[0] = numpy.array([21600])
QUERY_TO_MODEL_TIMES_TABLE_NEXT_INTERP[
    nwp_model_utils.MODEL_TIMES_COLUMN].values[1] = numpy.array([32400])
QUERY_TO_MODEL_TIMES_TABLE_NEXT_INTERP[
    nwp_model_utils.MODEL_TIMES_COLUMN].values[2] = numpy.array([43200])

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
    nwp_model_utils.MODEL_TIMES_COLUMN].values[0] = numpy.array([10800, 21600])
QUERY_TO_MODEL_TIMES_TABLE_SUB_AND_LINEAR_INTERP[
    nwp_model_utils.MODEL_TIMES_COLUMN].values[1] = numpy.array([21600, 32400])
QUERY_TO_MODEL_TIMES_TABLE_SUB_AND_LINEAR_INTERP[
    nwp_model_utils.MODEL_TIMES_COLUMN].values[2] = numpy.array([32400, 43200])

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
        [0, 10800, 21600, 32400])
QUERY_TO_MODEL_TIMES_TABLE_SUPERLINEAR_INTERP[
    nwp_model_utils.MODEL_TIMES_COLUMN].values[1] = numpy.array(
        [10800, 21600, 32400, 43200])
QUERY_TO_MODEL_TIMES_TABLE_SUPERLINEAR_INTERP[
    nwp_model_utils.MODEL_TIMES_COLUMN].values[2] = numpy.array(
        [21600, 32400, 43200, 54000])

QUERY_TO_MODEL_TIMES_TABLE_SUPERLINEAR_INTERP[
    nwp_model_utils.MODEL_TIMES_NEEDED_COLUMN].values[0] = numpy.array(
        [True, True, True, True, False, False], dtype=bool)
QUERY_TO_MODEL_TIMES_TABLE_SUPERLINEAR_INTERP[
    nwp_model_utils.MODEL_TIMES_NEEDED_COLUMN].values[1] = numpy.array(
        [False, True, True, True, True, False], dtype=bool)
QUERY_TO_MODEL_TIMES_TABLE_SUPERLINEAR_INTERP[
    nwp_model_utils.MODEL_TIMES_NEEDED_COLUMN].values[2] = numpy.array(
        [False, False, True, True, True, True], dtype=bool)

# The following constants are used to test rotate_winds_to_earth_relative and
# rotate_winds_to_grid_relative.
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

    def test_check_model_name_rap(self):
        """Ensures correct output from check_model_name when model is RAP."""

        nwp_model_utils.check_model_name(nwp_model_utils.RAP_MODEL_NAME)

    def test_check_model_name_ruc(self):
        """Ensures correct output from check_model_name when model is RUC."""

        nwp_model_utils.check_model_name(nwp_model_utils.RUC_MODEL_NAME)

    def test_check_model_name_narr(self):
        """Ensures correct output from check_model_name when model is NARR."""

        nwp_model_utils.check_model_name(nwp_model_utils.NARR_MODEL_NAME)

    def test_check_model_name_fake(self):
        """Ensures correct output from check_model_name when model is fake."""

        with self.assertRaises(ValueError):
            nwp_model_utils.check_model_name(FAKE_MODEL_NAME)

    def test_check_grid_id_narr(self):
        """Ensures correct output from check_grid_id when model is NARR."""

        nwp_model_utils.check_grid_id(nwp_model_utils.NARR_MODEL_NAME)

    def test_check_grid_id_rap130(self):
        """Ensures correct output from check_grid_id for RAP on 130 grid."""

        nwp_model_utils.check_grid_id(nwp_model_utils.RAP_MODEL_NAME,
                                      nwp_model_utils.ID_FOR_130GRID)

    def test_check_grid_id_rap252(self):
        """Ensures correct output from check_grid_id for RAP on 252 grid."""

        nwp_model_utils.check_grid_id(nwp_model_utils.RAP_MODEL_NAME,
                                      nwp_model_utils.ID_FOR_252GRID)

    def test_check_grid_id_rap236(self):
        """Ensures correct output from check_grid_id for RAP on 236 grid."""

        with self.assertRaises(ValueError):
            nwp_model_utils.check_grid_id(nwp_model_utils.RAP_MODEL_NAME,
                                          nwp_model_utils.ID_FOR_236GRID)

    def test_check_grid_id_ruc130(self):
        """Ensures correct output from check_grid_id for RUC on 130 grid."""

        nwp_model_utils.check_grid_id(nwp_model_utils.RUC_MODEL_NAME,
                                      nwp_model_utils.ID_FOR_130GRID)

    def test_check_grid_id_ruc252(self):
        """Ensures correct output from check_grid_id for RUC on 252 grid."""

        nwp_model_utils.check_grid_id(nwp_model_utils.RUC_MODEL_NAME,
                                      nwp_model_utils.ID_FOR_252GRID)

    def test_check_grid_id_ruc236(self):
        """Ensures correct output from check_grid_id for RUC on 236 grid."""

        nwp_model_utils.check_grid_id(nwp_model_utils.RUC_MODEL_NAME,
                                      nwp_model_utils.ID_FOR_236GRID)

    def test_dimensions_to_grid_id_221(self):
        """Ensures correct output from dimensions_to_grid_id for 221 grid."""

        these_dimensions = numpy.array(nwp_model_utils.get_grid_dimensions(
            nwp_model_utils.NARR_MODEL_NAME))

        this_grid_id = nwp_model_utils.dimensions_to_grid_id(these_dimensions)
        self.assertTrue(this_grid_id == nwp_model_utils.ID_FOR_221GRID)

    def test_dimensions_to_grid_id_130(self):
        """Ensures correct output from dimensions_to_grid_id for 130 grid."""

        these_dimensions = numpy.array(nwp_model_utils.get_grid_dimensions(
            nwp_model_utils.RUC_MODEL_NAME, nwp_model_utils.ID_FOR_130GRID))

        this_grid_id = nwp_model_utils.dimensions_to_grid_id(these_dimensions)
        self.assertTrue(this_grid_id == nwp_model_utils.ID_FOR_130GRID)

    def test_dimensions_to_grid_id_252(self):
        """Ensures correct output from dimensions_to_grid_id for 252 grid."""

        these_dimensions = numpy.array(nwp_model_utils.get_grid_dimensions(
            nwp_model_utils.RUC_MODEL_NAME, nwp_model_utils.ID_FOR_252GRID))

        this_grid_id = nwp_model_utils.dimensions_to_grid_id(these_dimensions)
        self.assertTrue(this_grid_id == nwp_model_utils.ID_FOR_252GRID)

    def test_dimensions_to_grid_id_236(self):
        """Ensures correct output from dimensions_to_grid_id for 236 grid."""

        these_dimensions = numpy.array(nwp_model_utils.get_grid_dimensions(
            nwp_model_utils.RUC_MODEL_NAME, nwp_model_utils.ID_FOR_236GRID))

        this_grid_id = nwp_model_utils.dimensions_to_grid_id(these_dimensions)
        self.assertTrue(this_grid_id == nwp_model_utils.ID_FOR_236GRID)

    def test_dimensions_to_grid_id_fake(self):
        """Ensures correct output from dimensions_to_grid_id for fake grid."""

        with self.assertRaises(ValueError):
            nwp_model_utils.dimensions_to_grid_id(FAKE_GRID_DIMENSIONS)

    def test_get_times_needed_for_interp_previous(self):
        """Ensures correct output from get_times_needed_for_interp.

        In this case, interpolation method is previous-neighbour.
        """

        these_model_times_unix_sec, this_query_to_model_times_table = (
            nwp_model_utils.get_times_needed_for_interp(
                query_times_unix_sec=QUERY_TIMES_UNIX_SEC,
                model_time_step_hours=MODEL_TIME_STEP_HOURS,
                method_string=interp.PREVIOUS_INTERP_METHOD))

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
                method_string=interp.NEXT_INTERP_METHOD))

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

    def test_get_times_needed_for_interp_nearest(self):
        """Ensures correct output from get_times_needed_for_interp.

        In this case, interpolation method is nearest-neighbour.
        """

        these_model_times_unix_sec, this_query_to_model_times_table = (
            nwp_model_utils.get_times_needed_for_interp(
                query_times_unix_sec=QUERY_TIMES_UNIX_SEC,
                model_time_step_hours=MODEL_TIME_STEP_HOURS,
                method_string=interp.NEAREST_INTERP_METHOD))

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
                method_string=interp.LINEAR_INTERP_METHOD))

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
                method_string=interp.SPLINE2_INTERP_METHOD))

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
                method_string=interp.SPLINE3_INTERP_METHOD))

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

    def test_rotate_winds_to_earth_relative(self):
        """Ensures correct output from rotate_winds_to_earth_relative."""

        (these_u_winds_earth_relative_m_s01,
         these_v_winds_earth_relative_m_s01) = (
             nwp_model_utils.rotate_winds_to_earth_relative(
                 u_winds_grid_relative_m_s01=U_WINDS_GRID_RELATIVE_M_S01,
                 v_winds_grid_relative_m_s01=V_WINDS_GRID_RELATIVE_M_S01,
                 rotation_angle_cosines=ROTATION_ANGLE_COSINES,
                 rotation_angle_sines=ROTATION_ANGLE_SINES))

        self.assertTrue(numpy.allclose(
            these_u_winds_earth_relative_m_s01, U_WINDS_EARTH_RELATIVE_M_S01,
            atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_v_winds_earth_relative_m_s01, V_WINDS_EARTH_RELATIVE_M_S01,
            atol=TOLERANCE))

    def test_rotate_winds_to_grid_relative(self):
        """Ensures correct output from rotate_winds_to_grid_relative."""

        (these_u_winds_grid_relative_m_s01,
         these_v_winds_grid_relative_m_s01) = (
             nwp_model_utils.rotate_winds_to_grid_relative(
                 u_winds_earth_relative_m_s01=U_WINDS_EARTH_RELATIVE_M_S01,
                 v_winds_earth_relative_m_s01=V_WINDS_EARTH_RELATIVE_M_S01,
                 rotation_angle_cosines=ROTATION_ANGLE_COSINES,
                 rotation_angle_sines=ROTATION_ANGLE_SINES))

        self.assertTrue(numpy.allclose(
            these_u_winds_grid_relative_m_s01, U_WINDS_GRID_RELATIVE_M_S01,
            atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_v_winds_grid_relative_m_s01, V_WINDS_GRID_RELATIVE_M_S01,
            atol=TOLERANCE))

    def test_projection_grid130(self):
        """Ensures approx correctness of Lambert projection for NCEP 130 grid.

        This method ensures that x-y coordinates for all grid points can be
        generated accurately from lat-long coordinates.  Specifically, the mean
        and max distance errors must be <= 100 and 500 metres, respectively.

        NOTE: This test relies on several methods, so it is not a unit test.
        """

        num_grid_rows, num_grid_columns = nwp_model_utils.get_grid_dimensions(
            model_name=nwp_model_utils.RAP_MODEL_NAME,
            grid_id=nwp_model_utils.ID_FOR_130GRID)

        (grid_point_lng_vector_deg, grid_point_lat_vector_deg) = numpy.loadtxt(
            GRID130_LATLNG_FILE_NAME, unpack=True)
        grid_point_lat_matrix_deg = numpy.reshape(
            grid_point_lat_vector_deg, (num_grid_rows, num_grid_columns))
        grid_point_lng_matrix_deg = numpy.reshape(
            grid_point_lng_vector_deg, (num_grid_rows, num_grid_columns))

        grid_point_x_matrix_metres, grid_point_y_matrix_metres = (
            nwp_model_utils.project_latlng_to_xy(
                grid_point_lat_matrix_deg, grid_point_lng_matrix_deg,
                model_name=nwp_model_utils.RAP_MODEL_NAME,
                grid_id=nwp_model_utils.ID_FOR_130GRID))

        (expected_grid_point_x_matrix_metres,
         expected_grid_point_y_matrix_metres) = (
             nwp_model_utils.get_xy_grid_point_matrices(
                 model_name=nwp_model_utils.RAP_MODEL_NAME,
                 grid_id=nwp_model_utils.ID_FOR_130GRID))

        x_error_matrix_metres = (
            grid_point_x_matrix_metres - expected_grid_point_x_matrix_metres)
        y_error_matrix_metres = (
            grid_point_y_matrix_metres - expected_grid_point_y_matrix_metres)
        distance_error_matrix_metres = numpy.sqrt(
            x_error_matrix_metres ** 2 + y_error_matrix_metres ** 2)

        self.assertTrue(numpy.mean(
            distance_error_matrix_metres) <= MAX_MEAN_DISTANCE_ERROR_RAP_METRES)
        self.assertTrue(numpy.max(
            distance_error_matrix_metres) <= MAX_MAX_DISTANCE_ERROR_RAP_METRES)

    def test_projection_grid252(self):
        """Ensures approx correctness of Lambert projection for NCEP 252 grid.

        See documentation for test_projection_grid130.
        """

        num_grid_rows, num_grid_columns = nwp_model_utils.get_grid_dimensions(
            model_name=nwp_model_utils.RAP_MODEL_NAME,
            grid_id=nwp_model_utils.ID_FOR_252GRID)

        (grid_point_lng_vector_deg, grid_point_lat_vector_deg) = numpy.loadtxt(
            GRID252_LATLNG_FILE_NAME, unpack=True)
        grid_point_lat_matrix_deg = numpy.reshape(
            grid_point_lat_vector_deg, (num_grid_rows, num_grid_columns))
        grid_point_lng_matrix_deg = numpy.reshape(
            grid_point_lng_vector_deg, (num_grid_rows, num_grid_columns))

        grid_point_x_matrix_metres, grid_point_y_matrix_metres = (
            nwp_model_utils.project_latlng_to_xy(
                grid_point_lat_matrix_deg, grid_point_lng_matrix_deg,
                model_name=nwp_model_utils.RAP_MODEL_NAME,
                grid_id=nwp_model_utils.ID_FOR_252GRID))

        (expected_grid_point_x_matrix_metres,
         expected_grid_point_y_matrix_metres) = (
             nwp_model_utils.get_xy_grid_point_matrices(
                 model_name=nwp_model_utils.RAP_MODEL_NAME,
                 grid_id=nwp_model_utils.ID_FOR_252GRID))

        x_error_matrix_metres = (
            grid_point_x_matrix_metres - expected_grid_point_x_matrix_metres)
        y_error_matrix_metres = (
            grid_point_y_matrix_metres - expected_grid_point_y_matrix_metres)
        distance_error_matrix_metres = numpy.sqrt(
            x_error_matrix_metres ** 2 + y_error_matrix_metres ** 2)

        self.assertTrue(numpy.mean(
            distance_error_matrix_metres) <= MAX_MEAN_DISTANCE_ERROR_RAP_METRES)
        self.assertTrue(numpy.max(
            distance_error_matrix_metres) <= MAX_MAX_DISTANCE_ERROR_RAP_METRES)

    def test_projection_grid236(self):
        """Ensures approx correctness of Lambert projection for NCEP 236 grid.

        See documentation for test_projection_grid130.
        """

        num_grid_rows, num_grid_columns = nwp_model_utils.get_grid_dimensions(
            model_name=nwp_model_utils.RUC_MODEL_NAME,
            grid_id=nwp_model_utils.ID_FOR_236GRID)

        (grid_point_lng_vector_deg, grid_point_lat_vector_deg) = numpy.loadtxt(
            GRID236_LATLNG_FILE_NAME, unpack=True)
        grid_point_lat_matrix_deg = numpy.reshape(
            grid_point_lat_vector_deg, (num_grid_rows, num_grid_columns))
        grid_point_lng_matrix_deg = numpy.reshape(
            grid_point_lng_vector_deg, (num_grid_rows, num_grid_columns))

        grid_point_x_matrix_metres, grid_point_y_matrix_metres = (
            nwp_model_utils.project_latlng_to_xy(
                grid_point_lat_matrix_deg, grid_point_lng_matrix_deg,
                model_name=nwp_model_utils.RUC_MODEL_NAME,
                grid_id=nwp_model_utils.ID_FOR_236GRID))

        (expected_grid_point_x_matrix_metres,
         expected_grid_point_y_matrix_metres) = (
             nwp_model_utils.get_xy_grid_point_matrices(
                 model_name=nwp_model_utils.RUC_MODEL_NAME,
                 grid_id=nwp_model_utils.ID_FOR_236GRID))

        x_error_matrix_metres = (
            grid_point_x_matrix_metres - expected_grid_point_x_matrix_metres)
        y_error_matrix_metres = (
            grid_point_y_matrix_metres - expected_grid_point_y_matrix_metres)
        distance_error_matrix_metres = numpy.sqrt(
            x_error_matrix_metres ** 2 + y_error_matrix_metres ** 2)

        self.assertTrue(numpy.mean(
            distance_error_matrix_metres) <= MAX_MEAN_DISTANCE_ERROR_RUC_METRES)
        self.assertTrue(numpy.max(
            distance_error_matrix_metres) <= MAX_MAX_DISTANCE_ERROR_RUC_METRES)

    def test_projection_narr(self):
        """Ensures approx correctness of Lambert projection for NARR grid.

        See documentation for test_projection_grid130.
        """

        num_grid_rows, num_grid_columns = nwp_model_utils.get_grid_dimensions(
            nwp_model_utils.NARR_MODEL_NAME)

        (grid_point_lng_vector_deg, grid_point_lat_vector_deg) = numpy.loadtxt(
            NARR_LATLNG_FILE_NAME, unpack=True)
        grid_point_lat_matrix_deg = numpy.reshape(
            grid_point_lat_vector_deg, (num_grid_rows, num_grid_columns))
        grid_point_lng_matrix_deg = numpy.reshape(
            grid_point_lng_vector_deg, (num_grid_rows, num_grid_columns))

        grid_point_x_matrix_metres, grid_point_y_matrix_metres = (
            nwp_model_utils.project_latlng_to_xy(
                grid_point_lat_matrix_deg, grid_point_lng_matrix_deg,
                model_name=nwp_model_utils.NARR_MODEL_NAME))

        (expected_grid_point_x_matrix_metres,
         expected_grid_point_y_matrix_metres) = (
             nwp_model_utils.get_xy_grid_point_matrices(
                 nwp_model_utils.NARR_MODEL_NAME))

        x_error_matrix_metres = (
            grid_point_x_matrix_metres - expected_grid_point_x_matrix_metres)
        y_error_matrix_metres = (
            grid_point_y_matrix_metres - expected_grid_point_y_matrix_metres)
        distance_error_matrix_metres = numpy.sqrt(
            x_error_matrix_metres ** 2 + y_error_matrix_metres ** 2)

        self.assertTrue(numpy.mean(distance_error_matrix_metres) <=
                        MAX_MEAN_DISTANCE_ERROR_NARR_METRES)
        self.assertTrue(numpy.max(
            distance_error_matrix_metres) <= MAX_MAX_DISTANCE_ERROR_NARR_METRES)

    def test_wind_rotation_angles_grid130(self):
        """Ensures approx correctness of wind-rotation angles on NCEP 130 grid.

        This method ensures that the wind-rotation angle for all grid points can
        be generated accurately from lat-long coordinates.  Specifically, the
        mean sine and cosine error must be <= 10^-5 and max error must be
        <= 10^-4.

        NOTE: This test relies on methods other than get_wind_rotation_angles,
              so it is not a unit test.
        """

        num_grid_rows, num_grid_columns = nwp_model_utils.get_grid_dimensions(
            model_name=nwp_model_utils.RAP_MODEL_NAME,
            grid_id=nwp_model_utils.ID_FOR_130GRID)

        expected_cos_vector, expected_sin_vector = numpy.loadtxt(
            GRID130_WIND_ROTATION_FILE_NAME, unpack=True)
        expected_cos_matrix = numpy.reshape(
            expected_cos_vector, (num_grid_rows, num_grid_columns))
        expected_sin_matrix = numpy.reshape(
            expected_sin_vector, (num_grid_rows, num_grid_columns))

        grid_point_lat_matrix_deg, grid_point_lng_matrix_deg = (
            nwp_model_utils.get_latlng_grid_point_matrices(
                model_name=nwp_model_utils.RAP_MODEL_NAME,
                grid_id=nwp_model_utils.ID_FOR_130GRID))
        rotation_angle_cos_matrix, rotation_angle_sin_matrix = (
            nwp_model_utils.get_wind_rotation_angles(
                grid_point_lat_matrix_deg, grid_point_lng_matrix_deg,
                model_name=nwp_model_utils.RAP_MODEL_NAME))

        cos_error_matrix = numpy.absolute(
            rotation_angle_cos_matrix - expected_cos_matrix)
        sin_error_matrix = numpy.absolute(
            rotation_angle_sin_matrix - expected_sin_matrix)

        self.assertTrue(
            numpy.mean(cos_error_matrix) <= MAX_MEAN_SIN_OR_COS_ERROR)
        self.assertTrue(numpy.max(cos_error_matrix) <= MAX_MAX_SIN_OR_COS_ERROR)

        self.assertTrue(
            numpy.mean(sin_error_matrix) <= MAX_MEAN_SIN_OR_COS_ERROR)
        self.assertTrue(numpy.max(sin_error_matrix) <= MAX_MAX_SIN_OR_COS_ERROR)

    def test_wind_rotation_angles_grid252(self):
        """Ensures approx correctness of wind-rotation angles on NCEP 252 grid.

        See documentation for test_wind_rotation_angles_grid130.
        """

        num_grid_rows, num_grid_columns = nwp_model_utils.get_grid_dimensions(
            model_name=nwp_model_utils.RAP_MODEL_NAME,
            grid_id=nwp_model_utils.ID_FOR_252GRID)

        expected_cos_vector, expected_sin_vector = numpy.loadtxt(
            GRID252_WIND_ROTATION_FILE_NAME, unpack=True)
        expected_cos_matrix = numpy.reshape(
            expected_cos_vector, (num_grid_rows, num_grid_columns))
        expected_sin_matrix = numpy.reshape(
            expected_sin_vector, (num_grid_rows, num_grid_columns))

        grid_point_lat_matrix_deg, grid_point_lng_matrix_deg = (
            nwp_model_utils.get_latlng_grid_point_matrices(
                model_name=nwp_model_utils.RAP_MODEL_NAME,
                grid_id=nwp_model_utils.ID_FOR_252GRID))
        rotation_angle_cos_matrix, rotation_angle_sin_matrix = (
            nwp_model_utils.get_wind_rotation_angles(
                grid_point_lat_matrix_deg, grid_point_lng_matrix_deg,
                model_name=nwp_model_utils.RAP_MODEL_NAME))

        cos_error_matrix = numpy.absolute(
            rotation_angle_cos_matrix - expected_cos_matrix)
        sin_error_matrix = numpy.absolute(
            rotation_angle_sin_matrix - expected_sin_matrix)

        self.assertTrue(
            numpy.mean(cos_error_matrix) <= MAX_MEAN_SIN_OR_COS_ERROR)
        self.assertTrue(numpy.max(cos_error_matrix) <= MAX_MAX_SIN_OR_COS_ERROR)

        self.assertTrue(
            numpy.mean(sin_error_matrix) <= MAX_MEAN_SIN_OR_COS_ERROR)
        self.assertTrue(numpy.max(sin_error_matrix) <= MAX_MAX_SIN_OR_COS_ERROR)


if __name__ == '__main__':
    unittest.main()
