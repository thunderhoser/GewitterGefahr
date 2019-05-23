"""Unit tests for nwp_model_utils.py."""

import os.path
import unittest
import numpy
import pandas
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import interp

TOLERANCE = 1e-6

MAX_MEAN_DISTANCE_ERROR_RAP_METRES = 100.
MAX_MAX_DISTANCE_ERROR_RAP_METRES = 500.
MAX_MEAN_DISTANCE_ERROR_NARR_METRES = 250.
MAX_MAX_DISTANCE_ERROR_NARR_METRES = 1000.

GRID130_LATLNG_FILE_NAME = '{0:s}/grid_point_latlng_grid130.data'.format(
    os.path.dirname(__file__)
)
GRID252_LATLNG_FILE_NAME = '{0:s}/grid_point_latlng_grid252.data'.format(
    os.path.dirname(__file__)
)
NARR_LATLNG_FILE_NAME = '{0:s}/grid_point_latlng_grid221.data'.format(
    os.path.dirname(__file__)
)

MAX_MEAN_SIN_OR_COS_ERROR = 1e-5
MAX_MAX_SIN_OR_COS_ERROR = 1e-4

GRID130_WIND_ROTATION_FILE_NAME = (
    '{0:s}/wind_rotation_angles_grid130.data'
).format(os.path.dirname(__file__))
GRID252_WIND_ROTATION_FILE_NAME = (
    '{0:s}/wind_rotation_angles_grid252.data'
).format(os.path.dirname(__file__))

# The following constants are used to test get_times_needed_for_interp.
MODEL_TIME_STEP_HOURS = 3
QUERY_TIMES_UNIX_SEC = numpy.array(
    [14400, 18000, 21600, 25200, 28800, 32400, 36000], dtype=int
)

MIN_QUERY_TIMES_UNIX_SEC = numpy.array([10800, 21600, 32400], dtype=int)
MAX_QUERY_TIMES_UNIX_SEC = numpy.array([21600, 32400, 43200], dtype=int)

MODEL_TIMES_PREV_INTERP_UNIX_SEC = numpy.array(
    [10800, 21600, 32400], dtype=int
)
MODEL_TIMES_NEXT_INTERP_UNIX_SEC = numpy.array(
    [21600, 32400, 43200], dtype=int
)
MODEL_TIMES_SUB_AND_LINEAR_INTERP_UNIX_SEC = numpy.array(
    [10800, 21600, 32400, 43200], dtype=int
)
MODEL_TIMES_SUPERLINEAR_INTERP_UNIX_SEC = numpy.array(
    [0, 10800, 21600, 32400, 43200, 54000], dtype=int
)

THIS_DICT = {
    nwp_model_utils.MIN_QUERY_TIME_COLUMN: MIN_QUERY_TIMES_UNIX_SEC,
    nwp_model_utils.MAX_QUERY_TIME_COLUMN: MAX_QUERY_TIMES_UNIX_SEC
}

INTERP_TIME_TABLE_PREV_INTERP = pandas.DataFrame.from_dict(THIS_DICT)
INTERP_TIME_TABLE_NEXT_INTERP = pandas.DataFrame.from_dict(THIS_DICT)
INTERP_TIME_TABLE_SUB_AND_LINEAR_INTERP = pandas.DataFrame.from_dict(THIS_DICT)
INTERP_TIME_TABLE_SUPERLINEAR_INTERP = pandas.DataFrame.from_dict(THIS_DICT)

THIS_NESTED_ARRAY = INTERP_TIME_TABLE_PREV_INTERP[[
    nwp_model_utils.MIN_QUERY_TIME_COLUMN,
    nwp_model_utils.MIN_QUERY_TIME_COLUMN
]].values.tolist()

THIS_ARGUMENT_DICT = {
    nwp_model_utils.MODEL_TIMES_COLUMN: THIS_NESTED_ARRAY,
    nwp_model_utils.MODEL_TIMES_NEEDED_COLUMN: THIS_NESTED_ARRAY
}

INTERP_TIME_TABLE_PREV_INTERP = INTERP_TIME_TABLE_PREV_INTERP.assign(
    **THIS_ARGUMENT_DICT)
INTERP_TIME_TABLE_NEXT_INTERP = INTERP_TIME_TABLE_NEXT_INTERP.assign(
    **THIS_ARGUMENT_DICT)
INTERP_TIME_TABLE_SUB_AND_LINEAR_INTERP = (
    INTERP_TIME_TABLE_SUB_AND_LINEAR_INTERP.assign(**THIS_ARGUMENT_DICT)
)
INTERP_TIME_TABLE_SUPERLINEAR_INTERP = (
    INTERP_TIME_TABLE_SUPERLINEAR_INTERP.assign(**THIS_ARGUMENT_DICT)
)

INTERP_TIME_TABLE_PREV_INTERP[nwp_model_utils.MODEL_TIMES_COLUMN].values[0] = (
    numpy.array([10800], dtype=int)
)
INTERP_TIME_TABLE_PREV_INTERP[nwp_model_utils.MODEL_TIMES_COLUMN].values[1] = (
    numpy.array([21600], dtype=int)
)
INTERP_TIME_TABLE_PREV_INTERP[nwp_model_utils.MODEL_TIMES_COLUMN].values[2] = (
    numpy.array([32400], dtype=int)
)

INTERP_TIME_TABLE_PREV_INTERP[
    nwp_model_utils.MODEL_TIMES_NEEDED_COLUMN
].values[0] = numpy.array([1, 0, 0], dtype=bool)

INTERP_TIME_TABLE_PREV_INTERP[
    nwp_model_utils.MODEL_TIMES_NEEDED_COLUMN
].values[1] = numpy.array([0, 1, 0], dtype=bool)

INTERP_TIME_TABLE_PREV_INTERP[
    nwp_model_utils.MODEL_TIMES_NEEDED_COLUMN
].values[2] = numpy.array([0, 0, 1], dtype=bool)

INTERP_TIME_TABLE_NEXT_INTERP[nwp_model_utils.MODEL_TIMES_COLUMN].values[0] = (
    numpy.array([21600], dtype=int)
)
INTERP_TIME_TABLE_NEXT_INTERP[nwp_model_utils.MODEL_TIMES_COLUMN].values[1] = (
    numpy.array([32400], dtype=int)
)
INTERP_TIME_TABLE_NEXT_INTERP[nwp_model_utils.MODEL_TIMES_COLUMN].values[2] = (
    numpy.array([43200], dtype=int)
)

INTERP_TIME_TABLE_NEXT_INTERP[
    nwp_model_utils.MODEL_TIMES_NEEDED_COLUMN
].values[0] = numpy.array([1, 0, 0], dtype=bool)

INTERP_TIME_TABLE_NEXT_INTERP[
    nwp_model_utils.MODEL_TIMES_NEEDED_COLUMN
].values[1] = numpy.array([0, 1, 0], dtype=bool)

INTERP_TIME_TABLE_NEXT_INTERP[
    nwp_model_utils.MODEL_TIMES_NEEDED_COLUMN
].values[2] = numpy.array([0, 0, 1], dtype=bool)

INTERP_TIME_TABLE_SUB_AND_LINEAR_INTERP[
    nwp_model_utils.MODEL_TIMES_COLUMN
].values[0] = numpy.array([10800, 21600], dtype=int)

INTERP_TIME_TABLE_SUB_AND_LINEAR_INTERP[
    nwp_model_utils.MODEL_TIMES_COLUMN
].values[1] = numpy.array([21600, 32400], dtype=int)

INTERP_TIME_TABLE_SUB_AND_LINEAR_INTERP[
    nwp_model_utils.MODEL_TIMES_COLUMN
].values[2] = numpy.array([32400, 43200], dtype=int)

INTERP_TIME_TABLE_SUB_AND_LINEAR_INTERP[
    nwp_model_utils.MODEL_TIMES_NEEDED_COLUMN
].values[0] = numpy.array([1, 1, 0, 0], dtype=bool)

INTERP_TIME_TABLE_SUB_AND_LINEAR_INTERP[
    nwp_model_utils.MODEL_TIMES_NEEDED_COLUMN
].values[1] = numpy.array([0, 1, 1, 0], dtype=bool)

INTERP_TIME_TABLE_SUB_AND_LINEAR_INTERP[
    nwp_model_utils.MODEL_TIMES_NEEDED_COLUMN
].values[2] = numpy.array([0, 0, 1, 1], dtype=bool)

INTERP_TIME_TABLE_SUPERLINEAR_INTERP[
    nwp_model_utils.MODEL_TIMES_COLUMN
].values[0] = numpy.array([0, 10800, 21600, 32400], dtype=int)

INTERP_TIME_TABLE_SUPERLINEAR_INTERP[
    nwp_model_utils.MODEL_TIMES_COLUMN
].values[1] = numpy.array([10800, 21600, 32400, 43200], dtype=int)

INTERP_TIME_TABLE_SUPERLINEAR_INTERP[
    nwp_model_utils.MODEL_TIMES_COLUMN
].values[2] = numpy.array([21600, 32400, 43200, 54000], dtype=int)

INTERP_TIME_TABLE_SUPERLINEAR_INTERP[
    nwp_model_utils.MODEL_TIMES_NEEDED_COLUMN
].values[0] = numpy.array([1, 1, 1, 1, 0, 0], dtype=bool)

INTERP_TIME_TABLE_SUPERLINEAR_INTERP[
    nwp_model_utils.MODEL_TIMES_NEEDED_COLUMN
].values[1] = numpy.array([0, 1, 1, 1, 1, 0], dtype=bool)

INTERP_TIME_TABLE_SUPERLINEAR_INTERP[
    nwp_model_utils.MODEL_TIMES_NEEDED_COLUMN
].values[2] = numpy.array([0, 0, 1, 1, 1, 1], dtype=bool)

# The following constants are used to test rotate_winds_to_earth_relative and
# rotate_winds_to_grid_relative.
HALF_ROOT3 = numpy.sqrt(3) / 2

U_WINDS_GRID_RELATIVE_M_S01 = numpy.array([
    [0, 5, 10], [0, -5, -10]
], dtype=float)

V_WINDS_GRID_RELATIVE_M_S01 = numpy.array([
    [10, 15, 20], [-10, -15, -20]
], dtype=float)

ROTATION_ANGLE_COSINES = numpy.array([
    [1, 0.5, -0.5], [-1, -0.5, 0.5]
])

ROTATION_ANGLE_SINES = numpy.array([
    [0, HALF_ROOT3, HALF_ROOT3], [0, -HALF_ROOT3, -HALF_ROOT3]
])

U_WINDS_EARTH_RELATIVE_M_S01 = numpy.array([
    [0, 2.5 + 15 * HALF_ROOT3, -5 + 20 * HALF_ROOT3],
    [0, 2.5 + 15 * HALF_ROOT3, -5 + 20 * HALF_ROOT3]
])

V_WINDS_EARTH_RELATIVE_M_S01 = numpy.array([
    [10, 7.5 - 5 * HALF_ROOT3, -10 - 10 * HALF_ROOT3],
    [10, 7.5 - 5 * HALF_ROOT3, -10 - 10 * HALF_ROOT3]
])


def _compare_interp_time_tables(first_interp_time_table,
                                second_interp_time_table):
    """Compares two tables with interpolation times.

    :param first_interp_time_table: pandas DataFrame created by
        `nwp_model_utils.get_times_needed_for_interp`.
    :param second_interp_time_table: Same.
    :return: are_tables_equal: Boolean flag.
    """

    first_column_names = list(first_interp_time_table)
    second_column_names = list(second_interp_time_table)
    if set(first_column_names) != set(second_column_names):
        return False

    first_num_rows = len(first_interp_time_table.index)
    second_num_rows = len(second_interp_time_table.index)
    if first_num_rows != second_num_rows:
        return False

    for i in range(first_num_rows):
        for this_column in first_column_names:
            if not numpy.array_equal(
                    first_interp_time_table[this_column].values[i],
                    second_interp_time_table[this_column].values[i]
            ):
                return False

    return True


class NwpModelUtilsTests(unittest.TestCase):
    """Each method is a unit test for nwp_model_utils.py."""

    def test_check_grid_name_narr221(self):
        """Ensures correct output from check_grid_name.

        In this case, model is NARR and grid is NCEP 221.
        """

        nwp_model_utils.check_grid_name(
            model_name=nwp_model_utils.NARR_MODEL_NAME,
            grid_name=nwp_model_utils.NAME_OF_221GRID)

    def test_check_grid_name_narr_extended221(self):
        """Ensures correct output from check_grid_name.

        In this case, model is NARR and grid is extended NCEP 221.
        """

        nwp_model_utils.check_grid_name(
            model_name=nwp_model_utils.NARR_MODEL_NAME,
            grid_name=nwp_model_utils.NAME_OF_EXTENDED_221GRID)

    def test_check_grid_name_narr130(self):
        """Ensures correct output from check_grid_name.

        In this case, model is NARR and grid is NCEP 130.
        """

        with self.assertRaises(ValueError):
            nwp_model_utils.check_grid_name(
                model_name=nwp_model_utils.NARR_MODEL_NAME,
                grid_name=nwp_model_utils.NAME_OF_130GRID)

    def test_check_grid_name_rap221(self):
        """Ensures correct output from check_grid_name.

        In this case, model is RAP and grid is NCEP 221.
        """

        with self.assertRaises(ValueError):
            nwp_model_utils.check_grid_name(
                model_name=nwp_model_utils.RAP_MODEL_NAME,
                grid_name=nwp_model_utils.NAME_OF_221GRID)

    def test_check_grid_name_rap_extended221(self):
        """Ensures correct output from check_grid_name.

        In this case, model is RAP and grid is extended NCEP 221.
        """

        with self.assertRaises(ValueError):
            nwp_model_utils.check_grid_name(
                model_name=nwp_model_utils.RAP_MODEL_NAME,
                grid_name=nwp_model_utils.NAME_OF_EXTENDED_221GRID)

    def test_check_grid_name_rap130(self):
        """Ensures correct output from check_grid_name.

        In this case, model is RAP and grid is NCEP 130.
        """

        nwp_model_utils.check_grid_name(
            model_name=nwp_model_utils.RAP_MODEL_NAME,
            grid_name=nwp_model_utils.NAME_OF_130GRID)

    def test_dimensions_to_grid_221(self):
        """Ensures correct output from dimensions_to_grid for NCEP 221 grid."""

        this_num_rows, this_num_columns = nwp_model_utils.get_grid_dimensions(
            model_name=nwp_model_utils.NARR_MODEL_NAME,
            grid_name=nwp_model_utils.NAME_OF_221GRID)

        this_grid_name = nwp_model_utils.dimensions_to_grid(
            num_rows=this_num_rows, num_columns=this_num_columns)

        self.assertTrue(this_grid_name == nwp_model_utils.NAME_OF_221GRID)

    def test_dimensions_to_grid_extended221(self):
        """Ensures correctness of dimensions_to_grid for extended 221 grid."""

        this_num_rows, this_num_columns = nwp_model_utils.get_grid_dimensions(
            model_name=nwp_model_utils.NARR_MODEL_NAME,
            grid_name=nwp_model_utils.NAME_OF_EXTENDED_221GRID)

        this_grid_name = nwp_model_utils.dimensions_to_grid(
            num_rows=this_num_rows, num_columns=this_num_columns)

        self.assertTrue(
            this_grid_name == nwp_model_utils.NAME_OF_EXTENDED_221GRID
        )

    def test_dimensions_to_grid_130(self):
        """Ensures correct output from dimensions_to_grid for NCEP 130 grid."""

        this_num_rows, this_num_columns = nwp_model_utils.get_grid_dimensions(
            model_name=nwp_model_utils.RAP_MODEL_NAME,
            grid_name=nwp_model_utils.NAME_OF_130GRID)

        this_grid_name = nwp_model_utils.dimensions_to_grid(
            num_rows=this_num_rows, num_columns=this_num_columns)

        self.assertTrue(this_grid_name == nwp_model_utils.NAME_OF_130GRID)

    def test_dimensions_to_grid_252(self):
        """Ensures correct output from dimensions_to_grid for NCEP 252 grid."""

        this_num_rows, this_num_columns = nwp_model_utils.get_grid_dimensions(
            model_name=nwp_model_utils.RAP_MODEL_NAME,
            grid_name=nwp_model_utils.NAME_OF_252GRID)

        this_grid_name = nwp_model_utils.dimensions_to_grid(
            num_rows=this_num_rows, num_columns=this_num_columns)

        self.assertTrue(this_grid_name == nwp_model_utils.NAME_OF_252GRID)

    def test_dimensions_to_grid_fake(self):
        """Ensures correct output from dimensions_to_grid for fake grid."""

        with self.assertRaises(ValueError):
            nwp_model_utils.dimensions_to_grid(num_rows=1, num_columns=2)

    def test_rotate_winds_to_earth_relative(self):
        """Ensures correct output from rotate_winds_to_earth_relative."""

        these_u_winds_m_s01, these_v_winds_m_s01 = (
            nwp_model_utils.rotate_winds_to_earth_relative(
                u_winds_grid_relative_m_s01=U_WINDS_GRID_RELATIVE_M_S01,
                v_winds_grid_relative_m_s01=V_WINDS_GRID_RELATIVE_M_S01,
                rotation_angle_cosines=ROTATION_ANGLE_COSINES,
                rotation_angle_sines=ROTATION_ANGLE_SINES)
        )

        self.assertTrue(numpy.allclose(
            these_u_winds_m_s01, U_WINDS_EARTH_RELATIVE_M_S01, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            these_v_winds_m_s01, V_WINDS_EARTH_RELATIVE_M_S01, atol=TOLERANCE
        ))

    def test_rotate_winds_to_grid_relative(self):
        """Ensures correct output from rotate_winds_to_grid_relative."""

        these_u_winds_m_s01, these_v_winds_m_s01 = (
            nwp_model_utils.rotate_winds_to_grid_relative(
                u_winds_earth_relative_m_s01=U_WINDS_EARTH_RELATIVE_M_S01,
                v_winds_earth_relative_m_s01=V_WINDS_EARTH_RELATIVE_M_S01,
                rotation_angle_cosines=ROTATION_ANGLE_COSINES,
                rotation_angle_sines=ROTATION_ANGLE_SINES)
        )

        self.assertTrue(numpy.allclose(
            these_u_winds_m_s01, U_WINDS_GRID_RELATIVE_M_S01, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            these_v_winds_m_s01, V_WINDS_GRID_RELATIVE_M_S01, atol=TOLERANCE
        ))

    def test_get_times_needed_for_interp_previous(self):
        """Ensures correct output from get_times_needed_for_interp.

        In this case, interpolation method is previous-neighbour.
        """

        these_model_times_unix_sec, this_interp_time_table = (
            nwp_model_utils.get_times_needed_for_interp(
                query_times_unix_sec=QUERY_TIMES_UNIX_SEC,
                model_time_step_hours=MODEL_TIME_STEP_HOURS,
                method_string=interp.PREV_NEIGHBOUR_METHOD_STRING)
        )

        self.assertTrue(numpy.array_equal(
            these_model_times_unix_sec, MODEL_TIMES_PREV_INTERP_UNIX_SEC
        ))

        self.assertTrue(_compare_interp_time_tables(
            this_interp_time_table, INTERP_TIME_TABLE_PREV_INTERP
        ))

    def test_get_times_needed_for_interp_next(self):
        """Ensures correct output from get_times_needed_for_interp.

        In this case, interpolation method is next-neighbour.
        """

        these_model_times_unix_sec, this_interp_time_table = (
            nwp_model_utils.get_times_needed_for_interp(
                query_times_unix_sec=QUERY_TIMES_UNIX_SEC,
                model_time_step_hours=MODEL_TIME_STEP_HOURS,
                method_string=interp.NEXT_NEIGHBOUR_METHOD_STRING)
        )

        self.assertTrue(numpy.array_equal(
            these_model_times_unix_sec, MODEL_TIMES_NEXT_INTERP_UNIX_SEC
        ))

        self.assertTrue(_compare_interp_time_tables(
            this_interp_time_table, INTERP_TIME_TABLE_NEXT_INTERP
        ))

    def test_get_times_needed_for_interp_nearest(self):
        """Ensures correct output from get_times_needed_for_interp.

        In this case, interpolation method is nearest-neighbour.
        """

        these_model_times_unix_sec, this_interp_time_table = (
            nwp_model_utils.get_times_needed_for_interp(
                query_times_unix_sec=QUERY_TIMES_UNIX_SEC,
                model_time_step_hours=MODEL_TIME_STEP_HOURS,
                method_string=interp.NEAREST_NEIGHBOUR_METHOD_STRING)
        )

        self.assertTrue(numpy.array_equal(
            these_model_times_unix_sec,
            MODEL_TIMES_SUB_AND_LINEAR_INTERP_UNIX_SEC
        ))

        self.assertTrue(_compare_interp_time_tables(
            this_interp_time_table, INTERP_TIME_TABLE_SUB_AND_LINEAR_INTERP
        ))

    def test_get_times_needed_for_interp_linear(self):
        """Ensures correct output from get_times_needed_for_interp.

        In this case, interpolation method is linear.
        """

        these_model_times_unix_sec, this_interp_time_table = (
            nwp_model_utils.get_times_needed_for_interp(
                query_times_unix_sec=QUERY_TIMES_UNIX_SEC,
                model_time_step_hours=MODEL_TIME_STEP_HOURS,
                method_string=interp.LINEAR_METHOD_STRING)
        )

        self.assertTrue(numpy.array_equal(
            these_model_times_unix_sec,
            MODEL_TIMES_SUB_AND_LINEAR_INTERP_UNIX_SEC
        ))

        self.assertTrue(_compare_interp_time_tables(
            this_interp_time_table, INTERP_TIME_TABLE_SUB_AND_LINEAR_INTERP
        ))

    def test_get_times_needed_for_interp_quadratic(self):
        """Ensures correct output from get_times_needed_for_interp.

        In this case, interpolation method is quadratic.
        """

        these_model_times_unix_sec, this_interp_time_table = (
            nwp_model_utils.get_times_needed_for_interp(
                query_times_unix_sec=QUERY_TIMES_UNIX_SEC,
                model_time_step_hours=MODEL_TIME_STEP_HOURS,
                method_string=interp.SPLINE2_METHOD_STRING)
        )

        self.assertTrue(numpy.array_equal(
            these_model_times_unix_sec,
            MODEL_TIMES_SUPERLINEAR_INTERP_UNIX_SEC
        ))

        self.assertTrue(_compare_interp_time_tables(
            this_interp_time_table, INTERP_TIME_TABLE_SUPERLINEAR_INTERP
        ))

    def test_get_times_needed_for_interp_cubic(self):
        """Ensures correct output from get_times_needed_for_interp.

        In this case, interpolation method is cubic.
        """

        these_model_times_unix_sec, this_interp_time_table = (
            nwp_model_utils.get_times_needed_for_interp(
                query_times_unix_sec=QUERY_TIMES_UNIX_SEC,
                model_time_step_hours=MODEL_TIME_STEP_HOURS,
                method_string=interp.SPLINE3_METHOD_STRING)
        )

        self.assertTrue(numpy.array_equal(
            these_model_times_unix_sec,
            MODEL_TIMES_SUPERLINEAR_INTERP_UNIX_SEC
        ))

        self.assertTrue(_compare_interp_time_tables(
            this_interp_time_table, INTERP_TIME_TABLE_SUPERLINEAR_INTERP
        ))

    def test_projection_grid130(self):
        """Ensures approx correctness of Lambert proj for NCEP 130 grid."""

        num_grid_rows, num_grid_columns = nwp_model_utils.get_grid_dimensions(
            model_name=nwp_model_utils.RAP_MODEL_NAME,
            grid_name=nwp_model_utils.NAME_OF_130GRID)

        unique_longitudes_deg, unique_latitudes_deg = numpy.loadtxt(
            GRID130_LATLNG_FILE_NAME, unpack=True)
        latitude_matrix_deg = numpy.reshape(
            unique_latitudes_deg, (num_grid_rows, num_grid_columns)
        )
        longitude_matrix_deg = numpy.reshape(
            unique_longitudes_deg, (num_grid_rows, num_grid_columns)
        )

        x_matrix_metres, y_matrix_metres = nwp_model_utils.project_latlng_to_xy(
            latitudes_deg=latitude_matrix_deg,
            longitudes_deg=longitude_matrix_deg,
            model_name=nwp_model_utils.RAP_MODEL_NAME,
            grid_name=nwp_model_utils.NAME_OF_130GRID)

        expected_x_matrix_metres, expected_y_matrix_metres = (
            nwp_model_utils.get_xy_grid_point_matrices(
                model_name=nwp_model_utils.RAP_MODEL_NAME,
                grid_name=nwp_model_utils.NAME_OF_130GRID)
        )

        x_error_matrix_metres = x_matrix_metres - expected_x_matrix_metres
        y_error_matrix_metres = y_matrix_metres - expected_y_matrix_metres
        distance_error_matrix_metres = numpy.sqrt(
            x_error_matrix_metres ** 2 + y_error_matrix_metres ** 2
        )

        self.assertTrue(
            numpy.mean(distance_error_matrix_metres) <=
            MAX_MEAN_DISTANCE_ERROR_RAP_METRES
        )

        self.assertTrue(
            numpy.max(distance_error_matrix_metres) <=
            MAX_MAX_DISTANCE_ERROR_RAP_METRES
        )

    def test_projection_grid252(self):
        """Ensures approx correctness of Lambert proj for NCEP 252 grid."""

        num_grid_rows, num_grid_columns = nwp_model_utils.get_grid_dimensions(
            model_name=nwp_model_utils.RAP_MODEL_NAME,
            grid_name=nwp_model_utils.NAME_OF_252GRID)

        unique_longitudes_deg, unique_latitudes_deg = numpy.loadtxt(
            GRID252_LATLNG_FILE_NAME, unpack=True)
        latitude_matrix_deg = numpy.reshape(
            unique_latitudes_deg, (num_grid_rows, num_grid_columns)
        )
        longitude_matrix_deg = numpy.reshape(
            unique_longitudes_deg, (num_grid_rows, num_grid_columns)
        )

        x_matrix_metres, y_matrix_metres = nwp_model_utils.project_latlng_to_xy(
            latitudes_deg=latitude_matrix_deg,
            longitudes_deg=longitude_matrix_deg,
            model_name=nwp_model_utils.RAP_MODEL_NAME,
            grid_name=nwp_model_utils.NAME_OF_252GRID)

        expected_x_matrix_metres, expected_y_matrix_metres = (
            nwp_model_utils.get_xy_grid_point_matrices(
                model_name=nwp_model_utils.RAP_MODEL_NAME,
                grid_name=nwp_model_utils.NAME_OF_252GRID)
        )

        x_error_matrix_metres = x_matrix_metres - expected_x_matrix_metres
        y_error_matrix_metres = y_matrix_metres - expected_y_matrix_metres
        distance_error_matrix_metres = numpy.sqrt(
            x_error_matrix_metres ** 2 + y_error_matrix_metres ** 2
        )

        self.assertTrue(
            numpy.mean(distance_error_matrix_metres) <=
            MAX_MEAN_DISTANCE_ERROR_RAP_METRES
        )

        self.assertTrue(
            numpy.max(distance_error_matrix_metres) <=
            MAX_MAX_DISTANCE_ERROR_RAP_METRES
        )

    def test_projection_grid221(self):
        """Ensures approx correctness of Lambert proj for NCEP 221 grid."""

        num_grid_rows, num_grid_columns = nwp_model_utils.get_grid_dimensions(
            model_name=nwp_model_utils.NARR_MODEL_NAME,
            grid_name=nwp_model_utils.NAME_OF_221GRID)

        unique_longitudes_deg, unique_latitudes_deg = numpy.loadtxt(
            NARR_LATLNG_FILE_NAME, unpack=True)
        latitude_matrix_deg = numpy.reshape(
            unique_latitudes_deg, (num_grid_rows, num_grid_columns)
        )
        longitude_matrix_deg = numpy.reshape(
            unique_longitudes_deg, (num_grid_rows, num_grid_columns)
        )

        x_matrix_metres, y_matrix_metres = nwp_model_utils.project_latlng_to_xy(
            latitudes_deg=latitude_matrix_deg,
            longitudes_deg=longitude_matrix_deg,
            model_name=nwp_model_utils.NARR_MODEL_NAME,
            grid_name=nwp_model_utils.NAME_OF_221GRID)

        expected_x_matrix_metres, expected_y_matrix_metres = (
            nwp_model_utils.get_xy_grid_point_matrices(
                model_name=nwp_model_utils.NARR_MODEL_NAME,
                grid_name=nwp_model_utils.NAME_OF_221GRID)
        )

        x_error_matrix_metres = x_matrix_metres - expected_x_matrix_metres
        y_error_matrix_metres = y_matrix_metres - expected_y_matrix_metres
        distance_error_matrix_metres = numpy.sqrt(
            x_error_matrix_metres ** 2 + y_error_matrix_metres ** 2
        )

        self.assertTrue(
            numpy.mean(distance_error_matrix_metres) <=
            MAX_MEAN_DISTANCE_ERROR_NARR_METRES
        )

        self.assertTrue(
            numpy.max(distance_error_matrix_metres) <=
            MAX_MAX_DISTANCE_ERROR_NARR_METRES
        )

    def test_projection_extended221(self):
        """Ensures approx correctness of Lambert proj for extended 221 grid."""

        num_grid_rows, num_grid_columns = nwp_model_utils.get_grid_dimensions(
            model_name=nwp_model_utils.NARR_MODEL_NAME,
            grid_name=nwp_model_utils.NAME_OF_221GRID)

        unique_longitudes_deg, unique_latitudes_deg = numpy.loadtxt(
            NARR_LATLNG_FILE_NAME, unpack=True)
        latitude_matrix_deg = numpy.reshape(
            unique_latitudes_deg, (num_grid_rows, num_grid_columns)
        )
        longitude_matrix_deg = numpy.reshape(
            unique_longitudes_deg, (num_grid_rows, num_grid_columns)
        )

        x_matrix_metres, y_matrix_metres = nwp_model_utils.project_latlng_to_xy(
            latitudes_deg=latitude_matrix_deg,
            longitudes_deg=longitude_matrix_deg,
            model_name=nwp_model_utils.NARR_MODEL_NAME,
            grid_name=nwp_model_utils.NAME_OF_221GRID)

        expected_x_matrix_metres, expected_y_matrix_metres = (
            nwp_model_utils.get_xy_grid_point_matrices(
                model_name=nwp_model_utils.NARR_MODEL_NAME,
                grid_name=nwp_model_utils.NAME_OF_EXTENDED_221GRID)
        )

        expected_x_matrix_metres = expected_x_matrix_metres[100:-100, 100:-100]
        expected_y_matrix_metres = expected_y_matrix_metres[100:-100, 100:-100]
        expected_x_matrix_metres -= expected_x_matrix_metres[0, 0]
        expected_y_matrix_metres -= expected_y_matrix_metres[0, 0]

        x_error_matrix_metres = x_matrix_metres - expected_x_matrix_metres
        y_error_matrix_metres = y_matrix_metres - expected_y_matrix_metres
        distance_error_matrix_metres = numpy.sqrt(
            x_error_matrix_metres ** 2 + y_error_matrix_metres ** 2
        )

        self.assertTrue(
            numpy.mean(distance_error_matrix_metres) <=
            MAX_MEAN_DISTANCE_ERROR_NARR_METRES
        )

        self.assertTrue(
            numpy.max(distance_error_matrix_metres) <=
            MAX_MAX_DISTANCE_ERROR_NARR_METRES
        )

    def test_wind_rotation_angles_grid130(self):
        """Ensures approx correctness of rotation angles for NCEP 130 grid."""

        num_grid_rows, num_grid_columns = nwp_model_utils.get_grid_dimensions(
            model_name=nwp_model_utils.RAP_MODEL_NAME,
            grid_name=nwp_model_utils.NAME_OF_130GRID)

        expected_cos_vector, expected_sin_vector = numpy.loadtxt(
            GRID130_WIND_ROTATION_FILE_NAME, unpack=True)
        expected_cos_matrix = numpy.reshape(
            expected_cos_vector, (num_grid_rows, num_grid_columns)
        )
        expected_sin_matrix = numpy.reshape(
            expected_sin_vector, (num_grid_rows, num_grid_columns)
        )

        latitude_matrix_deg, longitude_matrix_deg = (
            nwp_model_utils.get_latlng_grid_point_matrices(
                model_name=nwp_model_utils.RAP_MODEL_NAME,
                grid_name=nwp_model_utils.NAME_OF_130GRID)
        )

        rotation_angle_cos_matrix, rotation_angle_sin_matrix = (
            nwp_model_utils.get_wind_rotation_angles(
                latitudes_deg=latitude_matrix_deg,
                longitudes_deg=longitude_matrix_deg,
                model_name=nwp_model_utils.RAP_MODEL_NAME)
        )

        cos_error_matrix = numpy.absolute(
            rotation_angle_cos_matrix - expected_cos_matrix
        )
        sin_error_matrix = numpy.absolute(
            rotation_angle_sin_matrix - expected_sin_matrix
        )

        self.assertTrue(
            numpy.mean(cos_error_matrix) <= MAX_MEAN_SIN_OR_COS_ERROR
        )
        self.assertTrue(numpy.max(cos_error_matrix) <= MAX_MAX_SIN_OR_COS_ERROR)

        self.assertTrue(
            numpy.mean(sin_error_matrix) <= MAX_MEAN_SIN_OR_COS_ERROR
        )
        self.assertTrue(numpy.max(sin_error_matrix) <= MAX_MAX_SIN_OR_COS_ERROR)

    def test_wind_rotation_angles_grid252(self):
        """Ensures approx correctness of rotation angles for NCEP 252 grid."""

        num_grid_rows, num_grid_columns = nwp_model_utils.get_grid_dimensions(
            model_name=nwp_model_utils.RAP_MODEL_NAME,
            grid_name=nwp_model_utils.NAME_OF_252GRID)

        expected_cos_vector, expected_sin_vector = numpy.loadtxt(
            GRID252_WIND_ROTATION_FILE_NAME, unpack=True)
        expected_cos_matrix = numpy.reshape(
            expected_cos_vector, (num_grid_rows, num_grid_columns)
        )
        expected_sin_matrix = numpy.reshape(
            expected_sin_vector, (num_grid_rows, num_grid_columns)
        )

        latitude_matrix_deg, longitude_matrix_deg = (
            nwp_model_utils.get_latlng_grid_point_matrices(
                model_name=nwp_model_utils.RAP_MODEL_NAME,
                grid_name=nwp_model_utils.NAME_OF_252GRID)
        )

        rotation_angle_cos_matrix, rotation_angle_sin_matrix = (
            nwp_model_utils.get_wind_rotation_angles(
                latitudes_deg=latitude_matrix_deg,
                longitudes_deg=longitude_matrix_deg,
                model_name=nwp_model_utils.RAP_MODEL_NAME)
        )

        cos_error_matrix = numpy.absolute(
            rotation_angle_cos_matrix - expected_cos_matrix
        )
        sin_error_matrix = numpy.absolute(
            rotation_angle_sin_matrix - expected_sin_matrix
        )

        self.assertTrue(
            numpy.mean(cos_error_matrix) <= MAX_MEAN_SIN_OR_COS_ERROR
        )
        self.assertTrue(numpy.max(cos_error_matrix) <= MAX_MAX_SIN_OR_COS_ERROR)

        self.assertTrue(
            numpy.mean(sin_error_matrix) <= MAX_MEAN_SIN_OR_COS_ERROR
        )
        self.assertTrue(numpy.max(sin_error_matrix) <= MAX_MAX_SIN_OR_COS_ERROR)


if __name__ == '__main__':
    unittest.main()
