"""Unit tests for interp.py."""

import copy
import unittest
import numpy
from gewittergefahr.gg_utils import interp
from gewittergefahr.gg_utils import nwp_model_utils

TOLERANCE = 1e-6

# The following constants are used to test _find_nearest_value.
SORTED_ARRAY = numpy.array([-5., -3., -1., 0., 1.5, 4., 9.])
TEST_VALUES = numpy.array([-6., -5., -4., -3., 5., 8., 9., 10.])
NEAREST_INDICES_FOR_TEST_VALUES = numpy.array([0., 0., 1., 1., 5., 6., 6., 6.])

# The following constants are used to test _get_wind_rotation_metadata.
FIELD_NAMES_GRIB1 = [
    'HGT:500 mb', 'UGRD:500 mb', 'VGRD:500 mb', 'HGT:700 mb', 'UGRD:700 mb',
    'HGT:850 mb', 'VGRD:850 mb', 'SPFH:2 m above gnd', 'UGRD:10 m above gnd',
    'VGRD:10 m above gnd'
]

ROTATE_WIND_FLAGS_FOR_NARR = numpy.full(
    len(FIELD_NAMES_GRIB1), False, dtype=bool)
ROTATE_WIND_FLAGS_FOR_RAPRUC = numpy.array(
    [0, 1, 1, 0, 1, 0, 1, 0, 1, 1], dtype=bool)

FIELD_NAMES_OTHER_COMPONENT_NARR = [''] * len(FIELD_NAMES_GRIB1)
FIELD_NAMES_OTHER_COMPONENT_RAPRUC = [
    '', 'VGRD:500 mb', 'UGRD:500 mb', '', 'VGRD:700 mb', '', 'UGRD:850 mb', '',
    'VGRD:10 m above gnd', 'UGRD:10 m above gnd'
]

OTHER_WIND_COMPONENT_INDICES_NARR = numpy.full(
    len(FIELD_NAMES_GRIB1), -1, dtype=int)
OTHER_WIND_COMPONENT_INDICES_RAPRUC = numpy.array(
    [-1, 2, 1, -1, -1, -1, -1, -1, 9, 8], dtype=int)

WIND_ROTATION_METADATA_DICT_NARR = {
    interp.FIELD_NAMES_GRIB1_KEY: FIELD_NAMES_GRIB1,
    interp.ROTATE_WIND_FLAGS_KEY: ROTATE_WIND_FLAGS_FOR_NARR,
    interp.FIELD_NAMES_OTHER_COMPONENT_KEY: FIELD_NAMES_OTHER_COMPONENT_NARR,
    interp.OTHER_WIND_COMPONENT_INDICES_KEY: OTHER_WIND_COMPONENT_INDICES_NARR
}

WIND_ROTATION_METADATA_DICT_RAPRUC = {
    interp.FIELD_NAMES_GRIB1_KEY: FIELD_NAMES_GRIB1,
    interp.ROTATE_WIND_FLAGS_KEY: ROTATE_WIND_FLAGS_FOR_RAPRUC,
    interp.FIELD_NAMES_OTHER_COMPONENT_KEY: FIELD_NAMES_OTHER_COMPONENT_RAPRUC,
    interp.OTHER_WIND_COMPONENT_INDICES_KEY: OTHER_WIND_COMPONENT_INDICES_RAPRUC
}


# The following constants are used to test _stack_1d_arrays_horizontally.
LIST_OF_1D_ARRAYS = [numpy.array([1., 2., 3]),
                     numpy.array([0., 5., 10.]),
                     numpy.array([6., 6., 6.])]
MATRIX_FIRST_ARRAY = numpy.reshape(LIST_OF_1D_ARRAYS[0], (3, 1))
MATRIX_FIRST_2ARRAYS = numpy.transpose(
    numpy.array([[1., 2., 3.], [0., 5., 10.]]))
MATRIX_FIRST_3ARRAYS = numpy.transpose(
    numpy.array([[1., 2., 3.], [0., 5., 10.], [6., 6., 6.]]))

# The following constants are used to test _find_heights_with_temperature.
TARGET_TEMPERATURE_KELVINS = 10.
WARM_TEMPERATURES_KELVINS = numpy.array(
    [10., 10., 10., 11., 11., 11., 12., 12., 12., numpy.nan, numpy.nan, 12.])
COLD_TEMPERATURES_KELVINS = numpy.array(
    [8., 9., 10., 8., 9., 10., 8., 9., 10., numpy.nan, 8., numpy.nan])
WARM_HEIGHTS_M_ASL = numpy.full(9, 2000.)
COLD_HEIGHTS_M_ASL = numpy.full(9, 2500.)

TARGET_HEIGHTS_M_ASL = numpy.array(
    [2000., 2000., numpy.nan, 2166.666667, 2250., 2500., 2250., 2333.333333,
     2500., numpy.nan, numpy.nan, numpy.nan])

# The following constants are used to test interp_in_time.
INPUT_MATRIX_TIME0 = numpy.array([[0., 2., 5., 10.],
                                  [-2., 1., 3., 6.],
                                  [-3.5, -2.5, 3., 8.]])
INPUT_MATRIX_TIME4 = numpy.array([[2., 5., 7., 15.],
                                  [0., 2., 5., 8.],
                                  [-1.5, -2.5, 0., 4.]])

INPUT_TIMES_UNIX_SEC = numpy.array([0, 4])
QUERY_TIMES_FOR_LINEAR_UNIX_SEC = numpy.array([1, 2, 3])
QUERY_TIMES_FOR_EXTRAP_UNIX_SEC = numpy.array([8])
INPUT_MATRIX_FOR_TEMPORAL_INTERP = numpy.stack(
    (INPUT_MATRIX_TIME0, INPUT_MATRIX_TIME4), axis=-1)

THIS_INTERP_MATRIX_TIME1 = numpy.array([[0.5, 2.75, 5.5, 11.25],
                                        [-1.5, 1.25, 3.5, 6.5],
                                        [-3., -2.5, 2.25, 7.]])
THIS_INTERP_MATRIX_TIME2 = numpy.array([[1., 3.5, 6., 12.5],
                                        [-1., 1.5, 4., 7.],
                                        [-2.5, -2.5, 1.5, 6.]])
THIS_INTERP_MATRIX_TIME3 = numpy.array([[1.5, 4.25, 6.5, 13.75],
                                        [-0.5, 1.75, 4.5, 7.5],
                                        [-2., -2.5, 0.75, 5.]])
THIS_INTERP_MATRIX_TIME8 = numpy.array([[4., 8., 9., 20.],
                                        [2., 3., 7., 10.],
                                        [0.5, -2.5, -3., 0.]])

INTERP_MATRIX_LINEAR_IN_TIME = numpy.stack(
    (THIS_INTERP_MATRIX_TIME1, THIS_INTERP_MATRIX_TIME2,
     THIS_INTERP_MATRIX_TIME3), axis=-1)
TIME_EXTRAP_MATRIX = numpy.expand_dims(THIS_INTERP_MATRIX_TIME8, axis=-1)

QUERY_TIMES_FOR_PREV_NEIGH_UNIX_SEC = numpy.array([1, 2, 3, 8])
THIS_INTERP_MATRIX_TIME1 = copy.deepcopy(INPUT_MATRIX_TIME0)
THIS_INTERP_MATRIX_TIME2 = copy.deepcopy(INPUT_MATRIX_TIME0)
THIS_INTERP_MATRIX_TIME3 = copy.deepcopy(INPUT_MATRIX_TIME0)
THIS_INTERP_MATRIX_TIME8 = copy.deepcopy(INPUT_MATRIX_TIME4)
INTERP_MATRIX_PREVIOUS_TIME = numpy.stack(
    (THIS_INTERP_MATRIX_TIME1, THIS_INTERP_MATRIX_TIME2,
     THIS_INTERP_MATRIX_TIME3, THIS_INTERP_MATRIX_TIME8), axis=-1)

QUERY_TIMES_FOR_NEXT_NEIGH_UNIX_SEC = numpy.array([1, 2, 3])
THIS_INTERP_MATRIX_TIME1 = copy.deepcopy(INPUT_MATRIX_TIME4)
THIS_INTERP_MATRIX_TIME2 = copy.deepcopy(INPUT_MATRIX_TIME4)
THIS_INTERP_MATRIX_TIME3 = copy.deepcopy(INPUT_MATRIX_TIME4)
INTERP_MATRIX_NEXT_TIME = numpy.stack(
    (THIS_INTERP_MATRIX_TIME1, THIS_INTERP_MATRIX_TIME2,
     THIS_INTERP_MATRIX_TIME3), axis=-1)

# The following constants are used to test interp_from_xy_grid_to_points.
INPUT_MATRIX_FOR_SPATIAL_INTERP = numpy.array([[17., 24., 1., 8.],
                                               [23., 5., 7., 14.],
                                               [4., 6., 13., 20.],
                                               [10., 12., 19., 21.],
                                               [11., 18., 25., 2.]])

SPLINE_DEGREE = 1
GRID_POINT_X_METRES = numpy.array([0., 1., 2., 3.])
GRID_POINT_Y_METRES = numpy.array([0., 2., 4., 6., 8.])
QUERY_X_FOR_SPLINE_METRES = numpy.array([0., 0.5, 1., 1.5, 2., 2.5, 3.])
QUERY_Y_FOR_SPLINE_METRES = numpy.array([0., 2., 2.5, 3., 5., 6., 7.5])
INTERP_VALUES_SPLINE = numpy.array([17., 14., 5.25, 7.75, 16., 20., 6.75])

QUERY_X_FOR_NEAREST_NEIGH_METRES = numpy.array(
    [0., 0.3, 0.7, 1.2, 1.8, 2.1, 2.9])
QUERY_Y_FOR_NEAREST_NEIGH_METRES = numpy.array(
    [0.5, 1.5, 2.5, 4., 5.5, 6.5, 7.7])
INTERP_VALUES_NEAREST_NEIGH = numpy.array([17., 23., 5., 6., 19., 19., 2])

QUERY_X_FOR_EXTRAP_METRES = numpy.array([-1., 4.])
QUERY_Y_FOR_EXTRAP_METRES = numpy.array([-2., 10.])
SPATIAL_EXTRAP_VALUES = numpy.array([17., 2.])


def _compare_metadata_dicts(first_metadata_dict, second_metadata_dict):
    """Compares two dicts created by `interp._get_wind_rotation_metadata`.

    :param first_metadata_dict: First dictionary.
    :param second_metadata_dict: Second dictionary.
    :return: are_dicts_equal: Boolean flag.
    """

    first_keys = first_metadata_dict.keys()
    second_keys = second_metadata_dict.keys()
    if set(first_keys) != set(second_keys):
        return False

    for this_key in first_keys:
        if this_key in [interp.FIELD_NAMES_GRIB1_KEY,
                        interp.FIELD_NAMES_OTHER_COMPONENT_KEY]:
            if first_metadata_dict[this_key] != second_metadata_dict[this_key]:
                return False
        else:
            if not numpy.array_equal(first_metadata_dict[this_key],
                                     second_metadata_dict[this_key]):
                return False

    return True


class InterpTests(unittest.TestCase):
    """Each method is a unit test for interp.py."""

    def test_find_nearest_value(self):
        """Ensures correct output from _find_nearest_value."""

        these_nearest_indices = numpy.full(len(TEST_VALUES), -1)
        for i in range(len(TEST_VALUES)):
            _, these_nearest_indices[i] = interp._find_nearest_value(
                sorted_input_values=SORTED_ARRAY, test_value=TEST_VALUES[i])

        self.assertTrue(numpy.array_equal(
            these_nearest_indices, NEAREST_INDICES_FOR_TEST_VALUES))

    def test_get_wind_rotation_metadata_narr(self):
        """Ensures correct output from _get_wind_rotation_metadata.

        In this case, the NWP model is the NARR.
        """

        this_metadata_dict = interp._get_wind_rotation_metadata(
            field_names_grib1=FIELD_NAMES_GRIB1,
            model_name=nwp_model_utils.NARR_MODEL_NAME)

        self.assertTrue(_compare_metadata_dicts(
            this_metadata_dict, WIND_ROTATION_METADATA_DICT_NARR))

    def test_get_wind_rotation_metadata_ruc(self):
        """Ensures correct output from _get_wind_rotation_metadata.

        In this case, the NWP model is the RUC.
        """

        this_metadata_dict = interp._get_wind_rotation_metadata(
            field_names_grib1=FIELD_NAMES_GRIB1,
            model_name=nwp_model_utils.RUC_MODEL_NAME)

        self.assertTrue(_compare_metadata_dicts(
            this_metadata_dict, WIND_ROTATION_METADATA_DICT_RAPRUC))

    def test_get_wind_rotation_metadata_rap(self):
        """Ensures correct output from _get_wind_rotation_metadata.

        In this case, the NWP model is the RAP.
        """

        this_metadata_dict = interp._get_wind_rotation_metadata(
            field_names_grib1=FIELD_NAMES_GRIB1,
            model_name=nwp_model_utils.RUC_MODEL_NAME)

        self.assertTrue(_compare_metadata_dicts(
            this_metadata_dict, WIND_ROTATION_METADATA_DICT_RAPRUC))

    def test_stack_1d_arrays_horizontally_1array(self):
        """Ensures correct output from _stack_1d_arrays_horizontally.

        In this case there is one input array.
        """

        these_indices = numpy.array([0], dtype=int)
        this_matrix = interp._stack_1d_arrays_horizontally(
            [LIST_OF_1D_ARRAYS[i] for i in these_indices])

        self.assertTrue(numpy.allclose(
            this_matrix, MATRIX_FIRST_ARRAY, atol=TOLERANCE))

    def test_stack_1d_arrays_horizontally_2arrays(self):
        """Ensures correct output from _stack_1d_arrays_horizontally.

        In this case there are 2 input arrays.
        """

        these_indices = numpy.array([0, 1], dtype=int)
        this_matrix = interp._stack_1d_arrays_horizontally(
            [LIST_OF_1D_ARRAYS[i] for i in these_indices])

        self.assertTrue(numpy.allclose(
            this_matrix, MATRIX_FIRST_2ARRAYS, atol=TOLERANCE))

    def test_stack_1d_arrays_horizontally_3arrays(self):
        """Ensures correct output from _stack_1d_arrays_horizontally.

        In this case there are 3 input arrays.
        """

        these_indices = numpy.array([0, 1, 2], dtype=int)
        this_matrix = interp._stack_1d_arrays_horizontally(
            [LIST_OF_1D_ARRAYS[i] for i in these_indices])

        self.assertTrue(numpy.allclose(
            this_matrix, MATRIX_FIRST_3ARRAYS, atol=TOLERANCE))

    def test_find_heights_with_temperature(self):
        """Ensures correct output from _find_heights_with_temperature."""

        these_heights_m_asl = interp._find_heights_with_temperature(
            warm_temperatures_kelvins=WARM_TEMPERATURES_KELVINS,
            cold_temperatures_kelvins=COLD_TEMPERATURES_KELVINS,
            warm_heights_m_asl=WARM_HEIGHTS_M_ASL,
            cold_heights_m_asl=COLD_HEIGHTS_M_ASL,
            target_temperature_kelvins=TARGET_TEMPERATURE_KELVINS)

        self.assertTrue(numpy.allclose(
            these_heights_m_asl, TARGET_HEIGHTS_M_ASL, atol=TOLERANCE,
            equal_nan=True))

    def test_check_temporal_interp_method_valid(self):
        """Ensures correct output from check_temporal_interp_method.

        In this case, input is a valid temporal-interp method.
        """

        interp.check_temporal_interp_method(
            interp.NEAREST_NEIGHBOUR_METHOD_STRING)

    def test_check_temporal_interp_method_invalid(self):
        """Ensures correct output from check_temporal_interp_method.

        In this case, input is *not* a valid temporal-interp method.
        """

        with self.assertRaises(ValueError):
            interp.check_temporal_interp_method(interp.SPLINE_METHOD_STRING)

    def test_check_spatial_interp_method_valid(self):
        """Ensures correct output from check_spatial_interp_method.

        In this case, input is a valid spatial-interp method.
        """

        interp.check_spatial_interp_method(interp.SPLINE_METHOD_STRING)

    def test_check_spatial_interp_method_invalid(self):
        """Ensures correct output from check_spatial_interp_method.

        In this case, input is *not* a valid spatial-interp method.
        """

        with self.assertRaises(ValueError):
            interp.check_spatial_interp_method(interp.SPLINE0_METHOD_STRING)

    def test_interp_in_time_linear(self):
        """Ensures correct output from interp_in_time.

        In this case, interpolation method is linear.
        """

        this_interp_matrix = interp.interp_in_time(
            input_matrix=INPUT_MATRIX_FOR_TEMPORAL_INTERP,
            sorted_input_times_unix_sec=INPUT_TIMES_UNIX_SEC,
            query_times_unix_sec=QUERY_TIMES_FOR_LINEAR_UNIX_SEC,
            method_string=interp.LINEAR_METHOD_STRING, extrapolate=False)

        self.assertTrue(numpy.allclose(
            this_interp_matrix, INTERP_MATRIX_LINEAR_IN_TIME, atol=TOLERANCE))

    def test_interp_in_time_extrap(self):
        """Ensures correct output from interp_in_time.

        In this case, interp_in_time will do only extrapolation.
        """

        this_interp_matrix = interp.interp_in_time(
            input_matrix=INPUT_MATRIX_FOR_TEMPORAL_INTERP,
            sorted_input_times_unix_sec=INPUT_TIMES_UNIX_SEC,
            query_times_unix_sec=QUERY_TIMES_FOR_EXTRAP_UNIX_SEC,
            method_string=interp.LINEAR_METHOD_STRING, extrapolate=True)

        self.assertTrue(numpy.allclose(
            this_interp_matrix, TIME_EXTRAP_MATRIX, atol=TOLERANCE))

    def test_interp_in_time_previous(self):
        """Ensures correct output from interp_in_time.

        In this case, interpolation method is previous-neighbour.
        """

        this_interp_matrix = interp.interp_in_time(
            input_matrix=INPUT_MATRIX_FOR_TEMPORAL_INTERP,
            sorted_input_times_unix_sec=INPUT_TIMES_UNIX_SEC,
            query_times_unix_sec=QUERY_TIMES_FOR_PREV_NEIGH_UNIX_SEC,
            method_string=interp.PREV_NEIGHBOUR_METHOD_STRING,
            extrapolate=False)

        self.assertTrue(numpy.allclose(
            this_interp_matrix, INTERP_MATRIX_PREVIOUS_TIME, atol=TOLERANCE))

    def test_interp_in_time_next(self):
        """Ensures correct output from interp_in_time.

        In this case, interpolation method is next-neighbour.
        """

        this_interp_matrix = interp.interp_in_time(
            input_matrix=INPUT_MATRIX_FOR_TEMPORAL_INTERP,
            sorted_input_times_unix_sec=INPUT_TIMES_UNIX_SEC,
            query_times_unix_sec=QUERY_TIMES_FOR_NEXT_NEIGH_UNIX_SEC,
            method_string=interp.NEXT_NEIGHBOUR_METHOD_STRING,
            extrapolate=False)

        self.assertTrue(numpy.allclose(
            this_interp_matrix, INTERP_MATRIX_NEXT_TIME, atol=TOLERANCE))

    def test_interp_from_xy_grid_to_points_spline(self):
        """Ensures correct output from interp_from_xy_grid_to_points.

        In this case, interpolation method is linear spline.
        """

        these_interp_values = interp.interp_from_xy_grid_to_points(
            input_matrix=INPUT_MATRIX_FOR_SPATIAL_INTERP,
            sorted_grid_point_x_metres=GRID_POINT_X_METRES,
            sorted_grid_point_y_metres=GRID_POINT_Y_METRES,
            query_x_coords_metres=QUERY_X_FOR_SPLINE_METRES,
            query_y_coords_metres=QUERY_Y_FOR_SPLINE_METRES,
            method_string=interp.SPLINE_METHOD_STRING,
            spline_degree=SPLINE_DEGREE, extrapolate=False)

        self.assertTrue(numpy.allclose(
            these_interp_values, INTERP_VALUES_SPLINE, atol=TOLERANCE))

    def test_interp_from_xy_grid_to_points_spline_extrap(self):
        """Ensures correct output from interp_from_xy_grid_to_points.

        In this case, interpolation method is linear spline and extrapolation is
        allowed.
        """

        these_interp_values = interp.interp_from_xy_grid_to_points(
            input_matrix=INPUT_MATRIX_FOR_SPATIAL_INTERP,
            sorted_grid_point_x_metres=GRID_POINT_X_METRES,
            sorted_grid_point_y_metres=GRID_POINT_Y_METRES,
            query_x_coords_metres=QUERY_X_FOR_EXTRAP_METRES,
            query_y_coords_metres=QUERY_Y_FOR_EXTRAP_METRES,
            method_string=interp.SPLINE_METHOD_STRING,
            spline_degree=SPLINE_DEGREE, extrapolate=True)

        self.assertTrue(numpy.allclose(
            these_interp_values, SPATIAL_EXTRAP_VALUES, atol=TOLERANCE))

    def test_interp_from_xy_grid_to_points_nearest(self):
        """Ensures correct output from interp_from_xy_grid_to_points.

        In this case, interpolation method is nearest-neighbour.
        """

        these_interp_values = interp.interp_from_xy_grid_to_points(
            input_matrix=INPUT_MATRIX_FOR_SPATIAL_INTERP,
            sorted_grid_point_x_metres=GRID_POINT_X_METRES,
            sorted_grid_point_y_metres=GRID_POINT_Y_METRES,
            query_x_coords_metres=QUERY_X_FOR_NEAREST_NEIGH_METRES,
            query_y_coords_metres=QUERY_Y_FOR_NEAREST_NEIGH_METRES,
            method_string=interp.NEAREST_NEIGHBOUR_METHOD_STRING,
            extrapolate=False)

        self.assertTrue(numpy.allclose(
            these_interp_values, INTERP_VALUES_NEAREST_NEIGH, atol=TOLERANCE))

    def test_interp_from_xy_grid_to_points_nearest_extrap(self):
        """Ensures correct output from interp_from_xy_grid_to_points.

        In this case, interpolation method is nearest-neighbour and
        extrapolation is allowed.
        """

        these_interp_values = interp.interp_from_xy_grid_to_points(
            input_matrix=INPUT_MATRIX_FOR_SPATIAL_INTERP,
            sorted_grid_point_x_metres=GRID_POINT_X_METRES,
            sorted_grid_point_y_metres=GRID_POINT_Y_METRES,
            query_x_coords_metres=QUERY_X_FOR_EXTRAP_METRES,
            query_y_coords_metres=QUERY_Y_FOR_EXTRAP_METRES,
            method_string=interp.NEAREST_NEIGHBOUR_METHOD_STRING,
            extrapolate=True)

        self.assertTrue(numpy.allclose(
            these_interp_values, SPATIAL_EXTRAP_VALUES, atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
