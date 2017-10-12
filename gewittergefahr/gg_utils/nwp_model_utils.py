"""Processing methods for NWP (numerical weather prediction) data."""

import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import number_rounding as rounder
from gewittergefahr.gg_utils import error_checking

TIME_FORMAT = '%Y%m%d-%H%M%S'

MAIN_TEMPERATURE_COLUMN = 'temperature_kelvins'
MAIN_RH_COLUMN = 'relative_humidity'
MAIN_SPFH_COLUMN = 'specific_humidity'
MAIN_GPH_COLUMN = 'geopotential_height_metres'
MAIN_U_WIND_COLUMN = 'u_wind_m_s01'
MAIN_V_WIND_COLUMN = 'v_wind_m_s01'

LINEAR_AND_SUBLINEAR_INTERP_METHODS = ['linear', 'nearest', 'zero', 'slinear']
SUPERLINEAR_INTERP_METHODS = ['quadratic', 'cubic']
TEMPORAL_INTERP_METHODS = (
    LINEAR_AND_SUBLINEAR_INTERP_METHODS + SUPERLINEAR_INTERP_METHODS)


def get_times_needed_for_interp(min_query_time_unix_sec=None,
                                model_time_step_sec=None, method_string=None):
    """Finds model times needed for interpolation to a range of query times.

    The range of query times will be [min_query_time_unix_sec,
    min_query_time_unix_sec + model_time_step_sec].

    :param min_query_time_unix_sec: Minimum query time (Unix format).
    :param model_time_step_sec: Model time step.  If interpolating between
        forecast times (from the same initialization), this should be the
        model's time resolution (seconds between successive forecasts).  If
        interpolating between model runs (forecasts for the same valid time but
        from different initializations), this should be the model's refresh time
        (seconds between successive model runs).
    :param method_string: Interpolation method.  Valid options are listed in
        `TEMPORAL_INTERP_METHODS` and described in the documentation for
        `scipy.interpolate.interp1d`.
    :return: model_times_unix_sec: 1-D numpy array of model times needed for
        interpolation.
    :raises: ValueError: if min_query_time_unix_sec is not a multiple of
        model_time_step_sec.
    :raises: ValueError: if method_string not in TEMPORAL_INTERP_METHODS.
    """

    error_checking.assert_is_integer(min_query_time_unix_sec)
    error_checking.assert_is_integer(model_time_step_sec)
    error_checking.assert_is_string(method_string)

    rounded_min_query_time_unix_sec = rounder.round_to_nearest(
        min_query_time_unix_sec, model_time_step_sec)
    if min_query_time_unix_sec != rounded_min_query_time_unix_sec:
        min_query_time_string = time_conversion.unix_sec_to_string(
            min_query_time_unix_sec, TIME_FORMAT)

        error_string = (
            'Minimum query time (' + min_query_time_string +
            ') is not a multiple of model time step (' +
            str(model_time_step_sec) + ' seconds).')
        raise ValueError(error_string)

    if method_string not in TEMPORAL_INTERP_METHODS:
        error_string = (
            '\n\n' + str(TEMPORAL_INTERP_METHODS) +
            '\n\nValid temporal-interp methods (listed above) do not include "'
            + method_string + '".')
        raise ValueError(error_string)

    max_query_time_unix_sec = min_query_time_unix_sec + model_time_step_sec

    if method_string in LINEAR_AND_SUBLINEAR_INTERP_METHODS:
        model_times_unix_sec = numpy.array(
            [min_query_time_unix_sec, max_query_time_unix_sec], dtype=int)
    else:
        model_times_unix_sec = numpy.linspace(
            min_query_time_unix_sec - model_time_step_sec,
            max_query_time_unix_sec + model_time_step_sec, num=4, dtype=int)

    return model_times_unix_sec


def rotate_winds(u_winds_grid_relative_m_s01=None,
                 v_winds_grid_relative_m_s01=None, rotation_angle_cosines=None,
                 rotation_angle_sines=None):
    """Rotates wind vectors from grid-relative to Earth-relative.

    The equation is as follows, where alpha is the rotation angle.

    u_Earth = u_grid * cos(alpha) + v_grid * sin(alpha)
    v_Earth = v_grid * cos(alpha) - u_grid * sin(alpha)

    M = number of rows (unique grid-point latitudes)
    N = number of columns (unique grid-point longitudes)

    :param u_winds_grid_relative_m_s01: M-by-N numpy array of grid-relative
        u-winds (towards positive x-direction).
    :param v_winds_grid_relative_m_s01: M-by-N numpy array of grid-relative
        v-winds (towards positive y-direction).
    :param rotation_angle_cosines: M-by-N numpy array with cosines of wind-
        rotation angles.
    :param rotation_angle_sines: M-by-N numpy array with sines of wind-
        rotation angles.
    :return: u_winds_earth_relative_m_s01: M-by-N numpy array of Earth-relative
        (eastward) u-winds.
    :return: v_winds_earth_relative_m_s01: M-by-N numpy array of Earth-relative
        (northward) v-winds.
    """

    error_checking.assert_is_real_numpy_array(u_winds_grid_relative_m_s01)
    error_checking.assert_is_numpy_array(u_winds_grid_relative_m_s01,
                                         num_dimensions=2)
    num_grid_rows = u_winds_grid_relative_m_s01.shape[0]
    num_grid_columns = u_winds_grid_relative_m_s01.shape[1]

    error_checking.assert_is_real_numpy_array(v_winds_grid_relative_m_s01)
    error_checking.assert_is_numpy_array(
        v_winds_grid_relative_m_s01,
        exact_dimensions=numpy.array([num_grid_rows, num_grid_columns]))

    error_checking.assert_is_geq_numpy_array(rotation_angle_cosines, -1)
    error_checking.assert_is_leq_numpy_array(rotation_angle_cosines, 1)
    error_checking.assert_is_numpy_array(
        rotation_angle_cosines,
        exact_dimensions=numpy.array([num_grid_rows, num_grid_columns]))

    error_checking.assert_is_geq_numpy_array(rotation_angle_sines, -1)
    error_checking.assert_is_leq_numpy_array(rotation_angle_sines, 1)
    error_checking.assert_is_numpy_array(
        rotation_angle_sines,
        exact_dimensions=numpy.array([num_grid_rows, num_grid_columns]))

    u_winds_earth_relative_m_s01 = (
        rotation_angle_cosines * u_winds_grid_relative_m_s01 +
        rotation_angle_sines * v_winds_grid_relative_m_s01)
    v_winds_earth_relative_m_s01 = (
        rotation_angle_cosines * v_winds_grid_relative_m_s01 -
        rotation_angle_sines * u_winds_grid_relative_m_s01)
    return u_winds_earth_relative_m_s01, v_winds_earth_relative_m_s01
