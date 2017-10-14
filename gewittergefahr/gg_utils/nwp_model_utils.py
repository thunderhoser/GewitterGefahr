"""Processing methods for NWP (numerical weather prediction) data."""

import copy
import numpy
import pandas
from gewittergefahr.gg_utils import number_rounding as rounder
from gewittergefahr.gg_utils import error_checking

TIME_FORMAT = '%Y%m%d-%H%M%S'
HOURS_TO_SECONDS = 3600

TEMPERATURE_COLUMN_FOR_SOUNDING_TABLES = 'temperature_kelvins'
RH_COLUMN_FOR_SOUNDING_TABLES = 'relative_humidity_percent'
SPFH_COLUMN_FOR_SOUNDING_TABLES = 'specific_humidity'
HEIGHT_COLUMN_FOR_SOUNDING_TABLES = 'geopotential_height_metres'
U_WIND_COLUMN_FOR_SOUNDING_TABLES = 'u_wind_m_s01'
V_WIND_COLUMN_FOR_SOUNDING_TABLES = 'v_wind_m_s01'

TEMPERATURE_COLUMN_FOR_SOUNDING_TABLES_GRIB1 = 'TMP'
RH_COLUMN_FOR_SOUNDING_TABLES_GRIB1 = 'RH'
SPFH_COLUMN_FOR_SOUNDING_TABLES_GRIB1 = 'SPFH'
HEIGHT_COLUMN_FOR_SOUNDING_TABLES_GRIB1 = 'HGT'
U_WIND_COLUMN_FOR_SOUNDING_TABLES_GRIB1 = 'UGRD'
V_WIND_COLUMN_FOR_SOUNDING_TABLES_GRIB1 = 'VGRD'

MIN_QUERY_TIME_COLUMN = 'min_query_time_unix_sec'
MAX_QUERY_TIME_COLUMN = 'max_query_time_unix_sec'
MODEL_TIMES_COLUMN = 'model_times_unix_sec'
MODEL_TIMES_NEEDED_COLUMN = 'model_time_needed_flags'

PREVIOUS_INTERP_METHOD = 'previous'
NEXT_INTERP_METHOD = 'next'
NOT_REALLY_INTERP_METHODS = [PREVIOUS_INTERP_METHOD, NEXT_INTERP_METHOD]
SUB_AND_LINEAR_INTERP_METHODS = ['linear', 'nearest', 'zero', 'slinear']
SUPERLINEAR_INTERP_METHODS = ['quadratic', 'cubic']
TEMPORAL_INTERP_METHODS = (
    NOT_REALLY_INTERP_METHODS + SUB_AND_LINEAR_INTERP_METHODS +
    SUPERLINEAR_INTERP_METHODS)


def get_times_needed_for_interp(query_times_unix_sec=None,
                                model_time_step_hours=None, method_string=None):
    """Finds model times needed for interpolation to each range of query times.

    Q = number of query times
    M = number of model times needed

    :param query_times_unix_sec: length-Q numpy array of query times (Unix
        format).
    :param model_time_step_hours: Model time step.  If interpolating between
        forecast times (from the same initialization), this should be the
        model's time resolution (hours between successive forecasts).  If
        interpolating between model runs (forecasts for the same valid time but
        from different initializations), this should be the model's refresh time
        (hours between successive model runs).
    :param method_string: Interpolation method.  Valid options are "previous",
        "next", "linear", "nearest", "zero", "slinear", "quadratic", and
        "cubic".  The last 6 methods are described in the documentation for
        `scipy.interpolate.interp1d`.
    :return: model_times_unix_sec: length-M numpy array of model times needed
        (Unix format).
    :return: query_to_model_times_table: pandas DataFrame with the following
        columns.  Each row corresponds to one range of query times.
    query_to_model_times_table.min_query_time_unix_sec: Minimum query time for
        this range.
    query_to_model_times_table.max_query_time_unix_sec: Max query time for this
        range.
    query_to_model_times_table.model_times_unix_sec: 1-D numpy array of model
        times needed for this range.
    query_to_model_times_table.model_time_needed_flags: length-M numpy array of
        Boolean flags.  If model_time_needed_flags[i] = True at row j, this
        means the [i]th model time is needed for interp to the [j]th range of
        query times.

    :raises: ValueError: if method_string not in TEMPORAL_INTERP_METHODS.
    """

    error_checking.assert_is_integer_numpy_array(query_times_unix_sec)
    error_checking.assert_is_numpy_array_without_nan(query_times_unix_sec)
    error_checking.assert_is_numpy_array(query_times_unix_sec, num_dimensions=1)
    error_checking.assert_is_integer(model_time_step_hours)
    error_checking.assert_is_string(method_string)

    if method_string not in TEMPORAL_INTERP_METHODS:
        error_string = (
            '\n\n' + str(TEMPORAL_INTERP_METHODS) +
            '\n\nValid temporal-interp methods (listed above) do not include "'
            + method_string + '".')
        raise ValueError(error_string)

    model_time_step_sec = model_time_step_hours * HOURS_TO_SECONDS
    min_min_query_time_unix_sec = rounder.floor_to_nearest(
        float(numpy.min(query_times_unix_sec)), model_time_step_sec)
    max_max_query_time_unix_sec = rounder.ceiling_to_nearest(
        float(numpy.max(query_times_unix_sec)), model_time_step_sec)

    num_ranges = int((max_max_query_time_unix_sec -
                      min_min_query_time_unix_sec) / model_time_step_sec)
    min_query_times_unix_sec = numpy.linspace(
        min_min_query_time_unix_sec,
        max_max_query_time_unix_sec - model_time_step_sec, num=num_ranges,
        dtype=int)
    max_query_times_unix_sec = numpy.linspace(
        min_min_query_time_unix_sec + model_time_step_sec,
        max_max_query_time_unix_sec, num=num_ranges, dtype=int)

    if method_string == PREVIOUS_INTERP_METHOD:
        min_model_time_unix_sec = copy.deepcopy(min_min_query_time_unix_sec)
        max_model_time_unix_sec = (
            max_max_query_time_unix_sec - model_time_step_sec)
    elif method_string == NEXT_INTERP_METHOD:
        min_model_time_unix_sec = (
            min_min_query_time_unix_sec + model_time_step_sec)
        max_model_time_unix_sec = copy.deepcopy(max_max_query_time_unix_sec)
    elif method_string in SUB_AND_LINEAR_INTERP_METHODS:
        min_model_time_unix_sec = copy.deepcopy(min_min_query_time_unix_sec)
        max_model_time_unix_sec = copy.deepcopy(max_max_query_time_unix_sec)
    else:
        min_model_time_unix_sec = (
            min_min_query_time_unix_sec - model_time_step_sec)
        max_model_time_unix_sec = (
            max_max_query_time_unix_sec + model_time_step_sec)

    num_model_times = int((max_model_time_unix_sec -
                           min_model_time_unix_sec) / model_time_step_sec) + 1
    model_times_unix_sec = numpy.linspace(
        min_model_time_unix_sec, max_model_time_unix_sec, num=num_model_times,
        dtype=int)

    query_to_model_times_dict = {
        MIN_QUERY_TIME_COLUMN: min_query_times_unix_sec,
        MAX_QUERY_TIME_COLUMN: max_query_times_unix_sec}
    query_to_model_times_table = pandas.DataFrame.from_dict(
        query_to_model_times_dict)

    nested_array = query_to_model_times_table[[
        MIN_QUERY_TIME_COLUMN, MIN_QUERY_TIME_COLUMN]].values.tolist()
    argument_dict = {MODEL_TIMES_COLUMN: nested_array,
                     MODEL_TIMES_NEEDED_COLUMN: nested_array}
    query_to_model_times_table = query_to_model_times_table.assign(
        **argument_dict)

    for i in range(num_ranges):
        if method_string == PREVIOUS_INTERP_METHOD:
            these_model_times_unix_sec = numpy.array(
                [min_query_times_unix_sec[i]], dtype=int)
        elif method_string == NEXT_INTERP_METHOD:
            these_model_times_unix_sec = numpy.array(
                [max_query_times_unix_sec[i]], dtype=int)
        elif method_string in SUB_AND_LINEAR_INTERP_METHODS:
            these_model_times_unix_sec = numpy.array(
                [min_query_times_unix_sec[i], max_query_times_unix_sec[i]],
                dtype=int)
        else:
            these_model_times_unix_sec = numpy.array(
                [min_query_times_unix_sec[i] - model_time_step_sec,
                 min_query_times_unix_sec[i], max_query_times_unix_sec[i],
                 max_query_times_unix_sec[i] + model_time_step_sec], dtype=int)

        query_to_model_times_table[
            MODEL_TIMES_COLUMN].values[i] = these_model_times_unix_sec
        query_to_model_times_table[MODEL_TIMES_NEEDED_COLUMN].values[i] = [
            t in these_model_times_unix_sec for t in model_times_unix_sec]

    return model_times_unix_sec, query_to_model_times_table


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
