"""Processing methods for NWP (numerical weather prediction) data."""

import copy
import numpy
import pandas
from gewittergefahr.gg_utils import number_rounding as rounder
from gewittergefahr.gg_utils import error_checking

HOURS_TO_SECONDS = 3600
TIME_FORMAT = '%Y%m%d-%H%M%S'

RAP_MODEL_NAME = 'rap'
NARR_MODEL_NAME = 'narr'
MODEL_NAMES = [RAP_MODEL_NAME, NARR_MODEL_NAME]

ID_FOR_130GRID = '130'
ID_FOR_252GRID = '252'
RAP_GRID_IDS = [ID_FOR_130GRID, ID_FOR_252GRID]

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


def check_model_name(model_name):
    """Ensures that model name is valid.

    :param model_name: Name of model (either "rap" or "narr" -- this list will
        be expanded if/when more NWP models are included in GewitterGefahr).
    :raises: ValueError: model_name is not in list of valid names.
    """

    error_checking.assert_is_string(model_name)
    if model_name not in MODEL_NAMES:
        error_string = (
            '\n\n' + str(MODEL_NAMES) +
            '\n\nValid model names (listed above) do not include "' +
            model_name + '".')
        raise ValueError(error_string)


def check_grid_id(model_name, grid_id=None):
    """Ensures that grid ID is valid for the given model.

    :param model_name: Name of model (examples: "rap" and "narr").
    :param grid_id: String ID for RAP grid (either "130" or "252").  If
        model_name != "rap", this can be left as None.
    :raises: ValueError: if grid ID is not recognized for the given model.
    """

    check_model_name(model_name)
    if model_name == RAP_MODEL_NAME and grid_id not in RAP_GRID_IDS:
        error_string = (
            '\n\n' + str(RAP_GRID_IDS) + '\n\nValid grid IDs for ' +
            model_name.upper() +
            ' model (listed above) do not include the following: "' + grid_id +
            '"')
        raise ValueError(error_string)


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

    :param u_winds_grid_relative_m_s01: numpy array of grid-relative u-winds
        (towards positive x-direction).
    :param v_winds_grid_relative_m_s01: equivalent-shape numpy array of grid-
        relative v-winds (towards positive y-direction).
    :param rotation_angle_cosines: equivalent-shape numpy array with cosines of
        rotation angles.
    :param rotation_angle_sines: equivalent-shape numpy array with sines of
        rotation angles.
    :return: u_winds_earth_relative_m_s01: equivalent-shape numpy array of
        Earth-relative (northward) u-winds.
    :return: v_winds_earth_relative_m_s01: equivalent-shape numpy array of
        Earth-relative (eastward) v-winds.
    """

    error_checking.assert_is_real_numpy_array(u_winds_grid_relative_m_s01)
    array_dimensions = numpy.asarray(u_winds_grid_relative_m_s01.shape)

    error_checking.assert_is_real_numpy_array(v_winds_grid_relative_m_s01)
    error_checking.assert_is_numpy_array(
        v_winds_grid_relative_m_s01, exact_dimensions=array_dimensions)

    error_checking.assert_is_geq_numpy_array(rotation_angle_cosines, -1)
    error_checking.assert_is_leq_numpy_array(rotation_angle_cosines, 1)
    error_checking.assert_is_numpy_array(
        rotation_angle_cosines, exact_dimensions=array_dimensions)

    error_checking.assert_is_geq_numpy_array(rotation_angle_sines, -1)
    error_checking.assert_is_leq_numpy_array(rotation_angle_sines, 1)
    error_checking.assert_is_numpy_array(
        rotation_angle_sines, exact_dimensions=array_dimensions)

    u_winds_earth_relative_m_s01 = (
        rotation_angle_cosines * u_winds_grid_relative_m_s01 +
        rotation_angle_sines * v_winds_grid_relative_m_s01)
    v_winds_earth_relative_m_s01 = (
        rotation_angle_cosines * v_winds_grid_relative_m_s01 -
        rotation_angle_sines * u_winds_grid_relative_m_s01)
    return u_winds_earth_relative_m_s01, v_winds_earth_relative_m_s01
