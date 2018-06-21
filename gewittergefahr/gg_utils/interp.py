"""Interpolation methods."""

import copy
import math
import os.path
import numpy
import pandas
import scipy.interpolate
from gewittergefahr.gg_io import grib_io
from gewittergefahr.gg_io import nwp_model_io
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import error_checking

LARGE_TIME_DIFFERENCE_SEC = int(1e10)

HEIGHT_NAME = 'geopotential_height_metres'
HEIGHT_NAME_GRIB1 = 'HGT'
TEMPERATURE_NAME = 'temperature_kelvins'
TEMPERATURE_NAME_GRIB1 = 'TMP'

NEAREST_NEIGHBOUR_METHOD_STRING = 'nearest'
SPLINE_METHOD_STRING = 'spline'
SPATIAL_INTERP_METHOD_STRINGS = [
    NEAREST_NEIGHBOUR_METHOD_STRING, SPLINE_METHOD_STRING]

DEFAULT_SPLINE_DEGREE = 3
SMOOTHING_FACTOR_FOR_SPATIAL_INTERP = 0

PREV_NEIGHBOUR_METHOD_STRING = 'previous'
NEXT_NEIGHBOUR_METHOD_STRING = 'next'
LINEAR_METHOD_STRING = 'linear'
SPLINE0_METHOD_STRING = 'zero'
SPLINE1_METHOD_STRING = 'slinear'
SPLINE2_METHOD_STRING = 'quadratic'
SPLINE3_METHOD_STRING = 'cubic'
TEMPORAL_INTERP_METHOD_STRINGS = [
    PREV_NEIGHBOUR_METHOD_STRING, NEXT_NEIGHBOUR_METHOD_STRING,
    NEAREST_NEIGHBOUR_METHOD_STRING, LINEAR_METHOD_STRING,
    SPLINE0_METHOD_STRING, SPLINE1_METHOD_STRING, SPLINE2_METHOD_STRING,
    SPLINE3_METHOD_STRING]

QUERY_TIME_COLUMN = 'unix_time_sec'
QUERY_LAT_COLUMN = 'latitude_deg'
QUERY_LNG_COLUMN = 'longitude_deg'
QUERY_X_COLUMN = 'x_coordinate_metres'
QUERY_Y_COLUMN = 'y_coordinate_metres'

GRID_POINT_X_KEY = 'grid_point_x_metres'
GRID_POINT_Y_KEY = 'grid_point_y_metres'
FIELD_NAMES_GRIB1_KEY = 'field_names_grib1'
ROTATE_WIND_FLAGS_KEY = 'rotate_wind_flags'
FIELD_NAMES_OTHER_COMPONENT_KEY = 'field_names_other_wind_component_grib1'
OTHER_WIND_COMPONENT_INDICES_KEY = 'other_wind_component_indices'
ROTATION_SINES_KEY = 'rotation_sine_by_query_point'
ROTATION_COSINES_KEY = 'rotation_cosine_by_query_point'

# TODO(thunderhoser): Allow this module to interpolate between different lead
# times from the same model initialization, rather than just interpolating
# between zero-hour analyses from the same initialization.
FORECAST_LEAD_TIME_HOURS = 0


def _interp_to_previous_time(
        input_matrix, input_times_unix_sec, query_times_unix_sec):
    """Previous-neighbour temporal interpolation.

    :param input_matrix: See documentation for `interp_in_time`.
    :param input_times_unix_sec: Same.
    :param query_times_unix_sec: Same.
    :return: interp_matrix: Same.
    """

    error_checking.assert_is_geq_numpy_array(
        query_times_unix_sec, numpy.min(input_times_unix_sec))

    num_query_times = len(query_times_unix_sec)
    list_of_interp_matrices = []

    for i in range(num_query_times):
        these_time_diffs_sec = query_times_unix_sec[i] - input_times_unix_sec
        these_time_diffs_sec[
            these_time_diffs_sec < 0] = LARGE_TIME_DIFFERENCE_SEC

        this_previous_index = numpy.argmin(these_time_diffs_sec)
        list_of_interp_matrices.append(
            numpy.take(input_matrix, this_previous_index, axis=-1))

    return numpy.stack(list_of_interp_matrices, axis=-1)


def _interp_to_next_time(
        input_matrix, input_times_unix_sec, query_times_unix_sec):
    """Next-neighbour temporal interpolation.

    :param input_matrix: See documentation for `interp_in_time`.
    :param input_times_unix_sec: Same.
    :param query_times_unix_sec: Same.
    :return: interp_matrix: Same.
    """

    error_checking.assert_is_leq_numpy_array(
        query_times_unix_sec, numpy.max(input_times_unix_sec))

    num_query_times = len(query_times_unix_sec)
    list_of_interp_matrices = []

    for i in range(num_query_times):
        these_time_diffs_sec = input_times_unix_sec - query_times_unix_sec[i]
        these_time_diffs_sec[
            these_time_diffs_sec < 0] = LARGE_TIME_DIFFERENCE_SEC

        this_next_index = numpy.argmin(these_time_diffs_sec)
        list_of_interp_matrices.append(
            numpy.take(input_matrix, this_next_index, axis=-1))

    return numpy.stack(list_of_interp_matrices, axis=-1)


def _find_nearest_value(sorted_input_values, test_value):
    """Finds nearest value in array to test value.

    This method is based on the following:

    https://stackoverflow.com/posts/26026189/revisions

    :param sorted_input_values: 1-D numpy array.  Must be sorted in ascending
        order.
    :param test_value: Test value.
    :return: nearest_value: Nearest value in `sorted_input_values` to
        `test_value`.
    :return: nearest_index: Array index of nearest value.
    """

    nearest_index = numpy.searchsorted(
        sorted_input_values, test_value, side='left')

    subtract_one = nearest_index > 0 and (
        nearest_index == len(sorted_input_values) or
        math.fabs(test_value - sorted_input_values[nearest_index - 1]) <
        math.fabs(test_value - sorted_input_values[nearest_index]))
    if subtract_one:
        nearest_index -= 1

    return sorted_input_values[nearest_index], nearest_index


def _nn_interp_from_xy_grid_to_points(
        input_matrix, sorted_grid_point_x_metres, sorted_grid_point_y_metres,
        query_x_coords_metres, query_y_coords_metres):
    """Nearest-neighbour interpolation from x-y grid to scattered points.

    :param input_matrix: See doc for `interp_from_xy_grid_to_points`.
    :param sorted_grid_point_x_metres: Same.
    :param sorted_grid_point_y_metres: Same.
    :param query_x_coords_metres: Same.
    :param query_y_coords_metres: Same.
    :return: interp_values: Same.
    """

    num_query_points = len(query_x_coords_metres)
    interp_values = numpy.full(num_query_points, numpy.nan)

    for i in range(num_query_points):
        _, this_row = _find_nearest_value(
            sorted_grid_point_y_metres, query_y_coords_metres[i])
        _, this_column = _find_nearest_value(
            sorted_grid_point_x_metres, query_x_coords_metres[i])
        interp_values[i] = input_matrix[this_row, this_column]

    return interp_values


def _get_wind_rotation_metadata(field_names_grib1, model_name):
    """Returns metadata for wind rotation.

    Metadata may be used to rotate NWP winds from grid-relative to
    Earth-relative.

    F = number of NWP fields

    :param field_names_grib1: length-F list of field names in grib1 format.
    :param model_name: Model name (must be accepted by
        `nwp_model_utils.check_model_name`).
    :return: metadata_dict: Dictionary with the following keys.
    metadata_dict['field_names_grib1']: length-F list of field names in grib1
        format.
    metadata_dict['rotate_wind_flags']: length-F numpy array of Boolean flag,
        indicating which fields need to be rotated (only grid-relative u-wind
        and v-wind must be rotated).
    metadata_dict['field_names_other_wind_component_grib1']: length-F list of
        field names.  If field_names_grib1[j] is a u-wind component that must be
        rotated, field_names_other_wind_component_grib1[j] is the corresponding
        v-wind component.  If field_names_grib1[j] is a v-wind component that
        must be rotated, field_names_other_wind_component_grib1[j] is the
        corresponding u-wind component.  Otherwise,
        field_names_other_wind_component_grib1[j] is None.
    metadata_dict['other_wind_component_indices']: length-F numpy array of
        indices.  If other_wind_component_index[j] = k, field_name_grib1[j]
        and field_name_grib1[k] are components in the same vector, so will be
        rotated together.  If other_wind_component_index[j] = -1, the [j]th
        field need not be rotated.
    """

    nwp_model_utils.check_model_name(model_name)

    num_fields = len(field_names_grib1)
    rotate_wind_flags = numpy.full(num_fields, False, dtype=bool)
    field_names_other_wind_component_grib1 = [''] * num_fields
    other_wind_component_indices = numpy.full(num_fields, -1, dtype=int)

    if model_name != nwp_model_utils.NARR_MODEL_NAME:
        for j in range(num_fields):
            rotate_wind_flags[j] = grib_io.is_wind_field(field_names_grib1[j])
            if not rotate_wind_flags[j]:
                continue

            field_names_other_wind_component_grib1[j] = (
                grib_io.switch_uv_in_field_name(field_names_grib1[j]))
            if (field_names_other_wind_component_grib1[j]
                    not in field_names_grib1):
                continue

            other_wind_component_indices[j] = field_names_grib1.index(
                field_names_other_wind_component_grib1[j])

    return {
        FIELD_NAMES_GRIB1_KEY: field_names_grib1,
        ROTATE_WIND_FLAGS_KEY: rotate_wind_flags,
        FIELD_NAMES_OTHER_COMPONENT_KEY: field_names_other_wind_component_grib1,
        OTHER_WIND_COMPONENT_INDICES_KEY: other_wind_component_indices
    }


def _read_nwp_for_interp(
        init_times_unix_sec, query_to_model_times_row, field_name_grib1,
        field_name_other_wind_component_grib1, list_of_model_grids,
        list_of_model_grids_other_wind_component, model_name,
        top_grib_directory_name, grid_id=None,
        wgrib_exe_name=grib_io.WGRIB_EXE_NAME_DEFAULT,
        wgrib2_exe_name=grib_io.WGRIB2_EXE_NAME_DEFAULT,
        raise_error_if_missing=False):
    """Reads NWP data needed for interpolation to a range of query times.

    T = number of model-initialization times

    :param init_times_unix_sec: length-T numpy array of model-initialization
        times.
    :param query_to_model_times_row: Single row of pandas DataFrame created by
        `nwp_model_utils.get_times_needed_for_interp`.
    :param field_name_grib1: Name of interpoland in grib1 format.
    :param field_name_other_wind_component_grib1: Name of other wind component
        in grib1 format.  If `field_name_grib1` is anything other than
        grid-relative u-wind or v-wind, this argument should be None.
    :param list_of_model_grids: length-T list, where the [i]th element is either
        None or the full grid (of `field_name_grib1`) from the [i]th
        initialization time.
    :param list_of_model_grids_other_wind_component: Same as
        `list_of_model_grids`, but for `field_name_other_wind_component_grib1`
        rather than `field_name_grib1`.
    :param model_name: Model name (must be accepted by
        `nwp_model_utils.check_model_name`).
    :param top_grib_directory_name: Name of top-level directory with grib files
        containing NWP data.
    :param grid_id: ID for model grid (must be accepted by
        `nwp_model_utils.check_grid_id`).
    :param wgrib_exe_name: Path to wgrib executable.
    :param wgrib2_exe_name: Path to wgrib2 executable.
    :param raise_error_if_missing: Boolean flag.  If any grib file cannot be
        read (because it is missing or corrupt) and raise_error_if_missing =
        True, this method will error out.  If any grib file cannot be read and
        raise_error_if_missing = False, this method will return None for the
        relevant entries in `list_of_model_grids` and
        `list_of_model_grids_other_wind_component`.
    :return: list_of_model_grids: Same as input, except that different elements
        are filled and different elements are None.
    :return: list_of_model_grids_other_wind_component: See above.
    :return: missing_data: Boolean flag.  If True, some data needed for the
        interpolation are missing.
    """

    missing_data = False
    rotate_wind = field_name_other_wind_component_grib1 is not None

    init_time_needed_flags = query_to_model_times_row[
        nwp_model_utils.MODEL_TIMES_NEEDED_COLUMN].values[0]
    num_init_times = len(init_times_unix_sec)

    for i in range(num_init_times):
        if init_time_needed_flags[i]:
            continue

        list_of_model_grids[i] = None
        list_of_model_grids_other_wind_component[i] = None

    for i in range(num_init_times):
        if not init_time_needed_flags[i]:
            continue

        read_this_file = list_of_model_grids[i] is None or (
            rotate_wind and list_of_model_grids_other_wind_component[i] is None)
        if not read_this_file:
            continue

        this_grib_file_name = nwp_model_io.find_grib_file(
            top_directory_name=top_grib_directory_name,
            init_time_unix_sec=init_times_unix_sec[i], model_name=model_name,
            grid_id=grid_id, lead_time_hours=FORECAST_LEAD_TIME_HOURS,
            raise_error_if_missing=raise_error_if_missing)

        if not os.path.isfile(this_grib_file_name):
            missing_data = True
            list_of_model_grids[i] = None
            list_of_model_grids_other_wind_component[i] = None
            continue

        list_of_model_grids[i] = nwp_model_io.read_field_from_grib_file(
            grib_file_name=this_grib_file_name,
            field_name_grib1=field_name_grib1, model_name=model_name,
            grid_id=grid_id, wgrib_exe_name=wgrib_exe_name,
            wgrib2_exe_name=wgrib2_exe_name,
            raise_error_if_fails=raise_error_if_missing)

        if list_of_model_grids[i] is None:
            missing_data = True
            list_of_model_grids_other_wind_component[i] = None
            continue

        if rotate_wind:
            list_of_model_grids_other_wind_component[i] = (
                nwp_model_io.read_field_from_grib_file(
                    grib_file_name=this_grib_file_name,
                    field_name_grib1=field_name_other_wind_component_grib1,
                    model_name=model_name, grid_id=grid_id,
                    wgrib_exe_name=wgrib_exe_name,
                    wgrib2_exe_name=wgrib2_exe_name,
                    raise_error_if_fails=raise_error_if_missing))

            if list_of_model_grids_other_wind_component[i] is None:
                missing_data = True
                list_of_model_grids[i] = None
                continue

    if not rotate_wind:
        list_of_model_grids_other_wind_component = None

    return (list_of_model_grids, list_of_model_grids_other_wind_component,
            missing_data)


def _get_grids_for_model(model_name):
    """Returns list of grids used to interpolate from the given model.

    :param model_name: Model name (must be accepted by
        `nwp_model_utils.check_model_name`).
    :return: grid_ids: 1-D list of grid IDs (strings).
    """

    if model_name == nwp_model_utils.NARR_MODEL_NAME:
        return [nwp_model_utils.ID_FOR_221GRID]
    return [nwp_model_utils.ID_FOR_130GRID, nwp_model_utils.ID_FOR_252GRID]


def _read_nwp_for_interp_any_grid(
        init_times_unix_sec, query_to_model_times_row, field_name_grib1,
        field_name_other_wind_component_grib1, list_of_model_grids,
        list_of_model_grids_other_wind_component, top_grib_directory_name,
        model_name, wgrib_exe_name=grib_io.WGRIB_EXE_NAME_DEFAULT,
        wgrib2_exe_name=grib_io.WGRIB2_EXE_NAME_DEFAULT,
        raise_error_if_missing=False):
    """Reads NWP data needed for interpolation to a range of query times.

    This method reads data from the highest-resolution grid available at each
    time.

    :param init_times_unix_sec: See doc for `_read_nwp_for_interp`.
    :param query_to_model_times_row: Same.
    :param field_name_grib1: Same.
    :param field_name_other_wind_component_grib1: Same.
    :param list_of_model_grids: Same.
    :param list_of_model_grids_other_wind_component: Same.
    :param top_grib_directory_name: Same.
    :param model_name: Same.
    :param wgrib_exe_name: Same.
    :param wgrib2_exe_name: Same.
    :param raise_error_if_missing: Same.
    :return: list_of_model_grids: Same.
    :return: list_of_model_grids_other_wind_component: Same.
    :return: missing_data: Same.
    """

    grid_ids = _get_grids_for_model(model_name)

    for i in range(len(grid_ids)):
        (list_of_model_grids, list_of_model_grids_other_wind_component,
         missing_data
        ) = _read_nwp_for_interp(
            init_times_unix_sec=init_times_unix_sec,
            query_to_model_times_row=query_to_model_times_row,
            field_name_grib1=field_name_grib1,
            field_name_other_wind_component_grib1=
            field_name_other_wind_component_grib1,
            list_of_model_grids=list_of_model_grids,
            list_of_model_grids_other_wind_component=
            list_of_model_grids_other_wind_component, model_name=model_name,
            top_grib_directory_name=top_grib_directory_name,
            grid_id=grid_ids[i], wgrib_exe_name=wgrib_exe_name,
            wgrib2_exe_name=wgrib2_exe_name,
            raise_error_if_missing=(
                raise_error_if_missing and i == len(grid_ids) - 1))

        if missing_data:
            continue

    return (list_of_model_grids, list_of_model_grids_other_wind_component,
            missing_data)


def _stack_1d_arrays_horizontally(list_of_1d_arrays):
    """Stacks 1-D numpy arrays horizontally.

    The result is a 2-D matrix, where each column is one of the input arrays.

    :param list_of_1d_arrays: 1-D list of 1-D numpy arrays.
    :return: matrix_2d: Resulting matrix.
    """

    return numpy.stack(tuple(list_of_1d_arrays), axis=-1)


def _prep_to_interp_nwp_from_xy_grid(
        query_point_table, model_name, grid_id, field_names, field_names_grib1):
    """Prepares input data for `interp_nwp_from_xy_grid`.

    M = number of rows (unique y-coordinates at grid points)
    N = number of columns (unique x-coordinates at grid points)
    F = number of fields to interpolate
    Q = number of query points

    :param query_point_table: See doc for `interp_nwp_from_xy_grid`.
    :param model_name: Model name (must be accepted by
        `nwp_model_utils.check_model_name`).
    :param grid_id: ID for model grid (must be accepted by
        `nwp_model_utils.check_grid_id`).
    :param field_names: length-F list of field names in GewitterGefahr format.
    :param field_names_grib1: length-F list of field names in grib1 format.
    :return: query_point_table: Same as input, except that columns
        "latitude_deg" and "longitude_deg" are replaced with
        "x_coordinate_metres" and "y_coordinate_metres".
    :return: interp_table: Same as output from `interp_nwp_from_xy_grid`, except
        that all values are NaN.
    :return: metadata_dict: Dictionary with the following keys.
    metadata_dict['grid_point_x_metres']: length-N numpy array with x-coords
        of grid points.
    metadata_dict['grid_point_y_metres']: length-M numpy array with y-coords
        of grid points.
    metadata_dict['field_names_grib1']: See doc for
        `_get_wind_rotation_metadata`.
    metadata_dict['rotate_wind_flags']: Same.
    metadata_dict['field_names_other_wind_component_grib1']: Same.
    metadata_dict['other_wind_component_indices']: Same.
    metadata_dict.rotation_sine_by_query_point: length-Q numpy array of sines,
        which will be used to rotate winds from grid-relative to Earth-relative.
    metadata_dict.rotation_cosine_by_query_point: Same but for cosines.
    """

    # Error-checking.
    error_checking.assert_is_string_list(field_names)
    error_checking.assert_is_string_list(field_names_grib1)
    error_checking.assert_is_numpy_array(
        numpy.asarray(field_names), num_dimensions=1)

    num_fields = len(field_names)
    error_checking.assert_is_numpy_array(
        numpy.asarray(field_names_grib1),
        exact_dimensions=numpy.array([num_fields]))

    # Find grid points for model.
    grid_point_x_metres, grid_point_y_metres = (
        nwp_model_utils.get_xy_grid_points(
            model_name=model_name, grid_id=grid_id))

    # Project query points to model coords.
    query_x_metres, query_y_metres = (
        nwp_model_utils.project_latlng_to_xy(
            latitudes_deg=query_point_table[QUERY_LAT_COLUMN].values,
            longitudes_deg=query_point_table[QUERY_LNG_COLUMN].values,
            model_name=model_name, grid_id=grid_id))
    argument_dict = {
        QUERY_X_COLUMN: query_x_metres,
        QUERY_Y_COLUMN: query_y_metres
    }
    query_point_table = query_point_table.assign(**argument_dict)

    # Create interp_table.
    num_query_points = len(query_point_table.index)
    nan_array = numpy.full(num_query_points, numpy.nan)
    interp_dict = {}
    for j in range(num_fields):
        interp_dict.update({field_names[j]: nan_array})
    interp_table = pandas.DataFrame.from_dict(interp_dict)

    # Create the rest of the metadata.
    metadata_dict = _get_wind_rotation_metadata(
        field_names_grib1=field_names_grib1, model_name=model_name)
    metadata_dict.update({
        GRID_POINT_X_KEY: grid_point_x_metres,
        GRID_POINT_Y_KEY: grid_point_y_metres
    })

    if numpy.any(metadata_dict[ROTATE_WIND_FLAGS_KEY]):
        rotation_cosine_by_query_point, rotation_sine_by_query_point = (
            nwp_model_utils.get_wind_rotation_angles(
                latitudes_deg=query_point_table[QUERY_LAT_COLUMN].values,
                longitudes_deg=query_point_table[QUERY_LNG_COLUMN].values,
                model_name=model_name))
    else:
        rotation_cosine_by_query_point = None
        rotation_sine_by_query_point = None

    metadata_dict.update({
        ROTATION_COSINES_KEY: rotation_cosine_by_query_point,
        ROTATION_SINES_KEY: rotation_sine_by_query_point
    })

    query_point_table.drop(
        [QUERY_LAT_COLUMN, QUERY_LNG_COLUMN], axis=1, inplace=True)
    return query_point_table, interp_table, metadata_dict


def _find_heights_with_temperature(
        warm_temperatures_kelvins, cold_temperatures_kelvins,
        warm_heights_m_asl, cold_heights_m_asl, target_temperature_kelvins):
    """At each horizontal point, finds height with the target temperature.

    P = number of horizontal points

    :param warm_temperatures_kelvins: length-P numpy array of temperatures on
        warm side.
    :param cold_temperatures_kelvins: length-P numpy array of temperatures on
        cold side.
    :param warm_heights_m_asl: length-P numpy array of heights (metres above sea
        level) corresponding to `warm_temperatures_kelvins`.
    :param cold_heights_m_asl: length-P numpy array of heights (metres above sea
        level) corresponding to `cold_temperatures_kelvins`.
    :param target_temperature_kelvins: Target temperature.
    :return: target_heights_m_asl: length-P numpy array of heights (metres above
        sea level) with the target temperature, estimated by interpolation.
    """

    bad_point_flags = numpy.logical_or(
        numpy.isnan(warm_temperatures_kelvins),
        numpy.isnan(cold_temperatures_kelvins))

    num_points = len(warm_temperatures_kelvins)
    target_heights_m_asl = numpy.full(num_points, numpy.nan)
    if numpy.all(bad_point_flags):
        return target_heights_m_asl

    good_point_indices = numpy.where(numpy.invert(bad_point_flags))[0]

    warm_minus_cold_kelvins = (
        warm_temperatures_kelvins[good_point_indices] -
        cold_temperatures_kelvins[good_point_indices])
    target_minus_cold_kelvins = (
        target_temperature_kelvins -
        cold_temperatures_kelvins[good_point_indices])

    warm_minus_cold_metres = (warm_heights_m_asl[good_point_indices] -
                              cold_heights_m_asl[good_point_indices])
    target_minus_cold_metres = warm_minus_cold_metres * (
        target_minus_cold_kelvins / warm_minus_cold_kelvins)
    target_heights_m_asl[good_point_indices] = (
        cold_heights_m_asl[good_point_indices] + target_minus_cold_metres)

    return target_heights_m_asl


def check_temporal_interp_method(interp_method_string):
    """Ensures that temporal-interpolation method is valid.

    :param interp_method_string: Interp method.
    :raises: ValueError: if `interp_method_string not in
        TEMPORAL_INTERP_METHOD_STRINGS`.
    """

    error_checking.assert_is_string(interp_method_string)
    if interp_method_string not in TEMPORAL_INTERP_METHOD_STRINGS:
        error_string = (
            '\n\n{0:s}\nValid temporal-interp methods (listed above) do not '
            'include "{1:s}".'
        ).format(str(TEMPORAL_INTERP_METHOD_STRINGS), interp_method_string)
        raise ValueError(error_string)


def check_spatial_interp_method(interp_method_string):
    """Ensures that spatial-interpolation method is valid.

    :param interp_method_string: Interp method.
    :raises: ValueError: if `interp_method_string not in
        SPATIAL_INTERP_METHOD_STRINGS`.
    """

    error_checking.assert_is_string(interp_method_string)
    if interp_method_string not in SPATIAL_INTERP_METHOD_STRINGS:
        error_string = (
            '\n\n{0:s}\nValid spatial-interp methods (listed above) do not '
            'include "{1:s}".'
        ).format(str(SPATIAL_INTERP_METHOD_STRINGS), interp_method_string)
        raise ValueError(error_string)


def interp_in_time(
        input_matrix, sorted_input_times_unix_sec, query_times_unix_sec,
        method_string, extrapolate=False):
    """Temporal interpolation.

    D = number of dimensions (for both input_matrix and interp_matrix)
    N = number of input times
    Q = number of query times

    :param input_matrix: D-dimensional numpy array, where time increases along
        the last axis (length N).
    :param sorted_input_times_unix_sec: length-N numpy array of input times,
        sorted in ascending order.
    :param query_times_unix_sec: length-Q numpy array of output times.
    :param method_string: Interp method (must be accepted by
        `check_temporal_interp_method`).
    :param extrapolate: Boolean flag.  If True, will extrapolate to times
        outside the range of `sorted_input_times_unix_sec`.  If False, will
        throw an error if `query_times_unix_sec` includes times outside the
        range of `sorted_input_times_unix_sec`.
    :return: interp_matrix: D-dimensional numpy array, where the last axis
        represents time (length Q).  The last axis is ordered in the same way as
        `query_times_unix_sec`, so that query_times_unix_sec[i] corresponds to
        `interp_matrix[..., i]`.
    """

    # error_checking.assert_is_numpy_array_without_nan(input_matrix)
    check_temporal_interp_method(method_string)
    error_checking.assert_is_boolean(extrapolate)

    error_checking.assert_is_integer_numpy_array(sorted_input_times_unix_sec)
    error_checking.assert_is_numpy_array_without_nan(
        sorted_input_times_unix_sec)
    error_checking.assert_is_numpy_array(
        sorted_input_times_unix_sec, num_dimensions=1)

    error_checking.assert_is_integer_numpy_array(query_times_unix_sec)
    error_checking.assert_is_numpy_array_without_nan(query_times_unix_sec)
    error_checking.assert_is_numpy_array(query_times_unix_sec, num_dimensions=1)

    if method_string == PREV_NEIGHBOUR_METHOD_STRING:
        return _interp_to_previous_time(
            input_matrix=input_matrix,
            input_times_unix_sec=sorted_input_times_unix_sec,
            query_times_unix_sec=query_times_unix_sec)

    if method_string == NEXT_NEIGHBOUR_METHOD_STRING:
        return _interp_to_next_time(
            input_matrix=input_matrix,
            input_times_unix_sec=sorted_input_times_unix_sec,
            query_times_unix_sec=query_times_unix_sec)

    if extrapolate:
        interp_object = scipy.interpolate.interp1d(
            sorted_input_times_unix_sec, input_matrix, kind=method_string,
            bounds_error=False, fill_value='extrapolate', assume_sorted=True)
    else:
        interp_object = scipy.interpolate.interp1d(
            sorted_input_times_unix_sec, input_matrix, kind=method_string,
            bounds_error=True, assume_sorted=True)

    return interp_object(query_times_unix_sec)


def interp_from_xy_grid_to_points(
        input_matrix, sorted_grid_point_x_metres, sorted_grid_point_y_metres,
        query_x_coords_metres, query_y_coords_metres,
        method_string=NEAREST_NEIGHBOUR_METHOD_STRING,
        spline_degree=DEFAULT_SPLINE_DEGREE, extrapolate=False):
    """Interpolation from x-y grid to scattered points.

    M = number of rows (unique y-coordinates at grid points)
    N = number of columns (unique x-coordinates at grid points)
    Q = number of query points

    :param input_matrix: M-by-N numpy array of gridded data.
    :param sorted_grid_point_x_metres: length-N numpy array with x-coordinates
        of grid points.  Must be sorted in ascending order.  Also,
        sorted_grid_point_x_metres[j] must match input_matrix[:, j].
    :param sorted_grid_point_y_metres: length-M numpy array with y-coordinates
        of grid points.  Must be sorted in ascending order.  Also,
        sorted_grid_point_y_metres[i] must match input_matrix[i, :].
    :param query_x_coords_metres: length-Q numpy array with x-coordinates of
        query points.
    :param query_y_coords_metres: length-Q numpy array with y-coordinates of
        query points.
    :param method_string: Interpolation method (must be accepted by
        `check_spatial_interp_method`).
    :param spline_degree: [used only if method_string = "spline"]
        Polynomial degree for spline interpolation (1 for linear, 2 for
        quadratic, 3 for cubic).
    :param extrapolate: Boolean flag.  If True, will extrapolate to points
        outside the domain (specified by `sorted_grid_point_x_metres` and
        `sorted_grid_point_y_metres`).  If False, will throw an error if there
        are query points outside the domain.
    :return: interp_values: length-Q numpy array of interpolated values.
    """

    error_checking.assert_is_numpy_array_without_nan(sorted_grid_point_x_metres)
    error_checking.assert_is_numpy_array(
        sorted_grid_point_x_metres, num_dimensions=1)
    num_grid_columns = len(sorted_grid_point_x_metres)

    error_checking.assert_is_numpy_array_without_nan(sorted_grid_point_y_metres)
    error_checking.assert_is_numpy_array(
        sorted_grid_point_y_metres, num_dimensions=1)
    num_grid_rows = len(sorted_grid_point_y_metres)

    error_checking.assert_is_real_numpy_array(input_matrix)
    error_checking.assert_is_numpy_array(
        input_matrix, exact_dimensions=numpy.array(
            [num_grid_rows, num_grid_columns]))

    error_checking.assert_is_numpy_array_without_nan(query_x_coords_metres)
    error_checking.assert_is_numpy_array(
        query_x_coords_metres, num_dimensions=1)
    num_query_points = len(query_x_coords_metres)

    error_checking.assert_is_numpy_array_without_nan(query_y_coords_metres)
    error_checking.assert_is_numpy_array(
        query_y_coords_metres, exact_dimensions=numpy.array([num_query_points]))

    error_checking.assert_is_boolean(extrapolate)
    if not extrapolate:
        error_checking.assert_is_geq_numpy_array(
            query_x_coords_metres, numpy.min(sorted_grid_point_x_metres))
        error_checking.assert_is_leq_numpy_array(
            query_x_coords_metres, numpy.max(sorted_grid_point_x_metres))
        error_checking.assert_is_geq_numpy_array(
            query_y_coords_metres, numpy.min(sorted_grid_point_y_metres))
        error_checking.assert_is_leq_numpy_array(
            query_y_coords_metres, numpy.max(sorted_grid_point_y_metres))

    check_spatial_interp_method(method_string)
    if method_string == NEAREST_NEIGHBOUR_METHOD_STRING:
        return _nn_interp_from_xy_grid_to_points(
            input_matrix=input_matrix,
            sorted_grid_point_x_metres=sorted_grid_point_x_metres,
            sorted_grid_point_y_metres=sorted_grid_point_y_metres,
            query_x_coords_metres=query_x_coords_metres,
            query_y_coords_metres=query_y_coords_metres)

    interp_object = scipy.interpolate.RectBivariateSpline(
        sorted_grid_point_y_metres, sorted_grid_point_x_metres, input_matrix,
        kx=spline_degree, ky=spline_degree,
        s=SMOOTHING_FACTOR_FOR_SPATIAL_INTERP)

    return interp_object(
        query_y_coords_metres, query_x_coords_metres, grid=False)


def interp_nwp_from_xy_grid(
        query_point_table, field_names, field_names_grib1, model_name,
        top_grib_directory_name, use_all_grids=True, grid_id=None,
        temporal_interp_method_string=PREV_NEIGHBOUR_METHOD_STRING,
        spatial_interp_method_string=NEAREST_NEIGHBOUR_METHOD_STRING,
        spline_degree=DEFAULT_SPLINE_DEGREE,
        wgrib_exe_name=grib_io.WGRIB_EXE_NAME_DEFAULT,
        wgrib2_exe_name=grib_io.WGRIB2_EXE_NAME_DEFAULT,
        raise_error_if_missing=False):
    """Interpolates NWP data from x-y grid in both space and time.

    Each query point consists of (latitude, longitude, time).  Before
    interpolation, query points will be projected to the same x-y space as the
    model.

    F = number of fields to interpolate
    Q = number of query points

    :param query_point_table: Q-row pandas DataFrame with the following columns.
    query_point_table.unix_time_sec: Time.
    query_point_table.latitude_deg: Latitude (deg N).
    query_point_table.longitude_deg: Longitude (deg E).

    :param field_names: length-F list of field names in GewitterGefahr format.
    :param field_names_grib1: length-F list of field names in grib1 format.
    :param model_name: Model name (must be accepted by
        `nwp_model_utils.check_model_name`).
    :param top_grib_directory_name: Name of top-level directory with grib files
        containing NWP data.
    :param use_all_grids: Boolean flag.  If True, this method will interp from
        the highest-resolution grid available at each model-initialization time.
        If False, will interpolate from only one grid.
    :param grid_id: [used only if use_all_grids = False]
        Model grid (must be accepted by `nwp_model_utils.check_grid_id`).
    :param temporal_interp_method_string: Temporal interp method (must be
        accepted by `check_temporal_interp_method`).
    :param spatial_interp_method_string: Spatial interp method (must be
        accepted by `check_spatial_interp_method`).
    :param spline_degree: See doc for `interp_from_xy_grid_to_points`.
    :param wgrib_exe_name: Path to wgrib executable.
    :param wgrib2_exe_name: Path to wgrib2 executable.
    :param raise_error_if_missing: See doc for `_read_nwp_for_interp`.
    :return: interp_table: pandas DataFrame, where each column is one field and
        each row is one query point.  Column names are taken directly from the
        input list `field_names`.
    """

    error_checking.assert_is_boolean(use_all_grids)
    nwp_model_utils.check_model_name(model_name)

    if model_name == nwp_model_utils.NARR_MODEL_NAME or use_all_grids:
        grid_ids = _get_grids_for_model(model_name)
    else:
        grid_ids = [grid_id]

    num_grids = len(grid_ids)
    x_points_by_grid_metres = [numpy.array([])] * num_grids
    y_points_by_grid_metres = [numpy.array([])] * num_grids
    query_point_table_by_grid = [pandas.DataFrame()] * num_grids

    for g in range(num_grids):
        (query_point_table_by_grid[g], interp_table, metadata_dict
        ) = _prep_to_interp_nwp_from_xy_grid(
            query_point_table=copy.deepcopy(query_point_table),
            model_name=model_name, grid_id=grid_ids[g], field_names=field_names,
            field_names_grib1=field_names_grib1)

        x_points_by_grid_metres[g] = metadata_dict[GRID_POINT_X_KEY]
        y_points_by_grid_metres[g] = metadata_dict[GRID_POINT_Y_KEY]

    rotate_wind_flags = metadata_dict[ROTATE_WIND_FLAGS_KEY]
    field_names_other_wind_component_grib1 = metadata_dict[
        FIELD_NAMES_OTHER_COMPONENT_KEY]
    other_wind_component_indices = metadata_dict[
        OTHER_WIND_COMPONENT_INDICES_KEY]
    rotation_sine_by_query_point = metadata_dict[ROTATION_SINES_KEY]
    rotation_cosine_by_query_point = metadata_dict[ROTATION_COSINES_KEY]

    _, init_time_step_hours = nwp_model_utils.get_time_steps(model_name)
    init_times_unix_sec, query_to_model_times_table = (
        nwp_model_utils.get_times_needed_for_interp(
            query_times_unix_sec=query_point_table[QUERY_TIME_COLUMN].values,
            model_time_step_hours=init_time_step_hours,
            method_string=temporal_interp_method_string))

    num_init_times = len(init_times_unix_sec)
    num_query_time_ranges = len(query_to_model_times_table.index)
    num_fields = len(field_names)
    interp_done_by_field = numpy.full(num_fields, False, dtype=bool)

    for j in range(num_fields):
        if interp_done_by_field[j]:
            continue

        list_of_2d_grids = [None] * num_init_times
        list_of_2d_grids_other_wind_component = [None] * num_init_times

        for i in range(num_query_time_ranges):
            if i == num_query_time_ranges - 1:
                query_indices_in_this_range = numpy.where(
                    query_point_table[QUERY_TIME_COLUMN].values >=
                    query_to_model_times_table[
                        nwp_model_utils.MIN_QUERY_TIME_COLUMN].values[-1]
                )[0]
            else:
                query_indices_in_this_range = numpy.where(numpy.logical_and(
                    query_point_table[QUERY_TIME_COLUMN].values >=
                    query_to_model_times_table[
                        nwp_model_utils.MIN_QUERY_TIME_COLUMN].values[i],
                    query_point_table[QUERY_TIME_COLUMN].values <
                    query_to_model_times_table[
                        nwp_model_utils.MAX_QUERY_TIME_COLUMN].values[i]
                ))[0]

            (list_of_2d_grids, list_of_2d_grids_other_wind_component,
             missing_data
            ) = _read_nwp_for_interp_any_grid(
                init_times_unix_sec=init_times_unix_sec,
                query_to_model_times_row=query_to_model_times_table.iloc[[i]],
                field_name_grib1=field_names_grib1[j],
                field_name_other_wind_component_grib1=
                field_names_other_wind_component_grib1[j],
                list_of_model_grids=list_of_2d_grids,
                list_of_model_grids_other_wind_component=
                list_of_2d_grids_other_wind_component, model_name=model_name,
                top_grib_directory_name=top_grib_directory_name,
                wgrib_exe_name=wgrib_exe_name,
                wgrib2_exe_name=wgrib2_exe_name,
                raise_error_if_missing=raise_error_if_missing)

            if missing_data:
                continue

            list_of_spatial_interp_arrays = [numpy.array([])] * num_init_times
            list_of_sinterp_arrays_other_wind_component = (
                [numpy.array([])] * num_init_times)

            init_time_needed_indices = numpy.where(
                query_to_model_times_table[
                    nwp_model_utils.MODEL_TIMES_NEEDED_COLUMN
                ].values[i])[0]

            for t in init_time_needed_indices:
                this_grid_id = nwp_model_utils.dimensions_to_grid_id(
                    numpy.array(list_of_2d_grids[t].shape))
                this_grid_index = grid_ids.index(this_grid_id)

                list_of_spatial_interp_arrays[
                    t
                ] = interp_from_xy_grid_to_points(
                    input_matrix=list_of_2d_grids[t],
                    sorted_grid_point_x_metres=x_points_by_grid_metres[
                        this_grid_index],
                    sorted_grid_point_y_metres=y_points_by_grid_metres[
                        this_grid_index],
                    query_x_coords_metres=query_point_table_by_grid[
                        this_grid_index][QUERY_X_COLUMN].values[
                            query_indices_in_this_range],
                    query_y_coords_metres=query_point_table_by_grid[
                        this_grid_index][QUERY_Y_COLUMN].values[
                            query_indices_in_this_range],
                    method_string=spatial_interp_method_string,
                    spline_degree=spline_degree, extrapolate=True)

                if rotate_wind_flags[j]:
                    list_of_sinterp_arrays_other_wind_component[
                        t
                    ] = interp_from_xy_grid_to_points(
                        input_matrix=list_of_2d_grids_other_wind_component[t],
                        sorted_grid_point_x_metres=x_points_by_grid_metres[
                            this_grid_index],
                        sorted_grid_point_y_metres=y_points_by_grid_metres[
                            this_grid_index],
                        query_x_coords_metres=query_point_table_by_grid[
                            this_grid_index][QUERY_X_COLUMN].values[
                                query_indices_in_this_range],
                        query_y_coords_metres=query_point_table_by_grid[
                            this_grid_index][QUERY_Y_COLUMN].values[
                                query_indices_in_this_range],
                        method_string=spatial_interp_method_string,
                        spline_degree=spline_degree, extrapolate=True)

                    if grib_io.is_u_wind_field(field_names_grib1[j]):
                        (list_of_spatial_interp_arrays[t],
                         list_of_sinterp_arrays_other_wind_component[t]
                        ) = nwp_model_utils.rotate_winds_to_earth_relative(
                            u_winds_grid_relative_m_s01=
                            list_of_spatial_interp_arrays[t],
                            v_winds_grid_relative_m_s01=
                            list_of_sinterp_arrays_other_wind_component[t],
                            rotation_angle_cosines=
                            rotation_cosine_by_query_point[
                                query_indices_in_this_range],
                            rotation_angle_sines=rotation_sine_by_query_point[
                                query_indices_in_this_range])

                    if grib_io.is_v_wind_field(field_names_grib1[j]):
                        (list_of_sinterp_arrays_other_wind_component[t],
                         list_of_spatial_interp_arrays[t]
                        ) = nwp_model_utils.rotate_winds_to_earth_relative(
                            u_winds_grid_relative_m_s01=
                            list_of_sinterp_arrays_other_wind_component[t],
                            v_winds_grid_relative_m_s01=
                            list_of_spatial_interp_arrays[t],
                            rotation_angle_cosines=
                            rotation_cosine_by_query_point[
                                query_indices_in_this_range],
                            rotation_angle_sines=rotation_sine_by_query_point[
                                query_indices_in_this_range])

            spatial_interp_matrix_2d = _stack_1d_arrays_horizontally(
                [list_of_spatial_interp_arrays[t] for
                 t in init_time_needed_indices])

            if rotate_wind_flags[j]:
                sinterp_matrix_2d_other_wind_component = (
                    _stack_1d_arrays_horizontally(
                        [list_of_sinterp_arrays_other_wind_component[t] for
                         t in init_time_needed_indices]))

            (these_unique_query_times_unix_sec,
             these_query_times_orig_to_unique
            ) = numpy.unique(
                query_point_table[QUERY_TIME_COLUMN].values[
                    query_indices_in_this_range], return_inverse=True)

            for k in range(len(these_unique_query_times_unix_sec)):
                these_indices = numpy.where(
                    these_query_times_orig_to_unique == k)[0]
                query_indices_at_this_time = query_indices_in_this_range[
                    these_indices]

                these_interp_values = interp_in_time(
                    input_matrix=spatial_interp_matrix_2d[
                        query_indices_in_this_range, :],
                    sorted_input_times_unix_sec=init_times_unix_sec[
                        init_time_needed_indices],
                    query_times_unix_sec=these_unique_query_times_unix_sec[[k]],
                    method_string=temporal_interp_method_string,
                    extrapolate=False)
                interp_table[field_names[j]].values[
                    query_indices_at_this_time] = these_interp_values[:, 0]

                if other_wind_component_indices[j] != -1:
                    these_interp_values = interp_in_time(
                        input_matrix=sinterp_matrix_2d_other_wind_component[
                            query_indices_in_this_range, :],
                        sorted_input_times_unix_sec=init_times_unix_sec[
                            init_time_needed_indices],
                        query_times_unix_sec=
                        these_unique_query_times_unix_sec[[k]],
                        method_string=temporal_interp_method_string,
                        extrapolate=False)

                    this_field_name = field_names[
                        other_wind_component_indices[j]]
                    interp_table[this_field_name].values[
                        query_indices_at_this_time] = these_interp_values[:, 0]

        interp_done_by_field[j] = True
        if other_wind_component_indices[j] != -1:
            interp_done_by_field[other_wind_component_indices[j]] = True

    return interp_table
