"""Interpolation methods."""

import numpy
import pandas
import scipy.interpolate
from gewittergefahr.gg_io import grib_io
from gewittergefahr.gg_io import nwp_model_io
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import error_checking

# TODO(thunderhoser): Allow interp_nwp_fields_from_xy_grid to deal with real
# forecasts, not only zero-hour analyses.

VERY_LARGE_INTEGER = int(1e10)

NEAREST_INTERP_METHOD = 'nearest'
SPLINE_INTERP_METHOD = 'spline'
SPATIAL_INTERP_METHODS = [NEAREST_INTERP_METHOD, SPLINE_INTERP_METHOD]

DEFAULT_SPLINE_DEGREE = 3
SMOOTHING_FACTOR_FOR_SPATIAL_INTERP = 0

PREVIOUS_INTERP_METHOD = 'previous'
NEXT_INTERP_METHOD = 'next'
LINEAR_INTERP_METHOD = 'linear'
SPLINE0_INTERP_METHOD = 'zero'
SPLINE1_INTERP_METHOD = 'slinear'
SPLINE2_INTERP_METHOD = 'quadratic'
SPLINE3_INTERP_METHOD = 'cubic'
TEMPORAL_INTERP_METHODS = [
    PREVIOUS_INTERP_METHOD, NEXT_INTERP_METHOD, NEAREST_INTERP_METHOD,
    LINEAR_INTERP_METHOD, SPLINE0_INTERP_METHOD, SPLINE1_INTERP_METHOD,
    SPLINE2_INTERP_METHOD, SPLINE3_INTERP_METHOD]

QUERY_TIME_COLUMN = 'unix_time_sec'
QUERY_LAT_COLUMN = 'latitude_deg'
QUERY_LNG_COLUMN = 'longitude_deg'
QUERY_X_COLUMN = 'x_coordinate_metres'
QUERY_Y_COLUMN = 'y_coordinate_metres'

FIELD_NAME_GRIB1_COLUMN = 'field_name_grib1'
ROTATE_WIND_COLUMN = 'rotate_wind'
FIELD_NAME_OTHER_WIND_COMPONENT_COLUMN = 'field_name_other_wind_component_grib1'
OTHER_WIND_COMPONENT_INDICES_COLUMN = 'other_wind_component_indices'

# interp_nwp_fields_from_xy_grid works only for zero-hour analyses, not real
# forecasts.
FORECAST_LEAD_TIME_HOURS = 0


def _interp_to_previous_time(input_matrix, input_times_unix_sec=None,
                             query_times_unix_sec=None):
    """Interpolates data in time, using the previous-neighbour method.

    :param input_matrix: See documentation for interp_in_time.
    :param input_times_unix_sec: See documentation for interp_in_time.
    :param query_times_unix_sec: See documentation for interp_in_time.
    :return: interp_matrix: See documentation for interp_in_time.
    """

    error_checking.assert_is_geq_numpy_array(
        query_times_unix_sec, numpy.min(input_times_unix_sec))

    num_query_times = len(query_times_unix_sec)
    list_of_interp_matrices = []

    for i in range(num_query_times):
        these_time_diffs_sec = query_times_unix_sec[i] - input_times_unix_sec
        these_time_diffs_sec[these_time_diffs_sec < 0] = VERY_LARGE_INTEGER
        this_previous_index = numpy.argmin(these_time_diffs_sec)
        list_of_interp_matrices.append(
            numpy.take(input_matrix, this_previous_index, axis=-1))

    return numpy.stack(list_of_interp_matrices, axis=-1)


def _interp_to_next_time(input_matrix, input_times_unix_sec=None,
                         query_times_unix_sec=None):
    """Interpolates data in time, using the next-neighbour method.

    :param input_matrix: See documentation for interp_in_time.
    :param input_times_unix_sec: See documentation for interp_in_time.
    :param query_times_unix_sec: See documentation for interp_in_time.
    :return: interp_matrix: See documentation for interp_in_time.
    """

    error_checking.assert_is_leq_numpy_array(
        query_times_unix_sec, numpy.max(input_times_unix_sec))

    num_query_times = len(query_times_unix_sec)
    list_of_interp_matrices = []

    for i in range(num_query_times):
        these_time_diffs_sec = input_times_unix_sec - query_times_unix_sec[i]
        these_time_diffs_sec[these_time_diffs_sec < 0] = VERY_LARGE_INTEGER
        this_next_index = numpy.argmin(these_time_diffs_sec)
        list_of_interp_matrices.append(
            numpy.take(input_matrix, this_next_index, axis=-1))

    return numpy.stack(list_of_interp_matrices, axis=-1)


def _nn_interp_from_xy_grid_to_points(input_matrix,
                                      sorted_grid_point_x_metres=None,
                                      sorted_grid_point_y_metres=None,
                                      query_x_metres=None, query_y_metres=None):
    """Performs nearest-neighbour interp from x-y grid to scattered points.

    :param input_matrix: See documentation for interp_from_xy_grid_to_points.
    :param sorted_grid_point_x_metres: See documentation for
        interp_from_xy_grid_to_points.
    :param sorted_grid_point_y_metres: See documentation for
        interp_from_xy_grid_to_points.
    :param query_x_metres: See documentation for interp_from_xy_grid_to_points.
    :param query_y_metres: See documentation for interp_from_xy_grid_to_points.
    :return: interp_values: See documentation for interp_from_xy_grid_to_points.
    """

    error_checking.assert_is_geq_numpy_array(
        query_x_metres, numpy.min(sorted_grid_point_x_metres))
    error_checking.assert_is_leq_numpy_array(
        query_x_metres, numpy.max(sorted_grid_point_x_metres))
    error_checking.assert_is_geq_numpy_array(
        query_y_metres, numpy.min(sorted_grid_point_y_metres))
    error_checking.assert_is_leq_numpy_array(
        query_y_metres, numpy.max(sorted_grid_point_y_metres))

    num_query_points = len(query_x_metres)
    interp_values = numpy.full(num_query_points, numpy.nan)

    for i in range(num_query_points):
        this_row = numpy.argmin(numpy.absolute(
            sorted_grid_point_y_metres - query_y_metres[i]))
        this_column = numpy.argmin(numpy.absolute(
            sorted_grid_point_x_metres - query_x_metres[i]))
        interp_values[i] = input_matrix[this_row, this_column]

    return interp_values


def _get_wind_rotation_metadata(field_names_grib1, model_name):
    """Gets metadata for rotation of winds from NWP model.

    F = number of fields to interpolate

    :param field_names_grib1: length-F list of field names in grib1 format.
        These will be used to read fields from grib files.
    :param model_name: Name of model.
    :return: wind_rotation_metadata_table: F-row pandas DataFrame with the
        following columns.
    wind_rotation_metadata_table.field_name_grib1: Same as input.
    wind_rotation_metadata_table.rotate_wind: Boolean flag.  If the [j]th row
        has rotate_wind = True, the [j]th field is a wind component that must be
        rotated.
    wind_rotation_metadata_table.field_name_other_wind_component_grib1: If the
        [j]th field is a wind component that must be rotated, this is the name
        (in grib1 format) of the other component in the wind vector.
    wind_rotation_metadata_table.other_wind_component_indices: If the [j]th
        field is a wind component that must be rotated and
        other_wind_component_indices[j] = k, this means the other component in
        the wind vector is the [k]th field.  If other_wind_component_indices[j]
        = -1, this means the other wind component was not included in
        field_names_grib1.
    """

    nwp_model_utils.check_model_name(model_name)

    num_fields = len(field_names_grib1)
    rotate_wind_flags = numpy.full(num_fields, False, dtype=bool)
    field_names_other_wind_component_grib1 = [None] * num_fields
    other_wind_component_indices = numpy.full(num_fields, -1, dtype=int)

    if model_name == nwp_model_utils.RAP_MODEL_NAME:
        for j in range(num_fields):
            rotate_wind_flags[j] = grib_io.is_wind_field(field_names_grib1[j])
            if not rotate_wind_flags[j]:
                continue

            field_names_other_wind_component_grib1[j] = (
                grib_io.field_name_switch_u_and_v(field_names_grib1[j]))
            if field_names_other_wind_component_grib1[
                    j] not in field_names_grib1:
                continue

            other_wind_component_indices[j] = field_names_grib1.index(
                field_names_other_wind_component_grib1[j])

    wind_rotation_metadata_dict = {
        FIELD_NAME_GRIB1_COLUMN: field_names_grib1,
        ROTATE_WIND_COLUMN: rotate_wind_flags,
        FIELD_NAME_OTHER_WIND_COMPONENT_COLUMN:
            field_names_other_wind_component_grib1,
        OTHER_WIND_COMPONENT_INDICES_COLUMN: other_wind_component_indices
    }
    return pandas.DataFrame.from_dict(wind_rotation_metadata_dict)


def _read_nwp_fields_for_interp(init_times_unix_sec=None,
                                query_to_model_times_row=None,
                                field_name_grib1=None,
                                field_name_other_wind_component_grib1=None,
                                list_of_model_grids=None,
                                list_of_model_grids_other_component=None,
                                model_name=None, grid_id=None, grib_type=None,
                                top_grib_directory_name=None):
    """Reads NWP fields needed for interpolation to range of query times.

    T = number of model-initialization times

    :param init_times_unix_sec: length-T of model-init times (Unix format).
    :param query_to_model_times_row: Single row of pandas DataFrame created by
        `nwp_model_utils.get_times_needed_for_interp`.
    :param field_name_grib1: Name of interpoland (grib1 format).
    :param field_name_other_wind_component_grib1: If interpoland is a wind
        component that must be rotated, this is the name of the other wind
        component in the same vector.
    :param list_of_model_grids: length-T list, where the [i]th element is either
        None or the model grid from the [i]th initialization time.
    :param list_of_model_grids_other_component: Same as above, but for other
        wind component.
    :param model_name: Name of model.
    :param grid_id: String ID for grid.
    :param grib_type: Type of grib files ("grib1" or "grib2") used by model.
    :param top_grib_directory_name: Name of top-level directory with grib files
        for the given model.
    :return: list_of_model_grids: Same as input, except that the [i]th element
        is filled (None) only if the [i]th time is (not) needed.
    :return: list_of_model_grids_other_component: Same as input, except that the
        [i]th element is filled (None) only if the [i]th time is (not) needed.
    """

    rotate_wind = field_name_other_wind_component_grib1 is not None

    init_time_needed_flags = query_to_model_times_row[
        nwp_model_utils.MODEL_TIMES_NEEDED_COLUMN].values[0]
    init_time_needed_indices = numpy.where(init_time_needed_flags)[0]
    init_time_obsolete_indices = numpy.where(
        numpy.invert(init_time_needed_flags))[0]

    for this_index in init_time_obsolete_indices:
        list_of_model_grids[this_index] = None
        if rotate_wind:
            list_of_model_grids_other_component[this_index] = None

    for this_index in init_time_needed_indices:
        this_grib_file_name = nwp_model_io.find_grib_file(
            init_times_unix_sec[this_index], lead_time_hours=0,
            model_name=model_name, grid_id=grid_id,
            top_directory_name=top_grib_directory_name,
            raise_error_if_missing=True)

        list_of_model_grids[this_index], _ = (
            nwp_model_io.read_field_from_grib_file(
                this_grib_file_name,
                init_time_unix_sec=init_times_unix_sec[this_index],
                lead_time_hours=0, model_name=model_name, grid_id=grid_id,
                top_single_field_dir_name=top_grib_directory_name,
                grib1_field_name=field_name_grib1,
                delete_single_field_file=True, raise_error_if_fails=True))

        if rotate_wind:
            list_of_model_grids_other_component[this_index], _ = (
                nwp_model_io.read_field_from_grib_file(
                    this_grib_file_name,
                    init_time_unix_sec=init_times_unix_sec[this_index],
                    lead_time_hours=0, model_name=model_name, grid_id=grid_id,
                    top_single_field_dir_name=top_grib_directory_name,
                    grib1_field_name=field_name_other_wind_component_grib1,
                    delete_single_field_file=True, raise_error_if_fails=True))

    if not rotate_wind:
        list_of_model_grids_other_component = None
    return list_of_model_grids, list_of_model_grids_other_component


def check_temporal_interp_method(temporal_interp_method):
    """Ensures that temporal-interpolation method is valid.

    :param temporal_interp_method: Interp method.
    :raises: ValueError: if `temporal_interp_method not in
        TEMPORAL_INTERP_METHODS`.
    """

    error_checking.assert_is_string(temporal_interp_method)
    if temporal_interp_method not in TEMPORAL_INTERP_METHODS:
        error_string = (
            '\n\n' + str(TEMPORAL_INTERP_METHODS) + '\n\nValid temporal-' +
            'interp methods (listed above) do not include the following: "' +
            temporal_interp_method + '"')
        raise ValueError(error_string)


def check_spatial_interp_method(spatial_interp_method):
    """Ensures that spatial-interpolation method is valid.

    :param spatial_interp_method: Interp method.
    :raises: ValueError: if `spatial_interp_method not in
        SPATIAL_INTERP_METHODS`.
    """

    error_checking.assert_is_string(spatial_interp_method)
    if spatial_interp_method not in SPATIAL_INTERP_METHODS:
        error_string = (
            '\n\n' + str(SPATIAL_INTERP_METHODS) + '\n\nValid spatial-' +
            'interp methods (listed above) do not include the following: "' +
            spatial_interp_method + '"')
        raise ValueError(error_string)


def interp_in_time(input_matrix, sorted_input_times_unix_sec=None,
                   query_times_unix_sec=None,
                   method_string=LINEAR_INTERP_METHOD, allow_extrap=False):
    """Interpolates data in time.

    D = number of dimensions (for both input_matrix and interp_matrix)
    N = number of input time steps
    P = number of query times

    :param input_matrix: D-dimensional numpy array of input data, where the last
        axis is time (length N).
    :param sorted_input_times_unix_sec: length-N numpy array of input times
        (Unix format).  Must be in ascending order.
    :param query_times_unix_sec: length-P numpy array of query times (Unix
        format).
    :param method_string: Interpolation method.  See documentation of
        `scipy.interpolate.interp1d` for valid options.
    :param allow_extrap: Boolean flag.  If True, this method may extrapolate
        outside the time range of the original data.  If False, this method may
        *not* extrapolate.  If False and query_times_unix_sec includes a value
        outside the range of sorted_input_times_unix_sec,
        `interp_object(query_times_unix_sec)` will raise an error.
    :return: interp_matrix: D-dimensional numpy array of interpolated values,
        where the last axis is time (length P).  The first (D - 1) dimensions
        have the same length as in input_matrix.
    """

    # error_checking.assert_is_numpy_array_without_nan(input_matrix)
    error_checking.assert_is_integer_numpy_array(sorted_input_times_unix_sec)
    error_checking.assert_is_numpy_array_without_nan(
        sorted_input_times_unix_sec)
    error_checking.assert_is_numpy_array(sorted_input_times_unix_sec,
                                         num_dimensions=1)
    error_checking.assert_is_boolean(allow_extrap)

    error_checking.assert_is_integer_numpy_array(query_times_unix_sec)
    error_checking.assert_is_numpy_array_without_nan(query_times_unix_sec)
    error_checking.assert_is_numpy_array(query_times_unix_sec, num_dimensions=1)

    check_temporal_interp_method(method_string)
    if method_string == PREVIOUS_INTERP_METHOD:
        return _interp_to_previous_time(
            input_matrix, input_times_unix_sec=sorted_input_times_unix_sec,
            query_times_unix_sec=query_times_unix_sec)

    if method_string == NEXT_INTERP_METHOD:
        return _interp_to_next_time(
            input_matrix, input_times_unix_sec=sorted_input_times_unix_sec,
            query_times_unix_sec=query_times_unix_sec)

    if allow_extrap:
        interp_object = scipy.interpolate.interp1d(
            sorted_input_times_unix_sec, input_matrix, kind=method_string,
            bounds_error=False, fill_value='extrapolate', assume_sorted=True)
    else:
        interp_object = scipy.interpolate.interp1d(
            sorted_input_times_unix_sec, input_matrix, kind=method_string,
            bounds_error=True, assume_sorted=True)

    return interp_object(query_times_unix_sec)


def interp_from_xy_grid_to_points(input_matrix, sorted_grid_point_x_metres=None,
                                  sorted_grid_point_y_metres=None,
                                  query_x_metres=None, query_y_metres=None,
                                  method_string=NEAREST_INTERP_METHOD,
                                  spline_degree=DEFAULT_SPLINE_DEGREE):
    """Interpolates data from x-y grid to scattered points.

    M = number of rows (unique y-coordinates of grid points)
    N = number of columns (unique x-coordinates of grid points)
    Q = number of query points

    :param input_matrix: M-by-N numpy array of input data.
    :param sorted_grid_point_x_metres: length-N numpy array with x-coordinates
        of grid points.  Must be in ascending order.
    :param sorted_grid_point_y_metres: length-M numpy array with y-coordinates
        of grid points.  Must be in ascending order.
    :param query_x_metres: length-Q numpy array with x-coords of query points.
    :param query_y_metres: length-Q numpy array with y-coords of query points.
    :param method_string: Interp method (either "nearest" or "spline").
    :param spline_degree: Polynomial degree for spline interpolation (1 for
        linear, 2 for quadratic, 3 for cubic).
    :return: interp_values: length-Q numpy array of interpolated values from
        input_matrix.
    :raises: ValueError: if method_string is neither "nearest" nor "spline".
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

    error_checking.assert_is_numpy_array_without_nan(query_x_metres)
    error_checking.assert_is_numpy_array(query_x_metres, num_dimensions=1)
    num_query_points = len(query_x_metres)

    error_checking.assert_is_numpy_array_without_nan(query_y_metres)
    error_checking.assert_is_numpy_array(
        query_y_metres, exact_dimensions=numpy.array([num_query_points]))

    check_spatial_interp_method(method_string)
    if method_string == NEAREST_INTERP_METHOD:
        return _nn_interp_from_xy_grid_to_points(
            input_matrix, sorted_grid_point_x_metres=sorted_grid_point_x_metres,
            sorted_grid_point_y_metres=sorted_grid_point_y_metres,
            query_x_metres=query_x_metres, query_y_metres=query_y_metres)

    interp_object = scipy.interpolate.RectBivariateSpline(
        sorted_grid_point_y_metres, sorted_grid_point_x_metres, input_matrix,
        kx=spline_degree, ky=spline_degree,
        s=SMOOTHING_FACTOR_FOR_SPATIAL_INTERP)

    return interp_object(query_y_metres, query_x_metres, grid=False)


def interp_nwp_fields_from_xy_grid(query_point_table, model_name=None,
                                   grid_id=None, field_names=None,
                                   field_names_grib1=None,
                                   top_grib_directory_name=None,
                                   temporal_interp_method=
                                   PREVIOUS_INTERP_METHOD,
                                   spatial_interp_method=NEAREST_INTERP_METHOD,
                                   spline_degree=DEFAULT_SPLINE_DEGREE):
    """Interpolates data from x-y grid in both space and time.

    Each query point consists of (latitude, longitude, time).

    F = number of fields to interpolate

    :param query_point_table: pandas DataFrame with the following columns.
    query_point_table.unix_time_sec: Time in Unix format.
    query_point_table.latitude_deg: Latitude (deg N).
    query_point_table.longitude_deg: Longitude (deg E).

    :param model_name: Name of model.
    :param grid_id: String ID for grid.
    :param field_names: length-F list of field names in GewitterGefahr format.
        These will become column names in the output table.
    :param field_names_grib1: length-F list of field names in grib1 format.
        These will be used to read fields from grib files.
    :param top_grib_directory_name: Name of top-level directory with grib files
        for given model.
    :param temporal_interp_method: See documentation for interp_in_time.
    :param spatial_interp_method: See documentation for
        interp_from_xy_grid_to_points.
    :param spline_degree: See documentation for interp_from_xy_grid_to_points.
    :return: interp_table: pandas DataFrame, where each column is one field and
        each row is one query point.  Column names are given by the input
        `field_names`.
    """

    # TODO(thunderhoser): I need to simplify this method, but I'm not sure how
    # yet.

    error_checking.assert_is_string_list(field_names)
    error_checking.assert_is_string_list(field_names_grib1)
    error_checking.assert_is_numpy_array(numpy.asarray(field_names),
                                         num_dimensions=1)

    num_fields = len(field_names)
    error_checking.assert_is_numpy_array(
        numpy.asarray(field_names_grib1),
        exact_dimensions=numpy.array([num_fields]))

    grib_type = nwp_model_utils.get_grib_type(model_name)
    _, init_time_step_hours = nwp_model_utils.get_time_steps(model_name)
    grid_point_x_metres, grid_point_y_metres = (
        nwp_model_utils.get_xy_grid_points(model_name, grid_id))

    query_x_metres, query_y_metres = nwp_model_utils.project_latlng_to_xy(
        query_point_table[QUERY_LAT_COLUMN].values,
        query_point_table[QUERY_LNG_COLUMN].values, model_name=model_name,
        grid_id=grid_id)
    argument_dict = {
        QUERY_X_COLUMN: query_x_metres, QUERY_Y_COLUMN: query_y_metres}
    query_point_table = query_point_table.assign(**argument_dict)

    num_query_points = len(query_point_table.index)
    nan_array = numpy.full(num_query_points, numpy.nan)
    interp_dict = {}
    for j in range(num_fields):
        interp_dict.update({field_names[j]: nan_array})
    interp_table = pandas.DataFrame.from_dict(interp_dict)

    wind_rotation_metadata_table = _get_wind_rotation_metadata(
        field_names_grib1, model_name)

    rotate_wind_flags = wind_rotation_metadata_table[ROTATE_WIND_COLUMN].values
    field_names_other_wind_component_grib1 = wind_rotation_metadata_table[
        FIELD_NAME_OTHER_WIND_COMPONENT_COLUMN].values
    other_wind_component_indices = wind_rotation_metadata_table[
        OTHER_WIND_COMPONENT_INDICES_COLUMN].values

    if numpy.any(rotate_wind_flags):
        rotation_angle_cosines, rotation_angle_sines = (
            nwp_model_utils.get_wind_rotation_angles(
                query_point_table[QUERY_LAT_COLUMN].values,
                query_point_table[QUERY_LNG_COLUMN].values,
                model_name=model_name))

    query_point_table.drop(
        [QUERY_LAT_COLUMN, QUERY_LNG_COLUMN], axis=1, inplace=True)

    init_times_unix_sec, query_to_model_times_table = (
        nwp_model_utils.get_times_needed_for_interp(
            query_times_unix_sec=query_point_table[QUERY_TIME_COLUMN].values,
            model_time_step_hours=init_time_step_hours,
            method_string=temporal_interp_method))

    num_init_times = len(init_times_unix_sec)
    num_query_time_ranges = len(query_to_model_times_table.index)
    interp_done_flags = numpy.full(num_fields, False, dtype=bool)

    for j in range(num_fields):
        if interp_done_flags[j]:
            continue

        list_of_2d_model_grids = [None] * num_init_times
        list_of_2d_grids_other_wind_component = [None] * num_init_times

        for i in range(num_query_time_ranges):
            if i == num_query_time_ranges - 1:
                in_range_flags = (
                    query_point_table[QUERY_TIME_COLUMN].values >=
                    query_to_model_times_table[
                        nwp_model_utils.MIN_QUERY_TIME_COLUMN].values[-1])
            else:
                in_range_flags = numpy.logical_and(
                    query_point_table[QUERY_TIME_COLUMN].values >=
                    query_to_model_times_table[
                        nwp_model_utils.MIN_QUERY_TIME_COLUMN].values[i],
                    query_point_table[QUERY_TIME_COLUMN].values <
                    query_to_model_times_table[
                        nwp_model_utils.MAX_QUERY_TIME_COLUMN].values[i]
                )

            in_range_indices = numpy.where(in_range_flags)[0]

            (list_of_2d_model_grids, list_of_2d_grids_other_wind_component) = (
                _read_nwp_fields_for_interp(
                    init_times_unix_sec=init_times_unix_sec,
                    query_to_model_times_row=
                    query_to_model_times_table.iloc[[i]],
                    field_name_grib1=field_names_grib1[j],
                    field_name_other_wind_component_grib1=
                    field_names_other_wind_component_grib1[j],
                    list_of_model_grids=list_of_2d_model_grids,
                    list_of_model_grids_other_component=
                    list_of_2d_grids_other_wind_component,
                    model_name=model_name, grid_id=grid_id, grib_type=grib_type,
                    top_grib_directory_name=top_grib_directory_name))

            init_time_needed_flags = query_to_model_times_table[
                nwp_model_utils.MODEL_TIMES_NEEDED_COLUMN].values[i]
            init_time_needed_indices = numpy.where(init_time_needed_flags)[0]

            model_grids_to_stack = [
                list_of_2d_model_grids[t] for t in init_time_needed_indices]
            model_grid_3d = numpy.stack(model_grids_to_stack, axis=-1)

            if rotate_wind_flags[j]:
                model_grids_to_stack = [
                    list_of_2d_grids_other_wind_component[t] for t in
                    init_time_needed_indices]
                model_grid_3d_other_wind_component = numpy.stack(
                    model_grids_to_stack, axis=-1)

            (these_unique_query_times_unix_sec,
             these_query_times_orig_to_unique) = numpy.unique(
                 query_point_table[QUERY_TIME_COLUMN].values[in_range_indices],
                 return_inverse=True)

            for k in range(len(these_unique_query_times_unix_sec)):
                interp_model_grid = interp_in_time(
                    model_grid_3d,
                    sorted_input_times_unix_sec=
                    init_times_unix_sec[init_time_needed_indices],
                    query_times_unix_sec=these_unique_query_times_unix_sec[[k]],
                    method_string=temporal_interp_method, allow_extrap=False)
                interp_model_grid = interp_model_grid[:, :, 0]

                if rotate_wind_flags[j]:
                    interp_model_grid_other_wind_component = interp_in_time(
                        model_grid_3d_other_wind_component,
                        sorted_input_times_unix_sec=
                        init_times_unix_sec[init_time_needed_indices],
                        query_times_unix_sec=
                        these_unique_query_times_unix_sec[[k]],
                        method_string=temporal_interp_method,
                        allow_extrap=False)
                    interp_model_grid_other_wind_component = (
                        interp_model_grid_other_wind_component[:, :, 0])

                these_query_indices = numpy.where(
                    these_query_times_orig_to_unique == k)[0]
                these_query_indices = in_range_indices[these_query_indices]

                these_interp_values = interp_from_xy_grid_to_points(
                    interp_model_grid,
                    sorted_grid_point_x_metres=grid_point_x_metres,
                    sorted_grid_point_y_metres=grid_point_y_metres,
                    query_x_metres=query_point_table[QUERY_X_COLUMN].values[
                        these_query_indices],
                    query_y_metres=query_point_table[QUERY_Y_COLUMN].values[
                        these_query_indices],
                    method_string=spatial_interp_method,
                    spline_degree=spline_degree)

                if rotate_wind_flags[j]:
                    these_interp_values_other_wind_component = (
                        interp_from_xy_grid_to_points(
                            interp_model_grid_other_wind_component,
                            sorted_grid_point_x_metres=grid_point_x_metres,
                            sorted_grid_point_y_metres=grid_point_y_metres,
                            query_x_metres=query_point_table[
                                QUERY_X_COLUMN].values[these_query_indices],
                            query_y_metres=query_point_table[
                                QUERY_Y_COLUMN].values[these_query_indices],
                            method_string=spatial_interp_method,
                            spline_degree=spline_degree))

                    if grib_io.is_u_wind_field(field_names_grib1[j]):
                        (these_interp_values,
                         these_interp_values_other_wind_component) = (
                             nwp_model_utils.rotate_winds(
                                 these_interp_values,
                                 these_interp_values_other_wind_component,
                                 rotation_angle_cosines=rotation_angle_cosines,
                                 rotation_angle_sines=rotation_angle_sines))
                    else:
                        (these_interp_values_other_wind_component,
                         these_interp_values) = nwp_model_utils.rotate_winds(
                             these_interp_values_other_wind_component,
                             these_interp_values,
                             rotation_angle_cosines=rotation_angle_cosines,
                             rotation_angle_sines=rotation_angle_sines)

                    if other_wind_component_indices[j] != -1:
                        interp_table[field_names[
                            other_wind_component_indices[j]]].values[
                                these_query_indices] = (
                                    these_interp_values_other_wind_component)
                        interp_done_flags[
                            other_wind_component_indices[j]] = True

                interp_table[field_names[j]].values[
                    these_query_indices] = these_interp_values

    return interp_table
