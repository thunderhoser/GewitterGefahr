"""Interpolation methods."""

import numpy
import scipy.interpolate
from gewittergefahr.gg_utils import error_checking

DEFAULT_TEMPORAL_INTERP_METHOD = 'linear'
NEAREST_INTERP_METHOD = 'nearest'
SPLINE_INTERP_METHOD = 'spline'
SPATIAL_INTERP_METHODS = [NEAREST_INTERP_METHOD, SPLINE_INTERP_METHOD]

DEFAULT_SPLINE_DEGREE = 3
SMOOTHING_FACTOR_FOR_SPATIAL_INTERP = 0


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


def interp_in_time(input_matrix, sorted_input_times_unix_sec=None,
                   query_times_unix_sec=None,
                   method_string=DEFAULT_TEMPORAL_INTERP_METHOD,
                   allow_extrap=False):
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

    error_checking.assert_is_numpy_array_without_nan(input_matrix)
    error_checking.assert_is_integer_numpy_array(sorted_input_times_unix_sec)
    error_checking.assert_is_numpy_array_without_nan(
        sorted_input_times_unix_sec)
    error_checking.assert_is_numpy_array(sorted_input_times_unix_sec,
                                         num_dimensions=1)
    error_checking.assert_is_boolean(allow_extrap)

    error_checking.assert_is_integer_numpy_array(query_times_unix_sec)
    error_checking.assert_is_numpy_array_without_nan(query_times_unix_sec)
    error_checking.assert_is_numpy_array(query_times_unix_sec, num_dimensions=1)

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

    error_checking.assert_is_string(method_string)
    if method_string not in SPATIAL_INTERP_METHODS:
        error_string = (
            '\n\n' + str(SPATIAL_INTERP_METHODS) + '\n\nValid spatial-interp ' +
            'methods (listed above) do not include the following: "' +
            method_string + '"')
        raise ValueError(error_string)

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
