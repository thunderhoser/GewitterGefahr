"""Interpolation methods."""

import scipy.interpolate

DEFAULT_TEMPORAL_INTERP_METHOD = 'linear'
DEFAULT_DEGREE_FOR_SPATIAL_INTERP = 3
SMOOTHING_FACTOR_FOR_SPATIAL_INTERP = 0


def interp_in_time(input_matrix, sorted_input_times_unix_sec=None,
                   query_times_unix_sec=None,
                   method_string=DEFAULT_TEMPORAL_INTERP_METHOD):
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
    :return: interp_matrix: D-dimensional numpy array of interpolated values,
        where the last axis is time (length P).  The first (D - 1) dimensions
        have the same length as in input_matrix.
    """

    interp_object = scipy.interpolate.interp1d(
        sorted_input_times_unix_sec, input_matrix, kind=method_string,
        bounds_error=True, assume_sorted=True)
    return interp_object(query_times_unix_sec)


def interp_from_xy_grid_to_points(
        input_matrix, sorted_grid_point_x_metres=None,
        sorted_grid_point_y_metres=None, query_x_metres=None,
        query_y_metres=None,
        polynomial_degree=DEFAULT_DEGREE_FOR_SPATIAL_INTERP):
    """Interpolates data from x-y grid to scattered points.

    M = number of grid rows (unique y-coordinates of grid points)
    N = number of grid columns (unique x-coordinates of grid points)
    P = number of query points

    :param input_matrix: M-by-N numpy array of input data.
    :param sorted_grid_point_x_metres: length-N numpy array with x-coordinates
        of grid points.  Must be in ascending order.
    :param sorted_grid_point_y_metres: length-M numpy array with y-coordinates
        of grid points.  Must be in ascending order.
    :param query_x_metres: length-P numpy array with x-coordinates of query
        points.
    :param query_y_metres: length-P numpy array with y-coordinates of query
        points.
    :param polynomial_degree: Polynomial degree for interpolation.  1 for
        linear, 2 for quadratic, 3 for cubic.
    :return: interp_values: length-P numpy array with data interpolated to query
        points.
    """

    interp_object = scipy.interpolate.RectBivariateSpline(
        sorted_grid_point_y_metres, sorted_grid_point_x_metres, input_matrix,
        kx=polynomial_degree, ky=polynomial_degree,
        s=SMOOTHING_FACTOR_FOR_SPATIAL_INTERP)

    return interp_object(query_y_metres, query_x_metres, grid=False)
