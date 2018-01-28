"""Methods for smoothing data over a grid."""

import numpy
from scipy.ndimage.filters import generic_filter
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import error_checking

TOLERANCE = 1e-6
EFOLDING_TO_CUTOFF_RADIUS_DEFAULT = 3.


def _get_distances_from_center_point(
        grid_spacing_x, grid_spacing_y, cutoff_radius):
    """For each grid point in smoothing window, computes distance from center.

    This method makes two key assumptions:

    [1] The grid is equidistant.
    [2] All input args have the same units (e.g., metres).

    m = number of grid rows used for smoothing at each point
    n = number of grid columns used for smoothing at each point

    :param grid_spacing_x: Spacing between adjacent grid points in x-direction.
    :param grid_spacing_y: Spacing between adjacent grid points in y-direction.
    :param cutoff_radius: Cutoff radius for smoother.
    :return: distance_from_center_matrix: m-by-n numpy array of distances from
        center point.  m and n are always odd, and the center element is always
        zero.
    """

    error_checking.assert_is_greater(grid_spacing_x, 0.)
    error_checking.assert_is_greater(grid_spacing_y, 0.)
    error_checking.assert_is_geq(cutoff_radius, grid_spacing_x)
    error_checking.assert_is_geq(cutoff_radius, grid_spacing_y)

    half_width_x_pixels = int(numpy.floor(cutoff_radius / grid_spacing_x))
    half_width_y_pixels = int(numpy.floor(cutoff_radius / grid_spacing_y))
    width_x_pixels = 2 * half_width_x_pixels + 1
    width_y_pixels = 2 * half_width_y_pixels + 1

    unique_relative_x_coords = numpy.linspace(
        -half_width_x_pixels * grid_spacing_x,
        half_width_x_pixels * grid_spacing_x, num=width_x_pixels)
    unique_relative_y_coords = numpy.linspace(
        -half_width_y_pixels * grid_spacing_y,
        half_width_y_pixels * grid_spacing_y, num=width_y_pixels)

    relative_x_matrix, relative_y_matrix = grids.xy_vectors_to_matrices(
        unique_relative_x_coords, unique_relative_y_coords)
    return numpy.sqrt(relative_x_matrix ** 2 + relative_y_matrix ** 2)


def _get_weights_for_gaussian(
        grid_spacing_x, grid_spacing_y, e_folding_radius, cutoff_radius):
    """Computes weights for Gaussian smoother.

    m = number of grid rows used for smoothing at each point
    n = number of grid columns used for smoothing at each point

    :param grid_spacing_x: Spacing between adjacent grid points in x-direction.
    :param grid_spacing_y: Spacing between adjacent grid points in y-direction.
    :param e_folding_radius: e-folding radius for Gaussian smoother.
    :param cutoff_radius: Cutoff radius for Gaussian smoother.
    :return: weight_matrix: m-by-n numpy array of weights to be applied at each
        point.
    """

    distance_from_center_matrix = _get_distances_from_center_point(
        grid_spacing_x, grid_spacing_y, cutoff_radius)

    weight_matrix = numpy.exp(
        -(distance_from_center_matrix / e_folding_radius) ** 2)
    return weight_matrix / numpy.sum(weight_matrix)


def _get_weights_for_cressman(grid_spacing_x, grid_spacing_y, cutoff_radius):
    """Computes weights for Gaussian smoother.

    m = number of grid rows used for smoothing at each point
    n = number of grid columns used for smoothing at each point

    :param grid_spacing_x: Spacing between adjacent grid points in x-direction.
    :param grid_spacing_y: Spacing between adjacent grid points in y-direction.
    :param cutoff_radius: Cutoff radius for Cressman smoother.
    :return: weight_matrix: m-by-n numpy array of weights to be applied at each
        point.
    """

    distance_from_center_matrix = _get_distances_from_center_point(
        grid_spacing_x, grid_spacing_y, cutoff_radius)

    weight_matrix = (
        (cutoff_radius ** 2 - distance_from_center_matrix ** 2) /
        (cutoff_radius ** 2 + distance_from_center_matrix ** 2))
    return weight_matrix / numpy.sum(weight_matrix)


def _apply_smoother_at_one_point(values, weight_vector):
    """Applies any kind of smoother at one grid point.

    P = number of points used to smooth each target point

    :param values: length-P numpy array of input values.
    :param weight_vector: length-P numpy array of corresponding weights.
    :return: smoothed_value: Smoothed value (to replace point at center of input
        array).
    """

    return numpy.sum(values * weight_vector)


def _apply_smoother_at_all_points(input_matrix, weight_matrix):
    """Applies any kind of smoother at all grid points.

    M = number of grid rows
    N = number of grid columns
    m = number of grid rows used for smoothing at each point
    n = number of grid columns used for smoothing at each point

    This method treats all NaN's as zero.

    :param input_matrix: M-by-N numpy array of input data.
    :param weight_matrix: m-by-n numpy array of weights.
    :return: output_matrix: M-by-N numpy array of smoothed input values.
    """

    weight_vector = numpy.reshape(weight_matrix, weight_matrix.size)
    input_matrix[numpy.isnan(input_matrix)] = 0.

    output_matrix = generic_filter(
        input_matrix, function=_apply_smoother_at_one_point,
        size=(weight_matrix.shape[0], weight_matrix.shape[1]), mode='constant',
        cval=0., extra_arguments=(weight_vector,))

    output_matrix[numpy.absolute(output_matrix) < TOLERANCE] = numpy.nan
    return output_matrix


def apply_gaussian(input_matrix, grid_spacing_x, grid_spacing_y,
                   e_folding_radius, cutoff_radius=None):
    """Applies Gaussian smoother.

    M = number of rows (unique grid-point y-coordinates)
    N = number of columns (unique grid-point x-coordinates)

    :param input_matrix: M-by-N numpy array of input data.
    :param grid_spacing_x: Spacing between adjacent grid points in x-direction
        (i.e., between adjacent columns).
    :param grid_spacing_y: Spacing between adjacent grid points in y-direction
        (i.e., between adjacent rows).
    :param e_folding_radius: e-folding radius for Gaussian smoother.
    :param cutoff_radius: Cutoff radius for Gaussian smoother.  Default is
        3 * e-folding radius.
    :return: output_matrix: M-by-N numpy array of smoothed input values.
    """

    # TODO(thunderhoser): For some reason, using
    # `scipy.ndimage.filters.gaussian_filter` is way faster.  I should change
    # this module to use said method.

    error_checking.assert_is_greater(e_folding_radius, 0.)
    if cutoff_radius is None:
        cutoff_radius = EFOLDING_TO_CUTOFF_RADIUS_DEFAULT * e_folding_radius
    error_checking.assert_is_geq(cutoff_radius, e_folding_radius)

    weight_matrix = _get_weights_for_gaussian(
        grid_spacing_x, grid_spacing_y, e_folding_radius, cutoff_radius)
    return _apply_smoother_at_all_points(input_matrix, weight_matrix)


def apply_cressman(input_matrix, grid_spacing_x, grid_spacing_y, cutoff_radius):
    """Applies Cressman smoother.

    M = number of rows (unique grid-point y-coordinates)
    N = number of columns (unique grid-point x-coordinates)

    :param input_matrix: M-by-N numpy array of input data.
    :param grid_spacing_x: Spacing between adjacent grid points in x-direction
        (i.e., between adjacent columns).
    :param grid_spacing_y: Spacing between adjacent grid points in y-direction
        (i.e., between adjacent rows).
    :param cutoff_radius: Cutoff radius for Cressman smoother.
    :return: output_matrix: M-by-N numpy array of smoothed input values.
    """

    weight_matrix = _get_weights_for_cressman(
        grid_spacing_x, grid_spacing_y, cutoff_radius)
    return _apply_smoother_at_all_points(input_matrix, weight_matrix)
