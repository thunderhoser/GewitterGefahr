"""General helper methods (ones that don't belong in another "utils" module)."""

import math
import numpy
from scipy.ndimage import median_filter
from gewittergefahr.gg_utils import error_checking


def find_nearest_value(sorted_input_values, test_value):
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
        math.fabs(test_value - sorted_input_values[nearest_index])
    )

    if subtract_one:
        nearest_index -= 1

    return sorted_input_values[nearest_index], nearest_index


def split_array_by_nan(input_array):
    """Splits numpy array into list of contiguous subarrays without NaN.

    :param input_array: 1-D numpy array.
    :return: list_of_arrays: 1-D list of 1-D numpy arrays.  Each numpy array is
        without NaN.
    """

    error_checking.assert_is_real_numpy_array(input_array)
    error_checking.assert_is_numpy_array(input_array, num_dimensions=1)

    return [
        input_array[i] for i in
        numpy.ma.clump_unmasked(numpy.ma.masked_invalid(input_array))
    ]


def apply_median_filter(input_matrix, num_cells_in_half_window):
    """Applies median filter to any-dimensional grid.

    :param input_matrix: numpy array with any dimensions.
    :param num_cells_in_half_window: Number of grid cells in half-window of
        smoothing filter.
    :return: output_matrix: numpy array after smoothing (same dimensions as
        input).
    """

    error_checking.assert_is_numpy_array(input_matrix)
    error_checking.assert_is_integer(num_cells_in_half_window)
    error_checking.assert_is_geq(num_cells_in_half_window, 1)

    return median_filter(
        input_matrix, size=2 * num_cells_in_half_window + 1, mode='nearest',
        origin=0)
