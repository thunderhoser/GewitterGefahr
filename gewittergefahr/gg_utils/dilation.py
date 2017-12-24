"""Methods for dilation (the mathematical-morphology operation)."""

import numpy
from scipy.ndimage.filters import percentile_filter
from gewittergefahr.gg_utils import error_checking

DEFAULT_HALF_WIDTH = 2
TOLERANCE = 1e-6


def dilate_2d_matrix(input_matrix, percentile_level,
                     half_width_in_pixels=DEFAULT_HALF_WIDTH,
                     take_largest_absolute_value=False):
    """Dilates 2-D matrix.  NaN's are treated as zeros.

    M = number of rows
    N = number of columns

    :param input_matrix: M-by-N numpy array of input data.
    :param percentile_level: Percentile level (ranging from 0...100).  At each
        pixel [i, j], the [q]th percentile in the dilation window will be taken,
        where q = `percentile_level`.
    :param half_width_in_pixels: Half-width of dilation window.
    :param take_largest_absolute_value: Boolean flag.  If True, this method will
        perform two dilations: one with the [q]th percentile and one with the
        [100 - q]th percentile, where q = `percentile_level`.  At each pixel,
        this method will keep the greatest absolute value (but preserve its
        sign) between the two dilations.
    :return: output_matrix: M-by-N numpy array of dilated input values.
    """

    error_checking.assert_is_numpy_array(input_matrix, num_dimensions=2)
    error_checking.assert_is_real_numpy_array(input_matrix)
    error_checking.assert_is_geq(percentile_level, 0.)
    error_checking.assert_is_leq(percentile_level, 100.)
    error_checking.assert_is_integer(half_width_in_pixels)
    error_checking.assert_is_greater(half_width_in_pixels, 0)
    error_checking.assert_is_boolean(take_largest_absolute_value)

    width_in_pixels = 2 * half_width_in_pixels + 1
    input_matrix[numpy.isnan(input_matrix)] = 0.

    if take_largest_absolute_value:
        output_matrix_orig_percentile = percentile_filter(
            input_matrix, percentile=percentile_level, size=width_in_pixels,
            mode='constant', cval=0.)
        output_matrix_opposite_percentile = percentile_filter(
            input_matrix, percentile=100. - percentile_level,
            size=width_in_pixels, mode='constant', cval=0.)

        output_matrix = numpy.dstack((
            output_matrix_orig_percentile, output_matrix_opposite_percentile))
        max_index_matrix = numpy.argmax(numpy.absolute(output_matrix), axis=2)

        num_rows = output_matrix.shape[0]
        num_columns = output_matrix.shape[1]
        row_indices = numpy.linspace(0, num_rows - 1, num=num_rows, dtype=int)
        column_indices = numpy.linspace(
            0, num_columns - 1, num=num_columns, dtype=int)

        column_index_matrix, row_index_matrix = numpy.meshgrid(
            column_indices, row_indices)
        output_matrix = output_matrix[
            row_index_matrix, column_index_matrix, max_index_matrix]

    else:
        output_matrix = percentile_filter(
            input_matrix, percentile=percentile_level, size=width_in_pixels,
            mode='constant', cval=0.)

    output_matrix[numpy.absolute(output_matrix) < TOLERANCE] = numpy.nan
    return output_matrix
