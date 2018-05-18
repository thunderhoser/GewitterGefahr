"""General helper methods (ones that don't belong in another "utils" module)."""

import numpy
from gewittergefahr.gg_utils import error_checking


def split_array_by_nan(input_array):
    """Splits numpy array into list of contiguous subarrays without NaN.

    :param input_array: 1-D numpy array.
    :return: list_of_arrays: 1-D list of 1-D numpy arrays.  Each numpy array is
        without NaN.
    """

    error_checking.assert_is_real_numpy_array(input_array)
    error_checking.assert_is_numpy_array(input_array, num_dimensions=1)
    return [input_array[i] for i in
            numpy.ma.clump_unmasked(numpy.ma.masked_invalid(input_array))]
