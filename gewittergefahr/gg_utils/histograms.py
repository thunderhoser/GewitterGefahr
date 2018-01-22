"""Creation of histograms."""

import numpy
from gewittergefahr.gg_utils import error_checking


def create_histogram(input_values, num_bins, min_value, max_value):
    """Creates a histogram with uniform bin-spacing.

    N = number of input values
    K = number of bins

    :param input_values: length-N numpy array of input values (to be binned).
    :param num_bins: Number of bins.
    :param min_value: Minimum value to include in histogram.  Any input value <
        `min_value` will be assigned to the first bin.
    :param max_value: Maximum value to include in histogram.  Any input value >
        `max_value` will be assigned to the last bin.
    :return: input_to_bin_indices: length-N numpy array of bin indices.  If
        input_values[i] = j, the [i]th input value belongs in the [j]th bin.
    :return: num_examples_by_bin: length-K numpy array, where the [j]th value is
        the number of inputs assigned to the [j]th bin.
    """

    error_checking.assert_is_numpy_array_without_nan(input_values)
    error_checking.assert_is_numpy_array(input_values, num_dimensions=1)
    error_checking.assert_is_integer(num_bins)
    error_checking.assert_is_geq(num_bins, 2)
    error_checking.assert_is_greater(max_value, min_value)

    bin_cutoffs = numpy.linspace(min_value, max_value, num=num_bins + 1)
    input_to_bin_indices = numpy.digitize(
        input_values, bin_cutoffs, right=False) - 1
    input_to_bin_indices[input_to_bin_indices < 0] = 0
    input_to_bin_indices[input_to_bin_indices > num_bins - 1] = num_bins - 1

    num_examples_by_bin = numpy.full(num_bins, -1, dtype=int)
    for j in range(num_bins):
        num_examples_by_bin[j] = numpy.sum(input_to_bin_indices == j)

    return input_to_bin_indices, num_examples_by_bin
