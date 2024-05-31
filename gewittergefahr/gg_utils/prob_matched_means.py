"""Probability-matched means (PMM).

--- REFERENCE ---

Ebert, E.E., 2001: "Ability of a poor man's ensemble to predict the probability
and distribution of precipitation". Monthly Weather Review, 129 (10), 2461-2480,
https://doi.org/10.1175/1520-0493(2001)129%3C2461:AOAPMS%3E2.0.CO;2.
"""

import numpy
from scipy.interpolate import interp1d
from gewittergefahr.gg_utils import error_checking

DEFAULT_MAX_PERCENTILE_LEVEL = 99.
MAX_PERCENTILE_KEY = 'max_percentile_level'


def _run_pmm_one_variable(
        input_matrix, max_percentile_level=DEFAULT_MAX_PERCENTILE_LEVEL):
    """Applies PMM to one variable.

    E = number of examples (realizations over which to average)

    :param input_matrix: numpy array.  The first axis must have length E.  Other
        axes are assumed to be spatial dimensions.  Thus, input_matrix[i, ...]
        is the spatial field for the [i]th example.
    :param max_percentile_level: Maximum percentile.  No output value will
        exceed the [q]th percentile of `input_matrix`, where q =
        `max_percentile_level`.  Similarly, no output value will be less than
        the [100 - q]th percentile of `input_matrix`.
    :return: mean_field_matrix: numpy array of probability-matched means.  Will
        have the same dimensions as `input_matrix`, except without the first
        axis.  For example, if `input_matrix` is E x 32 x 32 x 12, this will be
        32 x 32 x 12.
    """

    # Pool values over all dimensions and remove extremes.
    pooled_values = numpy.ravel(input_matrix)
    pooled_values = numpy.sort(pooled_values)

    max_pooled_value = numpy.percentile(pooled_values, max_percentile_level)
    pooled_values = pooled_values[pooled_values <= max_pooled_value]

    min_pooled_value = numpy.percentile(
        pooled_values, 100 - max_percentile_level)
    pooled_values = pooled_values[pooled_values >= min_pooled_value]

    # Find ensemble mean at each grid point.
    mean_field_matrix = numpy.mean(input_matrix, axis=0)
    mean_field_flattened = numpy.ravel(mean_field_matrix)

    # At each grid point, replace ensemble mean with the same percentile from
    # pooled array.
    pooled_value_percentiles = numpy.linspace(
        0, 100, num=len(pooled_values), dtype=float)
    mean_value_percentiles = numpy.linspace(
        0, 100, num=len(mean_field_flattened), dtype=float)

    sort_indices = numpy.argsort(mean_field_flattened)
    unsort_indices = numpy.argsort(sort_indices)

    interp_object = interp1d(
        pooled_value_percentiles, pooled_values, kind='linear',
        bounds_error=True, assume_sorted=True)

    mean_field_flattened = interp_object(mean_value_percentiles)
    mean_field_flattened = mean_field_flattened[unsort_indices]
    mean_field_matrix = numpy.reshape(
        mean_field_flattened, mean_field_matrix.shape)

    return mean_field_matrix


def check_input_args(input_matrix, max_percentile_level):
    """Error-checks input arguments.

    :param input_matrix: See doc for `run_pmm_many_variables`.
    :param max_percentile_level: Same.
    :return: metadata_dict: Dictionary with the following keys.
    metadata_dict['max_percentile_level']: See input doc.
    """

    error_checking.assert_is_numpy_array_without_nan(input_matrix)
    num_spatial_dimensions = len(input_matrix.shape) - 2
    error_checking.assert_is_geq(num_spatial_dimensions, 0)

    error_checking.assert_is_geq(max_percentile_level, 90.)
    error_checking.assert_is_leq(max_percentile_level, 100.)

    return {
        MAX_PERCENTILE_KEY: max_percentile_level
    }


def run_pmm_many_variables(
        input_matrix, max_percentile_level=DEFAULT_MAX_PERCENTILE_LEVEL):
    """Applies PMM to each variable separately.

    E = number of examples (realizations over which to average)
    V = number of variables

    :param input_matrix: numpy array.  The first axis must have length E, and the
        last axis must have length V.  Other axes are assumed to be spatial
        dimensions.  Thus, input_matrix[i, ..., j] is the spatial field for the
        [j]th variable and [i]th example.
    :param max_percentile_level: See doc for `_run_pmm_one_variable`.
    :return: mean_field_matrix: numpy array of probability-matched means.  Will
        have the same dimensions as `input_matrix`, except without the first
        axis.  For example, if `input_matrix` is E x 32 x 32 x 12 x V, this will
        be 32 x 32 x 12 x V.
    """

    metadata_dict = check_input_args(
        input_matrix=input_matrix, max_percentile_level=max_percentile_level)

    max_percentile_level = metadata_dict[MAX_PERCENTILE_KEY]

    num_variables = input_matrix.shape[-1]
    mean_field_matrix = numpy.full(input_matrix.shape[1:], numpy.nan)

    for j in range(num_variables):
        mean_field_matrix[..., j] = _run_pmm_one_variable(
            input_matrix=input_matrix[..., j],
            max_percentile_level=max_percentile_level
        )

    return mean_field_matrix
