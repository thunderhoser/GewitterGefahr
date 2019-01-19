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

MINIMUM_STRING = 'min'
MAXIMUM_STRING = 'max'
VALID_THRESHOLD_TYPE_STRINGS = [MINIMUM_STRING, MAXIMUM_STRING]

MAX_PERCENTILE_KEY = 'max_percentile_level'
THRESHOLD_VAR_KEY = 'threshold_var_index'
THRESHOLD_VALUE_KEY = 'threshold_value'
THRESHOLD_TYPE_KEY = 'threshold_type_string'


def _check_threshold_type(threshold_type_string):
    """Error-checks threshold type.

    :param threshold_type_string: See doc for `run_pmm`.
    :raises: ValueError: if
        `threshold_type_string not in VALID_THRESHOLD_TYPE_STRINGS`.
    """

    error_checking.assert_is_string(threshold_type_string)

    if threshold_type_string not in VALID_THRESHOLD_TYPE_STRINGS:
        error_string = (
            '\n{0:s}\nValid threshold types (listed above) do not include '
            '"{1:s}".'
        ).format(str(VALID_THRESHOLD_TYPE_STRINGS), threshold_type_string)

        raise ValueError(error_string)


def _run_pmm_one_variable(
        input_matrix, max_percentile_level=DEFAULT_MAX_PERCENTILE_LEVEL,
        threshold_value=None, threshold_type_string=None):
    """Applies PMM to one variable.

    E = number of examples (realizations over which to average)

    :param input_matrix: numpy array.  The first axis must have length E.  Other
        axes are assumed to be spatial dimensions.  Thus, input_matrix[i, ...]
        is the spatial field for the [i]th example.
    :param max_percentile_level: Maximum percentile.  No output value will
        exceed the [q]th percentile of `input_matrix`, where q =
        `max_percentile_level`.  Similarly, no output value will be less than
        the [100 - q]th percentile of `input_matrix`.
    :param threshold_value: At each grid point, this method will count the
        number of examples that meet this threshold.  If you do not care about
        these counts, leave this argument as None.
    :param threshold_type_string: Threshold type (must be accepted by
        `_check_threshold_type`).  If type is "min", then at each grid point,
        this method will count the number of examples with value >=
        `threshold_value`.
    :return: mean_field_matrix: numpy array of probability-matched means.  Will
        have the same dimensions as `input_matrix`, except without the first
        axis.  For example, if `input_matrix` is E x 32 x 32 x 12, this will be
        32 x 32 x 12.
    :return: threshold_count_matrix: numpy array of threshold counts.  Will have
        the same dimensions as `input_matrix`, except without the first axis.
        If no thresholding was done, this will be None.
    """

    use_threshold = not (
        threshold_value is None and threshold_type_string is None
    )

    if use_threshold:
        if threshold_type_string == MINIMUM_STRING:
            threshold_count_matrix = numpy.sum(
                input_matrix >= threshold_value, axis=0)
        else:
            threshold_count_matrix = numpy.sum(
                input_matrix <= threshold_value, axis=0)
    else:
        threshold_count_matrix = None

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
    # mean_field_flattened = numpy.ravel(mean_field_matrix)
    #
    # # At each grid point, replace ensemble mean with the same percentile from
    # # pooled array.
    # pooled_value_percentiles = numpy.linspace(
    #     0, 100, num=len(pooled_values), dtype=float)
    # mean_value_percentiles = numpy.linspace(
    #     0, 100, num=len(mean_field_flattened), dtype=float)
    #
    # sort_indices = numpy.argsort(mean_field_flattened)
    # mean_value_percentiles = mean_value_percentiles[sort_indices]
    #
    # interp_object = interp1d(
    #     pooled_value_percentiles, pooled_values, kind='linear',
    #     bounds_error=True, assume_sorted=True)
    #
    # mean_field_flattened = interp_object(mean_value_percentiles)
    # mean_field_matrix = numpy.reshape(
    #     mean_field_flattened, mean_field_matrix.shape)
    #
    # return mean_field_matrix, threshold_count_matrix

    return mean_field_matrix


def check_input_args(
        input_matrix, max_percentile_level, threshold_var_index,
        threshold_value, threshold_type_string):
    """Error-checks input arguments.

    :param input_matrix: See doc for `run_pmm_many_variables`.
    :param max_percentile_level: Same.
    :param threshold_var_index: Same.
    :param threshold_value: Same.
    :param threshold_type_string: Same.
    :return: metadata_dict: Dictionary with the following keys.
    metadata_dict['max_percentile_level']: See input doc.
    metadata_dict['threshold_var_index']: See input doc.
    metadata_dict['threshold_value']: See input doc.
    metadata_dict['threshold_type_string']: See input doc.
    """

    error_checking.assert_is_numpy_array_without_nan(input_matrix)
    num_spatial_dimensions = len(input_matrix.shape) - 2
    error_checking.assert_is_geq(num_spatial_dimensions, 1)

    error_checking.assert_is_greater(max_percentile_level, 50.)
    error_checking.assert_is_leq(max_percentile_level, 100.)

    use_threshold = not (
        threshold_var_index is None
        and threshold_value is None
        and threshold_type_string is None
    )

    if use_threshold:
        _check_threshold_type(threshold_type_string)
        error_checking.assert_is_not_nan(threshold_value)

        error_checking.assert_is_integer(threshold_var_index)
        error_checking.assert_is_geq(threshold_var_index, 0)

        num_variables = input_matrix.shape[-1]
        error_checking.assert_is_less_than(threshold_var_index, num_variables)
    else:
        threshold_var_index = -1

    return {
        MAX_PERCENTILE_KEY: max_percentile_level,
        THRESHOLD_VAR_KEY: threshold_var_index,
        THRESHOLD_VALUE_KEY: threshold_value,
        THRESHOLD_TYPE_KEY: threshold_type_string
    }


def run_pmm_many_variables(
        input_matrix, max_percentile_level=DEFAULT_MAX_PERCENTILE_LEVEL,
        threshold_var_index=None, threshold_value=None,
        threshold_type_string=None):
    """Applies PMM to each variable separately.

    E = number of examples (realizations over which to average)
    V = number of variables

    :param input_matrix: numpy array.  The first axis must have length E, and the
        last axis must have length V.  Other axes are assumed to be spatial
        dimensions.  Thus, input_matrix[i, ..., j] is the spatial field for the
        [j]th variable and [i]th example.
    :param max_percentile_level: See doc for `_run_pmm_one_variable`.
    :param threshold_var_index: Determines variable to which threshold will be
        applied.  If `threshold_var_index = j`, threshold will be applied to
        [j]th variable.  See `_run_pmm_one_variable` for more about thresholds.
    :param threshold_value: See doc for `_run_pmm_one_variable`.
    :param threshold_type_string: Same.
    :return: mean_field_matrix: numpy array of probability-matched means.  Will
        have the same dimensions as `input_matrix`, except without the first
        axis.  For example, if `input_matrix` is E x 32 x 32 x 12 x V, this will
        be 32 x 32 x 12 x V.
    :return: threshold_count_matrix: numpy array of threshold counts.  Will have
        the same dimensions as `input_matrix`, except without the first and last
        axes.  If no thresholding was done, this will be None.
    """

    metadata_dict = check_input_args(
        input_matrix=input_matrix, max_percentile_level=max_percentile_level,
        threshold_var_index=threshold_var_index,
        threshold_value=threshold_value,
        threshold_type_string=threshold_type_string)

    max_percentile_level = metadata_dict[MAX_PERCENTILE_KEY]
    threshold_var_index = metadata_dict[THRESHOLD_VAR_KEY]
    threshold_value = metadata_dict[THRESHOLD_VALUE_KEY]
    threshold_type_string = metadata_dict[THRESHOLD_TYPE_KEY]

    num_variables = input_matrix.shape[-1]
    mean_field_matrix = numpy.full(input_matrix.shape[1:], numpy.nan)
    threshold_count_matrix = None

    for j in range(num_variables):
        if threshold_var_index == j:
            mean_field_matrix[..., j], threshold_count_matrix = (
                _run_pmm_one_variable(
                    input_matrix=input_matrix[..., j],
                    max_percentile_level=max_percentile_level,
                    threshold_value=threshold_value,
                    threshold_type_string=threshold_type_string)
            )
        else:
            mean_field_matrix[..., j] = _run_pmm_one_variable(
                input_matrix=input_matrix[..., j],
                max_percentile_level=max_percentile_level
            )[0]

    return mean_field_matrix, threshold_count_matrix
