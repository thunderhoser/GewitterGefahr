"""Handles Monte Carlo significance-testing."""

import numpy
from scipy.stats import percentileofscore
from gewittergefahr.gg_utils import prob_matched_means as pmm
from gewittergefahr.gg_utils import error_checking

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TRIAL_PMM_MATRICES_KEY = 'list_of_trial_pmm_matrices'
PERCENTILE_MATRICES_KEY = 'list_of_percentile_matrices'
P_VALUE_MATRICES_KEY = 'list_of_p_value_matrices'
MAX_PMM_PERCENTILE_KEY = 'max_pmm_percentile_level'
NUM_ITERATIONS_KEY = 'num_iterations'
BASELINE_FILE_KEY = 'baseline_file_name'


def _check_input_args(
        list_of_baseline_matrices, list_of_trial_matrices, num_iterations):
    """Error-checks input args for Monte Carlo test.

    :param list_of_baseline_matrices: See doc for `run_monte_carlo_test`.
    :param list_of_trial_matrices: Same.
    :param num_iterations: Same.
    :raises: ValueError: if number of baseline matrices (input tensors to model)
        != number of trial matrices.
    :raises: TypeError: if all "input matrices" are None.
    :return: num_examples_per_set: Number of examples in each set.
    """

    error_checking.assert_is_integer(num_iterations)
    error_checking.assert_is_geq(num_iterations, 100)

    num_baseline_matrices = len(list_of_baseline_matrices)
    num_trial_matrices = len(list_of_trial_matrices)

    if num_baseline_matrices != num_trial_matrices:
        error_string = (
            'Number of baseline matrices ({0:d}) should = number of trial '
            'matrices ({1:d}).'
        ).format(num_baseline_matrices, num_trial_matrices)

        raise ValueError(error_string)

    num_matrices = num_trial_matrices
    num_examples_per_set = None

    for i in range(num_matrices):
        if (list_of_baseline_matrices[i] is None
                and list_of_trial_matrices[i] is None):
            continue

        error_checking.assert_is_numpy_array(list_of_baseline_matrices[i])

        if num_examples_per_set is None:
            num_examples_per_set = list_of_baseline_matrices[i].shape[0]

        these_expected_dim = numpy.array(
            (num_examples_per_set,) + list_of_baseline_matrices[i].shape[1:],
            dtype=int
        )

        error_checking.assert_is_numpy_array(
            list_of_baseline_matrices[i], exact_dimensions=these_expected_dim)

        these_expected_dim = numpy.array(
            list_of_baseline_matrices[i].shape, dtype=int)

        error_checking.assert_is_numpy_array(
            list_of_trial_matrices[i], exact_dimensions=these_expected_dim)

    if num_examples_per_set is None:
        raise TypeError('All "input matrices" are None.')

    return num_examples_per_set


def check_output(monte_carlo_dict):
    """Error-checks output from Monte Carlo test.

    :param monte_carlo_dict: Dictionary created by `run_monte_carlo_test`.
    :raises: ValueError: if not (number of trial matrices = number of MIN
        matrices = number of MAX matrices).
    """

    error_checking.assert_is_greater(
        monte_carlo_dict[MAX_PMM_PERCENTILE_KEY], 50.
    )
    error_checking.assert_is_leq(monte_carlo_dict[MAX_PMM_PERCENTILE_KEY], 100.)
    error_checking.assert_is_integer(monte_carlo_dict[NUM_ITERATIONS_KEY])
    error_checking.assert_is_geq(monte_carlo_dict[NUM_ITERATIONS_KEY], 100)

    num_trial_matrices = len(monte_carlo_dict[TRIAL_PMM_MATRICES_KEY])
    num_p_value_matrices = len(monte_carlo_dict[P_VALUE_MATRICES_KEY])

    if not num_trial_matrices == num_p_value_matrices:
        error_string = (
            'Number of trial ({0:d}), and p-value ({1:d}) matrices should be '
            'the same.'
        ).format(num_trial_matrices, num_p_value_matrices)

        raise ValueError(error_string)

    num_matrices = num_trial_matrices

    for i in range(num_matrices):
        if (
                monte_carlo_dict[TRIAL_PMM_MATRICES_KEY][i] is None
                and monte_carlo_dict[P_VALUE_MATRICES_KEY][i] is None
        ):
            continue

        error_checking.assert_is_numpy_array(
            monte_carlo_dict[TRIAL_PMM_MATRICES_KEY][i]
        )

        these_expected_dim = numpy.array(
            monte_carlo_dict[TRIAL_PMM_MATRICES_KEY][i].shape, dtype=int
        )
        error_checking.assert_is_numpy_array(
            monte_carlo_dict[P_VALUE_MATRICES_KEY][i],
            exact_dimensions=these_expected_dim
        )


def run_monte_carlo_test(
        list_of_baseline_matrices, list_of_trial_matrices,
        max_pmm_percentile_level, num_iterations):
    """Runs Monte Carlo significance test.

    E = number of examples in each set
    T = number of matrices in each set

    :param list_of_baseline_matrices: length-T list of numpy arrays, where the
        first axis of each numpy array has length E.
    :param list_of_trial_matrices: See above.
    :param max_pmm_percentile_level: Max percentile for probability-matched
        means (PMM).  For more details, see documentation for
        `pmm.run_pmm_many_variables`.
    :param num_iterations: Number of Monte Carlo iterations.
    :return: monte_carlo_dict: Dictionary with the following keys.
    monte_carlo_dict['list_of_trial_pmm_matrices']: length-T list of numpy
        arrays, where list_of_trial_pmm_matrices[i] is the PMM composite over
        list_of_trial_matrices[i].  Thus, list_of_trial_pmm_matrices[i] has the
        same dimensions as list_of_trial_matrices[i], except without the first
        axis.
    monte_carlo_dict['list_of_p_value_matrices']: Same as above but containing
        p-values.
    monte_carlo_dict['list_of_percentile_matrices']: Same as above but
        containing percentiles.
    monte_carlo_dict['max_pmm_percentile_level']: Same as input.
    monte_carlo_dict['num_iterations']: Same as input.
    """

    num_examples_per_set = _check_input_args(
        list_of_baseline_matrices=list_of_baseline_matrices,
        list_of_trial_matrices=list_of_trial_matrices,
        num_iterations=num_iterations
    )

    example_indices = numpy.linspace(
        0, 2 * num_examples_per_set - 1, num=2 * num_examples_per_set,
        dtype=int
    )

    num_matrices = len(list_of_trial_matrices)
    list_of_shuffled_pmm_matrices = [None] * num_matrices
    print(SEPARATOR_STRING)

    for i in range(num_iterations):
        if numpy.mod(i, 25) == 0:
            print('Have run {0:d} of {1:d} Monte Carlo iterations...'.format(
                i, num_iterations
            ))

        these_indices = numpy.random.choice(
            example_indices, size=num_examples_per_set, replace=False)

        these_baseline_indices = these_indices[
            these_indices < num_examples_per_set]

        these_trial_indices = (
            these_indices[these_indices >= num_examples_per_set] -
            num_examples_per_set
        )

        for j in range(num_matrices):
            if list_of_trial_matrices[j] is None:
                continue

            this_shuffled_matrix = numpy.concatenate((
                list_of_baseline_matrices[j][these_baseline_indices, ...],
                list_of_trial_matrices[j][these_trial_indices, ...]
            ))

            this_shuffled_pmm_matrix = pmm.run_pmm_many_variables(
                input_matrix=this_shuffled_matrix,
                max_percentile_level=max_pmm_percentile_level)

            if list_of_shuffled_pmm_matrices[j] is None:
                dimensions = numpy.array(
                    (num_iterations,) + this_shuffled_pmm_matrix.shape,
                    dtype=int
                )
                list_of_shuffled_pmm_matrices[j] = numpy.full(
                    dimensions, numpy.nan)

            list_of_shuffled_pmm_matrices[j][i, ...] = this_shuffled_pmm_matrix

    print('Have run all {0:d} Monte Carlo iterations!'.format(num_iterations))
    print(SEPARATOR_STRING)

    list_of_trial_pmm_matrices = [None] * num_matrices
    list_of_percentile_matrices = [None] * num_matrices
    list_of_p_value_matrices = [None] * num_matrices

    for j in range(num_matrices):
        if list_of_trial_matrices[j] is None:
            continue

        list_of_trial_pmm_matrices[j] = pmm.run_pmm_many_variables(
            input_matrix=list_of_trial_matrices[j],
            max_percentile_level=max_pmm_percentile_level
        )

        trial_pmm_values = numpy.ravel(list_of_trial_pmm_matrices[j])
        percentiles = numpy.full(trial_pmm_values.shape, numpy.nan)

        for linear_index in range(len(trial_pmm_values)):
            if numpy.mod(linear_index, 25) == 0:
                print('Have computed {0:d} of {1:d} p-values...'.format(
                    linear_index, len(trial_pmm_values)
                ))

            these_indices = numpy.unravel_index(
                linear_index, list_of_trial_pmm_matrices[j].shape
            )
            shuffled_pmm_matrix = list_of_shuffled_pmm_matrices[j] + 0.

            for this_index in these_indices[::-1]:
                shuffled_pmm_matrix = shuffled_pmm_matrix[..., this_index]

            percentiles[linear_index] = 0.01 * percentileofscore(
                a=numpy.ravel(shuffled_pmm_matrix),
                score=trial_pmm_values[linear_index],
                kind='mean'
            )

        list_of_percentile_matrices[j] = numpy.reshape(
            percentiles, list_of_trial_pmm_matrices[j].shape
        )

        p_values = percentiles + 0.
        bottom_indices = numpy.where(p_values < 0.5)[0]
        top_indices = numpy.where(p_values >= 0.5)[0]
        p_values[bottom_indices] = 2 * p_values[bottom_indices]
        p_values[top_indices] = 2 * (1. - p_values[top_indices])

        print('Fraction of p-values <= 0.05: {0:.4f}'.format(
            numpy.mean(p_values <= 0.05)
        ))

        list_of_p_value_matrices[j] = numpy.reshape(
            p_values, list_of_trial_pmm_matrices[j].shape
        )

    return {
        TRIAL_PMM_MATRICES_KEY: list_of_trial_pmm_matrices,
        PERCENTILE_MATRICES_KEY: list_of_percentile_matrices,
        P_VALUE_MATRICES_KEY: list_of_p_value_matrices,
        MAX_PMM_PERCENTILE_KEY: max_pmm_percentile_level,
        NUM_ITERATIONS_KEY: num_iterations
    }


def find_sig_grid_points(p_value_matrix, max_false_discovery_rate):
    """Finds grid points with statistically significant values.

    This method implements Equation 3 of Wilks et al. (2016), which you can find
    here:

    https://journals.ametsoc.org/doi/full/10.1175/BAMS-D-15-00267.1

    :param p_value_matrix: numpy array of p-values.
    :param max_false_discovery_rate: Max false-discovery rate.
    :return: significance_matrix: numpy array of Boolean flags, with same shape
        as `p_value_matrix`.
    """

    error_checking.assert_is_greater(max_false_discovery_rate, 0.)
    error_checking.assert_is_less_than(max_false_discovery_rate, 1.)

    p_values_sorted = numpy.ravel(p_value_matrix)
    p_values_sorted = p_values_sorted[
        numpy.invert(numpy.isnan(p_values_sorted))
    ]
    p_values_sorted = numpy.sort(p_values_sorted)

    num_grid_cells = len(p_values_sorted)
    grid_cell_indices = numpy.linspace(
        1, num_grid_cells, num_grid_cells, dtype=float
    )

    significant_flags = (
        p_values_sorted <=
        (grid_cell_indices / num_grid_cells) * max_false_discovery_rate
    )
    significant_indices = numpy.where(significant_flags)[0]

    if len(significant_indices) == 0:
        max_p_value = 0.
    else:
        max_p_value = p_values_sorted[significant_indices[-1]]

    print('Max p-value for Wilks test = {0:.2e}'.format(max_p_value))

    significance_matrix = p_value_matrix <= max_p_value
    print('Number of significant grid points = {0:d}/{1:d}'.format(
        numpy.sum(significance_matrix),
        numpy.sum(numpy.invert(numpy.isnan(p_value_matrix)))
    ))

    return significance_matrix
