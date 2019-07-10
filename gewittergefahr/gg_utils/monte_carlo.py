"""Handles Monte Carlo significance-testing."""

import numpy
from gewittergefahr.gg_utils import prob_matched_means as pmm
from gewittergefahr.gg_utils import error_checking

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TRIAL_PMM_MATRICES_KEY = 'list_of_trial_pmm_matrices'
MIN_MATRICES_KEY = 'list_of_min_matrices'
MAX_MATRICES_KEY = 'list_of_max_matrices'
MAX_PMM_PERCENTILE_KEY = 'max_pmm_percentile_level'
NUM_ITERATIONS_KEY = 'num_iterations'
CONFIDENCE_LEVEL_KEY = 'confidence_level'
BASELINE_FILE_KEY = 'baseline_file_name'


def _check_input_args(
        list_of_baseline_matrices, list_of_trial_matrices, num_iterations,
        confidence_level):
    """Error-checks input args for Monte Carlo test.

    :param list_of_baseline_matrices: See doc for `run_monte_carlo_test`.
    :param list_of_trial_matrices: Same.
    :param num_iterations: Same.
    :param confidence_level: Same.
    :raises: ValueError: if number of baseline matrices (input tensors to model)
        != number of trial matrices.
    :raises: TypeError: if all "input matrices" are None.
    :return: num_examples_per_set: Number of examples in each set.
    """

    error_checking.assert_is_integer(num_iterations)
    error_checking.assert_is_geq(num_iterations, 100)
    error_checking.assert_is_geq(confidence_level, 0.)
    error_checking.assert_is_leq(confidence_level, 1.)

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
    error_checking.assert_is_geq(monte_carlo_dict[CONFIDENCE_LEVEL_KEY], 0.)
    error_checking.assert_is_leq(monte_carlo_dict[CONFIDENCE_LEVEL_KEY], 1.)

    num_trial_matrices = len(monte_carlo_dict[TRIAL_PMM_MATRICES_KEY])
    num_min_matrices = len(monte_carlo_dict[MIN_MATRICES_KEY])
    num_max_matrices = len(monte_carlo_dict[MAX_MATRICES_KEY])

    if not num_trial_matrices == num_min_matrices == num_max_matrices:
        error_string = (
            'Number of trial ({0:d}), MIN ({1:d}), and MAX ({2:d}) matrices '
            'should be the same.'
        ).format(num_trial_matrices, num_min_matrices, num_max_matrices)

        raise ValueError(error_string)

    num_matrices = num_trial_matrices

    for i in range(num_matrices):
        if (monte_carlo_dict[TRIAL_PMM_MATRICES_KEY][i] is None
                and monte_carlo_dict[MIN_MATRICES_KEY][i] is None
                and monte_carlo_dict[MAX_MATRICES_KEY][i] is None):
            continue

        error_checking.assert_is_numpy_array(
            monte_carlo_dict[TRIAL_PMM_MATRICES_KEY][i]
        )

        these_expected_dim = numpy.array(
            monte_carlo_dict[TRIAL_PMM_MATRICES_KEY][i].shape, dtype=int
        )

        error_checking.assert_is_numpy_array(
            monte_carlo_dict[MIN_MATRICES_KEY][i],
            exact_dimensions=these_expected_dim
        )

        error_checking.assert_is_numpy_array(
            monte_carlo_dict[MAX_MATRICES_KEY][i],
            exact_dimensions=these_expected_dim
        )


def run_monte_carlo_test(
        list_of_baseline_matrices, list_of_trial_matrices,
        max_pmm_percentile_level, num_iterations, confidence_level):
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
    :param confidence_level: Confidence level for statistical significance.
    :return: monte_carlo_dict: Dictionary with the following keys.
    monte_carlo_dict['list_of_trial_pmm_matrices']: length-T list of numpy
        arrays, where list_of_trial_pmm_matrices[i] is the PMM composite over
        list_of_trial_matrices[i].  Thus, list_of_trial_pmm_matrices[i] has the
        same dimensions as list_of_trial_matrices[i], except without the first
        axis.
    monte_carlo_dict['list_of_min_matrices']: length-T list of numpy arrays,
        where list_of_min_matrices[i] has the same dimensions as
        list_of_baseline_matrices[i], except without the first axis.  Each
        matrix defines MIN thresholds for stat significance.  In other words, if
        list_of_trial_pmm_matrices[i][j] < list_of_min_matrices[i][j],
        list_of_trial_pmm_matrices[i][j] is significantly different from the
        baseline.
    monte_carlo_dict['list_of_max_matrices']: Same but for MAX thresholds.
    monte_carlo_dict['max_pmm_percentile_level']: Same as input.
    monte_carlo_dict['num_iterations']: Same as input.
    monte_carlo_dict['confidence_level']: Same as input.
    """

    num_examples_per_set = _check_input_args(
        list_of_baseline_matrices=list_of_baseline_matrices,
        list_of_trial_matrices=list_of_trial_matrices,
        num_iterations=num_iterations, confidence_level=confidence_level)

    example_indices = numpy.linspace(
        0, 2 * num_examples_per_set - 1, num=2 * num_examples_per_set,
        dtype=int)

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
                max_percentile_level=max_pmm_percentile_level
            )[0]

            this_shuffled_pmm_matrix = numpy.expand_dims(
                this_shuffled_pmm_matrix, axis=0)

            if list_of_shuffled_pmm_matrices[j] is None:
                list_of_shuffled_pmm_matrices[j] = this_shuffled_pmm_matrix + 0.
            else:
                list_of_shuffled_pmm_matrices[j] = numpy.concatenate((
                    list_of_shuffled_pmm_matrices[j], this_shuffled_pmm_matrix
                ))

    print('Have run all {0:d} Monte Carlo iterations!'.format(num_iterations))
    print(SEPARATOR_STRING)

    list_of_min_matrices = [None] * num_matrices
    list_of_max_matrices = [None] * num_matrices
    list_of_trial_pmm_matrices = [None] * num_matrices

    for j in range(num_matrices):
        if list_of_trial_matrices[j] is None:
            continue

        list_of_min_matrices[j] = numpy.percentile(
            a=list_of_shuffled_pmm_matrices[j], q=50. * (1 - confidence_level),
            axis=0
        )

        list_of_max_matrices[j] = numpy.percentile(
            a=list_of_shuffled_pmm_matrices[j], q=50. * (1 + confidence_level),
            axis=0
        )

        list_of_trial_pmm_matrices[j] = pmm.run_pmm_many_variables(
            input_matrix=list_of_trial_matrices[j],
            max_percentile_level=max_pmm_percentile_level
        )[0]

        this_num_low_significant = numpy.sum(
            list_of_trial_pmm_matrices[j] < list_of_min_matrices[j]
        )
        this_num_high_significant = numpy.sum(
            list_of_trial_pmm_matrices[j] > list_of_max_matrices[j]
        )

        print((
            'Number of elements in {0:d}th matrix = {1:d} ... num significant '
            'on low end = {2:d} ... num significant on high end = {3:d}'
        ).format(
            j + 1, list_of_trial_pmm_matrices[j].size, this_num_low_significant,
            this_num_high_significant
        ))

    return {
        TRIAL_PMM_MATRICES_KEY: list_of_trial_pmm_matrices,
        MIN_MATRICES_KEY: list_of_min_matrices,
        MAX_MATRICES_KEY: list_of_max_matrices,
        MAX_PMM_PERCENTILE_KEY: max_pmm_percentile_level,
        NUM_ITERATIONS_KEY: num_iterations,
        CONFIDENCE_LEVEL_KEY: confidence_level
    }
