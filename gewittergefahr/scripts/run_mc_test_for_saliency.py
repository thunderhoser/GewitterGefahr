"""Runs Monte Carlo significance test for saliency maps."""

import argparse
import numpy
from gewittergefahr.gg_utils import prob_matched_means as pmm
from gewittergefahr.deep_learning import saliency_maps

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

BASELINE_FILE_ARG_NAME = 'baseline_saliency_file_name'
TRIAL_FILE_ARG_NAME = 'trial_saliency_file_name'
MAX_PERCENTILE_ARG_NAME = 'max_pmm_percentile_level'
NUM_ITERATIONS_ARG_NAME = 'num_iterations'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'

BASELINE_FILE_HELP_STRING = (
    'Path to file with saliency maps for baseline set.  Will be read by '
    '`saliency_maps.read_standard_file`.')

TRIAL_FILE_HELP_STRING = (
    'Path to file with saliency maps for trial set.  Will be read by '
    '`saliency_maps.read_standard_file`.  This script will find grid points in '
    'the trial set at which saliency is significantly different than the '
    'baseline set.')

MAX_PERCENTILE_HELP_STRING = (
    'Max percentile used in PMM procedure.  See '
    '`prob_matched_means.run_pmm_one_variable` for details.')

NUM_ITERATIONS_HELP_STRING = 'Number of Monte Carlo iterations.'

CONFIDENCE_LEVEL_HELP_STRING = (
    'Confidence level for statistical significance.  Grid point G in the trial '
    'set will be deemed statistically significant, iff its saliency is outside '
    'this confidence interval from the Monte Carlo iterations.  For example, if'
    ' the confidence level is 0.95, grid point G will be significant iff its '
    'saliency is outside the [2.5th, 97.5th] percentiles from Monte Carlo '
    'iterations.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + BASELINE_FILE_ARG_NAME, type=str, required=True,
    help=BASELINE_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TRIAL_FILE_ARG_NAME, type=str, required=True,
    help=TRIAL_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILE_ARG_NAME, type=float, required=False,
    default=pmm.DEFAULT_MAX_PERCENTILE_LEVEL, help=MAX_PERCENTILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_ITERATIONS_ARG_NAME, type=int, required=False, default=5000,
    help=NUM_ITERATIONS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CONFIDENCE_LEVEL_ARG_NAME, type=float, required=False, default=0.95,
    help=CONFIDENCE_LEVEL_HELP_STRING)


def _run(baseline_saliency_file_name, trial_saliency_file_name,
         max_pmm_percentile_level, num_iterations, confidence_level):
    """Runs Monte Carlo significance test for saliency maps.

    This is effectively the main method.

    :param baseline_saliency_file_name: See documentation at top of file.
    :param trial_saliency_file_name: Same.
    :param max_pmm_percentile_level: Same.
    :param num_iterations: Same.
    :param confidence_level: Same.
    :raises: ValueError: if number of baseline matrices (input tensors to model)
        != number of trial matrices.
    :raises: ValueError: if number of baseline examples != number of trial
        examples.
    """

    print('Reading baseline set from: "{0:s}"...'.format(
        baseline_saliency_file_name))

    baseline_saliency_dict = saliency_maps.read_standard_file(
        baseline_saliency_file_name)

    baseline_saliency_matrices = baseline_saliency_dict[
        saliency_maps.SALIENCY_MATRICES_KEY]

    print('Reading trial set from: "{0:s}"...'.format(trial_saliency_file_name))
    trial_saliency_dict = saliency_maps.read_standard_file(
        trial_saliency_file_name)
    trial_saliency_matrices = trial_saliency_dict[
        saliency_maps.SALIENCY_MATRICES_KEY]

    num_baseline_matrices = len(baseline_saliency_matrices)
    num_trial_matrices = len(trial_saliency_matrices)

    if num_baseline_matrices != num_trial_matrices:
        error_string = (
            'Number of baseline matrices ({0:d}) should = number of trial '
            'matrices ({1:d}).'
        ).format(num_baseline_matrices, num_trial_matrices)

        raise ValueError(error_string)

    num_baseline_examples = baseline_saliency_matrices[0].shape[0]
    num_trial_examples = trial_saliency_matrices[0].shape[0]

    if num_baseline_examples != num_trial_examples:
        error_string = (
            'Number of baseline examples ({0:d}) should = number of trial '
            'examples ({1:d}).'
        ).format(num_baseline_examples, num_trial_examples)

        raise ValueError(error_string)

    num_examples_per_set = num_baseline_examples
    example_indices = numpy.linspace(
        0, 2 * num_examples_per_set - 1, num=2 * num_examples_per_set,
        dtype=int)

    num_matrices = num_trial_matrices
    random_pmm_saliency_matrices = [None] * num_matrices
    print(SEPARATOR_STRING)

    for i in range(num_iterations):
        if numpy.mod(i, 25) == 0:
            print('Have run {0:d} of {1:d} Monte Carlo iterations...'.format(
                i, num_iterations
            ))

        these_indices = numpy.random.choice(
            example_indices, size=num_trial_examples, replace=False)

        these_baseline_indices = these_indices[
            these_indices < num_examples_per_set]

        these_trial_indices = (
            these_indices[these_indices >= num_examples_per_set] -
            num_examples_per_set
        )

        for j in range(num_matrices):
            this_saliency_matrix = numpy.concatenate((
                baseline_saliency_matrices[j][these_baseline_indices, ...],
                trial_saliency_matrices[j][these_trial_indices, ...]
            ))

            this_saliency_matrix = pmm.run_pmm_many_variables(
                input_matrix=this_saliency_matrix,
                max_percentile_level=max_pmm_percentile_level
            )[0]

            this_saliency_matrix = numpy.expand_dims(
                this_saliency_matrix, axis=0)

            if random_pmm_saliency_matrices[j] is None:
                random_pmm_saliency_matrices[j] = this_saliency_matrix + 0.
            else:
                random_pmm_saliency_matrices[j] = numpy.concatenate((
                    random_pmm_saliency_matrices[j], this_saliency_matrix
                ))

    print('Have run all {0:d} Monte Carlo iterations!'.format(num_iterations))
    print(SEPARATOR_STRING)

    min_saliency_matrices = [None] * num_matrices
    max_saliency_matrices = [None] * num_matrices
    trial_pmm_saliency_matrices = [None] * num_matrices

    for j in range(num_matrices):
        min_saliency_matrices[j] = numpy.percentile(
            a=random_pmm_saliency_matrices[j], q=50. * (1 - confidence_level),
            axis=0
        )

        max_saliency_matrices[j] = numpy.percentile(
            a=random_pmm_saliency_matrices[j], q=50. * (1 + confidence_level),
            axis=0
        )

        trial_pmm_saliency_matrices[j] = pmm.run_pmm_many_variables(
            input_matrix=trial_saliency_matrices[j],
            max_percentile_level=max_pmm_percentile_level
        )[0]


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        baseline_saliency_file_name=getattr(
            INPUT_ARG_OBJECT, BASELINE_FILE_ARG_NAME),
        trial_saliency_file_name=getattr(INPUT_ARG_OBJECT, TRIAL_FILE_ARG_NAME),
        max_pmm_percentile_level=getattr(
            INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME),
        num_iterations=getattr(INPUT_ARG_OBJECT, NUM_ITERATIONS_ARG_NAME),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME)
    )
