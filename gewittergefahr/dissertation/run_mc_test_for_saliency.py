"""Runs Monte Carlo test for saliency maps.

This test ensures that actual and dummy saliency maps are significantly
different.  The dummy maps may be produced by one of three "sanity checks":
the edge-detector test, data-randomization test, or
model-parameter-randomization test.
"""

import pickle
import argparse
import numpy
import scipy.stats
from gewittergefahr.gg_utils import general_utils
from gewittergefahr.gg_utils import monte_carlo
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.deep_learning import saliency_maps

ACTUAL_FILE_ARG_NAME = 'actual_saliency_file_name'
DUMMY_FILE_ARG_NAME = 'dummy_saliency_file_name'
SMOOTHING_RADIUS_ARG_NAME = 'smoothing_radius_grid_cells'
MAX_PERCENTILE_ARG_NAME = 'max_pmm_percentile_level'
NUM_ITERATIONS_ARG_NAME = 'num_iterations'
COMPARE_RANKED_ARG_NAME = 'compare_ranked_values'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

ACTUAL_FILE_HELP_STRING = (
    'Path to file with actual saliency maps (will be read by '
    '`saliency.read_file`).'
)
DUMMY_FILE_HELP_STRING = (
    'Path to file with dummy saliency maps, from edge-detector test or model-'
    'parameter-randomization test (will be read by `saliency.read_file`).'
)
SMOOTHING_RADIUS_HELP_STRING = (
    'e-folding radius for Gaussian smoother (num grid cells).  If you do not '
    'want to smooth saliency maps, make this negative.'
)
MAX_PERCENTILE_HELP_STRING = (
    'Max percentile level for probability-matched means (PMM).'
)
NUM_ITERATIONS_HELP_STRING = 'Number of iterations for Monte Carlo test.'
COMPARE_RANKED_HELP_STRING = (
    'Boolean flag.  If 1 (0), will compare spatially ranked (raw) class '
    'activations.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file (will be written by `_write_results`).'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + ACTUAL_FILE_ARG_NAME, type=str, required=True,
    help=ACTUAL_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DUMMY_FILE_ARG_NAME, type=str, required=True,
    help=DUMMY_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SMOOTHING_RADIUS_ARG_NAME, type=float, required=False,
    default=-1, help=SMOOTHING_RADIUS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILE_ARG_NAME, type=float, required=False,
    default=99, help=MAX_PERCENTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_ITERATIONS_ARG_NAME, type=int, required=False, default=20000,
    help=NUM_ITERATIONS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + COMPARE_RANKED_ARG_NAME, type=int, required=False, default=1,
    help=COMPARE_RANKED_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _write_results(monte_carlo_dict, pickle_file_name):
    """Writes results to Pickle file.

    :param monte_carlo_dict: Dictionary created by
        `monte_carlo.run_monte_carlo_test`.
    :param pickle_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(monte_carlo_dict, pickle_file_handle)
    pickle_file_handle.close()


def _smooth_maps(saliency_matrices, smoothing_radius_grid_cells):
    """Smooths saliency maps via Gaussian filter.

    T = number of input tensors to the model

    :param saliency_matrices: length-T list of numpy arrays.
    :param smoothing_radius_grid_cells: e-folding radius (number of grid cells).
    :return: saliency_matrices: Smoothed version of input.
    """

    print((
        'Smoothing saliency maps with Gaussian filter (e-folding radius of '
        '{0:.1f} grid cells)...'
    ).format(
        smoothing_radius_grid_cells
    ))

    num_matrices = len(saliency_matrices)
    num_examples = saliency_matrices[0].shape[0]

    for j in range(num_matrices):
        this_num_channels = saliency_matrices[j].shape[-1]

        for i in range(num_examples):
            for k in range(this_num_channels):
                saliency_matrices[j][i, ..., k] = (
                    general_utils.apply_gaussian_filter(
                        input_matrix=saliency_matrices[j][i, ..., k],
                        e_folding_radius_grid_cells=smoothing_radius_grid_cells
                    )
                )

    return saliency_matrices


def _run(actual_file_name, dummy_file_name, smoothing_radius_grid_cells,
         max_pmm_percentile_level, num_iterations, compare_ranked_values,
         output_file_name):
    """Runs Monte Carlo test for saliency maps.

    This is effectively the main method.

    :param actual_file_name: See documentation at top of file.
    :param dummy_file_name: Same.
    :param smoothing_radius_grid_cells: Same.
    :param max_pmm_percentile_level: Same.
    :param num_iterations: Same.
    :param compare_ranked_values: Same.
    :param output_file_name: Same.
    """

    if smoothing_radius_grid_cells <= 0:
        smoothing_radius_grid_cells = None

    # Read saliency maps.
    print('Reading actual saliency maps from: "{0:s}"...'.format(
        actual_file_name
    ))
    actual_saliency_dict = saliency_maps.read_file(actual_file_name)[0]
    actual_saliency_matrices = (
        actual_saliency_dict[saliency_maps.SALIENCY_MATRICES_KEY]
    )

    print('Reading dummy saliency maps from: "{0:s}"...'.format(
        dummy_file_name
    ))
    dummy_saliency_dict = saliency_maps.read_file(dummy_file_name)[0]
    dummy_saliency_matrices = (
        dummy_saliency_dict[saliency_maps.SALIENCY_MATRICES_KEY]
    )

    # Ensure that the two files contain the same examples.
    assert (
        actual_saliency_dict[saliency_maps.FULL_STORM_IDS_KEY] ==
        dummy_saliency_dict[saliency_maps.FULL_STORM_IDS_KEY]
    )

    assert numpy.array_equal(
        actual_saliency_dict[saliency_maps.STORM_TIMES_KEY],
        dummy_saliency_dict[saliency_maps.STORM_TIMES_KEY]
    )

    if smoothing_radius_grid_cells is not None:
        actual_saliency_matrices = _smooth_maps(
            saliency_matrices=actual_saliency_matrices,
            smoothing_radius_grid_cells=smoothing_radius_grid_cells
        )
        dummy_saliency_matrices = _smooth_maps(
            saliency_matrices=dummy_saliency_matrices,
            smoothing_radius_grid_cells=smoothing_radius_grid_cells
        )

    # Convert saliency from absolute values to percentiles.
    if compare_ranked_values:
        num_matrices = len(actual_saliency_matrices)
        num_examples = actual_saliency_matrices[0].shape[0]

        for j in range(num_matrices):
            for i in range(num_examples):
                this_num_channels = actual_saliency_matrices[j].shape[-1]

                for k in range(this_num_channels):
                    this_flat_array = numpy.ravel(
                        actual_saliency_matrices[j][i, ..., k]
                    )
                    these_flat_ranks = (
                        scipy.stats.rankdata(this_flat_array, method='average')
                        / len(this_flat_array)
                    )
                    actual_saliency_matrices[j][i, ..., k] = numpy.reshape(
                        these_flat_ranks,
                        actual_saliency_matrices[j][i, ..., k].shape
                    )

                    this_flat_array = numpy.ravel(
                        dummy_saliency_matrices[j][i, ..., k]
                    )
                    these_flat_ranks = (
                        scipy.stats.rankdata(this_flat_array, method='average')
                        / len(this_flat_array)
                    )
                    dummy_saliency_matrices[j][i, ..., k] = numpy.reshape(
                        these_flat_ranks,
                        dummy_saliency_matrices[j][i, ..., k].shape
                    )

    # Do Monte Carlo test.
    monte_carlo_dict = monte_carlo.run_monte_carlo_test(
        list_of_baseline_matrices=dummy_saliency_matrices,
        list_of_trial_matrices=actual_saliency_matrices,
        max_pmm_percentile_level=max_pmm_percentile_level,
        num_iterations=num_iterations
    )

    print('Writing results of Monte Carlo test to file: "{0:s}"...'.format(
        output_file_name
    ))
    _write_results(
        monte_carlo_dict=monte_carlo_dict, pickle_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        actual_file_name=getattr(INPUT_ARG_OBJECT, ACTUAL_FILE_ARG_NAME),
        dummy_file_name=getattr(INPUT_ARG_OBJECT, DUMMY_FILE_ARG_NAME),
        smoothing_radius_grid_cells=getattr(
            INPUT_ARG_OBJECT, SMOOTHING_RADIUS_ARG_NAME),
        max_pmm_percentile_level=getattr(
            INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME),
        num_iterations=getattr(INPUT_ARG_OBJECT, NUM_ITERATIONS_ARG_NAME),
        compare_ranked_values=bool(
            getattr(INPUT_ARG_OBJECT, COMPARE_RANKED_ARG_NAME)
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME),
    )
