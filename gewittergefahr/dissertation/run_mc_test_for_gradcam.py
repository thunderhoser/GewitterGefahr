"""Runs Monte Carlo test for class-activation maps (CAM).

This test ensures that actual and dummy CAMs are significantly different.  The
dummy CAMs may be produced by one of three "sanity checks": the edge-detector
test, data-randomization test, or model-parameter-randomization test.
"""

import pickle
import argparse
import numpy
import scipy.stats
from gewittergefahr.gg_utils import general_utils
from gewittergefahr.gg_utils import monte_carlo
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.deep_learning import gradcam

ACTUAL_FILE_ARG_NAME = 'actual_gradcam_file_name'
DUMMY_FILE_ARG_NAME = 'dummy_gradcam_file_name'
SMOOTHING_RADIUS_ARG_NAME = 'smoothing_radius_grid_cells'
MAX_PERCENTILE_ARG_NAME = 'max_pmm_percentile_level'
NUM_ITERATIONS_ARG_NAME = 'num_iterations'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

ACTUAL_FILE_HELP_STRING = (
    'Path to file with actual class-activation maps (will be read by '
    '`gradcam.read_file`).'
)
DUMMY_FILE_HELP_STRING = (
    'Path to file with dummy class-activation maps (will be read by '
    '`gradcam.read_file`).'
)
SMOOTHING_RADIUS_HELP_STRING = (
    'e-folding radius for Gaussian smoother (num grid cells).  If you do not '
    'want to smooth class-activation maps, make this negative.'
)
MAX_PERCENTILE_HELP_STRING = (
    'Max percentile level for probability-matched means (PMM).'
)
NUM_ITERATIONS_HELP_STRING = 'Number of iterations for Monte Carlo test.'
CONFIDENCE_LEVEL_HELP_STRING = (
    'Confidence level (in range 0...1) for Monte Carlo test.'
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
    '--' + CONFIDENCE_LEVEL_ARG_NAME, type=float, required=False, default=0.95,
    help=CONFIDENCE_LEVEL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _write_results(monte_carlo_dict_unguided, monte_carlo_dict_guided,
                   pickle_file_name):
    """Writes results to Pickle file.

    :param monte_carlo_dict_unguided: Dictionary created by
        `monte_carlo.run_monte_carlo_test` for unguided CAMs.
    :param monte_carlo_dict_guided: Same but for guided CAMs.
    :param pickle_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(monte_carlo_dict_unguided, pickle_file_handle)
    pickle.dump(monte_carlo_dict_guided, pickle_file_handle)
    pickle_file_handle.close()


def _smooth_maps(cam_matrices, guided_cam_matrices,
                 smoothing_radius_grid_cells):
    """Smooths guided and unguided class-activation maps, using Gaussian filter.

    T = number of input tensors to the model

    :param cam_matrices: length-T list of numpy arrays with unguided class-
        activation maps (CAMs).
    :param guided_cam_matrices: length-T list of numpy arrays with guided CAMs.
    :param smoothing_radius_grid_cells: e-folding radius (number of grid cells).
    :return: cam_matrices: Smoothed version of input.
    :return: guided_cam_matrices: Smoothed version of input.
    """

    print((
        'Smoothing CAMs with Gaussian filter (e-folding radius of {0:.1f} grid '
        'cells)...'
    ).format(
        smoothing_radius_grid_cells
    ))

    num_matrices = len(cam_matrices)

    for j in range(num_matrices):
        if cam_matrices[j] is None:
            continue

        num_examples = cam_matrices[j].shape[0]
        this_num_channels = guided_cam_matrices[j].shape[-1]

        for i in range(num_examples):
            cam_matrices[j][i, ...] = general_utils.apply_gaussian_filter(
                input_matrix=cam_matrices[j][i, ...],
                e_folding_radius_grid_cells=smoothing_radius_grid_cells
            )

            for k in range(this_num_channels):
                guided_cam_matrices[j][i, ..., k] = (
                    general_utils.apply_gaussian_filter(
                        input_matrix=guided_cam_matrices[j][i, ..., k],
                        e_folding_radius_grid_cells=smoothing_radius_grid_cells
                    )
                )

    return cam_matrices, guided_cam_matrices


def _run(actual_file_name, dummy_file_name, smoothing_radius_grid_cells,
         max_pmm_percentile_level, num_iterations, confidence_level,
         output_file_name):
    """Runs Monte Carlo test for class-activation maps (CAM).

    This is effectively the main method.

    :param actual_file_name: See documentation at top of file.
    :param dummy_file_name: Same.
    :param smoothing_radius_grid_cells: Same.
    :param max_pmm_percentile_level: Same.
    :param num_iterations: Same.
    :param confidence_level: Same.
    :param output_file_name: Same.
    """

    if smoothing_radius_grid_cells <= 0:
        smoothing_radius_grid_cells = None

    # Read class-activation maps.
    print('Reading actual CAMs from: "{0:s}"...'.format(actual_file_name))
    actual_gradcam_dict = gradcam.read_file(actual_file_name)[0]
    actual_cam_matrices = actual_gradcam_dict[gradcam.CAM_MATRICES_KEY]
    actual_guided_cam_matrices = (
        actual_gradcam_dict[gradcam.GUIDED_CAM_MATRICES_KEY]
    )

    print('Reading dummy CAMsfrom: "{0:s}"...'.format(dummy_file_name))
    dummy_gradcam_dict = gradcam.read_file(dummy_file_name)[0]
    dummy_cam_matrices = dummy_gradcam_dict[gradcam.CAM_MATRICES_KEY]
    dummy_guided_cam_matrices = (
        dummy_gradcam_dict[gradcam.GUIDED_CAM_MATRICES_KEY]
    )

    # Ensure that the two files contain the same examples.
    assert (
        actual_gradcam_dict[gradcam.FULL_STORM_IDS_KEY] ==
        dummy_gradcam_dict[gradcam.FULL_STORM_IDS_KEY]
    )

    assert numpy.array_equal(
        actual_gradcam_dict[gradcam.STORM_TIMES_KEY],
        dummy_gradcam_dict[gradcam.STORM_TIMES_KEY]
    )

    if smoothing_radius_grid_cells is not None:
        actual_cam_matrices, actual_guided_cam_matrices = _smooth_maps(
            cam_matrices=actual_cam_matrices,
            guided_cam_matrices=actual_guided_cam_matrices,
            smoothing_radius_grid_cells=smoothing_radius_grid_cells
        )

        dummy_cam_matrices, dummy_guided_cam_matrices = _smooth_maps(
            cam_matrices=dummy_cam_matrices,
            guided_cam_matrices=dummy_guided_cam_matrices,
            smoothing_radius_grid_cells=smoothing_radius_grid_cells
        )

    # Convert from absolute values to percentiles.
    num_matrices = len(actual_cam_matrices)

    for j in range(num_matrices):
        if actual_cam_matrices[j] is None:
            continue

        num_examples = actual_cam_matrices[j].shape[0]
        this_num_channels = actual_guided_cam_matrices[j].shape[-1]

        for i in range(num_examples):
            this_flat_array = numpy.ravel(actual_cam_matrices[j][i, ...])
            these_flat_ranks = (
                scipy.stats.rankdata(this_flat_array, method='average') /
                len(this_flat_array)
            )
            actual_cam_matrices[j][i, ...] = numpy.reshape(
                these_flat_ranks, actual_cam_matrices[j][i, ...].shape
            )

            this_flat_array = numpy.ravel(dummy_cam_matrices[j][i, ...])
            these_flat_ranks = (
                scipy.stats.rankdata(this_flat_array, method='average') /
                len(this_flat_array)
            )
            dummy_cam_matrices[j][i, ...] = numpy.reshape(
                these_flat_ranks, dummy_cam_matrices[j][i, ...].shape
            )

            for k in range(this_num_channels):
                this_flat_array = numpy.ravel(
                    actual_guided_cam_matrices[j][i, ..., k]
                )
                these_flat_ranks = (
                    scipy.stats.rankdata(this_flat_array, method='average') /
                    len(this_flat_array)
                )
                actual_guided_cam_matrices[j][i, ..., k] = numpy.reshape(
                    these_flat_ranks,
                    actual_guided_cam_matrices[j][i, ..., k].shape
                )

                this_flat_array = numpy.ravel(
                    dummy_guided_cam_matrices[j][i, ..., k]
                )
                these_flat_ranks = (
                    scipy.stats.rankdata(this_flat_array, method='average') /
                    len(this_flat_array)
                )
                dummy_guided_cam_matrices[j][i, ..., k] = numpy.reshape(
                    these_flat_ranks,
                    dummy_guided_cam_matrices[j][i, ..., k].shape
                )

    # Do Monte Carlo test.
    actual_cam_matrices = [
        None if a is None else numpy.expand_dims(a, axis=-1)
        for a in actual_cam_matrices
    ]
    dummy_cam_matrices = [
        None if a is None else numpy.expand_dims(a, axis=-1)
        for a in dummy_cam_matrices
    ]

    monte_carlo_dict_unguided = monte_carlo.run_monte_carlo_test(
        list_of_baseline_matrices=dummy_cam_matrices,
        list_of_trial_matrices=actual_cam_matrices,
        max_pmm_percentile_level=max_pmm_percentile_level,
        num_iterations=num_iterations, confidence_level=confidence_level
    )

    monte_carlo_dict_unguided[monte_carlo.TRIAL_PMM_MATRICES_KEY] = [
        None if a is None else a[..., 0]
        for a in monte_carlo_dict_unguided[monte_carlo.TRIAL_PMM_MATRICES_KEY]
    ]
    monte_carlo_dict_unguided[monte_carlo.MIN_MATRICES_KEY] = [
        None if a is None else a[..., 0]
        for a in monte_carlo_dict_unguided[monte_carlo.MIN_MATRICES_KEY]
    ]
    monte_carlo_dict_unguided[monte_carlo.MAX_MATRICES_KEY] = [
        None if a is None else a[..., 0]
        for a in monte_carlo_dict_unguided[monte_carlo.MAX_MATRICES_KEY]
    ]

    monte_carlo_dict_guided = monte_carlo.run_monte_carlo_test(
        list_of_baseline_matrices=dummy_guided_cam_matrices,
        list_of_trial_matrices=actual_guided_cam_matrices,
        max_pmm_percentile_level=max_pmm_percentile_level,
        num_iterations=num_iterations, confidence_level=confidence_level
    )

    print('Writing results of Monte Carlo test to file: "{0:s}"...'.format(
        output_file_name
    ))
    _write_results(
        monte_carlo_dict_unguided=monte_carlo_dict_unguided,
        monte_carlo_dict_guided=monte_carlo_dict_guided,
        pickle_file_name=output_file_name
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
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME),
    )
