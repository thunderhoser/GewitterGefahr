"""Stitches Monte Carlo files (different iters but same params) together."""

import copy
import pickle
import argparse
import numpy
from gewittergefahr.gg_utils import monte_carlo
from gewittergefahr.gg_utils import file_system_utils

TOLERANCE = 1e-6

INPUT_FILES_ARG_NAME = 'input_file_names'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_FILES_HELP_STRING = (
    'List of paths to input files, each containing results for a different set '
    'of iterations.  These files will be read by `_read_results`.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Results for all iterations will be written here by '
    '`_write_results`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=INPUT_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _read_results(pickle_file_name):
    """Reads results from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: monte_carlo_dict: Dictionary in format created by
        `monte_carlo.run_monte_carlo_test`.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    monte_carlo_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    return monte_carlo_dict


def _write_results(monte_carlo_dict, pickle_file_name):
    """Writes results to Pickle file.

    :param monte_carlo_dict: Dictionary in format created by
        `monte_carlo.run_monte_carlo_test`.
    :param pickle_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(monte_carlo_dict, pickle_file_handle)
    pickle_file_handle.close()


def _run(input_file_names, output_file_name):
    """Stitches Monte Carlo files (different iters but same params) together.

    This is effectively the main method.

    :param input_file_names: See documentation at top of file.
    :param output_file_name: Same.
    """

    num_input_files = len(input_file_names)
    monte_carlo_dict = dict()

    for i in range(num_input_files):
        print('Reading data from: "{0:s}"...'.format(input_file_names[i]))
        this_monte_carlo_dict = _read_results(input_file_names[i])

        if i == 0:
            monte_carlo_dict[monte_carlo.TRIAL_PMM_MATRICES_KEY] = (
                copy.deepcopy(
                    this_monte_carlo_dict[monte_carlo.TRIAL_PMM_MATRICES_KEY]
                )
            )
            monte_carlo_dict[monte_carlo.PERCENTILE_MATRICES_KEY] = (
                copy.deepcopy(
                    this_monte_carlo_dict[monte_carlo.PERCENTILE_MATRICES_KEY]
                )
            )
            monte_carlo_dict[monte_carlo.MAX_PMM_PERCENTILE_KEY] = (
                this_monte_carlo_dict[monte_carlo.MAX_PMM_PERCENTILE_KEY] + 0.
            )
            monte_carlo_dict[monte_carlo.NUM_ITERATIONS_KEY] = (
                this_monte_carlo_dict[monte_carlo.NUM_ITERATIONS_KEY] + 0
            )

        assert numpy.isclose(
            monte_carlo_dict[monte_carlo.MAX_PMM_PERCENTILE_KEY],
            this_monte_carlo_dict[monte_carlo.MAX_PMM_PERCENTILE_KEY],
            atol=TOLERANCE
        )
        assert (
            monte_carlo_dict[monte_carlo.NUM_ITERATIONS_KEY] ==
            this_monte_carlo_dict[monte_carlo.NUM_ITERATIONS_KEY]
        )

        if i == 0:
            continue

        num_matrices = len(
            monte_carlo_dict[monte_carlo.PERCENTILE_MATRICES_KEY]
        )

        for j in range(num_matrices):
            if monte_carlo_dict[monte_carlo.PERCENTILE_MATRICES_KEY][j] is None:
                continue

            monte_carlo_dict[monte_carlo.PERCENTILE_MATRICES_KEY][j] += (
                this_monte_carlo_dict[monte_carlo.PERCENTILE_MATRICES_KEY][j]
            )

    num_matrices = len(monte_carlo_dict[monte_carlo.PERCENTILE_MATRICES_KEY])
    p_value_matrices = [None] * num_matrices

    for j in range(num_matrices):
        if monte_carlo_dict[monte_carlo.PERCENTILE_MATRICES_KEY][j] is None:
            continue

        monte_carlo_dict[monte_carlo.PERCENTILE_MATRICES_KEY][j] = (
            monte_carlo_dict[monte_carlo.PERCENTILE_MATRICES_KEY][j] /
            num_input_files
        )

        assert numpy.all(
            monte_carlo_dict[monte_carlo.PERCENTILE_MATRICES_KEY][j] >= 0.
        )
        assert numpy.all(
            monte_carlo_dict[monte_carlo.PERCENTILE_MATRICES_KEY][j] <= 1.
        )

        p_values = numpy.ravel(
            monte_carlo_dict[monte_carlo.PERCENTILE_MATRICES_KEY][j]
        )
        bottom_indices = numpy.where(p_values < 0.5)[0]
        top_indices = numpy.where(p_values >= 0.5)[0]
        p_values[bottom_indices] = 2 * p_values[bottom_indices]
        p_values[top_indices] = 2 * (1. - p_values[top_indices])

        print('Fraction of p-values <= 0.05: {0:.4f}'.format(
            numpy.mean(p_values <= 0.05)
        ))

        p_value_matrices[j] = numpy.reshape(
            p_values,
            monte_carlo_dict[monte_carlo.PERCENTILE_MATRICES_KEY][j].shape
        )

    monte_carlo_dict[monte_carlo.P_VALUE_MATRICES_KEY] = p_value_matrices

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    _write_results(
        monte_carlo_dict=monte_carlo_dict, pickle_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_names=getattr(INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME),
    )
