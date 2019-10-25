"""Plots results of permutation test."""

import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.deep_learning import permutation
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import permutation_plotting

FIGURE_RESOLUTION_DPI = 300

INPUT_FILE_ARG_NAME = 'input_file_name'
NUM_PREDICTORS_ARG_NAME = 'num_predictors_to_plot'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file (will be read by `permutation.read_results`).')

NUM_PREDICTORS_HELP_STRING = (
    'Will plot only the `{0:s}` most important predictors in each figure.  To '
    'plot all predictors, leave this argument alone.'
).format(NUM_PREDICTORS_ARG_NAME)

CONFIDENCE_LEVEL_HELP_STRING = (
    'Confidence level for error bars (in range 0...1).')

OUTPUT_DIR_HELP_STRING = (
    'Path to output directory (figures will be saved here).')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_PREDICTORS_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_PREDICTORS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CONFIDENCE_LEVEL_ARG_NAME, type=float, required=False, default=0.95,
    help=CONFIDENCE_LEVEL_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(input_file_name, num_predictors_to_plot, confidence_level,
         output_dir_name):
    """Plots results of permutation test.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param num_predictors_to_plot: Same.
    :param confidence_level: Same.
    :param output_dir_name: Same.
    """

    if num_predictors_to_plot <= 0:
        num_predictors_to_plot = None

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    permutation_dict = permutation.read_results(input_file_name)

    figure_object, axes_object_matrix = plotting_utils.create_paneled_figure(
        num_rows=1, num_columns=2, shared_x_axis=False, shared_y_axis=True,
        keep_aspect_ratio=False)

    permutation_plotting.plot_single_pass_test(
        permutation_dict=permutation_dict, axes_object=axes_object_matrix[0, 0],
        num_predictors_to_plot=num_predictors_to_plot,
        plot_percent_increase=False, confidence_level=confidence_level)

    axes_object_matrix[0, 0].set_title('Single-pass test')

    permutation_plotting.plot_multipass_test(
        permutation_dict=permutation_dict, axes_object=axes_object_matrix[0, 1],
        num_predictors_to_plot=num_predictors_to_plot,
        plot_percent_increase=False, confidence_level=confidence_level)

    axes_object_matrix[0, 1].set_ylabel('')
    axes_object_matrix[0, 1].set_title('Multi-pass test')

    figure_file_name = '{0:s}/permutation_test_abs-values.jpg'.format(
        output_dir_name)

    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight')
    pyplot.close(figure_object)

    _, axes_object_matrix = plotting_utils.create_paneled_figure(
        num_rows=1, num_columns=2, shared_x_axis=False, shared_y_axis=True,
        keep_aspect_ratio=False)

    permutation_plotting.plot_single_pass_test(
        permutation_dict=permutation_dict, axes_object=axes_object_matrix[0, 0],
        num_predictors_to_plot=num_predictors_to_plot,
        plot_percent_increase=True, confidence_level=confidence_level)

    axes_object_matrix[0, 1].set_title('Single-pass test')

    permutation_plotting.plot_multipass_test(
        permutation_dict=permutation_dict, axes_object=axes_object_matrix[0, 1],
        num_predictors_to_plot=num_predictors_to_plot,
        plot_percent_increase=True, confidence_level=confidence_level)

    axes_object_matrix[0, 1].set_ylabel('')
    axes_object_matrix[0, 1].set_title('Multi-pass test')

    figure_file_name = '{0:s}/permutation_test_percentage.jpg'.format(
        output_dir_name)

    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight')
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        num_predictors_to_plot=getattr(
            INPUT_ARG_OBJECT, NUM_PREDICTORS_ARG_NAME),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
