"""Plots results of permutation test."""

import os.path
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
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file (will be read by `permutation.read_results`).')

NUM_PREDICTORS_HELP_STRING = (
    'Will plot only the `{0:s}` most important predictors in each figure.  To '
    'plot all predictors, leave this argument alone.'
).format(NUM_PREDICTORS_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Path to output directory (figures will be saved here).  If you want output'
    ' directory = input directory, leave this argument alone.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_PREDICTORS_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_PREDICTORS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=False, default='',
    help=OUTPUT_DIR_HELP_STRING)


def _run(input_file_name, num_predictors_to_plot, output_dir_name):
    """Plots results of permutation test.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param output_dir_name: Same.
    """

    if num_predictors_to_plot <= 0:
        num_predictors_to_plot = None

    if output_dir_name in ['', 'None']:
        output_dir_name = os.path.split(input_file_name)[0]

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    print('Reading permutation results from: "{0:s}"...'.format(input_file_name))
    permutation_dict = permutation.read_results(input_file_name)

    _, axes_object_matrix = plotting_utils.create_paneled_figure(
        num_rows=1, num_columns=2, shared_x_axis=False, shared_y_axis=True,
        keep_aspect_ratio=False)

    permutation_plotting.plot_breiman_results(
        permutation_dict=permutation_dict,
        axes_object=axes_object_matrix[0, 0],
        plot_percent_increase=False,
        num_predictors_to_plot=num_predictors_to_plot)

    axes_object_matrix[0, 0].set_xlabel('AUC')
    axes_object_matrix[0, 0].set_title('Single-pass')

    permutation_plotting.plot_lakshmanan_results(
        permutation_dict=permutation_dict,
        axes_object=axes_object_matrix[0, 1],
        plot_percent_increase=False, num_steps_to_plot=num_predictors_to_plot)

    axes_object_matrix[0, 1].set_xlabel('AUC')
    axes_object_matrix[0, 1].set_ylabel('')
    axes_object_matrix[0, 1].set_title('Multi-pass')

    pyplot.tight_layout()
    absolute_value_file_name = '{0:s}/permutation_absolute-values.jpg'.format(
        output_dir_name)

    print('Saving figure to file: "{0:s}"...'.format(absolute_value_file_name))
    pyplot.savefig(absolute_value_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()

    _, axes_object_matrix = plotting_utils.create_paneled_figure(
        num_rows=1, num_columns=2, shared_x_axis=False, shared_y_axis=True,
        keep_aspect_ratio=False)

    permutation_plotting.plot_breiman_results(
        permutation_dict=permutation_dict,
        axes_object=axes_object_matrix[0, 0],
        plot_percent_increase=True,
        num_predictors_to_plot=num_predictors_to_plot)

    axes_object_matrix[0, 0].set_title('Single-pass')

    permutation_plotting.plot_lakshmanan_results(
        permutation_dict=permutation_dict,
        axes_object=axes_object_matrix[0, 1],
        plot_percent_increase=True, num_steps_to_plot=num_predictors_to_plot)

    axes_object_matrix[0, 1].set_ylabel('')
    axes_object_matrix[0, 1].set_title('Multi-pass')

    pyplot.tight_layout()
    percentage_file_name = '{0:s}/permutation_percentage.jpg'.format(
        output_dir_name)

    print('Saving figure to file: "{0:s}"...'.format(percentage_file_name))
    pyplot.savefig(percentage_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        num_predictors_to_plot=getattr(
            INPUT_ARG_OBJECT, NUM_PREDICTORS_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
