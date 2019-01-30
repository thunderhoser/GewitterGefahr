"""Plots results of permutation test."""

import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.deep_learning import permutation
from gewittergefahr.plotting import permutation_plotting

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 600

INPUT_FILE_ARG_NAME = 'input_file_name'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file (will be read by `permutation.read_results`).')

OUTPUT_FILE_HELP_STRING = 'Path to output file (figure will be saved here).'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _init_panels(num_rows, num_columns, figure_width_inches,
                 figure_height_inches):
    """Initializes multi-panel figure.

    :param num_rows: Number of rows of panels.
    :param num_columns: Number of columns of panels.
    :param figure_width_inches: Figure width.
    :param figure_height_inches: Figure height.
    :return: figure_object: Instance of `matplotlib.figure.Figure`.
    :return: axes_objects_2d_list: 2-D list, where axes_objects_2d_list[i][j] is
        a `matplotlib.axes._subplots.AxesSubplot` object for the [i]th row and
        [j]th column.
    """

    figure_object, axes_objects_2d_list = pyplot.subplots(
        num_rows, num_columns, sharex=False, sharey=True,
        figsize=(figure_width_inches, figure_height_inches)
    )

    if num_rows == num_columns == 1:
        axes_objects_2d_list = [[axes_objects_2d_list]]
    elif num_columns == 1:
        axes_objects_2d_list = [[a] for a in axes_objects_2d_list]
    elif num_rows == 1:
        axes_objects_2d_list = [axes_objects_2d_list]

    pyplot.subplots_adjust(
        left=0.02, bottom=0.02, right=0.98, top=0.95, hspace=0, wspace=0
    )

    return figure_object, axes_objects_2d_list


def _run(input_file_name, output_file_name):
    """Plots results of permutation test.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param output_file_name: Same.
    """

    print 'Reading permutation results from: "{0:s}"...'.format(input_file_name)
    permutation_dict = permutation.read_results(input_file_name)

    _, axes_objects_2d_list = _init_panels(
        num_rows=2, num_columns=2, figure_width_inches=FIGURE_WIDTH_INCHES,
        figure_height_inches=FIGURE_HEIGHT_INCHES)

    permutation_plotting.plot_breiman_results(
        permutation_dict=permutation_dict,
        axes_object=axes_objects_2d_list[0][0],
        plot_percent_increase=False, num_predictors_to_plot=None)

    axes_objects_2d_list[0][0].set_xlabel('')
    axes_objects_2d_list[0][0].set_title('Single-pass')

    permutation_plotting.plot_breiman_results(
        permutation_dict=permutation_dict,
        axes_object=axes_objects_2d_list[0][1],
        plot_percent_increase=True, num_predictors_to_plot=None)

    axes_objects_2d_list[0][1].set_xlabel('')
    axes_objects_2d_list[0][1].set_ylabel('')
    axes_objects_2d_list[0][1].set_title('Single-pass')

    permutation_plotting.plot_lakshmanan_results(
        permutation_dict=permutation_dict,
        axes_object=axes_objects_2d_list[1][0],
        plot_percent_increase=False, num_steps_to_plot=None)

    axes_objects_2d_list[1][0].set_xlabel('AUC')
    axes_objects_2d_list[1][0].set_title('Multi-pass')

    permutation_plotting.plot_lakshmanan_results(
        permutation_dict=permutation_dict,
        axes_object=axes_objects_2d_list[1][1],
        plot_percent_increase=True, num_steps_to_plot=None)

    axes_objects_2d_list[1][1].set_xlabel('Percent of original AUC')
    axes_objects_2d_list[1][1].set_ylabel('')
    axes_objects_2d_list[1][1].set_title('Multi-pass')

    pyplot.tight_layout()
    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    print 'Saving figure to file: "{0:s}"...'.format(output_file_name)
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
