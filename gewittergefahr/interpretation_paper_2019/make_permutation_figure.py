"""Makes figure with results of all 4 permutation tests."""

import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.deep_learning import permutation_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import permutation_plotting

FIGURE_RESOLUTION_DPI = 300

FORWARD_FILE_ARG_NAME = 'forward_test_file_name'
BACKWARDS_FILE_ARG_NAME = 'backwards_test_file_name'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

FORWARD_FILE_HELP_STRING = (
    'Path to file with results of forward (both single- and multi-pass) '
    'permutation tests.  Will be read by `permutation_utils.read_results`.'
)

BACKWARDS_FILE_HELP_STRING = (
    'Same as `{0:s}` but for backwards tests.'
).format(FORWARD_FILE_ARG_NAME)

CONFIDENCE_LEVEL_HELP_STRING = (
    'Confidence level for error bars (in range 0...1).'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file (figure will be saved here).'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + FORWARD_FILE_ARG_NAME, type=str, required=True,
    help=FORWARD_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + BACKWARDS_FILE_ARG_NAME, type=str, required=True,
    help=BACKWARDS_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CONFIDENCE_LEVEL_ARG_NAME, type=float, required=False, default=0.95,
    help=CONFIDENCE_LEVEL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(forward_test_file_name, backwards_test_file_name, confidence_level,
         output_file_name):
    """Makes figure with results of all 4 permutation tests.

    This is effectively the main method.

    :param forward_test_file_name: See documentation at top of file.
    :param backwards_test_file_name: Same.
    :param confidence_level: Same.
    :param output_file_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    print('Reading data from: "{0:s}"...'.format(forward_test_file_name))
    forward_test_dict = permutation_utils.read_results(forward_test_file_name)

    print('Reading data from: "{0:s}"...'.format(backwards_test_file_name))
    backwards_test_dict = permutation_utils.read_results(
        backwards_test_file_name
    )

    figure_object, axes_object_matrix = plotting_utils.create_paneled_figure(
        num_rows=2, num_columns=2, shared_x_axis=False, shared_y_axis=True,
        keep_aspect_ratio=False, horizontal_spacing=0.1, vertical_spacing=0.05
    )

    permutation_plotting.plot_single_pass_test(
        permutation_dict=forward_test_dict,
        axes_object=axes_object_matrix[0, 0],
        plot_percent_increase=False, confidence_level=confidence_level
    )

    axes_object_matrix[0, 0].set_title('Forward single-pass test')
    axes_object_matrix[0, 0].set_xticks([])
    axes_object_matrix[0, 0].set_xlabel('')
    plotting_utils.label_axes(
        axes_object=axes_object_matrix[0, 0], label_string='(a)',
        x_coord_normalized=-0.01, y_coord_normalized=0.925
    )

    permutation_plotting.plot_multipass_test(
        permutation_dict=forward_test_dict,
        axes_object=axes_object_matrix[0, 1],
        plot_percent_increase=False, confidence_level=confidence_level
    )

    axes_object_matrix[0, 1].set_title('Forward multi-pass test')
    axes_object_matrix[0, 1].set_xticks([])
    axes_object_matrix[0, 1].set_xlabel('')
    axes_object_matrix[0, 1].set_ylabel('')
    plotting_utils.label_axes(
        axes_object=axes_object_matrix[0, 1], label_string='(b)',
        x_coord_normalized=1.15, y_coord_normalized=0.925
    )

    permutation_plotting.plot_single_pass_test(
        permutation_dict=backwards_test_dict,
        axes_object=axes_object_matrix[1, 0],
        plot_percent_increase=False, confidence_level=confidence_level
    )

    axes_object_matrix[1, 0].set_title('Backward single-pass test')
    axes_object_matrix[1, 0].set_xlabel('Area under ROC curve')
    plotting_utils.label_axes(
        axes_object=axes_object_matrix[1, 0], label_string='(c)',
        x_coord_normalized=-0.01, y_coord_normalized=0.925
    )

    permutation_plotting.plot_multipass_test(
        permutation_dict=backwards_test_dict,
        axes_object=axes_object_matrix[1, 1],
        plot_percent_increase=False, confidence_level=confidence_level
    )

    axes_object_matrix[1, 1].set_title('Backward multi-pass test')
    axes_object_matrix[1, 1].set_xlabel('Area under ROC curve')
    axes_object_matrix[1, 1].set_ylabel('')
    plotting_utils.label_axes(
        axes_object=axes_object_matrix[1, 1], label_string='(d)',
        x_coord_normalized=1.15, y_coord_normalized=0.925
    )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        forward_test_file_name=getattr(INPUT_ARG_OBJECT, FORWARD_FILE_ARG_NAME),
        backwards_test_file_name=getattr(
            INPUT_ARG_OBJECT, BACKWARDS_FILE_ARG_NAME
        ),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
