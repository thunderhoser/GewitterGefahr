"""Plots Laplacian kernel used for edge-detector test."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils

THIS_FIRST_MATRIX = numpy.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
])

THIS_SECOND_MATRIX = numpy.array([
    [0, 1, 0],
    [1, -6, 1],
    [0, 1, 0]
])

KERNEL_MATRIX_3D = numpy.stack(
    (THIS_FIRST_MATRIX, THIS_SECOND_MATRIX, THIS_FIRST_MATRIX), axis=-1
).astype(float)

THIS_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
THIS_COLOUR = matplotlib.colors.to_rgba(THIS_COLOUR, 0.5)
COLOUR_MAP_OBJECT = matplotlib.colors.ListedColormap([THIS_COLOUR])

ZERO_FONT_COLOUR = numpy.full(3, 0.)
POSITIVE_FONT_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255

FONT_SIZE = 25
AXIS_LINE_WIDTH = 4
FIGURE_RESOLUTION_DPI = 600

pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('axes', linewidth=AXIS_LINE_WIDTH)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

OUTPUT_FILE_ARG_NAME = 'output_file_name'
OUTPUT_FILE_HELP_STRING = 'Path to output file (figure will be saved here).'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _plot_kernel_one_height(kernel_matrix_2d, axes_object):
    """Plots kernel at one height.

    M = number of rows in kernel
    N = number of columns in kernel

    :param kernel_matrix_2d: M-by-N numpy array of kernel values.
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    dummy_matrix = numpy.ma.masked_where(
        kernel_matrix_2d == 0, kernel_matrix_2d)

    axes_object.imshow(
        dummy_matrix, cmap=COLOUR_MAP_OBJECT, vmin=numpy.min(kernel_matrix_2d),
        vmax=numpy.max(kernel_matrix_2d), origin='upper'
    )

    axes_object.set_xticks([], [])
    axes_object.set_yticks([], [])

    num_rows = kernel_matrix_2d.shape[0]
    num_columns = kernel_matrix_2d.shape[1]

    for i in range(num_rows):
        for j in range(num_columns):
            this_label_string = '{0:d}'.format(
                int(numpy.round(kernel_matrix_2d[i, j]))
            )

            if kernel_matrix_2d[i, j] == 0:
                this_colour = ZERO_FONT_COLOUR
            else:
                this_colour = POSITIVE_FONT_COLOUR

            axes_object.text(
                j, i, this_label_string, fontsize=FONT_SIZE, color=this_colour,
                horizontalalignment='center', verticalalignment='center')


def _run(output_file_name):
    """Plots Laplacian kernel used for edge-detector test.

    This is effectively the main method.

    :param output_file_name: See documentation at top of file.
    """

    num_heights = KERNEL_MATRIX_3D.shape[-1]

    figure_object, axes_object_matrix = plotting_utils.create_paneled_figure(
        num_rows=1, num_columns=num_heights, horizontal_spacing=0.1,
        vertical_spacing=0.1, shared_x_axis=False, shared_y_axis=False,
        keep_aspect_ratio=True)

    for k in range(num_heights):
        _plot_kernel_one_height(
            kernel_matrix_2d=KERNEL_MATRIX_3D[..., k],
            axes_object=axes_object_matrix[0, k]
        )

    axes_object_matrix[0, 0].set_title('Bottom height')
    axes_object_matrix[0, 1].set_title('Middle height')
    axes_object_matrix[0, 2].set_title('Top height')

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
