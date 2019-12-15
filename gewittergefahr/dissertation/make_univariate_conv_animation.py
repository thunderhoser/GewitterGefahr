"""Makes animation to explain univariate convolution."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.deep_learning import standalone_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import imagemagick_utils

DEFAULT_FONT_SIZE = 25
DEFAULT_LINE_WIDTH = 4

pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', linewidth=DEFAULT_LINE_WIDTH)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

COLOUR_LIST = [
    numpy.full(3, 1.),
    numpy.array([27, 158, 119], dtype=float) / 255
]

COLOUR_LIST[1] = matplotlib.colors.to_rgba(COLOUR_LIST[1], 0.5)

COLOUR_MAP_OBJECT = matplotlib.colors.ListedColormap(COLOUR_LIST)
DEFAULT_FONT_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
SPECIAL_FONT_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255
FEATURE_TO_KERNEL_LINE_COLOUR = numpy.full(3, 152. / 255)

PANEL_LETTER_FONT_SIZE = 30
FEATURE_TO_KERNEL_LINE_WIDTH = 2

INPUT_FEATURE_MATRIX = numpy.array([
    [-10, -8, -6, -4, -2, 0],
    [-8, -6, -4, -2, 0.5, 1],
    [-6, -4, 1.5, 1.75, 2, 2],
    [-4, 2, 2.25, 2.5, 2.75, 3]
])

INPUT_FEATURE_MATRIX = numpy.expand_dims(INPUT_FEATURE_MATRIX, axis=-1)

KERNEL_MATRIX = numpy.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
])

KERNEL_MATRIX = numpy.expand_dims(KERNEL_MATRIX, axis=-1)
KERNEL_MATRIX = numpy.expand_dims(KERNEL_MATRIX, axis=-1)

NUM_PANEL_ROWS = 1
NUM_PANEL_COLUMNS = 3
FIGURE_RESOLUTION_DPI = 300

FIGURE_CAPTION = (
    'Schematic for univariate convolution in two spatial dimensions.\n'
    '[a] Input map.\n'
    '[b] Convolutional filter.  Filter weights are set to values known to be '
    'useful for edge\n'
    'detection.  In a real CNN these weights are learned during training.\n'
    '[c] Feature map produced by convolution.  At each filter position, the '
    'highlighted\n'
    'value in c is the sum of elementwise products of highlighted values in a '
    'and b.\n'
    'When the filter runs off the edge of the 2-D grid, the missing input '
    'values are\n'
    'assumed to be zero.'
)

INCLUDE_CAPTION_ARG_NAME = 'include_caption'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INCLUDE_CAPTION_HELP_STRING = (
    'Boolean flag.  If 1, will include caption in each frame of animation.')

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (individual frames and full GIF will be saved '
    'here).')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INCLUDE_CAPTION_ARG_NAME, type=int, required=False, default=1,
    help=INCLUDE_CAPTION_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _plot_feature_map(feature_matrix_2d, kernel_row, kernel_column,
                      is_output_map, axes_object):
    """Plots one feature map.

    M = number of rows in grid
    N = number of columns in grid

    :param feature_matrix_2d: M-by-N numpy array of feature values.
    :param kernel_row: Row index at center of kernel.
    :param kernel_column: Column index at center of kernel.
    :param is_output_map: Boolean flag.
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    num_rows = feature_matrix_2d.shape[0]
    num_columns = feature_matrix_2d.shape[1]

    if is_output_map:
        first_highlighted_row = kernel_row
        last_highlighted_row = kernel_row
        first_highlighted_column = kernel_column
        last_highlighted_column = kernel_column
    else:
        first_highlighted_row = max([kernel_row - 1, 0])
        last_highlighted_row = min([kernel_row + 1, num_rows - 1])
        first_highlighted_column = max([kernel_column - 1, 0])
        last_highlighted_column = min([kernel_column + 1, num_columns - 1])

    dummy_matrix = numpy.full(feature_matrix_2d.shape, numpy.nan)
    dummy_matrix[
        first_highlighted_row:(last_highlighted_row + 1),
        first_highlighted_column:(last_highlighted_column + 1)
    ] = 0

    dummy_matrix = numpy.ma.masked_where(
        numpy.isnan(dummy_matrix), dummy_matrix
    )

    axes_object.imshow(
        dummy_matrix, cmap=COLOUR_MAP_OBJECT, vmin=-1., vmax=0., origin='upper'
    )

    axes_object.set_xticks([], [])
    axes_object.set_yticks([], [])

    for i in range(num_rows):
        for j in range(num_columns):
            if i == kernel_row and j == kernel_column and is_output_map:
                this_colour = SPECIAL_FONT_COLOUR
            else:
                this_colour = DEFAULT_FONT_COLOUR

            this_label_string = '{0:d}'.format(
                int(numpy.round(feature_matrix_2d[i, j]))
            )

            axes_object.text(
                j, i, this_label_string, fontsize=DEFAULT_FONT_SIZE,
                color=this_colour,
                horizontalalignment='center', verticalalignment='center')


def _plot_kernel(kernel_matrix_2d, feature_matrix_2d, feature_row_at_center,
                 feature_column_at_center, axes_object):
    """Plots convolution kernel (filter).

    M = number of rows in feature map
    N = number of columns in feature map
    J = number of rows in kernel
    K = number of columns in kernel

    :param kernel_matrix_2d: J-by-K numpy array of weights.
    :param feature_matrix_2d: M-by-N numpy array of feature values.
    :param feature_row_at_center: Feature-map row at center of kernel.
    :param feature_column_at_center: Feature-map column at center of kernel.
    :param axes_object: See doc for `_plot_feature_map`.
    """

    num_kernel_rows = kernel_matrix_2d.shape[0]
    num_kernel_columns = kernel_matrix_2d.shape[1]
    num_feature_rows = feature_matrix_2d.shape[0]
    num_feature_columns = feature_matrix_2d.shape[1]

    center_row_from_bottom = num_feature_rows - feature_row_at_center - 1
    center_column_from_right = (
        num_feature_columns - feature_column_at_center - 1
    )

    first_highlighted_row = max([1 - feature_row_at_center, 0])
    last_highlighted_row = min([
        1 + center_row_from_bottom, num_kernel_rows - 1
    ])
    first_highlighted_column = max([1 - feature_column_at_center, 0])
    last_highlighted_column = min([
        1 + center_column_from_right, num_kernel_columns - 1
    ])

    dummy_matrix = numpy.full(kernel_matrix_2d.shape, numpy.nan)
    dummy_matrix[
        first_highlighted_row:(last_highlighted_row + 1),
        first_highlighted_column:(last_highlighted_column + 1)
    ] = 0

    dummy_matrix = numpy.ma.masked_where(
        numpy.isnan(dummy_matrix), dummy_matrix
    )

    axes_object.imshow(
        dummy_matrix, cmap=COLOUR_MAP_OBJECT, vmin=-1., vmax=0., origin='upper'
    )

    axes_object.set_xticks([], [])
    axes_object.set_yticks([], [])

    for i in range(num_kernel_rows):
        for j in range(num_kernel_columns):
            this_label_string = '{0:d}'.format(
                int(numpy.round(kernel_matrix_2d[i, j]))
            )

            axes_object.text(
                j, i, this_label_string, fontsize=DEFAULT_FONT_SIZE,
                color=DEFAULT_FONT_COLOUR,
                horizontalalignment='center', verticalalignment='center')


def _plot_feature_to_kernel_lines(
        kernel_matrix_2d, feature_matrix_2d, feature_row_at_center,
        feature_column_at_center, kernel_axes_object, feature_axes_object):
    """Plots lines between feature map and kernel.

    :param kernel_matrix_2d: See doc for `_plot_kernel`.
    :param feature_matrix_2d: Same.
    :param feature_row_at_center: Same.
    :param feature_column_at_center: Same.
    :param kernel_axes_object: Axes on which kernel is plotted (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param feature_axes_object: Axes on which feature map is plotted (instance
        of `matplotlib.axes._subplots.AxesSubplot`).
    """

    num_feature_rows = feature_matrix_2d.shape[0]
    num_feature_columns = feature_matrix_2d.shape[1]
    num_kernel_rows = kernel_matrix_2d.shape[0]

    first_feature_row = max([feature_row_at_center - 1.5, -0.5])
    last_feature_row = min([
        feature_row_at_center + 1.5, num_feature_rows - 0.5
    ])
    last_feature_column = min([
        feature_column_at_center + 1.5, num_feature_columns - 0.5
    ])

    center_row_from_bottom = num_feature_rows - feature_row_at_center - 1
    first_kernel_row = -0.5 + max([1 - feature_row_at_center, 0])
    last_kernel_row = 0.5 + min([
        1 + center_row_from_bottom, num_kernel_rows - 1
    ])
    first_kernel_column = -0.5 + max([1 - feature_column_at_center, 0])

    this_connection_object = ConnectionPatch(
        xyA=(first_kernel_column, first_kernel_row),
        xyB=(last_feature_column, first_feature_row),
        coordsA='data', coordsB='data',
        axesA=kernel_axes_object, axesB=feature_axes_object,
        color=FEATURE_TO_KERNEL_LINE_COLOUR,
        linewidth=FEATURE_TO_KERNEL_LINE_WIDTH, linestyle='dashed'
    )

    kernel_axes_object.add_artist(this_connection_object)

    this_connection_object = ConnectionPatch(
        xyA=(first_kernel_column, last_kernel_row),
        xyB=(last_feature_column, last_feature_row),
        coordsA='data', coordsB='data',
        axesA=kernel_axes_object, axesB=feature_axes_object,
        color=FEATURE_TO_KERNEL_LINE_COLOUR,
        linewidth=FEATURE_TO_KERNEL_LINE_WIDTH, linestyle='dashed'
    )

    kernel_axes_object.add_artist(this_connection_object)


def _run(include_caption, output_dir_name):
    """Makes animation to explain multivariate convolution.

    This is effectively the main method.

    :param include_caption: See documentation at top of file.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    output_feature_matrix = standalone_utils.do_2d_convolution(
        feature_matrix=INPUT_FEATURE_MATRIX, kernel_matrix=KERNEL_MATRIX,
        pad_edges=True, stride_length_px=1)

    output_feature_matrix = output_feature_matrix[0, ..., 0]

    num_grid_rows = INPUT_FEATURE_MATRIX.shape[0]
    num_grid_columns = INPUT_FEATURE_MATRIX.shape[1]
    image_file_names = []

    kernel_width_ratio = float(KERNEL_MATRIX.shape[1]) / num_grid_columns
    kernel_height_ratio = float(KERNEL_MATRIX.shape[0]) / num_grid_rows

    for i in range(num_grid_rows):
        for j in range(num_grid_columns):
            this_figure_object, this_axes_object_matrix = (
                plotting_utils.create_paneled_figure(
                    num_rows=NUM_PANEL_ROWS, num_columns=NUM_PANEL_COLUMNS,
                    horizontal_spacing=0.2, vertical_spacing=0.,
                    shared_x_axis=False, shared_y_axis=False,
                    keep_aspect_ratio=True)
            )

            letter_label = None

            _plot_feature_map(
                feature_matrix_2d=INPUT_FEATURE_MATRIX[..., 0],
                kernel_row=i, kernel_column=j, is_output_map=False,
                axes_object=this_axes_object_matrix[0, 0]
            )

            if letter_label is None:
                letter_label = 'a'
            else:
                letter_label = chr(ord(letter_label) + 1)

            plotting_utils.label_axes(
                axes_object=this_axes_object_matrix[0, 0],
                label_string='({0:s})'.format(letter_label),
                font_size=PANEL_LETTER_FONT_SIZE,
                y_coord_normalized=1.04, x_coord_normalized=0.1
            )

            _plot_feature_map(
                feature_matrix_2d=output_feature_matrix,
                kernel_row=i, kernel_column=j, is_output_map=True,
                axes_object=this_axes_object_matrix[0, 2]
            )

            this_bbox_object = this_axes_object_matrix[0, 1].get_position()
            this_width = kernel_width_ratio * (
                this_bbox_object.x1 - this_bbox_object.x0
            )
            this_height = kernel_height_ratio * (
                this_bbox_object.y1 - this_bbox_object.y0
            )

            this_bbox_object.x0 += 0.5 * this_width
            this_bbox_object.y0 = (
                this_axes_object_matrix[0, 0].get_position().y0 + 0.1
            )
            this_bbox_object.x1 = this_bbox_object.x0 + this_width
            this_bbox_object.y1 = this_bbox_object.y0 + this_height

            this_axes_object_matrix[0, 1].set_position(this_bbox_object)

            _plot_kernel(
                kernel_matrix_2d=KERNEL_MATRIX[..., 0, 0],
                feature_matrix_2d=INPUT_FEATURE_MATRIX[..., 0],
                feature_row_at_center=i, feature_column_at_center=j,
                axes_object=this_axes_object_matrix[0, 1]
            )

            letter_label = chr(ord(letter_label) + 1)

            plotting_utils.label_axes(
                axes_object=this_axes_object_matrix[0, 1],
                label_string='({0:s})'.format(letter_label),
                font_size=PANEL_LETTER_FONT_SIZE,
                y_coord_normalized=1.04, x_coord_normalized=0.2
            )

            _plot_feature_to_kernel_lines(
                kernel_matrix_2d=KERNEL_MATRIX[..., 0, 0],
                feature_matrix_2d=INPUT_FEATURE_MATRIX[..., 0],
                feature_row_at_center=i, feature_column_at_center=j,
                kernel_axes_object=this_axes_object_matrix[0, 1],
                feature_axes_object=this_axes_object_matrix[0, 0]
            )

            letter_label = chr(ord(letter_label) + 1)

            plotting_utils.label_axes(
                axes_object=this_axes_object_matrix[0, 2],
                label_string='({0:s})'.format(letter_label),
                font_size=PANEL_LETTER_FONT_SIZE,
                y_coord_normalized=1.04, x_coord_normalized=0.1
            )

            if include_caption:
                this_figure_object.text(
                    0.5, 0.35, FIGURE_CAPTION,
                    fontsize=DEFAULT_FONT_SIZE, color='k',
                    horizontalalignment='center', verticalalignment='top')

            image_file_names.append(
                '{0:s}/conv_animation_row{1:d}_column{2:d}.jpg'.format(
                    output_dir_name, i, j)
            )

            print('Saving figure to: "{0:s}"...'.format(image_file_names[-1]))
            this_figure_object.savefig(
                image_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
                pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(this_figure_object)

    animation_file_name = '{0:s}/conv_animation.gif'.format(output_dir_name)
    print('Creating animation: "{0:s}"...'.format(animation_file_name))

    imagemagick_utils.create_gif(
        input_file_names=image_file_names, output_file_name=animation_file_name,
        num_seconds_per_frame=0.5, resize_factor=0.5)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        include_caption=bool(
            getattr(INPUT_ARG_OBJECT, INCLUDE_CAPTION_ARG_NAME)
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
