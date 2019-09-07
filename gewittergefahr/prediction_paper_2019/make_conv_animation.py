"""Makes animation to explain convolution."""

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

COLOUR_MAP_OBJECT = matplotlib.colors.ListedColormap(COLOUR_LIST)
DEFAULT_FONT_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255
SPECIAL_FONT_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
FEATURE_TO_KERNEL_LINE_COLOUR = numpy.full(3, 152. / 255)

PANEL_LETTER_FONT_SIZE = 30
FEATURE_TO_KERNEL_LINE_WIDTH = 2

TEMPERATURE_MATRIX = numpy.array([
    [-10, -8, -6, -4, -2, 0],
    [-8, -6, -4, -2, 0.5, 1],
    [-6, -4, 1.5, 1.75, 2, 2],
    [-4, 2, 2.25, 2.5, 2.75, 3]
])

U_WIND_MATRIX = numpy.array([
    [1, 1, 1, 1, 1, -2],
    [1, 1, 1, -2, -2, -2],
    [1, -2, -2, -2, -2, -2],
    [-2, -2, -2, -2, -2, -2]
])

V_WIND_MATRIX = numpy.array([
    [-6, -6, -6, -6, -6, -6],
    [-6, -6, -6, -6, -6, 3],
    [-6, -6, -6, 3, 3, 3],
    [-6, -6, 3, 3, 3, 3]
])

INPUT_FEATURE_MATRIX = numpy.stack(
    (TEMPERATURE_MATRIX, U_WIND_MATRIX, V_WIND_MATRIX), axis=-1
)

# for k in range(INPUT_FEATURE_MATRIX.shape[-1]):
#     this_numerator = (
#         INPUT_FEATURE_MATRIX[..., k] - numpy.mean(INPUT_FEATURE_MATRIX[..., k])
#     )
#
#     INPUT_FEATURE_MATRIX[..., k] = (
#         this_numerator / numpy.std(INPUT_FEATURE_MATRIX[..., k], ddof=1)
#     )

TEMPERATURE_KERNEL_MATRIX = numpy.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
])

U_WIND_KERNEL_MATRIX = numpy.array([
    [1, 0, -1],
    [0, 0, 0],
    [-1, 0, 1]
])

V_WIND_KERNEL_MATRIX = numpy.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
])

KERNEL_MATRIX = numpy.stack(
    (TEMPERATURE_KERNEL_MATRIX, U_WIND_KERNEL_MATRIX, V_WIND_KERNEL_MATRIX),
    axis=-1
).astype(float)

KERNEL_MATRIX = numpy.expand_dims(KERNEL_MATRIX, axis=-1)

NUM_PANEL_ROWS = 3
NUM_PANEL_COLUMNS = 3
FIGURE_RESOLUTION_DPI = 300

# OUTPUT_DIR_NAME = (
#     '/condo/swatwork/ralager/prediction_paper_2019/conv_animation'
# )
OUTPUT_DIR_NAME = (
    '/localdata/ryan.lagerquist/eager/prediction_paper_2019/conv_animation'
)


def _plot_feature_map(feature_matrix_2d, kernel_row, kernel_column,
                      axes_object):
    """Plots one feature map.

    M = number of rows in grid
    N = number of columns in grid

    :param feature_matrix_2d: M-by-N numpy array of feature values.
    :param kernel_row: Row index at center of kernel.
    :param kernel_column: Column index at center of kernel.
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    num_rows = feature_matrix_2d.shape[0]
    num_columns = feature_matrix_2d.shape[1]

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
            if i == kernel_row and j == kernel_column:
                this_colour = DEFAULT_FONT_COLOUR
            else:
                this_colour = SPECIAL_FONT_COLOUR

            # this_label_string = '{0:.1f}'.format(feature_matrix_2d[i, j])
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
                color=SPECIAL_FONT_COLOUR,
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


def _run():
    """Makes animation to explain convolution.

    This is effectively the main method.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=OUTPUT_DIR_NAME)

    output_feature_matrix = standalone_utils.do_2d_convolution(
        feature_matrix=INPUT_FEATURE_MATRIX, kernel_matrix=KERNEL_MATRIX,
        pad_edges=True, stride_length_px=1)

    output_feature_matrix = output_feature_matrix[0, ..., 0]

    num_grid_rows = INPUT_FEATURE_MATRIX.shape[0]
    num_grid_columns = INPUT_FEATURE_MATRIX.shape[1]
    num_input_channels = INPUT_FEATURE_MATRIX.shape[2]
    image_file_names = []

    for i in range(num_grid_rows):
        for j in range(num_grid_columns):
            this_figure_object, this_axes_object_matrix = (
                plotting_utils.create_paneled_figure(
                    num_rows=NUM_PANEL_ROWS, num_columns=NUM_PANEL_COLUMNS,
                    horizontal_spacing=0.2, vertical_spacing=0.,
                    shared_x_axis=False, shared_y_axis=False,
                    keep_aspect_ratio=True)
            )

            this_axes_object_matrix[0, 2].axis('off')
            this_axes_object_matrix[2, 2].axis('off')
            letter_label = None

            for k in range(num_input_channels):
                _plot_feature_map(
                    feature_matrix_2d=INPUT_FEATURE_MATRIX[..., k],
                    kernel_row=i, kernel_column=j,
                    axes_object=this_axes_object_matrix[k, 0]
                )

                if letter_label is None:
                    letter_label = 'a'
                else:
                    letter_label = chr(ord(letter_label) + 1)

                plotting_utils.label_axes(
                    axes_object=this_axes_object_matrix[k, 0],
                    label_string='({0:s})'.format(letter_label),
                    font_size=PANEL_LETTER_FONT_SIZE,
                    y_coord_normalized=1.02, x_coord_normalized=0.1
                )

            _plot_feature_map(
                feature_matrix_2d=output_feature_matrix,
                kernel_row=i, kernel_column=j,
                axes_object=this_axes_object_matrix[1, 2]
            )

            for k in range(num_input_channels):
                _plot_kernel(
                    kernel_matrix_2d=KERNEL_MATRIX[k, ..., 0],
                    feature_matrix_2d=INPUT_FEATURE_MATRIX[..., k],
                    feature_row_at_center=i, feature_column_at_center=j,
                    axes_object=this_axes_object_matrix[k, 1]
                )

                letter_label = chr(ord(letter_label) + 1)

                plotting_utils.label_axes(
                    axes_object=this_axes_object_matrix[k, 1],
                    label_string='({0:s})'.format(letter_label),
                    font_size=PANEL_LETTER_FONT_SIZE,
                    y_coord_normalized=1.02, x_coord_normalized=0.1
                )

                _plot_feature_to_kernel_lines(
                    kernel_matrix_2d=KERNEL_MATRIX[k, ..., 0],
                    feature_matrix_2d=INPUT_FEATURE_MATRIX[..., k],
                    feature_row_at_center=i, feature_column_at_center=j,
                    kernel_axes_object=this_axes_object_matrix[k, 1],
                    feature_axes_object=this_axes_object_matrix[k, 0]
                )

            letter_label = chr(ord(letter_label) + 1)

            plotting_utils.label_axes(
                axes_object=this_axes_object_matrix[1, 2],
                label_string='({0:s})'.format(letter_label),
                font_size=PANEL_LETTER_FONT_SIZE,
                y_coord_normalized=1.02, x_coord_normalized=0.1
            )

            image_file_names.append(
                '{0:s}/conv_animation_row{1:d}_column{2:d}.jpg'.format(
                    OUTPUT_DIR_NAME, i, j)
            )

            print('Saving figure to: "{0:s}"...'.format(image_file_names[-1]))
            this_figure_object.savefig(
                image_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
                pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(this_figure_object)

    animation_file_name = '{0:s}/conv_animation.gif'.format(OUTPUT_DIR_NAME)
    print('Creating animation: "{0:s}"...'.format(animation_file_name))

    imagemagick_utils.create_gif(
        input_file_names=image_file_names, output_file_name=animation_file_name,
        num_seconds_per_frame=0.5, resize_factor=0.5)


if __name__ == '__main__':
    _run()
