"""Makes figure to explain one convolution block."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.deep_learning import standalone_utils
from gewittergefahr.deep_learning import architecture_utils
from gewittergefahr.deep_learning import input_examples
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.plotting import plotting_utils

RADAR_FIELD_NAME = radar_utils.REFL_NAME
RADAR_HEIGHT_M_ASL = 3000
NORMALIZATION_TYPE_STRING = dl_utils.Z_NORMALIZATION_TYPE_STRING
DUMMY_TARGET_NAME = 'tornado_lead-time=0000-3600sec_distance=00000-10000m'

FIRST_KERNEL_MATRIX = numpy.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
])

SECOND_KERNEL_MATRIX = numpy.array([
    [1, 0, -1],
    [0, 0, 0],
    [-1, 0, 1]
])

THIRD_KERNEL_MATRIX = numpy.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
])

KERNEL_MATRIX = numpy.stack(
    (FIRST_KERNEL_MATRIX, SECOND_KERNEL_MATRIX, THIRD_KERNEL_MATRIX), axis=-1
).astype(float)

KERNEL_MATRIX = numpy.expand_dims(KERNEL_MATRIX, axis=-2)

COLOUR_MAP_OBJECT = pyplot.get_cmap('seismic')
MAX_COLOUR_PERCENTILE = 99.
FONT_SIZE = 20

NUM_PANEL_ROWS = 3
NUM_PANEL_COLUMNS = 5
FIGURE_RESOLUTION_DPI = 300

EXAMPLE_FILE_ARG_NAME = 'input_example_file_name'
EXAMPLE_INDICES_ARG_NAME = 'example_indices'
NORMALIZATION_FILE_ARG_NAME = 'normalization_file_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

EXAMPLE_FILE_HELP_STRING = (
    'Name of example file.  The reflectivity field for one example (storm '
    'object) will be read from here by `input_examples.read_example_file`.')

EXAMPLE_INDICES_HELP_STRING = (
    'List of example indices.  One figure will be created for each example.')

NORMALIZATION_FILE_HELP_STRING = (
    'Path to normalization file.  Will be read by `deep_learning_utils.'
    'read_normalization_params_from_file` and used to normalize reflectivity '
    'field.')

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (figures will be saved here).')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_FILE_ARG_NAME, type=str, required=True,
    help=EXAMPLE_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_INDICES_ARG_NAME, type=int, nargs='+', required=True,
    help=EXAMPLE_INDICES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NORMALIZATION_FILE_ARG_NAME, type=str, required=True,
    help=NORMALIZATION_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _plot_one_feature_map(feature_matrix_2d, max_colour_value, plot_colour_bar,
                          axes_object):
    """Plots one feature map.

    M = number of rows in grid
    N = number of columns in grid

    :param feature_matrix_2d: M-by-N numpy array of feature values.
    :param max_colour_value: Max value in colour scheme.
    :param plot_colour_bar: Boolean flag.
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    min_colour_value = -1 * max_colour_value

    axes_object.pcolormesh(
        feature_matrix_2d, cmap=COLOUR_MAP_OBJECT, vmin=min_colour_value,
        vmax=max_colour_value, shading='flat', edgecolors='None')

    axes_object.set_xlim(0., feature_matrix_2d.shape[1])
    axes_object.set_ylim(0., feature_matrix_2d.shape[0])
    axes_object.set_xticks([])
    axes_object.set_yticks([])

    if not plot_colour_bar:
        return

    colour_bar_object = plotting_utils.plot_linear_colour_bar(
        axes_object_or_matrix=axes_object, data_matrix=feature_matrix_2d,
        colour_map_object=COLOUR_MAP_OBJECT, min_value=min_colour_value,
        max_value=max_colour_value, orientation_string='horizontal',
        fraction_of_axis_length=0.9, extend_min=True, extend_max=True,
        font_size=FONT_SIZE)

    tick_values = colour_bar_object.ax.get_xticks()
    tick_label_strings = ['{0:.1f}'.format(x) for x in tick_values]
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_label_strings)


def _plot_one_example(
        input_feature_matrix, feature_matrix_after_conv,
        feature_matrix_after_activn, feature_matrix_after_bn,
        feature_matrix_after_pooling, output_file_name):
    """Plots entire figure for one example (storm object).

    :param input_feature_matrix: 2-D numpy array with input features.
    :param feature_matrix_after_conv: 2-D numpy array with features after
        convolution.
    :param feature_matrix_after_activn: 2-D numpy array with features after
        activation.
    :param feature_matrix_after_bn: 2-D numpy array with features after batch
        normalization.
    :param feature_matrix_after_pooling: 2-D numpy array with features after
        pooling.
    :param output_file_name: Path to output file.  Figure will be saved here.
    """

    figure_object, axes_object_matrix = plotting_utils.create_paneled_figure(
        num_rows=NUM_PANEL_ROWS, num_columns=NUM_PANEL_COLUMNS,
        horizontal_spacing=0., vertical_spacing=0.,
        shared_x_axis=False, shared_y_axis=False, keep_aspect_ratio=True)

    max_colour_value = numpy.percentile(
        numpy.absolute(input_feature_matrix), MAX_COLOUR_PERCENTILE
    )

    axes_object_matrix[0, 0].set_title('Input', fontsize=FONT_SIZE)

    for k in range(NUM_PANEL_ROWS):
        if k == 0:
            _plot_one_feature_map(
                feature_matrix_2d=input_feature_matrix[..., k],
                max_colour_value=max_colour_value, plot_colour_bar=False,
                axes_object=axes_object_matrix[k, 0]
            )

            continue

        axes_object_matrix[k, 0].axis('off')

    colour_bar_object = plotting_utils.plot_linear_colour_bar(
        axes_object_or_matrix=axes_object_matrix[NUM_PANEL_ROWS - 1, 0],
        data_matrix=input_feature_matrix[..., 0],
        colour_map_object=COLOUR_MAP_OBJECT, min_value=-1 * max_colour_value,
        max_value=max_colour_value, orientation_string='horizontal',
        fraction_of_axis_length=0.9, extend_min=True, extend_max=True,
        font_size=FONT_SIZE)

    tick_values = colour_bar_object.ax.get_xticks()
    tick_label_strings = ['{0:.1f}'.format(x) for x in tick_values]
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_label_strings)

    letter_label = 'a'
    plotting_utils.label_axes(
        axes_object=axes_object_matrix[0, 0],
        label_string='({0:s})'.format(letter_label),
        x_coord_normalized=0., y_coord_normalized=1.05
    )

    this_matrix = numpy.stack(
        (feature_matrix_after_conv, feature_matrix_after_activn), axis=0
    )
    max_colour_value = numpy.percentile(
        numpy.absolute(this_matrix), MAX_COLOUR_PERCENTILE
    )

    axes_object_matrix[0, 1].set_title('After convolution', fontsize=FONT_SIZE)

    for k in range(NUM_PANEL_ROWS):
        _plot_one_feature_map(
            feature_matrix_2d=feature_matrix_after_conv[..., k],
            max_colour_value=max_colour_value,
            plot_colour_bar=k == NUM_PANEL_ROWS - 1,
            axes_object=axes_object_matrix[k, 1]
        )

        letter_label = chr(ord(letter_label) + 1)

        plotting_utils.label_axes(
            axes_object=axes_object_matrix[k, 1],
            label_string='({0:s})'.format(letter_label),
            x_coord_normalized=0., y_coord_normalized=1.05
        )

    axes_object_matrix[0, 2].set_title('After activation', fontsize=FONT_SIZE)

    for k in range(NUM_PANEL_ROWS):
        _plot_one_feature_map(
            feature_matrix_2d=feature_matrix_after_activn[..., k],
            max_colour_value=max_colour_value,
            plot_colour_bar=k == NUM_PANEL_ROWS - 1,
            axes_object=axes_object_matrix[k, 2]
        )

        letter_label = chr(ord(letter_label) + 1)

        plotting_utils.label_axes(
            axes_object=axes_object_matrix[k, 2],
            label_string='({0:s})'.format(letter_label),
            x_coord_normalized=0., y_coord_normalized=1.05
        )

    max_colour_value = numpy.percentile(
        numpy.absolute(feature_matrix_after_bn), MAX_COLOUR_PERCENTILE
    )

    axes_object_matrix[0, 3].set_title('After batch norm', fontsize=FONT_SIZE)

    for k in range(NUM_PANEL_ROWS):
        _plot_one_feature_map(
            feature_matrix_2d=feature_matrix_after_bn[..., k],
            max_colour_value=max_colour_value,
            plot_colour_bar=k == NUM_PANEL_ROWS - 1,
            axes_object=axes_object_matrix[k, 3]
        )

        letter_label = chr(ord(letter_label) + 1)

        plotting_utils.label_axes(
            axes_object=axes_object_matrix[k, 3],
            label_string='({0:s})'.format(letter_label),
            x_coord_normalized=0., y_coord_normalized=1.05
        )

    max_colour_value = numpy.percentile(
        numpy.absolute(feature_matrix_after_pooling), MAX_COLOUR_PERCENTILE
    )

    axes_object_matrix[0, 4].set_title('After max-pooling', fontsize=FONT_SIZE)

    for k in range(NUM_PANEL_ROWS):
        _plot_one_feature_map(
            feature_matrix_2d=feature_matrix_after_pooling[..., k],
            max_colour_value=max_colour_value,
            plot_colour_bar=k == NUM_PANEL_ROWS - 1,
            axes_object=axes_object_matrix[k, 4]
        )

        letter_label = chr(ord(letter_label) + 1)

        plotting_utils.label_axes(
            axes_object=axes_object_matrix[k, 4],
            label_string='({0:s})'.format(letter_label),
            x_coord_normalized=0., y_coord_normalized=1.05
        )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run(example_file_name, example_indices, normalization_file_name,
         output_dir_name):
    """Makes figure to explain one convolution block.

    :param example_file_name: See documentation at top of file.
    :param example_indices: Same.
    :param normalization_file_name: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    print('Reading data from: "{0:s}"...'.format(example_file_name))
    example_dict = input_examples.read_example_file(
        netcdf_file_name=example_file_name, read_all_target_vars=False,
        target_name=DUMMY_TARGET_NAME, include_soundings=False,
        radar_heights_to_keep_m_agl=numpy.array([RADAR_HEIGHT_M_ASL], dtype=int)
    )

    # print(numpy.where(example_dict[input_examples.TARGET_VALUES_KEY] == 1))

    if input_examples.REFL_IMAGE_MATRIX_KEY in example_dict:
        input_feature_matrix = example_dict[input_examples.REFL_IMAGE_MATRIX_KEY]
    else:
        field_index = example_dict[input_examples.RADAR_FIELDS_KEY].index(
            RADAR_FIELD_NAME
        )

        input_feature_matrix = example_dict[
            input_examples.RADAR_IMAGE_MATRIX_KEY
        ][..., [field_index]]

    num_examples = input_feature_matrix.shape[0]
    error_checking.assert_is_geq_numpy_array(example_indices, 0)
    error_checking.assert_is_less_than_numpy_array(
        example_indices, num_examples)

    input_feature_matrix = dl_utils.normalize_radar_images(
        radar_image_matrix=input_feature_matrix, field_names=[RADAR_FIELD_NAME],
        normalization_type_string=NORMALIZATION_TYPE_STRING,
        normalization_param_file_name=normalization_file_name)

    if len(input_feature_matrix.shape) == 4:
        input_feature_matrix = input_feature_matrix[..., 0]
    else:
        input_feature_matrix = input_feature_matrix[..., 0, 0]

    input_feature_matrix = numpy.expand_dims(input_feature_matrix, axis=-1)

    print('Doing convolution for all {0:d} examples...'.format(num_examples))
    feature_matrix_after_conv = None

    for i in range(num_examples):
        this_feature_matrix = standalone_utils.do_2d_convolution(
            feature_matrix=input_feature_matrix[i, ...] + 0,
            kernel_matrix=KERNEL_MATRIX, pad_edges=False, stride_length_px=1
        )[0, ...]

        if feature_matrix_after_conv is None:
            feature_matrix_after_conv = numpy.full(
                (num_examples,) + this_feature_matrix.shape, numpy.nan
            )

        feature_matrix_after_conv[i, ...] = this_feature_matrix

    print('Doing activation for all {0:d} examples...'.format(num_examples))
    feature_matrix_after_activn = standalone_utils.do_activation(
        input_values=feature_matrix_after_conv + 0,
        function_name=architecture_utils.RELU_FUNCTION_STRING, alpha=0.2)

    print('Doing batch norm for all {0:d} examples...'.format(num_examples))
    feature_matrix_after_bn = standalone_utils.do_batch_normalization(
        feature_matrix=feature_matrix_after_activn + 0
    )

    print('Doing max-pooling for all {0:d} examples...\n'.format(num_examples))
    feature_matrix_after_pooling = None

    for i in range(num_examples):
        this_feature_matrix = standalone_utils.do_2d_pooling(
            feature_matrix=feature_matrix_after_bn[i, ...], stride_length_px=2,
            pooling_type_string=standalone_utils.MAX_POOLING_TYPE_STRING
        )[0, ...]

        if feature_matrix_after_pooling is None:
            feature_matrix_after_pooling = numpy.full(
                (num_examples,) + this_feature_matrix.shape, numpy.nan
            )

        feature_matrix_after_pooling[i, ...] = this_feature_matrix

    for i in example_indices:
        this_output_file_name = '{0:s}/convolution_block{1:06d}.jpg'.format(
            output_dir_name, i)

        _plot_one_example(
            input_feature_matrix=input_feature_matrix[i, ...],
            feature_matrix_after_conv=feature_matrix_after_conv[i, ...],
            feature_matrix_after_activn=feature_matrix_after_activn[i, ...],
            feature_matrix_after_bn=feature_matrix_after_bn[i, ...],
            feature_matrix_after_pooling=feature_matrix_after_pooling[i, ...],
            output_file_name=this_output_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        example_file_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_FILE_ARG_NAME),
        example_indices=numpy.array(
            getattr(INPUT_ARG_OBJECT, EXAMPLE_INDICES_ARG_NAME), dtype=int
        ),
        normalization_file_name=getattr(
            INPUT_ARG_OBJECT, NORMALIZATION_FILE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
