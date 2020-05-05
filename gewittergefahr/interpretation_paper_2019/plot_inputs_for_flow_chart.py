"""Plots inputs (radar images) for CNN flow chart."""

import os.path
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import model_activation
from gewittergefahr.deep_learning import testing_io
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.plotting import imagemagick_utils
from gewittergefahr.scripts import plot_input_examples as plot_examples

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

RADAR_HEIGHTS_M_AGL = numpy.array([2000, 6000, 10000], dtype=int)

COLOUR_BAR_LENGTH = 0.25
PANEL_NAME_FONT_SIZE = 30
COLOUR_BAR_FONT_SIZE = 25

CONVERT_EXE_NAME = '/usr/bin/convert'
TITLE_FONT_SIZE = 150
TITLE_FONT_NAME = 'DejaVu-Sans-Bold'

FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)

ACTIVATION_FILE_ARG_NAME = 'input_activation_file_name'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

ACTIVATION_FILE_HELP_STRING = (
    'Path to activation file (will be read by `model_activation.read_file`).'
)

NUM_EXAMPLES_HELP_STRING = (
    'Number of examples to read from `{0:s}`.  If you want to read all '
    'examples, leave this alone.'
).format(ACTIVATION_FILE_ARG_NAME)

EXAMPLE_DIR_HELP_STRING = (
    'Name of top-level directory with examples.  Files therein will be found by'
    ' `input_examples.find_example_file` and read by'
    ' `input_examples.read_example_file`.')

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (figures will be saved here).')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + ACTIVATION_FILE_ARG_NAME, type=str, required=False, default='',
    help=ACTIVATION_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_EXAMPLES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _plot_one_example(
        radar_matrix, sounding_matrix, sounding_pressures_pascals,
        full_storm_id_string, storm_time_unix_sec, model_metadata_dict,
        output_dir_name):
    """Plots predictors for one example.

    M = number of rows in radar grid
    N = number of columns in radar grid
    H_r = number of heights in radar grid
    F_r = number of radar fields
    H_s = number of sounding heights
    F_s = number of sounding fields

    :param radar_matrix: numpy array (1 x M x N x H_r x F_r) of radar values.
    :param sounding_matrix: numpy array (1 x H_s x F_s) of sounding values.
    :param sounding_pressures_pascals: numpy array (length H_s) of sounding
        pressures.
    :param full_storm_id_string: Full storm ID.
    :param storm_time_unix_sec: Valid time.
    :param model_metadata_dict: Dictionary returned by
        `cnn.read_model_metadata`.
    :param output_dir_name: Name of output directory (figures will be saved
        here).
    :return: radar_figure_file_name: Path to radar figure created by this
        method.
    """

    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    radar_field_names = training_option_dict[trainval_io.RADAR_FIELDS_KEY]

    num_radar_fields = radar_matrix.shape[-1]
    num_radar_heights = radar_matrix.shape[-2]

    handle_dict = plot_examples.plot_one_example(
        list_of_predictor_matrices=[radar_matrix, sounding_matrix],
        model_metadata_dict=model_metadata_dict,
        pmm_flag=False, example_index=0, plot_sounding=True,
        sounding_pressures_pascals=sounding_pressures_pascals,
        allow_whitespace=True, plot_panel_names=True,
        panel_name_font_size=PANEL_NAME_FONT_SIZE,
        add_titles=False, label_colour_bars=True,
        colour_bar_length=COLOUR_BAR_LENGTH,
        colour_bar_font_size=COLOUR_BAR_FONT_SIZE,
        num_panel_rows=num_radar_heights
    )

    sounding_file_name = plot_examples.metadata_to_file_name(
        output_dir_name=output_dir_name, is_sounding=True, pmm_flag=False,
        full_storm_id_string=full_storm_id_string,
        storm_time_unix_sec=storm_time_unix_sec
    )

    print('Saving figure to: "{0:s}"...'.format(sounding_file_name))

    sounding_figure_object = handle_dict[plot_examples.SOUNDING_FIGURE_KEY]
    sounding_figure_object.savefig(
        sounding_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(sounding_figure_object)

    figure_objects = handle_dict[plot_examples.RADAR_FIGURES_KEY]
    panel_file_names = [None] * num_radar_fields

    for k in range(num_radar_fields):
        panel_file_names[k] = plot_examples.metadata_to_file_name(
            output_dir_name=output_dir_name, is_sounding=False, pmm_flag=False,
            full_storm_id_string=full_storm_id_string,
            storm_time_unix_sec=storm_time_unix_sec,
            radar_field_name=radar_field_names[k]
        )

        print('Saving figure to: "{0:s}"...'.format(panel_file_names[k]))

        figure_objects[k].savefig(
            panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_objects[k])

    radar_figure_file_name = plot_examples.metadata_to_file_name(
        output_dir_name=output_dir_name, is_sounding=False, pmm_flag=False,
        full_storm_id_string=full_storm_id_string,
        storm_time_unix_sec=storm_time_unix_sec
    )

    print('Concatenating panels to: "{0:s}"...'.format(radar_figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=radar_figure_file_name,
        num_panel_rows=1, num_panel_columns=num_radar_fields,
        border_width_pixels=50
    )
    imagemagick_utils.resize_image(
        input_file_name=radar_figure_file_name,
        output_file_name=radar_figure_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=radar_figure_file_name,
        output_file_name=radar_figure_file_name,
        border_width_pixels=10
    )

    for this_file_name in panel_file_names:
        os.remove(this_file_name)

    return radar_figure_file_name


def _run(activation_file_name, num_examples, top_example_dir_name,
         output_dir_name):
    """Plots inputs (radar images) for CNN flow chart.

    This is effectively the main method.

    :param activation_file_name: See documentation at top of file.
    :param num_examples: Same.
    :param top_example_dir_name: Same.
    :param output_dir_name: Same.
    :raises: ValueError: if activation file contains activations for more than
        one model component.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    print('Reading data from: "{0:s}"...'.format(activation_file_name))
    activation_matrix, activation_metadata_dict = (
        model_activation.read_file(activation_file_name)
    )

    num_model_components = activation_matrix.shape[1]
    if num_model_components > 1:
        error_string = (
            'The file should contain activations for only one model '
            'component, not {0:d}.'
        ).format(num_model_components)

        raise TypeError(error_string)

    full_storm_id_strings = (
        activation_metadata_dict[model_activation.FULL_IDS_KEY]
    )
    storm_times_unix_sec = (
        activation_metadata_dict[model_activation.STORM_TIMES_KEY]
    )
    model_file_name = (
        activation_metadata_dict[model_activation.MODEL_FILE_NAME_KEY]
    )
    model_metafile_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0]
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = cnn.read_model_metadata(model_metafile_name)

    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    training_option_dict[trainval_io.NORMALIZATION_TYPE_KEY] = None
    training_option_dict[trainval_io.SAMPLING_FRACTIONS_KEY] = None
    training_option_dict[trainval_io.REFLECTIVITY_MASK_KEY] = None
    training_option_dict[trainval_io.RADAR_HEIGHTS_KEY] = RADAR_HEIGHTS_M_AGL

    model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY] = training_option_dict
    model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY][
        trainval_io.UPSAMPLE_REFLECTIVITY_KEY
    ] = False

    if 0 < num_examples < len(full_storm_id_strings):
        full_storm_id_strings = full_storm_id_strings[:num_examples]
        storm_times_unix_sec = storm_times_unix_sec[:num_examples]

    print(SEPARATOR_STRING)
    example_dict = testing_io.read_predictors_specific_examples(
        top_example_dir_name=top_example_dir_name,
        desired_full_id_strings=full_storm_id_strings,
        desired_times_unix_sec=storm_times_unix_sec,
        option_dict=model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY],
        layer_operation_dicts=model_metadata_dict[cnn.LAYER_OPERATIONS_KEY]
    )
    print(SEPARATOR_STRING)

    radar_matrix = example_dict[testing_io.INPUT_MATRICES_KEY][0]
    sounding_matrix = example_dict[testing_io.INPUT_MATRICES_KEY][-1]
    sounding_pressure_matrix_pa = (
        example_dict[testing_io.SOUNDING_PRESSURES_KEY]
    )
    num_examples = len(full_storm_id_strings)

    for i in range(num_examples):
        _plot_one_example(
            radar_matrix=radar_matrix[[i], ...],
            sounding_matrix=sounding_matrix[[i], ...],
            sounding_pressures_pascals=sounding_pressure_matrix_pa[i, ...],
            full_storm_id_string=full_storm_id_strings[i],
            storm_time_unix_sec=storm_times_unix_sec[i],
            model_metadata_dict=model_metadata_dict,
            output_dir_name=output_dir_name
        )
        print('\n')


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        activation_file_name=getattr(
            INPUT_ARG_OBJECT, ACTIVATION_FILE_ARG_NAME
        ),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        top_example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
