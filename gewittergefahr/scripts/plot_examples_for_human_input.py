"""Plots one or more examples (storm objects) for human input.

The images saved by this script are meant to be drawn on by humans.
"""

import os.path
import argparse
import numpy
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import input_examples
from gewittergefahr.deep_learning import model_activation
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.deep_learning import testing_io
from gewittergefahr.scripts import plot_input_examples as plot_examples

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

DUMMY_TARGET_NAME = plot_examples.DUMMY_TARGET_NAME
SHEAR_FIELD_NAMES = [radar_utils.LOW_LEVEL_SHEAR_NAME]
REFL_HEIGHTS_M_AGL = numpy.array([1000, 2000, 3000], dtype=int)

THIS_DICT = {
    input_examples.RADAR_FIELD_KEY: radar_utils.REFL_NAME,
    input_examples.OPERATION_NAME_KEY: input_examples.MAX_OPERATION_NAME,
    input_examples.MIN_HEIGHT_KEY: 1000,
    input_examples.MAX_HEIGHT_KEY: 3000
}

LAYER_OPERATION_DICTS = [THIS_DICT]

ACTIVATION_FILE_ARG_NAME = plot_examples.ACTIVATION_FILE_ARG_NAME
STORM_METAFILE_ARG_NAME = plot_examples.STORM_METAFILE_ARG_NAME
NUM_EXAMPLES_ARG_NAME = plot_examples.NUM_EXAMPLES_ARG_NAME
EXAMPLE_DIR_ARG_NAME = plot_examples.EXAMPLE_DIR_ARG_NAME
NUM_ROWS_ARG_NAME = plot_examples.NUM_ROWS_ARG_NAME
NUM_COLUMNS_ARG_NAME = plot_examples.NUM_COLUMNS_ARG_NAME
ALLOW_WHITESPACE_ARG_NAME = plot_examples.ALLOW_WHITESPACE_ARG_NAME
CBAR_LENGTH_ARG_NAME = plot_examples.CBAR_LENGTH_ARG_NAME
OUTPUT_DIR_ARG_NAME = plot_examples.OUTPUT_DIR_ARG_NAME

ACTIVATION_FILE_HELP_STRING = plot_examples.ACTIVATION_FILE_HELP_STRING
STORM_METAFILE_HELP_STRING = plot_examples.STORM_METAFILE_HELP_STRING
NUM_EXAMPLES_HELP_STRING = plot_examples.NUM_EXAMPLES_HELP_STRING
EXAMPLE_DIR_HELP_STRING = plot_examples.EXAMPLE_DIR_HELP_STRING
NUM_ROWS_HELP_STRING = plot_examples.NUM_ROWS_HELP_STRING
NUM_COLUMNS_HELP_STRING = plot_examples.NUM_COLUMNS_HELP_STRING
ALLOW_WHITESPACE_HELP_STRING = plot_examples.ALLOW_WHITESPACE_HELP_STRING
CBAR_LENGTH_HELP_STRING = plot_examples.CBAR_LENGTH_HELP_STRING
OUTPUT_DIR_HELP_STRING = plot_examples.OUTPUT_DIR_HELP_STRING

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + ACTIVATION_FILE_ARG_NAME, type=str, required=False, default='',
    help=ACTIVATION_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + STORM_METAFILE_ARG_NAME, type=str, required=False, default='',
    help=STORM_METAFILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_EXAMPLES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_ROWS_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_ROWS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_COLUMNS_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_COLUMNS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + ALLOW_WHITESPACE_ARG_NAME, type=int, required=False, default=1,
    help=ALLOW_WHITESPACE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CBAR_LENGTH_ARG_NAME, type=float, required=False,
    default=0.9, help=CBAR_LENGTH_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(activation_file_name, storm_metafile_name, num_examples,
         top_example_dir_name, num_radar_rows, num_radar_columns,
         allow_whitespace, colour_bar_length, output_dir_name):
    """Plots one or more examples (storm objects) for human input.

    This is effectively the main method.

    :param activation_file_name: See documentation at top of file.
    :param storm_metafile_name: Same.
    :param num_examples: Same.
    :param top_example_dir_name: Same.
    :param num_radar_rows: Same.
    :param num_radar_columns: Same.
    :param allow_whitespace: Same.
    :param colour_bar_length: Same.
    :param output_dir_name: Same.
    """

    if num_radar_rows <= 0:
        num_radar_rows = None
    if num_radar_columns <= 0:
        num_radar_columns = None

    if activation_file_name in ['', 'None']:
        activation_file_name = None

    if activation_file_name is None:
        print('Reading data from: "{0:s}"...'.format(storm_metafile_name))
        full_storm_id_strings, storm_times_unix_sec = (
            tracking_io.read_ids_and_times(storm_metafile_name)
        )

        training_option_dict = dict()
        training_option_dict[trainval_io.SOUNDING_FIELDS_KEY] = None
        training_option_dict[trainval_io.SOUNDING_HEIGHTS_KEY] = None

        training_option_dict[trainval_io.NUM_ROWS_KEY] = num_radar_rows
        training_option_dict[trainval_io.NUM_COLUMNS_KEY] = num_radar_columns
        training_option_dict[trainval_io.NORMALIZATION_TYPE_KEY] = None
        training_option_dict[trainval_io.TARGET_NAME_KEY] = DUMMY_TARGET_NAME
        training_option_dict[trainval_io.BINARIZE_TARGET_KEY] = False
        training_option_dict[trainval_io.SAMPLING_FRACTIONS_KEY] = None
        training_option_dict[trainval_io.REFLECTIVITY_MASK_KEY] = None

        model_metadata_dict = {cnn.LAYER_OPERATIONS_KEY: None}
    else:
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

        full_storm_id_strings = activation_metadata_dict[
            model_activation.FULL_IDS_KEY]
        storm_times_unix_sec = activation_metadata_dict[
            model_activation.STORM_TIMES_KEY]

        model_file_name = activation_metadata_dict[
            model_activation.MODEL_FILE_NAME_KEY]
        model_metafile_name = '{0:s}/model_metadata.p'.format(
            os.path.split(model_file_name)[0]
        )

        print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
        model_metadata_dict = cnn.read_model_metadata(model_metafile_name)

        training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
        training_option_dict[trainval_io.NORMALIZATION_TYPE_KEY] = None
        training_option_dict[trainval_io.SAMPLING_FRACTIONS_KEY] = None
        training_option_dict[trainval_io.REFLECTIVITY_MASK_KEY] = None

    training_option_dict[trainval_io.RADAR_FIELDS_KEY] = SHEAR_FIELD_NAMES
    training_option_dict[trainval_io.RADAR_HEIGHTS_KEY] = REFL_HEIGHTS_M_AGL
    training_option_dict[trainval_io.UPSAMPLE_REFLECTIVITY_KEY] = True

    model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY] = training_option_dict
    # model_metadata_dict[cnn.LAYER_OPERATIONS_KEY] = LAYER_OPERATION_DICTS

    if 0 < num_examples < len(full_storm_id_strings):
        full_storm_id_strings = full_storm_id_strings[:num_examples]
        storm_times_unix_sec = storm_times_unix_sec[:num_examples]

    print(SEPARATOR_STRING)
    predictor_matrices = testing_io.read_specific_examples(
        desired_full_id_strings=full_storm_id_strings,
        desired_times_unix_sec=storm_times_unix_sec,
        option_dict=model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY],
        top_example_dir_name=top_example_dir_name,
        list_of_layer_operation_dicts=model_metadata_dict[
            cnn.LAYER_OPERATIONS_KEY]
    )[0]
    print(SEPARATOR_STRING)

    example_dict = {
        input_examples.RADAR_FIELDS_KEY: SHEAR_FIELD_NAMES,
        input_examples.REFL_IMAGE_MATRIX_KEY: predictor_matrices[0],
        input_examples.AZ_SHEAR_IMAGE_MATRIX_KEY: predictor_matrices[1],
        input_examples.RADAR_HEIGHTS_KEY: REFL_HEIGHTS_M_AGL
    }

    example_dict = input_examples.reduce_examples_3d_to_2d(
        example_dict=example_dict,
        list_of_operation_dicts=LAYER_OPERATION_DICTS)

    print(example_dict[input_examples.RADAR_IMAGE_MATRIX_KEY].shape)

    print(len(predictor_matrices))
    for this_matrix in predictor_matrices:
        print(this_matrix.shape)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        activation_file_name=getattr(
            INPUT_ARG_OBJECT, ACTIVATION_FILE_ARG_NAME),
        storm_metafile_name=getattr(INPUT_ARG_OBJECT, STORM_METAFILE_ARG_NAME),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        top_example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        num_radar_rows=getattr(INPUT_ARG_OBJECT, NUM_ROWS_ARG_NAME),
        num_radar_columns=getattr(INPUT_ARG_OBJECT, NUM_COLUMNS_ARG_NAME),
        allow_whitespace=bool(getattr(
            INPUT_ARG_OBJECT, ALLOW_WHITESPACE_ARG_NAME
        )),
        colour_bar_length=getattr(INPUT_ARG_OBJECT, CBAR_LENGTH_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
