"""Applies CNN to one example file."""

import os.path
import argparse
import numpy
import keras.backend as K
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import testing_io
from gewittergefahr.deep_learning import training_validation_io as trainval_io

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)))

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

LARGE_INTEGER = int(1e10)
INPUT_TIME_FORMAT = '%Y-%m-%d-%H%M%S'

MODEL_FILE_ARG_NAME = 'input_model_file_name'
EXAMPLE_FILE_ARG_NAME = 'input_example_file_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
OUTPUT_FILE_ARG_NAME = 'output_prediction_file_name'

MODEL_FILE_HELP_STRING = (
    'Path to trained CNN.  Will be read by `cnn.read_model`.')

EXAMPLE_FILE_HELP_STRING = (
    'Path to example file (will be read by `input_examples.read_example_file`).'
    '  The CNN will be applied to all examples in this file in the time period '
    '`{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

TIME_HELP_STRING = (
    'Time (format "yyyy-mm-dd-HHMMSS").  The CNN will be applied to all '
    'examples in the file `{0:s}` in the time period `{1:s}`...`{2:s}`.'
).format(EXAMPLE_FILE_ARG_NAME, FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

# TODO(thunderhoser): Fix this help string.
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Predictions will be written here by ``.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_FILE_ARG_NAME, type=str, required=True,
    help=EXAMPLE_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _run(model_file_name, example_file_name, first_time_string,
         last_time_string, output_file_name):
    """Applies CNN to one example file.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param example_file_name: Same.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param output_file_name: Same.
    """

    print 'Reading model from: "{0:s}"...'.format(model_file_name)
    model_object = cnn.read_model(model_file_name)

    model_directory_name, _ = os.path.split(model_file_name)
    model_metafile_name = '{0:s}/model_metadata.p'.format(model_directory_name)

    print 'Reading metadata from: "{0:s}"...'.format(model_metafile_name)
    model_metadata_dict = cnn.read_model_metadata(model_metafile_name)
    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]

    training_option_dict[trainval_io.SAMPLING_FRACTIONS_KEY] = None
    training_option_dict[trainval_io.EXAMPLE_FILES_KEY] = [example_file_name]
    training_option_dict[trainval_io.FIRST_STORM_TIME_KEY] = (
        time_conversion.string_to_unix_sec(first_time_string, INPUT_TIME_FORMAT)
    )
    training_option_dict[trainval_io.LAST_STORM_TIME_KEY] = (
        time_conversion.string_to_unix_sec(last_time_string, INPUT_TIME_FORMAT)
    )

    if model_metadata_dict[cnn.LAYER_OPERATIONS_KEY] is not None:
        generator_object = testing_io.gridrad_generator_2d_reduced(
            option_dict=training_option_dict,
            list_of_operation_dicts=model_metadata_dict[
                cnn.LAYER_OPERATIONS_KEY],
            num_examples_total=LARGE_INTEGER
        )

    elif model_metadata_dict[cnn.USE_2D3D_CONVOLUTION_KEY]:
        generator_object = testing_io.myrorss_generator_2d3d(
            option_dict=training_option_dict, num_examples_total=LARGE_INTEGER)
    else:
        generator_object = testing_io.generator_2d_or_3d(
            option_dict=training_option_dict, num_examples_total=LARGE_INTEGER)

    storm_object_dict = next(generator_object)
    print SEPARATOR_STRING

    observed_labels = storm_object_dict[testing_io.TARGET_ARRAY_KEY]
    list_of_predictor_matrices = storm_object_dict[
        testing_io.INPUT_MATRICES_KEY]

    if model_metadata_dict[cnn.USE_2D3D_CONVOLUTION_KEY]:
        if len(list_of_predictor_matrices) == 3:
            this_sounding_matrix = list_of_predictor_matrices[2]
        else:
            this_sounding_matrix = None

        forecast_probabilities = cnn.apply_2d3d_cnn(
            model_object=model_object,
            reflectivity_matrix_dbz=list_of_predictor_matrices[0],
            azimuthal_shear_matrix_s01=list_of_predictor_matrices[1],
            sounding_matrix=this_sounding_matrix, verbose=True
        )[:, -1]

    else:
        if len(list_of_predictor_matrices) == 2:
            this_sounding_matrix = list_of_predictor_matrices[1]
        else:
            this_sounding_matrix = None

        forecast_probabilities = cnn.apply_2d_or_3d_cnn(
            model_object=model_object,
            radar_image_matrix=list_of_predictor_matrices[0],
            sounding_matrix=this_sounding_matrix, verbose=True
        )[:, -1]

    print SEPARATOR_STRING

    for k in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        print '{0:d}th percentile of {1:d} forecast probs = {2:.4f}'.format(
            k, len(forecast_probabilities),
            numpy.percentile(forecast_probabilities, k)
        )

    print '\n'

    for i in range(len(forecast_probabilities)):
        print 'Observed label = {0:d} ... forecast prob = {1:.4f}'.format(
            observed_labels[i], forecast_probabilities[i]
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        example_file_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_FILE_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
