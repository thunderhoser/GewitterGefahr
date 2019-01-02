"""Runs permutation test for predictor importance."""

import copy
import os.path
import argparse
import random
import numpy
from keras import backend as K
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import input_examples
from gewittergefahr.deep_learning import testing_io
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.deep_learning import permutation

random.seed(6695)
numpy.random.seed(6695)

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)))

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MODEL_FILE_ARG_NAME = 'input_model_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
FIRST_SPC_DATE_ARG_NAME = 'first_spc_date_string'
LAST_SPC_DATE_ARG_NAME = 'last_spc_date_string'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
CLASS_FRACTION_KEYS_ARG_NAME = 'class_fraction_keys'
CLASS_FRACTION_VALUES_ARG_NAME = 'class_fraction_values'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

MODEL_FILE_HELP_STRING = (
    'Path to file with trained CNN.  Will be read by `cnn.read_model`.')

EXAMPLE_DIR_HELP_STRING = (
    'Name of top-level directory with input examples.  Files therein will be '
    'found by `input_examples.find_example_file` and read by '
    '`input_examples.read_example_file`.')

SPC_DATE_HELP_STRING = (
    'SPC date (format "yyyymmdd").  This script will evaluate predictions on '
    'examples from the period `{0:s}`...`{1:s}`.'
).format(FIRST_SPC_DATE_ARG_NAME, LAST_SPC_DATE_ARG_NAME)

NUM_EXAMPLES_HELP_STRING = 'Number of examples to use.'

CLASS_FRACTION_KEYS_HELP_STRING = (
    'List of keys used to create input `class_to_sampling_fraction_dict` for '
    '`deep_learning_utils.sample_by_class`.  If you do not want class-'
    'conditional sampling, leave this alone.'
)

CLASS_FRACTION_VALUES_HELP_STRING = (
    'List of values used to create input `class_to_sampling_fraction_dict` for '
    '`deep_learning_utils.sample_by_class`.  If you do not want class-'
    'conditional sampling, leave this alone.'
)

OUTPUT_FILE_HELP_STRING = (
    'Path to output (Pickle) file.  Will be written by'
    '`permutation_importance.write_results`.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=True,
    help=NUM_EXAMPLES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CLASS_FRACTION_KEYS_ARG_NAME, type=int, nargs='+',
    required=False, default=[0], help=CLASS_FRACTION_KEYS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CLASS_FRACTION_VALUES_ARG_NAME, type=float, nargs='+',
    required=False, default=[0.], help=CLASS_FRACTION_VALUES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _run(model_file_name, top_example_dir_name,
         first_spc_date_string, last_spc_date_string, num_examples,
         class_fraction_keys, class_fraction_values, output_file_name):
    """Runs permutation test for predictor importance.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param top_example_dir_name: Same.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
    :param num_examples: Same.
    :param class_fraction_keys: Same.
    :param class_fraction_values: Same.
    :param output_file_name: Same.
    """

    print 'Reading model from: "{0:s}"...'.format(model_file_name)
    model_object = cnn.read_model(model_file_name)

    model_directory_name, _ = os.path.split(model_file_name)
    metadata_file_name = '{0:s}/model_metadata.p'.format(model_directory_name)

    print 'Reading metadata from: "{0:s}"...'.format(metadata_file_name)
    model_metadata_dict = cnn.read_model_metadata(metadata_file_name)
    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]

    if len(class_fraction_keys) > 1:
        class_to_sampling_fraction_dict = dict(zip(
            class_fraction_keys, class_fraction_values))
    else:
        class_to_sampling_fraction_dict = None

    training_option_dict[
        trainval_io.SAMPLING_FRACTIONS_KEY] = class_to_sampling_fraction_dict

    example_file_names = input_examples.find_many_example_files(
        top_directory_name=top_example_dir_name, shuffled=False,
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string,
        raise_error_if_any_missing=False)

    training_option_dict[trainval_io.EXAMPLE_FILES_KEY] = example_file_names
    training_option_dict[trainval_io.FIRST_STORM_TIME_KEY] = (
        time_conversion.get_start_of_spc_date(first_spc_date_string)
    )
    training_option_dict[trainval_io.LAST_STORM_TIME_KEY] = (
        time_conversion.get_end_of_spc_date(last_spc_date_string)
    )

    if model_metadata_dict[cnn.USE_2D3D_CONVOLUTION_KEY]:
        generator_object = testing_io.myrorss_generator_2d3d(
            option_dict=training_option_dict, num_examples_total=num_examples)
    else:
        generator_object = testing_io.generator_2d_or_3d(
            option_dict=training_option_dict, num_examples_total=num_examples)

    storm_ids = []
    storm_times_unix_sec = numpy.array([], dtype=int)
    target_values = numpy.array([], dtype=int)
    list_of_predictor_matrices = None

    for _ in range(len(example_file_names)):
        try:
            this_storm_object_dict = next(generator_object)
            print SEPARATOR_STRING
        except StopIteration:
            break

        storm_ids += this_storm_object_dict[testing_io.STORM_IDS_KEY]
        storm_times_unix_sec = numpy.concatenate((
            storm_times_unix_sec,
            this_storm_object_dict[testing_io.STORM_TIMES_KEY]
        ))

        these_target_values = this_storm_object_dict[
            testing_io.TARGET_ARRAY_KEY]
        if len(these_target_values.shape) > 1:
            these_target_values = numpy.argmax(these_target_values, axis=1)

        target_values = numpy.concatenate((
            target_values, these_target_values))

        these_predictor_matrices = this_storm_object_dict[
            testing_io.INPUT_MATRICES_KEY]

        if list_of_predictor_matrices is None:
            list_of_predictor_matrices = copy.deepcopy(these_predictor_matrices)
        else:
            for k in range(len(list_of_predictor_matrices)):
                list_of_predictor_matrices[k] = numpy.concatenate((
                    list_of_predictor_matrices[k], these_predictor_matrices[k]
                ))

    if model_metadata_dict[cnn.USE_2D3D_CONVOLUTION_KEY]:
        prediction_function = permutation.prediction_function_2d3d_cnn
        predictor_names_by_matrix = [
            training_option_dict[trainval_io.RADAR_FIELDS_KEY],
            [radar_utils.REFL_NAME]
        ]

        if len(list_of_predictor_matrices) == 3:
            predictor_names_by_matrix.append(
                training_option_dict[trainval_io.SOUNDING_FIELDS_KEY]
            )

    else:
        predictor_names_by_matrix = [
            training_option_dict[trainval_io.RADAR_FIELDS_KEY]
        ]

        if len(list_of_predictor_matrices) == 2:
            predictor_names_by_matrix.append(
                training_option_dict[trainval_io.SOUNDING_FIELDS_KEY]
            )

        num_radar_dimensions = len(list_of_predictor_matrices[0].shape) - 2
        if num_radar_dimensions == 2:
            prediction_function = permutation.prediction_function_2d_cnn
        else:
            prediction_function = permutation.prediction_function_3d_cnn

    result_dict = permutation.run_permutation_test(
        model_object=model_object,
        list_of_input_matrices=list_of_predictor_matrices,
        predictor_names_by_matrix=predictor_names_by_matrix,
        target_values=target_values, prediction_function=prediction_function,
        cost_function=permutation.cross_entropy_function)

    result_dict[permutation.MODEL_FILE_KEY] = model_file_name
    result_dict[permutation.TARGET_VALUES_KEY] = target_values
    result_dict[permutation.STORM_IDS_KEY] = storm_ids
    result_dict[permutation.STORM_TIMES_KEY] = storm_times_unix_sec

    print 'Writing results to: "{0:s}"...'.format(output_file_name)
    permutation.write_results(
        result_dict=result_dict, pickle_file_name=output_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        top_example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        first_spc_date_string=getattr(
            INPUT_ARG_OBJECT, FIRST_SPC_DATE_ARG_NAME),
        last_spc_date_string=getattr(INPUT_ARG_OBJECT, LAST_SPC_DATE_ARG_NAME),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        class_fraction_keys=numpy.array(
            getattr(INPUT_ARG_OBJECT, CLASS_FRACTION_KEYS_ARG_NAME), dtype=int),
        class_fraction_values=numpy.array(
            getattr(INPUT_ARG_OBJECT, CLASS_FRACTION_VALUES_ARG_NAME),
            dtype=float),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
