"""Runs permutation test for predictor importance."""

import copy
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse
import numpy
from keras import backend as K
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import input_examples
from gewittergefahr.deep_learning import testing_io
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.deep_learning import permutation
from gewittergefahr.deep_learning import permutation_utils
from gewittergefahr.deep_learning import correlation

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
    allow_soft_placement=False
)))

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
NUM_EXAMPLES_PER_BATCH = 1000

MODEL_FILE_ARG_NAME = 'input_model_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
FIRST_DATE_ARG_NAME = 'first_spc_date_string'
LAST_DATE_ARG_NAME = 'last_spc_date_string'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
DO_BACKWARDS_ARG_NAME = 'do_backwards_test'
SEPARATE_HEIGHTS_ARG_NAME = 'separate_radar_heights'
DOWNSAMPLING_KEYS_ARG_NAME = 'downsampling_keys'
DOWNSAMPLING_VALUES_ARG_NAME = 'downsampling_values'
NUM_BOOTSTRAP_ARG_NAME = 'num_bootstrap_reps'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

MODEL_FILE_HELP_STRING = (
    'Path to file with trained CNN.  Will be read by `cnn.read_model`.')

EXAMPLE_DIR_HELP_STRING = (
    'Name of top-level directory with input examples.  Files therein will be '
    'found by `input_examples.find_example_file` and read by '
    '`input_examples.read_example_file`.')

SPC_DATE_HELP_STRING = (
    'SPC date (format "yyyymmdd").  This script will use examples from the '
    'period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

NUM_EXAMPLES_HELP_STRING = (
    'Number of examples to use.  If you want to use all examples, make this a '
    'very large number.')

DO_BACKWARDS_HELP_STRING = (
    'Boolean flag.  If 1, will run backwards test.  If 0, will run forward '
    'test.')

SEPARATE_HEIGHTS_HELP_STRING = (
    'Boolean flag.  If 1, 3-D radar fields will be separated by height, so each'
    ' step will involve permuting one variable at one height.  If 0, each step '
    'will involve permuting one variable at all heights.')

DOWNSAMPLING_KEYS_HELP_STRING = (
    'Keys (class labels) used to create dictionary for '
    '`deep_learning_utils.sample_by_class`.  If you do not want downsampling by'
    ' class, leave this alone.')

DOWNSAMPLING_VALUES_HELP_STRING = (
    'Values (class fractions) used to create dictionary for '
    '`deep_learning_utils.sample_by_class`.  If you do not want downsampling by'
    ' class, leave this alone.')

NUM_BOOTSTRAP_HELP_STRING = (
    'Number of bootstrap replicates (used to compute the cost function after '
    'each permutation).  If you do not want bootstrapping, make this <= 1.')

OUTPUT_FILE_HELP_STRING = (
    'Path to output (Pickle) file.  Will be written by '
    '`permutation_utils.write_results`.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=True,
    help=NUM_EXAMPLES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + DO_BACKWARDS_ARG_NAME, type=int, required=False, default=0,
    help=DO_BACKWARDS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SEPARATE_HEIGHTS_ARG_NAME, type=int, required=False, default=0,
    help=SEPARATE_HEIGHTS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + DOWNSAMPLING_KEYS_ARG_NAME, type=int, nargs='+',
    required=False, default=[0], help=DOWNSAMPLING_KEYS_HELP_STRING
)

INPUT_ARG_PARSER.add_argument(
    '--' + DOWNSAMPLING_VALUES_ARG_NAME, type=float, nargs='+',
    required=False, default=[0.], help=DOWNSAMPLING_VALUES_HELP_STRING
)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_BOOTSTRAP_ARG_NAME, type=int, required=False, default=1,
    help=NUM_BOOTSTRAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _run(model_file_name, top_example_dir_name, first_spc_date_string,
         last_spc_date_string, num_examples, do_backwards_test,
         separate_radar_heights, downsampling_keys, downsampling_values,
         num_bootstrap_reps, output_file_name):
    """Runs permutation test for predictor importance.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param top_example_dir_name: Same.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
    :param num_examples: Same.
    :param do_backwards_test: Same.
    :param separate_radar_heights: Same.
    :param downsampling_keys: Same.
    :param downsampling_values: Same.
    :param num_bootstrap_reps: Same.
    :param output_file_name: Same.
    """

    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = cnn.read_model(model_file_name)

    metafile_name = cnn.find_metafile(model_file_name=model_file_name)
    print('Reading metadata from: "{0:s}"...'.format(metafile_name))
    cnn_metadata_dict = cnn.read_model_metadata(metafile_name)

    if len(downsampling_keys) > 1:
        downsampling_dict = dict(list(zip(
            downsampling_keys, downsampling_values
        )))
    else:
        downsampling_dict = None

    training_option_dict = cnn_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    training_option_dict[trainval_io.SAMPLING_FRACTIONS_KEY] = downsampling_dict

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
    training_option_dict[trainval_io.NUM_EXAMPLES_PER_BATCH_KEY] = (
        NUM_EXAMPLES_PER_BATCH
    )

    if cnn_metadata_dict[cnn.LAYER_OPERATIONS_KEY] is not None:
        generator_object = testing_io.gridrad_generator_2d_reduced(
            option_dict=training_option_dict,
            desired_num_examples=num_examples,
            list_of_operation_dicts=cnn_metadata_dict[
                cnn.LAYER_OPERATIONS_KEY]
        )

    elif cnn_metadata_dict[cnn.CONV_2D3D_KEY]:
        generator_object = testing_io.myrorss_generator_2d3d(
            option_dict=training_option_dict,
            desired_num_examples=num_examples)
    else:
        generator_object = testing_io.generator_2d_or_3d(
            option_dict=training_option_dict,
            desired_num_examples=num_examples)

    full_storm_id_strings = []
    storm_times_unix_sec = numpy.array([], dtype=int)
    target_values = numpy.array([], dtype=int)
    predictor_matrices = None

    print(SEPARATOR_STRING)

    for _ in range(len(example_file_names)):
        try:
            this_storm_object_dict = next(generator_object)
            print(SEPARATOR_STRING)
        except StopIteration:
            break

        full_storm_id_strings += this_storm_object_dict[testing_io.FULL_IDS_KEY]
        storm_times_unix_sec = numpy.concatenate((
            storm_times_unix_sec,
            this_storm_object_dict[testing_io.STORM_TIMES_KEY]
        ))

        these_target_values = this_storm_object_dict[
            testing_io.TARGET_ARRAY_KEY
        ]
        if len(these_target_values.shape) > 1:
            these_target_values = numpy.argmax(these_target_values, axis=1)

        target_values = numpy.concatenate((
            target_values, these_target_values
        ))

        these_predictor_matrices = this_storm_object_dict[
            testing_io.INPUT_MATRICES_KEY]

        if predictor_matrices is None:
            predictor_matrices = copy.deepcopy(these_predictor_matrices)
        else:
            for k in range(len(predictor_matrices)):
                predictor_matrices[k] = numpy.concatenate((
                    predictor_matrices[k], these_predictor_matrices[k]
                ))

    print(SEPARATOR_STRING)
    correlation_matrix, predictor_names = correlation.get_pearson_correlations(
        predictor_matrices=predictor_matrices,
        cnn_metadata_dict=cnn_metadata_dict,
        separate_radar_heights=separate_radar_heights)
    print(SEPARATOR_STRING)

    num_predictors = len(predictor_names)

    for i in range(num_predictors):
        for j in range(i, num_predictors):
            print((
                'Pearson correlation between "{0:s}" and "{1:s}" = {2:.3f}'
            ).format(
                predictor_names[i], predictor_names[j], correlation_matrix[i, j]
            ))

    print(SEPARATOR_STRING)

    if do_backwards_test:
        result_dict = permutation.run_backwards_test(
            model_object=model_object, predictor_matrices=predictor_matrices,
            target_values=target_values, cnn_metadata_dict=cnn_metadata_dict,
            cost_function=permutation_utils.negative_auc_function,
            separate_radar_heights=separate_radar_heights,
            num_bootstrap_reps=num_bootstrap_reps)
    else:
        result_dict = permutation.run_forward_test(
            model_object=model_object, predictor_matrices=predictor_matrices,
            target_values=target_values, cnn_metadata_dict=cnn_metadata_dict,
            cost_function=permutation_utils.negative_auc_function,
            separate_radar_heights=separate_radar_heights,
            num_bootstrap_reps=num_bootstrap_reps)

    print(SEPARATOR_STRING)

    result_dict[permutation_utils.MODEL_FILE_KEY] = model_file_name
    result_dict[permutation_utils.TARGET_VALUES_KEY] = target_values
    result_dict[permutation_utils.FULL_IDS_KEY] = full_storm_id_strings
    result_dict[permutation_utils.STORM_TIMES_KEY] = storm_times_unix_sec

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    permutation_utils.write_results(
        result_dict=result_dict, pickle_file_name=output_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        top_example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        first_spc_date_string=getattr(
            INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_spc_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        do_backwards_test=bool(getattr(
            INPUT_ARG_OBJECT, DO_BACKWARDS_ARG_NAME
        )),
        separate_radar_heights=bool(getattr(
            INPUT_ARG_OBJECT, SEPARATE_HEIGHTS_ARG_NAME
        )),
        downsampling_keys=numpy.array(
            getattr(INPUT_ARG_OBJECT, DOWNSAMPLING_KEYS_ARG_NAME), dtype=int),
        downsampling_values=numpy.array(
            getattr(INPUT_ARG_OBJECT, DOWNSAMPLING_VALUES_ARG_NAME),
            dtype=float
        ),
        num_bootstrap_reps=getattr(INPUT_ARG_OBJECT, NUM_BOOTSTRAP_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
