"""Runs permutation test with specific examples (storm objects)."""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse
from keras import backend as K
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import testing_io
from gewittergefahr.deep_learning import permutation
from gewittergefahr.deep_learning import correlation

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
    allow_soft_placement=False
)))

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MODEL_FILE_ARG_NAME = 'input_model_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
STORM_METAFILE_ARG_NAME = 'input_storm_metafile_name'
DO_BACKWARDS_ARG_NAME = 'do_backwards_test'
SEPARATE_HEIGHTS_ARG_NAME = 'separate_radar_heights'
NUM_BOOTSTRAP_ARG_NAME = 'num_bootstrap_reps'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

MODEL_FILE_HELP_STRING = (
    'Path to file with trained CNN.  Will be read by `cnn.read_model`.')

EXAMPLE_DIR_HELP_STRING = (
    'Name of top-level directory with examples.  Files therein will be '
    'found by `input_examples.find_example_file` and read by '
    '`input_examples.read_example_file`.')

STORM_METAFILE_HELP_STRING = (
    'Path to Pickle file with storm IDs and times (will be read by '
    '`storm_tracking_io.read_ids_and_times`).  Permutation test will be done '
    'with only these storm objects (examples).')

DO_BACKWARDS_HELP_STRING = (
    'Boolean flag.  If 1, will run backwards test.  If 0, will run forward '
    'test.')

SEPARATE_HEIGHTS_HELP_STRING = (
    'Boolean flag.  If 1, 3-D radar fields will be separated by height, so each'
    ' step will involve permuting one variable at one height.  If 0, each step '
    'will involve permuting one variable at all heights.')

NUM_BOOTSTRAP_HELP_STRING = (
    'Number of bootstrap replicates (used to compute the cost function after '
    'each permutation).  If you do not want bootstrapping, make this <= 1.')

OUTPUT_FILE_HELP_STRING = (
    'Path to output (Pickle) file.  Will be written by '
    '`permutation.write_results`.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + STORM_METAFILE_ARG_NAME, type=str, required=True,
    help=STORM_METAFILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + DO_BACKWARDS_ARG_NAME, type=int, required=False, default=0,
    help=DO_BACKWARDS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SEPARATE_HEIGHTS_ARG_NAME, type=int, required=False, default=0,
    help=SEPARATE_HEIGHTS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_BOOTSTRAP_ARG_NAME, type=int, required=False, default=1,
    help=NUM_BOOTSTRAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _run(model_file_name, top_example_dir_name, storm_metafile_name,
         do_backwards_test, separate_radar_heights, num_bootstrap_reps,
         output_file_name):
    """Runs permutation test with specific examples (storm objects).

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param top_example_dir_name: Same.
    :param storm_metafile_name: Same.
    :param do_backwards_test: Same.
    :param separate_radar_heights: Same.
    :param num_bootstrap_reps: Same.
    :param output_file_name: Same.
    """

    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = cnn.read_model(model_file_name)
    metafile_name = cnn.find_metafile(model_file_name=model_file_name)

    print('Reading metadata from: "{0:s}"...'.format(metafile_name))
    cnn_metadata_dict = cnn.read_model_metadata(metafile_name)
    training_option_dict = cnn_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]

    print('Reading storm metadata from: "{0:s}"...'.format(storm_metafile_name))
    full_storm_id_strings, storm_times_unix_sec = (
        tracking_io.read_ids_and_times(storm_metafile_name)
    )
    print(SEPARATOR_STRING)

    example_dict = testing_io.read_predictors_specific_examples(
        top_example_dir_name=top_example_dir_name,
        desired_full_id_strings=full_storm_id_strings,
        desired_times_unix_sec=storm_times_unix_sec,
        option_dict=training_option_dict,
        layer_operation_dicts=cnn_metadata_dict[cnn.LAYER_OPERATIONS_KEY]
    )
    print(SEPARATOR_STRING)

    predictor_matrices = example_dict[testing_io.INPUT_MATRICES_KEY]
    target_values = example_dict[testing_io.TARGET_ARRAY_KEY]

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
            cost_function=permutation.negative_auc_function,
            separate_radar_heights=separate_radar_heights,
            num_bootstrap_reps=num_bootstrap_reps)
    else:
        result_dict = permutation.run_forward_test(
            model_object=model_object, predictor_matrices=predictor_matrices,
            target_values=target_values, cnn_metadata_dict=cnn_metadata_dict,
            cost_function=permutation.negative_auc_function,
            separate_radar_heights=separate_radar_heights,
            num_bootstrap_reps=num_bootstrap_reps)

    print(SEPARATOR_STRING)

    result_dict[permutation.MODEL_FILE_KEY] = model_file_name
    result_dict[permutation.TARGET_VALUES_KEY] = target_values
    result_dict[permutation.FULL_IDS_KEY] = full_storm_id_strings
    result_dict[permutation.STORM_TIMES_KEY] = storm_times_unix_sec

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    permutation.write_results(
        result_dict=result_dict, pickle_file_name=output_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        top_example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        storm_metafile_name=getattr(INPUT_ARG_OBJECT, STORM_METAFILE_ARG_NAME),
        do_backwards_test=bool(getattr(
            INPUT_ARG_OBJECT, DO_BACKWARDS_ARG_NAME
        )),
        separate_radar_heights=bool(getattr(
            INPUT_ARG_OBJECT, SEPARATE_HEIGHTS_ARG_NAME
        )),
        num_bootstrap_reps=getattr(INPUT_ARG_OBJECT, NUM_BOOTSTRAP_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
