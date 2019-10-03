"""Runs novelty detection.

--- REFERENCE ---

Wagstaff, K., and J. Lee: "Interpretable discovery in large image data sets."
arXiv e-prints, 1806, https://arxiv.org/abs/1806.08340
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse
import numpy
from keras import backend as K
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import testing_io
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.deep_learning import novelty_detection
from gewittergefahr.deep_learning import model_interpretation

# TODO(thunderhoser): Maybe allow this script to handle soundings or multiple
# radar tensors?

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
    allow_soft_placement=False
)))

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TRIAL_STORM_IDS_KEY = novelty_detection.TRIAL_STORM_IDS_KEY
TRIAL_STORM_TIMES_KEY = novelty_detection.TRIAL_STORM_TIMES_KEY
BASELINE_STORM_IDS_KEY = novelty_detection.BASELINE_STORM_IDS_KEY
BASELINE_STORM_TIMES_KEY = novelty_detection.BASELINE_STORM_TIMES_KEY
NUM_NOVEL_EXAMPLES_KEY = 'num_novel_examples'

CNN_FILE_ARG_NAME = 'input_cnn_file_name'
UPCONVNET_FILE_ARG_NAME = 'input_upconvnet_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
BASELINE_FILE_ARG_NAME = 'input_baseline_metafile_name'
TRIAL_FILE_ARG_NAME = 'input_trial_metafile_name'
NUM_BASELINE_EX_ARG_NAME = 'num_baseline_examples'
NUM_TRIAL_EX_ARG_NAME = 'num_trial_examples'
NUM_NOVEL_EX_ARG_NAME = 'num_novel_examples'
FEATURE_LAYER_ARG_NAME = 'cnn_feature_layer_name'
PERCENT_VARIANCE_ARG_NAME = 'percent_variance_to_keep'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

CNN_FILE_HELP_STRING = (
    'Path to file with trained CNN.  Will be read by `cnn.read_model`.')

UPCONVNET_FILE_HELP_STRING = (
    'Path to file with trained upconvnet.  Will be read by `cnn.read_model`.')

EXAMPLE_DIR_HELP_STRING = (
    'Name of top-level directory with input examples (will be fed to CNN).  '
    'Files therein will be found by `input_examples.find_example_file` and read'
    ' by `input_examples.read_example_file`.')

BASELINE_FILE_HELP_STRING = (
    'Path to file with metadata for baseline examples (storm objects).  Will be'
    ' read by `storm_tracking_io.read_ids_and_times`.')

TRIAL_FILE_HELP_STRING = (
    'Path to file with metadata for trial examples (storm objects).  Will be '
    'read by `storm_tracking_io.read_ids_and_times`.')

NUM_BASELINE_EX_HELP_STRING = (
    'Number of examples (storm objects) in baseline set.  Will use the first '
    '`{0:s}` examples from `{1:s}`.  If you want to use all examples in '
    '`{1:s}`, leave this argument alone.'
).format(NUM_BASELINE_EX_ARG_NAME, BASELINE_FILE_ARG_NAME)

NUM_TRIAL_EX_HELP_STRING = (
    'Number of examples (storm objects) in trial set.  Will use the first '
    '`{0:s}` examples from `{1:s}`.  If you want to use all examples in '
    '`{1:s}`, leave this argument alone.'
).format(NUM_BASELINE_EX_ARG_NAME, TRIAL_FILE_ARG_NAME)

NUM_NOVEL_EX_HELP_STRING = (
    'Number of novel examples (storm objects) to find.  This script will find '
    'the `{0:s}` most novel examples in the trial set.  To find the novelty of '
    'every trial example, leave this argument alone.'
).format(NUM_NOVEL_EX_ARG_NAME)

FEATURE_LAYER_HELP_STRING = (
    'Name of feature layer in CNN.  Outputs of this layer will be inputs to the'
    ' upconvnet.')

PERCENT_VARIANCE_HELP_STRING = (
    'Percentage of variance to retain in SVD (singular-value decomposition) '
    'model.')

OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Will be written by '
    '`novelty_detection.write_standard_file`.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + CNN_FILE_ARG_NAME, type=str, required=True,
    help=CNN_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + UPCONVNET_FILE_ARG_NAME, type=str, required=True,
    help=UPCONVNET_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + BASELINE_FILE_ARG_NAME, type=str, required=True,
    help=BASELINE_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TRIAL_FILE_ARG_NAME, type=str, required=True,
    help=TRIAL_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_BASELINE_EX_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_BASELINE_EX_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_TRIAL_EX_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_TRIAL_EX_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_NOVEL_EX_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_NOVEL_EX_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FEATURE_LAYER_ARG_NAME, type=str, required=True,
    help=FEATURE_LAYER_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PERCENT_VARIANCE_ARG_NAME, type=float, required=False,
    default=novelty_detection.DEFAULT_PCT_VARIANCE_TO_KEEP,
    help=PERCENT_VARIANCE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _check_dimensions(cnn_model_object, upconvnet_model_object):
    """Ensures that CNN and upconvnet dimensions work together.

    :param cnn_model_object: Trained CNN (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param upconvnet_model_object: Same but for trained upconvnet.
    :raises: ValueError: if dimensions of first CNN input tensor != dimensions
        of upconvnet output.
    """

    ucn_output_dimensions = numpy.array(
        upconvnet_model_object.output.get_shape().as_list()[1:], dtype=int
    )

    if isinstance(cnn_model_object.input, list):
        first_cnn_input_tensor = cnn_model_object.input[0]
    else:
        first_cnn_input_tensor = cnn_model_object.input

    cnn_input_dimensions = numpy.array(
        first_cnn_input_tensor.get_shape().as_list()[1:], dtype=int
    )

    if not numpy.array_equal(cnn_input_dimensions, ucn_output_dimensions):
        error_string = (
            'Dimensions of first CNN input tensor ({0:s}) should equal '
            'dimensions of upconvnet output ({1:s}).'
        ).format(str(cnn_input_dimensions), str(ucn_output_dimensions))

        raise ValueError(error_string)


def _filter_examples(
        trial_full_id_strings, trial_times_unix_sec, num_trial_examples,
        baseline_full_id_strings, baseline_times_unix_sec,
        num_baseline_examples, num_novel_examples):
    """Filters trial and baseline examples (storm objects).

    T = original num trial examples
    t = desired num trial examples
    B = original num baseline examples
    b = desired num baseline examples

    :param trial_full_id_strings: length-T list of storm IDs.
    :param trial_times_unix_sec: length-T numpy array of storm times.
    :param num_trial_examples: t in the above discussion.  To keep all trial
        examples, make this non-positive.
    :param baseline_full_id_strings: length-B list of storm IDs.
    :param baseline_times_unix_sec: length-B numpy array of storm times.
    :param num_baseline_examples: b in the above discussion.  To keep all
        baseline examples, make this non-positive.
    :param num_novel_examples: Number of novel examples to find.
    :return: metadata_dict: Dictionary with the following keys.
    metadata_dict["trial_full_id_strings"]: length-t list of storm IDs.
    metadata_dict["trial_times_unix_sec"]: length-t numpy array of storm times.
    metadata_dict["baseline_full_id_strings"]: length-b list of storm IDs.
    metadata_dict["baseline_times_unix_sec"]: length-b numpy array of storm
        times.
    metadata_dict["num_novel_examples"]: Number of novel examples to find.
    """

    if 0 < num_trial_examples < len(trial_full_id_strings):
        trial_full_id_strings = trial_full_id_strings[:num_trial_examples]
        trial_times_unix_sec = trial_times_unix_sec[:num_trial_examples]

    num_trial_examples = len(trial_full_id_strings)
    if num_novel_examples <= 0:
        num_novel_examples = num_trial_examples + 0

    num_novel_examples = min([num_novel_examples, num_trial_examples])
    print('Number of novel examples to find: {0:d}'.format(num_novel_examples))

    bad_baseline_indices = tracking_utils.find_storm_objects(
        all_id_strings=baseline_full_id_strings,
        all_times_unix_sec=baseline_times_unix_sec,
        id_strings_to_keep=trial_full_id_strings,
        times_to_keep_unix_sec=trial_times_unix_sec, allow_missing=True)

    print('Removing {0:d} trial examples from baseline set...'.format(
        len(bad_baseline_indices)
    ))

    baseline_times_unix_sec = numpy.delete(
        baseline_times_unix_sec, bad_baseline_indices
    )
    baseline_full_id_strings = numpy.delete(
        numpy.array(baseline_full_id_strings), bad_baseline_indices
    ).tolist()

    if 0 < num_baseline_examples < len(baseline_full_id_strings):
        baseline_full_id_strings = baseline_full_id_strings[
            :num_baseline_examples]
        baseline_times_unix_sec = baseline_times_unix_sec[
            :num_baseline_examples]

    return {
        TRIAL_STORM_IDS_KEY: trial_full_id_strings,
        TRIAL_STORM_TIMES_KEY: trial_times_unix_sec,
        BASELINE_STORM_IDS_KEY: baseline_full_id_strings,
        BASELINE_STORM_TIMES_KEY: baseline_times_unix_sec,
        NUM_NOVEL_EXAMPLES_KEY: num_novel_examples
    }


def _run(cnn_file_name, upconvnet_file_name, top_example_dir_name,
         baseline_storm_metafile_name, trial_storm_metafile_name,
         num_baseline_examples, num_trial_examples, num_novel_examples,
         cnn_feature_layer_name, percent_variance_to_keep, output_file_name):
    """Runs novelty detection.

    This is effectively the main method.

    :param cnn_file_name: See documentation at top of file.
    :param upconvnet_file_name: Same.
    :param top_example_dir_name: Same.
    :param baseline_storm_metafile_name: Same.
    :param trial_storm_metafile_name: Same.
    :param num_baseline_examples: Same.
    :param num_trial_examples: Same.
    :param num_novel_examples: Same.
    :param cnn_feature_layer_name: Same.
    :param percent_variance_to_keep: Same.
    :param output_file_name: Same.
    :raises: ValueError: if dimensions of first CNN input matrix != dimensions
        of upconvnet output.
    """

    print('Reading trained CNN from: "{0:s}"...'.format(cnn_file_name))
    cnn_model_object = cnn.read_model(cnn_file_name)

    print('Reading trained upconvnet from: "{0:s}"...'.format(
        upconvnet_file_name))
    upconvnet_model_object = cnn.read_model(upconvnet_file_name)
    _check_dimensions(cnn_model_object=cnn_model_object,
                      upconvnet_model_object=upconvnet_model_object)

    print('Reading metadata for baseline examples from: "{0:s}"...'.format(
        baseline_storm_metafile_name))
    baseline_full_id_strings, baseline_times_unix_sec = (
        tracking_io.read_ids_and_times(baseline_storm_metafile_name)
    )

    print('Reading metadata for trial examples from: "{0:s}"...'.format(
        trial_storm_metafile_name))
    trial_full_id_strings, trial_times_unix_sec = (
        tracking_io.read_ids_and_times(trial_storm_metafile_name)
    )

    this_dict = _filter_examples(
        trial_full_id_strings=trial_full_id_strings,
        trial_times_unix_sec=trial_times_unix_sec,
        num_trial_examples=num_trial_examples,
        baseline_full_id_strings=baseline_full_id_strings,
        baseline_times_unix_sec=baseline_times_unix_sec,
        num_baseline_examples=num_baseline_examples,
        num_novel_examples=num_novel_examples)

    trial_full_id_strings = this_dict[TRIAL_STORM_IDS_KEY]
    trial_times_unix_sec = this_dict[TRIAL_STORM_TIMES_KEY]
    baseline_full_id_strings = this_dict[BASELINE_STORM_IDS_KEY]
    baseline_times_unix_sec = this_dict[BASELINE_STORM_TIMES_KEY]
    num_novel_examples = this_dict[NUM_NOVEL_EXAMPLES_KEY]

    cnn_metafile_name = '{0:s}/model_metadata.p'.format(
        os.path.split(cnn_file_name)[0]
    )

    print('Reading CNN metadata from: "{0:s}"...'.format(cnn_metafile_name))
    cnn_metadata_dict = cnn.read_model_metadata(cnn_metafile_name)
    print(SEPARATOR_STRING)

    baseline_predictor_matrices, _ = testing_io.read_specific_examples(
        top_example_dir_name=top_example_dir_name,
        desired_full_id_strings=baseline_full_id_strings,
        desired_times_unix_sec=baseline_times_unix_sec,
        option_dict=cnn_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY],
        list_of_layer_operation_dicts=cnn_metadata_dict[
            cnn.LAYER_OPERATIONS_KEY]
    )
    print(SEPARATOR_STRING)

    trial_predictor_matrices, _ = testing_io.read_specific_examples(
        top_example_dir_name=top_example_dir_name,
        desired_full_id_strings=trial_full_id_strings,
        desired_times_unix_sec=trial_times_unix_sec,
        option_dict=cnn_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY],
        list_of_layer_operation_dicts=cnn_metadata_dict[
            cnn.LAYER_OPERATIONS_KEY]
    )
    print(SEPARATOR_STRING)

    novelty_dict = novelty_detection.do_novelty_detection(
        baseline_predictor_matrices=baseline_predictor_matrices,
        trial_predictor_matrices=trial_predictor_matrices,
        cnn_model_object=cnn_model_object,
        cnn_feature_layer_name=cnn_feature_layer_name,
        upconvnet_model_object=upconvnet_model_object,
        num_novel_examples=num_novel_examples, multipass=False,
        percent_variance_to_keep=percent_variance_to_keep)

    print(SEPARATOR_STRING)
    print('Denormalizing inputs and outputs of novelty detection...')

    cnn_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY][
        trainval_io.SOUNDING_FIELDS_KEY] = None

    novelty_dict[novelty_detection.BASELINE_MATRIX_KEY] = (
        model_interpretation.denormalize_data(
            list_of_input_matrices=baseline_predictor_matrices[[0]],
            model_metadata_dict=cnn_metadata_dict)
    )

    novelty_dict[novelty_detection.TRIAL_MATRIX_KEY] = (
        model_interpretation.denormalize_data(
            list_of_input_matrices=trial_predictor_matrices[[0]],
            model_metadata_dict=cnn_metadata_dict)
    )

    novelty_dict[novelty_detection.UPCONV_MATRIX_KEY] = (
        model_interpretation.denormalize_data(
            list_of_input_matrices=[
                novelty_dict[novelty_detection.UPCONV_NORM_MATRIX_KEY]
            ],
            model_metadata_dict=cnn_metadata_dict)
    )[0]
    novelty_dict.pop(novelty_detection.UPCONV_NORM_MATRIX_KEY)

    novelty_dict[novelty_detection.UPCONV_SVD_MATRIX_KEY] = (
        model_interpretation.denormalize_data(
            list_of_input_matrices=[
                novelty_dict[novelty_detection.UPCONV_NORM_SVD_MATRIX_KEY]
            ],
            model_metadata_dict=cnn_metadata_dict)
    )[0]
    novelty_dict.pop(novelty_detection.UPCONV_NORM_SVD_MATRIX_KEY)

    novelty_dict = novelty_detection.add_metadata(
        novelty_dict=novelty_dict,
        baseline_full_id_strings=baseline_full_id_strings,
        baseline_times_unix_sec=baseline_times_unix_sec,
        trial_full_id_strings=trial_full_id_strings,
        trial_times_unix_sec=trial_times_unix_sec,
        cnn_file_name=cnn_file_name, upconvnet_file_name=upconvnet_file_name)

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    novelty_detection.write_standard_file(
        novelty_dict=novelty_dict, pickle_file_name=output_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        cnn_file_name=getattr(INPUT_ARG_OBJECT, CNN_FILE_ARG_NAME),
        upconvnet_file_name=getattr(INPUT_ARG_OBJECT, UPCONVNET_FILE_ARG_NAME),
        top_example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        baseline_storm_metafile_name=getattr(
            INPUT_ARG_OBJECT, BASELINE_FILE_ARG_NAME),
        trial_storm_metafile_name=getattr(
            INPUT_ARG_OBJECT, TRIAL_FILE_ARG_NAME),
        num_baseline_examples=getattr(
            INPUT_ARG_OBJECT, NUM_BASELINE_EX_ARG_NAME),
        num_trial_examples=getattr(INPUT_ARG_OBJECT, NUM_TRIAL_EX_ARG_NAME),
        num_novel_examples=getattr(INPUT_ARG_OBJECT, NUM_NOVEL_EX_ARG_NAME),
        cnn_feature_layer_name=getattr(
            INPUT_ARG_OBJECT, FEATURE_LAYER_ARG_NAME),
        percent_variance_to_keep=getattr(
            INPUT_ARG_OBJECT, PERCENT_VARIANCE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
