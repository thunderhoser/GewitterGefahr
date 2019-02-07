"""Runs novelty detection.

--- REFERENCE ---

Wagstaff, K., and J. Lee: "Interpretable discovery in large image data sets."
arXiv e-prints, 1806, https://arxiv.org/abs/1806.08340
"""

import os.path
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

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)))

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

CNN_FILE_ARG_NAME = 'input_cnn_file_name'
UPCONVNET_FILE_ARG_NAME = 'input_upconvnet_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
BASELINE_FILE_ARG_NAME = 'input_baseline_metafile_name'
TRIAL_FILE_ARG_NAME = 'input_trial_metafile_name'
NUM_BASELINE_EX_ARG_NAME = 'num_baseline_examples'
NUM_TRIAL_EX_ARG_NAME = 'num_trial_examples'
NUM_NOVEL_EX_ARG_NAME = 'num_novel_examples'
FEATURE_LAYER_ARG_NAME = 'cnn_feature_layer_name'
PERCENT_VARIANCE_ARG_NAME = 'percent_svd_variance_to_keep'
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


def _run(cnn_file_name, upconvnet_file_name, top_example_dir_name,
         baseline_storm_metafile_name, trial_storm_metafile_name,
         num_baseline_examples, num_trial_examples, num_novel_examples,
         cnn_feature_layer_name, percent_svd_variance_to_keep,
         output_file_name):
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
    :param percent_svd_variance_to_keep: Same.
    :param output_file_name: Same.
    :raises: ValueError: if dimensions of first CNN input matrix != dimensions
        of upconvnet output.
    """

    print 'Reading trained CNN from: "{0:s}"...'.format(cnn_file_name)
    cnn_model_object = cnn.read_model(cnn_file_name)
    cnn_metafile_name = '{0:s}/model_metadata.p'.format(
        os.path.split(cnn_file_name)[0]
    )

    print 'Reading trained upconvnet from: "{0:s}"...'.format(
        upconvnet_file_name)
    upconvnet_model_object = cnn.read_model(upconvnet_file_name)

    # ucn_output_dimensions = numpy.array(
    #     upconvnet_model_object.output.get_shape().as_list()[1:], dtype=int
    # )

    if isinstance(cnn_model_object.input, list):
        first_cnn_input_tensor = cnn_model_object.input[0]
    else:
        first_cnn_input_tensor = cnn_model_object.input

    cnn_input_dimensions = numpy.array(
        first_cnn_input_tensor.get_shape().as_list()[1:], dtype=int
    )

    # if not numpy.array_equal(cnn_input_dimensions, ucn_output_dimensions):
    #     error_string = (
    #         'Dimensions of first CNN input matrix ({0:s}) should equal '
    #         'dimensions of upconvnet output ({1:s}).'
    #     ).format(str(cnn_input_dimensions), str(ucn_output_dimensions))
    #
    #     raise ValueError(error_string)

    print 'Reading CNN metadata from: "{0:s}"...'.format(cnn_metafile_name)
    cnn_metadata_dict = cnn.read_model_metadata(cnn_metafile_name)

    print 'Reading metadata for baseline examples from: "{0:s}"...'.format(
        baseline_storm_metafile_name)
    baseline_storm_ids, baseline_times_unix_sec = (
        tracking_io.read_ids_and_times(baseline_storm_metafile_name)
    )

    print 'Reading metadata for trial examples from: "{0:s}"...'.format(
        trial_storm_metafile_name)
    trial_storm_ids, trial_times_unix_sec = tracking_io.read_ids_and_times(
        trial_storm_metafile_name)

    if 0 < num_baseline_examples < len(baseline_storm_ids):
        baseline_storm_ids = baseline_storm_ids[:num_baseline_examples]
        baseline_times_unix_sec = baseline_times_unix_sec[
            :num_baseline_examples]

    if 0 < num_trial_examples < len(trial_storm_ids):
        trial_storm_ids = trial_storm_ids[:num_trial_examples]
        trial_times_unix_sec = trial_times_unix_sec[:num_trial_examples]

    if num_novel_examples <= 0:
        num_novel_examples = num_trial_examples + 0
    num_novel_examples = min([num_novel_examples, num_trial_examples])

    bad_baseline_indices = tracking_utils.find_storm_objects(
        all_storm_ids=baseline_storm_ids,
        all_times_unix_sec=baseline_times_unix_sec,
        storm_ids_to_keep=trial_storm_ids,
        times_to_keep_unix_sec=trial_times_unix_sec, allow_missing=True)

    print 'Removing {0:d} trial examples from baseline set...'.format(
        len(bad_baseline_indices)
    )

    baseline_storm_ids = numpy.delete(
        numpy.array(baseline_storm_ids), bad_baseline_indices)
    baseline_storm_ids = baseline_storm_ids.tolist()
    baseline_times_unix_sec = numpy.delete(
        baseline_times_unix_sec, bad_baseline_indices)

    # num_baseline_examples = len(baseline_storm_ids)
    num_trial_examples = len(trial_storm_ids)

    print SEPARATOR_STRING

    list_of_baseline_input_matrices, _ = testing_io.read_specific_examples(
        top_example_dir_name=top_example_dir_name,
        desired_storm_ids=baseline_storm_ids,
        desired_times_unix_sec=baseline_times_unix_sec,
        option_dict=cnn_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY],
        list_of_layer_operation_dicts=cnn_metadata_dict[
            cnn.LAYER_OPERATIONS_KEY]
    )

    print SEPARATOR_STRING

    list_of_trial_input_matrices, _ = testing_io.read_specific_examples(
        top_example_dir_name=top_example_dir_name,
        desired_storm_ids=trial_storm_ids,
        desired_times_unix_sec=trial_times_unix_sec,
        option_dict=cnn_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY],
        list_of_layer_operation_dicts=cnn_metadata_dict[
            cnn.LAYER_OPERATIONS_KEY]
    )

    print SEPARATOR_STRING

    novelty_dict = novelty_detection.do_novelty_detection(
        list_of_baseline_input_matrices=list_of_baseline_input_matrices,
        list_of_trial_input_matrices=list_of_trial_input_matrices,
        cnn_model_object=cnn_model_object,
        cnn_feature_layer_name=cnn_feature_layer_name,
        upconvnet_model_object=upconvnet_model_object,
        num_novel_examples=num_novel_examples, multipass=False,
        percent_svd_variance_to_keep=percent_svd_variance_to_keep)

    print SEPARATOR_STRING

    print 'Adding metadata to novelty-detection results...'
    novelty_dict = novelty_detection.add_metadata(
        novelty_dict=novelty_dict, baseline_storm_ids=baseline_storm_ids,
        baseline_storm_times_unix_sec=baseline_times_unix_sec,
        trial_storm_ids=trial_storm_ids,
        trial_storm_times_unix_sec=trial_times_unix_sec,
        cnn_file_name=cnn_file_name, upconvnet_file_name=upconvnet_file_name)

    print 'Denormalizing inputs and outputs of novelty detection...'

    novelty_dict[novelty_detection.BASELINE_INPUTS_KEY] = (
        model_interpretation.denormalize_data(
            list_of_input_matrices=novelty_dict[
                novelty_detection.BASELINE_INPUTS_KEY
            ],
            model_metadata_dict=cnn_metadata_dict)
    )

    novelty_dict[novelty_detection.TRIAL_INPUTS_KEY] = (
        model_interpretation.denormalize_data(
            list_of_input_matrices=novelty_dict[
                novelty_detection.TRIAL_INPUTS_KEY
            ],
            model_metadata_dict=cnn_metadata_dict)
    )

    cnn_metadata_dict[
        cnn.TRAINING_OPTION_DICT_KEY][trainval_io.SOUNDING_FIELDS_KEY] = None

    novelty_dict[novelty_detection.NOVEL_IMAGES_UPCONV_KEY] = (
        model_interpretation.denormalize_data(
            list_of_input_matrices=[
                novelty_dict[novelty_detection.NOVEL_IMAGES_UPCONV_KEY]
            ],
            model_metadata_dict=cnn_metadata_dict)
    )[0]

    novelty_dict[novelty_detection.NOVEL_IMAGES_UPCONV_SVD_KEY] = (
        model_interpretation.denormalize_data(
            list_of_input_matrices=[
                novelty_dict[novelty_detection.NOVEL_IMAGES_UPCONV_SVD_KEY]
            ],
            model_metadata_dict=cnn_metadata_dict)
    )[0]

    print 'Writing results to: "{0:s}"...'.format(output_file_name)
    novelty_detection.write_standard_file(novelty_dict=novelty_dict,
                                          pickle_file_name=output_file_name)


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
        percent_svd_variance_to_keep=getattr(
            INPUT_ARG_OBJECT, PERCENT_VARIANCE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
