"""Plots upconvnet reconstructions of many examples (storm objects)."""

import os.path
import argparse
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import upconvnet
from gewittergefahr.deep_learning import testing_io
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.deep_learning import model_interpretation
from gewittergefahr.scripts import plot_input_examples

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

UPCONVNET_FILE_ARG_NAME = 'input_upconvnet_file_name'
STORM_METAFILE_ARG_NAME = 'input_storm_metafile_name'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

UPCONVNET_FILE_HELP_STRING = (
    'Path to file with trained upconvnet (will be read by `cnn.read_model`).')

STORM_METAFILE_HELP_STRING = (
    'Path to file with storm IDs and times (will be read by '
    '`storm_tracking_io.read_ids_and_times`).')

NUM_EXAMPLES_HELP_STRING = (
    'Number of examples to plot.  Will plot the first `{0:s}` examples (storm '
    'objects) from `{1:s}`.  If you want to plot all examples, leave this '
    'argument alone.'
).format(NUM_EXAMPLES_ARG_NAME, STORM_METAFILE_ARG_NAME)

EXAMPLE_DIR_HELP_STRING = (
    'Name of top-level directory with images (to be reconstructed).  Files '
    'therein will be found by `input_examples.find_example_file` and read by '
    '`input_examples.read_example_file`.')

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level output directory.  Figures will be saved here.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + UPCONVNET_FILE_ARG_NAME, type=str, required=True,
    help=UPCONVNET_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + STORM_METAFILE_ARG_NAME, type=str, required=True,
    help=STORM_METAFILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_EXAMPLES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(upconvnet_file_name, storm_metafile_name, num_examples,
         top_example_dir_name, top_output_dir_name):
    """Plots upconvnet reconstructions of many examples (storm objects).

    This is effectively the main method.

    :param upconvnet_file_name: See documentation at top of file.
    :param storm_metafile_name: Same.
    :param num_examples: Same.
    :param top_example_dir_name: Same.
    :param top_output_dir_name: Same.
    """

    print 'Reading trained upconvnet from: "{0:s}"...'.format(
        upconvnet_file_name)
    upconvnet_model_object = cnn.read_model(upconvnet_file_name)
    upconvnet_metafile_name = '{0:s}/model_metadata.p'.format(
        os.path.split(upconvnet_file_name)[0]
    )

    print 'Reading upconvnet metadata from: "{0:s}"...'.format(
        upconvnet_metafile_name)
    upconvnet_metadata_dict = upconvnet.read_model_metadata(
        upconvnet_metafile_name)
    cnn_file_name = upconvnet_metadata_dict[upconvnet.CNN_FILE_KEY]

    print 'Reading trained CNN from: "{0:s}"...'.format(cnn_file_name)
    cnn_model_object = cnn.read_model(cnn_file_name)
    cnn_metafile_name = '{0:s}/model_metadata.p'.format(
        os.path.split(cnn_file_name)[0]
    )

    print 'Reading CNN metadata from: "{0:s}"...'.format(cnn_metafile_name)
    cnn_metadata_dict = cnn.read_model_metadata(cnn_metafile_name)
    training_option_dict = cnn_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]

    print 'Reading storm IDs and times from: "{0:s}"...'.format(
        storm_metafile_name)
    storm_ids, storm_times_unix_sec = tracking_io.read_ids_and_times(
        storm_metafile_name)

    if 0 < num_examples < len(storm_ids):
        storm_ids = storm_ids[:num_examples]
        storm_times_unix_sec = storm_times_unix_sec[:num_examples]

    print SEPARATOR_STRING
    list_of_predictor_matrices = testing_io.read_specific_examples(
        desired_storm_ids=storm_ids,
        desired_times_unix_sec=storm_times_unix_sec,
        option_dict=training_option_dict,
        top_example_dir_name=top_example_dir_name,
        list_of_layer_operation_dicts=cnn_metadata_dict[
            cnn.LAYER_OPERATIONS_KEY]
    )[0]
    print SEPARATOR_STRING

    actual_radar_matrix = list_of_predictor_matrices[0]
    have_soundings = training_option_dict[trainval_io.SOUNDING_FIELDS_KEY]

    if have_soundings:
        sounding_matrix = list_of_predictor_matrices[-1]
    else:
        sounding_matrix = None

    feature_matrix = cnn.apply_2d_or_3d_cnn(
        model_object=cnn_model_object, radar_image_matrix=actual_radar_matrix,
        sounding_matrix=sounding_matrix, verbose=True, return_features=True,
        feature_layer_name=upconvnet_metadata_dict[
            upconvnet.CNN_FEATURE_LAYER_KEY]
    )
    print '\n'

    reconstructed_radar_matrix = upconvnet.apply_upconvnet(
        model_object=upconvnet_model_object, feature_matrix=feature_matrix,
        verbose=True)
    print '\n'

    print 'Denormalizing actual and reconstructed radar images...'

    cnn_metadata_dict[
        cnn.TRAINING_OPTION_DICT_KEY][trainval_io.SOUNDING_FIELDS_KEY] = None

    actual_radar_matrix = model_interpretation.denormalize_data(
        list_of_input_matrices=[actual_radar_matrix],
        model_metadata_dict=cnn_metadata_dict
    )[0]

    reconstructed_radar_matrix = model_interpretation.denormalize_data(
        list_of_input_matrices=[reconstructed_radar_matrix],
        model_metadata_dict=cnn_metadata_dict
    )[0]

    print SEPARATOR_STRING

    actual_output_dir_name = '{0:s}/actual_images'.format(top_output_dir_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=actual_output_dir_name)

    # TODO(thunderhoser): Calling a method in another script is hacky.  If this
    # method is going to be reused, should be in a module.
    plot_input_examples.plot_examples(
        list_of_predictor_matrices=[actual_radar_matrix], storm_ids=storm_ids,
        storm_times_unix_sec=storm_times_unix_sec,
        model_metadata_dict=cnn_metadata_dict,
        output_dir_name=actual_output_dir_name)
    print SEPARATOR_STRING

    reconstructed_output_dir_name = '{0:s}/reconstructed_images'.format(
        top_output_dir_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=reconstructed_output_dir_name)

    plot_input_examples.plot_examples(
        list_of_predictor_matrices=[reconstructed_radar_matrix],
        storm_ids=storm_ids, storm_times_unix_sec=storm_times_unix_sec,
        model_metadata_dict=cnn_metadata_dict,
        output_dir_name=reconstructed_output_dir_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        upconvnet_file_name=getattr(
            INPUT_ARG_OBJECT, UPCONVNET_FILE_ARG_NAME),
        storm_metafile_name=getattr(INPUT_ARG_OBJECT, STORM_METAFILE_ARG_NAME),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        top_example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
