"""Plots one or more radar images and their upconvnet reconstructions."""

import argparse
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import upconvnet
from gewittergefahr.deep_learning import testing_io
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.scripts import plot_input_examples as plot_examples

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

PREDICTION_FILE_ARG_NAME = 'input_prediction_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
COLOUR_MAP_ARG_NAME = 'diff_colour_map_name'
MAX_PERCENTILE_ARG_NAME = 'max_diff_percentile'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'
NUM_EXAMPLES_ARG_NAME = plot_examples.NUM_EXAMPLES_ARG_NAME
ALLOW_WHITESPACE_ARG_NAME = plot_examples.ALLOW_WHITESPACE_ARG_NAME
PLOT_PANEL_NAMES_ARG_NAME = plot_examples.PLOT_PANEL_NAMES_ARG_NAME
ADD_TITLES_ARG_NAME = plot_examples.ADD_TITLES_ARG_NAME
LABEL_CBARS_ARG_NAME = plot_examples.LABEL_CBARS_ARG_NAME
CBAR_LENGTH_ARG_NAME = plot_examples.CBAR_LENGTH_ARG_NAME

PREDICTION_FILE_HELP_STRING = (
    'Path to file with upconvnet predictions (reconstructed radar images).  '
    'Will be read by `upconvnet.read_predictions`.')

EXAMPLE_DIR_HELP_STRING = (
    'Name of top-level directory with actual (non-reconstructed) examples.  '
    'Files therein will be found by `input_examples.find_example_file` and read'
    ' by `input_examples.read_example_file`.')

COLOUR_MAP_HELP_STRING = (
    'Name of colour map (must be accepted by `pyplot.get_cmap`).  Will be used '
    'to plot differences (reconstructed minus actual).')

MAX_PERCENTILE_HELP_STRING = (
    'Determines max value in colour scheme for differences.  For each example '
    'and radar field, max value will be [q]th percentile of absolute '
    'differences, where q = `{0:s}`.'
).format(MAX_PERCENTILE_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Figures will be saved here.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_FILE_ARG_NAME, type=str, required=True,
    help=PREDICTION_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + COLOUR_MAP_ARG_NAME, type=str, required=False, default='seismic',
    help=COLOUR_MAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILE_ARG_NAME, type=float, required=False,
    default=99., help=MAX_PERCENTILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=False, default=-1,
    help=plot_examples.NUM_EXAMPLES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + ALLOW_WHITESPACE_ARG_NAME, type=int, required=False, default=1,
    help=plot_examples.ALLOW_WHITESPACE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_PANEL_NAMES_ARG_NAME, type=int, required=False, default=1,
    help=plot_examples.PLOT_PANEL_NAMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + ADD_TITLES_ARG_NAME, type=int, required=False, default=1,
    help=plot_examples.ADD_TITLES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LABEL_CBARS_ARG_NAME, type=int, required=False, default=0,
    help=plot_examples.LABEL_CBARS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CBAR_LENGTH_ARG_NAME, type=float, required=False, default=0.8,
    help=plot_examples.CBAR_LENGTH_HELP_STRING)


def _run(prediction_file_name, top_example_dir_name, diff_colour_map_name,
         max_diff_percentile, num_examples, allow_whitespace, plot_panel_names,
         add_titles, label_colour_bars, colour_bar_length, top_output_dir_name):
    """Plots one or more radar images and their upconvnet reconstructions.

    This is effectively the main method.

    :param prediction_file_name: See documentation at top of file.
    :param top_example_dir_name: Same.
    :param diff_colour_map_name: Same.
    :param max_diff_percentile: Same.
    :param num_examples: Same.
    :param allow_whitespace: Same.
    :param plot_panel_names: Same.
    :param add_titles: Same.
    :param label_colour_bars: Same.
    :param colour_bar_length: Same.
    :param top_output_dir_name: Same.
    """

    diff_colour_map_object = pyplot.get_cmap(diff_colour_map_name)

    # Read data.
    print('Reading reconstructed radar images from: "{0:s}"...'.format(
        prediction_file_name
    ))
    prediction_dict = upconvnet.read_predictions(prediction_file_name)

    reconstructed_radar_matrix = prediction_dict[
        upconvnet.RECON_IMAGE_MATRIX_KEY]
    full_storm_id_strings = prediction_dict[upconvnet.FULL_STORM_IDS_KEY]
    storm_times_unix_sec = prediction_dict[upconvnet.STORM_TIMES_KEY]

    if 0 < num_examples < len(full_storm_id_strings):
        reconstructed_radar_matrix = reconstructed_radar_matrix[
            :num_examples, ...]
        full_storm_id_strings = full_storm_id_strings[:num_examples]
        storm_times_unix_sec = storm_times_unix_sec[:num_examples]

    upconvnet_file_name = prediction_dict[upconvnet.UPCONVNET_FILE_KEY]
    upconvnet_metafile_name = cnn.find_metafile(upconvnet_file_name)

    print('Reading upconvnet metadata from: "{0:s}"...'.format(
        upconvnet_metafile_name
    ))
    upconvnet_metadata_dict = upconvnet.read_model_metadata(
        upconvnet_metafile_name)
    cnn_file_name = upconvnet_metadata_dict[upconvnet.CNN_FILE_KEY]
    cnn_metafile_name = cnn.find_metafile(cnn_file_name)

    print('Reading CNN metadata from: "{0:s}"...'.format(cnn_metafile_name))
    cnn_metadata_dict = cnn.read_model_metadata(cnn_metafile_name)

    training_option_dict = cnn_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    training_option_dict[trainval_io.NORMALIZATION_TYPE_KEY] = None
    training_option_dict[trainval_io.SOUNDING_FIELDS_KEY] = None
    cnn_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY] = training_option_dict

    print(SEPARATOR_STRING)
    example_dict = testing_io.read_predictors_specific_examples(
        top_example_dir_name=top_example_dir_name,
        desired_full_id_strings=full_storm_id_strings,
        desired_times_unix_sec=storm_times_unix_sec,
        option_dict=training_option_dict,
        layer_operation_dicts=cnn_metadata_dict[cnn.LAYER_OPERATIONS_KEY]
    )
    print(SEPARATOR_STRING)

    actual_radar_matrix = example_dict[testing_io.INPUT_MATRICES_KEY][0]

    plot_examples.plot_examples(
        list_of_predictor_matrices=[actual_radar_matrix],
        model_metadata_dict=cnn_metadata_dict,
        output_dir_name='{0:s}/actual_images'.format(top_output_dir_name),
        pmm_flag=False, plot_soundings=False, plot_radar_diffs=False,
        allow_whitespace=allow_whitespace, plot_panel_names=plot_panel_names,
        add_titles=add_titles, label_colour_bars=label_colour_bars,
        colour_bar_length=colour_bar_length,
        full_storm_id_strings=full_storm_id_strings,
        storm_times_unix_sec=storm_times_unix_sec)
    print(SEPARATOR_STRING)

    plot_examples.plot_examples(
        list_of_predictor_matrices=[reconstructed_radar_matrix],
        model_metadata_dict=cnn_metadata_dict,
        output_dir_name=
        '{0:s}/reconstructed_images'.format(top_output_dir_name),
        pmm_flag=False, plot_soundings=False, plot_radar_diffs=False,
        allow_whitespace=allow_whitespace, plot_panel_names=plot_panel_names,
        add_titles=add_titles, label_colour_bars=label_colour_bars,
        colour_bar_length=colour_bar_length,
        full_storm_id_strings=full_storm_id_strings,
        storm_times_unix_sec=storm_times_unix_sec)
    print(SEPARATOR_STRING)

    difference_matrix = reconstructed_radar_matrix - actual_radar_matrix

    plot_examples.plot_examples(
        list_of_predictor_matrices=[difference_matrix],
        model_metadata_dict=cnn_metadata_dict,
        output_dir_name='{0:s}/differences'.format(top_output_dir_name),
        pmm_flag=False, plot_soundings=False, plot_radar_diffs=True,
        diff_colour_map_object=diff_colour_map_object,
        max_diff_percentile=max_diff_percentile,
        allow_whitespace=allow_whitespace, plot_panel_names=plot_panel_names,
        add_titles=add_titles, label_colour_bars=label_colour_bars,
        colour_bar_length=colour_bar_length,
        full_storm_id_strings=full_storm_id_strings,
        storm_times_unix_sec=storm_times_unix_sec)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_FILE_ARG_NAME
        ),
        top_example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        diff_colour_map_name=getattr(INPUT_ARG_OBJECT, COLOUR_MAP_ARG_NAME),
        max_diff_percentile=getattr(
            INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME
        ),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        allow_whitespace=bool(getattr(
            INPUT_ARG_OBJECT, ALLOW_WHITESPACE_ARG_NAME
        )),
        plot_panel_names=bool(getattr(
            INPUT_ARG_OBJECT, PLOT_PANEL_NAMES_ARG_NAME
        )),
        add_titles=bool(getattr(INPUT_ARG_OBJECT, ADD_TITLES_ARG_NAME)),
        label_colour_bars=bool(getattr(
            INPUT_ARG_OBJECT, LABEL_CBARS_ARG_NAME
        )),
        colour_bar_length=getattr(INPUT_ARG_OBJECT, CBAR_LENGTH_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
