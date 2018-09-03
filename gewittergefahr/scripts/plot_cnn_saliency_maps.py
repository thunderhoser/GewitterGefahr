"""Plots saliency maps for a CNN (convolutional neural network)."""

import os.path
import argparse
import numpy
from gewittergefahr.gg_utils import soundings
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import saliency_maps
from gewittergefahr.deep_learning import storm_images
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.plotting import saliency_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
SOUNDING_HEIGHTS_M_AGL = soundings.DEFAULT_HEIGHT_LEVELS_M_AGL + 0

INPUT_FILE_ARG_NAME = 'input_file_name'
MAX_COLOUR_VALUE_ARG_NAME = 'max_colour_value'
MAX_COLOUR_PRCTILE_ARG_NAME = 'max_colour_percentile'
NUM_PANEL_ROWS_ARG_NAME = 'num_panel_rows'
TEMP_DIRECTORY_ARG_NAME = 'temp_directory_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by `saliency_maps.read_file`.')

MAX_COLOUR_VALUE_HELP_STRING = (
    'Max saliency value in colour scheme.  Minimum saliency in colour scheme '
    'will be -1 * `{0:s}`.  To use `{1:s}` instead, leave this argument alone.'
).format(MAX_COLOUR_VALUE_ARG_NAME, MAX_COLOUR_PRCTILE_ARG_NAME)

MAX_COLOUR_PRCTILE_HELP_STRING = (
    'Max saliency value in colour scheme will be the `{0:s}`th percentile of '
    'absolute values in `{1:s}` (over all storm objects, radar field/height '
    'pairs, and sounding field/height pairs).  Minimum saliency in colour '
    'scheme will be -1 * max value.  To use `{2:s}` instead, leave this '
    'argument alone.'
).format(MAX_COLOUR_PRCTILE_ARG_NAME, INPUT_FILE_ARG_NAME,
         MAX_COLOUR_VALUE_ARG_NAME)

NUM_PANEL_ROWS_HELP_STRING = (
    'Number of panel rows in each radar figure.  If radar images are 3-D, there'
    ' will be one figure per storm object and field, containing all heights.  '
    'If radar images are 2-D, there will be one figure per storm object, '
    'containing all field/height pairs.')

TEMP_DIRECTORY_HELP_STRING = (
    'Name of temporary directory.  If `{0:s}` contains soundings, this script '
    'will plot one sounding figure per storm object, containing the actual '
    'sounding and saliency map.  The actual sounding and saliency map will be '
    'saved to `{1:s}` before they are concatenated, then deleted.  To use the '
    'default temp directory on the local machine, leave this argument alone.')

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_COLOUR_VALUE_ARG_NAME, type=float, required=False, default=-1,
    help=MAX_COLOUR_VALUE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_COLOUR_PRCTILE_ARG_NAME, type=float, required=False, default=-1,
    help=MAX_COLOUR_PRCTILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_PANEL_ROWS_ARG_NAME, type=int, required=False, default=3,
    help=NUM_PANEL_ROWS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TEMP_DIRECTORY_ARG_NAME, type=str, required=False, default='',
    help=TEMP_DIRECTORY_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(input_file_name, max_colour_value, max_colour_percentile,
         num_panel_rows, temp_directory_name, output_dir_name):
    """Plots saliency maps for a CNN (convolutional neural network).

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param max_colour_value: Same.
    :param max_colour_percentile: Same.
    :param num_panel_rows: Same.
    :param temp_directory_name: Same.
    :param output_dir_name: Same.
    :raises: ValueError: if both `max_colour_value` and `max_colour_percentile`
        are non-positive.
    :raises: TypeError: if saliency maps come from a model that does 2-D and 3-D
        convolution.
    """

    # Check input args.
    if max_colour_value <= 0:
        max_colour_value = None
    if max_colour_percentile <= 0:
        max_colour_percentile = None
    if temp_directory_name == '':
        temp_directory_name = None

    if max_colour_value is None and max_colour_percentile is None:
        raise ValueError(
            'max_colour_value and max_colour_percentile cannot both be None.')

    # Read saliency maps.
    print 'Reading saliency maps from: "{0:s}"...'.format(input_file_name)
    (list_of_input_matrices, list_of_saliency_matrices, saliency_metadata_dict
    ) = saliency_maps.read_file(input_file_name)

    if max_colour_value is None:
        all_saliency_values = numpy.array([])
        for this_matrix in list_of_saliency_matrices:
            all_saliency_values = numpy.concatenate(
                (all_saliency_values, numpy.ravel(this_matrix)))

        max_colour_value = numpy.percentile(
            numpy.absolute(all_saliency_values), max_colour_percentile)
        del all_saliency_values

    print 'Max saliency value in colour scheme = {0:.3e}\n'.format(
        max_colour_value)
    saliency_option_dict = {
        saliency_plotting.MAX_COLOUR_VALUE_KEY: max_colour_value
    }

    # Read metadata for the CNN that generated the saliency maps.
    model_file_name = saliency_metadata_dict[saliency_maps.MODEL_FILE_NAME_KEY]
    model_metadata_file_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0])

    print 'Reading metadata from: "{0:s}"...'.format(model_metadata_file_name)
    model_metadata_dict = cnn.read_model_metadata(model_metadata_file_name)
    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]

    if model_metadata_dict[cnn.USE_2D3D_CONVOLUTION_KEY]:
        raise TypeError('This script cannot yet handle models that do 2-D and '
                        '3-D convolution.')

    # Plot saliency maps.
    training_radar_file_name_matrix = training_option_dict[
        trainval_io.RADAR_FILE_NAMES_KEY]
    num_radar_dimensions = len(training_radar_file_name_matrix.shape)

    if num_radar_dimensions == 3:
        radar_field_names = [
            storm_images.image_file_name_to_field(f) for f in
            training_radar_file_name_matrix[0, :, 0]
        ]
        radar_heights_m_agl = numpy.array(
            [storm_images.image_file_name_to_height(f)
             for f in training_radar_file_name_matrix[0, 0, :]],
            dtype=int)

        saliency_plotting.plot_saliency_with_radar_3d_fields(
            radar_matrix=list_of_input_matrices[0],
            saliency_matrix=list_of_saliency_matrices[0],
            saliency_metadata_dict=saliency_metadata_dict,
            radar_field_names=radar_field_names,
            radar_heights_m_agl=radar_heights_m_agl,
            one_fig_per_storm_object=True, num_panel_rows=num_panel_rows,
            output_dir_name=output_dir_name,
            saliency_option_dict=saliency_option_dict)
    else:
        field_name_by_pair = [
            storm_images.image_file_name_to_field(f) for f in
            training_radar_file_name_matrix[0, :]
        ]
        height_by_pair_m_agl = numpy.array(
            [storm_images.image_file_name_to_height(f)
             for f in training_radar_file_name_matrix[0, :]],
            dtype=int)

        saliency_plotting.plot_saliency_with_radar_2d_fields(
            radar_matrix=list_of_input_matrices[0],
            saliency_matrix=list_of_saliency_matrices[0],
            saliency_metadata_dict=saliency_metadata_dict,
            field_name_by_pair=field_name_by_pair,
            height_by_pair_m_agl=height_by_pair_m_agl,
            one_fig_per_storm_object=True, num_panel_rows=num_panel_rows,
            output_dir_name=output_dir_name,
            saliency_option_dict=saliency_option_dict)

    print SEPARATOR_STRING

    if training_option_dict[trainval_io.SOUNDING_FIELDS_KEY] is not None:
        saliency_plotting.plot_saliency_with_soundings(
            sounding_matrix=list_of_input_matrices[-1],
            saliency_matrix=list_of_saliency_matrices[-1],
            saliency_metadata_dict=saliency_metadata_dict,
            sounding_field_names=training_option_dict[
                trainval_io.SOUNDING_FIELDS_KEY],
            output_dir_name=output_dir_name,
            saliency_option_dict=saliency_option_dict,
            temp_directory_name=temp_directory_name)

    print SEPARATOR_STRING


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        max_colour_value=getattr(INPUT_ARG_OBJECT, MAX_COLOUR_VALUE_ARG_NAME),
        max_colour_percentile=getattr(
            INPUT_ARG_OBJECT, MAX_COLOUR_PRCTILE_ARG_NAME),
        num_panel_rows=getattr(INPUT_ARG_OBJECT, NUM_PANEL_ROWS_ARG_NAME),
        temp_directory_name=getattr(INPUT_ARG_OBJECT, TEMP_DIRECTORY_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME))
