"""Plots saliency maps for a CNN (convolutional neural network)."""

import os.path
import argparse
import numpy
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import saliency_maps
from gewittergefahr.deep_learning import storm_images
from gewittergefahr.deep_learning import model_interpretation
from gewittergefahr.plotting import saliency_plotting

# TODO(thunderhoser): Maybe add other saliency-plotting options to this script.

INPUT_FILE_ARG_NAME = 'input_file_name'
MAX_CONTOUR_VALUE_ARG_NAME = 'max_contour_value'
MAX_CONTOUR_PRCTILE_ARG_NAME = 'max_contour_percentile'
ONE_FIG_PER_OBJECT_ARG_NAME = 'one_fig_per_storm_object'
NUM_PANEL_ROWS_ARG_NAME = 'num_panel_rows'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by `saliency_maps.read_file`.')

MAX_CONTOUR_VALUE_HELP_STRING = (
    'Max saliency contour.  Minimum saliency contour will be -1 * `{0:s}`.  To '
    'use `{1:s}` instead, leave this argument alone.'
).format(MAX_CONTOUR_VALUE_ARG_NAME, MAX_CONTOUR_PRCTILE_ARG_NAME)

MAX_CONTOUR_PRCTILE_HELP_STRING = (
    'Max saliency contour will be the `{0:s}`th percentile in `{1:s}` (over all'
    ' storm objects and field/height pairs).  Minimum saliency contour will be '
    '-1 * the `{0:s}`th percentile.  To use `{2:s}` instead, leave this '
    'argument alone.'
).format(MAX_CONTOUR_PRCTILE_ARG_NAME, INPUT_FILE_ARG_NAME,
         MAX_CONTOUR_VALUE_ARG_NAME)

ONE_FIG_PER_OBJECT_HELP_STRING = (
    'Boolean flag.  If 1, this script will create one figure per storm object, '
    'where each panel is a different radar field/height.  If 0, will create one'
    ' figure per radar field/height, where each panel is a different storm '
    'object.')

NUM_PANEL_ROWS_HELP_STRING = 'Number of panel rows in each figure.'
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_CONTOUR_VALUE_ARG_NAME, type=float, required=False, default=-1,
    help=MAX_CONTOUR_VALUE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_CONTOUR_PRCTILE_ARG_NAME, type=float, required=False, default=-1,
    help=MAX_CONTOUR_PRCTILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + ONE_FIG_PER_OBJECT_ARG_NAME, type=int, required=False, default=1,
    help=ONE_FIG_PER_OBJECT_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_PANEL_ROWS_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_PANEL_ROWS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(input_file_name, max_contour_value, max_contour_percentile,
         one_fig_per_storm_object, num_panel_rows, output_dir_name):
    """Plots saliency maps for a CNN (convolutional neural network).

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param max_contour_value: Same.
    :param max_contour_percentile: Same.
    :param one_fig_per_storm_object: Same.
    :param num_panel_rows: Same.
    :param output_dir_name: Same.
    :raises: ValueError: if both `max_contour_value` and
        `max_contour_percentile` are non-positive.
    :raises: TypeError: if saliency maps come from a model that does 2-D and 3-D
        convolution.
    """

    # Check input args.
    if max_contour_value <= 0:
        max_contour_value = None
    if max_contour_percentile <= 0:
        max_contour_percentile = None
    if max_contour_value is None and max_contour_percentile is None:
        raise ValueError(
            'max_contour_value and max_contour_percentile cannot both be None.')

    # Read saliency maps.
    print 'Reading saliency maps from: "{0:s}"...'.format(input_file_name)
    (list_of_input_matrices, list_of_saliency_matrices, saliency_metadata_dict
    ) = saliency_maps.read_file(input_file_name)

    if max_contour_value is None:
        all_saliency_values = numpy.array([])
        for this_matrix in list_of_saliency_matrices:
            all_saliency_values = numpy.concatenate(
                all_saliency_values, numpy.ravel(this_matrix))

        max_contour_value = numpy.percentile(
            all_saliency_values, max_contour_percentile)
        del all_saliency_values

    print 'Max saliency contour = {0:.3e}\n'.format(max_contour_value)
    saliency_option_dict = {
        saliency_plotting.MAX_CONTOUR_VALUE_KEY: max_contour_value
    }

    # Read metadata for the CNN that generated the saliency maps.
    model_file_name = saliency_metadata_dict[saliency_maps.MODEL_FILE_NAME_KEY]
    model_metadata_file_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0])

    print 'Reading metadata from: "{0:s}"...'.format(model_metadata_file_name)
    model_metadata_dict = cnn.read_model_metadata(model_metadata_file_name)
    if model_metadata_dict[cnn.USE_2D3D_CONVOLUTION_KEY]:
        raise TypeError('This script cannot yet handle models that do 2-D and '
                        '3-D convolution.')

    print 'Denormalizing optimized inputs...'
    list_of_input_matrices = model_interpretation.denormalize_data(
        list_of_input_matrices=list_of_input_matrices,
        model_metadata_dict=model_metadata_dict)

    # Plot saliency maps.
    training_radar_file_name_matrix = model_metadata_dict[
        cnn.TRAINING_FILE_NAMES_KEY]
    num_radar_dimensions = len(training_radar_file_name_matrix.shape)

    if num_radar_dimensions == 3:
        radar_field_names = [
            storm_images.image_file_name_to_field(f) for f in
            training_radar_file_name_matrix[0, :, 0]
        ]
        radar_heights_m_asl = numpy.array(
            [storm_images.image_file_name_to_height(f)
             for f in training_radar_file_name_matrix[0, 0, :]],
            dtype=int)

        saliency_plotting.plot_many_saliency_fields_3d(
            radar_field_matrix=list_of_input_matrices[0],
            saliency_field_matrix=list_of_saliency_matrices[0],
            saliency_metadata_dict=saliency_metadata_dict,
            radar_field_names=radar_field_names,
            radar_heights_m_asl=radar_heights_m_asl,
            one_fig_per_storm_object=one_fig_per_storm_object,
            num_panel_rows=num_panel_rows, output_dir_name=output_dir_name,
            saliency_option_dict=saliency_option_dict)
    else:
        field_name_by_pair = [
            storm_images.image_file_name_to_field(f) for f in
            training_radar_file_name_matrix[0, :]
        ]
        height_by_pair_m_asl = numpy.array(
            [storm_images.image_file_name_to_height(f)
             for f in training_radar_file_name_matrix[0, :]],
            dtype=int)

        saliency_plotting.plot_many_saliency_fields_2d(
            radar_field_matrix=list_of_input_matrices[0],
            saliency_field_matrix=list_of_saliency_matrices[0],
            saliency_metadata_dict=saliency_metadata_dict,
            field_name_by_pair=field_name_by_pair,
            height_by_pair_m_asl=height_by_pair_m_asl,
            one_fig_per_storm_object=one_fig_per_storm_object,
            num_panel_rows=num_panel_rows, output_dir_name=output_dir_name,
            saliency_option_dict=saliency_option_dict)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        max_contour_value=getattr(INPUT_ARG_OBJECT, MAX_CONTOUR_VALUE_ARG_NAME),
        max_contour_percentile=getattr(
            INPUT_ARG_OBJECT, MAX_CONTOUR_PRCTILE_ARG_NAME),
        one_fig_per_storm_object=bool(getattr(
            INPUT_ARG_OBJECT, ONE_FIG_PER_OBJECT_ARG_NAME)),
        num_panel_rows=getattr(INPUT_ARG_OBJECT, NUM_PANEL_ROWS_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME))
