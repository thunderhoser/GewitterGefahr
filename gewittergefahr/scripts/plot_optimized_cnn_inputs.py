"""Plots optimized input examples (synthetic storm objects) for a CNN.

CNN = convolutional neural network
"""

import os.path
import argparse
import numpy
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import storm_images
from gewittergefahr.deep_learning import model_interpretation
from gewittergefahr.deep_learning import feature_optimization
from gewittergefahr.plotting import \
    feature_optimization_plotting as fopt_plotting

INPUT_FILE_ARG_NAME = 'input_file_name'
ONE_FIG_PER_COMPONENT_ARG_NAME = 'one_figure_per_component'
NUM_PANEL_ROWS_ARG_NAME = 'num_panel_rows'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by `feature_optimization.read_file`.')
ONE_FIG_PER_COMPONENT_HELP_STRING = (
    'Boolean flag.  If 1, this script will create one figure per model '
    'component, where each panel is a different radar field/height.  If 0, will'
    ' create one figure per radar field/height, where each panel is a different'
    ' model component.')
NUM_PANEL_ROWS_HELP_STRING = 'Number of panel rows in each figure.'
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + ONE_FIG_PER_COMPONENT_ARG_NAME, type=int, required=False, default=1,
    help=ONE_FIG_PER_COMPONENT_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_PANEL_ROWS_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_PANEL_ROWS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(input_file_name, one_figure_per_component, num_panel_rows,
         output_dir_name):
    """Plots optimized input examples (synthetic storm objects) for a CNN.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param one_figure_per_component: Same.
    :param num_panel_rows: Same.
    :param output_dir_name: Same.
    :raises: TypeError: if input examples were optimized for a model that does
        2-D and 3-D convolution.
    """

    print 'Reading optimized inputs from: "{0:s}"...'.format(input_file_name)
    (list_of_optimized_input_matrices, fopt_metadata_dict
    ) = feature_optimization.read_file(input_file_name)

    model_file_name = fopt_metadata_dict[
        feature_optimization.MODEL_FILE_NAME_KEY]
    model_metadata_file_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0])

    print 'Reading metadata from: "{0:s}"...'.format(model_metadata_file_name)
    model_metadata_dict = cnn.read_model_metadata(model_metadata_file_name)
    if model_metadata_dict[cnn.USE_2D3D_CONVOLUTION_KEY]:
        raise TypeError('This script cannot yet handle models that do 2-D and '
                        '3-D convolution.')

    print 'Denormalizing optimized inputs...'
    list_of_optimized_input_matrices = model_interpretation.denormalize_data(
        list_of_input_matrices=list_of_optimized_input_matrices,
        model_metadata_dict=model_metadata_dict)

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

        fopt_plotting.plot_many_optimized_fields_3d(
            radar_field_matrix=list_of_optimized_input_matrices[0],
            radar_field_names=radar_field_names,
            radar_heights_m_asl=radar_heights_m_asl,
            one_figure_per_component=one_figure_per_component,
            component_type_string=fopt_metadata_dict[
                feature_optimization.COMPONENT_TYPE_KEY],
            output_dir_name=output_dir_name,
            num_panel_rows=num_panel_rows,
            target_class=fopt_metadata_dict[
                feature_optimization.TARGET_CLASS_KEY],
            layer_name=fopt_metadata_dict[feature_optimization.LAYER_NAME_KEY],
            neuron_index_matrix=fopt_metadata_dict[
                feature_optimization.NEURON_INDICES_KEY],
            channel_indices=fopt_metadata_dict[
                feature_optimization.CHANNEL_INDICES_KEY])
    else:
        field_name_by_pair = [
            storm_images.image_file_name_to_field(f) for f in
            training_radar_file_name_matrix[0, :]
        ]
        height_by_pair_m_asl = numpy.array(
            [storm_images.image_file_name_to_height(f)
             for f in training_radar_file_name_matrix[0, :]],
            dtype=int)

        fopt_plotting.plot_many_optimized_fields_2d(
            radar_field_matrix=list_of_optimized_input_matrices[0],
            field_name_by_pair=field_name_by_pair,
            height_by_pair_m_asl=height_by_pair_m_asl,
            one_figure_per_component=one_figure_per_component,
            num_panel_rows=num_panel_rows,
            component_type_string=fopt_metadata_dict[
                feature_optimization.COMPONENT_TYPE_KEY],
            output_dir_name=output_dir_name,
            target_class=fopt_metadata_dict[
                feature_optimization.TARGET_CLASS_KEY],
            layer_name=fopt_metadata_dict[feature_optimization.LAYER_NAME_KEY],
            neuron_index_matrix=fopt_metadata_dict[
                feature_optimization.NEURON_INDICES_KEY],
            channel_indices=fopt_metadata_dict[
                feature_optimization.CHANNEL_INDICES_KEY])


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        one_figure_per_component=bool(
            getattr(INPUT_ARG_OBJECT, ONE_FIG_PER_COMPONENT_ARG_NAME)),
        num_panel_rows=getattr(INPUT_ARG_OBJECT, NUM_PANEL_ROWS_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME))
