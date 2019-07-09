"""For each example (storm object), plots feature maps for one CNN layer."""

import random
import os.path
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from keras import backend as K
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import testing_io
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import feature_map_plotting

random.seed(6695)
numpy.random.seed(6695)

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
    allow_soft_placement=False
)))

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MAIN_FONT_SIZE = 20
SMALL_FONT_SIZE = 15
TINY_FONT_SIZE = 10
NUM_PANELS_FOR_SMALL_FONT = 128
NUM_PANELS_FOR_TINY_FONT = 256
NUM_PANELS_FOR_NO_FONT = 384

FIGURE_RESOLUTION_DPI = 300

MODEL_FILE_ARG_NAME = 'input_model_file_name'
LAYER_NAMES_ARG_NAME = 'layer_names'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
STORM_METAFILE_ARG_NAME = 'input_storm_metafile_name'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MODEL_FILE_HELP_STRING = (
    'Path to file with trained CNN.  Will be read by `cnn.read_model`.')

LAYER_NAMES_HELP_STRING = (
    'Layer names.  Feature maps will be plotted for each pair of layer and '
    'example (storm object).')

EXAMPLE_DIR_HELP_STRING = (
    'Name of top-level directory with input examples.  Files therein will be '
    'found by `input_examples.find_example_file` and read by '
    '`input_examples.read_example_file`.')

STORM_METAFILE_HELP_STRING = (
    'Path to Pickle file with storm IDs and times.  Will be read by '
    '`storm_tracking_io.read_ids_and_times`.')

NUM_EXAMPLES_HELP_STRING = (
    'Number of examples (storm objects) to read from `{0:s}`.  If you want to '
    'read all examples, make this non-positive.'
).format(STORM_METAFILE_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level output directory.  Figures will be saved here (one '
    'subdirectory per layer).')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAYER_NAMES_ARG_NAME, type=str, nargs='+', required=True,
    help=LAYER_NAMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + STORM_METAFILE_ARG_NAME, type=str, required=True,
    help=STORM_METAFILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_EXAMPLES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _plot_feature_maps_one_layer(
        feature_matrix, full_id_strings, storm_times_unix_sec, layer_name,
        output_dir_name):
    """Plots all feature maps for one layer.

    E = number of examples (storm objects)
    M = number of spatial rows
    N = number of spatial columns
    H = number of spatial depths (heights)
    C = number of channels

    :param feature_matrix: numpy array (E x M x N x C or E x M x N x H x C) of
        feature maps.
    :param full_id_strings: length-E list of full storm IDs.
    :param storm_times_unix_sec: length-E numpy array of storm times.
    :param layer_name: Name of layer.
    :param output_dir_name: Name of output directory for this layer.
    """

    num_spatial_dimensions = len(feature_matrix.shape) - 2
    num_storm_objects = feature_matrix.shape[0]
    num_channels = feature_matrix.shape[-1]

    if num_spatial_dimensions == 3:
        num_heights = feature_matrix.shape[-2]
    else:
        num_heights = None

    num_panel_rows = int(numpy.round(numpy.sqrt(num_channels)))
    annotation_string_by_channel = [None] * num_channels

    # annotation_string_by_channel = [
    #     'Filter {0:d}'.format(c + 1) for c in range(num_channels)
    # ]

    if num_channels >= NUM_PANELS_FOR_NO_FONT:
        annotation_string_by_channel = [''] * num_channels
        font_size = TINY_FONT_SIZE + 0
    elif num_channels >= NUM_PANELS_FOR_TINY_FONT:
        font_size = TINY_FONT_SIZE + 0
    elif num_channels >= NUM_PANELS_FOR_SMALL_FONT:
        font_size = SMALL_FONT_SIZE + 0
    else:
        font_size = MAIN_FONT_SIZE + 0

    max_colour_value = numpy.percentile(numpy.absolute(feature_matrix), 99)
    min_colour_value = -1 * max_colour_value

    for i in range(num_storm_objects):
        this_time_string = time_conversion.unix_sec_to_string(
            storm_times_unix_sec[i], TIME_FORMAT)

        if num_spatial_dimensions == 2:
            _, this_axes_object_matrix = (
                feature_map_plotting.plot_many_2d_feature_maps(
                    feature_matrix=numpy.flip(feature_matrix[i, ...], axis=0),
                    annotation_string_by_panel=annotation_string_by_channel,
                    num_panel_rows=num_panel_rows,
                    colour_map_object=pyplot.cm.seismic,
                    min_colour_value=min_colour_value,
                    max_colour_value=max_colour_value, font_size=font_size)
            )

            plotting_utils.plot_linear_colour_bar(
                axes_object_or_matrix=this_axes_object_matrix,
                data_matrix=feature_matrix[i, ...],
                colour_map_object=pyplot.cm.seismic, min_value=min_colour_value,
                max_value=max_colour_value, orientation_string='horizontal',
                extend_min=True, extend_max=True)

            this_title_string = 'Layer "{0:s}", storm "{1:s}" at {2:s}'.format(
                layer_name, full_id_strings[i], this_time_string
            )
            pyplot.suptitle(this_title_string, fontsize=MAIN_FONT_SIZE)

            this_figure_file_name = (
                '{0:s}/storm={1:s}_{2:s}_features.jpg'
            ).format(
                output_dir_name, full_id_strings[i].replace('_', '-'),
                this_time_string
            )

            print('Saving figure to: "{0:s}"...'.format(this_figure_file_name))
            pyplot.savefig(this_figure_file_name, dpi=FIGURE_RESOLUTION_DPI)
            pyplot.close()

        else:
            for k in range(num_heights):
                _, this_axes_object_matrix = (
                    feature_map_plotting.plot_many_2d_feature_maps(
                        feature_matrix=numpy.flip(
                            feature_matrix[i, :, :, k, :], axis=0),
                        annotation_string_by_panel=annotation_string_by_channel,
                        num_panel_rows=num_panel_rows,
                        colour_map_object=pyplot.cm.seismic,
                        min_colour_value=min_colour_value,
                        max_colour_value=max_colour_value, font_size=font_size)
                )

                plotting_utils.plot_linear_colour_bar(
                    axes_object_or_matrix=this_axes_object_matrix,
                    data_matrix=feature_matrix[i, :, :, k, :],
                    colour_map_object=pyplot.cm.seismic,
                    min_value=min_colour_value, max_value=max_colour_value,
                    orientation_string='horizontal', extend_min=True,
                    extend_max=True)

                this_title_string = (
                    'Layer "{0:s}", height {1:d} of {2:d}, storm "{3:s}" at '
                    '{4:s}'
                ).format(
                    layer_name, k + 1, num_heights, full_id_strings[i],
                    this_time_string
                )

                pyplot.suptitle(this_title_string, fontsize=MAIN_FONT_SIZE)

                this_figure_file_name = (
                    '{0:s}/storm={1:s}_{2:s}_features_height{3:02d}.jpg'
                ).format(
                    output_dir_name, full_id_strings[i].replace('_', '-'),
                    this_time_string, k + 1
                )

                print('Saving figure to: "{0:s}"...'.format(
                    this_figure_file_name))

                pyplot.savefig(this_figure_file_name, dpi=FIGURE_RESOLUTION_DPI)
                pyplot.close()


def _run(model_file_name, layer_names, top_example_dir_name,
         storm_metafile_name, num_examples, top_output_dir_name):
    """Evaluates CNN (convolutional neural net) predictions.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param layer_names: Same.
    :param top_example_dir_name: Same.
    :param storm_metafile_name: Same.
    :param num_examples: Same.
    :param top_output_dir_name: Same.
    :raises: ValueError: if feature maps do not have 2 or 3 spatial dimensions.
    """

    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = cnn.read_model(model_file_name)

    model_metafile_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0]
    )

    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = cnn.read_model_metadata(model_metafile_name)
    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    training_option_dict[trainval_io.REFLECTIVITY_MASK_KEY] = None

    print('Reading storm metadata from: "{0:s}"...'.format(storm_metafile_name))
    full_id_strings, storm_times_unix_sec = tracking_io.read_ids_and_times(
        storm_metafile_name)

    print(SEPARATOR_STRING)

    if 0 < num_examples < len(full_id_strings):
        full_id_strings = full_id_strings[:num_examples]
        storm_times_unix_sec = storm_times_unix_sec[:num_examples]

    list_of_predictor_matrices = testing_io.read_specific_examples(
        top_example_dir_name=top_example_dir_name,
        desired_full_id_strings=full_id_strings,
        desired_times_unix_sec=storm_times_unix_sec,
        option_dict=training_option_dict,
        list_of_layer_operation_dicts=model_metadata_dict[
            cnn.LAYER_OPERATIONS_KEY]
    )[0]

    print(SEPARATOR_STRING)

    num_layers = len(layer_names)
    feature_matrix_by_layer = [None] * num_layers

    for k in range(num_layers):
        if model_metadata_dict[cnn.USE_2D3D_CONVOLUTION_KEY]:
            if len(list_of_predictor_matrices) == 3:
                sounding_matrix = list_of_predictor_matrices[-1]
            else:
                sounding_matrix = None

            feature_matrix_by_layer[k] = cnn.apply_2d3d_cnn(
                model_object=model_object,
                reflectivity_matrix_dbz=list_of_predictor_matrices[0],
                azimuthal_shear_matrix_s01=list_of_predictor_matrices[1],
                sounding_matrix=sounding_matrix,
                return_features=True, feature_layer_name=layer_names[k]
            )
        else:
            if len(list_of_predictor_matrices) == 2:
                sounding_matrix = list_of_predictor_matrices[-1]
            else:
                sounding_matrix = None

            num_radar_dimensions = len(list_of_predictor_matrices[0].shape) - 2

            if num_radar_dimensions == 2:
                feature_matrix_by_layer[k] = cnn.apply_2d_or_3d_cnn(
                    model_object=model_object,
                    radar_image_matrix=list_of_predictor_matrices[0],
                    sounding_matrix=sounding_matrix,
                    return_features=True, feature_layer_name=layer_names[k]
                )
            else:
                feature_matrix_by_layer[k] = cnn.apply_2d_or_3d_cnn(
                    model_object=model_object,
                    radar_image_matrix=list_of_predictor_matrices[0],
                    sounding_matrix=sounding_matrix,
                    return_features=True, feature_layer_name=layer_names[k]
                )

    for k in range(num_layers):
        this_output_dir_name = '{0:s}/{1:s}'.format(
            top_output_dir_name, layer_names[k]
        )

        file_system_utils.mkdir_recursive_if_necessary(
            directory_name=this_output_dir_name)

        _plot_feature_maps_one_layer(
            feature_matrix=feature_matrix_by_layer[k],
            full_id_strings=full_id_strings,
            storm_times_unix_sec=storm_times_unix_sec,
            layer_name=layer_names[k],
            output_dir_name=this_output_dir_name)

        print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        layer_names=getattr(INPUT_ARG_OBJECT, LAYER_NAMES_ARG_NAME),
        top_example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        storm_metafile_name=getattr(INPUT_ARG_OBJECT, STORM_METAFILE_ARG_NAME),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
