"""For each example (storm object), plots feature maps for one CNN layer."""

import random
import os.path
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from keras import backend as K
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import input_examples
from gewittergefahr.deep_learning import testing_io
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import feature_map_plotting

random.seed(6695)
numpy.random.seed(6695)

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)))

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TITLE_FONT_SIZE = 20
FIGURE_RESOLUTION_DPI = 300

MODEL_FILE_ARG_NAME = 'input_model_file_name'
LAYER_NAMES_ARG_NAME = 'layer_names'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
FIRST_SPC_DATE_ARG_NAME = 'first_spc_date_string'
LAST_SPC_DATE_ARG_NAME = 'last_spc_date_string'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
CLASS_FRACTION_KEYS_ARG_NAME = 'class_fraction_keys'
CLASS_FRACTION_VALUES_ARG_NAME = 'class_fraction_values'
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

SPC_DATE_HELP_STRING = (
    'SPC date (format "yyyymmdd").  This script will evaluate predictions on '
    'examples from the period `{0:s}`...`{1:s}`.'
).format(FIRST_SPC_DATE_ARG_NAME, LAST_SPC_DATE_ARG_NAME)

NUM_EXAMPLES_HELP_STRING = 'Number of examples to use.'

CLASS_FRACTION_KEYS_HELP_STRING = (
    'List of keys used to create input `class_to_sampling_fraction_dict` for '
    '`deep_learning_utils.sample_by_class`.  If you do not want class-'
    'conditional sampling, leave this alone.'
)

CLASS_FRACTION_VALUES_HELP_STRING = (
    'List of values used to create input `class_to_sampling_fraction_dict` for '
    '`deep_learning_utils.sample_by_class`.  If you do not want class-'
    'conditional sampling, leave this alone.'
)

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
    '--' + FIRST_SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=True,
    help=NUM_EXAMPLES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CLASS_FRACTION_KEYS_ARG_NAME, type=int, nargs='+',
    required=False, default=[0], help=CLASS_FRACTION_KEYS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CLASS_FRACTION_VALUES_ARG_NAME, type=float, nargs='+',
    required=False, default=[0.], help=CLASS_FRACTION_VALUES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _plot_feature_maps_one_layer(
        feature_matrix, storm_ids, storm_times_unix_sec, layer_name,
        output_dir_name):
    """Plots all feature maps for one layer.

    E = number of examples (storm objects)
    M = number of spatial rows
    N = number of spatial columns
    H = number of spatial depths (heights)
    C = number of channels

    :param feature_matrix: numpy array (E x M x N x C or E x M x N x H x C) of
        feature maps.
    :param storm_ids: length-E list of storm IDs.
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
    annotation_string_by_channel = [
        'Channel {0:d}'.format(c + 1) for c in range(num_channels)
    ]

    max_colour_value = numpy.percentile(numpy.absolute(feature_matrix), 99)
    min_colour_value = -1 * max_colour_value

    for i in range(num_storm_objects):
        this_time_string = time_conversion.unix_sec_to_string(
            storm_times_unix_sec[i], TIME_FORMAT)

        if num_spatial_dimensions == 2:
            _, these_axes_objects = (
                feature_map_plotting.plot_many_2d_feature_maps(
                    feature_matrix=feature_matrix[i, ...],
                    annotation_string_by_panel=annotation_string_by_channel,
                    num_panel_rows=num_panel_rows,
                    colour_map_object=pyplot.cm.seismic,
                    min_colour_value=min_colour_value,
                    max_colour_value=max_colour_value)
            )

            plotting_utils.add_linear_colour_bar(
                axes_object_or_list=these_axes_objects,
                values_to_colour=feature_matrix[i, ...],
                colour_map=pyplot.cm.seismic, colour_min=min_colour_value,
                colour_max=max_colour_value, orientation='horizontal',
                extend_min=True, extend_max=True)

            this_title_string = 'Layer "{0:s}", storm "{1:s}" at {2:s}'.format(
                layer_name, storm_ids[i], this_time_string)
            pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)

            this_figure_file_name = '{0:s}/{1:s}_{2:s}_features.jpg'.format(
                output_dir_name, storm_ids[i].replace('_', '-'),
                this_time_string)

            print 'Saving figure to: "{0:s}"...'.format(this_figure_file_name)
            pyplot.savefig(this_figure_file_name, dpi=FIGURE_RESOLUTION_DPI)

        else:
            for k in range(num_heights):
                _, these_axes_objects = (
                    feature_map_plotting.plot_many_2d_feature_maps(
                        feature_matrix=feature_matrix[i, :, :, k, :],
                        annotation_string_by_panel=annotation_string_by_channel,
                        num_panel_rows=num_panel_rows,
                        colour_map_object=pyplot.cm.seismic,
                        min_colour_value=min_colour_value,
                        max_colour_value=max_colour_value)
                )

                plotting_utils.add_linear_colour_bar(
                    axes_object_or_list=these_axes_objects,
                    values_to_colour=feature_matrix[i, :, :, k, :],
                    colour_map=pyplot.cm.seismic, colour_min=min_colour_value,
                    colour_max=max_colour_value, orientation='horizontal',
                    extend_min=True, extend_max=True)

                this_title_string = (
                    'Layer "{0:s}", height {1:d} of {2:d}, storm "{3:s}" at '
                    '{4:s}'
                ).format(
                    layer_name, k + 1, num_heights, storm_ids[i],
                    this_time_string)

                pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)

                this_figure_file_name = (
                    '{0:s}/{1:s}_{2:s}_features_height{3:02d}.jpg'
                ).format(
                    output_dir_name, storm_ids[i].replace('_', '-'),
                    this_time_string, k + 1)

                print 'Saving figure to: "{0:s}"...'.format(
                    this_figure_file_name)
                pyplot.savefig(this_figure_file_name, dpi=FIGURE_RESOLUTION_DPI)


def _run(model_file_name, layer_names, top_example_dir_name,
         first_spc_date_string, last_spc_date_string, num_examples,
         class_fraction_keys, class_fraction_values, top_output_dir_name):
    """Evaluates CNN (convolutional neural net) predictions.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param layer_names: Same.
    :param top_example_dir_name: Same.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
    :param num_examples: Same.
    :param class_fraction_keys: Same.
    :param class_fraction_values: Same.
    :param top_output_dir_name: Same.
    :raises: ValueError: if feature maps do not have 2 or 3 spatial dimensions.
    """

    print 'Reading model from: "{0:s}"...'.format(model_file_name)
    model_object = cnn.read_model(model_file_name)

    model_directory_name, _ = os.path.split(model_file_name)
    metadata_file_name = '{0:s}/model_metadata.p'.format(model_directory_name)

    print 'Reading metadata from: "{0:s}"...'.format(metadata_file_name)
    model_metadata_dict = cnn.read_model_metadata(metadata_file_name)
    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]

    if len(class_fraction_keys) > 1:
        class_to_sampling_fraction_dict = dict(zip(
            class_fraction_keys, class_fraction_values))
    else:
        class_to_sampling_fraction_dict = None

    training_option_dict[
        trainval_io.SAMPLING_FRACTIONS_KEY] = class_to_sampling_fraction_dict

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

    if model_metadata_dict[cnn.USE_2D3D_CONVOLUTION_KEY]:
        generator_object = testing_io.example_generator_2d3d_myrorss(
            option_dict=training_option_dict, num_examples_total=num_examples)
    else:
        generator_object = testing_io.example_generator_2d_or_3d(
            option_dict=training_option_dict, num_examples_total=num_examples)

    num_layers = len(layer_names)
    feature_matrix_by_layer = [None] * num_layers
    storm_ids = []
    storm_times_unix_sec = numpy.array([], dtype=int)

    for _ in range(len(example_file_names)):
        this_feature_matrix_by_layer = [None] * num_layers

        try:
            this_storm_object_dict = next(generator_object)
            print SEPARATOR_STRING
        except StopIteration:
            break

        these_predictor_matrices = this_storm_object_dict[
            testing_io.INPUT_MATRICES_KEY]

        if model_metadata_dict[cnn.USE_2D3D_CONVOLUTION_KEY]:
            if len(these_predictor_matrices) == 3:
                this_sounding_matrix = these_predictor_matrices[2]
            else:
                this_sounding_matrix = None

            for k in range(num_layers):
                this_feature_matrix_by_layer[k] = cnn.apply_2d3d_cnn(
                    model_object=model_object,
                    reflectivity_image_matrix_dbz=these_predictor_matrices[0],
                    az_shear_image_matrix_s01=these_predictor_matrices[1],
                    sounding_matrix=this_sounding_matrix,
                    return_features=True, output_layer_name=layer_names[k])

        else:
            if len(these_predictor_matrices) == 2:
                this_sounding_matrix = these_predictor_matrices[1]
            else:
                this_sounding_matrix = None

            num_radar_dimensions = len(these_predictor_matrices[0].shape) - 2

            for k in range(num_layers):
                if num_radar_dimensions == 2:
                    this_feature_matrix_by_layer[k] = cnn.apply_2d_cnn(
                        model_object=model_object,
                        radar_image_matrix=these_predictor_matrices[0],
                        sounding_matrix=this_sounding_matrix,
                        return_features=True, output_layer_name=layer_names[k])
                else:
                    this_feature_matrix_by_layer[k] = cnn.apply_3d_cnn(
                        model_object=model_object,
                        radar_image_matrix=these_predictor_matrices[0],
                        sounding_matrix=this_sounding_matrix,
                        return_features=True, output_layer_name=layer_names[k])

        storm_ids += this_storm_object_dict[testing_io.STORM_IDS_KEY]
        storm_times_unix_sec = numpy.concatenate((
            storm_times_unix_sec,
            this_storm_object_dict[testing_io.STORM_TIMES_KEY]
        ))

        for k in range(num_layers):
            if feature_matrix_by_layer[k] is None:
                feature_matrix_by_layer[k] = (
                    this_feature_matrix_by_layer[k] + 0.
                )

                num_spatial_dimensions = (
                    len(feature_matrix_by_layer[k].shape) - 2
                )

                if num_spatial_dimensions not in [2, 3]:
                    error_string = (
                        'Feature maps should have 2 or 3 spatial dimensions.  '
                        'Instead, got {0:d}.'
                    ).format(num_spatial_dimensions)

                    raise ValueError(error_string)
            else:
                feature_matrix_by_layer[k] = numpy.concatenate(
                    (feature_matrix_by_layer[k],
                     this_feature_matrix_by_layer[k]), axis=0)

    for k in range(num_layers):
        this_output_dir_name = '{0:s}/{1:s}'.format(
            top_output_dir_name, layer_names[k])
        file_system_utils.mkdir_recursive_if_necessary(
            directory_name=this_output_dir_name)

        _plot_feature_maps_one_layer(
            feature_matrix=feature_matrix_by_layer[k], storm_ids=storm_ids,
            storm_times_unix_sec=storm_times_unix_sec,
            layer_name=layer_names[k],
            output_dir_name=this_output_dir_name)
        print SEPARATOR_STRING


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        layer_names=getattr(INPUT_ARG_OBJECT, LAYER_NAMES_ARG_NAME),
        top_example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        first_spc_date_string=getattr(
            INPUT_ARG_OBJECT, FIRST_SPC_DATE_ARG_NAME),
        last_spc_date_string=getattr(INPUT_ARG_OBJECT, LAST_SPC_DATE_ARG_NAME),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        class_fraction_keys=numpy.array(
            getattr(INPUT_ARG_OBJECT, CLASS_FRACTION_KEYS_ARG_NAME), dtype=int),
        class_fraction_values=numpy.array(
            getattr(INPUT_ARG_OBJECT, CLASS_FRACTION_VALUES_ARG_NAME),
            dtype=float),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
