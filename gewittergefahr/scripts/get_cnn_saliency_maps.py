"""Computes saliency map for each storm object and each CNN component.

CNN = convolutional neural network
"""

import copy
import os.path
import argparse
import numpy
from keras import backend as K
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import storm_images
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.deep_learning import model_interpretation
from gewittergefahr.deep_learning import saliency_maps

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)))

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

LARGE_INTEGER = int(1e10)

MODEL_FILE_ARG_NAME = 'model_file_name'
COMPONENT_TYPE_ARG_NAME = 'component_type_string'
TARGET_CLASS_ARG_NAME = 'target_class'
LAYER_NAME_ARG_NAME = 'layer_name'
IDEAL_ACTIVATION_ARG_NAME = 'ideal_activation'
NEURON_INDICES_ARG_NAME = 'neuron_indices'
CHANNEL_INDEX_ARG_NAME = 'channel_index'
RADAR_IMAGE_DIR_ARG_NAME = 'input_radar_image_dir_name'
SOUNDING_DIR_ARG_NAME = 'input_sounding_dir_name'
STORM_IDS_ARG_NAME = 'storm_ids'
STORM_TIMES_ARG_NAME = 'storm_times_unix_sec'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

MODEL_FILE_HELP_STRING = (
    'Path to input file, containing a trained CNN.  Will be read by '
    '`cnn.read_model`.')
COMPONENT_TYPE_HELP_STRING = (
    'Component type.  Saliency maps may be computed for one class, one/many '
    'neurons, or one/many channels.  Valid options are listed below.\n{0:s}'
).format(str(model_interpretation.VALID_COMPONENT_TYPE_STRINGS))
TARGET_CLASS_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] Saliency maps will be computed for each '
    'storm object and each class k, where k = `{2:s}`.'
).format(COMPONENT_TYPE_ARG_NAME,
         model_interpretation.CLASS_COMPONENT_TYPE_STRING,
         TARGET_CLASS_ARG_NAME)
LAYER_NAME_HELP_STRING = (
    '[used only if {0:s} = "{1:s}" or "{2:s}"] Name of layer with neurons or '
    'channels for which saliency maps will be computed.'
).format(COMPONENT_TYPE_ARG_NAME,
         model_interpretation.NEURON_COMPONENT_TYPE_STRING,
         model_interpretation.CLASS_COMPONENT_TYPE_STRING)
IDEAL_ACTIVATION_HELP_STRING = (
    '[used only if {0:s} = "{1:s}" or "{2:s}"] The loss function will be '
    '(neuron_activation - ideal_activation)**2 or [max(channel_activations) - '
    'ideal_activation]**2.  If {3:s} = -1, the loss function will be '
    '-sign(neuron_activation) * neuron_activation**2 or '
    '-sign(max(channel_activations)) * max(channel_activations)**2.'
).format(COMPONENT_TYPE_ARG_NAME,
         model_interpretation.NEURON_COMPONENT_TYPE_STRING,
         model_interpretation.CHANNEL_COMPONENT_TYPE_STRING,
         IDEAL_ACTIVATION_ARG_NAME)
NEURON_INDICES_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] Indices of neuron whose saliency map is to '
    'be computed.  For example, to compute saliency maps for neuron (0, 0, 2), '
    'this argument should be "0 0 2".'
).format(COMPONENT_TYPE_ARG_NAME,
         model_interpretation.NEURON_COMPONENT_TYPE_STRING)
CHANNEL_INDEX_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] Index of channel whose saliency map is to '
    'be computed.'
).format(COMPONENT_TYPE_ARG_NAME,
         model_interpretation.CHANNEL_COMPONENT_TYPE_STRING)
RADAR_IMAGE_DIR_HELP_STRING = (
    'Name of top-level directory with storm-centered radar images.  Files '
    'therein will be found by `storm_images.find_storm_image_file` and read by '
    '`storm_images.read_storm_images`.')
SOUNDING_DIR_HELP_STRING = (
    'Name of top-level directory with storm-centered soundings.  Files therein '
    'will be found by `soundings_only.find_sounding_file` and read by '
    '`soundings_only.read_soundings`.')
STORM_IDS_HELP_STRING = (
    'List of storm IDs (must have the same length as `{0:s}`).  Saliency maps '
    'will be computed for each storm object.'
).format(STORM_TIMES_ARG_NAME)
STORM_TIMES_HELP_STRING = (
    'List of storm times (must have the same length as `{0:s}`).  Saliency maps'
    ' will be computed for each storm object.'
).format(STORM_IDS_ARG_NAME)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file (will be written by `saliency_maps.write_file`).')

DEFAULT_TOP_RADAR_IMAGE_DIR_NAME = (
    '/condo/swatcommon/common/gridrad_final/myrorss_format/tracks/reanalyzed/'
    'storm_images_rotated')
DEFAULT_TOP_SOUNDING_DIR_NAME = (
    '/condo/swatcommon/common/gridrad_final/myrorss_format/tracks/reanalyzed/'
    'soundings')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + COMPONENT_TYPE_ARG_NAME, type=str, required=True,
    help=COMPONENT_TYPE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_CLASS_ARG_NAME, type=int, required=False, default=1,
    help=TARGET_CLASS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAYER_NAME_ARG_NAME, type=str, required=False, default='',
    help=LAYER_NAME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + IDEAL_ACTIVATION_ARG_NAME, type=float, required=False,
    default=saliency_maps.DEFAULT_IDEAL_ACTIVATION,
    help=IDEAL_ACTIVATION_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NEURON_INDICES_ARG_NAME, type=int, nargs='+', required=False,
    default=[-1], help=NEURON_INDICES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CHANNEL_INDEX_ARG_NAME, type=int, required=False, default=-1,
    help=CHANNEL_INDEX_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_IMAGE_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_TOP_RADAR_IMAGE_DIR_NAME,
    help=RADAR_IMAGE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SOUNDING_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_TOP_SOUNDING_DIR_NAME, help=SOUNDING_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + STORM_IDS_ARG_NAME, type=str, nargs='+', required=True,
    help=STORM_IDS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + STORM_TIMES_ARG_NAME, type=int, nargs='+', required=True,
    help=STORM_TIMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _run(
        model_file_name, component_type_string, target_class, layer_name,
        ideal_activation, neuron_indices, channel_index,
        top_radar_image_dir_name, top_sounding_dir_name, desired_storm_ids,
        desired_storm_times_unix_sec, output_file_name):
    """Computes saliency map for each storm object and each model component.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param component_type_string: Same.
    :param target_class: Same.
    :param layer_name: Same.
    :param ideal_activation: Same.
    :param neuron_indices_flattened: Same.
    :param channel_indices: Same.
    :param top_radar_image_dir_name: Same.
    :param top_sounding_dir_name: Same.
    :param desired_storm_ids: Same.
    :param desired_storm_times_unix_sec: Same.
    :param output_file_name: Same.
    """

    # Check input args.
    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)
    model_interpretation.check_component_type(component_type_string)

    # Read model and metadata.
    print 'Reading model from: "{0:s}"...'.format(model_file_name)
    model_object = cnn.read_model(model_file_name)

    metadata_file_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0])
    print 'Reading metadata from: "{0:s}"...'.format(metadata_file_name)
    model_metadata_dict = cnn.read_model_metadata(metadata_file_name)

    # Find model input.
    num_radar_dimensions = len(
        model_metadata_dict[cnn.TRAINING_FILE_NAMES_KEY].shape)
    print SEPARATOR_STRING

    if num_radar_dimensions == 2:
        radar_file_name_matrix = trainval_io.find_radar_files_2d(
            top_directory_name=top_radar_image_dir_name,
            radar_source=model_metadata_dict[cnn.RADAR_SOURCE_KEY],
            radar_field_names=model_metadata_dict[
                cnn.RADAR_FIELD_NAMES_KEY],
            first_file_time_unix_sec=numpy.min(desired_storm_times_unix_sec),
            last_file_time_unix_sec=numpy.max(desired_storm_times_unix_sec),
            one_file_per_time_step=False, shuffle_times=False,
            radar_heights_m_asl=model_metadata_dict[cnn.RADAR_HEIGHTS_KEY],
            reflectivity_heights_m_asl=model_metadata_dict[
                cnn.REFLECTIVITY_HEIGHTS_KEY])[0]
    else:
        radar_file_name_matrix = trainval_io.find_radar_files_3d(
            top_directory_name=top_radar_image_dir_name,
            radar_source=model_metadata_dict[cnn.RADAR_SOURCE_KEY],
            radar_field_names=model_metadata_dict[
                cnn.RADAR_FIELD_NAMES_KEY],
            radar_heights_m_asl=model_metadata_dict[cnn.RADAR_HEIGHTS_KEY],
            first_file_time_unix_sec=numpy.min(desired_storm_times_unix_sec),
            last_file_time_unix_sec=numpy.max(desired_storm_times_unix_sec),
            one_file_per_time_step=False, shuffle_times=False)[0]

    print SEPARATOR_STRING

    list_of_input_matrices = None
    list_of_saliency_matrices = None
    storm_ids = []
    storm_times_unix_sec = numpy.array([], dtype=int)
    num_spc_dates = radar_file_name_matrix.shape[0]

    for i in range(num_spc_dates):
        (this_list_of_input_matrices, these_storm_ids, these_times_unix_sec
        ) = model_interpretation.read_storms_one_spc_date(
            radar_file_name_matrix=radar_file_name_matrix,
            model_metadata_dict=model_metadata_dict,
            top_sounding_dir_name=top_sounding_dir_name, spc_date_index=i,
            desired_storm_ids=desired_storm_ids,
            desired_storm_times_unix_sec=desired_storm_times_unix_sec)
        print MINOR_SEPARATOR_STRING

        if this_list_of_input_matrices is None:
            continue

        storm_ids += these_storm_ids
        storm_times_unix_sec = numpy.concatenate((
            storm_times_unix_sec, these_times_unix_sec))

        if num_radar_dimensions == 2:
            _, this_spc_date_string = storm_images.image_file_name_to_time(
                radar_file_name_matrix[i, 0])
        else:
            _, this_spc_date_string = storm_images.image_file_name_to_time(
                radar_file_name_matrix[i, 0, 0])

        if (component_type_string ==
                model_interpretation.CLASS_COMPONENT_TYPE_STRING):
            print (
                'Computing saliency maps for target class {0:d} and SPC date '
                '"{1:s}"...'
            ).format(target_class, this_spc_date_string)

            this_list_of_saliency_matrices = (
                saliency_maps.get_saliency_maps_for_class_activation(
                    model_object=model_object, target_class=target_class,
                    list_of_input_matrices=this_list_of_input_matrices)
            )

        elif (component_type_string ==
              model_interpretation.NEURON_COMPONENT_TYPE_STRING):
            print (
                'Computing saliency maps for neuron {0:s} in layer "{1:s}"...'
            ).format(str(neuron_indices), layer_name)

            this_list_of_saliency_matrices = (
                saliency_maps.get_saliency_maps_for_neuron_activation(
                    model_object=model_object, layer_name=layer_name,
                    neuron_indices=neuron_indices,
                    list_of_input_matrices=this_list_of_input_matrices,
                    ideal_activation=ideal_activation)
            )

        else:
            print (
                'Computing saliency maps for channel {0:d} in layer "{1:s}"...'
            ).format(channel_index, layer_name)

            this_list_of_saliency_matrices = (
                saliency_maps.get_saliency_maps_for_channel_activation(
                    model_object=model_object, layer_name=layer_name,
                    channel_index=channel_index,
                    list_of_input_matrices=this_list_of_input_matrices,
                    stat_function_for_neuron_activations=K.max,
                    ideal_activation=ideal_activation)
            )

        if list_of_saliency_matrices is None:
            list_of_input_matrices = copy.deepcopy(this_list_of_input_matrices)
            list_of_saliency_matrices = copy.deepcopy(
                this_list_of_saliency_matrices)
        else:
            for k in range(len(list_of_saliency_matrices)):
                list_of_input_matrices[k] = numpy.concatenate(
                    (list_of_input_matrices[k], this_list_of_input_matrices[k]),
                    axis=0)

                list_of_saliency_matrices[k] = numpy.concatenate(
                    (list_of_saliency_matrices[k],
                     this_list_of_saliency_matrices[k]),
                    axis=0)

        if i == num_spc_dates - 1:
            print SEPARATOR_STRING
        else:
            print MINOR_SEPARATOR_STRING

    print 'Writing saliency maps to file: "{0:s}"...'.format(output_file_name)
    saliency_maps.write_file(
        pickle_file_name=output_file_name,
        list_of_input_matrices=list_of_input_matrices,
        list_of_saliency_matrices=list_of_saliency_matrices,
        model_file_name=model_file_name, storm_ids=storm_ids,
        storm_times_unix_sec=storm_times_unix_sec,
        component_type_string=component_type_string, target_class=target_class,
        layer_name=layer_name, ideal_activation=ideal_activation,
        neuron_indices=neuron_indices, channel_index=channel_index)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        component_type_string=getattr(
            INPUT_ARG_OBJECT, COMPONENT_TYPE_ARG_NAME),
        target_class=getattr(INPUT_ARG_OBJECT, TARGET_CLASS_ARG_NAME),
        layer_name=getattr(INPUT_ARG_OBJECT, LAYER_NAME_ARG_NAME),
        ideal_activation=getattr(INPUT_ARG_OBJECT, IDEAL_ACTIVATION_ARG_NAME),
        neuron_indices=numpy.array(
            getattr(INPUT_ARG_OBJECT, NEURON_INDICES_ARG_NAME), dtype=int),
        channel_index=getattr(INPUT_ARG_OBJECT, CHANNEL_INDEX_ARG_NAME),
        top_radar_image_dir_name=getattr(
            INPUT_ARG_OBJECT, RADAR_IMAGE_DIR_ARG_NAME),
        top_sounding_dir_name=getattr(INPUT_ARG_OBJECT, SOUNDING_DIR_ARG_NAME),
        desired_storm_ids=getattr(INPUT_ARG_OBJECT, STORM_IDS_ARG_NAME),
        desired_storm_times_unix_sec=numpy.array(
            getattr(INPUT_ARG_OBJECT, STORM_TIMES_ARG_NAME), dtype=int),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME))
