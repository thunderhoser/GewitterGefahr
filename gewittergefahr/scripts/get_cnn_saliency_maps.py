"""Computes saliency maps for the given class, neurons, or channels of a CNN.

CNN = convolutional neural network
"""

import copy
import os.path
import argparse
import numpy
from keras import backend as K
from gewittergefahr.gg_utils import general_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import deployment_io
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.deep_learning import feature_optimization

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)))

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

LARGE_INTEGER = int(1e10)
INPUT_TIME_FORMAT = '%Y-%m-%d-%H%M%S'

MODEL_FILE_ARG_NAME = 'model_file_name'
COMPONENT_TYPE_ARG_NAME = 'component_type_string'
TARGET_CLASS_ARG_NAME = 'target_class'
RETURN_PROBS_ARG_NAME = 'return_probs'
IDEAL_LOGIT_ARG_NAME = 'ideal_logit'
LAYER_NAME_ARG_NAME = 'layer_name'
IDEAL_ACTIVATION_ARG_NAME = 'ideal_activation'
NEURON_INDICES_ARG_NAME = 'neuron_indices'
CHANNEL_INDICES_ARG_NAME = 'channel_indices'
RADAR_IMAGE_DIR_ARG_NAME = 'input_radar_image_dir_name'
SOUNDING_DIR_ARG_NAME = 'input_sounding_dir_name'
STORM_ID_ARG_NAME = 'storm_id'
STORM_TIME_ARG_NAME = 'storm_time_string'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

MODEL_FILE_HELP_STRING = (
    'Path to input file, containing a trained CNN.  Will be read by '
    '`cnn.read_model`.')
COMPONENT_TYPE_HELP_STRING = (
    'Component type.  Saliency maps may be computed for one class, one/many '
    'neurons, or one/many channels.  Valid options are listed below.\n{0:s}'
).format(str(feature_optimization.VALID_COMPONENT_TYPE_STRINGS))
TARGET_CLASS_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] Saliency maps will be computed for class k,'
    ' where k = `{2:s}`.'
).format(COMPONENT_TYPE_ARG_NAME,
         feature_optimization.CLASS_COMPONENT_TYPE_STRING,
         TARGET_CLASS_ARG_NAME)
RETURN_PROBS_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] Boolean flag.  If 1, saliency maps will be '
    'created for the predicted probability of class k, where k = `{2:s}`.  If '
    '0, saliency maps will be created for the pre-softmax logit for class k.'
).format(COMPONENT_TYPE_ARG_NAME,
         feature_optimization.CLASS_COMPONENT_TYPE_STRING,
         TARGET_CLASS_ARG_NAME)
IDEAL_LOGIT_HELP_STRING = (
    '[used only if {0:s} = "{1:s}" and {2:s} = 0] The loss function will be '
    '(logit[k] - {3:s}) ** 2, where logit[k] is the logit for the target class.'
    '  If {3:s} = -1, the loss function will be -sign(logit[k]) * logit[k]**2, '
    'or the negative signed square of logit[k], so that loss always decreases '
    'as logit[k] increases.'
).format(COMPONENT_TYPE_ARG_NAME,
         feature_optimization.CLASS_COMPONENT_TYPE_STRING,
         RETURN_PROBS_ARG_NAME, IDEAL_LOGIT_ARG_NAME)
LAYER_NAME_HELP_STRING = (
    '[used only if {0:s} = "{1:s}" or "{2:s}"] Name of layer with neurons or '
    'channels for which saliency maps will be computed.'
).format(COMPONENT_TYPE_ARG_NAME,
         feature_optimization.NEURON_COMPONENT_TYPE_STRING,
         feature_optimization.CLASS_COMPONENT_TYPE_STRING)
IDEAL_ACTIVATION_HELP_STRING = (
    '[used only if {0:s} = "{1:s}" or "{2:s}"] The loss function will be '
    '(neuron_activation - ideal_activation)**2 or [max(channel_activations) - '
    'ideal_activation]**2.  If {3:s} = -1, the loss function will be '
    '-sign(neuron_activation) * neuron_activation**2 or '
    '-sign(max(channel_activations)) * max(channel_activations)**2.'
).format(COMPONENT_TYPE_ARG_NAME,
         feature_optimization.NEURON_COMPONENT_TYPE_STRING,
         feature_optimization.CHANNEL_COMPONENT_TYPE_STRING,
         IDEAL_ACTIVATION_ARG_NAME)
NEURON_INDICES_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] Indices for each neuron whose saliency map '
    'is to be computed.  For example, to compute saliency maps for neuron '
    '(0, 0, 2), this argument should be "0 0 2".  To compute saliency maps for '
    'neurons (0, 0, 2) and (1, 1, 2), this list should be "0 0 2 -1 1 1 2".  In'
    ' other words, use -1 to separate neurons.'
).format(COMPONENT_TYPE_ARG_NAME,
         feature_optimization.NEURON_COMPONENT_TYPE_STRING)
CHANNEL_INDICES_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] Index for each channel whose saliency map '
    'is to be computed.'
).format(COMPONENT_TYPE_ARG_NAME,
         feature_optimization.CHANNEL_COMPONENT_TYPE_STRING)
RADAR_IMAGE_DIR_HELP_STRING = (
    'Name of top-level directory with storm-centered radar images.  Files '
    'therein will be found by `storm_images.find_storm_image_file` and read by '
    '`storm_images.read_storm_images`.')
SOUNDING_DIR_HELP_STRING = (
    'Name of top-level directory with storm-centered soundings.  Files therein '
    'will be found by `soundings_only.find_sounding_file` and read by '
    '`soundings_only.read_soundings`.')
STORM_ID_HELP_STRING = (
    'Storm ID.  Saliency maps will be created only for this storm at time '
    '`{0:s}`.'
).format(STORM_TIME_ARG_NAME)
STORM_TIME_HELP_STRING = (
    'Storm time (format "yyyy-mm-dd-HHMMSS").  Saliency maps will be created '
    'only for storm `{0:s}` at this time.'
).format(STORM_ID_ARG_NAME)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file (will be written by `feature_optimization.'
    'write_saliency_maps_to_file`).')

DEFAULT_TOP_RADAR_IMAGE_DIR_NAME = (
    '/condo/swatcommon/common/gridrad_final/myrorss_format/tracks/reanalyzed/'
    'storm_images_with_rdp')
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
    '--' + RETURN_PROBS_ARG_NAME, type=int, required=False,
    default=1, help=RETURN_PROBS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + IDEAL_LOGIT_ARG_NAME, type=float, required=False,
    default=feature_optimization.DEFAULT_IDEAL_LOGIT,
    help=IDEAL_LOGIT_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAYER_NAME_ARG_NAME, type=str, required=False, default='',
    help=LAYER_NAME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + IDEAL_ACTIVATION_ARG_NAME, type=float, required=False,
    default=feature_optimization.DEFAULT_IDEAL_ACTIVATION,
    help=IDEAL_ACTIVATION_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NEURON_INDICES_ARG_NAME, type=int, nargs='+', required=False,
    default=[-1], help=NEURON_INDICES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CHANNEL_INDICES_ARG_NAME, type=int, nargs='+', required=False,
    default=[-1], help=CHANNEL_INDICES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_IMAGE_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_TOP_RADAR_IMAGE_DIR_NAME,
    help=RADAR_IMAGE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SOUNDING_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_TOP_SOUNDING_DIR_NAME, help=SOUNDING_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + STORM_ID_ARG_NAME, type=str, required=True,
    help=STORM_ID_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + STORM_TIME_ARG_NAME, type=str, required=True,
    help=STORM_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _read_input_one_storm_object(
        radar_file_name_matrix, model_metadata_dict, top_sounding_dir_name,
        storm_id, storm_time_unix_sec):
    """Reads model input for one storm object.

    :param radar_file_name_matrix: numpy array of file names, created by either
        `training_validation_io.find_radar_files_2d` or
        `training_validation_io.find_radar_files_3d`.
    :param model_metadata_dict: Dictionary created by `cnn.read_model_metadata`.
    :param top_sounding_dir_name: Name of top-level directory with storm-
        centered soundings.
    :param storm_id: String ID for storm cell.
    :param storm_time_unix_sec: Valid time for storm object.
    :return: list_of_input_matrices: length-T list of numpy arrays, where T =
        number of input tensors to the model.  Each array contains data for only
        the given storm object (ID-time pair).
    """

    if model_metadata_dict[cnn.USE_2D3D_CONVOLUTION_KEY]:
        example_dict = deployment_io.create_storm_images_2d3d_myrorss(
            radar_file_name_matrix=radar_file_name_matrix[[0], ...],
            num_examples_per_file_time=LARGE_INTEGER, return_target=False,
            target_name=model_metadata_dict[cnn.TARGET_NAME_KEY],
            radar_normalization_dict=model_metadata_dict[
                cnn.RADAR_NORMALIZATION_DICT_KEY],
            sounding_field_names=model_metadata_dict[
                cnn.SOUNDING_FIELD_NAMES_KEY],
            top_sounding_dir_name=top_sounding_dir_name,
            sounding_lag_time_for_convective_contamination_sec=
            model_metadata_dict[cnn.SOUNDING_LAG_TIME_KEY],
            sounding_normalization_dict=model_metadata_dict[
                cnn.SOUNDING_NORMALIZATION_DICT_KEY])
    else:
        num_radar_dimensions = len(
            model_metadata_dict[cnn.TRAINING_FILE_NAMES_KEY].shape)
        if num_radar_dimensions == 3:
            example_dict = deployment_io.create_storm_images_3d(
                radar_file_name_matrix=radar_file_name_matrix[[0], ...],
                num_examples_per_file_time=LARGE_INTEGER, return_target=False,
                target_name=model_metadata_dict[cnn.TARGET_NAME_KEY],
                radar_normalization_dict=model_metadata_dict[
                    cnn.RADAR_NORMALIZATION_DICT_KEY],
                refl_masking_threshold_dbz=model_metadata_dict[
                    cnn.REFL_MASKING_THRESHOLD_KEY],
                return_rotation_divergence_product=False,
                sounding_field_names=model_metadata_dict[
                    cnn.SOUNDING_FIELD_NAMES_KEY],
                top_sounding_dir_name=top_sounding_dir_name,
                sounding_lag_time_for_convective_contamination_sec=
                model_metadata_dict[cnn.SOUNDING_LAG_TIME_KEY],
                sounding_normalization_dict=model_metadata_dict[
                    cnn.SOUNDING_NORMALIZATION_DICT_KEY])
        else:
            example_dict = deployment_io.create_storm_images_2d(
                radar_file_name_matrix=radar_file_name_matrix[[0], ...],
                num_examples_per_file_time=LARGE_INTEGER, return_target=False,
                target_name=model_metadata_dict[cnn.TARGET_NAME_KEY],
                radar_normalization_dict=model_metadata_dict[
                    cnn.RADAR_NORMALIZATION_DICT_KEY],
                sounding_field_names=model_metadata_dict[
                    cnn.SOUNDING_FIELD_NAMES_KEY],
                top_sounding_dir_name=top_sounding_dir_name,
                sounding_lag_time_for_convective_contamination_sec=
                model_metadata_dict[cnn.SOUNDING_LAG_TIME_KEY],
                sounding_normalization_dict=model_metadata_dict[
                    cnn.SOUNDING_NORMALIZATION_DICT_KEY])

    storm_object_index_as_array = numpy.where(numpy.logical_and(
        numpy.array(example_dict[deployment_io.STORM_IDS_KEY]) == storm_id,
        example_dict[deployment_io.STORM_TIMES_KEY] == storm_time_unix_sec
    ))[0]

    if model_metadata_dict[cnn.USE_2D3D_CONVOLUTION_KEY]:
        list_of_input_matrices = [
            example_dict[deployment_io.REFLECTIVITY_MATRIX_KEY][
                storm_object_index_as_array, ...],
            example_dict[deployment_io.AZ_SHEAR_MATRIX_KEY][
                storm_object_index_as_array, ...]
        ]
    else:
        list_of_input_matrices = [
            example_dict[deployment_io.RADAR_IMAGE_MATRIX_KEY][
                storm_object_index_as_array, ...]
        ]

    if example_dict[deployment_io.SOUNDING_MATRIX_KEY] is not None:
        list_of_input_matrices.append(
            example_dict[deployment_io.SOUNDING_MATRIX_KEY][
                storm_object_index_as_array, ...])

    return list_of_input_matrices


def _run(
        model_file_name, component_type_string, target_class, return_probs,
        ideal_logit, layer_name, ideal_activation, neuron_indices_flattened,
        channel_indices, top_radar_image_dir_name, top_sounding_dir_name,
        storm_id, storm_time_string, output_file_name):
    """Computes saliency maps for given class, neurons, or channels of a CNN.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param component_type_string: Same.
    :param target_class: Same.
    :param return_probs: Same.
    :param ideal_logit: Same.
    :param layer_name: Same.
    :param ideal_activation: Same.
    :param neuron_indices_flattened: Same.
    :param channel_indices: Same.
    :param top_radar_image_dir_name: Same.
    :param top_sounding_dir_name: Same.
    :param storm_id: Same.
    :param storm_time_string: Same.
    :param output_file_name: Same.
    """

    # Check input args.
    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)
    feature_optimization.check_component_type(component_type_string)
    storm_time_unix_sec = time_conversion.string_to_unix_sec(
        storm_time_string, INPUT_TIME_FORMAT)

    if (component_type_string ==
            feature_optimization.CHANNEL_COMPONENT_TYPE_STRING):
        error_checking.assert_is_geq_numpy_array(channel_indices, 0)

    if (component_type_string ==
            feature_optimization.NEURON_COMPONENT_TYPE_STRING):
        neuron_indices_flattened = neuron_indices_flattened.astype(float)
        neuron_indices_flattened[neuron_indices_flattened < 0] = numpy.nan

        neuron_indices_2d_list = general_utils.split_array_by_nan(
            neuron_indices_flattened)
        neuron_index_matrix = numpy.array(neuron_indices_2d_list, dtype=int)
    else:
        neuron_index_matrix = None

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
            first_file_time_unix_sec=storm_time_unix_sec,
            last_file_time_unix_sec=storm_time_unix_sec,
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
            first_file_time_unix_sec=storm_time_unix_sec,
            last_file_time_unix_sec=storm_time_unix_sec,
            one_file_per_time_step=False, shuffle_times=False)[0]

    print SEPARATOR_STRING
    list_of_input_matrices = _read_input_one_storm_object(
        radar_file_name_matrix=radar_file_name_matrix,
        model_metadata_dict=model_metadata_dict,
        top_sounding_dir_name=top_sounding_dir_name,
        storm_id=storm_id, storm_time_unix_sec=storm_time_unix_sec)
    print SEPARATOR_STRING

    # Create saliency maps.
    list_of_saliency_matrices = None

    if (component_type_string ==
            feature_optimization.CLASS_COMPONENT_TYPE_STRING):
        print 'Computing saliency map for target class {0:d}...'.format(
            target_class)
        list_of_saliency_matrices = (
            feature_optimization.get_saliency_maps_for_class_activation(
                model_object=model_object, target_class=target_class,
                return_probs=return_probs,
                list_of_input_matrices=list_of_input_matrices,
                ideal_logit=ideal_logit))

    elif (component_type_string ==
          feature_optimization.NEURON_COMPONENT_TYPE_STRING):

        for j in range(neuron_index_matrix.shape[0]):
            print (
                'Computing saliency map for neuron {0:s} in layer "{1:s}"...'
            ).format(str(neuron_index_matrix[j, :]), layer_name)

            these_matrices = (
                feature_optimization.get_saliency_maps_for_neuron_activation(
                    model_object=model_object, layer_name=layer_name,
                    neuron_indices=neuron_index_matrix[j, ...],
                    list_of_input_matrices=list_of_input_matrices,
                    ideal_activation=ideal_activation))

            if list_of_saliency_matrices is None:
                list_of_saliency_matrices = copy.deepcopy(these_matrices)
            else:
                for k in range(len(list_of_saliency_matrices)):
                    list_of_saliency_matrices[k] = numpy.concatenate(
                        (list_of_saliency_matrices[k], these_matrices[k]),
                        axis=0)
    else:
        for this_channel_index in channel_indices:
            print (
                'Computing saliency map for channel {0:d} in layer "{1:s}"...'
            ).format(this_channel_index, layer_name)

            these_matrices = (
                feature_optimization.get_saliency_maps_for_channel_activation(
                    model_object=model_object, layer_name=layer_name,
                    channel_index=this_channel_index,
                    list_of_input_matrices=list_of_input_matrices,
                    stat_function_for_neuron_activations=K.max,
                    ideal_activation=ideal_activation))

            if list_of_saliency_matrices is None:
                list_of_saliency_matrices = copy.deepcopy(these_matrices)
            else:
                for k in range(len(list_of_saliency_matrices)):
                    list_of_saliency_matrices[k] = numpy.concatenate(
                        (list_of_saliency_matrices[k], these_matrices[k]),
                        axis=0)

    print SEPARATOR_STRING
    print 'Writing saliency maps to file: "{0:s}"...'.format(output_file_name)
    feature_optimization.write_saliency_maps_to_file(
        pickle_file_name=output_file_name,
        list_of_saliency_matrices=list_of_saliency_matrices,
        model_file_name=model_file_name, storm_id=storm_id,
        storm_time_unix_sec=storm_time_unix_sec,
        component_type_string=component_type_string, target_class=target_class,
        return_probs=return_probs, ideal_logit=ideal_logit,
        layer_name=layer_name, ideal_activation=ideal_activation,
        neuron_index_matrix=neuron_index_matrix,
        channel_indices=channel_indices)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        component_type_string=getattr(
            INPUT_ARG_OBJECT, COMPONENT_TYPE_ARG_NAME),
        target_class=getattr(INPUT_ARG_OBJECT, TARGET_CLASS_ARG_NAME),
        return_probs=bool(getattr(INPUT_ARG_OBJECT, RETURN_PROBS_ARG_NAME)),
        ideal_logit=getattr(INPUT_ARG_OBJECT, IDEAL_LOGIT_ARG_NAME),
        layer_name=getattr(INPUT_ARG_OBJECT, LAYER_NAME_ARG_NAME),
        ideal_activation=getattr(INPUT_ARG_OBJECT, IDEAL_ACTIVATION_ARG_NAME),
        neuron_indices_flattened=numpy.array(
            getattr(INPUT_ARG_OBJECT, NEURON_INDICES_ARG_NAME), dtype=int),
        channel_indices=numpy.array(
            getattr(INPUT_ARG_OBJECT, CHANNEL_INDICES_ARG_NAME), dtype=int),
        top_radar_image_dir_name=getattr(
            INPUT_ARG_OBJECT, RADAR_IMAGE_DIR_ARG_NAME),
        top_sounding_dir_name=getattr(INPUT_ARG_OBJECT, SOUNDING_DIR_ARG_NAME),
        storm_id=getattr(INPUT_ARG_OBJECT, STORM_ID_ARG_NAME),
        storm_time_string=getattr(INPUT_ARG_OBJECT, STORM_TIME_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME))
