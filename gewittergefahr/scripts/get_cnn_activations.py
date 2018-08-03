"""Computes activation for the given class, neurons, or channels of a CNN.

CNN = convolutional neural network
"""

import os.path
import argparse
import numpy
from keras import backend as K
from gewittergefahr.gg_utils import general_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import storm_images
from gewittergefahr.deep_learning import model_interpretation
from gewittergefahr.deep_learning import model_activation
from gewittergefahr.deep_learning import deployment_io
from gewittergefahr.deep_learning import training_validation_io as trainval_io

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)))

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

LARGE_INTEGER = int(1e10)

MODEL_FILE_ARG_NAME = 'model_file_name'
COMPONENT_TYPE_ARG_NAME = 'component_type_string'
TARGET_CLASS_ARG_NAME = 'target_class'
RETURN_PROBS_ARG_NAME = 'return_probs'
LAYER_NAME_ARG_NAME = 'layer_name'
NEURON_INDICES_ARG_NAME = 'neuron_indices'
CHANNEL_INDICES_ARG_NAME = 'channel_indices'
RADAR_IMAGE_DIR_ARG_NAME = 'input_radar_image_dir_name'
SOUNDING_DIR_ARG_NAME = 'input_sounding_dir_name'
FIRST_SPC_DATE_ARG_NAME = 'first_spc_date_string'
LAST_SPC_DATE_ARG_NAME = 'last_spc_date_string'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

MODEL_FILE_HELP_STRING = (
    'Path to input file, containing a trained CNN.  Will be read by '
    '`cnn.read_model`.')
COMPONENT_TYPE_HELP_STRING = (
    'Component type.  Activations may be computed for one class, one/many '
    'neurons, or one/many channels.  Valid options are listed below.\n{0:s}'
).format(str(model_interpretation.VALID_COMPONENT_TYPE_STRINGS))
TARGET_CLASS_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] Activations will be computed for class k, '
    'where k = `{2:s}`.'
).format(COMPONENT_TYPE_ARG_NAME,
         model_interpretation.CLASS_COMPONENT_TYPE_STRING,
         TARGET_CLASS_ARG_NAME)
RETURN_PROBS_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] Boolean flag.  If 1, the activation for '
    'each example will be the predicted probability of class k, where k = '
    '`{2:s}`.  If 0, the activation for each example will be pre-softmax logit '
    'for class k.'
).format(COMPONENT_TYPE_ARG_NAME,
         model_interpretation.CLASS_COMPONENT_TYPE_STRING,
         TARGET_CLASS_ARG_NAME)
LAYER_NAME_HELP_STRING = (
    '[used only if {0:s} = "{1:s}" or "{2:s}"] Name of layer with neurons or '
    'channels whose activations will be computed.'
).format(COMPONENT_TYPE_ARG_NAME,
         model_interpretation.NEURON_COMPONENT_TYPE_STRING,
         model_interpretation.CLASS_COMPONENT_TYPE_STRING)
NEURON_INDICES_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] Indices for each neuron whose activation is'
    ' to be computed.  For example, to compute activations for neuron (0, 0, 2)'
    ', this argument should be "0 0 2".  To compute activations for neurons '
    '(0, 0, 2) and (1, 1, 2), this list should be "0 0 2 -1 1 1 2".  In other '
    'words, use -1 to separate neurons.'
).format(COMPONENT_TYPE_ARG_NAME,
         model_interpretation.NEURON_COMPONENT_TYPE_STRING)
CHANNEL_INDICES_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] Index for each channel whose activation is '
    'to be computed.'
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
SPC_DATE_HELP_STRING = (
    'SPC (Storm Prediction Center) date in format "yyyymmdd".  Activation will '
    'be computed for each model component and each example (storm object) from '
    '`{0:s}`...`{1:s}`.'
).format(FIRST_SPC_DATE_ARG_NAME, LAST_SPC_DATE_ARG_NAME)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file (will be written by `model_activation.write_file`).')

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
    '--' + LAYER_NAME_ARG_NAME, type=str, required=False, default='',
    help=LAYER_NAME_HELP_STRING)

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
    '--' + FIRST_SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _read_input_one_spc_date(
        radar_file_name_matrix, model_metadata_dict, top_sounding_dir_name,
        spc_date_index):
    """Reads model input for one SPC date.

    E = number of examples (storm objects)

    :param radar_file_name_matrix: numpy array of file names, created by either
        `training_validation_io.find_radar_files_2d` or
        `training_validation_io.find_radar_files_3d`.
    :param model_metadata_dict: Dictionary created by `cnn.read_model_metadata`.
    :param top_sounding_dir_name: Name of top-level directory with storm-
        centered soundings.
    :param spc_date_index: Index of SPC date.  Data will be read from
        radar_file_name_matrix[i, ...], where i = `spc_date_index`.
    :return: list_of_input_matrices: length-T list of numpy arrays, where T =
        number of input tensors to the model.  The first dimension of each array
        has length E.
    :return: storm_ids: length-E list of storm IDs.
    :return: storm_times_unix_sec: length-E list of storm times.
    """

    if model_metadata_dict[cnn.USE_2D3D_CONVOLUTION_KEY]:
        example_dict = deployment_io.create_storm_images_2d3d_myrorss(
            radar_file_name_matrix=radar_file_name_matrix[
                [spc_date_index], ...],
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

        if example_dict is not None:
            list_of_input_matrices = [
                example_dict[deployment_io.REFLECTIVITY_MATRIX_KEY],
                example_dict[deployment_io.AZ_SHEAR_MATRIX_KEY]
            ]

    else:
        num_radar_dimensions = len(
            model_metadata_dict[cnn.TRAINING_FILE_NAMES_KEY].shape)
        if num_radar_dimensions == 3:
            example_dict = deployment_io.create_storm_images_3d(
                radar_file_name_matrix=radar_file_name_matrix[
                    [spc_date_index], ...],
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
                radar_file_name_matrix=radar_file_name_matrix[
                    [spc_date_index], ...],
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

        if example_dict is not None:
            list_of_input_matrices = [
                example_dict[deployment_io.RADAR_IMAGE_MATRIX_KEY]
            ]

    if example_dict is None:
        return None, None, None

    if example_dict[deployment_io.SOUNDING_MATRIX_KEY] is not None:
        list_of_input_matrices.append(
            example_dict[deployment_io.SOUNDING_MATRIX_KEY])

    return (list_of_input_matrices, example_dict[deployment_io.STORM_IDS_KEY],
            example_dict[deployment_io.STORM_TIMES_KEY])


def _run(
        model_file_name, component_type_string, target_class, return_probs,
        layer_name, neuron_indices_flattened, channel_indices,
        top_radar_image_dir_name, top_sounding_dir_name, first_spc_date_string,
        last_spc_date_string, output_file_name):
    """Creates activation maps for one class, neuron, or channel of a CNN.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param component_type_string: Same.
    :param target_class: Same.
    :param return_probs: Same.
    :param layer_name: Same.
    :param neuron_indices_flattened: Same.
    :param channel_indices: Same.
    :param top_radar_image_dir_name: Same.
    :param top_sounding_dir_name: Same.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
    :param output_file_name: Same.
    """

    # Check input args.
    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)
    model_interpretation.check_component_type(component_type_string)

    if (component_type_string ==
            model_interpretation.CHANNEL_COMPONENT_TYPE_STRING):
        error_checking.assert_is_geq_numpy_array(channel_indices, 0)

    if (component_type_string ==
            model_interpretation.NEURON_COMPONENT_TYPE_STRING):
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
            first_file_time_unix_sec=
            time_conversion.spc_date_string_to_unix_sec(
                first_spc_date_string),
            last_file_time_unix_sec=
            time_conversion.spc_date_string_to_unix_sec(
                last_spc_date_string),
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
            first_file_time_unix_sec=
            time_conversion.spc_date_string_to_unix_sec(
                first_spc_date_string),
            last_file_time_unix_sec=
            time_conversion.spc_date_string_to_unix_sec(
                last_spc_date_string),
            one_file_per_time_step=False, shuffle_times=False)[0]

    print SEPARATOR_STRING

    # Compute activation for each example (storm object) and model component.
    activation_matrix = None
    storm_ids = []
    storm_times_unix_sec = numpy.array([], dtype=int)
    num_spc_dates = radar_file_name_matrix.shape[0]

    for i in range(num_spc_dates):
        (this_list_of_input_matrices, these_storm_ids, these_times_unix_sec
        ) = _read_input_one_spc_date(
            radar_file_name_matrix=radar_file_name_matrix,
            model_metadata_dict=model_metadata_dict,
            top_sounding_dir_name=top_sounding_dir_name, spc_date_index=i)
        print '\n'

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
                'Computing activations for target class {0:d} and SPC date '
                '"{1:s}"...'
            ).format(target_class, this_spc_date_string)

            this_activation_matrix = (
                model_activation.get_class_activation_for_examples(
                    model_object=model_object, target_class=target_class,
                    return_probs=return_probs,
                    list_of_input_matrices=this_list_of_input_matrices))

            this_activation_matrix = numpy.reshape(
                this_activation_matrix, (len(this_activation_matrix), 1))

        elif (component_type_string ==
              model_interpretation.NEURON_COMPONENT_TYPE_STRING):
            this_activation_matrix = None

            for j in range(neuron_index_matrix.shape[0]):
                print (
                    'Computing activations for neuron {0:s} in layer "{1:s}", '
                    'SPC date "{2:s}"...'
                ).format(str(neuron_index_matrix[j, :]), layer_name,
                         this_spc_date_string)

                these_activations = (
                    model_activation.get_neuron_activation_for_examples(
                        model_object=model_object, layer_name=layer_name,
                        neuron_indices=neuron_index_matrix[j, :],
                        list_of_input_matrices=this_list_of_input_matrices))

                these_activations = numpy.reshape(
                    these_activations, (len(these_activations), 1))
                if this_activation_matrix is None:
                    this_activation_matrix = these_activations + 0.
                else:
                    this_activation_matrix = numpy.concatenate(
                        (this_activation_matrix, these_activations), axis=1)

        else:
            this_activation_matrix = None

            for this_channel_index in channel_indices:
                print (
                    'Computing activations for channel {0:d} in layer "{1:s}", '
                    'SPC date "{2:s}"...'
                ).format(this_channel_index, layer_name, this_spc_date_string)

                these_activations = (
                    model_activation.get_channel_activation_for_examples(
                        model_object=model_object, layer_name=layer_name,
                        channel_index=this_channel_index,
                        list_of_input_matrices=this_list_of_input_matrices,
                        stat_function_for_neuron_activations=K.max))

                these_activations = numpy.reshape(
                    these_activations, (len(these_activations), 1))
                if this_activation_matrix is None:
                    this_activation_matrix = these_activations + 0.
                else:
                    this_activation_matrix = numpy.concatenate(
                        (this_activation_matrix, these_activations), axis=1)

        if activation_matrix is None:
            activation_matrix = this_activation_matrix + 0.
        else:
            activation_matrix = numpy.concatenate(
                (activation_matrix, this_activation_matrix), axis=0)


        if i == num_spc_dates - 1:
            print SEPARATOR_STRING
        else:
            print MINOR_SEPARATOR_STRING

    print 'Writing activations to file: "{0:s}"...'.format(output_file_name)
    model_activation.write_file(
        pickle_file_name=output_file_name, activation_matrix=activation_matrix,
        storm_ids=storm_ids, storm_times_unix_sec=storm_times_unix_sec,
        model_file_name=model_file_name,
        component_type_string=component_type_string, target_class=target_class,
        return_probs=return_probs, layer_name=layer_name,
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
        layer_name=getattr(INPUT_ARG_OBJECT, LAYER_NAME_ARG_NAME),
        neuron_indices_flattened=numpy.array(
            getattr(INPUT_ARG_OBJECT, NEURON_INDICES_ARG_NAME), dtype=int),
        channel_indices=numpy.array(
            getattr(INPUT_ARG_OBJECT, CHANNEL_INDICES_ARG_NAME), dtype=int),
        top_radar_image_dir_name=getattr(
            INPUT_ARG_OBJECT, RADAR_IMAGE_DIR_ARG_NAME),
        top_sounding_dir_name=getattr(INPUT_ARG_OBJECT, SOUNDING_DIR_ARG_NAME),
        first_spc_date_string=getattr(
            INPUT_ARG_OBJECT, FIRST_SPC_DATE_ARG_NAME),
        last_spc_date_string=getattr(INPUT_ARG_OBJECT, LAST_SPC_DATE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME))
