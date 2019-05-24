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
from gewittergefahr.deep_learning import input_examples
from gewittergefahr.deep_learning import testing_io
from gewittergefahr.deep_learning import model_interpretation
from gewittergefahr.deep_learning import model_activation
from gewittergefahr.deep_learning import training_validation_io as trainval_io

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)))

LARGE_INTEGER = int(1e10)
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

CLASS_COMPONENT_TYPE_STRING = model_interpretation.CLASS_COMPONENT_TYPE_STRING
NEURON_COMPONENT_TYPE_STRING = model_interpretation.NEURON_COMPONENT_TYPE_STRING
CHANNEL_COMPONENT_TYPE_STRING = (
    model_interpretation.CHANNEL_COMPONENT_TYPE_STRING)

MODEL_FILE_ARG_NAME = 'model_file_name'
COMPONENT_TYPE_ARG_NAME = 'component_type_string'
TARGET_CLASS_ARG_NAME = 'target_class'
LAYER_NAME_ARG_NAME = 'layer_name'
NEURON_INDICES_ARG_NAME = 'neuron_indices'
CHANNEL_INDICES_ARG_NAME = 'channel_indices'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
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
).format(COMPONENT_TYPE_ARG_NAME, CLASS_COMPONENT_TYPE_STRING,
         TARGET_CLASS_ARG_NAME)

LAYER_NAME_HELP_STRING = (
    '[used only if {0:s} = "{1:s}" or "{2:s}"] Name of layer with neurons or '
    'channels whose activations will be computed.'
).format(COMPONENT_TYPE_ARG_NAME, NEURON_COMPONENT_TYPE_STRING,
         CLASS_COMPONENT_TYPE_STRING)

NEURON_INDICES_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] Indices for each neuron whose activation is'
    ' to be computed.  For example, to compute activations for neuron (0, 0, 2)'
    ', this argument should be "0 0 2".  To compute activations for neurons '
    '(0, 0, 2) and (1, 1, 2), this list should be "0 0 2 -1 1 1 2".  In other '
    'words, use -1 to separate neurons.'
).format(COMPONENT_TYPE_ARG_NAME, NEURON_COMPONENT_TYPE_STRING)

CHANNEL_INDICES_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] Index for each channel whose activation is '
    'to be computed.'
).format(COMPONENT_TYPE_ARG_NAME, CHANNEL_COMPONENT_TYPE_STRING)

EXAMPLE_DIR_HELP_STRING = (
    'Name of top-level directory with input examples.  Files therein will be '
    'found by `input_examples.find_example_file` and read by '
    '`input_examples.read_example_file`.')

SPC_DATE_HELP_STRING = (
    'SPC date (format "yyyymmdd").  For each model component, activation will '
    'be computed for each example (storm object) from `{0:s}`...`{1:s}`.'
).format(FIRST_SPC_DATE_ARG_NAME, LAST_SPC_DATE_ARG_NAME)

OUTPUT_FILE_HELP_STRING = (
    'Path to output file (will be written by `model_activation.write_file`).')

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
    '--' + NEURON_INDICES_ARG_NAME, type=int, nargs='+', required=False,
    default=[-1], help=NEURON_INDICES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CHANNEL_INDICES_ARG_NAME, type=int, nargs='+', required=False,
    default=[-1], help=CHANNEL_INDICES_HELP_STRING)

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
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _run(
        model_file_name, component_type_string, target_class, layer_name,
        neuron_indices_flattened, channel_indices, top_example_dir_name,
        first_spc_date_string, last_spc_date_string, output_file_name):
    """Creates activation maps for one class, neuron, or channel of a CNN.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param component_type_string: Same.
    :param target_class: Same.
    :param layer_name: Same.
    :param neuron_indices_flattened: Same.
    :param channel_indices: Same.
    :param top_example_dir_name: Same.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
    :param output_file_name: Same.
    """

    # Check input args.
    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)
    model_interpretation.check_component_type(component_type_string)

    if component_type_string == CHANNEL_COMPONENT_TYPE_STRING:
        error_checking.assert_is_geq_numpy_array(channel_indices, 0)
    if component_type_string == NEURON_COMPONENT_TYPE_STRING:
        neuron_indices_flattened = neuron_indices_flattened.astype(float)
        neuron_indices_flattened[neuron_indices_flattened < 0] = numpy.nan

        neuron_indices_2d_list = general_utils.split_array_by_nan(
            neuron_indices_flattened)
        neuron_index_matrix = numpy.array(neuron_indices_2d_list, dtype=int)
    else:
        neuron_index_matrix = None

    # Read model and metadata.
    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = cnn.read_model(model_file_name)

    metadata_file_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0]
    )

    print('Reading metadata from: "{0:s}"...'.format(metadata_file_name))
    model_metadata_dict = cnn.read_model_metadata(metadata_file_name)
    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]

    # Create generator.
    example_file_names = input_examples.find_many_example_files(
        top_directory_name=top_example_dir_name, shuffled=False,
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string,
        raise_error_if_any_missing=False)

    training_option_dict[trainval_io.SAMPLING_FRACTIONS_KEY] = None
    training_option_dict[trainval_io.EXAMPLE_FILES_KEY] = example_file_names
    training_option_dict[trainval_io.FIRST_STORM_TIME_KEY] = (
        time_conversion.get_start_of_spc_date(first_spc_date_string))
    training_option_dict[trainval_io.LAST_STORM_TIME_KEY] = (
        time_conversion.get_end_of_spc_date(last_spc_date_string))

    if model_metadata_dict[cnn.LAYER_OPERATIONS_KEY] is not None:
        generator_object = testing_io.gridrad_generator_2d_reduced(
            option_dict=training_option_dict,
            list_of_operation_dicts=model_metadata_dict[
                cnn.LAYER_OPERATIONS_KEY],
            num_examples_total=LARGE_INTEGER
        )

    elif model_metadata_dict[cnn.USE_2D3D_CONVOLUTION_KEY]:
        generator_object = testing_io.myrorss_generator_2d3d(
            option_dict=training_option_dict, num_examples_total=LARGE_INTEGER)
    else:
        generator_object = testing_io.generator_2d_or_3d(
            option_dict=training_option_dict, num_examples_total=LARGE_INTEGER)

    # Compute activation for each example (storm object) and model component.
    full_id_strings = []
    storm_times_unix_sec = numpy.array([], dtype=int)
    activation_matrix = None

    print(SEPARATOR_STRING)

    for _ in range(len(example_file_names)):
        try:
            this_storm_object_dict = next(generator_object)
        except StopIteration:
            break

        this_list_of_input_matrices = this_storm_object_dict[
            testing_io.INPUT_MATRICES_KEY]
        these_id_strings = this_storm_object_dict[testing_io.FULL_IDS_KEY]
        these_times_unix_sec = this_storm_object_dict[
            testing_io.STORM_TIMES_KEY]

        full_id_strings += these_id_strings
        storm_times_unix_sec = numpy.concatenate((
            storm_times_unix_sec, these_times_unix_sec))

        if component_type_string == CLASS_COMPONENT_TYPE_STRING:
            print('Computing activations for target class {0:d}...'.format(
                target_class))

            this_activation_matrix = (
                model_activation.get_class_activation_for_examples(
                    model_object=model_object, target_class=target_class,
                    list_of_input_matrices=this_list_of_input_matrices)
            )

            this_activation_matrix = numpy.reshape(
                this_activation_matrix, (len(this_activation_matrix), 1)
            )

        elif component_type_string == NEURON_COMPONENT_TYPE_STRING:
            this_activation_matrix = None

            for j in range(neuron_index_matrix.shape[0]):
                print((
                    'Computing activations for neuron {0:s} in layer "{1:s}"...'
                ).format(str(neuron_index_matrix[j, :]), layer_name))

                these_activations = (
                    model_activation.get_neuron_activation_for_examples(
                        model_object=model_object, layer_name=layer_name,
                        neuron_indices=neuron_index_matrix[j, :],
                        list_of_input_matrices=this_list_of_input_matrices)
                )

                these_activations = numpy.reshape(
                    these_activations, (len(these_activations), 1)
                )

                if this_activation_matrix is None:
                    this_activation_matrix = these_activations + 0.
                else:
                    this_activation_matrix = numpy.concatenate(
                        (this_activation_matrix, these_activations), axis=1)
        else:
            this_activation_matrix = None

            for this_channel_index in channel_indices:
                print((
                    'Computing activations for channel {0:d} in layer '
                    '"{1:s}"...'
                ).format(this_channel_index, layer_name))

                these_activations = (
                    model_activation.get_channel_activation_for_examples(
                        model_object=model_object, layer_name=layer_name,
                        channel_index=this_channel_index,
                        list_of_input_matrices=this_list_of_input_matrices,
                        stat_function_for_neuron_activations=K.max)
                )

                these_activations = numpy.reshape(
                    these_activations, (len(these_activations), 1)
                )

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

        print(SEPARATOR_STRING)

    print('Writing activations to file: "{0:s}"...'.format(output_file_name))
    model_activation.write_file(
        pickle_file_name=output_file_name, activation_matrix=activation_matrix,
        full_id_strings=full_id_strings,
        storm_times_unix_sec=storm_times_unix_sec,
        model_file_name=model_file_name,
        component_type_string=component_type_string, target_class=target_class,
        layer_name=layer_name, neuron_index_matrix=neuron_index_matrix,
        channel_indices=channel_indices)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        component_type_string=getattr(
            INPUT_ARG_OBJECT, COMPONENT_TYPE_ARG_NAME),
        target_class=getattr(INPUT_ARG_OBJECT, TARGET_CLASS_ARG_NAME),
        layer_name=getattr(INPUT_ARG_OBJECT, LAYER_NAME_ARG_NAME),
        neuron_indices_flattened=numpy.array(
            getattr(INPUT_ARG_OBJECT, NEURON_INDICES_ARG_NAME), dtype=int),
        channel_indices=numpy.array(
            getattr(INPUT_ARG_OBJECT, CHANNEL_INDICES_ARG_NAME), dtype=int),
        top_example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        first_spc_date_string=getattr(
            INPUT_ARG_OBJECT, FIRST_SPC_DATE_ARG_NAME),
        last_spc_date_string=getattr(INPUT_ARG_OBJECT, LAST_SPC_DATE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
