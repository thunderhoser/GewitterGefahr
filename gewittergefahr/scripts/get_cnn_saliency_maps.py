"""Computes saliency map for each storm object and each CNN component.

CNN = convolutional neural network
"""

import pickle
import os.path
import argparse
import numpy
from keras import backend as K
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import input_examples
from gewittergefahr.deep_learning import testing_io
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.deep_learning import model_interpretation
from gewittergefahr.deep_learning import saliency_maps

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)))

LARGE_INTEGER = int(1e10)
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

STORM_IDS_KEY = 'storm_ids'
STORM_TIMES_KEY = 'storm_times_unix_sec'

CLASS_COMPONENT_TYPE_STRING = model_interpretation.CLASS_COMPONENT_TYPE_STRING
NEURON_COMPONENT_TYPE_STRING = model_interpretation.NEURON_COMPONENT_TYPE_STRING
CHANNEL_COMPONENT_TYPE_STRING = (
    model_interpretation.CHANNEL_COMPONENT_TYPE_STRING)

MODEL_FILE_ARG_NAME = 'model_file_name'
COMPONENT_TYPE_ARG_NAME = 'component_type_string'
TARGET_CLASS_ARG_NAME = 'target_class'
LAYER_NAME_ARG_NAME = 'layer_name'
IDEAL_ACTIVATION_ARG_NAME = 'ideal_activation'
NEURON_INDICES_ARG_NAME = 'neuron_indices'
CHANNEL_INDEX_ARG_NAME = 'channel_index'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
STORM_DICT_FILE_ARG_NAME = 'input_storm_dict_file_name'
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
).format(COMPONENT_TYPE_ARG_NAME, CLASS_COMPONENT_TYPE_STRING,
         TARGET_CLASS_ARG_NAME)

LAYER_NAME_HELP_STRING = (
    '[used only if {0:s} = "{1:s}" or "{2:s}"] Name of layer with neurons or '
    'channels for which saliency maps will be computed.'
).format(COMPONENT_TYPE_ARG_NAME, NEURON_COMPONENT_TYPE_STRING,
         CLASS_COMPONENT_TYPE_STRING)

IDEAL_ACTIVATION_HELP_STRING = (
    '[used only if {0:s} = "{1:s}" or "{2:s}"] The loss function will be '
    '(neuron_activation - ideal_activation)**2 or [max(channel_activations) - '
    'ideal_activation]**2.  If {3:s} = -1, the loss function will be '
    '-sign(neuron_activation) * neuron_activation**2 or '
    '-sign(max(channel_activations)) * max(channel_activations)**2.'
).format(COMPONENT_TYPE_ARG_NAME, NEURON_COMPONENT_TYPE_STRING,
         CHANNEL_COMPONENT_TYPE_STRING, IDEAL_ACTIVATION_ARG_NAME)

NEURON_INDICES_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] Indices of neuron whose saliency map is to '
    'be computed.  For example, to compute saliency maps for neuron (0, 0, 2), '
    'this argument should be "0 0 2".'
).format(COMPONENT_TYPE_ARG_NAME, NEURON_COMPONENT_TYPE_STRING)

CHANNEL_INDEX_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] Index of channel whose saliency map is to '
    'be computed.'
).format(COMPONENT_TYPE_ARG_NAME, CHANNEL_COMPONENT_TYPE_STRING)

EXAMPLE_DIR_HELP_STRING = (
    'Name of top-level directory with input examples.  Files therein will be '
    'found by `input_examples.find_example_file` and read by '
    '`input_examples.read_example_file`.')

STORM_DICT_FILE_HELP_STRING = (
    'Path to Pickle file with "storm dictionary".  This file should contain '
    'only one dictionary, containing at least the keys "{0:s}" and "{1:s}".'
).format(STORM_IDS_KEY, STORM_TIMES_KEY)

OUTPUT_FILE_HELP_STRING = (
    'Path to output file (will be written by `saliency_maps.write_file`).')

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
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + STORM_DICT_FILE_ARG_NAME, type=str, required=True,
    help=STORM_DICT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _read_storm_metadata(pickle_file_name):
    """Reads storm metadata (IDs and times) from Pickle file.
    
    N = number of storm objects
    
    :param pickle_file_name: Path to input file.
    :return: storm_ids: length-N list of storm IDs (strings).
    :return: storm_times_unix_sec: length-N numpy array of valid times.
    :raises: ValueError: if dictionary cannot be found in Pickle file.
    """

    print 'Reading storm metadata from: "{0:s}"...'.format(pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'rb')
    while True:
        storm_metadata_dict = pickle.load(pickle_file_handle)
        if isinstance(storm_metadata_dict, dict):
            break

    pickle_file_handle.close()
    if not isinstance(storm_metadata_dict, dict):
        raise ValueError('Cannot find dictionary in file.')

    return (storm_metadata_dict[STORM_IDS_KEY],
            storm_metadata_dict[STORM_TIMES_KEY])


def _run(
        model_file_name, component_type_string, target_class, layer_name,
        ideal_activation, neuron_indices, channel_index, top_example_dir_name,
        input_storm_dict_file_name, output_file_name):
    """Computes saliency map for each storm object and each model component.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param component_type_string: Same.
    :param target_class: Same.
    :param layer_name: Same.
    :param ideal_activation: Same.
    :param neuron_indices: Same.
    :param channel_index: Same.
    :param top_example_dir_name: Same.
    :param input_storm_dict_file_name: Same.
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
    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]

    desired_storm_ids, desired_storm_times_unix_sec = _read_storm_metadata(
        input_storm_dict_file_name)
    print desired_storm_ids

    # Create saliency map for each storm object.
    desired_spc_dates_unix_sec = numpy.array([
        time_conversion.time_to_spc_date_unix_sec(t)
        for t in desired_storm_times_unix_sec
    ], dtype=int)

    unique_spc_dates_unix_sec = numpy.unique(desired_spc_dates_unix_sec)

    list_of_input_matrices = None
    list_of_saliency_matrices = None
    storm_ids = []
    storm_times_unix_sec = numpy.array([], dtype=int)
    sounding_pressure_matrix_pascals = None

    print SEPARATOR_STRING

    for this_spc_date_unix_sec in unique_spc_dates_unix_sec:
        this_spc_date_string = time_conversion.time_to_spc_date_string(
            this_spc_date_unix_sec)

        this_example_file_name = input_examples.find_example_file(
            top_directory_name=top_example_dir_name, shuffled=False,
            spc_date_string=this_spc_date_string)

        training_option_dict[trainval_io.SAMPLING_FRACTIONS_KEY] = None
        training_option_dict[
            trainval_io.EXAMPLE_FILES_KEY] = [this_example_file_name]
        training_option_dict[trainval_io.FIRST_STORM_TIME_KEY] = (
            time_conversion.get_start_of_spc_date(this_spc_date_string))
        training_option_dict[trainval_io.LAST_STORM_TIME_KEY] = (
            time_conversion.get_end_of_spc_date(this_spc_date_string))

        if model_metadata_dict[cnn.USE_2D3D_CONVOLUTION_KEY]:
            this_generator = testing_io.example_generator_2d3d_myrorss(
                option_dict=training_option_dict,
                num_examples_total=LARGE_INTEGER)
        else:
            this_generator = testing_io.example_generator_2d_or_3d(
                option_dict=training_option_dict,
                num_examples_total=LARGE_INTEGER)

        this_storm_object_dict = next(this_generator)

        this_start_time_unix_sec = time_conversion.get_start_of_spc_date(
            this_spc_date_string)
        this_end_time_unix_sec = time_conversion.get_end_of_spc_date(
            this_spc_date_string)
        these_indices = numpy.where(numpy.logical_and(
            storm_times_unix_sec >= this_start_time_unix_sec,
            storm_times_unix_sec <= this_end_time_unix_sec
        ))[0]

        these_indices = tracking_utils.find_storm_objects(
            all_storm_ids=this_storm_object_dict[testing_io.STORM_IDS_KEY],
            all_times_unix_sec=this_storm_object_dict[
                testing_io.STORM_TIMES_KEY],
            storm_ids_to_keep=[desired_storm_ids[k] for k in these_indices],
            times_to_keep_unix_sec=desired_storm_times_unix_sec[these_indices],
            allow_missing=False)

        these_storm_ids = [
            this_storm_object_dict[testing_io.STORM_IDS_KEY][k]
            for k in these_indices
        ]
        these_storm_times_unix_sec = this_storm_object_dict[
            testing_io.STORM_TIMES_KEY][these_indices]
        these_input_matrices = [
            a[these_indices, ...]
            for a in this_storm_object_dict[testing_io.INPUT_MATRICES_KEY]
        ]

        this_pressure_matrix_pascals = this_storm_object_dict[
            model_interpretation.SOUNDING_PRESSURES_KEY]
        if this_pressure_matrix_pascals is not None:
            this_pressure_matrix_pascals = this_pressure_matrix_pascals[
                these_indices, ...]

        if component_type_string == CLASS_COMPONENT_TYPE_STRING:
            print 'Computing saliency maps for target class {0:d}...'.format(
                target_class)

            these_saliency_matrices = (
                saliency_maps.get_saliency_maps_for_class_activation(
                    model_object=model_object, target_class=target_class,
                    list_of_input_matrices=these_input_matrices)
            )

        elif component_type_string == NEURON_COMPONENT_TYPE_STRING:
            print (
                'Computing saliency maps for neuron {0:s} in layer "{1:s}"...'
            ).format(str(neuron_indices), layer_name)

            these_saliency_matrices = (
                saliency_maps.get_saliency_maps_for_neuron_activation(
                    model_object=model_object, layer_name=layer_name,
                    neuron_indices=neuron_indices,
                    list_of_input_matrices=these_input_matrices,
                    ideal_activation=ideal_activation)
            )

        else:
            print (
                'Computing saliency maps for channel {0:d} in layer "{1:s}"...'
            ).format(channel_index, layer_name)

            these_saliency_matrices = (
                saliency_maps.get_saliency_maps_for_channel_activation(
                    model_object=model_object, layer_name=layer_name,
                    channel_index=channel_index,
                    list_of_input_matrices=these_input_matrices,
                    stat_function_for_neuron_activations=K.max,
                    ideal_activation=ideal_activation)
            )

        if list_of_input_matrices is None:
            storm_ids = these_storm_ids + []
            storm_times_unix_sec = these_storm_times_unix_sec + 0
            list_of_input_matrices = these_input_matrices + []
            list_of_saliency_matrices = these_saliency_matrices + []

            if this_pressure_matrix_pascals is not None:
                sounding_pressure_matrix_pascals = (
                    this_pressure_matrix_pascals + 0.)

        else:
            storm_ids += these_storm_ids
            storm_times_unix_sec = numpy.concatenate((
                storm_times_unix_sec, these_storm_times_unix_sec))

            for k in range(len(list_of_input_matrices)):
                list_of_input_matrices[k] = numpy.concatenate(
                    (list_of_input_matrices[k], these_input_matrices[k]),
                    axis=0)

                list_of_saliency_matrices[k] = numpy.concatenate(
                    (list_of_saliency_matrices[k], these_saliency_matrices[k]),
                    axis=0)

            if this_pressure_matrix_pascals is not None:
                sounding_pressure_matrix_pascals = numpy.concatenate(
                    (sounding_pressure_matrix_pascals,
                     this_pressure_matrix_pascals), axis=0)

        print SEPARATOR_STRING

    print 'Denormalizing model inputs...'
    list_of_input_matrices = model_interpretation.denormalize_data(
        list_of_input_matrices=list_of_input_matrices,
        model_metadata_dict=model_metadata_dict)

    print 'Writing saliency maps to file: "{0:s}"...'.format(output_file_name)
    saliency_maps.write_file(
        pickle_file_name=output_file_name,
        list_of_input_matrices=list_of_input_matrices,
        list_of_saliency_matrices=list_of_saliency_matrices,
        model_file_name=model_file_name, storm_ids=storm_ids,
        storm_times_unix_sec=storm_times_unix_sec,
        component_type_string=component_type_string, target_class=target_class,
        layer_name=layer_name, ideal_activation=ideal_activation,
        neuron_indices=neuron_indices, channel_index=channel_index,
        sounding_pressure_matrix_pascals=sounding_pressure_matrix_pascals)


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
        top_example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        input_storm_dict_file_name=getattr(
            INPUT_ARG_OBJECT, STORM_DICT_FILE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
