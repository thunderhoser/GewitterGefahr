"""Helper methods for computing activation.

--- NOTATION ---

The following letters are used throughout this module.

T = number of input tensors to the model
E = number of examples (storm objects)
"""

import pickle
import numpy
from keras import backend as K
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import model_interpretation

STORM_IDS_KEY = 'storm_ids'
STORM_TIMES_KEY = 'storm_times_unix_sec'
MODEL_FILE_NAME_KEY = 'model_file_name'
COMPONENT_TYPE_KEY = 'component_type_string'
TARGET_CLASS_KEY = 'target_class'
LAYER_NAME_KEY = 'layer_name'
NEURON_INDICES_KEY = 'neuron_index_matrix'
CHANNEL_INDICES_KEY = 'channel_indices'

HIT_INDICES_KEY = 'hit_indices'
MISS_INDICES_KEY = 'miss_indices'
FALSE_ALARM_INDICES_KEY = 'false_alarm_indices'
CORRECT_NULL_INDICES_KEY = 'correct_null_indices'


def check_metadata(
        component_type_string, target_class=None, layer_name=None,
        neuron_index_matrix=None, channel_indices=None):
    """Error-checks metadata for activation calculations.

    C = number of model components (classes, neurons, or channels) for which
        activations were computed

    :param component_type_string: Component type (must be accepted by
        `model_interpretation.check_component_type`).
    :param target_class: See doc for `get_class_activation_for_examples`.
    :param layer_name: See doc for `get_neuron_activation_for_examples` or
        `get_channel_activation_for_examples`.
    :param neuron_index_matrix: [used only if component_type_string = "neuron"]
        C-by-? numpy array, where neuron_index_matrix[j, :] contains array
        indices of the [j]th neuron whose activation was computed.
    :param channel_indices: [used only if component_type_string = "channel"]
        length-C numpy array, where channel_indices[j] is the index of the
        [j]th channel whose activation was computed.
    :return: num_components: Number of model components (classes, neurons, or
        channels) whose activation was computed.
    """

    model_interpretation.check_component_type(component_type_string)
    if (component_type_string ==
            model_interpretation.CLASS_COMPONENT_TYPE_STRING):
        error_checking.assert_is_integer(target_class)
        error_checking.assert_is_geq(target_class, 0)
        num_components = 1

    if component_type_string in [
            model_interpretation.NEURON_COMPONENT_TYPE_STRING,
            model_interpretation.CHANNEL_COMPONENT_TYPE_STRING
    ]:
        error_checking.assert_is_string(layer_name)

    if (component_type_string ==
            model_interpretation.NEURON_COMPONENT_TYPE_STRING):
        error_checking.assert_is_integer_numpy_array(neuron_index_matrix)
        error_checking.assert_is_geq_numpy_array(neuron_index_matrix, 0)
        error_checking.assert_is_numpy_array(
            neuron_index_matrix, num_dimensions=2)
        num_components = neuron_index_matrix.shape[0]

    if (component_type_string ==
            model_interpretation.CHANNEL_COMPONENT_TYPE_STRING):
        error_checking.assert_is_integer_numpy_array(channel_indices)
        error_checking.assert_is_geq_numpy_array(channel_indices, 0)
        num_components = len(channel_indices)

    return num_components


def get_class_activation_for_examples(
        model_object, target_class, list_of_input_matrices):
    """For each input example, returns predicted probability of target class.

    :param model_object: Instance of `keras.models.Model`.
    :param target_class: Predictions will be returned for this class.  Must be
        an integer in 0...(K - 1), where K = number of classes.
    :param list_of_input_matrices: length-T list of numpy arrays, comprising
        one or more examples (storm objects).  list_of_input_matrices[i] must
        have the same dimensions as the [i]th input tensor to the model.
    :return: activation_values: length-E numpy array, where activation_values[i]
        is the activation (predicted probability or logit) of the target class
        for the [i]th example.
    """

    check_metadata(
        component_type_string=model_interpretation.CLASS_COMPONENT_TYPE_STRING,
        target_class=target_class)

    if isinstance(model_object.input, list):
        list_of_input_tensors = model_object.input
    else:
        list_of_input_tensors = [model_object.input]

    num_output_neurons = model_object.layers[-1].output.get_shape().as_list()[
        -1]

    if num_output_neurons == 1:
        error_checking.assert_is_leq(target_class, 1)
        if target_class == 1:
            output_tensor = model_object.layers[-1].output[..., 0]
        else:
            output_tensor = 1. - model_object.layers[-1].output[..., 0]
    else:
        error_checking.assert_is_less_than(target_class, num_output_neurons)
        output_tensor = model_object.layers[-1].output[..., target_class]

    activation_function = K.function(
        list_of_input_tensors + [K.learning_phase()],
        [output_tensor])

    return activation_function(list_of_input_matrices + [0])[0]


def get_neuron_activation_for_examples(
        model_object, layer_name, neuron_indices, list_of_input_matrices):
    """For each input example, returns the activation of one neuron.

    :param model_object: Instance of `keras.models.Model`.
    :param layer_name: Name of layer containing the relevant neuron.
    :param neuron_indices: 1-D numpy array with indices of the relevant neuron.
        Must have length K - 1, where K = number of dimensions in layer output.
        The first dimension of the layer output is the example dimension, for
        which all indices from 0...(E - 1) are used.
    :param list_of_input_matrices: See doc for
        `get_class_activation_for_examples`.
    :return: activation_values: length-E numpy array, where activation_values[i]
        is the activation of the given neuron by the [i]th example.
    """

    check_metadata(
        component_type_string=model_interpretation.NEURON_COMPONENT_TYPE_STRING,
        layer_name=layer_name,
        neuron_index_matrix=numpy.expand_dims(neuron_indices, axis=0))

    if isinstance(model_object.input, list):
        list_of_input_tensors = model_object.input
    else:
        list_of_input_tensors = [model_object.input]

    activation_function = K.function(
        list_of_input_tensors + [K.learning_phase()],
        [model_object.get_layer(name=layer_name).output[..., neuron_indices]])

    return activation_function(list_of_input_matrices + [0])[0]


def get_channel_activation_for_examples(
        model_object, layer_name, channel_index, list_of_input_matrices,
        stat_function_for_neuron_activations):
    """For each input example, returns the activation of one channel.

    :param model_object: Instance of `keras.models.Model`.
    :param layer_name: Name of layer containing the relevant channel.
    :param channel_index: Index of the relevant channel.  This method computes
        activations for the [j]th output channel of `layer_name`, where
        j = `channel_index`.
    :param list_of_input_matrices: See doc for
        `get_class_activation_for_examples`.
    :param stat_function_for_neuron_activations: Function used to process neuron
        activations (needed because a channel generally has many neurons).  This
        function must take a Keras tensor (containing neuron activations) and
        return a single number.  Some examples are `keras.backend.max` and
        `keras.backend.mean`.
    :return: activation_values: length-E numpy array, where activation_values[i]
        is stat_function_for_neuron_activations(channel_activations) for the
        [i]th example.
    """

    check_metadata(
        component_type_string=
        model_interpretation.CHANNEL_COMPONENT_TYPE_STRING,
        layer_name=layer_name, channel_indices=numpy.array([channel_index]))

    if isinstance(model_object.input, list):
        list_of_input_tensors = model_object.input
    else:
        list_of_input_tensors = [model_object.input]

    activation_function = K.function(
        list_of_input_tensors + [K.learning_phase()],
        [stat_function_for_neuron_activations(
            model_object.get_layer(name=layer_name).output[..., channel_index])
        ]
    )

    return activation_function(list_of_input_matrices + [0])[0]


def get_hilo_activation_examples(
        storm_activations, num_high_activation_examples,
        num_low_activation_examples):
    """Finds examples (storm objects) with highest and lowest activations.

    E = number of examples

    :param storm_activations: length-E numpy array of model activations.
    :param num_high_activation_examples: Number of high-activation examples to
        return.
    :param num_low_activation_examples: Number of low-activation examples to
        return.
    :return: low_indices: 1-D numpy array with indices of low-activation
        examples.
    :return: high_indices: 1-D numpy array with indices of high-activation
        examples.
    """

    error_checking.assert_is_numpy_array(storm_activations, num_dimensions=1)
    error_checking.assert_is_integer(num_high_activation_examples)
    error_checking.assert_is_geq(num_high_activation_examples, 0)
    error_checking.assert_is_integer(num_low_activation_examples)
    error_checking.assert_is_geq(num_low_activation_examples, 0)
    error_checking.assert_is_greater(
        num_high_activation_examples + num_low_activation_examples, 0)

    num_low_activation_examples = min(
        [num_low_activation_examples, len(storm_activations)])
    num_high_activation_examples = min(
        [num_high_activation_examples, len(storm_activations)])

    sort_indices = numpy.argsort(storm_activations)
    if num_high_activation_examples > 0:
        high_indices = sort_indices[::-1][:num_high_activation_examples]
    else:
        high_indices = numpy.array([], dtype=int)

    if num_low_activation_examples > 0:
        low_indices = sort_indices[:num_low_activation_examples]
    else:
        low_indices = numpy.array([], dtype=int)

    return high_indices, low_indices


def get_contingency_table_extremes(
        storm_activations, storm_target_values, num_hits, num_misses,
        num_false_alarms, num_correct_nulls):
    """Returns "contingency-table extremes".

    Specifically, this method returns the following:

    - best hits (positive examples with the highest activations)
    - worst misses (positive examples with the lowest activations)
    - worst false alarms (negative examples with the highest activations)
    - best correct nulls (negative examples with the lowest activations)

    DEFINITIONS

    One "example" is one storm object.
    A "negative example" is a storm object with target = 0.
    A "positive example" is a storm object with target = 1.
    The target variable must be binary.

    E = number of examples

    :param storm_activations: length-E numpy array of model activations.
    :param storm_target_values: length-E numpy array of target values.  These
        must be integers from 0...1.
    :param num_hits: Number of best hits.
    :param num_misses: Number of worst misses.
    :param num_false_alarms: Number of worst false alarms.
    :param num_correct_nulls: Number of best correct nulls.
    :return: ct_extreme_dict: Dictionary with the following keys.
    ct_extreme_dict['hit_indices']: 1-D numpy array with indices of best hits.
    ct_extreme_dict['miss_indices']: 1-D numpy array with indices of worst
        misses.
    ct_extreme_dict['false_alarm_indices']: 1-D numpy array with indices of
        worst false alarms.
    ct_extreme_dict['correct_null_indices']: 1-D numpy array with indices of
        best correct nulls.
    """

    error_checking.assert_is_numpy_array(storm_activations, num_dimensions=1)
    error_checking.assert_is_integer_numpy_array(storm_target_values)
    error_checking.assert_is_geq_numpy_array(storm_target_values, 0)
    error_checking.assert_is_leq_numpy_array(storm_target_values, 1)

    num_storm_objects = len(storm_activations)
    error_checking.assert_is_numpy_array(
        storm_target_values, exact_dimensions=numpy.array([num_storm_objects]))

    error_checking.assert_is_integer(num_hits)
    error_checking.assert_is_geq(num_hits, 0)
    error_checking.assert_is_integer(num_misses)
    error_checking.assert_is_geq(num_misses, 0)
    error_checking.assert_is_integer(num_false_alarms)
    error_checking.assert_is_geq(num_false_alarms, 0)
    error_checking.assert_is_integer(num_correct_nulls)
    error_checking.assert_is_geq(num_correct_nulls, 0)
    error_checking.assert_is_greater(
        num_hits + num_misses + num_false_alarms + num_correct_nulls, 0)

    positive_indices = numpy.where(storm_target_values == 1)[0]
    num_hits = min([num_hits, len(positive_indices)])
    num_misses = min([num_misses, len(positive_indices)])

    if num_hits > 0:
        these_indices = numpy.argsort(storm_activations[positive_indices])[::-1]
        hit_indices = positive_indices[these_indices][:num_hits]
    else:
        hit_indices = numpy.array([], dtype=int)

    if num_misses > 0:
        these_indices = numpy.argsort(storm_activations[positive_indices])
        miss_indices = positive_indices[these_indices][:num_misses]
    else:
        miss_indices = numpy.array([], dtype=int)

    negative_indices = numpy.where(storm_target_values == 0)[0]
    num_false_alarms = min([num_false_alarms, len(negative_indices)])
    num_correct_nulls = min([num_correct_nulls, len(negative_indices)])

    if num_false_alarms > 0:
        these_indices = numpy.argsort(storm_activations[negative_indices])[::-1]
        false_alarm_indices = negative_indices[these_indices][:num_false_alarms]
    else:
        false_alarm_indices = numpy.array([], dtype=int)

    if num_correct_nulls > 0:
        these_indices = numpy.argsort(storm_activations[negative_indices])
        correct_null_indices = negative_indices[
            these_indices][:num_correct_nulls]
    else:
        correct_null_indices = numpy.array([], dtype=int)

    return {
        HIT_INDICES_KEY: hit_indices,
        MISS_INDICES_KEY: miss_indices,
        FALSE_ALARM_INDICES_KEY: false_alarm_indices,
        CORRECT_NULL_INDICES_KEY: correct_null_indices
    }


def write_file(
        pickle_file_name, activation_matrix, storm_ids, storm_times_unix_sec,
        model_file_name, component_type_string, target_class=None,
        layer_name=None, neuron_index_matrix=None, channel_indices=None):
    """Writes activations to Pickle file.

    E = number of examples (storm objects)
    C = number of model components (classes, neurons, or channels) for which
        activations were computed

    :param pickle_file_name: Path to output file.
    :param activation_matrix: E-by-C numpy array of activations, where
        activation_matrix[i, j] = activation of the [j]th model component for
        the [i]th example.
    :param storm_ids: length-E list of storm IDs.
    :param storm_times_unix_sec: length-E numpy array of storm times.
    :param model_file_name: Path to file with trained model.
    :param component_type_string: See doc for `check_metadata`.
    :param target_class: Same.
    :param layer_name: Same.
    :param neuron_index_matrix: Same.
    :param channel_indices: Same.
    """

    num_components = check_metadata(
        component_type_string=component_type_string, target_class=target_class,
        layer_name=layer_name, neuron_index_matrix=neuron_index_matrix,
        channel_indices=channel_indices)
    error_checking.assert_is_string(model_file_name)

    error_checking.assert_is_string_list(storm_ids)
    error_checking.assert_is_numpy_array(
        numpy.array(storm_ids), num_dimensions=1)
    num_examples = len(storm_ids)

    error_checking.assert_is_integer_numpy_array(storm_times_unix_sec)
    error_checking.assert_is_numpy_array(
        storm_times_unix_sec, exact_dimensions=numpy.array([num_examples]))

    error_checking.assert_is_numpy_array_without_nan(activation_matrix)
    error_checking.assert_is_numpy_array(
        activation_matrix,
        exact_dimensions=numpy.array([num_examples, num_components]))

    metadata_dict = {
        STORM_IDS_KEY: storm_ids,
        STORM_TIMES_KEY: storm_times_unix_sec,
        MODEL_FILE_NAME_KEY: model_file_name,
        COMPONENT_TYPE_KEY: component_type_string,
        TARGET_CLASS_KEY: target_class,
        LAYER_NAME_KEY: layer_name,
        NEURON_INDICES_KEY: neuron_index_matrix,
        CHANNEL_INDICES_KEY: channel_indices,
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(activation_matrix, pickle_file_handle)
    pickle.dump(metadata_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_file(pickle_file_name):
    """Reads activations from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: activation_matrix: See doc for `write_file`.
    :return: metadata_dict: Dictionary with the following keys.
    metadata_dict['storm_ids']: See doc for `write_file`.
    metadata_dict['storm_times_unix_sec']: Same.
    metadata_dict['model_file_name']: Same.
    metadata_dict['component_type_string']: Same.
    metadata_dict['target_class']: Same.
    metadata_dict['layer_name']: Same.
    metadata_dict['neuron_index_matrix']: Same.
    metadata_dict['channel_indices']: Same.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    activation_matrix = pickle.load(pickle_file_handle)
    metadata_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    return activation_matrix, metadata_dict
