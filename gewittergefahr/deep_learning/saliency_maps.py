"""Methods for creating saliency maps."""

import pickle
import numpy
from keras import backend as K
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import model_interpretation

DEFAULT_IDEAL_LOGIT = 7.
DEFAULT_IDEAL_ACTIVATION = 2.

MODEL_FILE_NAME_KEY = 'model_file_name'
STORM_ID_KEY = 'storm_id'
STORM_TIME_KEY = 'storm_time_unix_sec'
COMPONENT_TYPE_KEY = 'component_type_string'
TARGET_CLASS_KEY = 'target_class'
RETURN_PROBS_KEY = 'return_probs'
IDEAL_LOGIT_KEY = 'ideal_logit'
LAYER_NAME_KEY = 'layer_name'
IDEAL_ACTIVATION_KEY = 'ideal_activation'
NEURON_INDICES_KEY = 'neuron_index_matrix'
CHANNEL_INDICES_KEY = 'channel_indices'


def _do_saliency_calculations(
        model_object, loss_tensor, list_of_input_matrices):
    """Does saliency calculations.

    T = number of input tensors to the model
    E = number of examples (storm objects)

    :param model_object: Instance of `keras.models.Model`.
    :param loss_tensor: Keras tensor defining the loss function.
    :param list_of_input_matrices: length-T list of numpy arrays, comprising one
        or more examples (storm objects).  list_of_input_matrices[i] must have
        the same dimensions as the [i]th input tensor to the model.
    :return: list_of_saliency_matrices: length-T list of numpy arrays,
        comprising the saliency map for each example.
        list_of_saliency_matrices[i] has the same dimensions as
        list_of_input_matrices[i] and defines the "saliency" of each value x,
        which is the gradient of the loss function with respect to x.
    """

    if isinstance(model_object.input, list):
        list_of_input_tensors = model_object.input
    else:
        list_of_input_tensors = [model_object.input]

    list_of_gradient_tensors = K.gradients(loss_tensor, list_of_input_tensors)
    num_input_tensors = len(list_of_input_tensors)
    for i in range(num_input_tensors):
        list_of_gradient_tensors[i] /= K.maximum(
            K.std(list_of_gradient_tensors[i]), K.epsilon())

    inputs_to_gradients_function = K.function(
        list_of_input_tensors + [K.learning_phase()], list_of_gradient_tensors)
    list_of_saliency_matrices = inputs_to_gradients_function(
        list_of_input_matrices + [0])
    for i in range(num_input_tensors):
        list_of_saliency_matrices[i] *= -1

    return list_of_saliency_matrices


def check_metadata(
        component_type_string, target_class=None, return_probs=None,
        ideal_logit=None, layer_name=None, ideal_activation=None,
        neuron_index_matrix=None, channel_indices=None):
    """Error-checks metadata for saliency calculations.

    C = number of model components (classes, neurons, or channels) for which
        saliency maps were computed

    :param component_type_string: Component type (must be accepted by
        `model_interpretation.check_component_type`).
    :param target_class: See doc for `get_saliency_maps_for_class_activation`.
    :param return_probs: Same.
    :param ideal_logit: Same.
    :param layer_name: See doc for `get_saliency_maps_for_neuron_activation` or
        `get_saliency_maps_for_channel_activation`.
    :param ideal_activation: Same.
    :param neuron_index_matrix: [used only if component_type_string = "neuron"]
        C-by-? numpy array, where neuron_index_matrix[j, :] contains array
        indices of the [j]th neuron whose saliency map was computed.
    :param channel_indices: [used only if component_type_string = "channel"]
        length-C numpy array, where channel_indices[j] is the index of the
        [j]th channel whose saliency map was computed.
    :return: num_components: Number of model components (classes, neurons, or
        channels) whose saliency map was computed.
    """

    model_interpretation.check_component_type(component_type_string)
    if (component_type_string ==
            model_interpretation.CLASS_COMPONENT_TYPE_STRING):
        error_checking.assert_is_integer(target_class)
        error_checking.assert_is_geq(target_class, 0)
        error_checking.assert_is_boolean(return_probs)
        num_components = 1

        if return_probs:
            ideal_logit = None
        if ideal_logit is not None:
            error_checking.assert_is_greater(ideal_logit, 0.)

    if component_type_string in [
            model_interpretation.NEURON_COMPONENT_TYPE_STRING,
            model_interpretation.CHANNEL_COMPONENT_TYPE_STRING
    ]:
        error_checking.assert_is_string(layer_name)
        if ideal_activation is not None:
            error_checking.assert_is_greater(ideal_activation, 0.)

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


def get_saliency_maps_for_class_activation(
        model_object, target_class, return_probs, list_of_input_matrices,
        ideal_logit=DEFAULT_IDEAL_LOGIT):
    """For each input example, creates saliency map for target class.

    If `return_probs = True`, this method creates saliency maps for the
    predicted probability of the target class.

    If `return_probs = False`, creates saliency maps for the pre-softmax logit
    (unnormalized probability) of the target class.

    :param model_object: Instance of `keras.models.Model`.
    :param target_class: Saliency maps will be created for this class.  Must be
        an integer in 0...(K - 1), where K = number of classes.
    :param return_probs: See general discussion above.
    :param list_of_input_matrices: See doc for `_do_saliency_calculations`.
    :param ideal_logit: [used only if `return_probs = False`]
        The loss function will be (logit[k] - ideal_logit) ** 2, where logit[k]
        is the logit for the target class.  If `ideal_logit is None`, the loss
        function will be -sign(logit[k]) * logit[k]**2, or the negative signed
        square of logit[k], so that loss always decreases as logit[k] increases.
    :return: list_of_saliency_matrices: See doc for `_do_saliency_calculations`.
    :raises: TypeError: if `return_probs = False` and the output layer is not an
        activation layer.
    """

    check_metadata(
        component_type_string=model_interpretation.CLASS_COMPONENT_TYPE_STRING,
        target_class=target_class, return_probs=return_probs,
        ideal_logit=ideal_logit)

    if not return_probs:
        out_layer_type_string = type(model_object.layers[-1]).__name__
        if out_layer_type_string != 'Activation':
            error_string = (
                'If `return_probs = False`, the output layer must be an '
                '"Activation" layer (got "{0:s}" layer).  Otherwise, there is '
                'no way to access the pre-softmax logits (unnormalized '
                'probabilities).'
            ).format(out_layer_type_string)
            raise TypeError(error_string)

    num_output_neurons = model_object.layers[-1].output.get_shape().as_list()[
        -1]
    if num_output_neurons == 1 and target_class == 1:
        neuron_index = 0
    else:
        neuron_index = target_class

    if return_probs:
        loss_tensor = K.mean(
            (model_object.layers[-1].output[..., neuron_index] - 1) ** 2)
    else:
        if ideal_logit is None:
            loss_tensor = -K.mean(
                K.sign(model_object.layers[-1].input[..., neuron_index]) *
                model_object.layers[-1].input[..., neuron_index] ** 2)
        else:
            loss_tensor = K.mean(
                (model_object.layers[-1].input[..., neuron_index] -
                 ideal_logit) ** 2)

    return _do_saliency_calculations(
        model_object=model_object, loss_tensor=loss_tensor,
        list_of_input_matrices=list_of_input_matrices)


def get_saliency_maps_for_neuron_activation(
        model_object, layer_name, neuron_indices, list_of_input_matrices,
        ideal_activation=DEFAULT_IDEAL_ACTIVATION):
    """For each input example, creates saliency map for activatn of one neuron.

    :param model_object: Instance of `keras.models.Model`.
    :param layer_name: Name of layer containing the relevant neuron.
    :param neuron_indices: 1-D numpy array with indices of the relevant neuron.
        Must have length K - 1, where K = number of dimensions in layer output.
        The first dimension of the layer output is the example dimension, for
        which the index in this case is always 0.
    :param list_of_input_matrices: See doc for `_do_saliency_calculations`.
    :param ideal_activation: The loss function will be
        (neuron_activation - ideal_activation)** 2.  If
        `ideal_activation is None`, the loss function will be
        -sign(neuron_activation) * neuron_activation**2, or the negative signed
        square of neuron_activation, so that loss always decreases as
        neuron_activation increases.
    :return: list_of_saliency_matrices: See doc for `_do_saliency_calculations`.
    """

    check_metadata(
        component_type_string=model_interpretation.NEURON_COMPONENT_TYPE_STRING,
        layer_name=layer_name, ideal_activation=ideal_activation,
        neuron_index_matrix=numpy.expand_dims(neuron_indices, axis=0))

    if ideal_activation is None:
        loss_tensor = (
            -K.sign(
                model_object.get_layer(name=layer_name).output[
                    0, ..., neuron_indices]) *
            model_object.get_layer(name=layer_name).output[
                0, ..., neuron_indices] ** 2
        )
    else:
        loss_tensor = (
            model_object.get_layer(name=layer_name).output[
                0, ..., neuron_indices] -
            ideal_activation) ** 2

    return _do_saliency_calculations(
        model_object=model_object, loss_tensor=loss_tensor,
        list_of_input_matrices=list_of_input_matrices)


def get_saliency_maps_for_channel_activation(
        model_object, layer_name, channel_index, list_of_input_matrices,
        stat_function_for_neuron_activations,
        ideal_activation=DEFAULT_IDEAL_ACTIVATION):
    """For each input example, creates saliency map for activatn of one channel.

    :param model_object: Instance of `keras.models.Model`.
    :param layer_name: Name of layer containing the relevant channel.
    :param channel_index: Index of the relevant channel.  This method creates
        saliency maps for the [j]th output channel of `layer_name`, where
        j = `channel_index`.
    :param list_of_input_matrices: See doc for `_do_saliency_calculations`.
    :param stat_function_for_neuron_activations: Function used to process neuron
        activations.  In general, a channel contains many neurons, so there is
        an infinite number of ways to maximize the "channel activation," because
        there is an infinite number of ways to define "channel activation".
        This function must take a Keras tensor (containing neuron activations)
        and return a single number.  Some examples are `keras.backend.max` and
        `keras.backend.mean`.
    :param ideal_activation: See doc for
        `get_saliency_maps_for_neuron_activation`.
    :return: list_of_saliency_matrices: See doc for `_do_saliency_calculations`.
    """

    check_metadata(
        component_type_string=
        model_interpretation.CHANNEL_COMPONENT_TYPE_STRING,
        layer_name=layer_name, ideal_activation=ideal_activation,
        channel_indices=numpy.array([channel_index]))

    if ideal_activation is None:
        loss_tensor = -K.abs(stat_function_for_neuron_activations(
            model_object.get_layer(name=layer_name).output[
                0, ..., channel_index]))
    else:
        error_checking.assert_is_greater(ideal_activation, 0.)
        loss_tensor = K.abs(
            stat_function_for_neuron_activations(
                model_object.get_layer(name=layer_name).output[
                    0, ..., channel_index]) -
            ideal_activation)

    return _do_saliency_calculations(
        model_object=model_object, loss_tensor=loss_tensor,
        list_of_input_matrices=list_of_input_matrices)


def write_file(
        pickle_file_name, list_of_input_matrices, list_of_saliency_matrices,
        model_file_name, storm_id, storm_time_unix_sec, component_type_string,
        target_class=None, return_probs=None, ideal_logit=None, layer_name=None,
        ideal_activation=None, neuron_index_matrix=None, channel_indices=None):
    """Writes saliency maps to Pickle file.

    T = number of input tensors to the model
    C = number of model components (classes, neurons, or channels) for which
        saliency maps were computed

    All saliency maps are for the same example (storm object) but a different
    model component.

    :param pickle_file_name: Path to output file.
    :param list_of_input_matrices: length-T list of numpy arrays, comprising the
        input data for one storm object.  The first dimension of each array has
        length 1.
    :param list_of_saliency_matrices: length-T list of numpy arrays, comprising
        the saliency map for each model component.  The first dimension of each
        array has length C.  Otherwise, list_of_saliency_matrices[i] has the
        same dimensions as list_of_input_matrices[i].
    :param model_file_name: Path to file with trained model.
    :param storm_id: String ID for storm object.
    :param storm_time_unix_sec: Valid time for storm object.
    :param component_type_string: See doc for `check_metadata`.
    :param target_class: Same.
    :param return_probs: Same.
    :param ideal_logit: Same.
    :param layer_name: Same.
    :param ideal_activation: Same.
    :param neuron_index_matrix: Same.
    :param channel_indices: Same.
    :raises: ValueError: if `list_of_input_matrices` and
        `list_of_saliency_matrices` have different lengths.
    """

    num_components = check_metadata(
        component_type_string=component_type_string, target_class=target_class,
        return_probs=return_probs, ideal_logit=ideal_logit,
        layer_name=layer_name, ideal_activation=ideal_activation,
        neuron_index_matrix=neuron_index_matrix,
        channel_indices=channel_indices)

    error_checking.assert_is_string(model_file_name)
    error_checking.assert_is_string(storm_id)
    error_checking.assert_is_integer(storm_time_unix_sec)

    error_checking.assert_is_list(list_of_input_matrices)
    error_checking.assert_is_list(list_of_saliency_matrices)
    num_input_matrices = len(list_of_input_matrices)
    num_saliency_matrices = len(list_of_saliency_matrices)

    if num_input_matrices != num_saliency_matrices:
        error_string = (
            'Number of input matrices ({0:d}) should equal number of saliency '
            'matrices ({1:d}).'
        ).format(num_input_matrices, num_saliency_matrices)
        raise ValueError(error_string)

    for i in range(num_input_matrices):
        error_checking.assert_is_numpy_array(list_of_input_matrices[i])
        these_expected_dim = numpy.array(
            (1,) + list_of_input_matrices[i].shape[1:], dtype=int)
        error_checking.assert_is_numpy_array(
            list_of_input_matrices[i], exact_dimensions=these_expected_dim)

        these_expected_dim[0] = num_components
        error_checking.assert_is_numpy_array(
            list_of_saliency_matrices[i], exact_dimensions=these_expected_dim)

    metadata_dict = {
        MODEL_FILE_NAME_KEY: model_file_name,
        STORM_ID_KEY: storm_id,
        STORM_TIME_KEY: storm_time_unix_sec,
        COMPONENT_TYPE_KEY: component_type_string,
        TARGET_CLASS_KEY: target_class,
        RETURN_PROBS_KEY: return_probs,
        IDEAL_LOGIT_KEY: ideal_logit,
        LAYER_NAME_KEY: layer_name,
        IDEAL_ACTIVATION_KEY: ideal_activation,
        NEURON_INDICES_KEY: neuron_index_matrix,
        CHANNEL_INDICES_KEY: channel_indices,
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(list_of_input_matrices, pickle_file_handle)
    pickle.dump(list_of_saliency_matrices, pickle_file_handle)
    pickle.dump(metadata_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_file(pickle_file_name):
    """Reads saliency maps from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: list_of_input_matrices: See doc for `write_file`.
    :return: list_of_saliency_matrices: Same.
    :return: metadata_dict: Dictionary with the following keys.
    metadata_dict['model_file_name']: See doc for `write_file`.
    metadata_dict['storm_id']: Same.
    metadata_dict['storm_time_unix_sec']: Same.
    metadata_dict['component_type_string']: Same.
    metadata_dict['target_class']: Same.
    metadata_dict['return_probs']: Same.
    metadata_dict['ideal_logit']: Same.
    metadata_dict['layer_name']: Same.
    metadata_dict['ideal_activation']: Same.
    metadata_dict['neuron_index_matrix']: Same.
    metadata_dict['channel_indices']: Same.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    list_of_input_matrices = pickle.load(pickle_file_handle)
    list_of_saliency_matrices = pickle.load(pickle_file_handle)
    metadata_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    return list_of_input_matrices, list_of_saliency_matrices, metadata_dict
