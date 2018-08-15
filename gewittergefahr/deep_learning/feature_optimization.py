"""Helper methods for feature optimization.

--- REFERENCES ---

Olah, C., A. Mordvintsev, and L. Schubert, 2017: Feature visualization. Distill,
    doi:10.23915/distill.00007,
    URL https://distill.pub/2017/feature-visualization.
"""

import pickle
import numpy
from keras import backend as K
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import model_interpretation
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils

DEFAULT_IDEAL_ACTIVATION = 2.
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_NUM_ITERATIONS = 200

MODEL_FILE_NAME_KEY = 'model_file_name'
NUM_ITERATIONS_KEY = 'num_iterations'
LEARNING_RATE_KEY = 'learning_rate'
COMPONENT_TYPE_KEY = 'component_type_string'
INIT_FUNCTION_KEY = 'init_function'
TARGET_CLASS_KEY = 'target_class'
LAYER_NAME_KEY = 'layer_name'
IDEAL_ACTIVATION_KEY = 'ideal_activation'
NEURON_INDICES_KEY = 'neuron_index_matrix'
CHANNEL_INDICES_KEY = 'channel_indices'


def _do_gradient_descent(
        model_object, loss_tensor, init_function, num_iterations,
        learning_rate):
    """Does gradient descent for feature optimization.

    :param model_object: Instance of `keras.models.Model`.
    :param loss_tensor: Keras tensor defining the loss function.
    :param init_function: Function used to initialize input tensors.  See
        `create_gaussian_initializer` for an example.
    :param num_iterations: Number of iterations (number of times that the input
        tensors will be adjusted).
    :param learning_rate: Learning rate.  At each iteration, each input value x
        will be decremented by `learning_rate * gradient`, where `gradient` is
        the gradient of x with respect to the loss function.
    :return: list_of_optimized_input_matrices: length-T list of optimized input
        matrices (numpy arrays), where T = number of input tensors to the model.
    """

    if isinstance(model_object.input, list):
        list_of_input_tensors = model_object.input
    else:
        list_of_input_tensors = [model_object.input]

    list_of_gradient_tensors = K.gradients(loss_tensor, list_of_input_tensors)
    num_input_tensors = len(list_of_input_tensors)
    for i in range(num_input_tensors):
        list_of_gradient_tensors[i] /= K.maximum(
            K.sqrt(K.mean(list_of_gradient_tensors[i] ** 2)),
            K.epsilon())

    inputs_to_loss_and_gradients = K.function(
        list_of_input_tensors + [K.learning_phase()],
        ([loss_tensor] + list_of_gradient_tensors))

    list_of_optimized_input_matrices = [None] * num_input_tensors
    for i in range(num_input_tensors):
        these_dimensions = numpy.array(
            [1] + list_of_input_tensors[i].get_shape().as_list()[1:], dtype=int)
        list_of_optimized_input_matrices[i] = init_function(these_dimensions)

    for j in range(num_iterations):
        these_outputs = inputs_to_loss_and_gradients(
            list_of_optimized_input_matrices + [0])

        if numpy.mod(j, 100) == 0:
            print 'Loss at iteration {0:d} of {1:d}: {2:.2e}'.format(
                j + 1, num_iterations, these_outputs[0])

        for i in range(num_input_tensors):
            list_of_optimized_input_matrices[i] -= (
                these_outputs[i + 1] * learning_rate)

    print 'Loss after all {0:d} iterations: {1:.2e}'.format(
        num_iterations, these_outputs[0])
    return list_of_optimized_input_matrices


def check_metadata(
        num_iterations, learning_rate, component_type_string, target_class=None,
        layer_name=None, ideal_activation=None, neuron_index_matrix=None,
        channel_indices=None):
    """Error-checks metadata for feature optimization.

    C = number of model components (classes, neurons, or channels) for which
        input data were optimized

    :param num_iterations: See doc for `_do_gradient_descent`.
    :param learning_rate: Same.
    :param component_type_string: Component type (must be accepted by
        `model_interpretation.check_component_type`).
    :param target_class: See doc for `optimize_input_for_class`.
    :param layer_name: See doc for `optimize_input_for_neuron_activation` or
        `optimize_input_for_channel_activation`.
    :param ideal_activation: Same.
    :param neuron_index_matrix:
        [used only if component_type_string = "neuron"]
        C-by-? numpy array, where neuron_index_matrix[j, :] contains array
        indices of the [j]th neuron for which input data were optimized.
    :param channel_indices: [used only if component_type_string = "channel"]
        length-C numpy array, where channel_indices[j] is the index of the [j]th
        channel for which input data were optimized.
    :return: num_components: Number of model components (classes, neurons, or
        channels) for which input data were optimized.
    """

    error_checking.assert_is_integer(num_iterations)
    error_checking.assert_is_greater(num_iterations, 0)
    error_checking.assert_is_greater(learning_rate, 0.)
    error_checking.assert_is_less_than(learning_rate, 1.)
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


def create_gaussian_initializer(mean, standard_deviation):
    """Creates Gaussian initializer.

    :param mean: Mean of Gaussian distribution.
    :param standard_deviation: Standard deviation of Gaussian distribution.
    :return: init_function: Function (see below).
    """

    def init_function(array_dimensions):
        """Initializes numpy array with Gaussian distribution.

        :param array_dimensions: numpy array of dimensions.
        :return: array: Array with the given dimensions.  For example, if
            array_dimensions = [1, 5, 10], this array will be 1 x 5 x 10.
        """

        return numpy.random.normal(
            loc=mean, scale=standard_deviation, size=array_dimensions)

    return init_function


def create_uniform_random_initializer(min_value, max_value):
    """Creates uniform-random initializer.

    :param min_value: Minimum value in uniform distribution.
    :param max_value: Max value in uniform distribution.
    :return: init_function: Function (see below).
    """

    def init_function(array_dimensions):
        """Initializes numpy array with uniform distribution.

        :param array_dimensions: numpy array of dimensions.
        :return: array: Array with the given dimensions.
        """

        return numpy.random.uniform(
            low=min_value, high=max_value, size=array_dimensions)

    return init_function


def create_constant_initializer(constant_value):
    """Creates constant initializer.

    :param constant_value: Constant value with which to fill numpy array.
    :return: init_function: Function (see below).
    """

    def init_function(array_dimensions):
        """Initializes numpy array with constant value.

        :param array_dimensions: numpy array of dimensions.
        :return: array: Array with the given dimensions.
        """

        return numpy.full(array_dimensions, constant_value, dtype=float)

    return init_function


def create_climo_initializer(
        normalization_param_file_name, normalization_type_string,
        min_normalized_value=dl_utils.DEFAULT_MIN_NORMALIZED_VALUE,
        max_normalized_value=dl_utils.DEFAULT_MAX_NORMALIZED_VALUE,
        test_mode=False, sounding_field_names=None, sounding_pressures_mb=None,
        radar_field_names=None, radar_heights_m_asl=None,
        radar_field_name_by_channel=None, radar_height_by_channel_m_asl=None,
        radar_normalization_table=None, sounding_normalization_table=None):
    """Creates climatological initializer.

    Specifically, this function initializes each value to a climatological mean.
    There is one mean for each radar field/height and each sounding
    field/pressure.

    F_s = number of sounding fields in model input
    H_s = number of vertical sounding levels (pressures) in model input

    The following letters are used only for 3-D radar images.

    F_r = number of radar fields in model input
    H_r = number of radar heights in model input

    The following letters are used only for 2-D radar images.

    C = number of radar channels (field/height pairs) in model input

    :param normalization_param_file_name: See doc for
        `dl_utils.read_normalization_params_from_file`.
    :param normalization_type_string: Same.
    :param min_normalized_value: Same.
    :param max_normalized_value: Same.
    :param test_mode: For testing only.  Leave this alone.
    :param sounding_field_names:
        [if model input does not contain soundings, leave this as `None`]
        List (length F_s) with names of sounding fields, in the order that they
        appear in the corresponding input tensor.
    :param sounding_pressures_mb:
        [if model input does not contain soundings, leave this as `None`]
        numpy array (length H_s) of sounding pressure levels, in the order that
        they appear in the corresponding input tensor.
    :param radar_field_names:
        [if model input does not contain 3-D radar images, leave this as `None`]
        List (length F_r) with names of radar fields, in the order that they
        appear in the corresponding input tensor.
    :param radar_heights_m_asl:
        [if model input does not contain 3-D radar images, leave this as `None`]
        numpy array (length H_r) of radar heights (metres above sea level), in
        the order that they appear in the corresponding input tensor.
    :param radar_field_name_by_channel:
        [if model input does not contain 2-D radar images, leave this as `None`]
        Length-C list of radar fields, in the order that they appear in the
        corresponding input tensor.
    :param radar_height_by_channel_m_asl:
        [if model input does not contain 2-D radar images, leave this as `None`]
        Length-C numpy array of radar heights (metres above sea level), in the
        order that they appear in the corresponding input tensor.
    :param radar_normalization_table: For testing only.  Leave this alone.
    :param sounding_normalization_table: For testing only.  Leave this alone.
    :return: init_function: Function (see below).
    """

    error_checking.assert_is_boolean(test_mode)
    if not test_mode:
        (_, radar_normalization_table, _, sounding_normalization_table
        ) = dl_utils.read_normalization_params_from_file(
            normalization_param_file_name)

    if sounding_field_names is not None:
        error_checking.assert_is_string_list(sounding_field_names)
        error_checking.assert_is_numpy_array(
            numpy.array(sounding_field_names), num_dimensions=1)

        error_checking.assert_is_integer_numpy_array(sounding_pressures_mb)
        error_checking.assert_is_numpy_array(
            sounding_pressures_mb, num_dimensions=1)

    if radar_field_names is not None:
        error_checking.assert_is_string_list(radar_field_names)
        error_checking.assert_is_numpy_array(
            numpy.array(radar_field_names), num_dimensions=1)

        error_checking.assert_is_integer_numpy_array(radar_heights_m_asl)
        error_checking.assert_is_numpy_array(
            radar_heights_m_asl, num_dimensions=1)

    if radar_field_name_by_channel is not None:
        error_checking.assert_is_string_list(radar_field_name_by_channel)
        error_checking.assert_is_numpy_array(
            numpy.array(radar_field_name_by_channel), num_dimensions=1)

        error_checking.assert_is_integer_numpy_array(
            radar_height_by_channel_m_asl)
        error_checking.assert_is_numpy_array(
            radar_height_by_channel_m_asl,
            exact_dimensions=numpy.array([len(radar_field_name_by_channel)]))

    def init_function(array_dimensions):
        """Initializes numpy array with climatological means.

        If len(array_dimensions) = 3, this method assumes that the corresponding
        input tensor contains soundings.
        If len(array_dimensions) = 4 ... 2-D radar images.
        If len(array_dimensions) = 5 ... 3-D radar images.

        :param array_dimensions: numpy array of dimensions.
        :return: array: Array with the given dimensions.
        """

        array = numpy.full(array_dimensions, numpy.nan)
        if len(array_dimensions) == 5:
            for j in range(len(radar_field_names)):
                for k in range(len(radar_heights_m_asl)):
                    this_key = (radar_field_names[j], radar_heights_m_asl[k])
                    array[..., k, j] = radar_normalization_table[
                        dl_utils.MEAN_VALUE_COLUMN].loc[[this_key]].values[0]

            return dl_utils.normalize_radar_images(
                radar_image_matrix=array, field_names=radar_field_names,
                normalization_type_string=normalization_type_string,
                normalization_param_file_name=normalization_param_file_name,
                test_mode=test_mode, min_normalized_value=min_normalized_value,
                max_normalized_value=max_normalized_value,
                normalization_table=radar_normalization_table)

        if len(array_dimensions) == 4:
            for j in range(len(radar_field_name_by_channel)):
                this_key = (radar_field_name_by_channel[j],
                            radar_height_by_channel_m_asl[j])
                array[..., j] = radar_normalization_table[
                    dl_utils.MEAN_VALUE_COLUMN].loc[[this_key]].values[0]

            return dl_utils.normalize_radar_images(
                radar_image_matrix=array,
                field_names=radar_field_name_by_channel,
                normalization_type_string=normalization_type_string,
                normalization_param_file_name=normalization_param_file_name,
                test_mode=test_mode, min_normalized_value=min_normalized_value,
                max_normalized_value=max_normalized_value,
                normalization_table=radar_normalization_table)

        if len(array_dimensions) == 3:
            for j in range(len(sounding_field_names)):
                for k in range(len(sounding_pressures_mb)):
                    this_key = (
                        sounding_field_names[j], sounding_pressures_mb[k])
                    array[..., k, j] = sounding_normalization_table[
                        dl_utils.MEAN_VALUE_COLUMN].loc[[this_key]].values[0]

            return dl_utils.normalize_soundings(
                sounding_matrix=array,
                pressureless_field_names=sounding_field_names,
                normalization_type_string=normalization_type_string,
                normalization_param_file_name=normalization_param_file_name,
                test_mode=test_mode, min_normalized_value=min_normalized_value,
                max_normalized_value=max_normalized_value,
                normalization_table=sounding_normalization_table)

        return None

    return init_function


def optimize_input_for_class(
        model_object, target_class, init_function,
        num_iterations=DEFAULT_NUM_ITERATIONS,
        learning_rate=DEFAULT_LEARNING_RATE):
    """Optimizes synthetic input example for probability of target class.

    :param model_object: Instance of `keras.models.Model`.
    :param target_class: Synthetic input data will be optimized for this class.
        Must be an integer in 0...(K - 1), where K = number of classes.
    :param init_function: See doc for `_do_gradient_descent`.
    :param num_iterations: Same.
    :param learning_rate: Same.
    :return: list_of_optimized_input_matrices: See doc for
        `_do_gradient_descent`.
    """

    check_metadata(
        num_iterations=num_iterations, learning_rate=learning_rate,
        component_type_string=model_interpretation.CLASS_COMPONENT_TYPE_STRING,
        target_class=target_class)

    num_output_neurons = model_object.layers[-1].output.get_shape().as_list()[
        -1]

    if num_output_neurons == 1:
        error_checking.assert_is_leq(target_class, 1)
        if target_class == 1:
            loss_tensor = K.mean(
                (model_object.layers[-1].output[..., 0] - 1) ** 2)
        else:
            loss_tensor = K.mean(
                model_object.layers[-1].output[..., 0] ** 2)
    else:
        error_checking.assert_is_less_than(target_class, num_output_neurons)
        loss_tensor = K.mean(
            (model_object.layers[-1].output[..., target_class] - 1) ** 2)

    return _do_gradient_descent(
        model_object=model_object, loss_tensor=loss_tensor,
        init_function=init_function, num_iterations=num_iterations,
        learning_rate=learning_rate)


def optimize_input_for_neuron_activation(
        model_object, layer_name, neuron_indices, init_function,
        num_iterations=DEFAULT_NUM_ITERATIONS,
        learning_rate=DEFAULT_LEARNING_RATE,
        ideal_activation=DEFAULT_IDEAL_ACTIVATION):
    """Optimizes synthetic input example for activation of the given neuron.

    :param model_object: Instance of `keras.models.Model`.
    :param layer_name: Name of layer containing the relevant neuron.
    :param neuron_indices: 1-D numpy array with indices of the relevant neuron.
        Must have length K - 1, where K = number of dimensions in layer output.
        The first dimension of the layer output is the example dimension, for
        which the index in this case is always 0.
    :param init_function: See doc for `_do_gradient_descent`.
    :param num_iterations: Same.
    :param learning_rate: Same.
    :param ideal_activation: The loss function will be
        (neuron_activation - ideal_activation)** 2.  If
        `ideal_activation is None`, the loss function will be
        -sign(neuron_activation) * neuron_activation**2, or the negative signed
        square of neuron_activation, so that loss always decreases as
        neuron_activation increases.
    :return: list_of_optimized_input_matrices: See doc for
        `_do_gradient_descent`.
    """

    check_metadata(
        num_iterations=num_iterations, learning_rate=learning_rate,
        component_type_string=model_interpretation.NEURON_COMPONENT_TYPE_STRING,
        layer_name=layer_name, ideal_activation=ideal_activation,
        neuron_index_matrix=numpy.expand_dims(neuron_indices, axis=0))

    neuron_indices_as_tuple = (0,) + tuple(neuron_indices)

    if ideal_activation is None:
        loss_tensor = -(
            K.sign(
                model_object.get_layer(name=layer_name).output[
                    neuron_indices_as_tuple]) *
            model_object.get_layer(name=layer_name).output[
                neuron_indices_as_tuple] ** 2
        )
    else:
        loss_tensor = (
            model_object.get_layer(name=layer_name).output[
                neuron_indices_as_tuple] -
            ideal_activation) ** 2

    return _do_gradient_descent(
        model_object=model_object, loss_tensor=loss_tensor,
        init_function=init_function, num_iterations=num_iterations,
        learning_rate=learning_rate)


def optimize_input_for_channel_activation(
        model_object, layer_name, channel_index, init_function,
        stat_function_for_neuron_activations,
        num_iterations=DEFAULT_NUM_ITERATIONS,
        learning_rate=DEFAULT_LEARNING_RATE,
        ideal_activation=DEFAULT_IDEAL_ACTIVATION):
    """Optimizes synthetic input example for activation of the given channel.

    :param model_object: Instance of `keras.models.Model`.
    :param layer_name: Name of layer containing the relevant channel.
    :param channel_index: Index of the relevant channel.  This method optimizes
        synthetic input data for activation of the [j]th output channel of
        `layer_name`, where j = `channel_index`.
    :param init_function: See doc for `_do_gradient_descent`.
    :param stat_function_for_neuron_activations: Function used to process neuron
        activations.  In general, a channel contains many neurons, so there is
        an infinite number of ways to maximize the "channel activation," because
        there is an infinite number of ways to define "channel activation".
        This function must take a Keras tensor (containing neuron activations)
        and return a single number.  Some examples are `keras.backend.max` and
        `keras.backend.mean`.
    :param num_iterations: See doc for `_do_gradient_descent`.
    :param learning_rate: Same.
    :param ideal_activation: The loss function will be
        abs(stat_function_for_neuron_activations(neuron_activations) -
            ideal_activation).

        For example, if `stat_function_for_neuron_activations` is the mean,
        loss function will be abs(mean(neuron_activations) - ideal_activation).
        If `ideal_activation is None`, the loss function will be
        -1 * abs(stat_function_for_neuron_activations(neuron_activations) -
                 ideal_activation).

    :return: list_of_optimized_input_matrices: See doc for
        `_do_gradient_descent`.
    """

    check_metadata(
        num_iterations=num_iterations, learning_rate=learning_rate,
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

    return _do_gradient_descent(
        model_object=model_object, loss_tensor=loss_tensor,
        init_function=init_function, num_iterations=num_iterations,
        learning_rate=learning_rate)


def write_file(
        pickle_file_name, list_of_optimized_input_matrices, model_file_name,
        num_iterations, learning_rate, component_type_string, init_function,
        target_class=None, layer_name=None, ideal_activation=None,
        neuron_index_matrix=None, channel_indices=None):
    """Writes optimized input data to Pickle file.

    :param pickle_file_name: Path to output file.
    :param list_of_optimized_input_matrices: List of optimized input matrices,
        created by `_do_gradient_descent`.
    :param model_file_name: Path to file with trained model.
    :param num_iterations: See doc for `check_metadata`.
    :param learning_rate: Same.
    :param component_type_string: Same.
    :param init_function: Same.
    :param target_class: Same.
    :param layer_name: Same.
    :param ideal_activation: Same.
    :param neuron_index_matrix: Same.
    :param channel_indices: Same.
    """

    num_components = check_metadata(
        num_iterations=num_iterations, learning_rate=learning_rate,
        component_type_string=component_type_string, target_class=target_class,
        layer_name=layer_name, ideal_activation=ideal_activation,
        neuron_index_matrix=neuron_index_matrix,
        channel_indices=channel_indices)

    error_checking.assert_is_string(model_file_name)
    error_checking.assert_is_list(list_of_optimized_input_matrices)

    for this_array in list_of_optimized_input_matrices:
        error_checking.assert_is_numpy_array(this_array)
        these_expected_dim = numpy.array(
            (num_components,) + this_array.shape[1:], dtype=int)
        error_checking.assert_is_numpy_array(
            this_array, exact_dimensions=these_expected_dim)

    metadata_dict = {
        MODEL_FILE_NAME_KEY: model_file_name,
        NUM_ITERATIONS_KEY: num_iterations,
        LEARNING_RATE_KEY: learning_rate,
        COMPONENT_TYPE_KEY: component_type_string,
        INIT_FUNCTION_KEY: init_function,
        TARGET_CLASS_KEY: target_class,
        LAYER_NAME_KEY: layer_name,
        IDEAL_ACTIVATION_KEY: ideal_activation,
        NEURON_INDICES_KEY: neuron_index_matrix,
        CHANNEL_INDICES_KEY: channel_indices,
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(list_of_optimized_input_matrices, pickle_file_handle)
    pickle.dump(metadata_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_file(pickle_file_name):
    """Reads optimized input data from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: list_of_optimized_input_matrices: See doc for `write_file`.
    :return: metadata_dict: Dictionary with the following keys.
    metadata_dict['model_file_name']: See doc for `write_file`.
    metadata_dict['num_iterations']: Same.
    metadata_dict['learning_rate']: Same.
    metadata_dict['component_type_string']: Same.
    metadata_dict['init_function']: Same.
    metadata_dict['target_class']: Same.
    metadata_dict['layer_name']: Same.
    metadata_dict['ideal_activation']: Same.
    metadata_dict['neuron_index_matrix']: Same.
    metadata_dict['channel_indices']: Same.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    list_of_optimized_input_matrices = pickle.load(pickle_file_handle)
    metadata_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    return list_of_optimized_input_matrices, metadata_dict
