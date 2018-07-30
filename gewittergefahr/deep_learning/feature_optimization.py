"""Helper methods for feature optimization.

--- REFERENCES ---

Olah, C., A. Mordvintsev, and L. Schubert, 2017: Feature visualization. Distill,
    doi:10.23915/distill.00007,
    URL https://distill.pub/2017/feature-visualization.
"""

import numpy
from keras import backend as K
from gewittergefahr.gg_utils import error_checking

DEFAULT_IDEAL_LOGIT = 7.
DEFAULT_IDEAL_ACTIVATION = 2.

DEFAULT_LEARNING_RATE = 0.01
DEFAULT_NUM_ITERATIONS = 200

CLASS_OPTIMIZATION_TYPE_STRING = 'class'
NEURON_OPTIMIZATION_TYPE_STRING = 'neuron'
CHANNEL_OPTIMIZATION_TYPE_STRING = 'channel'
VALID_OPTIMIZATION_TYPE_STRINGS = [
    CLASS_OPTIMIZATION_TYPE_STRING, NEURON_OPTIMIZATION_TYPE_STRING,
    CHANNEL_OPTIMIZATION_TYPE_STRING
]


def _check_input_args(init_function, num_iterations, learning_rate):
    """Checks input args to optimization methods.

    :param init_function: Function used to initialize input tensors.  See
        `create_gaussian_initializer` for an example.
    :param num_iterations: Number of iterations for the optimization procedure.
        This is the number of times that the input tensors will be adjusted.
    :param learning_rate: Learning rate.  At each iteration, each input value x
        will be decremented by `learning_rate * gradient`, where `gradient` is
        the gradient of x with respect to the loss function.
    :raises: TypeError: if `init_function` is not a function.
    """

    error_checking.assert_is_integer(num_iterations)
    error_checking.assert_is_greater(num_iterations, 0)
    error_checking.assert_is_greater(learning_rate, 0.)
    error_checking.assert_is_less_than(learning_rate, 1.)

    if not callable(init_function):
        raise TypeError(
            '`init_function` is not callable (i.e., not a function).')


def _do_gradient_descent(
        model_object, loss_tensor, init_function, num_iterations,
        learning_rate):
    """Does gradient descent for feature optimization.

    :param model_object: Instance of `keras.models.Model`.
    :param loss_tensor: Keras tensor defining the loss function.
    :param init_function: See doc for `_check_input_args`.
    :param num_iterations: Same.
    :param learning_rate: Same.
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


def check_optimization_type(optimization_type_string):
    """Ensures that optimization type is valid.

    :param optimization_type_string: Optimization type.
    :raises: ValueError: if
        `optimization_type_string not in VALID_OPTIMIZATION_TYPE_STRINGS`.
    """

    error_checking.assert_is_string(optimization_type_string)
    if optimization_type_string not in VALID_OPTIMIZATION_TYPE_STRINGS:
        error_string = (
            '\n\n{0:s}\nValid optimization types (listed above) do not include '
            '"{1:s}".'
        ).format(str(VALID_OPTIMIZATION_TYPE_STRINGS), optimization_type_string)
        raise ValueError(error_string)


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


def optimize_input_for_class(
        model_object, target_class, optimize_for_probability, init_function,
        num_iterations=DEFAULT_NUM_ITERATIONS,
        learning_rate=DEFAULT_LEARNING_RATE, ideal_logit=DEFAULT_IDEAL_LOGIT):
    """Finds an input that maximizes prediction of the target class.

    If `optimize_for_probability = True`, this method finds an input that maxxes
    the predicted probability of the target class.  This also minimizes the sum
    of predicted probabilities of the other classes, since all probabilities
    must sum to 1.

    If `optimize_for_probability = False`, this method finds an input that
    maxxes the logit for the target class.  Each input to the prediction layer's
    activation function is a logit, and each output is a probability, so logits
    can be viewed as "unnormalized probabilities".  Maxxing the logit for the
    target class does not necessarily minimize the sum of logits for the other
    classes, because the sum of all logits is unbounded.

    This leads to the following recommendations:

    [1] If you want to maximize prediction of the target class while minimizing
        the prediction of all other classes, set
        `optimize_for_probability = True`.
    [2] If you want to maximize prediction of the target class, regardless of
        how this affects predictions of the other classes, set
        `optimize_for_probability = False`.

    According to Olah et al. (2017), "optimizing pre-softmax logits produces
    images of better visual quality".  However, this was for a multiclass
    problem.  The same may not be true for a binary problem.

    :param model_object: Instance of `keras.models.Model`.
    :param target_class: Input will be optimized for this class.  Must be an
        integer in 0...(K - 1), where K = number of classes.
    :param optimize_for_probability: See general discussion above.
    :param init_function: See doc for `_check_input_args`.
    :param num_iterations: Same.
    :param learning_rate: Same.
    :param ideal_logit: [used only if `optimize_for_probability = False`]
        The loss function will be (logit[k] - ideal_logit) ** 2, where logit[k]
        is the logit for the target class.  If `ideal_logit is None`, the loss
        function will be -sign(logit[k]) * logit[k]**2, or the negative signed
        square of logit[k], so that loss always decreases as logit[k] increases.
    :return: list_of_optimized_input_matrices: See doc for
        `_do_gradient_descent`.
    :raises: TypeError: if `optimize_for_probability = False` and the output
        layer is not an activation layer.
    """

    _check_input_args(
        init_function=init_function, num_iterations=num_iterations,
        learning_rate=learning_rate)

    error_checking.assert_is_integer(target_class)
    error_checking.assert_is_boolean(optimize_for_probability)
    if not optimize_for_probability:
        if ideal_logit is not None:
            error_checking.assert_is_greater(ideal_logit, 0.)

        out_layer_type_string = type(model_object.layers[-1]).__name__
        if out_layer_type_string != 'Activation':
            error_string = (
                'If `optimize_for_probability = False`, the output layer must '
                'be an "Activation" layer (got "{0:s}" layer).  Otherwise, '
                'there is no way to access the pre-softmax logits (unnormalized'
                ' probabilities).'
            ).format(out_layer_type_string)
            raise TypeError(error_string)

    if optimize_for_probability:
        loss_tensor = K.mean(
            (model_object.layers[-1].output[..., target_class] - 1) ** 2)
    else:
        if ideal_logit is None:
            loss_tensor = -K.mean(
                K.sign(model_object.layers[-1].input[..., target_class]) *
                model_object.layers[-1].input[..., target_class] ** 2)
        else:
            loss_tensor = K.mean(
                (model_object.layers[-1].input[..., target_class] -
                 ideal_logit) ** 2)

    return _do_gradient_descent(
        model_object=model_object, loss_tensor=loss_tensor,
        init_function=init_function, num_iterations=num_iterations,
        learning_rate=learning_rate)


def optimize_input_for_neuron_activation(
        model_object, layer_name, neuron_indices, init_function,
        num_iterations=DEFAULT_NUM_ITERATIONS,
        learning_rate=DEFAULT_LEARNING_RATE,
        ideal_activation=DEFAULT_IDEAL_ACTIVATION):
    """Finds an input that maximizes the activation of one neuron in one layer.

    :param model_object: Instance of `keras.models.Model`.
    :param layer_name: Name of layer with neuron whose activation is to be
        maximized.
    :param neuron_indices: 1-D numpy array with indices of neuron whose
        activation is to be maximized.  If the layer output has K dimensions,
        `neuron_indices` must have length K - 1.  (The first dimension of the
        layer output is the example dimension, for which the index is 0 in this
        case.)
    :param init_function: See doc for `_check_input_args`.
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

    _check_input_args(
        init_function=init_function, num_iterations=num_iterations,
        learning_rate=learning_rate)

    error_checking.assert_is_string(layer_name)
    error_checking.assert_is_integer_numpy_array(neuron_indices)
    error_checking.assert_is_numpy_array(neuron_indices, num_dimensions=1)
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
    """Finds an input that maximizes the activation of one channel in one layer.

    :param model_object: Instance of `keras.models.Model`.
    :param layer_name: Name of layer with channel whose activation is to be
        maximized.
    :param channel_index: Index of channel whose activation is to be maximized.
        If `channel_index = c`, the activation of the [c]th channel in the
        layer will be maximized.
    :param init_function: See doc for `_check_input_args`.
    :param stat_function_for_neuron_activations: Function used to process neuron
        activations.  In general, a channel contains many neurons, so there is
        an infinite number of ways to maximize the "channel activation," because
        there is an infinite number of ways to define "channel activation".
        This function must take a Keras tensor (containing neuron activations)
        and return a single number.  Some examples are `keras.backend.max` and
        `keras.backend.mean`.
    :param num_iterations: See doc for `_check_input_args`.
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
    :raises: TypeError: if `stat_function_for_neuron_activations` is not a
        function.
    """

    _check_input_args(
        init_function=init_function, num_iterations=num_iterations,
        learning_rate=learning_rate)

    error_checking.assert_is_string(layer_name)
    error_checking.assert_is_integer(channel_index)
    if not callable(stat_function_for_neuron_activations):
        raise TypeError('`stat_function_for_neuron_activations` is not callable'
                        ' (i.e., not a function).')

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


def sort_neurons_by_weight(model_object, layer_name):
    """Sorts neurons of the given layer in descending order by weight.

    K = number of dimensions in `weight_matrix`
    W = number of values in `weight_matrix`

    :param model_object: Instance of `keras.models.Model`.
    :param layer_name: Name of layer whose neurons are to be sorted.
    :return: weight_matrix: numpy array of weights, with the same dimensions as
        `model_object.get_layer(name=layer_name).get_weights()[0]`.

    If the layer is convolutional, dimensions of `weight_matrix` are as follows:

    - Last dimension = output channel
    - Second-last dimension = input channel
    - First dimensions = spatial dimensions

    For example, if the conv layer has a 3-by-5 kernel with 16 input channels
    and 32 output channels, `weight_matrix` will be 3 x 5 x 16 x 32.

    If the layer is dense (fully connected), `weight_matrix` is 1-D.

    :return: sort_indices_as_tuple: length-K tuple.  sort_indices_as_tuple[k] is
        a length-W numpy array, containing indices for the [k]th dimension of
        `weight_matrix`.  When these indices are applied to all dimensions of
        `weight_matrix` -- i.e., when sort_indices_as_tuple[k] is applied for
        k = 0...(K - 1) -- `weight_matrix` has been sorted in descending order.
    :raises: TypeError: if the given layer is neither dense nor convolutional.
    """

    error_checking.assert_is_string(layer_name)

    layer_type_string = type(model_object.get_layer(name=layer_name)).__name__
    valid_layer_type_strings = ['Dense', 'Conv1D', 'Conv2D', 'Conv3D']
    if layer_type_string not in valid_layer_type_strings:
        error_string = (
            '\n\n{0:s}\nLayer "{1:s}" has type "{2:s}", which is not in the '
            'above list.'
        ).format(str(valid_layer_type_strings), layer_name, layer_type_string)
        raise TypeError(error_string)

    weight_matrix = model_object.get_layer(name=layer_name).get_weights()[0]
    sort_indices_linear = numpy.argsort(
        -numpy.reshape(weight_matrix, weight_matrix.size))
    sort_indices_as_tuple = numpy.unravel_index(
        sort_indices_linear, weight_matrix.shape)

    return weight_matrix, sort_indices_as_tuple


def get_class_activation_for_examples(
        model_object, target_class, return_probs, list_of_input_matrices):
    """Returns prediction of one class for each input example.

    If `return_probs = True`, this method returns the predicted probability of
    the target class for each example.

    If `return_probs = False`, returns the logit of the target class for each
    example.  Each input to the prediction layer's activation function is a
    logit, and each output is a probability, so logits can be viewed as
    "unnormalized probabilities".

    :param model_object: Instance of `keras.models.Model`.
    :param target_class: Predictions will be returned for this class.  Must be
        an integer in 0...(K - 1), where K = number of classes.
    :param return_probs: See general discussion above.
    :param list_of_input_matrices: length-T list of numpy arrays, comprising one
        or more examples (storm objects).  list_of_input_matrices[i] must have
        the same dimensions as the [i]th input tensor to the model.
    :return: activation_values: length-E numpy array, where activation_values[i]
        is the activation (prediction) of the given class for the [i]th example.
    :raises: TypeError: if `return_probs = False` and the output layer is not an
        activation layer.
    """

    error_checking.assert_is_integer(target_class)
    error_checking.assert_is_boolean(return_probs)
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

    if isinstance(model_object.input, list):
        list_of_input_tensors = model_object.input
    else:
        list_of_input_tensors = [model_object.input]

    if return_probs:
        activation_function = K.function(
            list_of_input_tensors + [K.learning_phase()],
            [model_object.layers[-1].output[..., target_class]])
    else:
        activation_function = K.function(
            list_of_input_tensors + [K.learning_phase()],
            [model_object.layers[-1].input[..., target_class]])

    return activation_function(list_of_input_matrices + [0])[0]


def get_neuron_activation_for_examples(
        model_object, layer_name, neuron_indices, list_of_input_matrices):
    """Returns activation of one neuron by each input example.

    T = number of input tensors to the model
    E = number of examples (storm objects)

    :param model_object: Instance of `keras.models.Model`.
    :param layer_name: Name of layer with neuron whose activation is to be
        computed.
    :param neuron_indices: 1-D numpy array with indices of neuron whose
        activation is to be computed.  If the layer output has K dimensions,
        `neuron_indices` must have length K - 1.  (The first dimension of the
        layer output is the example dimension, for which all indices from
        0...[E - 1] are used.)
    :param list_of_input_matrices: See doc for
        `get_class_activation_for_examples`.
    :return: activation_values: length-E numpy array, where activation_values[i]
        is the activation of the given neuron by the [i]th example.
    """

    error_checking.assert_is_string(layer_name)
    error_checking.assert_is_integer_numpy_array(neuron_indices)
    error_checking.assert_is_numpy_array(neuron_indices, num_dimensions=1)

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
    """Returns activation of one channel by each input example.

    :param model_object: Instance of `keras.models.Model`.
    :param layer_name: Name of layer with channel whose activation is to be
        computed.
    :param channel_index: Index of channel whose activation is to be computed.
        If `channel_index = c`, the activation of the [c]th channel in the
        layer will be computed.
    :param list_of_input_matrices: See doc for
        `get_class_activation_for_examples`.
    :param stat_function_for_neuron_activations: See doc for
        `optimize_input_for_channel_activation`.
    :return: activation_values: length-E numpy array, where activation_values[i]
        is stat_function_for_neuron_activations(channel activations) for the
        [i]th example.
    """

    error_checking.assert_is_string(layer_name)
    error_checking.assert_is_integer(channel_index)
    if not callable(stat_function_for_neuron_activations):
        raise TypeError('`stat_function_for_neuron_activations` is not callable'
                        ' (i.e., not a function).')

    if isinstance(model_object.input, list):
        list_of_input_tensors = model_object.input
    else:
        list_of_input_tensors = [model_object.input]

    activation_function = K.function(
        list_of_input_tensors + [K.learning_phase()],
        [model_object.get_layer(name=layer_name).output[..., channel_index]])

    return activation_function(list_of_input_matrices + [0])[0]


def get_saliency_maps_for_class_activation(
        model_object, target_class, return_probs, list_of_input_matrices,
        ideal_logit=DEFAULT_IDEAL_LOGIT):
    """Creates saliency map for prediction of one class by each input example.

    :param model_object: Instance of `keras.models.Model`.
    :param target_class: See doc for `get_class_activation_for_examples`.
    :param return_probs: Same.
    :param list_of_input_matrices: See doc for `_do_saliency_calculations`.
    :param ideal_logit: See doc for `optimize_input_for_class`.
    :return: list_of_saliency_matrices: See doc for `_do_saliency_calculations`.
    """

    error_checking.assert_is_integer(target_class)
    error_checking.assert_is_boolean(return_probs)
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

    if return_probs:
        loss_tensor = K.mean(
            (model_object.layers[-1].output[..., target_class] - 1) ** 2)
    else:
        if ideal_logit is None:
            loss_tensor = -K.mean(
                K.sign(model_object.layers[-1].input[..., target_class]) *
                model_object.layers[-1].input[..., target_class] ** 2)
        else:
            loss_tensor = K.mean(
                (model_object.layers[-1].input[..., target_class] -
                 ideal_logit) ** 2)

    return _do_saliency_calculations(
        model_object=model_object, loss_tensor=loss_tensor,
        list_of_input_matrices=list_of_input_matrices)


def get_saliency_maps_for_neuron_activation(
        model_object, layer_name, neuron_indices, list_of_input_matrices,
        ideal_activation=DEFAULT_IDEAL_ACTIVATION):
    """Creates saliency map for activation of one neuron by each input example.

    T = number of input tensors to the model

    :param model_object: Instance of `keras.models.Model`.
    :param layer_name: See doc for `get_neuron_activation_for_examples`.
    :param neuron_indices: Same.
    :param list_of_input_matrices: See doc for `_do_saliency_calculations`.
    :param ideal_activation: See doc for `optimize_input_for_neuron_activation`.
    :return: list_of_saliency_matrices: See doc for `_do_saliency_calculations`.
    """

    error_checking.assert_is_string(layer_name)
    error_checking.assert_is_integer_numpy_array(neuron_indices)
    error_checking.assert_is_numpy_array(neuron_indices, num_dimensions=1)

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
    """Creates saliency map for activation of one channel by each input example.

    :param model_object: Instance of `keras.models.Model`.
    :param layer_name: See doc for `get_channel_activation_for_examples`.
    :param channel_index: Same.
    :param list_of_input_matrices: ee doc for `_do_saliency_calculations`.
    :param stat_function_for_neuron_activations: See doc for
        `optimize_input_for_channel_activation`.
    :param ideal_activation: Same.
    :return: list_of_saliency_matrices: ee doc for `_do_saliency_calculations`.
    """

    error_checking.assert_is_string(layer_name)
    error_checking.assert_is_integer(channel_index)
    if not callable(stat_function_for_neuron_activations):
        raise TypeError('`stat_function_for_neuron_activations` is not callable'
                        ' (i.e., not a function).')

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
