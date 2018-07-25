"""Helper methods for feature optimization."""

import numpy
from keras import backend as K
from gewittergefahr.gg_utils import error_checking

LARGE_NUMBER = 1e6

DEFAULT_LEARNING_RATE = 0.01
DEFAULT_NUM_ITERATIONS = 200


def create_gaussian_initializer(mean, standard_deviation):
    """Creates Gaussian initializer.

    :param mean: Mean of Gaussian distribution.
    :param standard_deviation: Standard deviation of Gaussian distribution.
    :return: initializer: Function (see below).
    """

    def initializer(array_dimensions):
        """Initializes numpy array with Gaussian distribution.

        :param array_dimensions: numpy array of dimensions.
        :return: array: Array with the given dimensions.  For example, if
            array_dimensions = [1, 5, 10], this array will be 1 x 5 x 10.
        """

        return numpy.random.normal(
            loc=mean, scale=standard_deviation, size=array_dimensions)

    return initializer


def create_uniform_random_initializer(min_value, max_value):
    """Creates uniform-random initializer.

    :param min_value: Minimum value in uniform distribution.
    :param max_value: Max value in uniform distribution.
    :return: initializer: Function (see below).
    """

    def initializer(array_dimensions):
        """Initializes numpy array with uniform distribution.

        :param array_dimensions: numpy array of dimensions.
        :return: array: Array with the given dimensions.
        """

        return numpy.random.uniform(
            low=min_value, high=max_value, size=array_dimensions)

    return initializer


def create_constant_initializer(constant_value):
    """Creates constant initializer.

    :param constant_value: Constant value with which to fill numpy array.
    :return: initializer: Function (see below).
    """

    def initializer(array_dimensions):
        """Initializes numpy array with constant value.

        :param array_dimensions: numpy array of dimensions.
        :return: array: Array with the given dimensions.
        """

        return numpy.full(array_dimensions, constant_value, dtype=float)

    return initializer


def optimize_input_for_class(
        model_object, target_class, optimize_for_probability, initializer,
        num_iterations=DEFAULT_NUM_ITERATIONS,
        learning_rate=DEFAULT_LEARNING_RATE):
    """Finds the input that maximizes prediction of the target class.

    If `optimize_for_probability = True`, this method finds an input that
    maximizes the probability of the target class.  I recommend setting
    `optimize_for_probability = True` if you want to maximize prediction of the
    target class while minimizing predictions of the other classes.  Since the
    sum of class probabilities must be 1, maximizing probability for the target
    class entails minimizing probabilities for the other classes.

    If `optimize_for_probability = True`, this method finds an input that
    maximizes the logit (softmax input) of the target class.  I recommend
    setting `optimize_for_probability = False` if you want to maximize
    prediction of the target class regardless of how it affects the other
    classes.  Since the sum of logits (softmax inputs) is unbounded, maximizing
    logit for the target class does not necessarily minimize logits for the
    other classes.

    N = number of input tensors

    :param model_object: Instance of `keras.models.Model`.
    :param target_class: Input will be optimized for this class.  Must be an
        integer in 0...(K - 1), where K = number of classes.
    :param optimize_for_probability: See general discussion above.
    :param initializer: Function used to initialize input tensors.  See
        `create_gaussian_initializer` for an example.
    :param num_iterations: Number of iterations for the optimization procedure.
        This is the number of times that the input tensors will be adjusted.
    :param learning_rate: Learning rate.  At each iteration, each input value x
        will be decremented by `learning_rate * gradient`, where `gradient` is
        the gradient of x with respect to the loss function.  The "loss
        function" is the difference between the actual and maximum possible
        predictions for the target class.
    :return: list_of_optimized_input_matrices: length-N list of optimized input
        matrices (numpy arrays).
    :raises: TypeError: if `initializer` is not a function.
    """

    error_checking.assert_is_integer(target_class)
    error_checking.assert_is_boolean(optimize_for_probability)
    error_checking.assert_is_integer(num_iterations)
    error_checking.assert_is_greater(num_iterations, 0)
    error_checking.assert_is_greater(learning_rate, 0.)
    error_checking.assert_is_less_than(learning_rate, 1.)

    if not callable(initializer):
        raise TypeError('`initializer` is not callable (i.e., not a function).')

    # Define loss tensor.
    if optimize_for_probability:
        loss_tensor = K.mean(
            (model_object.layers[-1].output[:, target_class] - 1) ** 2)
    else:
        loss_tensor = K.mean(
            (model_object.layers[-1].output[:, target_class] - LARGE_NUMBER)
            ** 2)

    # Define and scale gradient tensors.
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

    # Define function from input tensors to loss and gradient tensors.
    inputs_to_loss_and_gradients = K.function(
        list_of_input_tensors + [K.learning_phase()],
        ([loss_tensor] + list_of_gradient_tensors))

    list_of_optimized_input_matrices = [None] * num_input_tensors
    for i in range(num_input_tensors):
        these_dimensions = numpy.array(
            [1] + list_of_input_tensors[i].get_shape().as_list()[1:], dtype=int)
        list_of_optimized_input_matrices[i] = initializer(these_dimensions)

    # Do optimization.
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
