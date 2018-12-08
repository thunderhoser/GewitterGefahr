"""Methods for backwards optimization (or "feature optimization").

--- REFERENCES ---

Olah, C., A. Mordvintsev, and L. Schubert, 2017: Feature visualization. Distill,
    doi:10.23915/distill.00007,
    URL https://distill.pub/2017/feature-visualization.
"""

import copy
import pickle
import numpy
from keras import backend as K
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import model_interpretation
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.deep_learning import training_validation_io as trainval_io

DEFAULT_IDEAL_ACTIVATION = 2.
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_NUM_ITERATIONS = 200

MODEL_FILE_NAME_KEY = 'model_file_name'
NUM_ITERATIONS_KEY = 'num_iterations'
LEARNING_RATE_KEY = 'learning_rate'
COMPONENT_TYPE_KEY = 'component_type_string'
INIT_FUNCTION_KEY = 'init_function_name_or_matrices'
TARGET_CLASS_KEY = 'target_class'
LAYER_NAME_KEY = 'layer_name'
IDEAL_ACTIVATION_KEY = 'ideal_activation'
NEURON_INDICES_KEY = 'neuron_indices'
CHANNEL_INDEX_KEY = 'channel_index'

GAUSSIAN_INIT_FUNCTION_NAME = 'gaussian'
UNIFORM_INIT_FUNCTION_NAME = 'uniform'
CONSTANT_INIT_FUNCTION_NAME = 'constant'
CLIMO_INIT_FUNCTION_NAME = 'climo'

VALID_INIT_FUNCTION_NAMES = [
    GAUSSIAN_INIT_FUNCTION_NAME, UNIFORM_INIT_FUNCTION_NAME,
    CONSTANT_INIT_FUNCTION_NAME, CLIMO_INIT_FUNCTION_NAME
]


def _check_input_args(num_iterations, learning_rate, ideal_activation=None):
    """Error-checks input args for backwards optimization.

    :param num_iterations: See doc for `_do_gradient_descent`.
    :param learning_rate: Same.
    :param ideal_activation: See doc for `optimize_input_for_neuron_activation`
        or `optimize_input_for_channel_activation`.
    """

    error_checking.assert_is_integer(num_iterations)
    error_checking.assert_is_greater(num_iterations, 0)
    error_checking.assert_is_greater(learning_rate, 0.)
    error_checking.assert_is_less_than(learning_rate, 1.)

    if ideal_activation is not None:
        error_checking.assert_is_greater(ideal_activation, 0.)


def _do_gradient_descent(
        model_object, loss_tensor, init_function_or_matrices, num_iterations,
        learning_rate):
    """Does gradient descent (the nitty-gritty part of backwards optimization).

    :param model_object: Trained instance of `keras.models.Model` or
        `keras.models.Sequential`.
    :param loss_tensor: Keras tensor, defining the loss function to be
        minimized.
    :param init_function_or_matrices: Either a function or list of numpy arrays.

    If function, will be used to initialize input matrices.  See
    `create_gaussian_initializer` for an example.

    If list of numpy arrays, these are the input matrices themselves.  Matrices
    should be processed in the exact same way that training data were processed
    (e.g., normalization method).  Matrices must also be in the same order as
    training matrices, and the [q]th matrix in this list must have the same
    shape as the [q]th training matrix.

    :param num_iterations: Number of gradient-descent iterations (number of
        times that the input matrices are adjusted).
    :param learning_rate: Learning rate.  At each iteration, each input value x
        will be decremented by `learning_rate * gradient`, where `gradient` is
        the gradient of the loss function with respect to x.
    :return: list_of_optimized_input_matrices: length-T list of optimized input
        matrices (numpy arrays), where T = number of input tensors to the model.
        If the input arg `init_function_or_matrices` is a list of numpy arrays
        (rather than a function), `list_of_optimized_input_matrices` will have
        the exact same shape, just with different values.
    """

    if isinstance(model_object.input, list):
        list_of_input_tensors = model_object.input
    else:
        list_of_input_tensors = [model_object.input]

    num_input_tensors = len(list_of_input_tensors)
    print num_input_tensors

    list_of_gradient_tensors = K.gradients(loss_tensor, list_of_input_tensors)
    for i in range(num_input_tensors):
        list_of_gradient_tensors[i] /= K.maximum(
            K.sqrt(K.mean(list_of_gradient_tensors[i] ** 2)),
            K.epsilon()
        )

    inputs_to_loss_and_gradients = K.function(
        list_of_input_tensors + [K.learning_phase()],
        ([loss_tensor] + list_of_gradient_tensors)
    )

    if isinstance(init_function_or_matrices, list):
        list_of_optimized_input_matrices = copy.deepcopy(
            init_function_or_matrices)
    else:
        list_of_optimized_input_matrices = [None] * num_input_tensors

        for i in range(num_input_tensors):
            these_dimensions = numpy.array(
                [1] + list_of_input_tensors[i].get_shape().as_list()[1:],
                dtype=int)

            list_of_optimized_input_matrices[i] = init_function_or_matrices(
                these_dimensions)

    for j in range(num_iterations):
        these_outputs = inputs_to_loss_and_gradients(
            list_of_optimized_input_matrices + [0])

        if numpy.mod(j, 100) == 0:
            print 'Loss after {0:d} of {1:d} iterations: {2:.2e}'.format(
                j, num_iterations, these_outputs[0])

        for i in range(num_input_tensors):
            list_of_optimized_input_matrices[i] -= (
                these_outputs[i + 1] * learning_rate)

    print 'Loss after {0:d} iterations: {1:.2e}'.format(
        num_iterations, these_outputs[0])
    print len(list_of_optimized_input_matrices)
    return list_of_optimized_input_matrices


def check_init_function(init_function_name):
    """Error-checks initialization function.

    :param init_function_name: Name of initialization function.
    :raises: ValueError: if
        `init_function_name not in VALID_INIT_FUNCTION_NAMES`.
    """

    error_checking.assert_is_string(init_function_name)

    if init_function_name not in VALID_INIT_FUNCTION_NAMES:
        error_string = (
            '\n{0:s}\nValid init functions (listed above) do not include '
            '"{1:s}".'
        ).format(str(VALID_INIT_FUNCTION_NAMES), init_function_name)

        raise ValueError(error_string)


def create_gaussian_initializer(mean, standard_deviation):
    """Creates Gaussian initializer.

    :param mean: Mean of distribution (used to fill initialized model-input
        matrix).
    :param standard_deviation: Standard deviation of distribution.
    :return: init_function: Function (see below).
    """

    def init_function(matrix_dimensions):
        """Initializes model input to Gaussian distribution.

        :param matrix_dimensions: numpy array with desired dimensions
            for initialized model input.
        :return: initial_matrix: numpy array with the given dimensions.
        """

        return numpy.random.normal(
            loc=mean, scale=standard_deviation, size=matrix_dimensions)

    return init_function


def create_uniform_random_initializer(min_value, max_value):
    """Creates uniform-random initializer.

    :param min_value: Minimum value in distribution (used to fill initialized
        model-input matrix).
    :param max_value: Max value in distribution.
    :return: init_function: Function (see below).
    """

    def init_function(matrix_dimensions):
        """Initializes model input to uniform random distribution.

        :param matrix_dimensions: numpy array with desired dimensions
            for initialized model input.
        :return: initial_matrix: numpy array with the given dimensions.
        """

        return numpy.random.uniform(
            low=min_value, high=max_value, size=matrix_dimensions)

    return init_function


def create_constant_initializer(constant_value):
    """Creates constant initializer.

    :param constant_value: Constant value (repeated in initialized model-input
        matrix).
    :return: init_function: Function (see below).
    """

    def init_function(matrix_dimensions):
        """Initializes model input to constant value.

        :param matrix_dimensions: numpy array with desired dimensions
            for initialized model input.
        :return: initial_matrix: numpy array with the given dimensions.
        """

        return numpy.full(matrix_dimensions, constant_value, dtype=float)

    return init_function


def create_climo_initializer(
        training_option_dict, myrorss_2d3d, test_mode=False,
        radar_normalization_table=None, sounding_normalization_table=None):
    """Creates climatological initializer.

    :param training_option_dict: See doc for
        `training_validation_io.example_generator_2d_or_3d` or
        `training_validation_io.example_generator_2d3d_myrorss`.
    :param myrorss_2d3d: Boolean flag.  If True, this method will assume that
        model input contains 2-D azimuthal-shear images and 3-D reflectivity
        images.  In other words, this method will treat `training_option_dict`
        in the same way that
        `training_validation_io.example_generator_2d3d_myrorss` does.  If False,
        will treat `training_option_dict` in the same way that
        `training_validation_io.example_generator_2d_or_3d` does.
    :param test_mode: Never mind.  Leave this alone.
    :param radar_normalization_table: Never mind.  Leave this alone.
    :param sounding_normalization_table: Never mind.  Leave this alone.
    :return: init_function: Function (see below).
    """

    error_checking.assert_is_boolean(myrorss_2d3d)
    error_checking.assert_is_boolean(test_mode)

    if not test_mode:
        _, radar_normalization_table, _, sounding_normalization_table = (
            dl_utils.read_normalization_params_from_file(
                training_option_dict[trainval_io.NORMALIZATION_FILE_KEY]
            )
        )

    def init_function(matrix_dimensions):
        """Initializes model input to climatological means.

        This function uses one mean for each radar field/height pair and each
        sounding field/height pair, rather than one per field altogether, to
        create "realistic" vertical profiles.

        If len(matrix_dimensions) = 3, this function creates initial soundings.

        If len(matrix_dimensions) = 4 and myrorss_2d3d = False, creates initial
        2-D radar images with all fields in `training_option_dict`.

        If len(matrix_dimensions) = 4 and myrorss_2d3d = True, creates initial
        2-D radar images with only azimuthal-shear fields in
        `training_option_dict`.

        If len(matrix_dimensions) = 5 and myrorss_2d3d = False, creates initial
        3-D radar images with all fields in `training_option_dict`.

        If len(matrix_dimensions) = 5 and myrorss_2d3d = True, creates initial
        3-D reflectivity images.

        :param matrix_dimensions: numpy array with desired dimensions
            for initialized model input.
        :return: initial_matrix: numpy array with the given dimensions.
        """

        initial_matrix = numpy.full(matrix_dimensions, numpy.nan)

        if len(matrix_dimensions) == 5:
            if myrorss_2d3d:
                radar_field_names = [radar_utils.REFL_NAME]
            else:
                radar_field_names = training_option_dict[
                    trainval_io.RADAR_FIELDS_KEY]

            radar_heights_m_agl = training_option_dict[
                trainval_io.RADAR_HEIGHTS_KEY]

            for j in range(len(radar_field_names)):
                for k in range(len(radar_heights_m_agl)):
                    this_key = (radar_field_names[j], radar_heights_m_agl[k])

                    initial_matrix[..., k, j] = radar_normalization_table[
                        dl_utils.MEAN_VALUE_COLUMN].loc[[this_key]].values[0]

            return dl_utils.normalize_radar_images(
                radar_image_matrix=initial_matrix,
                field_names=radar_field_names,
                normalization_type_string=training_option_dict[
                    trainval_io.NORMALIZATION_TYPE_KEY],
                normalization_param_file_name=training_option_dict[
                    trainval_io.NORMALIZATION_FILE_KEY],
                test_mode=test_mode,
                min_normalized_value=training_option_dict[
                    trainval_io.MIN_NORMALIZED_VALUE_KEY],
                max_normalized_value=training_option_dict[
                    trainval_io.MAX_NORMALIZED_VALUE_KEY],
                normalization_table=radar_normalization_table)

        if len(matrix_dimensions) == 4:
            radar_field_names = training_option_dict[
                trainval_io.RADAR_FIELDS_KEY]
            radar_heights_m_agl = training_option_dict[
                trainval_io.RADAR_HEIGHTS_KEY]

            for j in range(len(radar_field_names)):
                this_key = (radar_field_names[j], radar_heights_m_agl[j])

                initial_matrix[..., j] = radar_normalization_table[
                    dl_utils.MEAN_VALUE_COLUMN].loc[[this_key]].values[0]

            return dl_utils.normalize_radar_images(
                radar_image_matrix=initial_matrix,
                field_names=radar_field_names,
                normalization_type_string=training_option_dict[
                    trainval_io.NORMALIZATION_TYPE_KEY],
                normalization_param_file_name=training_option_dict[
                    trainval_io.NORMALIZATION_FILE_KEY],
                test_mode=test_mode,
                min_normalized_value=training_option_dict[
                    trainval_io.MIN_NORMALIZED_VALUE_KEY],
                max_normalized_value=training_option_dict[
                    trainval_io.MAX_NORMALIZED_VALUE_KEY],
                normalization_table=radar_normalization_table)

        if len(matrix_dimensions) == 3:
            sounding_field_names = training_option_dict[
                trainval_io.SOUNDING_FIELDS_KEY]
            sounding_heights_m_agl = training_option_dict[
                trainval_io.SOUNDING_HEIGHTS_KEY]

            for j in range(len(sounding_field_names)):
                for k in range(len(sounding_heights_m_agl)):
                    this_key = (
                        sounding_field_names[j], sounding_heights_m_agl[k]
                    )

                    initial_matrix[..., k, j] = sounding_normalization_table[
                        dl_utils.MEAN_VALUE_COLUMN].loc[[this_key]].values[0]

            return dl_utils.normalize_soundings(
                sounding_matrix=initial_matrix,
                field_names=sounding_field_names,
                normalization_type_string=training_option_dict[
                    trainval_io.NORMALIZATION_TYPE_KEY],
                normalization_param_file_name=training_option_dict[
                    trainval_io.NORMALIZATION_FILE_KEY],
                test_mode=test_mode,
                min_normalized_value=training_option_dict[
                    trainval_io.MIN_NORMALIZED_VALUE_KEY],
                max_normalized_value=training_option_dict[
                    trainval_io.MAX_NORMALIZED_VALUE_KEY],
                normalization_table=sounding_normalization_table)

        return None

    return init_function


def optimize_input_for_class(
        model_object, target_class, init_function_or_matrices,
        num_iterations=DEFAULT_NUM_ITERATIONS,
        learning_rate=DEFAULT_LEARNING_RATE):
    """Creates synthetic input example to maximize probability of target class.

    :param model_object: Trained instance of `keras.models.Model` or
        `keras.models.Sequential`.
    :param target_class: Input data will be optimized for this class.  Must be
        an integer in 0...(K - 1), where K = number of classes.
    :param init_function_or_matrices: See doc for `_do_gradient_descent`.
    :param num_iterations: Same.
    :param learning_rate: Same.
    :return: list_of_optimized_input_matrices: Same.
    """

    model_interpretation.check_component_metadata(
        component_type_string=
        model_interpretation.CLASS_COMPONENT_TYPE_STRING,
        target_class=target_class)

    _check_input_args(
        num_iterations=num_iterations, learning_rate=learning_rate)

    num_output_neurons = (
        model_object.layers[-1].output.get_shape().as_list()[-1]
    )

    if num_output_neurons == 1:
        error_checking.assert_is_leq(target_class, 1)

        if target_class == 1:
            loss_tensor = K.mean(
                (model_object.layers[-1].output[..., 0] - 1) ** 2
            )
        else:
            loss_tensor = K.mean(
                model_object.layers[-1].output[..., 0] ** 2
            )
    else:
        error_checking.assert_is_less_than(target_class, num_output_neurons)

        loss_tensor = K.mean(
            (model_object.layers[-1].output[..., target_class] - 1) ** 2
        )

    return _do_gradient_descent(
        model_object=model_object, loss_tensor=loss_tensor,
        init_function_or_matrices=init_function_or_matrices,
        num_iterations=num_iterations, learning_rate=learning_rate)


def optimize_input_for_neuron(
        model_object, layer_name, neuron_indices, init_function_or_matrices,
        num_iterations=DEFAULT_NUM_ITERATIONS,
        learning_rate=DEFAULT_LEARNING_RATE,
        ideal_activation=DEFAULT_IDEAL_ACTIVATION):
    """Creates synthetic input example to maximize activation of neuron.

    :param model_object: Trained instance of `keras.models.Model` or
        `keras.models.Sequential`.
    :param layer_name: Name of layer containing the relevant neuron.
    :param neuron_indices: 1-D numpy array with indices of the relevant neuron.
        Must have length D - 1, where D = number of dimensions in layer output.
        The first dimension of layer output is the example dimension, for which
        the index in this case is always 0.
    :param init_function_or_matrices: See doc for `_do_gradient_descent`.
    :param num_iterations: Same.
    :param learning_rate: Same.
    :param ideal_activation: If this value is specified, the loss function will
        be (neuron_activation - ideal_activation)^2.

        If this value is None, the loss function will be
        -sign(neuron_activation) * neuron_activation^2.

    :return: list_of_optimized_input_matrices: See doc for
        `_do_gradient_descent`.
    """

    model_interpretation.check_component_metadata(
        component_type_string=
        model_interpretation.NEURON_COMPONENT_TYPE_STRING,
        layer_name=layer_name, neuron_indices=neuron_indices)

    _check_input_args(
        num_iterations=num_iterations, learning_rate=learning_rate,
        ideal_activation=ideal_activation)

    neuron_indices_as_tuple = (0,) + tuple(neuron_indices)

    if ideal_activation is None:
        loss_tensor = -(
            K.sign(
                model_object.get_layer(name=layer_name).output[
                    neuron_indices_as_tuple]
            ) *
            model_object.get_layer(name=layer_name).output[
                neuron_indices_as_tuple] ** 2
        )
    else:
        loss_tensor = (
            model_object.get_layer(name=layer_name).output[
                neuron_indices_as_tuple] -
            ideal_activation
        ) ** 2

    return _do_gradient_descent(
        model_object=model_object, loss_tensor=loss_tensor,
        init_function_or_matrices=init_function_or_matrices,
        num_iterations=num_iterations, learning_rate=learning_rate)


def optimize_input_for_channel(
        model_object, layer_name, channel_index, init_function_or_matrices,
        stat_function_for_neuron_activations,
        num_iterations=DEFAULT_NUM_ITERATIONS,
        learning_rate=DEFAULT_LEARNING_RATE,
        ideal_activation=DEFAULT_IDEAL_ACTIVATION):
    """Creates synthetic input example to maxx activation of neurons in channel.

    :param model_object: Trained instance of `keras.models.Model` or
        `keras.models.Sequential`.
    :param layer_name: Name of layer containing the relevant channel.
    :param channel_index: Index of the relevant channel.  Will optimize for
        activation of [j]th channel in layer, where j = `channel_index`.
    :param init_function_or_matrices: See doc for `_do_gradient_descent`.
    :param stat_function_for_neuron_activations: Function used to convert all
        neuron activations into a single number.  Some examples are
        `keras.backend.max` and `keras.backend.mean`.  The exact format of this
        function is given below.

        Input: Keras tensor of neuron activations.
        Output: Single number.

    :param num_iterations: See doc for `_do_gradient_descent`.
    :param learning_rate: Same.
    :param ideal_activation: If this value is specified, the loss function will
        be abs[stat_function_for_neuron_activations(neuron_activations) -
               ideal_activation].

    If this value is None, loss function will be
    -abs[stat_function_for_neuron_activations(neuron_activations)].

    :return: list_of_optimized_input_matrices: See doc for
        `_do_gradient_descent`.
    """

    model_interpretation.check_component_metadata(
        component_type_string=
        model_interpretation.CHANNEL_COMPONENT_TYPE_STRING,
        layer_name=layer_name, channel_index=channel_index)

    _check_input_args(
        num_iterations=num_iterations, learning_rate=learning_rate,
        ideal_activation=ideal_activation)

    if ideal_activation is None:
        loss_tensor = -K.abs(stat_function_for_neuron_activations(
            model_object.get_layer(name=layer_name).output[
                0, ..., channel_index]
        ))
    else:
        error_checking.assert_is_greater(ideal_activation, 0.)
        loss_tensor = K.abs(
            stat_function_for_neuron_activations(
                model_object.get_layer(name=layer_name).output[
                    0, ..., channel_index]
            ) - ideal_activation
        )

    return _do_gradient_descent(
        model_object=model_object, loss_tensor=loss_tensor,
        init_function_or_matrices=init_function_or_matrices,
        num_iterations=num_iterations, learning_rate=learning_rate)


def write_results(
        pickle_file_name, list_of_optimized_input_matrices, model_file_name,
        init_function_name_or_matrices, num_iterations, learning_rate,
        component_type_string, target_class=None, layer_name=None,
        neuron_indices=None, channel_index=None, ideal_activation=None):
    """Writes results of backwards optimization to Pickle file.

    :param pickle_file_name: Path to output file.
    :param list_of_optimized_input_matrices: Optimized input data (see doc for
        `_do_gradient_descent`).
    :param model_file_name: Path to file with trained model.
    :param init_function_name_or_matrices: See doc for `_do_gradient_descent`.
        The only difference here is that the variable must be a function *name*
        (rather than the function itself) or list of numpy arrays.
    :param num_iterations: See doc for `_do_gradient_descent`.
    :param learning_rate: Same.
    :param component_type_string: See doc for
        `model_interpretation.check_component_metadata`.
    :param target_class: Same.
    :param layer_name: Same.
    :param neuron_indices: Same.
    :param channel_index: Same.
    :param ideal_activation: See doc for `optimize_input_for_neuron_activation`
        or `optimize_input_for_channel_activation`.
    :raises: ValueError: if `init_function_name_or_matrices` is a list of numpy
        arrays and has a different length than
        `list_of_optimized_input_matrices`.
    """

    model_interpretation.check_component_metadata(
        component_type_string=component_type_string,
        target_class=target_class, layer_name=layer_name,
        neuron_indices=neuron_indices, channel_index=channel_index)

    _check_input_args(
        num_iterations=num_iterations, learning_rate=learning_rate,
        ideal_activation=ideal_activation)

    error_checking.assert_is_string(model_file_name)
    error_checking.assert_is_list(list_of_optimized_input_matrices)

    if not isinstance(init_function_name_or_matrices, str):
        num_init_matrices = len(init_function_name_or_matrices)
        num_optimized_matrices = len(list_of_optimized_input_matrices)

        if num_init_matrices != num_optimized_matrices:
            error_string = (
                'Number of initial matrices ({0:d}) should equal number of '
                'optimized matrices ({1:d}).'
            ).format(num_init_matrices, num_optimized_matrices)

            raise ValueError(error_string)

    num_matrices = len(list_of_optimized_input_matrices)

    for i in range(num_matrices):
        error_checking.assert_is_numpy_array(
            list_of_optimized_input_matrices[i])

        if not isinstance(init_function_name_or_matrices, str):
            error_checking.assert_is_numpy_array(
                init_function_name_or_matrices[i],
                exact_dimensions=numpy.array(
                    list_of_optimized_input_matrices[i].shape)
            )

    metadata_dict = {
        MODEL_FILE_NAME_KEY: model_file_name,
        NUM_ITERATIONS_KEY: num_iterations,
        LEARNING_RATE_KEY: learning_rate,
        COMPONENT_TYPE_KEY: component_type_string,
        INIT_FUNCTION_KEY: init_function_name_or_matrices,
        TARGET_CLASS_KEY: target_class,
        LAYER_NAME_KEY: layer_name,
        IDEAL_ACTIVATION_KEY: ideal_activation,
        NEURON_INDICES_KEY: neuron_indices,
        CHANNEL_INDEX_KEY: channel_index
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(list_of_optimized_input_matrices, pickle_file_handle)
    pickle.dump(metadata_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_results(pickle_file_name):
    """Reads results of backwards optimization from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: list_of_optimized_input_matrices: See doc for `write_file`.
    :return: metadata_dict: Dictionary with the following keys.
    metadata_dict['model_file_name']: See doc for `write_file`.
    metadata_dict['num_iterations']: Same.
    metadata_dict['learning_rate']: Same.
    metadata_dict['component_type_string']: Same.
    metadata_dict['init_function_name_or_matrices']: Same.
    metadata_dict['target_class']: Same.
    metadata_dict['layer_name']: Same.
    metadata_dict['ideal_activation']: Same.
    metadata_dict['neuron_indices']: Same.
    metadata_dict['channel_index']: Same.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    list_of_optimized_input_matrices = pickle.load(pickle_file_handle)
    metadata_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    return list_of_optimized_input_matrices, metadata_dict
