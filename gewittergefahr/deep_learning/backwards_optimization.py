"""Methods for backwards optimization (or "feature optimization").

--- REFERENCES ---

Olah, C., A. Mordvintsev, and L. Schubert, 2017: Feature visualization. Distill,
    doi:10.23915/distill.00007,
    URL https://distill.pub/2017/feature-visualization.
"""

import pickle
import numpy
from keras import backend as K
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import input_examples
from gewittergefahr.deep_learning import model_interpretation
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.deep_learning import training_validation_io as trainval_io

DEFAULT_LEARNING_RATE = 0.0025
DEFAULT_NUM_ITERATIONS = 200
DEFAULT_L2_WEIGHT = None

DEFAULT_IDEAL_ACTIVATION = 2.

INIT_FUNCTION_KEY = 'init_function_name_or_matrices'
OPTIMIZED_MATRICES_KEY = 'list_of_optimized_matrices'

FULL_IDS_KEY = tracking_io.FULL_IDS_KEY
STORM_TIMES_KEY = tracking_io.STORM_TIMES_KEY

MODEL_FILE_KEY = model_interpretation.MODEL_FILE_KEY
NUM_ITERATIONS_KEY = 'num_iterations'
LEARNING_RATE_KEY = 'learning_rate'
L2_WEIGHT_KEY = 'l2_weight'
COMPONENT_TYPE_KEY = 'component_type_string'
TARGET_CLASS_KEY = 'target_class'
LAYER_NAME_KEY = 'layer_name'
IDEAL_ACTIVATION_KEY = 'ideal_activation'
NEURON_INDICES_KEY = 'neuron_indices'
CHANNEL_INDEX_KEY = 'channel_index'

STANDARD_FILE_KEYS = [
    INIT_FUNCTION_KEY, OPTIMIZED_MATRICES_KEY, MODEL_FILE_KEY,
    NUM_ITERATIONS_KEY, LEARNING_RATE_KEY, L2_WEIGHT_KEY, COMPONENT_TYPE_KEY,
    TARGET_CLASS_KEY, LAYER_NAME_KEY, IDEAL_ACTIVATION_KEY, NEURON_INDICES_KEY,
    CHANNEL_INDEX_KEY, FULL_IDS_KEY, STORM_TIMES_KEY
]

MEAN_INPUT_MATRICES_KEY = model_interpretation.MEAN_INPUT_MATRICES_KEY
MEAN_OPTIMIZED_MATRICES_KEY = 'list_of_mean_optimized_matrices'
THRESHOLD_COUNTS_KEY = 'threshold_count_matrix'
STANDARD_FILE_NAME_KEY = 'standard_bwo_file_name'
PMM_METADATA_KEY = 'pmm_metadata_dict'

PMM_FILE_KEYS = [
    MEAN_INPUT_MATRICES_KEY, MEAN_OPTIMIZED_MATRICES_KEY, THRESHOLD_COUNTS_KEY,
    MODEL_FILE_KEY, STANDARD_FILE_NAME_KEY, PMM_METADATA_KEY
]

GAUSSIAN_INIT_FUNCTION_NAME = 'gaussian'
UNIFORM_INIT_FUNCTION_NAME = 'uniform'
CONSTANT_INIT_FUNCTION_NAME = 'constant'
CLIMO_INIT_FUNCTION_NAME = 'climo'

VALID_INIT_FUNCTION_NAMES = [
    GAUSSIAN_INIT_FUNCTION_NAME, UNIFORM_INIT_FUNCTION_NAME,
    CONSTANT_INIT_FUNCTION_NAME, CLIMO_INIT_FUNCTION_NAME
]


def _check_input_args(num_iterations, learning_rate, l2_weight=None,
                      ideal_activation=None):
    """Error-checks input args for backwards optimization.

    :param num_iterations: See doc for `_do_gradient_descent`.
    :param learning_rate: Same.
    :param l2_weight: Same.
    :param ideal_activation: See doc for `optimize_input_for_neuron_activation`
        or `optimize_input_for_channel_activation`.
    """

    error_checking.assert_is_integer(num_iterations)
    error_checking.assert_is_greater(num_iterations, 0)
    error_checking.assert_is_greater(learning_rate, 0.)
    error_checking.assert_is_less_than(learning_rate, 1.)

    if l2_weight is not None:
        error_checking.assert_is_greater(l2_weight, 0.)

    if ideal_activation is not None:
        error_checking.assert_is_greater(ideal_activation, 0.)


def _do_gradient_descent(
        model_object, loss_tensor, init_function_or_matrices, num_iterations,
        learning_rate, l2_weight=None):
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
    :param l2_weight: L2-regularization weight.  This will penalize the squared
        Euclidean distance between the original and synthetic (optimized) input
        tensors.
    :return: list_of_optimized_matrices: length-T list of optimized input
        matrices (numpy arrays), where T = number of input tensors to the model.
        If the input arg `init_function_or_matrices` is a list of numpy arrays
        (rather than a function), `list_of_optimized_matrices` will have
        the exact same shape, just with different values.
    """

    if isinstance(model_object.input, list):
        list_of_input_tensors = model_object.input
    else:
        list_of_input_tensors = [model_object.input]

    num_input_tensors = len(list_of_input_tensors)

    if isinstance(init_function_or_matrices, list):
        list_of_optimized_matrices = init_function_or_matrices + []
    else:
        list_of_optimized_matrices = [None] * num_input_tensors

        for i in range(num_input_tensors):
            these_dimensions = numpy.array(
                [1] + list_of_input_tensors[i].get_shape().as_list()[1:],
                dtype=int
            )

            list_of_optimized_matrices[i] = init_function_or_matrices(
                these_dimensions)

    if l2_weight is not None:
        for i in range(num_input_tensors):
            loss_tensor += l2_weight * K.sum(
                (list_of_input_tensors[i][0, ...] -
                 list_of_optimized_matrices[i][0, ...]) ** 2
            )

            # loss_tensor += l2_weight * K.sum(
            #     (model_object.layers[0].output[i] -
            #      list_of_optimized_matrices[i]) ** 2
            # )

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

    for j in range(num_iterations):
        these_outputs = inputs_to_loss_and_gradients(
            list_of_optimized_matrices + [0]
        )

        if numpy.mod(j, 100) == 0:
            print('Loss after {0:d} of {1:d} iterations: {2:.2e}'.format(
                j, num_iterations, these_outputs[0]
            ))

        for i in range(num_input_tensors):
            list_of_optimized_matrices[i] -= (
                these_outputs[i + 1] * learning_rate
            )

    print('Loss after {0:d} iterations: {1:.2e}'.format(
        num_iterations, these_outputs[0]
    ))

    return list_of_optimized_matrices


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
        model_metadata_dict, test_mode=False, radar_normalization_table=None,
        sounding_normalization_table=None):
    """Creates climatological initializer.

    :param model_metadata_dict: See doc for `cnn.read_model_metadata`.
    :param test_mode: Never mind.  Leave this alone.
    :param radar_normalization_table: Never mind.  Leave this alone.
    :param sounding_normalization_table: Never mind.  Leave this alone.
    :return: init_function: Function (see below).
    """

    error_checking.assert_is_boolean(test_mode)
    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]

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
            if model_metadata_dict[cnn.USE_2D3D_CONVOLUTION_KEY]:
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
            list_of_layer_operation_dicts = model_metadata_dict[
                cnn.LAYER_OPERATIONS_KEY]

            if list_of_layer_operation_dicts is None:
                radar_field_names = training_option_dict[
                    trainval_io.RADAR_FIELDS_KEY]
                radar_heights_m_agl = training_option_dict[
                    trainval_io.RADAR_HEIGHTS_KEY]

                for j in range(len(radar_field_names)):
                    this_key = (radar_field_names[j], radar_heights_m_agl[j])
                    initial_matrix[..., j] = radar_normalization_table[
                        dl_utils.MEAN_VALUE_COLUMN].loc[[this_key]].values[0]

            else:
                radar_field_names = [
                    d[input_examples.RADAR_FIELD_KEY]
                    for d in list_of_layer_operation_dicts
                ]
                min_heights_m_agl = numpy.array([
                    d[input_examples.MIN_HEIGHT_KEY]
                    for d in list_of_layer_operation_dicts
                ], dtype=int)
                max_heights_m_agl = numpy.array([
                    d[input_examples.MAX_HEIGHT_KEY]
                    for d in list_of_layer_operation_dicts
                ], dtype=int)

                for j in range(len(radar_field_names)):
                    this_key = (radar_field_names[j], min_heights_m_agl[j])
                    this_first_mean = radar_normalization_table[
                        dl_utils.MEAN_VALUE_COLUMN].loc[[this_key]].values[0]

                    this_key = (radar_field_names[j], max_heights_m_agl[j])
                    this_second_mean = radar_normalization_table[
                        dl_utils.MEAN_VALUE_COLUMN].loc[[this_key]].values[0]

                    initial_matrix[..., j] = numpy.mean(
                        [this_first_mean, this_second_mean])

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
        learning_rate=DEFAULT_LEARNING_RATE, l2_weight=DEFAULT_L2_WEIGHT):
    """Creates synthetic input example to maximize probability of target class.

    :param model_object: Trained instance of `keras.models.Model` or
        `keras.models.Sequential`.
    :param target_class: Input data will be optimized for this class.  Must be
        an integer in 0...(K - 1), where K = number of classes.
    :param init_function_or_matrices: See doc for `_do_gradient_descent`.
    :param num_iterations: Same.
    :param learning_rate: Same.
    :param l2_weight: Same.
    :return: list_of_optimized_matrices: Same.
    """

    model_interpretation.check_component_metadata(
        component_type_string=model_interpretation.CLASS_COMPONENT_TYPE_STRING,
        target_class=target_class)

    _check_input_args(
        num_iterations=num_iterations, learning_rate=learning_rate,
        l2_weight=l2_weight)

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
        num_iterations=num_iterations, learning_rate=learning_rate,
        l2_weight=l2_weight)


def optimize_input_for_neuron(
        model_object, layer_name, neuron_indices, init_function_or_matrices,
        num_iterations=DEFAULT_NUM_ITERATIONS,
        learning_rate=DEFAULT_LEARNING_RATE, l2_weight=DEFAULT_L2_WEIGHT,
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
    :param l2_weight: Same.
    :param ideal_activation: If this value is specified, the loss function will
        be (neuron_activation - ideal_activation)^2.

        If this value is None, the loss function will be
        -sign(neuron_activation) * neuron_activation^2.

    :return: list_of_optimized_matrices: See doc for `_do_gradient_descent`.
    """

    model_interpretation.check_component_metadata(
        component_type_string=model_interpretation.NEURON_COMPONENT_TYPE_STRING,
        layer_name=layer_name, neuron_indices=neuron_indices)

    _check_input_args(
        num_iterations=num_iterations, learning_rate=learning_rate,
        l2_weight=l2_weight, ideal_activation=ideal_activation)

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
        num_iterations=num_iterations, learning_rate=learning_rate,
        l2_weight=l2_weight)


def optimize_input_for_channel(
        model_object, layer_name, channel_index, init_function_or_matrices,
        stat_function_for_neuron_activations,
        num_iterations=DEFAULT_NUM_ITERATIONS,
        learning_rate=DEFAULT_LEARNING_RATE, l2_weight=DEFAULT_L2_WEIGHT,
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
    :param l2_weight: Same.
    :param ideal_activation: If this value is specified, the loss function will
        be abs[stat_function_for_neuron_activations(neuron_activations) -
               ideal_activation].

    If this value is None, loss function will be
    -abs[stat_function_for_neuron_activations(neuron_activations)].

    :return: list_of_optimized_matrices: See doc for `_do_gradient_descent`.
    """

    model_interpretation.check_component_metadata(
        component_type_string=
        model_interpretation.CHANNEL_COMPONENT_TYPE_STRING,
        layer_name=layer_name, channel_index=channel_index
    )

    _check_input_args(
        num_iterations=num_iterations, learning_rate=learning_rate,
        l2_weight=l2_weight, ideal_activation=ideal_activation)

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
        num_iterations=num_iterations, learning_rate=learning_rate,
        l2_weight=l2_weight)


def write_standard_file(
        pickle_file_name, init_function_name_or_matrices,
        list_of_optimized_matrices, model_file_name, num_iterations,
        learning_rate, component_type_string, l2_weight=None, target_class=None,
        layer_name=None, neuron_indices=None, channel_index=None,
        ideal_activation=None, full_id_strings=None, storm_times_unix_sec=None):
    """Writes optimized learning examples to Pickle file.

    E = number of examples (storm objects)

    :param pickle_file_name: Path to output file.
    :param init_function_name_or_matrices: See doc for `_do_gradient_descent`.
        The only difference here is that, if a function was used, the input
        argument must be the function *name* rather than the function itself.
    :param list_of_optimized_matrices: List of numpy arrays created by
        `_do_gradient_descent`.
    :param model_file_name: Path to file with trained model (readable by
        `cnn.read_model`).
    :param num_iterations: See doc for `_do_gradient_descent`.
    :param learning_rate: Same.
    :param component_type_string: See doc for
        `model_interpretation.check_component_metadata`.
    :param l2_weight: See doc for `_do_gradient_descent`.
    :param target_class: See doc for
        `model_interpretation.check_component_metadata`.
    :param layer_name: Same.
    :param neuron_indices: Same.
    :param channel_index: Same.
    :param ideal_activation: See doc for `optimize_input_for_neuron` or
        `optimize_input_for_channel`.
    :param full_id_strings:
        [used only if `init_function_name_or_matrices` is list of matrices]
        length-E list of full storm IDs.
    :param storm_times_unix_sec:
        [used only if `init_function_name_or_matrices` is list of matrices]
        length-E numpy array of storm times.
    :raises: ValueError: if `init_function_name_or_matrices` is a list of numpy
        arrays and has a different length than `list_of_optimized_matrices`.
    """

    model_interpretation.check_component_metadata(
        component_type_string=component_type_string,
        target_class=target_class, layer_name=layer_name,
        neuron_indices=neuron_indices, channel_index=channel_index)

    _check_input_args(
        num_iterations=num_iterations, learning_rate=learning_rate,
        l2_weight=l2_weight, ideal_activation=ideal_activation)

    error_checking.assert_is_string(model_file_name)
    error_checking.assert_is_list(list_of_optimized_matrices)

    if isinstance(init_function_name_or_matrices, str):
        num_storm_objects = None
    else:
        num_init_matrices = len(init_function_name_or_matrices)
        num_optimized_matrices = len(list_of_optimized_matrices)

        if num_init_matrices != num_optimized_matrices:
            error_string = (
                'Number of input matrices ({0:d}) should equal number of output'
                ' matrices ({1:d}).'
            ).format(num_init_matrices, num_optimized_matrices)

            raise ValueError(error_string)

        error_checking.assert_is_string_list(full_id_strings)
        error_checking.assert_is_numpy_array(
            numpy.array(full_id_strings), num_dimensions=1)

        num_storm_objects = len(full_id_strings)
        these_expected_dim = numpy.array([num_storm_objects], dtype=int)

        error_checking.assert_is_integer_numpy_array(storm_times_unix_sec)
        error_checking.assert_is_numpy_array(
            storm_times_unix_sec, exact_dimensions=these_expected_dim)

    num_matrices = len(list_of_optimized_matrices)

    for i in range(num_matrices):
        error_checking.assert_is_numpy_array_without_nan(
            list_of_optimized_matrices[i])

        if num_storm_objects is not None:
            these_expected_dim = numpy.array(
                (num_storm_objects,) + list_of_optimized_matrices[i].shape[1:],
                dtype=int)
            error_checking.assert_is_numpy_array(
                list_of_optimized_matrices[i],
                exact_dimensions=these_expected_dim)

        if not isinstance(init_function_name_or_matrices, str):
            error_checking.assert_is_numpy_array_without_nan(
                init_function_name_or_matrices[i])

            these_expected_dim = numpy.array(
                list_of_optimized_matrices[i].shape, dtype=int)

            error_checking.assert_is_numpy_array(
                init_function_name_or_matrices[i],
                exact_dimensions=these_expected_dim)

    optimization_dict = {
        INIT_FUNCTION_KEY: init_function_name_or_matrices,
        OPTIMIZED_MATRICES_KEY: list_of_optimized_matrices,
        MODEL_FILE_KEY: model_file_name,
        NUM_ITERATIONS_KEY: num_iterations,
        LEARNING_RATE_KEY: learning_rate,
        L2_WEIGHT_KEY: l2_weight,
        COMPONENT_TYPE_KEY: component_type_string,
        TARGET_CLASS_KEY: target_class,
        LAYER_NAME_KEY: layer_name,
        IDEAL_ACTIVATION_KEY: ideal_activation,
        NEURON_INDICES_KEY: neuron_indices,
        CHANNEL_INDEX_KEY: channel_index,
        FULL_IDS_KEY: full_id_strings,
        STORM_TIMES_KEY: storm_times_unix_sec
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(optimization_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_standard_file(pickle_file_name):
    """Reads optimized learning examples from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: optimization_dict: Dictionary with the following keys.
    optimization_dict['init_function_name_or_matrices']: See doc for
        `write_standard_file`.
    optimization_dict['list_of_optimized_matrices']: Same.
    optimization_dict['model_file_name']: Same.
    optimization_dict['num_iterations']: Same.
    optimization_dict['learning_rate']: Same.
    optimization_dict['l2_weight']: Same.
    optimization_dict['component_type_string']: Same.
    optimization_dict['target_class']: Same.
    optimization_dict['layer_name']: Same.
    optimization_dict['ideal_activation']: Same.
    optimization_dict['neuron_indices']: Same.
    optimization_dict['channel_index']: Same.
    optimization_dict['full_id_strings']: Same.
    optimization_dict['storm_times_unix_sec']: Same.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    optimization_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    missing_keys = list(set(STANDARD_FILE_KEYS) - set(optimization_dict.keys()))
    if len(missing_keys) == 0:
        return optimization_dict

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), pickle_file_name)

    raise ValueError(error_string)


def write_pmm_file(
        pickle_file_name, list_of_mean_input_matrices,
        list_of_mean_optimized_matrices, threshold_count_matrix,
        model_file_name, standard_bwo_file_name, pmm_metadata_dict):
    """Writes mean backwards-optimized map to Pickle file.

    This is a mean over many examples, created by PMM (probability-matched
    means).

    T = number of input tensors to the model

    :param pickle_file_name: Path to output file.
    :param list_of_mean_input_matrices: length-T list of numpy arrays, where
        list_of_mean_input_matrices[i] is the mean (over many examples) of the
        [i]th input tensor to the model.  list_of_mean_input_matrices[i] should
        have the same dimensions as the [i]th input tensor, except without the
        first axis.
    :param list_of_mean_optimized_matrices: Same as
        `list_of_mean_input_matrices` (and with the same dimensions), but for
        optimized learning examples.  In other words,
        `list_of_mean_input_matrices` contains the mean input and
        `list_of_mean_optimized_matrices` contains the mean output.
    :param threshold_count_matrix: See doc for
        `prob_matched_means.run_pmm_many_variables`.
    :param model_file_name: Path to file with trained model (readable by
        `cnn.read_model`).
    :param standard_bwo_file_name: Path to file with standard
        backwards-optimization output (readable by `read_standard_file`).
    :param pmm_metadata_dict: Dictionary created by
        `prob_matched_means.check_input_args`.
    :raises: ValueError: if `list_of_mean_input_matrices` and
        `list_of_mean_optimized_matrices` have different lengths.
    """

    error_checking.assert_is_string(model_file_name)
    error_checking.assert_is_string(standard_bwo_file_name)
    error_checking.assert_is_list(list_of_mean_input_matrices)
    error_checking.assert_is_list(list_of_mean_optimized_matrices)

    num_input_matrices = len(list_of_mean_input_matrices)
    num_output_matrices = len(list_of_mean_optimized_matrices)

    if num_input_matrices != num_output_matrices:
        error_string = (
            'Number of input matrices ({0:d}) should equal number of output '
            'matrices ({1:d}).'
        ).format(num_input_matrices, num_output_matrices)

        raise ValueError(error_string)

    for i in range(num_input_matrices):
        error_checking.assert_is_numpy_array_without_nan(
            list_of_mean_input_matrices[i])
        error_checking.assert_is_numpy_array_without_nan(
            list_of_mean_optimized_matrices[i])

        these_expected_dim = numpy.array(
            list_of_mean_input_matrices[i].shape, dtype=int)
        error_checking.assert_is_numpy_array(
            list_of_mean_optimized_matrices[i],
            exact_dimensions=these_expected_dim)

    if threshold_count_matrix is not None:
        error_checking.assert_is_integer_numpy_array(threshold_count_matrix)
        error_checking.assert_is_geq_numpy_array(threshold_count_matrix, 0)

        spatial_dimensions = numpy.array(
            list_of_mean_input_matrices[0].shape[:-1], dtype=int)
        error_checking.assert_is_numpy_array(
            threshold_count_matrix, exact_dimensions=spatial_dimensions)

    mean_optimization_dict = {
        MEAN_INPUT_MATRICES_KEY: list_of_mean_input_matrices,
        MEAN_OPTIMIZED_MATRICES_KEY: list_of_mean_optimized_matrices,
        THRESHOLD_COUNTS_KEY: threshold_count_matrix,
        MODEL_FILE_KEY: model_file_name,
        STANDARD_FILE_NAME_KEY: standard_bwo_file_name,
        PMM_METADATA_KEY: pmm_metadata_dict
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(mean_optimization_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_pmm_file(pickle_file_name):
    """Reads mean backwards-optimized map from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: mean_optimization_dict: Dictionary with the following keys.
    mean_optimization_dict['list_of_mean_input_matrices']: See doc for
        `write_pmm_file`.
    mean_optimization_dict['list_of_mean_optimized_matrices']: Same.
    mean_optimization_dict['threshold_count_matrix']: Same.
    mean_optimization_dict['model_file_name']: Same.
    mean_optimization_dict['standard_bwo_file_name']: Same.
    mean_optimization_dict['pmm_metadata_dict']: Same.

    :raises: ValueError: if any of the aforelisted keys are missing from the
        dictionary.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    mean_optimization_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    missing_keys = list(set(PMM_FILE_KEYS) - set(mean_optimization_dict.keys()))
    if len(missing_keys) == 0:
        return mean_optimization_dict

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), pickle_file_name)

    raise ValueError(error_string)
