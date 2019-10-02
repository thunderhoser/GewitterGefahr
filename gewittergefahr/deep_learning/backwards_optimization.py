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
from gewittergefahr.gg_utils import physical_constraints
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import input_examples
from gewittergefahr.deep_learning import model_interpretation
from gewittergefahr.deep_learning import saliency_maps
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.deep_learning import training_validation_io as trainval_io

TOLERANCE = 1e-6

DEFAULT_LEARNING_RATE = 0.001
DEFAULT_NUM_ITERATIONS = 200
DEFAULT_L2_WEIGHT = 1.2533
DEFAULT_IDEAL_ACTIVATION = 2.

NORM_INPUT_MATRICES_KEY = 'normalized_input_matrices'
NORM_OUTPUT_MATRICES_KEY = 'normalized_output_matrices'
INITIAL_ACTIVATION_KEY = 'initial_activation'
FINAL_ACTIVATION_KEY = 'final_activation'

INPUT_MATRICES_KEY = 'denorm_input_matrices'
OUTPUT_MATRICES_KEY = 'denorm_output_matrices'
INITIAL_ACTIVATIONS_KEY = 'initial_activations'
FINAL_ACTIVATIONS_KEY = 'final_activations'
MODEL_FILE_KEY = model_interpretation.MODEL_FILE_KEY
FULL_STORM_IDS_KEY = model_interpretation.FULL_STORM_IDS_KEY
STORM_TIMES_KEY = model_interpretation.STORM_TIMES_KEY
SOUNDING_PRESSURES_KEY = model_interpretation.SOUNDING_PRESSURES_KEY

NUM_ITERATIONS_KEY = 'num_iterations'
LEARNING_RATE_KEY = 'learning_rate'
L2_WEIGHT_KEY = 'l2_weight'
RADAR_CONSTRAINT_WEIGHT_KEY = 'radar_constraint_weight'
MINMAX_CONSTRAINT_WEIGHT_KEY = 'minmax_constraint_weight'
COMPONENT_TYPE_KEY = saliency_maps.COMPONENT_TYPE_KEY
TARGET_CLASS_KEY = saliency_maps.TARGET_CLASS_KEY
LAYER_NAME_KEY = saliency_maps.LAYER_NAME_KEY
IDEAL_ACTIVATION_KEY = saliency_maps.IDEAL_ACTIVATION_KEY
NEURON_INDICES_KEY = saliency_maps.NEURON_INDICES_KEY
CHANNEL_INDEX_KEY = saliency_maps.CHANNEL_INDEX_KEY

STANDARD_FILE_KEYS = [
    INPUT_MATRICES_KEY, OUTPUT_MATRICES_KEY,
    INITIAL_ACTIVATIONS_KEY, FINAL_ACTIVATIONS_KEY,
    MODEL_FILE_KEY, FULL_STORM_IDS_KEY, STORM_TIMES_KEY, SOUNDING_PRESSURES_KEY,
    NUM_ITERATIONS_KEY, LEARNING_RATE_KEY, L2_WEIGHT_KEY,
    RADAR_CONSTRAINT_WEIGHT_KEY, MINMAX_CONSTRAINT_WEIGHT_KEY,
    COMPONENT_TYPE_KEY, TARGET_CLASS_KEY, LAYER_NAME_KEY, IDEAL_ACTIVATION_KEY,
    NEURON_INDICES_KEY, CHANNEL_INDEX_KEY
]

MEAN_INPUT_MATRICES_KEY = 'mean_denorm_input_matrices'
MEAN_OUTPUT_MATRICES_KEY = 'mean_denorm_output_matrices'
MEAN_INITIAL_ACTIVATION_KEY = 'mean_initial_activation'
MEAN_FINAL_ACTIVATION_KEY = 'mean_final_activation'
NON_PMM_FILE_KEY = model_interpretation.NON_PMM_FILE_KEY
PMM_MAX_PERCENTILE_KEY = model_interpretation.PMM_MAX_PERCENTILE_KEY
MEAN_SOUNDING_PRESSURES_KEY = model_interpretation.MEAN_SOUNDING_PRESSURES_KEY

PMM_FILE_KEYS = [
    MEAN_INPUT_MATRICES_KEY, MEAN_OUTPUT_MATRICES_KEY,
    MEAN_INITIAL_ACTIVATION_KEY, MEAN_FINAL_ACTIVATION_KEY,
    MODEL_FILE_KEY, NON_PMM_FILE_KEY, PMM_MAX_PERCENTILE_KEY,
    MEAN_SOUNDING_PRESSURES_KEY
]

GAUSSIAN_INIT_FUNCTION_NAME = 'gaussian'
UNIFORM_INIT_FUNCTION_NAME = 'uniform'
CONSTANT_INIT_FUNCTION_NAME = 'constant'
CLIMO_INIT_FUNCTION_NAME = 'climo'

VALID_INIT_FUNCTION_NAMES = [
    GAUSSIAN_INIT_FUNCTION_NAME, UNIFORM_INIT_FUNCTION_NAME,
    CONSTANT_INIT_FUNCTION_NAME, CLIMO_INIT_FUNCTION_NAME
]


def _check_in_and_out_matrices(
        input_matrices, num_examples=None, output_matrices=None):
    """Error-checks input and output matrices.

    T = number of input tensors to the model
    E = number of examples (storm objects)

    :param input_matrices: length-T list of predictor matrices before backwards
        optimization.  Each item must be a numpy array.
    :param num_examples: E in the above discussion.  The first axis of each
        array must have length E.  If you don't know the number of examples,
        leave this as None.
    :param output_matrices: Same as `input_matrices` but after backwards
        optimization.
    :raises: ValueError: if `input_matrices` and `output_matrices` have
        different lengths.
    """

    error_checking.assert_is_list(input_matrices)
    num_matrices = len(input_matrices)

    if output_matrices is None:
        output_matrices = [None] * num_matrices

    error_checking.assert_is_list(output_matrices)
    num_output_matrices = len(output_matrices)

    if num_matrices != num_output_matrices:
        error_string = (
            'Number of input matrices ({0:d}) should = number of output '
            'matrices ({1:d}).'
        ).format(num_matrices, num_output_matrices)

        raise ValueError(error_string)

    for i in range(num_matrices):
        error_checking.assert_is_numpy_array_without_nan(input_matrices[i])

        if num_examples is not None:
            these_expected_dim = numpy.array(
                (num_examples,) + input_matrices[i].shape[1:], dtype=int
            )
            error_checking.assert_is_numpy_array(
                input_matrices[i], exact_dimensions=these_expected_dim
            )

        if output_matrices[i] is not None:
            error_checking.assert_is_numpy_array_without_nan(output_matrices[i])

            these_expected_dim = numpy.array(
                input_matrices[i].shape[:-1], dtype=int
            )
            error_checking.assert_is_numpy_array(
                output_matrices[i], exact_dimensions=these_expected_dim
            )


def _do_gradient_descent(
        model_object, activation_tensor, loss_tensor, init_function_or_matrices,
        num_iterations, learning_rate, l2_weight=None):
    """Does gradient descent (the nitty-gritty part of backwards optimization).

    :param model_object: Trained instance of `keras.models.Model` or
        `keras.models.Sequential`.
    :param activation_tensor: Keras tensor, defining the activation of the
        relevant model component.
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
    :return: result_dict: Dictionary with the following keys.
    result_dict["normalized_input_matrices"]: length-T list of input matrices
        (before backwards optimization).
    result_dict["normalized_output_matrices"]: length-T list of output matrices
        (after backwards optimization).
    result_dict["initial_activation"]: Initial activation (before backwards
        optimization).
    result_dict["final_activation"]: Final activation (after backwards
        optimization).
    """

    if isinstance(model_object.input, list):
        list_of_input_tensors = model_object.input
    else:
        list_of_input_tensors = [model_object.input]

    num_input_tensors = len(list_of_input_tensors)

    if isinstance(init_function_or_matrices, list):
        input_matrices = init_function_or_matrices
        output_matrices = copy.deepcopy(init_function_or_matrices)
    else:
        input_matrices = [None] * num_input_tensors

        for i in range(num_input_tensors):
            these_dimensions = numpy.array(
                [1] + list_of_input_tensors[i].get_shape().as_list()[1:],
                dtype=int
            )

            input_matrices[i] = init_function_or_matrices(these_dimensions)

        output_matrices = copy.deepcopy(input_matrices)

    if l2_weight is not None:
        for i in range(num_input_tensors):
            this_diff_matrix = (
                list_of_input_tensors[i][0, ...] - output_matrices[i][0, ...]
            )

            loss_tensor += l2_weight * K.mean(this_diff_matrix ** 2)

    list_of_gradient_tensors = K.gradients(loss_tensor, list_of_input_tensors)
    for i in range(num_input_tensors):
        list_of_gradient_tensors[i] /= K.maximum(
            K.sqrt(K.mean(list_of_gradient_tensors[i] ** 2)),
            K.epsilon()
        )

    inputs_to_act_loss_grad = K.function(
        list_of_input_tensors + [K.learning_phase()],
        ([activation_tensor, loss_tensor] + list_of_gradient_tensors)
    )

    initial_activation = None

    for j in range(num_iterations):
        these_outputs = inputs_to_act_loss_grad(output_matrices + [0])
        if j == 0:
            initial_activation = these_outputs[0][0]

        if numpy.mod(j, 100) == 0:
            print((
                'Loss after {0:d} of {1:d} iterations = {2:.2e} ... '
                'activation = {3:.2e}'
            ).format(
                j, num_iterations, these_outputs[1], these_outputs[0][0]
            ))

        for i in range(num_input_tensors):
            output_matrices[i] -= these_outputs[i + 2] * learning_rate

    final_activation = these_outputs[0][0]

    print((
        'Loss after {0:d} iterations = {1:.2e} ... activation = {2:.2e}'
    ).format(
        num_iterations, these_outputs[1], final_activation
    ))

    return {
        NORM_INPUT_MATRICES_KEY: input_matrices,
        NORM_OUTPUT_MATRICES_KEY: output_matrices,
        INITIAL_ACTIVATION_KEY: initial_activation,
        FINAL_ACTIVATION_KEY: final_activation
    }


def _radar_constraints_to_loss_fn(model_object, model_metadata_dict, weight):
    """Converts radar constraints to loss function.

    :param model_object: Trained instance of `keras.models.Model` or
        `keras.models.Sequential`.
    :param weight: Weight used to multiply this part of the loss function.
    :param model_metadata_dict:
        [used only if `radar_constraint_weight is not None`]
        Dictionary returned by `cnn.read_model_metadata`.
    :return: loss_tensor: Keras tensor defining the loss function.  This may be
        None.
    """

    if weight is None:
        return None

    list_of_layer_operation_dicts = model_metadata_dict[
        cnn.LAYER_OPERATIONS_KEY]

    if list_of_layer_operation_dicts is None:
        return None

    error_checking.assert_is_greater(weight, 0.)

    if isinstance(model_object.input, list):
        radar_tensor = model_object.input[0]
    else:
        radar_tensor = model_object.input

    return weight * physical_constraints.radar_constraints_to_loss_fn(
        radar_tensor=radar_tensor,
        list_of_layer_operation_dicts=list_of_layer_operation_dicts
    )


def _minmax_constraints_to_loss_fn(model_object, model_metadata_dict, weight):
    """Converts min-max constraints to loss function.

    :param model_object: See doc for `_radar_constraints_to_loss_fn`.
    :param weight: Same.
    :param model_metadata_dict: Same.
    :return: loss_tensor: Same.
    """

    if weight is None:
        return None

    if isinstance(model_object.input, list):
        list_of_input_tensors = model_object.input
    else:
        list_of_input_tensors = [model_object.input]

    return weight * physical_constraints.minima_and_maxima_to_loss_fn(
        list_of_input_tensors=list_of_input_tensors,
        cnn_metadata_dict=model_metadata_dict)


def check_metadata(
        component_type_string, num_iterations, learning_rate, target_class=None,
        layer_name=None, ideal_activation=None, neuron_indices=None,
        channel_index=None, l2_weight=None, radar_constraint_weight=None,
        minmax_constraint_weight=None):
    """Error-checks metadata.

    :param component_type_string: See doc for `saliency_maps.check_metadata`.
    :param num_iterations: Number of iterations.
    :param learning_rate: Learning rate.
    :param target_class: See doc for `saliency_maps.check_metadata`.
    :param layer_name: Same.
    :param ideal_activation: Same.
    :param neuron_indices: Same.
    :param channel_index: Same.
    :param l2_weight: Weight for L_2 regularization.
    :param radar_constraint_weight: Weight used to multiply part of loss
        function with radar constraints (see doc for
        `_radar_constraints_to_loss_fn`).
    :param minmax_constraint_weight: Weight used to multiply part of loss
        function with min-max constraints (see doc for
        `_minmax_constraints_to_loss_fn`).

    :return: metadata_dict: Dictionary with the following keys.
    metadata_dict['component_type_string']: See input doc.
    metadata_dict['num_iterations']: Same.
    metadata_dict['learning_rate']: Same.
    metadata_dict['target_class']: Same.
    metadata_dict['layer_name']: Same.
    metadata_dict['ideal_activation']: Same.
    metadata_dict['neuron_indices']: Same.
    metadata_dict['channel_index']: Same.
    metadata_dict['l2_weight']: Same.
    metadata_dict['radar_constraint_weight']: Same.
    metadata_dict['minmax_constraint_weight']: Same.
    """

    metadata_dict = saliency_maps.check_metadata(
        component_type_string=component_type_string, target_class=target_class,
        layer_name=layer_name, ideal_activation=ideal_activation,
        neuron_indices=neuron_indices, channel_index=channel_index)

    error_checking.assert_is_integer(num_iterations)
    error_checking.assert_is_greater(num_iterations, 0)
    error_checking.assert_is_greater(learning_rate, 0.)
    error_checking.assert_is_less_than(learning_rate, 1.)

    if l2_weight is not None:
        error_checking.assert_is_greater(l2_weight, 0.)
    if radar_constraint_weight is not None:
        error_checking.assert_is_greater(radar_constraint_weight, 0.)
    if minmax_constraint_weight is not None:
        error_checking.assert_is_greater(minmax_constraint_weight, 0.)

    metadata_dict.update({
        NUM_ITERATIONS_KEY: num_iterations,
        LEARNING_RATE_KEY: learning_rate,
        L2_WEIGHT_KEY: l2_weight,
        RADAR_CONSTRAINT_WEIGHT_KEY: radar_constraint_weight,
        MINMAX_CONSTRAINT_WEIGHT_KEY: minmax_constraint_weight
    })


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
            if model_metadata_dict[cnn.CONV_2D3D_KEY]:
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
                        dl_utils.MEAN_VALUE_COLUMN
                    ].loc[[this_key]].values[0]

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
                normalization_table=radar_normalization_table
            )

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
                        dl_utils.MEAN_VALUE_COLUMN
                    ].loc[[this_key]].values[0]

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
                        dl_utils.MEAN_VALUE_COLUMN
                    ].loc[[this_key]].values[0]

                    this_key = (radar_field_names[j], max_heights_m_agl[j])
                    this_second_mean = radar_normalization_table[
                        dl_utils.MEAN_VALUE_COLUMN
                    ].loc[[this_key]].values[0]

                    initial_matrix[..., j] = numpy.mean([
                        this_first_mean, this_second_mean
                    ])

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
                normalization_table=radar_normalization_table
            )

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
                        dl_utils.MEAN_VALUE_COLUMN
                    ].loc[[this_key]].values[0]

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
                normalization_table=sounding_normalization_table
            )

        return None

    return init_function


def optimize_input_for_class(
        model_object, target_class, init_function_or_matrices,
        num_iterations=DEFAULT_NUM_ITERATIONS,
        learning_rate=DEFAULT_LEARNING_RATE, l2_weight=DEFAULT_L2_WEIGHT,
        radar_constraint_weight=None, minmax_constraint_weight=None,
        model_metadata_dict=None):
    """Creates synthetic input example to maximize probability of target class.

    :param model_object: Trained instance of `keras.models.Model` or
        `keras.models.Sequential`.
    :param target_class: Input data will be optimized for this class.  Must be
        an integer in 0...(K - 1), where K = number of classes.
    :param init_function_or_matrices: See doc for `_do_gradient_descent`.
    :param num_iterations: Same.
    :param learning_rate: Same.
    :param l2_weight: Same.
    :param radar_constraint_weight: See doc for `_radar_constraints_to_loss_fn`.
    :param minmax_constraint_weight: See doc for
        `_minmax_constraints_to_loss_fn`.
    :param model_metadata_dict: Same.
    :return: result_dict: See doc for `_do_gradient_descent`.
    """

    check_metadata(
        component_type_string=model_interpretation.CLASS_COMPONENT_TYPE_STRING,
        num_iterations=num_iterations, learning_rate=learning_rate,
        target_class=target_class, l2_weight=l2_weight,
        radar_constraint_weight=radar_constraint_weight,
        minmax_constraint_weight=minmax_constraint_weight)

    num_output_neurons = (
        model_object.layers[-1].output.get_shape().as_list()[-1]
    )

    if num_output_neurons == 1:
        error_checking.assert_is_leq(target_class, 1)
        activation_tensor = model_object.layers[-1].output[..., 0]

        if target_class == 1:
            loss_tensor = K.mean((activation_tensor - 1) ** 2)
        else:
            loss_tensor = K.mean(activation_tensor ** 2)
    else:
        error_checking.assert_is_less_than(target_class, num_output_neurons)

        activation_tensor = model_object.layers[-1].output[..., target_class]
        loss_tensor = K.mean((activation_tensor - 1) ** 2)

    if radar_constraint_weight is not None:
        loss_tensor += _radar_constraints_to_loss_fn(
            model_object=model_object, model_metadata_dict=model_metadata_dict,
            weight=radar_constraint_weight)

    if minmax_constraint_weight is not None:
        loss_tensor += _minmax_constraints_to_loss_fn(
            model_object=model_object, model_metadata_dict=model_metadata_dict,
            weight=minmax_constraint_weight)

    return _do_gradient_descent(
        model_object=model_object, activation_tensor=activation_tensor,
        loss_tensor=loss_tensor,
        init_function_or_matrices=init_function_or_matrices,
        num_iterations=num_iterations, learning_rate=learning_rate,
        l2_weight=l2_weight)


def optimize_input_for_neuron(
        model_object, layer_name, neuron_indices, init_function_or_matrices,
        num_iterations=DEFAULT_NUM_ITERATIONS,
        learning_rate=DEFAULT_LEARNING_RATE, l2_weight=DEFAULT_L2_WEIGHT,
        ideal_activation=DEFAULT_IDEAL_ACTIVATION,
        radar_constraint_weight=None, minmax_constraint_weight=None,
        model_metadata_dict=None):
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

    :param radar_constraint_weight: See doc for `_radar_constraints_to_loss_fn`.
    :param minmax_constraint_weight: See doc for
        `_minmax_constraints_to_loss_fn`.
    :param model_metadata_dict: Same.
    :return: result_dict: See doc for `_do_gradient_descent`.
    """

    check_metadata(
        component_type_string=model_interpretation.NEURON_COMPONENT_TYPE_STRING,
        num_iterations=num_iterations, learning_rate=learning_rate,
        neuron_indices=neuron_indices, l2_weight=l2_weight,
        radar_constraint_weight=radar_constraint_weight,
        minmax_constraint_weight=minmax_constraint_weight)

    neuron_indices_as_tuple = (0,) + tuple(neuron_indices)
    activation_tensor = model_object.get_layer(name=layer_name).output[
        neuron_indices_as_tuple
    ]

    if ideal_activation is None:
        loss_tensor = -K.sign(activation_tensor) * activation_tensor ** 2
    else:
        loss_tensor = (activation_tensor - ideal_activation) ** 2

    if radar_constraint_weight is not None:
        loss_tensor += _radar_constraints_to_loss_fn(
            model_object=model_object, model_metadata_dict=model_metadata_dict,
            weight=radar_constraint_weight)

    if minmax_constraint_weight is not None:
        loss_tensor += _minmax_constraints_to_loss_fn(
            model_object=model_object, model_metadata_dict=model_metadata_dict,
            weight=minmax_constraint_weight)

    return _do_gradient_descent(
        model_object=model_object, activation_tensor=activation_tensor,
        loss_tensor=loss_tensor,
        init_function_or_matrices=init_function_or_matrices,
        num_iterations=num_iterations, learning_rate=learning_rate,
        l2_weight=l2_weight)


def optimize_input_for_channel(
        model_object, layer_name, channel_index, init_function_or_matrices,
        stat_function_for_neuron_activations,
        num_iterations=DEFAULT_NUM_ITERATIONS,
        learning_rate=DEFAULT_LEARNING_RATE, l2_weight=DEFAULT_L2_WEIGHT,
        ideal_activation=DEFAULT_IDEAL_ACTIVATION,
        radar_constraint_weight=None, minmax_constraint_weight=None,
        model_metadata_dict=None):
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

    :param radar_constraint_weight: See doc for `_radar_constraints_to_loss_fn`.
    :param minmax_constraint_weight: See doc for
        `_minmax_constraints_to_loss_fn`.
    :param model_metadata_dict: Same.
    :return: result_dict: See doc for `_do_gradient_descent`.
    """

    check_metadata(
        component_type_string=model_interpretation.NEURON_COMPONENT_TYPE_STRING,
        num_iterations=num_iterations, learning_rate=learning_rate,
        layer_name=layer_name, channel_index=channel_index, l2_weight=l2_weight,
        radar_constraint_weight=radar_constraint_weight,
        minmax_constraint_weight=minmax_constraint_weight)

    activation_tensor = stat_function_for_neuron_activations(
        model_object.get_layer(name=layer_name).output[0, ..., channel_index]
    )

    if ideal_activation is None:
        loss_tensor = -K.sign(activation_tensor) * activation_tensor ** 2
    else:
        loss_tensor = (activation_tensor - ideal_activation) ** 2

    if radar_constraint_weight is not None:
        loss_tensor += _radar_constraints_to_loss_fn(
            model_object=model_object, model_metadata_dict=model_metadata_dict,
            weight=radar_constraint_weight)

    if minmax_constraint_weight is not None:
        loss_tensor += _minmax_constraints_to_loss_fn(
            model_object=model_object, model_metadata_dict=model_metadata_dict,
            weight=minmax_constraint_weight)

    return _do_gradient_descent(
        model_object=model_object, activation_tensor=activation_tensor,
        loss_tensor=loss_tensor,
        init_function_or_matrices=init_function_or_matrices,
        num_iterations=num_iterations, learning_rate=learning_rate,
        l2_weight=l2_weight)


def write_standard_file(
        pickle_file_name, denorm_input_matrices, denorm_output_matrices,
        initial_activations, final_activations, model_file_name, metadata_dict,
        full_storm_id_strings=None, storm_times_unix_sec=None,
        sounding_pressure_matrix_pa=None):
    """Writes backwards-optimized examples to Pickle file.

    E = number of examples (storm objects)
    H = number of sounding heights

    If input matrices do not come from real examples, `full_storm_id_strings`
    and `storm_times_unix_sec` can be None.

    :param pickle_file_name: Path to output file.
    :param denorm_input_matrices: See doc for `_check_in_and_out_matrices`.
    :param denorm_output_matrices: Same.
    :param initial_activations: length-E numpy array of initial model
        activations (before backwards optimization).
    :param final_activations: length-E numpy array of final model activations
        (after backwards optimization).
    :param model_file_name: Path to model that created saliency maps (readable
        by `cnn.read_model`).
    :param metadata_dict: Dictionary created by `check_metadata`.
    :param full_storm_id_strings: length-E list of storm IDs.
    :param storm_times_unix_sec: length-E numpy array of storm times.
    :param sounding_pressure_matrix_pa: E-by-H numpy array of pressure
        levels.  Needed only if `denorm_input_matrices` contains soundings from
        real examples but without pressure as a predictor.
    """

    error_checking.assert_is_string(model_file_name)
    used_real_examples = not (
        full_storm_id_strings is None and storm_times_unix_sec is None
    )

    if used_real_examples:
        error_checking.assert_is_string_list(full_storm_id_strings)
        error_checking.assert_is_numpy_array(
            numpy.array(full_storm_id_strings), num_dimensions=1
        )

        num_examples = len(full_storm_id_strings)
        these_expected_dim = numpy.array([num_examples], dtype=int)

        error_checking.assert_is_integer_numpy_array(storm_times_unix_sec)
        error_checking.assert_is_numpy_array(
            storm_times_unix_sec, exact_dimensions=these_expected_dim)
    else:
        num_examples = denorm_input_matrices[0].shape[0]
        sounding_pressure_matrix_pa = None

    _check_in_and_out_matrices(
        input_matrices=denorm_input_matrices, num_examples=num_examples,
        output_matrices=denorm_output_matrices)

    these_expected_dim = numpy.array([num_examples], dtype=int)

    error_checking.assert_is_numpy_array_without_nan(initial_activations)
    error_checking.assert_is_numpy_array(
        initial_activations, exact_dimensions=these_expected_dim)

    error_checking.assert_is_numpy_array_without_nan(final_activations)
    error_checking.assert_is_numpy_array(
        final_activations, exact_dimensions=these_expected_dim)

    if sounding_pressure_matrix_pa is not None:
        error_checking.assert_is_numpy_array_without_nan(
            sounding_pressure_matrix_pa)
        error_checking.assert_is_greater_numpy_array(
            sounding_pressure_matrix_pa, 0.)
        error_checking.assert_is_numpy_array(
            sounding_pressure_matrix_pa, num_dimensions=2)

        these_expected_dim = numpy.array(
            (num_examples,) + sounding_pressure_matrix_pa.shape[1:],
            dtype=int
        )
        error_checking.assert_is_numpy_array(
            sounding_pressure_matrix_pa, exact_dimensions=these_expected_dim)

    bwo_dictionary = {
        INPUT_MATRICES_KEY: denorm_input_matrices,
        OUTPUT_MATRICES_KEY: denorm_output_matrices,
        INITIAL_ACTIVATIONS_KEY: initial_activations,
        FINAL_ACTIVATIONS_KEY: final_activations,
        MODEL_FILE_KEY: model_file_name,
        FULL_STORM_IDS_KEY: full_storm_id_strings,
        STORM_TIMES_KEY: storm_times_unix_sec,
        NUM_ITERATIONS_KEY: metadata_dict[NUM_ITERATIONS_KEY],
        LEARNING_RATE_KEY: metadata_dict[LEARNING_RATE_KEY],
        L2_WEIGHT_KEY: metadata_dict[L2_WEIGHT_KEY],
        RADAR_CONSTRAINT_WEIGHT_KEY: metadata_dict[RADAR_CONSTRAINT_WEIGHT_KEY],
        MINMAX_CONSTRAINT_WEIGHT_KEY:
            metadata_dict[MINMAX_CONSTRAINT_WEIGHT_KEY],
        COMPONENT_TYPE_KEY: metadata_dict[COMPONENT_TYPE_KEY],
        TARGET_CLASS_KEY: metadata_dict[TARGET_CLASS_KEY],
        LAYER_NAME_KEY: metadata_dict[LAYER_NAME_KEY],
        IDEAL_ACTIVATION_KEY: metadata_dict[IDEAL_ACTIVATION_KEY],
        NEURON_INDICES_KEY: metadata_dict[NEURON_INDICES_KEY],
        CHANNEL_INDEX_KEY: metadata_dict[CHANNEL_INDEX_KEY],
        SOUNDING_PRESSURES_KEY: sounding_pressure_matrix_pa
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(bwo_dictionary, pickle_file_handle)
    pickle_file_handle.close()


def write_pmm_file(
        pickle_file_name, mean_denorm_input_matrices,
        mean_denorm_output_matrices, mean_initial_activation,
        mean_final_activation, model_file_name, non_pmm_file_name,
        pmm_max_percentile_level, mean_sounding_pressures_pa=None):
    """Writes composite of backwards-optimized examples to Pickle file.

    The composite should be created by probability-matched means (PMM).

    T = number of input tensors to the model
    H = number of sounding heights

    :param pickle_file_name: Path to output file.
    :param mean_denorm_input_matrices: See doc for `_check_in_and_out_matrices`.
    :param mean_denorm_output_matrices: Same.
    :param mean_initial_activation: Mean initial activation (before backwards
        optimization).
    :param mean_final_activation: Mean final activation (after backwards
        optimization).
    :param model_file_name: Path to model used for backwards optimization
        (readable by `cnn.read_model`).
    :param non_pmm_file_name: Path to standard backwards-optimization file
        (containing non-composited results).
    :param pmm_max_percentile_level: Max percentile level for PMM.
    :param mean_sounding_pressures_pa: length-H numpy array of PMM-composited
        sounding pressures.  Needed only if `mean_denorm_input_matrices`
        contains soundings from real examples but without pressure as a
        predictor.
    """

    error_checking.assert_is_string(model_file_name)
    error_checking.assert_is_string(non_pmm_file_name)
    error_checking.assert_is_geq(pmm_max_percentile_level, 90.)
    error_checking.assert_is_leq(pmm_max_percentile_level, 100.)

    _check_in_and_out_matrices(
        input_matrices=mean_denorm_input_matrices, num_examples=None,
        output_matrices=mean_denorm_output_matrices)

    error_checking.assert_is_not_nan(mean_initial_activation)
    error_checking.assert_is_not_nan(mean_final_activation)

    if mean_sounding_pressures_pa is not None:
        num_heights = mean_denorm_input_matrices[-1].shape[-2]
        these_expected_dim = numpy.array([num_heights], dtype=int)

        error_checking.assert_is_geq_numpy_array(mean_sounding_pressures_pa, 0.)
        error_checking.assert_is_numpy_array(
            mean_sounding_pressures_pa, exact_dimensions=these_expected_dim)

    mean_bwo_dictionary = {
        MEAN_INPUT_MATRICES_KEY: mean_denorm_input_matrices,
        MEAN_OUTPUT_MATRICES_KEY: mean_denorm_output_matrices,
        MEAN_INITIAL_ACTIVATION_KEY: mean_initial_activation,
        MEAN_FINAL_ACTIVATION_KEY: mean_final_activation,
        MODEL_FILE_KEY: model_file_name,
        NON_PMM_FILE_KEY: non_pmm_file_name,
        PMM_MAX_PERCENTILE_KEY: pmm_max_percentile_level,
        MEAN_SOUNDING_PRESSURES_KEY: mean_sounding_pressures_pa
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(mean_bwo_dictionary, pickle_file_handle)
    pickle_file_handle.close()


def read_file(pickle_file_name):
    """Reads composite or non-composite results from Pickle file.

    :param pickle_file_name: Path to input file (created by
        `write_standard_file` or `write_pmm_file`).
    :return: bwo_dictionary: Has the following keys if not a composite...
    bwo_dictionary['denorm_input_matrices']: See doc for
        `write_standard_file`.
    bwo_dictionary['denorm_output_matrices']: Same.
    bwo_dictionary['initial_activations']: Same.
    bwo_dictionary['final_activations']: Same.
    bwo_dictionary['full_storm_id_strings']: Same.
    bwo_dictionary['storm_times_unix_sec']: Same.
    bwo_dictionary['model_file_name']: Same.
    bwo_dictionary['num_iterations']: Same.
    bwo_dictionary['learning_rate']: Same.
    bwo_dictionary['l2_weight']: Same.
    bwo_dictionary['radar_constraint_weight']: Same.
    bwo_dictionary['minmax_constraint_weight']: Same.
    bwo_dictionary['component_type_string']: Same.
    bwo_dictionary['target_class']: Same.
    bwo_dictionary['layer_name']: Same.
    bwo_dictionary['ideal_activation']: Same.
    bwo_dictionary['neuron_indices']: Same.
    bwo_dictionary['channel_index']: Same.
    bwo_dictionary['sounding_pressure_matrix_pa']: Same.

    ...or the following keys if composite...

    bwo_dictionary['mean_denorm_input_matrices']: See doc for `write_pmm_file`.
    bwo_dictionary['mean_denorm_output_matrices']: Same.
    bwo_dictionary['mean_initial_activation']: Same.
    bwo_dictionary['mean_final_activation']: Same.
    bwo_dictionary['model_file_name']: Same.
    bwo_dictionary['non_pmm_file_name']: Same.
    bwo_dictionary['pmm_max_percentile_level']: Same.
    bwo_dictionary['mean_sounding_pressures_pa']: Same.

    :return: pmm_flag: Boolean flag.  True if `bwo_dictionary` contains
        composite, False otherwise.

    :raises: ValueError: if dictionary does not contain expected keys.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    bwo_dictionary = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    pmm_flag = MEAN_INPUT_MATRICES_KEY in bwo_dictionary

    if pmm_flag:
        missing_keys = list(
            set(PMM_FILE_KEYS) - set(bwo_dictionary.keys())
        )
    else:
        missing_keys = list(
            set(STANDARD_FILE_KEYS) - set(bwo_dictionary.keys())
        )

    if len(missing_keys) == 0:
        return bwo_dictionary, pmm_flag

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), pickle_file_name)

    raise ValueError(error_string)
