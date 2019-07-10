"""Methods for Grad-CAM (gradient-weighted class-activation-mapping).

Most of this was scavenged from:
https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py

--- REFERENCE ---

Selvaraju, R.R., M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra,
2017: "Grad-CAM: Visual explanations from deep networks via gradient-based
localization".  International Conference on Computer Vision, IEEE,
https://doi.org/10.1109/ICCV.2017.74.
"""

import pickle
import numpy
import keras
from keras import backend as K
import tensorflow
from tensorflow.python.framework import ops as tensorflow_ops
from scipy.interpolate import (
    UnivariateSpline, RectBivariateSpline, RegularGridInterpolator
)
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import monte_carlo
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import model_interpretation

TOLERANCE = 1e-6

BACKPROP_FUNCTION_NAME = 'GuidedBackProp'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_MATRICES_KEY = 'list_of_input_matrices'
CAM_MATRICES_KEY = 'list_of_cam_matrices'
GUIDED_CAM_MATRICES_KEY = 'list_of_guided_cam_matrices'

FULL_IDS_KEY = tracking_io.FULL_IDS_KEY
STORM_TIMES_KEY = tracking_io.STORM_TIMES_KEY
MODEL_FILE_KEY = model_interpretation.MODEL_FILE_KEY
TARGET_CLASS_KEY = 'target_class'
TARGET_LAYER_KEY = 'target_layer_name'
SOUNDING_PRESSURES_KEY = 'sounding_pressure_matrix_pascals'

STANDARD_FILE_KEYS = [
    INPUT_MATRICES_KEY, CAM_MATRICES_KEY, GUIDED_CAM_MATRICES_KEY,
    FULL_IDS_KEY, STORM_TIMES_KEY, MODEL_FILE_KEY,
    TARGET_CLASS_KEY, TARGET_LAYER_KEY, SOUNDING_PRESSURES_KEY
]

MEAN_INPUT_MATRICES_KEY = model_interpretation.MEAN_INPUT_MATRICES_KEY
MEAN_CAM_MATRICES_KEY = 'list_of_mean_cam_matrices'
MEAN_GUIDED_CAM_MATRICES_KEY = 'list_of_mean_guided_cam_matrices'
STANDARD_FILE_NAME_KEY = 'standard_gradcam_file_name'
PMM_METADATA_KEY = 'pmm_metadata_dict'
CAM_MONTE_CARLO_KEY = 'cam_monte_carlo_dict'
GUIDED_CAM_MONTE_CARLO_KEY = 'guided_cam_monte_carlo_dict'

PMM_FILE_KEYS = [
    MEAN_INPUT_MATRICES_KEY, MEAN_CAM_MATRICES_KEY,
    MEAN_GUIDED_CAM_MATRICES_KEY, MODEL_FILE_KEY, STANDARD_FILE_NAME_KEY,
    PMM_METADATA_KEY, CAM_MONTE_CARLO_KEY, GUIDED_CAM_MONTE_CARLO_KEY
]


def _compute_gradients(loss_tensor, list_of_input_tensors):
    """Computes gradient of each input tensor with respect to loss tensor.

    :param loss_tensor: Loss tensor.
    :param list_of_input_tensors: 1-D list of input tensors.
    :return: list_of_gradient_tensors: 1-D list of gradient tensors.
    """

    list_of_gradient_tensors = tensorflow.gradients(
        loss_tensor, list_of_input_tensors)

    for i in range(len(list_of_gradient_tensors)):
        if list_of_gradient_tensors[i] is not None:
            continue

        list_of_gradient_tensors[i] = tensorflow.zeros_like(
            list_of_input_tensors[i]
        )

    return list_of_gradient_tensors


def _normalize_tensor(input_tensor):
    """Normalizes tensor by its L2 norm.

    :param input_tensor: Unnormalized tensor.
    :return: output_tensor: Normalized tensor.
    """

    rms_tensor = K.sqrt(K.mean(K.square(input_tensor)))
    return input_tensor / (rms_tensor + K.epsilon())


def _upsample_cam(class_activation_matrix, new_dimensions):
    """Upsamples class-activation matrix (CAM).

    CAM may be 1-D, 2-D, or 3-D.

    :param class_activation_matrix: numpy array containing 1-D, 2-D, or 3-D
        class-activation matrix.
    :param new_dimensions: numpy array of new dimensions.  If matrix is
        {1D, 2D, 3D}, this must be a length-{1, 2, 3} array, respectively.
    :return: class_activation_matrix: Upsampled version of input.
    """

    num_rows_new = new_dimensions[0]
    row_indices_new = numpy.linspace(
        1, num_rows_new, num=num_rows_new, dtype=float
    )
    row_indices_orig = numpy.linspace(
        1, num_rows_new, num=class_activation_matrix.shape[0], dtype=float
    )

    if len(new_dimensions) == 1:
        interp_object = UnivariateSpline(
            x=row_indices_orig, y=numpy.ravel(class_activation_matrix),
            k=1, s=0
        )

        return interp_object(row_indices_new)

    num_columns_new = new_dimensions[1]
    column_indices_new = numpy.linspace(
        1, num_columns_new, num=num_columns_new, dtype=float
    )
    column_indices_orig = numpy.linspace(
        1, num_columns_new, num=class_activation_matrix.shape[1], dtype=float
    )

    if len(new_dimensions) == 2:
        interp_object = RectBivariateSpline(
            x=row_indices_orig, y=column_indices_orig,
            z=class_activation_matrix, kx=1, ky=1, s=0
        )

        return interp_object(x=row_indices_new, y=column_indices_new, grid=True)

    num_heights_new = new_dimensions[2]
    height_indices_new = numpy.linspace(
        1, num_heights_new, num=num_heights_new, dtype=float
    )
    height_indices_orig = numpy.linspace(
        1, num_heights_new, num=class_activation_matrix.shape[2], dtype=float
    )

    interp_object = RegularGridInterpolator(
        points=(row_indices_orig, column_indices_orig, height_indices_orig),
        values=class_activation_matrix, method='linear'
    )

    row_index_matrix, column_index_matrix, height_index_matrix = (
        numpy.meshgrid(row_indices_new, column_indices_new, height_indices_new)
    )
    query_point_matrix = numpy.stack(
        (row_index_matrix, column_index_matrix, height_index_matrix), axis=-1
    )

    return interp_object(query_point_matrix)


def _register_guided_backprop():
    """Registers guided-backprop method with TensorFlow backend."""

    if (BACKPROP_FUNCTION_NAME not in
            tensorflow_ops._gradient_registry._registry):

        @tensorflow_ops.RegisterGradient(BACKPROP_FUNCTION_NAME)
        def _GuidedBackProp(operation, gradient_tensor):
            input_type = operation.inputs[0].dtype

            return (
                gradient_tensor *
                tensorflow.cast(gradient_tensor > 0., input_type) *
                tensorflow.cast(operation.inputs[0] > 0., input_type)
            )


def _change_backprop_function(model_object):
    """Changes backpropagation function for Keras model.

    :param model_object: Instance of `keras.models.Model` or
        `keras.models.Sequential`.
    :return: new_model_object: Same as `model_object` but with new backprop
        function.
    """

    model_dict = model_object.get_config()
    standard_function_names = ['relu', 'elu', 'selu']
    advanced_function_names = ['LeakyReLU', 'ELU']

    for this_layer_dict in model_dict['layers']:
        first_flag = this_layer_dict['class_name'] in advanced_function_names
        second_flag = (
            this_layer_dict['class_name'] == 'Activation' and
            this_layer_dict['config']['activation'] in standard_function_names
        )

        change_this_activation_function = first_flag or second_flag
        if change_this_activation_function:
            this_layer_dict['class_name'] = 'Activation'
            this_layer_dict['config']['activation'] = 'relu'

            if 'alpha' in this_layer_dict['config']:
                this_layer_dict['config'].pop('alpha')

    orig_to_new_operation_dict = {
        'Relu': BACKPROP_FUNCTION_NAME,
        'relu': BACKPROP_FUNCTION_NAME
    }

    graph_object = tensorflow.get_default_graph()

    with graph_object.gradient_override_map(orig_to_new_operation_dict):
        # new_model_object = keras.models.clone_model(model_object)
        new_model_object = keras.models.Model.from_config(model_dict)
        new_model_object.set_weights(model_object.get_weights())

    print(SEPARATOR_STRING)

    new_model_dict = new_model_object.get_config()
    for this_layer_dict in new_model_dict['layers']:
        print(this_layer_dict)

    print(SEPARATOR_STRING)
    return new_model_object


def _make_saliency_function(model_object, target_layer_name,
                            input_layer_indices):
    """Creates saliency function.

    This function computes the gradient of activations in the target layer with
    respect to each input value in the specified layers.

    :param model_object: Instance of `keras.models.Model` or
        `keras.models.Sequential`.
    :param target_layer_name: Target layer (numerator in gradient will be based
        on activations in this layer).
    :param input_layer_indices: 1-D numpy array of indices.  If the array
        contains j, the gradient will be computed with respect to every value in
        the [j]th input layer.
    :return: saliency_function: Instance of `keras.backend.function`.
    """

    output_tensor = model_object.get_layer(name=target_layer_name).output
    filter_maxxed_output_tensor = K.max(output_tensor, axis=-1)

    if isinstance(model_object.input, list):
        list_of_input_tensors = model_object.input
    else:
        list_of_input_tensors = [model_object.input]

    list_of_saliency_tensors = K.gradients(
        K.sum(filter_maxxed_output_tensor),
        [list_of_input_tensors[i] for i in input_layer_indices]
    )

    return K.function(
        list_of_input_tensors + [K.learning_phase()],
        list_of_saliency_tensors
    )


def _normalize_guided_gradcam_output(ggradcam_output_matrix):
    """Normalizes image produced by guided Grad-CAM.

    :param ggradcam_output_matrix: numpy array of output values from guided
        Grad-CAM.
    :return: ggradcam_output_matrix: Normalized version of input.  If the first
        axis had length 1, it has been removed ("squeezed out").
    """

    # Standardize.
    ggradcam_output_matrix -= numpy.mean(ggradcam_output_matrix)
    ggradcam_output_matrix /= (
        numpy.std(ggradcam_output_matrix, ddof=0) + K.epsilon()
    )

    # Force standard deviation of 0.1 and mean of 0.5.
    ggradcam_output_matrix = 0.5 + ggradcam_output_matrix * 0.1
    ggradcam_output_matrix[ggradcam_output_matrix < 0.] = 0.
    ggradcam_output_matrix[ggradcam_output_matrix > 1.] = 1.

    return ggradcam_output_matrix


def _get_connected_input_layers(model_object, list_of_input_matrices,
                                target_layer_name):
    """Finds input layers connected to target layer.

    :param model_object: See doc for `run_gradcam`.
    :param list_of_input_matrices: Same.
    :param target_layer_name: Same.
    :return: connected_to_input_layer_indices: 1-D numpy array of indices.  If
        the array contains j, the target layer is connected to the [j]th input
        layer, which has the same dimensions as list_of_input_matrices[j].
    """

    connected_layer_objects = cnn.get_connected_input_layers(
        model_object=model_object, target_layer_name=target_layer_name)

    num_input_matrices = len(list_of_input_matrices)
    num_connected_layers = len(connected_layer_objects)
    connected_to_input_layer_indices = numpy.full(
        num_connected_layers, -1, dtype=int)

    for i in range(num_connected_layers):
        these_first_dim = numpy.array(
            list(connected_layer_objects[i].input_shape[1:]), dtype=int
        )

        for j in range(num_input_matrices):
            these_second_dim = numpy.array(
                list_of_input_matrices[j].shape[1:], dtype=int
            )

            if not numpy.array_equal(these_first_dim, these_second_dim):
                continue

            connected_to_input_layer_indices[i] = j
            break

        if connected_to_input_layer_indices[i] >= 0:
            continue

        error_string = (
            'Cannot find matrix corresponding to input layer "{0:s}".  '
            'Dimensions of "{0:s}" (excluding example dimension) are {1:s}.'
        ).format(
            connected_layer_objects[i].name, str(these_first_dim)
        )

        raise ValueError(error_string)

    return connected_to_input_layer_indices


def _check_in_and_out_matrices(
        list_of_input_matrices, num_examples, list_of_cam_matrices=None,
        list_of_guided_cam_matrices=None):
    """Error-checks input and output matrices for Grad-CAM.

    T = number of input tensors to the model

    :param list_of_input_matrices: length-T list of numpy arrays, containing
        only one example (storm object).  list_of_input_matrices[i] must have
        the same dimensions as the [i]th input tensor to the model.
    :param num_examples: Number of examples expected.  If num_examples <= 0,
        this method will allow the example dimension to be missing from the
        matrices.
    :param list_of_cam_matrices: length-T list of numpy arrays.
        list_of_cam_matrices[i] will be the class-activation map for
        list_of_input_matrices[i].  It will have the same dimensions as
        list_of_input_matrices[i], just without the channel dimension.  If the
        target layer is not connected to the [i]th input tensor,
        list_of_cam_matrices[i] will be None.
    :param list_of_guided_cam_matrices: length-T list of numpy arrays.
        list_of_guided_cam_matrices[i] will be the guided class-activation map
        for list_of_input_matrices[i] and will have the same dimensions as
        list_of_input_matrices[i].  If the target layer is not connected to the
        [i]th input tensor, list_of_guided_cam_matrices[i] will be None.
    :return: list_of_input_matrices: Same as input but guaranteed to have
        example dimensions.
    """

    error_checking.assert_is_list(list_of_input_matrices)
    num_input_matrices = len(list_of_input_matrices)

    if list_of_cam_matrices is None:
        list_of_cam_matrices = [None] * num_input_matrices
    if list_of_guided_cam_matrices is None:
        list_of_guided_cam_matrices = [None] * num_input_matrices

    assert len(list_of_cam_matrices) == num_input_matrices
    assert len(list_of_guided_cam_matrices) == num_input_matrices

    for i in range(num_input_matrices):
        error_checking.assert_is_numpy_array_without_nan(
            list_of_input_matrices[i]
        )

        if num_examples == 1 and list_of_input_matrices[i].shape[0] != 1:
            list_of_input_matrices[i] = numpy.expand_dims(
                list_of_input_matrices[i], axis=0
            )

        if num_examples > 0:
            assert list_of_input_matrices[i].shape[0] == num_examples

        if list_of_cam_matrices[i] is not None:
            error_checking.assert_is_numpy_array_without_nan(
                list_of_cam_matrices[i]
            )

            these_expected_dim = numpy.array(
                list_of_input_matrices[i].shape[:-1], dtype=int
            )

            error_checking.assert_is_numpy_array(
                list_of_cam_matrices[i], exact_dimensions=these_expected_dim
            )

        if list_of_guided_cam_matrices[i] is not None:
            error_checking.assert_is_numpy_array_without_nan(
                list_of_guided_cam_matrices[i]
            )

            these_expected_dim = numpy.array(
                list_of_input_matrices[i].shape, dtype=int
            )

            error_checking.assert_is_numpy_array(
                list_of_guided_cam_matrices[i],
                exact_dimensions=these_expected_dim
            )


def run_gradcam(model_object, list_of_input_matrices, target_class,
                target_layer_name):
    """Runs Grad-CAM.

    T = number of input tensors to the model

    :param model_object: Trained model (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param list_of_input_matrices: See doc for `_check_in_and_out_matrices`.
    :param target_class: Class-activation maps will be created for this class.
        Must be an integer in 0...(K - 1), where K = number of classes.
    :param target_layer_name: Name of target layer.  Neuron-importance weights
        will be based on activations in this layer.  Must be a convolution
        layer.
    :return: list_of_cam_matrices: See doc for `_check_in_and_out_matrices`.
    """

    error_checking.assert_is_integer(target_class)
    error_checking.assert_is_geq(target_class, 0)
    error_checking.assert_is_string(target_layer_name)
    _check_in_and_out_matrices(
        list_of_input_matrices=list_of_input_matrices, num_examples=1)

    num_input_matrices = len(list_of_input_matrices)

    # Create loss tensor.
    output_layer_object = model_object.layers[-1].output
    num_output_neurons = output_layer_object.get_shape().as_list()[-1]

    if num_output_neurons == 1:
        error_checking.assert_is_leq(target_class, 1)

        if target_class == 1:
            # loss_tensor = model_object.layers[-1].output[..., 0]
            loss_tensor = model_object.layers[-1].input[..., 0]
        else:
            # loss_tensor = -1 * model_object.layers[-1].output[..., 0]
            loss_tensor = -1 * model_object.layers[-1].input[..., 0]
    else:
        error_checking.assert_is_less_than(target_class, num_output_neurons)
        # loss_tensor = model_object.layers[-1].output[..., target_class]
        loss_tensor = model_object.layers[-1].input[..., target_class]

    # Create gradient function.
    target_layer_activation_tensor = model_object.get_layer(
        name=target_layer_name
    ).output

    gradient_tensor = _compute_gradients(
        loss_tensor, [target_layer_activation_tensor]
    )[0]
    gradient_tensor = _normalize_tensor(gradient_tensor)

    if isinstance(model_object.input, list):
        list_of_input_tensors = model_object.input
    else:
        list_of_input_tensors = [model_object.input]

    gradient_function = K.function(
        list_of_input_tensors, [target_layer_activation_tensor, gradient_tensor]
    )

    # Evaluate gradient function.
    target_layer_activation_matrix, gradient_matrix = gradient_function(
        list_of_input_matrices)
    target_layer_activation_matrix = target_layer_activation_matrix[0, ...]
    gradient_matrix = gradient_matrix[0, ...]

    # Compute class-activation matrix in the target layer's space.
    mean_weight_by_filter = numpy.mean(gradient_matrix, axis=(0, 1))
    original_cam_matrix = numpy.ones(target_layer_activation_matrix.shape[:-1])

    num_filters = len(mean_weight_by_filter)
    for k in range(num_filters):
        original_cam_matrix += (
            mean_weight_by_filter[k] * target_layer_activation_matrix[..., k]
        )

    input_layer_indices = _get_connected_input_layers(
        model_object=model_object,
        list_of_input_matrices=list_of_input_matrices,
        target_layer_name=target_layer_name)

    list_of_cam_matrices = [None] * num_input_matrices

    for i in input_layer_indices:
        these_spatial_dim = numpy.array(
            list_of_input_matrices[i].shape[1:-1], dtype=int
        )

        add_height_dim = (
            len(original_cam_matrix.shape) == 2 and len(these_spatial_dim) == 3
        )

        if add_height_dim:
            list_of_cam_matrices[i] = _upsample_cam(
                class_activation_matrix=original_cam_matrix,
                new_dimensions=these_spatial_dim[:-1]
            )

            list_of_cam_matrices[i] = numpy.expand_dims(
                list_of_cam_matrices[i], axis=-1
            )

            list_of_cam_matrices[i] = numpy.repeat(
                a=list_of_cam_matrices[i], repeats=these_spatial_dim[-1],
                axis=-1
            )
        else:
            list_of_cam_matrices[i] = _upsample_cam(
                class_activation_matrix=original_cam_matrix,
                new_dimensions=these_spatial_dim)

        list_of_cam_matrices[i] = numpy.maximum(list_of_cam_matrices[i], 0.)
        list_of_cam_matrices[i] = numpy.expand_dims(
            list_of_cam_matrices[i], axis=0
        )

    # denominator = numpy.maximum(numpy.max(class_activation_matrix), K.epsilon())
    # return class_activation_matrix / denominator

    return list_of_cam_matrices


def run_guided_gradcam(
        orig_model_object, list_of_input_matrices, target_layer_name,
        list_of_cam_matrices, new_model_object=None):
    """Runs guided Grad-CAM.

    T = number of input tensors to the model

    :param orig_model_object: Original model (trained instance of
        `keras.models.Model` or `keras.models.Sequential`).
    :param list_of_input_matrices: See doc for `run_gradcam`.
    :param target_layer_name: Same.
    :param list_of_cam_matrices: Same.
    :param new_model_object: New model (created by `_change_backprop_function`)
        for use in guided backpropagation.  If this is None, it will be created
        on the fly.
    :return: list_of_guided_cam_matrices: See doc for
        `_check_in_and_out_matrices`.
    """

    error_checking.assert_is_string(target_layer_name)
    _check_in_and_out_matrices(
        list_of_input_matrices=list_of_input_matrices,
        list_of_cam_matrices=list_of_cam_matrices, num_examples=1)

    num_input_matrices = len(list_of_input_matrices)

    if new_model_object is None:
        _register_guided_backprop()
        new_model_object = _change_backprop_function(
            model_object=orig_model_object)

    input_layer_indices = _get_connected_input_layers(
        model_object=orig_model_object,
        list_of_input_matrices=list_of_input_matrices,
        target_layer_name=target_layer_name)

    saliency_function = _make_saliency_function(
        model_object=new_model_object, target_layer_name=target_layer_name,
        input_layer_indices=input_layer_indices)

    list_of_saliency_matrices = saliency_function(list_of_input_matrices + [0])

    min_saliency_values = numpy.array([
        numpy.min(s) for s in list_of_saliency_matrices
    ])
    max_saliency_values = numpy.array([
        numpy.max(s) for s in list_of_saliency_matrices
    ])

    print((
        'Min saliency by input matrix = {0:s} ... max saliency by matrix = '
        '{1:s}'
    ).format(
        str(min_saliency_values), str(max_saliency_values)
    ))

    list_of_guided_cam_matrices = [None] * num_input_matrices
    saliency_index = 0

    for i in range(num_input_matrices):
        if list_of_cam_matrices[i] is None:
            continue

        list_of_guided_cam_matrices[i] = (
            list_of_saliency_matrices[saliency_index] *
            list_of_cam_matrices[i][..., numpy.newaxis]
        )

        saliency_index += 1

    # ggradcam_output_matrix = _normalize_guided_gradcam_output(
    #     ggradcam_output_matrix[0, ...])

    return list_of_guided_cam_matrices, new_model_object


def write_pmm_file(
        pickle_file_name, list_of_mean_input_matrices,
        list_of_mean_cam_matrices, list_of_mean_guided_cam_matrices,
        model_file_name, standard_gradcam_file_name, pmm_metadata_dict,
        cam_monte_carlo_dict=None, guided_cam_monte_carlo_dict=None):
    """Writes PMM (probability-matched mean) over many CAMs to Pickle file.

    :param pickle_file_name: Path to output file.
    :param list_of_mean_input_matrices: See doc for `_check_in_and_out_matrices`
        with num_examples = 0.
    :param list_of_mean_cam_matrices: Same.
    :param list_of_mean_guided_cam_matrices: Same.
    :param model_file_name: Path to file with CNN used to create CAMs (readable
        by `cnn.read_model`).
    :param standard_gradcam_file_name: Path to file with standard Grad-CAM
        output (inputs to the composite).  Should be readable by
        `read_standard_file`.
    :param pmm_metadata_dict: Dictionary created by
        `prob_matched_means.check_input_args`.
    :param cam_monte_carlo_dict: Dictionary with results of Monte Carlo
        significance test.  Must contain keys listed in
        `monte_carlo.check_output`, plus the following.

    cam_monte_carlo_dict['baseline_file_name']: Path to saliency file for
        baseline set (readable by `read_standard_file`), against which the trial
        set (contained in `standard_gradcam_file_name`) was compared.

    :param guided_cam_monte_carlo_dict: Same but for guided CAMs.
    """

    # TODO(thunderhoser): This method currently does not deal with sounding
    # pressures.

    error_checking.assert_is_string(model_file_name)
    error_checking.assert_is_string(standard_gradcam_file_name)

    _check_in_and_out_matrices(
        list_of_input_matrices=list_of_mean_input_matrices, num_examples=0,
        list_of_cam_matrices=list_of_mean_cam_matrices,
        list_of_guided_cam_matrices=list_of_mean_guided_cam_matrices)

    num_input_matrices = len(list_of_mean_input_matrices)

    if (cam_monte_carlo_dict is not None or
            guided_cam_monte_carlo_dict is not None):
        monte_carlo.check_output(cam_monte_carlo_dict)
        error_checking.assert_is_string(
            cam_monte_carlo_dict[monte_carlo.BASELINE_FILE_KEY]
        )

        monte_carlo.check_output(guided_cam_monte_carlo_dict)
        error_checking.assert_is_string(
            guided_cam_monte_carlo_dict[monte_carlo.BASELINE_FILE_KEY]
        )

        for i in range(num_input_matrices):
            both_none = (
                list_of_mean_cam_matrices[i] is None and
                cam_monte_carlo_dict[
                    monte_carlo.TRIAL_PMM_MATRICES_KEY][i] is None
            )

            if both_none:
                continue

            assert numpy.allclose(
                list_of_mean_cam_matrices[i],
                cam_monte_carlo_dict[monte_carlo.TRIAL_PMM_MATRICES_KEY][i],
                atol=TOLERANCE
            )

            assert numpy.allclose(
                list_of_mean_guided_cam_matrices[i],
                guided_cam_monte_carlo_dict[
                    monte_carlo.TRIAL_PMM_MATRICES_KEY][i],
                atol=TOLERANCE
            )

    mean_gradcam_dict = {
        MEAN_INPUT_MATRICES_KEY: list_of_mean_input_matrices,
        MEAN_CAM_MATRICES_KEY: list_of_mean_cam_matrices,
        MEAN_GUIDED_CAM_MATRICES_KEY: list_of_mean_guided_cam_matrices,
        MODEL_FILE_KEY: model_file_name,
        STANDARD_FILE_NAME_KEY: standard_gradcam_file_name,
        PMM_METADATA_KEY: pmm_metadata_dict,
        CAM_MONTE_CARLO_KEY: cam_monte_carlo_dict,
        GUIDED_CAM_MONTE_CARLO_KEY: guided_cam_monte_carlo_dict
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(mean_gradcam_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_pmm_file(pickle_file_name):
    """Reads mean class-activation map from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: mean_gradcam_dict: Dictionary with the following keys.
    mean_gradcam_dict['list_of_mean_input_matrices']: See doc for
        `write_pmm_file`.
    mean_gradcam_dict['list_of_mean_cam_matrices']: Same.
    mean_gradcam_dict['list_of_mean_guided_cam_matrices']: Same.
    mean_gradcam_dict['model_file_name']: Same.
    mean_gradcam_dict['standard_gradcam_file_name']: Same.
    mean_gradcam_dict['pmm_metadata_dict']: Same.
    mean_gradcam_dict['cam_monte_carlo_dict']: Same.
    mean_gradcam_dict['guided_cam_monte_carlo_dict']: Same.

    :raises: ValueError: if any of the aforelisted keys are missing from the
        dictionary.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    mean_gradcam_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    if (CAM_MONTE_CARLO_KEY not in mean_gradcam_dict and
            GUIDED_CAM_MONTE_CARLO_KEY not in mean_gradcam_dict):
        mean_gradcam_dict[CAM_MONTE_CARLO_KEY] = None
        mean_gradcam_dict[GUIDED_CAM_MONTE_CARLO_KEY] = None

    missing_keys = list(set(PMM_FILE_KEYS) - set(mean_gradcam_dict.keys()))
    if len(missing_keys) == 0:
        return mean_gradcam_dict

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), pickle_file_name)

    raise ValueError(error_string)


def write_standard_file(
        pickle_file_name, list_of_input_matrices, list_of_cam_matrices,
        list_of_guided_cam_matrices, model_file_name, full_id_strings,
        storm_times_unix_sec, target_class, target_layer_name,
        sounding_pressure_matrix_pascals=None):
    """Writes class-activation maps (one for each example) to Pickle file.

    E = number of examples (storm objects)
    H = number of height levels per sounding

    :param pickle_file_name: Path to output file.
    :param list_of_input_matrices: See doc for `_check_in_and_out_matrices`.
    :param list_of_cam_matrices: Same.
    :param list_of_guided_cam_matrices: Same.
    :param model_file_name: Path to file with trained CNN (readable by
        `cnn.read_model`).
    :param full_id_strings: length-E list of full storm IDs.
    :param storm_times_unix_sec: length-E numpy array of storm times.
    :param target_class: See doc for `run_gradcam`.
    :param target_layer_name: Same.
    :param sounding_pressure_matrix_pascals: E-by-H numpy array of pressure
        levels in soundings.  Useful when model input contains soundings with no
        pressure variable, since pressure is needed to plot soundings.
    """

    error_checking.assert_is_string(model_file_name)
    error_checking.assert_is_integer(target_class)
    error_checking.assert_is_geq(target_class, 0)
    error_checking.assert_is_string(target_layer_name)

    error_checking.assert_is_string_list(full_id_strings)
    error_checking.assert_is_numpy_array(
        numpy.array(full_id_strings), num_dimensions=1
    )

    num_storm_objects = len(full_id_strings)
    these_expected_dim = numpy.array([num_storm_objects], dtype=int)

    error_checking.assert_is_integer_numpy_array(storm_times_unix_sec)
    error_checking.assert_is_numpy_array(
        storm_times_unix_sec, exact_dimensions=these_expected_dim)

    _check_in_and_out_matrices(
        list_of_input_matrices=list_of_input_matrices,
        list_of_cam_matrices=list_of_cam_matrices,
        list_of_guided_cam_matrices=list_of_guided_cam_matrices,
        num_examples=num_storm_objects)

    if sounding_pressure_matrix_pascals is not None:
        error_checking.assert_is_numpy_array(
            sounding_pressure_matrix_pascals, num_dimensions=2)

        these_expected_dim = numpy.array(
            (num_storm_objects,) + sounding_pressure_matrix_pascals.shape[1:],
            dtype=int
        )

        error_checking.assert_is_numpy_array(
            sounding_pressure_matrix_pascals,
            exact_dimensions=these_expected_dim)

    gradcam_dict = {
        INPUT_MATRICES_KEY: list_of_input_matrices,
        CAM_MATRICES_KEY: list_of_cam_matrices,
        GUIDED_CAM_MATRICES_KEY: list_of_guided_cam_matrices,
        MODEL_FILE_KEY: model_file_name,
        FULL_IDS_KEY: full_id_strings,
        STORM_TIMES_KEY: storm_times_unix_sec,
        TARGET_CLASS_KEY: target_class,
        TARGET_LAYER_KEY: target_layer_name,
        SOUNDING_PRESSURES_KEY: sounding_pressure_matrix_pascals
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(gradcam_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_standard_file(pickle_file_name):
    """Reads class-activation maps (one for each example) from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: gradcam_dict: Dictionary with the following keys.
    gradcam_dict['list_of_input_matrices']: See doc for `write_standard_file`.
    gradcam_dict['list_of_cam_matrices']: Same.
    gradcam_dict['list_of_guided_cam_matrices']: Same.
    gradcam_dict['model_file_name']: Same.
    gradcam_dict['full_id_strings']: Same.
    gradcam_dict['storm_times_unix_sec']: Same.
    gradcam_dict['target_class']: Same.
    gradcam_dict['target_layer_name']: Same.
    gradcam_dict['sounding_pressure_matrix_pascals']: Same.

    :raises: ValueError: if any of the aforelisted keys are missing from the
        dictionary.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    gradcam_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    missing_keys = list(set(STANDARD_FILE_KEYS) - set(gradcam_dict.keys()))
    if len(missing_keys) == 0:
        return gradcam_dict

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), pickle_file_name)

    raise ValueError(error_string)
