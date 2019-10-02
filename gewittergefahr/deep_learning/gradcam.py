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
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import model_interpretation

TOLERANCE = 1e-6
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

BACKPROP_FUNCTION_NAME = 'GuidedBackProp'

PREDICTOR_MATRICES_KEY = model_interpretation.PREDICTOR_MATRICES_KEY
CAM_MATRICES_KEY = 'cam_matrices'
GUIDED_CAM_MATRICES_KEY = 'guided_cam_matrices'
MODEL_FILE_KEY = model_interpretation.MODEL_FILE_KEY
FULL_STORM_IDS_KEY = model_interpretation.FULL_STORM_IDS_KEY
STORM_TIMES_KEY = model_interpretation.STORM_TIMES_KEY
TARGET_CLASS_KEY = 'target_class'
TARGET_LAYER_KEY = 'target_layer_name'
SOUNDING_PRESSURES_KEY = model_interpretation.SOUNDING_PRESSURES_KEY

STANDARD_FILE_KEYS = [
    PREDICTOR_MATRICES_KEY, CAM_MATRICES_KEY, GUIDED_CAM_MATRICES_KEY,
    FULL_STORM_IDS_KEY, STORM_TIMES_KEY, MODEL_FILE_KEY,
    TARGET_CLASS_KEY, TARGET_LAYER_KEY, SOUNDING_PRESSURES_KEY
]

MEAN_PREDICTOR_MATRICES_KEY = model_interpretation.MEAN_PREDICTOR_MATRICES_KEY
MEAN_CAM_MATRICES_KEY = 'mean_cam_matrices'
MEAN_GUIDED_CAM_MATRICES_KEY = 'mean_guided_cam_matrices'
NON_PMM_FILE_KEY = model_interpretation.NON_PMM_FILE_KEY
PMM_MAX_PERCENTILE_KEY = model_interpretation.PMM_MAX_PERCENTILE_KEY
MEAN_SOUNDING_PRESSURES_KEY = model_interpretation.MEAN_SOUNDING_PRESSURES_KEY

PMM_FILE_KEYS = [
    MEAN_PREDICTOR_MATRICES_KEY, MEAN_CAM_MATRICES_KEY,
    MEAN_GUIDED_CAM_MATRICES_KEY, MODEL_FILE_KEY, NON_PMM_FILE_KEY,
    PMM_MAX_PERCENTILE_KEY, MEAN_SOUNDING_PRESSURES_KEY
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
        # interp_object = UnivariateSpline(
        #     x=row_indices_orig, y=numpy.ravel(class_activation_matrix),
        #     k=1, s=0
        # )

        interp_object = UnivariateSpline(
            x=row_indices_orig, y=numpy.ravel(class_activation_matrix),
            k=3, s=0
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
            z=class_activation_matrix, kx=3, ky=3, s=0
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
        predictor_matrices, num_examples=None, cam_matrices=None,
        guided_cam_matrices=None):
    """Error-checks input and output matrices.

    T = number of input tensors to the model
    E = number of examples (storm objects)

    :param predictor_matrices: length-T list of predictor matrices.  Each item
        must be a numpy array.
    :param num_examples: E in the above discussion.  The first axis of each
        array must have length E.  If you don't know the number of examples,
        leave this as None.
    :param cam_matrices: length-T list of class-activation maps.
        cam_matrices[i] is the class-activation map for predictor_matrices[i]
        and should have the same dimensions as predictor_matrices[i], except
        without the last dimension, which is the channel dimension.  If the
        [i]th tensor is not connected to the target layer, cam_matrices[i]
        should be None.
    :param guided_cam_matrices: length-T list of guided class-activation maps.
        Same as cam_matrices[i], except that cam_matrices[i] should have the
        same dimensions as predictor_matrices[i], including the last dimension.
    :raises: ValueError: if `cam_matrices` or `guided_cam_matrices` has
        different length than `predictor_matrices`.
    """

    error_checking.assert_is_list(predictor_matrices)
    num_matrices = len(predictor_matrices)

    if cam_matrices is None:
        cam_matrices = [None] * num_matrices
    if guided_cam_matrices is None:
        guided_cam_matrices = [None] * num_matrices

    num_cam_matrices = len(cam_matrices)
    num_guided_cam_matrices = len(guided_cam_matrices)

    if num_matrices != num_cam_matrices:
        error_string = (
            'Number of predictor matrices ({0:d}) should = number of CAM '
            'matrices ({1:d}).'
        ).format(num_matrices, num_cam_matrices)

        raise ValueError(error_string)

    if num_matrices != num_guided_cam_matrices:
        error_string = (
            'Number of predictor matrices ({0:d}) should = number of guided-CAM'
            ' matrices ({1:d}).'
        ).format(num_matrices, num_guided_cam_matrices)

        raise ValueError(error_string)

    for i in range(num_matrices):
        error_checking.assert_is_numpy_array_without_nan(predictor_matrices[i])

        if num_examples is not None:
            these_expected_dim = numpy.array(
                (num_examples,) + predictor_matrices[i].shape[1:], dtype=int
            )
            error_checking.assert_is_numpy_array(
                predictor_matrices[i], exact_dimensions=these_expected_dim
            )

        if cam_matrices[i] is not None:
            error_checking.assert_is_numpy_array_without_nan(cam_matrices[i])

            these_expected_dim = numpy.array(
                predictor_matrices[i].shape[:-1], dtype=int
            )
            error_checking.assert_is_numpy_array(
                cam_matrices[i], exact_dimensions=these_expected_dim
            )

        if guided_cam_matrices[i] is not None:
            error_checking.assert_is_numpy_array_without_nan(
                guided_cam_matrices[i]
            )

            these_expected_dim = numpy.array(
                predictor_matrices[i].shape, dtype=int
            )
            error_checking.assert_is_numpy_array(
                guided_cam_matrices[i], exact_dimensions=these_expected_dim
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
        predictor_matrices=list_of_input_matrices, num_examples=None)

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
    :return: new_model_object: Model used for guided backpropagation.
    """

    _check_in_and_out_matrices(
        predictor_matrices=list_of_input_matrices, num_examples=None,
        cam_matrices=list_of_cam_matrices)

    error_checking.assert_is_string(target_layer_name)
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


def write_standard_file(
        pickle_file_name, denorm_predictor_matrices, cam_matrices,
        guided_cam_matrices, full_storm_id_strings, storm_times_unix_sec,
        model_file_name, target_class, target_layer_name,
        sounding_pressure_matrix_pa=None):
    """Writes class-activation maps (one per storm object) to Pickle file.

    E = number of examples (storm objects)
    H = number of sounding heights

    :param pickle_file_name: Path to output file.
    :param denorm_predictor_matrices: See doc for `_check_in_and_out_matrices`.
    :param cam_matrices: Same.
    :param guided_cam_matrices: Same.
    :param full_storm_id_strings: length-E list of storm IDs.
    :param storm_times_unix_sec: length-E numpy array of storm times.
    :param model_file_name: Path to model that created saliency maps (readable
        by `cnn.read_model`).
    :param target_class: Target class.  `cam_matrices` and `guided_cam_matrices`
        contain activations for the [k + 1]th class, where k = `target_class`.
    :param target_layer_name: Name of target layer.
    :param sounding_pressure_matrix_pa: E-by-H numpy array of pressure
        levels.  Needed only if the model is trained with soundings but without
        pressure as a predictor.
    """

    error_checking.assert_is_string(model_file_name)
    error_checking.assert_is_integer(target_class)
    error_checking.assert_is_geq(target_class, 0)
    error_checking.assert_is_string(target_layer_name)

    error_checking.assert_is_string_list(full_storm_id_strings)
    error_checking.assert_is_numpy_array(
        numpy.array(full_storm_id_strings), num_dimensions=1
    )

    num_examples = len(full_storm_id_strings)
    these_expected_dim = numpy.array([num_examples], dtype=int)

    error_checking.assert_is_integer_numpy_array(storm_times_unix_sec)
    error_checking.assert_is_numpy_array(
        storm_times_unix_sec, exact_dimensions=these_expected_dim)

    _check_in_and_out_matrices(
        predictor_matrices=denorm_predictor_matrices, num_examples=num_examples,
        cam_matrices=cam_matrices, guided_cam_matrices=guided_cam_matrices)

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

    gradcam_dict = {
        PREDICTOR_MATRICES_KEY: denorm_predictor_matrices,
        CAM_MATRICES_KEY: cam_matrices,
        GUIDED_CAM_MATRICES_KEY: guided_cam_matrices,
        MODEL_FILE_KEY: model_file_name,
        FULL_STORM_IDS_KEY: full_storm_id_strings,
        STORM_TIMES_KEY: storm_times_unix_sec,
        TARGET_CLASS_KEY: target_class,
        TARGET_LAYER_KEY: target_layer_name,
        SOUNDING_PRESSURES_KEY: sounding_pressure_matrix_pa
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(gradcam_dict, pickle_file_handle)
    pickle_file_handle.close()


def write_pmm_file(
        pickle_file_name, mean_denorm_predictor_matrices,
        mean_cam_matrices, mean_guided_cam_matrices, model_file_name,
        non_pmm_file_name, pmm_max_percentile_level,
        mean_sounding_pressures_pa=None):
    """Writes composite class-activation map to Pickle file.

    The composite should be created by probability-matched means (PMM).

    T = number of input tensors to the model
    H = number of sounding heights

    :param pickle_file_name: Path to output file.
    :param mean_denorm_predictor_matrices: See doc for
        `_check_in_and_out_matrices`.
    :param mean_cam_matrices: Same.
    :param mean_guided_cam_matrices: Same.
    :param model_file_name: Path to model that created CAMs (readable by
        `cnn.read_model`).
    :param non_pmm_file_name: Path to standard CAM file (containing
        non-composited CAMs).
    :param pmm_max_percentile_level: Max percentile level for PMM.
    :param mean_sounding_pressures_pa: length-H numpy array of PMM-composited
        sounding pressures.  Needed only if the model is trained with soundings
        but without pressure as a predictor.
    """

    error_checking.assert_is_string(model_file_name)
    error_checking.assert_is_string(non_pmm_file_name)
    error_checking.assert_is_geq(pmm_max_percentile_level, 90.)
    error_checking.assert_is_leq(pmm_max_percentile_level, 100.)

    _check_in_and_out_matrices(
        predictor_matrices=mean_denorm_predictor_matrices, num_examples=None,
        cam_matrices=mean_cam_matrices,
        guided_cam_matrices=mean_guided_cam_matrices)

    if mean_sounding_pressures_pa is not None:
        num_heights = mean_denorm_predictor_matrices[-1].shape[-2]
        these_expected_dim = numpy.array([num_heights], dtype=int)

        error_checking.assert_is_geq_numpy_array(mean_sounding_pressures_pa, 0.)
        error_checking.assert_is_numpy_array(
            mean_sounding_pressures_pa, exact_dimensions=these_expected_dim)

    mean_gradcam_dict = {
        MEAN_PREDICTOR_MATRICES_KEY: mean_denorm_predictor_matrices,
        MEAN_CAM_MATRICES_KEY: mean_cam_matrices,
        MEAN_GUIDED_CAM_MATRICES_KEY: mean_guided_cam_matrices,
        MODEL_FILE_KEY: model_file_name,
        NON_PMM_FILE_KEY: non_pmm_file_name,
        PMM_MAX_PERCENTILE_KEY: pmm_max_percentile_level,
        MEAN_SOUNDING_PRESSURES_KEY: mean_sounding_pressures_pa
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(mean_gradcam_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_file(pickle_file_name):
    """Reads composite or non-composite class-activation maps from Pickle file.

    :param pickle_file_name: Path to input file (created by
        `write_standard_file` or `write_pmm_file`).
    :return: gradcam_dict: Has the following keys if not a composite...
    gradcam_dict['denorm_predictor_matrices']: See doc for
        `write_standard_file`.
    gradcam_dict['cam_matrices']: Same.
    gradcam_dict['guided_cam_matrices']: Same.
    gradcam_dict['full_storm_id_strings']: Same.
    gradcam_dict['storm_times_unix_sec']: Same.
    gradcam_dict['model_file_name']: Same.
    gradcam_dict['target_class']: Same.
    gradcam_dict['target_layer_name']: Same.
    gradcam_dict['sounding_pressure_matrix_pa']: Same.

    ...or the following keys if composite...

    gradcam_dict['mean_denorm_predictor_matrices']: See doc for
        `write_pmm_file`.
    gradcam_dict['mean_cam_matrices']: Same.
    gradcam_dict['mean_guided_cam_matrices']: Same.
    gradcam_dict['model_file_name']: Same.
    gradcam_dict['non_pmm_file_name']: Same.
    gradcam_dict['pmm_max_percentile_level']: Same.
    gradcam_dict['mean_sounding_pressures_pa']: Same.

    :return: pmm_flag: Boolean flag.  True if `gradcam_dict` contains
        composite, False otherwise.

    :raises: ValueError: if dictionary does not contain expected keys.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    gradcam_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    pmm_flag = MEAN_PREDICTOR_MATRICES_KEY in gradcam_dict

    if pmm_flag:
        missing_keys = list(
            set(PMM_FILE_KEYS) - set(gradcam_dict.keys())
        )
    else:
        missing_keys = list(
            set(STANDARD_FILE_KEYS) - set(gradcam_dict.keys())
        )

    if len(missing_keys) == 0:
        return gradcam_dict, pmm_flag

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), pickle_file_name)

    raise ValueError(error_string)
