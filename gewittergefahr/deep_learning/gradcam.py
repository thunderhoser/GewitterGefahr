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
    UnivariateSpline, RectBivariateSpline, RegularGridInterpolator)
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

BACKPROP_FUNCTION_NAME = 'GuidedBackProp'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_MATRICES_KEY = 'list_of_input_matrices'
CLASS_ACTIVATIONS_KEY = 'class_activation_matrix'
GUIDED_GRADCAM_KEY = 'ggradcam_output_matrix'

STORM_IDS_KEY = tracking_io.STORM_IDS_KEY + ''
STORM_TIMES_KEY = tracking_io.STORM_TIMES_KEY + ''
MODEL_FILE_NAME_KEY = 'model_file_name'
TARGET_CLASS_KEY = 'target_class'
TARGET_LAYER_KEY = 'target_layer_name'
SOUNDING_PRESSURES_KEY = 'sounding_pressure_matrix_pascals'

STANDARD_FILE_KEYS = [
    INPUT_MATRICES_KEY, CLASS_ACTIVATIONS_KEY, GUIDED_GRADCAM_KEY,
    STORM_IDS_KEY, STORM_TIMES_KEY, MODEL_FILE_NAME_KEY,
    TARGET_CLASS_KEY, TARGET_LAYER_KEY, SOUNDING_PRESSURES_KEY
]

MEAN_INPUT_MATRICES_KEY = 'list_of_mean_input_matrices'
MEAN_CLASS_ACTIVATIONS_KEY = 'mean_class_activation_matrix'
MEAN_GUIDED_GRADCAM_KEY = 'mean_ggradcam_output_matrix'
THRESHOLD_COUNTS_KEY = 'threshold_count_matrix'
STANDARD_FILE_NAME_KEY = 'standard_gradcam_file_name'
PMM_METADATA_KEY = 'pmm_metadata_dict'

PMM_FILE_KEYS = [
    MEAN_INPUT_MATRICES_KEY, MEAN_CLASS_ACTIVATIONS_KEY,
    MEAN_GUIDED_GRADCAM_KEY, THRESHOLD_COUNTS_KEY, STANDARD_FILE_NAME_KEY,
    PMM_METADATA_KEY
]


def _find_relevant_input_matrix(list_of_input_matrices, num_spatial_dim):
    """Finds relevant input matrix (with desired number of spatial dimensions).

    :param list_of_input_matrices: See doc for `run_gradcam`.
    :param num_spatial_dim: Desired number of spatial dimensions.
    :return: relevant_index: Array index for relevant input matrix.  If
        `relevant_index = q`, then list_of_input_matrices[q] is the relevant
        matrix.
    :raises: TypeError: if this method does not find exactly one input matrix
        with desired number of spatial dimensions.
    """

    num_spatial_dim_by_input = numpy.array(
        [len(m.shape) - 2 for m in list_of_input_matrices], dtype=int)

    these_indices = numpy.where(num_spatial_dim_by_input == num_spatial_dim)[0]
    if len(these_indices) != 1:
        error_string = (
            'Expected one input matrix with {0:d} dimensions.  Found {1:d} such'
            ' input matrices.'
        ).format(num_spatial_dim, len(these_indices))

        raise TypeError(error_string)

    return these_indices[0]


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
            list_of_input_tensors[i])

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
        1, num_rows_new, num=num_rows_new, dtype=float)
    row_indices_orig = numpy.linspace(
        1, num_rows_new, num=class_activation_matrix.shape[0], dtype=float)

    if len(new_dimensions) == 1:
        interp_object = UnivariateSpline(
            x=row_indices_orig, y=numpy.ravel(class_activation_matrix),
            k=1, s=0)

        return interp_object(row_indices_new)

    num_columns_new = new_dimensions[1]
    column_indices_new = numpy.linspace(
        1, num_columns_new, num=num_columns_new, dtype=float)
    column_indices_orig = numpy.linspace(
        1, num_columns_new, num=class_activation_matrix.shape[1],
        dtype=float)

    if len(new_dimensions) == 2:
        interp_object = RectBivariateSpline(
            x=row_indices_orig, y=column_indices_orig,
            z=class_activation_matrix, kx=1, ky=1, s=0)

        return interp_object(x=row_indices_new, y=column_indices_new, grid=True)

    num_heights_new = new_dimensions[2]
    height_indices_new = numpy.linspace(
        1, num_heights_new, num=num_heights_new, dtype=float)
    height_indices_orig = numpy.linspace(
        1, num_heights_new, num=class_activation_matrix.shape[2],
        dtype=float)

    interp_object = RegularGridInterpolator(
        points=(row_indices_orig, column_indices_orig, height_indices_orig),
        values=class_activation_matrix, method='linear')

    row_index_matrix, column_index_matrix, height_index_matrix = (
        numpy.meshgrid(row_indices_new, column_indices_new, height_indices_new)
    )
    query_point_matrix = numpy.stack(
        (row_index_matrix, column_index_matrix, height_index_matrix), axis=-1)

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

    print SEPARATOR_STRING
    new_model_dict = new_model_object.get_config()
    for this_layer_dict in new_model_dict['layers']:
        print this_layer_dict

    print SEPARATOR_STRING
    return new_model_object


def _make_saliency_function(model_object, layer_name, input_index):
    """Creates saliency function.

    :param model_object: Instance of `keras.models.Model` or
        `keras.models.Sequential`.
    :param layer_name: Numerator in gradient will be based on activations in
        this layer.
    :param input_index: Denominators in gradient will be each value in the [q]th
        input tensor to the model, where q = `input_index`.
    :return: saliency_function: Instance of `keras.backend.function`.
    """

    output_tensor = model_object.get_layer(name=layer_name).output
    filter_maxxed_output_tensor = K.max(output_tensor, axis=-1)

    if isinstance(model_object.input, list):
        list_of_input_tensors = model_object.input
    else:
        list_of_input_tensors = [model_object.input]

    list_of_saliency_tensors = K.gradients(
        K.sum(filter_maxxed_output_tensor), list_of_input_tensors[input_index])

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


def run_gradcam(model_object, list_of_input_matrices, target_class,
                target_layer_name):
    """Runs Grad-CAM.

    T = number of input tensors to the model

    :param model_object: Trained instance of `keras.models.Model` or
        `keras.models.Sequential`.
    :param list_of_input_matrices: length-T list of numpy arrays, containing
        only one example (storm object).  list_of_input_matrices[i] must have
        the same dimensions as the [i]th input tensor to the model.
    :param target_class: Activation maps will be created for this class.  Must
        be an integer in 0...(K - 1), where K = number of classes.
    :param target_layer_name: Name of target layer.  Neuron-importance weights
        will be based on activations in this layer.
    :return: class_activation_matrix: Class-activation matrix.  Dimensions of
        this numpy array will be the spatial dimensions of whichever input
        tensor feeds into the target layer.  For example, if the given input
        tensor is 2-dimensional with M rows and N columns, this array will be
        M x N.
    """

    # Check input args.
    error_checking.assert_is_string(target_layer_name)
    error_checking.assert_is_list(list_of_input_matrices)

    for q in range(len(list_of_input_matrices)):
        error_checking.assert_is_numpy_array(list_of_input_matrices[q])

        if list_of_input_matrices[q].shape[0] != 1:
            list_of_input_matrices[q] = numpy.expand_dims(
                list_of_input_matrices[q], axis=0)

    # Create loss tensor.
    output_layer_object = model_object.layers[-1].output
    num_output_neurons = output_layer_object.get_shape().as_list()[-1]

    if num_output_neurons == 1:
        error_checking.assert_is_leq(target_class, 1)

        if target_class == 1:
            loss_tensor = model_object.layers[-1].output[..., 0]
        else:
            loss_tensor = 1 - model_object.layers[-1].output[..., 0]
    else:
        error_checking.assert_is_less_than(target_class, num_output_neurons)
        loss_tensor = model_object.layers[-1].output[..., target_class]

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

    # Compute class-activation matrix.
    mean_weight_by_filter = numpy.mean(gradient_matrix, axis=(0, 1))
    class_activation_matrix = numpy.ones(
        target_layer_activation_matrix.shape[:-1])

    num_filters = len(mean_weight_by_filter)
    for m in range(num_filters):
        class_activation_matrix += (
            mean_weight_by_filter[m] * target_layer_activation_matrix[..., m]
        )

    input_index = _find_relevant_input_matrix(
        list_of_input_matrices=list_of_input_matrices,
        num_spatial_dim=len(class_activation_matrix.shape)
    )

    spatial_dimensions = numpy.array(
        list_of_input_matrices[input_index].shape[1:-1], dtype=int)
    class_activation_matrix = _upsample_cam(
        class_activation_matrix=class_activation_matrix,
        new_dimensions=spatial_dimensions)

    class_activation_matrix[class_activation_matrix < 0.] = 0.
    # denominator = numpy.maximum(numpy.max(class_activation_matrix), K.epsilon())
    # return class_activation_matrix / denominator

    return class_activation_matrix


def run_guided_gradcam(
        orig_model_object, list_of_input_matrices, target_layer_name,
        class_activation_matrix, new_model_object=None):
    """Runs guided Grad-CAM.

    M = number of rows in grid
    N = number of columns in grid
    C = number of channels

    :param orig_model_object: Original model (trained instance of
        `keras.models.Model` or `keras.models.Sequential`).
    :param list_of_input_matrices: See doc for `run_gradcam`.
    :param target_layer_name: Same.
    :param class_activation_matrix: Same.
    :param new_model_object: New model (created by `_change_backprop_function`),
        to be used for guided backprop.
    :return: ggradcam_output_matrix: M-by-N-by-C numpy array of output values.
    :return: new_model_object: See input doc.
    """

    # Check input args.
    error_checking.assert_is_string(target_layer_name)
    error_checking.assert_is_list(list_of_input_matrices)
    error_checking.assert_is_numpy_array_without_nan(class_activation_matrix)

    for q in range(len(list_of_input_matrices)):
        error_checking.assert_is_numpy_array(list_of_input_matrices[q])

        if list_of_input_matrices[q].shape[0] != 1:
            list_of_input_matrices[q] = numpy.expand_dims(
                list_of_input_matrices[q], axis=0)

    # Do the dirty work.
    if new_model_object is None:
        _register_guided_backprop()
        new_model_object = _change_backprop_function(
            model_object=orig_model_object)

    input_index = _find_relevant_input_matrix(
        list_of_input_matrices=list_of_input_matrices,
        num_spatial_dim=len(class_activation_matrix.shape)
    )

    saliency_function = _make_saliency_function(
        model_object=new_model_object, layer_name=target_layer_name,
        input_index=input_index)

    saliency_matrix = saliency_function(list_of_input_matrices + [0])[0]
    print 'Minimum saliency = {0:.4e} ... max saliency = {1:.4e}'.format(
        numpy.min(saliency_matrix), numpy.max(saliency_matrix)
    )

    ggradcam_output_matrix = saliency_matrix * class_activation_matrix[
        ..., numpy.newaxis]
    ggradcam_output_matrix = ggradcam_output_matrix[0, ...]

    # ggradcam_output_matrix = _normalize_guided_gradcam_output(
    #     ggradcam_output_matrix[0, ...])

    return ggradcam_output_matrix, new_model_object


def write_pmm_file(
        pickle_file_name, list_of_mean_input_matrices,
        mean_class_activation_matrix, mean_ggradcam_output_matrix,
        threshold_count_matrix, standard_gradcam_file_name, pmm_metadata_dict):
    """Writes mean class-activation map to Pickle file.

    This is a mean over many examples, created by PMM (probability-matched
    means).

    :param pickle_file_name: Path to output file.
    :param list_of_mean_input_matrices: Same as input `list_of_input_matrices`
        to `write_standard_file`, but without the first axis.
    :param mean_class_activation_matrix: Same as input `class_activation_matrix`
        to `write_standard_file`, but without the first axis.
    :param mean_ggradcam_output_matrix: Same as input `ggradcam_output_matrix`
        to `write_standard_file`, but without the first axis.
    :param threshold_count_matrix: See doc for
        `prob_matched_means.run_pmm_many_variables`.
    :param standard_gradcam_file_name: Path to file with standard Grad-CAM
        output (readable by `read_standard_file`).
    :param pmm_metadata_dict: Dictionary created by
        `prob_matched_means.check_input_args`.
    """

    # TODO(thunderhoser): This method currently does not deal with sounding
    # pressures.

    error_checking.assert_is_string(standard_gradcam_file_name)

    error_checking.assert_is_numpy_array_without_nan(
        mean_class_activation_matrix)
    spatial_dimensions = numpy.array(
        mean_class_activation_matrix.shape, dtype=int)

    error_checking.assert_is_numpy_array_without_nan(
        mean_ggradcam_output_matrix)
    num_channels = mean_ggradcam_output_matrix.shape[-1]

    these_expected_dim = numpy.array(
        mean_class_activation_matrix.shape + (num_channels,), dtype=int)
    error_checking.assert_is_numpy_array(
        mean_ggradcam_output_matrix, exact_dimensions=these_expected_dim)

    if threshold_count_matrix is not None:
        error_checking.assert_is_integer_numpy_array(threshold_count_matrix)
        error_checking.assert_is_geq_numpy_array(threshold_count_matrix, 0)
        error_checking.assert_is_numpy_array(
            threshold_count_matrix, exact_dimensions=spatial_dimensions)

    error_checking.assert_is_list(list_of_mean_input_matrices)
    for this_input_matrix in list_of_mean_input_matrices:
        error_checking.assert_is_numpy_array_without_nan(this_input_matrix)

    mean_gradcam_dict = {
        MEAN_INPUT_MATRICES_KEY: list_of_mean_input_matrices,
        MEAN_CLASS_ACTIVATIONS_KEY: mean_class_activation_matrix,
        MEAN_GUIDED_GRADCAM_KEY: mean_ggradcam_output_matrix,
        THRESHOLD_COUNTS_KEY: threshold_count_matrix,
        STANDARD_FILE_NAME_KEY: standard_gradcam_file_name,
        PMM_METADATA_KEY: pmm_metadata_dict
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
    mean_gradcam_dict['mean_class_activation_matrix']: Same.
    mean_gradcam_dict['mean_ggradcam_output_matrix']: Same
    mean_gradcam_dict['threshold_count_matrix']: Same.
    mean_gradcam_dict['standard_gradcam_file_name']: Same.
    mean_gradcam_dict['pmm_metadata_dict']: Same.

    :raises: ValueError: if any of the aforelisted keys are missing from the
        dictionary.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    mean_gradcam_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    missing_keys = list(set(PMM_FILE_KEYS) - set(mean_gradcam_dict.keys()))
    if len(missing_keys) == 0:
        return mean_gradcam_dict

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), pickle_file_name)

    raise ValueError(error_string)


def write_standard_file(
        pickle_file_name, list_of_input_matrices, class_activation_matrix,
        ggradcam_output_matrix, model_file_name, storm_ids,
        storm_times_unix_sec, target_class, target_layer_name,
        sounding_pressure_matrix_pascals=None):
    """Writes class-activation maps (one for each example) to Pickle file.

    T = number of input tensors to the model
    E = number of examples (storm objects)
    H = number of height levels per sounding

    :param pickle_file_name: Path to output file.
    :param list_of_input_matrices: length-T list of numpy arrays, containing
        denormalized input data for all examples.  The first axis of each array
        must have length E.
    :param class_activation_matrix: numpy array, where
        class_activation_matrix[i, ...] is the activation matrix for the [i]th
        example, created by `run_gradcam`.
    :param ggradcam_output_matrix: numpy array, where
        ggradcam_output_matrix[i, ...] contains output of guided Grad-CAM for
        the [i]th example, created by `run_guided_gradcam`.
    :param model_file_name: Path to file with trained CNN (readable by
        `cnn.read_model`).
    :param storm_ids: length-E list of storm IDs (strings).
    :param storm_times_unix_sec: length-E numpy array of storm times.
    :param target_class: See doc for `run_gradcam`.
    :param target_layer_name: Same.
    :param sounding_pressure_matrix_pascals: E-by-H numpy array of pressure
        levels in soundings.  Useful when model input contains soundings with no
        pressure variable, since pressure is needed to plot soundings.
    :raises: ValueError: if `list_of_input_matrices` and
        `class_activation_matrix` have non-matching dimensions.
    """

    error_checking.assert_is_string(model_file_name)
    error_checking.assert_is_integer(target_class)
    error_checking.assert_is_geq(target_class, 0)
    error_checking.assert_is_string(target_layer_name)

    error_checking.assert_is_string_list(storm_ids)
    error_checking.assert_is_numpy_array(
        numpy.array(storm_ids), num_dimensions=1)

    num_storm_objects = len(storm_ids)
    error_checking.assert_is_integer_numpy_array(storm_times_unix_sec)
    error_checking.assert_is_numpy_array(
        storm_times_unix_sec, exact_dimensions=numpy.array([num_storm_objects])
    )

    error_checking.assert_is_numpy_array_without_nan(class_activation_matrix)
    error_checking.assert_is_geq(len(class_activation_matrix.shape), 2)

    these_expected_dim = numpy.array(
        (num_storm_objects,) + class_activation_matrix.shape[1:], dtype=int)
    error_checking.assert_is_numpy_array(
        class_activation_matrix, exact_dimensions=these_expected_dim)

    error_checking.assert_is_numpy_array_without_nan(ggradcam_output_matrix)
    num_channels = ggradcam_output_matrix.shape[-1]

    these_expected_dim = numpy.array(
        class_activation_matrix.shape + (num_channels,), dtype=int)
    error_checking.assert_is_numpy_array(
        ggradcam_output_matrix, exact_dimensions=these_expected_dim)

    error_checking.assert_is_list(list_of_input_matrices)

    for this_input_matrix in list_of_input_matrices:
        error_checking.assert_is_numpy_array_without_nan(this_input_matrix)

        these_expected_dim = numpy.array(
            (num_storm_objects,) + this_input_matrix.shape[1:], dtype=int)
        error_checking.assert_is_numpy_array(
            this_input_matrix, exact_dimensions=these_expected_dim)

    if sounding_pressure_matrix_pascals is not None:
        error_checking.assert_is_numpy_array(
            sounding_pressure_matrix_pascals, num_dimensions=2)

        these_expected_dim = numpy.array(
            (num_storm_objects,) + sounding_pressure_matrix_pascals.shape[1:],
            dtype=int)
        error_checking.assert_is_numpy_array(
            sounding_pressure_matrix_pascals,
            exact_dimensions=these_expected_dim)

    gradcam_dict = {
        INPUT_MATRICES_KEY: list_of_input_matrices,
        CLASS_ACTIVATIONS_KEY: class_activation_matrix,
        GUIDED_GRADCAM_KEY: ggradcam_output_matrix,
        MODEL_FILE_NAME_KEY: model_file_name,
        STORM_IDS_KEY: storm_ids,
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
    gradcam_dict['class_activation_matrix']: Same.
    gradcam_dict['ggradcam_output_matrix']: Same.
    gradcam_dict['model_file_name']: Same.
    gradcam_dict['storm_ids']: Same.
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
