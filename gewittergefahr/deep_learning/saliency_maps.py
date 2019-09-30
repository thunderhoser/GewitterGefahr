"""Methods for creating saliency maps."""

import pickle
import numpy
from keras import backend as K
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import monte_carlo
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import model_interpretation

TOLERANCE = 1e-6
DEFAULT_IDEAL_ACTIVATION = 2.

INPUT_MATRICES_KEY = 'list_of_input_matrices'
SALIENCY_MATRICES_KEY = 'list_of_saliency_matrices'

FULL_IDS_KEY = tracking_io.FULL_IDS_KEY
STORM_TIMES_KEY = tracking_io.STORM_TIMES_KEY
MODEL_FILE_KEY = model_interpretation.MODEL_FILE_KEY
COMPONENT_TYPE_KEY = 'component_type_string'
TARGET_CLASS_KEY = 'target_class'
LAYER_NAME_KEY = 'layer_name'
IDEAL_ACTIVATION_KEY = 'ideal_activation'
NEURON_INDICES_KEY = 'neuron_indices'
CHANNEL_INDEX_KEY = 'channel_index'
SOUNDING_PRESSURES_KEY = 'sounding_pressure_matrix_pascals'

STANDARD_FILE_KEYS = [
    INPUT_MATRICES_KEY, SALIENCY_MATRICES_KEY, FULL_IDS_KEY, STORM_TIMES_KEY,
    MODEL_FILE_KEY, COMPONENT_TYPE_KEY, TARGET_CLASS_KEY, LAYER_NAME_KEY,
    IDEAL_ACTIVATION_KEY, NEURON_INDICES_KEY, CHANNEL_INDEX_KEY,
    SOUNDING_PRESSURES_KEY
]

MEAN_INPUT_MATRICES_KEY = model_interpretation.MEAN_INPUT_MATRICES_KEY
MEAN_SALIENCY_MATRICES_KEY = 'list_of_mean_saliency_matrices'
STANDARD_FILE_NAME_KEY = 'standard_saliency_file_name'
PMM_METADATA_KEY = 'pmm_metadata_dict'
MONTE_CARLO_DICT_KEY = 'monte_carlo_dict'
MEAN_SOUNDING_PRESSURES_KEY = 'mean_sounding_pressures_pascals'

PMM_FILE_KEYS = [
    MEAN_INPUT_MATRICES_KEY, MEAN_SALIENCY_MATRICES_KEY, MODEL_FILE_KEY,
    STANDARD_FILE_NAME_KEY, PMM_METADATA_KEY, MONTE_CARLO_DICT_KEY,
    MEAN_SOUNDING_PRESSURES_KEY
]


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
            K.std(list_of_gradient_tensors[i]), K.epsilon()
        )

    inputs_to_gradients_function = K.function(
        list_of_input_tensors + [K.learning_phase()], list_of_gradient_tensors
    )

    # list_of_saliency_matrices = None
    # num_examples = list_of_input_matrices[0].shape[0]
    #
    # for i in range(num_examples):
    #     these_input_matrices = [a[[i], ...] for a in list_of_input_matrices]
    #     these_saliency_matrices = inputs_to_gradients_function(
    #         these_input_matrices + [0])
    #
    #     if list_of_saliency_matrices is None:
    #         list_of_saliency_matrices = these_saliency_matrices + []
    #     else:
    #         for i in range(num_input_tensors):
    #             list_of_saliency_matrices[i] = numpy.concatenate(
    #                 (list_of_saliency_matrices[i], these_saliency_matrices[i]),
    #                 axis=0)

    list_of_saliency_matrices = inputs_to_gradients_function(
        list_of_input_matrices + [0]
    )

    for i in range(num_input_tensors):
        list_of_saliency_matrices[i] *= -1

    return list_of_saliency_matrices


def check_metadata(
        component_type_string, target_class=None, layer_name=None,
        ideal_activation=None, neuron_indices=None, channel_index=None):
    """Error-checks metadata for saliency calculations.

    :param component_type_string: Component type (must be accepted by
        `model_interpretation.check_component_type`).
    :param target_class: See doc for `get_saliency_maps_for_class_activation`.
    :param layer_name: See doc for `get_saliency_maps_for_neuron_activation` or
        `get_saliency_maps_for_channel_activation`.
    :param ideal_activation: Same.
    :param neuron_indices: See doc for
        `get_saliency_maps_for_neuron_activation`.
    :param channel_index: See doc for `get_saliency_maps_for_class_activation`.

    :return: metadata_dict: Dictionary with the following keys.
    metadata_dict['component_type_string']: See input doc.
    metadata_dict['target_class']: Same.
    metadata_dict['layer_name']: Same.
    metadata_dict['ideal_activation']: Same.
    metadata_dict['neuron_indices']: Same.
    metadata_dict['channel_index']: Same.
    """

    model_interpretation.check_component_type(component_type_string)

    if (component_type_string ==
            model_interpretation.CLASS_COMPONENT_TYPE_STRING):
        error_checking.assert_is_integer(target_class)
        error_checking.assert_is_geq(target_class, 0)

    if component_type_string in [
            model_interpretation.NEURON_COMPONENT_TYPE_STRING,
            model_interpretation.CHANNEL_COMPONENT_TYPE_STRING
    ]:
        error_checking.assert_is_string(layer_name)
        if ideal_activation is not None:
            error_checking.assert_is_greater(ideal_activation, 0.)

    if (component_type_string ==
            model_interpretation.NEURON_COMPONENT_TYPE_STRING):
        error_checking.assert_is_integer_numpy_array(neuron_indices)
        error_checking.assert_is_geq_numpy_array(neuron_indices, 0)
        error_checking.assert_is_numpy_array(neuron_indices, num_dimensions=1)

    if (component_type_string ==
            model_interpretation.CHANNEL_COMPONENT_TYPE_STRING):
        error_checking.assert_is_integer(channel_index)
        error_checking.assert_is_geq(channel_index, 0)

    return {
        COMPONENT_TYPE_KEY: component_type_string,
        TARGET_CLASS_KEY: target_class,
        LAYER_NAME_KEY: layer_name,
        IDEAL_ACTIVATION_KEY: ideal_activation,
        NEURON_INDICES_KEY: neuron_indices,
        CHANNEL_INDEX_KEY: channel_index
    }


def get_saliency_maps_for_class_activation(
        model_object, target_class, list_of_input_matrices):
    """For each input example, creates saliency map for prob of target class.

    :param model_object: Instance of `keras.models.Model`.
    :param target_class: Saliency maps will be created for this class.  Must be
        an integer in 0...(K - 1), where K = number of classes.
    :param list_of_input_matrices: See doc for `_do_saliency_calculations`.
    :return: list_of_saliency_matrices: See doc for `_do_saliency_calculations`.
    """

    check_metadata(
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
            loss_tensor = K.mean(model_object.layers[-1].output[..., 0] ** 2)
    else:
        error_checking.assert_is_less_than(target_class, num_output_neurons)
        loss_tensor = K.mean(
            (model_object.layers[-1].output[..., target_class] - 1) ** 2)

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
        neuron_indices=neuron_indices)

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
        channel_index=channel_index
    )

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


def write_pmm_file(
        pickle_file_name, list_of_mean_input_matrices,
        list_of_mean_saliency_matrices, model_file_name,
        standard_saliency_file_name, pmm_metadata_dict,
        mean_sounding_pressures_pascals=None, monte_carlo_dict=None):
    """Writes mean saliency map to Pickle file.

    This is a mean over many examples, created by PMM (probability-matched
    means).

    H_s = number of sounding heights

    :param pickle_file_name: Path to output file.
    :param list_of_mean_input_matrices: Same as input `list_of_input_matrices`
        to `write_standard_file`, but without the first axis.
    :param list_of_mean_saliency_matrices: Same as input
        `list_of_saliency_matrices` to `write_standard_file`, but without the
        first axis.
    :param model_file_name: Path to file with trained model (readable by
        `cnn.read_model`).
    :param standard_saliency_file_name: Path to file with standard saliency
        output (readable by `read_standard_file`).
    :param pmm_metadata_dict: Dictionary created by
        `prob_matched_means.check_input_args`.
    :param mean_sounding_pressures_pascals:
        [used only if `list_of_mean_input_matrices` contains soundings]
        numpy array (length H_s) of mean pressures.
    :param monte_carlo_dict: Dictionary with results of Monte Carlo significance
        test.  Must contain keys listed in `monte_carlo.check_output`, plus the
        following.
    monte_carlo_dict['baseline_file_name']: Path to saliency file for baseline
        set (readable by `read_standard_file`), against which the trial set
        (contained in `standard_saliency_file_name`) was compared.

    :raises: ValueError: if `list_of_mean_input_matrices` and
        `list_of_mean_saliency_matrices` have different lengths.
    """

    error_checking.assert_is_string(model_file_name)
    error_checking.assert_is_string(standard_saliency_file_name)
    error_checking.assert_is_list(list_of_mean_input_matrices)
    error_checking.assert_is_list(list_of_mean_saliency_matrices)

    num_input_matrices = len(list_of_mean_input_matrices)
    num_saliency_matrices = len(list_of_mean_saliency_matrices)

    if num_input_matrices != num_saliency_matrices:
        error_string = (
            'Number of input matrices ({0:d}) should equal number of saliency '
            'matrices ({1:d}).'
        ).format(num_input_matrices, num_saliency_matrices)

        raise ValueError(error_string)

    for i in range(num_input_matrices):
        error_checking.assert_is_numpy_array_without_nan(
            list_of_mean_input_matrices[i]
        )
        error_checking.assert_is_numpy_array_without_nan(
            list_of_mean_saliency_matrices[i]
        )

        these_expected_dim = numpy.array(
            list_of_mean_input_matrices[i].shape, dtype=int
        )
        error_checking.assert_is_numpy_array(
            list_of_mean_saliency_matrices[i],
            exact_dimensions=these_expected_dim
        )

    if mean_sounding_pressures_pascals is not None:
        num_sounding_heights = list_of_mean_input_matrices[-1].shape[-2]
        these_expected_dim = numpy.array([num_sounding_heights], dtype=int)

        error_checking.assert_is_geq_numpy_array(
            mean_sounding_pressures_pascals, 0.)
        error_checking.assert_is_numpy_array(
            mean_sounding_pressures_pascals, exact_dimensions=these_expected_dim
        )

    if monte_carlo_dict is not None:
        monte_carlo.check_output(monte_carlo_dict)
        error_checking.assert_is_string(
            monte_carlo_dict[monte_carlo.BASELINE_FILE_KEY]
        )

        for i in range(num_input_matrices):
            assert numpy.allclose(
                list_of_mean_saliency_matrices[i],
                monte_carlo_dict[monte_carlo.TRIAL_PMM_MATRICES_KEY][i],
                atol=TOLERANCE
            )

    mean_saliency_dict = {
        MEAN_INPUT_MATRICES_KEY: list_of_mean_input_matrices,
        MEAN_SALIENCY_MATRICES_KEY: list_of_mean_saliency_matrices,
        MODEL_FILE_KEY: model_file_name,
        STANDARD_FILE_NAME_KEY: standard_saliency_file_name,
        PMM_METADATA_KEY: pmm_metadata_dict,
        MEAN_SOUNDING_PRESSURES_KEY: mean_sounding_pressures_pascals,
        MONTE_CARLO_DICT_KEY: monte_carlo_dict
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(mean_saliency_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_pmm_file(pickle_file_name):
    """Reads mean saliency map from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: mean_saliency_dict: Dictionary with the following keys.
    mean_saliency_dict['list_of_mean_input_matrices']: See doc for
        `write_pmm_file`.
    mean_saliency_dict['list_of_mean_saliency_matrices']: Same.
    mean_saliency_dict['model_file_name']: Same.
    mean_saliency_dict['standard_saliency_file_name']: Same.
    mean_saliency_dict['pmm_metadata_dict']: Same.
    mean_saliency_dict['mean_sounding_pressures_pascals']: Same.
    mean_saliency_dict['monte_carlo_dict']: Same.

    :raises: ValueError: if any of the aforelisted keys are missing from the
        dictionary.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    mean_saliency_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    if MONTE_CARLO_DICT_KEY not in mean_saliency_dict:
        mean_saliency_dict[MONTE_CARLO_DICT_KEY] = None

    missing_keys = list(set(PMM_FILE_KEYS) - set(mean_saliency_dict.keys()))
    if len(missing_keys) == 0:
        return mean_saliency_dict

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), pickle_file_name)

    raise ValueError(error_string)


def write_standard_file(
        pickle_file_name, list_of_input_matrices, list_of_saliency_matrices,
        full_id_strings, storm_times_unix_sec, model_file_name,
        saliency_metadata_dict, sounding_pressure_matrix_pascals=None):
    """Writes saliency maps (one per example) to Pickle file.

    T = number of input tensors to the model
    E = number of examples (storm objects)
    H = number of height levels per sounding

    :param pickle_file_name: Path to output file.
    :param list_of_input_matrices: length-T list of numpy arrays, containing
        predictors (inputs to the model).  The first dimension of each array
        must have length E.
    :param list_of_saliency_matrices: length-T list of numpy arrays, containing
        saliency values.  list_of_saliency_matrices[i] must have the same
        dimensions as list_of_input_matrices[i].
    :param full_id_strings: length-E list of full storm IDs.
    :param storm_times_unix_sec: length-E numpy array of storm times.
    :param model_file_name: Path to file with trained model (readable by
        `cnn.read_model`).
    :param saliency_metadata_dict: Dictionary created by `check_metadata`.
    :param sounding_pressure_matrix_pascals: E-by-H numpy array of pressure
        levels in soundings.  Useful only when the model input contains
        soundings with no pressure, because it is needed to plot soundings.
    :raises: ValueError: if `list_of_input_matrices` and
        `list_of_saliency_matrices` have different lengths.
    """

    error_checking.assert_is_string(model_file_name)
    error_checking.assert_is_string_list(full_id_strings)
    error_checking.assert_is_numpy_array(
        numpy.array(full_id_strings), num_dimensions=1)

    num_storm_objects = len(full_id_strings)
    these_expected_dim = numpy.array([num_storm_objects], dtype=int)

    error_checking.assert_is_integer_numpy_array(storm_times_unix_sec)
    error_checking.assert_is_numpy_array(
        storm_times_unix_sec, exact_dimensions=these_expected_dim)

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
        error_checking.assert_is_numpy_array_without_nan(
            list_of_input_matrices[i]
        )
        error_checking.assert_is_numpy_array_without_nan(
            list_of_saliency_matrices[i]
        )

        these_expected_dim = numpy.array(
            (num_storm_objects,) + list_of_input_matrices[i].shape[1:],
            dtype=int
        )
        error_checking.assert_is_numpy_array(
            list_of_input_matrices[i], exact_dimensions=these_expected_dim
        )

        these_expected_dim = numpy.array(
            list_of_input_matrices[i].shape, dtype=int
        )
        error_checking.assert_is_numpy_array(
            list_of_saliency_matrices[i], exact_dimensions=these_expected_dim
        )

    if sounding_pressure_matrix_pascals is not None:
        error_checking.assert_is_numpy_array_without_nan(
            sounding_pressure_matrix_pascals)
        error_checking.assert_is_greater_numpy_array(
            sounding_pressure_matrix_pascals, 0.)
        error_checking.assert_is_numpy_array(
            sounding_pressure_matrix_pascals, num_dimensions=2)

        these_expected_dim = numpy.array(
            (num_storm_objects,) + sounding_pressure_matrix_pascals.shape[1:],
            dtype=int
        )
        error_checking.assert_is_numpy_array(
            sounding_pressure_matrix_pascals,
            exact_dimensions=these_expected_dim)

    saliency_dict = {
        INPUT_MATRICES_KEY: list_of_input_matrices,
        SALIENCY_MATRICES_KEY: list_of_saliency_matrices,
        FULL_IDS_KEY: full_id_strings,
        STORM_TIMES_KEY: storm_times_unix_sec,
        MODEL_FILE_KEY: model_file_name,
        COMPONENT_TYPE_KEY: saliency_metadata_dict[COMPONENT_TYPE_KEY],
        TARGET_CLASS_KEY: saliency_metadata_dict[TARGET_CLASS_KEY],
        LAYER_NAME_KEY: saliency_metadata_dict[LAYER_NAME_KEY],
        IDEAL_ACTIVATION_KEY: saliency_metadata_dict[IDEAL_ACTIVATION_KEY],
        NEURON_INDICES_KEY: saliency_metadata_dict[NEURON_INDICES_KEY],
        CHANNEL_INDEX_KEY: saliency_metadata_dict[CHANNEL_INDEX_KEY],
        SOUNDING_PRESSURES_KEY: sounding_pressure_matrix_pascals
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(saliency_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_standard_file(pickle_file_name):
    """Reads saliency maps (one per example) from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: saliency_dict: Dictionary with the following keys.
    saliency_dict['list_of_input_matrices']: See doc for `write_standard_file`.
    saliency_dict['list_of_saliency_matrices']: Same.
    saliency_dict['full_id_strings']: Same.
    saliency_dict['storm_times_unix_sec']: Same.
    saliency_dict['model_file_name']: Same.
    saliency_dict['component_type_string']: Same.
    saliency_dict['target_class']: Same.
    saliency_dict['layer_name']: Same.
    saliency_dict['ideal_activation']: Same.
    saliency_dict['neuron_indices']: Same.
    saliency_dict['channel_index']: Same.
    saliency_dict['sounding_pressure_matrix_pascals']: Same.

    :raises: ValueError: if any of the aforelisted keys are missing from the
        dictionary.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    saliency_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    missing_keys = list(set(STANDARD_FILE_KEYS) - set(saliency_dict.keys()))
    if len(missing_keys) == 0:
        return saliency_dict

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), pickle_file_name)

    raise ValueError(error_string)
