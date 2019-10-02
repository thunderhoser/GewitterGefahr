"""Methods for creating saliency maps."""

import pickle
import numpy
from keras import backend as K
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import model_interpretation

TOLERANCE = 1e-6
DEFAULT_IDEAL_ACTIVATION = 2.

PREDICTOR_MATRICES_KEY = model_interpretation.PREDICTOR_MATRICES_KEY
SALIENCY_MATRICES_KEY = 'saliency_matrices'
MODEL_FILE_KEY = model_interpretation.MODEL_FILE_KEY
FULL_STORM_IDS_KEY = model_interpretation.FULL_STORM_IDS_KEY
STORM_TIMES_KEY = model_interpretation.STORM_TIMES_KEY
COMPONENT_TYPE_KEY = 'component_type_string'
TARGET_CLASS_KEY = 'target_class'
LAYER_NAME_KEY = 'layer_name'
IDEAL_ACTIVATION_KEY = 'ideal_activation'
NEURON_INDICES_KEY = 'neuron_indices'
CHANNEL_INDEX_KEY = 'channel_index'
SOUNDING_PRESSURES_KEY = model_interpretation.SOUNDING_PRESSURES_KEY

STANDARD_FILE_KEYS = [
    PREDICTOR_MATRICES_KEY, SALIENCY_MATRICES_KEY,
    FULL_STORM_IDS_KEY, STORM_TIMES_KEY, MODEL_FILE_KEY,
    COMPONENT_TYPE_KEY, TARGET_CLASS_KEY, LAYER_NAME_KEY, IDEAL_ACTIVATION_KEY,
    NEURON_INDICES_KEY, CHANNEL_INDEX_KEY,
    SOUNDING_PRESSURES_KEY
]

MEAN_PREDICTOR_MATRICES_KEY = model_interpretation.MEAN_PREDICTOR_MATRICES_KEY
MEAN_SALIENCY_MATRICES_KEY = 'mean_saliency_matrices'
NON_PMM_FILE_KEY = model_interpretation.NON_PMM_FILE_KEY
PMM_MAX_PERCENTILE_KEY = model_interpretation.PMM_MAX_PERCENTILE_KEY
MEAN_SOUNDING_PRESSURES_KEY = model_interpretation.MEAN_SOUNDING_PRESSURES_KEY

PMM_FILE_KEYS = [
    MEAN_PREDICTOR_MATRICES_KEY, MEAN_SALIENCY_MATRICES_KEY, MODEL_FILE_KEY,
    NON_PMM_FILE_KEY, PMM_MAX_PERCENTILE_KEY, MEAN_SOUNDING_PRESSURES_KEY
]


def _check_in_and_out_matrices(
        predictor_matrices, num_examples=None, saliency_matrices=None):
    """Error-checks input and output matrices.

    T = number of input tensors to the model
    E = number of examples (storm objects)

    :param predictor_matrices: length-T list of predictor matrices.  Each item
        must be a numpy array.
    :param num_examples: E in the above discussion.  The first axis of each
        array must have length E.  If you don't know the number of examples,
        leave this as None.
    :param saliency_matrices: Same as `predictor_matrices` but with saliency
        values.
    :raises: ValueError: if `predictor_matrices` and `saliency_matrices` have
        different lengths.
    """

    error_checking.assert_is_list(predictor_matrices)
    num_matrices = len(predictor_matrices)

    if saliency_matrices is None:
        saliency_matrices = [None] * num_matrices

    error_checking.assert_is_list(saliency_matrices)
    num_saliency_matrices = len(saliency_matrices)

    if num_matrices != num_saliency_matrices:
        error_string = (
            'Number of predictor matrices ({0:d}) should = number of saliency '
            'matrices ({1:d}).'
        ).format(num_matrices, num_saliency_matrices)

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

        if saliency_matrices[i] is not None:
            error_checking.assert_is_numpy_array_without_nan(
                saliency_matrices[i]
            )

            these_expected_dim = numpy.array(
                predictor_matrices[i].shape[:-1], dtype=int
            )
            error_checking.assert_is_numpy_array(
                saliency_matrices[i], exact_dimensions=these_expected_dim
            )


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
                (model_object.layers[-1].output[..., 0] - 1) ** 2
            )
        else:
            loss_tensor = K.mean(model_object.layers[-1].output[..., 0] ** 2)
    else:
        error_checking.assert_is_less_than(target_class, num_output_neurons)
        loss_tensor = K.mean(
            (model_object.layers[-1].output[..., target_class] - 1) ** 2
        )

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
                    0, ..., neuron_indices
                ]
            )
            * model_object.get_layer(name=layer_name).output[
                0, ..., neuron_indices
            ] ** 2
        )
    else:
        loss_tensor = (
            model_object.get_layer(name=layer_name).output[
                0, ..., neuron_indices
            ]
            - ideal_activation
        ) ** 2

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
                0, ..., channel_index
            ]
        ))
    else:
        error_checking.assert_is_greater(ideal_activation, 0.)
        loss_tensor = K.abs(
            stat_function_for_neuron_activations(
                model_object.get_layer(name=layer_name).output[
                    0, ..., channel_index]
            )
            - ideal_activation
        )

    return _do_saliency_calculations(
        model_object=model_object, loss_tensor=loss_tensor,
        list_of_input_matrices=list_of_input_matrices)


def write_standard_file(
        pickle_file_name, denorm_predictor_matrices, saliency_matrices,
        full_storm_id_strings, storm_times_unix_sec, model_file_name,
        metadata_dict, sounding_pressure_matrix_pa=None):
    """Writes saliency maps (one per storm object) to Pickle file.

    E = number of examples (storm objects)
    H = number of sounding heights

    :param pickle_file_name: Path to output file.
    :param denorm_predictor_matrices: See doc for `_check_in_and_out_matrices`.
    :param saliency_matrices: Same.
    :param full_storm_id_strings: length-E list of storm IDs.
    :param storm_times_unix_sec: length-E numpy array of storm times.
    :param model_file_name: Path to model that created saliency maps (readable
        by `cnn.read_model`).
    :param metadata_dict: Dictionary created by `check_metadata`.
    :param sounding_pressure_matrix_pa: E-by-H numpy array of pressure
        levels.  Needed only if the model is trained with soundings but without
        pressure as a predictor.
    """

    error_checking.assert_is_string(model_file_name)
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
        saliency_matrices=saliency_matrices)

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

    saliency_dict = {
        PREDICTOR_MATRICES_KEY: denorm_predictor_matrices,
        SALIENCY_MATRICES_KEY: saliency_matrices,
        FULL_STORM_IDS_KEY: full_storm_id_strings,
        STORM_TIMES_KEY: storm_times_unix_sec,
        MODEL_FILE_KEY: model_file_name,
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
    pickle.dump(saliency_dict, pickle_file_handle)
    pickle_file_handle.close()


def write_pmm_file(
        pickle_file_name, mean_denorm_predictor_matrices,
        mean_saliency_matrices, model_file_name, non_pmm_file_name,
        pmm_max_percentile_level, mean_sounding_pressures_pa=None):
    """Writes composite saliency map to Pickle file.

    The composite should be created by probability-matched means (PMM).

    H = number of sounding heights

    :param pickle_file_name: Path to output file.
    :param mean_denorm_predictor_matrices: See doc for
        `_check_in_and_out_matrices`.
    :param mean_saliency_matrices: Same.
    :param model_file_name: Path to model that created saliency maps (readable
        by `cnn.read_model`).
    :param non_pmm_file_name: Path to standard saliency file (containing
        non-composited saliency maps).
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
        saliency_matrices=mean_saliency_matrices)

    if mean_sounding_pressures_pa is not None:
        num_heights = mean_denorm_predictor_matrices[-1].shape[-2]
        these_expected_dim = numpy.array([num_heights], dtype=int)

        error_checking.assert_is_geq_numpy_array(mean_sounding_pressures_pa, 0.)
        error_checking.assert_is_numpy_array(
            mean_sounding_pressures_pa, exact_dimensions=these_expected_dim)

    mean_saliency_dict = {
        MEAN_PREDICTOR_MATRICES_KEY: mean_denorm_predictor_matrices,
        MEAN_SALIENCY_MATRICES_KEY: mean_saliency_matrices,
        MODEL_FILE_KEY: model_file_name,
        NON_PMM_FILE_KEY: non_pmm_file_name,
        PMM_MAX_PERCENTILE_KEY: pmm_max_percentile_level,
        MEAN_SOUNDING_PRESSURES_KEY: mean_sounding_pressures_pa
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(mean_saliency_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_file(pickle_file_name):
    """Reads composite or non-composite saliency maps from Pickle file.

    :param pickle_file_name: Path to input file (created by
        `write_standard_file` or `write_pmm_file`).
    :return: saliency_dict: Has the following keys if not a composite...
    saliency_dict['denorm_predictor_matrices']: See doc for
        `write_standard_file`.
    saliency_dict['saliency_matrices']: Same.
    saliency_dict['full_storm_id_strings']: Same.
    saliency_dict['storm_times_unix_sec']: Same.
    saliency_dict['model_file_name']: Same.
    saliency_dict['component_type_string']: Same.
    saliency_dict['target_class']: Same.
    saliency_dict['layer_name']: Same.
    saliency_dict['ideal_activation']: Same.
    saliency_dict['neuron_indices']: Same.
    saliency_dict['channel_index']: Same.
    saliency_dict['sounding_pressure_matrix_pa']: Same.

    ...or the following keys if composite...

    saliency_dict['mean_denorm_predictor_matrices']: See doc for
        `write_pmm_file`.
    saliency_dict['mean_saliency_matrices']: Same.
    saliency_dict['model_file_name']: Same.
    saliency_dict['non_pmm_file_name']: Same.
    saliency_dict['pmm_max_percentile_level']: Same.
    saliency_dict['mean_sounding_pressures_pa']: Same.

    :return: pmm_flag: Boolean flag.  True if `saliency_dict` contains
        composite, False otherwise.

    :raises: ValueError: if dictionary does not contain expected keys.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    saliency_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    pmm_flag = MEAN_PREDICTOR_MATRICES_KEY in saliency_dict

    if pmm_flag:
        missing_keys = list(
            set(PMM_FILE_KEYS) - set(saliency_dict.keys())
        )
    else:
        missing_keys = list(
            set(STANDARD_FILE_KEYS) - set(saliency_dict.keys())
        )

    if len(missing_keys) == 0:
        return saliency_dict, pmm_flag

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), pickle_file_name)

    raise ValueError(error_string)
