"""Deals with physical constraints.

Currently used only for backwards optimization, but I may find other uses.
"""

import numpy
from keras import backend as K
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import input_examples
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.deep_learning import training_validation_io as trainval_io

RADAR_TO_MINIMUM_DICT = {
    radar_utils.ECHO_TOP_18DBZ_NAME: 0.,        # km
    radar_utils.ECHO_TOP_40DBZ_NAME: 0.,        # km
    radar_utils.ECHO_TOP_50DBZ_NAME: 0.,        # km
    radar_utils.MESH_NAME: 0.,                  # mm
    radar_utils.REFL_NAME: 0.,                  # dBZ
    radar_utils.REFL_COLUMN_MAX_NAME: 0.,       # dBZ
    radar_utils.REFL_0CELSIUS_NAME: 0.,         # dBZ
    radar_utils.REFL_M10CELSIUS_NAME: 0.,       # dBZ
    radar_utils.REFL_M20CELSIUS_NAME: 0.,       # dBZ
    radar_utils.REFL_LOWEST_ALTITUDE_NAME: 0.,  # dBZ
    radar_utils.SHI_NAME: 0.,                   # unitless
    radar_utils.VIL_NAME: 0.,                   # mm
    radar_utils.CORRELATION_COEFF_NAME: 0.,     # unitless
    radar_utils.SPECTRUM_WIDTH_NAME: 0.,        # m s^-1
}

RADAR_TO_MAX_DICT = {
    radar_utils.CORRELATION_COEFF_NAME: 1.      # unitless
}

SOUNDING_TO_MINIMUM_DICT = {
    soundings.PRESSURE_NAME: 0.,                       # Pascals
    soundings.TEMPERATURE_NAME: 0.,                    # Kelvins
    soundings.VIRTUAL_POTENTIAL_TEMPERATURE_NAME: 0.,  # Kelvins
    soundings.SPECIFIC_HUMIDITY_NAME: 0.,              # kg kg^-1
    soundings.RELATIVE_HUMIDITY_NAME: 0.               # unitless
}

SOUNDING_TO_MAX_DICT = {
    soundings.SPECIFIC_HUMIDITY_NAME: 1.,              # kg kg^-1
    soundings.RELATIVE_HUMIDITY_NAME: 1.               # unitless
}


def __normalize_minmax_for_radar_field(
        field_name, cnn_metadata_dict, test_mode=False,
        normalization_table=None):
    """Converts min and max for radar field to normalized units.

    :param field_name: Name of radar field (must be accepted by
        `radar_utils.check_field_name`).
    :param cnn_metadata_dict: See doc for `_normalize_minima_and_maxima`.
    :param test_mode: Same.
    :param normalization_table: Same.
    :return: min_normalized_value: Minimum value in normalized units.
    :return: max_normalized_value: Max value in normalized units.
    """

    training_option_dict = cnn_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    normalization_type_string = training_option_dict[
        trainval_io.NORMALIZATION_TYPE_KEY]
    min_normalized_value = training_option_dict[
        trainval_io.MIN_NORMALIZED_VALUE_KEY]
    max_normalized_value = training_option_dict[
        trainval_io.MAX_NORMALIZED_VALUE_KEY]

    if test_mode:
        normalization_file_name = None
    else:
        normalization_file_name = training_option_dict[
            trainval_io.NORMALIZATION_FILE_KEY]

    if field_name in RADAR_TO_MINIMUM_DICT:
        if normalization_type_string == dl_utils.Z_NORMALIZATION_TYPE_STRING:
            min_physical_value = RADAR_TO_MINIMUM_DICT[field_name]
            dummy_radar_matrix = numpy.full((1, 1, 1, 1), min_physical_value)

            dummy_radar_matrix = dl_utils.normalize_radar_images(
                radar_image_matrix=dummy_radar_matrix, field_names=[field_name],
                normalization_type_string=normalization_type_string,
                normalization_param_file_name=normalization_file_name,
                min_normalized_value=min_normalized_value,
                max_normalized_value=max_normalized_value,
                test_mode=test_mode, normalization_table=normalization_table)

            min_normalized_value = numpy.ravel(dummy_radar_matrix)[0]
    else:
        min_normalized_value = numpy.nan

    if field_name in RADAR_TO_MAX_DICT:
        if normalization_type_string == dl_utils.Z_NORMALIZATION_TYPE_STRING:
            max_physical_value = RADAR_TO_MAX_DICT[field_name]
            dummy_radar_matrix = numpy.full((1, 1, 1, 1), max_physical_value)

            dummy_radar_matrix = dl_utils.normalize_radar_images(
                radar_image_matrix=dummy_radar_matrix, field_names=[field_name],
                normalization_type_string=normalization_type_string,
                normalization_param_file_name=normalization_file_name,
                min_normalized_value=min_normalized_value,
                max_normalized_value=max_normalized_value,
                test_mode=test_mode, normalization_table=normalization_table)

            max_normalized_value = numpy.ravel(dummy_radar_matrix)[0]
    else:
        max_normalized_value = numpy.nan

    return min_normalized_value, max_normalized_value


def __normalize_minmax_for_sounding_field(
        field_name, cnn_metadata_dict, test_mode=False,
        normalization_table=None):
    """Converts min and max for sounding field to normalized units.

    :param field_name: Name of sounding field (must be accepted by
        `soundings.check_field_name`).
    :param cnn_metadata_dict: See doc for `_normalize_minima_and_maxima`.
    :param test_mode: Same.
    :param normalization_table: Same.
    :return: min_normalized_value: Minimum value in normalized units.
    :return: max_normalized_value: Max value in normalized units.
    """

    training_option_dict = cnn_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    normalization_type_string = training_option_dict[
        trainval_io.NORMALIZATION_TYPE_KEY]
    min_normalized_value = training_option_dict[
        trainval_io.MIN_NORMALIZED_VALUE_KEY]
    max_normalized_value = training_option_dict[
        trainval_io.MAX_NORMALIZED_VALUE_KEY]

    if test_mode:
        normalization_file_name = None
    else:
        normalization_file_name = training_option_dict[
            trainval_io.NORMALIZATION_FILE_KEY]

    if field_name in SOUNDING_TO_MINIMUM_DICT:
        if normalization_type_string == dl_utils.Z_NORMALIZATION_TYPE_STRING:
            min_physical_value = SOUNDING_TO_MINIMUM_DICT[field_name]
            dummy_sounding_matrix = numpy.full((1, 1, 1), min_physical_value)

            dummy_sounding_matrix = dl_utils.normalize_soundings(
                sounding_matrix=dummy_sounding_matrix, field_names=[field_name],
                normalization_type_string=normalization_type_string,
                normalization_param_file_name=normalization_file_name,
                min_normalized_value=min_normalized_value,
                max_normalized_value=max_normalized_value,
                test_mode=test_mode, normalization_table=normalization_table)

            min_normalized_value = numpy.ravel(dummy_sounding_matrix)[0]
    else:
        min_normalized_value = numpy.nan

    if field_name in SOUNDING_TO_MAX_DICT:
        if normalization_type_string == dl_utils.Z_NORMALIZATION_TYPE_STRING:
            max_physical_value = SOUNDING_TO_MAX_DICT[field_name]
            dummy_sounding_matrix = numpy.full((1, 1, 1), max_physical_value)

            dummy_sounding_matrix = dl_utils.normalize_soundings(
                sounding_matrix=dummy_sounding_matrix, field_names=[field_name],
                normalization_type_string=normalization_type_string,
                normalization_param_file_name=normalization_file_name,
                min_normalized_value=min_normalized_value,
                max_normalized_value=max_normalized_value,
                test_mode=test_mode, normalization_table=normalization_table)

            max_normalized_value = numpy.ravel(dummy_sounding_matrix)[0]
    else:
        max_normalized_value = numpy.nan

    return min_normalized_value, max_normalized_value


def _find_constrained_radar_channels(list_of_layer_operation_dicts):
    """Finds pairs of constrained radar channels.

    C = number of radar channels
    P = number of pairs of constrained channels

    :param list_of_layer_operation_dicts: length-C list of dictionaries.  Each
        must be accepted by `input_examples._check_layer_operation`.
    :return: greater_indices: length-P numpy array of indices.  If
        greater_indices[i] = m and less_indices[i] = k, the [m]th channel must
        >= the [k]th channel.
    :return: less_indices: See above.
    """

    field_name_by_channel = [
        d[input_examples.RADAR_FIELD_KEY] for d in list_of_layer_operation_dicts
    ]
    operation_name_by_channel = [
        d[input_examples.OPERATION_NAME_KEY]
        for d in list_of_layer_operation_dicts
    ]
    min_height_by_channel_m_agl = numpy.array([
        d[input_examples.MIN_HEIGHT_KEY]
        for d in list_of_layer_operation_dicts
    ], dtype=int)
    max_height_by_channel_m_agl = numpy.array([
        d[input_examples.MAX_HEIGHT_KEY]
        for d in list_of_layer_operation_dicts
    ], dtype=int)

    num_channels = len(field_name_by_channel)
    greater_indices = []
    less_indices = []

    for m in range(num_channels):
        if operation_name_by_channel[m] == input_examples.MIN_OPERATION_NAME:
            continue

        same_layer_flags = numpy.logical_and(
            min_height_by_channel_m_agl == min_height_by_channel_m_agl[m],
            max_height_by_channel_m_agl == max_height_by_channel_m_agl[m]
        )

        same_field_flags = numpy.logical_and(
            same_layer_flags,
            numpy.array(field_name_by_channel) == field_name_by_channel[m]
        )

        same_field_indices = numpy.where(same_field_flags)[0]

        if len(same_field_indices) == 0:
            continue

        same_field_operation_names = [
            operation_name_by_channel[k] for k in same_field_indices
        ]

        if operation_name_by_channel[m] == input_examples.MEAN_OPERATION_NAME:
            try:
                this_subindex = same_field_operation_names.index(
                    input_examples.MIN_OPERATION_NAME)

                greater_indices.append(m)
                less_indices.append(same_field_indices[this_subindex])
            except ValueError:
                pass

            continue

        try:
            this_subindex = same_field_operation_names.index(
                input_examples.MEAN_OPERATION_NAME)

            greater_indices.append(m)
            less_indices.append(same_field_indices[this_subindex])
            continue
        except ValueError:
            pass

        try:
            this_subindex = same_field_operation_names.index(
                input_examples.MIN_OPERATION_NAME)

            greater_indices.append(m)
            less_indices.append(same_field_indices[this_subindex])
        except ValueError:
            pass

    greater_indices = numpy.array(greater_indices, dtype=int)
    less_indices = numpy.array(less_indices, dtype=int)
    return greater_indices, less_indices


def _normalize_minima_and_maxima(
        list_of_input_tensors, cnn_metadata_dict, test_mode=False,
        radar_normalization_table=None, sounding_normalization_table=None):
    """Converts min and max for each physical variable to normalized units.

    T = number of input tensors to the CNN

    :param list_of_input_tensors: length-T list of Keras tensors.
    :param cnn_metadata_dict: Dictionary returned by `cnn.read_model_metadata`.
    :param test_mode: Leave this alone.
    :param radar_normalization_table: Leave this alone.
    :param sounding_normalization_table: Leave this alone.
    :return: min_values_by_tensor: length-T list of numpy arrays.  Length of
        [i]th array = length of last axis (channel axis) in [i]th input tensor.
    :return: max_values_by_tensor: Same.
    """

    error_checking.assert_is_boolean(test_mode)

    training_option_dict = cnn_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]

    num_input_tensors = len(list_of_input_tensors)
    min_values_by_tensor = [numpy.array([])] * num_input_tensors
    max_values_by_tensor = [numpy.array([])] * num_input_tensors

    if cnn_metadata_dict[cnn.CONV_2D3D_KEY]:
        this_min_value, this_max_value = __normalize_minmax_for_radar_field(
            field_name=radar_utils.REFL_NAME,
            cnn_metadata_dict=cnn_metadata_dict, test_mode=test_mode,
            normalization_table=radar_normalization_table)

        min_values_by_tensor[0] = numpy.array([this_min_value])
        max_values_by_tensor[0] = numpy.array([this_max_value])

        az_shear_field_names = training_option_dict[
            trainval_io.RADAR_FIELDS_KEY]

        num_az_shear_fields = len(az_shear_field_names)
        min_values_by_tensor[1] = numpy.full(num_az_shear_fields, numpy.nan)
        max_values_by_tensor[1] = numpy.full(num_az_shear_fields, numpy.nan)

        for j in range(num_az_shear_fields):
            min_values_by_tensor[1][j], max_values_by_tensor[1][j] = (
                __normalize_minmax_for_radar_field(
                    field_name=az_shear_field_names[j],
                    cnn_metadata_dict=cnn_metadata_dict, test_mode=test_mode,
                    normalization_table=radar_normalization_table)
            )
    else:
        list_of_layer_operation_dicts = cnn_metadata_dict[
            cnn.LAYER_OPERATIONS_KEY]

        if list_of_layer_operation_dicts is None:
            radar_field_names = training_option_dict[
                trainval_io.RADAR_FIELDS_KEY]
        else:
            radar_field_names = [
                d[input_examples.RADAR_FIELD_KEY]
                for d in list_of_layer_operation_dicts
            ]

        num_radar_fields = len(radar_field_names)
        min_values_by_tensor[0] = numpy.full(num_radar_fields, numpy.nan)
        max_values_by_tensor[0] = numpy.full(num_radar_fields, numpy.nan)

        for j in range(num_radar_fields):
            min_values_by_tensor[0][j], max_values_by_tensor[0][j] = (
                __normalize_minmax_for_radar_field(
                    field_name=radar_field_names[j],
                    cnn_metadata_dict=cnn_metadata_dict, test_mode=test_mode,
                    normalization_table=radar_normalization_table)
            )

    sounding_field_names = training_option_dict[trainval_io.SOUNDING_FIELDS_KEY]
    if sounding_field_names is None:
        return min_values_by_tensor, max_values_by_tensor

    num_sounding_fields = len(sounding_field_names)
    min_values_by_tensor[-1] = numpy.full(num_sounding_fields, numpy.nan)
    max_values_by_tensor[-1] = numpy.full(num_sounding_fields, numpy.nan)

    for j in range(num_sounding_fields):
        min_values_by_tensor[-1][j], max_values_by_tensor[-1][j] = (
            __normalize_minmax_for_sounding_field(
                field_name=sounding_field_names[j],
                cnn_metadata_dict=cnn_metadata_dict, test_mode=test_mode,
                normalization_table=sounding_normalization_table)
        )

    return min_values_by_tensor, max_values_by_tensor


def radar_constraints_to_loss_fn(radar_tensor, list_of_layer_operation_dicts):
    """Converts radar constraints to loss function.

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid
    C = number of radar channels

    :param radar_tensor: E-by-M-by-N-by-C Keras tensor with radar data.
    :param list_of_layer_operation_dicts: See doc for
        `_find_constrained_radar_channels`.
    :return: loss_tensor: Keras tensor defining loss function.  If there are no
        constrained pairs of radar channels, this will be None.
    """

    radar_dimensions = numpy.array(
        radar_tensor.get_shape().as_list()[1:], dtype=int
    )

    error_checking.assert_is_numpy_array(
        radar_dimensions, exact_dimensions=numpy.array([3], dtype=int)
    )

    error_checking.assert_equals(
        radar_dimensions[-1], len(list_of_layer_operation_dicts)
    )

    greater_indices, less_indices = _find_constrained_radar_channels(
        list_of_layer_operation_dicts)

    num_constrained_pairs = len(greater_indices)
    loss_tensor = None

    for k in range(num_constrained_pairs):
        this_difference_tensor = (
            radar_tensor[..., less_indices[k]] -
            radar_tensor[..., greater_indices[k]]
        )

        if loss_tensor is None:
            loss_tensor = K.mean(K.maximum(this_difference_tensor, 0.) ** 2)
        else:
            loss_tensor += K.mean(K.maximum(this_difference_tensor, 0.) ** 2)

    return loss_tensor


def minima_and_maxima_to_loss_fn(
        list_of_input_tensors, cnn_metadata_dict):
    """Converts minima and maxima to loss functions.

    :param list_of_input_tensors: See doc for `_normalize_minima_and_maxima`.
    :param cnn_metadata_dict: Same.
    :return: loss_tensor: Keras tensor defining loss function.  If there is no
        constrained predictor, this will be None.
    """

    min_values_by_tensor, max_values_by_tensor = _normalize_minima_and_maxima(
        list_of_input_tensors=list_of_input_tensors,
        cnn_metadata_dict=cnn_metadata_dict)

    loss_tensor = None
    num_input_tensors = len(list_of_input_tensors)

    for i in range(num_input_tensors):
        this_num_channels = len(min_values_by_tensor[i])

        for j in range(this_num_channels):
            if numpy.isnan(min_values_by_tensor[i][j]):
                continue

            this_difference_tensor = (
                min_values_by_tensor[i][j] -
                list_of_input_tensors[i][..., j]
            )

            if loss_tensor is None:
                loss_tensor = K.mean(K.maximum(this_difference_tensor, 0.) ** 2)
            else:
                loss_tensor += K.mean(
                    K.maximum(this_difference_tensor, 0.) ** 2
                )

        for j in range(this_num_channels):
            if numpy.isnan(max_values_by_tensor[i][j]):
                continue

            this_difference_tensor = (
                list_of_input_tensors[i][..., j] -
                max_values_by_tensor[i][j]
            )

            if loss_tensor is None:
                loss_tensor = K.mean(K.maximum(this_difference_tensor, 0.) ** 2)
            else:
                loss_tensor += K.mean(
                    K.maximum(this_difference_tensor, 0.) ** 2
                )

    return loss_tensor
