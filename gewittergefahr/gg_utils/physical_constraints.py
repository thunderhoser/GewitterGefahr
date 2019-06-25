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
        list_of_input_tensors, cnn_metadata_dict, normalization_file_name,
        normalization_type_string, min_normalized_value=0.,
        max_normalized_value=1.):
    """Converts min and max for each physical variable to normalized units.

    T = number of input tensors to the CNN

    :param list_of_input_tensors: length-T list of Keras tensors.
    :param cnn_metadata_dict: Dictionary returned by `cnn.read_model_metadata`.
    :param normalization_file_name: Path to normalization file.  Will be read by
        `deep_learning_utils.read_normalization_params_from_file`.
    :param normalization_type_string: See doc for
        `deep_learning_utils.normalize_radar_images` or
        `deep_learning_utils.normalize_soundings`.
    :param min_normalized_value: Same.
    :param max_normalized_value: Same.
    :return: min_values_by_tensor: length-T list of numpy arrays.  Length of
        [i]th array = length of last axis (channel axis) in [i]th input tensor.
    :return: max_values_by_tensor: Same.
    """

    # TODO(thunderhoser): Modularize this.
    # TODO(thunderhoser): Check normalization params.
    training_option_dict = cnn_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]

    num_input_tensors = len(list_of_input_tensors)
    min_values_by_tensor = [] * num_input_tensors
    max_values_by_tensor = [] * num_input_tensors

    if cnn_metadata_dict[cnn.USE_2D3D_CONVOLUTION_KEY]:
        num_refl_heights = list_of_input_tensors[0].get_shape().as_list()[-2]
        num_az_shear_fields = list_of_input_tensors[1].get_shape().as_list()[-1]

        if (normalization_type_string ==
                dl_utils.MINMAX_NORMALIZATION_TYPE_STRING):

            min_values_by_tensor[0] = numpy.full(
                num_refl_heights, min_normalized_value)
            max_values_by_tensor[0] = numpy.full(
                num_refl_heights, max_normalized_value)

            min_values_by_tensor[1] = numpy.full(
                num_az_shear_fields, min_normalized_value)
            max_values_by_tensor[1] = numpy.full(
                num_az_shear_fields, max_normalized_value)

        else:
            if radar_utils.REFL_NAME in RADAR_TO_MINIMUM_DICT:
                this_min_value = RADAR_TO_MINIMUM_DICT[radar_utils.REFL_NAME]
                this_min_matrix = numpy.full((1, 1, 1, 1), this_min_value)

                this_min_matrix = dl_utils.normalize_radar_images(
                    radar_image_matrix=this_min_matrix,
                    field_names=[radar_utils.REFL_NAME],
                    normalization_type_string=normalization_type_string,
                    normalization_param_file_name=normalization_file_name,
                    min_normalized_value=min_normalized_value,
                    max_normalized_value=max_normalized_value)

                this_min_value = numpy.ravel(this_min_matrix)[0]
            else:
                this_min_value = numpy.nan

            min_values_by_tensor[0] = numpy.full(
                num_refl_heights, this_min_value)
            max_values_by_tensor[0] = numpy.full(num_refl_heights, numpy.nan)
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
            if (normalization_type_string ==
                    dl_utils.MINMAX_NORMALIZATION_TYPE_STRING):

                min_values_by_tensor[0][j] = (
                    min_normalized_value
                    if radar_field_names[j] in RADAR_TO_MINIMUM_DICT
                    else numpy.nan
                )

                max_values_by_tensor[0][j] = (
                    max_normalized_value
                    if radar_field_names[j] in RADAR_TO_MAX_DICT
                    else numpy.nan
                )

                continue

            if radar_field_names[j] in RADAR_TO_MINIMUM_DICT:
                this_min_value = RADAR_TO_MINIMUM_DICT[radar_field_names[j]]
                this_min_matrix = numpy.full((1, 1, 1, 1), this_min_value)

                this_min_matrix = dl_utils.normalize_radar_images(
                    radar_image_matrix=this_min_matrix,
                    field_names=[radar_field_names[j]],
                    normalization_type_string=normalization_type_string,
                    normalization_param_file_name=normalization_file_name,
                    min_normalized_value=min_normalized_value,
                    max_normalized_value=max_normalized_value)

                this_min_value = numpy.ravel(this_min_matrix)[0]
            else:
                this_min_value = numpy.nan

            min_values_by_tensor[0][j] = this_min_value

            if radar_field_names[j] in RADAR_TO_MAX_DICT:
                this_max_value = RADAR_TO_MAX_DICT[radar_field_names[j]]
                this_max_matrix = numpy.full((1, 1, 1, 1), this_max_value)

                this_max_matrix = dl_utils.normalize_radar_images(
                    radar_image_matrix=this_max_matrix,
                    field_names=[radar_field_names[j]],
                    normalization_type_string=normalization_type_string,
                    normalization_param_file_name=normalization_file_name,
                    min_normalized_value=min_normalized_value,
                    max_normalized_value=max_normalized_value)

                this_max_value = numpy.ravel(this_max_matrix)[0]
            else:
                this_max_value = numpy.nan

            max_values_by_tensor[0][j] = this_max_value

    sounding_field_names = training_option_dict[trainval_io.SOUNDING_FIELDS_KEY]
    if sounding_field_names is None:
        return min_values_by_tensor, max_values_by_tensor

    num_sounding_fields = len(sounding_field_names)
    min_values_by_tensor[-1] = numpy.full(num_sounding_fields, numpy.nan)
    max_values_by_tensor[-1] = numpy.full(num_sounding_fields, numpy.nan)

    for j in range(num_sounding_fields):
        if (normalization_type_string ==
                dl_utils.MINMAX_NORMALIZATION_TYPE_STRING):

            min_values_by_tensor[-1][j] = (
                min_normalized_value
                if sounding_field_names[j] in SOUNDING_TO_MINIMUM_DICT
                else numpy.nan
            )

            max_values_by_tensor[-1][j] = (
                max_normalized_value
                if sounding_field_names[j] in SOUNDING_TO_MAX_DICT
                else numpy.nan
            )

            continue

        if sounding_field_names[j] in SOUNDING_TO_MINIMUM_DICT:
            this_min_value = SOUNDING_TO_MINIMUM_DICT[sounding_field_names[j]]
            this_min_matrix = numpy.full((1, 1, 1), this_min_value)

            this_min_matrix = dl_utils.normalize_soundings(
                sounding_matrix=this_min_matrix,
                field_names=[sounding_field_names[j]],
                normalization_type_string=normalization_type_string,
                normalization_param_file_name=normalization_file_name,
                min_normalized_value=min_normalized_value,
                max_normalized_value=max_normalized_value)

            this_min_value = numpy.ravel(this_min_matrix)[0]
        else:
            this_min_value = numpy.nan

        min_values_by_tensor[-1][j] = this_min_value

        if sounding_field_names[j] in SOUNDING_TO_MAX_DICT:
            this_max_value = SOUNDING_TO_MAX_DICT[sounding_field_names[j]]
            this_max_matrix = numpy.full((1, 1, 1), this_max_value)

            this_max_matrix = dl_utils.normalize_soundings(
                sounding_matrix=this_max_matrix,
                field_names=[sounding_field_names[j]],
                normalization_type_string=normalization_type_string,
                normalization_param_file_name=normalization_file_name,
                min_normalized_value=min_normalized_value,
                max_normalized_value=max_normalized_value)

            this_max_value = numpy.ravel(this_max_matrix)[0]
        else:
            this_max_value = numpy.nan

        max_values_by_tensor[-1][j] = this_max_value

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
            loss_tensor = K.sum(K.maximum(this_difference_tensor, 0.) ** 2)
        else:
            loss_tensor += K.sum(K.maximum(this_difference_tensor, 0.) ** 2)

    return loss_tensor


def minima_and_maxima_to_loss_fn(
        list_of_input_tensors, cnn_metadata_dict, normalization_file_name,
        normalization_type_string, min_normalized_value=0.,
        max_normalized_value=1.):
    """Converts minima and maxima to loss functions.

    :param list_of_input_tensors: See doc for `_normalize_minima_and_maxima`.
    :param cnn_metadata_dict: Same.
    :param normalization_file_name: Same.
    :param normalization_type_string: Same.
    :param min_normalized_value: Same.
    :param max_normalized_value: Same.
    :return: loss_tensor: Keras tensor defining loss function.  If there is no
        constrained predictor, this will be None.
    """

    min_values_by_tensor, max_values_by_tensor = _normalize_minima_and_maxima(
        list_of_input_tensors=list_of_input_tensors,
        cnn_metadata_dict=cnn_metadata_dict,
        normalization_file_name=normalization_file_name,
        normalization_type_string=normalization_type_string,
        min_normalized_value=min_normalized_value,
        max_normalized_value=max_normalized_value)

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
                loss_tensor = K.sum(K.maximum(this_difference_tensor, 0.) ** 2)
            else:
                loss_tensor += K.sum(K.maximum(this_difference_tensor, 0.) ** 2)

        for j in range(this_num_channels):
            if numpy.isnan(max_values_by_tensor[i][j]):
                continue

            this_difference_tensor = (
                list_of_input_tensors[i][..., j] -
                max_values_by_tensor[i][j]
            )

            if loss_tensor is None:
                loss_tensor = K.sum(K.maximum(this_difference_tensor, 0.) ** 2)
            else:
                loss_tensor += K.sum(K.maximum(this_difference_tensor, 0.) ** 2)

    return loss_tensor
