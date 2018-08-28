"""Helper methods for model interpretation."""

import numpy
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import storm_images
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.deep_learning import deployment_io

LARGE_INTEGER = int(1e10)

CLASS_COMPONENT_TYPE_STRING = 'class'
NEURON_COMPONENT_TYPE_STRING = 'neuron'
CHANNEL_COMPONENT_TYPE_STRING = 'channel'
VALID_COMPONENT_TYPE_STRINGS = [
    CLASS_COMPONENT_TYPE_STRING, NEURON_COMPONENT_TYPE_STRING,
    CHANNEL_COMPONENT_TYPE_STRING
]


def check_component_type(component_type_string):
    """Ensures that model-component type is valid.

    :param component_type_string: Component type.
    :raises: ValueError: if
        `component_type_string not in VALID_COMPONENT_TYPE_STRINGS`.
    """

    error_checking.assert_is_string(component_type_string)
    if component_type_string not in VALID_COMPONENT_TYPE_STRINGS:
        error_string = (
            '\n\n{0:s}\nValid component types (listed above) do not include '
            '"{1:s}".'
        ).format(str(VALID_COMPONENT_TYPE_STRINGS), component_type_string)
        raise ValueError(error_string)


def check_component_metadata(
        component_type_string, target_class=None, layer_name=None,
        neuron_indices=None, channel_index=None):
    """Checks metadata for model component.

    :param component_type_string: Component type (must be accepted by
        `check_component_type`).
    :param target_class: [used only if component_type_string = "class"]
        Target class.  Integer from 0...(K - 1), where K = number of classes.
    :param layer_name:
        [used only if component_type_string = "neuron" or "channel"]
        Name of layer containing neuron or channel.
    :param neuron_indices: [used only if component_type_string = "neuron"]
        1-D numpy array with indices of neuron.
    :param channel_index: [used only if component_type_string = "channel"]
        Index of channel.
    """

    check_component_type(component_type_string)
    if component_type_string == CLASS_COMPONENT_TYPE_STRING:
        error_checking.assert_is_integer(target_class)
        error_checking.assert_is_geq(target_class, 0)

    if component_type_string in [NEURON_COMPONENT_TYPE_STRING,
                                 CHANNEL_COMPONENT_TYPE_STRING]:
        error_checking.assert_is_string(layer_name)

    if component_type_string == NEURON_COMPONENT_TYPE_STRING:
        error_checking.assert_is_integer_numpy_array(neuron_indices)
        error_checking.assert_is_geq_numpy_array(neuron_indices, 0)
        error_checking.assert_is_numpy_array(neuron_indices, num_dimensions=1)

    if component_type_string == CHANNEL_COMPONENT_TYPE_STRING:
        error_checking.assert_is_integer(channel_index)
        error_checking.assert_is_geq(channel_index, 0)


def model_component_to_string(
        component_type_string, target_class=None, layer_name=None,
        neuron_indices=None, channel_index=None):
    """Returns string descriptions for model component (class/neuron/channel).

    Specifically, this method creates two strings:

    - verbose string (to use in figure legends)
    - abbreviation (to use in file names)

    :param component_type_string: See doc for `check_component_metadata`.
    :param target_class: Same.
    :param layer_name: Same.
    :param neuron_indices: Same.
    :param channel_index: Same.
    :return: verbose_string: See general discussion above.
    :return: abbrev_string: See general discussion above.
    """

    check_component_metadata(
        component_type_string=component_type_string, target_class=target_class,
        layer_name=layer_name, neuron_indices=neuron_indices,
        channel_index=channel_index)

    if component_type_string == CLASS_COMPONENT_TYPE_STRING:
        verbose_string = 'Class {0:d}'.format(target_class)
        abbrev_string = 'class{0:d}'.format(target_class)
    else:
        verbose_string = 'Layer "{0:s}"'.format(layer_name)
        abbrev_string = 'layer={0:s}'.format(layer_name.replace('_', '-'))

    if component_type_string == CHANNEL_COMPONENT_TYPE_STRING:
        verbose_string += ', channel {0:d}'.format(channel_index)
        abbrev_string += '_channel{0:d}'.format(channel_index)

    if component_type_string == NEURON_COMPONENT_TYPE_STRING:
        this_neuron_string = ', '.join(
            ['{0:d}'.format(i) for i in neuron_indices])
        verbose_string += '; neuron ({0:s})'.format(this_neuron_string)

        this_neuron_string = ','.join(
            ['{0:d}'.format(i) for i in neuron_indices])
        abbrev_string += '_neuron{0:s}'.format(this_neuron_string)

    return verbose_string, abbrev_string


def sort_neurons_by_weight(model_object, layer_name):
    """Sorts neurons of the given layer in descending order by weight.

    K = number of dimensions in `weight_matrix`
    W = number of values in `weight_matrix`

    :param model_object: Instance of `keras.models.Model`.
    :param layer_name: Name of layer whose neurons are to be sorted.
    :return: weight_matrix: numpy array of weights, with the same dimensions as
        `model_object.get_layer(name=layer_name).get_weights()[0]`.

    If the layer is convolutional, dimensions of `weight_matrix` are as follows:

    - Last dimension = output channel
    - Second-last dimension = input channel
    - First dimensions = spatial dimensions

    For example, if the conv layer has a 3-by-5 kernel with 16 input channels
    and 32 output channels, `weight_matrix` will be 3 x 5 x 16 x 32.

    If the layer is dense (fully connected), `weight_matrix` is 1-D.

    :return: sort_indices_as_tuple: length-K tuple.  sort_indices_as_tuple[k] is
        a length-W numpy array, containing indices for the [k]th dimension of
        `weight_matrix`.  When these indices are applied to all dimensions of
        `weight_matrix` -- i.e., when sort_indices_as_tuple[k] is applied for
        k = 0...(K - 1) -- `weight_matrix` has been sorted in descending order.
    :raises: TypeError: if the given layer is neither dense nor convolutional.
    """

    layer_type_string = type(model_object.get_layer(name=layer_name)).__name__
    valid_layer_type_strings = ['Dense', 'Conv1D', 'Conv2D', 'Conv3D']
    if layer_type_string not in valid_layer_type_strings:
        error_string = (
            '\n\n{0:s}\nLayer "{1:s}" has type "{2:s}", which is not in the '
            'above list.'
        ).format(str(valid_layer_type_strings), layer_name, layer_type_string)
        raise TypeError(error_string)

    weight_matrix = model_object.get_layer(name=layer_name).get_weights()[0]
    sort_indices_linear = numpy.argsort(
        -numpy.reshape(weight_matrix, weight_matrix.size))
    sort_indices_as_tuple = numpy.unravel_index(
        sort_indices_linear, weight_matrix.shape)

    return weight_matrix, sort_indices_as_tuple


def read_storms_one_spc_date(
        radar_file_name_matrix, model_metadata_dict, top_sounding_dir_name,
        spc_date_index, desired_storm_ids=None,
        desired_storm_times_unix_sec=None):
    """Reads storm objects (model inputs) for one SPC date.

    E = number of examples (storm objects) returned

    If `desired_storm_ids is None or desired_storm_times_unix_sec is None`, data
    will be read for all storm objects on the given SPC date.  Otherwise, data
    will be read only for selected storm objects.

    :param radar_file_name_matrix: numpy array of file names, created by either
        `training_validation_io.find_radar_files_2d` or
        `training_validation_io.find_radar_files_3d`.
    :param model_metadata_dict: Dictionary with metadata for the relevant model,
        created by `cnn.read_model_metadata`.
    :param top_sounding_dir_name: Name of top-level directory with storm-
        centered soundings.
    :param spc_date_index: Index of SPC date.  Data will be read from
        radar_file_name_matrix[i, ...], where i = `spc_date_index`.
    :param desired_storm_ids: 1-D list of storm IDs.  Data will be read only for
        these storm objects.
    :param desired_storm_times_unix_sec: 1-D numpy array of storm times (same
        length as `desired_storm_ids`).  Data will be read only for these storm
        objects.
    :return: list_of_input_matrices: length-T list of numpy arrays, where T =
        number of input tensors to the model.  The first dimension of each array
        has length E.
    :return: storm_ids: length-E list of storm IDs.
    :return: storm_times_unix_sec: length-E list of storm times.
    """

    return_all_storm_objects = (
        desired_storm_ids is None or desired_storm_times_unix_sec is None)

    if not return_all_storm_objects:
        error_checking.assert_is_numpy_array(
            numpy.array(desired_storm_ids), num_dimensions=1)
        num_desired_objects = len(desired_storm_ids)

        error_checking.assert_is_integer_numpy_array(
            desired_storm_times_unix_sec)
        error_checking.assert_is_numpy_array(
            desired_storm_times_unix_sec,
            exact_dimensions=numpy.array([num_desired_objects]))

        this_radar_file_name = numpy.reshape(
            radar_file_name_matrix[spc_date_index, ...],
            radar_file_name_matrix[spc_date_index, ...].size)[0]
        _, spc_date_string = storm_images.image_file_name_to_time(
            this_radar_file_name)

        valid_indices = numpy.where(numpy.logical_and(
            desired_storm_times_unix_sec >=
            time_conversion.get_start_of_spc_date(spc_date_string),
            desired_storm_times_unix_sec <=
            time_conversion.get_end_of_spc_date(spc_date_string)
        ))[0]

        storm_ids = [desired_storm_ids[k] for k in valid_indices]
        storm_times_unix_sec = desired_storm_times_unix_sec[valid_indices]

    if model_metadata_dict[cnn.USE_2D3D_CONVOLUTION_KEY]:
        example_dict = deployment_io.create_storm_images_2d3d_myrorss(
            radar_file_name_matrix=radar_file_name_matrix[
                [spc_date_index], ...],
            num_examples_per_file=LARGE_INTEGER,
            normalization_type_string=model_metadata_dict[
                cnn.NORMALIZATION_TYPE_KEY],
            min_normalized_value=model_metadata_dict[
                cnn.MIN_NORMALIZED_VALUE_KEY],
            max_normalized_value=model_metadata_dict[
                cnn.MAX_NORMALIZED_VALUE_KEY],
            normalization_param_file_name=model_metadata_dict[
                cnn.NORMALIZATION_FILE_NAME_KEY],
            return_target=False,
            target_name=model_metadata_dict[cnn.TARGET_NAME_KEY],
            num_rows_to_keep=model_metadata_dict[cnn.NUM_ROWS_TO_KEEP_KEY],
            num_columns_to_keep=model_metadata_dict[
                cnn.NUM_COLUMNS_TO_KEEP_KEY],
            sounding_field_names=model_metadata_dict[
                cnn.SOUNDING_FIELD_NAMES_KEY],
            top_sounding_dir_name=top_sounding_dir_name,
            sounding_lag_time_for_convective_contamination_sec=
            model_metadata_dict[cnn.SOUNDING_LAG_TIME_KEY])
    else:
        num_radar_dimensions = len(
            model_metadata_dict[cnn.TRAINING_FILE_NAMES_KEY].shape)

        if num_radar_dimensions == 3:
            example_dict = deployment_io.create_storm_images_3d(
                radar_file_name_matrix=radar_file_name_matrix[
                    [spc_date_index], ...],
                num_examples_per_file=LARGE_INTEGER,
                normalization_type_string=model_metadata_dict[
                    cnn.NORMALIZATION_TYPE_KEY],
                min_normalized_value=model_metadata_dict[
                    cnn.MIN_NORMALIZED_VALUE_KEY],
                max_normalized_value=model_metadata_dict[
                    cnn.MAX_NORMALIZED_VALUE_KEY],
                normalization_param_file_name=model_metadata_dict[
                    cnn.NORMALIZATION_FILE_NAME_KEY],
                return_target=False,
                target_name=model_metadata_dict[cnn.TARGET_NAME_KEY],
                num_rows_to_keep=model_metadata_dict[cnn.NUM_ROWS_TO_KEEP_KEY],
                num_columns_to_keep=model_metadata_dict[
                    cnn.NUM_COLUMNS_TO_KEEP_KEY],
                refl_masking_threshold_dbz=model_metadata_dict[
                    cnn.REFL_MASKING_THRESHOLD_KEY],
                sounding_field_names=model_metadata_dict[
                    cnn.SOUNDING_FIELD_NAMES_KEY],
                top_sounding_dir_name=top_sounding_dir_name,
                sounding_lag_time_for_convective_contamination_sec=
                model_metadata_dict[cnn.SOUNDING_LAG_TIME_KEY])
        else:
            example_dict = deployment_io.create_storm_images_2d(
                radar_file_name_matrix=radar_file_name_matrix[
                    [spc_date_index], ...],
                num_examples_per_file=LARGE_INTEGER,
                normalization_type_string=model_metadata_dict[
                    cnn.NORMALIZATION_TYPE_KEY],
                min_normalized_value=model_metadata_dict[
                    cnn.MIN_NORMALIZED_VALUE_KEY],
                max_normalized_value=model_metadata_dict[
                    cnn.MAX_NORMALIZED_VALUE_KEY],
                normalization_param_file_name=model_metadata_dict[
                    cnn.NORMALIZATION_FILE_NAME_KEY],
                return_target=False,
                target_name=model_metadata_dict[cnn.TARGET_NAME_KEY],
                num_rows_to_keep=model_metadata_dict[cnn.NUM_ROWS_TO_KEEP_KEY],
                num_columns_to_keep=model_metadata_dict[
                    cnn.NUM_COLUMNS_TO_KEEP_KEY],
                sounding_field_names=model_metadata_dict[
                    cnn.SOUNDING_FIELD_NAMES_KEY],
                top_sounding_dir_name=top_sounding_dir_name,
                sounding_lag_time_for_convective_contamination_sec=
                model_metadata_dict[cnn.SOUNDING_LAG_TIME_KEY])

    if example_dict is None:
        return None, None, None

    if return_all_storm_objects:
        storm_ids = example_dict[deployment_io.STORM_IDS_KEY]
        storm_times_unix_sec = example_dict[deployment_io.STORM_TIMES_KEY]
        relevant_indices = numpy.linspace(
            0, len(storm_ids) - 1, num=len(storm_ids), dtype=int)
    else:
        relevant_indices = tracking_utils.find_storm_objects(
            all_storm_ids=example_dict[deployment_io.STORM_IDS_KEY],
            all_times_unix_sec=example_dict[deployment_io.STORM_TIMES_KEY],
            storm_ids_to_keep=storm_ids,
            times_to_keep_unix_sec=storm_times_unix_sec)

    if model_metadata_dict[cnn.USE_2D3D_CONVOLUTION_KEY]:
        list_of_input_matrices = [
            example_dict[deployment_io.REFLECTIVITY_MATRIX_KEY][
                relevant_indices, ...],
            example_dict[deployment_io.AZ_SHEAR_MATRIX_KEY][
                relevant_indices, ...]
        ]
    else:
        list_of_input_matrices = [
            example_dict[deployment_io.RADAR_IMAGE_MATRIX_KEY][
                relevant_indices, ...]
        ]

    if example_dict[deployment_io.SOUNDING_MATRIX_KEY] is not None:
        list_of_input_matrices.append(
            example_dict[deployment_io.SOUNDING_MATRIX_KEY][
                relevant_indices, ...])

    return list_of_input_matrices, storm_ids, storm_times_unix_sec


def denormalize_data(list_of_input_matrices, model_metadata_dict):
    """Denormalizes input data for a Keras model.

    :param list_of_input_matrices: length-T list of input matrices (numpy
        arrays), where T = number of input tensors to the model.
    :param model_metadata_dict: Dictionary with metadata for the relevant model,
        created by `cnn.read_model_metadata`.
    :return: list_of_input_matrices: Denormalized version of input (same
        dimensions).
    """

    if model_metadata_dict[cnn.USE_2D3D_CONVOLUTION_KEY]:
        radar_field_names = model_metadata_dict[cnn.RADAR_FIELD_NAMES_KEY]
        azimuthal_shear_indices = numpy.where(numpy.array(
            [f in radar_utils.SHEAR_NAMES for f in radar_field_names]))[0]
        azimuthal_shear_field_names = [
            radar_field_names[j] for j in azimuthal_shear_indices]

        list_of_input_matrices[0] = dl_utils.denormalize_radar_images(
            radar_image_matrix=list_of_input_matrices[0],
            field_names=[radar_utils.REFL_NAME],
            normalization_type_string=model_metadata_dict[
                cnn.NORMALIZATION_TYPE_KEY],
            normalization_param_file_name=model_metadata_dict[
                cnn.NORMALIZATION_FILE_NAME_KEY],
            min_normalized_value=model_metadata_dict[
                cnn.MIN_NORMALIZED_VALUE_KEY],
            max_normalized_value=model_metadata_dict[
                cnn.MAX_NORMALIZED_VALUE_KEY])

        list_of_input_matrices[1] = dl_utils.denormalize_radar_images(
            radar_image_matrix=list_of_input_matrices[1],
            field_names=azimuthal_shear_field_names,
            normalization_type_string=model_metadata_dict[
                cnn.NORMALIZATION_TYPE_KEY],
            normalization_param_file_name=model_metadata_dict[
                cnn.NORMALIZATION_FILE_NAME_KEY],
            min_normalized_value=model_metadata_dict[
                cnn.MIN_NORMALIZED_VALUE_KEY],
            max_normalized_value=model_metadata_dict[
                cnn.MAX_NORMALIZED_VALUE_KEY])
    else:
        radar_file_name_matrix = model_metadata_dict[
            cnn.TRAINING_FILE_NAMES_KEY]
        num_channels = radar_file_name_matrix.shape[1]
        field_name_by_channel = [''] * num_channels

        for j in range(num_channels):
            if len(radar_file_name_matrix.shape) == 3:
                field_name_by_channel[j] = (
                    storm_images.image_file_name_to_field(
                        radar_file_name_matrix[0, j, 0]))
            else:
                field_name_by_channel[j] = (
                    storm_images.image_file_name_to_field(
                        radar_file_name_matrix[0, j]))

        list_of_input_matrices[0] = dl_utils.denormalize_radar_images(
            radar_image_matrix=list_of_input_matrices[0],
            field_names=field_name_by_channel,
            normalization_type_string=model_metadata_dict[
                cnn.NORMALIZATION_TYPE_KEY],
            normalization_param_file_name=model_metadata_dict[
                cnn.NORMALIZATION_FILE_NAME_KEY],
            min_normalized_value=model_metadata_dict[
                cnn.MIN_NORMALIZED_VALUE_KEY],
            max_normalized_value=model_metadata_dict[
                cnn.MAX_NORMALIZED_VALUE_KEY])

    if model_metadata_dict[cnn.SOUNDING_FIELD_NAMES_KEY] is not None:
        list_of_input_matrices[-1] = dl_utils.denormalize_soundings(
            sounding_matrix=list_of_input_matrices[-1],
            field_names=model_metadata_dict[cnn.SOUNDING_FIELD_NAMES_KEY],
            normalization_type_string=model_metadata_dict[
                cnn.NORMALIZATION_TYPE_KEY],
            normalization_param_file_name=model_metadata_dict[
                cnn.NORMALIZATION_FILE_NAME_KEY],
            min_normalized_value=model_metadata_dict[
                cnn.MIN_NORMALIZED_VALUE_KEY],
            max_normalized_value=model_metadata_dict[
                cnn.MAX_NORMALIZED_VALUE_KEY])

    return list_of_input_matrices
