"""IO methods for training and on-the-fly validation of a deep-learning model.

Some variables in this module contain the string "pos_targets_only", which means
that they pertain only to storm objects with positive target values.  For
details on how to separate storm objects with positive target values, see
separate_myrorss_images_with_positive_targets.py.

--- NOTATION ---

The following letters will be used throughout this module.

E = number of examples (storm objects)
M = number of rows per radar image
N = number of columns per radar image
H_r = number of heights per radar image
F_r = number of radar fields (not including different heights)
H_s = number of vertical levels per sounding
F_s = number of sounding fields (not including different vertical levels)
C = number of field/height pairs per radar image
K = number of classes for target variable
T = number of file times (time steps or SPC dates)
"""

import copy
import numpy
import keras
from gewittergefahr.deep_learning import storm_images
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import soundings_only
from gewittergefahr.gg_utils import labels
from gewittergefahr.gg_utils import error_checking

MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

RADAR_IMAGE_MATRIX_KEY = 'radar_image_matrix'
REFLECTIVITY_IMAGE_MATRIX_KEY = 'reflectivity_image_matrix_dbz'
AZ_SHEAR_IMAGE_MATRIX_KEY = 'azimuthal_shear_image_matrix_s01'
SOUNDING_MATRIX_KEY = 'sounding_matrix'
SOUNDING_FIELD_NAMES_KEY = 'sounding_field_names'
TARGET_VALUES_KEY = 'target_values'

STORM_IDS_KEY = 'storm_ids'
STORM_TIMES_KEY = 'storm_times_unix_sec'


def _get_num_examples_per_batch_by_class(
        num_examples_per_batch, target_name, sampling_fraction_by_class_dict):
    """Returns number of examples needed per batch for each class.

    :param num_examples_per_batch: Total number of examples per batch.
    :param target_name: Name of target variable.
    :param sampling_fraction_by_class_dict: See doc for
        `storm_image_generator_2d`.
    :return: num_examples_per_batch_by_class_dict: Dictionary, where each key is
        the integer representing a class (-2 for "dead storm") and each value is
        the corresponding number of examples per batch.
    """

    num_extended_classes = labels.column_name_to_num_classes(
        column_name=target_name, include_dead_storms=True)

    if sampling_fraction_by_class_dict is None:
        num_classes = labels.column_name_to_num_classes(
            column_name=target_name, include_dead_storms=False)
        include_dead_storms = num_extended_classes > num_classes

        if include_dead_storms:
            first_keys = numpy.array([labels.DEAD_STORM_INTEGER], dtype=int)
            second_keys = numpy.linspace(
                0, num_classes - 1, num=num_classes, dtype=int)
            keys = numpy.concatenate((first_keys, second_keys))
        else:
            keys = numpy.linspace(
                0, num_extended_classes - 1, num=num_extended_classes,
                dtype=int)

        values = numpy.full(
            num_extended_classes, num_examples_per_batch, dtype=int)
        return dict(zip(keys, values))

    return dl_utils.class_fractions_to_num_examples(
        sampling_fraction_by_class_dict=sampling_fraction_by_class_dict,
        target_name=target_name, num_examples_total=num_examples_per_batch)


def _get_num_examples_left_by_class(
        num_examples_per_batch, num_file_times_per_batch,
        num_examples_per_batch_by_class_dict, num_examples_in_memory,
        num_file_times_in_memory, num_examples_in_memory_by_class_dict):
    """Returns number of examples needed in the current batch for each class.

    :param num_examples_per_batch: Total number of examples per batch.
    :param num_file_times_per_batch: Total number of file times per batch.  If
        each file contains one time step (SPC date), this is the number of time
        steps (SPC dates) per batch.
    :param num_examples_per_batch_by_class_dict: Dictionary created by
        `_get_num_examples_per_batch_by_class`.
    :param num_examples_in_memory: Number of examples in memory (in the current
        batch).
    :param num_file_times_in_memory: Number of file times in memory (in the
        current batch).
    :param num_examples_in_memory_by_class_dict: Dictionary created by
        `_determine_stopping_criterion`.
    :return: num_examples_left_by_class_dict: Dictionary, where each key is
        the integer representing a class (-2 for "dead storm") and each value is
        the corresponding number of examples needed in the current batch.
    """

    downsample = (
        num_file_times_in_memory >= num_file_times_per_batch and
        num_examples_in_memory >= num_examples_per_batch)

    if not downsample:
        return copy.deepcopy(num_examples_per_batch_by_class_dict)

    num_examples_left_by_class_dict = {}
    for this_key in num_examples_per_batch_by_class_dict.keys():
        this_value = (num_examples_per_batch_by_class_dict[this_key] -
                      num_examples_in_memory_by_class_dict[this_key])
        this_value = max([this_value, 0])
        num_examples_left_by_class_dict.update({this_key: this_value})

    return num_examples_left_by_class_dict


def _determine_stopping_criterion(
        num_examples_per_batch, num_file_times_per_batch,
        num_examples_per_batch_by_class_dict, num_file_times_in_memory,
        sampling_fraction_by_class_dict, target_values_in_memory):
    """Determines whether or not to stop generating examples.

    :param num_examples_per_batch: Total number of examples per batch.
    :param num_file_times_per_batch: Total number of file times per batch.  If
        each file contains one time step (SPC date), this is the number of time
        steps (SPC dates) per batch.
    :param num_examples_per_batch_by_class_dict: Dictionary created by
        `_get_num_examples_per_batch_by_class`.
    :param num_file_times_in_memory: Number of file times in memory (in the
        current batch).
    :param sampling_fraction_by_class_dict: See doc for
        `storm_image_generator_2d`.
    :param target_values_in_memory: 1-D numpy array of target values (class
        integers) currently in memory.
    :return: num_examples_in_memory_by_class_dict: Dictionary, where each key is
        the integer representing a class (-2 for "dead storm") and each value is
        the corresponding number of examples in memory.
    :return: stopping_criterion: Boolean flag.
    """

    num_examples_in_memory_by_class_dict = {}
    for this_key in num_examples_per_batch_by_class_dict.keys():
        this_value = numpy.sum(target_values_in_memory == this_key)
        num_examples_in_memory_by_class_dict.update({this_key: this_value})

    print 'Number of examples in memory by extended class: {0:s}'.format(
        str(num_examples_in_memory_by_class_dict))

    num_examples_in_memory = len(target_values_in_memory)
    stopping_criterion = (
        num_file_times_in_memory >= num_file_times_per_batch and
        num_examples_in_memory >= num_examples_per_batch)

    if stopping_criterion and sampling_fraction_by_class_dict is not None:
        for this_key in num_examples_per_batch_by_class_dict.keys():
            stopping_criterion = (
                stopping_criterion and
                num_examples_in_memory_by_class_dict[this_key] >=
                num_examples_per_batch_by_class_dict[this_key])

    return num_examples_in_memory_by_class_dict, stopping_criterion


def _select_batch(
        list_of_predictor_matrices, target_values, num_examples_per_batch,
        binarize_target, num_classes):
    """Randomly selects batch from examples in memory.

    E_m = number of examples in memory
    E_b = num_examples_per_batch

    :param list_of_predictor_matrices: 1-D list, where each item is a numpy
        array of predictors (radar images, soundings, or sounding statistics).
        The first axis of each numpy array must have length E_m.
    :param target_values: numpy array of target values (length E_m).
    :param num_examples_per_batch: Number of examples per batch.
    :param binarize_target: Boolean flag.  If True, will binarize target
        variable, so that the highest class becomes 1 and all other classes
        become 0.
    :param num_classes: Number of classes for target value.
    :return: list_of_predictor_matrices: Same as input, except that the first
        axis of each array has length E_b.
    :return: target_matrix: See output doc for `storm_image_generator_2d`.
    """

    num_examples_in_memory = len(target_values)
    example_indices = numpy.linspace(
        0, num_examples_in_memory - 1, num=num_examples_in_memory, dtype=int)

    if num_examples_in_memory > num_examples_per_batch:
        batch_indices = numpy.random.choice(
            example_indices, size=num_examples_per_batch, replace=False)
    else:
        batch_indices = example_indices + 0

    for i in range(len(list_of_predictor_matrices)):
        if list_of_predictor_matrices[i] is None:
            continue

        list_of_predictor_matrices[i] = list_of_predictor_matrices[
            i][batch_indices, ...].astype('float32')

    target_values[target_values == labels.DEAD_STORM_INTEGER] = 0

    if binarize_target:
        target_values = (target_values == num_classes - 1).astype(int)
        target_matrix = keras.utils.to_categorical(
            target_values[batch_indices], 2)
    else:
        target_matrix = keras.utils.to_categorical(
            target_values[batch_indices], num_classes)

    class_fractions = numpy.mean(target_matrix, axis=0)
    print 'Fraction of target values in each class: {0:s}\n'.format(
        str(class_fractions))

    return list_of_predictor_matrices, target_matrix


def _need_negative_target_values(num_examples_left_by_class_dict):
    """Determines whether or not negative target values are needed.

    :param num_examples_left_by_class_dict: Dictionary created by
        `_get_num_examples_left_by_class`.
    :return: need_neg_targets_flag: Boolean flag.
    """

    need_neg_targets_flag = num_examples_left_by_class_dict[0] > 0
    if labels.DEAD_STORM_INTEGER in num_examples_left_by_class_dict:
        need_neg_targets_flag = (
            need_neg_targets_flag or
            num_examples_left_by_class_dict[labels.DEAD_STORM_INTEGER] > 0)

    return need_neg_targets_flag


def _read_input_files_2d(
        radar_file_names, top_target_directory_name, target_name,
        num_examples_left_by_class_dict, radar_file_names_pos_targets_only=None,
        sounding_file_name=None, sounding_field_names=None):
    """Reads data for `storm_image_generator_2d`.

    t_0 = file time = time step or SPC date

    :param radar_file_names: length-C list of paths to radar files.  Each file
        should be readable by `storm_images.read_storm_images` and contain
        storm-centered radar images for a different field/height pair at t_0.
    :param top_target_directory_name: Name of top-level directory with target
        values (storm-hazard labels).  Files within this directory should be
        findable by `labels.find_label_file`.
    :param target_name: Name of target variable.
    :param num_examples_left_by_class_dict: Dictionary created by
        `_get_num_examples_left_by_class`.
    :param radar_file_names_pos_targets_only: Same as `radar_file_names`, but
        only for storm objects with positive target values.
    :param sounding_file_name: [optional] Path to sounding file.  Should be
        readable by `soundings_only.read_soundings` and contain storm-centered
        soundings for t_0.
    :param sounding_field_names:
        [used only if `sounding_file_names is not None`]
        1-D list with names of sounding fields (pressureless fields) to use.  If
        `sounding_field_names is None`, will use all sounding fields.
    :return: example_dict: Dictionary with the following keys.
    example_dict['radar_image_matrix']: E-by-M-by-N-by-C numpy array of
        storm-centered radar images.
    example_dict['target_values']: length-E numpy array of target values (class
        integers).
    example_dict['sounding_matrix']: numpy array (E x H_s x F_s) of soundings.
    example_dict['sounding_field_names']: list (length F_s) with names of
        sounding fields.
    """

    label_file_name = storm_images.find_storm_label_file(
        storm_image_file_name=radar_file_names[0],
        top_label_directory_name=top_target_directory_name,
        label_name=target_name, raise_error_if_missing=True)

    if radar_file_names_pos_targets_only is not None:
        if not _need_negative_target_values(num_examples_left_by_class_dict):
            radar_file_names = radar_file_names_pos_targets_only

    print 'Reading data from: "{0:s}" and "{1:s}"...'.format(
        radar_file_names[0], label_file_name)
    this_radar_image_dict = storm_images.read_storm_images_and_labels(
        image_file_name=radar_file_names[0],
        label_file_name=label_file_name, label_name=target_name,
        num_storm_objects_class_dict=num_examples_left_by_class_dict)

    if this_radar_image_dict is None:
        return None

    this_radar_image_dict = remove_storms_with_undefined_target(
        this_radar_image_dict)
    if not len(this_radar_image_dict[storm_images.STORM_IDS_KEY]):
        return None

    if sounding_file_name is None:
        sounding_matrix = None
    else:
        sounding_dict, this_radar_image_dict = read_soundings(
            sounding_file_name=sounding_file_name,
            sounding_field_names=sounding_field_names,
            radar_image_dict=this_radar_image_dict)

        if sounding_dict is None or not len(sounding_dict[STORM_IDS_KEY]):
            return None

        sounding_matrix = sounding_dict[soundings_only.SOUNDING_MATRIX_KEY]
        sounding_field_names = sounding_dict[
            soundings_only.PRESSURELESS_FIELD_NAMES_KEY]

    target_values = this_radar_image_dict[storm_images.LABEL_VALUES_KEY]
    storm_ids_to_keep = this_radar_image_dict[storm_images.STORM_IDS_KEY]
    storm_times_to_keep_unix_sec = this_radar_image_dict[
        storm_images.VALID_TIMES_KEY]

    num_channels = len(radar_file_names)
    tuple_of_image_matrices = ()

    for j in range(num_channels):
        if j != 0:
            print 'Reading data from: "{0:s}"...'.format(radar_file_names[j])
            this_radar_image_dict = storm_images.read_storm_images(
                netcdf_file_name=radar_file_names[j],
                storm_ids_to_keep=storm_ids_to_keep,
                valid_times_to_keep_unix_sec=storm_times_to_keep_unix_sec)

        tuple_of_image_matrices += (
            this_radar_image_dict[storm_images.STORM_IMAGE_MATRIX_KEY],)

    return {
        RADAR_IMAGE_MATRIX_KEY: dl_utils.stack_radar_fields(
            tuple_of_image_matrices),
        TARGET_VALUES_KEY: target_values,
        SOUNDING_MATRIX_KEY: sounding_matrix,
        SOUNDING_FIELD_NAMES_KEY: sounding_field_names
    }


def _read_input_files_3d(
        radar_file_name_matrix, top_target_directory_name, target_name,
        num_examples_left_by_class_dict, rdp_filter_threshold_s02=None,
        radar_file_name_matrix_pos_targets_only=None, sounding_file_name=None,
        sounding_field_names=None):
    """Reads data for `storm_image_generator_3d`.

    t_0 = file time = time step or SPC date

    :param radar_file_name_matrix: numpy array (F_r x H_r) of paths to radar
        files.  Each file should be readable by `storm_images.read_storm_images`
        and contain storm-centered radar images for a different field/height
        pair at t_0.
    :param top_target_directory_name: See doc for `_read_input_files_2d`.
    :param target_name: Same.
    :param num_examples_left_by_class_dict: Same.
    :param rdp_filter_threshold_s02: See doc for `storm_image_generator_3d`.
    :param radar_file_name_matrix_pos_targets_only: Same.
    :param sounding_file_name: See doc for `_read_input_files_2d`.
    :param sounding_field_names: Same.
    :return: example_dict: Dictionary with the following keys.
    example_dict['radar_image_matrix']: numpy array (E x M x N x H_r x F_r) of
        storm-centered radar images.
    example_dict['target_values']: See doc for `_read_input_files_2d`.
    example_dict['sounding_matrix']: Same.
    example_dict['sounding_field_names']: Same.
    """

    label_file_name = storm_images.find_storm_label_file(
        storm_image_file_name=radar_file_name_matrix[0, 0],
        top_label_directory_name=top_target_directory_name,
        label_name=target_name, raise_error_if_missing=True)

    if radar_file_name_matrix_pos_targets_only is not None:
        if not _need_negative_target_values(num_examples_left_by_class_dict):
            radar_file_name_matrix = radar_file_name_matrix_pos_targets_only

    print 'Reading data from: "{0:s}" and "{1:s}"...'.format(
        radar_file_name_matrix[0, 0], label_file_name)
    this_radar_image_dict = storm_images.read_storm_images_and_labels(
        image_file_name=radar_file_name_matrix[0, 0],
        label_file_name=label_file_name, label_name=target_name,
        num_storm_objects_class_dict=num_examples_left_by_class_dict)

    if this_radar_image_dict is None:
        return None

    this_radar_image_dict = remove_storms_with_undefined_target(
        this_radar_image_dict)
    if not len(this_radar_image_dict[storm_images.STORM_IDS_KEY]):
        return None

    if rdp_filter_threshold_s02 is not None:
        this_radar_image_dict = remove_storms_with_low_rdp(
            storm_image_dict=this_radar_image_dict,
            rdp_filter_threshold_s02=rdp_filter_threshold_s02)
        if not len(this_radar_image_dict[storm_images.STORM_IDS_KEY]):
            return None

    if sounding_file_name is None:
        sounding_matrix = None
    else:
        sounding_dict, this_radar_image_dict = read_soundings(
            sounding_file_name=sounding_file_name,
            sounding_field_names=sounding_field_names,
            radar_image_dict=this_radar_image_dict)

        if sounding_dict is None or not len(sounding_dict[STORM_IDS_KEY]):
            return None

        sounding_matrix = sounding_dict[soundings_only.SOUNDING_MATRIX_KEY]
        sounding_field_names = sounding_dict[
            soundings_only.PRESSURELESS_FIELD_NAMES_KEY]

    target_values = this_radar_image_dict[storm_images.LABEL_VALUES_KEY]
    storm_ids_to_keep = this_radar_image_dict[storm_images.STORM_IDS_KEY]
    storm_times_to_keep_unix_sec = this_radar_image_dict[
        storm_images.VALID_TIMES_KEY]

    num_radar_fields = radar_file_name_matrix.shape[0]
    num_radar_heights = radar_file_name_matrix.shape[1]
    tuple_of_4d_image_matrices = ()

    for k in range(num_radar_heights):
        tuple_of_3d_image_matrices = ()

        for j in range(num_radar_fields):
            if not j == k == 0:
                print 'Reading data from: "{0:s}"...'.format(
                    radar_file_name_matrix[j, k])
                this_radar_image_dict = storm_images.read_storm_images(
                    netcdf_file_name=radar_file_name_matrix[j, k],
                    storm_ids_to_keep=storm_ids_to_keep,
                    valid_times_to_keep_unix_sec=storm_times_to_keep_unix_sec)

            tuple_of_3d_image_matrices += (
                this_radar_image_dict[storm_images.STORM_IMAGE_MATRIX_KEY],)

        tuple_of_4d_image_matrices += (
            dl_utils.stack_radar_fields(tuple_of_3d_image_matrices),)

    return {
        RADAR_IMAGE_MATRIX_KEY: dl_utils.stack_radar_heights(
            tuple_of_4d_image_matrices),
        TARGET_VALUES_KEY: target_values,
        SOUNDING_MATRIX_KEY: sounding_matrix,
        SOUNDING_FIELD_NAMES_KEY: sounding_field_names
    }


def _read_input_files_2d3d(
        reflectivity_file_names, azimuthal_shear_file_names,
        top_target_directory_name, target_name, num_examples_left_by_class_dict,
        refl_file_names_pos_targets_only=None,
        az_shear_file_names_pos_targets_only=None, sounding_file_name=None,
        sounding_field_names=None):
    """Reads data for `storm_image_generator_2d3d_myrorss`.

    F_a = number of azimuthal-shear fields
    m = number of rows per reflectivity image
    n = number of columns per reflectivity image
    H_r = number of heights per reflectivity image
    M = number of rows per azimuthal-shear image
    N = number of columns per azimuthal-shear image

    t_0 = file time = time step or SPC date

    :param reflectivity_file_names: list (length H_r) of paths to reflectivity
        files.  Each file should be readable by `storm_images.read_storm_images`
        and contain storm-centered reflectivity images at t_0.
    :param azimuthal_shear_file_names: list (length F_a) of paths to
        azimuthal-shear files.  Each file should be readable by
        `storm_images.read_storm_images` and contain storm-centered
        azimuthal-shear images at t_0.
    :param top_target_directory_name: See doc for `_read_input_files_2d`.
    :param target_name: Same.
    :param num_examples_left_by_class_dict: Same.
    :param refl_file_names_pos_targets_only: Same as `reflectivity_file_names`,
        but only for storm objects with positive target values.
    :param az_shear_file_names_pos_targets_only: Same as
        `azimuthal_shear_file_names`, but only for storm objects with positive
        target values.
    :param sounding_file_name: See doc for `_read_input_files_2d`.
    :param sounding_field_names: Same.
    :return: example_dict: Dictionary with the following keys.
    example_dict['reflectivity_image_matrix_dbz']: numpy array
        (E x m x n x H_r x 1) of storm-centered reflectivity images.
    example_dict['azimuthal_shear_image_matrix_s01']: numpy array
        (E x M x N x F_a) of storm-centered azimuthal-shear images.
    example_dict['target_values']: See doc for `_read_input_files_2d`.
    example_dict['sounding_matrix']: Same.
    example_dict['sounding_field_names']: Same.
    """

    label_file_name = storm_images.find_storm_label_file(
        storm_image_file_name=reflectivity_file_names[0],
        top_label_directory_name=top_target_directory_name,
        label_name=target_name, raise_error_if_missing=True)

    if refl_file_names_pos_targets_only is not None:
        if not _need_negative_target_values(num_examples_left_by_class_dict):
            reflectivity_file_names = refl_file_names_pos_targets_only
            azimuthal_shear_file_names = az_shear_file_names_pos_targets_only

    print 'Reading data from: "{0:s}" and "{1:s}"...'.format(
        reflectivity_file_names[0], label_file_name)
    this_radar_image_dict = storm_images.read_storm_images_and_labels(
        image_file_name=reflectivity_file_names[0],
        label_file_name=label_file_name, label_name=target_name,
        num_storm_objects_class_dict=num_examples_left_by_class_dict)

    if this_radar_image_dict is None:
        return None

    this_radar_image_dict = remove_storms_with_undefined_target(
        this_radar_image_dict)
    if not len(this_radar_image_dict[storm_images.STORM_IDS_KEY]):
        return None

    if sounding_file_name is None:
        sounding_matrix = None
    else:
        sounding_dict, this_radar_image_dict = read_soundings(
            sounding_file_name=sounding_file_name,
            sounding_field_names=sounding_field_names,
            radar_image_dict=this_radar_image_dict)

        if sounding_dict is None or not len(sounding_dict[STORM_IDS_KEY]):
            return None

        sounding_matrix = sounding_dict[soundings_only.SOUNDING_MATRIX_KEY]
        sounding_field_names = sounding_dict[
            soundings_only.PRESSURELESS_FIELD_NAMES_KEY]

    target_values = this_radar_image_dict[storm_images.LABEL_VALUES_KEY]
    storm_ids_to_keep = this_radar_image_dict[storm_images.STORM_IDS_KEY]
    storm_times_to_keep_unix_sec = this_radar_image_dict[
        storm_images.VALID_TIMES_KEY]

    num_reflectivity_heights = len(reflectivity_file_names)
    tuple_of_4d_refl_matrices = ()

    for j in range(num_reflectivity_heights):
        if j != 0:
            print 'Reading data from: "{0:s}"...'.format(
                reflectivity_file_names[j])
            this_radar_image_dict = storm_images.read_storm_images(
                netcdf_file_name=reflectivity_file_names[j],
                storm_ids_to_keep=storm_ids_to_keep,
                valid_times_to_keep_unix_sec=storm_times_to_keep_unix_sec)

        this_4d_matrix = dl_utils.stack_radar_fields(
            (this_radar_image_dict[storm_images.STORM_IMAGE_MATRIX_KEY],))
        tuple_of_4d_refl_matrices += (this_4d_matrix,)

    num_azimuthal_shear_fields = len(azimuthal_shear_file_names)
    tuple_of_3d_az_shear_matrices = ()

    for j in range(num_azimuthal_shear_fields):
        print 'Reading data from: "{0:s}"...'.format(
            azimuthal_shear_file_names[j])
        this_radar_image_dict = storm_images.read_storm_images(
            netcdf_file_name=azimuthal_shear_file_names[j],
            storm_ids_to_keep=storm_ids_to_keep,
            valid_times_to_keep_unix_sec=storm_times_to_keep_unix_sec)

        tuple_of_3d_az_shear_matrices += (
            this_radar_image_dict[storm_images.STORM_IMAGE_MATRIX_KEY],)

    return {
        REFLECTIVITY_IMAGE_MATRIX_KEY: dl_utils.stack_radar_heights(
            tuple_of_4d_refl_matrices),
        AZ_SHEAR_IMAGE_MATRIX_KEY: dl_utils.stack_radar_fields(
            tuple_of_3d_az_shear_matrices),
        TARGET_VALUES_KEY: target_values,
        SOUNDING_MATRIX_KEY: sounding_matrix,
        SOUNDING_FIELD_NAMES_KEY: sounding_field_names
    }


def check_input_args(
        num_examples_per_batch, num_examples_per_file_time, normalize_by_batch,
        radar_file_name_matrix, num_radar_dimensions, binarize_target,
        radar_file_name_matrix_pos_targets_only=None,
        sounding_field_names=None):
    """Error-checking of input arguments to generator.

    T = number of storm times (initial times, not valid times)

    :param num_examples_per_batch: Number of examples (storm objects) per batch.
    :param num_examples_per_file_time: Number of examples (storm objects) per
        file.  If each file contains one time step (SPC date), this will be
        number of examples per time step (SPC date).
    :param normalize_by_batch: Used to normalize radar images (see doc for
        `deep_learning_utils.normalize_predictor_matrix`).
    :param radar_file_name_matrix: numpy array of paths to radar files.
        Should be created by `find_2d_input_files` or `find_3d_input_files`, and
        the length of the first axis should be T.
    :param num_radar_dimensions: Number of radar dimensions.  This should be the
        number of dimensions in `radar_file_name_matrix`.
    :param binarize_target: See doc for `_select_batch`.
    :param radar_file_name_matrix_pos_targets_only: [optional]
        Same as `radar_file_name_matrix`, but only for storm objects with
        positive target values.
    :param sounding_field_names: [optional]
        1-D list with names of sounding fields (pressureless fields) to use.  If
        `sounding_field_names is None`, will use all sounding fields.
    """

    error_checking.assert_is_integer(num_examples_per_batch)
    error_checking.assert_is_geq(num_examples_per_batch, 10)
    error_checking.assert_is_integer(num_examples_per_file_time)
    error_checking.assert_is_geq(num_examples_per_file_time, 2)
    error_checking.assert_is_boolean(normalize_by_batch)
    error_checking.assert_is_boolean(binarize_target)

    error_checking.assert_is_integer(num_radar_dimensions)
    error_checking.assert_is_geq(num_radar_dimensions, 2)
    error_checking.assert_is_leq(num_radar_dimensions, 3)
    error_checking.assert_is_numpy_array(
        radar_file_name_matrix, num_dimensions=num_radar_dimensions)

    if radar_file_name_matrix_pos_targets_only is not None:
        error_checking.assert_is_numpy_array(
            radar_file_name_matrix_pos_targets_only,
            exact_dimensions=numpy.array(radar_file_name_matrix.shape))

    if sounding_field_names is not None:
        error_checking.assert_is_numpy_array(
            numpy.array(sounding_field_names), num_dimensions=1)


def remove_storms_with_undefined_target(storm_image_dict):
    """Removes storm objects with undefined target value (-1).

    :param storm_image_dict: Dictionary created by
        `storm_images.read_storm_images_and_labels`.
    :return: storm_image_dict: Same as input, but maybe with fewer storm
        objects.
    """

    valid_indices = numpy.where(
        storm_image_dict[storm_images.LABEL_VALUES_KEY] !=
        labels.INVALID_STORM_INTEGER)[0]

    keys_to_change = [
        storm_images.STORM_IMAGE_MATRIX_KEY, storm_images.STORM_IDS_KEY,
        storm_images.VALID_TIMES_KEY, storm_images.LABEL_VALUES_KEY,
    ]
    if storm_images.ROTATION_DIVERGENCE_PRODUCTS_KEY in storm_image_dict:
        keys_to_change += [storm_images.ROTATION_DIVERGENCE_PRODUCTS_KEY]

    for this_key in keys_to_change:
        if this_key == storm_images.STORM_IDS_KEY:
            storm_image_dict[this_key] = [
                storm_image_dict[this_key][i] for i in valid_indices]
        else:
            storm_image_dict[this_key] = storm_image_dict[this_key][
                valid_indices, ...]

    return storm_image_dict


def remove_storms_with_low_rdp(storm_image_dict, rdp_filter_threshold_s02):
    """Removes storm objects with low RDP (rotation-divergence product).

    :param storm_image_dict: Dictionary created by
        `storm_images.read_storm_images_and_labels`.
    :param rdp_filter_threshold_s02: Minimum RDP.  All storm objects with RDP
        < `rdp_filter_threshold_s02` will be removed.
    :return: storm_image_dict: Same as input, but maybe with fewer storm
        objects.
    """

    error_checking.assert_is_greater(rdp_filter_threshold_s02, 0.)
    valid_indices = numpy.where(
        storm_image_dict[storm_images.ROTATION_DIVERGENCE_PRODUCTS_KEY] >=
        rdp_filter_threshold_s02)[0]

    keys_to_change = [
        storm_images.STORM_IMAGE_MATRIX_KEY, storm_images.STORM_IDS_KEY,
        storm_images.VALID_TIMES_KEY, storm_images.LABEL_VALUES_KEY,
        storm_images.ROTATION_DIVERGENCE_PRODUCTS_KEY
    ]

    for this_key in keys_to_change:
        if this_key == storm_images.STORM_IDS_KEY:
            storm_image_dict[this_key] = [
                storm_image_dict[this_key][i] for i in valid_indices]
        else:
            storm_image_dict[this_key] = storm_image_dict[this_key][
                valid_indices, ...]

    return storm_image_dict


def separate_radar_files_2d3d(
        radar_file_name_matrix, test_mode=False, field_name_by_pair=None,
        height_by_pair_m_asl=None):
    """Separates input files for `storm_image_generator_2d3d_myrorss`.

    Specifically, separates these files into two arrays: one for reflectivity
    and one for azimuthal shear.

    F_a = number of azimuthal-shear fields
    H_r = number of heights per reflectivity image

    :param radar_file_name_matrix: numpy array (T x [H_r + F_a]) of paths to
        image files.  This should be created by `find_radar_files_2d`.
    :param test_mode: Leave this False.
    :param field_name_by_pair: Leave this empty.
    :param height_by_pair_m_asl: Leave this empty.
    :return: reflectivity_file_name_matrix: numpy array (T x H_r) of paths to
        reflectivity files.
    :return: az_shear_file_name_matrix: numpy array (T x F_a) of paths to
        azimuthal-shear files.
    :raises: ValueError: if `radar_file_name_matrix` does not contain
        reflectivity files.
    :raises: ValueError: if `radar_file_name_matrix` does not contain azimuthal-
        shear files.
    :raises: ValueError: if `radar_file_name_matrix` contains any files other
        than reflectivity and azimuthal shear.
    """

    error_checking.assert_is_numpy_array(
        radar_file_name_matrix, num_dimensions=2)
    error_checking.assert_is_boolean(test_mode)
    num_field_height_pairs = radar_file_name_matrix.shape[1]

    if not test_mode:
        field_name_by_pair = [''] * num_field_height_pairs
        height_by_pair_m_asl = numpy.full(num_field_height_pairs, -1, dtype=int)

        for j in range(num_field_height_pairs):
            this_radar_image_dict = storm_images.read_storm_images(
                netcdf_file_name=radar_file_name_matrix[0, j])
            field_name_by_pair[j] = this_radar_image_dict[
                storm_images.RADAR_FIELD_NAME_KEY]
            height_by_pair_m_asl[j] = this_radar_image_dict[
                storm_images.RADAR_HEIGHT_KEY]

    reflectivity_indices = numpy.where(numpy.array(
        [s == radar_utils.REFL_NAME for s in field_name_by_pair]))[0]
    if not len(reflectivity_indices):
        error_string = (
            '\n\n{0:s}\nFields in `radar_file_name_matrix` (listed above) do '
            'not include "{1:s}".'
        ).format(str(field_name_by_pair), radar_utils.REFL_NAME)
        raise ValueError(error_string)

    azimuthal_shear_indices = numpy.where(numpy.array(
        [s in radar_utils.SHEAR_NAMES for s in field_name_by_pair]))[0]
    if not len(azimuthal_shear_indices):
        error_string = (
            '\n\n{0:s}\nFields in `radar_file_name_matrix` (listed above) do '
            'not include azimuthal shear.'
        ).format(str(field_name_by_pair))
        raise ValueError(error_string)

    other_field_flags = numpy.full(num_field_height_pairs, True, dtype=bool)
    other_field_flags[reflectivity_indices] = False
    other_field_flags[azimuthal_shear_indices] = False
    if numpy.any(other_field_flags):
        other_field_indices = numpy.where(other_field_flags)[0]
        other_field_names = [field_name_by_pair[i] for i in other_field_indices]

        error_string = (
            '\n\n{0:s}\n`radar_file_name_matrix` includes fields other than '
            'reflectivity and azimuthal shear (listed above).'
        ).format(str(other_field_names))
        raise ValueError(error_string)

    sort_indices = numpy.argsort(height_by_pair_m_asl[reflectivity_indices])
    reflectivity_indices = reflectivity_indices[sort_indices]
    reflectivity_file_name_matrix = radar_file_name_matrix[
        ..., reflectivity_indices]

    sort_indices = numpy.argsort(
        numpy.array(field_name_by_pair)[azimuthal_shear_indices])
    azimuthal_shear_indices = azimuthal_shear_indices[sort_indices]
    az_shear_file_name_matrix = radar_file_name_matrix[
        ..., azimuthal_shear_indices]

    return reflectivity_file_name_matrix, az_shear_file_name_matrix


def find_radar_files_2d(
        top_directory_name, radar_source, radar_field_names,
        first_file_time_unix_sec, last_file_time_unix_sec,
        one_file_per_time_step, shuffle_times=True,
        top_directory_name_pos_targets_only=None, radar_heights_m_asl=None,
        reflectivity_heights_m_asl=None):
    """Finds input files for either of the following generators.

    - storm_image_generator_2d
    - storm_image_generator_2d3d_myrorss

    If `one_file_per_time_step = True`, this method will return files containing
    one time step each.  Otherwise, will return files containing one SPC date
    each.

    :param top_directory_name: Name of top-level directory with storm-centered
        radar images.
    :param radar_source: Name of data source.
    :param radar_field_names: 1-D list with names of radar fields.
    :param first_file_time_unix_sec: Start time.  This method will seek files
        for all times from `first_file_time_unix_sec`...
        `last_file_time_unix_sec`.  If `one_file_per_time_step = False`, this
        can be any time on the first SPC date.
    :param last_file_time_unix_sec: See above.
    :param one_file_per_time_step: See general discussion above.
    :param shuffle_times: Boolean flag.  If True, will shuffle times randomly.
    :param top_directory_name_pos_targets_only: Name of top-level directory
        with storm-centered radar images, but only for storm objects with
        positive target values.
    :param radar_heights_m_asl: [used only if radar_source = "gridrad"]
        1-D numpy array of radar heights (metres above sea level).
    :param reflectivity_heights_m_asl: [used only if radar_source != "gridrad"]
        1-D numpy array of reflectivity heights (metres above sea level).
    :return: radar_file_name_matrix: T-by-C numpy array of paths to files with
        storm-centered radar images.
    :return: radar_file_name_matrix_pos_targets_only:
        [None if `top_directory_name_pos_targets_only is None`]
        Same as `radar_file_name_matrix`, but only for storm objects with
        positive target values.
    :return: file_times_unix_sec: length-T numpy array of file times.
    """

    radar_utils.check_data_source(radar_source)

    if radar_source == radar_utils.GRIDRAD_SOURCE_ID:
        storm_image_file_dict = storm_images.find_many_files_gridrad(
            top_directory_name=top_directory_name,
            radar_field_names=radar_field_names,
            radar_heights_m_asl=radar_heights_m_asl,
            start_time_unix_sec=first_file_time_unix_sec,
            end_time_unix_sec=last_file_time_unix_sec,
            one_file_per_time_step=one_file_per_time_step,
            raise_error_if_all_missing=True)
    else:
        storm_image_file_dict = storm_images.find_many_files_myrorss_or_mrms(
            top_directory_name=top_directory_name, radar_source=radar_source,
            radar_field_names=radar_field_names,
            start_time_unix_sec=first_file_time_unix_sec,
            end_time_unix_sec=last_file_time_unix_sec,
            one_file_per_time_step=one_file_per_time_step,
            reflectivity_heights_m_asl=reflectivity_heights_m_asl,
            raise_error_if_all_missing=True, raise_error_if_any_missing=False)

    radar_file_name_matrix = storm_image_file_dict[
        storm_images.IMAGE_FILE_NAME_MATRIX_KEY]
    image_times_unix_sec = storm_image_file_dict[storm_images.VALID_TIMES_KEY]

    if radar_source == radar_utils.GRIDRAD_SOURCE_ID:
        num_field_height_pairs = (
            radar_file_name_matrix.shape[1] * radar_file_name_matrix.shape[2])
        radar_file_name_matrix = numpy.reshape(
            radar_file_name_matrix,
            (len(image_times_unix_sec), num_field_height_pairs))

    time_missing_indices = numpy.unique(
        numpy.where(radar_file_name_matrix == '')[0])
    radar_file_name_matrix = numpy.delete(
        radar_file_name_matrix, time_missing_indices, axis=0)
    image_times_unix_sec = numpy.delete(
        image_times_unix_sec, time_missing_indices)

    if shuffle_times:
        num_image_times = radar_file_name_matrix.shape[0]
        image_time_indices = numpy.linspace(
            0, num_image_times - 1, num=num_image_times, dtype=int)
        numpy.random.shuffle(image_time_indices)

        radar_file_name_matrix = radar_file_name_matrix[image_time_indices, ...]
        image_times_unix_sec = image_times_unix_sec[image_time_indices]

    if top_directory_name_pos_targets_only is None:
        return radar_file_name_matrix, None, image_times_unix_sec

    num_image_times = radar_file_name_matrix.shape[0]
    num_field_height_pairs = radar_file_name_matrix.shape[1]
    radar_file_name_matrix_pos_targets_only = numpy.full(
        (num_image_times, num_field_height_pairs), '', dtype=object)

    for j in range(num_field_height_pairs):
        this_storm_image_dict = storm_images.read_storm_images(
            netcdf_file_name=radar_file_name_matrix[0, j], return_images=False)
        this_field_name = this_storm_image_dict[
            storm_images.RADAR_FIELD_NAME_KEY]
        this_height_m_asl = this_storm_image_dict[storm_images.RADAR_HEIGHT_KEY]

        for i in range(num_image_times):
            (this_time_unix_sec, this_spc_date_string
            ) = storm_images.image_file_name_to_time(
                radar_file_name_matrix[i, j])

            radar_file_name_matrix_pos_targets_only[i, j] = (
                storm_images.find_storm_image_file(
                    top_directory_name=top_directory_name_pos_targets_only,
                    spc_date_string=this_spc_date_string,
                    radar_source=radar_source, radar_field_name=this_field_name,
                    radar_height_m_asl=this_height_m_asl,
                    unix_time_sec=this_time_unix_sec,
                    raise_error_if_missing=True))

    return (radar_file_name_matrix, radar_file_name_matrix_pos_targets_only,
            image_times_unix_sec)


def find_radar_files_3d(
        top_directory_name, radar_source, radar_field_names,
        radar_heights_m_asl, first_file_time_unix_sec, last_file_time_unix_sec,
        one_file_per_time_step, shuffle_times=True,
        top_directory_name_pos_targets_only=None):
    """Finds input files for `storm_image_generator_3d`.

    :param top_directory_name: See doc for `find_radar_files_2d`.
    :param radar_source: Same.
    :param radar_field_names: Same.
    :param radar_heights_m_asl: Same.
    :param first_file_time_unix_sec: Same.
    :param last_file_time_unix_sec: Same.
    :param one_file_per_time_step: Same.
    :param shuffle_times: Same.
    :param top_directory_name_pos_targets_only: Same.
    :return: radar_file_name_matrix: numpy array (T x F_r x H_r) of paths to
        files with storm-centered radar images.
    :return: radar_file_name_matrix_pos_targets_only: See doc for
        `find_radar_files_2d`.
    :return: file_times_unix_sec: length-T numpy array of file times.
    """

    radar_utils.check_data_source(radar_source)

    if radar_source == radar_utils.GRIDRAD_SOURCE_ID:
        storm_image_file_dict = storm_images.find_many_files_gridrad(
            top_directory_name=top_directory_name,
            radar_field_names=radar_field_names,
            radar_heights_m_asl=radar_heights_m_asl,
            start_time_unix_sec=first_file_time_unix_sec,
            end_time_unix_sec=last_file_time_unix_sec,
            one_file_per_time_step=one_file_per_time_step,
            raise_error_if_all_missing=True)
    else:
        storm_image_file_dict = storm_images.find_many_files_myrorss_or_mrms(
            top_directory_name=top_directory_name, radar_source=radar_source,
            radar_field_names=[radar_utils.REFL_NAME],
            start_time_unix_sec=first_file_time_unix_sec,
            end_time_unix_sec=last_file_time_unix_sec,
            one_file_per_time_step=one_file_per_time_step,
            reflectivity_heights_m_asl=radar_heights_m_asl,
            raise_error_if_all_missing=True, raise_error_if_any_missing=False)

    radar_file_name_matrix = storm_image_file_dict[
        storm_images.IMAGE_FILE_NAME_MATRIX_KEY]
    image_times_unix_sec = storm_image_file_dict[storm_images.VALID_TIMES_KEY]

    if radar_source != radar_utils.GRIDRAD_SOURCE_ID:
        radar_file_name_matrix = numpy.reshape(
            radar_file_name_matrix,
            (len(image_times_unix_sec), 1, len(radar_heights_m_asl)))

    time_missing_indices = numpy.unique(
        numpy.where(radar_file_name_matrix == '')[0])
    radar_file_name_matrix = numpy.delete(
        radar_file_name_matrix, time_missing_indices, axis=0)
    image_times_unix_sec = numpy.delete(
        image_times_unix_sec, time_missing_indices)

    if shuffle_times:
        num_image_times = radar_file_name_matrix.shape[0]
        image_time_indices = numpy.linspace(
            0, num_image_times - 1, num=num_image_times, dtype=int)
        numpy.random.shuffle(image_time_indices)

        radar_file_name_matrix = radar_file_name_matrix[image_time_indices, ...]
        image_times_unix_sec = image_times_unix_sec[image_time_indices]

    if top_directory_name_pos_targets_only is None:
        return radar_file_name_matrix, None, image_times_unix_sec

    num_image_times = radar_file_name_matrix.shape[0]
    num_fields = radar_file_name_matrix.shape[1]
    num_heights = radar_file_name_matrix.shape[2]
    radar_file_name_matrix_pos_targets_only = numpy.full(
        (num_image_times, num_fields, num_heights), '', dtype=object)

    for j in range(num_fields):
        for k in range(num_heights):
            this_storm_image_dict = storm_images.read_storm_images(
                netcdf_file_name=radar_file_name_matrix[0, j, k],
                return_images=False)
            this_field_name = this_storm_image_dict[
                storm_images.RADAR_FIELD_NAME_KEY]
            this_height_m_asl = this_storm_image_dict[
                storm_images.RADAR_HEIGHT_KEY]

            for i in range(num_image_times):
                (this_time_unix_sec, this_spc_date_string
                ) = storm_images.image_file_name_to_time(
                    radar_file_name_matrix[i, j, k])

                radar_file_name_matrix_pos_targets_only[i, j, k] = (
                    storm_images.find_storm_image_file(
                        top_directory_name=top_directory_name_pos_targets_only,
                        spc_date_string=this_spc_date_string,
                        radar_source=radar_source,
                        radar_field_name=this_field_name,
                        radar_height_m_asl=this_height_m_asl,
                        unix_time_sec=this_time_unix_sec,
                        raise_error_if_missing=True))

    return (radar_file_name_matrix, radar_file_name_matrix_pos_targets_only,
            image_times_unix_sec)


def find_sounding_files(
        top_sounding_dir_name, radar_file_name_matrix, target_name,
        lag_time_for_convective_contamination_sec):
    """Finds sounding file for each radar time.

    :param top_sounding_dir_name: Name of top-level directory with sounding
        files, findable by `soundings_only.find_sounding_file`.
    :param radar_file_name_matrix: numpy array created by either
        `find_radar_files_2d` or `find_radar_files_3d`.  Length of the first
        axis is T.
    :param target_name: Name of target variable.
    :param lag_time_for_convective_contamination_sec: See doc for
        `soundings_only.interp_soundings_to_storm_objects`.
    :return: sounding_file_names: length-T list of paths to files with storm-
        centered soundings.
    """

    error_checking.assert_is_numpy_array(radar_file_name_matrix)
    num_radar_dimensions = len(radar_file_name_matrix.shape)
    error_checking.assert_is_geq(num_radar_dimensions, 2)
    error_checking.assert_is_leq(num_radar_dimensions, 3)

    target_param_dict = labels.column_name_to_label_params(target_name)
    min_lead_time_sec = target_param_dict[labels.MIN_LEAD_TIME_KEY]
    max_lead_time_sec = target_param_dict[labels.MAX_LEAD_TIME_KEY]
    mean_lead_time_sec = numpy.mean(
        numpy.array([min_lead_time_sec, max_lead_time_sec], dtype=float))
    mean_lead_time_sec = int(numpy.round(mean_lead_time_sec))

    print (
        'For each radar time, finding sounding file with lead time = {0:d} sec '
        'and lag time (for convective contamination) = {1:d} sec...'
    ).format(mean_lead_time_sec, lag_time_for_convective_contamination_sec)

    num_radar_times = radar_file_name_matrix.shape[0]
    sounding_file_names = [''] * num_radar_times

    for i in range(num_radar_times):
        if num_radar_dimensions == 2:
            this_file_name = radar_file_name_matrix[i, 0]
        else:
            this_file_name = radar_file_name_matrix[i, 0, 0]

        this_time_unix_sec, this_spc_date_string = (
            storm_images.image_file_name_to_time(this_file_name))
        sounding_file_names[i] = soundings_only.find_sounding_file(
            top_directory_name=top_sounding_dir_name,
            spc_date_string=this_spc_date_string,
            lead_time_seconds=mean_lead_time_sec,
            lag_time_for_convective_contamination_sec=
            lag_time_for_convective_contamination_sec,
            init_time_unix_sec=this_time_unix_sec, raise_error_if_missing=True)

    return sounding_file_names


def read_soundings(sounding_file_name, sounding_field_names, radar_image_dict):
    """Reads storm-centered soundings to match with storm-centered radar images.

    :param sounding_file_name: See doc for `_read_input_files_2d`.
    :param sounding_field_names: Same.
    :param radar_image_dict: Dictionary created by
        `storm_images.read_storm_images_and_labels`, for the same file time as
        the sounding file.
    :return: sounding_dict: See output doc for `soundings_only.read_soundings`.
    :return: radar_image_dict: Same as input, except that some storm objects may
        be removed.
    """

    print 'Reading data from: "{0:s}"...'.format(sounding_file_name)
    sounding_dict, _ = soundings_only.read_soundings(
        netcdf_file_name=sounding_file_name,
        pressureless_field_names_to_keep=sounding_field_names,
        storm_ids_to_keep=radar_image_dict[storm_images.STORM_IDS_KEY],
        init_times_to_keep_unix_sec=radar_image_dict[
            storm_images.VALID_TIMES_KEY])

    if not len(sounding_dict[soundings_only.STORM_IDS_KEY]):
        return None, None

    orig_storm_ids_as_numpy_array = numpy.array(
        radar_image_dict[storm_images.STORM_IDS_KEY])
    orig_storm_times_unix_sec = radar_image_dict[
        storm_images.VALID_TIMES_KEY] + 0

    num_storm_objects_kept = len(sounding_dict[soundings_only.STORM_IDS_KEY])
    storm_object_kept_indices = []

    for i in range(num_storm_objects_kept):
        this_index = numpy.where(numpy.logical_and(
            orig_storm_ids_as_numpy_array ==
            sounding_dict[soundings_only.STORM_IDS_KEY][i],
            orig_storm_times_unix_sec ==
            sounding_dict[soundings_only.INITIAL_TIMES_KEY][i]))[0][0]

        storm_object_kept_indices.append(this_index)

    storm_object_kept_indices = numpy.array(
        storm_object_kept_indices, dtype=int)

    radar_image_dict[storm_images.STORM_IDS_KEY] = sounding_dict[
        soundings_only.STORM_IDS_KEY]
    radar_image_dict[storm_images.VALID_TIMES_KEY] = sounding_dict[
        soundings_only.INITIAL_TIMES_KEY]
    radar_image_dict[storm_images.STORM_IMAGE_MATRIX_KEY] = radar_image_dict[
        storm_images.STORM_IMAGE_MATRIX_KEY][storm_object_kept_indices, ...]
    if storm_images.LABEL_VALUES_KEY in radar_image_dict:
        radar_image_dict[storm_images.LABEL_VALUES_KEY] = radar_image_dict[
            storm_images.LABEL_VALUES_KEY][storm_object_kept_indices]

    return sounding_dict, radar_image_dict


def storm_image_generator_2d(
        radar_file_name_matrix, top_target_directory_name,
        num_examples_per_batch, num_examples_per_file_time, target_name,
        radar_file_name_matrix_pos_targets_only=None, binarize_target=False,
        radar_normalization_dict=dl_utils.DEFAULT_RADAR_NORMALIZATION_DICT,
        sampling_fraction_by_class_dict=None, sounding_field_names=None,
        top_sounding_dir_name=None,
        sounding_lag_time_for_convective_contamination_sec=None,
        sounding_normalization_dict=
        dl_utils.DEFAULT_SOUNDING_NORMALIZATION_DICT,
        loop_thru_files_once=False):
    """Generates examples with 2-D radar images.

    Each example consists of storm-centered 2-D radar images, and possibly the
    storm-centered sounding, for one storm object.

    :param radar_file_name_matrix: T-by-C numpy array of paths to radar files,
        created by `find_radar_files_2d`.
    :param top_target_directory_name: Name of top-level directory with target
        values (storm-hazard labels).  Files within this directory should be
        findable by `labels.find_label_file`.
    :param num_examples_per_batch: See doc for `check_input_args`.
    :param num_examples_per_file_time: Same.
    :param target_name: Name of target variable.
    :param radar_file_name_matrix_pos_targets_only: [may be None]
        See doc for `find_radar_files_2d`.
    :param binarize_target: See doc for `_select_batch`.
    :param radar_normalization_dict: Used to normalize radar images (see doc for
        `deep_learning_utils.normalize_predictor_matrix`).
    :param sampling_fraction_by_class_dict: Dictionary, where each key is
        the integer representing a class (-2 for "dead storm") and each value is
        the corresponding sampling fraction.  Sampling fractions will be applied
        to each batch.
    :param sounding_field_names: list (length F_s) with names of sounding
        fields.  Each must be accepted by
        `soundings_only.check_pressureless_field_name`.
    :param top_sounding_dir_name: See doc for `find_sounding_files`.
    :param sounding_lag_time_for_convective_contamination_sec: Same.
    :param sounding_normalization_dict: Used to normalize soundings (see doc for
        `deep_learning_utils.normalize_sounding_matrix`).
    :param loop_thru_files_once: Boolean flag.  If True, this generator will
        loop through `radar_file_name_matrix` only once.  If False, once this
        generator has reached the end of `radar_file_name_matrix`, it will go
        back to the beginning and keep looping.

    If `sounding_field_names is None`, this method returns...

    :return: radar_image_matrix: E-by-M-by-N-by-C numpy array of storm-centered
        radar images.
    :return: target_matrix: E-by-K numpy array of target values (all 0 or 1, but
        with array type "float64").  If target_matrix[i, k] = 1, the [i]th
        storm object belongs to the [k]th class.  Classes are mutually exclusive
        and collectively exhaustive, so the sum across each row is 1.

    If `sounding_field_names is not None`, this method returns...

    :return: predictor_list: List with the following items.
    predictor_list[0] = radar_image_matrix: See documentation above.
    predictor_list[1] = sounding_matrix: numpy array (E x H_s x F_s) of storm-
        centered soundings.
    :return: target_matrix: See documentation above.
    """

    error_checking.assert_is_boolean(loop_thru_files_once)
    check_input_args(
        num_examples_per_batch=num_examples_per_batch,
        num_examples_per_file_time=num_examples_per_file_time,
        normalize_by_batch=False, radar_file_name_matrix=radar_file_name_matrix,
        num_radar_dimensions=2, binarize_target=binarize_target,
        radar_file_name_matrix_pos_targets_only=
        radar_file_name_matrix_pos_targets_only,
        sounding_field_names=sounding_field_names)

    if sounding_field_names is None:
        sounding_file_names = None
    else:
        sounding_file_names = find_sounding_files(
            top_sounding_dir_name=top_sounding_dir_name,
            radar_file_name_matrix=radar_file_name_matrix,
            target_name=target_name,
            lag_time_for_convective_contamination_sec=
            sounding_lag_time_for_convective_contamination_sec)

    num_channels = radar_file_name_matrix.shape[1]
    field_name_by_channel = [''] * num_channels
    for j in range(num_channels):
        this_radar_image_dict = storm_images.read_storm_images(
            netcdf_file_name=radar_file_name_matrix[0, j])
        field_name_by_channel[j] = this_radar_image_dict[
            storm_images.RADAR_FIELD_NAME_KEY]

    num_examples_per_batch_by_class_dict = _get_num_examples_per_batch_by_class(
        num_examples_per_batch=num_examples_per_batch, target_name=target_name,
        sampling_fraction_by_class_dict=sampling_fraction_by_class_dict)

    num_classes = labels.column_name_to_num_classes(
        column_name=target_name, include_dead_storms=False)
    num_file_times_per_batch = int(numpy.ceil(
        float(num_examples_per_batch) / num_examples_per_file_time))

    num_examples_in_memory_by_class_dict, _ = _determine_stopping_criterion(
        num_examples_per_batch=num_examples_per_batch,
        num_file_times_per_batch=num_file_times_per_batch,
        num_examples_per_batch_by_class_dict=
        num_examples_per_batch_by_class_dict,
        num_file_times_in_memory=0,
        sampling_fraction_by_class_dict=sampling_fraction_by_class_dict,
        target_values_in_memory=numpy.array([], dtype=int))

    num_file_times = radar_file_name_matrix.shape[0]

    # Initialize housekeeping variables.
    file_time_index = 0
    num_file_times_in_memory = 0
    full_radar_image_matrix = None
    full_sounding_matrix = None
    all_target_values = None

    while True:
        stopping_criterion = False
        if loop_thru_files_once and file_time_index >= num_file_times:
            raise StopIteration

        while not stopping_criterion:
            if file_time_index == num_file_times:
                if loop_thru_files_once:
                    if all_target_values is None:
                        raise StopIteration
                    break

                file_time_index = 0

            if all_target_values is None:
                num_examples_in_memory = 0
            else:
                num_examples_in_memory = len(all_target_values)

            num_examples_left_by_class_dict = _get_num_examples_left_by_class(
                num_examples_per_batch=num_examples_per_batch,
                num_file_times_per_batch=num_file_times_per_batch,
                num_examples_per_batch_by_class_dict=
                num_examples_per_batch_by_class_dict,
                num_examples_in_memory=num_examples_in_memory,
                num_file_times_in_memory=num_file_times_in_memory,
                num_examples_in_memory_by_class_dict=
                num_examples_in_memory_by_class_dict)

            if sounding_file_names is None:
                this_sounding_file_name = None
            else:
                this_sounding_file_name = sounding_file_names[file_time_index]

            if radar_file_name_matrix_pos_targets_only is None:
                these_radar_file_names_pos_targets_only = None
            else:
                these_radar_file_names_pos_targets_only = (
                    radar_file_name_matrix_pos_targets_only[
                        file_time_index, ...].tolist())

            this_example_dict = _read_input_files_2d(
                radar_file_names=radar_file_name_matrix[
                    file_time_index, ...].tolist(),
                top_target_directory_name=top_target_directory_name,
                target_name=target_name,
                num_examples_left_by_class_dict=num_examples_left_by_class_dict,
                radar_file_names_pos_targets_only=
                these_radar_file_names_pos_targets_only,
                sounding_file_name=this_sounding_file_name,
                sounding_field_names=sounding_field_names)
            print MINOR_SEPARATOR_STRING

            file_time_index += 1
            if this_example_dict is None:
                continue

            num_file_times_in_memory += 1
            sounding_field_names = this_example_dict[SOUNDING_FIELD_NAMES_KEY]

            if all_target_values is None:
                all_target_values = this_example_dict[TARGET_VALUES_KEY] + 0
                full_radar_image_matrix = this_example_dict[
                    RADAR_IMAGE_MATRIX_KEY] + 0.
                if sounding_file_names is not None:
                    full_sounding_matrix = this_example_dict[
                        SOUNDING_MATRIX_KEY] + 0.

            else:
                all_target_values = numpy.concatenate((
                    all_target_values, this_example_dict[TARGET_VALUES_KEY]))
                full_radar_image_matrix = numpy.concatenate(
                    (full_radar_image_matrix,
                     this_example_dict[RADAR_IMAGE_MATRIX_KEY]), axis=0)
                if sounding_file_names is not None:
                    full_sounding_matrix = numpy.concatenate(
                        (full_sounding_matrix,
                         this_example_dict[SOUNDING_MATRIX_KEY]), axis=0)

            (num_examples_in_memory_by_class_dict, stopping_criterion
            ) = _determine_stopping_criterion(
                num_examples_per_batch=num_examples_per_batch,
                num_file_times_per_batch=num_file_times_per_batch,
                num_examples_per_batch_by_class_dict=
                num_examples_per_batch_by_class_dict,
                num_file_times_in_memory=num_file_times_in_memory,
                sampling_fraction_by_class_dict=sampling_fraction_by_class_dict,
                target_values_in_memory=all_target_values)

        if sampling_fraction_by_class_dict is not None:
            batch_indices = dl_utils.sample_by_class(
                sampling_fraction_by_class_dict=sampling_fraction_by_class_dict,
                target_name=target_name, target_values=all_target_values,
                num_examples_total=num_examples_per_batch)

            full_radar_image_matrix = full_radar_image_matrix[
                batch_indices, ...]
            all_target_values = all_target_values[batch_indices]
            if sounding_file_names is not None:
                full_sounding_matrix = full_sounding_matrix[batch_indices, ...]

        full_radar_image_matrix = dl_utils.normalize_radar_images(
            radar_image_matrix=full_radar_image_matrix,
            field_names=field_name_by_channel,
            normalization_dict=radar_normalization_dict)

        if sounding_file_names is not None:
            full_sounding_matrix = dl_utils.normalize_soundings(
                sounding_matrix=full_sounding_matrix,
                pressureless_field_names=sounding_field_names,
                normalization_dict=sounding_normalization_dict)

        list_of_predictor_matrices, target_matrix = _select_batch(
            list_of_predictor_matrices=[
                full_radar_image_matrix, full_sounding_matrix],
            target_values=all_target_values,
            num_examples_per_batch=num_examples_per_batch,
            binarize_target=binarize_target, num_classes=num_classes)

        radar_image_matrix = list_of_predictor_matrices[0]
        sounding_matrix = list_of_predictor_matrices[1]

        # Update housekeeping variables.
        num_file_times_in_memory = 0
        full_radar_image_matrix = None
        full_sounding_matrix = None
        all_target_values = None

        for this_key in num_examples_in_memory_by_class_dict.keys():
            num_examples_in_memory_by_class_dict[this_key] = 0

        if sounding_file_names is None:
            yield (radar_image_matrix, target_matrix)
        else:
            yield ([radar_image_matrix, sounding_matrix], target_matrix)


def storm_image_generator_3d(
        radar_file_name_matrix, top_target_directory_name,
        num_examples_per_batch, num_examples_per_file_time, target_name,
        radar_file_name_matrix_pos_targets_only=None, binarize_target=False,
        radar_normalization_dict=dl_utils.DEFAULT_RADAR_NORMALIZATION_DICT,
        refl_masking_threshold_dbz=dl_utils.DEFAULT_REFL_MASK_THRESHOLD_DBZ,
        rdp_filter_threshold_s02=None, sampling_fraction_by_class_dict=None,
        sounding_field_names=None, top_sounding_dir_name=None,
        sounding_lag_time_for_convective_contamination_sec=None,
        sounding_normalization_dict=
        dl_utils.DEFAULT_SOUNDING_NORMALIZATION_DICT,
        loop_thru_files_once=False):
    """Generates examples with 3-D radar images.

    Each example consists of storm-centered 3-D radar images, and possibly the
    storm-centered sounding, for one storm object.

    :param radar_file_name_matrix: numpy array (T x F_r x H_r) of paths to
        radar-image files, created by `find_radar_files_3d`.
    :param top_target_directory_name: See doc for `storm_image_generator_2d`.
    :param num_examples_per_batch: Same.
    :param num_examples_per_file_time: Same.
    :param target_name: Same.
    :param radar_file_name_matrix_pos_targets_only: Same.
    :param binarize_target: Same.
    :param radar_normalization_dict: Same.
    :param refl_masking_threshold_dbz: Used to mask pixels with low reflectivity
        (see doc for `deep_learning_utils.mask_low_reflectivity_pixels`).
    :param rdp_filter_threshold_s02: Used as a pre-model filter, to remove storm
        objects with small rotation-divergence product (see doc for
        `storm_images.get_max_rdp_for_each_storm_object`).
    :param sampling_fraction_by_class_dict: See doc for
        `storm_image_generator_2d`.
    :param sounding_field_names: Same.
    :param top_sounding_dir_name: See doc for `find_sounding_files`.
    :param sounding_lag_time_for_convective_contamination_sec: Same.
    :param sounding_normalization_dict: See doc for `storm_image_generator_2d`.
    :param loop_thru_files_once: Same.

    If `sounding_field_names is None`, this method returns...

    :return: radar_image_matrix: numpy array (E x M x N x H_r x F_r) of storm-
        centered radar images.
    :return: target_matrix: See output doc for `storm_image_generator_2d`.

    If `sounding_field_names is not None`, this method returns...

    :return: predictor_list: List with the following items.
    predictor_list[0] = radar_image_matrix: See documentation above.
    predictor_list[1] = sounding_matrix: numpy array (E x H_s x F_s) of storm-
        centered soundings.
    :return: target_matrix: See documentation above.
    """

    error_checking.assert_is_boolean(loop_thru_files_once)
    check_input_args(
        num_examples_per_batch=num_examples_per_batch,
        num_examples_per_file_time=num_examples_per_file_time,
        normalize_by_batch=False, radar_file_name_matrix=radar_file_name_matrix,
        num_radar_dimensions=3, binarize_target=binarize_target,
        radar_file_name_matrix_pos_targets_only=
        radar_file_name_matrix_pos_targets_only,
        sounding_field_names=sounding_field_names)

    if sounding_field_names is None:
        sounding_file_names = None
    else:
        sounding_file_names = find_sounding_files(
            top_sounding_dir_name=top_sounding_dir_name,
            radar_file_name_matrix=radar_file_name_matrix,
            target_name=target_name,
            lag_time_for_convective_contamination_sec=
            sounding_lag_time_for_convective_contamination_sec)

    num_fields = radar_file_name_matrix.shape[1]
    radar_field_names = [''] * num_fields
    for j in range(num_fields):
        this_radar_image_dict = storm_images.read_storm_images(
            netcdf_file_name=radar_file_name_matrix[0, j, 0])
        radar_field_names[j] = this_radar_image_dict[
            storm_images.RADAR_FIELD_NAME_KEY]

    num_examples_per_batch_by_class_dict = _get_num_examples_per_batch_by_class(
        num_examples_per_batch=num_examples_per_batch, target_name=target_name,
        sampling_fraction_by_class_dict=sampling_fraction_by_class_dict)

    num_classes = labels.column_name_to_num_classes(
        column_name=target_name, include_dead_storms=False)
    num_file_times_per_batch = int(numpy.ceil(
        float(num_examples_per_batch) / num_examples_per_file_time))

    num_examples_in_memory_by_class_dict, _ = _determine_stopping_criterion(
        num_examples_per_batch=num_examples_per_batch,
        num_file_times_per_batch=num_file_times_per_batch,
        num_examples_per_batch_by_class_dict=
        num_examples_per_batch_by_class_dict,
        num_file_times_in_memory=0,
        sampling_fraction_by_class_dict=sampling_fraction_by_class_dict,
        target_values_in_memory=numpy.array([], dtype=int))

    num_file_times = radar_file_name_matrix.shape[0]

    # Initialize housekeeping variables.
    file_time_index = 0
    num_file_times_in_memory = 0
    full_radar_image_matrix = None
    full_sounding_matrix = None
    all_target_values = None

    while True:
        stopping_criterion = False
        if loop_thru_files_once and file_time_index >= num_file_times:
            raise StopIteration

        while not stopping_criterion:
            if file_time_index == num_file_times:
                if loop_thru_files_once:
                    if all_target_values is None:
                        raise StopIteration
                    break

                file_time_index = 0

            if full_radar_image_matrix is None:
                num_examples_in_memory = 0
            else:
                num_examples_in_memory = full_radar_image_matrix.shape[0]

            num_examples_left_by_class_dict = _get_num_examples_left_by_class(
                num_examples_per_batch=num_examples_per_batch,
                num_file_times_per_batch=num_file_times_per_batch,
                num_examples_per_batch_by_class_dict=
                num_examples_per_batch_by_class_dict,
                num_examples_in_memory=num_examples_in_memory,
                num_file_times_in_memory=num_file_times_in_memory,
                num_examples_in_memory_by_class_dict=
                num_examples_in_memory_by_class_dict)

            if sounding_file_names is None:
                this_sounding_file_name = None
            else:
                this_sounding_file_name = sounding_file_names[file_time_index]

            if radar_file_name_matrix_pos_targets_only is None:
                this_file_name_matrix_pos_targets_only = None
            else:
                this_file_name_matrix_pos_targets_only = (
                    radar_file_name_matrix_pos_targets_only[
                        file_time_index, ...])

            this_example_dict = _read_input_files_3d(
                radar_file_name_matrix=radar_file_name_matrix[
                    file_time_index, ...],
                top_target_directory_name=top_target_directory_name,
                target_name=target_name,
                num_examples_left_by_class_dict=num_examples_left_by_class_dict,
                rdp_filter_threshold_s02=rdp_filter_threshold_s02,
                radar_file_name_matrix_pos_targets_only=
                this_file_name_matrix_pos_targets_only,
                sounding_file_name=this_sounding_file_name,
                sounding_field_names=sounding_field_names)
            print MINOR_SEPARATOR_STRING

            file_time_index += 1
            if this_example_dict is None:
                continue

            num_file_times_in_memory += 1
            sounding_field_names = this_example_dict[SOUNDING_FIELD_NAMES_KEY]

            if all_target_values is None:
                all_target_values = this_example_dict[TARGET_VALUES_KEY] + 0
                full_radar_image_matrix = this_example_dict[
                    RADAR_IMAGE_MATRIX_KEY] + 0.
                if sounding_file_names is not None:
                    full_sounding_matrix = this_example_dict[
                        SOUNDING_MATRIX_KEY] + 0.

            else:
                all_target_values = numpy.concatenate((
                    all_target_values, this_example_dict[TARGET_VALUES_KEY]))
                full_radar_image_matrix = numpy.concatenate(
                    (full_radar_image_matrix,
                     this_example_dict[RADAR_IMAGE_MATRIX_KEY]), axis=0)
                if sounding_file_names is not None:
                    full_sounding_matrix = numpy.concatenate(
                        (full_sounding_matrix,
                         this_example_dict[SOUNDING_MATRIX_KEY]), axis=0)

            (num_examples_in_memory_by_class_dict, stopping_criterion
            ) = _determine_stopping_criterion(
                num_examples_per_batch=num_examples_per_batch,
                num_file_times_per_batch=num_file_times_per_batch,
                num_examples_per_batch_by_class_dict=
                num_examples_per_batch_by_class_dict,
                num_file_times_in_memory=num_file_times_in_memory,
                sampling_fraction_by_class_dict=sampling_fraction_by_class_dict,
                target_values_in_memory=all_target_values)

        if sampling_fraction_by_class_dict is not None:
            batch_indices = dl_utils.sample_by_class(
                sampling_fraction_by_class_dict=sampling_fraction_by_class_dict,
                target_name=target_name, target_values=all_target_values,
                num_examples_total=num_examples_per_batch)

            full_radar_image_matrix = full_radar_image_matrix[
                batch_indices, ...]
            all_target_values = all_target_values[batch_indices]
            if sounding_file_names is not None:
                full_sounding_matrix = full_sounding_matrix[batch_indices, ...]

        full_radar_image_matrix = dl_utils.mask_low_reflectivity_pixels(
            radar_image_matrix_3d=full_radar_image_matrix,
            field_names=radar_field_names,
            reflectivity_threshold_dbz=refl_masking_threshold_dbz)

        full_radar_image_matrix = dl_utils.normalize_radar_images(
            radar_image_matrix=full_radar_image_matrix,
            field_names=radar_field_names,
            normalization_dict=radar_normalization_dict)

        if sounding_file_names is not None:
            full_sounding_matrix = dl_utils.normalize_soundings(
                sounding_matrix=full_sounding_matrix,
                pressureless_field_names=sounding_field_names,
                normalization_dict=sounding_normalization_dict)

        list_of_predictor_matrices, target_matrix = _select_batch(
            list_of_predictor_matrices=[
                full_radar_image_matrix, full_sounding_matrix],
            target_values=all_target_values,
            num_examples_per_batch=num_examples_per_batch,
            binarize_target=binarize_target, num_classes=num_classes)

        radar_image_matrix = list_of_predictor_matrices[0]
        sounding_matrix = list_of_predictor_matrices[1]

        # Update housekeeping variables.
        num_file_times_in_memory = 0
        full_radar_image_matrix = None
        full_sounding_matrix = None
        all_target_values = None

        for this_key in num_examples_in_memory_by_class_dict.keys():
            num_examples_in_memory_by_class_dict[this_key] = 0

        if sounding_file_names is None:
            yield (radar_image_matrix, target_matrix)
        else:
            yield ([radar_image_matrix, sounding_matrix], target_matrix)


def storm_image_generator_2d3d_myrorss(
        radar_file_name_matrix, top_target_directory_name,
        num_examples_per_batch, num_examples_per_file_time, target_name,
        radar_file_name_matrix_pos_targets_only=None, binarize_target=False,
        radar_normalization_dict=dl_utils.DEFAULT_RADAR_NORMALIZATION_DICT,
        sampling_fraction_by_class_dict=None, sounding_field_names=None,
        top_sounding_dir_name=None,
        sounding_lag_time_for_convective_contamination_sec=None,
        sounding_normalization_dict=
        dl_utils.DEFAULT_SOUNDING_NORMALIZATION_DICT,
        loop_thru_files_once=False):
    """Generates examples with 2-D and 3-D radar images.

    Each example consists of a storm-centered 3-D reflectivity image, storm-
    centered 2-D azimuthal-shear images, and possibly the storm-centered
    sounding, for one storm object.

    F_a = number of azimuthal-shear fields
    m = number of rows per reflectivity image
    n = number of columns per reflectivity image
    H_r = number of heights per reflectivity image
    M = number of rows per azimuthal-shear image
    N = number of columns per azimuthal-shear image

    :param radar_file_name_matrix: See doc for `storm_image_generator_2d`.
    :param top_target_directory_name: Same.
    :param num_examples_per_batch: Same.
    :param num_examples_per_file_time: Same.
    :param target_name: Same.
    :param radar_file_name_matrix_pos_targets_only: Same.
    :param binarize_target: Same.
    :param radar_normalization_dict: Same.
    :param sampling_fraction_by_class_dict: Same.
    :param sounding_field_names: Same.
    :param top_sounding_dir_name: See doc for `find_sounding_files`.
    :param sounding_lag_time_for_convective_contamination_sec: Same.
    :param sounding_normalization_dict: See doc for `storm_image_generator_2d`.
    :param loop_thru_files_once: Same.

    If `sounding_field_names is None`, this method returns...

    :return: predictor_list: List with the following items.
    predictor_list[0] = reflectivity_image_matrix_dbz: numpy array
        (E x m x n x H_r x 1) of storm-centered reflectivity images.
    predictor_list[1] = azimuthal_shear_image_matrix_s01: numpy array
        (E x M x N x F_a) of storm-centered azimuthal-shear images.
    :return: target_matrix: See output doc for `storm_image_generator_2d`.

    If `sounding_field_names is not None`, this method returns...

    :return: predictor_list: List with the following items.
    predictor_list[0] = reflectivity_image_matrix_dbz: See documentation above.
    predictor_list[1] = azimuthal_shear_image_matrix_s01: See documentation
        above.
    predictor_list[2] = sounding_matrix: See output doc for
        `storm_image_generator_2d`.
    :return: target_matrix: See output doc for `storm_image_generator_2d`.
    """

    error_checking.assert_is_boolean(loop_thru_files_once)
    check_input_args(
        num_examples_per_batch=num_examples_per_batch,
        num_examples_per_file_time=num_examples_per_file_time,
        normalize_by_batch=False, radar_file_name_matrix=radar_file_name_matrix,
        num_radar_dimensions=2, binarize_target=binarize_target,
        radar_file_name_matrix_pos_targets_only=
        radar_file_name_matrix_pos_targets_only,
        sounding_field_names=sounding_field_names)

    if sounding_field_names is None:
        sounding_file_names = None
    else:
        sounding_file_names = find_sounding_files(
            top_sounding_dir_name=top_sounding_dir_name,
            radar_file_name_matrix=radar_file_name_matrix,
            target_name=target_name,
            lag_time_for_convective_contamination_sec=
            sounding_lag_time_for_convective_contamination_sec)

    (reflectivity_file_name_matrix, az_shear_file_name_matrix
    ) = separate_radar_files_2d3d(radar_file_name_matrix)

    if radar_file_name_matrix_pos_targets_only is not None:
        (refl_file_name_matrix_pos_targets_only,
         az_shear_file_name_matrix_pos_targets_only
        ) = separate_radar_files_2d3d(radar_file_name_matrix_pos_targets_only)

    num_azimuthal_shear_fields = az_shear_file_name_matrix.shape[1]
    azimuthal_shear_field_names = [''] * num_azimuthal_shear_fields
    for j in range(num_azimuthal_shear_fields):
        this_radar_image_dict = storm_images.read_storm_images(
            netcdf_file_name=az_shear_file_name_matrix[0, j])
        azimuthal_shear_field_names[j] = this_radar_image_dict[
            storm_images.RADAR_FIELD_NAME_KEY]

    num_examples_per_batch_by_class_dict = _get_num_examples_per_batch_by_class(
        num_examples_per_batch=num_examples_per_batch, target_name=target_name,
        sampling_fraction_by_class_dict=sampling_fraction_by_class_dict)

    num_classes = labels.column_name_to_num_classes(
        column_name=target_name, include_dead_storms=False)
    num_file_times_per_batch = int(numpy.ceil(
        float(num_examples_per_batch) / num_examples_per_file_time))

    num_examples_in_memory_by_class_dict, _ = _determine_stopping_criterion(
        num_examples_per_batch=num_examples_per_batch,
        num_file_times_per_batch=num_file_times_per_batch,
        num_examples_per_batch_by_class_dict=
        num_examples_per_batch_by_class_dict,
        num_file_times_in_memory=0,
        sampling_fraction_by_class_dict=sampling_fraction_by_class_dict,
        target_values_in_memory=numpy.array([], dtype=int))

    num_file_times = reflectivity_file_name_matrix.shape[0]

    # Initialize housekeeping variables.
    file_time_index = 0
    num_file_times_in_memory = 0
    full_reflectivity_matrix_dbz = None
    full_azimuthal_shear_matrix_s01 = None
    full_sounding_matrix = None
    all_target_values = None

    while True:
        stopping_criterion = False
        if loop_thru_files_once and file_time_index >= num_file_times:
            raise StopIteration

        while not stopping_criterion:
            if file_time_index == num_file_times:
                if loop_thru_files_once:
                    if all_target_values is None:
                        raise StopIteration
                    break

                file_time_index = 0

            if all_target_values is None:
                num_examples_in_memory = 0
            else:
                num_examples_in_memory = len(all_target_values)

            num_examples_left_by_class_dict = _get_num_examples_left_by_class(
                num_examples_per_batch=num_examples_per_batch,
                num_file_times_per_batch=num_file_times_per_batch,
                num_examples_per_batch_by_class_dict=
                num_examples_per_batch_by_class_dict,
                num_examples_in_memory=num_examples_in_memory,
                num_file_times_in_memory=num_file_times_in_memory,
                num_examples_in_memory_by_class_dict=
                num_examples_in_memory_by_class_dict)

            if sounding_file_names is None:
                this_sounding_file_name = None
            else:
                this_sounding_file_name = sounding_file_names[file_time_index]

            if radar_file_name_matrix_pos_targets_only is None:
                these_refl_file_names_pos_targets_only = None
                these_az_shear_file_names_pos_targets_only = None
            else:
                these_refl_file_names_pos_targets_only = (
                    refl_file_name_matrix_pos_targets_only[
                        file_time_index, ...].tolist())
                these_az_shear_file_names_pos_targets_only = (
                    az_shear_file_name_matrix_pos_targets_only[
                        file_time_index, ...].tolist())

            this_example_dict = _read_input_files_2d3d(
                reflectivity_file_names=reflectivity_file_name_matrix[
                    file_time_index, ...].tolist(),
                azimuthal_shear_file_names=az_shear_file_name_matrix[
                    file_time_index, ...].tolist(),
                top_target_directory_name=top_target_directory_name,
                target_name=target_name,
                num_examples_left_by_class_dict=num_examples_left_by_class_dict,
                refl_file_names_pos_targets_only=
                these_refl_file_names_pos_targets_only,
                az_shear_file_names_pos_targets_only=
                these_az_shear_file_names_pos_targets_only,
                sounding_file_name=this_sounding_file_name,
                sounding_field_names=sounding_field_names)
            print MINOR_SEPARATOR_STRING

            file_time_index += 1
            if this_example_dict is None:
                continue

            num_file_times_in_memory += 1
            sounding_field_names = this_example_dict[SOUNDING_FIELD_NAMES_KEY]

            if all_target_values is None:
                all_target_values = this_example_dict[TARGET_VALUES_KEY] + 0
                full_reflectivity_matrix_dbz = this_example_dict[
                    REFLECTIVITY_IMAGE_MATRIX_KEY] + 0.
                full_azimuthal_shear_matrix_s01 = this_example_dict[
                    AZ_SHEAR_IMAGE_MATRIX_KEY] + 0.

                if sounding_file_names is not None:
                    full_sounding_matrix = this_example_dict[
                        SOUNDING_MATRIX_KEY] + 0.

            else:
                all_target_values = numpy.concatenate((
                    all_target_values, this_example_dict[TARGET_VALUES_KEY]))
                full_reflectivity_matrix_dbz = numpy.concatenate(
                    (full_reflectivity_matrix_dbz,
                     this_example_dict[REFLECTIVITY_IMAGE_MATRIX_KEY]), axis=0)
                full_azimuthal_shear_matrix_s01 = numpy.concatenate(
                    (full_azimuthal_shear_matrix_s01,
                     this_example_dict[AZ_SHEAR_IMAGE_MATRIX_KEY]), axis=0)

                if sounding_file_names is not None:
                    full_sounding_matrix = numpy.concatenate(
                        (full_sounding_matrix,
                         this_example_dict[SOUNDING_MATRIX_KEY]), axis=0)

            (num_examples_in_memory_by_class_dict, stopping_criterion
            ) = _determine_stopping_criterion(
                num_examples_per_batch=num_examples_per_batch,
                num_file_times_per_batch=num_file_times_per_batch,
                num_examples_per_batch_by_class_dict=
                num_examples_per_batch_by_class_dict,
                num_file_times_in_memory=num_file_times_in_memory,
                sampling_fraction_by_class_dict=sampling_fraction_by_class_dict,
                target_values_in_memory=all_target_values)

        if sampling_fraction_by_class_dict is not None:
            batch_indices = dl_utils.sample_by_class(
                sampling_fraction_by_class_dict=sampling_fraction_by_class_dict,
                target_name=target_name, target_values=all_target_values,
                num_examples_total=num_examples_per_batch)

            full_reflectivity_matrix_dbz = full_reflectivity_matrix_dbz[
                batch_indices, ...]
            full_azimuthal_shear_matrix_s01 = full_azimuthal_shear_matrix_s01[
                batch_indices, ...]
            all_target_values = all_target_values[batch_indices]
            if sounding_file_names is not None:
                full_sounding_matrix = full_sounding_matrix[batch_indices, ...]

        full_reflectivity_matrix_dbz = dl_utils.normalize_radar_images(
            radar_image_matrix=full_reflectivity_matrix_dbz,
            field_names=[radar_utils.REFL_NAME],
            normalization_dict=radar_normalization_dict)

        full_azimuthal_shear_matrix_s01 = dl_utils.normalize_radar_images(
            radar_image_matrix=full_azimuthal_shear_matrix_s01,
            field_names=azimuthal_shear_field_names,
            normalization_dict=radar_normalization_dict)

        if sounding_file_names is not None:
            full_sounding_matrix = dl_utils.normalize_soundings(
                sounding_matrix=full_sounding_matrix,
                pressureless_field_names=sounding_field_names,
                normalization_dict=sounding_normalization_dict)

        list_of_predictor_matrices, target_matrix = _select_batch(
            list_of_predictor_matrices=[
                full_reflectivity_matrix_dbz, full_azimuthal_shear_matrix_s01,
                full_sounding_matrix],
            target_values=all_target_values,
            num_examples_per_batch=num_examples_per_batch,
            binarize_target=binarize_target, num_classes=num_classes)

        reflectivity_matrix_dbz = list_of_predictor_matrices[0]
        azimuthal_shear_matrix_s01 = list_of_predictor_matrices[1]
        sounding_matrix = list_of_predictor_matrices[2]

        # Update housekeeping variables.
        num_file_times_in_memory = 0
        full_reflectivity_matrix_dbz = None
        full_azimuthal_shear_matrix_s01 = None
        full_sounding_matrix = None
        all_target_values = None

        for this_key in num_examples_in_memory_by_class_dict.keys():
            num_examples_in_memory_by_class_dict[this_key] = 0

        if sounding_file_names is None:
            yield ([reflectivity_matrix_dbz, azimuthal_shear_matrix_s01],
                   target_matrix)
        else:
            yield ([reflectivity_matrix_dbz, azimuthal_shear_matrix_s01,
                    sounding_matrix],
                   target_matrix)
