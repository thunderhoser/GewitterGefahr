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
H_s = number of height levels per sounding
F_s = number of sounding fields (not including different heights)
C = number of field/height pairs per radar image
K = number of classes for target variable
T = number of file times (time steps or SPC dates)
"""

import copy
import numpy
import keras
from gewittergefahr.deep_learning import storm_images
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.deep_learning import data_augmentation
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import labels
from gewittergefahr.gg_utils import error_checking

MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

RADAR_IMAGE_MATRIX_KEY = 'radar_image_matrix'
REFLECTIVITY_IMAGE_MATRIX_KEY = 'reflectivity_image_matrix_dbz'
AZ_SHEAR_IMAGE_MATRIX_KEY = 'azimuthal_shear_image_matrix_s01'
SOUNDING_MATRIX_KEY = 'sounding_matrix'
SOUNDING_FIELDS_KEY = 'sounding_field_names'
TARGET_VALUES_KEY = 'target_values'

RADAR_FILE_NAMES_KEY = 'radar_file_name_matrix'
TARGET_DIRECTORY_KEY = 'top_target_dir_name'
NUM_EXAMPLES_PER_BATCH_KEY = 'num_examples_per_batch'
NUM_EXAMPLES_PER_FILE_KEY = 'num_examples_per_file'
TARGET_NAME_KEY = 'target_name'
NUM_ROWS_TO_KEEP_KEY = 'num_rows_to_keep'
NUM_COLUMNS_TO_KEEP_KEY = 'num_columns_to_keep'
NORMALIZATION_TYPE_KEY = 'normalization_type_string'
MIN_NORMALIZED_VALUE_KEY = 'min_normalized_value'
MAX_NORMALIZED_VALUE_KEY = 'max_normalized_value'
NORMALIZATION_FILE_KEY = 'normalization_param_file_name'
POSITIVE_RADAR_FILE_NAMES_KEY = 'radar_file_name_matrix_pos_targets_only'
BINARIZE_TARGET_KEY = 'binarize_target'
SAMPLING_FRACTIONS_KEY = 'sampling_fraction_by_class_dict'
SOUNDING_DIRECTORY_KEY = 'top_sounding_dir_name'
SOUNDING_LAG_TIME_KEY = 'sounding_lag_time_sec'
LOOP_ONCE_KEY = 'loop_thru_files_once'
REFLECTIVITY_MASK_KEY = 'refl_masking_threshold_dbz'

STORM_IDS_KEY = 'storm_ids'
STORM_TIMES_KEY = 'storm_times_unix_sec'

NUM_TRANSLATIONS_KEY = 'num_translations'
MAX_TRANSLATION_KEY = 'max_translation_pixels'
NUM_ROTATIONS_KEY = 'num_rotations'
MAX_ROTATION_KEY = 'max_absolute_rotation_angle_deg'
NUM_NOISINGS_KEY = 'num_noisings'
MAX_NOISE_KEY = 'max_noise_standard_deviation'

DEFAULT_GENERATOR_OPTION_DICT = {
    RADAR_FILE_NAMES_KEY: '',
    TARGET_DIRECTORY_KEY: '',
    NUM_EXAMPLES_PER_BATCH_KEY: '',
    NUM_EXAMPLES_PER_FILE_KEY: '',
    TARGET_NAME_KEY: '',
    NUM_ROWS_TO_KEEP_KEY: None,
    NUM_COLUMNS_TO_KEEP_KEY: None,
    NORMALIZATION_TYPE_KEY: None,
    MIN_NORMALIZED_VALUE_KEY: dl_utils.DEFAULT_MIN_NORMALIZED_VALUE,
    MAX_NORMALIZED_VALUE_KEY: dl_utils.DEFAULT_MAX_NORMALIZED_VALUE,
    NORMALIZATION_FILE_KEY: None,
    POSITIVE_RADAR_FILE_NAMES_KEY: None,
    BINARIZE_TARGET_KEY: False,
    SAMPLING_FRACTIONS_KEY: None,
    SOUNDING_FIELDS_KEY: None,
    SOUNDING_DIRECTORY_KEY: None,
    SOUNDING_LAG_TIME_KEY: None,
    LOOP_ONCE_KEY: False,
    REFLECTIVITY_MASK_KEY: dl_utils.DEFAULT_REFL_MASK_THRESHOLD_DBZ
}

DEFAULT_AUGMENTATION_OPTION_DICT = {
    NUM_TRANSLATIONS_KEY: 0,
    MAX_TRANSLATION_KEY: 3,
    NUM_ROTATIONS_KEY: 0,
    MAX_ROTATION_KEY: 15.,
    NUM_NOISINGS_KEY: 0,
    MAX_NOISE_KEY: 0.02
}

DEFAULT_GENERATOR_OPTION_DICT.update(DEFAULT_AUGMENTATION_OPTION_DICT)


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
        array of predictors (radar images or soundings).  The first axis of each
        array must have length E_m.
    :param target_values: numpy array of target values (length E_m).
    :param num_examples_per_batch: Number of examples per batch.
    :param binarize_target: Boolean flag.  If True, will binarize target
        variable, so that the highest class becomes 1 and all other classes
        become 0.
    :param num_classes: Number of classes for target value.
    :return: list_of_predictor_matrices: Same as input, except that the first
        axis of each array has length E_b.
    :return: target_array: See output doc for `storm_image_generator_2d`.
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
        num_classes_to_predict = 2
    else:
        num_classes_to_predict = num_classes + 0

    if num_classes_to_predict == 2:
        print 'Fraction of target values in positive class: {0:.3f}'.format(
            numpy.mean(target_values))
        return list_of_predictor_matrices, target_values

    target_matrix = keras.utils.to_categorical(
        target_values[batch_indices], num_classes_to_predict)

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
        radar_file_names, top_target_dir_name, target_name,
        num_examples_left_by_class_dict, num_rows_to_keep, num_columns_to_keep,
        radar_file_names_pos_targets_only=None, sounding_file_name=None,
        sounding_field_names=None):
    """Reads data for `storm_image_generator_2d`.

    t_0 = file time = time step or SPC date

    :param radar_file_names: length-C list of paths to radar files.  Each file
        should be readable by `storm_images.read_storm_images` and contain
        storm-centered radar images for a different field/height pair at t_0.
    :param top_target_dir_name: Name of top-level directory with target values
        (storm-hazard labels).  Files within this directory should be findable
        by `labels.find_label_file`.
    :param target_name: Name of target variable.
    :param num_examples_left_by_class_dict: Dictionary created by
        `_get_num_examples_left_by_class`.
    :param num_rows_to_keep: Number of rows to keep from each storm-centered
        radar image.  If less than total number of rows, images will be cropped
        around the center.
    :param num_columns_to_keep: Number of columns to keep from each storm-
        centered radar image.
    :param radar_file_names_pos_targets_only: Same as `radar_file_names`, but
        only for storm objects with positive target values.
    :param sounding_file_name: [optional] Path to sounding file.  Should be
        readable by `soundings.read_soundings` and contain storm-centered
        soundings for t_0.
    :param sounding_field_names:
        [used only if `sounding_file_names is not None`]
        1-D list with names of sounding fields to use.  If
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
        top_label_dir_name=top_target_dir_name, label_name=target_name,
        raise_error_if_missing=True)

    if radar_file_names_pos_targets_only is not None:
        if not _need_negative_target_values(num_examples_left_by_class_dict):
            radar_file_names = radar_file_names_pos_targets_only

    print 'Reading data from: "{0:s}" and "{1:s}"...'.format(
        radar_file_names[0], label_file_name)
    this_radar_image_dict = storm_images.read_storm_images_and_labels(
        image_file_name=radar_file_names[0],
        label_file_name=label_file_name, label_name=target_name,
        num_storm_objects_class_dict=num_examples_left_by_class_dict,
        num_rows_to_keep=num_rows_to_keep,
        num_columns_to_keep=num_columns_to_keep)

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

        sounding_matrix = sounding_dict[soundings.SOUNDING_MATRIX_KEY]
        sounding_field_names = sounding_dict[soundings.FIELD_NAMES_KEY]

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
                valid_times_to_keep_unix_sec=storm_times_to_keep_unix_sec,
                num_rows_to_keep=num_rows_to_keep,
                num_columns_to_keep=num_columns_to_keep)

        tuple_of_image_matrices += (
            this_radar_image_dict[storm_images.STORM_IMAGE_MATRIX_KEY],)

    return {
        RADAR_IMAGE_MATRIX_KEY: dl_utils.stack_radar_fields(
            tuple_of_image_matrices),
        TARGET_VALUES_KEY: target_values,
        SOUNDING_MATRIX_KEY: sounding_matrix,
        SOUNDING_FIELDS_KEY: sounding_field_names
    }


def _read_input_files_3d(
        radar_file_name_matrix, top_target_dir_name, target_name,
        num_examples_left_by_class_dict, num_rows_to_keep, num_columns_to_keep,
        radar_file_name_matrix_pos_targets_only=None, sounding_file_name=None,
        sounding_field_names=None):
    """Reads data for `storm_image_generator_3d`.

    t_0 = file time = time step or SPC date

    :param radar_file_name_matrix: numpy array (F_r x H_r) of paths to radar
        files.  Each file should be readable by `storm_images.read_storm_images`
        and contain storm-centered radar images for a different field/height
        pair at t_0.
    :param top_target_dir_name: See doc for `_read_input_files_2d`.
    :param target_name: Same.
    :param num_examples_left_by_class_dict: Same.
    :param num_rows_to_keep: Same.
    :param num_columns_to_keep: Same.
    :param radar_file_name_matrix_pos_targets_only: See doc for
        `storm_image_generator_3d`.
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
        top_label_dir_name=top_target_dir_name, label_name=target_name,
        raise_error_if_missing=True)

    if radar_file_name_matrix_pos_targets_only is not None:
        if not _need_negative_target_values(num_examples_left_by_class_dict):
            radar_file_name_matrix = radar_file_name_matrix_pos_targets_only

    print 'Reading data from: "{0:s}" and "{1:s}"...'.format(
        radar_file_name_matrix[0, 0], label_file_name)
    this_radar_image_dict = storm_images.read_storm_images_and_labels(
        image_file_name=radar_file_name_matrix[0, 0],
        label_file_name=label_file_name, label_name=target_name,
        num_storm_objects_class_dict=num_examples_left_by_class_dict,
        num_rows_to_keep=num_rows_to_keep,
        num_columns_to_keep=num_columns_to_keep)

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

        sounding_matrix = sounding_dict[soundings.SOUNDING_MATRIX_KEY]
        sounding_field_names = sounding_dict[soundings.FIELD_NAMES_KEY]

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
                    valid_times_to_keep_unix_sec=storm_times_to_keep_unix_sec,
                    num_rows_to_keep=num_rows_to_keep,
                    num_columns_to_keep=num_columns_to_keep)

            tuple_of_3d_image_matrices += (
                this_radar_image_dict[storm_images.STORM_IMAGE_MATRIX_KEY],)

        tuple_of_4d_image_matrices += (
            dl_utils.stack_radar_fields(tuple_of_3d_image_matrices),)

    return {
        RADAR_IMAGE_MATRIX_KEY: dl_utils.stack_radar_heights(
            tuple_of_4d_image_matrices),
        TARGET_VALUES_KEY: target_values,
        SOUNDING_MATRIX_KEY: sounding_matrix,
        SOUNDING_FIELDS_KEY: sounding_field_names
    }


def _read_input_files_2d3d(
        reflectivity_file_names, azimuthal_shear_file_names,
        top_target_dir_name, target_name, num_examples_left_by_class_dict,
        num_rows_to_keep, num_columns_to_keep,
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
    :param top_target_dir_name: See doc for `_read_input_files_2d`.
    :param target_name: Same.
    :param num_examples_left_by_class_dict: Same.
    :param num_rows_to_keep: Same.
    :param num_columns_to_keep: Same.
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
        top_label_dir_name=top_target_dir_name, label_name=target_name,
        raise_error_if_missing=True)

    if refl_file_names_pos_targets_only is not None:
        if not _need_negative_target_values(num_examples_left_by_class_dict):
            reflectivity_file_names = refl_file_names_pos_targets_only
            azimuthal_shear_file_names = az_shear_file_names_pos_targets_only

    print 'Reading data from: "{0:s}" and "{1:s}"...'.format(
        reflectivity_file_names[0], label_file_name)
    this_radar_image_dict = storm_images.read_storm_images_and_labels(
        image_file_name=reflectivity_file_names[0],
        label_file_name=label_file_name, label_name=target_name,
        num_storm_objects_class_dict=num_examples_left_by_class_dict,
        num_rows_to_keep=num_rows_to_keep,
        num_columns_to_keep=num_columns_to_keep)

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

        sounding_matrix = sounding_dict[soundings.SOUNDING_MATRIX_KEY]
        sounding_field_names = sounding_dict[soundings.FIELD_NAMES_KEY]

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
                valid_times_to_keep_unix_sec=storm_times_to_keep_unix_sec,
                num_rows_to_keep=num_rows_to_keep,
                num_columns_to_keep=num_columns_to_keep)

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
            valid_times_to_keep_unix_sec=storm_times_to_keep_unix_sec,
            num_rows_to_keep=num_rows_to_keep * 2,
            num_columns_to_keep=num_columns_to_keep * 2)

        tuple_of_3d_az_shear_matrices += (
            this_radar_image_dict[storm_images.STORM_IMAGE_MATRIX_KEY],)

    return {
        REFLECTIVITY_IMAGE_MATRIX_KEY: dl_utils.stack_radar_heights(
            tuple_of_4d_refl_matrices),
        AZ_SHEAR_IMAGE_MATRIX_KEY: dl_utils.stack_radar_fields(
            tuple_of_3d_az_shear_matrices),
        TARGET_VALUES_KEY: target_values,
        SOUNDING_MATRIX_KEY: sounding_matrix,
        SOUNDING_FIELDS_KEY: sounding_field_names
    }


def _augment_radar_images(
        list_of_predictor_matrices, target_array, option_dict=None):
    """Applies one or more data augmentations to each radar image.

    Q = total number of translations applied to each image.
    e = original number of examples
    E = number of examples after augmentation = e * (1 + Q)

    P = number of predictor matrices

    Currently this method applies each translation separately.  At some point I
    may apply multiple translations to the same image in series.

    :param list_of_predictor_matrices: length-P list, where each item is a numpy
        array of predictors (either radar images or soundings).  The first axis
        of each array must have length e.
    :param target_array: See doc for `storm_image_generator_2d`.  If the array
        is 1-D, it has length e.  If 2-D, it has dimensions e x K.
    :param option_dict: Dictionary with the following keys.
    option_dict['num_translations']: See doc for
        `data_augmentation.get_translations`.
    option_dict['max_translation_pixels']: Same.
    option_dict['num_rotations']: See doc for `data_augmentation.get_rotations`.
    option_dict['max_absolute_rotation_angle_deg']: Same.
    option_dict['num_noisings']: See doc for `data_augmentation.get_noisings`.
    option_dict['max_noise_standard_deviation']: Same.

    :return: list_of_predictor_matrices: Same as input, except that the first
        axis of each array now has length E.
    :return: target_array: See doc for `storm_image_generator_2d`.  If the array
        is 1-D, it has length E.  If 2-D, it has dimensions E x K.
    """

    if option_dict is None:
        orig_option_dict = {}
    else:
        orig_option_dict = option_dict.copy()

    option_dict = DEFAULT_AUGMENTATION_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    last_num_dimensions = len(list_of_predictor_matrices[-1].shape)
    soundings_included = last_num_dimensions == 3
    num_radar_matrices = (
        len(list_of_predictor_matrices) - int(soundings_included)
    )

    x_offsets_pixels, y_offsets_pixels = data_augmentation.get_translations(
        num_translations=option_dict[NUM_TRANSLATIONS_KEY],
        max_translation_pixels=option_dict[MAX_TRANSLATION_KEY],
        num_grid_rows=list_of_predictor_matrices[0].shape[1],
        num_grid_columns=list_of_predictor_matrices[0].shape[2])

    ccw_rotation_angles_deg = data_augmentation.get_rotations(
        num_rotations=option_dict[NUM_ROTATIONS_KEY],
        max_absolute_rotation_angle_deg=option_dict[MAX_ROTATION_KEY])

    noise_standard_deviations = data_augmentation.get_noisings(
        num_noisings=option_dict[NUM_NOISINGS_KEY],
        max_standard_deviation=option_dict[MAX_NOISE_KEY])

    print (
        'Augmenting radar images with {0:d} translations, {1:d} rotations, and '
        '{2:d} noisings each...'
    ).format(option_dict[NUM_TRANSLATIONS_KEY], option_dict[NUM_ROTATIONS_KEY],
             option_dict[NUM_NOISINGS_KEY])

    orig_num_examples = list_of_predictor_matrices[0].shape[0]

    for i in range(option_dict[NUM_TRANSLATIONS_KEY]):
        for j in range(num_radar_matrices):
            this_multiplier = j + 1  # Handles azimuthal shear.

            this_image_matrix = data_augmentation.shift_radar_images(
                radar_image_matrix=
                list_of_predictor_matrices[j][:orig_num_examples, ...],
                x_offset_pixels=this_multiplier * x_offsets_pixels[i],
                y_offset_pixels=this_multiplier * y_offsets_pixels[i])

            list_of_predictor_matrices[j] = numpy.concatenate(
                (list_of_predictor_matrices[j], this_image_matrix), axis=0)

        target_array = numpy.concatenate(
            (target_array, target_array[:orig_num_examples, ...]), axis=0)
        if soundings_included:
            list_of_predictor_matrices[-1] = numpy.concatenate(
                (list_of_predictor_matrices[-1],
                 list_of_predictor_matrices[-1][:orig_num_examples, ...]),
                axis=0)

    for i in range(option_dict[NUM_ROTATIONS_KEY]):
        for j in range(num_radar_matrices):
            this_image_matrix = data_augmentation.rotate_radar_images(
                radar_image_matrix=
                list_of_predictor_matrices[j][:orig_num_examples, ...],
                ccw_rotation_angle_deg=ccw_rotation_angles_deg[i])

            list_of_predictor_matrices[j] = numpy.concatenate(
                (list_of_predictor_matrices[j], this_image_matrix), axis=0)

        target_array = numpy.concatenate(
            (target_array, target_array[:orig_num_examples, ...]), axis=0)
        if soundings_included:
            list_of_predictor_matrices[-1] = numpy.concatenate(
                (list_of_predictor_matrices[-1],
                 list_of_predictor_matrices[-1][:orig_num_examples, ...]),
                axis=0)

    for i in range(option_dict[NUM_NOISINGS_KEY]):
        for j in range(num_radar_matrices):
            this_image_matrix = data_augmentation.noise_radar_images(
                radar_image_matrix=
                list_of_predictor_matrices[j][:orig_num_examples, ...],
                standard_deviation=noise_standard_deviations[i])

            list_of_predictor_matrices[j] = numpy.concatenate(
                (list_of_predictor_matrices[j], this_image_matrix), axis=0)

        target_array = numpy.concatenate(
            (target_array, target_array[:orig_num_examples, ...]), axis=0)
        if soundings_included:
            list_of_predictor_matrices[-1] = numpy.concatenate(
                (list_of_predictor_matrices[-1],
                 list_of_predictor_matrices[-1][:orig_num_examples, ...]),
                axis=0)

    return list_of_predictor_matrices, target_array


def check_input_args(
        num_examples_per_batch, num_examples_per_file, radar_file_name_matrix,
        num_radar_dimensions, binarize_target,
        radar_file_name_matrix_pos_targets_only=None,
        sounding_field_names=None):
    """Error-checking of input arguments to generator.

    T = number of storm times (initial times, not valid times)

    :param num_examples_per_batch: Number of examples (storm objects) per batch.
    :param num_examples_per_file: Number of examples (storm objects) per file.
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
        1-D list with names of sounding fields to use.  If
        `sounding_field_names is None`, will use all sounding fields.
    """

    error_checking.assert_is_integer(num_examples_per_batch)
    error_checking.assert_is_geq(num_examples_per_batch, 10)
    error_checking.assert_is_integer(num_examples_per_file)
    error_checking.assert_is_geq(num_examples_per_file, 2)

    error_checking.assert_is_integer(num_radar_dimensions)
    error_checking.assert_is_geq(num_radar_dimensions, 2)
    error_checking.assert_is_leq(num_radar_dimensions, 3)
    error_checking.assert_is_numpy_array(
        radar_file_name_matrix, num_dimensions=num_radar_dimensions)

    error_checking.assert_is_boolean(binarize_target)

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

    for this_key in keys_to_change:
        if this_key == storm_images.STORM_IDS_KEY:
            storm_image_dict[this_key] = [
                storm_image_dict[this_key][i] for i in valid_indices]
        else:
            storm_image_dict[this_key] = storm_image_dict[this_key][
                valid_indices, ...]

    return storm_image_dict


def separate_radar_files_2d3d(radar_file_name_matrix):
    """Separates input files for `storm_image_generator_2d3d_myrorss`.

    Specifically, separates these files into two arrays: one for reflectivity
    and one for azimuthal shear.

    F_a = number of azimuthal-shear fields
    H_r = number of heights per reflectivity image

    :param radar_file_name_matrix: numpy array (T x [H_r + F_a]) of paths to
        image files.  This should be created by `find_radar_files_2d`.
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

    num_field_height_pairs = radar_file_name_matrix.shape[1]
    field_name_by_pair = [''] * num_field_height_pairs
    height_by_pair_m_agl = numpy.full(num_field_height_pairs, -1, dtype=int)

    for j in range(num_field_height_pairs):
        field_name_by_pair[j] = storm_images.image_file_name_to_field(
            radar_file_name_matrix[0, j])
        height_by_pair_m_agl[j] = storm_images.image_file_name_to_height(
            radar_file_name_matrix[0, j])

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

    sort_indices = numpy.argsort(height_by_pair_m_agl[reflectivity_indices])
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
        top_directory_name_pos_targets_only=None, radar_heights_m_agl=None,
        reflectivity_heights_m_agl=None):
    """Finds input files for either of the following generators.

    - storm_image_generator_2d
    - storm_image_generator_2d3d_myrorss

    If `one_file_per_time_step = True`, this method will return files containing
    one time step each.  If `one_file_per_time_step = False`, will return files
    containing one SPC date each.

    :param top_directory_name: Name of top-level directory with storm-centered
        radar images.
    :param radar_source: Data source (must be accepted by
        `radar_utils.check_data_source`).
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
    :param radar_heights_m_agl: [used iff radar_source = "gridrad"]
        1-D numpy array of radar heights (metres above ground level).
    :param reflectivity_heights_m_agl: [used iff radar_source != "gridrad"]
        1-D numpy array of reflectivity heights (metres above ground level).
    :return: radar_file_name_matrix: T-by-C numpy array of paths to files with
        storm-centered radar images.
    :return: radar_file_name_matrix_pos_targets_only:
        [None if `top_directory_name_pos_targets_only is None`]
        Same as `radar_file_name_matrix`, but only for storm objects with
        positive target values.
    """

    radar_utils.check_data_source(radar_source)

    if radar_source == radar_utils.GRIDRAD_SOURCE_ID:
        storm_image_file_dict = storm_images.find_many_files_gridrad(
            top_directory_name=top_directory_name,
            radar_field_names=radar_field_names,
            radar_heights_m_agl=radar_heights_m_agl,
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
            reflectivity_heights_m_agl=reflectivity_heights_m_agl,
            raise_error_if_all_missing=True, raise_error_if_any_missing=False)

    radar_file_name_matrix = storm_image_file_dict[
        storm_images.IMAGE_FILE_NAMES_KEY]
    num_image_times = radar_file_name_matrix.shape[0]

    if radar_source == radar_utils.GRIDRAD_SOURCE_ID:
        num_field_height_pairs = (
            radar_file_name_matrix.shape[1] * radar_file_name_matrix.shape[2])
        radar_file_name_matrix = numpy.reshape(
            radar_file_name_matrix, (num_image_times, num_field_height_pairs))

    time_missing_indices = numpy.unique(
        numpy.where(radar_file_name_matrix == '')[0])
    radar_file_name_matrix = numpy.delete(
        radar_file_name_matrix, time_missing_indices, axis=0)
    num_image_times = radar_file_name_matrix.shape[0]

    if shuffle_times:
        image_time_indices = numpy.linspace(
            0, num_image_times - 1, num=num_image_times, dtype=int)
        numpy.random.shuffle(image_time_indices)
        radar_file_name_matrix = radar_file_name_matrix[image_time_indices, ...]

    if top_directory_name_pos_targets_only is None:
        return radar_file_name_matrix, None

    num_field_height_pairs = radar_file_name_matrix.shape[1]
    radar_file_name_matrix_pos_targets_only = numpy.full(
        (num_image_times, num_field_height_pairs), '', dtype=object)

    for j in range(num_field_height_pairs):
        this_field_name = storm_images.image_file_name_to_field(
            radar_file_name_matrix[0, j])
        this_height_m_agl = storm_images.image_file_name_to_height(
            radar_file_name_matrix[0, j])

        for i in range(num_image_times):
            (this_time_unix_sec, this_spc_date_string
            ) = storm_images.image_file_name_to_time(
                radar_file_name_matrix[i, j])

            radar_file_name_matrix_pos_targets_only[i, j] = (
                storm_images.find_storm_image_file(
                    top_directory_name=top_directory_name_pos_targets_only,
                    spc_date_string=this_spc_date_string,
                    radar_source=radar_source, radar_field_name=this_field_name,
                    radar_height_m_agl=this_height_m_agl,
                    unix_time_sec=this_time_unix_sec,
                    raise_error_if_missing=True))

    return radar_file_name_matrix, radar_file_name_matrix_pos_targets_only


def find_radar_files_3d(
        top_directory_name, radar_source, radar_field_names,
        radar_heights_m_agl, first_file_time_unix_sec, last_file_time_unix_sec,
        one_file_per_time_step, shuffle_times=True,
        top_directory_name_pos_targets_only=None):
    """Finds input files for `storm_image_generator_3d`.

    :param top_directory_name: See doc for `find_radar_files_2d`.
    :param radar_source: Same.
    :param radar_field_names: Same.
    :param radar_heights_m_agl: Same.
    :param first_file_time_unix_sec: Same.
    :param last_file_time_unix_sec: Same.
    :param one_file_per_time_step: Same.
    :param shuffle_times: Same.
    :param top_directory_name_pos_targets_only: Same.
    :return: radar_file_name_matrix: numpy array (T x F_r x H_r) of paths to
        files with storm-centered radar images.
    :return: radar_file_name_matrix_pos_targets_only: See doc for
        `find_radar_files_2d`.
    """

    radar_utils.check_data_source(radar_source)

    if radar_source == radar_utils.GRIDRAD_SOURCE_ID:
        storm_image_file_dict = storm_images.find_many_files_gridrad(
            top_directory_name=top_directory_name,
            radar_field_names=radar_field_names,
            radar_heights_m_agl=radar_heights_m_agl,
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
            reflectivity_heights_m_agl=radar_heights_m_agl,
            raise_error_if_all_missing=True, raise_error_if_any_missing=False)

    radar_file_name_matrix = storm_image_file_dict[
        storm_images.IMAGE_FILE_NAMES_KEY]
    num_image_times = radar_file_name_matrix.shape[0]

    if radar_source != radar_utils.GRIDRAD_SOURCE_ID:
        radar_file_name_matrix = numpy.reshape(
            radar_file_name_matrix,
            (num_image_times, 1, len(radar_heights_m_agl)))

    time_missing_indices = numpy.unique(
        numpy.where(radar_file_name_matrix == '')[0])
    radar_file_name_matrix = numpy.delete(
        radar_file_name_matrix, time_missing_indices, axis=0)
    num_image_times = radar_file_name_matrix.shape[0]

    if shuffle_times:
        image_time_indices = numpy.linspace(
            0, num_image_times - 1, num=num_image_times, dtype=int)
        numpy.random.shuffle(image_time_indices)
        radar_file_name_matrix = radar_file_name_matrix[image_time_indices, ...]

    if top_directory_name_pos_targets_only is None:
        return radar_file_name_matrix, None

    num_fields = radar_file_name_matrix.shape[1]
    num_heights = radar_file_name_matrix.shape[2]
    radar_file_name_matrix_pos_targets_only = numpy.full(
        (num_image_times, num_fields, num_heights), '', dtype=object)

    for j in range(num_fields):
        for k in range(num_heights):
            this_field_name = storm_images.image_file_name_to_field(
                radar_file_name_matrix[0, j, k])
            this_height_m_agl = storm_images.image_file_name_to_height(
                radar_file_name_matrix[0, j, k])

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
                        radar_height_m_agl=this_height_m_agl,
                        unix_time_sec=this_time_unix_sec,
                        raise_error_if_missing=True))

    return radar_file_name_matrix, radar_file_name_matrix_pos_targets_only


def find_sounding_files(
        top_sounding_dir_name, radar_file_name_matrix, target_name,
        lag_time_for_convective_contamination_sec):
    """Finds sounding file for each radar time.

    :param top_sounding_dir_name: Name of top-level directory with sounding
        files, findable by `soundings.find_sounding_file`.
    :param radar_file_name_matrix: numpy array created by either
        `find_radar_files_2d` or `find_radar_files_3d`.  Length of the first
        axis is T.
    :param target_name: Name of target variable.
    :param lag_time_for_convective_contamination_sec: See doc for
        `soundings.interp_soundings_to_storm_objects`.
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

        (this_time_unix_sec, this_spc_date_string
        ) = storm_images.image_file_name_to_time(this_file_name)

        sounding_file_names[i] = soundings.find_sounding_file(
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
    :return: sounding_dict: See output doc for `soundings.read_soundings`.
    :return: radar_image_dict: Same as input, except that some storm objects may
        be removed.
    """

    print 'Reading data from: "{0:s}"...'.format(sounding_file_name)
    sounding_dict, _ = soundings.read_soundings(
        netcdf_file_name=sounding_file_name,
        field_names_to_keep=sounding_field_names,
        storm_ids_to_keep=radar_image_dict[storm_images.STORM_IDS_KEY],
        init_times_to_keep_unix_sec=radar_image_dict[
            storm_images.VALID_TIMES_KEY])

    if not len(sounding_dict[soundings.STORM_IDS_KEY]):
        return None, None

    orig_storm_ids_as_numpy_array = numpy.array(
        radar_image_dict[storm_images.STORM_IDS_KEY])
    orig_storm_times_unix_sec = radar_image_dict[
        storm_images.VALID_TIMES_KEY] + 0

    num_storm_objects_kept = len(sounding_dict[soundings.STORM_IDS_KEY])
    storm_object_kept_indices = []

    for i in range(num_storm_objects_kept):
        this_index = numpy.where(numpy.logical_and(
            orig_storm_ids_as_numpy_array ==
            sounding_dict[soundings.STORM_IDS_KEY][i],
            orig_storm_times_unix_sec ==
            sounding_dict[soundings.INITIAL_TIMES_KEY][i]))[0][0]

        storm_object_kept_indices.append(this_index)

    storm_object_kept_indices = numpy.array(
        storm_object_kept_indices, dtype=int)

    radar_image_dict[storm_images.STORM_IDS_KEY] = sounding_dict[
        soundings.STORM_IDS_KEY]
    radar_image_dict[storm_images.VALID_TIMES_KEY] = sounding_dict[
        soundings.INITIAL_TIMES_KEY]
    radar_image_dict[storm_images.STORM_IMAGE_MATRIX_KEY] = radar_image_dict[
        storm_images.STORM_IMAGE_MATRIX_KEY][storm_object_kept_indices, ...]
    if storm_images.LABEL_VALUES_KEY in radar_image_dict:
        radar_image_dict[storm_images.LABEL_VALUES_KEY] = radar_image_dict[
            storm_images.LABEL_VALUES_KEY][storm_object_kept_indices]

    return sounding_dict, radar_image_dict


def storm_image_generator_2d(option_dict=None):
    """Generates examples with 2-D radar images.

    Each example consists of the following data for one storm object:

    - Storm-centered 2-D radar images (one per field)
    - One storm-centered sounding (maybe)
    - Target class

    :param option_dict: Dictionary with the following keys.
    option_dict['radar_file_name_matrix']: T-by-C numpy array of paths to radar
        files, created by `find_radar_files_2d`.
    option_dict['top_target_dir_name']: Name of top-level directory with target
        values (storm-hazard labels).  Files within this directory will be found
        by `labels.find_label_file`.
    option_dict['num_examples_per_batch']: See doc for `check_input_args`.
    option_dict['num_examples_per_file']: Same.
    option_dict['target_name']: Name of target variable.
    option_dict['num_rows_to_keep']: Number of rows to keep in each storm-
        centered radar image.  If less than total number of rows, images will be
        cropped around the center (so images will always have the same center,
        regardless of `num_rows_to_keep`).
    option_dict['num_columns_to_keep']: Same as above, but for columns.
    option_dict['normalization_type_string']: Normalization type (must be
        accepted by `dl_utils.check_normalization_type`).  If you don't want to
        normalize, leave this alone.
    option_dict['min_normalized_value']: See doc for
        `dl_utils.normalize_radar_images` or `dl_utils.normalize_soundings`.
    option_dict['max_normalized_value']: Same.
    option_dict['normalization_param_file_name']: Same.
    option_dict['radar_file_name_matrix_pos_targets_only']: See doc for
        `find_radar_files_2d`.  If you want to use only one set of radar files
        (most cases), leave this alone.
    option_dict['binarize_target']: See doc for `_select_batch`.
    option_dict['sampling_fraction_by_class_dict']: Dictionary, where each key
        is the integer for a target class (-2 for "dead storm") and each value
        is the corresponding sampling fraction.  Sampling fractions will be
        applied to each batch.
    option_dict['sounding_field_names']: list (length F_s) with names of
        sounding fields.  Each must be accepted by `soundings.check_field_name`.
    option_dict['top_sounding_dir_name']:
        [used iff `sounding_field_names is not None`]
        See doc for `find_sounding_files`.
    option_dict['sounding_lag_time_sec']: Same.
    option_dict['loop_thru_files_once']: Boolean flag.  If True, this generator
        will loop through files only once.  If False, once this generator has
        reached the last file, it will go back to the beginning and keep
        looping.
    option_dict['num_translations']: See doc for `_augment_radar_images`.
    option_dict['max_translation_pixels']: Same.
    option_dict['num_rotations']: Same.
    option_dict['max_absolute_rotation_angle_deg']: Same.
    option_dict['num_noisings']: Same.
    option_dict['max_noise_standard_deviation']: Same.

    If `sounding_field_names is None`, this method returns...

    :return: radar_image_matrix: E-by-M-by-N-by-C numpy array of storm-centered
        radar images.
    :return: target_array: If number of classes to predict = 2, this is a
        length-E numpy array, where each value is 0 or 1, indicating the
        negative or positive class.

        If number of classes to predict > 2, this is an E-by-K numpy array,
        where each value is 0 or 1.  If target_array[i, k] = 1, the [i]th storm
        object belongs to the [k]th class.  Classes are mutually exclusive and
        collectively exhaustive, so the sum across each row of the matrix is 1.

    If `sounding_field_names is not None`, this method returns...

    :return: predictor_list: List with the following items.
    predictor_list[0] = radar_image_matrix: See documentation above.
    predictor_list[1] = sounding_matrix: numpy array (E x H_s x F_s) of storm-
        centered soundings.
    :return: target_array: See documentation above.
    """

    if option_dict is None:
        orig_option_dict = {}
    else:
        orig_option_dict = option_dict.copy()

    option_dict = DEFAULT_GENERATOR_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    loop_thru_files_once = option_dict[LOOP_ONCE_KEY]
    error_checking.assert_is_boolean(loop_thru_files_once)

    num_examples_per_batch = option_dict[NUM_EXAMPLES_PER_BATCH_KEY]
    num_examples_per_file = option_dict[NUM_EXAMPLES_PER_FILE_KEY]
    radar_file_name_matrix = option_dict[RADAR_FILE_NAMES_KEY]
    binarize_target = option_dict[BINARIZE_TARGET_KEY]
    radar_file_name_matrix_pos_targets_only = option_dict[
        POSITIVE_RADAR_FILE_NAMES_KEY]
    sounding_field_names = option_dict[SOUNDING_FIELDS_KEY]

    check_input_args(
        num_examples_per_batch=num_examples_per_batch,
        num_examples_per_file=num_examples_per_file,
        radar_file_name_matrix=radar_file_name_matrix, num_radar_dimensions=2,
        binarize_target=binarize_target,
        radar_file_name_matrix_pos_targets_only=
        radar_file_name_matrix_pos_targets_only,
        sounding_field_names=sounding_field_names)

    target_name = option_dict[TARGET_NAME_KEY]
    top_target_dir_name = option_dict[TARGET_DIRECTORY_KEY]
    num_rows_to_keep = option_dict[NUM_ROWS_TO_KEEP_KEY]
    num_columns_to_keep = option_dict[NUM_COLUMNS_TO_KEEP_KEY]
    sampling_fraction_by_class_dict = option_dict[SAMPLING_FRACTIONS_KEY]

    normalization_type_string = option_dict[NORMALIZATION_TYPE_KEY]
    min_normalized_value = option_dict[MIN_NORMALIZED_VALUE_KEY]
    max_normalized_value = option_dict[MAX_NORMALIZED_VALUE_KEY]
    normalization_param_file_name = option_dict[NORMALIZATION_FILE_KEY]

    if sounding_field_names is None:
        sounding_file_names = None
    else:
        sounding_file_names = find_sounding_files(
            top_sounding_dir_name=option_dict[SOUNDING_DIRECTORY_KEY],
            radar_file_name_matrix=radar_file_name_matrix,
            target_name=target_name,
            lag_time_for_convective_contamination_sec=
            option_dict[SOUNDING_LAG_TIME_KEY])

    num_channels = radar_file_name_matrix.shape[1]
    field_name_by_channel = [''] * num_channels
    for j in range(num_channels):
        field_name_by_channel[j] = storm_images.image_file_name_to_field(
            radar_file_name_matrix[0, j])

    num_examples_per_batch_by_class_dict = _get_num_examples_per_batch_by_class(
        num_examples_per_batch=num_examples_per_batch, target_name=target_name,
        sampling_fraction_by_class_dict=sampling_fraction_by_class_dict)

    num_classes = labels.column_name_to_num_classes(
        column_name=target_name, include_dead_storms=False)
    num_file_times_per_batch = int(numpy.ceil(
        float(num_examples_per_batch) / num_examples_per_file))

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
                top_target_dir_name=top_target_dir_name,
                target_name=target_name,
                num_examples_left_by_class_dict=num_examples_left_by_class_dict,
                num_rows_to_keep=num_rows_to_keep,
                num_columns_to_keep=num_columns_to_keep,
                radar_file_names_pos_targets_only=
                these_radar_file_names_pos_targets_only,
                sounding_file_name=this_sounding_file_name,
                sounding_field_names=sounding_field_names)
            print MINOR_SEPARATOR_STRING

            file_time_index += 1
            if this_example_dict is None:
                continue

            num_file_times_in_memory += 1
            sounding_field_names = this_example_dict[SOUNDING_FIELDS_KEY]

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

        if normalization_type_string is not None:
            full_radar_image_matrix = dl_utils.normalize_radar_images(
                radar_image_matrix=full_radar_image_matrix,
                field_names=field_name_by_channel,
                normalization_type_string=normalization_type_string,
                normalization_param_file_name=normalization_param_file_name,
                min_normalized_value=min_normalized_value,
                max_normalized_value=max_normalized_value).astype('float32')

            if sounding_file_names is not None:
                full_sounding_matrix = dl_utils.normalize_soundings(
                    sounding_matrix=full_sounding_matrix,
                    field_names=sounding_field_names,
                    normalization_type_string=normalization_type_string,
                    normalization_param_file_name=normalization_param_file_name,
                    min_normalized_value=min_normalized_value,
                    max_normalized_value=max_normalized_value).astype('float32')

        list_of_predictor_matrices, target_array = _select_batch(
            list_of_predictor_matrices=[
                full_radar_image_matrix, full_sounding_matrix],
            target_values=all_target_values,
            num_examples_per_batch=num_examples_per_batch,
            binarize_target=binarize_target, num_classes=num_classes)

        list_of_predictor_matrices, target_array = _augment_radar_images(
            list_of_predictor_matrices=list_of_predictor_matrices,
            target_array=target_array, option_dict=option_dict)

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
            yield (radar_image_matrix, target_array)
        else:
            yield ([radar_image_matrix, sounding_matrix], target_array)


def storm_image_generator_3d(option_dict):
    """Generates examples with 3-D radar images.

    Each example consists of the following data for one storm object:

    - Storm-centered 3-D radar images (one per field)
    - One storm-centered sounding (maybe)
    - Target class

    :param option_dict: Dictionary with the following keys.
    option_dict['radar_file_name_matrix']: numpy array (T x F_r x H_r) of paths
        to radar files, created by `find_radar_files_3d`.
    option_dict['top_target_dir_name']: See doc for `storm_image_generator_2d`.
    option_dict['num_examples_per_batch']: Same.
    option_dict['num_examples_per_file']: Same.
    option_dict['target_name']: Same.
    option_dict['num_rows_to_keep']: Same.
    option_dict['num_columns_to_keep']: Same.
    option_dict['normalization_type_string']: Same.
    option_dict['min_normalized_value']: Same.
    option_dict['max_normalized_value']: Same.
    option_dict['normalization_param_file_name']: Same.
    option_dict['radar_file_name_matrix_pos_targets_only']: See doc for
        `find_radar_files_3d`.  If you want to use only one set of radar files
        (most cases), leave this alone.
    option_dict['binarize_target']: See doc for `storm_image_generator_2d`.
    option_dict['sampling_fraction_by_class_dict']: Same.
    option_dict['sounding_field_names']: Same.
    option_dict['top_sounding_dir_name']: Same.
    option_dict['sounding_lag_time_sec']: Same.
    option_dict['loop_thru_files_once']: Same.
    option_dict['num_translations']: Same.
    option_dict['max_translation_pixels']: Same.
    option_dict['num_rotations']: Same.
    option_dict['max_absolute_rotation_angle_deg']: Same.
    option_dict['num_noisings']: Same.
    option_dict['max_noise_standard_deviation']: Same.
    option_dict['refl_masking_threshold_dbz']: All pixels with reflectivity <
        `refl_masking_threshold_dbz` will be masked.  If you don't want a
        reflectivity mask, leave this alone.

    If `sounding_field_names is None`, this method returns...

    :return: radar_image_matrix: numpy array (E x M x N x H_r x F_r) of storm-
        centered radar images.
    :return: target_array: See output doc for `storm_image_generator_2d`.

    If `sounding_field_names is not None`, this method returns...

    :return: predictor_list: List with the following items.
    predictor_list[0] = radar_image_matrix: See documentation above.
    predictor_list[1] = sounding_matrix: numpy array (E x H_s x F_s) of storm-
        centered soundings.
    :return: target_array: See output doc for `storm_image_generator_2d`.
    """

    if option_dict is None:
        orig_option_dict = {}
    else:
        orig_option_dict = option_dict.copy()

    option_dict = DEFAULT_GENERATOR_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    loop_thru_files_once = option_dict[LOOP_ONCE_KEY]
    error_checking.assert_is_boolean(loop_thru_files_once)

    num_examples_per_batch = option_dict[NUM_EXAMPLES_PER_BATCH_KEY]
    num_examples_per_file = option_dict[NUM_EXAMPLES_PER_FILE_KEY]
    radar_file_name_matrix = option_dict[RADAR_FILE_NAMES_KEY]
    binarize_target = option_dict[BINARIZE_TARGET_KEY]
    radar_file_name_matrix_pos_targets_only = option_dict[
        POSITIVE_RADAR_FILE_NAMES_KEY]
    sounding_field_names = option_dict[SOUNDING_FIELDS_KEY]

    check_input_args(
        num_examples_per_batch=num_examples_per_batch,
        num_examples_per_file=num_examples_per_file,
        radar_file_name_matrix=radar_file_name_matrix, num_radar_dimensions=3,
        binarize_target=binarize_target,
        radar_file_name_matrix_pos_targets_only=
        radar_file_name_matrix_pos_targets_only,
        sounding_field_names=sounding_field_names)

    target_name = option_dict[TARGET_NAME_KEY]
    top_target_dir_name = option_dict[TARGET_DIRECTORY_KEY]
    num_rows_to_keep = option_dict[NUM_ROWS_TO_KEEP_KEY]
    num_columns_to_keep = option_dict[NUM_COLUMNS_TO_KEEP_KEY]
    sampling_fraction_by_class_dict = option_dict[SAMPLING_FRACTIONS_KEY]

    normalization_type_string = option_dict[NORMALIZATION_TYPE_KEY]
    min_normalized_value = option_dict[MIN_NORMALIZED_VALUE_KEY]
    max_normalized_value = option_dict[MAX_NORMALIZED_VALUE_KEY]
    normalization_param_file_name = option_dict[NORMALIZATION_FILE_KEY]
    refl_masking_threshold_dbz = option_dict[REFLECTIVITY_MASK_KEY]

    if sounding_field_names is None:
        sounding_file_names = None
    else:
        sounding_file_names = find_sounding_files(
            top_sounding_dir_name=option_dict[SOUNDING_DIRECTORY_KEY],
            radar_file_name_matrix=radar_file_name_matrix,
            target_name=target_name,
            lag_time_for_convective_contamination_sec=
            option_dict[SOUNDING_LAG_TIME_KEY])

    num_fields = radar_file_name_matrix.shape[1]
    radar_field_names = [''] * num_fields
    for j in range(num_fields):
        radar_field_names[j] = storm_images.image_file_name_to_field(
            radar_file_name_matrix[0, j, 0])

    num_examples_per_batch_by_class_dict = _get_num_examples_per_batch_by_class(
        num_examples_per_batch=num_examples_per_batch, target_name=target_name,
        sampling_fraction_by_class_dict=sampling_fraction_by_class_dict)

    num_classes = labels.column_name_to_num_classes(
        column_name=target_name, include_dead_storms=False)
    num_file_times_per_batch = int(numpy.ceil(
        float(num_examples_per_batch) / num_examples_per_file))

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
                top_target_dir_name=top_target_dir_name,
                target_name=target_name,
                num_examples_left_by_class_dict=num_examples_left_by_class_dict,
                num_rows_to_keep=num_rows_to_keep,
                num_columns_to_keep=num_columns_to_keep,
                radar_file_name_matrix_pos_targets_only=
                this_file_name_matrix_pos_targets_only,
                sounding_file_name=this_sounding_file_name,
                sounding_field_names=sounding_field_names)
            print MINOR_SEPARATOR_STRING

            file_time_index += 1
            if this_example_dict is None:
                continue

            num_file_times_in_memory += 1
            sounding_field_names = this_example_dict[SOUNDING_FIELDS_KEY]

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

        if refl_masking_threshold_dbz is not None:
            full_radar_image_matrix = dl_utils.mask_low_reflectivity_pixels(
                radar_image_matrix_3d=full_radar_image_matrix,
                field_names=radar_field_names,
                reflectivity_threshold_dbz=refl_masking_threshold_dbz)

        if normalization_type_string is not None:
            full_radar_image_matrix = dl_utils.normalize_radar_images(
                radar_image_matrix=full_radar_image_matrix,
                field_names=radar_field_names,
                normalization_type_string=normalization_type_string,
                normalization_param_file_name=normalization_param_file_name,
                min_normalized_value=min_normalized_value,
                max_normalized_value=max_normalized_value).astype('float32')

            if sounding_file_names is not None:
                full_sounding_matrix = dl_utils.normalize_soundings(
                    sounding_matrix=full_sounding_matrix,
                    field_names=sounding_field_names,
                    normalization_type_string=normalization_type_string,
                    normalization_param_file_name=normalization_param_file_name,
                    min_normalized_value=min_normalized_value,
                    max_normalized_value=max_normalized_value).astype('float32')

        list_of_predictor_matrices, target_array = _select_batch(
            list_of_predictor_matrices=[
                full_radar_image_matrix, full_sounding_matrix],
            target_values=all_target_values,
            num_examples_per_batch=num_examples_per_batch,
            binarize_target=binarize_target, num_classes=num_classes)

        list_of_predictor_matrices, target_array = _augment_radar_images(
            list_of_predictor_matrices=list_of_predictor_matrices,
            target_array=target_array, option_dict=option_dict)

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
            yield (radar_image_matrix, target_array)
        else:
            yield ([radar_image_matrix, sounding_matrix], target_array)


def storm_image_generator_2d3d_myrorss(option_dict):
    """Generates examples with 2-D and 3-D radar images.

    Each example consists of the following data for one storm object:

    - One storm-centered 3-D reflectivity image
    - Storm-centered 2-D azimuthal-shear images (one per field)
    - One storm-centered sounding (maybe)
    - Target class

    F_a = number of azimuthal-shear fields
    m = number of rows per reflectivity image
    n = number of columns per reflectivity image
    H_r = number of heights per reflectivity image
    M = number of rows per azimuthal-shear image
    N = number of columns per azimuthal-shear image

    :param option_dict: Dictionary with the following keys.
    option_dict['radar_file_name_matrix']: See doc for
        `storm_image_generator_2d`.
    option_dict['top_target_dir_name']: Same
    option_dict['num_examples_per_batch']: Same.
    option_dict['num_examples_per_file']: Same.
    option_dict['target_name']: Same.
    option_dict['num_rows_to_keep']: Same.
    option_dict['num_columns_to_keep']: Same.
    option_dict['normalization_type_string']: Same.
    option_dict['min_normalized_value']: Same.
    option_dict['max_normalized_value']: Same.
    option_dict['normalization_param_file_name']: Same.
    option_dict['radar_file_name_matrix_pos_targets_only']: Same.
    option_dict['binarize_target']: See doc for `storm_image_generator_2d`.
    option_dict['sampling_fraction_by_class_dict']: Same.
    option_dict['sounding_field_names']: Same.
    option_dict['top_sounding_dir_name']: Same.
    option_dict['sounding_lag_time_sec']: Same.
    option_dict['loop_thru_files_once']: Same.
    option_dict['num_translations']: Same.
    option_dict['max_translation_pixels']: Same.
    option_dict['num_rotations']: Same.
    option_dict['max_absolute_rotation_angle_deg']: Same.
    option_dict['num_noisings']: Same.
    option_dict['max_noise_standard_deviation']: Same.

    If `sounding_field_names is None`, this method returns...

    :return: predictor_list: List with the following items.
    predictor_list[0] = reflectivity_image_matrix_dbz: numpy array
        (E x m x n x H_r x 1) of storm-centered reflectivity images.
    predictor_list[1] = azimuthal_shear_image_matrix_s01: numpy array
        (E x M x N x F_a) of storm-centered azimuthal-shear images.
    :return: target_array: See output doc for `storm_image_generator_2d`.

    If `sounding_field_names is not None`, this method returns...

    :return: predictor_list: List with the following items.
    predictor_list[0] = reflectivity_image_matrix_dbz: See documentation above.
    predictor_list[1] = azimuthal_shear_image_matrix_s01: See documentation
        above.
    predictor_list[2] = sounding_matrix: See output doc for
        `storm_image_generator_2d`.
    :return: target_array: See output doc for `storm_image_generator_2d`.
    """

    if option_dict is None:
        orig_option_dict = {}
    else:
        orig_option_dict = option_dict.copy()

    option_dict = DEFAULT_GENERATOR_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    loop_thru_files_once = option_dict[LOOP_ONCE_KEY]
    error_checking.assert_is_boolean(loop_thru_files_once)

    num_examples_per_batch = option_dict[NUM_EXAMPLES_PER_BATCH_KEY]
    num_examples_per_file = option_dict[NUM_EXAMPLES_PER_FILE_KEY]
    radar_file_name_matrix = option_dict[RADAR_FILE_NAMES_KEY]
    binarize_target = option_dict[BINARIZE_TARGET_KEY]
    radar_file_name_matrix_pos_targets_only = option_dict[
        POSITIVE_RADAR_FILE_NAMES_KEY]
    sounding_field_names = option_dict[SOUNDING_FIELDS_KEY]

    check_input_args(
        num_examples_per_batch=num_examples_per_batch,
        num_examples_per_file=num_examples_per_file,
        radar_file_name_matrix=radar_file_name_matrix, num_radar_dimensions=2,
        binarize_target=binarize_target,
        radar_file_name_matrix_pos_targets_only=
        radar_file_name_matrix_pos_targets_only,
        sounding_field_names=sounding_field_names)

    target_name = option_dict[TARGET_NAME_KEY]
    top_target_dir_name = option_dict[TARGET_DIRECTORY_KEY]
    num_rows_to_keep = option_dict[NUM_ROWS_TO_KEEP_KEY]
    num_columns_to_keep = option_dict[NUM_COLUMNS_TO_KEEP_KEY]
    sampling_fraction_by_class_dict = option_dict[SAMPLING_FRACTIONS_KEY]

    normalization_type_string = option_dict[NORMALIZATION_TYPE_KEY]
    min_normalized_value = option_dict[MIN_NORMALIZED_VALUE_KEY]
    max_normalized_value = option_dict[MAX_NORMALIZED_VALUE_KEY]
    normalization_param_file_name = option_dict[NORMALIZATION_FILE_KEY]

    if sounding_field_names is None:
        sounding_file_names = None
    else:
        sounding_file_names = find_sounding_files(
            top_sounding_dir_name=option_dict[SOUNDING_DIRECTORY_KEY],
            radar_file_name_matrix=radar_file_name_matrix,
            target_name=target_name,
            lag_time_for_convective_contamination_sec=
            option_dict[SOUNDING_LAG_TIME_KEY])

    (reflectivity_file_name_matrix, az_shear_file_name_matrix
    ) = separate_radar_files_2d3d(radar_file_name_matrix)

    if radar_file_name_matrix_pos_targets_only is not None:
        (refl_file_name_matrix_pos_targets_only,
         az_shear_file_name_matrix_pos_targets_only
        ) = separate_radar_files_2d3d(radar_file_name_matrix_pos_targets_only)

    num_azimuthal_shear_fields = az_shear_file_name_matrix.shape[1]
    azimuthal_shear_field_names = [''] * num_azimuthal_shear_fields
    for j in range(num_azimuthal_shear_fields):
        azimuthal_shear_field_names[j] = storm_images.image_file_name_to_field(
            az_shear_file_name_matrix[0, j])

    num_examples_per_batch_by_class_dict = _get_num_examples_per_batch_by_class(
        num_examples_per_batch=num_examples_per_batch, target_name=target_name,
        sampling_fraction_by_class_dict=sampling_fraction_by_class_dict)

    num_classes = labels.column_name_to_num_classes(
        column_name=target_name, include_dead_storms=False)
    num_file_times_per_batch = int(numpy.ceil(
        float(num_examples_per_batch) / num_examples_per_file))

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
                        file_time_index, ...].tolist()
                )
                these_az_shear_file_names_pos_targets_only = (
                    az_shear_file_name_matrix_pos_targets_only[
                        file_time_index, ...].tolist()
                )

            this_example_dict = _read_input_files_2d3d(
                reflectivity_file_names=reflectivity_file_name_matrix[
                    file_time_index, ...].tolist(),
                azimuthal_shear_file_names=az_shear_file_name_matrix[
                    file_time_index, ...].tolist(),
                top_target_dir_name=top_target_dir_name,
                target_name=target_name,
                num_examples_left_by_class_dict=num_examples_left_by_class_dict,
                num_rows_to_keep=num_rows_to_keep,
                num_columns_to_keep=num_columns_to_keep,
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
            sounding_field_names = this_example_dict[SOUNDING_FIELDS_KEY]

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

        if normalization_type_string is not None:
            full_reflectivity_matrix_dbz = dl_utils.normalize_radar_images(
                radar_image_matrix=full_reflectivity_matrix_dbz,
                field_names=[radar_utils.REFL_NAME],
                normalization_type_string=normalization_type_string,
                normalization_param_file_name=normalization_param_file_name,
                min_normalized_value=min_normalized_value,
                max_normalized_value=max_normalized_value).astype('float32')

            full_azimuthal_shear_matrix_s01 = dl_utils.normalize_radar_images(
                radar_image_matrix=full_azimuthal_shear_matrix_s01,
                field_names=azimuthal_shear_field_names,
                normalization_type_string=normalization_type_string,
                normalization_param_file_name=normalization_param_file_name,
                min_normalized_value=min_normalized_value,
                max_normalized_value=max_normalized_value).astype('float32')

            if sounding_file_names is not None:
                full_sounding_matrix = dl_utils.normalize_soundings(
                    sounding_matrix=full_sounding_matrix,
                    field_names=sounding_field_names,
                    normalization_type_string=normalization_type_string,
                    normalization_param_file_name=normalization_param_file_name,
                    min_normalized_value=min_normalized_value,
                    max_normalized_value=max_normalized_value).astype('float32')

        list_of_predictor_matrices, target_array = _select_batch(
            list_of_predictor_matrices=[
                full_reflectivity_matrix_dbz, full_azimuthal_shear_matrix_s01,
                full_sounding_matrix],
            target_values=all_target_values,
            num_examples_per_batch=num_examples_per_batch,
            binarize_target=binarize_target, num_classes=num_classes)

        list_of_predictor_matrices, target_array = _augment_radar_images(
            list_of_predictor_matrices=list_of_predictor_matrices,
            target_array=target_array, option_dict=option_dict)

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
                   target_array)
        else:
            yield ([reflectivity_matrix_dbz, azimuthal_shear_matrix_s01,
                    sounding_matrix],
                   target_array)
