"""IO methods for training and on-the-fly validation.

--- NOTATION ---

The following letters will be used throughout this module.

E = number of examples (storm objects)
M = number of rows in each radar image
N = number of columns in each radar image
H_r = number of radar heights
F_r = number of radar fields (or "variables" or "channels")
H_s = number of sounding heights
F_s = number of sounding fields (or "variables" or "channels")
C = number of radar field/height pairs
"""

import numpy
import keras
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.deep_learning import data_augmentation
from gewittergefahr.deep_learning import input_examples
from gewittergefahr.gg_utils import target_val_utils
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import error_checking

# TODO(thunderhoser): Implement data augmentation in generators.

KM_TO_METRES = 1000

EXAMPLE_FILES_KEY = 'example_file_names'
FIRST_STORM_TIME_KEY = 'first_storm_time_unix_sec'
LAST_STORM_TIME_KEY = 'last_storm_time_unix_sec'
NUM_EXAMPLES_PER_BATCH_KEY = 'num_examples_per_batch'
RADAR_FIELDS_KEY = 'radar_field_names'
RADAR_HEIGHTS_KEY = 'radar_heights_m_agl'
SOUNDING_FIELDS_KEY = 'sounding_field_names'
SOUNDING_HEIGHTS_KEY = 'sounding_heights_m_agl'
NUM_ROWS_KEY = 'num_grid_rows'
NUM_COLUMNS_KEY = 'num_grid_columns'
NORMALIZATION_TYPE_KEY = 'normalization_type_string'
NORMALIZATION_FILE_KEY = 'normalization_param_file_name'
MIN_NORMALIZED_VALUE_KEY = 'min_normalized_value'
MAX_NORMALIZED_VALUE_KEY = 'max_normalized_value'
BINARIZE_TARGET_KEY = 'binarize_target'
SAMPLING_FRACTIONS_KEY = 'class_to_sampling_fraction_dict'
LOOP_ONCE_KEY = 'loop_thru_files_once'
REFLECTIVITY_MASK_KEY = 'refl_masking_threshold_dbz'

DEFAULT_GENERATOR_OPTION_DICT = {
    NORMALIZATION_TYPE_KEY: dl_utils.Z_NORMALIZATION_TYPE_STRING,
    MIN_NORMALIZED_VALUE_KEY: dl_utils.DEFAULT_MIN_NORMALIZED_VALUE,
    MAX_NORMALIZED_VALUE_KEY: dl_utils.DEFAULT_MAX_NORMALIZED_VALUE,
    BINARIZE_TARGET_KEY: False,
    SAMPLING_FRACTIONS_KEY: None,
    LOOP_ONCE_KEY: False,
    REFLECTIVITY_MASK_KEY: dl_utils.DEFAULT_REFL_MASK_THRESHOLD_DBZ
}

NUM_TRANSLATIONS_KEY = 'num_translations'
MAX_TRANSLATION_KEY = 'max_translation_pixels'
NUM_ROTATIONS_KEY = 'num_rotations'
MAX_ROTATION_KEY = 'max_absolute_rotation_angle_deg'
NUM_NOISINGS_KEY = 'num_noisings'
MAX_NOISE_KEY = 'max_noise_standard_deviation'

DEFAULT_AUGMENTATION_OPTION_DICT = {
    NUM_TRANSLATIONS_KEY: 0,
    MAX_TRANSLATION_KEY: 3,
    NUM_ROTATIONS_KEY: 0,
    MAX_ROTATION_KEY: 15.,
    NUM_NOISINGS_KEY: 0,
    MAX_NOISE_KEY: 0.02
}

DEFAULT_GENERATOR_OPTION_DICT.update(DEFAULT_AUGMENTATION_OPTION_DICT)


def _get_num_ex_per_batch_by_class(
        num_examples_per_batch, target_name, class_to_sampling_fraction_dict):
    """Returns number of examples needed for each class per batch.

    :param num_examples_per_batch: Total number of examples per batch.
    :param target_name: Name of target variable.  Must be accepted by
        `target_val_utils.target_name_to_params`.
    :param class_to_sampling_fraction_dict: See doc for `example_generator_3d`.
    :return: class_to_num_ex_per_batch_dict: Dictionary, where each key
        is the integer ID for a target class (-2 for "dead storm") and each
        value is the number of examples needed per batch.
    """

    num_extended_classes = target_val_utils.target_name_to_num_classes(
        target_name=target_name, include_dead_storms=True)

    if class_to_sampling_fraction_dict is None:
        num_classes = target_val_utils.target_name_to_num_classes(
            target_name=target_name, include_dead_storms=False)
        include_dead_storms = num_extended_classes > num_classes

        if include_dead_storms:
            first_keys = numpy.array(
                [target_val_utils.DEAD_STORM_INTEGER], dtype=int)
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
        sampling_fraction_by_class_dict=class_to_sampling_fraction_dict,
        target_name=target_name, num_examples_total=num_examples_per_batch)


def _get_num_examples_to_read_by_class(
        class_to_num_ex_per_batch_dict, target_values_in_memory):
    """Returns number of examples desired in next file for each class.

    :param class_to_num_ex_per_batch_dict: Dictionary created by
        `_get_num_ex_per_batch_by_class`.
    :param target_values_in_memory: 1-D numpy array of target values (integer
        class labels) in memory.
    :return: class_to_num_ex_to_read_dict: Dictionary, where each key
        is the integer ID for a target class (-2 for "dead storm") and each
        value is the number of examples to read.
    """

    if target_values_in_memory is None:
        return class_to_num_ex_per_batch_dict

    class_to_num_ex_to_read_dict = {}

    for this_class in class_to_num_ex_per_batch_dict.keys():
        this_num_examples = (
            class_to_num_ex_per_batch_dict[this_class] -
            numpy.sum(target_values_in_memory == this_class)
        )
        this_num_examples = max([this_num_examples, 0])
        class_to_num_ex_to_read_dict.update({this_class: this_num_examples})

    return class_to_num_ex_to_read_dict


def _check_stopping_criterion(
        num_examples_per_batch, class_to_num_ex_per_batch_dict,
        class_to_sampling_fraction_dict, target_values_in_memory):
    """Checks stopping criterion for generator.

    :param num_examples_per_batch: Total number of examples per batch.
    :param class_to_num_ex_per_batch_dict: Dictionary created by
        `_get_num_ex_per_batch_by_class`.
    :param class_to_sampling_fraction_dict: See doc for `example_generator_2d`.
    :param target_values_in_memory: 1-D numpy array of target values (integer
        class labels) currently in memory.
    :return: stop_generator: Boolean flag.
    """

    class_to_num_ex_in_memory_dict = {}
    for this_key in class_to_num_ex_per_batch_dict:
        this_value = numpy.sum(target_values_in_memory == this_key)
        class_to_num_ex_in_memory_dict.update({this_key: this_value})

    print 'Number of examples in memory by class:\n{0:s}\n'.format(
        str(class_to_num_ex_in_memory_dict))

    num_examples_in_memory = len(target_values_in_memory)
    stop_generator = num_examples_in_memory >= num_examples_per_batch

    if stop_generator and class_to_sampling_fraction_dict is not None:
        for this_key in class_to_num_ex_per_batch_dict.keys():
            stop_generator = (
                stop_generator and
                class_to_num_ex_in_memory_dict[this_key] >=
                class_to_num_ex_per_batch_dict[this_key]
            )

    return stop_generator


def _select_batch(
        list_of_predictor_matrices, target_values, num_examples_per_batch,
        binarize_target, num_classes):
    """Randomly selects batch from examples in memory.

    E = number of examples in memory
    e = number of examples in batch

    :param list_of_predictor_matrices: 1-D list of predictor matrices.  Each
        item should be a numpy array where the first axis has length E.
    :param target_values: length-E numpy array of target values (integer class
        labels).
    :param num_examples_per_batch: Number of examples per batch.
    :param binarize_target: Boolean flag.  If True, target variable will
        be binarized.  Only the highest class will be considered positive, and
        all others will be considered negative.
    :param num_classes: Number of target classes.
    :return: list_of_predictor_matrices: Same as input, but the first axis of
        each numpy array now has length e.
    :return: target_array: See output doc for `example_generator_2d`.
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

    target_values[target_values == target_val_utils.DEAD_STORM_INTEGER] = 0
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


def check_generator_input_args(option_dict):
    """Error-checks input arguments for generator.

    :param option_dict: See doc for `example_generator_2d_or_3d` or
        `example_generator_2d3d_myrorss`.
    :return: option_dict: Same as input, except that defaults have been added.
    """

    orig_option_dict = option_dict.copy()
    option_dict = DEFAULT_GENERATOR_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    error_checking.assert_is_string_list(option_dict[EXAMPLE_FILES_KEY])
    error_checking.assert_is_numpy_array(
        numpy.array(option_dict[EXAMPLE_FILES_KEY]), num_dimensions=1)

    error_checking.assert_is_integer(option_dict[NUM_EXAMPLES_PER_BATCH_KEY])
    error_checking.assert_is_geq(option_dict[NUM_EXAMPLES_PER_BATCH_KEY], 32)
    error_checking.assert_is_boolean(option_dict[BINARIZE_TARGET_KEY])
    error_checking.assert_is_boolean(option_dict[LOOP_ONCE_KEY])

    return option_dict


def example_generator_2d_or_3d(option_dict):
    """Generates examples with either all 2-D or all 3-D radar images.

    Each example corresponds to one storm object and contains the following
    data:

    - Storm-centered radar images (either one 2-D image for each storm object
      and field/height pair, or one 3-D image for each storm object and field)
    - Storm-centered sounding (optional)
    - Target class

    :param option_dict: Dictionary with the following keys.
    option_dict['example_file_names']: 1-D list of paths to input files (will be
        read by `input_examples.read_example_file`.
    option_dict['first_storm_time_unix_sec']: First storm time.  This generator
        will throw out any examples not in the period
        `first_storm_time_unix_sec`...`last_storm_time_unix_sec`.
    option_dict['last_storm_time_unix_sec']: See above.
    option_dict['num_examples_per_batch']: Number of examples in each training
        or validation batch.
    option_dict['radar_field_names']: Determines which radar fields are used.
        See doc for `input_examples.read_example_file`.
    option_dict['radar_heights_m_agl']: Determines which radar heights are used.
        See doc for `input_examples.read_example_file`.
    option_dict['sounding_field_names']: Determines which sounding fields (if
        any) are used.  See doc for `input_examples.read_example_file`.
    option_dict['sounding_heights_m_agl']: Determines which sounding heights (if
        any) are used.  See doc for `input_examples.read_example_file`.
    option_dict['num_grid_rows']: Number of rows in each radar image.  See doc
        for `input_examples.read_example_file`.
    option_dict['num_grid_columns']: Same but for columns.
    option_dict['normalization_type_string']: Used to normalize radar images and
        soundings.  See doc for `deep_learning_utils.normalize_radar_images` and
        `deep_learning_utils.normalize_soundings`.
    option_dict['normalization_param_file_name']: Same.
    option_dict['min_normalized_value']: Same.
    option_dict['max_normalized_value']: Same.
    option_dict['binarize_target']: Boolean flag.  If True, the target variable
        will be binarized, so that the highest class is positive (1) and all
        other classes are negative (0).
    option_dict['class_to_sampling_fraction_dict']: Used for downsampling based
        on target variable.  See doc for `deep_learning_utils.sample_by_class`.
    option_dict['loop_thru_files_once']: Boolean flag.  If True, this generator
        will loop through "example_file_names" only once.  If False, the
        generator will loop indefinitely (as many times as it is called).
    option_dict['refl_masking_threshold_dbz']:
        [used only if "example_file_names" contain 3-D radar images]
        Reflectivity-masking threshold.  All grid cells with reflectivity <
        threshold will be masked out.  If `refl_masking_threshold_dbz is None`,
        there will be no masking.
    option_dict['num_translations']: Used for data augmentation.  See doc for
        `_augment_radar_images`.
    option_dict['max_translation_pixels']: Same.
    option_dict['num_rotations']: Same.
    option_dict['max_absolute_rotation_angle_deg']: Same.
    option_dict['num_noisings']: Same.
    option_dict['max_noise_standard_deviation']: Same.

    If `sounding_field_names is None`, this method returns the following.

    :return: radar_image_matrix: numpy array (E x M x N x C or
        E x M x N x H_r x F_r) of storm-centered radar images.
    :return: target_array: If the problem is multiclass and
        `binarize_target = False`, this is an E-by-K numpy array with only zeros
        and ones (though technically the type is "float64").  If
        target_array[i, k] = 1, the [i]th example belongs to the [k]th class.

        If the problem is binary or `binarize_target = True`, this is a length-E
        numpy array of integers (0 for negative class, 1 for positive class).

    If `sounding_field_names is not None`, this method returns the following.

    :return: predictor_list: List with the following items.
    predictor_list[0] = radar_image_matrix: See above.
    predictor_list[1] = sounding_matrix: numpy array (E x H_s x F_s) of storm-
        centered soundings.
    :return: target_array: See above.
    """

    option_dict = check_generator_input_args(option_dict)

    example_file_names = option_dict[EXAMPLE_FILES_KEY]
    num_examples_per_batch = option_dict[NUM_EXAMPLES_PER_BATCH_KEY]
    binarize_target = option_dict[BINARIZE_TARGET_KEY]
    loop_thru_files_once = option_dict[LOOP_ONCE_KEY]

    radar_field_names = option_dict[RADAR_FIELDS_KEY]
    radar_heights_m_agl = option_dict[RADAR_HEIGHTS_KEY]
    sounding_field_names = option_dict[SOUNDING_FIELDS_KEY]
    sounding_heights_m_agl = option_dict[SOUNDING_HEIGHTS_KEY]
    first_storm_time_unix_sec = option_dict[FIRST_STORM_TIME_KEY]
    last_storm_time_unix_sec = option_dict[LAST_STORM_TIME_KEY]
    num_grid_rows = option_dict[NUM_ROWS_KEY]
    num_grid_columns = option_dict[NUM_COLUMNS_KEY]

    refl_masking_threshold_dbz = option_dict[REFLECTIVITY_MASK_KEY]
    normalization_type_string = option_dict[NORMALIZATION_TYPE_KEY]
    normalization_param_file_name = option_dict[NORMALIZATION_FILE_KEY]
    min_normalized_value = option_dict[MIN_NORMALIZED_VALUE_KEY]
    max_normalized_value = option_dict[MAX_NORMALIZED_VALUE_KEY]

    class_to_sampling_fraction_dict = option_dict[SAMPLING_FRACTIONS_KEY]
    this_example_dict = input_examples.read_example_file(
        netcdf_file_name=example_file_names[0], metadata_only=True)
    target_name = this_example_dict[input_examples.TARGET_NAME_KEY]

    class_to_num_ex_per_batch_dict = _get_num_ex_per_batch_by_class(
        num_examples_per_batch=num_examples_per_batch, target_name=target_name,
        class_to_sampling_fraction_dict=class_to_sampling_fraction_dict)

    num_classes = target_val_utils.target_name_to_num_classes(
        target_name=target_name, include_dead_storms=False)

    radar_image_matrix = None
    sounding_matrix = None
    target_values = None

    file_index = 0
    include_soundings = False
    num_radar_dimensions = -1

    while True:
        if loop_thru_files_once and file_index >= len(example_file_names):
            raise StopIteration

        stop_generator = False
        while not stop_generator:
            if file_index == len(example_file_names):
                if loop_thru_files_once:
                    if target_values is None:
                        raise StopIteration
                    break

                file_index = 0

            class_to_num_ex_to_read_dict = _get_num_examples_to_read_by_class(
                class_to_num_ex_per_batch_dict=class_to_num_ex_per_batch_dict,
                target_values_in_memory=target_values)

            print 'Reading data from: "{0:s}"...'.format(
                example_file_names[file_index])
            this_example_dict = input_examples.read_example_file(
                netcdf_file_name=example_file_names[file_index],
                include_soundings=sounding_field_names is not None,
                radar_field_names_to_keep=radar_field_names,
                radar_heights_to_keep_m_agl=radar_heights_m_agl,
                sounding_field_names_to_keep=sounding_field_names,
                sounding_heights_to_keep_m_agl=sounding_heights_m_agl,
                first_time_to_keep_unix_sec=first_storm_time_unix_sec,
                last_time_to_keep_unix_sec=last_storm_time_unix_sec,
                num_rows_to_keep=num_grid_rows,
                num_columns_to_keep=num_grid_columns,
                class_to_num_examples_dict=class_to_num_ex_to_read_dict)

            file_index += 1
            if this_example_dict is None:
                continue

            include_soundings = (
                input_examples.SOUNDING_MATRIX_KEY in this_example_dict)
            num_radar_dimensions = len(
                this_example_dict[input_examples.RADAR_IMAGE_MATRIX_KEY].shape
            ) - 2

            if target_values is None:
                radar_image_matrix = (
                    this_example_dict[input_examples.RADAR_IMAGE_MATRIX_KEY]
                    + 0.
                )
                target_values = (
                    this_example_dict[input_examples.TARGET_VALUES_KEY] + 0)

                if include_soundings:
                    sounding_matrix = (
                        this_example_dict[input_examples.SOUNDING_MATRIX_KEY]
                        + 0.
                    )
            else:
                radar_image_matrix = numpy.concatenate(
                    (radar_image_matrix,
                     this_example_dict[input_examples.RADAR_IMAGE_MATRIX_KEY]),
                    axis=0)
                target_values = numpy.concatenate((
                    target_values,
                    this_example_dict[input_examples.TARGET_VALUES_KEY]
                ))

                if include_soundings:
                    sounding_matrix = numpy.concatenate(
                        (sounding_matrix,
                         this_example_dict[input_examples.SOUNDING_MATRIX_KEY]),
                        axis=0)

            stop_generator = _check_stopping_criterion(
                num_examples_per_batch=num_examples_per_batch,
                class_to_num_ex_per_batch_dict=class_to_num_ex_per_batch_dict,
                class_to_sampling_fraction_dict=class_to_sampling_fraction_dict,
                target_values_in_memory=target_values)

        if class_to_sampling_fraction_dict is not None:
            indices_to_keep = dl_utils.sample_by_class(
                sampling_fraction_by_class_dict=class_to_sampling_fraction_dict,
                target_name=target_name, target_values=target_values,
                num_examples_total=num_examples_per_batch)

            radar_image_matrix = radar_image_matrix[indices_to_keep, ...]
            target_values = target_values[indices_to_keep]
            if include_soundings:
                sounding_matrix = sounding_matrix[indices_to_keep, ...]

        if refl_masking_threshold_dbz is not None and num_radar_dimensions == 3:
            radar_image_matrix = dl_utils.mask_low_reflectivity_pixels(
                radar_image_matrix_3d=radar_image_matrix,
                field_names=radar_field_names,
                reflectivity_threshold_dbz=refl_masking_threshold_dbz)

        if normalization_type_string is not None:
            radar_image_matrix = dl_utils.normalize_radar_images(
                radar_image_matrix=radar_image_matrix,
                field_names=radar_field_names,
                normalization_type_string=normalization_type_string,
                normalization_param_file_name=normalization_param_file_name,
                min_normalized_value=min_normalized_value,
                max_normalized_value=max_normalized_value).astype('float32')

            if include_soundings:
                sounding_matrix = dl_utils.normalize_soundings(
                    sounding_matrix=sounding_matrix,
                    field_names=sounding_field_names,
                    normalization_type_string=normalization_type_string,
                    normalization_param_file_name=normalization_param_file_name,
                    min_normalized_value=min_normalized_value,
                    max_normalized_value=max_normalized_value).astype('float32')

        list_of_predictor_matrices, target_array = _select_batch(
            list_of_predictor_matrices=[radar_image_matrix, sounding_matrix],
            target_values=target_values,
            num_examples_per_batch=num_examples_per_batch,
            binarize_target=binarize_target, num_classes=num_classes)

        radar_image_matrix = None
        sounding_matrix = None
        target_values = None

        if include_soundings:
            yield (list_of_predictor_matrices, target_array)
        else:
            yield (list_of_predictor_matrices[0], target_array)


def example_generator_2d3d_myrorss(option_dict):
    """Generates examples with both 2-D and 3-D radar images.

    Each example corresponds to one storm object and contains the following
    data:

    - Storm-centered azimuthal-shear images (one 2-D image for each
      azimuthal-shear field)
    - Storm-centered reflectivity image (3-D)
    - Storm-centered sounding (optional)
    - Target class

    M = number of rows in each reflectivity image
    N = number of columns in each reflectivity image

    :param option_dict: Same as input to `example_generator_2d_or_3d`, but with
        two exceptions.

    [1] No "refl_masking_threshold_dbz"
    [2] "num_grid_rows" and "num_grid_columns" apply only to reflectivity
        images.  They will be doubled for azimuthal-shear images (because
        az-shear images have twice the resolution, or half the grid spacing, of
        reflectivity images).

    :return: predictor_list: List with the following items.
    predictor_list[0] = reflectivity_image_matrix_dbz: numpy array
        (E x M x N x H_r x 1) of storm-centered reflectivity images.
    predictor_list[1] = az_shear_image_matrix_s01: numpy array (E x 2M x 2N x C)
        of storm-centered azimuthal-shear images.
    predictor_list[2] = sounding_matrix: numpy array (E x H_s x F_s) of storm-
        centered soundings.  If `sounding_field_names is None`, this item does
        not exist.

    :return: target_array: See doc for `example_generator_2d_or_3d`.
    """

    option_dict = check_generator_input_args(option_dict)

    example_file_names = option_dict[EXAMPLE_FILES_KEY]
    num_examples_per_batch = option_dict[NUM_EXAMPLES_PER_BATCH_KEY]
    binarize_target = option_dict[BINARIZE_TARGET_KEY]
    loop_thru_files_once = option_dict[LOOP_ONCE_KEY]

    azimuthal_shear_field_names = option_dict[RADAR_FIELDS_KEY]
    reflectivity_heights_m_agl = option_dict[RADAR_HEIGHTS_KEY]
    sounding_field_names = option_dict[SOUNDING_FIELDS_KEY]
    sounding_heights_m_agl = option_dict[SOUNDING_HEIGHTS_KEY]
    first_storm_time_unix_sec = option_dict[FIRST_STORM_TIME_KEY]
    last_storm_time_unix_sec = option_dict[LAST_STORM_TIME_KEY]
    num_grid_rows = option_dict[NUM_ROWS_KEY]
    num_grid_columns = option_dict[NUM_COLUMNS_KEY]

    normalization_type_string = option_dict[NORMALIZATION_TYPE_KEY]
    normalization_param_file_name = option_dict[NORMALIZATION_FILE_KEY]
    min_normalized_value = option_dict[MIN_NORMALIZED_VALUE_KEY]
    max_normalized_value = option_dict[MAX_NORMALIZED_VALUE_KEY]

    class_to_sampling_fraction_dict = option_dict[SAMPLING_FRACTIONS_KEY]
    this_example_dict = input_examples.read_example_file(
        netcdf_file_name=example_file_names[0], metadata_only=True)
    target_name = this_example_dict[input_examples.TARGET_NAME_KEY]

    class_to_num_ex_per_batch_dict = _get_num_ex_per_batch_by_class(
        num_examples_per_batch=num_examples_per_batch, target_name=target_name,
        class_to_sampling_fraction_dict=class_to_sampling_fraction_dict)

    num_classes = target_val_utils.target_name_to_num_classes(
        target_name=target_name, include_dead_storms=False)

    reflectivity_image_matrix_dbz = None
    az_shear_image_matrix_s01 = None
    sounding_matrix = None
    target_values = None

    file_index = 0
    include_soundings = False

    while True:
        if loop_thru_files_once and file_index >= len(example_file_names):
            raise StopIteration

        stop_generator = False
        while not stop_generator:
            if file_index == len(example_file_names):
                if loop_thru_files_once:
                    if target_values is None:
                        raise StopIteration
                    break

                file_index = 0

            class_to_num_ex_to_read_dict = _get_num_examples_to_read_by_class(
                class_to_num_ex_per_batch_dict=class_to_num_ex_per_batch_dict,
                target_values_in_memory=target_values)

            print 'Reading data from: "{0:s}"...'.format(
                example_file_names[file_index])
            this_example_dict = input_examples.read_example_file(
                netcdf_file_name=example_file_names[file_index],
                include_soundings=sounding_field_names is not None,
                radar_field_names_to_keep=azimuthal_shear_field_names,
                radar_heights_to_keep_m_agl=reflectivity_heights_m_agl,
                sounding_field_names_to_keep=sounding_field_names,
                sounding_heights_to_keep_m_agl=sounding_heights_m_agl,
                first_time_to_keep_unix_sec=first_storm_time_unix_sec,
                last_time_to_keep_unix_sec=last_storm_time_unix_sec,
                num_rows_to_keep=num_grid_rows,
                num_columns_to_keep=num_grid_columns,
                class_to_num_examples_dict=class_to_num_ex_to_read_dict)

            file_index += 1
            if this_example_dict is None:
                continue

            include_soundings = (
                input_examples.SOUNDING_MATRIX_KEY in this_example_dict)

            if target_values is None:
                reflectivity_image_matrix_dbz = (
                    this_example_dict[input_examples.REFL_IMAGE_MATRIX_KEY] + 0.
                )
                az_shear_image_matrix_s01 = (
                    this_example_dict[input_examples.AZ_SHEAR_IMAGE_MATRIX_KEY]
                    + 0.
                )
                target_values = (
                    this_example_dict[input_examples.TARGET_VALUES_KEY] + 0)

                if include_soundings:
                    sounding_matrix = (
                        this_example_dict[input_examples.SOUNDING_MATRIX_KEY]
                        + 0.
                    )
            else:
                reflectivity_image_matrix_dbz = numpy.concatenate(
                    (reflectivity_image_matrix_dbz,
                     this_example_dict[input_examples.REFL_IMAGE_MATRIX_KEY]),
                    axis=0)
                az_shear_image_matrix_s01 = numpy.concatenate((
                    az_shear_image_matrix_s01,
                    this_example_dict[input_examples.AZ_SHEAR_IMAGE_MATRIX_KEY]
                ), axis=0)
                target_values = numpy.concatenate((
                    target_values,
                    this_example_dict[input_examples.TARGET_VALUES_KEY]
                ))

                if include_soundings:
                    sounding_matrix = numpy.concatenate(
                        (sounding_matrix,
                         this_example_dict[input_examples.SOUNDING_MATRIX_KEY]),
                        axis=0)

            stop_generator = _check_stopping_criterion(
                num_examples_per_batch=num_examples_per_batch,
                class_to_num_ex_per_batch_dict=class_to_num_ex_per_batch_dict,
                class_to_sampling_fraction_dict=class_to_sampling_fraction_dict,
                target_values_in_memory=target_values)

        if class_to_sampling_fraction_dict is not None:
            indices_to_keep = dl_utils.sample_by_class(
                sampling_fraction_by_class_dict=class_to_sampling_fraction_dict,
                target_name=target_name, target_values=target_values,
                num_examples_total=num_examples_per_batch)

            reflectivity_image_matrix_dbz = reflectivity_image_matrix_dbz[
                indices_to_keep, ...]
            az_shear_image_matrix_s01 = az_shear_image_matrix_s01[
                indices_to_keep, ...]
            target_values = target_values[indices_to_keep]
            if include_soundings:
                sounding_matrix = sounding_matrix[indices_to_keep, ...]

        if normalization_type_string is not None:
            reflectivity_image_matrix_dbz = dl_utils.normalize_radar_images(
                radar_image_matrix=reflectivity_image_matrix_dbz,
                field_names=[radar_utils.REFL_NAME],
                normalization_type_string=normalization_type_string,
                normalization_param_file_name=normalization_param_file_name,
                min_normalized_value=min_normalized_value,
                max_normalized_value=max_normalized_value).astype('float32')

            az_shear_image_matrix_s01 = dl_utils.normalize_radar_images(
                radar_image_matrix=az_shear_image_matrix_s01,
                field_names=azimuthal_shear_field_names,
                normalization_type_string=normalization_type_string,
                normalization_param_file_name=normalization_param_file_name,
                min_normalized_value=min_normalized_value,
                max_normalized_value=max_normalized_value).astype('float32')

            if include_soundings:
                sounding_matrix = dl_utils.normalize_soundings(
                    sounding_matrix=sounding_matrix,
                    field_names=sounding_field_names,
                    normalization_type_string=normalization_type_string,
                    normalization_param_file_name=normalization_param_file_name,
                    min_normalized_value=min_normalized_value,
                    max_normalized_value=max_normalized_value).astype('float32')

        list_of_predictor_matrices, target_array = _select_batch(
            list_of_predictor_matrices=[
                reflectivity_image_matrix_dbz, az_shear_image_matrix_s01,
                sounding_matrix
            ],
            target_values=target_values,
            num_examples_per_batch=num_examples_per_batch,
            binarize_target=binarize_target, num_classes=num_classes)

        reflectivity_image_matrix_dbz = None
        az_shear_image_matrix_s01 = None
        sounding_matrix = None
        target_values = None

        if include_soundings:
            yield (list_of_predictor_matrices, target_array)
        else:
            yield (list_of_predictor_matrices[:-1], target_array)


def layer_ops_to_field_height_pairs(list_of_operation_dicts):
    """Converts list of layer operations to list of field-height pairs.

    These are the radar field-height pairs needed for input to the layer
    operations.

    :param list_of_operation_dicts: See doc for
        `input_examples.reduce_examples_3d_to_2d`.
    :return: unique_radar_field_names: 1-D list of field names.
    :return: unique_radar_heights_m_agl: 1-D numpy array of unique heights
        (metres above ground level).
    """

    unique_radar_field_names = [
        d[input_examples.RADAR_FIELD_KEY] for d in list_of_operation_dicts
    ]
    unique_radar_field_names = list(set(unique_radar_field_names))

    min_heights_m_agl = numpy.array([
        d[input_examples.MIN_HEIGHT_KEY] for d in list_of_operation_dicts
    ], dtype=int)

    max_heights_m_agl = numpy.array([
        d[input_examples.MAX_HEIGHT_KEY] for d in list_of_operation_dicts
    ], dtype=int)

    min_overall_height_m_agl = numpy.min(min_heights_m_agl)
    max_overall_height_m_agl = numpy.max(max_heights_m_agl)
    num_overall_heights = 1 + int(numpy.round(
        (max_overall_height_m_agl - min_overall_height_m_agl) / KM_TO_METRES
    ))

    unique_radar_heights_m_agl = numpy.linspace(
        min_overall_height_m_agl, max_overall_height_m_agl,
        num=num_overall_heights, dtype=int)

    return unique_radar_field_names, unique_radar_heights_m_agl


def gridrad_generator_2d_reduced(option_dict, list_of_operation_dicts):
    """Generates examples with 2-D GridRad images.

    These 2-D images are produced by applying layer operations to the native 3-D
    images.  The layer operations are specified by `list_of_operation_dicts`.

    :param option_dict: Same as input to `example_generator_2d_or_3d`, but
        without the following keys:
    - "refl_masking_threshold_dbz"
    - "radar_field_names"
    - "radar_heights_m_agl"

    :param list_of_operation_dicts: See doc for
        `input_examples.reduce_examples_3d_to_2d`.

    If `sounding_field_names is None`...

    :return: radar_image_matrix: See doc for `example_generator_2d_or_3d`.
    :return: target_array: Same.

    If `sounding_field_names is not None`...

    :return: predictor_list: See doc for `example_generator_2d_or_3d`.
    :return: target_array: Same.
    """

    option_dict = check_generator_input_args(option_dict)

    example_file_names = option_dict[EXAMPLE_FILES_KEY]
    first_storm_time_unix_sec = option_dict[FIRST_STORM_TIME_KEY]
    last_storm_time_unix_sec = option_dict[LAST_STORM_TIME_KEY]
    num_grid_rows = option_dict[NUM_ROWS_KEY]
    num_grid_columns = option_dict[NUM_COLUMNS_KEY]

    sounding_field_names = option_dict[SOUNDING_FIELDS_KEY]
    sounding_heights_m_agl = option_dict[SOUNDING_HEIGHTS_KEY]

    normalization_type_string = option_dict[NORMALIZATION_TYPE_KEY]
    normalization_param_file_name = option_dict[NORMALIZATION_FILE_KEY]
    min_normalized_value = option_dict[MIN_NORMALIZED_VALUE_KEY]
    max_normalized_value = option_dict[MAX_NORMALIZED_VALUE_KEY]

    num_examples_per_batch = option_dict[NUM_EXAMPLES_PER_BATCH_KEY]
    binarize_target = option_dict[BINARIZE_TARGET_KEY]
    loop_thru_files_once = option_dict[LOOP_ONCE_KEY]
    class_to_sampling_fraction_dict = option_dict[SAMPLING_FRACTIONS_KEY]

    this_example_dict = input_examples.read_example_file(
        netcdf_file_name=example_file_names[0], metadata_only=True)
    target_name = this_example_dict[input_examples.TARGET_NAME_KEY]

    class_to_num_ex_per_batch_dict = _get_num_ex_per_batch_by_class(
        num_examples_per_batch=num_examples_per_batch, target_name=target_name,
        class_to_sampling_fraction_dict=class_to_sampling_fraction_dict)

    num_classes = target_val_utils.target_name_to_num_classes(
        target_name=target_name, include_dead_storms=False)

    unique_radar_field_names, unique_radar_heights_m_agl = (
        layer_ops_to_field_height_pairs(list_of_operation_dicts)
    )

    radar_image_matrix = None
    sounding_matrix = None
    target_values = None

    file_index = 0
    include_soundings = False
    radar_field_names_2d = []

    while True:
        if loop_thru_files_once and file_index >= len(example_file_names):
            raise StopIteration

        stop_generator = False
        while not stop_generator:
            if file_index == len(example_file_names):
                if loop_thru_files_once:
                    if target_values is None:
                        raise StopIteration
                    break

                file_index = 0

            class_to_num_ex_to_read_dict = _get_num_examples_to_read_by_class(
                class_to_num_ex_per_batch_dict=class_to_num_ex_per_batch_dict,
                target_values_in_memory=target_values)

            print 'Reading data from: "{0:s}"...'.format(
                example_file_names[file_index])
            this_example_dict = input_examples.read_example_file(
                netcdf_file_name=example_file_names[file_index],
                include_soundings=sounding_field_names is not None,
                radar_field_names_to_keep=unique_radar_field_names,
                radar_heights_to_keep_m_agl=unique_radar_heights_m_agl,
                sounding_field_names_to_keep=sounding_field_names,
                sounding_heights_to_keep_m_agl=sounding_heights_m_agl,
                first_time_to_keep_unix_sec=first_storm_time_unix_sec,
                last_time_to_keep_unix_sec=last_storm_time_unix_sec,
                num_rows_to_keep=num_grid_rows,
                num_columns_to_keep=num_grid_columns,
                class_to_num_examples_dict=class_to_num_ex_to_read_dict)

            file_index += 1
            if this_example_dict is None:
                continue

            this_example_dict = input_examples.reduce_examples_3d_to_2d(
                example_dict=this_example_dict,
                list_of_operation_dicts=list_of_operation_dicts)
            radar_field_names_2d = this_example_dict[
                input_examples.RADAR_FIELDS_KEY]

            include_soundings = (
                input_examples.SOUNDING_MATRIX_KEY in this_example_dict)

            if target_values is None:
                radar_image_matrix = (
                    this_example_dict[input_examples.RADAR_IMAGE_MATRIX_KEY]
                    + 0.
                )
                target_values = (
                    this_example_dict[input_examples.TARGET_VALUES_KEY] + 0)

                if include_soundings:
                    sounding_matrix = (
                        this_example_dict[input_examples.SOUNDING_MATRIX_KEY]
                        + 0.
                    )
            else:
                radar_image_matrix = numpy.concatenate(
                    (radar_image_matrix,
                     this_example_dict[input_examples.RADAR_IMAGE_MATRIX_KEY]),
                    axis=0)
                target_values = numpy.concatenate((
                    target_values,
                    this_example_dict[input_examples.TARGET_VALUES_KEY]
                ))

                if include_soundings:
                    sounding_matrix = numpy.concatenate(
                        (sounding_matrix,
                         this_example_dict[input_examples.SOUNDING_MATRIX_KEY]),
                        axis=0)

            stop_generator = _check_stopping_criterion(
                num_examples_per_batch=num_examples_per_batch,
                class_to_num_ex_per_batch_dict=class_to_num_ex_per_batch_dict,
                class_to_sampling_fraction_dict=class_to_sampling_fraction_dict,
                target_values_in_memory=target_values)

        if class_to_sampling_fraction_dict is not None:
            indices_to_keep = dl_utils.sample_by_class(
                sampling_fraction_by_class_dict=class_to_sampling_fraction_dict,
                target_name=target_name, target_values=target_values,
                num_examples_total=num_examples_per_batch)

            radar_image_matrix = radar_image_matrix[indices_to_keep, ...]
            target_values = target_values[indices_to_keep]
            if include_soundings:
                sounding_matrix = sounding_matrix[indices_to_keep, ...]

        if normalization_type_string is not None:
            radar_image_matrix = dl_utils.normalize_radar_images(
                radar_image_matrix=radar_image_matrix,
                field_names=radar_field_names_2d,
                normalization_type_string=normalization_type_string,
                normalization_param_file_name=normalization_param_file_name,
                min_normalized_value=min_normalized_value,
                max_normalized_value=max_normalized_value).astype('float32')

            if include_soundings:
                sounding_matrix = dl_utils.normalize_soundings(
                    sounding_matrix=sounding_matrix,
                    field_names=sounding_field_names,
                    normalization_type_string=normalization_type_string,
                    normalization_param_file_name=normalization_param_file_name,
                    min_normalized_value=min_normalized_value,
                    max_normalized_value=max_normalized_value).astype('float32')

        list_of_predictor_matrices, target_array = _select_batch(
            list_of_predictor_matrices=[radar_image_matrix, sounding_matrix],
            target_values=target_values,
            num_examples_per_batch=num_examples_per_batch,
            binarize_target=binarize_target, num_classes=num_classes)

        radar_image_matrix = None
        sounding_matrix = None
        target_values = None

        if include_soundings:
            yield (list_of_predictor_matrices, target_array)
        else:
            yield (list_of_predictor_matrices[0], target_array)
