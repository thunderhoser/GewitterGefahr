"""IO methods for deployment of a trained model.

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

import copy
import numpy
import netCDF4
import keras.utils
from gewittergefahr.deep_learning import input_examples
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.gg_utils import target_val_utils
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import error_checking

INPUT_MATRICES_KEY = 'list_of_input_matrices'
SOUNDING_MATRIX_KEY = 'sounding_matrix'
TARGET_ARRAY_KEY = 'target_array'
FULL_IDS_KEY = 'full_storm_id_strings'
STORM_TIMES_KEY = 'storm_times_unix_sec'
SOUNDING_PRESSURES_KEY = 'sounding_pressure_matrix_pascals'

RADAR_FIELDS_KEY = input_examples.RADAR_FIELDS_KEY
MIN_RADAR_HEIGHTS_KEY = input_examples.MIN_RADAR_HEIGHTS_KEY
MAX_RADAR_HEIGHTS_KEY = input_examples.MAX_RADAR_HEIGHTS_KEY
RADAR_LAYER_OPERATION_NAMES_KEY = input_examples.RADAR_LAYER_OPERATION_NAMES_KEY

REDUCTION_METADATA_KEYS = [
    RADAR_FIELDS_KEY, MIN_RADAR_HEIGHTS_KEY, MAX_RADAR_HEIGHTS_KEY,
    RADAR_LAYER_OPERATION_NAMES_KEY
]

LARGE_INTEGER = int(1e10)


def _finalize_targets(target_values, binarize_target, num_classes):
    """Finalizes batch of target values.

    :param target_values: length-E numpy array of target values (integer class
        labels).
    :param binarize_target: Boolean flag.  If True, target variable will be
        binarized, where the highest class becomes 1 and all other classes
        become 0.  If False, the original classes will be kept, in which case
        the prediction task may be binary or multiclass.
    :param num_classes: Number of classes for target variable.
    :return: target_array: See output doc for `generator_2d_or_3d`.
    """

    target_values[target_values == target_val_utils.DEAD_STORM_INTEGER] = 0

    if binarize_target:
        target_values = (target_values == num_classes - 1).astype(int)
        num_classes_to_predict = 2
    else:
        num_classes_to_predict = num_classes + 0

    if num_classes_to_predict == 2:
        print('Fraction of {0:d} examples in positive class: {1:.3f}'.format(
            len(target_values), numpy.mean(target_values)
        ))
        return target_values

    target_matrix = keras.utils.to_categorical(
        target_values, num_classes_to_predict)

    class_fractions = numpy.mean(target_matrix, axis=0)
    print('Fraction of {0:d} examples in each class: {1:s}\n'.format(
        len(target_values), str(class_fractions)
    ))

    return target_matrix


def _find_examples_to_read(option_dict, num_examples_total):
    """Determines which examples to read.

    E = number of examples to read

    :param option_dict: See doc for any generator in this file.
    :param num_examples_total: Number of examples to generate.
    :return: full_id_strings: length-E list of full storm IDs.
    :return: storm_times_unix_sec: length-E numpy array of storm times.
    """

    error_checking.assert_is_integer(num_examples_total)
    error_checking.assert_is_greater(num_examples_total, 0)

    example_file_names = option_dict[trainval_io.EXAMPLE_FILES_KEY]
    target_name = option_dict[trainval_io.TARGET_NAME_KEY]

    # TODO(thunderhoser): This is a HACK for CNNs that take soundings only.
    if trainval_io.RADAR_FIELDS_KEY in option_dict:
        radar_field_names = option_dict[trainval_io.RADAR_FIELDS_KEY]
        radar_heights_m_agl = option_dict[trainval_io.RADAR_HEIGHTS_KEY]
        num_grid_rows = option_dict[trainval_io.NUM_ROWS_KEY]
        num_grid_columns = option_dict[trainval_io.NUM_COLUMNS_KEY]
    else:
        this_example_dict = input_examples.read_example_file(
            netcdf_file_name=example_file_names[0], read_all_target_vars=False,
            target_name=target_name, metadata_only=True)

        radar_field_names = [
            this_example_dict[input_examples.RADAR_FIELDS_KEY][0]
        ]
        radar_heights_m_agl = numpy.array(
            [this_example_dict[input_examples.RADAR_HEIGHTS_KEY][0]], dtype=int
        )

        num_grid_rows = None
        num_grid_columns = None

    first_storm_time_unix_sec = option_dict[trainval_io.FIRST_STORM_TIME_KEY]
    last_storm_time_unix_sec = option_dict[trainval_io.LAST_STORM_TIME_KEY]
    class_to_sampling_fraction_dict = option_dict[
        trainval_io.SAMPLING_FRACTIONS_KEY]

    full_id_strings = []
    storm_times_unix_sec = numpy.array([], dtype=int)
    target_values = numpy.array([], dtype=int)

    num_files = len(example_file_names)

    for i in range(num_files):
        print('Reading target values from: "{0:s}"...'.format(
            example_file_names[i]
        ))

        this_example_dict = input_examples.read_example_file(
            netcdf_file_name=example_file_names[i], read_all_target_vars=False,
            target_name=target_name, include_soundings=False,
            radar_field_names_to_keep=[radar_field_names[0]],
            radar_heights_to_keep_m_agl=radar_heights_m_agl[[0]],
            first_time_to_keep_unix_sec=first_storm_time_unix_sec,
            last_time_to_keep_unix_sec=last_storm_time_unix_sec,
            num_rows_to_keep=num_grid_rows,
            num_columns_to_keep=num_grid_columns)

        if this_example_dict is None:
            continue

        full_id_strings += this_example_dict[input_examples.FULL_IDS_KEY]
        storm_times_unix_sec = numpy.concatenate((
            storm_times_unix_sec,
            this_example_dict[input_examples.STORM_TIMES_KEY]
        ))
        target_values = numpy.concatenate((
            target_values, this_example_dict[input_examples.TARGET_VALUES_KEY]
        ))

    indices_to_keep = numpy.where(
        target_values != target_val_utils.INVALID_STORM_INTEGER
    )[0]

    full_id_strings = [full_id_strings[k] for k in indices_to_keep]
    storm_times_unix_sec = storm_times_unix_sec[indices_to_keep]
    target_values = target_values[indices_to_keep]
    num_examples_found = len(full_id_strings)

    if class_to_sampling_fraction_dict is None:
        indices_to_keep = numpy.linspace(
            0, num_examples_found - 1, num=num_examples_found, dtype=int)

        if num_examples_found > num_examples_total:
            indices_to_keep = numpy.random.choice(
                indices_to_keep, size=num_examples_total, replace=False)
    else:
        indices_to_keep = dl_utils.sample_by_class(
            sampling_fraction_by_class_dict=class_to_sampling_fraction_dict,
            target_name=target_name, target_values=target_values,
            num_examples_total=num_examples_total)

    full_id_strings = [full_id_strings[k] for k in indices_to_keep]
    storm_times_unix_sec = storm_times_unix_sec[indices_to_keep]

    return full_id_strings, storm_times_unix_sec


def generator_2d_or_3d(option_dict, num_examples_total):
    """Generates examples with either 2-D or 3-D radar images.

    Each example (storm object) consists of the following:

    - Storm-centered radar images (either one 2-D image for each field/height
      pair or one 3-D image for each field)
    - Storm-centered sounding (optional)
    - Target value (class)

    :param option_dict: Dictionary with the following keys.
    option_dict['example_file_names']: See doc for
        `training_validation_io.generator_2d_or_3d`.
    option_dict['binarize_target']: Same.
    option_dict['radar_field_names']: Same.
    option_dict['radar_heights_m_agl']: Same.
    option_dict['sounding_field_names']: Same.
    option_dict['sounding_heights_m_agl']: Same.
    option_dict['first_storm_time_unix_sec']: Same,
    option_dict['last_storm_time_unix_sec']: Same.
    option_dict['num_grid_rows']: Same.
    option_dict['num_grid_columns']: Same.
    option_dict['normalization_type_string']: Same.
    option_dict['normalization_param_file_name']: Same.
    option_dict['min_normalized_value']: Same.
    option_dict['max_normalized_value']: Same.
    option_dict['class_to_sampling_fraction_dict']: Same.
    option_dict['refl_masking_threshold_dbz']: Same.

    :param num_examples_total: Number of examples to generate.

    :return: storm_object_dict: Dictionary with the following keys.
    storm_object_dict['list_of_input_matrices']: length-T list of numpy arrays,
        where T = number of input tensors to model.  The first axis of each
        array has length E.
    storm_object_dict['full_storm_id_strings']: length-E list of full storm IDs.
    storm_object_dict['storm_times_unix_sec']: length-E numpy array of storm
        times.
    storm_object_dict['target_array']: See output doc for
        `training_validation_io.generator_2d_or_3d`.
    storm_object_dict['sounding_pressure_matrix_pascals']: numpy array (E x H_s)
        of pressures.  If soundings were not read, this is None.
    """

    full_id_strings, storm_times_unix_sec = _find_examples_to_read(
        option_dict=option_dict, num_examples_total=num_examples_total)
    print('\n')

    example_file_names = option_dict[trainval_io.EXAMPLE_FILES_KEY]
    target_name = option_dict[trainval_io.TARGET_NAME_KEY]

    first_storm_time_unix_sec = option_dict[trainval_io.FIRST_STORM_TIME_KEY]
    last_storm_time_unix_sec = option_dict[trainval_io.LAST_STORM_TIME_KEY]
    num_grid_rows = option_dict[trainval_io.NUM_ROWS_KEY]
    num_grid_columns = option_dict[trainval_io.NUM_COLUMNS_KEY]
    radar_field_names = option_dict[trainval_io.RADAR_FIELDS_KEY]
    radar_heights_m_agl = option_dict[trainval_io.RADAR_HEIGHTS_KEY]
    sounding_field_names = option_dict[trainval_io.SOUNDING_FIELDS_KEY]
    sounding_heights_m_agl = option_dict[trainval_io.SOUNDING_HEIGHTS_KEY]

    normalization_type_string = option_dict[trainval_io.NORMALIZATION_TYPE_KEY]

    if normalization_type_string is not None:
        normalization_param_file_name = option_dict[
            trainval_io.NORMALIZATION_FILE_KEY
        ]
        min_normalized_value = option_dict[trainval_io.MIN_NORMALIZED_VALUE_KEY]
        max_normalized_value = option_dict[trainval_io.MAX_NORMALIZED_VALUE_KEY]

    binarize_target = option_dict[trainval_io.BINARIZE_TARGET_KEY]
    refl_masking_threshold_dbz = option_dict[trainval_io.REFLECTIVITY_MASK_KEY]

    num_classes = target_val_utils.target_name_to_num_classes(
        target_name=target_name, include_dead_storms=False)

    if sounding_field_names is None:
        sounding_field_names_to_read = None
    else:
        if soundings.PRESSURE_NAME in sounding_field_names:
            sounding_field_names_to_read = sounding_field_names + []
        else:
            sounding_field_names_to_read = (
                sounding_field_names + [soundings.PRESSURE_NAME]
            )

    radar_image_matrix = None
    sounding_matrix = None
    target_values = None
    sounding_pressure_matrix_pascals = None
    file_index = 0

    while True:
        if file_index >= len(example_file_names):
            raise StopIteration

        print('Reading data from: "{0:s}"...'.format(
            example_file_names[file_index]
        ))

        this_example_dict = input_examples.read_example_file(
            netcdf_file_name=example_file_names[file_index],
            read_all_target_vars=False, target_name=target_name,
            include_soundings=sounding_field_names is not None,
            radar_field_names_to_keep=radar_field_names,
            radar_heights_to_keep_m_agl=radar_heights_m_agl,
            sounding_field_names_to_keep=sounding_field_names_to_read,
            sounding_heights_to_keep_m_agl=sounding_heights_m_agl,
            first_time_to_keep_unix_sec=first_storm_time_unix_sec,
            last_time_to_keep_unix_sec=last_storm_time_unix_sec,
            num_rows_to_keep=num_grid_rows,
            num_columns_to_keep=num_grid_columns)

        file_index += 1
        if this_example_dict is None:
            continue

        indices_to_keep = tracking_utils.find_storm_objects(
            all_id_strings=this_example_dict[input_examples.FULL_IDS_KEY],
            all_times_unix_sec=this_example_dict[
                input_examples.STORM_TIMES_KEY],
            id_strings_to_keep=full_id_strings,
            times_to_keep_unix_sec=storm_times_unix_sec, allow_missing=True)

        indices_to_keep = indices_to_keep[indices_to_keep >= 0]
        if len(indices_to_keep) == 0:
            continue

        this_example_dict = input_examples.subset_examples(
            example_dict=this_example_dict, indices_to_keep=indices_to_keep)

        include_soundings = (
            input_examples.SOUNDING_MATRIX_KEY in this_example_dict
        )

        if include_soundings:
            pressure_index = this_example_dict[
                input_examples.SOUNDING_FIELDS_KEY
            ].index(soundings.PRESSURE_NAME)

            this_pressure_matrix_pascals = this_example_dict[
                input_examples.SOUNDING_MATRIX_KEY][..., pressure_index]

            this_sounding_matrix = this_example_dict[
                input_examples.SOUNDING_MATRIX_KEY]

            if soundings.PRESSURE_NAME not in sounding_field_names:
                # this_sounding_matrix = this_sounding_matrix[..., :-1]
                this_sounding_matrix = numpy.delete(
                    this_sounding_matrix, pressure_index, axis=-1)

        if target_values is None:
            radar_image_matrix = (
                this_example_dict[input_examples.RADAR_IMAGE_MATRIX_KEY]
                + 0.
            )
            target_values = (
                this_example_dict[input_examples.TARGET_VALUES_KEY] + 0
            )

            if include_soundings:
                sounding_matrix = this_sounding_matrix + 0.
                sounding_pressure_matrix_pascals = (
                    this_pressure_matrix_pascals + 0.
                )
        else:
            radar_image_matrix = numpy.concatenate(
                (radar_image_matrix,
                 this_example_dict[input_examples.RADAR_IMAGE_MATRIX_KEY]),
                axis=0
            )
            target_values = numpy.concatenate((
                target_values,
                this_example_dict[input_examples.TARGET_VALUES_KEY]
            ))

            if include_soundings:
                sounding_matrix = numpy.concatenate(
                    (sounding_matrix, this_sounding_matrix), axis=0
                )
                sounding_pressure_matrix_pascals = numpy.concatenate(
                    (sounding_pressure_matrix_pascals,
                     this_pressure_matrix_pascals),
                    axis=0
                )

        num_radar_dimensions = len(
            this_example_dict[input_examples.RADAR_IMAGE_MATRIX_KEY].shape
        ) - 2

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

        list_of_predictor_matrices = [radar_image_matrix]
        if include_soundings:
            list_of_predictor_matrices.append(sounding_matrix)

        target_array = _finalize_targets(
            target_values=target_values, binarize_target=binarize_target,
            num_classes=num_classes)

        storm_object_dict = {
            INPUT_MATRICES_KEY: list_of_predictor_matrices,
            TARGET_ARRAY_KEY: target_array,
            FULL_IDS_KEY: this_example_dict[input_examples.FULL_IDS_KEY],
            STORM_TIMES_KEY: this_example_dict[input_examples.STORM_TIMES_KEY],
            SOUNDING_PRESSURES_KEY:
                copy.deepcopy(sounding_pressure_matrix_pascals)
        }

        radar_image_matrix = None
        sounding_matrix = None
        target_values = None
        sounding_pressure_matrix_pascals = None

        yield storm_object_dict


def myrorss_generator_2d3d(option_dict, num_examples_total):
    """Generates examples with both 2-D and 3-D radar images.

    Each example (storm object) consists of the following:

    - Storm-centered azimuthal shear (one 2-D image for each field)
    - Storm-centered reflectivity (one 3-D image)
    - Storm-centered sounding (optional)
    - Target value (class)

    :param option_dict: Dictionary with the following keys.
    option_dict['example_file_names']: See doc for
        `training_validation_io.myrorss_generator_2d3d`.
    option_dict['first_storm_time_unix_sec']: Same.
    option_dict['last_storm_time_unix_sec']: Same.
    option_dict['radar_field_names']: Same.
    option_dict['radar_heights_m_agl']: Same.
    option_dict['num_grid_rows']: Same.
    option_dict['num_grid_columns']: Same.
    option_dict['sounding_field_names']: Same.
    option_dict['sounding_heights_m_agl']: Same.
    option_dict['normalization_type_string']: Same.
    option_dict['normalization_param_file_name']: Same.
    option_dict['min_normalized_value']: Same.
    option_dict['max_normalized_value']: Same.
    option_dict['upsample_reflectivity']: Same.
    option_dict['target_name']: Same.
    option_dict['binarize_target']: Same.
    option_dict['class_to_sampling_fraction_dict']: Same.

    :param num_examples_total: Total number of examples to generate.

    :return: storm_object_dict: Dictionary with the following keys.
    storm_object_dict['list_of_input_matrices']: length-T list of numpy arrays,
        where T = number of input tensors to model.  The first axis of each
        array has length E.
    storm_object_dict['full_storm_id_strings']: length-E list of full storm IDs.
    storm_object_dict['storm_times_unix_sec']: length-E numpy array of storm
        times.
    storm_object_dict['target_array']: See output doc for
        `training_validation_io.myrorss_generator_2d3d`.
    storm_object_dict['sounding_pressure_matrix_pascals']: numpy array (E x H_s)
        of pressures.  If soundings were not read, this is None.
    """

    full_id_strings, storm_times_unix_sec = _find_examples_to_read(
        option_dict=option_dict, num_examples_total=num_examples_total)
    print('\n')

    example_file_names = option_dict[trainval_io.EXAMPLE_FILES_KEY]
    first_storm_time_unix_sec = option_dict[trainval_io.FIRST_STORM_TIME_KEY]
    last_storm_time_unix_sec = option_dict[trainval_io.LAST_STORM_TIME_KEY]

    azimuthal_shear_field_names = option_dict[trainval_io.RADAR_FIELDS_KEY]
    reflectivity_heights_m_agl = option_dict[trainval_io.RADAR_HEIGHTS_KEY]
    num_grid_rows = option_dict[trainval_io.NUM_ROWS_KEY]
    num_grid_columns = option_dict[trainval_io.NUM_COLUMNS_KEY]
    sounding_field_names = option_dict[trainval_io.SOUNDING_FIELDS_KEY]
    sounding_heights_m_agl = option_dict[trainval_io.SOUNDING_HEIGHTS_KEY]

    normalization_type_string = option_dict[trainval_io.NORMALIZATION_TYPE_KEY]

    if normalization_type_string is None:
        normalization_file_name = None
        min_normalized_value = None
        max_normalized_value = None
    else:
        normalization_file_name = option_dict[
            trainval_io.NORMALIZATION_FILE_KEY
        ]
        min_normalized_value = option_dict[trainval_io.MIN_NORMALIZED_VALUE_KEY]
        max_normalized_value = option_dict[trainval_io.MAX_NORMALIZED_VALUE_KEY]

    upsample_refl = option_dict[trainval_io.UPSAMPLE_REFLECTIVITY_KEY]

    if upsample_refl:
        radar_field_names = (
            [radar_utils.REFL_NAME] * len(reflectivity_heights_m_agl) +
            azimuthal_shear_field_names
        )
    else:
        radar_field_names = None

    target_name = option_dict[trainval_io.TARGET_NAME_KEY]
    binarize_target = option_dict[trainval_io.BINARIZE_TARGET_KEY]

    num_classes = target_val_utils.target_name_to_num_classes(
        target_name=target_name, include_dead_storms=False)

    if sounding_field_names is None:
        sounding_field_names_to_read = None
    else:
        if soundings.PRESSURE_NAME in sounding_field_names:
            sounding_field_names_to_read = sounding_field_names + []
        else:
            sounding_field_names_to_read = (
                sounding_field_names + [soundings.PRESSURE_NAME]
            )

    radar_image_matrix = None
    reflectivity_image_matrix_dbz = None
    az_shear_image_matrix_s01 = None
    sounding_matrix = None
    target_values = None

    file_index = 0
    sounding_pressure_matrix_pascals = None

    while True:
        if file_index >= len(example_file_names):
            raise StopIteration

        print('Reading data from: "{0:s}"...'.format(
            example_file_names[file_index]
        ))

        this_example_dict = input_examples.read_example_file(
            netcdf_file_name=example_file_names[file_index],
            read_all_target_vars=False, target_name=target_name,
            include_soundings=sounding_field_names is not None,
            radar_field_names_to_keep=azimuthal_shear_field_names,
            radar_heights_to_keep_m_agl=reflectivity_heights_m_agl,
            sounding_field_names_to_keep=sounding_field_names_to_read,
            sounding_heights_to_keep_m_agl=sounding_heights_m_agl,
            first_time_to_keep_unix_sec=first_storm_time_unix_sec,
            last_time_to_keep_unix_sec=last_storm_time_unix_sec,
            num_rows_to_keep=num_grid_rows,
            num_columns_to_keep=num_grid_columns)

        file_index += 1
        if this_example_dict is None:
            continue

        indices_to_keep = tracking_utils.find_storm_objects(
            all_id_strings=this_example_dict[input_examples.FULL_IDS_KEY],
            all_times_unix_sec=this_example_dict[
                input_examples.STORM_TIMES_KEY],
            id_strings_to_keep=full_id_strings,
            times_to_keep_unix_sec=storm_times_unix_sec, allow_missing=True)

        indices_to_keep = indices_to_keep[indices_to_keep >= 0]
        if len(indices_to_keep) == 0:
            continue

        this_example_dict = input_examples.subset_examples(
            example_dict=this_example_dict, indices_to_keep=indices_to_keep)

        include_soundings = (
            input_examples.SOUNDING_MATRIX_KEY in this_example_dict
        )

        if include_soundings:
            pressure_index = this_example_dict[
                input_examples.SOUNDING_FIELDS_KEY
            ].index(soundings.PRESSURE_NAME)

            this_pressure_matrix_pascals = this_example_dict[
                input_examples.SOUNDING_MATRIX_KEY][..., pressure_index]

            this_sounding_matrix = this_example_dict[
                input_examples.SOUNDING_MATRIX_KEY]

            if soundings.PRESSURE_NAME not in sounding_field_names:
                this_sounding_matrix = numpy.delete(
                    this_sounding_matrix, pressure_index, axis=-1)
        else:
            this_sounding_matrix = None
            this_pressure_matrix_pascals = None

        this_az_shear_matrix_s01 = this_example_dict[
            input_examples.AZ_SHEAR_IMAGE_MATRIX_KEY]

        if upsample_refl:
            this_refl_matrix_dbz = trainval_io.upsample_reflectivity(
                this_example_dict[input_examples.REFL_IMAGE_MATRIX_KEY][..., 0]
            )

            this_radar_matrix = numpy.concatenate(
                (this_refl_matrix_dbz, this_az_shear_matrix_s01), axis=-1
            )
        else:
            this_radar_matrix = None
            this_refl_matrix_dbz = this_example_dict[
                input_examples.REFL_IMAGE_MATRIX_KEY]

        if target_values is None:
            target_values = (
                this_example_dict[input_examples.TARGET_VALUES_KEY] + 0
            )

            if upsample_refl:
                radar_image_matrix = this_radar_matrix + 0.
            else:
                reflectivity_image_matrix_dbz = this_refl_matrix_dbz + 0.
                az_shear_image_matrix_s01 = this_az_shear_matrix_s01 + 0.

            if include_soundings:
                sounding_matrix = this_sounding_matrix + 0.
                sounding_pressure_matrix_pascals = (
                    this_pressure_matrix_pascals + 0.
                )
        else:
            target_values = numpy.concatenate((
                target_values,
                this_example_dict[input_examples.TARGET_VALUES_KEY]
            ))

            if upsample_refl:
                radar_image_matrix = numpy.concatenate(
                    (radar_image_matrix, this_radar_matrix), axis=0
                )
            else:
                reflectivity_image_matrix_dbz = numpy.concatenate(
                    (reflectivity_image_matrix_dbz, this_refl_matrix_dbz),
                    axis=0
                )
                az_shear_image_matrix_s01 = numpy.concatenate(
                    (az_shear_image_matrix_s01, this_az_shear_matrix_s01),
                    axis=0
                )

            if include_soundings:
                sounding_matrix = numpy.concatenate(
                    (sounding_matrix, this_sounding_matrix), axis=0
                )

                sounding_pressure_matrix_pascals = numpy.concatenate(
                    (sounding_pressure_matrix_pascals,
                     this_pressure_matrix_pascals),
                    axis=0
                )

        if normalization_type_string is not None:
            if upsample_refl:
                radar_image_matrix = dl_utils.normalize_radar_images(
                    radar_image_matrix=radar_image_matrix,
                    field_names=radar_field_names,
                    normalization_type_string=normalization_type_string,
                    normalization_param_file_name=normalization_file_name,
                    min_normalized_value=min_normalized_value,
                    max_normalized_value=max_normalized_value
                ).astype('float32')
            else:
                reflectivity_image_matrix_dbz = dl_utils.normalize_radar_images(
                    radar_image_matrix=reflectivity_image_matrix_dbz,
                    field_names=[radar_utils.REFL_NAME],
                    normalization_type_string=normalization_type_string,
                    normalization_param_file_name=normalization_file_name,
                    min_normalized_value=min_normalized_value,
                    max_normalized_value=max_normalized_value
                ).astype('float32')

                az_shear_image_matrix_s01 = dl_utils.normalize_radar_images(
                    radar_image_matrix=az_shear_image_matrix_s01,
                    field_names=azimuthal_shear_field_names,
                    normalization_type_string=normalization_type_string,
                    normalization_param_file_name=normalization_file_name,
                    min_normalized_value=min_normalized_value,
                    max_normalized_value=max_normalized_value
                ).astype('float32')

            if include_soundings:
                sounding_matrix = dl_utils.normalize_soundings(
                    sounding_matrix=sounding_matrix,
                    field_names=sounding_field_names,
                    normalization_type_string=normalization_type_string,
                    normalization_param_file_name=normalization_file_name,
                    min_normalized_value=min_normalized_value,
                    max_normalized_value=max_normalized_value).astype('float32')

        if upsample_refl:
            list_of_predictor_matrices = [radar_image_matrix]
        else:
            list_of_predictor_matrices = [
                reflectivity_image_matrix_dbz, az_shear_image_matrix_s01
            ]

        if include_soundings:
            list_of_predictor_matrices.append(sounding_matrix)

        target_array = _finalize_targets(
            target_values=target_values, binarize_target=binarize_target,
            num_classes=num_classes)

        storm_object_dict = {
            INPUT_MATRICES_KEY: list_of_predictor_matrices,
            TARGET_ARRAY_KEY: target_array,
            FULL_IDS_KEY: this_example_dict[input_examples.FULL_IDS_KEY],
            STORM_TIMES_KEY: this_example_dict[input_examples.STORM_TIMES_KEY],
            SOUNDING_PRESSURES_KEY:
                copy.deepcopy(sounding_pressure_matrix_pascals)
        }

        radar_image_matrix = None
        reflectivity_image_matrix_dbz = None
        az_shear_image_matrix_s01 = None
        sounding_matrix = None
        target_values = None
        sounding_pressure_matrix_pascals = None

        yield storm_object_dict


def sounding_generator(option_dict, num_examples_total):
    """Generates examples with soundings only.

    Each example (storm object) consists of the following:

    - Near-storm sounding
    - Target value (class)

    :param option_dict: Dictionary with the following keys.
    option_dict['example_file_names']: See doc for
        `training_validation_io.myrorss_generator_2d3d`.
    option_dict['first_storm_time_unix_sec']: Same.
    option_dict['last_storm_time_unix_sec']: Same.
    option_dict['sounding_field_names']: Same.
    option_dict['sounding_heights_m_agl']: Same.
    option_dict['normalization_type_string']: Same.
    option_dict['normalization_param_file_name']: Same.
    option_dict['min_normalized_value']: Same.
    option_dict['max_normalized_value']: Same.
    option_dict['target_name']: Same.
    option_dict['binarize_target']: Same.
    option_dict['class_to_sampling_fraction_dict']: Same.

    :param num_examples_total: Total number of examples to generate.

    :return: storm_object_dict: Dictionary with the following keys.
    storm_object_dict['sounding_matrix']: numpy array (E x H_s x F_s) of near-
        storm soundings.
    storm_object_dict['full_storm_id_strings']: See doc for
        `myrorss_generator_2d3d`.
    storm_object_dict['storm_times_unix_sec']: Same.
    storm_object_dict['target_array']: Same.
    storm_object_dict['sounding_pressure_matrix_pascals']: Same.
    """

    full_id_strings, storm_times_unix_sec = _find_examples_to_read(
        option_dict=option_dict, num_examples_total=num_examples_total)
    print('\n')

    example_file_names = option_dict[trainval_io.EXAMPLE_FILES_KEY]
    first_storm_time_unix_sec = option_dict[trainval_io.FIRST_STORM_TIME_KEY]
    last_storm_time_unix_sec = option_dict[trainval_io.LAST_STORM_TIME_KEY]

    sounding_field_names = option_dict[trainval_io.SOUNDING_FIELDS_KEY]
    sounding_heights_m_agl = option_dict[trainval_io.SOUNDING_HEIGHTS_KEY]
    normalization_type_string = option_dict[trainval_io.NORMALIZATION_TYPE_KEY]

    if normalization_type_string is not None:
        normalization_param_file_name = option_dict[
            trainval_io.NORMALIZATION_FILE_KEY
        ]
        min_normalized_value = option_dict[trainval_io.MIN_NORMALIZED_VALUE_KEY]
        max_normalized_value = option_dict[trainval_io.MAX_NORMALIZED_VALUE_KEY]

    target_name = option_dict[trainval_io.TARGET_NAME_KEY]
    binarize_target = option_dict[trainval_io.BINARIZE_TARGET_KEY]

    num_classes = target_val_utils.target_name_to_num_classes(
        target_name=target_name, include_dead_storms=False)

    if soundings.PRESSURE_NAME in sounding_field_names:
        sounding_field_names_to_read = sounding_field_names
    else:
        sounding_field_names_to_read = (
            sounding_field_names + [soundings.PRESSURE_NAME]
        )

    sounding_matrix = None
    target_values = None
    sounding_pressure_matrix_pascals = None
    file_index = 0

    this_example_dict = input_examples.read_example_file(
        netcdf_file_name=example_file_names[0], read_all_target_vars=False,
        target_name=target_name, metadata_only=True)

    dummy_radar_field_names = [
        this_example_dict[input_examples.RADAR_FIELDS_KEY][0]
    ]
    dummy_radar_heights_m_agl = numpy.array(
        [this_example_dict[input_examples.RADAR_HEIGHTS_KEY][0]], dtype=int
    )

    while True:
        if file_index >= len(example_file_names):
            raise StopIteration

        print('Reading data from: "{0:s}"...'.format(
            example_file_names[file_index]
        ))

        this_example_dict = input_examples.read_example_file(
            netcdf_file_name=example_file_names[file_index],
            read_all_target_vars=False, target_name=target_name,
            include_soundings=True,
            radar_field_names_to_keep=dummy_radar_field_names,
            radar_heights_to_keep_m_agl=dummy_radar_heights_m_agl,
            sounding_field_names_to_keep=sounding_field_names_to_read,
            sounding_heights_to_keep_m_agl=sounding_heights_m_agl,
            first_time_to_keep_unix_sec=first_storm_time_unix_sec,
            last_time_to_keep_unix_sec=last_storm_time_unix_sec)

        file_index += 1
        if this_example_dict is None:
            continue

        indices_to_keep = tracking_utils.find_storm_objects(
            all_id_strings=this_example_dict[input_examples.FULL_IDS_KEY],
            all_times_unix_sec=this_example_dict[
                input_examples.STORM_TIMES_KEY],
            id_strings_to_keep=full_id_strings,
            times_to_keep_unix_sec=storm_times_unix_sec, allow_missing=True)

        indices_to_keep = indices_to_keep[indices_to_keep >= 0]
        if len(indices_to_keep) == 0:
            continue

        this_example_dict = input_examples.subset_examples(
            example_dict=this_example_dict, indices_to_keep=indices_to_keep)

        pressure_index = this_example_dict[
            input_examples.SOUNDING_FIELDS_KEY
        ].index(soundings.PRESSURE_NAME)

        this_pressure_matrix_pascals = this_example_dict[
            input_examples.SOUNDING_MATRIX_KEY][..., pressure_index]

        this_sounding_matrix = this_example_dict[
            input_examples.SOUNDING_MATRIX_KEY]

        if soundings.PRESSURE_NAME not in sounding_field_names:
            this_sounding_matrix = numpy.delete(
                this_sounding_matrix, pressure_index, axis=-1)

        if sounding_matrix is None:
            sounding_matrix = this_sounding_matrix + 0.
            sounding_pressure_matrix_pascals = this_pressure_matrix_pascals + 0.
            target_values = (
                this_example_dict[input_examples.TARGET_VALUES_KEY] + 0
            )
        else:
            sounding_matrix = numpy.concatenate(
                (sounding_matrix, this_sounding_matrix), axis=0
            )

            sounding_pressure_matrix_pascals = numpy.concatenate(
                (sounding_pressure_matrix_pascals,
                 this_pressure_matrix_pascals), axis=0
            )

            target_values = numpy.concatenate((
                target_values,
                this_example_dict[input_examples.TARGET_VALUES_KEY]
            ))

        if normalization_type_string is not None:
            sounding_matrix = dl_utils.normalize_soundings(
                sounding_matrix=sounding_matrix,
                field_names=sounding_field_names,
                normalization_type_string=normalization_type_string,
                normalization_param_file_name=normalization_param_file_name,
                min_normalized_value=min_normalized_value,
                max_normalized_value=max_normalized_value
            ).astype('float32')

        target_array = _finalize_targets(
            target_values=target_values, binarize_target=binarize_target,
            num_classes=num_classes)

        storm_object_dict = {
            SOUNDING_MATRIX_KEY: sounding_matrix,
            TARGET_ARRAY_KEY: target_array,
            FULL_IDS_KEY: this_example_dict[input_examples.FULL_IDS_KEY],
            STORM_TIMES_KEY: this_example_dict[input_examples.STORM_TIMES_KEY],
            SOUNDING_PRESSURES_KEY: sounding_pressure_matrix_pascals + 0.
        }

        sounding_matrix = None
        target_values = None
        sounding_pressure_matrix_pascals = None

        yield storm_object_dict


def gridrad_generator_2d_reduced(option_dict, list_of_operation_dicts,
                                 num_examples_total):
    """Generates examples with 2-D GridRad images.

    These 2-D images are produced by applying layer operations to the native 3-D
    images.  The layer operations are specified by `list_of_operation_dicts`.

    Each example (storm object) consists of the following:

    - Storm-centered radar images (one 2-D image for each layer operation)
    - Storm-centered sounding (optional)
    - Target value (class)

    :param option_dict: Dictionary with the following keys.
    option_dict['example_file_names']: See doc for
        `training_validation_io.gridrad_generator_2d_reduced`.
    option_dict['binarize_target']: Same.
    option_dict['sounding_field_names']: Same.
    option_dict['sounding_heights_m_agl']: Same.
    option_dict['first_storm_time_unix_sec']: Same.
    option_dict['last_storm_time_unix_sec']: Same.
    option_dict['num_grid_rows']: Same.
    option_dict['num_grid_columns']: Same.
    option_dict['normalization_type_string']: Same.
    option_dict['normalization_param_file_name']: Same.
    option_dict['min_normalized_value']: Same.
    option_dict['max_normalized_value']: Same.
    option_dict['class_to_sampling_fraction_dict']: Same.

    :param list_of_operation_dicts: See doc for
        `input_examples.reduce_examples_3d_to_2d`.
    :param num_examples_total: Number of examples to generate.

    :return: storm_object_dict: Dictionary with the following keys.
    storm_object_dict['list_of_input_matrices']: length-T list of numpy arrays,
        where T = number of input tensors to model.  The first axis of each
        array has length E.
    storm_object_dict['full_storm_id_strings']: length-E list of full storm IDs.
    storm_object_dict['storm_times_unix_sec']: length-E numpy array of storm
        times.
    storm_object_dict['target_array']: See output doc for
        `training_validation_io.gridrad_generator_2d_reduced`.
    storm_object_dict['sounding_pressure_matrix_pascals']: numpy array (E x H_s)
        of pressures.  If soundings were not read, this is None.
    storm_object_dict['radar_field_names']: length-C list of field names, where
        the [j]th item corresponds to the [j]th channel of the 2-D radar images
        returned in "list_of_input_matrices".
    storm_object_dict['min_radar_heights_m_agl']: length-C numpy array with
        minimum height for each layer operation (used to reduce 3-D radar images
        to 2-D).
    storm_object_dict['max_radar_heights_m_agl']: Same but with max heights.
    storm_object_dict['radar_layer_operation_names']: length-C list with names
        of layer operations.  Each name must be accepted by
        `input_examples._check_layer_operation`.
    """

    unique_radar_field_names, unique_radar_heights_m_agl = (
        trainval_io.layer_ops_to_field_height_pairs(list_of_operation_dicts)
    )

    option_dict[trainval_io.RADAR_FIELDS_KEY] = unique_radar_field_names
    option_dict[trainval_io.RADAR_HEIGHTS_KEY] = unique_radar_heights_m_agl

    full_id_strings, storm_times_unix_sec = _find_examples_to_read(
        option_dict=option_dict, num_examples_total=num_examples_total)
    print('\n')

    example_file_names = option_dict[trainval_io.EXAMPLE_FILES_KEY]
    target_name = option_dict[trainval_io.TARGET_NAME_KEY]
    binarize_target = option_dict[trainval_io.BINARIZE_TARGET_KEY]

    first_storm_time_unix_sec = option_dict[trainval_io.FIRST_STORM_TIME_KEY]
    last_storm_time_unix_sec = option_dict[trainval_io.LAST_STORM_TIME_KEY]
    num_grid_rows = option_dict[trainval_io.NUM_ROWS_KEY]
    num_grid_columns = option_dict[trainval_io.NUM_COLUMNS_KEY]
    sounding_field_names = option_dict[trainval_io.SOUNDING_FIELDS_KEY]
    sounding_heights_m_agl = option_dict[trainval_io.SOUNDING_HEIGHTS_KEY]

    normalization_type_string = option_dict[trainval_io.NORMALIZATION_TYPE_KEY]

    if normalization_type_string is not None:
        normalization_param_file_name = option_dict[
            trainval_io.NORMALIZATION_FILE_KEY
        ]
        min_normalized_value = option_dict[trainval_io.MIN_NORMALIZED_VALUE_KEY]
        max_normalized_value = option_dict[trainval_io.MAX_NORMALIZED_VALUE_KEY]

    num_classes = target_val_utils.target_name_to_num_classes(
        target_name=target_name, include_dead_storms=False)

    if sounding_field_names is None:
        sounding_field_names_to_read = None
    else:
        if soundings.PRESSURE_NAME in sounding_field_names:
            sounding_field_names_to_read = sounding_field_names + []
        else:
            sounding_field_names_to_read = (
                sounding_field_names + [soundings.PRESSURE_NAME]
            )

    radar_image_matrix = None
    sounding_matrix = None
    target_values = None
    sounding_pressure_matrix_pascals = None

    reduction_metadata_dict = {}
    file_index = 0

    while True:
        if file_index >= len(example_file_names):
            raise StopIteration

        print('Reading data from: "{0:s}"...'.format(
            example_file_names[file_index]
        ))

        this_example_dict = input_examples.read_example_file(
            netcdf_file_name=example_file_names[file_index],
            read_all_target_vars=False, target_name=target_name,
            include_soundings=sounding_field_names is not None,
            radar_field_names_to_keep=unique_radar_field_names,
            radar_heights_to_keep_m_agl=unique_radar_heights_m_agl,
            sounding_field_names_to_keep=sounding_field_names_to_read,
            sounding_heights_to_keep_m_agl=sounding_heights_m_agl,
            first_time_to_keep_unix_sec=first_storm_time_unix_sec,
            last_time_to_keep_unix_sec=last_storm_time_unix_sec,
            num_rows_to_keep=num_grid_rows,
            num_columns_to_keep=num_grid_columns)

        file_index += 1
        if this_example_dict is None:
            continue

        indices_to_keep = tracking_utils.find_storm_objects(
            all_id_strings=this_example_dict[input_examples.FULL_IDS_KEY],
            all_times_unix_sec=this_example_dict[
                input_examples.STORM_TIMES_KEY],
            id_strings_to_keep=full_id_strings,
            times_to_keep_unix_sec=storm_times_unix_sec, allow_missing=True)

        indices_to_keep = indices_to_keep[indices_to_keep >= 0]
        if len(indices_to_keep) == 0:
            continue

        this_example_dict = input_examples.subset_examples(
            example_dict=this_example_dict, indices_to_keep=indices_to_keep)

        this_example_dict = input_examples.reduce_examples_3d_to_2d(
            example_dict=this_example_dict,
            list_of_operation_dicts=list_of_operation_dicts)

        radar_field_names_2d = this_example_dict[
            input_examples.RADAR_FIELDS_KEY]

        for this_key in REDUCTION_METADATA_KEYS:
            reduction_metadata_dict[this_key] = this_example_dict[this_key]

        include_soundings = (
            input_examples.SOUNDING_MATRIX_KEY in this_example_dict
        )

        if include_soundings:
            pressure_index = this_example_dict[
                input_examples.SOUNDING_FIELDS_KEY
            ].index(soundings.PRESSURE_NAME)

            this_pressure_matrix_pascals = this_example_dict[
                input_examples.SOUNDING_MATRIX_KEY][..., pressure_index]

            this_sounding_matrix = this_example_dict[
                input_examples.SOUNDING_MATRIX_KEY]

            if soundings.PRESSURE_NAME not in sounding_field_names:
                this_sounding_matrix = numpy.delete(
                    this_sounding_matrix, pressure_index, axis=-1)

        if target_values is None:
            radar_image_matrix = (
                this_example_dict[input_examples.RADAR_IMAGE_MATRIX_KEY]
                + 0.
            )
            target_values = (
                this_example_dict[input_examples.TARGET_VALUES_KEY] + 0
            )

            if include_soundings:
                sounding_matrix = this_sounding_matrix + 0.
                sounding_pressure_matrix_pascals = (
                    this_pressure_matrix_pascals + 0.)
        else:
            radar_image_matrix = numpy.concatenate(
                (radar_image_matrix,
                 this_example_dict[input_examples.RADAR_IMAGE_MATRIX_KEY]),
                axis=0
            )
            target_values = numpy.concatenate((
                target_values,
                this_example_dict[input_examples.TARGET_VALUES_KEY]
            ))

            if include_soundings:
                sounding_matrix = numpy.concatenate(
                    (sounding_matrix, this_sounding_matrix), axis=0
                )
                sounding_pressure_matrix_pascals = numpy.concatenate(
                    (sounding_pressure_matrix_pascals,
                     this_pressure_matrix_pascals), axis=0
                )

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

        list_of_predictor_matrices = [radar_image_matrix]
        if include_soundings:
            list_of_predictor_matrices.append(sounding_matrix)

        target_array = _finalize_targets(
            target_values=target_values, binarize_target=binarize_target,
            num_classes=num_classes)

        storm_object_dict = {
            INPUT_MATRICES_KEY: list_of_predictor_matrices,
            TARGET_ARRAY_KEY: target_array,
            FULL_IDS_KEY: this_example_dict[input_examples.FULL_IDS_KEY],
            STORM_TIMES_KEY: this_example_dict[input_examples.STORM_TIMES_KEY],
            SOUNDING_PRESSURES_KEY:
                copy.deepcopy(sounding_pressure_matrix_pascals)
        }

        for this_key in REDUCTION_METADATA_KEYS:
            storm_object_dict[this_key] = reduction_metadata_dict[this_key]

        radar_image_matrix = None
        sounding_matrix = None
        target_values = None
        sounding_pressure_matrix_pascals = None

        yield storm_object_dict


def read_specific_examples(
        top_example_dir_name, desired_full_id_strings, desired_times_unix_sec,
        option_dict, list_of_layer_operation_dicts=None):
    """Reads predictors for specific examples (storm objects).

    E = number of desired examples

    :param top_example_dir_name: Name of top-level directory with pre-processed
        examples.  Files therein will be found by
        `input_examples.find_example_file`.
    :param desired_full_id_strings: length-E list of full storm IDs.
    :param desired_times_unix_sec: length-E numpy array of storm times.
    :param option_dict: See doc for any generator in this file.
    :param list_of_layer_operation_dicts: See doc for
        `gridrad_generator_2d_reduced`.  If you do not want to reduce radar
        images from 3-D to 2-D, leave this as None.
    :return: list_of_predictor_matrices: length-T list of numpy arrays, where
        T = number of input tensors to model.  The first dimension of each numpy
        array has length E.
    :return: sounding_pressure_matrix_pascals: numpy array (E x H_s) of
        pressures.  If soundings were not read, this is None.
    """

    option_dict[trainval_io.SAMPLING_FRACTIONS_KEY] = None

    desired_spc_date_strings = [
        time_conversion.time_to_spc_date_string(t)
        for t in desired_times_unix_sec
    ]
    unique_spc_date_strings = numpy.unique(
        numpy.array(desired_spc_date_strings)
    ).tolist()

    myrorss_2d3d = None

    full_id_strings = []
    storm_times_unix_sec = numpy.array([], dtype=int)
    list_of_predictor_matrices = None
    sounding_pressure_matrix_pascals = None

    for this_spc_date_string in unique_spc_date_strings:
        this_start_time_unix_sec = time_conversion.get_start_of_spc_date(
            this_spc_date_string)
        this_end_time_unix_sec = time_conversion.get_end_of_spc_date(
            this_spc_date_string)

        this_example_file_name = input_examples.find_example_file(
            top_directory_name=top_example_dir_name, shuffled=False,
            spc_date_string=this_spc_date_string)

        option_dict[trainval_io.EXAMPLE_FILES_KEY] = [this_example_file_name]
        option_dict[trainval_io.FIRST_STORM_TIME_KEY] = this_start_time_unix_sec
        option_dict[trainval_io.LAST_STORM_TIME_KEY] = this_end_time_unix_sec

        if myrorss_2d3d is None:
            netcdf_dataset = netCDF4.Dataset(this_example_file_name)
            myrorss_2d3d = (
                input_examples.REFL_IMAGE_MATRIX_KEY in netcdf_dataset.variables
            )
            netcdf_dataset.close()

        if list_of_layer_operation_dicts is not None:
            this_generator = gridrad_generator_2d_reduced(
                option_dict=option_dict,
                list_of_operation_dicts=list_of_layer_operation_dicts,
                num_examples_total=LARGE_INTEGER)
        elif myrorss_2d3d:
            this_generator = myrorss_generator_2d3d(
                option_dict=option_dict, num_examples_total=LARGE_INTEGER)
        else:
            this_generator = generator_2d_or_3d(
                option_dict=option_dict, num_examples_total=LARGE_INTEGER)

        this_storm_object_dict = next(this_generator)

        these_desired_indices = numpy.where(numpy.logical_and(
            desired_times_unix_sec >= this_start_time_unix_sec,
            desired_times_unix_sec <= this_end_time_unix_sec
        ))[0]

        these_indices = tracking_utils.find_storm_objects(
            all_id_strings=this_storm_object_dict[FULL_IDS_KEY],
            all_times_unix_sec=this_storm_object_dict[STORM_TIMES_KEY],
            id_strings_to_keep=
            [desired_full_id_strings[k] for k in these_desired_indices],
            times_to_keep_unix_sec=
            desired_times_unix_sec[these_desired_indices],
            allow_missing=False
        )

        full_id_strings += [
            this_storm_object_dict[FULL_IDS_KEY][k] for k in these_indices
        ]
        storm_times_unix_sec = numpy.concatenate((
            storm_times_unix_sec,
            this_storm_object_dict[STORM_TIMES_KEY][these_indices]
        ))

        this_pressure_matrix_pascals = this_storm_object_dict[
            SOUNDING_PRESSURES_KEY]

        if this_pressure_matrix_pascals is not None:
            this_pressure_matrix_pascals = this_pressure_matrix_pascals[
                these_indices, ...]

            if sounding_pressure_matrix_pascals is None:
                sounding_pressure_matrix_pascals = (
                    this_pressure_matrix_pascals + 0.
                )
            else:
                sounding_pressure_matrix_pascals = numpy.concatenate(
                    (sounding_pressure_matrix_pascals,
                     this_pressure_matrix_pascals),
                    axis=0
                )

        if list_of_predictor_matrices is None:
            num_matrices = len(this_storm_object_dict[INPUT_MATRICES_KEY])
            list_of_predictor_matrices = [None] * num_matrices

        for k in range(len(list_of_predictor_matrices)):
            this_new_matrix = this_storm_object_dict[INPUT_MATRICES_KEY][k][
                these_indices, ...]

            if list_of_predictor_matrices[k] is None:
                list_of_predictor_matrices[k] = this_new_matrix + 0.
            else:
                list_of_predictor_matrices[k] = numpy.concatenate(
                    (list_of_predictor_matrices[k], this_new_matrix), axis=0
                )

            if numpy.any(numpy.isnan(list_of_predictor_matrices[k])):
                print(list_of_predictor_matrices[k].shape)
                print(list_of_predictor_matrices[k][96, ...])

                # print(numpy.where(numpy.isnan(list_of_predictor_matrices[k])))
                print('\n')

                print(this_new_matrix.shape)
                print(this_new_matrix[0, ...])
                # print(numpy.any(numpy.isnan(this_new_matrix)))
                print('\n\n*********\n\n')

    sort_indices = tracking_utils.find_storm_objects(
        all_id_strings=full_id_strings, all_times_unix_sec=storm_times_unix_sec,
        id_strings_to_keep=desired_full_id_strings,
        times_to_keep_unix_sec=desired_times_unix_sec, allow_missing=False)

    for k in range(len(list_of_predictor_matrices)):
        list_of_predictor_matrices[k] = list_of_predictor_matrices[k][
            sort_indices, ...]

    if sounding_pressure_matrix_pascals is not None:
        sounding_pressure_matrix_pascals = sounding_pressure_matrix_pascals[
            sort_indices, ...]

    return list_of_predictor_matrices, sounding_pressure_matrix_pascals
