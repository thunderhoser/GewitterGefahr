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
import keras.utils
from gewittergefahr.deep_learning import input_examples
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.gg_utils import target_val_utils
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import error_checking

INPUT_MATRICES_KEY = 'list_of_input_matrices'
TARGET_ARRAY_KEY = 'target_array'
STORM_IDS_KEY = 'storm_ids'
STORM_TIMES_KEY = 'storm_times_unix_sec'
SOUNDING_PRESSURES_KEY = 'sounding_pressure_matrix_pascals'


def _finalize_targets(target_values, binarize_target, num_classes):
    """Finalizes batch of target values.

    :param target_values: length-E numpy array of target values (integer class
        labels).
    :param binarize_target: Boolean flag.  If True, target variable will be
        binarized, so that the highest class becomes 1 and all others become 0.
    :param num_classes: Number of target classes.
    :return: target_array: See output doc for `example_generator_2d_or_3d` or
        `example_generator_2d3d_myrorss`.
    """

    target_values[target_values == target_val_utils.DEAD_STORM_INTEGER] = 0

    if binarize_target:
        target_values = (target_values == num_classes - 1).astype(int)
        num_classes_to_predict = 2
    else:
        num_classes_to_predict = num_classes + 0

    if num_classes_to_predict == 2:
        print (
            'Fraction of {0:d} target values in positive class: {1:.3f}'
        ).format(len(target_values), numpy.mean(target_values))
        return target_values

    target_matrix = keras.utils.to_categorical(
        target_values, num_classes_to_predict)

    class_fractions = numpy.mean(target_matrix, axis=0)
    print 'Fraction of {0:d} target values in each class: {1:s}\n'.format(
        len(target_values), str(class_fractions))

    return target_matrix


def _find_examples_to_read(option_dict, num_examples_total):
    """Determines which examples (storm objects) will be read by a generator.

    N = number of examples to read

    :param option_dict: See doc for `example_generator_2d_or_3d` or
        `example_generator_2d3d_myrorss`.
    :param num_examples_total: Same.
    :return: storm_ids: length-N list of storm IDs (strings).
    :return: storm_times_unix_sec: length-N numpy array of storm times.
    """

    error_checking.assert_is_integer(num_examples_total)
    error_checking.assert_is_greater(num_examples_total, 0)

    example_file_names = option_dict[trainval_io.EXAMPLE_FILES_KEY]
    radar_field_names = option_dict[trainval_io.RADAR_FIELDS_KEY]
    radar_heights_m_agl = option_dict[trainval_io.RADAR_HEIGHTS_KEY]
    first_storm_time_unix_sec = option_dict[trainval_io.FIRST_STORM_TIME_KEY]
    last_storm_time_unix_sec = option_dict[trainval_io.LAST_STORM_TIME_KEY]
    num_grid_rows = option_dict[trainval_io.NUM_ROWS_KEY]
    num_grid_columns = option_dict[trainval_io.NUM_COLUMNS_KEY]
    class_to_sampling_fraction_dict = option_dict[
        trainval_io.SAMPLING_FRACTIONS_KEY]

    storm_ids = []
    storm_times_unix_sec = numpy.array([], dtype=int)
    target_values = numpy.array([], dtype=int)

    target_name = None
    num_files = len(example_file_names)

    for i in range(num_files):
        print 'Reading target values from: "{0:s}"...'.format(
            example_file_names[i])

        this_example_dict = input_examples.read_example_file(
            netcdf_file_name=example_file_names[i], include_soundings=False,
            radar_field_names_to_keep=[radar_field_names[0]],
            radar_heights_to_keep_m_agl=radar_heights_m_agl[[0]],
            first_time_to_keep_unix_sec=first_storm_time_unix_sec,
            last_time_to_keep_unix_sec=last_storm_time_unix_sec,
            num_rows_to_keep=num_grid_rows,
            num_columns_to_keep=num_grid_columns)

        target_name = this_example_dict[input_examples.TARGET_NAME_KEY]

        storm_ids += this_example_dict[input_examples.STORM_IDS_KEY]
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

    storm_ids = [storm_ids[k] for k in indices_to_keep]
    storm_times_unix_sec = storm_times_unix_sec[indices_to_keep]
    target_values = target_values[indices_to_keep]
    num_examples_found = len(storm_ids)

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

    storm_ids = [storm_ids[k] for k in indices_to_keep]
    storm_times_unix_sec = storm_times_unix_sec[indices_to_keep]

    return storm_ids, storm_times_unix_sec


def example_generator_2d_or_3d(option_dict, num_examples_total):
    """Generates examples with either all 2-D or all 3-D radar images.

    Each example corresponds to one storm object and contains the following
    data:

    - Storm-centered radar images (either one 2-D image for each storm object
      and field/height pair, or one 3-D image for each storm object and field)
    - Storm-centered sounding (optional)
    - Target class

    :param option_dict: Dictionary with the following keys.
    option_dict['example_file_names']: See doc for
        `training_validation_io.example_generator_2d_or_3d`.
    option_dict['first_storm_time_unix_sec']: Same.
    option_dict['last_storm_time_unix_sec']: Same.
    option_dict['radar_field_names']: Same.
    option_dict['radar_heights_m_agl']: Same.
    option_dict['sounding_field_names']: Same.
    option_dict['sounding_heights_m_agl']: Same.
    option_dict['num_grid_rows']: Same.
    option_dict['num_grid_columns']: Same.
    option_dict['normalization_type_string']: Same.
    option_dict['normalization_param_file_name']: Same.
    option_dict['min_normalized_value']: Same.
    option_dict['max_normalized_value']: Same.
    option_dict['binarize_target']: Same.
    option_dict['class_to_sampling_fraction_dict']: Same.
    option_dict['refl_masking_threshold_dbz']: Same.

    :param num_examples_total: Total number of examples to generate.
    :return: storm_object_dict: Dictionary with the following keys.
    storm_object_dict['list_of_input_matrices']: length-T list of numpy arrays,
        where T = number of input tensors to the model.  The first dimension of
        each array has length E.
    storm_object_dict['storm_ids']: length-E list of storm IDs.
    storm_object_dict['storm_times_unix_sec']: length-E list of storm times.
    storm_object_dict['target_array']: If the problem is multiclass and
        `binarize_target = False`, this is an E-by-K numpy array with only zeros
        and ones (though technically the type is "float64").  If
        target_array[i, k] = 1, the [i]th example belongs to the [k]th class.

        If the problem is binary or `binarize_target = True`, this is a length-E
        numpy array of integers (0 for negative class, 1 for positive class).

    storm_object_dict['sounding_pressure_matrix_pascals']: numpy array (E x H_s)
        of pressures.  If soundings were not read, this is None.
    """

    storm_ids, storm_times_unix_sec = _find_examples_to_read(
        option_dict=option_dict, num_examples_total=num_examples_total)
    print '\n'

    example_file_names = option_dict[trainval_io.EXAMPLE_FILES_KEY]

    radar_field_names = option_dict[trainval_io.RADAR_FIELDS_KEY]
    radar_heights_m_agl = option_dict[trainval_io.RADAR_HEIGHTS_KEY]
    sounding_field_names = option_dict[trainval_io.SOUNDING_FIELDS_KEY]
    sounding_heights_m_agl = option_dict[trainval_io.SOUNDING_HEIGHTS_KEY]
    first_storm_time_unix_sec = option_dict[trainval_io.FIRST_STORM_TIME_KEY]
    last_storm_time_unix_sec = option_dict[trainval_io.LAST_STORM_TIME_KEY]
    num_grid_rows = option_dict[trainval_io.NUM_ROWS_KEY]
    num_grid_columns = option_dict[trainval_io.NUM_COLUMNS_KEY]

    binarize_target = option_dict[trainval_io.BINARIZE_TARGET_KEY]
    refl_masking_threshold_dbz = option_dict[trainval_io.REFLECTIVITY_MASK_KEY]

    normalization_type_string = option_dict[trainval_io.NORMALIZATION_TYPE_KEY]
    normalization_param_file_name = option_dict[
        trainval_io.NORMALIZATION_FILE_KEY]
    min_normalized_value = option_dict[trainval_io.MIN_NORMALIZED_VALUE_KEY]
    max_normalized_value = option_dict[trainval_io.MAX_NORMALIZED_VALUE_KEY]

    this_example_dict = input_examples.read_example_file(
        netcdf_file_name=example_file_names[0], metadata_only=True)
    target_name = this_example_dict[input_examples.TARGET_NAME_KEY]
    num_classes = target_val_utils.target_name_to_num_classes(
        target_name=target_name, include_dead_storms=False)

    if sounding_field_names is None:
        sounding_field_names_to_read = None
    else:
        if soundings.PRESSURE_NAME in sounding_field_names:
            sounding_field_names_to_read = sounding_field_names + []
        else:
            sounding_field_names_to_read = (
                sounding_field_names + [soundings.PRESSURE_NAME])

    radar_image_matrix = None
    sounding_matrix = None
    target_values = None
    sounding_pressure_matrix_pascals = None
    file_index = 0

    while True:
        if file_index >= len(example_file_names):
            raise StopIteration

        print 'Reading data from: "{0:s}"...'.format(
            example_file_names[file_index])
        this_example_dict = input_examples.read_example_file(
            netcdf_file_name=example_file_names[file_index],
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
            all_storm_ids=this_example_dict[input_examples.STORM_IDS_KEY],
            all_times_unix_sec=this_example_dict[
                input_examples.STORM_TIMES_KEY],
            storm_ids_to_keep=storm_ids,
            times_to_keep_unix_sec=storm_times_unix_sec, allow_missing=True)

        indices_to_keep = indices_to_keep[indices_to_keep >= 0]
        if len(indices_to_keep) == 0:
            continue

        this_example_dict = input_examples.subset_examples(
            example_dict=this_example_dict, indices_to_keep=indices_to_keep)

        include_soundings = (
            input_examples.SOUNDING_MATRIX_KEY in this_example_dict)

        if include_soundings:
            pressure_index = this_example_dict[
                input_examples.SOUNDING_FIELDS_KEY
            ].index(soundings.PRESSURE_NAME)

            this_pressure_matrix_pascals = this_example_dict[
                input_examples.SOUNDING_MATRIX_KEY][..., pressure_index]

            this_sounding_matrix = this_example_dict[
                input_examples.SOUNDING_MATRIX_KEY]
            if soundings.PRESSURE_NAME not in sounding_field_names:
                this_sounding_matrix = this_sounding_matrix[..., :-1]

        if target_values is None:
            radar_image_matrix = (
                this_example_dict[input_examples.RADAR_IMAGE_MATRIX_KEY]
                + 0.
            )
            target_values = (
                this_example_dict[input_examples.TARGET_VALUES_KEY] + 0)

            if include_soundings:
                sounding_matrix = this_sounding_matrix + 0.
                sounding_pressure_matrix_pascals = (
                    this_pressure_matrix_pascals + 0.)
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
                    (sounding_matrix, this_sounding_matrix), axis=0)
                sounding_pressure_matrix_pascals = numpy.concatenate(
                    (sounding_pressure_matrix_pascals,
                     this_pressure_matrix_pascals), axis=0)

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
            STORM_IDS_KEY: this_example_dict[input_examples.STORM_IDS_KEY],
            STORM_TIMES_KEY: this_example_dict[input_examples.STORM_TIMES_KEY],
            SOUNDING_PRESSURES_KEY:
                copy.deepcopy(sounding_pressure_matrix_pascals)
        }

        radar_image_matrix = None
        sounding_matrix = None
        target_values = None
        sounding_pressure_matrix_pascals = None

        yield storm_object_dict


def example_generator_2d3d_myrorss(option_dict, num_examples_total):
    """Generates examples with both 2-D and 3-D radar images.

    Each example corresponds to one storm object and contains the following
    data:

    - Storm-centered azimuthal-shear images (one 2-D image for each
      azimuthal-shear field)
    - Storm-centered reflectivity image (3-D)
    - Storm-centered sounding (optional)
    - Target class

    :param option_dict: Dictionary with the following keys.
    option_dict['example_file_names']: See doc for
        `training_validation_io.example_generator_2d3d_myrorss`.
    option_dict['first_storm_time_unix_sec']: Same.
    option_dict['last_storm_time_unix_sec']: Same.
    option_dict['radar_field_names']: Same.
    option_dict['radar_heights_m_agl']: Same.
    option_dict['sounding_field_names']: Same.
    option_dict['sounding_heights_m_agl']: Same.
    option_dict['num_grid_rows']: Same.
    option_dict['num_grid_columns']: Same.
    option_dict['normalization_type_string']: Same.
    option_dict['normalization_param_file_name']: Same.
    option_dict['min_normalized_value']: Same.
    option_dict['max_normalized_value']: Same.
    option_dict['binarize_target']: Same.
    option_dict['class_to_sampling_fraction_dict']: Same.

    :param num_examples_total: Total number of examples to generate.
    :return: storm_object_dict: Dictionary with the following keys.
    storm_object_dict['list_of_input_matrices']: length-T list of numpy arrays,
        where T = number of input tensors to the model.  The first dimension of
        each array has length E.
    storm_object_dict['storm_ids']: length-E list of storm IDs.
    storm_object_dict['storm_times_unix_sec']: length-E list of storm times.
    storm_object_dict['target_array']: See doc for `example_generator_2d_or_3d`.
    storm_object_dict['sounding_pressure_matrix_pascals']: numpy array (E x H_s)
        of pressures.  If soundings were not read, this is None.
    """

    storm_ids, storm_times_unix_sec = _find_examples_to_read(
        option_dict=option_dict, num_examples_total=num_examples_total)
    print '\n'

    example_file_names = option_dict[trainval_io.EXAMPLE_FILES_KEY]

    azimuthal_shear_field_names = option_dict[trainval_io.RADAR_FIELDS_KEY]
    reflectivity_heights_m_agl = option_dict[trainval_io.RADAR_HEIGHTS_KEY]
    sounding_field_names = option_dict[trainval_io.SOUNDING_FIELDS_KEY]
    sounding_heights_m_agl = option_dict[trainval_io.SOUNDING_HEIGHTS_KEY]
    first_storm_time_unix_sec = option_dict[trainval_io.FIRST_STORM_TIME_KEY]
    last_storm_time_unix_sec = option_dict[trainval_io.LAST_STORM_TIME_KEY]
    num_grid_rows = option_dict[trainval_io.NUM_ROWS_KEY]
    num_grid_columns = option_dict[trainval_io.NUM_COLUMNS_KEY]

    binarize_target = option_dict[trainval_io.BINARIZE_TARGET_KEY]

    normalization_type_string = option_dict[trainval_io.NORMALIZATION_TYPE_KEY]
    normalization_param_file_name = option_dict[
        trainval_io.NORMALIZATION_FILE_KEY]
    min_normalized_value = option_dict[trainval_io.MIN_NORMALIZED_VALUE_KEY]
    max_normalized_value = option_dict[trainval_io.MAX_NORMALIZED_VALUE_KEY]

    this_example_dict = input_examples.read_example_file(
        netcdf_file_name=example_file_names[0], metadata_only=True)
    target_name = this_example_dict[input_examples.TARGET_NAME_KEY]
    num_classes = target_val_utils.target_name_to_num_classes(
        target_name=target_name, include_dead_storms=False)

    if sounding_field_names is None:
        sounding_field_names_to_read = None
    else:
        if soundings.PRESSURE_NAME in sounding_field_names:
            sounding_field_names_to_read = sounding_field_names + []
        else:
            sounding_field_names_to_read = (
                sounding_field_names + [soundings.PRESSURE_NAME])

    reflectivity_image_matrix_dbz = None
    az_shear_image_matrix_s01 = None
    sounding_matrix = None
    target_values = None
    sounding_pressure_matrix_pascals = None
    file_index = 0

    while True:
        if file_index >= len(example_file_names):
            raise StopIteration

        print 'Reading data from: "{0:s}"...'.format(
            example_file_names[file_index])
        this_example_dict = input_examples.read_example_file(
            netcdf_file_name=example_file_names[file_index],
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
            all_storm_ids=this_example_dict[input_examples.STORM_IDS_KEY],
            all_times_unix_sec=this_example_dict[
                input_examples.STORM_TIMES_KEY],
            storm_ids_to_keep=storm_ids,
            times_to_keep_unix_sec=storm_times_unix_sec, allow_missing=True)

        indices_to_keep = indices_to_keep[indices_to_keep >= 0]
        if len(indices_to_keep) == 0:
            continue

        this_example_dict = input_examples.subset_examples(
            example_dict=this_example_dict, indices_to_keep=indices_to_keep)

        include_soundings = (
            input_examples.SOUNDING_MATRIX_KEY in this_example_dict)

        if include_soundings:
            pressure_index = this_example_dict[
                input_examples.SOUNDING_FIELDS_KEY
            ].index(soundings.PRESSURE_NAME)

            this_pressure_matrix_pascals = this_example_dict[
                input_examples.SOUNDING_MATRIX_KEY][..., pressure_index]

            this_sounding_matrix = this_example_dict[
                input_examples.SOUNDING_MATRIX_KEY]
            if soundings.PRESSURE_NAME not in sounding_field_names:
                this_sounding_matrix = this_sounding_matrix[..., -1]

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
                sounding_matrix = this_sounding_matrix + 0.
                sounding_pressure_matrix_pascals = (
                    this_pressure_matrix_pascals + 0.)
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
                    (sounding_matrix, this_sounding_matrix), axis=0)
                sounding_pressure_matrix_pascals = numpy.concatenate(
                    (sounding_pressure_matrix_pascals,
                     this_pressure_matrix_pascals), axis=0)

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
            STORM_IDS_KEY: this_example_dict[input_examples.STORM_IDS_KEY],
            STORM_TIMES_KEY: this_example_dict[input_examples.STORM_TIMES_KEY],
            SOUNDING_PRESSURES_KEY: sounding_pressure_matrix_pascals + 0.
        }

        reflectivity_image_matrix_dbz = None
        az_shear_image_matrix_s01 = None
        sounding_matrix = None
        target_values = None
        sounding_pressure_matrix_pascals = None

        yield storm_object_dict
