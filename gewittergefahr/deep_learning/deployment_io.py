"""IO methods for testing and deployment of a deep-learning model.

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

import os.path
import numpy
from gewittergefahr.deep_learning import storm_images
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import labels
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import error_checking

STORM_IDS_KEY = 'storm_ids'
STORM_TIMES_KEY = 'storm_times_unix_sec'
RADAR_IMAGE_MATRIX_KEY = 'radar_image_matrix'
REFLECTIVITY_MATRIX_KEY = 'reflectivity_image_matrix_dbz'
AZ_SHEAR_MATRIX_KEY = 'azimuthal_shear_image_matrix_s01'
SOUNDING_MATRIX_KEY = 'sounding_matrix'
TARGET_VALUES_KEY = 'target_values'


def _randomly_subset_radar_images(radar_image_dict, num_examples_to_keep):
    """Randomly subsets radar images.

    :param radar_image_dict: Dictionary created by
        `storm_images.read_storm_images` or
        `storm_images.read_storm_images_and_labels`.
    :param num_examples_to_keep: Will keep radar images for this many examples
        (storm objects).
    :return: radar_image_dict: Same as input, but may contain fewer storm
        objects.
    """

    num_examples_total = len(radar_image_dict[storm_images.STORM_IDS_KEY])
    if num_examples_total <= num_examples_to_keep:
        return radar_image_dict

    example_to_keep_indices = numpy.linspace(
        0, num_examples_to_keep - 1, num=num_examples_to_keep, dtype=int)
    example_to_keep_indices = numpy.random.choice(
        example_to_keep_indices, size=num_examples_to_keep, replace=False)

    radar_image_dict[storm_images.STORM_IMAGE_MATRIX_KEY] = radar_image_dict[
        storm_images.STORM_IMAGE_MATRIX_KEY][example_to_keep_indices, ...]
    radar_image_dict[storm_images.STORM_IDS_KEY] = [
        radar_image_dict[storm_images.STORM_IDS_KEY][i]
        for i in example_to_keep_indices]
    radar_image_dict[storm_images.VALID_TIMES_KEY] = radar_image_dict[
        storm_images.VALID_TIMES_KEY][example_to_keep_indices]

    if storm_images.LABEL_VALUES_KEY in radar_image_dict:
        radar_image_dict[storm_images.LABEL_VALUES_KEY] = radar_image_dict[
            storm_images.LABEL_VALUES_KEY][example_to_keep_indices]

    return radar_image_dict


def create_storm_images_2d(
        radar_file_name_matrix, num_examples_per_file, num_rows_to_keep=None,
        num_columns_to_keep=None, normalization_type_string=None,
        min_normalized_value=dl_utils.DEFAULT_MIN_NORMALIZED_VALUE,
        max_normalized_value=dl_utils.DEFAULT_MAX_NORMALIZED_VALUE,
        normalization_param_file_name=None, return_target=True,
        target_name=None, binarize_target=False, top_target_directory_name=None,
        sounding_field_names=None, top_sounding_dir_name=None,
        sounding_lag_time_for_convective_contamination_sec=None):
    """Creates examples with 2-D radar images.

    Each example corresponds to one storm object and consists of the following.

    - Radar images: one storm-centered image for each field/height pair
    - Sounding [optional]: one storm-centered sounding
    - Target value [optional]: integer representing the wind-speed or tornado
      class

    :param radar_file_name_matrix: T-by-C numpy array of paths to radar files.
        Should be created by `training_validation_io.find_radar_files_2d`.
    :param num_examples_per_file: Number of examples (storm objects) per file.
    :param num_rows_to_keep: Number of rows to keep from each storm-centered
        radar image.  If less than total number of rows, images will be cropped
        around the center.
    :param num_columns_to_keep: Number of columns to keep from each storm-
        centered radar image.
    :param normalization_type_string: Normalization type (must be accepted by
        `dl_utils._check_normalization_type`).  If you don't want to normalize,
        leave this as None.
    :param min_normalized_value: See doc for `dl_utils.normalize_radar_images`
        or `dl_utils.normalize_soundings`.
    :param max_normalized_value: Same.
    :param normalization_param_file_name: Same.
    :param return_target: Boolean flag.  If True, will return target values.
    :param target_name: [used only if return_target = True]
        Name of target variable.
    :param binarize_target: [used only if return_target = True]
        Boolean flag.  If True, will binarize target variable, so that the
        highest class becomes 1 and all other classes become 0.
    :param top_target_directory_name: [used only if return_target = True]
        Name of top-level directory with target values (storm-hazard labels).
        Files within this directory should be findable by
        `labels.find_label_file`.
    :param sounding_field_names: list (length F_s) with names of sounding
        fields.  Each must be accepted by `soundings.check_field_name`.
    :param top_sounding_dir_name: See doc for
        `training_validation_io.find_sounding_files`.
    :param sounding_lag_time_for_convective_contamination_sec: Same.
    :return: example_dict: Dictionary with the following keys.
    example_dict['radar_image_matrix']: E-by-M-by-N-by-C numpy array of
        storm-centered radar images.
    example_dict['sounding_matrix']: numpy array (E x H_s x F_s) of
        storm-centered soundings.  If `sounding_field_names is None`, this is
        `None`.
    example_dict['target_values']: length-E numpy array of target values.  If
        target_values[i] = k, the [i]th storm object belongs to the [k]th class.
        If `return_target = False`, this is `None`.
    """

    error_checking.assert_is_boolean(return_target)
    trainval_io.check_input_args(
        num_examples_per_batch=100, num_examples_per_file=num_examples_per_file,
        radar_file_name_matrix=radar_file_name_matrix, num_radar_dimensions=2,
        binarize_target=binarize_target,
        sounding_field_names=sounding_field_names)

    if sounding_field_names is None:
        sounding_file_names = None
    else:
        sounding_file_names = trainval_io.find_sounding_files(
            top_sounding_dir_name=top_sounding_dir_name,
            radar_file_name_matrix=radar_file_name_matrix,
            target_name=target_name,
            lag_time_for_convective_contamination_sec=
            sounding_lag_time_for_convective_contamination_sec)

    num_channels = radar_file_name_matrix.shape[1]
    field_name_by_channel = [''] * num_channels
    for j in range(num_channels):
        field_name_by_channel[j] = storm_images.image_file_name_to_field(
            radar_file_name_matrix[0, j])

    storm_ids = []
    storm_times_unix_sec = numpy.array([], dtype=int)
    radar_image_matrix = None
    sounding_matrix = None
    target_values = numpy.array([], dtype=int)

    num_file_times = radar_file_name_matrix.shape[0]
    for i in range(num_file_times):
        if return_target:
            label_file_name = storm_images.find_storm_label_file(
                storm_image_file_name=radar_file_name_matrix[i, 0],
                top_label_dir_name=top_target_directory_name,
                label_name=target_name, raise_error_if_missing=False,
                warn_if_missing=True)
            if not os.path.isfile(label_file_name):
                continue

            print 'Reading data from: "{0:s}" and "{1:s}"...'.format(
                radar_file_name_matrix[i, 0], label_file_name)
            this_radar_image_dict = storm_images.read_storm_images_and_labels(
                image_file_name=radar_file_name_matrix[i, 0],
                label_file_name=label_file_name, label_name=target_name,
                num_rows_to_keep=num_rows_to_keep,
                num_columns_to_keep=num_columns_to_keep)
            if not len(this_radar_image_dict[storm_images.STORM_IDS_KEY]):
                continue

            this_radar_image_dict = (
                trainval_io.remove_storms_with_undefined_target(
                    this_radar_image_dict))
            if not len(this_radar_image_dict[storm_images.STORM_IDS_KEY]):
                continue
        else:
            print 'Reading data from: "{0:s}"...'.format(
                radar_file_name_matrix[i, 0])
            this_radar_image_dict = storm_images.read_storm_images(
                netcdf_file_name=radar_file_name_matrix[i, 0],
                num_rows_to_keep=num_rows_to_keep,
                num_columns_to_keep=num_columns_to_keep)

        this_radar_image_dict = _randomly_subset_radar_images(
            radar_image_dict=this_radar_image_dict,
            num_examples_to_keep=num_examples_per_file)

        if sounding_file_names is not None:
            (this_sounding_dict, this_radar_image_dict
            ) = trainval_io.read_soundings(
                sounding_file_name=sounding_file_names[i],
                sounding_field_names=sounding_field_names,
                radar_image_dict=this_radar_image_dict)

            if (this_sounding_dict is None or
                    not len(this_sounding_dict[soundings.STORM_IDS_KEY])):
                continue

            this_sounding_matrix = this_sounding_dict[
                soundings.SOUNDING_MATRIX_KEY]
            sounding_field_names = this_sounding_dict[soundings.FIELD_NAMES_KEY]

            if sounding_matrix is None:
                sounding_matrix = this_sounding_matrix + 0.
            else:
                sounding_matrix = numpy.concatenate(
                    (sounding_matrix, this_sounding_matrix), axis=0)

        these_storm_ids = this_radar_image_dict[storm_images.STORM_IDS_KEY]
        these_storm_times_unix_sec = this_radar_image_dict[
            storm_images.VALID_TIMES_KEY]

        storm_ids += these_storm_ids
        storm_times_unix_sec = numpy.concatenate((
            storm_times_unix_sec, these_storm_times_unix_sec))
        if return_target:
            target_values = numpy.concatenate((
                target_values,
                this_radar_image_dict[storm_images.LABEL_VALUES_KEY]))

        tuple_of_image_matrices = ()
        for j in range(num_channels):
            if j != 0:
                print 'Reading data from: "{0:s}"...'.format(
                    radar_file_name_matrix[i, j])
                this_radar_image_dict = storm_images.read_storm_images(
                    netcdf_file_name=radar_file_name_matrix[i, j],
                    storm_ids_to_keep=these_storm_ids,
                    valid_times_to_keep_unix_sec=these_storm_times_unix_sec,
                    num_rows_to_keep=num_rows_to_keep,
                    num_columns_to_keep=num_columns_to_keep)

            tuple_of_image_matrices += (
                this_radar_image_dict[storm_images.STORM_IMAGE_MATRIX_KEY],)

        this_radar_image_matrix = dl_utils.stack_radar_fields(
            tuple_of_image_matrices)
        if radar_image_matrix is None:
            radar_image_matrix = this_radar_image_matrix + 0.
        else:
            radar_image_matrix = numpy.concatenate(
                (radar_image_matrix, this_radar_image_matrix), axis=0)

    if radar_image_matrix is None:
        return None

    if normalization_type_string is not None:
        radar_image_matrix = dl_utils.normalize_radar_images(
            radar_image_matrix=radar_image_matrix,
            field_names=field_name_by_channel,
            normalization_type_string=normalization_type_string,
            normalization_param_file_name=normalization_param_file_name,
            min_normalized_value=min_normalized_value,
            max_normalized_value=max_normalized_value).astype('float32')

        if sounding_file_names is not None:
            dl_utils.normalize_soundings(
                sounding_matrix=sounding_matrix,
                field_names=sounding_field_names,
                normalization_type_string=normalization_type_string,
                normalization_param_file_name=normalization_param_file_name,
                min_normalized_value=min_normalized_value,
                max_normalized_value=max_normalized_value).astype('float32')

    if binarize_target:
        num_classes = labels.column_name_to_num_classes(target_name)
        target_values = (target_values == num_classes - 1).astype(int)

    return {
        STORM_IDS_KEY: storm_ids,
        STORM_TIMES_KEY: storm_times_unix_sec,
        RADAR_IMAGE_MATRIX_KEY: radar_image_matrix,
        SOUNDING_MATRIX_KEY: sounding_matrix,
        TARGET_VALUES_KEY: target_values
    }


def create_storm_images_3d(
        radar_file_name_matrix, num_examples_per_file, num_rows_to_keep=None,
        num_columns_to_keep=None, normalization_type_string=None,
        min_normalized_value=dl_utils.DEFAULT_MIN_NORMALIZED_VALUE,
        max_normalized_value=dl_utils.DEFAULT_MAX_NORMALIZED_VALUE,
        normalization_param_file_name=None, return_target=True,
        target_name=None, binarize_target=False, top_target_directory_name=None,
        refl_masking_threshold_dbz=dl_utils.DEFAULT_REFL_MASK_THRESHOLD_DBZ,
        sounding_field_names=None, top_sounding_dir_name=None,
        sounding_lag_time_for_convective_contamination_sec=None):
    """Creates examples with 3-D radar images.

    Each example corresponds to one storm object and consists of the following.

    - Radar images: one storm-centered image for each field/height pair
    - Sounding [optional]: one storm-centered sounding
    - Target value [optional]: integer representing the wind-speed or tornado
      class

    :param radar_file_name_matrix: numpy array (T x F_r x H_r) of paths to
        radar-image files.  Should be created by `find_radar_files_3d`.
    :param num_examples_per_file: See doc for `create_storm_images_2d`.
    :param num_rows_to_keep: Same.
    :param num_columns_to_keep: Same.
    :param normalization_type_string: Same.
    :param min_normalized_value: Same.
    :param max_normalized_value: Same.
    :param normalization_param_file_name: Same.
    :param return_target: Same.
    :param target_name: Same.
    :param binarize_target: Same.
    :param top_target_directory_name: Same.
    :param refl_masking_threshold_dbz: Used to mask pixels with low reflectivity
        (see doc for `deep_learning_utils.mask_low_reflectivity_pixels`).  If
        you want no masking, leave this as `None`.
    :param sounding_field_names: list (length F_s) with names of sounding
        fields.  Each must be accepted by `soundings.check_field_name`.
    :param top_sounding_dir_name: See doc for
        `training_validation_io.find_sounding_files`.
    :param sounding_lag_time_for_convective_contamination_sec: Same.
    :param example_dict: Dictionary with the following keys.
    example_dict['radar_image_matrix']: numpy array (E x M x N x H_r x F_r) of
        storm-centered radar images.
    example_dict['sounding_matrix']: See doc for `create_storm_images_2d`.
    example_dict['target_values']: Same.
    """

    error_checking.assert_is_boolean(return_target)
    if not return_target:
        binarize_target = False

    trainval_io.check_input_args(
        num_examples_per_batch=100, num_examples_per_file=num_examples_per_file,
        radar_file_name_matrix=radar_file_name_matrix, num_radar_dimensions=3,
        binarize_target=binarize_target,
        sounding_field_names=sounding_field_names)

    if sounding_field_names is None:
        sounding_file_names = None
    else:
        sounding_file_names = trainval_io.find_sounding_files(
            top_sounding_dir_name=top_sounding_dir_name,
            radar_file_name_matrix=radar_file_name_matrix,
            target_name=target_name,
            lag_time_for_convective_contamination_sec=
            sounding_lag_time_for_convective_contamination_sec)

    num_fields = radar_file_name_matrix.shape[1]
    radar_field_names = [''] * num_fields
    for j in range(num_fields):
        radar_field_names[j] = storm_images.image_file_name_to_field(
            radar_file_name_matrix[0, j, 0])

    storm_ids = []
    storm_times_unix_sec = numpy.array([], dtype=int)
    radar_image_matrix = None
    sounding_matrix = None
    target_values = numpy.array([], dtype=int)

    num_file_times = radar_file_name_matrix.shape[0]
    num_heights = radar_file_name_matrix.shape[2]

    for i in range(num_file_times):
        if return_target:
            label_file_name = storm_images.find_storm_label_file(
                storm_image_file_name=radar_file_name_matrix[i, 0, 0],
                top_label_dir_name=top_target_directory_name,
                label_name=target_name, raise_error_if_missing=False,
                warn_if_missing=True)
            if not os.path.isfile(label_file_name):
                continue

            print 'Reading data from: "{0:s}" and "{1:s}"...'.format(
                radar_file_name_matrix[i, 0, 0], label_file_name)
            this_radar_image_dict = storm_images.read_storm_images_and_labels(
                image_file_name=radar_file_name_matrix[i, 0, 0],
                label_file_name=label_file_name, label_name=target_name,
                num_rows_to_keep=num_rows_to_keep,
                num_columns_to_keep=num_columns_to_keep)
            if not len(this_radar_image_dict[storm_images.STORM_IDS_KEY]):
                continue

            this_radar_image_dict = (
                trainval_io.remove_storms_with_undefined_target(
                    this_radar_image_dict))
            if not len(this_radar_image_dict[storm_images.STORM_IDS_KEY]):
                continue
        else:
            print 'Reading data from: "{0:s}"...'.format(
                radar_file_name_matrix[i, 0, 0])
            this_radar_image_dict = storm_images.read_storm_images(
                netcdf_file_name=radar_file_name_matrix[i, 0, 0],
                num_rows_to_keep=num_rows_to_keep,
                num_columns_to_keep=num_columns_to_keep)

        this_radar_image_dict = _randomly_subset_radar_images(
            radar_image_dict=this_radar_image_dict,
            num_examples_to_keep=num_examples_per_file)

        if sounding_file_names is not None:
            (this_sounding_dict, this_radar_image_dict
            ) = trainval_io.read_soundings(
                sounding_file_name=sounding_file_names[i],
                sounding_field_names=sounding_field_names,
                radar_image_dict=this_radar_image_dict)

            if (this_sounding_dict is None or
                    not len(this_sounding_dict[soundings.STORM_IDS_KEY])):
                continue

            this_sounding_matrix = this_sounding_dict[
                soundings.SOUNDING_MATRIX_KEY]
            sounding_field_names = this_sounding_dict[soundings.FIELD_NAMES_KEY]

            if sounding_matrix is None:
                sounding_matrix = this_sounding_matrix + 0.
            else:
                sounding_matrix = numpy.concatenate(
                    (sounding_matrix, this_sounding_matrix), axis=0)

        these_storm_ids = this_radar_image_dict[storm_images.STORM_IDS_KEY]
        these_storm_times_unix_sec = this_radar_image_dict[
            storm_images.VALID_TIMES_KEY]
        storm_ids += these_storm_ids
        storm_times_unix_sec = numpy.concatenate((
            storm_times_unix_sec, these_storm_times_unix_sec))

        if return_target:
            target_values = numpy.concatenate((
                target_values,
                this_radar_image_dict[storm_images.LABEL_VALUES_KEY]))

        tuple_of_4d_image_matrices = ()
        for k in range(num_heights):
            tuple_of_3d_image_matrices = ()

            for j in range(num_fields):
                if not j == k == 0:
                    print 'Reading data from: "{0:s}"...'.format(
                        radar_file_name_matrix[i, j, k])
                    this_radar_image_dict = storm_images.read_storm_images(
                        netcdf_file_name=radar_file_name_matrix[i, j, k],
                        storm_ids_to_keep=these_storm_ids,
                        valid_times_to_keep_unix_sec=these_storm_times_unix_sec,
                        num_rows_to_keep=num_rows_to_keep,
                        num_columns_to_keep=num_columns_to_keep)

                tuple_of_3d_image_matrices += (
                    this_radar_image_dict[storm_images.STORM_IMAGE_MATRIX_KEY],)

            tuple_of_4d_image_matrices += (
                dl_utils.stack_radar_fields(tuple_of_3d_image_matrices),)

        this_radar_image_matrix = dl_utils.stack_radar_heights(
            tuple_of_4d_image_matrices)
        if radar_image_matrix is None:
            radar_image_matrix = this_radar_image_matrix + 0.
        else:
            radar_image_matrix = numpy.concatenate(
                (radar_image_matrix, this_radar_image_matrix), axis=0)

    if radar_image_matrix is None:
        return None

    if refl_masking_threshold_dbz is not None:
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

        if sounding_file_names is not None:
            dl_utils.normalize_soundings(
                sounding_matrix=sounding_matrix,
                field_names=sounding_field_names,
                normalization_type_string=normalization_type_string,
                normalization_param_file_name=normalization_param_file_name,
                min_normalized_value=min_normalized_value,
                max_normalized_value=max_normalized_value).astype('float32')

    if binarize_target:
        num_classes = labels.column_name_to_num_classes(target_name)
        target_values = (target_values == num_classes - 1).astype(int)

    return {
        STORM_IDS_KEY: storm_ids,
        STORM_TIMES_KEY: storm_times_unix_sec,
        RADAR_IMAGE_MATRIX_KEY: radar_image_matrix,
        SOUNDING_MATRIX_KEY: sounding_matrix,
        TARGET_VALUES_KEY: target_values
    }


def create_storm_images_2d3d_myrorss(
        radar_file_name_matrix, num_examples_per_file, num_rows_to_keep=None,
        num_columns_to_keep=None, normalization_type_string=None,
        min_normalized_value=dl_utils.DEFAULT_MIN_NORMALIZED_VALUE,
        max_normalized_value=dl_utils.DEFAULT_MAX_NORMALIZED_VALUE,
        normalization_param_file_name=None, return_target=True,
        target_name=None, binarize_target=False, top_target_directory_name=None,
        sounding_field_names=None, top_sounding_dir_name=None,
        sounding_lag_time_for_convective_contamination_sec=None):
    """Creates examples with 2-D and 3-D radar images.

    Each example corresponds to one storm object and consists of the following.

    - Reflectivity images: one storm-centered image
    - Azimuthal-shear images: one storm-centered image for each azimuthal-shear
      field (possible fields are low-level and mid-level azimuthal shear)
    - Sounding [optional]: one storm-centered sounding
    - Target value [optional]: integer representing the wind-speed or tornado
      class

    F_a = number of azimuthal-shear fields
    m = number of rows per reflectivity image
    n = number of columns per reflectivity image
    H_r = number of heights per reflectivity image
    M = number of rows per azimuthal-shear image
    N = number of columns per azimuthal-shear image

    :param radar_file_name_matrix: See doc for
        `training_validation_io.separate_radar_files_2d3d`.
    :param num_examples_per_file: See doc for `create_storm_images_2d`.
    :param num_rows_to_keep: Same.
    :param num_columns_to_keep: Same.
    :param normalization_type_string: Same.
    :param min_normalized_value: Same.
    :param max_normalized_value: Same.
    :param normalization_param_file_name: Same.
    :param return_target: Same.
    :param target_name: Same.
    :param binarize_target: Same.
    :param top_target_directory_name: Same.
    :param sounding_field_names: list (length F_s) with names of sounding
        fields.  Each must be accepted by `soundings.check_field_name`.
    :param top_sounding_dir_name: See doc for
        `training_validation_io.find_sounding_files`.
    :param sounding_lag_time_for_convective_contamination_sec: Same.
    :return: example_dict: Dictionary with the following keys.
    example_dict['reflectivity_image_matrix_dbz']: numpy array
        (E x m x n x H_r x 1) of storm-centered reflectivity images.
    example_dict['azimuthal_shear_image_matrix_s01']: numpy array
        (E x M x N x F_a) of storm-centered azimuthal-shear images.
    example_dict['sounding_matrix']: See doc for `create_storm_images_2d`.
    example_dict['target_values']: Same.
    """

    error_checking.assert_is_boolean(return_target)
    trainval_io.check_input_args(
        num_examples_per_batch=100, num_examples_per_file=num_examples_per_file,
        radar_file_name_matrix=radar_file_name_matrix, num_radar_dimensions=2,
        binarize_target=binarize_target,
        sounding_field_names=sounding_field_names)

    if sounding_field_names is None:
        sounding_file_names = None
    else:
        sounding_file_names = trainval_io.find_sounding_files(
            top_sounding_dir_name=top_sounding_dir_name,
            radar_file_name_matrix=radar_file_name_matrix,
            target_name=target_name,
            lag_time_for_convective_contamination_sec=
            sounding_lag_time_for_convective_contamination_sec)

    (reflectivity_file_name_matrix, az_shear_file_name_matrix
    ) = trainval_io.separate_radar_files_2d3d(
        radar_file_name_matrix=radar_file_name_matrix)

    num_azimuthal_shear_fields = az_shear_file_name_matrix.shape[1]
    azimuthal_shear_field_names = [''] * num_azimuthal_shear_fields
    for j in range(num_azimuthal_shear_fields):
        azimuthal_shear_field_names[j] = storm_images.image_file_name_to_field(
            az_shear_file_name_matrix[0, j])

    storm_ids = []
    storm_times_unix_sec = numpy.array([], dtype=int)
    reflectivity_image_matrix_dbz = None
    azimuthal_shear_image_matrix_s01 = None
    sounding_matrix = None
    target_values = numpy.array([], dtype=int)

    num_file_times = reflectivity_file_name_matrix.shape[0]
    num_reflectivity_heights = reflectivity_file_name_matrix.shape[1]

    for i in range(num_file_times):
        if return_target:
            label_file_name = storm_images.find_storm_label_file(
                storm_image_file_name=reflectivity_file_name_matrix[i, 0],
                top_label_dir_name=top_target_directory_name,
                label_name=target_name, raise_error_if_missing=False,
                warn_if_missing=True)
            if not os.path.isfile(label_file_name):
                continue

            print 'Reading data from: "{0:s}" and "{1:s}"...'.format(
                reflectivity_file_name_matrix[i, 0], label_file_name)
            this_radar_image_dict = storm_images.read_storm_images_and_labels(
                image_file_name=reflectivity_file_name_matrix[i, 0],
                label_file_name=label_file_name, label_name=target_name,
                num_rows_to_keep=num_rows_to_keep,
                num_columns_to_keep=num_columns_to_keep)
            if not len(this_radar_image_dict[storm_images.STORM_IDS_KEY]):
                continue

            this_radar_image_dict = (
                trainval_io.remove_storms_with_undefined_target(
                    this_radar_image_dict))
            if not len(this_radar_image_dict[storm_images.STORM_IDS_KEY]):
                continue
        else:
            print 'Reading data from: "{0:s}"...'.format(
                reflectivity_file_name_matrix[i, 0])
            this_radar_image_dict = storm_images.read_storm_images(
                netcdf_file_name=reflectivity_file_name_matrix[i, 0],
                num_rows_to_keep=num_rows_to_keep,
                num_columns_to_keep=num_columns_to_keep)

        this_radar_image_dict = _randomly_subset_radar_images(
            radar_image_dict=this_radar_image_dict,
            num_examples_to_keep=num_examples_per_file)

        if sounding_file_names is not None:
            (this_sounding_dict, this_radar_image_dict
            ) = trainval_io.read_soundings(
                sounding_file_name=sounding_file_names[i],
                sounding_field_names=sounding_field_names,
                radar_image_dict=this_radar_image_dict)

            if (this_sounding_dict is None or
                    not len(this_sounding_dict[soundings.STORM_IDS_KEY])):
                continue

            this_sounding_matrix = this_sounding_dict[
                soundings.SOUNDING_MATRIX_KEY]
            sounding_field_names = this_sounding_dict[soundings.FIELD_NAMES_KEY]

            if sounding_matrix is None:
                sounding_matrix = this_sounding_matrix + 0.
            else:
                sounding_matrix = numpy.concatenate(
                    (sounding_matrix, this_sounding_matrix), axis=0)

        these_storm_ids = this_radar_image_dict[storm_images.STORM_IDS_KEY]
        these_storm_times_unix_sec = this_radar_image_dict[
            storm_images.VALID_TIMES_KEY]
        storm_ids += these_storm_ids
        storm_times_unix_sec = numpy.concatenate((
            storm_times_unix_sec, these_storm_times_unix_sec))

        if return_target:
            target_values = numpy.concatenate((
                target_values,
                this_radar_image_dict[storm_images.LABEL_VALUES_KEY]))

        tuple_of_4d_refl_matrices = ()
        for k in range(num_reflectivity_heights):
            if k != 0:
                print 'Reading data from: "{0:s}"...'.format(
                    reflectivity_file_name_matrix[i, k])
                this_radar_image_dict = storm_images.read_storm_images(
                    netcdf_file_name=reflectivity_file_name_matrix[i, k],
                    storm_ids_to_keep=these_storm_ids,
                    valid_times_to_keep_unix_sec=these_storm_times_unix_sec,
                    num_rows_to_keep=num_rows_to_keep,
                    num_columns_to_keep=num_columns_to_keep)

            this_4d_matrix = dl_utils.stack_radar_fields(
                (this_radar_image_dict[storm_images.STORM_IMAGE_MATRIX_KEY],))
            tuple_of_4d_refl_matrices += (this_4d_matrix,)

        this_reflectivity_matrix_dbz = dl_utils.stack_radar_heights(
            tuple_of_4d_refl_matrices)
        if reflectivity_image_matrix_dbz is None:
            reflectivity_image_matrix_dbz = this_reflectivity_matrix_dbz + 0.
        else:
            reflectivity_image_matrix_dbz = numpy.concatenate(
                (reflectivity_image_matrix_dbz, this_reflectivity_matrix_dbz),
                axis=0)

        tuple_of_3d_az_shear_matrices = ()
        for j in range(num_azimuthal_shear_fields):
            print 'Reading data from: "{0:s}"...'.format(
                az_shear_file_name_matrix[i, j])

            this_radar_image_dict = storm_images.read_storm_images(
                netcdf_file_name=az_shear_file_name_matrix[i, j],
                storm_ids_to_keep=these_storm_ids,
                valid_times_to_keep_unix_sec=these_storm_times_unix_sec,
                num_rows_to_keep=num_rows_to_keep,
                num_columns_to_keep=num_columns_to_keep)

            tuple_of_3d_az_shear_matrices += (
                this_radar_image_dict[storm_images.STORM_IMAGE_MATRIX_KEY],)

        this_azimuthal_shear_matrix_s01 = dl_utils.stack_radar_fields(
            tuple_of_3d_az_shear_matrices)
        if azimuthal_shear_image_matrix_s01 is None:
            azimuthal_shear_image_matrix_s01 = (
                this_azimuthal_shear_matrix_s01 + 0.)
        else:
            azimuthal_shear_image_matrix_s01 = numpy.concatenate(
                (azimuthal_shear_image_matrix_s01,
                 this_azimuthal_shear_matrix_s01), axis=0)

    if reflectivity_image_matrix_dbz is None:
        return None

    if normalization_type_string is not None:
        reflectivity_image_matrix_dbz = dl_utils.normalize_radar_images(
            radar_image_matrix=reflectivity_image_matrix_dbz,
            field_names=[radar_utils.REFL_NAME],
            normalization_type_string=normalization_type_string,
            normalization_param_file_name=normalization_param_file_name,
            min_normalized_value=min_normalized_value,
            max_normalized_value=max_normalized_value).astype('float32')

        azimuthal_shear_image_matrix_s01 = dl_utils.normalize_radar_images(
            radar_image_matrix=azimuthal_shear_image_matrix_s01,
            field_names=azimuthal_shear_field_names,
            normalization_type_string=normalization_type_string,
            normalization_param_file_name=normalization_param_file_name,
            min_normalized_value=min_normalized_value,
            max_normalized_value=max_normalized_value).astype('float32')

        if sounding_file_names is not None:
            dl_utils.normalize_soundings(
                sounding_matrix=sounding_matrix,
                field_names=sounding_field_names,
                normalization_type_string=normalization_type_string,
                normalization_param_file_name=normalization_param_file_name,
                min_normalized_value=min_normalized_value,
                max_normalized_value=max_normalized_value).astype('float32')

    if binarize_target:
        num_classes = labels.column_name_to_num_classes(target_name)
        target_values = (target_values == num_classes - 1).astype(int)

    return {
        STORM_IDS_KEY: storm_ids,
        STORM_TIMES_KEY: storm_times_unix_sec,
        REFLECTIVITY_MATRIX_KEY: reflectivity_image_matrix_dbz,
        AZ_SHEAR_MATRIX_KEY: azimuthal_shear_image_matrix_s01,
        SOUNDING_MATRIX_KEY: sounding_matrix,
        TARGET_VALUES_KEY: target_values
    }
