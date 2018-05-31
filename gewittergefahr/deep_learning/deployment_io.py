"""IO methods for deployment of a deep-learning model.

--- NOTATION ---

In this module, the following letters will be used to denote matrix dimensions.

K = number of classes (possible values of target variable)
E = number of examples
M = number of pixel rows per image
N = number of pixel columns per image
D = number of pixel depths per image
C = number of channels (predictor variables) per image
S = number of sounding statistics
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


def _check_input_args(
        image_file_name_matrix, num_image_dimensions, return_target,
        binarize_target, sounding_statistic_file_names=None,
        sounding_statistic_names=None):
    """Error-checks input arguments to example-creator.

    :param image_file_name_matrix: numpy array of paths to radar-image files.
        This should be created by `find_2d_input_files` or
        `find_3d_input_files`.
    :param num_image_dimensions: Number of dimensions expected in
        `image_file_name_matrix`.
    :param return_target: See documentation for `create_2d_storm_images`.
    :param binarize_target: Same.
    :param sounding_statistic_file_names: Same.
    :param sounding_statistic_names: Same.
    """

    error_checking.assert_is_integer(num_image_dimensions)
    error_checking.assert_is_geq(num_image_dimensions, 2)
    error_checking.assert_is_leq(num_image_dimensions, 3)
    error_checking.assert_is_numpy_array(
        image_file_name_matrix, num_dimensions=num_image_dimensions)

    error_checking.assert_is_boolean(return_target)
    error_checking.assert_is_boolean(binarize_target)

    if sounding_statistic_file_names is not None:
        num_init_times = image_file_name_matrix.shape[0]
        error_checking.assert_is_numpy_array(
            numpy.array(sounding_statistic_file_names),
            exact_dimensions=numpy.array([num_init_times]))

        if sounding_statistic_names is not None:
            error_checking.assert_is_numpy_array(
                numpy.array(sounding_statistic_names), num_dimensions=1)


def create_2d_storm_images(
        image_file_name_matrix, target_name, return_target=True,
        binarize_target=False, top_target_directory_name=None,
        radar_normalization_dict=dl_utils.DEFAULT_NORMALIZATION_DICT,
        sounding_statistic_file_names=None, sounding_statistic_names=None,
        sounding_stat_metadata_table=None):
    """Creates examples with 2-D radar images.

    Each example corresponds to one storm object and consists of the following.

    - Radar images: one storm-centered image for each predictor variable
      (field/height pair).
    - Sounding statistics [optional]: a 1-D array of sounding statistics.
    - Target value [optional]: predictand.

    T = number of storm times (initial times, not valid times)
    F = number of radar fields
    C = number of channels = num predictor variables = num field/height pairs

    :param image_file_name_matrix: T-by-C numpy array of paths to radar-image
        files.  This should be created by
        `training_validation_io.find_2d_input_files`.
    :param target_name: Name of target variable.
    :param return_target: Boolean flag.  If True (False), will (not) return
        target values.
    :param binarize_target: Boolean flag.  If True (False), will (not) binarize
        target values:
    :param top_target_directory_name: Name of top-level directory with target
        files.
    :param radar_normalization_dict: Used to normalize radar images (see doc for
        `deep_learning_utils.normalize_predictor_matrix`).
    :param sounding_statistic_file_names: length-T list of files with sounding
        statistics.  This should be created by
        `training_validation_io.find_sounding_statistic_files`.
    :param sounding_statistic_names: length-S list with names of sounding
        statistics.
    :param sounding_stat_metadata_table: Used to normalize sounding stats (see
        doc for `deep_learning_utils.normalize_sounding_statistics`).
    :return: image_matrix: E-by-M-by-N-by-C numpy array of storm-centered radar
        images.
    :return: sounding_stat_matrix: [may be None] E-by-S numpy array of sounding
        stats.
    :return: target_values: [may be None] length-E numpy array of target values.
        If target_values[i] = k, the [i]th example belongs to the [k]th class.
    """

    _check_input_args(
        image_file_name_matrix=image_file_name_matrix, num_image_dimensions=2,
        return_target=True, binarize_target=binarize_target,
        sounding_statistic_file_names=sounding_statistic_file_names,
        sounding_statistic_names=sounding_statistic_names)

    num_times = image_file_name_matrix.shape[0]
    num_channels = image_file_name_matrix.shape[1]
    field_name_by_channel = [''] * num_channels

    for j in range(num_channels):
        this_storm_image_dict = storm_images.read_storm_images(
            netcdf_file_name=image_file_name_matrix[0, j])
        field_name_by_channel[j] = this_storm_image_dict[
            storm_images.RADAR_FIELD_NAME_KEY]

    image_matrix = None
    sounding_stat_matrix = None
    target_values = numpy.array([], dtype=int)

    for i in range(num_times):
        if return_target:
            this_label_file_name = storm_images.find_storm_label_file(
                storm_image_file_name=image_file_name_matrix[i, 0],
                top_label_directory_name=top_target_directory_name,
                label_name=target_name, raise_error_if_missing=False)
            if not os.path.isfile(this_label_file_name):
                print (
                    'PROBLEM.  Cannot find label file.  Expected at: "{0:s}"'
                ).format(this_label_file_name)
                continue

            print (
                'Reading storm-centered radar images and hazard labels from '
                '"{0:s}" and "{1:s}"...'
            ).format(image_file_name_matrix[i, 0], this_label_file_name)

            this_storm_image_dict = storm_images.read_storm_images_and_labels(
                image_file_name=image_file_name_matrix[i, 0],
                label_file_name=this_label_file_name, label_name=target_name)
            these_storm_ids = this_storm_image_dict[storm_images.STORM_IDS_KEY]
            if not len(these_storm_ids):
                continue

            this_storm_image_dict, these_valid_storm_indices = (
                trainval_io.remove_storms_with_undef_target(
                    this_storm_image_dict))
            if not len(these_valid_storm_indices):
                continue

            these_storm_ids_to_keep = this_storm_image_dict[
                storm_images.STORM_IDS_KEY]
            these_valid_times_to_keep_unix_sec = this_storm_image_dict[
                storm_images.VALID_TIMES_KEY]
            target_values = numpy.concatenate((
                target_values,
                this_storm_image_dict[storm_images.LABEL_VALUES_KEY]))

            if binarize_target:
                num_classes = labels.column_name_to_num_classes(target_name)
                target_values = (target_values == num_classes - 1).astype(int)
        else:
            print 'Reading storm-centered radar images from "{0:s}"...'.format(
                image_file_name_matrix[i, 0])

            this_storm_image_dict = storm_images.read_storm_images(
                netcdf_file_name=image_file_name_matrix[i, 0])
            these_storm_ids_to_keep = None
            these_valid_times_to_keep_unix_sec = None

        if sounding_statistic_file_names is not None:
            print 'Reading sounding statistics from "{0:s}"...'.format(
                sounding_statistic_file_names[i])
            this_sounding_stat_dict = soundings.read_sounding_statistics(
                netcdf_file_name=sounding_statistic_file_names[i],
                storm_ids_to_keep=these_storm_ids_to_keep,
                statistic_names_to_keep=sounding_statistic_names)

            if sounding_statistic_names is None:
                sounding_statistic_names = this_sounding_stat_dict[
                    soundings.STATISTIC_NAMES_KEY]

            if sounding_stat_matrix is None:
                sounding_stat_matrix = this_sounding_stat_dict[
                    soundings.STATISTIC_MATRIX_KEY]
            else:
                sounding_stat_matrix = numpy.concatenate(
                    (sounding_stat_matrix,
                     this_sounding_stat_dict[soundings.STATISTIC_MATRIX_KEY]),
                    axis=0)

        tuple_of_image_matrices = ()

        for j in range(num_channels):
            if j != 0:
                print (
                    'Reading storm-centered radar images from "{0:s}"...'
                ).format(image_file_name_matrix[i, j])

                this_storm_image_dict = storm_images.read_storm_images(
                    netcdf_file_name=image_file_name_matrix[i, j],
                    storm_ids_to_keep=these_storm_ids_to_keep,
                    valid_times_to_keep_unix_sec=
                    these_valid_times_to_keep_unix_sec)

            tuple_of_image_matrices += (
                this_storm_image_dict[storm_images.STORM_IMAGE_MATRIX_KEY],)

        this_image_matrix = dl_utils.stack_predictor_variables(
            tuple_of_image_matrices)
        if image_matrix is None:
            image_matrix = this_image_matrix + 0.
        else:
            image_matrix = numpy.concatenate(
                (image_matrix, this_image_matrix), axis=0)

    if image_matrix is None:
        return image_matrix, sounding_stat_matrix, target_values

    image_matrix = dl_utils.normalize_predictor_matrix(
        predictor_matrix=image_matrix, normalize_by_batch=False,
        predictor_names=field_name_by_channel,
        normalization_dict=radar_normalization_dict).astype('float32')

    if sounding_statistic_file_names is not None:
        sounding_stat_matrix = dl_utils.normalize_sounding_statistics(
            sounding_stat_matrix=sounding_stat_matrix,
            statistic_names=sounding_statistic_names,
            metadata_table=sounding_stat_metadata_table,
            normalize_by_batch=False).astype('float32')

    return image_matrix, sounding_stat_matrix, target_values


def create_2d3d_storm_images(
        image_file_name_matrix, target_name, return_target=True,
        binarize_target=False, top_target_directory_name=None,
        radar_normalization_dict=dl_utils.DEFAULT_NORMALIZATION_DICT):
    """Creates examples with 2-D and 3-D radar images.

    Each example corresponds to one storm object and consists of the following.

    - Reflectivity: one 3-D storm-centered reflectivity image.
    - Azimuthal shear: one 2-D storm-centered image per variable (e.g.,
      low-level or mid-level az shear).
    - Target value [optional]: predictand.

    T = number of storm times (initial times, not valid times)
    F = number of azimuthal-shear fields
    m = number of pixel rows per reflectivity image
    n = number of pixel columns per reflectivity image
    D = number of pixel heights (depths) per reflectivity image
    M = number of pixel rows per az-shear image
    N = number of pixel columns per az-shear image

    :param image_file_name_matrix: T-by-(D + F) numpy array of paths to image
        files.  This should be created by
        `training_validation_io.find_2d_input_files`.
    :param target_name: See documentation for `create_2d_storm_images`.
    :param return_target: Same.
    :param binarize_target: Same.
    :param top_target_directory_name: Name of top-level directory with target
        files.
    :param radar_normalization_dict: Same.
    :return: reflectivity_image_matrix_dbz: E-by-m-by-n-by-D-by-1 numpy
        array of storm-centered reflectivity images.
    :return: azimuthal_shear_image_matrix_s01: E-by-M-by-N-by-F numpy
        array of storm-centered az-shear images.
    :return: target_values: [may be None] length-E numpy array of target values.
        If target_values[i] = k, the [i]th example belongs to the [k]th class.
    """

    _check_input_args(
        image_file_name_matrix=image_file_name_matrix, num_image_dimensions=2,
        return_target=True, binarize_target=binarize_target)

    reflectivity_file_name_matrix, az_shear_file_name_matrix = (
        trainval_io.separate_input_files_for_2d3d_myrorss(
            image_file_name_matrix))

    num_times = reflectivity_file_name_matrix.shape[0]
    num_reflectivity_heights = reflectivity_file_name_matrix.shape[1]
    num_az_shear_fields = az_shear_file_name_matrix.shape[1]

    az_shear_field_names = [''] * num_az_shear_fields
    for j in range(num_az_shear_fields):
        this_storm_image_dict = storm_images.read_storm_images(
            netcdf_file_name=az_shear_file_name_matrix[0, j])
        az_shear_field_names[j] = this_storm_image_dict[
            storm_images.RADAR_FIELD_NAME_KEY]

    reflectivity_image_matrix_dbz = None
    azimuthal_shear_image_matrix_s01 = None
    target_values = numpy.array([], dtype=int)

    for i in range(num_times):
        if return_target:
            this_label_file_name = storm_images.find_storm_label_file(
                storm_image_file_name=reflectivity_file_name_matrix[i, 0],
                top_label_directory_name=top_target_directory_name,
                label_name=target_name, raise_error_if_missing=False)
            if not os.path.isfile(this_label_file_name):
                print (
                    'PROBLEM.  Cannot find label file.  Expected at: "{0:s}"'
                ).format(this_label_file_name)
                continue

            print (
                'Reading storm-centered radar images and hazard labels from '
                '"{0:s}" and "{1:s}"...'
            ).format(reflectivity_file_name_matrix[i, 0], this_label_file_name)

            this_storm_image_dict = storm_images.read_storm_images_and_labels(
                image_file_name=reflectivity_file_name_matrix[i, 0],
                label_file_name=this_label_file_name, label_name=target_name)
            these_storm_ids = this_storm_image_dict[storm_images.STORM_IDS_KEY]
            if not len(these_storm_ids):
                continue

            this_storm_image_dict, these_valid_storm_indices = (
                trainval_io.remove_storms_with_undef_target(
                    this_storm_image_dict))
            if not len(these_valid_storm_indices):
                continue

            these_storm_ids_to_keep = this_storm_image_dict[
                storm_images.STORM_IDS_KEY]
            these_valid_times_to_keep_unix_sec = this_storm_image_dict[
                storm_images.VALID_TIMES_KEY]
            target_values = numpy.concatenate((
                target_values,
                this_storm_image_dict[storm_images.LABEL_VALUES_KEY]))

            if binarize_target:
                num_classes = labels.column_name_to_num_classes(target_name)
                target_values = (target_values == num_classes - 1).astype(int)
        else:
            print 'Reading storm-centered radar images from "{0:s}"...'.format(
                reflectivity_file_name_matrix[i, 0])

            this_storm_image_dict = storm_images.read_storm_images(
                netcdf_file_name=reflectivity_file_name_matrix[i, 0])
            these_storm_ids_to_keep = None
            these_valid_times_to_keep_unix_sec = None

        tuple_of_4d_refl_matrices = ()
        for k in range(num_reflectivity_heights):
            if k != 0:
                print (
                    'Reading storm-centered radar images from "{0:s}"...'
                ).format(reflectivity_file_name_matrix[i, k])

                this_storm_image_dict = storm_images.read_storm_images(
                    netcdf_file_name=reflectivity_file_name_matrix[i, k],
                    storm_ids_to_keep=these_storm_ids_to_keep,
                    valid_times_to_keep_unix_sec=
                    these_valid_times_to_keep_unix_sec)

            this_4d_matrix = dl_utils.stack_predictor_variables(
                (this_storm_image_dict[storm_images.STORM_IMAGE_MATRIX_KEY],))
            tuple_of_4d_refl_matrices += (this_4d_matrix,)

        this_refl_matrix_dbz = dl_utils.stack_heights(tuple_of_4d_refl_matrices)
        if reflectivity_image_matrix_dbz is None:
            reflectivity_image_matrix_dbz = this_refl_matrix_dbz + 0.
        else:
            reflectivity_image_matrix_dbz = numpy.concatenate(
                (reflectivity_image_matrix_dbz, this_refl_matrix_dbz), axis=0)

        tuple_of_3d_az_shear_matrices = ()
        for j in range(num_az_shear_fields):
            print (
                'Reading storm-centered radar images from "{0:s}"...'
            ).format(az_shear_file_name_matrix[i, j])

            this_storm_image_dict = storm_images.read_storm_images(
                netcdf_file_name=az_shear_file_name_matrix[i, j],
                storm_ids_to_keep=these_storm_ids_to_keep,
                valid_times_to_keep_unix_sec=
                these_valid_times_to_keep_unix_sec)

            tuple_of_3d_az_shear_matrices += (
                this_storm_image_dict[storm_images.STORM_IMAGE_MATRIX_KEY],)

        this_az_shear_matrix_s01 = dl_utils.stack_predictor_variables(
            tuple_of_3d_az_shear_matrices)
        if azimuthal_shear_image_matrix_s01 is None:
            azimuthal_shear_image_matrix_s01 = this_az_shear_matrix_s01 + 0.
        else:
            azimuthal_shear_image_matrix_s01 = numpy.concatenate(
                (azimuthal_shear_image_matrix_s01, this_az_shear_matrix_s01),
                axis=0)

    if reflectivity_image_matrix_dbz is None:
        return (reflectivity_image_matrix_dbz, azimuthal_shear_image_matrix_s01,
                target_values)

    reflectivity_image_matrix_dbz = dl_utils.normalize_predictor_matrix(
        predictor_matrix=reflectivity_image_matrix_dbz,
        normalize_by_batch=False, predictor_names=[radar_utils.REFL_NAME],
        normalization_dict=radar_normalization_dict).astype('float32')
    azimuthal_shear_image_matrix_s01 = dl_utils.normalize_predictor_matrix(
        predictor_matrix=azimuthal_shear_image_matrix_s01,
        normalize_by_batch=False, predictor_names=az_shear_field_names,
        normalization_dict=radar_normalization_dict).astype('float32')

    return (reflectivity_image_matrix_dbz, azimuthal_shear_image_matrix_s01,
            target_values)


def create_3d_storm_images(
        image_file_name_matrix, target_name, return_target=True,
        binarize_target=False, top_target_directory_name=None,
        radar_normalization_dict=dl_utils.DEFAULT_NORMALIZATION_DICT,
        sounding_statistic_file_names=None, sounding_statistic_names=None,
        sounding_stat_metadata_table=None):
    """Creates examples with 3-D radar images.

    Each example corresponds to one storm object and consists of the following.

    - Radar images: one storm-centered image for each predictor variable
      (field/height pair).
    - Sounding statistics [optional]: a 1-D array of sounding statistics.
    - Target value [optional]: predictand.

    T = number of storm times (initial times, not valid times)
    C = number of radar fields
    D = number of radar heights

    :param image_file_name_matrix: T-by-F-by-H numpy array of paths to radar-
        image files.  This should be created by
        `training_validation_io.find_3d_input_files`.
    :param target_name: See documentation for `create_2d_storm_images`.
    :param return_target: Same.
    :param binarize_target: Same.
    :param top_target_directory_name: Name of top-level directory with target
        files.
    :param radar_normalization_dict: Same.
    :param sounding_statistic_file_names: Same.
    :param sounding_statistic_names: Same.
    :param sounding_stat_metadata_table: Same.
    :return: image_matrix: E-by-M-by-N-by-D-by-C numpy array of storm-centered
        radar images.
    :return: sounding_stat_matrix: [may be None] E-by-S numpy array of sounding
        stats.
    :return: target_values: [may be None] length-E numpy array of target values.
        If target_values[i] = k, the [i]th example belongs to the [k]th class.
    """

    _check_input_args(
        image_file_name_matrix=image_file_name_matrix, num_image_dimensions=3,
        return_target=True, binarize_target=binarize_target,
        sounding_statistic_file_names=sounding_statistic_file_names,
        sounding_statistic_names=sounding_statistic_names)

    num_times = image_file_name_matrix.shape[0]
    num_fields = image_file_name_matrix.shape[1]
    num_heights = image_file_name_matrix.shape[2]
    radar_field_names = [''] * num_fields

    for j in range(num_fields):
        this_storm_image_dict = storm_images.read_storm_images(
            netcdf_file_name=image_file_name_matrix[0, j, 0])
        radar_field_names[j] = this_storm_image_dict[
            storm_images.RADAR_FIELD_NAME_KEY]

    image_matrix = None
    sounding_stat_matrix = None
    target_values = numpy.array([], dtype=int)

    for i in range(num_times):
        if return_target:
            this_label_file_name = storm_images.find_storm_label_file(
                storm_image_file_name=image_file_name_matrix[i, 0, 0],
                top_label_directory_name=top_target_directory_name,
                label_name=target_name, raise_error_if_missing=False)
            if not os.path.isfile(this_label_file_name):
                print (
                    'PROBLEM.  Cannot find label file.  Expected at: "{0:s}"'
                ).format(this_label_file_name)
                continue

            print (
                'Reading storm-centered radar images and hazard labels from '
                '"{0:s}" and "{1:s}"...'
            ).format(image_file_name_matrix[i, 0, 0], this_label_file_name)

            this_storm_image_dict = storm_images.read_storm_images_and_labels(
                image_file_name=image_file_name_matrix[i, 0, 0],
                label_file_name=this_label_file_name, label_name=target_name)
            these_storm_ids = this_storm_image_dict[storm_images.STORM_IDS_KEY]
            if not len(these_storm_ids):
                continue

            this_storm_image_dict, these_valid_storm_indices = (
                trainval_io.remove_storms_with_undef_target(
                    this_storm_image_dict))
            if not len(these_valid_storm_indices):
                continue

            these_storm_ids_to_keep = this_storm_image_dict[
                storm_images.STORM_IDS_KEY]
            these_valid_times_to_keep_unix_sec = this_storm_image_dict[
                storm_images.VALID_TIMES_KEY]
            target_values = numpy.concatenate((
                target_values,
                this_storm_image_dict[storm_images.LABEL_VALUES_KEY]))

            if binarize_target:
                num_classes = labels.column_name_to_num_classes(target_name)
                target_values = (target_values == num_classes - 1).astype(int)
        else:
            print 'Reading storm-centered radar images from "{0:s}"...'.format(
                image_file_name_matrix[i, 0, 0])

            this_storm_image_dict = storm_images.read_storm_images(
                netcdf_file_name=image_file_name_matrix[i, 0, 0])
            these_storm_ids_to_keep = None
            these_valid_times_to_keep_unix_sec = None

        if sounding_statistic_file_names is not None:
            print 'Reading sounding statistics from "{0:s}"...'.format(
                sounding_statistic_file_names[i])
            this_sounding_stat_dict = soundings.read_sounding_statistics(
                netcdf_file_name=sounding_statistic_file_names[i],
                storm_ids_to_keep=these_storm_ids_to_keep,
                statistic_names_to_keep=sounding_statistic_names)

            if sounding_statistic_names is None:
                sounding_statistic_names = this_sounding_stat_dict[
                    soundings.STATISTIC_NAMES_KEY]

            if sounding_stat_matrix is None:
                sounding_stat_matrix = this_sounding_stat_dict[
                    soundings.STATISTIC_MATRIX_KEY]
            else:
                sounding_stat_matrix = numpy.concatenate(
                    (sounding_stat_matrix,
                     this_sounding_stat_dict[soundings.STATISTIC_MATRIX_KEY]),
                    axis=0)

        tuple_of_4d_image_matrices = ()

        for k in range(num_heights):
            tuple_of_3d_image_matrices = ()

            for j in range(num_fields):
                if not j == k == 0:
                    print (
                        'Reading storm-centered radar images from "{0:s}"...'
                    ).format(image_file_name_matrix[i, j, k])
                    this_storm_image_dict = storm_images.read_storm_images(
                        netcdf_file_name=image_file_name_matrix[i, j, k],
                        storm_ids_to_keep=these_storm_ids_to_keep,
                        valid_times_to_keep_unix_sec=
                        these_valid_times_to_keep_unix_sec)

                tuple_of_3d_image_matrices += (
                    this_storm_image_dict[storm_images.STORM_IMAGE_MATRIX_KEY],)

            tuple_of_4d_image_matrices += (
                dl_utils.stack_predictor_variables(tuple_of_3d_image_matrices),)

        this_image_matrix = dl_utils.stack_heights(tuple_of_4d_image_matrices)
        if image_matrix is None:
            image_matrix = this_image_matrix + 0.
        else:
            image_matrix = numpy.concatenate(
                (image_matrix, this_image_matrix), axis=0)

    if image_matrix is None:
        return image_matrix, sounding_stat_matrix, target_values

    image_matrix = dl_utils.normalize_predictor_matrix(
        predictor_matrix=image_matrix, normalize_by_batch=False,
        predictor_names=radar_field_names,
        normalization_dict=radar_normalization_dict).astype('float32')

    if sounding_statistic_file_names is not None:
        sounding_stat_matrix = dl_utils.normalize_sounding_statistics(
            sounding_stat_matrix=sounding_stat_matrix,
            statistic_names=sounding_statistic_names,
            metadata_table=sounding_stat_metadata_table,
            normalize_by_batch=False).astype('float32')

    return image_matrix, sounding_stat_matrix, target_values
