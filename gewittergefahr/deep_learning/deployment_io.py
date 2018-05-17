"""IO methods for deployment of a deep-learning model.

--- NOTATION ---

In this module, the following letters will be used to denote matrix dimensions.

K = number of classes (possible values of target variable)
E = number of examples
M = number of pixel rows per image
N = number of pixel columns per image
D = number of pixel depths per image
C = number of channels (predictor variables) per image
"""

import numpy
from gewittergefahr.deep_learning import storm_images
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import gridrad_utils
from gewittergefahr.gg_utils import labels
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'


# TODO(thunderhoser): Handle missing files in this module.


def _read_sounding_statistics(
        top_directory_name, init_time_unix_sec, target_name, valid_storm_ids,
        statistic_names, metadata_table):
    """Reads sounding statistics from file.

    E = number of examples (storm objects)
    S = number of statistics

    :param top_directory_name: Name of top-level directory with sounding stats.
    :param init_time_unix_sec: Initial time for storm objects.
    :param target_name: Name of target variable.
    :param valid_storm_ids: length-E list of storm IDs.  Will read sounding
        stats only for these storm objects.  If `valid_storm_ids is None`, will
        read all storm objects.
    :param statistic_names: length-S list of statistic names.  Will read only
        these stats.
    :param metadata_table: Used to normalize statistics (see doc for
        `deep_learning_utils.normalize_sounding_statistics`).
    :return: sounding_stat_matrix: E-by-S numpy array of sounding stats.
    """

    target_param_dict = labels.column_name_to_label_params(target_name)
    lead_time_sec = (
        (target_param_dict[labels.MIN_LEAD_TIME_KEY] +
         target_param_dict[labels.MAX_LEAD_TIME_KEY]) / 2)

    sounding_stat_file_name = soundings.find_sounding_statistic_file(
        top_directory_name=top_directory_name,
        init_time_unix_sec=init_time_unix_sec, lead_time_sec=lead_time_sec,
        spc_date_string=time_conversion.time_to_spc_date_string(
            init_time_unix_sec),
        raise_error_if_missing=True)

    sounding_stat_dict = soundings.read_sounding_statistics(
        netcdf_file_name=sounding_stat_file_name,
        storm_ids_to_keep=valid_storm_ids,
        statistic_names_to_keep=statistic_names)
    sounding_stat_names = sounding_stat_dict[soundings.STATISTIC_NAMES_KEY]

    return dl_utils.normalize_sounding_statistics(
        sounding_stat_matrix=sounding_stat_dict[
            soundings.STATISTIC_MATRIX_KEY],
        statistic_names=sounding_stat_names, metadata_table=metadata_table,
        normalize_by_batch=False)


def create_2d_storm_images_one_time(
        top_storm_image_dir_name, init_time_unix_sec, radar_source,
        radar_field_names, target_name, return_target=True,
        radar_heights_m_asl=None, reflectivity_heights_m_asl=None,
        radar_normalization_dict=dl_utils.DEFAULT_NORMALIZATION_DICT,
        top_sounding_stat_dir_name=None, sounding_stat_names=None,
        sounding_stat_metadata_table=None):
    """Creates examples for all storm objects at one time.

    Each example consists of the following.

    - Radar images: one 2-D storm-centered image for each storm object and
      predictor variable (field/height pair).
    - Sounding statistics [optional]: a set of sounding statistics for each
      storm object.
    - Targets [optional]: one target value for each storm object.

    F = number of radar fields
    C = num image channels = num predictor variables = num field/height pairs
    S = number of sounding stats

    :param top_storm_image_dir_name: Name of top-level with storm-centered radar
        images.
    :param init_time_unix_sec: Initial time for storm objects.
    :param radar_source: Name of data source.
    :param radar_field_names: length-F list with names of radar fields.
    :param target_name: Name of target variable.
    :param return_target: Boolean flag.  If True, will return target values and
        predictors.  If False, will return only predictors.
    :param radar_heights_m_asl: [used only if radar_source == "gridrad"]
        1-D numpy array of radar heights (metres above sea level).  These will
        apply to each field.
    :param reflectivity_heights_m_asl: [used only if radar_source != "gridrad"]
        1-D numpy array of radar heights (metres above sea level).  These will
        apply only to the field "reflectivity_dbz".
    :param radar_normalization_dict: Used to normalize radar images (see doc for
        `deep_learning_utils.normalize_predictor_matrix`).
    :param top_sounding_stat_dir_name: Name of top-level directory with sounding
        stats.  If None, this method will not return sounding stats.
    :param sounding_stat_names:
        [used only if `top_sounding_stat_dir_name is not None`]
        length-S list with names of sounding statistics.
    :param sounding_stat_metadata_table:
        [used only if `top_sounding_stat_dir_name is not None`]
        Used to normalize sounding stats (see doc for
        `deep_learning_utils.normalize_sounding_statistics`).
    :return: image_matrix: E-by-M-by-N-by-C numpy array of storm-centered radar
        images.
    :return: sounding_stat_matrix:
        [None if `top_sounding_stat_dir_name is None`]
        E-by-S numpy array of sounding stats.
    :return: target_values: [None if return_target = False]
        length-E numpy array of target values (integers).  If target_values[i] =
        k, the [i]th example belongs to the [k]th class.
    """

    radar_utils.check_data_source(radar_source)
    error_checking.assert_is_boolean(return_target)

    if radar_source == radar_utils.GRIDRAD_SOURCE_ID:
        image_file_name_matrix, _ = storm_images.find_many_files_gridrad(
            top_directory_name=top_storm_image_dir_name,
            start_time_unix_sec=init_time_unix_sec,
            end_time_unix_sec=init_time_unix_sec,
            radar_field_names=radar_field_names,
            radar_heights_m_asl=radar_heights_m_asl,
            raise_error_if_missing=True)

        field_name_by_channel, _ = (
            gridrad_utils.fields_and_refl_heights_to_pairs(
                field_names=radar_field_names,
                heights_m_asl=radar_heights_m_asl))

    else:
        image_file_name_matrix, _, field_name_by_channel, _ = (
            storm_images.find_many_files_myrorss_or_mrms(
                top_directory_name=top_storm_image_dir_name,
                start_time_unix_sec=init_time_unix_sec,
                end_time_unix_sec=init_time_unix_sec,
                radar_source=radar_source, radar_field_names=radar_field_names,
                reflectivity_heights_m_asl=reflectivity_heights_m_asl,
                raise_error_if_missing=True))

    print SEPARATOR_STRING

    num_channels = len(field_name_by_channel)
    image_file_names = numpy.reshape(image_file_name_matrix, num_channels)
    print 'Reading data from: "{0:s}"...'.format(image_file_names[0])

    if return_target:
        this_label_file_name = storm_images.find_storm_label_file(
            storm_image_file_name=image_file_names[0],
            raise_error_if_missing=True)

        this_storm_image_dict = storm_images.read_storm_images_and_labels(
            image_file_name=image_file_names[0],
            label_file_name=this_label_file_name, return_label_name=target_name)

        target_values = this_storm_image_dict[storm_images.LABEL_VALUES_KEY]
        valid_storm_indices = numpy.where(target_values >= 0)[0]
        valid_storm_ids = [
            this_storm_image_dict[storm_images.STORM_IDS_KEY][i]
            for i in valid_storm_indices]

        target_values = target_values[valid_storm_indices]
    else:
        this_storm_image_dict = storm_images.read_storm_images_only(
            image_file_names[0])

        target_values = None
        valid_storm_ids = None

    return_sounding_stats = top_sounding_stat_dir_name is not None
    if return_sounding_stats:
        sounding_stat_matrix = _read_sounding_statistics(
            top_directory_name=top_sounding_stat_dir_name,
            init_time_unix_sec=init_time_unix_sec, target_name=target_name,
            valid_storm_ids=valid_storm_ids,
            statistic_names=sounding_stat_names,
            metadata_table=sounding_stat_metadata_table)
    else:
        sounding_stat_matrix = None

    tuple_of_image_matrices = ()
    for j in range(num_channels):
        if j != 0:
            print 'Reading data from: "{0:s}"...'.format(image_file_names[j])
            this_storm_image_dict = storm_images.read_storm_images_only(
                image_file_names[j])

        this_image_matrix = this_storm_image_dict[
            storm_images.STORM_IMAGE_MATRIX_KEY]
        if return_target:
            this_image_matrix = this_image_matrix[valid_storm_indices, ...]

        tuple_of_image_matrices += (this_image_matrix,)

    image_matrix = dl_utils.stack_predictor_variables(
        tuple_of_image_matrices).astype('float32')
    image_matrix = dl_utils.normalize_predictor_matrix(
        predictor_matrix=image_matrix, normalize_by_batch=False,
        predictor_names=field_name_by_channel,
        normalization_dict=radar_normalization_dict)

    return image_matrix, sounding_stat_matrix, target_values


def create_3d_storm_images_one_time(
        top_storm_image_dir_name, init_time_unix_sec, radar_source,
        radar_field_names, radar_heights_m_asl, target_name, return_target=True,
        radar_normalization_dict=dl_utils.DEFAULT_NORMALIZATION_DICT,
        top_sounding_stat_dir_name=None, sounding_stat_names=None,
        sounding_stat_metadata_table=None):
    """Creates examples for all storm objects at one time.

    This method is the 3-D version of `create_2d_storm_images_one_time`.

    C = number of radar fields
    D = number of radar heights

    :param top_storm_image_dir_name: See doc for
        `create_2d_storm_images_one_time`.
    :param init_time_unix_sec: Same.
    :param radar_source: Same.
    :param radar_field_names: Same.
    :param radar_heights_m_asl: length-D numpy array of radar heights (metres
        above sea level).  These will apply to each field.
    :param target_name: See doc for
        `create_2d_storm_images_one_time`.
    :param return_target: Same.
    :param radar_normalization_dict: Same.
    :param top_sounding_stat_dir_name: Same.
    :param sounding_stat_names: Same.
    :param sounding_stat_metadata_table: Same.
    :return: image_matrix: E-by-M-by-N-by-D-by-C numpy array of storm-centered
        radar images.
    :return: sounding_stat_matrix: See doc for
        `create_2d_storm_images_one_time`.
    :return: target_values: Same.
    """

    radar_utils.check_data_source(radar_source)
    error_checking.assert_is_boolean(return_target)

    if radar_source == radar_utils.GRIDRAD_SOURCE_ID:
        image_file_name_matrix, _ = storm_images.find_many_files_gridrad(
            top_directory_name=top_storm_image_dir_name,
            start_time_unix_sec=init_time_unix_sec,
            end_time_unix_sec=init_time_unix_sec,
            radar_field_names=radar_field_names,
            radar_heights_m_asl=radar_heights_m_asl,
            raise_error_if_missing=True)

        image_file_name_matrix = image_file_name_matrix[0, ...]

    else:
        radar_field_names = [radar_utils.REFL_NAME]

        image_file_name_matrix, _, _, _ = (
            storm_images.find_many_files_myrorss_or_mrms(
                top_directory_name=top_storm_image_dir_name,
                start_time_unix_sec=init_time_unix_sec,
                end_time_unix_sec=init_time_unix_sec,
                radar_source=radar_source, radar_field_names=radar_field_names,
                reflectivity_heights_m_asl=radar_heights_m_asl,
                raise_error_if_missing=True))

        num_heights = len(radar_heights_m_asl)
        image_file_name_matrix = numpy.reshape(
            image_file_name_matrix[0, ...], (1, num_heights))

    print SEPARATOR_STRING

    num_fields = len(radar_field_names)
    num_heights = len(radar_heights_m_asl)
    print 'Reading data from: "{0:s}"...'.format(image_file_name_matrix[0, 0])

    if return_target:
        this_label_file_name = storm_images.find_storm_label_file(
            storm_image_file_name=image_file_name_matrix[0, 0],
            raise_error_if_missing=True)

        this_storm_image_dict = storm_images.read_storm_images_and_labels(
            image_file_name=image_file_name_matrix[0, 0],
            label_file_name=this_label_file_name, return_label_name=target_name)

        target_values = this_storm_image_dict[storm_images.LABEL_VALUES_KEY]
        valid_storm_indices = numpy.where(target_values >= 0)[0]
        valid_storm_ids = [
            this_storm_image_dict[storm_images.STORM_IDS_KEY][i]
            for i in valid_storm_indices]

        target_values = target_values[valid_storm_indices]
    else:
        this_storm_image_dict = storm_images.read_storm_images_only(
            image_file_name_matrix[0, 0])

        target_values = None
        valid_storm_ids = None

    return_sounding_stats = top_sounding_stat_dir_name is not None
    if return_sounding_stats:
        sounding_stat_matrix = _read_sounding_statistics(
            top_directory_name=top_sounding_stat_dir_name,
            init_time_unix_sec=init_time_unix_sec, target_name=target_name,
            valid_storm_ids=valid_storm_ids,
            statistic_names=sounding_stat_names,
            metadata_table=sounding_stat_metadata_table)
    else:
        sounding_stat_matrix = None

    tuple_of_4d_image_matrices = ()
    for k in range(num_heights):
        tuple_of_3d_image_matrices = ()

        for j in range(num_fields):
            if not j == k == 0:
                print 'Reading data from: "{0:s}"...'.format(
                    image_file_name_matrix[j, k])
                this_storm_image_dict = storm_images.read_storm_images_only(
                    image_file_name_matrix[j, k])

            this_image_matrix = this_storm_image_dict[
                storm_images.STORM_IMAGE_MATRIX_KEY]
            if target_name is not None:
                this_image_matrix = this_image_matrix[valid_storm_indices, ...]

            tuple_of_3d_image_matrices += (this_image_matrix,)

        tuple_of_4d_image_matrices += (dl_utils.stack_predictor_variables(
            tuple_of_3d_image_matrices),)

    image_matrix = dl_utils.stack_heights(
        tuple_of_4d_image_matrices).astype('float32')
    image_matrix = dl_utils.normalize_predictor_matrix(
        predictor_matrix=image_matrix, normalize_by_batch=False,
        predictor_names=radar_field_names,
        normalization_dict=radar_normalization_dict)

    return image_matrix, sounding_stat_matrix, target_values
