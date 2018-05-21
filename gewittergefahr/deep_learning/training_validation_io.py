"""IO methods for training or validation of a deep-learning model.

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

import copy
import os.path
import numpy
import keras
from gewittergefahr.deep_learning import storm_images
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import gridrad_utils
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import labels
from gewittergefahr.gg_utils import error_checking

LARGE_INTEGER = 1e10


def _check_input_args(
        num_examples_per_batch, num_examples_per_init_time, normalize_by_batch,
        image_file_name_matrix, num_image_dimensions, binarize_target,
        sounding_statistic_file_names=None, sounding_statistic_names=None):
    """Error-checks input arguments to generator.

    T = number of storm times (initial times, not valid times)

    :param num_examples_per_batch: Number of examples (storm objects) per batch.
    :param num_examples_per_init_time: Number of examples (storm objects) per
        initial time.
    :param normalize_by_batch: Used to normalize predictor values (see doc for
        `deep_learning_utils.normalize_predictor_matrix`).
    :param image_file_name_matrix: numpy array of paths to radar-image files.
        This should be created by `find_2d_input_files` or
        `find_3d_input_files`.  Length of the first axis should be T.
    :param num_image_dimensions: Number of image dimensions.  This should be the
        number of dimensions in `image_file_name_matrix`.
    :param binarize_target: See documentation for `_select_batch`.
    :param sounding_statistic_file_names: [optional] length-T list of paths to
        sounding-statistic files.  This should be created by
        `find_sounding_statistic_files`.
    :param sounding_statistic_names:
        [used only if `sounding_statistic_file_names` is not None]
        length-S list with names of sounding statistics to use.  If None, will
        use all sounding statistics.
    """

    error_checking.assert_is_integer(num_examples_per_batch)
    error_checking.assert_is_geq(num_examples_per_batch, 10)
    error_checking.assert_is_integer(num_examples_per_init_time)
    error_checking.assert_is_geq(num_examples_per_init_time, 2)
    error_checking.assert_is_boolean(normalize_by_batch)

    error_checking.assert_is_integer(num_image_dimensions)
    error_checking.assert_is_geq(num_image_dimensions, 2)
    error_checking.assert_is_leq(num_image_dimensions, 3)
    error_checking.assert_is_numpy_array(
        image_file_name_matrix, num_dimensions=num_image_dimensions)

    error_checking.assert_is_boolean(binarize_target)

    if sounding_statistic_file_names is not None:
        num_init_times = image_file_name_matrix.shape[0]
        error_checking.assert_is_numpy_array(
            numpy.array(sounding_statistic_file_names),
            exact_dimensions=numpy.array([num_init_times]))

        if sounding_statistic_names is not None:
            error_checking.assert_is_numpy_array(
                numpy.array(sounding_statistic_names), num_dimensions=1)


def _shuffle_times(image_file_name_matrix, sounding_statistic_file_names=None):
    """Shuffles file arrays by time.

    :param image_file_name_matrix: See doc for `_check_input_args`.
    :param sounding_statistic_file_names: Same.
    :return: image_file_name_matrix: Same as input, but shuffled by time (along
        first axis).
    :return: sounding_statistic_file_names: Same as input, but shuffled by time
        (along only axis).
    """

    num_init_times = image_file_name_matrix.shape[0]
    init_time_indices = numpy.linspace(
        0, num_init_times - 1, num=num_init_times, dtype=int)
    numpy.random.shuffle(init_time_indices)

    image_file_name_matrix = image_file_name_matrix[init_time_indices, ...]
    if sounding_statistic_file_names is not None:
        sounding_statistic_file_names = [
            sounding_statistic_file_names[i] for i in init_time_indices]

    return image_file_name_matrix, sounding_statistic_file_names


def _get_num_examples_per_batch_by_xclass(
        num_examples_per_batch, target_name, xclass_fractions_to_sample):
    """Returns number of examples needed per batch for each extended class.

    K_x = number of extended classes

    :param num_examples_per_batch: Number of examples needed per batch.
    :param target_name: Name of target variable.
    :param xclass_fractions_to_sample: See doc for `storm_image_generator_2d` or
        `storm_image_generator_3d`.
    :return: num_examples_per_batch_by_xclass: numpy array (length K_x) with
        number of examples needed per batch in each extended class.
    """

    num_extended_classes = labels.column_name_to_num_classes(
        column_name=target_name, include_dead_storms=True)

    if xclass_fractions_to_sample is None:
        num_examples_per_batch_by_xclass = numpy.full(
            num_extended_classes, LARGE_INTEGER, dtype=int)
    else:
        num_examples_per_batch_by_xclass = (
            dl_utils.class_fractions_to_num_points(
                class_fractions=xclass_fractions_to_sample,
                num_points_to_sample=num_examples_per_batch))

        error_checking.assert_is_numpy_array(
            num_examples_per_batch_by_xclass,
            exact_dimensions=numpy.array([num_extended_classes]))

    return num_examples_per_batch_by_xclass


def _get_num_examples_remaining_by_xclass(
        num_examples_per_batch, num_init_times_per_batch,
        num_examples_per_batch_by_xclass, num_examples_in_memory,
        num_init_times_in_memory, num_examples_in_memory_by_xclass):
    """Returns number of examples still needed for each extended class.

    K_x = number of extended classes

    :param num_examples_per_batch: Number of examples needed per batch.
    :param num_init_times_per_batch: Number of initial times needed per batch.
    :param num_examples_per_batch_by_xclass: numpy array (length K_x) with
        number of examples needed per batch in each extended class.
    :param num_examples_in_memory: Number of examples currently in memory.
    :param num_init_times_in_memory: Number of initial times currently in
        memory.
    :param num_examples_in_memory_by_xclass: numpy array (length K_x) with
        number of examples currently in memory for each extended class.
    :return: num_examples_remaining_by_xclass: numpy array (length K_x) with
        number of remaining examples needed for each extended class.
    """

    downsample = (
        num_init_times_in_memory >= num_init_times_per_batch and
        num_examples_in_memory >= num_examples_per_batch)

    if downsample:
        num_examples_remaining_by_xclass = (
            num_examples_per_batch_by_xclass - num_examples_in_memory_by_xclass)
        num_examples_remaining_by_xclass[
            num_examples_remaining_by_xclass < 0] = 0
    else:
        num_examples_remaining_by_xclass = None

    return num_examples_remaining_by_xclass


def _determine_stopping_criterion(
        num_examples_per_batch, num_init_times_per_batch,
        num_examples_per_batch_by_xclass, num_init_times_in_memory,
        xclass_fractions_to_sample, target_values_in_memory, target_name):
    """Determines whether or not to stop generating examples.

    K_x = number of extended classes

    :param num_examples_per_batch: Number of examples needed per batch.
    :param num_init_times_per_batch: Number of initial times needed per batch.
    :param num_examples_per_batch_by_xclass: numpy array (length K_x) with
        number of examples needed per batch in each extended class.
    :param num_init_times_in_memory: Number of initial times currently in
        memory.
    :param xclass_fractions_to_sample: See doc for `storm_image_generator_2d` or
        `storm_image_generator_3d`.
    :param target_values_in_memory: 1-D numpy array with all target values
        currently in memory.
    :param target_name: Name of target variable.
    :return: num_examples_in_memory_by_xclass: numpy array (length K_x) with
        number of examples currently in memory for each extended class.
    :return: stopping_criterion: Boolean flag.
    """

    num_classes = labels.column_name_to_num_classes(
        column_name=target_name, include_dead_storms=False)
    num_extended_classes = labels.column_name_to_num_classes(
        column_name=target_name, include_dead_storms=True)
    include_dead_storms = num_extended_classes > num_classes

    num_examples_in_memory_by_xclass = [
        numpy.sum(target_values_in_memory == k) for k in range(num_classes)]
    if include_dead_storms:
        num_dead_storms = numpy.sum(
            target_values_in_memory == labels.DEAD_STORM_INTEGER)
        num_examples_in_memory_by_xclass = (
            [num_dead_storms] + num_examples_in_memory_by_xclass)

    num_examples_in_memory_by_xclass = numpy.array(
        num_examples_in_memory_by_xclass, dtype=int)
    print 'Number of examples in memory by extended class: {0:s}'.format(
        str(num_examples_in_memory_by_xclass))

    num_examples_in_memory = len(target_values_in_memory)
    stopping_criterion = (
        num_init_times_in_memory >= num_init_times_per_batch and
        num_examples_in_memory >= num_examples_per_batch and
        (xclass_fractions_to_sample is None or numpy.all(
            num_examples_in_memory_by_xclass >=
            num_examples_per_batch_by_xclass)))

    return num_examples_in_memory_by_xclass, stopping_criterion


def _select_batch(
        list_of_image_matrices, target_values, num_examples_per_batch,
        binarize_target=False, num_classes=None):
    """Selects batch randomly from examples in memory.

    E_m = number of examples in memory
    E_b = number of examples in batch

    :param list_of_image_matrices: 1-D list, where each item is a numpy array of
        storm-centered radar images, where the first axis has length E_m.
    :param target_values: numpy array of target values (length E_m).
    :param num_examples_per_batch: Number of examples needed per batch.
    :param binarize_target: Boolean flag.  If True, will binarize target
        variable, so that the highest class is 1 (positive) and all other
        classes are 0 (negative).
    :param num_classes: Number of classes for target value.
    :return: list_of_image_matrices: Same as input, except that the first axis
        of each array now has length E_b.
    :return: target_matrix: See output doc for `storm_image_generator_2d` or
        `storm_image_generator_3d`.
    """

    num_examples_in_memory = len(target_values)
    example_indices = numpy.linspace(
        0, num_examples_in_memory - 1, num=num_examples_in_memory, dtype=int)
    batch_indices = numpy.random.choice(
        example_indices, size=num_examples_per_batch, replace=False)

    for i in range(len(list_of_image_matrices)):
        if list_of_image_matrices[i] is None:
            continue

        list_of_image_matrices[i] = list_of_image_matrices[
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
    print 'Fraction of target values in each class:\n{0:s}\n'.format(
        str(class_fractions))

    return list_of_image_matrices, target_matrix


def remove_storms_with_undef_target(storm_image_dict):
    """Removes storm objects with undefined target value (-1).

    E = number of valid storm objects

    :param storm_image_dict: Dictionary created by
        `storm_images.read_storm_images_and_labels`.
    :return: storm_image_dict: Same as input, but maybe with fewer storm
        objects.
    :return: valid_storm_indices: length-E numpy array with indices of valid
        storm objects.  These are indices into the original dictionary, used to
        obtain the new dictionary.
    """

    valid_storm_indices = numpy.where(
        storm_image_dict[storm_images.LABEL_VALUES_KEY] !=
        labels.INVALID_STORM_INTEGER)[0]

    keys_to_change = [
        storm_images.STORM_IMAGE_MATRIX_KEY, storm_images.STORM_IDS_KEY,
        storm_images.LABEL_VALUES_KEY]
    for this_key in keys_to_change:
        if this_key == storm_images.STORM_IDS_KEY:
            storm_image_dict[this_key] = [
                storm_image_dict[this_key][i] for i in valid_storm_indices]
        else:
            storm_image_dict[this_key] = storm_image_dict[this_key][
                valid_storm_indices, ...]

    return storm_image_dict, valid_storm_indices


def separate_input_files_for_2d3d_myrorss(
        image_file_name_matrix, test_mode=False, field_name_by_pair=None,
        height_by_pair_m_asl=None):
    """Separates input files into reflectivity and azimuthal shear.

    These should be input files for `storm_image_generator_2d3d_myrorss`.

    T = number of storm times (initial times, not valid times)
    F = number of azimuthal-shear fields
    D = number of pixel heights (depths) per reflectivity image

    :param image_file_name_matrix: T-by-(D + F) numpy array of paths to image
        files.  This should be created by `find_2d_input_files`.
    :param test_mode: Leave this False.
    :param field_name_by_pair: Leave this empty.
    :param height_by_pair_m_asl: Leave this empty.
    :return: reflectivity_file_name_matrix: T-by-D numpy array of paths to
        reflectivity files.
    :return: az_shear_file_name_matrix: T-by-F numpy array of paths to
        azimuthal-shear files.
    :raises: ValueError: if `image_file_name_matrix` does not contain
        reflectivity files.
    :raises: ValueError: if `image_file_name_matrix` does not contain azimuthal-
        shear files.
    :raises: ValueError: if `image_file_name_matrix` contains files with any
        other field other than reflectivity or azimuthal shear.
    """

    error_checking.assert_is_numpy_array(
        image_file_name_matrix, num_dimensions=2)
    error_checking.assert_is_boolean(test_mode)
    num_field_height_pairs = image_file_name_matrix.shape[1]

    if not test_mode:
        field_name_by_pair = [''] * num_field_height_pairs
        height_by_pair_m_asl = numpy.full(num_field_height_pairs, -1, dtype=int)

        for j in range(num_field_height_pairs):
            this_storm_image_dict = storm_images.read_storm_images_only(
                image_file_name_matrix[0, j])
            field_name_by_pair[j] = str(
                this_storm_image_dict[storm_images.RADAR_FIELD_NAME_KEY])
            height_by_pair_m_asl[j] = this_storm_image_dict[
                storm_images.RADAR_HEIGHT_KEY]

    reflectivity_indices = numpy.where(numpy.array(
        [s == radar_utils.REFL_NAME for s in field_name_by_pair]))[0]
    if not len(reflectivity_indices):
        error_string = (
            '\n\n{0:s}\nRadar fields in `image_file_name_matrix` (listed above)'
            ' do not include "{1:s}".'
        ).format(str(field_name_by_pair), radar_utils.REFL_NAME)
        raise ValueError(error_string)

    azimuthal_shear_indices = numpy.where(numpy.array(
        [s in radar_utils.SHEAR_NAMES for s in field_name_by_pair]))[0]
    if not len(azimuthal_shear_indices):
        error_string = (
            '\n\n{0:s}\nRadar fields in `image_file_name_matrix` (listed above)'
            ' do not include azimuthal shear.'
        ).format(str(field_name_by_pair))
        raise ValueError(error_string)

    other_field_flags = numpy.full(num_field_height_pairs, True, dtype=bool)
    other_field_flags[reflectivity_indices] = False
    other_field_flags[azimuthal_shear_indices] = False
    if numpy.any(other_field_flags):
        other_field_indices = numpy.where(other_field_flags)[0]
        other_field_names = [field_name_by_pair[i] for i in other_field_indices]

        error_string = (
            '\n\n{0:s}\n`image_file_name_matrix` includes fields other than '
            'reflectivity and azimuthal shear (listed above).'
        ).format(str(other_field_names))
        raise ValueError(error_string)

    sort_indices = numpy.argsort(height_by_pair_m_asl[reflectivity_indices])
    reflectivity_indices = reflectivity_indices[sort_indices]
    reflectivity_file_name_matrix = image_file_name_matrix[
        ..., reflectivity_indices]

    sort_indices = numpy.argsort(
        numpy.array(field_name_by_pair)[azimuthal_shear_indices])
    azimuthal_shear_indices = azimuthal_shear_indices[sort_indices]
    az_shear_file_name_matrix = image_file_name_matrix[
        ..., azimuthal_shear_indices]

    return reflectivity_file_name_matrix, az_shear_file_name_matrix


def find_2d_input_files(
        top_directory_name, radar_source, radar_field_names,
        first_image_time_unix_sec, last_image_time_unix_sec,
        radar_heights_m_asl=None, reflectivity_heights_m_asl=None):
    """Finds input files for `storm_image_generator_2d`.

    T = number of image times
    F = number of radar fields
    C = number of channels = num predictor variables = num field/height pairs

    :param top_directory_name: Name of top-level directory with storm-centered
        radar images.
    :param radar_source: Data source (must be accepted by
        `radar_utils.check_data_source`).
    :param radar_field_names: length-F list with names of radar fields.
    :param first_image_time_unix_sec: First image time.  Files will be sought
        for all time steps from `first_image_time_unix_sec`...
        `last_image_time_unix_sec`.
    :param last_image_time_unix_sec: See above.
    :param radar_heights_m_asl: [used only if radar_source = "gridrad"]
        1-D numpy array of radar heights (metres above sea level).  These will
        be applied to each field in `radar_field_names`.  In other words, if
        there are F fields and H heights, there will be F*H predictor variables.
    :param reflectivity_heights_m_asl: [used only if radar_source != "gridrad"]
        1-D numpy array of radar heights (metres above sea level).  These will
        be applied only to "reflectivity_dbz", if "reflectivity_dbz" is in
        `radar_field_names`.  In other words, if there are F fields and H
        heights, there will be (F + H - 1) predictor variables.
    :return: image_file_name_matrix: T-by-C numpy array of paths to storm-image
        files.
    :return: image_times_unix_sec: length-T numpy array of image times.
    """

    radar_utils.check_data_source(radar_source)

    if radar_source == radar_utils.GRIDRAD_SOURCE_ID:
        image_file_name_matrix, image_times_unix_sec = (
            storm_images.find_many_files_gridrad(
                top_directory_name=top_directory_name,
                start_time_unix_sec=first_image_time_unix_sec,
                end_time_unix_sec=last_image_time_unix_sec,
                radar_field_names=radar_field_names,
                radar_heights_m_asl=radar_heights_m_asl,
                raise_error_if_all_missing=True))

        field_name_by_channel, _ = (
            gridrad_utils.fields_and_refl_heights_to_pairs(
                field_names=radar_field_names,
                heights_m_asl=radar_heights_m_asl))

        num_image_times = len(image_times_unix_sec)
        num_channels = len(field_name_by_channel)
        image_file_name_matrix = numpy.reshape(
            image_file_name_matrix, (num_image_times, num_channels))

    else:
        image_file_name_matrix, image_times_unix_sec, _, _ = (
            storm_images.find_many_files_myrorss_or_mrms(
                top_directory_name=top_directory_name,
                start_time_unix_sec=first_image_time_unix_sec,
                end_time_unix_sec=last_image_time_unix_sec,
                radar_source=radar_source, radar_field_names=radar_field_names,
                reflectivity_heights_m_asl=reflectivity_heights_m_asl,
                raise_error_if_all_missing=True,
                raise_error_if_any_missing=False))

    time_missing_indices = numpy.unique(
        numpy.where(image_file_name_matrix == '')[0])
    image_file_name_matrix = numpy.delete(
        image_file_name_matrix, time_missing_indices, axis=0)
    image_times_unix_sec = numpy.delete(
        image_times_unix_sec, time_missing_indices)

    return image_file_name_matrix, image_times_unix_sec


def find_3d_input_files(
        top_directory_name, radar_source, radar_field_names,
        radar_heights_m_asl, first_image_time_unix_sec,
        last_image_time_unix_sec):
    """Finds input files for `storm_image_generator_3d`.

    T = number of image times
    F = number of radar fields
    D = number of radar heights

    :param top_directory_name: Name of top-level directory with storm-centered
        radar images.
    :param radar_source: Data source (must be accepted by
        `radar_utils.check_data_source`).
    :param radar_field_names: length-F list with names of radar fields.
    :param first_image_time_unix_sec: First image time.  Files will be sought
        for all time steps from `first_image_time_unix_sec`...
        `last_image_time_unix_sec`.
    :param last_image_time_unix_sec: See above.
    :param radar_heights_m_asl: length-H numpy array of radar heights (metres
        above sea level).  These will be applied to each field in
        `radar_field_names`.  In other words, if there are F fields and H
        heights, there will be F*H predictor variables.
    :return: image_file_name_matrix: T-by-F-by-H numpy array of paths to
        storm-image files.
    """

    radar_utils.check_data_source(radar_source)

    if radar_source == radar_utils.GRIDRAD_SOURCE_ID:
        image_file_name_matrix, image_times_unix_sec = (
            storm_images.find_many_files_gridrad(
                top_directory_name=top_directory_name,
                start_time_unix_sec=first_image_time_unix_sec,
                end_time_unix_sec=last_image_time_unix_sec,
                radar_field_names=radar_field_names,
                radar_heights_m_asl=radar_heights_m_asl,
                raise_error_if_all_missing=True))

    else:
        radar_field_names = [radar_utils.REFL_NAME]

        image_file_name_matrix, image_times_unix_sec, _, _ = (
            storm_images.find_many_files_myrorss_or_mrms(
                top_directory_name=top_directory_name,
                start_time_unix_sec=first_image_time_unix_sec,
                end_time_unix_sec=last_image_time_unix_sec,
                radar_source=radar_source, radar_field_names=radar_field_names,
                reflectivity_heights_m_asl=radar_heights_m_asl,
                raise_error_if_all_missing=True,
                raise_error_if_any_missing=False))

        num_image_times = len(image_times_unix_sec)
        num_heights = len(radar_heights_m_asl)
        image_file_name_matrix = numpy.reshape(
            image_file_name_matrix, (num_image_times, 1, num_heights))

    time_missing_indices = numpy.unique(
        numpy.where(image_file_name_matrix == '')[0])
    image_file_name_matrix = numpy.delete(
        image_file_name_matrix, time_missing_indices, axis=0)
    image_times_unix_sec = numpy.delete(
        image_times_unix_sec, time_missing_indices)

    return image_file_name_matrix, image_times_unix_sec


def find_sounding_statistic_files(
        image_file_name_matrix, image_times_unix_sec, top_sounding_dir_name,
        sounding_lead_time_sec, raise_error_if_any_missing=True):
    """Locates files with sounding statistics.

    These files may be used as input for `storm_image_generator_2d` or
    `storm_image_generator_3d`.

    T = number of time steps in output

    :param image_file_name_matrix: numpy array of paths to storm-image files,
        created by either `find_2d_input_files` or `find_3d_input_files`.
    :param image_times_unix_sec: numpy array of corresponding time steps,
        created by either `find_2d_input_files` or `find_3d_input_files`.
    :param top_sounding_dir_name: Name of top-level directory with sounding
        stats.
    :param sounding_lead_time_sec: Lead time for sounding stats.  See
        `soundings.get_sounding_stats_for_storm_objects` for more details.
    :param raise_error_if_any_missing: Boolean flag.  If any sounding-statistic
        file is missing and `raise_error_if_any_missing = True`, this method
        will error out.  If any sounding-statistic file is missing and
        `raise_error_if_any_missing = False`, this method will skip the relevant
        time step and delete entries in `image_file_name_matrix` for the
        relevant time step.
    :return: sounding_statistic_file_names: length-T list of paths to sounding-
        statistic files.
    :return: image_file_name_matrix: Same as input, but maybe with fewer time
        steps.  The length of the first axis is T.
    :raises: ValueError: if no sounding-statistic files are found.
    """

    error_checking.assert_is_integer_numpy_array(image_times_unix_sec)
    error_checking.assert_is_numpy_array(image_times_unix_sec, num_dimensions=1)
    error_checking.assert_is_boolean(raise_error_if_any_missing)

    error_checking.assert_is_numpy_array(image_file_name_matrix)
    num_dimensions = len(image_file_name_matrix.shape)
    error_checking.assert_is_geq(num_dimensions, 2)
    error_checking.assert_is_leq(num_dimensions, 3)

    num_image_times = len(image_times_unix_sec)
    expected_dimensions = (num_image_times,) + image_file_name_matrix.shape[1:]
    error_checking.assert_is_numpy_array(
        image_file_name_matrix, exact_dimensions=expected_dimensions)

    sounding_statistic_file_names = [''] * num_image_times
    for i in range(num_image_times):
        this_file_name = soundings.find_sounding_statistic_file(
            top_directory_name=top_sounding_dir_name,
            init_time_unix_sec=image_times_unix_sec[i],
            lead_time_sec=sounding_lead_time_sec,
            spc_date_string=time_conversion.time_to_spc_date_string(
                image_times_unix_sec[i]),
            raise_error_if_missing=raise_error_if_any_missing)

        if not os.path.isfile(this_file_name):
            continue

        sounding_statistic_file_names[i] = this_file_name

    found_file_indices = numpy.where(
        numpy.array([f != '' for f in sounding_statistic_file_names]))[0]
    if not len(found_file_indices):
        error_string = (
            'Could not find any files with sounding statistics in directory '
            '"{0:s}" at lead time of {1:d} seconds.'
        ).format(top_sounding_dir_name, sounding_lead_time_sec)
        raise ValueError(error_string)

    sounding_statistic_file_names = [
        sounding_statistic_file_names[i] for i in found_file_indices]
    image_file_name_matrix = image_file_name_matrix[found_file_indices, ...]

    return sounding_statistic_file_names, image_file_name_matrix


def storm_image_generator_2d(
        image_file_name_matrix, num_examples_per_batch,
        num_examples_per_init_time, target_name, binarize_target=False,
        radar_normalization_dict=dl_utils.DEFAULT_NORMALIZATION_DICT,
        xclass_fractions_to_sample=None, sounding_statistic_file_names=None,
        sounding_statistic_names=None, sounding_stat_metadata_table=None):
    """Generates examples with 2-D radar images.

    Each example consists of a 2-D radar image, and possibly sounding
    statistics, for one storm object.

    T = number of storm times (initial times, not valid times)
    F = number of radar fields
    C = number of channels = num predictor variables = num field/height pairs
    K_x = number of extended classes

    :param image_file_name_matrix: T-by-C numpy array of paths to radar-image
        files.  This should be created by `find_2d_input_files`.
    :param num_examples_per_batch: See doc for `_check_input_args`.
    :param num_examples_per_init_time: Same.
    :param target_name: Name of target variable.
    :param binarize_target: See documentation for `_select_batch`.
    :param radar_normalization_dict: Used to normalize radar images (see doc for
        `deep_learning_utils.normalize_predictor_matrix`).
    :param xclass_fractions_to_sample: numpy array (length K_x) for class-
        conditional sampling.  xclass_fractions_to_sample[k] is the fraction of
        examples in the [k]th extended class to be included in each batch.
    :param sounding_statistic_file_names: See doc for `_check_input_args`.
    :param sounding_statistic_names: Same.
    :param sounding_stat_metadata_table: Used to normalize sounding stats (see
        doc for `deep_learning_utils.normalize_sounding_statistics`).

    If `sounding_statistic_file_names is None`, this method returns...

    :return: image_matrix: E-by-M-by-N-by-C numpy array of storm-centered radar
        images.
    :return: target_matrix: E-by-K numpy array of target values (all 0 or 1, but
        the array type is "float64").  If target_matrix[i, k] = 1, the [k]th
        class is the outcome for the [i]th example.  Classes are mutually
        exclusive and collectively exhaustive, so the sum across each row is 1.

    If `sounding_statistic_file_names is not None`, this method returns...

    :return: predictor_list: List with the following items.
    predictor_list[0] = image_matrix: See documentation above.
    predictor_list[1] = sounding_stat_matrix: E-by-S numpy array of sounding
        statistics.
    :return: target_matrix: See documentation above.
    """

    _check_input_args(
        num_examples_per_batch=num_examples_per_batch,
        num_examples_per_init_time=num_examples_per_init_time,
        normalize_by_batch=False, image_file_name_matrix=image_file_name_matrix,
        num_image_dimensions=2, binarize_target=binarize_target,
        sounding_statistic_file_names=sounding_statistic_file_names,
        sounding_statistic_names=sounding_statistic_names)

    num_channels = image_file_name_matrix.shape[1]
    field_name_by_channel = [''] * num_channels
    for j in range(num_channels):
        this_storm_image_dict = storm_images.read_storm_images_only(
            image_file_name_matrix[0, j])
        field_name_by_channel[j] = str(
            this_storm_image_dict[storm_images.RADAR_FIELD_NAME_KEY])

    image_file_name_matrix, sounding_statistic_file_names = _shuffle_times(
        image_file_name_matrix=image_file_name_matrix,
        sounding_statistic_file_names=sounding_statistic_file_names)
    num_init_times = image_file_name_matrix.shape[0]

    num_examples_per_batch_by_xclass = _get_num_examples_per_batch_by_xclass(
        num_examples_per_batch=num_examples_per_batch, target_name=target_name,
        xclass_fractions_to_sample=xclass_fractions_to_sample)

    num_classes = labels.column_name_to_num_classes(
        column_name=target_name, include_dead_storms=False)
    num_extended_classes = labels.column_name_to_num_classes(
        column_name=target_name, include_dead_storms=True)

    # Initialize housekeeping variables.
    init_time_index = 0
    num_init_times_in_memory = 0
    num_examples_in_memory_by_xclass = numpy.full(
        num_extended_classes, 0, dtype=int)
    num_init_times_per_batch = int(numpy.ceil(
        float(num_examples_per_batch) / num_examples_per_init_time))

    full_image_matrix = None
    full_sounding_stat_matrix = None
    all_target_values = None

    while True:
        stopping_criterion = False

        while not stopping_criterion:
            print '\n'
            tuple_of_image_matrices = ()

            this_label_file_name = storm_images.find_storm_label_file(
                storm_image_file_name=image_file_name_matrix[
                    init_time_index, 0],
                raise_error_if_missing=False, warn_if_missing=True)
            if not os.path.isfile(this_label_file_name):
                init_time_index = numpy.mod(init_time_index + 1, num_init_times)
                continue

            print 'Reading data from: "{0:s}"...'.format(
                image_file_name_matrix[init_time_index, 0])

            if all_target_values is None:
                num_examples_in_memory = 0
            else:
                num_examples_in_memory = len(all_target_values)

            num_examples_remaining_by_xclass = (
                _get_num_examples_remaining_by_xclass(
                    num_examples_per_batch=num_examples_per_batch,
                    num_init_times_per_batch=num_init_times_per_batch,
                    num_examples_per_batch_by_xclass=
                    num_examples_per_batch_by_xclass,
                    num_examples_in_memory=num_examples_in_memory,
                    num_init_times_in_memory=num_init_times_in_memory,
                    num_examples_in_memory_by_xclass=
                    num_examples_in_memory_by_xclass))

            this_storm_image_dict = storm_images.read_storm_images_and_labels(
                image_file_name=image_file_name_matrix[init_time_index, 0],
                label_file_name=this_label_file_name,
                return_label_name=target_name,
                num_storm_objects_by_xclass=num_examples_remaining_by_xclass)

            if this_storm_image_dict is None:
                init_time_index = numpy.mod(init_time_index + 1, num_init_times)
                continue

            this_storm_image_dict, these_valid_storm_indices = (
                remove_storms_with_undef_target(this_storm_image_dict))
            if not len(these_valid_storm_indices):
                init_time_index = numpy.mod(init_time_index + 1, num_init_times)
                continue

            these_valid_storm_ids = this_storm_image_dict[
                storm_images.STORM_IDS_KEY]

            if all_target_values is None:
                all_target_values = this_storm_image_dict[
                    storm_images.LABEL_VALUES_KEY]
            else:
                all_target_values = numpy.concatenate((
                    all_target_values,
                    this_storm_image_dict[storm_images.LABEL_VALUES_KEY]))

            if sounding_statistic_file_names is not None:
                this_sounding_stat_dict = soundings.read_sounding_statistics(
                    netcdf_file_name=sounding_statistic_file_names[
                        init_time_index],
                    storm_ids_to_keep=these_valid_storm_ids,
                    statistic_names_to_keep=sounding_statistic_names)

                if sounding_statistic_names is None:
                    sounding_statistic_names = this_sounding_stat_dict[
                        soundings.STATISTIC_NAMES_KEY]

                if full_sounding_stat_matrix is None:
                    full_sounding_stat_matrix = this_sounding_stat_dict[
                        soundings.STATISTIC_MATRIX_KEY]
                else:
                    full_sounding_stat_matrix = numpy.concatenate(
                        (full_sounding_stat_matrix,
                         this_sounding_stat_dict[
                             soundings.STATISTIC_MATRIX_KEY]),
                        axis=0)

            for j in range(num_channels):
                if j != 0:
                    print 'Reading data from: "{0:s}"...'.format(
                        image_file_name_matrix[init_time_index, j])
                    this_storm_image_dict = storm_images.read_storm_images_only(
                        netcdf_file_name=image_file_name_matrix[
                            init_time_index, j],
                        indices_to_keep=these_valid_storm_indices)

                this_field_image_matrix = this_storm_image_dict[
                    storm_images.STORM_IMAGE_MATRIX_KEY]
                tuple_of_image_matrices += (this_field_image_matrix,)

            # Update housekeeping variables.
            num_init_times_in_memory += 1
            init_time_index = numpy.mod(init_time_index + 1, num_init_times)

            this_image_matrix = dl_utils.stack_predictor_variables(
                tuple_of_image_matrices)
            if full_image_matrix is None:
                full_image_matrix = copy.deepcopy(this_image_matrix)
            else:
                full_image_matrix = numpy.concatenate(
                    (full_image_matrix, this_image_matrix), axis=0)

            num_examples_in_memory_by_xclass, stopping_criterion = (
                _determine_stopping_criterion(
                    num_examples_per_batch=num_examples_per_batch,
                    num_init_times_per_batch=num_init_times_per_batch,
                    num_examples_per_batch_by_xclass=
                    num_examples_per_batch_by_xclass,
                    num_init_times_in_memory=num_init_times_in_memory,
                    xclass_fractions_to_sample=xclass_fractions_to_sample,
                    target_values_in_memory=all_target_values,
                    target_name=target_name))

        if xclass_fractions_to_sample is not None:
            batch_indices = dl_utils.sample_points_by_extended_class(
                target_values=all_target_values, target_name=target_name,
                extended_class_fractions=xclass_fractions_to_sample,
                num_points_to_sample=num_examples_per_batch)

            full_image_matrix = full_image_matrix[batch_indices, ...]
            all_target_values = all_target_values[batch_indices]
            if sounding_statistic_file_names is not None:
                full_sounding_stat_matrix = full_sounding_stat_matrix[
                    batch_indices, ...]

        full_image_matrix = dl_utils.normalize_predictor_matrix(
            predictor_matrix=full_image_matrix,
            normalize_by_batch=False, predictor_names=field_name_by_channel,
            normalization_dict=radar_normalization_dict)

        if sounding_statistic_file_names is not None:
            full_sounding_stat_matrix = dl_utils.normalize_sounding_statistics(
                sounding_stat_matrix=full_sounding_stat_matrix,
                statistic_names=sounding_statistic_names,
                metadata_table=sounding_stat_metadata_table,
                normalize_by_batch=False)

        list_of_image_matrices, target_matrix = _select_batch(
            list_of_image_matrices=[
                full_image_matrix, full_sounding_stat_matrix],
            target_values=all_target_values,
            num_examples_per_batch=num_examples_per_batch,
            binarize_target=binarize_target, num_classes=num_classes)

        image_matrix = list_of_image_matrices[0]
        sounding_stat_matrix = list_of_image_matrices[1]

        # Update housekeeping variables.
        full_image_matrix = None
        all_target_values = None
        full_sounding_stat_matrix = None
        num_init_times_in_memory = 0
        num_examples_in_memory_by_xclass = numpy.full(
            num_extended_classes, 0, dtype=int)

        if sounding_statistic_file_names is None:
            yield (image_matrix, target_matrix)
        else:
            yield ([image_matrix, sounding_stat_matrix], target_matrix)


def storm_image_generator_2d3d_myrorss(
        image_file_name_matrix, num_examples_per_batch,
        num_examples_per_init_time, target_name, binarize_target=False,
        radar_normalization_dict=dl_utils.DEFAULT_NORMALIZATION_DICT,
        xclass_fractions_to_sample=None):
    """Generates examples with both 2-D and 3-D radar images.

    Each example consists of a 3-D reflectivity image and one or more 2-D
    azimuthal-shear images.

    T = number of storm times (initial times, not valid times)
    F = number of azimuthal-shear fields
    m = number of pixel rows per reflectivity image
    n = number of pixel columns per reflectivity image
    D = number of pixel heights (depths) per reflectivity image
    M = number of pixel rows per az-shear image
    N = number of pixel columns per az-shear image

    :param image_file_name_matrix: T-by-(D + F) numpy array of paths to image
        files.  This should be created by `find_2d_input_files`.
    :param num_examples_per_batch: See doc for `_check_input_args`.
    :param num_examples_per_init_time: Same.
    :param target_name: Name of target variable.
    :param binarize_target: See documentation for `_select_batch`.
    :param radar_normalization_dict: See doc for `storm_image_generator_2d`.
    :param xclass_fractions_to_sample: Same.
    :return: predictor_list: List with the following items.
    predictor_list[0] = reflectivity_image_matrix: E-by-m-by-n-by-D-by-1 numpy
        array of storm-centered reflectivity images.
    predictor_list[1] = azimuthal_shear_image_matrix: E-by-M-by-N-by-F numpy
        array of storm-centered az-shear images.
    :return: target_matrix: See doc for `storm_image_generator_2d`.
    """

    _check_input_args(
        num_examples_per_batch=num_examples_per_batch,
        num_examples_per_init_time=num_examples_per_init_time,
        normalize_by_batch=False, image_file_name_matrix=image_file_name_matrix,
        num_image_dimensions=2, binarize_target=binarize_target)

    image_file_name_matrix, _ = _shuffle_times(image_file_name_matrix)
    reflectivity_file_name_matrix, az_shear_file_name_matrix = (
        separate_input_files_for_2d3d_myrorss(image_file_name_matrix))

    num_init_times = reflectivity_file_name_matrix.shape[0]
    num_reflectivity_heights = reflectivity_file_name_matrix.shape[1]
    num_az_shear_fields = az_shear_file_name_matrix.shape[1]

    az_shear_field_names = [''] * num_az_shear_fields
    for j in range(num_az_shear_fields):
        this_storm_image_dict = storm_images.read_storm_images_only(
            az_shear_file_name_matrix[0, j])
        az_shear_field_names[j] = str(
            this_storm_image_dict[storm_images.RADAR_FIELD_NAME_KEY])

    num_examples_per_batch_by_xclass = _get_num_examples_per_batch_by_xclass(
        num_examples_per_batch=num_examples_per_batch, target_name=target_name,
        xclass_fractions_to_sample=xclass_fractions_to_sample)

    num_classes = labels.column_name_to_num_classes(
        column_name=target_name, include_dead_storms=False)
    num_extended_classes = labels.column_name_to_num_classes(
        column_name=target_name, include_dead_storms=True)

    # Initialize housekeeping variables.
    init_time_index = 0
    num_init_times_in_memory = 0
    num_examples_in_memory_by_xclass = numpy.full(
        num_extended_classes, 0, dtype=int)
    num_init_times_per_batch = int(numpy.ceil(
        float(num_examples_per_batch) / num_examples_per_init_time))

    full_reflectivity_matrix_dbz = None
    full_az_shear_matrix_s01 = None
    all_target_values = None

    while True:
        stopping_criterion = False

        while not stopping_criterion:
            print '\n'
            tuple_of_4d_refl_matrices = ()
            tuple_of_3d_az_shear_matrices = ()

            this_label_file_name = storm_images.find_storm_label_file(
                storm_image_file_name=reflectivity_file_name_matrix[
                    init_time_index, 0],
                raise_error_if_missing=False, warn_if_missing=True)
            if not os.path.isfile(this_label_file_name):
                init_time_index = numpy.mod(init_time_index + 1, num_init_times)
                continue

            print 'Reading data from: "{0:s}"...'.format(
                reflectivity_file_name_matrix[init_time_index, 0])

            if all_target_values is None:
                num_examples_in_memory = 0
            else:
                num_examples_in_memory = len(all_target_values)

            num_examples_remaining_by_xclass = (
                _get_num_examples_remaining_by_xclass(
                    num_examples_per_batch=num_examples_per_batch,
                    num_init_times_per_batch=num_init_times_per_batch,
                    num_examples_per_batch_by_xclass=
                    num_examples_per_batch_by_xclass,
                    num_examples_in_memory=num_examples_in_memory,
                    num_init_times_in_memory=num_init_times_in_memory,
                    num_examples_in_memory_by_xclass=
                    num_examples_in_memory_by_xclass))

            this_storm_image_dict = storm_images.read_storm_images_and_labels(
                image_file_name=reflectivity_file_name_matrix[
                    init_time_index, 0],
                label_file_name=this_label_file_name,
                return_label_name=target_name,
                num_storm_objects_by_xclass=num_examples_remaining_by_xclass)

            if this_storm_image_dict is None:
                init_time_index = numpy.mod(init_time_index + 1, num_init_times)
                continue

            this_storm_image_dict, these_valid_storm_indices = (
                remove_storms_with_undef_target(this_storm_image_dict))
            if not len(these_valid_storm_indices):
                init_time_index = numpy.mod(init_time_index + 1, num_init_times)
                continue

            if all_target_values is None:
                all_target_values = this_storm_image_dict[
                    storm_images.LABEL_VALUES_KEY]
            else:
                all_target_values = numpy.concatenate((
                    all_target_values,
                    this_storm_image_dict[storm_images.LABEL_VALUES_KEY]))

            for j in range(num_reflectivity_heights):
                if j != 0:
                    print 'Reading data from: "{0:s}"...'.format(
                        reflectivity_file_name_matrix[init_time_index, j])
                    this_storm_image_dict = storm_images.read_storm_images_only(
                        netcdf_file_name=reflectivity_file_name_matrix[
                            init_time_index, j],
                        indices_to_keep=these_valid_storm_indices)

                this_4d_matrix = dl_utils.stack_predictor_variables((
                    this_storm_image_dict[storm_images.STORM_IMAGE_MATRIX_KEY],
                ))
                tuple_of_4d_refl_matrices += (this_4d_matrix,)

            for j in range(num_az_shear_fields):
                print 'Reading data from: "{0:s}"...'.format(
                    az_shear_file_name_matrix[init_time_index, j])
                this_storm_image_dict = storm_images.read_storm_images_only(
                    netcdf_file_name=az_shear_file_name_matrix[
                        init_time_index, j],
                    indices_to_keep=these_valid_storm_indices)

                tuple_of_3d_az_shear_matrices += (
                    this_storm_image_dict[storm_images.STORM_IMAGE_MATRIX_KEY],)

            # Update housekeeping variables.
            num_init_times_in_memory += 1
            init_time_index = numpy.mod(init_time_index + 1, num_init_times)

            this_reflectivity_matrix_dbz = dl_utils.stack_heights(
                tuple_of_4d_refl_matrices)
            this_az_shear_matrix_s01 = dl_utils.stack_predictor_variables(
                tuple_of_3d_az_shear_matrices)

            if full_reflectivity_matrix_dbz is None:
                full_reflectivity_matrix_dbz = copy.deepcopy(
                    this_reflectivity_matrix_dbz)
                full_az_shear_matrix_s01 = copy.deepcopy(
                    this_az_shear_matrix_s01)
            else:
                full_reflectivity_matrix_dbz = numpy.concatenate(
                    (full_reflectivity_matrix_dbz,
                     this_reflectivity_matrix_dbz),
                    axis=0)
                full_az_shear_matrix_s01 = numpy.concatenate(
                    (full_az_shear_matrix_s01, this_az_shear_matrix_s01),
                    axis=0)

            num_examples_in_memory_by_xclass, stopping_criterion = (
                _determine_stopping_criterion(
                    num_examples_per_batch=num_examples_per_batch,
                    num_init_times_per_batch=num_init_times_per_batch,
                    num_examples_per_batch_by_xclass=
                    num_examples_per_batch_by_xclass,
                    num_init_times_in_memory=num_init_times_in_memory,
                    xclass_fractions_to_sample=xclass_fractions_to_sample,
                    target_values_in_memory=all_target_values,
                    target_name=target_name))

        if xclass_fractions_to_sample is not None:
            batch_indices = dl_utils.sample_points_by_extended_class(
                target_values=all_target_values, target_name=target_name,
                extended_class_fractions=xclass_fractions_to_sample,
                num_points_to_sample=num_examples_per_batch)

            full_reflectivity_matrix_dbz = full_reflectivity_matrix_dbz[
                batch_indices, ...]
            full_az_shear_matrix_s01 = full_az_shear_matrix_s01[
                batch_indices, ...]
            all_target_values = all_target_values[batch_indices]

        full_reflectivity_matrix_dbz = dl_utils.normalize_predictor_matrix(
            predictor_matrix=full_reflectivity_matrix_dbz,
            normalize_by_batch=False, predictor_names=[radar_utils.REFL_NAME],
            normalization_dict=radar_normalization_dict)

        full_az_shear_matrix_s01 = dl_utils.normalize_predictor_matrix(
            predictor_matrix=full_az_shear_matrix_s01, normalize_by_batch=False,
            predictor_names=az_shear_field_names,
            normalization_dict=radar_normalization_dict)

        list_of_image_matrices, target_matrix = _select_batch(
            list_of_image_matrices=[
                full_reflectivity_matrix_dbz, full_az_shear_matrix_s01],
            target_values=all_target_values,
            num_examples_per_batch=num_examples_per_batch,
            binarize_target=binarize_target, num_classes=num_classes)

        reflectivity_matrix_dbz = list_of_image_matrices[0]
        azimuthal_shear_matrix_s01 = list_of_image_matrices[1]

        # Update housekeeping variables.
        full_reflectivity_matrix_dbz = None
        full_az_shear_matrix_s01 = None
        all_target_values = None
        num_init_times_in_memory = 0
        num_examples_in_memory_by_xclass = numpy.full(
            num_extended_classes, 0, dtype=int)

        yield ([reflectivity_matrix_dbz, azimuthal_shear_matrix_s01],
               target_matrix)


def storm_image_generator_3d(
        image_file_name_matrix, num_examples_per_batch,
        num_examples_per_init_time, target_name, binarize_target=False,
        radar_normalization_dict=dl_utils.DEFAULT_NORMALIZATION_DICT,
        xclass_fractions_to_sample=None, sounding_statistic_file_names=None,
        sounding_statistic_names=None, sounding_stat_metadata_table=None):
    """Generates examples with 3-D radar images.

    Each example consists of a 3-D radar image, and possibly sounding
    statistics, for one storm object.

    T = number of storm times (initial times, not valid times)
    F = number of radar fields
    D = number of radar heights

    :param image_file_name_matrix: T-by-F-by-H numpy array of paths to radar-
        image files.  This should be created by `find_3d_input_files`.
    :param num_examples_per_batch: See doc for `_check_input_args`.
    :param num_examples_per_init_time: Same.
    :param target_name: Name of target variable.
    :param binarize_target: See documentation for `_select_batch`.
    :param radar_normalization_dict: See doc for `storm_image_generator_2d`.
    :param xclass_fractions_to_sample: Same.
    :param sounding_statistic_file_names: Same.
    :param sounding_statistic_names: Same.
    :param sounding_stat_metadata_table: Same.

    If `sounding_statistic_file_names is None`, this method returns...

    :return: image_matrix: E-by-M-by-N-by-D-by-C numpy array of storm-centered
        radar images.
    :return: target_matrix: See doc for `storm_image_generator_2d`.

    If `sounding_statistic_file_names is not None`, this method returns...

    :return: predictor_list: List with the following items.
    predictor_list[0] = image_matrix: See documentation above.
    predictor_list[1] = sounding_stat_matrix: E-by-S numpy array of sounding
        statistics.
    :return: target_matrix: See documentation above.
    """

    _check_input_args(
        num_examples_per_batch=num_examples_per_batch,
        num_examples_per_init_time=num_examples_per_init_time,
        normalize_by_batch=False, image_file_name_matrix=image_file_name_matrix,
        num_image_dimensions=3, binarize_target=binarize_target,
        sounding_statistic_file_names=sounding_statistic_file_names,
        sounding_statistic_names=sounding_statistic_names)

    num_fields = image_file_name_matrix.shape[1]
    num_heights = image_file_name_matrix.shape[2]
    radar_field_names = [''] * num_fields
    for j in range(num_fields):
        this_storm_image_dict = storm_images.read_storm_images_only(
            image_file_name_matrix[0, j, 0])
        radar_field_names[j] = str(
            this_storm_image_dict[storm_images.RADAR_FIELD_NAME_KEY])

    image_file_name_matrix, sounding_statistic_file_names = _shuffle_times(
        image_file_name_matrix=image_file_name_matrix,
        sounding_statistic_file_names=sounding_statistic_file_names)
    num_init_times = image_file_name_matrix.shape[0]

    num_examples_per_batch_by_xclass = _get_num_examples_per_batch_by_xclass(
        num_examples_per_batch=num_examples_per_batch, target_name=target_name,
        xclass_fractions_to_sample=xclass_fractions_to_sample)

    num_classes = labels.column_name_to_num_classes(
        column_name=target_name, include_dead_storms=False)
    num_extended_classes = labels.column_name_to_num_classes(
        column_name=target_name, include_dead_storms=True)

    # Initialize housekeeping variables.
    init_time_index = 0
    num_init_times_in_memory = 0
    num_examples_in_memory_by_xclass = numpy.full(
        num_extended_classes, 0, dtype=int)
    num_init_times_per_batch = int(numpy.ceil(
        float(num_examples_per_batch) / num_examples_per_init_time))

    full_image_matrix = None
    all_target_values = None
    full_sounding_stat_matrix = None

    while True:
        stopping_criterion = False

        while not stopping_criterion:
            print '\n'
            tuple_of_4d_image_matrices = ()

            this_label_file_name = storm_images.find_storm_label_file(
                storm_image_file_name=image_file_name_matrix[
                    init_time_index, 0, 0],
                raise_error_if_missing=False, warn_if_missing=True)
            if not os.path.isfile(this_label_file_name):
                init_time_index = numpy.mod(init_time_index + 1, num_init_times)
                continue

            print 'Reading data from: "{0:s}"...'.format(
                image_file_name_matrix[init_time_index, 0, 0])

            if full_image_matrix is None:
                num_examples_in_memory = 0
            else:
                num_examples_in_memory = full_image_matrix.shape[0]

            num_examples_remaining_by_xclass = (
                _get_num_examples_remaining_by_xclass(
                    num_examples_per_batch=num_examples_per_batch,
                    num_init_times_per_batch=num_init_times_per_batch,
                    num_examples_per_batch_by_xclass=
                    num_examples_per_batch_by_xclass,
                    num_examples_in_memory=num_examples_in_memory,
                    num_init_times_in_memory=num_init_times_in_memory,
                    num_examples_in_memory_by_xclass=
                    num_examples_in_memory_by_xclass))

            this_storm_image_dict = storm_images.read_storm_images_and_labels(
                image_file_name=image_file_name_matrix[init_time_index, 0, 0],
                label_file_name=this_label_file_name,
                return_label_name=target_name,
                num_storm_objects_by_xclass=num_examples_remaining_by_xclass)

            if this_storm_image_dict is None:
                init_time_index = numpy.mod(init_time_index + 1, num_init_times)
                continue

            this_storm_image_dict, these_valid_storm_indices = (
                remove_storms_with_undef_target(this_storm_image_dict))
            if not len(these_valid_storm_indices):
                init_time_index = numpy.mod(init_time_index + 1, num_init_times)
                continue

            these_valid_storm_ids = this_storm_image_dict[
                storm_images.STORM_IDS_KEY]

            if all_target_values is None:
                all_target_values = this_storm_image_dict[
                    storm_images.LABEL_VALUES_KEY]
            else:
                all_target_values = numpy.concatenate((
                    all_target_values,
                    this_storm_image_dict[storm_images.LABEL_VALUES_KEY]))

            if sounding_statistic_file_names is not None:
                this_sounding_stat_dict = soundings.read_sounding_statistics(
                    netcdf_file_name=sounding_statistic_file_names[
                        init_time_index],
                    storm_ids_to_keep=these_valid_storm_ids,
                    statistic_names_to_keep=sounding_statistic_names)

                if sounding_statistic_names is None:
                    sounding_statistic_names = this_sounding_stat_dict[
                        soundings.STATISTIC_NAMES_KEY]

                if full_sounding_stat_matrix is None:
                    full_sounding_stat_matrix = this_sounding_stat_dict[
                        soundings.STATISTIC_MATRIX_KEY]
                else:
                    full_sounding_stat_matrix = numpy.concatenate(
                        (full_sounding_stat_matrix,
                         this_sounding_stat_dict[
                             soundings.STATISTIC_MATRIX_KEY]),
                        axis=0)

            for k in range(num_heights):
                tuple_of_3d_image_matrices = ()

                for j in range(num_fields):
                    if not j == k == 0:
                        print 'Reading data from: "{0:s}"...'.format(
                            image_file_name_matrix[init_time_index, j, k])
                        this_storm_image_dict = (
                            storm_images.read_storm_images_only(
                                netcdf_file_name=
                                image_file_name_matrix[init_time_index, j, k],
                                indices_to_keep=these_valid_storm_indices))

                    this_3d_image_matrix = this_storm_image_dict[
                        storm_images.STORM_IMAGE_MATRIX_KEY]
                    tuple_of_3d_image_matrices += (this_3d_image_matrix,)

                tuple_of_4d_image_matrices += (
                    dl_utils.stack_predictor_variables(
                        tuple_of_3d_image_matrices),)

            # Update housekeeping variables.
            num_init_times_in_memory += 1
            init_time_index = numpy.mod(init_time_index + 1, num_init_times)

            this_image_matrix = dl_utils.stack_heights(
                tuple_of_4d_image_matrices)
            if full_image_matrix is None:
                full_image_matrix = copy.deepcopy(this_image_matrix)
            else:
                full_image_matrix = numpy.concatenate(
                    (full_image_matrix, this_image_matrix), axis=0)

            num_examples_in_memory_by_xclass, stopping_criterion = (
                _determine_stopping_criterion(
                    num_examples_per_batch=num_examples_per_batch,
                    num_init_times_per_batch=num_init_times_per_batch,
                    num_examples_per_batch_by_xclass=
                    num_examples_per_batch_by_xclass,
                    num_init_times_in_memory=num_init_times_in_memory,
                    xclass_fractions_to_sample=xclass_fractions_to_sample,
                    target_values_in_memory=all_target_values,
                    target_name=target_name))

        if xclass_fractions_to_sample is not None:
            batch_indices = dl_utils.sample_points_by_extended_class(
                target_values=all_target_values, target_name=target_name,
                extended_class_fractions=xclass_fractions_to_sample,
                num_points_to_sample=num_examples_per_batch)

            full_image_matrix = full_image_matrix[batch_indices, ...]
            all_target_values = all_target_values[batch_indices]
            if sounding_statistic_file_names is not None:
                full_sounding_stat_matrix = full_sounding_stat_matrix[
                    batch_indices, ...]

        full_image_matrix = dl_utils.normalize_predictor_matrix(
            predictor_matrix=full_image_matrix,
            normalize_by_batch=False, predictor_names=radar_field_names,
            normalization_dict=radar_normalization_dict)

        if sounding_statistic_file_names is not None:
            full_sounding_stat_matrix = dl_utils.normalize_sounding_statistics(
                sounding_stat_matrix=full_sounding_stat_matrix,
                statistic_names=sounding_statistic_names,
                metadata_table=sounding_stat_metadata_table,
                normalize_by_batch=False)

        list_of_image_matrices, target_matrix = _select_batch(
            list_of_image_matrices=[
                full_image_matrix, full_sounding_stat_matrix],
            target_values=all_target_values,
            num_examples_per_batch=num_examples_per_batch,
            binarize_target=binarize_target, num_classes=num_classes)

        image_matrix = list_of_image_matrices[0]
        sounding_stat_matrix = list_of_image_matrices[1]

        # Update housekeeping variables.
        full_image_matrix = None
        all_target_values = None
        full_sounding_stat_matrix = None
        num_init_times_in_memory = 0
        num_examples_in_memory_by_xclass = numpy.full(
            num_extended_classes, 0, dtype=int)

        if sounding_statistic_file_names is None:
            yield (image_matrix, target_matrix)
        else:
            yield ([image_matrix, sounding_stat_matrix], target_matrix)
