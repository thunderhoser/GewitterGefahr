"""IO methods for training or validation of a deep-learning model.

--- NOTATION ---

In this module, the following letters will be used to denote matrix dimensions.

K = number of classes (possible values of target variable)
E = number of examples
M = number of pixel rows per image
N = number of pixel columns per image
D = number of pixel depths per image
C = number of channels (predictor variables) per image
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


def _check_input_args(
        num_examples_per_batch, num_examples_per_init_time, normalize_by_batch,
        image_file_name_matrix, sounding_statistic_file_names=None,
        sounding_statistic_names=None):
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

    error_checking.assert_is_numpy_array(image_file_name_matrix)
    num_dimensions = image_file_name_matrix.shape
    error_checking.assert_is_geq(num_dimensions, 2)
    error_checking.assert_is_leq(num_dimensions, 2)

    if sounding_statistic_file_names is not None:
        num_init_times = image_file_name_matrix.shape[0]
        error_checking.assert_is_numpy_array(
            numpy.array(sounding_statistic_file_names),
            exact_dimensions=numpy.array([num_init_times]))

        if sounding_statistic_file_names is not None:
            error_checking.assert_is_numpy_array(
                numpy.array(sounding_statistic_names), num_dimensions=1)


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
                raise_error_if_missing=True))

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
                raise_error_if_missing=True))

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
                raise_error_if_missing=True))

    else:
        radar_field_names = [radar_utils.REFL_NAME]

        image_file_name_matrix, image_times_unix_sec, _, _ = (
            storm_images.find_many_files_myrorss_or_mrms(
                top_directory_name=top_directory_name,
                start_time_unix_sec=first_image_time_unix_sec,
                end_time_unix_sec=last_image_time_unix_sec,
                radar_source=radar_source, radar_field_names=radar_field_names,
                reflectivity_heights_m_asl=radar_heights_m_asl,
                raise_error_if_missing=True))

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

    # TODO(thunderhoser): Add `raise_error_if_any_missing` to methods that find
    # image files.

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
        num_examples_per_init_time, target_name,
        sounding_statistic_file_names=None, sounding_statistic_names=None,
        normalize_by_batch=False,
        normalization_dict=dl_utils.DEFAULT_NORMALIZATION_DICT,
        percentile_offset_for_normalization=
        dl_utils.DEFAULT_PERCENTILE_OFFSET_FOR_NORMALIZATION,
        class_fractions_to_sample=None):
    """Generates examples with 2-D radar images.

    Each example consists of a 2-D radar image, and possibly sounding
    statistics, for one storm object.

    T = number of storm times (initial times, not valid times)
    F = number of radar fields
    C = number of channels = num predictor variables = num field/height pairs
    S = number of sounding statistics

    :param image_file_name_matrix: T-by-C numpy array of paths to radar-image
        files.  This should be created by `find_2d_input_files`.
    :param num_examples_per_batch: See documentation for `_check_input_args`.
    :param num_examples_per_init_time: Same.
    :param target_name: Name of target variable.
    :param sounding_statistic_file_names: See doc for `_check_input_args`.
    :param sounding_statistic_names: Same.
    :param normalize_by_batch: Used to normalize predictor values (see doc for
        `deep_learning_utils.normalize_predictor_matrix`).
    :param normalization_dict: Same.
    :param percentile_offset_for_normalization: Same.
    :param class_fractions_to_sample: length-K numpy array used for class-
        conditional sampling.  class_fractions_to_sample[k] is the fraction of
        examples from the [k]th class to be returned in each batch.

    If `sounding_statistic_file_names is None`, this method returns the
    following.

    :return: image_matrix: E-by-M-by-N-by-C numpy array of storm-centered radar
        images.
    :return: target_matrix: E-by-K numpy array of target values (all 0 or 1, but
        technically the type is "float64").  If target_matrix[i, k] = 1, the
        [k]th class is the outcome for the [i]th example.  The sum across each
        row is 1 (classes are mutually exclusive and collectively exhaustive).

    If `sounding_statistic_file_names is not None`, this method returns the
    following.

    :return: predictor_list: List with the following items.
    predictor_list[0] = image_matrix: See documentation above.
    predictor_list[1] = sounding_stat_matrix: E-by-S numpy array of sounding
        statistics.
    :return: target_matrix: See documentation above.
    """

    # Check input arguments.
    _check_input_args(
        num_examples_per_batch=num_examples_per_batch,
        num_examples_per_init_time=num_examples_per_init_time,
        normalize_by_batch=normalize_by_batch,
        image_file_name_matrix=image_file_name_matrix,
        sounding_statistic_file_names=sounding_statistic_file_names,
        sounding_statistic_names=sounding_statistic_names)

    # Find names of radar fields.
    num_channels = image_file_name_matrix.shape[1]
    field_name_by_channel = [''] * num_channels

    for j in range(num_channels):
        this_storm_image_dict = storm_images.read_storm_images_only(
            image_file_name_matrix[0, j])
        field_name_by_channel[j] = str(
            this_storm_image_dict[storm_images.RADAR_FIELD_NAME_KEY])

    # Shuffle files by time.
    num_init_times = image_file_name_matrix.shape[0]
    init_time_indices = numpy.linspace(
        0, num_init_times - 1, num=num_init_times, dtype=int)
    numpy.random.shuffle(init_time_indices)

    image_file_name_matrix = image_file_name_matrix[init_time_indices, ...]
    if sounding_statistic_file_names is not None:
        sounding_statistic_file_names = [
            sounding_statistic_file_names[i] for i in init_time_indices]

    # Determine number of examples needed per class.
    num_classes = labels.column_name_to_num_classes(target_name)
    if class_fractions_to_sample is None:
        num_examples_per_batch_by_class = numpy.full(
            num_classes, 1e10, dtype=int)
    else:
        num_examples_per_batch_by_class = (
            dl_utils.class_fractions_to_num_points(
                class_fractions=class_fractions_to_sample,
                num_points_to_sample=num_examples_per_batch))

        error_checking.assert_is_numpy_array(
            num_examples_per_batch_by_class,
            exact_dimensions=numpy.array([num_classes]))

    # Initialize variables.
    init_time_index = 0
    num_init_times_in_memory = 0
    num_examples_in_memory_by_class = numpy.full(num_classes, 0, dtype=int)
    num_init_times_per_batch = int(numpy.ceil(
        float(num_examples_per_batch) / num_examples_per_init_time))

    full_image_matrix = None
    all_target_values = None
    full_sounding_stat_matrix = None

    while True:
        stopping_criterion = False

        while not stopping_criterion:
            print '\n'
            tuple_of_image_matrices = ()

            # Read images for the [0]th channel at the [i]th time (where i =
            # init_time_index).
            print 'Reading data from: "{0:s}"...'.format(
                image_file_name_matrix[init_time_index, 0])

            this_label_file_name = storm_images.find_storm_label_file(
                storm_image_file_name=image_file_name_matrix[
                    init_time_index, 0],
                raise_error_if_missing=True)

            downsampling_criterion = (
                num_init_times_in_memory >= num_init_times_per_batch and
                full_image_matrix is not None and
                full_image_matrix.shape[0] >= num_examples_per_batch)

            if downsampling_criterion:
                num_examples_needed_by_class = (
                    num_examples_per_batch_by_class -
                    num_examples_in_memory_by_class)
                num_examples_needed_by_class[
                    num_examples_needed_by_class < 0] = 0
            else:
                num_examples_needed_by_class = None

            this_storm_image_dict = storm_images.read_storm_images_and_labels(
                image_file_name=image_file_name_matrix[init_time_index, 0],
                label_file_name=this_label_file_name,
                return_label_name=target_name,
                num_storm_objects_by_class=num_examples_needed_by_class)

            if this_storm_image_dict is None:
                init_time_index += 1
                if init_time_index >= num_init_times:
                    init_time_index = 0
                continue

            these_target_values = this_storm_image_dict[
                storm_images.LABEL_VALUES_KEY]
            these_valid_storm_indices = numpy.where(these_target_values >= 0)[0]
            these_target_values = these_target_values[these_valid_storm_indices]

            if all_target_values is None:
                all_target_values = copy.deepcopy(these_target_values)
            else:
                all_target_values = numpy.concatenate((
                    all_target_values, these_target_values))

            if sounding_statistic_file_names is not None:
                this_sounding_stat_dict = soundings.read_sounding_statistics(
                    netcdf_file_name=sounding_statistic_file_names[
                        init_time_index],
                    storm_ids_to_keep=this_storm_image_dict[
                        storm_images.STORM_IDS_KEY],
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
                    # Read images for the [j]th channel at the [i]th time (where
                    # i = init_time_index).
                    print 'Reading data from: "{0:s}"...'.format(
                        image_file_name_matrix[init_time_index, j])
                    this_storm_image_dict = storm_images.read_storm_images_only(
                        netcdf_file_name=image_file_name_matrix[
                            init_time_index, j],
                        indices_to_keep=None)

                this_field_image_matrix = this_storm_image_dict[
                    storm_images.STORM_IMAGE_MATRIX_KEY
                ][these_valid_storm_indices, ...]
                tuple_of_image_matrices += (this_field_image_matrix,)

            # Housekeeping.
            num_init_times_in_memory += 1
            init_time_index += 1
            if init_time_index >= num_init_times:
                init_time_index = 0

            # Add images from [i]th time (where i = init_time_index) to
            # full_image_matrix, which contains radar images for all times.
            this_image_matrix = dl_utils.stack_predictor_variables(
                tuple_of_image_matrices)

            if full_image_matrix is None:
                full_image_matrix = copy.deepcopy(this_image_matrix)
            else:
                full_image_matrix = numpy.concatenate(
                    (full_image_matrix, this_image_matrix), axis=0)

            # Determine stopping criterion.
            num_examples_in_memory_by_class = numpy.array(
                [numpy.sum(all_target_values == k) for k in range(num_classes)],
                dtype=int)
            print 'Number of examples by class: {0:s}'.format(
                str(num_examples_in_memory_by_class))

            stopping_criterion = (
                num_init_times_in_memory >= num_init_times_per_batch and
                full_image_matrix.shape[0] >= num_examples_per_batch and
                (class_fractions_to_sample is None or numpy.all(
                    num_examples_in_memory_by_class >=
                    num_examples_per_batch_by_class)))

        # Downsample data.
        if class_fractions_to_sample is not None:
            batch_indices = dl_utils.sample_points_by_class(
                target_values=all_target_values,
                class_fractions=class_fractions_to_sample,
                num_points_to_sample=num_examples_per_batch)

            full_image_matrix = full_image_matrix[batch_indices, ...].astype(
                'float32')
            all_target_values = all_target_values[batch_indices]

        # Normalize images.
        full_image_matrix = dl_utils.normalize_predictor_matrix(
            predictor_matrix=full_image_matrix,
            normalize_by_batch=normalize_by_batch,
            predictor_names=field_name_by_channel,
            normalization_dict=normalization_dict,
            percentile_offset=percentile_offset_for_normalization)

        # Randomly select E examples (where E = num_examples_per_batch).
        num_examples = full_image_matrix.shape[0]
        example_indices = numpy.linspace(
            0, num_examples - 1, num=num_examples, dtype=int)
        batch_indices = numpy.random.choice(
            example_indices, size=num_examples_per_batch, replace=False)

        image_matrix = full_image_matrix[batch_indices, ...].astype('float32')
        target_values = all_target_values[batch_indices]
        if sounding_statistic_file_names is not None:
            sounding_stat_matrix = full_sounding_stat_matrix[
                batch_indices, ...].astype('float32')

        # Housekeeping.
        full_image_matrix = None
        all_target_values = None
        full_sounding_stat_matrix = None
        num_init_times_in_memory = 0
        num_examples_in_memory_by_class = numpy.full(num_classes, 0, dtype=int)

        # Turn 1-D array of target values into 2-D Boolean matrix.
        target_matrix = keras.utils.to_categorical(target_values, num_classes)
        class_fractions = numpy.mean(target_matrix, axis=0)
        print 'Fraction of target values in each class:\n{0:s}\n'.format(
            str(class_fractions))

        if sounding_statistic_file_names is None:
            yield (image_matrix, target_matrix)
        else:
            yield ([image_matrix, sounding_stat_matrix], target_matrix)


def storm_image_generator_3d(
        image_file_name_matrix, num_examples_per_batch,
        num_examples_per_init_time, target_name,
        sounding_statistic_file_names=None, sounding_statistic_names=None,
        normalize_by_batch=False,
        normalization_dict=dl_utils.DEFAULT_NORMALIZATION_DICT,
        percentile_offset_for_normalization=
        dl_utils.DEFAULT_PERCENTILE_OFFSET_FOR_NORMALIZATION,
        class_fractions_to_sample=None):
    """Generates examples with 3-D radar images.

    Each example consists of a 3-D radar image, and possibly sounding
    statistics, for one storm object.

    T = number of storm times (initial times, not valid times)
    F = number of radar fields
    D = number of radar heights
    S = number of sounding statistics

    :param image_file_name_matrix: T-by-F-by-H numpy array of paths to radar-
        image files.  This should be created by `find_3d_input_files`.
    :param num_examples_per_batch: See documentation for `_check_input_args`.
    :param num_examples_per_init_time: Same.
    :param target_name: Name of target variable.
    :param sounding_statistic_file_names: See doc for `_check_input_args`.
    :param sounding_statistic_names: Same.
    :param normalize_by_batch: Used to normalize predictor values (see doc for
        `deep_learning_utils.normalize_predictor_matrix`).
    :param normalization_dict: Same.
    :param percentile_offset_for_normalization: Same.
    :param class_fractions_to_sample: length-K numpy array used for class-
        conditional sampling.  class_fractions_to_sample[k] is the fraction of
        examples from the [k]th class to be returned in each batch.

    If `sounding_statistic_file_names is None`, this method returns the
    following.

    :return: image_matrix: E-by-M-by-N-by-D-by-C numpy array of storm-centered
        radar images.
    :return: target_matrix: E-by-K numpy array of target values (all 0 or 1, but
        technically the type is "float64").  If target_matrix[i, k] = 1, the
        [k]th class is the outcome for the [i]th example.  The sum across each
        row is 1 (classes are mutually exclusive and collectively exhaustive).

    If `sounding_statistic_file_names is not None`, this method returns the
    following.

    :return: predictor_list: List with the following items.
    predictor_list[0] = image_matrix: See documentation above.
    predictor_list[1] = sounding_stat_matrix: E-by-S numpy array of sounding
        statistics.
    :return: target_matrix: See documentation above.
    """

    # Check input arguments.
    _check_input_args(
        num_examples_per_batch=num_examples_per_batch,
        num_examples_per_init_time=num_examples_per_init_time,
        normalize_by_batch=normalize_by_batch,
        image_file_name_matrix=image_file_name_matrix,
        sounding_statistic_file_names=sounding_statistic_file_names,
        sounding_statistic_names=sounding_statistic_names)

    # Find names of radar fields.
    num_fields = image_file_name_matrix.shape[1]
    num_heights = image_file_name_matrix.shape[2]
    radar_field_names = [''] * num_fields

    for j in range(num_fields):
        this_storm_image_dict = storm_images.read_storm_images_only(
            image_file_name_matrix[0, j, 0])
        radar_field_names[j] = str(
            this_storm_image_dict[storm_images.RADAR_FIELD_NAME_KEY])

    # Shuffle files by time.
    num_init_times = image_file_name_matrix.shape[0]
    init_time_indices = numpy.linspace(
        0, num_init_times - 1, num=num_init_times, dtype=int)
    numpy.random.shuffle(init_time_indices)

    image_file_name_matrix = image_file_name_matrix[init_time_indices, ...]
    if sounding_statistic_file_names is not None:
        sounding_statistic_file_names = [
            sounding_statistic_file_names[i] for i in init_time_indices]

    # Determine number of examples needed per class.
    num_classes = labels.column_name_to_num_classes(target_name)
    if class_fractions_to_sample is None:
        num_examples_per_batch_by_class = numpy.full(
            num_classes, 1e10, dtype=int)
    else:
        num_examples_per_batch_by_class = (
            dl_utils.class_fractions_to_num_points(
                class_fractions=class_fractions_to_sample,
                num_points_to_sample=num_examples_per_batch))

        error_checking.assert_is_numpy_array(
            num_examples_per_batch_by_class,
            exact_dimensions=numpy.array([num_classes]))

    # Initialize variables.
    init_time_index = 0
    num_init_times_in_memory = 0
    num_examples_in_memory_by_class = numpy.full(num_classes, 0, dtype=int)
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

            # Read images for the [0]th field at the [0]th height and [i]th time
            # (where i = init_time_index).
            print 'Reading data from: "{0:s}"...'.format(
                image_file_name_matrix[init_time_index, 0, 0])

            this_label_file_name = storm_images.find_storm_label_file(
                storm_image_file_name=image_file_name_matrix[
                    init_time_index, 0, 0],
                raise_error_if_missing=True)

            downsampling_criterion = (
                num_init_times_in_memory >= num_init_times_per_batch and
                full_image_matrix is not None and
                full_image_matrix.shape[0] >= num_examples_per_batch)

            if downsampling_criterion:
                num_examples_needed_by_class = (
                    num_examples_per_batch_by_class -
                    num_examples_in_memory_by_class)
                num_examples_needed_by_class[
                    num_examples_needed_by_class < 0] = 0
            else:
                num_examples_needed_by_class = None

            this_storm_image_dict = storm_images.read_storm_images_and_labels(
                image_file_name=image_file_name_matrix[init_time_index, 0, 0],
                label_file_name=this_label_file_name,
                return_label_name=target_name,
                num_storm_objects_by_class=num_examples_needed_by_class)

            if this_storm_image_dict is None:
                init_time_index += 1
                if init_time_index >= num_init_times:
                    init_time_index = 0
                continue

            these_target_values = this_storm_image_dict[
                storm_images.LABEL_VALUES_KEY]
            these_valid_storm_indices = numpy.where(these_target_values >= 0)[0]
            these_target_values = these_target_values[these_valid_storm_indices]

            if all_target_values is None:
                all_target_values = copy.deepcopy(these_target_values)
            else:
                all_target_values = numpy.concatenate((
                    all_target_values, these_target_values))

            if sounding_statistic_file_names is not None:
                this_sounding_stat_dict = soundings.read_sounding_statistics(
                    netcdf_file_name=sounding_statistic_file_names[
                        init_time_index],
                    storm_ids_to_keep=this_storm_image_dict[
                        storm_images.STORM_IDS_KEY],
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
                        # Read images for the [j]th field at the [k]th height
                        # and [i]th time (where i = init_time_index).
                        print 'Reading data from: "{0:s}"...'.format(
                            image_file_name_matrix[init_time_index, j, k])
                        this_storm_image_dict = (
                            storm_images.read_storm_images_only(
                                netcdf_file_name=
                                image_file_name_matrix[init_time_index, j, k],
                                indices_to_keep=None))

                    this_3d_image_matrix = this_storm_image_dict[
                        storm_images.STORM_IMAGE_MATRIX_KEY][
                            these_valid_storm_indices, ...]
                    tuple_of_3d_image_matrices += (this_3d_image_matrix,)

                tuple_of_4d_image_matrices += (
                    dl_utils.stack_predictor_variables(
                        tuple_of_3d_image_matrices),)

            # Housekeeping.
            num_init_times_in_memory += 1
            init_time_index += 1
            if init_time_index >= num_init_times:
                init_time_index = 0

            # Add images from [i]th time (where i = init_time_index) to
            # full_image_matrix, which contains radar images for all times.
            this_image_matrix = dl_utils.stack_heights(
                tuple_of_4d_image_matrices)

            if full_image_matrix is None:
                full_image_matrix = copy.deepcopy(this_image_matrix)
            else:
                full_image_matrix = numpy.concatenate(
                    (full_image_matrix, this_image_matrix), axis=0)

            # Determine stopping criterion.
            num_examples_in_memory_by_class = numpy.array(
                [numpy.sum(all_target_values == k) for k in range(num_classes)],
                dtype=int)
            print 'Number of examples by class: {0:s}'.format(
                str(num_examples_in_memory_by_class))

            stopping_criterion = (
                num_init_times_in_memory >= num_init_times_per_batch and
                full_image_matrix.shape[0] >= num_examples_per_batch and
                (class_fractions_to_sample is None or numpy.all(
                    num_examples_in_memory_by_class >=
                    num_examples_per_batch_by_class)))

        # Downsample data.
        if class_fractions_to_sample is not None:
            batch_indices = dl_utils.sample_points_by_class(
                target_values=all_target_values,
                class_fractions=class_fractions_to_sample,
                num_points_to_sample=num_examples_per_batch)

            full_image_matrix = full_image_matrix[batch_indices, ...].astype(
                'float32')
            all_target_values = all_target_values[batch_indices]

        # Normalize images.
        full_image_matrix = dl_utils.normalize_predictor_matrix(
            predictor_matrix=full_image_matrix,
            normalize_by_batch=normalize_by_batch,
            predictor_names=radar_field_names,
            normalization_dict=normalization_dict,
            percentile_offset=percentile_offset_for_normalization)

        # Randomly select E examples (where E = num_examples_per_batch).
        num_examples = full_image_matrix.shape[0]
        example_indices = numpy.linspace(
            0, num_examples - 1, num=num_examples, dtype=int)
        batch_indices = numpy.random.choice(
            example_indices, size=num_examples_per_batch, replace=False)

        image_matrix = full_image_matrix[batch_indices, ...].astype('float32')
        target_values = all_target_values[batch_indices]
        if sounding_statistic_file_names is not None:
            sounding_stat_matrix = full_sounding_stat_matrix[
                batch_indices, ...].astype('float32')

        # Housekeeping.
        full_image_matrix = None
        all_target_values = None
        full_sounding_stat_matrix = None
        num_init_times_in_memory = 0
        num_examples_in_memory_by_class = numpy.full(num_classes, 0, dtype=int)

        # Turn 1-D array of target values into 2-D Boolean matrix.
        target_matrix = keras.utils.to_categorical(target_values, num_classes)
        class_fractions = numpy.mean(target_matrix, axis=0)
        print 'Fraction of target values in each class:\n{0:s}\n'.format(
            str(class_fractions))

        if sounding_statistic_file_names is None:
            yield (image_matrix, target_matrix)
        else:
            yield ([image_matrix, sounding_stat_matrix], target_matrix)
