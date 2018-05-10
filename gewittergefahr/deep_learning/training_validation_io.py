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
import numpy
import keras
from gewittergefahr.deep_learning import storm_images
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import gridrad_utils
from gewittergefahr.gg_utils import labels
from gewittergefahr.gg_utils import error_checking


def _check_input_args(
        radar_source, num_examples_per_batch, num_examples_per_image_time,
        normalize_by_batch):
    """Error-checks input arguments to generator.

    :param radar_source: See documentation for `storm_image_generator_2d`.
    :param num_examples_per_batch: Same.
    :param num_examples_per_image_time: Same.
    :param normalize_by_batch: Same.
    """

    radar_utils.check_data_source(radar_source)
    error_checking.assert_is_integer(num_examples_per_batch)
    error_checking.assert_is_geq(num_examples_per_batch, 10)
    error_checking.assert_is_integer(num_examples_per_image_time)
    error_checking.assert_is_geq(num_examples_per_image_time, 2)
    error_checking.assert_is_boolean(normalize_by_batch)


def storm_image_generator_2d(
        top_directory_name, radar_source, radar_field_names,
        num_examples_per_batch, num_examples_per_image_time,
        first_image_time_unix_sec, last_image_time_unix_sec, target_name,
        radar_heights_m_asl=None, reflectivity_heights_m_asl=None,
        normalize_by_batch=False,
        normalization_dict=dl_utils.DEFAULT_NORMALIZATION_DICT,
        percentile_offset_for_normalization=
        dl_utils.DEFAULT_PERCENTILE_OFFSET_FOR_NORMALIZATION,
        class_fractions_to_sample=None):
    """Generates examples with 2-D storm-centered radar images.

    F = number of radar fields
    C = number of channels = num predictor variables = num field/height pairs

    :param top_directory_name: Name of top-level directory with storm-centered
        radar images.
    :param radar_source: Data source (must be accepted by
        `radar_utils.check_data_source`).
    :param radar_field_names: length-F list with names of radar fields.
    :param num_examples_per_batch: Number of examples per batch.
    :param num_examples_per_image_time: Number of examples per image time.
    :param first_image_time_unix_sec: First image time.  Examples will be
        created for random times in `first_image_time_unix_sec`...
        `last_image_time_unix_sec`.
    :param last_image_time_unix_sec: See above.
    :param target_name: Name of target variable.
    :param radar_heights_m_asl: [used only if radar_source = "gridrad"]
        1-D numpy array of radar heights (metres above sea level).  These apply
        to each field.
    :param reflectivity_heights_m_asl: [used only if radar_source != "gridrad"]
        1-D numpy array of heights (metres above sea level) for
        "reflectivity_dbz".  If "reflectivity_dbz" is not in the list
        `radar_field_names`, you can leave this as None.
    :param normalize_by_batch: Used to normalize predictor values (see doc for
        `deep_learning_utils.normalize_predictor_matrix`).
    :param normalization_dict: Same.
    :param percentile_offset_for_normalization: Same.
    :param class_fractions_to_sample: length-K numpy array with fraction of data
        points (examples) to keep from each class.  This can be used to achieve
        the desired class balance.  If you don't care about class balance, leave
        this as None.
    :return: predictor_matrix: E-by-M-by-N-by-C numpy array of storm-centered
        radar images.
    :return: target_matrix: E-by-K numpy array of target values (all 0 or 1, but
        technically the type is "float64").  If target_matrix[i, k] = 1, the
        [k]th class is the outcome for the [i]th example.  The sum across each
        row is 1 (classes are mutually exclusive and collectively exhaustive).
    """

    _check_input_args(
        radar_source=radar_source,
        num_examples_per_batch=num_examples_per_batch,
        num_examples_per_image_time=num_examples_per_image_time,
        normalize_by_batch=normalize_by_batch)

    # Find input files (containing storm-centered radar images).
    if radar_source == radar_utils.GRIDRAD_SOURCE_ID:
        image_file_name_matrix, _ = storm_images.find_many_files_gridrad(
            top_directory_name=top_directory_name,
            start_time_unix_sec=first_image_time_unix_sec,
            end_time_unix_sec=last_image_time_unix_sec,
            radar_field_names=radar_field_names,
            radar_heights_m_asl=radar_heights_m_asl,
            raise_error_if_missing=True)

        field_name_by_predictor, _ = (
            gridrad_utils.fields_and_refl_heights_to_pairs(
                field_names=radar_field_names,
                heights_m_asl=radar_heights_m_asl))

        num_image_times = image_file_name_matrix.shape[0]
        num_predictors = len(field_name_by_predictor)
        image_file_name_matrix = numpy.reshape(
            image_file_name_matrix, (num_image_times, num_predictors))

    else:
        image_file_name_matrix, _, field_name_by_predictor, _ = (
            storm_images.find_many_files_myrorss_or_mrms(
                top_directory_name=top_directory_name,
                start_time_unix_sec=first_image_time_unix_sec,
                end_time_unix_sec=last_image_time_unix_sec,
                radar_source=radar_source, radar_field_names=radar_field_names,
                reflectivity_heights_m_asl=reflectivity_heights_m_asl,
                raise_error_if_missing=True))

    # Remove any time step with one or more missing files.
    time_missing_indices = numpy.unique(
        numpy.where(image_file_name_matrix == '')[0])
    image_file_name_matrix = numpy.delete(
        image_file_name_matrix, time_missing_indices, axis=0)

    num_image_times = image_file_name_matrix.shape[0]
    num_predictors = len(field_name_by_predictor)

    # Shuffle files by time.
    image_time_indices = numpy.linspace(
        0, num_image_times - 1, num=num_image_times, dtype=int)
    numpy.random.shuffle(image_time_indices)
    image_file_name_matrix = image_file_name_matrix[image_time_indices, ...]

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
    image_time_index = 0
    num_image_times_in_memory = 0
    num_examples_in_memory_by_class = numpy.full(num_classes, 0, dtype=int)
    num_image_times_per_batch = int(numpy.ceil(
        float(num_examples_per_batch) / num_examples_per_image_time))

    full_predictor_matrix = None
    all_target_values = None

    while True:
        stopping_criterion = False

        while not stopping_criterion:
            print '\n'
            tuple_of_predictor_matrices = ()

            # Read images for the [0]th predictor at the [i]th time (where i =
            # image_time_index).
            print 'Reading data from: "{0:s}"...'.format(
                image_file_name_matrix[image_time_index, 0])

            this_label_file_name = storm_images.find_storm_label_file(
                storm_image_file_name=
                image_file_name_matrix[image_time_index, 0],
                raise_error_if_missing=True)

            num_examples_needed_by_class = (num_examples_per_batch_by_class -
                                            num_examples_in_memory_by_class)
            num_examples_needed_by_class[num_examples_needed_by_class < 0] = 0

            this_storm_image_dict = storm_images.read_storm_images_and_labels(
                image_file_name=image_file_name_matrix[image_time_index, 0],
                label_file_name=this_label_file_name,
                return_label_name=target_name,
                num_storm_objects_by_class=num_examples_needed_by_class)

            if this_storm_image_dict is None:
                image_time_index += 1
                if image_time_index >= num_image_times:
                    image_time_index = 0
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

            for j in range(num_predictors):
                if j != 0:

                    # Read images for the [j]th predictor at the [i]th time
                    # (where i = image_time_index).
                    print 'Reading data from: "{0:s}"...'.format(
                        image_file_name_matrix[image_time_index, j])
                    this_storm_image_dict = storm_images.read_storm_images_only(
                        netcdf_file_name=
                        image_file_name_matrix[image_time_index, j],
                        indices_to_keep=None)

                this_field_predictor_matrix = this_storm_image_dict[
                    storm_images.STORM_IMAGE_MATRIX_KEY][
                        these_valid_storm_indices, ...]
                tuple_of_predictor_matrices += (this_field_predictor_matrix,)

            # Housekeeping.
            num_image_times_in_memory += 1
            image_time_index += 1
            if image_time_index >= num_image_times:
                image_time_index = 0

            # Add images from [i]th time (where i = image_time_index) to
            # full_predictor_matrix, which contains radar images for all times.
            this_predictor_matrix = dl_utils.stack_predictor_variables(
                tuple_of_predictor_matrices)

            if full_predictor_matrix is None:
                full_predictor_matrix = copy.deepcopy(this_predictor_matrix)
            else:
                full_predictor_matrix = numpy.concatenate(
                    (full_predictor_matrix, this_predictor_matrix), axis=0)

            # Determine stopping criterion.
            num_examples_in_memory_by_class = numpy.array(
                [numpy.sum(all_target_values == k) for k in range(num_classes)],
                dtype=int)
            print 'Number of examples by class: {0:s}'.format(
                str(num_examples_in_memory_by_class))

            stopping_criterion = (
                num_image_times_in_memory >= num_image_times_per_batch and
                full_predictor_matrix.shape[0] >= num_examples_per_batch and
                (class_fractions_to_sample is None or numpy.all(
                    num_examples_in_memory_by_class >=
                    num_examples_per_batch_by_class)))

        # Downsample data.
        if class_fractions_to_sample is not None:
            batch_indices = dl_utils.sample_points_by_class(
                target_values=all_target_values,
                class_fractions=class_fractions_to_sample,
                num_points_to_sample=num_examples_per_batch)

            full_predictor_matrix = full_predictor_matrix[
                batch_indices, ...].astype('float32')
            all_target_values = all_target_values[batch_indices]

        # Normalize images.
        full_predictor_matrix = dl_utils.normalize_predictor_matrix(
            predictor_matrix=full_predictor_matrix,
            normalize_by_batch=normalize_by_batch,
            predictor_names=field_name_by_predictor,
            normalization_dict=normalization_dict,
            percentile_offset=percentile_offset_for_normalization)

        # Randomly select E examples (where E = num_examples_per_batch).
        num_examples = full_predictor_matrix.shape[0]
        example_indices = numpy.linspace(
            0, num_examples - 1, num=num_examples, dtype=int)
        batch_indices = numpy.random.choice(
            example_indices, size=num_examples_per_batch, replace=False)

        predictor_matrix = full_predictor_matrix[
            batch_indices, ...].astype('float32')
        target_values = all_target_values[batch_indices]

        # Housekeeping.
        full_predictor_matrix = None
        all_target_values = None
        num_image_times_in_memory = 0
        num_examples_in_memory_by_class = numpy.full(num_classes, 0, dtype=int)

        # Turn 1-D array of target values into 2-D Boolean matrix.
        target_matrix = keras.utils.to_categorical(target_values, num_classes)
        class_fractions = numpy.mean(target_matrix, axis=0)
        print 'Fraction of target values in each class:\n{0:s}\n'.format(
            str(class_fractions))

        yield (predictor_matrix, target_matrix)


def storm_image_generator_3d(
        top_directory_name, radar_source, radar_heights_m_asl,
        num_examples_per_batch, num_examples_per_image_time,
        first_image_time_unix_sec, last_image_time_unix_sec, target_name,
        radar_field_names=None, normalize_by_batch=False,
        normalization_dict=dl_utils.DEFAULT_NORMALIZATION_DICT,
        percentile_offset_for_normalization=
        dl_utils.DEFAULT_PERCENTILE_OFFSET_FOR_NORMALIZATION,
        class_fractions_to_sample=None):
    """Generates examples with 3-D storm-centered radar images.

    F = number of radar fields
    C = number of channels = num predictor variables = num field/height pairs

    :param top_directory_name: See documentation for `storm_image_generator_3d`.
    :param radar_source: Same.
    :param radar_heights_m_asl: 1-D numpy array of radar heights (metres above
        sea level).  These apply to each field.
    :param num_examples_per_batch: See doc for `storm_image_generator_3d`.
    :param num_examples_per_image_time: Same.
    :param first_image_time_unix_sec: Same.
    :param last_image_time_unix_sec: Same.
    :param target_name: Same.
    :param radar_field_names: If radar_source = "myrorss" or "mrms", this
        argument is not used, because the only field available for 3-D images is
        "reflectivity_dbz".  If radar_source = "gridrad", this is a length-F
        list with names of radar fields.
    :param normalize_by_batch: Same.
    :param normalization_dict: Same.
    :param percentile_offset_for_normalization: Same.
    :param class_fractions_to_sample: Same.
    :return: predictor_matrix: E-by-M-by-N-by-D-by-C numpy array of
        storm-centered radar images.
    :return: target_matrix: See doc for `storm_image_generator_3d`.
    """

    _check_input_args(
        radar_source=radar_source,
        num_examples_per_batch=num_examples_per_batch,
        num_examples_per_image_time=num_examples_per_image_time,
        normalize_by_batch=normalize_by_batch)

    # Find input files (containing storm-centered radar images).
    if radar_source == radar_utils.GRIDRAD_SOURCE_ID:
        image_file_name_matrix, _ = storm_images.find_many_files_gridrad(
            top_directory_name=top_directory_name,
            start_time_unix_sec=first_image_time_unix_sec,
            end_time_unix_sec=last_image_time_unix_sec,
            radar_field_names=radar_field_names,
            radar_heights_m_asl=radar_heights_m_asl,
            raise_error_if_missing=True)

    else:
        radar_field_names = [radar_utils.REFL_NAME]

        image_file_name_matrix, _, _, _ = (
            storm_images.find_many_files_myrorss_or_mrms(
                top_directory_name=top_directory_name,
                start_time_unix_sec=first_image_time_unix_sec,
                end_time_unix_sec=last_image_time_unix_sec,
                radar_source=radar_source, radar_field_names=radar_field_names,
                reflectivity_heights_m_asl=radar_heights_m_asl,
                raise_error_if_missing=True))

        num_heights = len(radar_heights_m_asl)
        num_image_times = image_file_name_matrix.shape[0]
        image_file_name_matrix = numpy.reshape(
            image_file_name_matrix, (num_image_times, 1, num_heights))

    # Remove any time step with one or more missing files.
    time_missing_indices = numpy.unique(
        numpy.where(image_file_name_matrix == '')[0])
    image_file_name_matrix = numpy.delete(
        image_file_name_matrix, time_missing_indices, axis=0)

    num_image_times = image_file_name_matrix.shape[0]
    num_fields = len(radar_field_names)
    num_heights = len(radar_heights_m_asl)

    # Shuffle files by time.
    image_time_indices = numpy.linspace(
        0, num_image_times - 1, num=num_image_times, dtype=int)
    numpy.random.shuffle(image_time_indices)
    image_file_name_matrix = image_file_name_matrix[image_time_indices, ...]

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
    image_time_index = 0
    num_image_times_in_memory = 0
    num_examples_in_memory_by_class = numpy.full(num_classes, 0, dtype=int)
    num_image_times_per_batch = int(numpy.ceil(
        float(num_examples_per_batch) / num_examples_per_image_time))

    full_predictor_matrix = None
    all_target_values = None

    while True:
        stopping_criterion = False

        while not stopping_criterion:
            print '\n'
            tuple_of_4d_predictor_matrices = ()

            # Read images for the [0]th predictor at the [0]th height and [i]th
            # time (where i = image_time_index).
            print 'Reading data from: "{0:s}"...'.format(
                image_file_name_matrix[image_time_index, 0, 0])

            this_label_file_name = storm_images.find_storm_label_file(
                storm_image_file_name=
                image_file_name_matrix[image_time_index, 0, 0],
                raise_error_if_missing=True)

            num_examples_needed_by_class = (num_examples_per_batch_by_class -
                                            num_examples_in_memory_by_class)
            num_examples_needed_by_class[num_examples_needed_by_class < 0] = 0

            this_storm_image_dict = storm_images.read_storm_images_and_labels(
                image_file_name=image_file_name_matrix[image_time_index, 0, 0],
                label_file_name=this_label_file_name,
                return_label_name=target_name,
                num_storm_objects_by_class=num_examples_needed_by_class)

            if this_storm_image_dict is None:
                image_time_index += 1
                if image_time_index >= num_image_times:
                    image_time_index = 0
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

            for k in range(num_heights):
                tuple_of_3d_predictor_matrices = ()

                for j in range(num_fields):
                    if not j == k == 0:

                        # Read images for the [j]th predictor at the [k]th
                        # height and [i]th time (where i = image_time_index).
                        print 'Reading data from: "{0:s}"...'.format(
                            image_file_name_matrix[image_time_index, j, k])
                        this_storm_image_dict = (
                            storm_images.read_storm_images_only(
                                netcdf_file_name=
                                image_file_name_matrix[image_time_index, j, k],
                                indices_to_keep=None))

                    this_3d_predictor_matrix = this_storm_image_dict[
                        storm_images.STORM_IMAGE_MATRIX_KEY][
                            these_valid_storm_indices, ...]
                    tuple_of_3d_predictor_matrices += (
                        this_3d_predictor_matrix,)

                tuple_of_4d_predictor_matrices += (
                    dl_utils.stack_predictor_variables(
                        tuple_of_3d_predictor_matrices),)

            # Housekeeping.
            num_image_times_in_memory += 1
            image_time_index += 1
            if image_time_index >= num_image_times:
                image_time_index = 0

            # Add images from [i]th time (where i = image_time_index) to
            # full_predictor_matrix, which contains radar images for all times.
            this_predictor_matrix = dl_utils.stack_heights(
                tuple_of_4d_predictor_matrices)

            if full_predictor_matrix is None:
                full_predictor_matrix = copy.deepcopy(this_predictor_matrix)
            else:
                full_predictor_matrix = numpy.concatenate(
                    (full_predictor_matrix, this_predictor_matrix), axis=0)

            # Determine stopping criterion.
            num_examples_in_memory_by_class = numpy.array(
                [numpy.sum(all_target_values == k) for k in range(num_classes)],
                dtype=int)
            print 'Number of examples by class: {0:s}'.format(
                str(num_examples_in_memory_by_class))

            stopping_criterion = (
                num_image_times_in_memory >= num_image_times_per_batch and
                full_predictor_matrix.shape[0] >= num_examples_per_batch and
                (class_fractions_to_sample is None or numpy.all(
                    num_examples_in_memory_by_class >=
                    num_examples_per_batch_by_class)))

        # Downsample data.
        if class_fractions_to_sample is not None:
            batch_indices = dl_utils.sample_points_by_class(
                target_values=all_target_values,
                class_fractions=class_fractions_to_sample,
                num_points_to_sample=num_examples_per_batch)

            full_predictor_matrix = full_predictor_matrix[
                batch_indices, ...].astype('float32')
            all_target_values = all_target_values[batch_indices]

        # Normalize images.
        full_predictor_matrix = dl_utils.normalize_predictor_matrix(
            predictor_matrix=full_predictor_matrix,
            normalize_by_batch=normalize_by_batch,
            predictor_names=radar_field_names,
            normalization_dict=normalization_dict,
            percentile_offset=percentile_offset_for_normalization)

        # Randomly select E examples (where E = num_examples_per_batch).
        num_examples = full_predictor_matrix.shape[0]
        example_indices = numpy.linspace(
            0, num_examples - 1, num=num_examples, dtype=int)
        batch_indices = numpy.random.choice(
            example_indices, size=num_examples_per_batch, replace=False)

        predictor_matrix = full_predictor_matrix[
            batch_indices, ...].astype('float32')
        target_values = all_target_values[batch_indices]

        # Housekeeping.
        full_predictor_matrix = None
        all_target_values = None
        num_image_times_in_memory = 0
        num_examples_in_memory_by_class = numpy.full(num_classes, 0, dtype=int)

        # Turn 1-D array of target values into 2-D Boolean matrix.
        target_matrix = keras.utils.to_categorical(target_values, num_classes)
        class_fractions = numpy.mean(target_matrix, axis=0)
        print 'Fraction of target values in each class:\n{0:s}\n'.format(
            str(class_fractions))

        yield (predictor_matrix, target_matrix)
