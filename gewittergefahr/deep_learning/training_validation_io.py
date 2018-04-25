"""IO methods for training or validation of a deep-learning model.

--- NOTATION ---

In this module, the following letters will be used to denote matrix dimensions.

K = number of classes (possible values of target variable)
E = number of examples
M = number of pixel rows per image
N = number of pixel columns per image
P = number of channels (predictor variables) per image
"""

import numpy
import keras
from gewittergefahr.deep_learning import storm_images
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import gridrad_utils
from gewittergefahr.gg_utils import error_checking


def storm_image_generator(
        top_directory_name, radar_source, radar_field_names,
        num_examples_per_batch, num_examples_per_image_time,
        first_image_time_unix_sec, last_image_time_unix_sec, min_lead_time_sec,
        max_lead_time_sec, min_target_distance_metres,
        max_target_distance_metres, event_type_string, radar_heights_m_asl=None,
        reflectivity_heights_m_asl=None, wind_speed_percentile_level=None,
        wind_speed_class_cutoffs_kt=None, normalize_by_batch=False,
        normalization_dict=dl_utils.DEFAULT_NORMALIZATION_DICT,
        percentile_offset_for_normalization=
        dl_utils.DEFAULT_PERCENTILE_OFFSET_FOR_NORMALIZATION):
    """Generates training examples with storm-centered radar images.

    F = number of radar fields
    P = number of channels = num predictor variables = num field/height pairs

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
    :param min_lead_time_sec: Used to select target variable (see doc for
        `labels.get_column_name_for_classification_label`).
    :param max_lead_time_sec: Same.
    :param min_target_distance_metres: Same.
    :param max_target_distance_metres: Same.
    :param event_type_string: Same.
    :param radar_heights_m_asl: [used only if radar_source = "gridrad"]
        1-D numpy array of radar heights (metres above sea level).  These apply
        to each field.
    :param reflectivity_heights_m_asl: [used only if radar_source != "gridrad"]
        1-D numpy array of heights (metres above sea level) for
        "reflectivity_dbz".  If "reflectivity_dbz" is not in the list
        `radar_field_names`, you can leave this as None.
    :param wind_speed_percentile_level: Same.
    :param wind_speed_class_cutoffs_kt: Same.
    :param normalize_by_batch: Used to normalize predictor values (see doc for
        `deep_learning_utils.normalize_predictor_matrix`).
    :param normalization_dict: Same.
    :param percentile_offset_for_normalization: Same.
    :return: predictor_matrix: E-by-M-by-N-by-P numpy array of storm-centered
        radar images.
    :return: target_matrix: E-by-K numpy array of target values (all 0 or 1, but
        technically the type is "float64").  If target_matrix[i, k] = 1, the
        [k]th class is the outcome for the [i]th example.  The sum across each
        row is 1 (classes are mutually exclusive and collectively exhaustive).
    """

    # Check input arguments.
    radar_utils.check_data_source(radar_source)
    error_checking.assert_is_integer(num_examples_per_batch)
    error_checking.assert_is_geq(num_examples_per_batch, 10)
    error_checking.assert_is_integer(num_examples_per_image_time)
    error_checking.assert_is_geq(num_examples_per_image_time, 2)
    error_checking.assert_is_boolean(normalize_by_batch)

    # Find input files (with storm-centered radar images).
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
        (image_file_name_matrix, _, field_name_by_predictor, _) = (
            storm_images.find_many_files_myrorss_or_mrms(
                top_directory_name=top_directory_name,
                start_time_unix_sec=first_image_time_unix_sec,
                end_time_unix_sec=last_image_time_unix_sec,
                radar_source=radar_source, radar_field_names=radar_field_names,
                reflectivity_heights_m_asl=reflectivity_heights_m_asl,
                raise_error_if_missing=True))

    num_image_times = image_file_name_matrix.shape[0]
    num_predictors = len(field_name_by_predictor)

    # Shuffle files by time.
    image_time_indices = numpy.linspace(
        0, num_image_times - 1, num=num_image_times, dtype=int)
    numpy.random.shuffle(image_time_indices)
    image_file_name_matrix = image_file_name_matrix[image_time_indices, ...]

    # Initialize variables.
    image_time_index = 0
    num_image_times_in_memory = 0
    num_image_times_per_batch = int(numpy.ceil(
        float(num_examples_per_batch) / num_examples_per_image_time))

    full_predictor_matrix = None
    all_target_values = None
    num_classes = None

    while True:

        # While more files need to be read...
        while (num_image_times_in_memory < num_image_times_per_batch or
               full_predictor_matrix is None or
               full_predictor_matrix.shape[0] < num_examples_per_batch):
            print '\n'
            tuple_of_predictor_matrices = ()

            for j in range(num_predictors):
                print 'Reading data from: "{0:s}"...'.format(
                    image_file_name_matrix[image_time_index, j])

                if j == 0:

                    # Read radar images for the [j]th predictor at the [i]th
                    # time and target values for the [i]th time (where i =
                    # image_time_index).
                    (this_field_predictor_matrix, these_storm_ids, _, _, _,
                     this_storm_to_events_table) = (
                         storm_images.read_storm_images(
                             image_file_name_matrix[image_time_index, j]))

                    these_target_values = storm_images.extract_label_values(
                        storm_ids=these_storm_ids,
                        storm_to_events_table=this_storm_to_events_table,
                        min_lead_time_sec=min_lead_time_sec,
                        max_lead_time_sec=max_lead_time_sec,
                        min_link_distance_metres=min_target_distance_metres,
                        max_link_distance_metres=max_target_distance_metres,
                        event_type_string=event_type_string,
                        wind_speed_percentile_level=wind_speed_percentile_level,
                        wind_speed_class_cutoffs_kt=wind_speed_class_cutoffs_kt)

                    if num_classes is None:
                        if wind_speed_class_cutoffs_kt is None:
                            num_classes = 2
                        else:
                            num_classes = len(wind_speed_class_cutoffs_kt) + 1

                    all_target_values = numpy.concatenate((
                        all_target_values, these_target_values))
                else:

                    # Read radar images for the [j]th predictor at the [i]th
                    # time (where i = image_time_index).
                    this_field_predictor_matrix, _, _, _, _, _ = (
                        storm_images.read_storm_images(
                            image_file_name_matrix[image_time_index, j]))

                tuple_of_predictor_matrices += (this_field_predictor_matrix,)

            # Housekeeping.
            num_image_times_in_memory += 1
            image_time_index += 1
            if image_time_index >= num_image_times:
                image_time_index = 0

            # Add radar images from [i]th time (where i = image_time_index) to
            # full_predictor_matrix, which contains radar images for all times.
            this_predictor_matrix = dl_utils.stack_predictor_variables(
                tuple_of_predictor_matrices)
            full_predictor_matrix = numpy.concatenate(
                (full_predictor_matrix, this_predictor_matrix), axis=0)

        if normalize_by_batch:  # Normalize radar images.
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

        if not normalize_by_batch:  # Normalize radar images.
            predictor_matrix = dl_utils.normalize_predictor_matrix(
                predictor_matrix=predictor_matrix,
                normalize_by_batch=normalize_by_batch,
                predictor_names=field_name_by_predictor,
                normalization_dict=normalization_dict,
                percentile_offset=percentile_offset_for_normalization)

        # Housekeeping.
        full_predictor_matrix = None
        all_target_values = None
        num_image_times_in_memory = 0

        # Turn 1-D array of target values into 2-D Boolean matrix.
        target_matrix = keras.utils.to_categorical(target_values, num_classes)
        class_fractions = numpy.mean(target_matrix, axis=0)
        print 'Fraction of target values in each class:\n{0:s}\n'.format(
            str(class_fractions))

        yield (predictor_matrix, target_matrix)
