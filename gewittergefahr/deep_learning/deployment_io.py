"""IO methods for deployment of a deep-learning model.

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
from gewittergefahr.gg_utils import labels


def create_storm_images_with_targets(
        top_directory_name, image_time_unix_sec, radar_source,
        radar_field_names, target_name, radar_heights_m_asl=None,
        reflectivity_heights_m_asl=None,
        normalization_dict=dl_utils.DEFAULT_NORMALIZATION_DICT):
    """Creates examples for a single image time.

    Each example consists of storm-centered radar images (predictors) and one
    target value.

    F = number of radar fields
    P = number of channels = num predictor variables = num field/height pairs

    :param top_directory_name: Name of top-level directory with storm-centered
        radar images.
    :param image_time_unix_sec: Image time.
    :param radar_source: Data source (must be accepted by
        `radar_utils.check_data_source`).
    :param radar_field_names: length-F list with names of radar fields.
    :param target_name: Name of target variable.
    :param radar_heights_m_asl: [used only if radar_source = "gridrad"]
        1-D numpy array of radar heights (metres above sea level).  These apply
        to each field.
    :param reflectivity_heights_m_asl: [used only if radar_source != "gridrad"]
        1-D numpy array of heights (metres above sea level) for
        "reflectivity_dbz".  If "reflectivity_dbz" is not in the list
        `radar_field_names`, you can leave this as None.
    :param normalization_dict: Used to normalize predictor values (see doc for
        `deep_learning_utils.normalize_predictor_matrix`).
    :return: predictor_matrix: E-by-M-by-N-by-P numpy array of storm-centered
        radar images.
    :return: target_matrix: E-by-K numpy array of target values (all 0 or 1, but
        technically the type is "float64").  If target_matrix[i, k] = 1, the
        [k]th class is the outcome for the [i]th example.  The sum across each
        row is 1 (classes are mutually exclusive and collectively exhaustive).
    """

    radar_utils.check_data_source(radar_source)

    if radar_source == radar_utils.GRIDRAD_SOURCE_ID:
        image_file_name_matrix, _ = storm_images.find_many_files_gridrad(
            top_directory_name=top_directory_name,
            start_time_unix_sec=image_time_unix_sec,
            end_time_unix_sec=image_time_unix_sec,
            radar_field_names=radar_field_names,
            radar_heights_m_asl=radar_heights_m_asl,
            raise_error_if_missing=True)

        field_name_by_predictor, _ = (
            gridrad_utils.fields_and_refl_heights_to_pairs(
                field_names=radar_field_names,
                heights_m_asl=radar_heights_m_asl))

    else:
        image_file_name_matrix, _, field_name_by_predictor, _ = (
            storm_images.find_many_files_myrorss_or_mrms(
                top_directory_name=top_directory_name,
                start_time_unix_sec=image_time_unix_sec,
                end_time_unix_sec=image_time_unix_sec,
                radar_source=radar_source, radar_field_names=radar_field_names,
                reflectivity_heights_m_asl=reflectivity_heights_m_asl,
                raise_error_if_missing=True))

    num_predictors = len(field_name_by_predictor)
    image_file_names = numpy.reshape(image_file_name_matrix, num_predictors)

    num_classes = None
    target_values = None
    tuple_of_predictor_matrices = ()

    for j in range(num_predictors):
        print 'Reading data from: "{0:s}"...'.format(image_file_names[j])

        if j == 0:
            (this_predictor_matrix, these_storm_ids, _, _, _,
             this_storm_to_events_table) = (
                 storm_images.read_storm_images(image_file_names[j]))

            target_values = storm_images.extract_label_values(
                storm_ids=these_storm_ids,
                storm_to_events_table=this_storm_to_events_table,
                label_column=target_name)

            if num_classes is None:
                target_param_dict = labels.column_name_to_label_params(
                    target_name)
                wind_speed_class_cutoffs_kt = target_param_dict[
                    labels.WIND_SPEED_CLASS_CUTOFFS_KEY]

                if wind_speed_class_cutoffs_kt is None:
                    num_classes = 2
                else:
                    num_classes = len(wind_speed_class_cutoffs_kt) + 1

        else:
            this_predictor_matrix, _, _, _, _, _ = (
                storm_images.read_storm_images(image_file_names[j]))

        tuple_of_predictor_matrices += (this_predictor_matrix,)

    predictor_matrix = dl_utils.stack_predictor_variables(
        tuple_of_predictor_matrices).astype('float32')
    predictor_matrix = dl_utils.normalize_predictor_matrix(
        predictor_matrix=predictor_matrix, normalize_by_batch=False,
        predictor_names=field_name_by_predictor,
        normalization_dict=normalization_dict)

    target_matrix = keras.utils.to_categorical(target_values, num_classes)
    class_fractions = numpy.mean(target_matrix, axis=0)
    print 'Fraction of target values in each class:\n{0:s}\n'.format(
        str(class_fractions))

    return predictor_matrix, target_matrix


def create_storm_images_sans_targets(
        top_directory_name, image_time_unix_sec, radar_source,
        radar_field_names, radar_heights_m_asl=None,
        reflectivity_heights_m_asl=None,
        normalization_dict=dl_utils.DEFAULT_NORMALIZATION_DICT):
    """Creates examples for a single image time.

    Each example consists only of storm-centered radar images (predictors), with
    no target value.

    :param top_directory_name: See documentation for
        `create_storm_images_with_targets`.
    :param image_time_unix_sec: Same.
    :param radar_source: Same.
    :param radar_field_names: Same.
    :param radar_heights_m_asl: Same.
    :param reflectivity_heights_m_asl: Same.
    :param normalization_dict: Same.
    :return: predictor_matrix: Same.
    """

    radar_utils.check_data_source(radar_source)

    if radar_source == radar_utils.GRIDRAD_SOURCE_ID:
        image_file_name_matrix, _ = storm_images.find_many_files_gridrad(
            top_directory_name=top_directory_name,
            start_time_unix_sec=image_time_unix_sec,
            end_time_unix_sec=image_time_unix_sec,
            radar_field_names=radar_field_names,
            radar_heights_m_asl=radar_heights_m_asl,
            raise_error_if_missing=True)

        field_name_by_predictor, _ = (
            gridrad_utils.fields_and_refl_heights_to_pairs(
                field_names=radar_field_names,
                heights_m_asl=radar_heights_m_asl))

    else:
        image_file_name_matrix, _, field_name_by_predictor, _ = (
            storm_images.find_many_files_myrorss_or_mrms(
                top_directory_name=top_directory_name,
                start_time_unix_sec=image_time_unix_sec,
                end_time_unix_sec=image_time_unix_sec,
                radar_source=radar_source, radar_field_names=radar_field_names,
                reflectivity_heights_m_asl=reflectivity_heights_m_asl,
                raise_error_if_missing=True))

    num_predictors = len(field_name_by_predictor)
    image_file_names = numpy.reshape(image_file_name_matrix, num_predictors)
    tuple_of_predictor_matrices = ()

    for j in range(num_predictors):
        print 'Reading data from: "{0:s}"...'.format(image_file_names[j])

        this_predictor_matrix, _, _, _, _, _ = (
            storm_images.read_storm_images(image_file_names[j]))
        tuple_of_predictor_matrices += (this_predictor_matrix,)

    predictor_matrix = dl_utils.stack_predictor_variables(
        tuple_of_predictor_matrices).astype('float32')
    predictor_matrix = dl_utils.normalize_predictor_matrix(
        predictor_matrix=predictor_matrix, normalize_by_batch=False,
        predictor_names=field_name_by_predictor,
        normalization_dict=normalization_dict)

    return predictor_matrix
