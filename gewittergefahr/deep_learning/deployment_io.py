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
import keras
from gewittergefahr.deep_learning import storm_images
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import gridrad_utils
from gewittergefahr.gg_utils import labels


def create_2d_storm_images_one_time(
        top_directory_name, image_time_unix_sec, radar_source,
        radar_field_names, target_name=None, radar_heights_m_asl=None,
        reflectivity_heights_m_asl=None,
        normalization_dict=dl_utils.DEFAULT_NORMALIZATION_DICT):
    """Creates examples for a single image time.

    Each example consists of the following:

    - Predictors: 2-D storm-centered radar images.  One for each storm object
      and predictor variable (field/height pair).  The two dimensions are x and
      y (longitude and latitude).
    - Targets (optional): One target value for each storm object.

    F = number of radar fields
    C = num image channels = num predictor variables = num field/height pairs

    :param top_directory_name: Name of top-level directory with storm-centered
        radar images.
    :param image_time_unix_sec: Image time.
    :param radar_source: Data source (must be accepted by
        `radar_utils.check_data_source`).
    :param radar_field_names: length-F list with names of radar fields.
    :param target_name: Name of target variable.  If None, this method will
        return only predictors, not targets.
    :param radar_heights_m_asl: [used only if radar_source = "gridrad"]
        1-D numpy array of radar heights (metres above sea level).  These apply
        to each field.
    :param reflectivity_heights_m_asl: [used only if radar_source != "gridrad"]
        1-D numpy array of heights (metres above sea level) for
        "reflectivity_dbz".
    :param normalization_dict: Used to normalize predictor values (see doc for
        `deep_learning_utils.normalize_predictor_matrix`).
    :return: predictor_matrix: E-by-M-by-N-by-C numpy array of storm-centered
        radar images.
    :return: target_matrix: E-by-K numpy array of target values (all 0 or 1, but
        technically the type is "float64").  If target_matrix[i, k] = 1, the
        [k]th class is the outcome for the [i]th example.  The sum across each
        row is 1 (classes are mutually exclusive and collectively exhaustive).
    """

    # TODO(thunderhoser): Handle missing storm-image files.

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

    print 'Reading data from: "{0:s}"...'.format(image_file_names[0])

    if target_name is None:
        this_storm_image_dict = storm_images.read_storm_images_only(
            image_file_names[0])
    else:
        this_label_file_name = storm_images.find_storm_label_file(
            storm_image_file_name=image_file_names[0],
            raise_error_if_missing=True)

        this_storm_image_dict = storm_images.read_storm_images_and_labels(
            image_file_name=image_file_names[0],
            label_file_name=this_label_file_name, return_label_name=target_name)

        target_values = this_storm_image_dict[storm_images.LABEL_VALUES_KEY]
        valid_storm_indices = numpy.where(target_values >= 0)[0]
        target_values = target_values[valid_storm_indices]

        if num_classes is None:
            target_param_dict = labels.column_name_to_label_params(target_name)
            wind_speed_class_cutoffs_kt = target_param_dict[
                labels.WIND_SPEED_CLASS_CUTOFFS_KEY]

            if wind_speed_class_cutoffs_kt is None:
                num_classes = 2
            else:
                num_classes = len(wind_speed_class_cutoffs_kt) + 1

    for j in range(num_predictors):
        if j != 0:
            print 'Reading data from: "{0:s}"...'.format(image_file_names[j])
            this_storm_image_dict = storm_images.read_storm_images_only(
                image_file_names[j])

        this_predictor_matrix = this_storm_image_dict[
            storm_images.STORM_IMAGE_MATRIX_KEY]
        if target_name is not None:
            this_predictor_matrix = this_predictor_matrix[
                valid_storm_indices, ...]

        tuple_of_predictor_matrices += (this_predictor_matrix,)

    predictor_matrix = dl_utils.stack_predictor_variables(
        tuple_of_predictor_matrices).astype('float32')
    predictor_matrix = dl_utils.normalize_predictor_matrix(
        predictor_matrix=predictor_matrix, normalize_by_batch=False,
        predictor_names=field_name_by_predictor,
        normalization_dict=normalization_dict)

    if target_name is None:
        target_matrix = None
    else:
        target_matrix = keras.utils.to_categorical(target_values, num_classes)
        class_fractions = numpy.mean(target_matrix, axis=0)
        print 'Fraction of target values in each class:\n{0:s}\n'.format(
            str(class_fractions))

    return predictor_matrix, target_matrix


def create_3d_storm_images_one_time(
        top_directory_name, image_time_unix_sec, radar_source,
        radar_field_names, radar_heights_m_asl, target_name=None,
        normalization_dict=dl_utils.DEFAULT_NORMALIZATION_DICT):
    """Creates examples for a single image time.

    Each example consists of the following:

    - Predictors: 3-D storm-centered radar images.  One for each storm object,
      radar field, and radar height.  The three dimensions are x, y, and z
      (longitude, latitude, height).
    - Targets (optional): One target value for each storm object.

    C = number of radar fields
    D = number of radar heights

    :param top_directory_name: Name of top-level directory with storm-centered
        radar images.
    :param image_time_unix_sec: Image time.
    :param radar_source: Data source (must be accepted by
        `radar_utils.check_data_source`).
    :param radar_field_names: length-C list with names of radar fields.
    :param radar_heights_m_asl: length-D numpy array of radar heights (metres
        above sea level).  These apply to each field.
    :param target_name: Name of target variable.
    :param normalization_dict: Used to normalize predictor values (see doc for
        `deep_learning_utils.normalize_predictor_matrix`).
    :return: predictor_matrix: E-by-M-by-N-by-D-by-C numpy array of
        storm-centered radar images.
    :return: target_matrix: See documentation for
        `create_3d_storm_images_one_time`.
    """

    # TODO(thunderhoser): Handle missing storm-image files.

    if radar_source == radar_utils.GRIDRAD_SOURCE_ID:
        image_file_name_matrix, _ = storm_images.find_many_files_gridrad(
            top_directory_name=top_directory_name,
            start_time_unix_sec=image_time_unix_sec,
            end_time_unix_sec=image_time_unix_sec,
            radar_field_names=radar_field_names,
            radar_heights_m_asl=radar_heights_m_asl,
            raise_error_if_missing=True)

        image_file_name_matrix = image_file_name_matrix[0, ...]

    else:
        radar_field_names = [radar_utils.REFL_NAME]

        image_file_name_matrix, _, _, _ = (
            storm_images.find_many_files_myrorss_or_mrms(
                top_directory_name=top_directory_name,
                start_time_unix_sec=image_time_unix_sec,
                end_time_unix_sec=image_time_unix_sec,
                radar_source=radar_source, radar_field_names=radar_field_names,
                reflectivity_heights_m_asl=radar_heights_m_asl,
                raise_error_if_missing=True))

        num_heights = len(radar_heights_m_asl)
        image_file_name_matrix = numpy.reshape(
            image_file_name_matrix[0, ...], (1, num_heights))

    num_fields = len(radar_field_names)
    num_heights = len(radar_heights_m_asl)

    num_classes = None
    target_values = None
    tuple_of_4d_predictor_matrices = ()

    print 'Reading data from: "{0:s}"...'.format(image_file_name_matrix[0, 0])

    if target_name is None:
        this_storm_image_dict = storm_images.read_storm_images_only(
            image_file_name_matrix[0, 0])
    else:
        this_label_file_name = storm_images.find_storm_label_file(
            storm_image_file_name=image_file_name_matrix[0, 0],
            raise_error_if_missing=True)

        this_storm_image_dict = storm_images.read_storm_images_and_labels(
            image_file_name=image_file_name_matrix[0, 0],
            label_file_name=this_label_file_name, return_label_name=target_name)

        target_values = this_storm_image_dict[storm_images.LABEL_VALUES_KEY]
        valid_storm_indices = numpy.where(target_values >= 0)[0]
        target_values = target_values[valid_storm_indices]

        if num_classes is None:
            target_param_dict = labels.column_name_to_label_params(target_name)
            wind_speed_class_cutoffs_kt = target_param_dict[
                labels.WIND_SPEED_CLASS_CUTOFFS_KEY]

            if wind_speed_class_cutoffs_kt is None:
                num_classes = 2
            else:
                num_classes = len(wind_speed_class_cutoffs_kt) + 1

    for k in range(num_heights):
        tuple_of_3d_predictor_matrices = ()

        for j in range(num_fields):
            if not j == k == 0:
                print 'Reading data from: "{0:s}"...'.format(
                    image_file_name_matrix[j, k])
                this_storm_image_dict = storm_images.read_storm_images_only(
                    image_file_name_matrix[j, k])

            this_predictor_matrix = this_storm_image_dict[
                storm_images.STORM_IMAGE_MATRIX_KEY]
            if target_name is not None:
                this_predictor_matrix = this_predictor_matrix[
                    valid_storm_indices, ...]

            tuple_of_3d_predictor_matrices += (this_predictor_matrix,)

        tuple_of_4d_predictor_matrices += (dl_utils.stack_predictor_variables(
            tuple_of_3d_predictor_matrices),)

    predictor_matrix = dl_utils.stack_heights(
        tuple_of_4d_predictor_matrices).astype('float32')
    predictor_matrix = dl_utils.normalize_predictor_matrix(
        predictor_matrix=predictor_matrix, normalize_by_batch=False,
        predictor_names=radar_field_names,
        normalization_dict=normalization_dict)

    if target_name is not None:
        target_matrix = None
    else:
        target_matrix = keras.utils.to_categorical(target_values, num_classes)
        class_fractions = numpy.mean(target_matrix, axis=0)
        print 'Fraction of target values in each class:\n{0:s}\n'.format(
            str(class_fractions))

    return predictor_matrix, target_matrix
