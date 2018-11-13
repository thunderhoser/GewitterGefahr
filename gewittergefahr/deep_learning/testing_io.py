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

import numpy
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.gg_utils import error_checking


def create_examples_2d_or_3d(option_dict, num_examples_per_file):
    """Creates examples with either all 2-D or all 3-D radar images.

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

    If `sounding_field_names is None`, this method returns the following.

    :return: radar_image_matrix: See doc for
        `training_validation_io.example_generator_2d_or_3d`.
    :return: target_array: Same.

    If `sounding_field_names is not None`, this method returns the following.

    :return: predictor_list: See doc for
        `training_validation_io.example_generator_2d_or_3d`.
    :return: target_array: Same.
    """

    error_checking.assert_is_integer(num_examples_per_file)
    error_checking.assert_is_geq(num_examples_per_file, 32)

    example_file_names = option_dict[trainval_io.EXAMPLE_FILES_KEY]
    error_checking.assert_is_numpy_array(
        numpy.array(example_file_names), num_dimensions=1)

    option_dict.update({
        trainval_io.NUM_EXAMPLES_PER_BATCH_KEY: num_examples_per_file,
        trainval_io.LOOP_ONCE_KEY: True,
        trainval_io.NUM_TRANSLATIONS_KEY: 0,
        trainval_io.NUM_ROTATIONS_KEY: 0,
        trainval_io.NUM_NOISINGS_KEY: 0
    })

    num_example_files = len(example_file_names)
    list_of_predictor_matrices = None
    target_array = None

    for i in range(num_example_files):
        option_dict[trainval_io.EXAMPLE_FILES_KEY] = [example_file_names[i]]
        this_generator_object = trainval_io.example_generator_2d_or_3d(
            option_dict)

        these_predictor_matrices, this_target_array = next(
            this_generator_object)
        if isinstance(these_predictor_matrices, numpy.ndarray):
            these_predictor_matrices = [these_predictor_matrices]

        if target_array is None:
            list_of_predictor_matrices = these_predictor_matrices + []
            target_array = this_target_array + 0
        else:
            for k in range(len(list_of_predictor_matrices)):
                list_of_predictor_matrices[k] = numpy.concatenate((
                    list_of_predictor_matrices[k], these_predictor_matrices[k]
                ), axis=0)

            target_array = numpy.concatenate(
                (target_array, this_target_array), axis=0)

    if len(list_of_predictor_matrices) == 1:
        return list_of_predictor_matrices[0], target_array
    return list_of_predictor_matrices, target_array


def create_examples_2d3d_myrorss(option_dict, num_examples_per_file):
    """Creates examples with both 2-D and 3-D radar images.

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

    :return: predictor_list: See doc for
        `training_validation_io.example_generator_2d3d_myrorss`.
    :return: target_array: Same.
    """

    error_checking.assert_is_integer(num_examples_per_file)
    error_checking.assert_is_geq(num_examples_per_file, 32)

    example_file_names = option_dict[trainval_io.EXAMPLE_FILES_KEY]
    error_checking.assert_is_numpy_array(
        numpy.array(example_file_names), num_dimensions=1)

    option_dict.update({
        trainval_io.NUM_EXAMPLES_PER_BATCH_KEY: num_examples_per_file,
        trainval_io.LOOP_ONCE_KEY: True,
        trainval_io.NUM_TRANSLATIONS_KEY: 0,
        trainval_io.NUM_ROTATIONS_KEY: 0,
        trainval_io.NUM_NOISINGS_KEY: 0
    })

    num_example_files = len(example_file_names)
    list_of_predictor_matrices = None
    target_array = None

    for i in range(num_example_files):
        option_dict[trainval_io.EXAMPLE_FILES_KEY] = [example_file_names[i]]
        this_generator_object = trainval_io.example_generator_2d3d_myrorss(
            option_dict)
        these_predictor_matrices, this_target_array = next(
            this_generator_object)

        if target_array is None:
            list_of_predictor_matrices = these_predictor_matrices + []
            target_array = this_target_array + 0
        else:
            for k in range(len(list_of_predictor_matrices)):
                list_of_predictor_matrices[k] = numpy.concatenate((
                    list_of_predictor_matrices[k], these_predictor_matrices[k]
                ), axis=0)

            target_array = numpy.concatenate(
                (target_array, this_target_array), axis=0)

    return list_of_predictor_matrices, target_array
