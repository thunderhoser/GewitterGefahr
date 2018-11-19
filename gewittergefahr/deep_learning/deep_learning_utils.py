"""Helper methods for deep learning.

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

import copy
import pickle
import numpy
from gewittergefahr.gg_utils import labels
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import moisture_conversions
from gewittergefahr.gg_utils import temperature_conversions
from gewittergefahr.gg_utils import standard_atmosphere as standard_atmo
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

TOLERANCE_FOR_FREQUENCY_SUM = 1e-3
DEFAULT_REFL_MASK_THRESHOLD_DBZ = 15.

PASCALS_TO_MB = 0.01
METRES_PER_SECOND_TO_KT = 3.6 / 1.852

MEAN_VALUE_COLUMN = 'mean_value'
STANDARD_DEVIATION_COLUMN = 'standard_deviation'
MIN_VALUE_COLUMN = 'min_value'
MAX_VALUE_COLUMN = 'max_value'

NORMALIZATION_COLUMNS_NO_HEIGHT = [
    MEAN_VALUE_COLUMN, STANDARD_DEVIATION_COLUMN, MIN_VALUE_COLUMN,
    MAX_VALUE_COLUMN
]
NORMALIZATION_COLUMNS_WITH_HEIGHT = [
    MEAN_VALUE_COLUMN, STANDARD_DEVIATION_COLUMN
]

MINMAX_NORMALIZATION_TYPE_STRING = 'minmax'
Z_NORMALIZATION_TYPE_STRING = 'z_score'
VALID_NORMALIZATION_TYPE_STRINGS = [
    MINMAX_NORMALIZATION_TYPE_STRING, Z_NORMALIZATION_TYPE_STRING]

DEFAULT_MIN_NORMALIZED_VALUE = -1.
DEFAULT_MAX_NORMALIZED_VALUE = 1.


def _check_normalization_type(normalization_type_string):
    """Ensures that normalization type is valid.

    :param normalization_type_string: Normalization type.
    :raises: ValueError: if
        `normalization_type_string not in VALID_NORMALIZATION_TYPE_STRINGS`.
    """

    error_checking.assert_is_string(normalization_type_string)
    if normalization_type_string not in VALID_NORMALIZATION_TYPE_STRINGS:
        error_string = (
            '\n\n{0:s}\nValid normalization types (listed above) do not include'
            ' "{1:s}".'
        ).format(str(VALID_NORMALIZATION_TYPE_STRINGS),
                 normalization_type_string)
        raise ValueError(error_string)


def check_class_fractions(sampling_fraction_by_class_dict, target_name):
    """Error-checks sampling fractions (one for each class of target variable).

    :param sampling_fraction_by_class_dict: Dictionary, where each key is
        the integer representing a class (-2 for "dead storm") and each value is
        the corresponding sampling fraction.
    :param target_name: Name of target variable (must be accepted by
        `labels.column_name_to_num_classes`).
    :raises: KeyError: if dictionary does not contain the expected keys (class
        integers).
    :raises: ValueError: if sum(class_fractions) != 1.
    """

    num_classes = labels.column_name_to_num_classes(
        column_name=target_name, include_dead_storms=False)
    num_extended_classes = labels.column_name_to_num_classes(
        column_name=target_name, include_dead_storms=True)

    expected_keys = numpy.linspace(
        0, num_classes - 1, num=num_classes, dtype=int).tolist()
    if num_extended_classes > num_classes:
        expected_keys.append(labels.DEAD_STORM_INTEGER)

    if set(expected_keys) != set(sampling_fraction_by_class_dict.keys()):
        error_string = (
            '\n\n{0:s}\nExpected sampling_fraction_by_class_dict to contain the'
            ' keys listed above.  Instead, contains the keys listed below.'
            '\n{1:s}'
        ).format(str(expected_keys),
                 str(sampling_fraction_by_class_dict.keys()))
        raise KeyError(error_string)

    sum_of_class_fractions = numpy.sum(
        numpy.array(sampling_fraction_by_class_dict.values()))
    absolute_diff = numpy.absolute(sum_of_class_fractions - 1.)
    if absolute_diff > TOLERANCE_FOR_FREQUENCY_SUM:
        error_string = (
            '\n\n{0:s}\nSum of sampling fractions (listed above) should be 1.  '
            'Instead, got sum = {1:.4f}.'
        ).format(sampling_fraction_by_class_dict.values(),
                 sum_of_class_fractions)
        raise ValueError(error_string)


def class_fractions_to_num_examples(
        sampling_fraction_by_class_dict, target_name, num_examples_total):
    """For each target class, converts sampling fraction to number of examples.

    :param sampling_fraction_by_class_dict: See doc for `check_class_fractions`.
    :param target_name: Same.
    :param num_examples_total: Total number of examples to draw.
    :return: num_examples_by_class_dict: Dictionary, where each key is the
        integer representing a class (-2 for "dead storm") and each value is the
        corresponding number of examples to draw.
    """

    check_class_fractions(
        sampling_fraction_by_class_dict=sampling_fraction_by_class_dict,
        target_name=target_name)

    num_extended_classes = len(sampling_fraction_by_class_dict.keys())
    error_checking.assert_is_integer(num_examples_total)
    error_checking.assert_is_geq(num_examples_total, num_extended_classes)

    num_examples_by_class_dict = {}
    num_examples_used = 0
    num_classes_used = 0

    for this_key in sampling_fraction_by_class_dict.keys()[:-1]:
        this_num_examples = int(numpy.round(
            sampling_fraction_by_class_dict[this_key] * num_examples_total))
        this_num_examples = max([this_num_examples, 1])

        num_classes_used += 1
        num_classes_left = num_extended_classes - num_classes_used
        this_num_examples = min(
            [this_num_examples,
             num_examples_total - num_examples_used - num_classes_left])

        num_examples_by_class_dict.update({this_key: this_num_examples})
        num_examples_used += this_num_examples

    this_key = sampling_fraction_by_class_dict.keys()[-1]
    this_num_examples = num_examples_total - num_examples_used
    num_examples_by_class_dict.update({this_key: this_num_examples})
    return num_examples_by_class_dict


def class_fractions_to_weights(
        sampling_fraction_by_class_dict, target_name, binarize_target):
    """For each target class, converts sampling fraction to loss-fctn weight.

    :param sampling_fraction_by_class_dict: See doc for `check_class_fractions`.
    :param target_name: Same.
    :param binarize_target: Boolean flag.  If True, the target variable will be
        binarized, so that the highest class = 1 and all other classes = 0.
        Otherwise, the original number of classes will be retained, except that
        -2 ("dead storm") will be mapped to 0 (the lowest class).
    :return: lf_weight_by_class_dict: Dictionary, where each key is the integer
        representing a class and each value is the corresponding loss-function
        weight.
    """

    check_class_fractions(
        sampling_fraction_by_class_dict=sampling_fraction_by_class_dict,
        target_name=target_name)
    error_checking.assert_is_boolean(binarize_target)

    if binarize_target:
        max_key = numpy.max(numpy.array(sampling_fraction_by_class_dict.keys()))
        positive_fraction = sampling_fraction_by_class_dict[max_key]
        negative_fraction = 1. - positive_fraction
        new_sampling_fraction_dict = {
            0: negative_fraction, 1: positive_fraction
        }
    else:
        new_sampling_fraction_dict = copy.deepcopy(
            sampling_fraction_by_class_dict)

        if labels.DEAD_STORM_INTEGER in sampling_fraction_by_class_dict.keys():
            new_sampling_fraction_dict[0] = (
                new_sampling_fraction_dict[0] +
                new_sampling_fraction_dict[labels.DEAD_STORM_INTEGER])
            del new_sampling_fraction_dict[labels.DEAD_STORM_INTEGER]

    loss_function_weights = (
        1. / numpy.array(new_sampling_fraction_dict.values()))
    loss_function_weights = (
        loss_function_weights / numpy.sum(loss_function_weights))
    return dict(zip(new_sampling_fraction_dict.keys(), loss_function_weights))


def check_radar_images(
        radar_image_matrix, min_num_dimensions=3, max_num_dimensions=5):
    """Error-checks storm-centered radar images.

    :param radar_image_matrix: numpy array of radar images.  Dimensions may be
        E x M x N, E x M x N x C, or E x M x N x H_r x F_r.
    :param min_num_dimensions: Minimum dimensionality of `radar_image_matrix`.
    :param max_num_dimensions: Maximum dimensionality of `radar_image_matrix`.
    """

    error_checking.assert_is_integer(min_num_dimensions)
    error_checking.assert_is_geq(min_num_dimensions, 3)
    error_checking.assert_is_leq(min_num_dimensions, 5)

    error_checking.assert_is_integer(max_num_dimensions)
    error_checking.assert_is_geq(max_num_dimensions, 3)
    error_checking.assert_is_leq(max_num_dimensions, 5)
    error_checking.assert_is_geq(max_num_dimensions, min_num_dimensions)

    error_checking.assert_is_numpy_array_without_nan(radar_image_matrix)
    num_dimensions = len(radar_image_matrix.shape)
    error_checking.assert_is_geq(num_dimensions, min_num_dimensions)
    error_checking.assert_is_leq(num_dimensions, max_num_dimensions)


def check_soundings(sounding_matrix, num_examples=None, num_height_levels=None,
                    num_fields=None):
    """Error-checks storm-centered soundings.

    :param sounding_matrix: numpy array (E x H_s x F_s) of soundings.
    :param num_examples: Number of examples in `sounding_matrix`.
    :param num_height_levels: Number of height levels expected in
        `sounding_matrix`.
    :param num_fields: Number of fields expected in `sounding_matrix`.
    """

    error_checking.assert_is_real_numpy_array(sounding_matrix)
    error_checking.assert_is_numpy_array(sounding_matrix, num_dimensions=3)

    expected_dimensions = []
    if num_examples is None:
        expected_dimensions += [sounding_matrix.shape[0]]
    else:
        expected_dimensions += [num_examples]

    if num_height_levels is None:
        expected_dimensions += [sounding_matrix.shape[1]]
    else:
        expected_dimensions += [num_height_levels]

    if num_fields is None:
        expected_dimensions += [sounding_matrix.shape[2]]
    else:
        expected_dimensions += [num_fields]

    error_checking.assert_is_numpy_array(
        sounding_matrix, exact_dimensions=numpy.array(expected_dimensions))


def check_target_array(target_array, num_dimensions, num_classes):
    """Error-checks target values.

    :param target_array: numpy array in one of two formats.
    [1] length-E integer numpy array of target values.  All values are -2
        ("dead storm") or 0...[K - 1], where K = number of classes.
    [2] E-by-K numpy array, where each value is 0 or 1.  If target_array[i, k] =
        1, the [i]th storm object belongs to the [k]th class.  Classes are
        mutually exclusive and collectively exhaustive, so the sum across each
        row of the matrix is 1.

    :param num_dimensions: Number of dimensions expected in `target_array`.
    :param num_classes: Number of classes that should be represented in
        `target_array`.
    """

    error_checking.assert_is_integer(num_dimensions)
    error_checking.assert_is_geq(num_dimensions, 1)
    error_checking.assert_is_leq(num_dimensions, 2)
    error_checking.assert_is_integer(num_classes)
    error_checking.assert_is_geq(num_classes, 2)

    num_examples = target_array.shape[0]

    if num_dimensions == 1:
        error_checking.assert_is_numpy_array(
            target_array, exact_dimensions=numpy.array([num_examples]))
        error_checking.assert_is_integer_numpy_array(target_array)

        live_storm_object_indices = numpy.where(
            target_array != labels.DEAD_STORM_INTEGER)[0]
        error_checking.assert_is_geq_numpy_array(
            target_array[live_storm_object_indices], 0)
        error_checking.assert_is_less_than_numpy_array(
            target_array, num_classes)
    else:
        error_checking.assert_is_numpy_array(
            target_array,
            exact_dimensions=numpy.array([num_examples, num_classes]))
        error_checking.assert_is_geq_numpy_array(target_array, 0)
        error_checking.assert_is_leq_numpy_array(target_array, 1)


def stack_radar_fields(tuple_of_3d_matrices):
    """Stacks radar images with different field/height pairs.

    :param tuple_of_3d_matrices: length-C tuple, where each item is an
        E-by-M-by-N numpy array of radar images.
    :return: radar_image_matrix: E-by-M-by-N-by-C numpy array of radar images.
    """

    radar_image_matrix = numpy.stack(tuple_of_3d_matrices, axis=-1)
    check_radar_images(
        radar_image_matrix=radar_image_matrix, min_num_dimensions=4,
        max_num_dimensions=4)

    return radar_image_matrix


def stack_radar_heights(tuple_of_4d_matrices):
    """Stacks radar images with different heights.

    :param tuple_of_4d_matrices: tuple (length H_r), where each item is a numpy
        array (E x M x N x F_r) of radar images.
    :return: radar_image_matrix: numpy array (E x M x N x H_r x F_r) of radar
        images.
    """

    radar_image_matrix = numpy.stack(tuple_of_4d_matrices, axis=-2)
    check_radar_images(
        radar_image_matrix=radar_image_matrix, min_num_dimensions=5,
        max_num_dimensions=5)

    return radar_image_matrix


def normalize_radar_images(
        radar_image_matrix, field_names, normalization_type_string,
        normalization_param_file_name, test_mode=False, min_normalized_value=0.,
        max_normalized_value=1., normalization_table=None):
    """Normalizes radar images.

    If normalization_type_string = "z", z-score normalization is done for each
    field independently.  Means and standard deviations are read from the
    normalization file.

    If normalization_type_string = "minmax", min-max normalization is done for
    each field independently, using the following equations.  Climatological
    minima and maxima are read from the normalization file.

    x_unscaled(i, j) = [x(i, j) - x_min] / [x_max - x_min]

    x_scaled(i, j) = x_unscaled(i, j) * [
        max_normalized_value - min_normalized_value
    ] + min_normalized_value

    x(i, j) = original value at pixel (i, j)
    x_min = climatological minimum for field x
    x_max = climatological max for field x
    x_unscaled(i, j) = normalized but unscaled value at pixel (i, j)
    min_normalized_value: from input args
    max_normalized_value: from input args
    x_scaled(i, j) = normalized and scaled value at pixel (i, j)

    :param radar_image_matrix: numpy array of radar images.  Dimensions may be
        E x M x N x C or E x M x N x H_r x F_r.
    :param field_names: 1-D list with names of radar fields, in the order that
        they appear in radar_image_matrix.  If radar_image_matrix is
        4-dimensional, field_names must have length C.  If radar_image_matrix is
        5-dimensional, field_names must have length F_r.  Each field name must
        be accepted by `radar_utils.check_field_name`.
    :param normalization_type_string: Normalization type (must be accepted by
        `_check_normalization_type`).
    :param normalization_param_file_name: Path to file with normalization
        params.  Will be read by `read_normalization_params_from_file`.
    :param test_mode: For testing only.  Leave this alone.
    :param min_normalized_value:
        [used only if normalization_type_string = "minmax"]
        Minimum normalized value.
    :param max_normalized_value:
        [used only if normalization_type_string = "minmax"]
        Max normalized value.
    :param normalization_table: For testing only.  Leave this alone.
    :return: radar_image_matrix: Normalized version of input, with the same
        dimensions.
    """

    error_checking.assert_is_boolean(test_mode)
    if not test_mode:
        normalization_table = read_normalization_params_from_file(
            normalization_param_file_name)[0]

    check_radar_images(
        radar_image_matrix=radar_image_matrix, min_num_dimensions=4,
        max_num_dimensions=5)
    num_fields = radar_image_matrix.shape[-1]

    error_checking.assert_is_string_list(field_names)
    error_checking.assert_is_numpy_array(
        numpy.array(field_names),
        exact_dimensions=numpy.array([num_fields]))

    _check_normalization_type(normalization_type_string)
    if normalization_type_string == MINMAX_NORMALIZATION_TYPE_STRING:
        error_checking.assert_is_greater(
            max_normalized_value, min_normalized_value)

    for j in range(num_fields):
        if normalization_type_string == MINMAX_NORMALIZATION_TYPE_STRING:
            this_min_value = normalization_table[
                MIN_VALUE_COLUMN].loc[field_names[j]]
            this_max_value = normalization_table[
                MAX_VALUE_COLUMN].loc[field_names[j]]

            radar_image_matrix[..., j] = (
                (radar_image_matrix[..., j] - this_min_value) /
                (this_max_value - this_min_value))

            radar_image_matrix[..., j] = min_normalized_value + (
                radar_image_matrix[..., j] *
                (max_normalized_value - min_normalized_value))
        else:
            this_mean = normalization_table[
                MEAN_VALUE_COLUMN].loc[field_names[j]]
            this_standard_deviation = normalization_table[
                STANDARD_DEVIATION_COLUMN].loc[field_names[j]]

            radar_image_matrix[..., j] = (
                (radar_image_matrix[..., j] - this_mean) /
                this_standard_deviation)

    return radar_image_matrix


def denormalize_radar_images(
        radar_image_matrix, field_names, normalization_type_string,
        normalization_param_file_name, test_mode=False, min_normalized_value=0.,
        max_normalized_value=1., normalization_table=None):
    """Denormalizes radar images.

    This method is the inverse of `normalize_radar_images`.

    :param radar_image_matrix: See doc for `normalize_radar_images`.
    :param field_names: Same.
    :param normalization_type_string: Same.
    :param normalization_param_file_name: Path to file with normalization
        params.  Will be read by `read_normalization_params_from_file`.
    :param test_mode: For testing only.  Leave this alone.
    :param min_normalized_value: Same.
    :param max_normalized_value: Same.
    :param normalization_table: For testing only.  Leave this alone.
    :return: radar_image_matrix: Denormalized version of input, with the same
        dimensions.
    """

    error_checking.assert_is_boolean(test_mode)
    if not test_mode:
        normalization_table = read_normalization_params_from_file(
            normalization_param_file_name)[0]

    check_radar_images(
        radar_image_matrix=radar_image_matrix, min_num_dimensions=4,
        max_num_dimensions=5)
    num_fields = radar_image_matrix.shape[-1]

    error_checking.assert_is_string_list(field_names)
    error_checking.assert_is_numpy_array(
        numpy.array(field_names),
        exact_dimensions=numpy.array([num_fields]))

    _check_normalization_type(normalization_type_string)
    if normalization_type_string == MINMAX_NORMALIZATION_TYPE_STRING:
        error_checking.assert_is_greater(
            max_normalized_value, min_normalized_value)
        # error_checking.assert_is_geq_numpy_array(
        #     radar_image_matrix, min_normalized_value)
        # error_checking.assert_is_leq_numpy_array(
        #     radar_image_matrix, max_normalized_value)

    for j in range(num_fields):
        if normalization_type_string == MINMAX_NORMALIZATION_TYPE_STRING:
            this_min_value = normalization_table[
                MIN_VALUE_COLUMN].loc[field_names[j]]
            this_max_value = normalization_table[
                MAX_VALUE_COLUMN].loc[field_names[j]]

            radar_image_matrix[..., j] = (
                (radar_image_matrix[..., j] - min_normalized_value) /
                (max_normalized_value - min_normalized_value))

            radar_image_matrix[..., j] = this_min_value + (
                radar_image_matrix[..., j] * (this_max_value - this_min_value))
        else:
            this_mean = normalization_table[
                MEAN_VALUE_COLUMN].loc[field_names[j]]
            this_standard_deviation = normalization_table[
                STANDARD_DEVIATION_COLUMN].loc[field_names[j]]

            radar_image_matrix[..., j] = this_mean + (
                this_standard_deviation * radar_image_matrix[..., j])

    return radar_image_matrix


def mask_low_reflectivity_pixels(
        radar_image_matrix_3d, field_names,
        reflectivity_threshold_dbz=DEFAULT_REFL_MASK_THRESHOLD_DBZ):
    """Masks pixels with low reflectivity.

    Specifically, at each pixel with low reflectivity, this method sets all
    variables to zero.

    :param radar_image_matrix_3d: numpy array of radar images.  Dimensions must
        be E x M x N x H_r x F_r.
    :param field_names: List of field names (length F_r).  Each field name must
        be accepted by `radar_utils.check_field_name`.
    :param reflectivity_threshold_dbz: Threshold used to define "low
        reflectivity" (units of dBZ).
    :return: radar_image_matrix_3d: Same as input, but low-reflectivity pixels
        are masked.
    :raises: ValueError: if `"reflectivity_dbz" not in field_names`.
    """

    # TODO(thunderhoser): Maybe values shouldn't always be set to zero?

    check_radar_images(
        radar_image_matrix=radar_image_matrix_3d, min_num_dimensions=5,
        max_num_dimensions=5)
    num_fields = radar_image_matrix_3d.shape[-1]

    error_checking.assert_is_string_list(field_names)
    error_checking.assert_is_numpy_array(
        numpy.array(field_names), exact_dimensions=numpy.array([num_fields]))
    error_checking.assert_is_greater(reflectivity_threshold_dbz, 0.)

    if radar_utils.REFL_NAME not in field_names:
        error_string = (
            'Cannot find "{0:s}" (needed to find low-reflectivity pixels) in '
            'list `{1:s}`.'
        ).format(radar_utils.REFL_NAME, field_names)
        raise ValueError(error_string)

    reflectivity_index = field_names.index(radar_utils.REFL_NAME)
    tuple_of_bad_indices = numpy.where(
        radar_image_matrix_3d[..., reflectivity_index] <
        reflectivity_threshold_dbz)
    num_bad_values = len(tuple_of_bad_indices[0])

    for j in range(num_fields):
        these_last_indices = numpy.full(num_bad_values, j, dtype=int)
        radar_image_matrix_3d[
            tuple_of_bad_indices[0], tuple_of_bad_indices[1],
            tuple_of_bad_indices[2], tuple_of_bad_indices[3],
            these_last_indices] = 0.

    return radar_image_matrix_3d


def normalize_soundings(
        sounding_matrix, field_names, normalization_type_string,
        normalization_param_file_name, test_mode=False, min_normalized_value=0.,
        max_normalized_value=1., normalization_table=None):
    """Normalizes soundings.

    This method uses the same equations as `normalize_radar_images`.

    :param sounding_matrix: numpy array (E x H_s x F_s) of soundings.
    :param field_names: list (length F_s) of field names, in the order that they
        appear in `sounding_matrix`.
    :param normalization_type_string: Normalization type (must be accepted by
        `_check_normalization_type`).
    :param normalization_param_file_name: Path to file with normalization
        params.  Will be read by `read_normalization_params_from_file`.
    :param test_mode: For testing only.  Leave this alone.
    :param min_normalized_value:
        [used only if normalization_type_string = "minmax"]
        Minimum normalized value.
    :param max_normalized_value:
        [used only if normalization_type_string = "minmax"]
        Max normalized value.
    :param normalization_table: For testing only.  Leave this alone.
    :return: sounding_matrix: Normalized version of input, with the same
        dimensions.
    """

    error_checking.assert_is_boolean(test_mode)
    if not test_mode:
        normalization_table = read_normalization_params_from_file(
            normalization_param_file_name)[2]

    error_checking.assert_is_string_list(field_names)
    error_checking.assert_is_numpy_array(
        numpy.array(field_names), num_dimensions=1)

    num_fields = len(field_names)
    check_soundings(sounding_matrix=sounding_matrix, num_fields=num_fields)
    _check_normalization_type(normalization_type_string)

    if normalization_type_string == MINMAX_NORMALIZATION_TYPE_STRING:
        error_checking.assert_is_greater(
            max_normalized_value, min_normalized_value)

    for j in range(num_fields):
        if normalization_type_string == MINMAX_NORMALIZATION_TYPE_STRING:
            this_min_value = normalization_table[
                MIN_VALUE_COLUMN].loc[field_names[j]]
            this_max_value = normalization_table[
                MAX_VALUE_COLUMN].loc[field_names[j]]

            sounding_matrix[..., j] = (
                (sounding_matrix[..., j] - this_min_value) /
                (this_max_value - this_min_value)
            )
            sounding_matrix[..., j] = min_normalized_value + (
                sounding_matrix[..., j] *
                (max_normalized_value - min_normalized_value)
            )
        else:
            this_mean = normalization_table[
                MEAN_VALUE_COLUMN].loc[field_names[j]]
            this_standard_deviation = normalization_table[
                STANDARD_DEVIATION_COLUMN
            ].loc[field_names[j]]

            sounding_matrix[..., j] = (
                (sounding_matrix[..., j] - this_mean) / this_standard_deviation
            )

    return sounding_matrix


def denormalize_soundings(
        sounding_matrix, field_names, normalization_type_string,
        normalization_param_file_name, test_mode=False, min_normalized_value=0.,
        max_normalized_value=1., normalization_table=None):
    """Denormalizes soundings.

    This method is the inverse of `normalize_soundings`.

    :param sounding_matrix: See doc for `normalize_soundings`.
    :param field_names: Same.
    :param normalization_type_string: Same.
    :param normalization_param_file_name: Path to file with normalization
        params.  Will be read by `read_normalization_params_from_file`.
    :param test_mode: For testing only.  Leave this alone.
    :param min_normalized_value: Same.
    :param max_normalized_value: Same.
    :param normalization_table: For testing only.  Leave this alone.
    :return: sounding_matrix: Denormalized version of input, with the same
        dimensions.
    """

    error_checking.assert_is_boolean(test_mode)
    if not test_mode:
        normalization_table = read_normalization_params_from_file(
            normalization_param_file_name)[2]

    error_checking.assert_is_string_list(field_names)
    error_checking.assert_is_numpy_array(
        numpy.array(field_names), num_dimensions=1)

    num_fields = len(field_names)
    check_soundings(sounding_matrix=sounding_matrix, num_fields=num_fields)
    _check_normalization_type(normalization_type_string)

    if normalization_type_string == MINMAX_NORMALIZATION_TYPE_STRING:
        error_checking.assert_is_greater(
            max_normalized_value, min_normalized_value)
        # error_checking.assert_is_geq_numpy_array(
        #     sounding_matrix, min_normalized_value)
        # error_checking.assert_is_leq_numpy_array(
        #     sounding_matrix, max_normalized_value)

    for j in range(num_fields):
        if normalization_type_string == MINMAX_NORMALIZATION_TYPE_STRING:
            this_min_value = normalization_table[
                MIN_VALUE_COLUMN].loc[field_names[j]]
            this_max_value = normalization_table[
                MAX_VALUE_COLUMN].loc[field_names[j]]

            sounding_matrix[..., j] = (
                (sounding_matrix[..., j] - min_normalized_value) /
                (max_normalized_value - min_normalized_value)
            )
            sounding_matrix[..., j] = this_min_value + (
                sounding_matrix[..., j] * (this_max_value - this_min_value)
            )
        else:
            this_mean = normalization_table[
                MEAN_VALUE_COLUMN].loc[field_names[j]]
            this_standard_deviation = normalization_table[
                STANDARD_DEVIATION_COLUMN
            ].loc[field_names[j]]

            sounding_matrix[..., j] = this_mean + (
                this_standard_deviation * sounding_matrix[..., j]
            )

    return sounding_matrix


def soundings_to_metpy_dictionaries(
        sounding_matrix, field_names, height_levels_m_agl=None,
        storm_elevations_m_asl=None):
    """Converts soundings to format required by MetPy.

    If `sounding_matrix` contains pressures, `height_levels_m_agl` and
    `storm_elevations_m_asl` will not be used.

    Otherwise, `height_levels_m_agl` and `storm_elevations_m_asl` will be used
    to estimate the pressure levels for each sounding.

    :param sounding_matrix: numpy array (E x H_s x F_s) of soundings.
    :param field_names: list (length F_s) of field names, in the order that they
        appear in `sounding_matrix`.
    :param height_levels_m_agl: numpy array (length H_s) of height levels
        (metres above ground level), in the order that they appear in
        `sounding_matrix`.
    :param storm_elevations_m_asl: length-E numpy array of storm elevations
        (metres above sea level).
    :return: list_of_metpy_dictionaries: length-E list of dictionaries.  The
        format of each dictionary is described in the input doc for
        `sounding_plotting.plot_sounding`.
    """

    error_checking.assert_is_string_list(field_names)
    error_checking.assert_is_numpy_array(
        numpy.array(field_names), num_dimensions=1)
    check_soundings(
        sounding_matrix=sounding_matrix, num_fields=len(field_names))

    try:
        pressure_index = field_names.index(soundings.PRESSURE_NAME)
        pressure_matrix_pascals = sounding_matrix[..., pressure_index]
    except ValueError:
        error_checking.assert_is_geq_numpy_array(height_levels_m_agl, 0)
        error_checking.assert_is_numpy_array(
            height_levels_m_agl, num_dimensions=1)

        error_checking.assert_is_numpy_array_without_nan(storm_elevations_m_asl)
        error_checking.assert_is_numpy_array(
            storm_elevations_m_asl, num_dimensions=1)

        num_height_levels = len(height_levels_m_agl)
        num_examples = len(storm_elevations_m_asl)
        check_soundings(
            sounding_matrix=sounding_matrix, num_examples=num_examples,
            num_height_levels=num_height_levels)

        height_matrix_m_asl = numpy.full(
            (num_examples, num_height_levels), numpy.nan)
        for i in range(num_examples):
            height_matrix_m_asl[i, ...] = (
                height_levels_m_agl + storm_elevations_m_asl[i])

        pressure_matrix_pascals = standard_atmo.height_to_pressure(
            height_matrix_m_asl)

    try:
        temperature_index = field_names.index(soundings.TEMPERATURE_NAME)
        temperature_matrix_kelvins = sounding_matrix[..., temperature_index]
    except ValueError:
        virtual_pot_temp_index = field_names.index(
            soundings.VIRTUAL_POTENTIAL_TEMPERATURE_NAME)
        temperature_matrix_kelvins = (
            temperature_conversions.temperatures_from_potential_temperatures(
                potential_temperatures_kelvins=sounding_matrix[
                    ..., virtual_pot_temp_index],
                total_pressures_pascals=pressure_matrix_pascals)
        )

    try:
        specific_humidity_index = field_names.index(
            soundings.SPECIFIC_HUMIDITY_NAME)
        dewpoint_matrix_kelvins = (
            moisture_conversions.specific_humidity_to_dewpoint(
                specific_humidities_kg_kg01=sounding_matrix[
                    ..., specific_humidity_index],
                total_pressures_pascals=pressure_matrix_pascals)
        )
    except ValueError:
        relative_humidity_index = field_names.index(
            soundings.RELATIVE_HUMIDITY_NAME)
        dewpoint_matrix_kelvins = (
            moisture_conversions.relative_humidity_to_dewpoint(
                relative_humidities=sounding_matrix[
                    ..., relative_humidity_index],
                temperatures_kelvins=temperature_matrix_kelvins,
                total_pressures_pascals=pressure_matrix_pascals)
        )

    temperature_matrix_celsius = temperature_conversions.kelvins_to_celsius(
        temperature_matrix_kelvins)
    dewpoint_matrix_celsius = temperature_conversions.kelvins_to_celsius(
        dewpoint_matrix_kelvins)

    try:
        u_wind_index = field_names.index(
            soundings.U_WIND_NAME)
        v_wind_index = field_names.index(
            soundings.V_WIND_NAME)
        include_wind = True
    except ValueError:
        include_wind = False

    num_examples = sounding_matrix.shape[0]
    list_of_metpy_dictionaries = [None] * num_examples

    for i in range(num_examples):
        list_of_metpy_dictionaries[i] = {
            soundings.PRESSURE_COLUMN_METPY:
                pressure_matrix_pascals[i, :] * PASCALS_TO_MB,
            soundings.TEMPERATURE_COLUMN_METPY:
                temperature_matrix_celsius[i, :],
            soundings.DEWPOINT_COLUMN_METPY: dewpoint_matrix_celsius[i, :],
        }

        if include_wind:
            list_of_metpy_dictionaries[i].update({
                soundings.U_WIND_COLUMN_METPY:
                    (sounding_matrix[i, ..., u_wind_index] *
                     METRES_PER_SECOND_TO_KT),
                soundings.V_WIND_COLUMN_METPY:
                    (sounding_matrix[i, ..., v_wind_index] *
                     METRES_PER_SECOND_TO_KT)
            })

    return list_of_metpy_dictionaries


def sample_by_class(
        sampling_fraction_by_class_dict, target_name, target_values,
        num_examples_total, test_mode=False):
    """Randomly draws examples, respecting the fraction for each target class.

    In other words, this method allows "oversampling" and "undersampling" of
    different classes.

    :param sampling_fraction_by_class_dict: See doc for `check_class_fractions`.
    :param target_name: Same.
    :param target_values: See doc for `check_target_array` (format 1).
    :param num_examples_total: See doc for `class_fractions_to_num_examples`.
    :param test_mode: Leave this argument alone.
    :return: indices_to_keep: 1-D numpy array with indices of examples to keep.
        These are array indices into `target_values`.
    """

    num_desired_examples_by_class_dict = class_fractions_to_num_examples(
        sampling_fraction_by_class_dict=sampling_fraction_by_class_dict,
        target_name=target_name, num_examples_total=num_examples_total)

    num_classes = labels.column_name_to_num_classes(
        column_name=target_name, include_dead_storms=False)
    check_target_array(
        target_array=target_values, num_dimensions=1, num_classes=num_classes)

    indices_to_keep_by_class_dict = {}
    num_desired_examples_by_extended_class = []
    num_avail_examples_by_extended_class = []

    for this_key in num_desired_examples_by_class_dict.keys():
        these_indices = numpy.where(target_values == this_key)[0]
        num_desired_examples_by_extended_class.append(
            num_desired_examples_by_class_dict[this_key])
        num_avail_examples_by_extended_class.append(len(these_indices))

        if (num_desired_examples_by_extended_class[-1] > 0 and
                num_avail_examples_by_extended_class[-1] == 0):
            return None

        indices_to_keep_by_class_dict.update({this_key: these_indices})

    num_desired_examples_by_extended_class = numpy.array(
        num_desired_examples_by_extended_class)
    num_avail_examples_by_extended_class = numpy.array(
        num_avail_examples_by_extended_class)

    if numpy.any(num_avail_examples_by_extended_class <
                 num_desired_examples_by_extended_class):

        avail_to_desired_ratio_by_extended_class = (
            num_avail_examples_by_extended_class.astype(float) /
            num_desired_examples_by_extended_class)
        num_examples_total = int(numpy.floor(
            num_examples_total *
            numpy.min(avail_to_desired_ratio_by_extended_class)))

        num_desired_examples_by_class_dict = class_fractions_to_num_examples(
            sampling_fraction_by_class_dict=sampling_fraction_by_class_dict,
            target_name=target_name, num_examples_total=num_examples_total)

    indices_to_keep = numpy.array([], dtype=int)
    for this_key in indices_to_keep_by_class_dict.keys():
        if not test_mode:
            numpy.random.shuffle(indices_to_keep_by_class_dict[this_key])

        this_num_examples = num_desired_examples_by_class_dict[this_key]
        indices_to_keep = numpy.concatenate((
            indices_to_keep,
            indices_to_keep_by_class_dict[this_key][:this_num_examples]))

    return indices_to_keep


def write_normalization_params(
        pickle_file_name, radar_table_no_height, radar_table_with_height,
        sounding_table_no_height, sounding_table_with_height):
    """Writes normalization parameters to Pickle file.

    :param pickle_file_name: Path to output file.
    :param radar_table_no_height: Single-indexed pandas DataFrame.  Each index
        is a field name (accepted by `radar_utils.check_field_name`).  Must
        contain the following columns.
    radar_table_no_height.mean_value: Mean value for the given field.
    radar_table_no_height.standard_deviation: Standard deviation.
    radar_table_no_height.min_value: Minimum value.
    radar_table_no_height.max_value: Max value.

    :param radar_table_with_height: Double-indexed pandas DataFrame.  Each index
        is a tuple with (field_name, height_m_agl), where `field_name` is
        accepted by `radar_utils.check_field_name` and `height_m_agl` is in
        metres above ground level.  Must contain the following columns.
    radar_table_with_height.mean_value: Mean value for the given field.
    radar_table_with_height.standard_deviation: Standard deviation.

    :param sounding_table_no_height: Single-indexed pandas DataFrame.  Each
        index is a field name (accepted by `soundings.check_field_name`).
        Columns should be the same as in `radar_table_no_height`.
    :param sounding_table_with_height: Double-indexed pandas DataFrame.  Each
        index is a tuple with (field_name, height_m_agl), where `field_name` is
        accepted by `soundings.check_field_name` and `height_m_agl` is in metres
        above ground level.  Columns should be the same as in
        `radar_table_with_height`.
    """

    # TODO(thunderhoser): Move this to normalization.py or something.
    error_checking.assert_columns_in_dataframe(
        radar_table_no_height, NORMALIZATION_COLUMNS_NO_HEIGHT)
    error_checking.assert_columns_in_dataframe(
        radar_table_with_height, NORMALIZATION_COLUMNS_WITH_HEIGHT)
    error_checking.assert_columns_in_dataframe(
        sounding_table_no_height, NORMALIZATION_COLUMNS_NO_HEIGHT)
    error_checking.assert_columns_in_dataframe(
        sounding_table_with_height, NORMALIZATION_COLUMNS_WITH_HEIGHT)

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(radar_table_no_height, pickle_file_handle)
    pickle.dump(radar_table_with_height, pickle_file_handle)
    pickle.dump(sounding_table_no_height, pickle_file_handle)
    pickle.dump(sounding_table_with_height, pickle_file_handle)
    pickle_file_handle.close()


def read_normalization_params_from_file(pickle_file_name):
    """Reads normalization parameters from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: radar_table_no_height: See doc for `write_normalization_params`.
    :return: radar_table_with_height: Same.
    :return: sounding_table_no_height: Same.
    :return: sounding_table_with_height: Same.
    """

    # TODO(thunderhoser): Move this to normalization.py or something.
    pickle_file_handle = open(pickle_file_name, 'rb')
    radar_table_no_height = pickle.load(pickle_file_handle)
    radar_table_with_height = pickle.load(pickle_file_handle)
    sounding_table_no_height = pickle.load(pickle_file_handle)
    sounding_table_with_height = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    error_checking.assert_columns_in_dataframe(
        radar_table_no_height, NORMALIZATION_COLUMNS_NO_HEIGHT)
    error_checking.assert_columns_in_dataframe(
        radar_table_with_height, NORMALIZATION_COLUMNS_WITH_HEIGHT)
    error_checking.assert_columns_in_dataframe(
        sounding_table_no_height, NORMALIZATION_COLUMNS_NO_HEIGHT)
    error_checking.assert_columns_in_dataframe(
        sounding_table_with_height, NORMALIZATION_COLUMNS_WITH_HEIGHT)

    return (radar_table_no_height, radar_table_with_height,
            sounding_table_no_height, sounding_table_with_height)
