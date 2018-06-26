"""Helper methods for deep learning.

--- NOTATION ---

The following letters will be used throughout this module.

E = number of examples (storm objects)
M = number of rows per radar image
N = number of columns per radar image
H_r = number of heights per radar image
F_r = number of radar fields (not including different heights)
H_s = number of vertical levels per sounding
F_s = number of sounding fields (not including different vertical levels)
C = number of field/height pairs per radar image
K = number of classes for target variable
T = number of file times (time steps or SPC dates)
"""

import copy
import numpy
from gewittergefahr.gg_utils import labels
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import soundings_only
from gewittergefahr.gg_utils import error_checking

TOLERANCE_FOR_FREQUENCY_SUM = 1e-3

DEFAULT_PERCENTILE_OFFSET_FOR_NORMALIZATION = 1.
MAX_PERCENTILE_OFFSET_FOR_NORMALIZATION = 5.

DEFAULT_RADAR_NORMALIZATION_DICT = {
    radar_utils.ECHO_TOP_18DBZ_NAME: numpy.array([0., 15.]),  # km
    radar_utils.ECHO_TOP_40DBZ_NAME: numpy.array([0., 15.]),  # km
    radar_utils.ECHO_TOP_50DBZ_NAME: numpy.array([0., 15.]),  # km
    radar_utils.LOW_LEVEL_SHEAR_NAME: numpy.array([-0.01, 0.02]),  # s^-1
    radar_utils.MID_LEVEL_SHEAR_NAME: numpy.array([-0.01, 0.02]),  # s^-1
    radar_utils.MESH_NAME: numpy.array([0., 90.]),  # mm
    radar_utils.REFL_NAME: numpy.array([0., 75.]),  # dBZ
    radar_utils.REFL_COLUMN_MAX_NAME: numpy.array([0., 75.]),  # dBZ
    radar_utils.REFL_0CELSIUS_NAME: numpy.array([0., 75.]),  # dBZ
    radar_utils.REFL_M10CELSIUS_NAME: numpy.array([0., 75.]),  # dBZ
    radar_utils.REFL_M20CELSIUS_NAME: numpy.array([0., 75.]),  # dBZ
    radar_utils.REFL_LOWEST_ALTITUDE_NAME: numpy.array([0., 75.]),  # dBZ
    radar_utils.SHI_NAME: numpy.array([0., 450.]),  # unitless
    radar_utils.VIL_NAME: numpy.array([0., 75.]),  # mm
    radar_utils.DIFFERENTIAL_REFL_NAME: numpy.array([-1., 4.]),  # dB
    radar_utils.SPEC_DIFF_PHASE_NAME: numpy.array([-1., 4.]),  # deg km^-1
    radar_utils.CORRELATION_COEFF_NAME: numpy.array([0.7, 1.]),  # unitless
    radar_utils.SPECTRUM_WIDTH_NAME: numpy.array([0., 10.]),  # m s^-1
    radar_utils.VORTICITY_NAME: numpy.array([-0.0075, 0.0075]),  # s^-1
    radar_utils.DIVERGENCE_NAME: numpy.array([-0.0075, 0.0075])  # s^-1
}

DEFAULT_SOUNDING_NORMALIZATION_DICT = {
    soundings_only.RELATIVE_HUMIDITY_NAME: numpy.array([0., 1.]),  # unitless
    soundings_only.TEMPERATURE_NAME: numpy.array([197.4, 311.8]),  # Kelvins
    soundings_only.WIND_SPEED_KEY: numpy.array([0., 64.1]),  # m s^-1
    soundings_only.SPECIFIC_HUMIDITY_NAME:
        numpy.array([0., 0.0223]),  # kg kg^-1
    soundings_only.VIRTUAL_POTENTIAL_TEMPERATURE_NAME:
        numpy.array([285.2, 421.6])  # Kelvins
}


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


def check_soundings(
        sounding_matrix, num_examples=None, num_vertical_levels=None,
        num_pressureless_fields=None):
    """Error-checks storm-centered soundings.

    :param sounding_matrix: numpy array (E x H_s x F_s) of soundings.
    :param num_examples: Number of examples expected in `sounding_matrix`.
    :param num_vertical_levels: Number of vertical levels expected in
        `sounding_matrix`.
    :param num_pressureless_fields: Number of pressureless fields expected in
        `sounding_matrix`.
    """

    error_checking.assert_is_real_numpy_array(sounding_matrix)
    error_checking.assert_is_numpy_array(sounding_matrix, num_dimensions=3)

    expected_dimensions = []
    if num_examples is None:
        expected_dimensions += [sounding_matrix.shape[0]]
    else:
        expected_dimensions += [num_examples]

    if num_vertical_levels is None:
        expected_dimensions += [sounding_matrix.shape[1]]
    else:
        expected_dimensions += [num_vertical_levels]

    if num_pressureless_fields is None:
        expected_dimensions += [sounding_matrix.shape[2]]
    else:
        expected_dimensions += [num_pressureless_fields]

    error_checking.assert_is_numpy_array(
        sounding_matrix, exact_dimensions=numpy.array(expected_dimensions))


def check_target_values(target_values, num_dimensions, num_classes):
    """Error-checks target values.

    :param target_values: numpy array in one of two formats.
    [1] length-E integer numpy array of target values.  All values are -2
        ("dead storm") or 0...[K - 1], where K = number of classes.
    [2] E-by-K numpy array of target values (all 0 or 1, but with array type
        "float64").  If target_matrix[i, k] = 1, the [i]th storm object belongs
        to the [k]th class.  Classes are mutually exclusive and collectively
        exhaustive, so the sum across each row is 1.

    :param num_dimensions: Number of dimensions expected in `target_values`.
    :param num_classes: Number of classes that should be represented in
        `target_values`.
    """

    error_checking.assert_is_integer(num_dimensions)
    error_checking.assert_is_geq(num_dimensions, 1)
    error_checking.assert_is_leq(num_dimensions, 2)
    error_checking.assert_is_integer(num_classes)
    error_checking.assert_is_geq(num_classes, 2)

    num_examples = target_values.shape[0]

    if num_dimensions == 1:
        error_checking.assert_is_numpy_array(
            target_values, exact_dimensions=numpy.array([num_examples]))
        error_checking.assert_is_integer_numpy_array(target_values)

        live_storm_object_indices = numpy.where(
            target_values != labels.DEAD_STORM_INTEGER)[0]
        error_checking.assert_is_geq_numpy_array(
            target_values[live_storm_object_indices], 0)
        error_checking.assert_is_less_than_numpy_array(
            target_values, num_classes)
    else:
        error_checking.assert_is_numpy_array(
            target_values,
            exact_dimensions=numpy.array([num_examples, num_classes]))
        error_checking.assert_is_geq_numpy_array(target_values, 0)
        error_checking.assert_is_leq_numpy_array(target_values, 1)


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
        radar_image_matrix, normalize_by_batch=False, field_names=None,
        normalization_dict=DEFAULT_RADAR_NORMALIZATION_DICT,
        percentile_offset=DEFAULT_PERCENTILE_OFFSET_FOR_NORMALIZATION):
    """Normalizes radar images.

    Specifically, for each field x and each pixel (i, j), the normalization is
    done as follows.

    x_new(i, j) = [x_old(i, j) - x_min] / [x_max - x_min]

    If normalize_by_batch = True, radar_image_matrix is treated as one batch and
    variables have the following meanings.

    x_min = minimum (or a very small percentile) of x-values in the batch
    x_max = maximum (or a very large percentile) of x-values in the batch

    If normalize_by_batch = False, variables have the following meanings.

    x_min = climatological minimum for x (taken from normalization_dict)
    x_max = climatological maximum for x (taken from normalization_dict)

    :param radar_image_matrix: numpy array of radar images.  Dimensions may be
        E x M x N x C or E x M x N x H_r x F_r.
    :param normalize_by_batch: Boolean flag.  See general discussion above.
    :param field_names: [used only if normalize_by_batch = False]
        1-D list with names of radar fields, in the order that they appear in
        radar_image_matrix.  If radar_image_matrix is 4-dimensional, field_names
        must have length C.  If radar_image_matrix is 5-dimensional, field_names
        must have length F_r.  Each field name must be accepted by
        `radar_utils.check_field_name`.
    :param normalization_dict: [used only if normalize_by_batch = False]
        Dictionary, where each key is a field name (from the list `field_names`)
        and each value is a length-2 numpy array with [x_min, x_max].
    :param percentile_offset: [used only if normalize_by_batch = True]
        Offset for x_min and x_max (reduces the effect of outliers).  If
        percentile_offset = q, x_min will be the [q]th percentile and x_max will
        be the [100 - q]th percentile.
    :return: radar_image_matrix: Normalized version of input.  Dimensions are
        the same.
    """

    check_radar_images(
        radar_image_matrix=radar_image_matrix, min_num_dimensions=4,
        max_num_dimensions=5)
    num_fields = radar_image_matrix.shape[-1]

    error_checking.assert_is_boolean(normalize_by_batch)

    if normalize_by_batch:
        error_checking.assert_is_geq(percentile_offset, 0.)
        error_checking.assert_is_leq(
            percentile_offset, MAX_PERCENTILE_OFFSET_FOR_NORMALIZATION)

        for k in range(num_fields):
            this_min_value = numpy.nanpercentile(
                radar_image_matrix[..., k], percentile_offset)
            this_max_value = numpy.nanpercentile(
                radar_image_matrix[..., k], 100. - percentile_offset)
            radar_image_matrix[..., k] = (
                (radar_image_matrix[..., k] - this_min_value) /
                (this_max_value - this_min_value))
    else:
        error_checking.assert_is_string_list(field_names)
        error_checking.assert_is_numpy_array(
            numpy.array(field_names),
            exact_dimensions=numpy.array([num_fields]))

        for k in range(num_fields):
            this_min_value = normalization_dict[field_names[k]][0]
            this_max_value = normalization_dict[field_names[k]][1]
            radar_image_matrix[..., k] = (
                (radar_image_matrix[..., k] - this_min_value) /
                (this_max_value - this_min_value))

    return radar_image_matrix


def normalize_soundings(
        sounding_matrix, pressureless_field_names,
        normalization_dict=DEFAULT_SOUNDING_NORMALIZATION_DICT):
    """Normalizes soundings.

    For wind at each pixel (i, j), the normalization is done as follows.

    wind_speed_new(i, j) = [wind_speed(i, j) - min_wind_speed] /
                           [max_wind_speed - min_wind_speed]
    normalization_ratio = wind_speed_new(i, j) / wind_speed(i, j)
    u_wind_new(i, j) = u_wind(i, j) * normalization_ratio
    v_wind_new(i, j) = v_wind(i, j) * normalization_ratio

    For each other field f at each pixel (i, j), the normalization is done as
    follows.

    x_new(i, j) = [x_old(i, j) - x_min] / [x_max - x_min]

    x_min, x_max, min_wind_speed, and max_wind_speed are climatological extrema
    taken from `normalization_dict`.

    :param sounding_matrix: numpy array (E x H_s x F_s) of soundings.
    :param pressureless_field_names: list (length F_s) with names of
        pressureless fields, in the order that they appear in sounding_matrix.
    :param normalization_dict: Dictionary, where each key is a field name (
        either "wind_speed_m_s01" or an item in `pressureless_field_names`) and
        each value is a length-2 numpy array with [x_min, x_max].
    :return: sounding_matrix: Normalized version of input.  Dimensions are the
        same.
    """

    error_checking.assert_is_string_list(pressureless_field_names)
    error_checking.assert_is_numpy_array(
        numpy.array(pressureless_field_names), num_dimensions=1)

    num_pressureless_fields = len(pressureless_field_names)
    check_soundings(sounding_matrix=sounding_matrix,
                    num_pressureless_fields=num_pressureless_fields)

    done_wind_speed = False

    for k in range(num_pressureless_fields):
        if pressureless_field_names[k] in [soundings_only.U_WIND_NAME,
                                           soundings_only.V_WIND_NAME]:
            if done_wind_speed:
                continue

            u_wind_index = pressureless_field_names.index(
                soundings_only.U_WIND_NAME)
            v_wind_index = pressureless_field_names.index(
                soundings_only.V_WIND_NAME)
            wind_speeds_m_s01 = numpy.sqrt(
                sounding_matrix[..., u_wind_index] ** 2 +
                sounding_matrix[..., v_wind_index] ** 2)

            min_wind_speed_m_s01 = normalization_dict[
                soundings_only.WIND_SPEED_KEY][0]
            max_wind_speed_m_s01 = normalization_dict[
                soundings_only.WIND_SPEED_KEY][1]
            normalized_wind_speeds_m_s01 = (
                (wind_speeds_m_s01 - min_wind_speed_m_s01) /
                (max_wind_speed_m_s01 - min_wind_speed_m_s01)
            )

            normalization_ratios = (
                normalized_wind_speeds_m_s01 / wind_speeds_m_s01)
            normalization_ratios[numpy.isnan(normalization_ratios)] = 0.

            sounding_matrix[..., u_wind_index] = (
                sounding_matrix[..., u_wind_index] * normalization_ratios)
            sounding_matrix[..., v_wind_index] = (
                sounding_matrix[..., v_wind_index] * normalization_ratios)

            done_wind_speed = True
            continue

        this_min_value = normalization_dict[pressureless_field_names[k]][0]
        this_max_value = normalization_dict[pressureless_field_names[k]][1]
        sounding_matrix[..., k] = (
            (sounding_matrix[..., k] - this_min_value) /
            (this_max_value - this_min_value)
        )

    return sounding_matrix


def sample_by_class(
        sampling_fraction_by_class_dict, target_name, target_values,
        num_examples_total, test_mode=False):
    """Randomly draws examples, respecting the fraction for each target class.

    In other words, this method allows "oversampling" and "undersampling" of
    different classes.

    :param sampling_fraction_by_class_dict: See doc for `check_class_fractions`.
    :param target_name: Same.
    :param target_values: See doc for `check_target_values` (format 1).
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
    check_target_values(
        target_values=target_values, num_dimensions=1, num_classes=num_classes)

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
