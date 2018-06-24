"""Helper methods for deep learning.

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
import pandas
from gewittergefahr.gg_utils import labels
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import soundings_only
from gewittergefahr.gg_utils import error_checking

TOLERANCE_FOR_FREQUENCY_SUM = 1e-3

DEFAULT_PERCENTILE_OFFSET_FOR_NORMALIZATION = 1.
MAX_PERCENTILE_OFFSET_FOR_NORMALIZATION = 5.

DEFAULT_NORMALIZATION_DICT = {
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
    soundings_only.SPECIFIC_HUMIDITY_NAME: numpy.array([0., 0.0223]),  # kg kg^-1
    soundings_only.VIRTUAL_POTENTIAL_TEMPERATURE_NAME:
        numpy.array([285.2, 421.6])  # Kelvins
}

VECTOR_SUFFIXES_TO_NORMALIZE = [
    soundings.X_COMPONENT_SUFFIX, soundings.Y_COMPONENT_SUFFIX]
VECTOR_SUFFIXES_TO_NOT_NORMALIZE = [
    soundings.SINE_SUFFIX, soundings.COSINE_SUFFIX, soundings.MAGNITUDE_SUFFIX]

# TODO(thunderhoser): Stop using the word "predictor" to mean "radar image".
# There are many different kinds of predictors.


def _check_class_fractions(class_fractions):
    """Checks class fractions for errors.

    :param class_fractions: length-K numpy array, where the [k]th element is the
        desired fraction of data points for the [k]th class.
    :raises: ValueError: if sum(class_fractions) != 1.
    """

    error_checking.assert_is_numpy_array(class_fractions, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(class_fractions, 0.)
    error_checking.assert_is_leq_numpy_array(class_fractions, 1.)

    sum_of_class_fractions = numpy.sum(class_fractions)
    absolute_diff = numpy.absolute(sum_of_class_fractions - 1.)
    if absolute_diff > TOLERANCE_FOR_FREQUENCY_SUM:
        error_string = (
            '\n\n{0:s}\nSum of class fractions (shown above) should be 1.  '
            'Instead, got {1:.4f}.').format(str(class_fractions),
                                            sum_of_class_fractions)
        raise ValueError(error_string)


def class_fractions_to_num_points(class_fraction_dict, num_points_to_sample):
    """For each class, converts fraction of points to number of points.

    :param class_fraction_dict: Dictionary, where each key is a class integer
        (-2 for dead storms) and each value is the corresponding fraction of
        examples to include in each batch.
    :param num_points_to_sample: Number of points to sample.
    :return: num_points_class_dict: Dictionary, where each key is a class
        integer (-2 for dead storms) and each value is the corresponding number
        of examples to include in each batch.
    """

    _check_class_fractions(numpy.array(class_fraction_dict.values()))
    num_classes = len(class_fraction_dict.keys())
    error_checking.assert_is_integer(num_points_to_sample)
    error_checking.assert_is_geq(num_points_to_sample, num_classes)

    num_points_class_dict = {}
    num_points_used = 0
    num_classes_used = 0

    for this_key in class_fraction_dict.keys()[:-1]:
        this_num_points = int(numpy.round(
            class_fraction_dict[this_key] * num_points_to_sample))
        this_num_points = max([this_num_points, 1])

        num_classes_used += 1
        num_classes_left = num_classes - num_classes_used
        this_num_points = min(
            [this_num_points,
             num_points_to_sample - num_points_used - num_classes_left])

        num_points_class_dict.update({this_key: this_num_points})
        num_points_used += this_num_points

    this_key = class_fraction_dict.keys()[-1]
    this_num_points = num_points_to_sample - num_points_used
    num_points_class_dict.update({this_key: this_num_points})
    return num_points_class_dict


def class_fractions_to_weights(class_fraction_dict, binarize_target):
    """Converts sampling fractions to loss-function weights.

    :param class_fraction_dict: Dictionary, where each key is a class integer
        (-2 for dead storms) and each value is the corresponding fraction of
        examples to include in each batch.
    :param binarize_target: Boolean flag.  If True, the target variable will be
        binarized, so that the highest class = 1 and all other classes = 0.
        Otherwise, the target variable will retain the original number of
        classes (except the dead-storm class, which will be mapped from -2 to
        0).
    :return: class_weight_dict: Dictionary, where each key is a class integer
        (no dead storms) and each value is the corresponding weight.
    """

    error_checking.assert_is_boolean(binarize_target)

    if binarize_target:
        max_key = numpy.max(numpy.array(class_fraction_dict.keys()))
        positive_fraction = class_fraction_dict[max_key]
        negative_fraction = 1. - positive_fraction
        new_class_fraction_dict = {0: negative_fraction, 1: positive_fraction}

    else:
        new_class_fraction_dict = copy.deepcopy(class_fraction_dict)

        if labels.DEAD_STORM_INTEGER in class_fraction_dict.keys():
            new_class_fraction_dict[0] = (
                new_class_fraction_dict[0] +
                new_class_fraction_dict[labels.DEAD_STORM_INTEGER])
            del new_class_fraction_dict[labels.DEAD_STORM_INTEGER]

    class_weights = 1. / numpy.array(new_class_fraction_dict.values())
    class_weights = class_weights / numpy.sum(class_weights)
    return dict(zip(new_class_fraction_dict.keys(), class_weights))


def check_predictor_matrix(
        predictor_matrix, min_num_dimensions=3, max_num_dimensions=5):
    """Checks predictor matrix for errors.

    :param predictor_matrix: numpy array of predictor images.  Dimensions may be
        E x M x N or E x M x N x C.
    :param min_num_dimensions: Minimum number of dimensions expected in
        `predictor_matrix`.
    :param max_num_dimensions: Max number of dimensions expected in
        `predictor_matrix`.
    """

    error_checking.assert_is_integer(min_num_dimensions)
    error_checking.assert_is_geq(min_num_dimensions, 3)
    error_checking.assert_is_leq(min_num_dimensions, 5)

    error_checking.assert_is_integer(max_num_dimensions)
    error_checking.assert_is_geq(max_num_dimensions, 3)
    error_checking.assert_is_leq(max_num_dimensions, 5)
    error_checking.assert_is_geq(max_num_dimensions, min_num_dimensions)

    error_checking.assert_is_numpy_array_without_nan(predictor_matrix)

    num_dimensions = len(predictor_matrix.shape)
    error_checking.assert_is_geq(num_dimensions, min_num_dimensions)
    error_checking.assert_is_leq(num_dimensions, max_num_dimensions)


def check_sounding_stat_matrix(sounding_stat_matrix, num_examples=None):
    """Checks matrix of sounding statistics for errors.

    :param sounding_stat_matrix: E-by-S numpy array of sounding statistics.
    :param num_examples: Number of examples expected.
    """

    error_checking.assert_is_real_numpy_array(sounding_stat_matrix)
    error_checking.assert_is_numpy_array(sounding_stat_matrix, num_dimensions=2)
    if num_examples is None:
        return

    expected_dimensions = (num_examples, sounding_stat_matrix.shape[1])
    error_checking.assert_is_numpy_array(
        sounding_stat_matrix, exact_dimensions=expected_dimensions)


def check_sounding_matrix(
        sounding_matrix, num_examples=None, num_vertical_levels=None,
        num_pressureless_fields=None):
    """Checks sounding matrix for errors.

    E = number of examples
    H = number of vertical levels
    V = number of sounding variables (pressureless fields)

    :param sounding_matrix: E-by-H-by-V numpy array of soundings.
    :param num_examples: Expected number of examples (E).
    :param num_vertical_levels: Expected number of vertical levels (H).
    :param num_pressureless_fields: Expected number of pressureless fields (V).
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
    """Checks array of target values for errors.

    :param target_values: numpy array of target values.  This may be in one of
        two formats.
    [1] length-E numpy array of target values.  These are integers from
        0...(K - 1), where K = number of classes.
    [2] E-by-K numpy array of flags (all 0 or 1, but technically the type is
        "float64").  If target_matrix[i, k] = 1, the [k]th class is the outcome
        for the [i]th example.  The sum across each row is 1 (classes are
        mutually exclusive and collectively exhaustive).

    :param num_dimensions: Number of dimensions expected in `target_values`
        (either 1 or 2).
    :param num_classes: Number of classes expected in `target_values`.
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

        error_checking.assert_is_leq_numpy_array(target_values, 1)


def stack_predictor_variables(tuple_of_3d_predictor_matrices):
    """Stacks images with different predictor variables.

    C = number of channels (predictor variables)

    :param tuple_of_3d_predictor_matrices: length-C tuple, where each element is
        an E-by-M-by-N numpy array of predictor images.
    :return: predictor_matrix: E-by-M-by-N-by-C numpy array of predictor
        images.
    """

    predictor_matrix = numpy.stack(tuple_of_3d_predictor_matrices, axis=-1)
    check_predictor_matrix(
        predictor_matrix=predictor_matrix, min_num_dimensions=4,
        max_num_dimensions=4)

    return predictor_matrix


def stack_heights(tuple_of_4d_predictor_matrices):
    """Stacks predictor images from different heights.

    D = number of heights

    :param tuple_of_4d_predictor_matrices: length-D tuple, where each element is
        an E-by-M-by-N-by-C numpy array of predictor images.
    :return: predictor_matrix: E-by-M-by-N-by-D-by-C numpy array of predictor
        images.
    """

    predictor_matrix = numpy.stack(tuple_of_4d_predictor_matrices, axis=-2)
    check_predictor_matrix(
        predictor_matrix=predictor_matrix, min_num_dimensions=5,
        max_num_dimensions=5)

    return predictor_matrix


def normalize_predictor_matrix(
        predictor_matrix, normalize_by_batch=False, predictor_names=None,
        normalization_dict=DEFAULT_NORMALIZATION_DICT, percentile_offset=1.):
    """Normalizes predictor matrix.

    Specifically, for each predictor x and each pixel (i, j), the normalization
    is done as follows.

    x_new(i, j) = [x_old(i, j) - x_min] / [x_max - x_min]

    If normalize_by_batch = True, predictor_matrix is treated as a batch and
    variables have the following meanings.

    x_min = minimum (or a very small percentile) of all x-values in the batch
    x_max = max (or a very large percentile) of all x-values in the batch

    If normalize_by_batch = False, variables have the following meanings.

    x_min = climatological minimum for x (taken from normalization_dict)
    x_max = climatological max for x (taken from normalization_dict)

    :param predictor_matrix: numpy array of predictor images.  Dimensions may be
        E x M x N x C or E x M x N x D x C.
    :param normalize_by_batch: Boolean flag.  If True, x_min and x_max are based
        on the batch.  If False, x_min and x_max are based on climatology.
    :param predictor_names: [used only if normalize_by_batch = False]
        length-C list of predictor names, indicating the order of predictors in
        predictor_matrix.
    :param normalization_dict: [used only if normalize_by_batch = False]
        Dictionary, where each key is a predictor name and each value is a
        length-2 numpy array with (x_min, x_max).
    :param percentile_offset: [used only if normalize_by_batch = True]
        Offset for x_min and x_max (reduces the effect of outliers).  For
        example, if percentile_offset = 1, x_min and x_max will actually be the
        1st and 99th percentiles, respectively.  If you want the strict min and
        max, set this to 0.
    :return: predictor_matrix: Normalized version of input.  Dimensions are the
        same.
    """

    check_predictor_matrix(
        predictor_matrix, min_num_dimensions=4, max_num_dimensions=5)
    num_predictors = predictor_matrix.shape[-1]

    error_checking.assert_is_boolean(normalize_by_batch)

    if normalize_by_batch:
        error_checking.assert_is_geq(percentile_offset, 0.)
        error_checking.assert_is_leq(
            percentile_offset, MAX_PERCENTILE_OFFSET_FOR_NORMALIZATION)

        for m in range(num_predictors):
            this_min_value = numpy.nanpercentile(
                predictor_matrix[..., m], percentile_offset)
            this_max_value = numpy.nanpercentile(
                predictor_matrix[..., m], 100. - percentile_offset)
            predictor_matrix[..., m] = (
                (predictor_matrix[..., m] - this_min_value) /
                (this_max_value - this_min_value))

    else:
        error_checking.assert_is_string_list(predictor_names)
        error_checking.assert_is_numpy_array(
            numpy.array(predictor_names),
            exact_dimensions=numpy.array([num_predictors]))

        for m in range(num_predictors):
            this_min_value = normalization_dict[predictor_names[m]][0]
            this_max_value = normalization_dict[predictor_names[m]][1]
            predictor_matrix[..., m] = (
                (predictor_matrix[..., m] - this_min_value) /
                (this_max_value - this_min_value))

    return predictor_matrix


def normalize_sounding_statistics(
        sounding_stat_matrix, statistic_names, metadata_table,
        normalize_by_batch=False, percentile_offset=1.):
    """Normalizes sounding statistics.

    For each scalar statistic c and example i, the normalization is as follows.

    c_new(i) = [c_old(i) - c_min] / [c_max - c_min]

    For each vector statistic and example i, the x- and y-components are
    normalized as shown below.  Then the sine, cosine, and magnitude are
    computed accordingly.

    x_new(i) = [x_old(i) - x_min] / [x_max - x_min]
    y_new(i) = [y_old(i) - y_min] / [y_max - y_min]

    If normalize_by_batch = True, sounding_stat_matrix is treated as a batch and
    variables have the following meanings.

    c_min = minimum (or small percentile) of all c-values in the batch
    c_max = max (or large percentile) of all c-values in the batch

    If normalize_by_batch = False, variables have the following meanings.

    c_min = climatological minimum for c (taken from metadata_table)
    c_max = climatological max for c (taken from metadata_table)

    --- DEFINITIONS ---

    S = number of statistics

    :param sounding_stat_matrix: E-by-S numpy array of sounding stats.
    :param statistic_names: length-S list with names of sounding stats.
    :param metadata_table: pandas DataFrame created by
        `soundings.read_metadata_for_statistics`.
    :param normalize_by_batch: See discussion above.
    :param percentile_offset: [used only if normalize_by_batch = True]
        See doc for `normalize_predictor_matrix`.
    :return: sounding_stat_matrix: Normalized version of input.  Dimensions are
        the same.
    """

    error_checking.assert_is_numpy_array(sounding_stat_matrix, num_dimensions=2)
    num_statistics = sounding_stat_matrix.shape[1]

    error_checking.assert_is_boolean(normalize_by_batch)
    if normalize_by_batch:
        error_checking.assert_is_geq(percentile_offset, 0.)
        error_checking.assert_is_leq(
            percentile_offset, MAX_PERCENTILE_OFFSET_FOR_NORMALIZATION)

    error_checking.assert_is_string_list(statistic_names)
    error_checking.assert_is_numpy_array(
        numpy.array(statistic_names),
        exact_dimensions=numpy.array([num_statistics]))

    basic_statistic_names = [''] * num_statistics
    vector_suffixes = [''] * num_statistics
    for j in range(num_statistics):
        basic_statistic_names[j], vector_suffixes[j] = (
            soundings.remove_vector_suffix_from_stat_name(
                statistic_name=statistic_names[j],
                metadata_table=metadata_table))

    sounding_stat_table = pandas.DataFrame(
        sounding_stat_matrix, columns=statistic_names)

    for j in range(num_statistics):
        if vector_suffixes[j] in VECTOR_SUFFIXES_TO_NOT_NORMALIZE:
            continue

        if normalize_by_batch:
            this_min_value = numpy.nanpercentile(
                sounding_stat_table[statistic_names[j]].values,
                percentile_offset)
            this_max_value = numpy.nanpercentile(
                sounding_stat_table[statistic_names[j]].values,
                100. - percentile_offset)
        else:
            this_row = numpy.where(
                metadata_table[soundings.STATISTIC_NAME_COLUMN].values ==
                basic_statistic_names[j])[0][0]

            if vector_suffixes[j] == soundings.Y_COMPONENT_SUFFIX:
                this_min_value = metadata_table[
                    soundings.MIN_VALUES_FOR_NORM_COLUMN].values[this_row][1]
                this_max_value = metadata_table[
                    soundings.MAX_VALUES_FOR_NORM_COLUMN].values[this_row][1]
            else:
                this_min_value = metadata_table[
                    soundings.MIN_VALUES_FOR_NORM_COLUMN].values[this_row][0]
                this_max_value = metadata_table[
                    soundings.MAX_VALUES_FOR_NORM_COLUMN].values[this_row][0]

        sounding_stat_table[statistic_names[j]] = (
            (sounding_stat_table[statistic_names[j]] - this_min_value) /
            (this_max_value - this_min_value))

    for j in range(num_statistics):
        if vector_suffixes[j] != soundings.X_COMPONENT_SUFFIX:
            continue

        this_x_component_name = soundings.add_vector_suffix_to_stat_name(
            basic_statistic_name=basic_statistic_names[j],
            vector_suffix=soundings.X_COMPONENT_SUFFIX)
        this_y_component_name = soundings.add_vector_suffix_to_stat_name(
            basic_statistic_name=basic_statistic_names[j],
            vector_suffix=soundings.Y_COMPONENT_SUFFIX)
        this_sine_name = soundings.add_vector_suffix_to_stat_name(
            basic_statistic_name=basic_statistic_names[j],
            vector_suffix=soundings.SINE_SUFFIX)
        this_cosine_name = soundings.add_vector_suffix_to_stat_name(
            basic_statistic_name=basic_statistic_names[j],
            vector_suffix=soundings.COSINE_SUFFIX)
        this_magnitude_name = soundings.add_vector_suffix_to_stat_name(
            basic_statistic_name=basic_statistic_names[j],
            vector_suffix=soundings.MAGNITUDE_SUFFIX)

        sounding_stat_table[this_magnitude_name] = numpy.sqrt(
            sounding_stat_table[this_x_component_name].values ** 2 +
            sounding_stat_table[this_y_component_name].values ** 2)
        sounding_stat_table[this_sine_name] = (
            sounding_stat_table[this_y_component_name].values /
            sounding_stat_table[this_magnitude_name].values)
        sounding_stat_table[this_cosine_name] = (
            sounding_stat_table[this_x_component_name].values /
            sounding_stat_table[this_magnitude_name].values)

        bad_indices = numpy.where(
            sounding_stat_table[this_magnitude_name].values > 1.)[0]
        sounding_stat_table[this_magnitude_name].values[bad_indices] = 1.
        sounding_stat_table[this_x_component_name].values[bad_indices] = (
            sounding_stat_table[this_cosine_name].values[bad_indices] *
            sounding_stat_table[this_magnitude_name].values[bad_indices])
        sounding_stat_table[this_y_component_name].values[bad_indices] = (
            sounding_stat_table[this_sine_name].values[bad_indices] *
            sounding_stat_table[this_magnitude_name].values[bad_indices])

    sounding_stat_matrix = sounding_stat_table.as_matrix()
    sounding_stat_matrix[sounding_stat_matrix < 0.] = 0.
    sounding_stat_matrix[sounding_stat_matrix > 1.] = 1.
    sounding_stat_matrix[numpy.isnan(sounding_stat_matrix)] = 0.5
    return sounding_stat_matrix


def normalize_soundings(
        sounding_matrix, pressureless_field_names,
        normalization_dict=DEFAULT_SOUNDING_NORMALIZATION_DICT):
    """Normalizes soundings.

    E = number of examples
    H = number of vertical levels
    V = number of sounding variables (pressureless fields)

    For the wind vector at each pixel (i, j), the normalization is done as
    follows.

    normalized_wind_speed(i, j) = [wind_speed(i, j) - min_wind_speed] /
                                  [max_wind_speed - min_wind_speed]
    normalization_ratio = normalized_wind_speed(i, j) / wind_speed(i, j)
    normalized_u_wind(i, j) = u_wind(i, j) * normalization_ratio
    normalized_v_wind(i, j) = v_wind(i, j) * normalization_ratio

    For each pressureless field x (other than wind) and each pixel (i, j), the
    normalization is done as follows.

    x_normalized(i, j) = [x(i, j) - x_min] / [x_max - x_min]

    x_min, x_max, min_wind_speed, and max_wind_speed come from the input
    argument `normalization_dict`.

    :param sounding_matrix: E-by-H-by-V numpy array of soundings.
    :param pressureless_field_names: length-V numpy array with names of
        pressureless fields.
    :param normalization_dict: Dictionary, where each key is the name of a
        pressureless field and each value is a length-2 numpy array with
        (x_min, x_max).
    :return: sounding_matrix: Normalized version of input.  Dimensions are the
        same.
    """

    error_checking.assert_is_string_list(pressureless_field_names)
    error_checking.assert_is_numpy_array(
        numpy.array(pressureless_field_names), num_dimensions=1)

    num_pressureless_fields = len(pressureless_field_names)
    check_sounding_matrix(
        sounding_matrix=sounding_matrix,
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


def sample_points_by_class(
        target_values, target_name, class_fraction_dict, num_points_to_sample,
        test_mode=False):
    """Samples data points to achieve desired class balance.

    If any class is missing from `target_values`, this method will return None.

    In other words, this method allows "oversampling" and "undersampling" of
    different classes.

    :param target_values: length-E numpy array of target values.  These are
        integers from 0...(K - 1), where K = number of classes.  If the target
        variable is based on wind speed, some values may be -2 ("dead storm").
    :param target_name: Name of target variable.
    :param class_fraction_dict: Dictionary, where each key is a class integer
        (-2 for dead storms) and each value is the corresponding fraction of
        examples to include in each batch.
    :param num_points_to_sample: Number of data points to keep.
    :param test_mode: Boolean flag.  Always leave this False.
    :return: indices_to_keep: 1-D numpy array with indices of data points to
        keep.  These are indices into `target_values`.
    """

    num_desired_points_class_dict = class_fractions_to_num_points(
        class_fraction_dict=class_fraction_dict,
        num_points_to_sample=num_points_to_sample)

    num_classes = labels.column_name_to_num_classes(
        column_name=target_name, include_dead_storms=False)
    num_extended_classes = labels.column_name_to_num_classes(
        column_name=target_name, include_dead_storms=True)

    if num_extended_classes != len(num_desired_points_class_dict.keys()):
        error_string = (
            'Target variable ("{0:s}") has {1:d} extended classes, but '
            '`num_desired_points_class_dict` has {2:d} entries.'
        ).format(target_name, num_extended_classes,
                 num_desired_points_class_dict.keys())
        raise ValueError(error_string)

    check_target_values(
        target_values, num_dimensions=1, num_classes=num_classes)

    indices_by_class_dict = {}
    num_desired_points_by_class = []
    num_avail_points_by_class = []

    for this_key in num_desired_points_class_dict.keys():
        these_values = numpy.where(target_values == this_key)[0]
        num_desired_points_by_class.append(
            num_desired_points_class_dict[this_key])
        num_avail_points_by_class.append(len(these_values))

        if (num_desired_points_by_class[-1] > 0 and
                num_avail_points_by_class[-1] == 0):
            return None

        indices_by_class_dict.update({this_key: these_values})

    num_desired_points_by_class = numpy.array(num_desired_points_by_class)
    num_avail_points_by_class = numpy.array(num_avail_points_by_class)

    if numpy.any(num_avail_points_by_class < num_desired_points_by_class):
        avail_to_desired_ratio_by_class = num_avail_points_by_class.astype(
            float) / num_desired_points_by_class
        num_points_to_sample = int(numpy.floor(
            num_points_to_sample * numpy.min(avail_to_desired_ratio_by_class)))

        num_desired_points_class_dict = class_fractions_to_num_points(
            class_fraction_dict=class_fraction_dict,
            num_points_to_sample=num_points_to_sample)

    indices_to_keep = numpy.array([], dtype=int)
    for this_key in indices_by_class_dict.keys():
        if not test_mode:
            numpy.random.shuffle(indices_by_class_dict[this_key])

        this_num_points = num_desired_points_class_dict[this_key]
        indices_to_keep = numpy.concatenate((
            indices_to_keep, indices_by_class_dict[this_key][:this_num_points]))

    return indices_to_keep
