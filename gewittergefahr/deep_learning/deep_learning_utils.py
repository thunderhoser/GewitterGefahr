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
from gewittergefahr.gg_utils import radar_utils
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


def class_fractions_to_num_points(class_fractions, num_points_to_sample):
    """For each class, converts fraction of points to number of points.

    :param class_fractions: See documentation for `_check_class_fractions`.
    :param num_points_to_sample: Number of points to sample.
    :return: num_points_by_class: length-K numpy array, where the [k]th element
        is the number of points to sample for the [k]th class.
    :raises: ValueError: if sum(class_fractions) != 1.
    """

    _check_class_fractions(class_fractions)
    num_classes = len(class_fractions)
    error_checking.assert_is_integer(num_points_to_sample)
    error_checking.assert_is_geq(num_points_to_sample, num_classes)

    num_points_by_class = numpy.full(num_classes, -1, dtype=int)

    for k in range(num_classes - 1):
        num_points_by_class[k] = int(
            numpy.round(class_fractions[k] * num_points_to_sample))
        num_points_by_class[k] = max([num_points_by_class[k], 1])

        num_points_left = num_points_to_sample - numpy.sum(
            num_points_by_class[:k])
        num_classes_left = num_classes - 1 - k
        num_points_by_class[k] = min(
            [num_points_by_class[k], num_points_left - num_classes_left])

    num_points_by_class[-1] = num_points_to_sample - numpy.sum(
        num_points_by_class[:-1])

    return num_points_by_class


def class_fractions_to_weights(class_fractions):
    """Creates dictionary of class weights.

    This dictionary may be used to weight the loss function for a Keras model.

    :param class_fractions: See documentation for `_check_class_fractions`.
    :return: class_weight_dict: Dictionary, where each key is an integer from
        0...(K - 1) and each value is the corresponding weight.
    """

    class_weights = 1. / class_fractions
    class_weights = class_weights / numpy.sum(class_weights)
    class_weight_dict = {}

    for k in range(len(class_weights)):
        class_weight_dict.update({k: class_weights[k]})
    return class_weight_dict


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

    error_checking.assert_is_geq_numpy_array(target_values, 0)
    num_examples = target_values.shape[0]

    if num_dimensions == 1:
        error_checking.assert_is_numpy_array(
            target_values, exact_dimensions=numpy.array([num_examples]))

        error_checking.assert_is_integer_numpy_array(target_values)
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


def sample_points_by_class(
        target_values, class_fractions, num_points_to_sample, test_mode=False):
    """Samples data points to achieve desired class balance.

    If any class is missing from `target_values`, this method will return None.

    In other words, this method allows "oversampling" and "undersampling" of
    different classes.

    :param target_values: length-E numpy array of target values.  These are
        integers from 0...(K - 1), where K = number of classes.
    :param class_fractions: length-K numpy array of desired class fractions.
    :param num_points_to_sample: Number of data points to keep.
    :param test_mode: Boolean flag.  Always leave this False.
    :return: indices_to_keep: 1-D numpy array with indices of data points to
        keep.  These are indices into `target_values`.
    """

    num_desired_points_by_class = class_fractions_to_num_points(
        class_fractions=class_fractions,
        num_points_to_sample=num_points_to_sample)

    num_classes = len(class_fractions)
    check_target_values(
        target_values, num_dimensions=1, num_classes=num_classes)

    indices_by_class = [
        numpy.where(target_values == k)[0] for k in range(num_classes)]
    num_avail_points_by_class = numpy.array(
        [len(indices_by_class[k]) for k in range(num_classes)])
    if numpy.any(num_avail_points_by_class == 0):
        return None

    if numpy.any(num_avail_points_by_class < num_desired_points_by_class):
        avail_to_desired_ratio_by_class = num_avail_points_by_class.astype(
            float) / num_desired_points_by_class
        num_points_to_sample = int(numpy.floor(
            num_points_to_sample * numpy.min(avail_to_desired_ratio_by_class)))

        num_desired_points_by_class = class_fractions_to_num_points(
            class_fractions=class_fractions,
            num_points_to_sample=num_points_to_sample)

    for k in range(num_classes):
        if not test_mode:
            numpy.random.shuffle(indices_by_class[k])

        indices_by_class[k] = indices_by_class[
            k][:num_desired_points_by_class[k]]

    indices_to_keep = copy.deepcopy(indices_by_class[0])
    for k in range(1, num_classes):
        indices_to_keep = numpy.concatenate((
            indices_to_keep, indices_by_class[k]))

    return indices_to_keep
