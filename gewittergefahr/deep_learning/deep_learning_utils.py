"""Helper methods for deep learning.

In this module, the following letters will be used to denote matrix dimensions.

K = number of classes (possible values of target variable)
E = number of examples
M = number of pixel rows per image
N = number of pixel columns per image
P = number of channels (predictor variables) per image
"""

import numpy
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import error_checking

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


def _check_predictor_matrix(
        predictor_matrix, min_num_dimensions=3, max_num_dimensions=4):
    """Checks predictor matrix for errors.

    :param predictor_matrix: numpy array of predictor images.  Dimensions may be
        E x M x N or E x M x N x C.
    :param min_num_dimensions: Minimum number of dimensions expected in
        `predictor_matrix`.
    :param max_num_dimensions: Max number of dimensions expected in
        `predictor_matrix`.
    """

    error_checking.assert_is_numpy_array_without_nan(predictor_matrix)

    num_dimensions = len(predictor_matrix.shape)
    error_checking.assert_is_geq(num_dimensions, min_num_dimensions)
    error_checking.assert_is_leq(num_dimensions, max_num_dimensions)


def stack_predictor_variables(tuple_of_3d_predictor_matrices):
    """Stacks images with different predictor variables.

    P = number of channels (predictor variables)

    :param tuple_of_3d_predictor_matrices: length-C tuple, where each element is
        an E-by-M-by-N numpy array of predictor images.
    :return: predictor_matrix: E-by-M-by-N-by-C numpy array of predictor
        images.
    """

    predictor_matrix = numpy.stack(tuple_of_3d_predictor_matrices, axis=-1)
    _check_predictor_matrix(
        predictor_matrix=predictor_matrix, min_num_dimensions=4,
        max_num_dimensions=4)

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

    :param predictor_matrix: numpy array of predictor images.  Dimensions must
        be E x M x N x C.
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

    _check_predictor_matrix(
        predictor_matrix, min_num_dimensions=4, max_num_dimensions=4)
    num_predictors = predictor_matrix.shape[-1]

    error_checking.assert_is_boolean(normalize_by_batch)

    if normalize_by_batch:
        error_checking.assert_is_geq(percentile_offset, 0.)
        error_checking.assert_is_leq(
            percentile_offset, MAX_PERCENTILE_OFFSET_FOR_NORMALIZATION)

        num_examples = predictor_matrix.shape[0]
        for i in range(num_examples):
            for m in range(num_predictors):
                this_min_value = numpy.nanpercentile(
                    predictor_matrix[i, ..., m], percentile_offset)
                this_max_value = numpy.nanpercentile(
                    predictor_matrix[i, ..., m], 100. - percentile_offset)

                predictor_matrix[i, ..., m] = (
                    (predictor_matrix[i, ..., m] - this_min_value) /
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
