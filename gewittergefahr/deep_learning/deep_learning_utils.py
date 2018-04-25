"""Helper methods for deep learning.

In this module, the following letters will be used to denote matrix dimensions.

K = number of classes (possible values of target variable)
E = number of examples
M = number of pixel rows per image
N = number of pixel columns per image
P = number of channels (predictor variables) per image
"""

import numpy
from gewittergefahr.gg_utils import error_checking


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
