"""Unit tests for deep_learning_utils.py"""

import copy
import unittest
import numpy
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils

TOLERANCE = 1e-6

# The following constants are used to test _check_predictor_matrix.
PREDICTOR_MATRIX_1D = numpy.array([1, 2, 3, 4], dtype=numpy.float32)
PREDICTOR_MATRIX_2D = numpy.array([[1, 2, 3, 4],
                                   [5, 6, 7, 8],
                                   [9, 10, 11, 12]], dtype=numpy.float32)

TUPLE_OF_2D_PREDICTOR_MATRICES = (PREDICTOR_MATRIX_2D, PREDICTOR_MATRIX_2D)
PREDICTOR_MATRIX_3D = numpy.stack(TUPLE_OF_2D_PREDICTOR_MATRICES, axis=0)

TUPLE_OF_3D_PREDICTOR_MATRICES = (
    PREDICTOR_MATRIX_3D, PREDICTOR_MATRIX_3D, PREDICTOR_MATRIX_3D,
    PREDICTOR_MATRIX_3D, PREDICTOR_MATRIX_3D, PREDICTOR_MATRIX_3D)
PREDICTOR_MATRIX_4D = numpy.stack(TUPLE_OF_3D_PREDICTOR_MATRICES, axis=-1)

# The following constants are used to test normalize_predictor_matrix.
PERCENTILE_OFFSET_FOR_NORMALIZATION = 0.
PREDICTOR_NAMES = [radar_utils.REFL_NAME, radar_utils.DIFFERENTIAL_REFL_NAME]
PREDICTOR_NORMALIZATION_DICT = {
    radar_utils.DIFFERENTIAL_REFL_NAME: numpy.array([-10., 10.]),
    radar_utils.REFL_NAME: numpy.array([1., 10.])
}

REFL_MATRIX = numpy.array(
    [[0, 1, 2, 3],
     [4, 5, 6, 7]], dtype=numpy.float32)
DIFFERENTIAL_REFL_MATRIX = numpy.array(
    [[2, 4, 6, 8],
     [-1, -3, -5, -7]], dtype=numpy.float32)

THIS_SINGLE_EXAMPLE_MATRIX = numpy.stack(
    (REFL_MATRIX, DIFFERENTIAL_REFL_MATRIX), axis=-1)
PREDICTOR_MATRIX_UNNORMALIZED = numpy.stack(
    (THIS_SINGLE_EXAMPLE_MATRIX, THIS_SINGLE_EXAMPLE_MATRIX), axis=0)

REFL_MATRIX_NORMALIZED_BY_BATCH = (REFL_MATRIX + 0) / 7
DIFF_REFL_MATRIX_NORMALIZED_BY_BATCH = (DIFFERENTIAL_REFL_MATRIX + 7) / 15

THIS_SINGLE_EXAMPLE_MATRIX = numpy.stack(
    (REFL_MATRIX_NORMALIZED_BY_BATCH, DIFF_REFL_MATRIX_NORMALIZED_BY_BATCH),
    axis=-1)
PREDICTOR_MATRIX_NORMALIZED_BY_BATCH = numpy.stack(
    (THIS_SINGLE_EXAMPLE_MATRIX, THIS_SINGLE_EXAMPLE_MATRIX), axis=0)

REFL_MATRIX_NORMALIZED_BY_CLIMO = (REFL_MATRIX - 1) / 9
DIFF_REFL_MATRIX_NORMALIZED_BY_CLIMO = (DIFFERENTIAL_REFL_MATRIX + 10) / 20

THIS_SINGLE_EXAMPLE_MATRIX = numpy.stack(
    (REFL_MATRIX_NORMALIZED_BY_CLIMO, DIFF_REFL_MATRIX_NORMALIZED_BY_CLIMO),
    axis=-1)
PREDICTOR_MATRIX_NORMALIZED_BY_CLIMO = numpy.stack(
    (THIS_SINGLE_EXAMPLE_MATRIX, THIS_SINGLE_EXAMPLE_MATRIX), axis=0)


class DeepLearningUtilsTests(unittest.TestCase):
    """Each method is a unit test for deep_learning_utils.py."""

    def test_check_predictor_matrix_1d(self):
        """Ensures correct output from _check_predictor_matrix.

        In this case, input matrix is 1-D (bad).
        """

        with self.assertRaises(ValueError):
            dl_utils._check_predictor_matrix(PREDICTOR_MATRIX_1D)

    def test_check_predictor_matrix_2d(self):
        """Ensures correct output from _check_predictor_matrix.

        In this case, input matrix is 2-D (bad).
        """

        with self.assertRaises(ValueError):
            dl_utils._check_predictor_matrix(PREDICTOR_MATRIX_2D)

    def test_check_predictor_matrix_3d(self):
        """Ensures correct output from _check_predictor_matrix.

        In this case, input matrix is 3-D (good).
        """

        dl_utils._check_predictor_matrix(PREDICTOR_MATRIX_3D)

    def test_check_predictor_matrix_4d(self):
        """Ensures correct output from _check_predictor_matrix.

        In this case, input matrix is 4-D (good).
        """

        dl_utils._check_predictor_matrix(PREDICTOR_MATRIX_4D)

    def test_stack_predictor_variables(self):
        """Ensures correct output from stack_predictor_variables."""

        this_matrix = dl_utils.stack_predictor_variables(
            TUPLE_OF_3D_PREDICTOR_MATRICES)
        self.assertTrue(numpy.allclose(
            this_matrix, PREDICTOR_MATRIX_4D, atol=TOLERANCE, equal_nan=True))

    def test_normalize_predictor_matrix_by_batch(self):
        """Ensures correct output from normalize_predictor_matrix.

        In this case, normalize_by_batch = True.
        """

        this_predictor_matrix = copy.deepcopy(PREDICTOR_MATRIX_UNNORMALIZED)
        this_predictor_matrix = dl_utils.normalize_predictor_matrix(
            predictor_matrix=this_predictor_matrix, normalize_by_batch=True,
            percentile_offset=PERCENTILE_OFFSET_FOR_NORMALIZATION)

        self.assertTrue(numpy.allclose(
            this_predictor_matrix, PREDICTOR_MATRIX_NORMALIZED_BY_BATCH,
            atol=TOLERANCE, equal_nan=True))

    def test_normalize_predictor_matrix_by_climo(self):
        """Ensures correct output from normalize_predictor_matrix.

        In this case, normalize_by_batch = False.
        """

        this_predictor_matrix = copy.deepcopy(PREDICTOR_MATRIX_UNNORMALIZED)
        this_predictor_matrix = dl_utils.normalize_predictor_matrix(
            predictor_matrix=this_predictor_matrix, normalize_by_batch=False,
            predictor_names=PREDICTOR_NAMES,
            normalization_dict=PREDICTOR_NORMALIZATION_DICT)

        self.assertTrue(numpy.allclose(
            this_predictor_matrix, PREDICTOR_MATRIX_NORMALIZED_BY_CLIMO,
            atol=TOLERANCE, equal_nan=True))


if __name__ == '__main__':
    unittest.main()
