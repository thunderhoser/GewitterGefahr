"""Unit tests for deep_learning_utils.py"""

import unittest
import numpy
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


if __name__ == '__main__':
    unittest.main()
