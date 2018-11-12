"""Unit tests for standalone_utils.py."""

import unittest
import numpy
from gewittergefahr.deep_learning import standalone_utils
from gewittergefahr.deep_learning import architecture_utils

TOLERANCE = 1e-6

# The following constants are used to test do_2d_convolution.
THIS_FIRST_MATRIX = numpy.array([[1, 2, 3, 4, 5, 6, 7],
                                 [8, 9, 10, 11, 12, 13, 14],
                                 [15, 16, 17, 18, 19, 20, 21],
                                 [8, 9, 10, 11, 12, 13, 14],
                                 [1, 2, 3, 4, 5, 6, 7]])

THIS_SECOND_MATRIX = numpy.array([[0, 4, 8, 32, 34, 36, 38],
                                  [1, 5, 9, 33, 35, 37, 39],
                                  [2, 6, 10, 34, 36, 38, 40],
                                  [3, 7, 11, 35, 37, 39, 41],
                                  [4, 8, 12, 36, 38, 40, 42]])

ORIG_FEATURE_MATRIX_2D = numpy.stack(
    (THIS_FIRST_MATRIX, THIS_SECOND_MATRIX), axis=-1
).astype(float)

KERNEL_MATRIX_2D = numpy.array([[0, 1, 0],
                                [1, -4, 1],
                                [0, 1, 0]], dtype=float)
KERNEL_MATRIX_2D = numpy.stack((KERNEL_MATRIX_2D, KERNEL_MATRIX_2D), axis=-1)
KERNEL_MATRIX_2D = numpy.expand_dims(KERNEL_MATRIX_2D, axis=-1)

THIS_FIRST_MATRIX = numpy.array([[0, 0, 0, 0, 0],
                                 [-14, -14, -14, -14, -14],
                                 [0, 0, 0, 0, 0]])

THIS_SECOND_MATRIX = numpy.array([[0, 20, -22, 0, 0],
                                  [0, 20, -22, 0, 0],
                                  [0, 20, -22, 0, 0]])

FEATURE_MATRIX_2D_PADDING0_STRIDE1 = THIS_FIRST_MATRIX + THIS_SECOND_MATRIX
FEATURE_MATRIX_2D_PADDING0_STRIDE1 = numpy.expand_dims(
    FEATURE_MATRIX_2D_PADDING0_STRIDE1, axis=0
).astype(float)
FEATURE_MATRIX_2D_PADDING0_STRIDE1 = numpy.expand_dims(
    FEATURE_MATRIX_2D_PADDING0_STRIDE1, axis=-1)

THIS_FIRST_MATRIX = THIS_FIRST_MATRIX[::2, ::2]
THIS_SECOND_MATRIX = THIS_SECOND_MATRIX[::2, ::2]

FEATURE_MATRIX_2D_PADDING0_STRIDE2 = THIS_FIRST_MATRIX + THIS_SECOND_MATRIX
FEATURE_MATRIX_2D_PADDING0_STRIDE2 = numpy.expand_dims(
    FEATURE_MATRIX_2D_PADDING0_STRIDE2, axis=0
).astype(float)
FEATURE_MATRIX_2D_PADDING0_STRIDE2 = numpy.expand_dims(
    FEATURE_MATRIX_2D_PADDING0_STRIDE2, axis=-1)

THIS_FIRST_MATRIX = numpy.array([[6, 5, 4, 3, 2, 1, -8],
                                 [-7, 0, 0, 0, 0, 0, -15],
                                 [-28, -14, -14, -14, -14, -14, -36],
                                 [-7, 0, 0, 0, 0, 0, -15],
                                 [6, 5, 4, 3, 2, 1, -8]])

THIS_SECOND_MATRIX = numpy.array([[5, -3, 13, -53, -33, -35, -77],
                                  [3, 0, 20, -22, 0, 0, -41],
                                  [2, 0, 20, -22, 0, 0, -42],
                                  [1, 0, 20, -22, 0, 0, -43],
                                  [-5, -9, 7, -59, -39, -41, -87]])

FEATURE_MATRIX_2D_PADDING1_STRIDE1 = THIS_FIRST_MATRIX + THIS_SECOND_MATRIX
FEATURE_MATRIX_2D_PADDING1_STRIDE1 = numpy.expand_dims(
    FEATURE_MATRIX_2D_PADDING1_STRIDE1, axis=0
).astype(float)
FEATURE_MATRIX_2D_PADDING1_STRIDE1 = numpy.expand_dims(
    FEATURE_MATRIX_2D_PADDING1_STRIDE1, axis=-1)

THIS_FIRST_MATRIX = THIS_FIRST_MATRIX[::2, ::2]
THIS_SECOND_MATRIX = THIS_SECOND_MATRIX[::2, ::2]

FEATURE_MATRIX_2D_PADDING1_STRIDE2 = THIS_FIRST_MATRIX + THIS_SECOND_MATRIX
FEATURE_MATRIX_2D_PADDING1_STRIDE2 = numpy.expand_dims(
    FEATURE_MATRIX_2D_PADDING1_STRIDE2, axis=0
).astype(float)
FEATURE_MATRIX_2D_PADDING1_STRIDE2 = numpy.expand_dims(
    FEATURE_MATRIX_2D_PADDING1_STRIDE2, axis=-1)

# The following constants are used to test do_activation.
ACTIVATION_FUNCTION_NAME = architecture_utils.RELU_FUNCTION_STRING
ALPHA_FOR_ACTIVATION = 0.

PRE_ACTIVATION_MATRIX = numpy.array([[1, -2, 3, -4, 5],
                                     [-2, 4, -6, 8, -10],
                                     [0, 0, 0, 0, 0]], dtype=float)

POST_ACTIVATION_MATRIX = numpy.array([[1, 0, 3, 0, 5],
                                      [0, 4, 0, 8, 0],
                                      [0, 0, 0, 0, 0]], dtype=float)

# The following constants are used to test do_2d_pooling.
POOLING_STRIDE_LENGTH_PX = 2

UNPOOLED_MATRIX = numpy.array([[1, 2, 3, 4, 5, 6, 7],
                               [8, 9, 10, 11, 12, 13, 14],
                               [15, 16, 17, 18, 19, 20, 21],
                               [8, 9, 10, 11, 12, 13, 14],
                               [1, 2, 3, 4, 5, 6, 7]])

UNPOOLED_MATRIX = numpy.expand_dims(UNPOOLED_MATRIX, axis=0).astype(float)
UNPOOLED_MATRIX = numpy.expand_dims(UNPOOLED_MATRIX, axis=-1)

MAX_POOLED_MATRIX = numpy.array([[9, 11, 13],
                                 [16, 18, 20]])

MAX_POOLED_MATRIX = numpy.expand_dims(MAX_POOLED_MATRIX, axis=0).astype(float)
MAX_POOLED_MATRIX = numpy.expand_dims(MAX_POOLED_MATRIX, axis=-1)

MEAN_POOLED_MATRIX = numpy.array([[5, 7, 9],
                                  [12, 14, 16]])

MEAN_POOLED_MATRIX = numpy.expand_dims(MEAN_POOLED_MATRIX, axis=0).astype(float)
MEAN_POOLED_MATRIX = numpy.expand_dims(MEAN_POOLED_MATRIX, axis=-1)

# The following constants are used to test do_batch_normalization.
THIS_FIRST_MATRIX = numpy.array([[1, 2, 3, 4],
                                 [5, 6, 7, 8]])

THIS_SECOND_MATRIX = numpy.array([[0, 0, 0, 0],
                                  [-2, -2, -2, -2]])

THIS_THIRD_MATRIX = numpy.array([[2, 4, 6, 8],
                                 [0, -1, -2, -3]])

UNNORMALIZED_MATRIX = numpy.stack(
    (THIS_FIRST_MATRIX, THIS_SECOND_MATRIX, THIS_THIRD_MATRIX), axis=0
).astype(float)
UNNORMALIZED_MATRIX = numpy.expand_dims(UNNORMALIZED_MATRIX, axis=-1)

MEAN_MATRIX = numpy.array([[1, 2, 3, 4],
                           [1, 1, 1, 1]])
MEAN_MATRIX = numpy.stack(
    (MEAN_MATRIX, MEAN_MATRIX, MEAN_MATRIX), axis=0
).astype(float)
MEAN_MATRIX = numpy.expand_dims(MEAN_MATRIX, axis=-1)

STANDARD_DEVIATION_MATRIX = numpy.array([[2, 8, 18, 32],
                                         [26, 38, 54, 74]], dtype=float)
STANDARD_DEVIATION_MATRIX = numpy.sqrt(STANDARD_DEVIATION_MATRIX / 2)

STANDARD_DEVIATION_MATRIX = numpy.stack(
    (STANDARD_DEVIATION_MATRIX, STANDARD_DEVIATION_MATRIX,
     STANDARD_DEVIATION_MATRIX),
    axis=0)
STANDARD_DEVIATION_MATRIX = numpy.expand_dims(
    STANDARD_DEVIATION_MATRIX, axis=-1)

CENTERED_NORMALIZED_MATRIX = (
    (UNNORMALIZED_MATRIX - MEAN_MATRIX) / STANDARD_DEVIATION_MATRIX
)

SHIFT_PARAMETER = 5.
SCALE_PARAMETER = 0.5
UNCENTERED_NORMALIZED_MATRIX = (
    SHIFT_PARAMETER + SCALE_PARAMETER * CENTERED_NORMALIZED_MATRIX)


class StandaloneUtilsTests(unittest.TestCase):
    """Each method is a unit test for standalone_utils.py."""

    def test_do_2d_convolution_padding0_stride1(self):
        """Ensures correct output from do_2d_convolution.

        In this case, edges are *not* padded and stride length = 1.
        """

        this_feature_matrix = standalone_utils.do_2d_convolution(
            feature_matrix=ORIG_FEATURE_MATRIX_2D + 0.,
            kernel_matrix=KERNEL_MATRIX_2D, pad_edges=False, stride_length_px=1)

        self.assertTrue(numpy.allclose(
            this_feature_matrix, FEATURE_MATRIX_2D_PADDING0_STRIDE1,
            atol=TOLERANCE))

    def test_do_2d_convolution_padding0_stride2(self):
        """Ensures correct output from do_2d_convolution.

        In this case, edges are *not* padded and stride length = 2.
        """

        this_feature_matrix = standalone_utils.do_2d_convolution(
            feature_matrix=ORIG_FEATURE_MATRIX_2D + 0.,
            kernel_matrix=KERNEL_MATRIX_2D, pad_edges=False, stride_length_px=2)

        self.assertTrue(numpy.allclose(
            this_feature_matrix, FEATURE_MATRIX_2D_PADDING0_STRIDE2,
            atol=TOLERANCE))

    def test_do_2d_convolution_padding1_stride1(self):
        """Ensures correct output from do_2d_convolution.

        In this case, edges are padded and stride length = 1.
        """

        this_feature_matrix = standalone_utils.do_2d_convolution(
            feature_matrix=ORIG_FEATURE_MATRIX_2D + 0.,
            kernel_matrix=KERNEL_MATRIX_2D, pad_edges=True, stride_length_px=1)

        self.assertTrue(numpy.allclose(
            this_feature_matrix, FEATURE_MATRIX_2D_PADDING1_STRIDE1,
            atol=TOLERANCE))

    def test_do_2d_convolution_padding1_stride2(self):
        """Ensures correct output from do_2d_convolution.

        In this case, edges are padded and stride length = 2.
        """

        this_feature_matrix = standalone_utils.do_2d_convolution(
            feature_matrix=ORIG_FEATURE_MATRIX_2D + 0.,
            kernel_matrix=KERNEL_MATRIX_2D, pad_edges=True, stride_length_px=2)

        self.assertTrue(numpy.allclose(
            this_feature_matrix, FEATURE_MATRIX_2D_PADDING1_STRIDE2,
            atol=TOLERANCE))

    def test_do_activation(self):
        """Ensures correct output from do_activation."""

        this_feature_matrix = standalone_utils.do_activation(
            input_values=PRE_ACTIVATION_MATRIX + 0.,
            function_name=ACTIVATION_FUNCTION_NAME, alpha=ALPHA_FOR_ACTIVATION)

        self.assertTrue(numpy.allclose(
            this_feature_matrix, POST_ACTIVATION_MATRIX, atol=TOLERANCE))

    def test_do_2d_pooling_max(self):
        """Ensures correct output from do_2d_pooling.

        This case is MAX-pooling, rather than MEAN-pooling.
        """

        this_feature_matrix = standalone_utils.do_2d_pooling(
            feature_matrix=UNPOOLED_MATRIX + 0.,
            stride_length_px=POOLING_STRIDE_LENGTH_PX,
            pooling_type_string=standalone_utils.MAX_POOLING_TYPE_STRING)

        self.assertTrue(numpy.allclose(
            this_feature_matrix, MAX_POOLED_MATRIX, atol=TOLERANCE))

    def test_do_2d_pooling_mean(self):
        """Ensures correct output from do_2d_pooling.

        This case is MEAN-pooling, rather than MAX-pooling.
        """

        this_feature_matrix = standalone_utils.do_2d_pooling(
            feature_matrix=UNPOOLED_MATRIX + 0.,
            stride_length_px=POOLING_STRIDE_LENGTH_PX,
            pooling_type_string=standalone_utils.MEAN_POOLING_TYPE_STRING)

        self.assertTrue(numpy.allclose(
            this_feature_matrix, MEAN_POOLED_MATRIX, atol=TOLERANCE))

    def test_do_batch_normalization_centered(self):
        """Ensures correct output from do_batch_normalization.

        In this case the resulting distribution is centered, because scale = 1
        and shift = 0.
        """

        this_feature_matrix = standalone_utils.do_batch_normalization(
            feature_matrix=UNNORMALIZED_MATRIX + 0., scale_parameter=1.,
            shift_parameter=0.)

        self.assertTrue(numpy.allclose(
            this_feature_matrix, CENTERED_NORMALIZED_MATRIX, atol=TOLERANCE))

    def test_do_batch_normalization_uncentered(self):
        """Ensures correct output from do_batch_normalization.

        In this case the resulting distribution is uncentered, because
        scale != 1 and shift != 0.
        """

        this_feature_matrix = standalone_utils.do_batch_normalization(
            feature_matrix=UNNORMALIZED_MATRIX + 0.,
            scale_parameter=SCALE_PARAMETER, shift_parameter=SHIFT_PARAMETER)

        self.assertTrue(numpy.allclose(
            this_feature_matrix, UNCENTERED_NORMALIZED_MATRIX, atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
