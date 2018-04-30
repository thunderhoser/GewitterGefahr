"""Unit tests for deep_learning_utils.py"""

import copy
import unittest
import numpy
import keras
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils

TOLERANCE = 1e-6

# The following constants are used to test check_predictor_matrix.
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

TUPLE_OF_4D_PREDICTOR_MATRICES = (
    PREDICTOR_MATRIX_4D, PREDICTOR_MATRIX_4D, PREDICTOR_MATRIX_4D,
    PREDICTOR_MATRIX_4D, PREDICTOR_MATRIX_4D)
PREDICTOR_MATRIX_5D = numpy.stack(TUPLE_OF_4D_PREDICTOR_MATRICES, axis=-2)

# The following constants are used to test check_target_values.
TOY_TARGET_VALUES_1D_BINARY = numpy.array(
    [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1], dtype=int)
TOY_TARGET_VALUES_1D_MULTICLASS = numpy.array(
    [1, 2, 3, 4, 0, 2, 0, 1, 1, 5, 2], dtype=int)

TOY_TARGET_MATRIX_BINARY = keras.utils.to_categorical(
    TOY_TARGET_VALUES_1D_BINARY, 2)
TOY_TARGET_MATRIX_MULTICLASS = keras.utils.to_categorical(
    TOY_TARGET_VALUES_1D_MULTICLASS,
    numpy.max(TOY_TARGET_VALUES_1D_MULTICLASS) + 1)

# The following constants are used to test _class_fractions_to_num_points.
TOY_CLASS_FRACTIONS_BINARY = numpy.array([0.1, 0.9])
TOY_CLASS_FRACTIONS_TERNARY = numpy.array([0.1, 0.2, 0.7])

NUM_POINTS_TO_SAMPLE_LARGE = 17
NUM_POINTS_BY_CLASS_BINARY_LARGE = numpy.array([2, 15], dtype=int)
NUM_POINTS_BY_CLASS_TERNARY_LARGE = numpy.array([2, 3, 12], dtype=int)

NUM_POINTS_TO_SAMPLE_MEDIUM = 8
NUM_POINTS_BY_CLASS_BINARY_MEDIUM = numpy.array([1, 7], dtype=int)
NUM_POINTS_BY_CLASS_TERNARY_MEDIUM = numpy.array([1, 2, 5], dtype=int)

NUM_POINTS_TO_SAMPLE_SMALL = 4
NUM_POINTS_BY_CLASS_BINARY_SMALL = numpy.array([1, 3], dtype=int)
NUM_POINTS_BY_CLASS_TERNARY_SMALL = numpy.array([1, 1, 2], dtype=int)

NUM_POINTS_TO_SAMPLE_XSMALL = 3
NUM_POINTS_BY_CLASS_BINARY_XSMALL = numpy.array([1, 2], dtype=int)
NUM_POINTS_BY_CLASS_TERNARY_XSMALL = numpy.array([1, 1, 1], dtype=int)

# The following constants are used to test normalize_predictor_matrix.
PERCENTILE_OFFSET_FOR_NORMALIZATION = 0.
PREDICTOR_NAMES = [radar_utils.REFL_NAME, radar_utils.DIFFERENTIAL_REFL_NAME]
PREDICTOR_NORMALIZATION_DICT = {
    radar_utils.DIFFERENTIAL_REFL_NAME: numpy.array([-8., 8.]),
    radar_utils.REFL_NAME: numpy.array([1., 10.])
}

REFL_MATRIX_EXAMPLE1_HEIGHT1 = numpy.array(
    [[0, 1, 2, 3],
     [4, 5, 6, 7]], dtype=numpy.float32)
REFL_MATRIX_EXAMPLE1_HEIGHT2 = numpy.array(
    [[2, 4, 6, 8],
     [10, 12, 14, 16]], dtype=numpy.float32)
REFL_MATRIX_EXAMPLE1_HEIGHT3 = numpy.array(
    [[-5, 2, 0, 5],
     [3, -3, 11, 10]], dtype=numpy.float32)

REFL_MATRIX_EXAMPLE2_HEIGHT1 = numpy.array(
    [[3, 2, 1, 2],
     [6, 10, 16, -6]], dtype=numpy.float32)
REFL_MATRIX_EXAMPLE2_HEIGHT2 = numpy.array(
    [[0, 0, 0, 0],
     [0, 1, 1, 2]], dtype=numpy.float32)
REFL_MATRIX_EXAMPLE2_HEIGHT3 = numpy.array(
    [[17, 7, 0, 3],
     [6, 7, 8, 4]], dtype=numpy.float32)

DIFF_REFL_MATRIX_EXAMPLE1_HEIGHT1 = -1 * REFL_MATRIX_EXAMPLE1_HEIGHT1
DIFF_REFL_MATRIX_EXAMPLE1_HEIGHT2 = -1 * REFL_MATRIX_EXAMPLE1_HEIGHT2
DIFF_REFL_MATRIX_EXAMPLE1_HEIGHT3 = -1 * REFL_MATRIX_EXAMPLE1_HEIGHT3
DIFF_REFL_MATRIX_EXAMPLE2_HEIGHT1 = -1 * REFL_MATRIX_EXAMPLE2_HEIGHT1
DIFF_REFL_MATRIX_EXAMPLE2_HEIGHT2 = -1 * REFL_MATRIX_EXAMPLE2_HEIGHT2
DIFF_REFL_MATRIX_EXAMPLE2_HEIGHT3 = -1 * REFL_MATRIX_EXAMPLE2_HEIGHT3

EXAMPLE1_HEIGHT1_MATRIX_UNNORMALIZED = numpy.stack(
    (REFL_MATRIX_EXAMPLE1_HEIGHT1, DIFF_REFL_MATRIX_EXAMPLE1_HEIGHT1), axis=-1)
EXAMPLE2_HEIGHT1_MATRIX_UNNORMALIZED = numpy.stack(
    (REFL_MATRIX_EXAMPLE2_HEIGHT1, DIFF_REFL_MATRIX_EXAMPLE2_HEIGHT1), axis=-1)
PREDICTOR_MATRIX_4D_UNNORMALIZED = numpy.stack(
    (EXAMPLE1_HEIGHT1_MATRIX_UNNORMALIZED,
     EXAMPLE2_HEIGHT1_MATRIX_UNNORMALIZED), axis=0)

EXAMPLE1_HEIGHT1_MATRIX_NORM_BY_BATCH = numpy.stack(
    ((REFL_MATRIX_EXAMPLE1_HEIGHT1 + 6) / 22,
     (DIFF_REFL_MATRIX_EXAMPLE1_HEIGHT1 + 16) / 22), axis=-1)
EXAMPLE2_HEIGHT1_MATRIX_NORM_BY_BATCH = numpy.stack(
    ((REFL_MATRIX_EXAMPLE2_HEIGHT1 + 6) / 22,
     (DIFF_REFL_MATRIX_EXAMPLE2_HEIGHT1 + 16) / 22), axis=-1)
PREDICTOR_MATRIX_4D_NORM_BY_BATCH = numpy.stack(
    (EXAMPLE1_HEIGHT1_MATRIX_NORM_BY_BATCH,
     EXAMPLE2_HEIGHT1_MATRIX_NORM_BY_BATCH), axis=0)

EXAMPLE1_HEIGHT1_MATRIX_NORM_BY_CLIMO = numpy.stack(
    ((REFL_MATRIX_EXAMPLE1_HEIGHT1 - 1) / 9,
     (DIFF_REFL_MATRIX_EXAMPLE1_HEIGHT1 + 8) / 16), axis=-1)
EXAMPLE2_HEIGHT1_MATRIX_NORM_BY_CLIMO = numpy.stack(
    ((REFL_MATRIX_EXAMPLE2_HEIGHT1 - 1) / 9,
     (DIFF_REFL_MATRIX_EXAMPLE2_HEIGHT1 + 8) / 16), axis=-1)
PREDICTOR_MATRIX_4D_NORM_BY_CLIMO = numpy.stack(
    (EXAMPLE1_HEIGHT1_MATRIX_NORM_BY_CLIMO,
     EXAMPLE2_HEIGHT1_MATRIX_NORM_BY_CLIMO), axis=0)

EXAMPLE1_HEIGHT2_MATRIX_UNNORMALIZED = numpy.stack(
    (REFL_MATRIX_EXAMPLE1_HEIGHT2, DIFF_REFL_MATRIX_EXAMPLE1_HEIGHT2), axis=-1)
EXAMPLE2_HEIGHT2_MATRIX_UNNORMALIZED = numpy.stack(
    (REFL_MATRIX_EXAMPLE2_HEIGHT2, DIFF_REFL_MATRIX_EXAMPLE2_HEIGHT2), axis=-1)
HEIGHT2_MATRIX_UNNORMALIZED = numpy.stack(
    (EXAMPLE1_HEIGHT2_MATRIX_UNNORMALIZED,
     EXAMPLE2_HEIGHT2_MATRIX_UNNORMALIZED), axis=0)

EXAMPLE1_HEIGHT3_MATRIX_UNNORMALIZED = numpy.stack(
    (REFL_MATRIX_EXAMPLE1_HEIGHT3, DIFF_REFL_MATRIX_EXAMPLE1_HEIGHT3), axis=-1)
EXAMPLE2_HEIGHT3_MATRIX_UNNORMALIZED = numpy.stack(
    (REFL_MATRIX_EXAMPLE2_HEIGHT3, DIFF_REFL_MATRIX_EXAMPLE2_HEIGHT3), axis=-1)
HEIGHT3_MATRIX_UNNORMALIZED = numpy.stack(
    (EXAMPLE1_HEIGHT3_MATRIX_UNNORMALIZED,
     EXAMPLE2_HEIGHT3_MATRIX_UNNORMALIZED), axis=0)

PREDICTOR_MATRIX_5D_UNNORMALIZED = numpy.stack(
    (PREDICTOR_MATRIX_4D_UNNORMALIZED, HEIGHT2_MATRIX_UNNORMALIZED,
     HEIGHT3_MATRIX_UNNORMALIZED), axis=-2)

EXAMPLE1_HEIGHT1_MATRIX_NORM_BY_BATCH = numpy.stack(
    ((REFL_MATRIX_EXAMPLE1_HEIGHT1 + 6) / 23,
     (DIFF_REFL_MATRIX_EXAMPLE1_HEIGHT1 + 17) / 23), axis=-1)
EXAMPLE2_HEIGHT1_MATRIX_NORM_BY_BATCH = numpy.stack(
    ((REFL_MATRIX_EXAMPLE2_HEIGHT1 + 6) / 23,
     (DIFF_REFL_MATRIX_EXAMPLE2_HEIGHT1 + 17) / 23), axis=-1)
HEIGHT1_MATRIX_NORM_BY_BATCH = numpy.stack(
    (EXAMPLE1_HEIGHT1_MATRIX_NORM_BY_BATCH,
     EXAMPLE2_HEIGHT1_MATRIX_NORM_BY_BATCH), axis=0)

EXAMPLE1_HEIGHT2_MATRIX_NORM_BY_BATCH = numpy.stack(
    ((REFL_MATRIX_EXAMPLE1_HEIGHT2 + 6) / 23,
     (DIFF_REFL_MATRIX_EXAMPLE1_HEIGHT2 + 17) / 23), axis=-1)
EXAMPLE2_HEIGHT2_MATRIX_NORM_BY_BATCH = numpy.stack(
    ((REFL_MATRIX_EXAMPLE2_HEIGHT2 + 6) / 23,
     (DIFF_REFL_MATRIX_EXAMPLE2_HEIGHT2 + 17) / 23), axis=-1)
HEIGHT2_MATRIX_NORM_BY_BATCH = numpy.stack(
    (EXAMPLE1_HEIGHT2_MATRIX_NORM_BY_BATCH,
     EXAMPLE2_HEIGHT2_MATRIX_NORM_BY_BATCH), axis=0)

EXAMPLE1_HEIGHT3_MATRIX_NORM_BY_BATCH = numpy.stack(
    ((REFL_MATRIX_EXAMPLE1_HEIGHT3 + 6) / 23,
     (DIFF_REFL_MATRIX_EXAMPLE1_HEIGHT3 + 17) / 23), axis=-1)
EXAMPLE2_HEIGHT3_MATRIX_NORM_BY_BATCH = numpy.stack(
    ((REFL_MATRIX_EXAMPLE2_HEIGHT3 + 6) / 23,
     (DIFF_REFL_MATRIX_EXAMPLE2_HEIGHT3 + 17) / 23), axis=-1)
HEIGHT3_MATRIX_NORM_BY_BATCH = numpy.stack(
    (EXAMPLE1_HEIGHT3_MATRIX_NORM_BY_BATCH,
     EXAMPLE2_HEIGHT3_MATRIX_NORM_BY_BATCH), axis=0)

PREDICTOR_MATRIX_5D_NORM_BY_BATCH = numpy.stack(
    (HEIGHT1_MATRIX_NORM_BY_BATCH, HEIGHT2_MATRIX_NORM_BY_BATCH,
     HEIGHT3_MATRIX_NORM_BY_BATCH), axis=-2)

EXAMPLE1_HEIGHT2_MATRIX_NORM_BY_CLIMO = numpy.stack(
    ((REFL_MATRIX_EXAMPLE1_HEIGHT2 - 1) / 9,
     (DIFF_REFL_MATRIX_EXAMPLE1_HEIGHT2 + 8) / 16), axis=-1)
EXAMPLE2_HEIGHT2_MATRIX_NORM_BY_CLIMO = numpy.stack(
    ((REFL_MATRIX_EXAMPLE2_HEIGHT2 - 1) / 9,
     (DIFF_REFL_MATRIX_EXAMPLE2_HEIGHT2 + 8) / 16), axis=-1)
HEIGHT2_MATRIX_NORM_BY_CLIMO = numpy.stack(
    (EXAMPLE1_HEIGHT2_MATRIX_NORM_BY_CLIMO,
     EXAMPLE2_HEIGHT2_MATRIX_NORM_BY_CLIMO), axis=0)

EXAMPLE1_HEIGHT3_MATRIX_NORM_BY_CLIMO = numpy.stack(
    ((REFL_MATRIX_EXAMPLE1_HEIGHT3 - 1) / 9,
     (DIFF_REFL_MATRIX_EXAMPLE1_HEIGHT3 + 8) / 16), axis=-1)
EXAMPLE2_HEIGHT3_MATRIX_NORM_BY_CLIMO = numpy.stack(
    ((REFL_MATRIX_EXAMPLE2_HEIGHT3 - 1) / 9,
     (DIFF_REFL_MATRIX_EXAMPLE2_HEIGHT3 + 8) / 16), axis=-1)
HEIGHT3_MATRIX_NORM_BY_CLIMO = numpy.stack(
    (EXAMPLE1_HEIGHT3_MATRIX_NORM_BY_CLIMO,
     EXAMPLE2_HEIGHT3_MATRIX_NORM_BY_CLIMO), axis=0)

PREDICTOR_MATRIX_5D_NORM_BY_CLIMO = numpy.stack(
    (PREDICTOR_MATRIX_4D_NORM_BY_CLIMO, HEIGHT2_MATRIX_NORM_BY_CLIMO,
     HEIGHT3_MATRIX_NORM_BY_CLIMO), axis=-2)

# The following constants are used to test sample_points_by_class.
TARGET_VALUES_TO_SAMPLE_BINARY = numpy.array(
    [0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0,
     1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    dtype=int)

NUM_POINTS_TO_SAMPLE = 30
CLASS_FRACTIONS_BINARY = numpy.array([0.5, 0.5])
INDICES_TO_KEEP_BINARY = numpy.array(
    [0, 1, 2, 4, 7, 8, 9, 10, 11, 13, 3, 5, 6, 12, 15, 16, 19, 25, 45, 48],
    dtype=int)

TARGET_VALUES_TO_SAMPLE_TERNARY = numpy.array(
    [2, 2, 2, 2, 2, 0, 2, 1, 2, 0, 0, 2, 1, 0, 2, 2, 2, 0, 0, 1, 2, 0, 1, 2, 0,
     2, 0, 1, 1, 1, 0, 2, 1, 2, 1, 0, 0, 1, 0, 0, 1, 0, 2, 2, 2, 0, 1, 1, 2, 2],
    dtype=int)

CLASS_FRACTIONS_TERNARY = numpy.array([0.1, 0.6, 0.3])
INDICES_TO_KEEP_TERNARY = numpy.array(
    [5, 9, 7, 12, 19, 22, 27, 28, 29, 32, 34, 37, 40, 46, 47, 0, 1, 2, 3, 4, 6],
    dtype=int)


class DeepLearningUtilsTests(unittest.TestCase):
    """Each method is a unit test for deep_learning_utils.py."""

    def test_check_predictor_matrix_1d(self):
        """Ensures correct output from check_predictor_matrix.

        In this case, input matrix is 1-D (bad).
        """

        with self.assertRaises(ValueError):
            dl_utils.check_predictor_matrix(PREDICTOR_MATRIX_1D)

    def test_check_predictor_matrix_2d(self):
        """Ensures correct output from check_predictor_matrix.

        In this case, input matrix is 2-D (bad).
        """

        with self.assertRaises(ValueError):
            dl_utils.check_predictor_matrix(PREDICTOR_MATRIX_2D)

    def test_check_predictor_matrix_3d(self):
        """Ensures correct output from check_predictor_matrix.

        In this case, input matrix is 3-D (good).
        """

        dl_utils.check_predictor_matrix(PREDICTOR_MATRIX_3D)

    def test_check_predictor_matrix_4d(self):
        """Ensures correct output from check_predictor_matrix.

        In this case, input matrix is 4-D (good).
        """

        dl_utils.check_predictor_matrix(PREDICTOR_MATRIX_4D)

    def test_check_target_values_1d_2classes_good(self):
        """Ensures correct output from check_target_values.

        In this case, input array is 1-D and two-class, as expected.
        """

        dl_utils.check_target_values(
            TOY_TARGET_VALUES_1D_BINARY, num_dimensions=1, num_classes=2)

    def test_check_target_values_1d_bad_dim(self):
        """Ensures correct output from check_target_values.

        In this case, 2-D input array is expected and 1-D array is passed.
        """

        with self.assertRaises(TypeError):
            dl_utils.check_target_values(
                TOY_TARGET_VALUES_1D_BINARY, num_dimensions=2, num_classes=2)

    def test_check_target_values_1d_2classes_bad_class_num(self):
        """Ensures correct output from check_target_values.

        In this case, multiclass input array is expected and two-class input
        array is passed.  However, there is no way to ascertain that the two-
        class array is wrong (maybe higher classes just never occur).
        """

        dl_utils.check_target_values(
            TOY_TARGET_VALUES_1D_BINARY, num_dimensions=1,
            num_classes=TOY_TARGET_MATRIX_MULTICLASS.shape[1])

    def test_check_target_values_1d_multiclass_good(self):
        """Ensures correct output from check_target_values.

        In this case, input array is 1-D and multiclass, as expected.
        """

        dl_utils.check_target_values(
            TOY_TARGET_VALUES_1D_MULTICLASS, num_dimensions=1,
            num_classes=TOY_TARGET_MATRIX_MULTICLASS.shape[1])

    def test_check_target_values_1d_multiclass_bad_class_num(self):
        """Ensures correct output from check_target_values.

        In this case, two-class input array is expected and multiclass array is
        passed.
        """

        with self.assertRaises(ValueError):
            dl_utils.check_target_values(
                TOY_TARGET_VALUES_1D_MULTICLASS, num_dimensions=1, num_classes=2)

    def test_check_target_values_2d_2classes_good(self):
        """Ensures correct output from check_target_values.

        In this case, input array is 2-D and two-class, as expected.
        """

        dl_utils.check_target_values(
            TOY_TARGET_MATRIX_BINARY, num_dimensions=2, num_classes=2)

    def test_check_target_values_2d_bad_dim(self):
        """Ensures correct output from check_target_values.

        In this case, 1-D input array is expected and 2-D array is passed.
        """

        with self.assertRaises(TypeError):
            dl_utils.check_target_values(
                TOY_TARGET_MATRIX_BINARY, num_dimensions=1, num_classes=2)

    def test_check_target_values_2d_2classes_bad_class_num(self):
        """Ensures correct output from check_target_values.

        In this case, multiclass input matrix is expected and two-class matrix
        is passed.
        """

        with self.assertRaises(TypeError):
            dl_utils.check_target_values(
                TOY_TARGET_MATRIX_BINARY, num_dimensions=2,
                num_classes=TOY_TARGET_MATRIX_MULTICLASS.shape[1])

    def test_check_target_values_2d_multiclass_good(self):
        """Ensures correct output from check_target_values.

        In this case, input array is 2-D and multiclass, as expected.
        """

        dl_utils.check_target_values(
            TOY_TARGET_MATRIX_MULTICLASS, num_dimensions=2,
            num_classes=TOY_TARGET_MATRIX_MULTICLASS.shape[1])

    def test_check_target_values_2d_multiclass_bad_class_num(self):
        """Ensures correct output from check_target_values.

        In this case, two-class input matrix is expected and multiclass matrix
        is passed.
        """

        with self.assertRaises(TypeError):
            dl_utils.check_target_values(
                TOY_TARGET_MATRIX_MULTICLASS, num_dimensions=2, num_classes=2)

    def test_class_fractions_to_num_points_binary_large(self):
        """Ensures correct output from _class_fractions_to_num_points.

        In this case, there are 2 classes and number of points is large.
        """

        this_num_points_by_class = dl_utils._class_fractions_to_num_points(
            class_fractions=TOY_CLASS_FRACTIONS_BINARY,
            num_points_to_sample=NUM_POINTS_TO_SAMPLE_LARGE)

        self.assertTrue(numpy.array_equal(
            this_num_points_by_class, NUM_POINTS_BY_CLASS_BINARY_LARGE))

    def test_class_fractions_to_num_points_ternary_large(self):
        """Ensures correct output from _class_fractions_to_num_points.

        In this case, there are 3 classes and number of points is large.
        """

        this_num_points_by_class = dl_utils._class_fractions_to_num_points(
            class_fractions=TOY_CLASS_FRACTIONS_TERNARY,
            num_points_to_sample=NUM_POINTS_TO_SAMPLE_LARGE)

        self.assertTrue(numpy.array_equal(
            this_num_points_by_class, NUM_POINTS_BY_CLASS_TERNARY_LARGE))

    def test_class_fractions_to_num_points_binary_medium(self):
        """Ensures correct output from _class_fractions_to_num_points.

        In this case, there are 2 classes and number of points is medium.
        """

        this_num_points_by_class = dl_utils._class_fractions_to_num_points(
            class_fractions=TOY_CLASS_FRACTIONS_BINARY,
            num_points_to_sample=NUM_POINTS_TO_SAMPLE_MEDIUM)

        self.assertTrue(numpy.array_equal(
            this_num_points_by_class, NUM_POINTS_BY_CLASS_BINARY_MEDIUM))

    def test_class_fractions_to_num_points_ternary_medium(self):
        """Ensures correct output from _class_fractions_to_num_points.

        In this case, there are 3 classes and number of points is medium.
        """

        this_num_points_by_class = dl_utils._class_fractions_to_num_points(
            class_fractions=TOY_CLASS_FRACTIONS_TERNARY,
            num_points_to_sample=NUM_POINTS_TO_SAMPLE_MEDIUM)

        self.assertTrue(numpy.array_equal(
            this_num_points_by_class, NUM_POINTS_BY_CLASS_TERNARY_MEDIUM))

    def test_class_fractions_to_num_points_binary_small(self):
        """Ensures correct output from _class_fractions_to_num_points.

        In this case, there are 2 classes and number of points is small.
        """

        this_num_points_by_class = dl_utils._class_fractions_to_num_points(
            class_fractions=TOY_CLASS_FRACTIONS_BINARY,
            num_points_to_sample=NUM_POINTS_TO_SAMPLE_SMALL)

        self.assertTrue(numpy.array_equal(
            this_num_points_by_class, NUM_POINTS_BY_CLASS_BINARY_SMALL))

    def test_class_fractions_to_num_points_ternary_small(self):
        """Ensures correct output from _class_fractions_to_num_points.

        In this case, there are 3 classes and number of points is small.
        """

        this_num_points_by_class = dl_utils._class_fractions_to_num_points(
            class_fractions=TOY_CLASS_FRACTIONS_TERNARY,
            num_points_to_sample=NUM_POINTS_TO_SAMPLE_SMALL)

        self.assertTrue(numpy.array_equal(
            this_num_points_by_class, NUM_POINTS_BY_CLASS_TERNARY_SMALL))

    def test_class_fractions_to_num_points_binary_xsmall(self):
        """Ensures correct output from _class_fractions_to_num_points.

        In this case, there are 2 classes and number of points is extra small.
        """

        this_num_points_by_class = dl_utils._class_fractions_to_num_points(
            class_fractions=TOY_CLASS_FRACTIONS_BINARY,
            num_points_to_sample=NUM_POINTS_TO_SAMPLE_XSMALL)

        self.assertTrue(numpy.array_equal(
            this_num_points_by_class, NUM_POINTS_BY_CLASS_BINARY_XSMALL))

    def test_class_fractions_to_num_points_ternary_xsmall(self):
        """Ensures correct output from _class_fractions_to_num_points.

        In this case, there are 3 classes and number of points is extra small.
        """

        this_num_points_by_class = dl_utils._class_fractions_to_num_points(
            class_fractions=TOY_CLASS_FRACTIONS_TERNARY,
            num_points_to_sample=NUM_POINTS_TO_SAMPLE_XSMALL)

        self.assertTrue(numpy.array_equal(
            this_num_points_by_class, NUM_POINTS_BY_CLASS_TERNARY_XSMALL))

    def test_stack_predictor_variables(self):
        """Ensures correct output from stack_predictor_variables."""

        this_matrix = dl_utils.stack_predictor_variables(
            TUPLE_OF_3D_PREDICTOR_MATRICES)
        self.assertTrue(numpy.allclose(
            this_matrix, PREDICTOR_MATRIX_4D, atol=TOLERANCE, equal_nan=True))

    def test_normalize_predictor_matrix_4d_by_batch(self):
        """Ensures correct output from normalize_predictor_matrix.

        In this case, predictor matrix is 4-D and normalize_by_batch = True.
        """

        this_predictor_matrix = copy.deepcopy(PREDICTOR_MATRIX_4D_UNNORMALIZED)
        this_predictor_matrix = dl_utils.normalize_predictor_matrix(
            predictor_matrix=this_predictor_matrix, normalize_by_batch=True,
            percentile_offset=PERCENTILE_OFFSET_FOR_NORMALIZATION)

        self.assertTrue(numpy.allclose(
            this_predictor_matrix, PREDICTOR_MATRIX_4D_NORM_BY_BATCH,
            atol=TOLERANCE, equal_nan=True))

    def test_normalize_predictor_matrix_4d_by_climo(self):
        """Ensures correct output from normalize_predictor_matrix.

        In this case, predictor matrix is 4-D and normalize_by_batch = False.
        """

        this_predictor_matrix = copy.deepcopy(PREDICTOR_MATRIX_4D_UNNORMALIZED)
        this_predictor_matrix = dl_utils.normalize_predictor_matrix(
            predictor_matrix=this_predictor_matrix, normalize_by_batch=False,
            predictor_names=PREDICTOR_NAMES,
            normalization_dict=PREDICTOR_NORMALIZATION_DICT)

        self.assertTrue(numpy.allclose(
            this_predictor_matrix, PREDICTOR_MATRIX_4D_NORM_BY_CLIMO,
            atol=TOLERANCE, equal_nan=True))

    def test_normalize_predictor_matrix_5d_by_batch(self):
        """Ensures correct output from normalize_predictor_matrix.

        In this case, predictor matrix is 5-D and normalize_by_batch = True.
        """

        this_predictor_matrix = copy.deepcopy(PREDICTOR_MATRIX_5D_UNNORMALIZED)
        this_predictor_matrix = dl_utils.normalize_predictor_matrix(
            predictor_matrix=this_predictor_matrix, normalize_by_batch=True,
            percentile_offset=PERCENTILE_OFFSET_FOR_NORMALIZATION)

        self.assertTrue(numpy.allclose(
            this_predictor_matrix, PREDICTOR_MATRIX_5D_NORM_BY_BATCH,
            atol=TOLERANCE, equal_nan=True))

    def test_normalize_predictor_matrix_5d_by_climo(self):
        """Ensures correct output from normalize_predictor_matrix.

        In this case, predictor matrix is 5-D and normalize_by_batch = False.
        """

        this_predictor_matrix = copy.deepcopy(PREDICTOR_MATRIX_5D_UNNORMALIZED)
        this_predictor_matrix = dl_utils.normalize_predictor_matrix(
            predictor_matrix=this_predictor_matrix, normalize_by_batch=False,
            predictor_names=PREDICTOR_NAMES,
            normalization_dict=PREDICTOR_NORMALIZATION_DICT)

        self.assertTrue(numpy.allclose(
            this_predictor_matrix, PREDICTOR_MATRIX_5D_NORM_BY_CLIMO,
            atol=TOLERANCE, equal_nan=True))

    def test_sample_points_by_class_binary(self):
        """Ensures correct output from sample_points_by_class.

        In this case there are 2 classes.
        """

        these_indices = dl_utils.sample_points_by_class(
            target_values=TARGET_VALUES_TO_SAMPLE_BINARY,
            class_fractions=CLASS_FRACTIONS_BINARY,
            num_points_to_sample=NUM_POINTS_TO_SAMPLE, test_mode=True)

        self.assertTrue(numpy.array_equal(
            these_indices, INDICES_TO_KEEP_BINARY))

    def test_sample_points_by_class_ternary(self):
        """Ensures correct output from sample_points_by_class.

        In this case there are 3 classes.
        """

        these_indices = dl_utils.sample_points_by_class(
            target_values=TARGET_VALUES_TO_SAMPLE_TERNARY,
            class_fractions=CLASS_FRACTIONS_TERNARY,
            num_points_to_sample=NUM_POINTS_TO_SAMPLE, test_mode=True)

        self.assertTrue(numpy.array_equal(
            these_indices, INDICES_TO_KEEP_TERNARY))


if __name__ == '__main__':
    unittest.main()
