"""Unit tests for permutation.py."""

import copy
import unittest
import numpy
from sklearn.metrics import log_loss as sklearn_cross_entropy
from gewittergefahr.deep_learning import permutation

TOLERANCE = 1e-6
XENTROPY_TOLERANCE = 1e-4

# The following constants are used to test _flatten_last_two_dim.
THIS_HEIGHT1_MATRIX = numpy.array([
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0]
], dtype=float)

THIS_HEIGHT2_MATRIX = THIS_HEIGHT1_MATRIX * 10
THIS_HEIGHT3_MATRIX = THIS_HEIGHT1_MATRIX * 5
THIS_FIRST_EXAMPLE_MATRIX = numpy.stack(
    (THIS_HEIGHT1_MATRIX, THIS_HEIGHT2_MATRIX, THIS_HEIGHT3_MATRIX), axis=-1
)

THIS_SECOND_EXAMPLE_MATRIX = THIS_FIRST_EXAMPLE_MATRIX + 10
THIS_THIRD_EXAMPLE_MATRIX = THIS_FIRST_EXAMPLE_MATRIX + 20
THIS_FOURTH_EXAMPLE_MATRIX = THIS_FIRST_EXAMPLE_MATRIX + 30

REFLECTIVITY_MATRIX_DBZ = numpy.stack((
    THIS_FIRST_EXAMPLE_MATRIX, THIS_SECOND_EXAMPLE_MATRIX,
    THIS_THIRD_EXAMPLE_MATRIX, THIS_FOURTH_EXAMPLE_MATRIX
), axis=0)

VORTICITY_MATRIX_S01 = 0.001 * REFLECTIVITY_MATRIX_DBZ

RADAR_MATRIX_5D = numpy.stack(
    (REFLECTIVITY_MATRIX_DBZ, VORTICITY_MATRIX_S01), axis=-1
)

RADAR_MATRIX_4D = numpy.concatenate(
    (REFLECTIVITY_MATRIX_DBZ, VORTICITY_MATRIX_S01), axis=-1
)

# The following constants are used to test _permute_one_predictor.
THIS_TEMPERATURE_MATRIX_CELSIUS = numpy.array([
    [5, 4, 3, 2, 1, 0, -1],
    [12, 11, 10, 9, 8, 7, 6],
    [19, 18, 17, 16, 15, 14, 13],
    [26, 25, 24, 23, 22, 21, 20]
], dtype=float)

THIS_TEMPERATURE_MATRIX_KELVINS = 273.15 + THIS_TEMPERATURE_MATRIX_CELSIUS
THIS_RH_MATRIX_UNITLESS = 0.01 * (THIS_TEMPERATURE_MATRIX_CELSIUS + 1.)

SOUNDING_MATRIX = numpy.stack(
    (THIS_TEMPERATURE_MATRIX_KELVINS, THIS_RH_MATRIX_UNITLESS), axis=-1
)

PREDICTOR_MATRICES = [RADAR_MATRIX_5D, SOUNDING_MATRIX]

FIRST_SEPARATE_HEIGHTS_FLAG = True
FIRST_MATRIX_INDEX = 0
FIRST_PREDICTOR_INDEX = 1

SECOND_SEPARATE_HEIGHTS_FLAG = True
SECOND_MATRIX_INDEX = 1
SECOND_PREDICTOR_INDEX = 0

THIRD_SEPARATE_HEIGHTS_FLAG = False
THIRD_MATRIX_INDEX = 0
THIRD_PREDICTOR_INDEX = 1

# The following constants are used to test cross_entropy_function,
# negative_auc_function, and _bootstrap_cost.
BINARY_TARGET_VALUES = numpy.array([0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0], dtype=int)
BINARY_PROBABILITY_MATRIX = numpy.array([
    [0, 1],
    [1, 0],
    [0, 1],
    [1, 0],
    [0.5, 0.5],
    [0.5, 0.5],
    [0, 1],
    [1, 0],
    [0, 1],
    [1, 0],
    [0.5, 0.5]
])

NEGATIVE_AUC = -0.75
BINARY_CROSS_ENTROPY = sklearn_cross_entropy(
    BINARY_TARGET_VALUES, BINARY_PROBABILITY_MATRIX[:, 1]
)

TERNARY_TARGET_VALUES = numpy.array(
    [2, 0, 1, 1, 2, 1, 2, 0, 2, 0, 0], dtype=int
)

TERNARY_PROBABILITY_MATRIX = numpy.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [1. / 3, 1. / 3, 1. / 3],
    [0.5, 0.25, 0.25],
    [0.25, 0.5, 0.25],
    [1. / 3, 0, 2. / 3],
    [1, 0, 0],
    [0, 0.5, 0.5],
    [0.75, 0, 0.25],
    [0.6, 0.2, 0.2]
])

TERNARY_CROSS_ENTROPY = sklearn_cross_entropy(
    TERNARY_TARGET_VALUES, TERNARY_PROBABILITY_MATRIX)


def _compare_before_and_after_permutn(
        orig_predictor_matrices, new_predictor_matrices, separate_radar_heights,
        matrix_index, predictor_index):
    """Compares predictor matrices before and after permutation.

    T = number of predictor matrices

    :param orig_predictor_matrices: length-T list of numpy arrays.
    :param new_predictor_matrices: length-T list of numpy arrays with matching
        dimensions.
    :param separate_radar_heights: See doc for
        `permutation._permute_one_predictor`.
    :param matrix_index: Same.
    :param predictor_index: Same.
    :return: all_good: Boolean flag, indicating whether or not results are as
        expected.
    """

    num_matrices = len(orig_predictor_matrices)
    all_good = True

    for i in range(num_matrices):
        if i == 0 and separate_radar_heights:
            this_orig_matrix = permutation._flatten_last_two_dim(
                orig_predictor_matrices[i]
            )[0]
            this_new_matrix = permutation._flatten_last_two_dim(
                new_predictor_matrices[i]
            )[0]
        else:
            this_orig_matrix = orig_predictor_matrices[i]
            this_new_matrix = new_predictor_matrices[i]

        this_num_predictors = this_orig_matrix.shape[-1]

        for j in range(this_num_predictors):
            if i == matrix_index and j == predictor_index:
                all_good = all_good and not numpy.allclose(
                    this_orig_matrix[..., j], this_new_matrix[..., j],
                    atol=TOLERANCE
                )
            else:
                all_good = all_good and numpy.allclose(
                    this_orig_matrix[..., j], this_new_matrix[..., j],
                    atol=TOLERANCE
                )

            if not all_good:
                return False

    return all_good


class PermutationTests(unittest.TestCase):
    """Each method is a unit test for permutation.py."""

    def test_flatten_last_two_dim(self):
        """Ensures correct output from _flatten_last_two_dim."""

        this_radar_matrix_4d, this_orig_shape = (
            permutation._flatten_last_two_dim(RADAR_MATRIX_5D)
        )

        self.assertTrue(numpy.allclose(
            this_radar_matrix_4d, RADAR_MATRIX_4D, atol=TOLERANCE
        ))
        self.assertTrue(numpy.array_equal(
            this_orig_shape, numpy.array(RADAR_MATRIX_5D.shape, dtype=int)
        ))

    def test_bootstrap_cost_1replicate(self):
        """Ensures correct output from _bootstrap_cost.

        In this case there is only one bootstrap replicate.
        """

        this_cost_array = permutation._bootstrap_cost(
            target_values=BINARY_TARGET_VALUES,
            class_probability_matrix=BINARY_PROBABILITY_MATRIX,
            cost_function=permutation.negative_auc_function, num_replicates=1)

        self.assertTrue(numpy.allclose(
            this_cost_array, numpy.array([NEGATIVE_AUC]), atol=TOLERANCE
        ))

    def test_bootstrap_cost_1000replicates(self):
        """Ensures correct output from _bootstrap_cost.

        In this case there are 1000 replicates.
        """

        this_cost_array = permutation._bootstrap_cost(
            target_values=BINARY_TARGET_VALUES,
            class_probability_matrix=BINARY_PROBABILITY_MATRIX,
            cost_function=permutation.cross_entropy_function,
            num_replicates=1000)

        self.assertTrue(len(this_cost_array) == 1000)

    def test_permute_one_predictor_first(self):
        """Ensures correct output from _permute_one_predictor.

        In this case, using first set of inputs.
        """

        these_new_matrices, these_permuted_values = (
            permutation._permute_one_predictor(
                predictor_matrices=copy.deepcopy(PREDICTOR_MATRICES),
                separate_radar_heights=FIRST_SEPARATE_HEIGHTS_FLAG,
                matrix_index=FIRST_MATRIX_INDEX,
                predictor_index=FIRST_PREDICTOR_INDEX, permuted_values=None)
        )

        self.assertTrue(_compare_before_and_after_permutn(
            orig_predictor_matrices=PREDICTOR_MATRICES,
            new_predictor_matrices=these_new_matrices,
            separate_radar_heights=FIRST_SEPARATE_HEIGHTS_FLAG,
            matrix_index=FIRST_MATRIX_INDEX,
            predictor_index=FIRST_PREDICTOR_INDEX
        ))

        these_newnew_matrices = permutation._permute_one_predictor(
            predictor_matrices=copy.deepcopy(these_new_matrices),
            separate_radar_heights=FIRST_SEPARATE_HEIGHTS_FLAG,
            matrix_index=FIRST_MATRIX_INDEX,
            predictor_index=FIRST_PREDICTOR_INDEX,
            permuted_values=these_permuted_values
        )[0]

        for i in range(len(these_new_matrices)):
            self.assertTrue(numpy.allclose(
                these_newnew_matrices[i], these_new_matrices[i],
                atol=TOLERANCE
            ))

    def test_permute_one_predictor_second(self):
        """Ensures correct output from _permute_one_predictor.

        In this case, using second set of inputs.
        """

        these_new_matrices, these_permuted_values = (
            permutation._permute_one_predictor(
                predictor_matrices=copy.deepcopy(PREDICTOR_MATRICES),
                separate_radar_heights=SECOND_SEPARATE_HEIGHTS_FLAG,
                matrix_index=SECOND_MATRIX_INDEX,
                predictor_index=SECOND_PREDICTOR_INDEX, permuted_values=None)
        )

        self.assertTrue(_compare_before_and_after_permutn(
            orig_predictor_matrices=PREDICTOR_MATRICES,
            new_predictor_matrices=these_new_matrices,
            separate_radar_heights=SECOND_SEPARATE_HEIGHTS_FLAG,
            matrix_index=SECOND_MATRIX_INDEX,
            predictor_index=SECOND_PREDICTOR_INDEX
        ))

        these_newnew_matrices = permutation._permute_one_predictor(
            predictor_matrices=copy.deepcopy(these_new_matrices),
            separate_radar_heights=SECOND_SEPARATE_HEIGHTS_FLAG,
            matrix_index=SECOND_MATRIX_INDEX,
            predictor_index=SECOND_PREDICTOR_INDEX,
            permuted_values=these_permuted_values
        )[0]

        for i in range(len(these_new_matrices)):
            self.assertTrue(numpy.allclose(
                these_newnew_matrices[i], these_new_matrices[i],
                atol=TOLERANCE
            ))

    def test_permute_one_predictor_third(self):
        """Ensures correct output from _permute_one_predictor.

        In this case, using third set of inputs.
        """

        these_new_matrices, these_permuted_values = (
            permutation._permute_one_predictor(
                predictor_matrices=copy.deepcopy(PREDICTOR_MATRICES),
                separate_radar_heights=THIRD_SEPARATE_HEIGHTS_FLAG,
                matrix_index=THIRD_MATRIX_INDEX,
                predictor_index=THIRD_PREDICTOR_INDEX, permuted_values=None)
        )

        self.assertTrue(_compare_before_and_after_permutn(
            orig_predictor_matrices=PREDICTOR_MATRICES,
            new_predictor_matrices=these_new_matrices,
            separate_radar_heights=THIRD_SEPARATE_HEIGHTS_FLAG,
            matrix_index=THIRD_MATRIX_INDEX,
            predictor_index=THIRD_PREDICTOR_INDEX
        ))

        these_newnew_matrices = permutation._permute_one_predictor(
            predictor_matrices=copy.deepcopy(these_new_matrices),
            separate_radar_heights=THIRD_SEPARATE_HEIGHTS_FLAG,
            matrix_index=THIRD_MATRIX_INDEX,
            predictor_index=THIRD_PREDICTOR_INDEX,
            permuted_values=these_permuted_values
        )[0]

        for i in range(len(these_new_matrices)):
            self.assertTrue(numpy.allclose(
                these_newnew_matrices[i], these_new_matrices[i],
                atol=TOLERANCE
            ))

    def test_cross_entropy_function_binary(self):
        """Ensures correct output from cross_entropy_function.

        In this case there are 2 target classes (binary).
        """

        this_cross_entropy = permutation.cross_entropy_function(
            target_values=BINARY_TARGET_VALUES,
            class_probability_matrix=BINARY_PROBABILITY_MATRIX, test_mode=True)

        self.assertTrue(numpy.isclose(
            this_cross_entropy, BINARY_CROSS_ENTROPY, atol=XENTROPY_TOLERANCE
        ))

    def test_cross_entropy_function_ternary(self):
        """Ensures correct output from cross_entropy_function.

        In this case there are 3 target classes (ternary).
        """

        this_cross_entropy = permutation.cross_entropy_function(
            target_values=TERNARY_TARGET_VALUES,
            class_probability_matrix=TERNARY_PROBABILITY_MATRIX, test_mode=True)

        self.assertTrue(numpy.isclose(
            this_cross_entropy, TERNARY_CROSS_ENTROPY, atol=XENTROPY_TOLERANCE
        ))

    def test_negative_auc_function_binary(self):
        """Ensures correct output from negative_auc_function.

        In this case there are 2 target classes (binary).
        """

        this_negative_auc = permutation.negative_auc_function(
            target_values=BINARY_TARGET_VALUES,
            class_probability_matrix=BINARY_PROBABILITY_MATRIX)

        self.assertTrue(numpy.isclose(
            this_negative_auc, NEGATIVE_AUC, atol=TOLERANCE
        ))

    def test_negative_auc_function_ternary(self):
        """Ensures correct output from negative_auc_function.

        In this case there are 3 target classes (ternary).
        """

        with self.assertRaises(TypeError):
            permutation.negative_auc_function(
                target_values=TERNARY_TARGET_VALUES,
                class_probability_matrix=TERNARY_PROBABILITY_MATRIX)


if __name__ == '__main__':
    unittest.main()
