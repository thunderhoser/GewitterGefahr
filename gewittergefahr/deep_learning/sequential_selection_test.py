"""Unit tests for sequential_selection.py."""

import unittest
import numpy
from gewittergefahr.deep_learning import sequential_selection

TOLERANCE = 1e-6

# The following constants are used to test _subset_input_matrices.
THIS_MATRIX1 = numpy.random.uniform(low=0., high=1., size=(100, 32, 32, 12, 4))
THIS_MATRIX2 = numpy.random.uniform(low=0., high=1., size=(100, 49, 5))
LIST_OF_TRAINING_MATRICES = [THIS_MATRIX1 + 0., THIS_MATRIX2 + 0.]

THIS_MATRIX1 = numpy.random.uniform(low=0., high=1., size=(100, 32, 32, 12, 4))
THIS_MATRIX2 = numpy.random.uniform(low=0., high=1., size=(100, 49, 5))
LIST_OF_VALIDATION_MATRICES = [THIS_MATRIX1, THIS_MATRIX2]

PREDICTOR_NAMES_BY_MATRIX = [['a', 'b', 'c', 'd'], ['1', '2', '3', '4', '5']]

FIRST_DESIRED_PREDICTOR_NAMES = ['4', 'b', 'a', '5', '2']
SECOND_DESIRED_PREDICTOR_NAMES = ['4', '5', '2']
THIRD_DESIRED_PREDICTOR_NAMES = ['b', 'a']

EMPTY_INDICES = numpy.array([], dtype=int)
THESE_INDICES1 = numpy.array([1, 0], dtype=int)
THESE_INDICES2 = numpy.array([3, 4, 1], dtype=int)

FIRST_SUBSET_TRAINING_MATRICES = [
    LIST_OF_TRAINING_MATRICES[0][..., THESE_INDICES1],
    LIST_OF_TRAINING_MATRICES[1][..., THESE_INDICES2]
]
FIRST_SUBSET_VALIDATION_MATRICES = [
    LIST_OF_VALIDATION_MATRICES[0][..., THESE_INDICES1],
    LIST_OF_VALIDATION_MATRICES[1][..., THESE_INDICES2]
]

SECOND_SUBSET_TRAINING_MATRICES = [
    LIST_OF_TRAINING_MATRICES[0][..., EMPTY_INDICES],
    LIST_OF_TRAINING_MATRICES[1][..., THESE_INDICES2]
]
SECOND_SUBSET_VALIDATION_MATRICES = [
    LIST_OF_VALIDATION_MATRICES[0][..., EMPTY_INDICES],
    LIST_OF_VALIDATION_MATRICES[1][..., THESE_INDICES2]
]

THIRD_SUBSET_TRAINING_MATRICES = [
    LIST_OF_TRAINING_MATRICES[0][..., THESE_INDICES1],
    LIST_OF_TRAINING_MATRICES[1][..., EMPTY_INDICES]
]
THIRD_SUBSET_VALIDATION_MATRICES = [
    LIST_OF_VALIDATION_MATRICES[0][..., THESE_INDICES1],
    LIST_OF_VALIDATION_MATRICES[1][..., EMPTY_INDICES]
]

# The following constants are used to test _eval_sfs_stopping_criterion.
MIN_LOSS_DECREASE = 0.01
MIN_PERCENTAGE_LOSS_DECREASE = 1.
NUM_STEPS_FOR_LOSS_DECREASE = 5

LOWEST_COST_BY_STEP_FIRST = numpy.array(
    [3.9, 3.8, 3.78, 3.77, 3.768, 3.766, 3.765])
LOWEST_COST_BY_STEP_SECOND = numpy.array(
    [0.9, 0.75, 0.7, 0.69, 0.689, 0.687, 0.686, 0.685, 0.683])


def _compare_lists_of_matrices(first_list_of_matrices, second_list_of_matrices):
    """Compares two lists of matrices (numpy arrays).

    :param first_list_of_matrices: First list of matrices.
    :param second_list_of_matrices: Second list of matrices.
    :return: are_lists_equal: Boolean flag.
    """

    if len(first_list_of_matrices) != len(second_list_of_matrices):
        return False

    for i in range(len(first_list_of_matrices)):
        if first_list_of_matrices[i].size == 0:
            if second_list_of_matrices[i].size != 0:
                return False

            continue

        if not numpy.allclose(
                first_list_of_matrices[i], second_list_of_matrices[i],
                atol=TOLERANCE):
            return False

    return True


class SequentialSelectionTests(unittest.TestCase):
    """Each method is a unit test for sequential_selection.py."""

    def test_subset_input_matrices_first(self):
        """Ensures correct output from _subset_input_matrices.

        In this case, desired predictors are those in
        `FIRST_DESIRED_PREDICTOR_NAMES`.
        """

        these_training_matrices, these_validation_matrices = (
            sequential_selection._subset_input_matrices(
                list_of_training_matrices=LIST_OF_TRAINING_MATRICES,
                list_of_validation_matrices=LIST_OF_VALIDATION_MATRICES,
                predictor_names_by_matrix=PREDICTOR_NAMES_BY_MATRIX,
                desired_predictor_names=FIRST_DESIRED_PREDICTOR_NAMES)
        )

        self.assertTrue(_compare_lists_of_matrices(
            these_training_matrices, FIRST_SUBSET_TRAINING_MATRICES))
        self.assertTrue(_compare_lists_of_matrices(
            these_validation_matrices, FIRST_SUBSET_VALIDATION_MATRICES))

    def test_subset_input_matrices_second(self):
        """Ensures correct output from _subset_input_matrices.

        In this case, desired predictors are those in
        `SECOND_DESIRED_PREDICTOR_NAMES`.
        """

        these_training_matrices, these_validation_matrices = (
            sequential_selection._subset_input_matrices(
                list_of_training_matrices=LIST_OF_TRAINING_MATRICES,
                list_of_validation_matrices=LIST_OF_VALIDATION_MATRICES,
                predictor_names_by_matrix=PREDICTOR_NAMES_BY_MATRIX,
                desired_predictor_names=SECOND_DESIRED_PREDICTOR_NAMES)
        )

        self.assertTrue(_compare_lists_of_matrices(
            these_training_matrices, SECOND_SUBSET_TRAINING_MATRICES))
        self.assertTrue(_compare_lists_of_matrices(
            these_validation_matrices, SECOND_SUBSET_VALIDATION_MATRICES))

    def test_subset_input_matrices_third(self):
        """Ensures correct output from _subset_input_matrices.

        In this case, desired predictors are those in
        `THIRD_DESIRED_PREDICTOR_NAMES`.
        """

        these_training_matrices, these_validation_matrices = (
            sequential_selection._subset_input_matrices(
                list_of_training_matrices=LIST_OF_TRAINING_MATRICES,
                list_of_validation_matrices=LIST_OF_VALIDATION_MATRICES,
                predictor_names_by_matrix=PREDICTOR_NAMES_BY_MATRIX,
                desired_predictor_names=THIRD_DESIRED_PREDICTOR_NAMES)
        )

        self.assertTrue(_compare_lists_of_matrices(
            these_training_matrices, THIRD_SUBSET_TRAINING_MATRICES))
        self.assertTrue(_compare_lists_of_matrices(
            these_validation_matrices, THIRD_SUBSET_VALIDATION_MATRICES))

    def test_eval_sfs_stopping_criterion_absolute_first(self):
        """Ensures correct output from _eval_sfs_stopping_criterion.

        In this case, stopping criterion is `min_loss_decrease` and array of
        lowest costs is `LOWEST_COST_BY_STEP_FIRST`.
        """

        this_flag = sequential_selection._eval_sfs_stopping_criterion(
            min_loss_decrease=MIN_LOSS_DECREASE,
            min_percentage_loss_decrease=None,
            num_steps_for_loss_decrease=NUM_STEPS_FOR_LOSS_DECREASE,
            lowest_cost_by_step=LOWEST_COST_BY_STEP_FIRST)

        self.assertFalse(this_flag)

    def test_eval_sfs_stopping_criterion_percent_first(self):
        """Ensures correct output from _eval_sfs_stopping_criterion.

        In this case, stopping criterion is `min_percentage_loss_decrease` and
        array of lowest costs is `LOWEST_COST_BY_STEP_FIRST`.
        """

        this_flag = sequential_selection._eval_sfs_stopping_criterion(
            min_loss_decrease=None,
            min_percentage_loss_decrease=MIN_PERCENTAGE_LOSS_DECREASE,
            num_steps_for_loss_decrease=NUM_STEPS_FOR_LOSS_DECREASE,
            lowest_cost_by_step=LOWEST_COST_BY_STEP_FIRST)

        self.assertTrue(this_flag)

    def test_eval_sfs_stopping_criterion_absolute_second(self):
        """Ensures correct output from _eval_sfs_stopping_criterion.

        In this case, stopping criterion is `min_loss_decrease` and array of
        lowest costs is `LOWEST_COST_BY_STEP_SECOND`.
        """

        this_flag = sequential_selection._eval_sfs_stopping_criterion(
            min_loss_decrease=MIN_LOSS_DECREASE,
            min_percentage_loss_decrease=None,
            num_steps_for_loss_decrease=NUM_STEPS_FOR_LOSS_DECREASE,
            lowest_cost_by_step=LOWEST_COST_BY_STEP_SECOND)

        self.assertTrue(this_flag)

    def test_eval_sfs_stopping_criterion_percent_second(self):
        """Ensures correct output from _eval_sfs_stopping_criterion.

        In this case, stopping criterion is `min_percentage_loss_decrease` and
        array of lowest costs is `LOWEST_COST_BY_STEP_SECOND`.
        """

        this_flag = sequential_selection._eval_sfs_stopping_criterion(
            min_loss_decrease=None,
            min_percentage_loss_decrease=MIN_PERCENTAGE_LOSS_DECREASE,
            num_steps_for_loss_decrease=NUM_STEPS_FOR_LOSS_DECREASE,
            lowest_cost_by_step=LOWEST_COST_BY_STEP_SECOND)

        self.assertFalse(this_flag)


if __name__ == '__main__':
    unittest.main()
