"""Unit tests for prob_matched_means.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import prob_matched_means as pmm

TOLERANCE = 1e-6
MAX_PERCENTILE_LEVEL = 99.

# The following constants are used to test _run_pmm_one_variable.
THIS_EXAMPLE_MATRIX = numpy.array([
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
], dtype=float)

INPUT_MATRIX_ONE_VAR = numpy.stack((
    THIS_EXAMPLE_MATRIX, 2 * THIS_EXAMPLE_MATRIX, 3 * THIS_EXAMPLE_MATRIX,
    4 * THIS_EXAMPLE_MATRIX
), axis=0)

MEAN_FIELD_MATRIX_ONE_VAR = numpy.array([
    [2, 4, 6, 8, 9, 12, 13, 15, 17, 19, 21, 23, 24],
    [27, 30, 34, 38, 42, 45, 48, 52, 60, 66, 72, 80, 96]
], dtype=float)

# The following constants are used to test run_pmm_many_variables.
INPUT_MATRIX_MANY_VARS = numpy.stack(
    (INPUT_MATRIX_ONE_VAR, 2 * INPUT_MATRIX_ONE_VAR, 3 * INPUT_MATRIX_ONE_VAR),
    axis=-1
)

MEAN_FIELD_MATRIX_MANY_VARS = numpy.stack((
    MEAN_FIELD_MATRIX_ONE_VAR, 2 * MEAN_FIELD_MATRIX_ONE_VAR,
    3 * MEAN_FIELD_MATRIX_ONE_VAR
), axis=-1)


class ProbMatchedMeansTests(unittest.TestCase):
    """Each method is a unit test for prob_matched_means.py."""

    def test_run_pmm_one_variable(self):
        """Ensures correct output from _run_pmm_one_variable."""

        this_mean_field_matrix = pmm._run_pmm_one_variable(
            input_matrix=INPUT_MATRIX_ONE_VAR,
            max_percentile_level=MAX_PERCENTILE_LEVEL)

        self.assertTrue(numpy.allclose(
            this_mean_field_matrix, MEAN_FIELD_MATRIX_ONE_VAR, atol=TOLERANCE
        ))

    def test_run_pmm_many_variables(self):
        """Ensures correct output from run_pmm_many_variables."""

        this_mean_field_matrix = pmm.run_pmm_many_variables(
            input_matrix=INPUT_MATRIX_MANY_VARS,
            max_percentile_level=MAX_PERCENTILE_LEVEL)

        self.assertTrue(numpy.allclose(
            this_mean_field_matrix, MEAN_FIELD_MATRIX_MANY_VARS, atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
