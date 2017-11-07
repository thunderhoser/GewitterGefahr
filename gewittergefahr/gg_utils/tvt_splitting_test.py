"""Unit tests for tvt_splitting.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import tvt_splitting

# The following constants are used to test _apply_time_separation and
# check_time_separation.
UNIX_TIMES_SEC = numpy.array(
    [0, 10000, 100, 3000, 300, 2233, 500, 2100, 800, 2000, 1000, 1500],
    dtype=int)

TIME_SEPARATION_SEC = 1000
EARLY_INDICES_NO_SEPARATION = numpy.array([0, 2, 4, 6, 8, 10], dtype=int)
LATE_INDICES_NO_SEPARATION = numpy.array([1, 3, 5, 7, 9, 11], dtype=int)
EARLY_INDICES_WITH_SEPARATION = numpy.array([0, 2, 4, 6], dtype=int)
LATE_INDICES_WITH_SEPARATION = numpy.array([1, 3, 5, 7, 9], dtype=int)

# The following constants are used to test split_training_validation_testing.
VALIDATION_FRACTION = 0.333333
TESTING_FRACTION = 0.166667
TRAINING_INDICES_NO_BIAS_CORRECTION = numpy.array([0, 2, 4, 6], dtype=int)
VALIDATION_INDICES_NO_BIAS_CORRECTION = numpy.array([7, 9], dtype=int)
TESTING_INDICES_NO_BIAS_CORRECTION = numpy.array([1], dtype=int)

# The following constants are used to test split_training_for_bias_correction.
BASE_MODEL_TIMES_UNIX_SEC = numpy.array(
    [-2000, 1234567, -1500, 20000, -1000, 10000, -500, 5000, 0, 2000, 250, 1000,
     300, 500, 400], dtype=int)
BIAS_CORRECTION_TIMES_UNIX_SEC = numpy.array(
    [0, 20000, 100, 15000, 200, 9999, 500, 8888, 750, 7777, 1000, 6666, 1500,
     5555, 2000, 4000, 2500, 3250], dtype=int)

BIAS_CORRECTION_VALIDATION_INDICES = numpy.array([7, 9, 11, 13, 15], dtype=int)
BIAS_CORRECTION_TESTING_INDICES = numpy.array([1, 3, 5], dtype=int)
FIRST_NON_TRAINING_TIME_UNIX_SEC = 4000

BASE_MODEL_TRAINING_FRACTION = 0.6
BASE_MODEL_TRAINING_INDICES = numpy.array(
    [0, 2, 4, 6, 8, 10, 11, 12, 13, 14], dtype=int)
BIAS_CORRECTION_TRAINING_INDICES = numpy.array([14, 16], dtype=int)


class TvtSplittingTests(unittest.TestCase):
    """Each method is a unit test for tvt_splitting.py."""

    def test_apply_time_separation_necessary(self):
        """Ensures correct output from _apply_time_separation.

        In this case, early and late sets are not yet separated by 1000 seconds.
        """

        these_early_indices, these_late_indices = (
            tvt_splitting._apply_time_separation(
                UNIX_TIMES_SEC, early_indices=EARLY_INDICES_NO_SEPARATION,
                late_indices=LATE_INDICES_NO_SEPARATION,
                time_separation_sec=TIME_SEPARATION_SEC))

        self.assertTrue(set(these_early_indices) ==
                        set(EARLY_INDICES_WITH_SEPARATION))
        self.assertTrue(set(these_late_indices) ==
                        set(LATE_INDICES_WITH_SEPARATION))

    def test_apply_time_separation_unnecessary(self):
        """Ensures correct output from _apply_time_separation.

        In this case, early and late sets are already separated by 1000 seconds.
        """

        these_early_indices, these_late_indices = (
            tvt_splitting._apply_time_separation(
                UNIX_TIMES_SEC, early_indices=EARLY_INDICES_WITH_SEPARATION,
                late_indices=LATE_INDICES_WITH_SEPARATION,
                time_separation_sec=TIME_SEPARATION_SEC))

        self.assertTrue(set(these_early_indices) ==
                        set(EARLY_INDICES_WITH_SEPARATION))
        self.assertTrue(set(these_late_indices) ==
                        set(LATE_INDICES_WITH_SEPARATION))

    def test_check_time_separation_bad(self):
        """Ensures correct output from check_time_separation.

        In this case, early and late sets are *not* separated by 1000 seconds.
        """

        with self.assertRaises(ValueError):
            tvt_splitting.check_time_separation(
                UNIX_TIMES_SEC, early_indices=EARLY_INDICES_NO_SEPARATION,
                late_indices=LATE_INDICES_NO_SEPARATION,
                time_separation_sec=TIME_SEPARATION_SEC)

    def test_check_time_separation_good(self):
        """Ensures correct output from check_time_separation.

        In this case, early and late sets are separated by 1000 seconds.
        """

        tvt_splitting.check_time_separation(
            UNIX_TIMES_SEC, early_indices=EARLY_INDICES_WITH_SEPARATION,
            late_indices=LATE_INDICES_WITH_SEPARATION,
            time_separation_sec=TIME_SEPARATION_SEC)

    def test_split_training_validation_testing(self):
        """Ensures correct output from split_training_validation_testing."""

        (these_training_indices,
         these_validation_indices,
         these_testing_indices) = (
             tvt_splitting.split_training_validation_testing(
                 UNIX_TIMES_SEC, validation_fraction=VALIDATION_FRACTION,
                 testing_fraction=TESTING_FRACTION,
                 time_separation_sec=TIME_SEPARATION_SEC))

        self.assertTrue(set(these_training_indices) ==
                        set(TRAINING_INDICES_NO_BIAS_CORRECTION))
        self.assertTrue(set(these_validation_indices) ==
                        set(VALIDATION_INDICES_NO_BIAS_CORRECTION))
        self.assertTrue(set(these_testing_indices) ==
                        set(TESTING_INDICES_NO_BIAS_CORRECTION))

    def test_split_training_for_bias_correction(self):
        """Ensures correct output from split_training_for_bias_correction."""

        (these_base_model_training_indices,
         these_bias_correction_training_indices) = (
             tvt_splitting.split_training_for_bias_correction(
                 all_base_model_times_unix_sec=BASE_MODEL_TIMES_UNIX_SEC,
                 all_bias_correction_times_unix_sec=
                 BIAS_CORRECTION_TIMES_UNIX_SEC,
                 base_model_fraction=BASE_MODEL_TRAINING_FRACTION,
                 first_non_training_time_unix_sec=
                 FIRST_NON_TRAINING_TIME_UNIX_SEC,
                 time_separation_sec=TIME_SEPARATION_SEC))

        self.assertTrue(set(these_base_model_training_indices) ==
                        set(BASE_MODEL_TRAINING_INDICES))
        self.assertTrue(set(these_bias_correction_training_indices) ==
                        set(BIAS_CORRECTION_TRAINING_INDICES))

    def test_split_tvt_for_bias_correction(self):
        """Ensures correct output from split_tvt_for_bias_correction."""

        (these_base_model_training_indices,
         these_bias_correction_training_indices,
         these_validation_indices,
         these_testing_indices) = tvt_splitting.split_tvt_for_bias_correction(
             base_model_times_unix_sec=BASE_MODEL_TIMES_UNIX_SEC,
             bias_correction_times_unix_sec=BIAS_CORRECTION_TIMES_UNIX_SEC,
             validation_fraction=VALIDATION_FRACTION,
             testing_fraction=TESTING_FRACTION,
             base_model_training_fraction=BASE_MODEL_TRAINING_FRACTION,
             time_separation_sec=TIME_SEPARATION_SEC)

        self.assertTrue(set(these_base_model_training_indices) ==
                        set(BASE_MODEL_TRAINING_INDICES))
        self.assertTrue(set(these_bias_correction_training_indices) ==
                        set(BIAS_CORRECTION_TRAINING_INDICES))
        self.assertTrue(set(these_validation_indices) ==
                        set(BIAS_CORRECTION_VALIDATION_INDICES))
        self.assertTrue(set(these_testing_indices) ==
                        set(BIAS_CORRECTION_TESTING_INDICES))


if __name__ == '__main__':
    unittest.main()
