"""Unit tests for fancy_downsampling.py."""

import copy
import unittest
import numpy
from gewittergefahr.deep_learning import fancy_downsampling

# The following constants are used to test _find_uncovered_times.
ALL_TIMES_UNIX_SEC = numpy.array([
    0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5,
    5, 5, 6
], dtype=int)

COVERED_TIMES_UNIX_SEC = numpy.array(
    [0, 4, 1, 1, 5, 0, 4, 0, 0, 3, 1, 3, 3, 5, 3], dtype=int
)
UNCOVERED_INDICES = numpy.array([9, 10, 11, 12, 13, 14, 27], dtype=int)

# The following constants are used to test _find_storm_cells.
OBJECT_ID_STRINGS = [
    'a', 'b', 'c', 'd', 'e', 'a', 'b', 'c', 'e', 'f', 'a', 'b', 'e', 'f', 'a',
    'b', 'f', 'g', 'b', 'f', 'g'
]
DESIRED_CELL_ID_STRINGS_NONE_MISSING = [
    'a', 'a', 'a', 'a', 'd', 'd', 'd', 'c', 'c', 'c',
]
DESIRED_CELL_ID_STRINGS_SOME_MISSING = [
    'a', 'a', 'a', 'a', 'd', 'd', 'd', 'h', 'h', 'c', 'c', 'c', 'i'
]

DESIRED_OBJECT_INDICES = numpy.array([0, 2, 3, 5, 7, 10, 14], dtype=int)

# The following constants are used to test downsample_for_non_training.
TARGET_NAME = 'tornado_lead-time=0000-3600sec_distance=00000-10000m'
CLASS_FRACTION_DICT = {0: 0.5, 1: 0.5}

MAIN_ID_STRINGS = [
    'b', 'b', 'b', 'b', 'e', 'e', 'f', 'd', 'f', 'd', 'f',
    'a', 'c', 'd', 'f', 'a', 'c', 'd', 'f', 'a', 'c', 'd', 'f',
    'a', 'c', 'a', 'c', 'c', 'c'
]

MAIN_STORM_TIMES_UNIX_SEC = numpy.array([
    1, 2, 3, 4, 5, 6, 8, 9, 9, 10, 10,
    11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13,
    14, 14, 15, 15, 16, 17
], dtype=int)

MAIN_TARGET_VALUES = numpy.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1,
    1, 1, 0, 1, 0, 0
], dtype=int)

THESE_INDICES = numpy.array(
    [5, 6, 7, 8, 9, 10, 11, 12, 15, 13, 14, 17, 18, 19, 22, 23, 24, 26],
    dtype=int
)

NON_TRAINING_ID_STRINGS = [MAIN_ID_STRINGS[k] for k in THESE_INDICES]
NON_TRAINING_TIMES_UNIX_SEC = MAIN_STORM_TIMES_UNIX_SEC[THESE_INDICES]
NON_TRAINING_TARGET_VALUES = MAIN_TARGET_VALUES[THESE_INDICES]

# The following constants are used to test downsample_for_training.
THESE_INDICES = numpy.linspace(
    5, len(MAIN_ID_STRINGS) - 1, num=len(MAIN_ID_STRINGS) - 5, dtype=int
)

TRAINING_ID_STRINGS = [MAIN_ID_STRINGS[k] for k in THESE_INDICES]
TRAINING_TIMES_UNIX_SEC = MAIN_STORM_TIMES_UNIX_SEC[THESE_INDICES]
TRAINING_TARGET_VALUES = MAIN_TARGET_VALUES[THESE_INDICES]


class FancyDownsamplingTests(unittest.TestCase):
    """Each method is a unit test for fancy_downsampling.py."""

    def test_find_storm_cells_none_good(self):
        """Ensures correct output from find_storm_cells.

        In this case no desired storm IDs are missing.
        """

        these_indices = fancy_downsampling._find_storm_cells(
            object_id_strings=OBJECT_ID_STRINGS,
            desired_cell_id_strings=DESIRED_CELL_ID_STRINGS_NONE_MISSING)

        self.assertTrue(numpy.array_equal(
            these_indices, DESIRED_OBJECT_INDICES))

    def test_find_storm_cells_none_bad(self):
        """Ensures correct output from find_storm_cells.

        In this case some desired storm IDs are missing.
        """

        with self.assertRaises(ValueError):
            fancy_downsampling._find_storm_cells(
                object_id_strings=OBJECT_ID_STRINGS,
                desired_cell_id_strings=DESIRED_CELL_ID_STRINGS_SOME_MISSING)

    def test_find_uncovered_times_good(self):
        """Ensures correct output from _find_uncovered_times.

        In this case all covered times are found in the main input array
        (`all_times_unix_sec`).
        """

        these_indices = fancy_downsampling._find_uncovered_times(
            all_times_unix_sec=ALL_TIMES_UNIX_SEC,
            covered_times_unix_sec=COVERED_TIMES_UNIX_SEC)

        self.assertTrue(numpy.array_equal(these_indices, UNCOVERED_INDICES))

    def test_find_uncovered_times_bad(self):
        """Ensures correct output from _find_uncovered_times.

        In this case some covered times are *not* found in the main input array
        (`all_times_unix_sec`).
        """

        extra_times_unix_sec = numpy.array(
            [numpy.max(ALL_TIMES_UNIX_SEC) + 1], dtype=int
        )
        these_covered_times_unix_sec = numpy.concatenate((
            COVERED_TIMES_UNIX_SEC, extra_times_unix_sec
        ))

        with self.assertRaises(ValueError):
            fancy_downsampling._find_uncovered_times(
                all_times_unix_sec=ALL_TIMES_UNIX_SEC,
                covered_times_unix_sec=these_covered_times_unix_sec)

    def test_downsample_for_non_training(self):
        """Ensures correct output from downsample_for_non_training."""

        these_id_strings, these_times_unix_sec, these_target_values = (
            fancy_downsampling.downsample_for_non_training(
                full_id_strings=copy.deepcopy(MAIN_ID_STRINGS),
                storm_times_unix_sec=MAIN_STORM_TIMES_UNIX_SEC + 0,
                target_values=MAIN_TARGET_VALUES + 0, target_name=TARGET_NAME,
                class_fraction_dict=CLASS_FRACTION_DICT, test_mode=True)
        )

        self.assertTrue(NON_TRAINING_ID_STRINGS == these_id_strings)
        self.assertTrue(numpy.array_equal(
            NON_TRAINING_TIMES_UNIX_SEC, these_times_unix_sec
        ))
        self.assertTrue(numpy.array_equal(
            NON_TRAINING_TARGET_VALUES, these_target_values
        ))

    def test_downsample_for_training(self):
        """Ensures correct output from downsample_for_training."""

        these_id_strings, these_times_unix_sec, these_target_values = (
            fancy_downsampling.downsample_for_training(
                full_id_strings=copy.deepcopy(MAIN_ID_STRINGS),
                storm_times_unix_sec=MAIN_STORM_TIMES_UNIX_SEC + 0,
                target_values=MAIN_TARGET_VALUES + 0, target_name=TARGET_NAME,
                class_fraction_dict=CLASS_FRACTION_DICT, test_mode=True)
        )

        self.assertTrue(TRAINING_ID_STRINGS == these_id_strings)
        self.assertTrue(numpy.array_equal(
            TRAINING_TIMES_UNIX_SEC, these_times_unix_sec
        ))
        self.assertTrue(numpy.array_equal(
            TRAINING_TARGET_VALUES, these_target_values
        ))


if __name__ == '__main__':
    unittest.main()
