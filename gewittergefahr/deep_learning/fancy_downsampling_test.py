"""Unit tests for fancy_downsampling.py."""

import unittest
import numpy
from gewittergefahr.deep_learning import fancy_downsampling

# The following constants are used to test _find_uncovered_times.
ALL_TIMES_UNIX_SEC = numpy.array(
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5,
     5, 5, 6],
    dtype=int)
COVERED_TIMES_UNIX_SEC = numpy.array(
    [0, 4, 1, 1, 5, 0, 4, 0, 0, 3, 1, 3, 3, 5, 3], dtype=int)
UNCOVERED_INDICES = numpy.array([9, 10, 11, 12, 13, 14, 27], dtype=int)

# The following constants are used to test _find_storm_cells.
STORM_ID_BY_OBJECT = [
    'a', 'b', 'c', 'd', 'e', 'a', 'b', 'c', 'e', 'f', 'a', 'b', 'e', 'f', 'a',
    'b', 'f', 'g', 'b', 'f', 'g'
]
DESIRED_CELL_IDS_NONE_MISSING = [
    'a', 'a', 'a', 'a', 'd', 'd', 'd', 'c', 'c', 'c',
]
DESIRED_CELL_IDS_SOME_MISSING = [
    'a', 'a', 'a', 'a', 'd', 'd', 'd', 'h', 'h', 'c', 'c', 'c', 'i'
]

DESIRED_OBJECT_INDICES = numpy.array([0, 2, 3, 5, 7, 10, 14], dtype=int)


class FancyDownsamplingTests(unittest.TestCase):
    """Each method is a unit test for fancy_downsampling.py."""

    def test_find_storm_cells_none_good(self):
        """Ensures correct output from find_storm_cells.

        In this case no desired storm IDs are missing.
        """

        these_indices = fancy_downsampling._find_storm_cells(
            storm_id_by_object=STORM_ID_BY_OBJECT,
            desired_storm_cell_ids=DESIRED_CELL_IDS_NONE_MISSING)

        self.assertTrue(numpy.array_equal(
            these_indices, DESIRED_OBJECT_INDICES))

    def test_find_storm_cells_none_bad(self):
        """Ensures correct output from find_storm_cells.

        In this case some desired storm IDs are missing.
        """

        with self.assertRaises(ValueError):
            fancy_downsampling._find_storm_cells(
                storm_id_by_object=STORM_ID_BY_OBJECT,
                desired_storm_cell_ids=DESIRED_CELL_IDS_SOME_MISSING)

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
            [numpy.max(ALL_TIMES_UNIX_SEC) + 1], dtype=int)
        these_covered_times_unix_sec = numpy.concatenate((
            COVERED_TIMES_UNIX_SEC, extra_times_unix_sec))

        with self.assertRaises(ValueError):
            fancy_downsampling._find_uncovered_times(
                all_times_unix_sec=ALL_TIMES_UNIX_SEC,
                covered_times_unix_sec=these_covered_times_unix_sec)


if __name__ == '__main__':
    unittest.main()
