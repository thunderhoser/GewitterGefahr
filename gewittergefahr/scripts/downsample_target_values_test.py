"""Unit tests for downsample_target_values.py."""

import unittest
import numpy
from gewittergefahr.scripts import downsample_target_values

ALL_TIMES_UNIX_SEC = numpy.array(
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5,
     5, 5, 6],
    dtype=int)
COVERED_TIMES_UNIX_SEC = numpy.array(
    [0, 4, 1, 1, 5, 0, 4, 0, 0, 3, 1, 3, 3, 5, 3], dtype=int)
UNCOVERED_INDICES = numpy.array([9, 10, 11, 12, 13, 14, 27], dtype=int)


class DownsampleTargetValuesTests(unittest.TestCase):
    """Each method is a unit test for downsample_target_values.py."""

    def test_find_uncovered_times_good(self):
        """Ensures correct output from _find_uncovered_times.

        In this case all covered times are found in the main input array
        (`all_times_unix_sec`).
        """

        these_indices = downsample_target_values._find_uncovered_times(
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
            downsample_target_values._find_uncovered_times(
                all_times_unix_sec=ALL_TIMES_UNIX_SEC,
                covered_times_unix_sec=these_covered_times_unix_sec)


if __name__ == '__main__':
    unittest.main()
