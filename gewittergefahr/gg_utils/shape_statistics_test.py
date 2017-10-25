"""Unit tests for shape_statistics.py."""

import unittest
from gewittergefahr.gg_utils import shape_statistics as shape_stats

FAKE_STATISTIC_NAME = 'foo'


class ShapeStatisticsTests(unittest.TestCase):
    """Each method is a unit test for shape_statistics.py."""

    def test_check_statistic_names_all_valid(self):
        """Ensures correct output from _check_statistic_names.

        In this case, all input names are valid.
        """

        shape_stats._check_statistic_names(shape_stats.VALID_STATISTIC_NAMES)

    def test_check_statistic_names_one_invalid(self):
        """Ensures correct output from _check_statistic_names.

        In this case, one input name is invalid.
        """

        with self.assertRaises(ValueError):
            shape_stats._check_statistic_names(
                shape_stats.VALID_STATISTIC_NAMES + [FAKE_STATISTIC_NAME])

    def test_stat_name_new_to_orig_different(self):
        """Ensures correct output from _stat_name_new_to_orig.

        In this case the statistic is area, for which new and original names are
        different.
        """

        this_statistic_name_orig = shape_stats._stat_name_new_to_orig(
            shape_stats.AREA_NAME)
        self.assertTrue(this_statistic_name_orig == shape_stats.AREA_NAME_ORIG)

    def test_stat_name_new_to_orig_same(self):
        """Ensures correct output from _stat_name_new_to_orig.

        In this case the statistic is solidity, for which new and original names
        are the same.
        """

        this_statistic_name_orig = shape_stats._stat_name_new_to_orig(
            shape_stats.SOLIDITY_NAME)
        self.assertTrue(
            this_statistic_name_orig == shape_stats.SOLIDITY_NAME_ORIG)


if __name__ == '__main__':
    unittest.main()
