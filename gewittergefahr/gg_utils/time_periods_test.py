"""Unit tests for time_periods.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import time_periods

ESTIMATED_START_TIME_UNIX_SEC = 1506999783  # 030303 UTC 3 Oct 2017
ESTIMATED_END_TIME_UNIX_SEC = 1507004664  # 042424 UTC 3 Oct 2017
TIME_INTERVAL_SEC = 600

TIMES_WITH_ENDPOINT_UNIX_SEC = numpy.array(
    [1506999600, 1507000200, 1507000800, 1507001400, 1507002000, 1507002600,
     1507003200, 1507003800, 1507004400, 1507005000])
TIMES_WITHOUT_ENDPOINT_UNIX_SEC = numpy.array(
    [1506999600, 1507000200, 1507000800, 1507001400, 1507002000, 1507002600,
     1507003200, 1507003800, 1507004400])

TIME_IN_PERIOD_UNIX_SEC = 1507003444  # 040404 UTC 3 Oct 2017
PERIOD_LENGTH_SEC = 5400
START_TIME_UNIX_SEC = 1506999600
END_TIME_UNIX_SEC = 1507005000


class TimePeriodsTests(unittest.TestCase):
    """Each method is a unit test for time_periods.py."""

    def test_range_and_interval_to_list_include_endpoint(self):
        """Ensures correct output from range_and_interval_to_list.

        In this case, endpoint of period is included in list of exact times.
        """

        these_times_unix_sec = time_periods.range_and_interval_to_list(
            start_time_unix_sec=ESTIMATED_START_TIME_UNIX_SEC,
            end_time_unix_sec=ESTIMATED_END_TIME_UNIX_SEC,
            time_interval_sec=TIME_INTERVAL_SEC, include_endpoint=True)
        self.assertTrue(numpy.array_equal(these_times_unix_sec,
                                          TIMES_WITH_ENDPOINT_UNIX_SEC))

    def test_range_and_interval_to_list_exclude_endpoint(self):
        """Ensures correct output from range_and_interval_to_list.

        In this case, endpoint of period is excluded from list of exact times.
        """

        these_times_unix_sec = time_periods.range_and_interval_to_list(
            start_time_unix_sec=ESTIMATED_START_TIME_UNIX_SEC,
            end_time_unix_sec=ESTIMATED_END_TIME_UNIX_SEC,
            time_interval_sec=TIME_INTERVAL_SEC, include_endpoint=False)
        self.assertTrue(numpy.array_equal(these_times_unix_sec,
                                          TIMES_WITHOUT_ENDPOINT_UNIX_SEC))

    def test_time_and_period_length_to_range(self):
        """Ensures correct output from time_and_period_length_to_range."""

        (this_start_time_unix_sec,
         this_end_time_unix_sec) = time_periods.time_and_period_length_to_range(
             TIME_IN_PERIOD_UNIX_SEC, PERIOD_LENGTH_SEC)

        self.assertTrue(this_start_time_unix_sec == START_TIME_UNIX_SEC)
        self.assertTrue(this_end_time_unix_sec == END_TIME_UNIX_SEC)

    def test_time_and_period_length_and_interval_to_list_include_endpoint(self):
        """Ensures correctness of time_and_period_length_and_interval_to_list.

        In this case, endpoint of period is included in list of exact times.
        """

        these_times_unix_sec = (
            time_periods.time_and_period_length_and_interval_to_list(
                unix_time_sec=TIME_IN_PERIOD_UNIX_SEC,
                period_length_sec=PERIOD_LENGTH_SEC,
                time_interval_sec=TIME_INTERVAL_SEC, include_endpoint=True))
        self.assertTrue(numpy.array_equal(these_times_unix_sec,
                                          TIMES_WITH_ENDPOINT_UNIX_SEC))

    def test_time_and_period_length_and_interval_to_list_exclude_endpoint(self):
        """Ensures correctness of time_and_period_length_and_interval_to_list.

        In this case, endpoint of period is excluded from list of exact times.
        """

        these_times_unix_sec = (
            time_periods.time_and_period_length_and_interval_to_list(
                unix_time_sec=TIME_IN_PERIOD_UNIX_SEC,
                period_length_sec=PERIOD_LENGTH_SEC,
                time_interval_sec=TIME_INTERVAL_SEC, include_endpoint=False))
        self.assertTrue(numpy.array_equal(these_times_unix_sec,
                                          TIMES_WITHOUT_ENDPOINT_UNIX_SEC))


if __name__ == '__main__':
    unittest.main()
