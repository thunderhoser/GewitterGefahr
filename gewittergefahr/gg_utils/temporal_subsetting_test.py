"""Unit tests for temporal_subsetting.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import temporal_subsetting

# The following constants are used to test get_monthly_chunks.
CHUNK_TO_MONTHS_DICT_1EACH = {
    0: 1,
    1: 2,
    2: 3,
    3: 4,
    4: 5,
    5: 6,
    6: 7,
    7: 8,
    8: 9,
    9: 10,
    10: 11,
    11: 12
}

for THIS_KEY in CHUNK_TO_MONTHS_DICT_1EACH:
    CHUNK_TO_MONTHS_DICT_1EACH[THIS_KEY] = numpy.array(
        [CHUNK_TO_MONTHS_DICT_1EACH[THIS_KEY]], dtype=int
    )

# The following constants are used to test get_hourly_chunks.
CHUNK_TO_HOURS_DICT_1EACH = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    12: 12,
    13: 13,
    14: 14,
    15: 15,
    16: 16,
    17: 17,
    18: 18,
    19: 19,
    20: 20,
    21: 21,
    22: 22,
    23: 23
}

for THIS_KEY in CHUNK_TO_HOURS_DICT_1EACH:
    CHUNK_TO_HOURS_DICT_1EACH[THIS_KEY] = numpy.array(
        [CHUNK_TO_HOURS_DICT_1EACH[THIS_KEY]], dtype=int
    )

CHUNK_TO_HOURS_DICT_3EACH = {
    0: numpy.array([0, 1, 2]),
    1: numpy.array([3, 4, 5]),
    2: numpy.array([6, 7, 8]),
    3: numpy.array([9, 10, 11]),
    4: numpy.array([12, 13, 14]),
    5: numpy.array([15, 16, 17]),
    6: numpy.array([18, 19, 20]),
    7: numpy.array([21, 22, 23])
}

for THIS_KEY in CHUNK_TO_HOURS_DICT_3EACH:
    CHUNK_TO_HOURS_DICT_3EACH[THIS_KEY] = numpy.array(
        CHUNK_TO_HOURS_DICT_3EACH[THIS_KEY], dtype=int
    )

CHUNK_TO_HOURS_DICT_6EACH = {
    0: numpy.array([0, 1, 2, 3, 4, 5]),
    1: numpy.array([6, 7, 8, 9, 10, 11]),
    2: numpy.array([12, 13, 14, 15, 16, 17]),
    3: numpy.array([18, 19, 20, 21, 22, 23])
}

for THIS_KEY in CHUNK_TO_HOURS_DICT_6EACH:
    CHUNK_TO_HOURS_DICT_6EACH[THIS_KEY] = numpy.array(
        CHUNK_TO_HOURS_DICT_6EACH[THIS_KEY], dtype=int
    )

# The following constants are used to test get_events_in_months and
# get_events_in_hours.
EVENT_TIME_STRINGS = [
    '4001-01-01-01', '4002-02-02-02', '4003-03-03-03', '4004-04-04-04',
    '4005-05-05-05', '4006-06-06-06', '4007-07-07-07', '4008-08-08-08',
    '4009-09-09-09', '4010-10-10-10', '4011-11-11-11', '4012-12-12-12'
]
EVENT_TIMES_UNIX_SEC = numpy.array([
    time_conversion.string_to_unix_sec(t, '%Y-%m-%d-%H')
    for t in EVENT_TIME_STRINGS
], dtype=int)

EVENT_MONTHS = numpy.linspace(1, 12, num=12, dtype=int)
EVENT_HOURS = numpy.linspace(1, 12, num=12, dtype=int)

DESIRED_MONTHS = numpy.array([12, 1, 2], dtype=int)
INDICES_IN_DESIRED_MONTHS = numpy.array([0, 1, 11], dtype=int)

MORNING_HOURS = numpy.array([6, 7, 8, 9, 10, 11], dtype=int)
MORNING_INDICES = numpy.array([5, 6, 7, 8, 9, 10], dtype=int)

EVENING_HOURS = numpy.array([18, 19, 20, 21, 22, 23], dtype=int)
EVENING_INDICES = numpy.array([], dtype=int)


def _compare_chunk_dicts(first_dict, second_dict):
    """Compare two dictionaries, mapping chunk index to either hours or months.

    :param first_dict: First dictionary.
    :param second_dict: Second dictionary.
    :return: are_dicts_equal: Boolean flag.
    """

    first_keys = list(first_dict.keys())
    second_keys = list(second_dict.keys())
    if set(first_keys) != set(second_keys):
        return False

    for this_key in first_keys:
        if numpy.array_equal(first_dict[this_key], second_dict[this_key]):
            continue

        return False

    return True


class TemporalSubsettingTests(unittest.TestCase):
    """Each method is a unit test for temporal_subsetting.py."""

    def test_get_monthly_chunks_1each(self):
        """Ensures correct output from get_monthly_chunks.

        In this case there is one month per chunk.
        """

        this_chunk_to_months_dict = temporal_subsetting.get_monthly_chunks(
            num_months_per_chunk=1, verbose=False)

        self.assertTrue(_compare_chunk_dicts(
            this_chunk_to_months_dict, CHUNK_TO_MONTHS_DICT_1EACH
        ))

    def test_get_hourly_chunks_1each(self):
        """Ensures correct output from get_hourly_chunks.

        In this case there is one hour per chunk.
        """

        this_chunk_to_hours_dict = temporal_subsetting.get_hourly_chunks(
            num_hours_per_chunk=1, verbose=False)

        self.assertTrue(_compare_chunk_dicts(
            this_chunk_to_hours_dict, CHUNK_TO_HOURS_DICT_1EACH
        ))

    def test_get_hourly_chunks_3each(self):
        """Ensures correct output from get_hourly_chunks.

        In this case there are 3 hours per chunk.
        """

        this_chunk_to_hours_dict = temporal_subsetting.get_hourly_chunks(
            num_hours_per_chunk=3, verbose=False)

        self.assertTrue(_compare_chunk_dicts(
            this_chunk_to_hours_dict, CHUNK_TO_HOURS_DICT_3EACH
        ))

    def test_get_hourly_chunks_6(self):
        """Ensures correct output from get_hourly_chunks.

        In this case there are 6 hours per chunk.
        """

        this_chunk_to_hours_dict = temporal_subsetting.get_hourly_chunks(
            num_hours_per_chunk=6, verbose=False)

        self.assertTrue(_compare_chunk_dicts(
            this_chunk_to_hours_dict, CHUNK_TO_HOURS_DICT_6EACH
        ))

    def test_get_events_in_months(self):
        """Ensures correct output from get_events_in_months."""

        these_indices, these_months = temporal_subsetting.get_events_in_months(
            desired_months=DESIRED_MONTHS,
            event_times_unix_sec=EVENT_TIMES_UNIX_SEC, verbose=False)

        self.assertTrue(numpy.array_equal(
            these_indices, INDICES_IN_DESIRED_MONTHS
        ))
        self.assertTrue(numpy.array_equal(these_months, EVENT_MONTHS))

    def test_get_events_in_morning(self):
        """Ensures correct output from get_events_in_hours.

        In this case, only morning hours are desired.
        """

        these_indices, these_hours = temporal_subsetting.get_events_in_hours(
            desired_hours=MORNING_HOURS,
            event_times_unix_sec=EVENT_TIMES_UNIX_SEC, verbose=False)

        self.assertTrue(numpy.array_equal(these_indices, MORNING_INDICES))
        self.assertTrue(numpy.array_equal(these_hours, EVENT_HOURS))

    def test_get_events_in_evening(self):
        """Ensures correct output from get_events_in_hours.

        In this case, only evening hours are desired.
        """

        these_indices, these_hours = temporal_subsetting.get_events_in_hours(
            desired_hours=EVENING_HOURS,
            event_times_unix_sec=EVENT_TIMES_UNIX_SEC, verbose=False)

        self.assertTrue(numpy.array_equal(these_indices, EVENING_INDICES))
        self.assertTrue(numpy.array_equal(these_hours, EVENT_HOURS))


if __name__ == '__main__':
    unittest.main()
