"""Unit tests for match_storms_between_datasets.py."""

import unittest
import numpy
import pandas
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.scripts import match_storms_between_datasets as match_storms

# The following constants are used to test _match_all_times.
MAX_TIME_DIFF_SECONDS = 180

FIRST_SOURCE_TIMES_UNIX_SEC = numpy.array([0, 300, 600, 900, 1200], dtype=int)
FIRST_TARGET_TIMES_UNIX_SEC = numpy.array([1033, 723, 847, 382, -54], dtype=int)
FIRST_TIME_MATCH_INDICES = numpy.array([4, 3, 1, 2, 0], dtype=int)

SECOND_SOURCE_TIMES_UNIX_SEC = numpy.array(
    [-54, 382, 723, 847, 1033], dtype=int
)
SECOND_TARGET_TIMES_UNIX_SEC = numpy.array([900, 0, 1200, 300, 600], dtype=int)
SECOND_TIME_MATCH_INDICES = numpy.array([1, 3, 4, 0, 0], dtype=int)

# The following constants are used to test _match_locations_one_time.
MAX_DISTANCE_METRES = 2e5

SOURCE_LATITUDES_DEG = numpy.array([51.0, 53.5, 52.3, 49.7, 55.2])
SOURCE_LONGITUDES_DEG = numpy.array([-114.1, -113.5, -113.8, -112.8, -118.8])
SOURCE_ID_STRINGS = ['a', 'b', 'c', 'd', 'e']
SOURCE_TIMES_UNIX_SEC = numpy.array([0, 0, 0, 0, 0], dtype=int)

SOURCE_OBJECT_TABLE = pandas.DataFrame.from_dict({
    tracking_utils.CENTROID_LATITUDE_COLUMN: SOURCE_LATITUDES_DEG,
    tracking_utils.CENTROID_LONGITUDE_COLUMN: SOURCE_LONGITUDES_DEG,
    tracking_utils.FULL_ID_COLUMN: SOURCE_ID_STRINGS,
    tracking_utils.VALID_TIME_COLUMN: SOURCE_TIMES_UNIX_SEC
})

TARGET_LATITUDES_DEG = numpy.array([55, 54, 53, 52, 51, 50, 49], dtype=float)
TARGET_LONGITUDES_DEG = numpy.array(
    [-119, -118, -117, -116, -115, -114, -113], dtype=float
)
TARGET_ID_STRINGS = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
TARGET_TIMES_UNIX_SEC = numpy.array([1, 1, 1, 1, 1, 1, 1], dtype=int)

TARGET_OBJECT_TABLE = pandas.DataFrame.from_dict({
    tracking_utils.CENTROID_LATITUDE_COLUMN: TARGET_LATITUDES_DEG,
    tracking_utils.CENTROID_LONGITUDE_COLUMN: TARGET_LONGITUDES_DEG,
    tracking_utils.FULL_ID_COLUMN: TARGET_ID_STRINGS,
    tracking_utils.VALID_TIME_COLUMN: TARGET_TIMES_UNIX_SEC
})

SOURCE_TO_TARGET_DICT = dict()
SOURCE_TO_TARGET_DICT[('a', 0)] = ['E', 1]
SOURCE_TO_TARGET_DICT[('b', 0)] = None
SOURCE_TO_TARGET_DICT[('c', 0)] = ['D', 1]
SOURCE_TO_TARGET_DICT[('d', 0)] = ['G', 1]
SOURCE_TO_TARGET_DICT[('e', 0)] = ['A', 1]


class MatchStormsTests(unittest.TestCase):
    """Each method is a unit test for match_storms_between_datasets.py."""

    def test_match_all_times_first(self):
        """Ensures correct output from _match_all_times.

        In this case, using first set of times.
        """

        these_indices = match_storms._match_all_times(
            source_times_unix_sec=FIRST_SOURCE_TIMES_UNIX_SEC,
            target_times_unix_sec=FIRST_TARGET_TIMES_UNIX_SEC,
            max_diff_seconds=MAX_TIME_DIFF_SECONDS)

        self.assertTrue(numpy.array_equal(
            these_indices, FIRST_TIME_MATCH_INDICES
        ))

    def test_match_all_times_second(self):
        """Ensures correct output from _match_all_times.

        In this case, using second set of times.
        """

        these_indices = match_storms._match_all_times(
            source_times_unix_sec=SECOND_SOURCE_TIMES_UNIX_SEC,
            target_times_unix_sec=SECOND_TARGET_TIMES_UNIX_SEC,
            max_diff_seconds=MAX_TIME_DIFF_SECONDS)

        self.assertTrue(numpy.array_equal(
            these_indices, SECOND_TIME_MATCH_INDICES
        ))

    def test_match_all_times_error(self):
        """Ensures correct output from _match_all_times.

        In this case the max allowed time difference is smaller than that found
        in the arrays, leading to a ValueError.
        """

        with self.assertRaises(ValueError):
            match_storms._match_all_times(
                source_times_unix_sec=SECOND_SOURCE_TIMES_UNIX_SEC,
                target_times_unix_sec=SECOND_TARGET_TIMES_UNIX_SEC,
                max_diff_seconds=1)

    def test_match_locations_one_time(self):
        """Ensures correct output from _match_locations_one_time."""

        this_source_to_target_dict = match_storms._match_locations_one_time(
            source_object_table=SOURCE_OBJECT_TABLE,
            target_object_table=TARGET_OBJECT_TABLE,
            max_distance_metres=MAX_DISTANCE_METRES)

        self.assertTrue(this_source_to_target_dict == SOURCE_TO_TARGET_DICT)


if __name__ == '__main__':
    unittest.main()
