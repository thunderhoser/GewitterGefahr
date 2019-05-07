"""Unit tests for storm_tracking_utils.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils

TOLERANCE = 1e-6

MIN_BUFFER_DISTANCE_METRES = 0.
MAX_BUFFER_DISTANCE_METRES = 5000.
BUFFER_COLUMN_NAME_INCLUSIVE = 'polygon_object_latlng_deg_buffer_5000m'
BUFFER_COLUMN_NAME_EXCLUSIVE = 'polygon_object_latlng_deg_buffer_0m_5000m'
BUFFER_COLUMN_NAME_FAKE = 'foobar'

# The following constants are used to test find_storm_objects.
ALL_STORM_ID_STRINGS = ['a', 'b', 'c', 'd', 'a', 'c', 'e', 'f', 'e']
ALL_TIMES_UNIX_SEC = numpy.array([0, 0, 0, 0, 1, 1, 1, 1, 2], dtype=int)

KEPT_ID_STRINGS_0MISSING = ['a', 'c', 'a', 'e', 'e', 'e']
KEPT_TIMES_UNIX_SEC_0MISSING = numpy.array([0, 0, 1, 1, 2, 1], dtype=int)
RELEVANT_INDICES_0MISSING = numpy.array([0, 2, 4, 6, 8, 6], dtype=int)

KEPT_ID_STRINGS_1MISSING = ['a', 'c', 'a', 'e', 'e', 'e', 'a']
KEPT_TIMES_UNIX_SEC_1MISSING = numpy.array([0, 0, 1, 1, 2, 1, 2], dtype=int)
RELEVANT_INDICES_1MISSING = numpy.array([0, 2, 4, 6, 8, 6, -1], dtype=int)


class StormTrackingUtilsTests(unittest.TestCase):
    """Each method is a unit test for storm_tracking_utils.py."""

    def test_column_name_to_buffer_inclusive(self):
        """Ensures correct output from column_name_to_buffer.

        In this case the distance buffer includes the storm object.
        """

        this_min_distance_metres, this_max_distance_metres = (
            tracking_utils.column_name_to_buffer(BUFFER_COLUMN_NAME_INCLUSIVE)
        )

        self.assertTrue(numpy.isnan(this_min_distance_metres))
        self.assertTrue(numpy.isclose(
            this_max_distance_metres, MAX_BUFFER_DISTANCE_METRES, atol=TOLERANCE
        ))

    def test_column_name_to_buffer_exclusive(self):
        """Ensures correct output from column_name_to_buffer.

        In this case the distance buffer does not include the storm object.
        """

        this_min_distance_metres, this_max_distance_metres = (
            tracking_utils.column_name_to_buffer(BUFFER_COLUMN_NAME_EXCLUSIVE)
        )

        self.assertTrue(numpy.isclose(
            this_min_distance_metres, MIN_BUFFER_DISTANCE_METRES, atol=TOLERANCE
        ))

        self.assertTrue(numpy.isclose(
            this_max_distance_metres, MAX_BUFFER_DISTANCE_METRES, atol=TOLERANCE
        ))

    def test_column_name_to_buffer_fake(self):
        """Ensures correct output from column_name_to_buffer.

        In this case the column name is malformatted.
        """

        this_min_distance_metres, this_max_distance_metres = (
            tracking_utils.column_name_to_buffer(BUFFER_COLUMN_NAME_FAKE)
        )

        self.assertTrue(this_min_distance_metres is None)
        self.assertTrue(this_max_distance_metres is None)

    def test_buffer_to_column_name_inclusive(self):
        """Ensures correct output from buffer_to_column_name.

        In this case the distance buffer includes the storm object.
        """

        this_column_name = tracking_utils.buffer_to_column_name(
            min_distance_metres=numpy.nan,
            max_distance_metres=MAX_BUFFER_DISTANCE_METRES)

        self.assertTrue(this_column_name == BUFFER_COLUMN_NAME_INCLUSIVE)

    def test_buffer_to_column_name_exclusive(self):
        """Ensures correct output from buffer_to_column_name.

        In this case the distance buffer does not include the storm object.
        """

        this_column_name = tracking_utils.buffer_to_column_name(
            min_distance_metres=MIN_BUFFER_DISTANCE_METRES,
            max_distance_metres=MAX_BUFFER_DISTANCE_METRES)

        self.assertTrue(this_column_name == BUFFER_COLUMN_NAME_EXCLUSIVE)

    def test_find_storm_objects_0missing(self):
        """Ensures correct output from find_storm_objects.

        In this case, no desired storm objects are missing.
        """

        these_indices = tracking_utils.find_storm_objects(
            all_id_strings=ALL_STORM_ID_STRINGS,
            all_times_unix_sec=ALL_TIMES_UNIX_SEC,
            id_strings_to_keep=KEPT_ID_STRINGS_0MISSING,
            times_to_keep_unix_sec=KEPT_TIMES_UNIX_SEC_0MISSING,
            allow_missing=False)

        self.assertTrue(numpy.array_equal(
            these_indices, RELEVANT_INDICES_0MISSING
        ))

    def test_find_storm_objects_allow_missing_false(self):
        """Ensures correct output from find_storm_objects.

        In this case, one desired storm object is missing and
        `allow_missing = False`.
        """

        with self.assertRaises(ValueError):
            tracking_utils.find_storm_objects(
                all_id_strings=ALL_STORM_ID_STRINGS,
                all_times_unix_sec=ALL_TIMES_UNIX_SEC,
                id_strings_to_keep=KEPT_ID_STRINGS_1MISSING,
                times_to_keep_unix_sec=KEPT_TIMES_UNIX_SEC_1MISSING,
                allow_missing=False)

    def test_find_storm_objects_allow_missing_true(self):
        """Ensures correct output from find_storm_objects.

        In this case, one desired storm object is missing and
        `allow_missing = True`.
        """

        these_indices = tracking_utils.find_storm_objects(
            all_id_strings=ALL_STORM_ID_STRINGS,
            all_times_unix_sec=ALL_TIMES_UNIX_SEC,
            id_strings_to_keep=KEPT_ID_STRINGS_1MISSING,
            times_to_keep_unix_sec=KEPT_TIMES_UNIX_SEC_1MISSING,
            allow_missing=True)

        self.assertTrue(numpy.array_equal(
            these_indices, RELEVANT_INDICES_1MISSING
        ))


if __name__ == '__main__':
    unittest.main()
