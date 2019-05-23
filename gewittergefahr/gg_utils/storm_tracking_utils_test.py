"""Unit tests for storm_tracking_utils.py."""

import unittest
import numpy
import pandas
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

# The following constants are used to test storm_objects_to_tracks.
THESE_ID_STRINGS = [
    'foo', 'bar', 'hal', 'foo', 'bar', 'moo', 'empty', 'foo', 'moo', 'empty'
]
THESE_TIMES_UNIX_SEC = numpy.array(
    [0, 0, 0, 300, 300, 300, 300, 600, 600, 600], dtype=int
)
THESE_X_COORDS_METRES = numpy.array(
    [10, 0, 20, 11, 1, 30, numpy.nan, 12, 31, numpy.nan]
)
THESE_Y_COORDS_METRES = numpy.array(
    [100, 0, 200, 105, 5, 300, numpy.nan, 110, 305, numpy.nan]
)

THIS_DICT = {
    tracking_utils.PRIMARY_ID_COLUMN: THESE_ID_STRINGS,
    tracking_utils.VALID_TIME_COLUMN: THESE_TIMES_UNIX_SEC,
    tracking_utils.CENTROID_LATITUDE_COLUMN: THESE_Y_COORDS_METRES,
    tracking_utils.CENTROID_LONGITUDE_COLUMN: THESE_X_COORDS_METRES,
    tracking_utils.CENTROID_X_COLUMN: THESE_X_COORDS_METRES,
    tracking_utils.CENTROID_Y_COLUMN: THESE_Y_COORDS_METRES
}

STORM_OBJECT_TABLE = pandas.DataFrame.from_dict(THIS_DICT)

THESE_ID_STRINGS = ['bar', 'empty', 'foo', 'hal', 'moo']

THIS_DICT = {
    tracking_utils.PRIMARY_ID_COLUMN: THESE_ID_STRINGS,
    tracking_utils.TRACK_TIMES_COLUMN:
        [[0, 300], [300, 600], [0, 300, 600], [0], [300, 600]],
    tracking_utils.OBJECT_INDICES_COLUMN:
        [[1, 4], [6, 9], [0, 3, 7], [2], [5, 8]],
    tracking_utils.TRACK_X_COORDS_COLUMN:
        [[0, 1], [numpy.nan, numpy.nan], [10, 11, 12], [20], [30, 31]],
    tracking_utils.TRACK_Y_COORDS_COLUMN:
        [[0, 5], [numpy.nan, numpy.nan], [100, 105, 110], [200], [300, 305]]
}

STORM_TRACK_TABLE = pandas.DataFrame.from_dict(THIS_DICT)

STORM_TRACK_TABLE[tracking_utils.TRACK_LATITUDES_COLUMN] = STORM_TRACK_TABLE[
    tracking_utils.TRACK_Y_COORDS_COLUMN]
STORM_TRACK_TABLE[tracking_utils.TRACK_LONGITUDES_COLUMN] = STORM_TRACK_TABLE[
    tracking_utils.TRACK_X_COORDS_COLUMN]


def _compare_storm_track_tables(
        first_storm_track_table, second_storm_track_table):
    """Compares two tables with storm tracks.

    :param first_storm_track_table: First table (pandas DataFrame).
    :param second_storm_track_table: Second table.
    :return: are_tables_equal: Boolean flag.
    """

    first_column_names = list(first_storm_track_table)
    second_column_names = list(second_storm_track_table)
    if set(first_column_names) != set(second_column_names):
        return False

    first_num_tracks = len(first_storm_track_table.index)
    second_num_tracks = len(first_storm_track_table.index)
    if first_num_tracks != second_num_tracks:
        return False

    for this_column in first_column_names:
        if this_column == tracking_utils.PRIMARY_ID_COLUMN:
            if not numpy.array_equal(
                    first_storm_track_table[this_column].values,
                    second_storm_track_table[this_column].values):
                return False

        else:
            for i in range(first_num_tracks):
                if not numpy.allclose(
                        first_storm_track_table[this_column].values[i],
                        second_storm_track_table[this_column].values[i],
                        atol=TOLERANCE, equal_nan=True):
                    return False

    return True


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

    def test_storm_objects_to_tracks(self):
        """Ensures correct output from storm_objects_to_tracks."""

        this_storm_track_table = tracking_utils.storm_objects_to_tracks(
            STORM_OBJECT_TABLE)

        this_storm_track_table.sort_values(
            tracking_utils.PRIMARY_ID_COLUMN, axis=0, ascending=True,
            inplace=True)

        self.assertTrue(_compare_storm_track_tables(
            this_storm_track_table, STORM_TRACK_TABLE
        ))


if __name__ == '__main__':
    unittest.main()
