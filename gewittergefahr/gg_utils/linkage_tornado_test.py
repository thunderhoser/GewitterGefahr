"""Unit tests for linkage.py.

This file specifically tests methods that deal with linking tornado occurrence
(not genesis) to storms.
"""

import copy
import unittest
import numpy
import pandas
from gewittergefahr.gg_io import tornado_io
from gewittergefahr.gg_utils import linkage
from gewittergefahr.gg_utils import linkage_test
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils

TOLERANCE = 1e-6
INTERP_TIME_INTERVAL_SEC = 1

STORM_OBJECT_TABLE = linkage_test.create_storm_objects()

THESE_START_TIMES_UNIX_SEC = numpy.array([1, 5, 5, 5, 60], dtype=int)
# THESE_END_TIMES_UNIX_SEC = numpy.array([11, 10, 12, 14, 60], dtype=int)

THESE_START_LATITUDES_DEG = numpy.array([59.5, 61, 51, 49, 89])
# THESE_END_LATITUDES_DEG = numpy.array([59.5, 66, 58, 58, 89])

THESE_START_LONGITUDES_DEG = numpy.array([271, 275, 242.5, 242.5, 300])
# THESE_END_LONGITUDES_DEG = numpy.array([281, 275, 242.5, 242.5, 300])

THESE_FUJITA_STRINGS = ['EF1', 'EF2', 'EF3', 'EF4', 'EF5']

THESE_TIMES_UNIX_SEC = numpy.array([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    5, 6, 7, 8, 9, 10,
    5, 6, 7, 8, 9, 10, 11, 12,
    5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
    60
], dtype=int)

THESE_LATITUDES_DEG = numpy.array([
    59.5, 59.5, 59.5, 59.5, 59.5, 59.5, 59.5, 59.5, 59.5, 59.5, 59.5,
    61, 62, 63, 64, 65, 66,
    51, 52, 53, 54, 55, 56, 57, 58,
    49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    89
])

THESE_LONGITUDES_DEG = numpy.array([
    271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281,
    275, 275, 275, 275, 275, 275,
    242.5, 242.5, 242.5, 242.5, 242.5, 242.5, 242.5, 242.5,
    242.5, 242.5, 242.5, 242.5, 242.5, 242.5, 242.5, 242.5, 242.5, 242.5,
    300
])

THESE_UNIQUE_ID_STRINGS = [
    tornado_io.create_tornado_id(
        start_time_unix_sec=t, start_latitude_deg=y, start_longitude_deg=x
    ) for t, x, y in
    zip(THESE_START_TIMES_UNIX_SEC, THESE_START_LONGITUDES_DEG,
        THESE_START_LATITUDES_DEG)
]

THESE_ID_STRINGS = (
    [THESE_UNIQUE_ID_STRINGS[0]] * 11 + [THESE_UNIQUE_ID_STRINGS[1]] * 6 +
    [THESE_UNIQUE_ID_STRINGS[2]] * 8 + [THESE_UNIQUE_ID_STRINGS[3]] * 10 +
    [THESE_UNIQUE_ID_STRINGS[4]] * 1
)

THESE_FUJITA_STRINGS = (
    [THESE_FUJITA_STRINGS[0]] * 11 + [THESE_FUJITA_STRINGS[1]] * 6 +
    [THESE_FUJITA_STRINGS[2]] * 8 + [THESE_FUJITA_STRINGS[3]] * 10 +
    [THESE_FUJITA_STRINGS[4]] * 1
)

TORNADO_TABLE = pandas.DataFrame.from_dict({
    linkage.EVENT_TIME_COLUMN: THESE_TIMES_UNIX_SEC,
    linkage.EVENT_LATITUDE_COLUMN: THESE_LATITUDES_DEG,
    linkage.EVENT_LONGITUDE_COLUMN: THESE_LONGITUDES_DEG,
    linkage.EVENT_Y_COLUMN: THESE_LATITUDES_DEG,
    linkage.EVENT_X_COLUMN: THESE_LONGITUDES_DEG,
    tornado_io.TORNADO_ID_COLUMN: THESE_ID_STRINGS,
    tornado_io.FUJITA_RATING_COLUMN: THESE_FUJITA_STRINGS
})

# The following constants are used to test _find_nearest_storms.
MAX_EXTRAPOLATION_TIME_SEC = 2
MAX_LINK_DISTANCE_METRES = 12.

THESE_SECONDARY_ID_STRINGS = [
    'B2', 'B3', 'B5', 'B5', 'B5', 'B5', 'B5', 'B5', None, None, None,
    'B4', 'B4', 'B4', 'B4', None, None,
    'A2', 'A2', 'A2', 'A2', 'A4', 'A4', 'A4', 'A4',
    'A3', 'A3', 'A3', 'A3', 'A4', 'A4', 'A4', 'A4', 'A4', None,
    None
]

THESE_TIMES_UNIX_SEC = numpy.array([
    1, 2, 3, 4, 5, 6, 7, 8, -1, -1, -1,
    5, 6, 7, 8, -1, -1,
    5, 6, 7, 8, 9, 10, 11, 12,
    5, 6, 7, 8, 9, 10, 11, 12, 13, -1,
    -1
], dtype=int)

# TODO(thunderhoser): Fix distances.
THESE_DISTANCES_METRES = numpy.full(len(THESE_TIMES_UNIX_SEC), 0.)
THESE_DISTANCES_METRES[THESE_TIMES_UNIX_SEC == -1] = numpy.nan

TORNADO_TO_STORM_TABLE = TORNADO_TABLE.assign(**{
    linkage.NEAREST_SECONDARY_ID_COLUMN: THESE_SECONDARY_ID_STRINGS,
    linkage.NEAREST_TIME_COLUMN: THESE_TIMES_UNIX_SEC,
    linkage.LINKAGE_DISTANCE_COLUMN: THESE_DISTANCES_METRES
})

# The following constants are used to test _reverse_tornado_linkages.
STORM_TO_TORNADOES_TABLE = copy.deepcopy(STORM_OBJECT_TABLE)

THIS_NESTED_ARRAY = STORM_TO_TORNADOES_TABLE[[
    tracking_utils.VALID_TIME_COLUMN, tracking_utils.VALID_TIME_COLUMN
]].values.tolist()

THESE_COLUMNS = [
    linkage.EVENT_LATITUDES_COLUMN, linkage.EVENT_LONGITUDES_COLUMN,
    linkage.FUJITA_RATINGS_COLUMN, linkage.TORNADO_IDS_COLUMN,
    linkage.LINKAGE_DISTANCES_COLUMN, linkage.RELATIVE_EVENT_TIMES_COLUMN,
    linkage.MAIN_OBJECT_FLAGS_COLUMN
]

NUM_STORM_OBJECTS = len(STORM_TO_TORNADOES_TABLE.index)

for this_column in THESE_COLUMNS:
    STORM_TO_TORNADOES_TABLE = STORM_TO_TORNADOES_TABLE.assign(
        **{this_column: THIS_NESTED_ARRAY}
    )

    for k in range(NUM_STORM_OBJECTS):
        if this_column in [linkage.FUJITA_RATINGS_COLUMN,
                           linkage.TORNADO_IDS_COLUMN]:
            STORM_TO_TORNADOES_TABLE[this_column].values[k] = []
        elif this_column == linkage.RELATIVE_EVENT_TIMES_COLUMN:
            STORM_TO_TORNADOES_TABLE[this_column].values[k] = numpy.array(
                [], dtype=int
            )
        elif this_column == linkage.MAIN_OBJECT_FLAGS_COLUMN:
            STORM_TO_TORNADOES_TABLE[this_column].values[k] = numpy.array(
                [], dtype=bool
            )
        else:
            STORM_TO_TORNADOES_TABLE[this_column].values[k] = numpy.array([])

STORM_ROW_TO_TORNADO_ROWS = {
    0: [17, 18, 19, 20, 25, 26, 27, 28],
    2: [0],
    3: [17, 18, 19, 20, 25, 26, 27, 28],
    4: [0],
    5: [17, 18, 19, 20, 25, 26, 27, 28],
    6: [1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14],
    7: [17, 18, 19, 20, 25, 26, 27, 28],
    8: [11, 12, 13, 14],
    9: [17, 18, 19, 20],
    10: [25, 26, 27, 28],
    11: [11, 12, 13, 14],
    12: [4, 5, 6, 7],
    13: [18, 19, 20],
    14: [12, 13, 14],
    15: [5, 6, 7],
    16: [19, 20],
    17: [27, 28],
    18: [22, 23, 24, 30, 31, 32, 33],
    19: [23, 24, 31, 32, 33]
}

STORM_ROW_TO_MAIN_OBJECT_FLAGS = {
    0: [0, 0, 0, 0, 0, 0, 0, 0],
    2: [0],
    3: [0, 0, 0, 0, 0, 0, 0, 0],
    4: [1],
    5: [0, 0, 0, 0, 0, 0, 0, 0],
    6: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    7: [0, 0, 0, 0, 0, 0, 0, 0],
    8: [0, 0, 0, 0],
    9: [1, 0, 0, 0],
    10: [1, 1, 0, 0],
    11: [1, 0, 0, 0],
    12: [1, 0, 0, 0],
    13: [1, 0, 0],
    14: [1, 1, 1],
    15: [1, 1, 1],
    16: [1, 1],
    17: [1, 1],
    18: [1, 0, 0, 1, 0, 0, 0],
    19: [1, 1, 1, 1, 1]
}

for this_storm_row in STORM_ROW_TO_TORNADO_ROWS:
    these_tornado_rows = numpy.array(
        STORM_ROW_TO_TORNADO_ROWS[this_storm_row], dtype=int
    )

    STORM_TO_TORNADOES_TABLE[linkage.EVENT_LATITUDES_COLUMN].values[
        this_storm_row
    ] = TORNADO_TO_STORM_TABLE[linkage.EVENT_LATITUDE_COLUMN].values[
        these_tornado_rows]

    STORM_TO_TORNADOES_TABLE[linkage.EVENT_LONGITUDES_COLUMN].values[
        this_storm_row
    ] = TORNADO_TO_STORM_TABLE[linkage.EVENT_LONGITUDE_COLUMN].values[
        these_tornado_rows]

    STORM_TO_TORNADOES_TABLE[linkage.LINKAGE_DISTANCES_COLUMN].values[
        this_storm_row
    ] = TORNADO_TO_STORM_TABLE[linkage.LINKAGE_DISTANCE_COLUMN].values[
        these_tornado_rows]

    STORM_TO_TORNADOES_TABLE[linkage.FUJITA_RATINGS_COLUMN].values[
        this_storm_row
    ] = TORNADO_TO_STORM_TABLE[tornado_io.FUJITA_RATING_COLUMN].values[
        these_tornado_rows
    ].tolist()

    STORM_TO_TORNADOES_TABLE[linkage.TORNADO_IDS_COLUMN].values[
        this_storm_row
    ] = TORNADO_TO_STORM_TABLE[tornado_io.TORNADO_ID_COLUMN].values[
        these_tornado_rows
    ].tolist()

    these_relative_times_sec = (
        TORNADO_TO_STORM_TABLE[linkage.EVENT_TIME_COLUMN].values[
            these_tornado_rows] -
        STORM_TO_TORNADOES_TABLE[tracking_utils.VALID_TIME_COLUMN].values[
            this_storm_row]
    )

    STORM_TO_TORNADOES_TABLE[linkage.RELATIVE_EVENT_TIMES_COLUMN].values[
        this_storm_row
    ] = these_relative_times_sec

    these_main_object_flags = numpy.array(
        STORM_ROW_TO_MAIN_OBJECT_FLAGS[this_storm_row], dtype=bool
    )

    STORM_TO_TORNADOES_TABLE[linkage.MAIN_OBJECT_FLAGS_COLUMN].values[
        this_storm_row
    ] = these_main_object_flags

# The following constants are used to test _remove_redundant_tornado_linkages
# and _share_tornado_linkages.
EARLY_STORM_OBJECT_TABLE = STORM_OBJECT_TABLE.loc[
    STORM_OBJECT_TABLE[tracking_utils.VALID_TIME_COLUMN] <= 5
]

EARLY_TORNADO_TABLE = TORNADO_TABLE.loc[
    TORNADO_TABLE[linkage.EVENT_TIME_COLUMN] < 60
]

LATE_STORM_OBJECT_TABLE = STORM_OBJECT_TABLE.loc[
    STORM_OBJECT_TABLE[tracking_utils.VALID_TIME_COLUMN] > 5
]

LATE_TORNADO_TABLE = TORNADO_TABLE.loc[
    TORNADO_TABLE[linkage.EVENT_TIME_COLUMN] <= 60
]


def _sort_tornadoes_for_each_storm(storm_to_tornadoes_table):
    """Sorts tornadoes (by ID, then time) for each storm.

    :param storm_to_tornadoes_table: pandas DataFrame created by
        `linkage._reverse_tornado_linkages`.
    :return: storm_to_tornadoes_table: Same but with tornadoes sorted for each
        storm.
    """

    columns_to_sort = [
        linkage.EVENT_LATITUDES_COLUMN, linkage.EVENT_LONGITUDES_COLUMN,
        linkage.FUJITA_RATINGS_COLUMN, linkage.TORNADO_IDS_COLUMN,
        linkage.LINKAGE_DISTANCES_COLUMN, linkage.RELATIVE_EVENT_TIMES_COLUMN,
        linkage.MAIN_OBJECT_FLAGS_COLUMN
    ]

    num_storm_objects = len(storm_to_tornadoes_table.index)

    for i in range(num_storm_objects):
        these_tornado_id_strings = storm_to_tornadoes_table[
            linkage.TORNADO_IDS_COLUMN].values[i]
        these_relative_times_sec = storm_to_tornadoes_table[
            linkage.RELATIVE_EVENT_TIMES_COLUMN].values[i]

        these_strings = numpy.array([
            '{0:s}_{1:d}'.format(s, t) for s, t in
            zip(these_tornado_id_strings, these_relative_times_sec)
        ])

        these_sort_indices = numpy.argsort(these_strings)

        for this_column in columns_to_sort:
            this_array = storm_to_tornadoes_table[this_column].values[i]

            if isinstance(this_array, numpy.ndarray):
                storm_to_tornadoes_table[this_column].values[i] = (
                    this_array[these_sort_indices]
                )
            else:
                storm_to_tornadoes_table[this_column].values[i] = [
                    this_array[k] for k in these_sort_indices
                ]

    return storm_to_tornadoes_table


class LinkageTests(unittest.TestCase):
    """Each method is a unit test for linkage.py."""

    def test_find_nearest_storms(self):
        """Ensures correct output from _find_nearest_storms."""

        this_tornado_to_storm_table = linkage._find_nearest_storms(
            storm_object_table=STORM_OBJECT_TABLE,
            event_table=TORNADO_TABLE,
            max_time_before_storm_start_sec=MAX_EXTRAPOLATION_TIME_SEC,
            max_time_after_storm_end_sec=MAX_EXTRAPOLATION_TIME_SEC,
            max_link_distance_metres=MAX_LINK_DISTANCE_METRES,
            interp_time_interval_sec=INTERP_TIME_INTERVAL_SEC,
            event_type_string=linkage.TORNADO_EVENT_STRING)

        self.assertTrue(
            this_tornado_to_storm_table[
                linkage.NEAREST_SECONDARY_ID_COLUMN].values.tolist() ==
            TORNADO_TO_STORM_TABLE[
                linkage.NEAREST_SECONDARY_ID_COLUMN].values.tolist()
        )

        self.assertTrue(numpy.array_equal(
            this_tornado_to_storm_table[linkage.NEAREST_TIME_COLUMN].values,
            TORNADO_TO_STORM_TABLE[linkage.NEAREST_TIME_COLUMN].values
        ))

    def test_reverse_tornado_linkages(self):
        """Ensures correct output from _reverse_tornado_linkages."""

        this_storm_to_tornadoes_table = linkage._reverse_tornado_linkages(
            storm_object_table=STORM_OBJECT_TABLE,
            tornado_to_storm_table=TORNADO_TO_STORM_TABLE)

        self.assertTrue(linkage_test.compare_storm_to_events_tables(
            this_storm_to_tornadoes_table, STORM_TO_TORNADOES_TABLE
        ))

    def test_remove_redundant_tornado_linkages(self):
        """Ensures correct output from _remove_redundant_tornado_linkages."""

        early_tornado_to_storm_table = linkage._find_nearest_storms(
            storm_object_table=EARLY_STORM_OBJECT_TABLE,
            event_table=EARLY_TORNADO_TABLE,
            max_time_before_storm_start_sec=MAX_EXTRAPOLATION_TIME_SEC,
            max_time_after_storm_end_sec=MAX_EXTRAPOLATION_TIME_SEC,
            max_link_distance_metres=MAX_LINK_DISTANCE_METRES,
            interp_time_interval_sec=INTERP_TIME_INTERVAL_SEC,
            event_type_string=linkage.TORNADO_EVENT_STRING)

        late_tornado_to_storm_table = linkage._find_nearest_storms(
            storm_object_table=LATE_STORM_OBJECT_TABLE,
            event_table=LATE_TORNADO_TABLE,
            max_time_before_storm_start_sec=MAX_EXTRAPOLATION_TIME_SEC,
            max_time_after_storm_end_sec=MAX_EXTRAPOLATION_TIME_SEC,
            max_link_distance_metres=MAX_LINK_DISTANCE_METRES,
            interp_time_interval_sec=INTERP_TIME_INTERVAL_SEC,
            event_type_string=linkage.TORNADO_EVENT_STRING)

        early_tornado_to_storm_table, late_tornado_to_storm_table = (
            linkage._remove_redundant_tornado_linkages(
                early_tornado_to_storm_table=early_tornado_to_storm_table,
                late_tornado_to_storm_table=late_tornado_to_storm_table)
        )

        tornado_to_storm_table = pandas.concat(
            [early_tornado_to_storm_table, late_tornado_to_storm_table],
            axis=0, ignore_index=True)

        # These assertions only make sure that redundant tornado segments
        # ("events") have been removed.  `test_share_tornado_linkages` makes
        # sure that this leads to linkages being shared properly.

        self.assertTrue(numpy.array_equal(
            tornado_to_storm_table[tornado_io.TORNADO_ID_COLUMN].values,
            TORNADO_TABLE[tornado_io.TORNADO_ID_COLUMN].values
        ))

        self.assertTrue(numpy.array_equal(
            tornado_to_storm_table[linkage.EVENT_TIME_COLUMN].values,
            TORNADO_TABLE[linkage.EVENT_TIME_COLUMN].values
        ))

    def test_share_tornado_linkages(self):
        """Ensures correct output from _share_tornado_linkages."""

        early_tornado_to_storm_table = linkage._find_nearest_storms(
            storm_object_table=EARLY_STORM_OBJECT_TABLE,
            event_table=EARLY_TORNADO_TABLE,
            max_time_before_storm_start_sec=MAX_EXTRAPOLATION_TIME_SEC,
            max_time_after_storm_end_sec=MAX_EXTRAPOLATION_TIME_SEC,
            max_link_distance_metres=MAX_LINK_DISTANCE_METRES,
            interp_time_interval_sec=INTERP_TIME_INTERVAL_SEC,
            event_type_string=linkage.TORNADO_EVENT_STRING)

        late_tornado_to_storm_table = linkage._find_nearest_storms(
            storm_object_table=LATE_STORM_OBJECT_TABLE,
            event_table=LATE_TORNADO_TABLE,
            max_time_before_storm_start_sec=MAX_EXTRAPOLATION_TIME_SEC,
            max_time_after_storm_end_sec=MAX_EXTRAPOLATION_TIME_SEC,
            max_link_distance_metres=MAX_LINK_DISTANCE_METRES,
            interp_time_interval_sec=INTERP_TIME_INTERVAL_SEC,
            event_type_string=linkage.TORNADO_EVENT_STRING)

        early_storm_to_tornadoes_table, late_storm_to_tornadoes_table = (
            linkage._share_tornado_linkages(
                early_tornado_to_storm_table=early_tornado_to_storm_table,
                late_tornado_to_storm_table=late_tornado_to_storm_table,
                early_storm_object_table=STORM_OBJECT_TABLE,
                late_storm_object_table=STORM_OBJECT_TABLE,
                max_time_before_storm_start_sec=MAX_EXTRAPOLATION_TIME_SEC,
                max_time_after_storm_end_sec=MAX_EXTRAPOLATION_TIME_SEC)
        )

        this_actual_table = pandas.concat(
            [early_storm_to_tornadoes_table, late_storm_to_tornadoes_table],
            axis=0, ignore_index=True)

        this_actual_table = _sort_tornadoes_for_each_storm(this_actual_table)
        this_expected_table = _sort_tornadoes_for_each_storm(
            copy.deepcopy(STORM_TO_TORNADOES_TABLE)
        )

        self.assertTrue(linkage_test.compare_storm_to_events_tables(
            this_actual_table, this_expected_table
        ))


if __name__ == '__main__':
    unittest.main()
