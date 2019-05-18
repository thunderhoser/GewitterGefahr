"""Additional unit tests for temporal_tracking.py.

This file specifically tests methods that find predecessors and successors.  I
created this file, rather than putting all the unit tests in
temporal_tracking_test.py, because temporal_tracking_test.py was getting
unwieldy.
"""

import unittest
import numpy
import pandas
from gewittergefahr.gg_utils import temporal_tracking
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils

THESE_TIMES_UNIX_SEC = numpy.array([
    0, 0, 0,
    1, 1,
    2, 2,
    4, 4,
    5, 5, 5, 5,
    6, 6, 6,
    7, 7,
    10,
    11
])

THESE_SECONDARY_ID_STRINGS = [
    'A1', 'B1', 'B2',
    'A1', 'B2',
    'A1', 'B3',
    'A1', 'B4',
    'A2', 'A3', 'B4', 'B5',
    'A2', 'B4', 'B5',
    'A2', 'A3',
    'A4',
    'A4'
]

THESE_FIRST_PREV_SEC_ID_STRINGS = [
    '', '', '',
    'A1', 'B2',
    'A1', 'B1',
    'A1', 'B3',
    'A1', 'A1', 'B4', 'B3',
    'A2', 'B4', 'B5',
    'A2', 'A3',
    'A2',
    'A4'
]

THESE_SECOND_PREV_SEC_ID_STRINGS = [
    '', '', '',
    '', '',
    '', 'B2',
    '', '',
    '', '', '', '',
    '', '', '',
    '', '',
    'A3',
    ''
]

THESE_FIRST_NEXT_SEC_ID_STRINGS = [
    'A1', 'B3', 'B2',
    'A1', 'B3',
    'A1', 'B4',
    'A2', 'B4',
    'A2', 'A3', 'B4', 'B5',
    'A2', '', '',
    'A4', 'A4',
    'A4',
    ''
]

THESE_SECOND_NEXT_SEC_ID_STRINGS = [
    '', '', '',
    '', '',
    '', 'B5',
    'A3', '',
    '', '', '', '',
    '', '', '',
    '', '',
    '',
    ''
]

STORM_OBJECT_TABLE = pandas.DataFrame.from_dict({
    tracking_utils.SECONDARY_ID_COLUMN: THESE_SECONDARY_ID_STRINGS,
    tracking_utils.VALID_TIME_COLUMN: THESE_TIMES_UNIX_SEC,
    tracking_utils.FIRST_PREV_SECONDARY_ID_COLUMN:
        THESE_FIRST_PREV_SEC_ID_STRINGS,
    tracking_utils.SECOND_PREV_SECONDARY_ID_COLUMN:
        THESE_SECOND_PREV_SEC_ID_STRINGS,
    tracking_utils.FIRST_NEXT_SECONDARY_ID_COLUMN:
        THESE_FIRST_NEXT_SEC_ID_STRINGS,
    tracking_utils.SECOND_NEXT_SECONDARY_ID_COLUMN:
        THESE_SECOND_NEXT_SEC_ID_STRINGS
})

ALL_PREDECESSOR_DICT_5SECONDS = {
    0: [], 1: [], 2: [],
    3: [0], 4: [2],
    5: [0], 6: [1, 2],
    7: [0], 8: [1, 2],
    9: [0], 10: [0], 11: [1, 2], 12: [1, 2],
    13: [3], 14: [4], 15: [4],
    16: [5], 17: [5],
    18: [9, 10],
    19: [13, 17]
}

# THESE_SECONDARY_ID_STRINGS = [
#     'A1', 'B1', 'B2',          time 0
#     'A1', 'B2',                time 1
#     'A1', 'B3',                time 2
#     'A1', 'B4',                time 4
#     'A2', 'A3', 'B4', 'B5',    time 5
#     'A2', 'B4', 'B5',          time 6
#     'A2', 'A3',                time 7
#     'A4',                      time 10
#     'A4'                       time 11
# ]

ALL_PREDECESSOR_DICT_2SECONDS = {
    0: [], 1: [], 2: [],
    3: [0], 4: [2],
    5: [0], 6: [1, 2],
    7: [5], 8: [6],
    9: [7], 10: [7], 11: [8], 12: [],
    13: [7], 14: [8], 15: [12],
    16: [9], 17: [10],
    18: [],
    19: [18]
}

IMMED_PREDECESSOR_DICT_5SECONDS = {
    0: [], 1: [], 2: [],
    3: [0], 4: [2],
    5: [3], 6: [1, 4],
    7: [5], 8: [6],
    9: [7], 10: [7], 11: [8], 12: [6],
    13: [9], 14: [11], 15: [12],
    16: [13], 17: [10],
    18: [16, 17],
    19: [18]
}

IMMED_SUCCESSOR_DICT_5SECONDS = {
    0: [3], 1: [6], 2: [4],
    3: [5], 4: [6],
    5: [7], 6: [8, 12],
    7: [9, 10], 8: [11],
    9: [13], 10: [17], 11: [14], 12: [15],
    13: [16], 14: [], 15: [],
    16: [18], 17: [18],
    18: [19],
    19: []
}

IMMED_PREDECESSOR_DICT_2SECONDS = {
    0: [], 1: [], 2: [],
    3: [0], 4: [2],
    5: [3], 6: [1, 4],
    7: [5], 8: [6],
    9: [7], 10: [7], 11: [8], 12: [],
    13: [9], 14: [11], 15: [12],
    16: [13], 17: [10],
    18: [],
    19: [18]
}

IMMED_SUCCESSOR_DICT_2SECONDS = {
    0: [3], 1: [6], 2: [4],
    3: [5], 4: [6],
    5: [7], 6: [8],
    7: [9, 10], 8: [11],
    9: [13], 10: [17], 11: [14], 12: [15],
    13: [16], 14: [], 15: [],
    16: [], 17: [],
    18: [19],
    19: []
}

IMMED_PREDECESSOR_DICT_1SECOND = {
    0: [], 1: [], 2: [],
    3: [0], 4: [2],
    5: [3], 6: [4],
    7: [], 8: [],
    9: [7], 10: [7], 11: [8], 12: [],
    13: [9], 14: [11], 15: [12],
    16: [13], 17: [],
    18: [],
    19: [18]
}

IMMED_SUCCESSOR_DICT_1SECOND = {
    0: [3], 1: [], 2: [4],
    3: [5], 4: [6],
    5: [], 6: [],
    7: [9, 10], 8: [11],
    9: [13], 10: [], 11: [14], 12: [15],
    13: [16], 14: [], 15: [],
    16: [], 17: [],
    18: [19],
    19: []
}


class TemporalTrackingPredsuccTests(unittest.TestCase):
    """Each method is a unit test for temporal_tracking.py."""

    def test_find_immediate_predecessors_5sec(self):
        """Ensures correct output from find_immediate_predecessors.

        In this case, max time difference = 5 seconds.
        """

        this_num_storm_objects = len(STORM_OBJECT_TABLE.index)

        for i in range(this_num_storm_objects):
            these_predecessor_rows = (
                temporal_tracking.find_immediate_predecessors(
                    storm_object_table=STORM_OBJECT_TABLE,
                    target_row=i, max_time_diff_seconds=5)
            )

            these_predecessor_rows = numpy.sort(these_predecessor_rows)
            these_expected_rows = numpy.sort(numpy.array(
                IMMED_PREDECESSOR_DICT_5SECONDS[i], dtype=int
            ))

            self.assertTrue(numpy.array_equal(
                these_predecessor_rows, these_expected_rows
            ))

    def test_find_immediate_successors_5sec(self):
        """Ensures correct output from find_immediate_successors.

        In this case, max time difference = 5 seconds.
        """

        this_num_storm_objects = len(STORM_OBJECT_TABLE.index)

        for i in range(this_num_storm_objects):
            these_successor_rows = temporal_tracking.find_immediate_successors(
                storm_object_table=STORM_OBJECT_TABLE, target_row=i,
                max_time_diff_seconds=5)

            these_successor_rows = numpy.sort(these_successor_rows)
            these_expected_rows = numpy.sort(numpy.array(
                IMMED_SUCCESSOR_DICT_5SECONDS[i], dtype=int
            ))

            self.assertTrue(numpy.array_equal(
                these_successor_rows, these_expected_rows
            ))

    def test_find_immediate_predecessors_2sec(self):
        """Ensures correct output from find_immediate_predecessors.

        In this case, max time difference = 2 seconds.
        """

        this_num_storm_objects = len(STORM_OBJECT_TABLE.index)

        for i in range(this_num_storm_objects):
            these_predecessor_rows = (
                temporal_tracking.find_immediate_predecessors(
                    storm_object_table=STORM_OBJECT_TABLE,
                    target_row=i, max_time_diff_seconds=2)
            )

            these_predecessor_rows = numpy.sort(these_predecessor_rows)
            these_expected_rows = numpy.sort(numpy.array(
                IMMED_PREDECESSOR_DICT_2SECONDS[i], dtype=int
            ))

            self.assertTrue(numpy.array_equal(
                these_predecessor_rows, these_expected_rows
            ))

    def test_find_immediate_successors_2sec(self):
        """Ensures correct output from find_immediate_successors.

        In this case, max time difference = 2 seconds.
        """

        this_num_storm_objects = len(STORM_OBJECT_TABLE.index)

        for i in range(this_num_storm_objects):
            these_successor_rows = temporal_tracking.find_immediate_successors(
                storm_object_table=STORM_OBJECT_TABLE, target_row=i,
                max_time_diff_seconds=2)

            these_successor_rows = numpy.sort(these_successor_rows)
            these_expected_rows = numpy.sort(numpy.array(
                IMMED_SUCCESSOR_DICT_2SECONDS[i], dtype=int
            ))

            self.assertTrue(numpy.array_equal(
                these_successor_rows, these_expected_rows
            ))

    def test_find_immediate_predecessors_1sec(self):
        """Ensures correct output from find_immediate_predecessors.

        In this case, max time difference = 1 second.
        """

        this_num_storm_objects = len(STORM_OBJECT_TABLE.index)

        for i in range(this_num_storm_objects):
            these_predecessor_rows = (
                temporal_tracking.find_immediate_predecessors(
                    storm_object_table=STORM_OBJECT_TABLE,
                    target_row=i, max_time_diff_seconds=1)
            )

            these_predecessor_rows = numpy.sort(these_predecessor_rows)
            these_expected_rows = numpy.sort(numpy.array(
                IMMED_PREDECESSOR_DICT_1SECOND[i], dtype=int
            ))

            self.assertTrue(numpy.array_equal(
                these_predecessor_rows, these_expected_rows
            ))

    def test_find_immediate_successors_1sec(self):
        """Ensures correct output from find_immediate_successors.

        In this case, max time difference = 1 second.
        """

        this_num_storm_objects = len(STORM_OBJECT_TABLE.index)

        for i in range(this_num_storm_objects):
            these_successor_rows = temporal_tracking.find_immediate_successors(
                storm_object_table=STORM_OBJECT_TABLE, target_row=i,
                max_time_diff_seconds=1)

            these_successor_rows = numpy.sort(these_successor_rows)
            these_expected_rows = numpy.sort(numpy.array(
                IMMED_SUCCESSOR_DICT_1SECOND[i], dtype=int
            ))

            self.assertTrue(numpy.array_equal(
                these_successor_rows, these_expected_rows
            ))

    def test_find_predecessors_5sec(self):
        """Ensures correct output from find_predecessors.

        In this case, max time difference = 5 seconds.
        """

        this_num_storm_objects = len(STORM_OBJECT_TABLE.index)

        for i in range(this_num_storm_objects):
            these_predecessor_rows = temporal_tracking.find_predecessors(
                storm_object_table=STORM_OBJECT_TABLE, target_row=i,
                num_seconds_back=5)

            these_predecessor_rows = numpy.sort(these_predecessor_rows)
            these_expected_rows = numpy.sort(numpy.array(
                ALL_PREDECESSOR_DICT_5SECONDS[i], dtype=int
            ))

            print i
            print these_expected_rows
            print these_predecessor_rows
            print '\n\n'

            self.assertTrue(numpy.array_equal(
                these_predecessor_rows, these_expected_rows
            ))

    def test_find_predecessors_2sec(self):
        """Ensures correct output from find_predecessors.

        In this case, max time difference = 2 seconds.
        """

        this_num_storm_objects = len(STORM_OBJECT_TABLE.index)

        for i in range(this_num_storm_objects):
            these_predecessor_rows = temporal_tracking.find_predecessors(
                storm_object_table=STORM_OBJECT_TABLE, target_row=i,
                num_seconds_back=2)

            these_predecessor_rows = numpy.sort(these_predecessor_rows)
            these_expected_rows = numpy.sort(numpy.array(
                ALL_PREDECESSOR_DICT_2SECONDS[i], dtype=int
            ))

            self.assertTrue(numpy.array_equal(
                these_predecessor_rows, these_expected_rows
            ))


if __name__ == '__main__':
    unittest.main()
