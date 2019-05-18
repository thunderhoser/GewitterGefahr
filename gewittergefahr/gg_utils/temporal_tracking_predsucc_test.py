"""Additional unit tests for temporal_tracking.py.

This file specifically tests methods that find predecessors and successors.  I
created this file, rather than putting all the unit tests in
temporal_tracking_test.py, because temporal_tracking_test.py was getting
unwieldy.
"""

import copy
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

FIRST_STORM_OBJECT_TABLE = pandas.DataFrame.from_dict({
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

FIRST_IMMED_PREDECESSOR_DICT_5SEC = {
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

FIRST_IMMED_SUCCESSOR_DICT_5SEC = {
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

FIRST_IMMED_PREDECESSOR_DICT_2SEC = {
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

FIRST_IMMED_SUCCESSOR_DICT_2SEC = {
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

FIRST_IMMED_PREDECESSOR_DICT_1SEC = {
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

FIRST_IMMED_SUCCESSOR_DICT_1SEC = {
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

FIRST_PREDECESSOR_DICT_5SEC = {
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

FIRST_PREDECESSOR_DICT_2SEC = {
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

FIRST_PREDECESSOR_DICT_1SEC = copy.deepcopy(FIRST_IMMED_PREDECESSOR_DICT_1SEC)

THESE_TIMES_UNIX_SEC = numpy.array([
    0, 0, 0, 0, 0, 0,
    10, 10, 10, 10, 10, 10, 10, 10,
    15, 15, 15, 15, 15, 15, 15
])

THESE_SECONDARY_ID_STRINGS = [
    '000000', '000001', '000002', '000003', '000004', '000005',
    '000007', '000005', '000009', '000008', '000004', '000006', '000001',
    '000010',
    '000011', '000005', '000016', '000014', '000012', '000013', '000015'
]

THESE_FIRST_PREV_SEC_ID_STRINGS = [
    '', '', '', '', '', '',
    '000003', '000005', '', '000003', '000004', '000000', '000001', '',
    '000007', '000005', '', '000010', '000006', '000006', '000010'
]

THESE_SECOND_PREV_SEC_ID_STRINGS = [
    '', '', '', '', '', '',
    '', '', '', '', '', '000002', '', '',
    '000009', '', '', '', '', '', ''
]

THESE_FIRST_NEXT_SEC_ID_STRINGS = [
    '000006', '000001', '000006', '000007', '000004', '000005',
    '000011', '000005', '000011', '', '', '000012', '', '000014',
    '', '', '', '', '', '', ''
]

THESE_SECOND_NEXT_SEC_ID_STRINGS = [
    '', '', '', '000008', '', '',
    '', '', '', '', '', '000013', '', '000015',
    '', '', '', '', '', '', ''
]

SECOND_STORM_OBJECT_TABLE = pandas.DataFrame.from_dict({
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

SECOND_IMMED_PREDECESSOR_DICT_15SEC = {
    0: [], 1: [], 2: [], 3: [], 4: [], 5: [],
    6: [3], 7: [5], 8: [], 9: [3], 10: [4], 11: [0, 2], 12: [1], 13: [],
    14: [6, 8], 15: [7], 16: [], 17: [13], 18: [11], 19: [11], 20: [13]
}

SECOND_IMMED_PREDECESSOR_DICT_5SEC = {
    0: [], 1: [], 2: [], 3: [], 4: [], 5: [],
    6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [], 13: [],
    14: [6, 8], 15: [7], 16: [], 17: [13], 18: [11], 19: [11], 20: [13]
}

SECOND_IMMED_SUCCESSOR_DICT_15SEC = {
    0: [11], 1: [12], 2: [11], 3: [6, 9], 4: [10], 5: [7],
    6: [14], 7: [15], 8: [14], 9: [], 10: [], 11: [18, 19], 12: [],
    13: [17, 20],
    14: [], 15: [], 16: [], 17: [], 18: [], 19: [], 20: []
}

SECOND_IMMED_SUCCESSOR_DICT_5SEC = {
    0: [], 1: [], 2: [], 3: [], 4: [], 5: [],
    6: [14], 7: [15], 8: [14], 9: [], 10: [], 11: [18, 19], 12: [],
    13: [17, 20],
    14: [], 15: [], 16: [], 17: [], 18: [], 19: [], 20: []
}

SECOND_PREDECESSOR_DICT_15SEC = {
    0: [], 1: [], 2: [], 3: [], 4: [], 5: [],
    6: [3], 7: [5], 8: [], 9: [3], 10: [4], 11: [0, 2], 12: [1], 13: [],
    14: [3, 8], 15: [5], 16: [], 17: [13], 18: [0, 2], 19: [0, 2], 20: [13]
}

SECOND_PREDECESSOR_DICT_5SEC = {
    0: [], 1: [], 2: [], 3: [], 4: [], 5: [],
    6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [], 13: [],
    14: [6, 8], 15: [7], 16: [], 17: [13], 18: [11], 19: [11], 20: [13]
}


class TemporalTrackingPredsuccTests(unittest.TestCase):
    """Each method is a unit test for temporal_tracking.py."""

    def test_find_immediate_predecessors_first_5sec(self):
        """Ensures correct output from find_immediate_predecessors.

        In this case, working on the first table with max time difference =
        5 seconds.
        """

        this_num_storm_objects = len(FIRST_STORM_OBJECT_TABLE.index)

        for i in range(this_num_storm_objects):
            these_predecessor_rows = (
                temporal_tracking.find_immediate_predecessors(
                    storm_object_table=FIRST_STORM_OBJECT_TABLE,
                    target_row=i, max_time_diff_seconds=5)
            )

            these_predecessor_rows = numpy.sort(these_predecessor_rows)
            these_expected_rows = numpy.sort(numpy.array(
                FIRST_IMMED_PREDECESSOR_DICT_5SEC[i], dtype=int
            ))

            self.assertTrue(numpy.array_equal(
                these_predecessor_rows, these_expected_rows
            ))

    def test_find_immediate_successors_first_5sec(self):
        """Ensures correct output from find_immediate_successors.

        In this case, working on the first table with max time difference =
        5 seconds.
        """

        this_num_storm_objects = len(FIRST_STORM_OBJECT_TABLE.index)

        for i in range(this_num_storm_objects):
            these_successor_rows = temporal_tracking.find_immediate_successors(
                storm_object_table=FIRST_STORM_OBJECT_TABLE, target_row=i,
                max_time_diff_seconds=5)

            these_successor_rows = numpy.sort(these_successor_rows)
            these_expected_rows = numpy.sort(numpy.array(
                FIRST_IMMED_SUCCESSOR_DICT_5SEC[i], dtype=int
            ))

            self.assertTrue(numpy.array_equal(
                these_successor_rows, these_expected_rows
            ))

    def test_find_predecessors_first_5sec(self):
        """Ensures correct output from find_predecessors.

        In this case, working on the first table with max time difference =
        5 seconds.
        """

        this_num_storm_objects = len(FIRST_STORM_OBJECT_TABLE.index)

        for i in range(this_num_storm_objects):
            these_predecessor_rows = temporal_tracking.find_predecessors(
                storm_object_table=FIRST_STORM_OBJECT_TABLE, target_row=i,
                num_seconds_back=5)

            these_predecessor_rows = numpy.sort(these_predecessor_rows)
            these_expected_rows = numpy.sort(numpy.array(
                FIRST_PREDECESSOR_DICT_5SEC[i], dtype=int
            ))

            self.assertTrue(numpy.array_equal(
                these_predecessor_rows, these_expected_rows
            ))

    def test_find_immediate_predecessors_first_2sec(self):
        """Ensures correct output from find_immediate_predecessors.

        In this case, working on the first table with max time difference =
        2 seconds.
        """

        this_num_storm_objects = len(FIRST_STORM_OBJECT_TABLE.index)

        for i in range(this_num_storm_objects):
            these_predecessor_rows = (
                temporal_tracking.find_immediate_predecessors(
                    storm_object_table=FIRST_STORM_OBJECT_TABLE,
                    target_row=i, max_time_diff_seconds=2)
            )

            these_predecessor_rows = numpy.sort(these_predecessor_rows)
            these_expected_rows = numpy.sort(numpy.array(
                FIRST_IMMED_PREDECESSOR_DICT_2SEC[i], dtype=int
            ))

            self.assertTrue(numpy.array_equal(
                these_predecessor_rows, these_expected_rows
            ))

    def test_find_immediate_successors_first_2sec(self):
        """Ensures correct output from find_immediate_successors.

        In this case, working on the first table with max time difference =
        2 seconds.
        """

        this_num_storm_objects = len(FIRST_STORM_OBJECT_TABLE.index)

        for i in range(this_num_storm_objects):
            these_successor_rows = temporal_tracking.find_immediate_successors(
                storm_object_table=FIRST_STORM_OBJECT_TABLE, target_row=i,
                max_time_diff_seconds=2)

            these_successor_rows = numpy.sort(these_successor_rows)
            these_expected_rows = numpy.sort(numpy.array(
                FIRST_IMMED_SUCCESSOR_DICT_2SEC[i], dtype=int
            ))

            self.assertTrue(numpy.array_equal(
                these_successor_rows, these_expected_rows
            ))

    def test_find_predecessors_first_2sec(self):
        """Ensures correct output from find_predecessors.

        In this case, working on the first table with max time difference =
        2 seconds.
        """

        this_num_storm_objects = len(FIRST_STORM_OBJECT_TABLE.index)

        for i in range(this_num_storm_objects):
            these_predecessor_rows = temporal_tracking.find_predecessors(
                storm_object_table=FIRST_STORM_OBJECT_TABLE, target_row=i,
                num_seconds_back=2)

            these_predecessor_rows = numpy.sort(these_predecessor_rows)
            these_expected_rows = numpy.sort(numpy.array(
                FIRST_PREDECESSOR_DICT_2SEC[i], dtype=int
            ))

            self.assertTrue(numpy.array_equal(
                these_predecessor_rows, these_expected_rows
            ))

    def test_find_immediate_predecessors_first_1sec(self):
        """Ensures correct output from find_immediate_predecessors.

        In this case, working on the first table with max time difference =
        1 second.
        """

        this_num_storm_objects = len(FIRST_STORM_OBJECT_TABLE.index)

        for i in range(this_num_storm_objects):
            these_predecessor_rows = (
                temporal_tracking.find_immediate_predecessors(
                    storm_object_table=FIRST_STORM_OBJECT_TABLE,
                    target_row=i, max_time_diff_seconds=1)
            )

            these_predecessor_rows = numpy.sort(these_predecessor_rows)
            these_expected_rows = numpy.sort(numpy.array(
                FIRST_IMMED_PREDECESSOR_DICT_1SEC[i], dtype=int
            ))

            self.assertTrue(numpy.array_equal(
                these_predecessor_rows, these_expected_rows
            ))

    def test_find_immediate_successors_first_1sec(self):
        """Ensures correct output from find_immediate_successors.

        In this case, working on the first table with max time difference =
        1 second.
        """

        this_num_storm_objects = len(FIRST_STORM_OBJECT_TABLE.index)

        for i in range(this_num_storm_objects):
            these_successor_rows = temporal_tracking.find_immediate_successors(
                storm_object_table=FIRST_STORM_OBJECT_TABLE, target_row=i,
                max_time_diff_seconds=1)

            these_successor_rows = numpy.sort(these_successor_rows)
            these_expected_rows = numpy.sort(numpy.array(
                FIRST_IMMED_SUCCESSOR_DICT_1SEC[i], dtype=int
            ))

            self.assertTrue(numpy.array_equal(
                these_successor_rows, these_expected_rows
            ))

    def test_find_predecessors_first_1sec(self):
        """Ensures correct output from find_predecessors.

        In this case, working on the first table with max time difference =
        1 second.
        """

        this_num_storm_objects = len(FIRST_STORM_OBJECT_TABLE.index)

        for i in range(this_num_storm_objects):
            these_predecessor_rows = temporal_tracking.find_predecessors(
                storm_object_table=FIRST_STORM_OBJECT_TABLE, target_row=i,
                num_seconds_back=1)

            these_predecessor_rows = numpy.sort(these_predecessor_rows)
            these_expected_rows = numpy.sort(numpy.array(
                FIRST_PREDECESSOR_DICT_1SEC[i], dtype=int
            ))

            self.assertTrue(numpy.array_equal(
                these_predecessor_rows, these_expected_rows
            ))

    def test_find_immediate_predecessors_second_15sec(self):
        """Ensures correct output from find_immediate_predecessors.

        In this case, working on the second table with max time difference =
        15 seconds.
        """

        this_num_storm_objects = len(SECOND_STORM_OBJECT_TABLE.index)

        for i in range(this_num_storm_objects):
            these_predecessor_rows = (
                temporal_tracking.find_immediate_predecessors(
                    storm_object_table=SECOND_STORM_OBJECT_TABLE,
                    target_row=i, max_time_diff_seconds=15)
            )

            these_predecessor_rows = numpy.sort(these_predecessor_rows)
            these_expected_rows = numpy.sort(numpy.array(
                SECOND_IMMED_PREDECESSOR_DICT_15SEC[i], dtype=int
            ))

            self.assertTrue(numpy.array_equal(
                these_predecessor_rows, these_expected_rows
            ))

    def test_find_immediate_successors_second_15sec(self):
        """Ensures correct output from find_immediate_successors.

        In this case, working on the second table with max time difference =
        15 seconds.
        """

        this_num_storm_objects = len(SECOND_STORM_OBJECT_TABLE.index)

        for i in range(this_num_storm_objects):
            these_successor_rows = temporal_tracking.find_immediate_successors(
                storm_object_table=SECOND_STORM_OBJECT_TABLE, target_row=i,
                max_time_diff_seconds=15)

            these_successor_rows = numpy.sort(these_successor_rows)
            these_expected_rows = numpy.sort(numpy.array(
                SECOND_IMMED_SUCCESSOR_DICT_15SEC[i], dtype=int
            ))

            self.assertTrue(numpy.array_equal(
                these_successor_rows, these_expected_rows
            ))

    def test_find_predecessors_second_15sec(self):
        """Ensures correct output from find_predecessors.

        In this case, working on the second table with max time difference =
        15 seconds.
        """

        this_num_storm_objects = len(SECOND_STORM_OBJECT_TABLE.index)

        for i in range(this_num_storm_objects):
            these_predecessor_rows = temporal_tracking.find_predecessors(
                storm_object_table=SECOND_STORM_OBJECT_TABLE, target_row=i,
                num_seconds_back=15)

            these_predecessor_rows = numpy.sort(these_predecessor_rows)
            these_expected_rows = numpy.sort(numpy.array(
                SECOND_PREDECESSOR_DICT_15SEC[i], dtype=int
            ))

            self.assertTrue(numpy.array_equal(
                these_predecessor_rows, these_expected_rows
            ))

    def test_find_immediate_predecessors_second_5sec(self):
        """Ensures correct output from find_immediate_predecessors.

        In this case, working on the second table with max time difference =
        5 seconds.
        """

        this_num_storm_objects = len(SECOND_STORM_OBJECT_TABLE.index)

        for i in range(this_num_storm_objects):
            these_predecessor_rows = (
                temporal_tracking.find_immediate_predecessors(
                    storm_object_table=SECOND_STORM_OBJECT_TABLE,
                    target_row=i, max_time_diff_seconds=5)
            )

            these_predecessor_rows = numpy.sort(these_predecessor_rows)
            these_expected_rows = numpy.sort(numpy.array(
                SECOND_IMMED_PREDECESSOR_DICT_5SEC[i], dtype=int
            ))

            self.assertTrue(numpy.array_equal(
                these_predecessor_rows, these_expected_rows
            ))

    def test_find_immediate_successors_second_5sec(self):
        """Ensures correct output from find_immediate_successors.

        In this case, working on the second table with max time difference =
        5 seconds.
        """

        this_num_storm_objects = len(SECOND_STORM_OBJECT_TABLE.index)

        for i in range(this_num_storm_objects):
            these_successor_rows = temporal_tracking.find_immediate_successors(
                storm_object_table=SECOND_STORM_OBJECT_TABLE, target_row=i,
                max_time_diff_seconds=5)

            these_successor_rows = numpy.sort(these_successor_rows)
            these_expected_rows = numpy.sort(numpy.array(
                SECOND_IMMED_SUCCESSOR_DICT_5SEC[i], dtype=int
            ))

            self.assertTrue(numpy.array_equal(
                these_successor_rows, these_expected_rows
            ))

    def test_find_predecessors_second_5sec(self):
        """Ensures correct output from find_predecessors.

        In this case, working on the second table with max time difference =
        5 seconds.
        """

        this_num_storm_objects = len(SECOND_STORM_OBJECT_TABLE.index)

        for i in range(this_num_storm_objects):
            these_predecessor_rows = temporal_tracking.find_predecessors(
                storm_object_table=SECOND_STORM_OBJECT_TABLE, target_row=i,
                num_seconds_back=5)

            these_predecessor_rows = numpy.sort(these_predecessor_rows)
            these_expected_rows = numpy.sort(numpy.array(
                SECOND_PREDECESSOR_DICT_5SEC[i], dtype=int
            ))

            self.assertTrue(numpy.array_equal(
                these_predecessor_rows, these_expected_rows
            ))


if __name__ == '__main__':
    unittest.main()
