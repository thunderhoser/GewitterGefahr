"""Unit tests for subset_predictions_by_time.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.deep_learning import prediction_io
from gewittergefahr.scripts import subset_predictions_by_time as subsetting

# The following constants are used to test _get_months_in_each_chunk.
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

# The following constants are used to test _get_hours_in_each_chunk.
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

# The following constants are used to test _find_storm_objects_in_months and
# _find_storm_objects_in_hours.
TARGET_NAME = 'foo'
THESE_ID_STRINGS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']

THESE_TIME_STRINGS = [
    '4001-01-01-01', '4002-02-02-02', '4003-03-03-03', '4004-04-04-04',
    '4005-05-05-05', '4006-06-06-06', '4007-07-07-07', '4008-08-08-08',
    '4009-09-09-09', '4010-10-10-10', '4011-11-11-11', '4012-12-12-12'
]
THESE_TIMES_UNIX_SEC = [
    time_conversion.string_to_unix_sec(t, '%Y-%m-%d-%H')
    for t in THESE_TIME_STRINGS
]

THIS_PROBABILITY_MATRIX = numpy.array([
    [1, 0],
    [0.1, 0.9],
    [0.2, 0.8],
    [0.3, 0.7],
    [0.4, 0.6],
    [0.5, 0.5],
    [0.6, 0.4],
    [0.7, 0.3],
    [0.8, 0.2],
    [0.9, 0.1],
    [1, 0],
    [0.75, 0.25]
])

FULL_PREDICTION_DICT = {
    prediction_io.STORM_TIMES_KEY: THESE_TIMES_UNIX_SEC,
    prediction_io.STORM_IDS_KEY: THESE_ID_STRINGS,
    prediction_io.PROBABILITY_MATRIX_KEY: THIS_PROBABILITY_MATRIX,
    prediction_io.TARGET_NAME_KEY: TARGET_NAME
}

DESIRED_MONTHS = numpy.array([12, 1, 2], dtype=int)

THESE_ID_STRINGS = ['A', 'B', 'L']
THESE_TIME_STRINGS = ['4001-01-01-01', '4002-02-02-02', '4012-12-12-12']
THESE_TIMES_UNIX_SEC = [
    time_conversion.string_to_unix_sec(t, '%Y-%m-%d-%H')
    for t in THESE_TIME_STRINGS
]

THIS_PROBABILITY_MATRIX = numpy.array([
    [1, 0],
    [0.1, 0.9],
    [0.75, 0.25]
])

PREDICTION_DICT_DESIRED_MONTHS = {
    prediction_io.STORM_TIMES_KEY: THESE_TIMES_UNIX_SEC,
    prediction_io.STORM_IDS_KEY: THESE_ID_STRINGS,
    prediction_io.PROBABILITY_MATRIX_KEY: THIS_PROBABILITY_MATRIX,
    prediction_io.TARGET_NAME_KEY: TARGET_NAME
}

MORNING_HOURS = numpy.array([6, 7, 8, 9, 10, 11], dtype=int)

THESE_ID_STRINGS = ['F', 'G', 'H', 'I', 'J', 'K']
THESE_TIME_STRINGS = [
    '4006-06-06-06', '4007-07-07-07', '4008-08-08-08',
    '4009-09-09-09', '4010-10-10-10', '4011-11-11-11'
]
THESE_TIMES_UNIX_SEC = [
    time_conversion.string_to_unix_sec(t, '%Y-%m-%d-%H')
    for t in THESE_TIME_STRINGS
]

THIS_PROBABILITY_MATRIX = numpy.array([
    [0.5, 0.5],
    [0.6, 0.4],
    [0.7, 0.3],
    [0.8, 0.2],
    [0.9, 0.1],
    [1, 0]
])

PREDICTION_DICT_MORNING = {
    prediction_io.STORM_TIMES_KEY: THESE_TIMES_UNIX_SEC,
    prediction_io.STORM_IDS_KEY: THESE_ID_STRINGS,
    prediction_io.PROBABILITY_MATRIX_KEY: THIS_PROBABILITY_MATRIX,
    prediction_io.TARGET_NAME_KEY: TARGET_NAME
}

EVENING_HOURS = numpy.array([18, 19, 20, 21, 22, 23], dtype=int)

PREDICTION_DICT_EVENING = {
    prediction_io.STORM_TIMES_KEY: [],
    prediction_io.STORM_IDS_KEY: [],
    prediction_io.PROBABILITY_MATRIX_KEY: numpy.full((0, 2), 0.),
    prediction_io.TARGET_NAME_KEY: TARGET_NAME
}


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


def _compare_prediction_dicts(first_dict, second_dict):
    """Compare two dictionaries with predictions.

    :param first_dict: First dictionary.
    :param second_dict: Second dictionary.
    :return: are_dicts_equal: Boolean flag.
    """

    first_keys = list(first_dict.keys())
    second_keys = list(second_dict.keys())
    if set(first_keys) != set(second_keys):
        return False

    for this_key in first_keys:
        if isinstance(first_dict[this_key], list):
            if first_dict[this_key] == second_dict[this_key]:
                continue
        elif isinstance(first_dict[this_key], numpy.ndarray):
            if numpy.array_equal(first_dict[this_key], second_dict[this_key]):
                continue
        else:
            if first_dict[this_key] == second_dict[this_key]:
                continue

        return False

    return True


class SubsetPredictionsByTimeTests(unittest.TestCase):
    """Each method is a unit test for subset_predictions_by_time.py."""

    def test_get_months_in_each_chunk(self):
        """Ensures correct output from _get_months_in_each_chunk."""

        this_chunk_to_months_dict = subsetting._get_months_in_each_chunk(1)
        self.assertTrue(_compare_chunk_dicts(
            this_chunk_to_months_dict, CHUNK_TO_MONTHS_DICT_1EACH
        ))

    def test_get_hours_in_each_chunk_1(self):
        """Ensures correct output from _get_hours_in_each_chunk.

        In this case there is one hour per chunk.
        """

        this_chunk_to_hours_dict = subsetting._get_hours_in_each_chunk(1)
        self.assertTrue(_compare_chunk_dicts(
            this_chunk_to_hours_dict, CHUNK_TO_HOURS_DICT_1EACH
        ))

    def test_get_hours_in_each_chunk_3(self):
        """Ensures correct output from _get_hours_in_each_chunk.

        In this case there are 3 hours per chunk.
        """

        this_chunk_to_hours_dict = subsetting._get_hours_in_each_chunk(3)
        self.assertTrue(_compare_chunk_dicts(
            this_chunk_to_hours_dict, CHUNK_TO_HOURS_DICT_3EACH
        ))

    def test_get_hours_in_each_chunk_6(self):
        """Ensures correct output from _get_hours_in_each_chunk.

        In this case there are 6 hours per chunk.
        """

        this_chunk_to_hours_dict = subsetting._get_hours_in_each_chunk(6)
        self.assertTrue(_compare_chunk_dicts(
            this_chunk_to_hours_dict, CHUNK_TO_HOURS_DICT_6EACH
        ))

    def test_find_storm_objects_in_months(self):
        """Ensures correct output from _find_storm_objects_in_months."""

        this_prediction_dict = subsetting._find_storm_objects_in_months(
            desired_months=DESIRED_MONTHS, prediction_dict=FULL_PREDICTION_DICT
        )[0]

        self.assertTrue(_compare_prediction_dicts(
            this_prediction_dict, PREDICTION_DICT_DESIRED_MONTHS
        ))

    def test_find_storm_objects_in_morning(self):
        """Ensures correct output from _find_storm_objects_in_hours.

        In this case, only morning hours are desired.
        """

        this_prediction_dict = subsetting._find_storm_objects_in_hours(
            desired_hours=MORNING_HOURS, prediction_dict=FULL_PREDICTION_DICT
        )[0]

        self.assertTrue(_compare_prediction_dicts(
            this_prediction_dict, PREDICTION_DICT_MORNING
        ))

    def test_find_storm_objects_in_evening(self):
        """Ensures correct output from _find_storm_objects_in_hours.

        In this case, only evening hours are desired.
        """

        this_prediction_dict = subsetting._find_storm_objects_in_hours(
            desired_hours=EVENING_HOURS, prediction_dict=FULL_PREDICTION_DICT
        )[0]

        self.assertTrue(_compare_prediction_dicts(
            this_prediction_dict, PREDICTION_DICT_EVENING
        ))


if __name__ == '__main__':
    unittest.main()
