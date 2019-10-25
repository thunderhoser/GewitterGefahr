"""Unit tests for find_extreme_examples_2models.py."""

import unittest
import numpy
from gewittergefahr.deep_learning import prediction_io
from gewittergefahr.scripts import find_extreme_examples_2models as fee_2models

# FIRST_STORM_ID_STRINGS = [
#     'a', 'b', 'c',
#     'a', 'c', 'd',
#     'a', 'd', 'e',
#     'a', 'd',
#     'd'
# ]
#
# FIRST_STORM_TIMES_UNIX_SEC = numpy.array([
#     0, 0, 0,
#     300, 300, 300,
#     600, 600, 600,
#     900, 900,
#     1200
# ], dtype=int)

FIRST_STORM_ID_STRINGS = ['a', 'b', 'c']
FIRST_STORM_TIMES_UNIX_SEC = numpy.array([0, 0, 0], dtype=int)
FIRST_OBSERVED_LABELS = numpy.array([1, 0, 1], dtype=int)

FIRST_PREDICTION_DICT = {
    prediction_io.STORM_IDS_KEY: FIRST_STORM_ID_STRINGS,
    prediction_io.STORM_TIMES_KEY: FIRST_STORM_TIMES_UNIX_SEC,
    prediction_io.OBSERVED_LABELS_KEY: FIRST_OBSERVED_LABELS
}

# SECOND_STORM_ID_STRINGS = [
#     'i', 'ii', 'iii', 'vi',
#     'i', 'iii', 'vi',
#     'i', 'iv', 'vii', 'viii',
#     'i', 'vii', 'viii', 'x',
#     'viii', 'x'
# ]
#
# SECOND_STORM_TIMES_UNIX_SEC = numpy.array([
#     50, 50, 50, 50,
#     340, 340, 340,
#     666, 666, 666, 666,
#     963, 963, 963, 963,
#     1244, 1244
# ], dtype=int)

SECOND_STORM_ID_STRINGS = ['iii', 'vi', 'ii', 'i']
SECOND_STORM_TIMES_UNIX_SEC = numpy.array([50, 50, 50, 50], dtype=int)
SECOND_OBSERVED_LABELS = numpy.array([0, 1, 0, 1], dtype=int)

SECOND_PREDICTION_DICT = {
    prediction_io.STORM_IDS_KEY: SECOND_STORM_ID_STRINGS,
    prediction_io.STORM_TIMES_KEY: SECOND_STORM_TIMES_UNIX_SEC,
    prediction_io.OBSERVED_LABELS_KEY: SECOND_OBSERVED_LABELS
}

# MATCH_DICT = {
#     ('a', 0): ['i', 50],
#     ('b', 0): ['ii', 50],
#     ('c', 0): ['iii', 50],
#     ('a', 300): ['i', 340],
#     ('c', 300): ['iii', 340],
#     ('a', 600): ['i', 666],
#     ('d', 600): ['iv', 666],
#     ('a', 900): ['i', 963]
# }

MATCH_DICT = {
    ('a', 0): ['i', 50],
    ('b', 0): ['ii', 50],
    ('c', 0): ['iii', 50]
}

FIRST_INDICES = numpy.array([0, 1], dtype=int)
SECOND_INDICES = numpy.array([3, 2], dtype=int)


class FindExtremeExamples2ModelsTests(unittest.TestCase):
    """Each method is a unit test for find_extreme_examples_2models.py."""

    def test_match_storm_objects_one_time(self):
        """Ensures correct output from _match_storm_objects_one_time."""

        these_first_indices, these_second_indices = (
            fee_2models._match_storm_objects_one_time(
                first_prediction_dict=FIRST_PREDICTION_DICT,
                second_prediction_dict=SECOND_PREDICTION_DICT,
                match_dict=MATCH_DICT, allow_missing=False)
        )

        self.assertTrue(numpy.array_equal(these_first_indices, FIRST_INDICES))
        self.assertTrue(numpy.array_equal(these_second_indices, SECOND_INDICES))


if __name__ == '__main__':
    unittest.main()
