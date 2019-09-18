"""Unit tests for prediction_io.py."""

import copy
import unittest
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.deep_learning import prediction_io

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
    [0.9, 0.1],
    [0.8, 0.2],
    [0.7, 0.3],
    [0.6, 0.4],
    [0.5, 0.5],
    [0.4, 0.6],
    [0.3, 0.7],
    [0.2, 0.8],
    [0.1, 0.9],
    [0, 1],
    [0.75, 0.25]
])
THESE_OBSERVED_LABELS = numpy.array(
    [0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0], dtype=int
)

FULL_PREDICTION_DICT_SANS_OBS = {
    prediction_io.TARGET_NAME_KEY: TARGET_NAME,
    prediction_io.STORM_IDS_KEY: THESE_ID_STRINGS,
    prediction_io.STORM_TIMES_KEY: THESE_TIMES_UNIX_SEC,
    prediction_io.PROBABILITY_MATRIX_KEY: THIS_PROBABILITY_MATRIX,
    prediction_io.OBSERVED_LABELS_KEY: None
}

FULL_PREDICTION_DICT_WITH_OBS = copy.deepcopy(FULL_PREDICTION_DICT_SANS_OBS)
FULL_PREDICTION_DICT_WITH_OBS[
    prediction_io.OBSERVED_LABELS_KEY] = THESE_OBSERVED_LABELS

DESIRED_INDICES = numpy.array([0, 1, 11], dtype=int)

THESE_ID_STRINGS = ['A', 'B', 'L']
THESE_TIME_STRINGS = ['4001-01-01-01', '4002-02-02-02', '4012-12-12-12']
THESE_TIMES_UNIX_SEC = [
    time_conversion.string_to_unix_sec(t, '%Y-%m-%d-%H')
    for t in THESE_TIME_STRINGS
]
THIS_PROBABILITY_MATRIX = numpy.array([
    [1, 0],
    [0.9, 0.1],
    [0.75, 0.25]
])
THESE_OBSERVED_LABELS = numpy.array([0, 0, 0], dtype=int)

SMALL_PREDICTION_DICT_SANS_OBS = {
    prediction_io.TARGET_NAME_KEY: TARGET_NAME,
    prediction_io.STORM_IDS_KEY: THESE_ID_STRINGS,
    prediction_io.STORM_TIMES_KEY: THESE_TIMES_UNIX_SEC,
    prediction_io.PROBABILITY_MATRIX_KEY: THIS_PROBABILITY_MATRIX,
    prediction_io.OBSERVED_LABELS_KEY: None
}

SMALL_PREDICTION_DICT_WITH_OBS = copy.deepcopy(SMALL_PREDICTION_DICT_SANS_OBS)
SMALL_PREDICTION_DICT_WITH_OBS[
    prediction_io.OBSERVED_LABELS_KEY] = THESE_OBSERVED_LABELS


def _compare_ungridded_predictions(first_dict, second_dict):
    """Compare two dictionaries with ungridded predictions.

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


class PredictionIoTests(unittest.TestCase):
    """Each method is a unit test for prediction_io.py."""

    def test_subset_ungridded_predictions_sans_obs(self):
        """Ensures correct output from subset_ungridded_predictions.

        In this case, labels (observations) are *not* included.
        """

        this_prediction_dict = prediction_io.subset_ungridded_predictions(
            prediction_dict=FULL_PREDICTION_DICT_SANS_OBS,
            desired_storm_indices=DESIRED_INDICES)

        self.assertTrue(_compare_ungridded_predictions(
            this_prediction_dict, SMALL_PREDICTION_DICT_SANS_OBS
        ))

    def test_subset_ungridded_predictions_with_obs(self):
        """Ensures correct output from subset_ungridded_predictions.

        In this case, labels (observations) are included.
        """

        this_prediction_dict = prediction_io.subset_ungridded_predictions(
            prediction_dict=FULL_PREDICTION_DICT_WITH_OBS,
            desired_storm_indices=DESIRED_INDICES)

        self.assertTrue(_compare_ungridded_predictions(
            this_prediction_dict, SMALL_PREDICTION_DICT_WITH_OBS
        ))


if __name__ == '__main__':
    unittest.main()
