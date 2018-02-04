"""Unit tests for labels_new.py."""

import copy
import unittest
import numpy
import pandas
from gewittergefahr.gg_utils import labels_new
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import link_events_to_storms as events2storms

TOLERANCE = 1e-6

# The following constants are used to test
# _remove_data_near_end_of_tracking_period.
MAX_LEAD_TIME_SEC = 3600

THESE_END_TIMES_UNIX_SEC = numpy.array(
    [10000, 10000, 10000, 10000, 20000, 20000, 20000, 20000], dtype=int)
THESE_VALID_TIMES_UNIX_SEC = numpy.array(
    [2500, 5000, 7500, 10000, 12500, 15000, 16400, 17500], dtype=int)

THIS_DICTIONARY = {
    tracking_utils.TRACKING_END_TIME_COLUMN: THESE_END_TIMES_UNIX_SEC,
    tracking_utils.TIME_COLUMN: THESE_VALID_TIMES_UNIX_SEC
}
STORM_TO_EVENTS_TABLE_WITH_END_OF_PERIOD = pandas.DataFrame.from_dict(
    THIS_DICTIONARY)

THESE_INVALID_ROWS = numpy.array([2, 3, 7], dtype=int)
STORM_TO_EVENTS_TABLE_SANS_END_OF_PERIOD = (
    STORM_TO_EVENTS_TABLE_WITH_END_OF_PERIOD.drop(
        STORM_TO_EVENTS_TABLE_WITH_END_OF_PERIOD.index[THESE_INVALID_ROWS],
        axis=0, inplace=False))

# The following constants are used to test get_column_name_for_regression_label,
# get_column_name_for_num_wind_obs, and
# get_column_name_for_classification_label.
MIN_LEAD_TIME_SEC = 900
MIN_LINK_DISTANCE_METRES = 1.
MAX_LINK_DISTANCE_METRES = 5000.
WIND_SPEED_PERCENTILE_LEVEL = 97.5
WIND_CLASS_CUTOFFS_KT = numpy.array([10., 20., 30., 40., 50.])

REGRESSION_LABEL_COLUMN_NAME = (
    'wind-speed-m-s01_percentile=097.5_lead-time=0900-3600sec_'
    'distance=00001-05000m')
NUM_WIND_OBS_COLUMN_NAME = (
    'num-wind-observations_lead-time=0900-3600sec_distance=00001-05000m')
WIND_CLASSIFICATION_LABEL_COLUMN_NAME = (
    'wind-speed_percentile=097.5_lead-time=0900-3600sec_distance=00001-05000m'
    '_cutoffs=10-20-30-40-50kt')
TORNADO_LABEL_COLUMN_NAME = (
    'tornado_lead-time=0900-3600sec_distance=00001-05000m')

# The following constants are used to test column_name_to_label_params.
PARAM_DICT_FOR_REGRESSION_LABEL = {
    labels_new.MIN_LEAD_TIME_KEY: MIN_LEAD_TIME_SEC,
    labels_new.MAX_LEAD_TIME_KEY: MAX_LEAD_TIME_SEC,
    labels_new.MIN_LINKAGE_DISTANCE_KEY: MIN_LINK_DISTANCE_METRES,
    labels_new.MAX_LINKAGE_DISTANCE_KEY: MAX_LINK_DISTANCE_METRES,
    labels_new.EVENT_TYPE_KEY: events2storms.WIND_EVENT_TYPE_STRING,
    labels_new.WIND_SPEED_PERCENTILE_LEVEL_KEY: WIND_SPEED_PERCENTILE_LEVEL,
    labels_new.WIND_SPEED_CLASS_CUTOFFS_KEY: None
}
PARAM_DICT_FOR_WIND_CLASSIFICATION_LABEL = {
    labels_new.MIN_LEAD_TIME_KEY: MIN_LEAD_TIME_SEC,
    labels_new.MAX_LEAD_TIME_KEY: MAX_LEAD_TIME_SEC,
    labels_new.MIN_LINKAGE_DISTANCE_KEY: MIN_LINK_DISTANCE_METRES,
    labels_new.MAX_LINKAGE_DISTANCE_KEY: MAX_LINK_DISTANCE_METRES,
    labels_new.EVENT_TYPE_KEY: events2storms.WIND_EVENT_TYPE_STRING,
    labels_new.WIND_SPEED_PERCENTILE_LEVEL_KEY: WIND_SPEED_PERCENTILE_LEVEL,
    labels_new.WIND_SPEED_CLASS_CUTOFFS_KEY: WIND_CLASS_CUTOFFS_KT
}
PARAM_DICT_FOR_TORNADO_LABEL = {
    labels_new.MIN_LEAD_TIME_KEY: MIN_LEAD_TIME_SEC,
    labels_new.MAX_LEAD_TIME_KEY: MAX_LEAD_TIME_SEC,
    labels_new.MIN_LINKAGE_DISTANCE_KEY: MIN_LINK_DISTANCE_METRES,
    labels_new.MAX_LINKAGE_DISTANCE_KEY: MAX_LINK_DISTANCE_METRES,
    labels_new.EVENT_TYPE_KEY: events2storms.TORNADO_EVENT_TYPE_STRING,
    labels_new.WIND_SPEED_PERCENTILE_LEVEL_KEY: None,
    labels_new.WIND_SPEED_CLASS_CUTOFFS_KEY: None
}

# The following constants are used to test get_columns_with_labels,
# get_columns_with_num_wind_obs, and check_label_table.
THIS_DICTIONARY = {
    REGRESSION_LABEL_COLUMN_NAME: numpy.array([]),
    NUM_WIND_OBS_COLUMN_NAME: numpy.array([], dtype=int),
    WIND_CLASSIFICATION_LABEL_COLUMN_NAME: numpy.array([], dtype=int),
    TORNADO_LABEL_COLUMN_NAME: numpy.array([], dtype=int)
}
LABEL_TABLE = pandas.DataFrame.from_dict(THIS_DICTIONARY)


class LabelsNewTests(unittest.TestCase):
    """Each method is a unit test for labels_new.py."""

    def test_remove_data_near_end_of_tracking_period(self):
        """Ensures correctness of _remove_data_near_end_of_tracking_period."""

        this_orig_table = copy.deepcopy(
            STORM_TO_EVENTS_TABLE_WITH_END_OF_PERIOD)
        this_new_table = labels_new._remove_data_near_end_of_tracking_period(
            storm_to_events_table=this_orig_table,
            max_lead_time_sec=MAX_LEAD_TIME_SEC)

        self.assertTrue(this_new_table.equals(
            STORM_TO_EVENTS_TABLE_SANS_END_OF_PERIOD))

    def test_get_column_name_for_regression_label(self):
        """Ensures correct output from get_column_name_for_regression_label."""

        this_column_name = labels_new.get_column_name_for_regression_label(
            min_lead_time_sec=MIN_LEAD_TIME_SEC,
            max_lead_time_sec=MAX_LEAD_TIME_SEC,
            min_link_distance_metres=MIN_LINK_DISTANCE_METRES,
            max_link_distance_metres=MAX_LINK_DISTANCE_METRES,
            wind_speed_percentile_level=WIND_SPEED_PERCENTILE_LEVEL)
        self.assertTrue(this_column_name == REGRESSION_LABEL_COLUMN_NAME)

    def test_get_column_name_for_num_wind_obs(self):
        """Ensures correct output from get_column_name_for_num_wind_obs."""

        this_column_name = labels_new.get_column_name_for_num_wind_obs(
            min_lead_time_sec=MIN_LEAD_TIME_SEC,
            max_lead_time_sec=MAX_LEAD_TIME_SEC,
            min_link_distance_metres=MIN_LINK_DISTANCE_METRES,
            max_link_distance_metres=MAX_LINK_DISTANCE_METRES)
        self.assertTrue(this_column_name == NUM_WIND_OBS_COLUMN_NAME)

    def test_get_column_name_for_classification_label_wind(self):
        """Ensures correct output from get_column_name_for_classification_label.

        In this case, event type is wind speed.
        """

        this_column_name = labels_new.get_column_name_for_classification_label(
            min_lead_time_sec=MIN_LEAD_TIME_SEC,
            max_lead_time_sec=MAX_LEAD_TIME_SEC,
            min_link_distance_metres=MIN_LINK_DISTANCE_METRES,
            max_link_distance_metres=MAX_LINK_DISTANCE_METRES,
            event_type_string=events2storms.WIND_EVENT_TYPE_STRING,
            wind_speed_percentile_level=WIND_SPEED_PERCENTILE_LEVEL,
            wind_speed_class_cutoffs_kt=WIND_CLASS_CUTOFFS_KT)

        self.assertTrue(
            this_column_name == WIND_CLASSIFICATION_LABEL_COLUMN_NAME)

    def test_get_column_name_for_classification_label_tornado(self):
        """Ensures correct output from get_column_name_for_classification_label.

        In this case, event type is tornado occurrence.
        """

        this_column_name = labels_new.get_column_name_for_classification_label(
            min_lead_time_sec=MIN_LEAD_TIME_SEC,
            max_lead_time_sec=MAX_LEAD_TIME_SEC,
            min_link_distance_metres=MIN_LINK_DISTANCE_METRES,
            max_link_distance_metres=MAX_LINK_DISTANCE_METRES,
            event_type_string=events2storms.TORNADO_EVENT_TYPE_STRING)
        self.assertTrue(this_column_name == TORNADO_LABEL_COLUMN_NAME)

    def test_column_name_to_label_params_regression(self):
        """Ensures correct output from column_name_to_label_params.

        In this case, learning goal is regression.
        """

        this_parameter_dict = labels_new.column_name_to_label_params(
            REGRESSION_LABEL_COLUMN_NAME)
        self.assertTrue(this_parameter_dict == PARAM_DICT_FOR_REGRESSION_LABEL)


    def test_column_name_to_label_params_wind_classification(self):
        """Ensures correct output from column_name_to_label_params.

        In this case, learning goal is classification and event type is wind
        speed.
        """

        this_parameter_dict = labels_new.column_name_to_label_params(
            WIND_CLASSIFICATION_LABEL_COLUMN_NAME)

        self.assertTrue(numpy.array_equal(
            this_parameter_dict[labels_new.WIND_SPEED_CLASS_CUTOFFS_KEY],
            PARAM_DICT_FOR_WIND_CLASSIFICATION_LABEL[
                labels_new.WIND_SPEED_CLASS_CUTOFFS_KEY]))

        this_parameter_dict.pop(labels_new.WIND_SPEED_CLASS_CUTOFFS_KEY, None)
        PARAM_DICT_FOR_WIND_CLASSIFICATION_LABEL.pop(
            labels_new.WIND_SPEED_CLASS_CUTOFFS_KEY, None)

        self.assertTrue(
            this_parameter_dict == PARAM_DICT_FOR_WIND_CLASSIFICATION_LABEL)

    def test_column_name_to_label_params_tornado(self):
        """Ensures correct output from column_name_to_label_params.

        In this case, learning goal is classification and event type is tornado
        occurrence.
        """

        this_parameter_dict = labels_new.column_name_to_label_params(
            TORNADO_LABEL_COLUMN_NAME)
        self.assertTrue(this_parameter_dict == PARAM_DICT_FOR_TORNADO_LABEL)

    def test_get_columns_with_labels_regression(self):
        """Ensures correct output from get_columns_with_labels.

        In this case, learning goal is regression.
        """

        these_column_names = labels_new.get_columns_with_labels(
            label_table=LABEL_TABLE,
            goal_string=labels_new.REGRESSION_GOAL_STRING,
            event_type_string=events2storms.WIND_EVENT_TYPE_STRING)
        self.assertTrue(these_column_names == [REGRESSION_LABEL_COLUMN_NAME])

    def test_get_columns_with_labels_wind_classification(self):
        """Ensures correct output from get_columns_with_labels.

        In this case, learning goal is classification and event type is wind
        speed.
        """

        these_column_names = labels_new.get_columns_with_labels(
            label_table=LABEL_TABLE,
            goal_string=labels_new.CLASSIFICATION_GOAL_STRING,
            event_type_string=events2storms.WIND_EVENT_TYPE_STRING)

        self.assertTrue(
            these_column_names == [WIND_CLASSIFICATION_LABEL_COLUMN_NAME])

    def test_get_columns_with_labels_tornado(self):
        """Ensures correct output from get_columns_with_labels.

        In this case, learning goal is classification and event type is tornado
        occurrence.
        """

        these_column_names = labels_new.get_columns_with_labels(
            label_table=LABEL_TABLE,
            goal_string=labels_new.CLASSIFICATION_GOAL_STRING,
            event_type_string=events2storms.TORNADO_EVENT_TYPE_STRING)
        self.assertTrue(these_column_names == [TORNADO_LABEL_COLUMN_NAME])

    def test_get_columns_with_num_wind_obs_regression(self):
        """Ensures correct output from get_columns_with_num_wind_obs.

        In this case, learning goal is regression.
        """

        these_column_names = labels_new.get_columns_with_num_wind_obs(
            label_table=LABEL_TABLE,
            label_column_names=[REGRESSION_LABEL_COLUMN_NAME])
        self.assertTrue(these_column_names == [NUM_WIND_OBS_COLUMN_NAME])

    def test_get_columns_with_num_wind_obs_classification(self):
        """Ensures correct output from get_columns_with_num_wind_obs.

        In this case, learning goal is classification.
        """

        these_column_names = labels_new.get_columns_with_num_wind_obs(
            label_table=LABEL_TABLE,
            label_column_names=[WIND_CLASSIFICATION_LABEL_COLUMN_NAME])
        self.assertTrue(these_column_names == [NUM_WIND_OBS_COLUMN_NAME])


if __name__ == '__main__':
    unittest.main()
