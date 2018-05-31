"""Unit tests for labels.py."""

import unittest
import numpy
import pandas
from gewittergefahr.gg_utils import labels
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import link_events_to_storms as events2storms

TOLERANCE = 1e-6

# The following constants are used to test
# _find_storms_near_end_of_tracking_period.
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

INVALID_STORM_OBJECT_INDICES = numpy.array([2, 3, 7], dtype=int)

# The following constants are used to test _find_dead_storms.
MIN_LEAD_TIME_FOR_DEAD_SEC = 3600

THESE_END_TIMES_UNIX_SEC = numpy.array(
    [10000, 10000, 10000, 10000, 20000, 20000, 20000, 20000], dtype=int)
THESE_VALID_TIMES_UNIX_SEC = numpy.array(
    [2500, 5000, 7500, 10000, 12500, 15000, 16400, 17500], dtype=int)

THIS_DICTIONARY = {
    tracking_utils.CELL_END_TIME_COLUMN: THESE_END_TIMES_UNIX_SEC,
    tracking_utils.TIME_COLUMN: THESE_VALID_TIMES_UNIX_SEC
}
STORM_TO_EVENTS_TABLE_WITH_DEAD_STORMS = pandas.DataFrame.from_dict(
    THIS_DICTIONARY)

DEAD_STORM_OBJECT_INDICES = numpy.array([2, 3, 7], dtype=int)

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

WIND_CLASSIFN_LABEL_NAME_ZERO_LEAD_TIME = (
    'wind-speed_percentile=097.5_lead-time=0000-3600sec_distance=00001-05000m'
    '_cutoffs=10-20-30-40-50kt')

# The following constants are used to test column_name_to_label_params and
# column_name_to_num_classes.
PARAM_DICT_FOR_REGRESSION_LABEL = {
    labels.MIN_LEAD_TIME_KEY: MIN_LEAD_TIME_SEC,
    labels.MAX_LEAD_TIME_KEY: MAX_LEAD_TIME_SEC,
    labels.MIN_LINKAGE_DISTANCE_KEY: MIN_LINK_DISTANCE_METRES,
    labels.MAX_LINKAGE_DISTANCE_KEY: MAX_LINK_DISTANCE_METRES,
    labels.EVENT_TYPE_KEY: events2storms.WIND_EVENT_TYPE_STRING,
    labels.WIND_SPEED_PERCENTILE_LEVEL_KEY: WIND_SPEED_PERCENTILE_LEVEL,
    labels.WIND_SPEED_CLASS_CUTOFFS_KEY: None
}
PARAM_DICT_FOR_WIND_CLASSIFICATION_LABEL = {
    labels.MIN_LEAD_TIME_KEY: MIN_LEAD_TIME_SEC,
    labels.MAX_LEAD_TIME_KEY: MAX_LEAD_TIME_SEC,
    labels.MIN_LINKAGE_DISTANCE_KEY: MIN_LINK_DISTANCE_METRES,
    labels.MAX_LINKAGE_DISTANCE_KEY: MAX_LINK_DISTANCE_METRES,
    labels.EVENT_TYPE_KEY: events2storms.WIND_EVENT_TYPE_STRING,
    labels.WIND_SPEED_PERCENTILE_LEVEL_KEY: WIND_SPEED_PERCENTILE_LEVEL,
    labels.WIND_SPEED_CLASS_CUTOFFS_KEY: WIND_CLASS_CUTOFFS_KT
}
PARAM_DICT_FOR_TORNADO_LABEL = {
    labels.MIN_LEAD_TIME_KEY: MIN_LEAD_TIME_SEC,
    labels.MAX_LEAD_TIME_KEY: MAX_LEAD_TIME_SEC,
    labels.MIN_LINKAGE_DISTANCE_KEY: MIN_LINK_DISTANCE_METRES,
    labels.MAX_LINKAGE_DISTANCE_KEY: MAX_LINK_DISTANCE_METRES,
    labels.EVENT_TYPE_KEY: events2storms.TORNADO_EVENT_TYPE_STRING,
    labels.WIND_SPEED_PERCENTILE_LEVEL_KEY: None,
    labels.WIND_SPEED_CLASS_CUTOFFS_KEY: None
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

# The following constants are used to test find_label_file.
TOP_DIRECTORY_NAME = 'labels'
FILE_TIME_UNIX_SEC = 1517523991  # 222631 1 Feb 2018
FILE_SPC_DATE_STRING = '20180201'

WIND_LABELS_ONE_TIME_FILE_NAME = (
    'labels/2018/20180201/wind_labels_2018-02-01-222631.p')
WIND_LABELS_SPC_DATE_FILE_NAME = 'labels/2018/wind_labels_20180201.p'
TORNADO_LABELS_ONE_TIME_FILE_NAME = (
    'labels/2018/20180201/tornado_labels_2018-02-01-222631.p')
TORNADO_LABELS_SPC_DATE_FILE_NAME = 'labels/2018/tornado_labels_20180201.p'


class LabelsTests(unittest.TestCase):
    """Each method is a unit test for labels.py."""

    def test_find_storms_near_end_of_tracking_period(self):
        """Ensures correctness of _find_storms_near_end_of_tracking_period."""

        these_indices = labels._find_storms_near_end_of_tracking_period(
            storm_to_events_table=STORM_TO_EVENTS_TABLE_WITH_END_OF_PERIOD,
            max_lead_time_sec=MAX_LEAD_TIME_SEC)

        self.assertTrue(numpy.array_equal(
            these_indices, INVALID_STORM_OBJECT_INDICES))

    def test_find_dead_storms(self):
        """Ensures correct output from _find_dead_storms."""

        these_indices = labels._find_dead_storms(
            storm_to_events_table=STORM_TO_EVENTS_TABLE_WITH_DEAD_STORMS,
            min_lead_time_sec=MIN_LEAD_TIME_FOR_DEAD_SEC)

        self.assertTrue(numpy.array_equal(
            these_indices, DEAD_STORM_OBJECT_INDICES))

    def test_get_column_name_for_regression_label(self):
        """Ensures correct output from get_column_name_for_regression_label."""

        this_column_name = labels.get_column_name_for_regression_label(
            min_lead_time_sec=MIN_LEAD_TIME_SEC,
            max_lead_time_sec=MAX_LEAD_TIME_SEC,
            min_link_distance_metres=MIN_LINK_DISTANCE_METRES,
            max_link_distance_metres=MAX_LINK_DISTANCE_METRES,
            wind_speed_percentile_level=WIND_SPEED_PERCENTILE_LEVEL)
        self.assertTrue(this_column_name == REGRESSION_LABEL_COLUMN_NAME)

    def test_get_column_name_for_num_wind_obs(self):
        """Ensures correct output from get_column_name_for_num_wind_obs."""

        this_column_name = labels.get_column_name_for_num_wind_obs(
            min_lead_time_sec=MIN_LEAD_TIME_SEC,
            max_lead_time_sec=MAX_LEAD_TIME_SEC,
            min_link_distance_metres=MIN_LINK_DISTANCE_METRES,
            max_link_distance_metres=MAX_LINK_DISTANCE_METRES)
        self.assertTrue(this_column_name == NUM_WIND_OBS_COLUMN_NAME)

    def test_get_column_name_for_classification_label_wind(self):
        """Ensures correct output from get_column_name_for_classification_label.

        In this case, event type is wind speed.
        """

        this_column_name = labels.get_column_name_for_classification_label(
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

        this_column_name = labels.get_column_name_for_classification_label(
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

        this_parameter_dict = labels.column_name_to_label_params(
            REGRESSION_LABEL_COLUMN_NAME)
        self.assertTrue(this_parameter_dict == PARAM_DICT_FOR_REGRESSION_LABEL)


    def test_column_name_to_label_params_wind_classification(self):
        """Ensures correct output from column_name_to_label_params.

        In this case, learning goal is classification and event type is wind
        speed.
        """

        this_parameter_dict = labels.column_name_to_label_params(
            WIND_CLASSIFICATION_LABEL_COLUMN_NAME)

        self.assertTrue(numpy.array_equal(
            this_parameter_dict[labels.WIND_SPEED_CLASS_CUTOFFS_KEY],
            PARAM_DICT_FOR_WIND_CLASSIFICATION_LABEL[
                labels.WIND_SPEED_CLASS_CUTOFFS_KEY]))

        this_parameter_dict.pop(labels.WIND_SPEED_CLASS_CUTOFFS_KEY, None)
        PARAM_DICT_FOR_WIND_CLASSIFICATION_LABEL.pop(
            labels.WIND_SPEED_CLASS_CUTOFFS_KEY, None)

        self.assertTrue(
            this_parameter_dict == PARAM_DICT_FOR_WIND_CLASSIFICATION_LABEL)

    def test_column_name_to_label_params_tornado(self):
        """Ensures correct output from column_name_to_label_params.

        In this case, learning goal is classification and event type is tornado
        occurrence.
        """

        this_parameter_dict = labels.column_name_to_label_params(
            TORNADO_LABEL_COLUMN_NAME)
        self.assertTrue(this_parameter_dict == PARAM_DICT_FOR_TORNADO_LABEL)

    def test_column_name_to_num_classes_regression(self):
        """Ensures correct output from column_name_to_num_classes.

        In this case, learning goal is regression.
        """

        this_num_classes = labels.column_name_to_num_classes(
            REGRESSION_LABEL_COLUMN_NAME)
        self.assertTrue(this_num_classes is None)

    def test_column_name_to_num_classes_wind_with_dead_storms(self):
        """Ensures correct output from column_name_to_num_classes.

        In this case, learning goal is classification; event type is wind speed;
        and the "dead storm" label is included in the number of classes.
        """

        this_num_classes = labels.column_name_to_num_classes(
            WIND_CLASSIFICATION_LABEL_COLUMN_NAME, include_dead_storms=True)
        self.assertTrue(this_num_classes == len(WIND_CLASS_CUTOFFS_KT) + 2)

    def test_column_name_to_num_classes_wind_sans_dead_storms(self):
        """Ensures correct output from column_name_to_num_classes.

        In this case, learning goal is classification; event type is wind speed;
        and the "dead storm" label is *not* included in the number of classes.
        """

        this_num_classes = labels.column_name_to_num_classes(
            WIND_CLASSIFICATION_LABEL_COLUMN_NAME, include_dead_storms=False)
        self.assertTrue(this_num_classes == len(WIND_CLASS_CUTOFFS_KT) + 1)

    def test_column_name_to_num_classes_tornado(self):
        """Ensures correct output from column_name_to_num_classes.

        In this case, learning goal is classification and event type is tornado
        occurrence.
        """

        this_num_classes = labels.column_name_to_num_classes(
            TORNADO_LABEL_COLUMN_NAME)
        self.assertTrue(this_num_classes == 2)

    def test_column_name_to_num_classes_wind_zero_lead_time(self):
        """Ensures correct output from column_name_to_num_classes.

        In this case, learning goal is classification; event type is wind speed;
        and the lead-time range includes 0 seconds, so dead storms are
        impossible.
        """

        this_num_classes = labels.column_name_to_num_classes(
            WIND_CLASSIFN_LABEL_NAME_ZERO_LEAD_TIME, include_dead_storms=True)
        self.assertTrue(this_num_classes == len(WIND_CLASS_CUTOFFS_KT) + 1)

    def test_get_columns_with_labels_regression(self):
        """Ensures correct output from get_columns_with_labels.

        In this case, learning goal is regression.
        """

        these_column_names = labels.get_columns_with_labels(
            label_table=LABEL_TABLE,
            goal_string=labels.REGRESSION_GOAL_STRING,
            event_type_string=events2storms.WIND_EVENT_TYPE_STRING)
        self.assertTrue(these_column_names == [REGRESSION_LABEL_COLUMN_NAME])

    def test_get_columns_with_labels_wind_classification(self):
        """Ensures correct output from get_columns_with_labels.

        In this case, learning goal is classification and event type is wind
        speed.
        """

        these_column_names = labels.get_columns_with_labels(
            label_table=LABEL_TABLE,
            goal_string=labels.CLASSIFICATION_GOAL_STRING,
            event_type_string=events2storms.WIND_EVENT_TYPE_STRING)

        self.assertTrue(
            these_column_names == [WIND_CLASSIFICATION_LABEL_COLUMN_NAME])

    def test_get_columns_with_labels_tornado(self):
        """Ensures correct output from get_columns_with_labels.

        In this case, learning goal is classification and event type is tornado
        occurrence.
        """

        these_column_names = labels.get_columns_with_labels(
            label_table=LABEL_TABLE,
            goal_string=labels.CLASSIFICATION_GOAL_STRING,
            event_type_string=events2storms.TORNADO_EVENT_TYPE_STRING)
        self.assertTrue(these_column_names == [TORNADO_LABEL_COLUMN_NAME])

    def test_get_columns_with_num_wind_obs_regression(self):
        """Ensures correct output from get_columns_with_num_wind_obs.

        In this case, learning goal is regression.
        """

        these_column_names = labels.get_columns_with_num_wind_obs(
            label_table=LABEL_TABLE,
            label_column_names=[REGRESSION_LABEL_COLUMN_NAME])
        self.assertTrue(these_column_names == [NUM_WIND_OBS_COLUMN_NAME])

    def test_get_columns_with_num_wind_obs_classification(self):
        """Ensures correct output from get_columns_with_num_wind_obs.

        In this case, learning goal is classification.
        """

        these_column_names = labels.get_columns_with_num_wind_obs(
            label_table=LABEL_TABLE,
            label_column_names=[WIND_CLASSIFICATION_LABEL_COLUMN_NAME])
        self.assertTrue(these_column_names == [NUM_WIND_OBS_COLUMN_NAME])

    def test_find_label_file_one_time_wind(self):
        """Ensures correct output from find_label_file.

        In this case, event type is damaging wind and file contains data for one
        time step.
        """

        this_file_name = labels.find_label_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            event_type_string=events2storms.WIND_EVENT_TYPE_STRING,
            raise_error_if_missing=False, unix_time_sec=FILE_TIME_UNIX_SEC)
        self.assertTrue(this_file_name == WIND_LABELS_ONE_TIME_FILE_NAME)

    def test_find_label_file_one_time_tornado(self):
        """Ensures correct output from find_label_file.

        In this case, event type is tornado and file contains data for one time
        step.
        """

        this_file_name = labels.find_label_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            event_type_string=events2storms.TORNADO_EVENT_TYPE_STRING,
            raise_error_if_missing=False, unix_time_sec=FILE_TIME_UNIX_SEC)
        self.assertTrue(this_file_name == TORNADO_LABELS_ONE_TIME_FILE_NAME)

    def test_find_label_file_spc_date_wind(self):
        """Ensures correct output from find_label_file.

        In this case, event type is damaging wind and file contains data for one
        SPC date.
        """

        this_file_name = labels.find_label_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            event_type_string=events2storms.WIND_EVENT_TYPE_STRING,
            raise_error_if_missing=False, spc_date_string=FILE_SPC_DATE_STRING)
        self.assertTrue(this_file_name == WIND_LABELS_SPC_DATE_FILE_NAME)

    def test_find_label_file_spc_date_tornado(self):
        """Ensures correct output from find_label_file.

        In this case, event type is tornado and file contains data for one SPC
        date.
        """

        this_file_name = labels.find_label_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            event_type_string=events2storms.TORNADO_EVENT_TYPE_STRING,
            raise_error_if_missing=False, spc_date_string=FILE_SPC_DATE_STRING)
        self.assertTrue(this_file_name == TORNADO_LABELS_SPC_DATE_FILE_NAME)


if __name__ == '__main__':
    unittest.main()
