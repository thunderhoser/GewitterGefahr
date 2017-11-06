"""Unit tests for feature_vectors_test.py."""

import unittest
import numpy
import pandas
from gewittergefahr.gg_io import radar_io
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import radar_statistics as radar_stats
from gewittergefahr.gg_utils import shape_statistics as shape_stats
from gewittergefahr.gg_utils import feature_vectors
from gewittergefahr.gg_utils import labels
from gewittergefahr.linkage import storm_to_winds

TOLERANCE = 1e-6
KT_TO_METRES_PER_SECOND = 1.852 / 3.6

# Create table with radar statistic (standard deviation of VIL [vertically
# integrated liquid]) for each storm object.
NUM_STORM_OBJECTS = 25
STORM_IDS = ['storm{0:02d}'.format(s) for s in range(NUM_STORM_OBJECTS)]
STORM_OBJECT_TIMES_UNIX_SEC = numpy.linspace(
    0, 24000, num=NUM_STORM_OBJECTS, dtype=int)

RADAR_STATISTIC_NAME = radar_stats._radar_field_and_statistic_to_column_name(
    radar_field_name=radar_io.VIL_NAME,
    statistic_name=radar_stats.STANDARD_DEVIATION_NAME)
RADAR_STATISTIC_VALUES = numpy.full(NUM_STORM_OBJECTS, 3.)
RADAR_STATISTIC_DICT = {
    tracking_io.STORM_ID_COLUMN: STORM_IDS,
    tracking_io.TIME_COLUMN: STORM_OBJECT_TIMES_UNIX_SEC,
    RADAR_STATISTIC_NAME: RADAR_STATISTIC_VALUES
}
RADAR_STATISTIC_TABLE = pandas.DataFrame.from_dict(RADAR_STATISTIC_DICT)

# Create table with shape statistic (area) for each storm object.
SHAPE_STATISTIC_NAME = shape_stats.AREA_NAME
SHAPE_STATISTIC_VALUES = numpy.full(NUM_STORM_OBJECTS, 200.)
SHAPE_STATISTIC_DICT = {
    tracking_io.STORM_ID_COLUMN: STORM_IDS,
    tracking_io.TIME_COLUMN: STORM_OBJECT_TIMES_UNIX_SEC,
    SHAPE_STATISTIC_NAME: SHAPE_STATISTIC_VALUES
}
SHAPE_STATISTIC_TABLE = pandas.DataFrame.from_dict(SHAPE_STATISTIC_DICT)

# Create table with sounding index (storm speed) for each storm object.
SOUNDING_INDEX_NAME = 'storm_velocity_m_s01_magnitude'
SOUNDING_INDEX_VALUES = numpy.full(NUM_STORM_OBJECTS, 10.)
SOUNDING_INDEX_DICT = {
    tracking_io.STORM_ID_COLUMN: STORM_IDS,
    tracking_io.TIME_COLUMN: STORM_OBJECT_TIMES_UNIX_SEC,
    SOUNDING_INDEX_NAME: SOUNDING_INDEX_VALUES
}
SOUNDING_INDEX_TABLE = pandas.DataFrame.from_dict(SOUNDING_INDEX_DICT)

# Create table with regression and classification label for each storm object.
MIN_LEAD_TIME_SEC = 2700
MAX_LEAD_TIME_SEC = 3600
MIN_DISTANCE_METRES = 1.
MAX_DISTANCE_METRES = 5000.
PERCENTILE_LEVEL = 100.
CLASS_CUTOFFS_KT = numpy.array([50.])

REGRESSION_LABEL_COLUMN_NAME = labels.get_column_name_for_regression_label(
    min_lead_time_sec=MIN_LEAD_TIME_SEC, max_lead_time_sec=MAX_LEAD_TIME_SEC,
    min_distance_metres=MIN_DISTANCE_METRES,
    max_distance_metres=MAX_DISTANCE_METRES, percentile_level=PERCENTILE_LEVEL)

CLASSIFICATION_LABEL_COLUMN_NAME = (
    labels.get_column_name_for_classification_label(
        min_lead_time_sec=MIN_LEAD_TIME_SEC,
        max_lead_time_sec=MAX_LEAD_TIME_SEC,
        min_distance_metres=MIN_DISTANCE_METRES,
        max_distance_metres=MAX_DISTANCE_METRES,
        percentile_level=PERCENTILE_LEVEL, class_cutoffs_kt=CLASS_CUTOFFS_KT))

DEAD_STORM_OBJECT_INDICES = numpy.linspace(0, 4, num=5, dtype=int)
LIVE_STORM_OBJECT_INDICES = numpy.linspace(5, 24, num=20, dtype=int)

REMAINING_STORM_LIFETIMES_SEC = numpy.full(NUM_STORM_OBJECTS, 3600, dtype=int)
REMAINING_STORM_LIFETIMES_SEC[DEAD_STORM_OBJECT_INDICES] = 1800
STORM_END_TIMES_UNIX_SEC = (
    STORM_OBJECT_TIMES_UNIX_SEC + REMAINING_STORM_LIFETIMES_SEC)

NUM_OBSERVATIONS_BY_STORM_OBJECT = numpy.array([0, 0, 0, 0, 0,
                                                26, 25, 24,
                                                3, 100, 30, 2, 50,
                                                75, 40, 30, 4,
                                                11, 33, 22,
                                                45, 25, 5,
                                                3, 13], dtype=int)
REGRESSION_LABELS_M_S01 = numpy.array([0., 0., 0., 0., 0.,
                                       1., 3., 5.,
                                       10., 12.5, 15., 17.5, 19.9,
                                       20., 25., 27., 28.,
                                       30., 35., 39.9,
                                       40., 45., 49.9,
                                       50., 60.]) * KT_TO_METRES_PER_SECOND

CLASSIFICATION_LABELS = numpy.full(NUM_STORM_OBJECTS, 0, dtype=int)
CLASSIFICATION_LABELS[-2:] = 1

STORM_TO_WINDS_DICT = {
    tracking_io.STORM_ID_COLUMN: STORM_IDS,
    tracking_io.TIME_COLUMN: STORM_OBJECT_TIMES_UNIX_SEC,
    storm_to_winds.END_TIME_COLUMN: STORM_END_TIMES_UNIX_SEC,
    labels.NUM_OBSERVATIONS_FOR_LABEL_COLUMN: NUM_OBSERVATIONS_BY_STORM_OBJECT,
    REGRESSION_LABEL_COLUMN_NAME: REGRESSION_LABELS_M_S01,
    CLASSIFICATION_LABEL_COLUMN_NAME: CLASSIFICATION_LABELS
}
STORM_TO_WINDS_TABLE = pandas.DataFrame.from_dict(STORM_TO_WINDS_DICT)

# The following constants are used to test _select_storms_uniformly_by_category
# and sample_by_uniform_wind_speed.
CATEGORIES_FOR_UNIFORM_SAMPLING = numpy.array([-1, -1, -1, -1, -1,
                                               0, 0, 0,
                                               1, 1, 1, 1, 1,
                                               2, 2, 2, 2,
                                               3, 3, 3,
                                               4, 4, 4,
                                               5, 5], dtype=int)
LIVE_SELECTED_INDICES_FOR_UNIF_SAMPLING = numpy.array(
    [0, 1, 5, 6, 9, 12, 13, 14, 18, 19, 20, 21, 23, 24])
LIVE_SELECTED_INDICES_FOR_MIN_OBS_SAMPLING = numpy.array(
    [5, 6, 9, 10, 12, 13, 14, 15, 18, 20, 21])
LIVE_SELECTED_INDICES_FOR_MIN_OBS_PLUS_SAMPLING = numpy.array(
    [5, 6, 9, 10, 12, 13, 14, 15, 18, 20, 21, 23, 24])
NUM_DEAD_STORM_OBJECTS_TO_SELECT = 3

# The following constants are used to test check_feature_table.
FEATURE_COLUMN_NAMES = [
    RADAR_STATISTIC_NAME, SHAPE_STATISTIC_NAME, SOUNDING_INDEX_NAME]

# The following constants are used to test
# join_features_and_label_for_storm_objects.
FEATURE_DICT = {
    tracking_io.STORM_ID_COLUMN: STORM_IDS,
    tracking_io.TIME_COLUMN: STORM_OBJECT_TIMES_UNIX_SEC,
    storm_to_winds.END_TIME_COLUMN: STORM_END_TIMES_UNIX_SEC,
    labels.NUM_OBSERVATIONS_FOR_LABEL_COLUMN: NUM_OBSERVATIONS_BY_STORM_OBJECT,
    REGRESSION_LABEL_COLUMN_NAME: REGRESSION_LABELS_M_S01,
    CLASSIFICATION_LABEL_COLUMN_NAME: CLASSIFICATION_LABELS,
    RADAR_STATISTIC_NAME: RADAR_STATISTIC_VALUES,
    SHAPE_STATISTIC_NAME: SHAPE_STATISTIC_VALUES,
    SOUNDING_INDEX_NAME: SOUNDING_INDEX_VALUES
}

FEATURE_TABLE = pandas.DataFrame.from_dict(FEATURE_DICT)
INTEGER_AND_STRING_COLUMNS = [
    storm_to_winds.END_TIME_COLUMN, tracking_io.STORM_ID_COLUMN,
    tracking_io.TIME_COLUMN, CLASSIFICATION_LABEL_COLUMN_NAME]

# The following constants are used to test find_unsampled_file_one_time.
FILE_TIME_UNIX_SEC = 1509936790  # 025310 6 Nov 2017
FILE_SPC_DATE_UNIX_SEC = 1509936790
TOP_DIRECTORY_NAME = 'feature_vectors'
FEATURE_FILE_NAME_ONE_TIME = (
    'feature_vectors/20171105/features_2017-11-06-025310.p')

# The following constants are used to test find_unsampled_file_time_period.
FILE_START_TIME_UNIX_SEC = 1509883200  # 1200 UTC 5 Nov 2017
FILE_END_TIME_UNIX_SEC = 1509969599  # 115959 UTC 6 Nov 2017
FEATURE_FILE_NAME_TIME_PERIOD = (
    'feature_vectors/features_2017-11-05-120000_2017-11-06-115959.p')


class FeatureVectorsTests(unittest.TestCase):
    """Each method is a unit test for feature_vectors_test.py."""

    def test_find_live_and_dead_storms(self):
        """Ensures correct output from _find_live_and_dead_storms."""

        these_live_indices, these_dead_indices = (
            feature_vectors._find_live_and_dead_storms(FEATURE_TABLE))
        self.assertTrue(numpy.array_equal(
            these_live_indices, LIVE_STORM_OBJECT_INDICES))
        self.assertTrue(numpy.array_equal(
            these_dead_indices, DEAD_STORM_OBJECT_INDICES))

    def test_select_dead_storms(self):
        """Ensures correct output from _select_dead_storms."""

        these_selected_indices = feature_vectors._select_dead_storms(
            live_indices=LIVE_STORM_OBJECT_INDICES,
            dead_indices=DEAD_STORM_OBJECT_INDICES,
            live_selected_indices=LIVE_SELECTED_INDICES_FOR_MIN_OBS_SAMPLING)

        expected_num_indices = NUM_DEAD_STORM_OBJECTS_TO_SELECT + len(
            LIVE_SELECTED_INDICES_FOR_MIN_OBS_SAMPLING)
        self.assertTrue(len(these_selected_indices) == expected_num_indices)

        these_selected_indices = set(these_selected_indices)
        for this_index in LIVE_SELECTED_INDICES_FOR_MIN_OBS_SAMPLING:
            self.assertTrue(this_index in these_selected_indices)

        this_num_dead_selected = 0
        for this_index in DEAD_STORM_OBJECT_INDICES:
            if this_index in these_selected_indices:
                this_num_dead_selected += 1

        self.assertTrue(
            this_num_dead_selected == NUM_DEAD_STORM_OBJECTS_TO_SELECT)

    def test_select_storms_uniformly_by_category(self):
        """Ensures correct output from _select_storms_uniformly_by_category."""

        these_selected_indices = (
            feature_vectors._select_storms_uniformly_by_category(
                CATEGORIES_FOR_UNIFORM_SAMPLING,
                NUM_OBSERVATIONS_BY_STORM_OBJECT))
        self.assertTrue(numpy.array_equal(
            these_selected_indices, LIVE_SELECTED_INDICES_FOR_UNIF_SAMPLING))

    def test_check_feature_table(self):
        """Ensures correct output from check_feature_table."""

        (these_feature_column_names,
         this_regression_label_column_name,
         this_classification_label_column_name) = (
            feature_vectors.check_feature_table(
                FEATURE_TABLE, require_storm_objects=True))

        self.assertTrue(set(these_feature_column_names) ==
                        set(FEATURE_COLUMN_NAMES))
        self.assertTrue(this_regression_label_column_name ==
                        REGRESSION_LABEL_COLUMN_NAME)
        self.assertTrue(this_classification_label_column_name ==
                        CLASSIFICATION_LABEL_COLUMN_NAME)

    def test_join_features_and_label_for_storm_objects(self):
        """Ensures correctness of join_features_and_label_for_storm_objects."""

        this_feature_table = (
            feature_vectors.join_features_and_label_for_storm_objects(
                radar_statistic_table=RADAR_STATISTIC_TABLE,
                shape_statistic_table=SHAPE_STATISTIC_TABLE,
                sounding_index_table=SOUNDING_INDEX_TABLE,
                storm_to_winds_table=STORM_TO_WINDS_TABLE,
                label_column_name=CLASSIFICATION_LABEL_COLUMN_NAME))

        self.assertTrue(set(list(this_feature_table)) ==
                        set(list(FEATURE_TABLE)))

        for this_column_name in list(this_feature_table):
            if this_column_name in INTEGER_AND_STRING_COLUMNS:
                self.assertTrue(numpy.array_equal(
                    this_feature_table[this_column_name].values,
                    FEATURE_TABLE[this_column_name].values))
            else:
                self.assertTrue(numpy.allclose(
                    this_feature_table[this_column_name].values,
                    FEATURE_TABLE[this_column_name].values, atol=TOLERANCE))

    def test_sample_by_min_observations_return_table(self):
        """Ensures correct output from sample_by_min_observations.

        In this case, return_table = True.
        """

        this_feature_table, _ = feature_vectors.sample_by_min_observations(
            FEATURE_TABLE, return_table=True)
        this_feature_table = this_feature_table.iloc[
                             NUM_DEAD_STORM_OBJECTS_TO_SELECT:]
        this_expected_feature_table = FEATURE_TABLE.iloc[
            LIVE_SELECTED_INDICES_FOR_MIN_OBS_SAMPLING]

        self.assertTrue(set(list(this_feature_table)) ==
                        set(list(this_expected_feature_table)))

        for this_column_name in list(this_feature_table):
            if this_column_name in INTEGER_AND_STRING_COLUMNS:
                self.assertTrue(numpy.array_equal(
                    this_feature_table[this_column_name].values,
                    this_expected_feature_table[this_column_name].values))
            else:
                self.assertTrue(numpy.allclose(
                    this_feature_table[this_column_name].values,
                    this_expected_feature_table[this_column_name].values,
                    atol=TOLERANCE))

    def test_sample_by_min_observations_return_dict(self):
        """Ensures correct output from sample_by_min_observations.

        In this case, return_table = False.
        """

        _, this_metadata_dict = feature_vectors.sample_by_min_observations(
            FEATURE_TABLE, return_table=False)

        self.assertTrue(numpy.array_equal(
            this_metadata_dict[feature_vectors.LIVE_SELECTED_INDICES_KEY],
            LIVE_SELECTED_INDICES_FOR_MIN_OBS_SAMPLING))
        self.assertTrue(
            this_metadata_dict[feature_vectors.NUM_LIVE_STORMS_KEY] ==
            len(LIVE_STORM_OBJECT_INDICES))
        self.assertTrue(
            this_metadata_dict[feature_vectors.NUM_DEAD_STORMS_KEY] ==
            len(DEAD_STORM_OBJECT_INDICES))

    def test_sample_by_min_observations_plus_return_table(self):
        """Ensures correct output from sample_by_min_observations_plus.

        In this case, return_table = True.
        """

        this_feature_table, _ = feature_vectors.sample_by_min_observations_plus(
            FEATURE_TABLE, return_table=True)
        this_feature_table = this_feature_table.iloc[
                             NUM_DEAD_STORM_OBJECTS_TO_SELECT:]
        this_expected_feature_table = FEATURE_TABLE.iloc[
            LIVE_SELECTED_INDICES_FOR_MIN_OBS_PLUS_SAMPLING]

        self.assertTrue(set(list(this_feature_table)) ==
                        set(list(this_expected_feature_table)))

        for this_column_name in list(this_feature_table):
            if this_column_name in INTEGER_AND_STRING_COLUMNS:
                self.assertTrue(numpy.array_equal(
                    this_feature_table[this_column_name].values,
                    this_expected_feature_table[this_column_name].values))
            else:
                self.assertTrue(numpy.allclose(
                    this_feature_table[this_column_name].values,
                    this_expected_feature_table[this_column_name].values,
                    atol=TOLERANCE))

    def test_sample_by_min_observations_plus_return_dict(self):
        """Ensures correct output from sample_by_min_observations_plus.

        In this case, return_table = False.
        """

        _, this_metadata_dict = feature_vectors.sample_by_min_observations_plus(
            FEATURE_TABLE, return_table=False)

        self.assertTrue(numpy.array_equal(
            this_metadata_dict[feature_vectors.LIVE_SELECTED_INDICES_KEY],
            LIVE_SELECTED_INDICES_FOR_MIN_OBS_PLUS_SAMPLING))
        self.assertTrue(
            this_metadata_dict[feature_vectors.NUM_LIVE_STORMS_KEY] ==
            len(LIVE_STORM_OBJECT_INDICES))
        self.assertTrue(
            this_metadata_dict[feature_vectors.NUM_DEAD_STORMS_KEY] ==
            len(DEAD_STORM_OBJECT_INDICES))

    def test_sample_by_uniform_wind_speed_return_table(self):
        """Ensures correct output from sample_by_uniform_wind_speed.

        In this case, return_table = True.
        """

        this_feature_table, _ = feature_vectors.sample_by_uniform_wind_speed(
            FEATURE_TABLE, return_table=True)
        this_expected_feature_table = FEATURE_TABLE.iloc[
            LIVE_SELECTED_INDICES_FOR_UNIF_SAMPLING]

        self.assertTrue(set(list(this_feature_table)) ==
                        set(list(this_expected_feature_table)))

        for this_column_name in list(this_feature_table):
            if this_column_name in INTEGER_AND_STRING_COLUMNS:
                self.assertTrue(numpy.array_equal(
                    this_feature_table[this_column_name].values,
                    this_expected_feature_table[this_column_name].values))
            else:
                self.assertTrue(numpy.allclose(
                    this_feature_table[this_column_name].values,
                    this_expected_feature_table[this_column_name].values,
                    atol=TOLERANCE))

    def test_sample_by_uniform_wind_speed_return_dict(self):
        """Ensures correct output from sample_by_uniform_wind_speed.

        In this case, return_table = False.
        """

        _, this_metadata_dict = feature_vectors.sample_by_uniform_wind_speed(
            FEATURE_TABLE, return_table=False)

        self.assertTrue(numpy.array_equal(
            this_metadata_dict[
                feature_vectors.SPEED_CATEGORY_KEY_FOR_UNIFORM_SAMPLING],
            CATEGORIES_FOR_UNIFORM_SAMPLING))
        self.assertTrue(numpy.array_equal(
            this_metadata_dict[feature_vectors.NUM_OBSERVATIONS_KEY],
            NUM_OBSERVATIONS_BY_STORM_OBJECT))

    def test_find_unsampled_file_one_time(self):
        """Ensures correct output from find_unsampled_file_one_time."""

        this_file_name = feature_vectors.find_unsampled_file_one_time(
            unix_time_sec=FILE_TIME_UNIX_SEC,
            spc_date_unix_sec=FILE_SPC_DATE_UNIX_SEC,
            top_directory_name=TOP_DIRECTORY_NAME, raise_error_if_missing=False)
        self.assertTrue(this_file_name == FEATURE_FILE_NAME_ONE_TIME)

    def test_find_unsampled_file_time_period(self):
        """Ensures correct output from find_unsampled_file_time_period."""

        this_file_name = feature_vectors.find_unsampled_file_time_period(
            FILE_START_TIME_UNIX_SEC, FILE_END_TIME_UNIX_SEC,
            directory_name=TOP_DIRECTORY_NAME, raise_error_if_missing=False)
        self.assertTrue(this_file_name == FEATURE_FILE_NAME_TIME_PERIOD)


if __name__ == '__main__':
    unittest.main()
