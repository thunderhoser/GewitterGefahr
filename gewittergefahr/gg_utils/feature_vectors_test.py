"""Unit tests for feature_vectors.py."""

import unittest
import numpy
import pandas
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import radar_statistics as radar_stats
from gewittergefahr.gg_utils import shape_statistics as shape_stats
from gewittergefahr.gg_utils import feature_vectors
from gewittergefahr.gg_utils import labels
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import link_events_to_storms as events2storms

TOLERANCE = 1e-6
KT_TO_METRES_PER_SECOND = 1.852 / 3.6

# Create table with radar statistic for each storm object.
NUM_STORM_OBJECTS = 25
STORM_IDS = ['storm{0:02d}'.format(s) for s in range(NUM_STORM_OBJECTS)]
STORM_TIMES_UNIX_SEC = numpy.linspace(
    0, 24000, num=NUM_STORM_OBJECTS, dtype=int)

RADAR_STATISTIC_NAME = radar_stats.radar_field_and_statistic_to_column_name(
    radar_field_name=radar_utils.VIL_NAME,
    radar_height_m_asl=250, statistic_name=radar_stats.STANDARD_DEVIATION_NAME)
RADAR_STATISTIC_VALUES = numpy.full(NUM_STORM_OBJECTS, 3.)
THIS_DICTIONARY = {
    tracking_utils.STORM_ID_COLUMN: STORM_IDS,
    tracking_utils.TIME_COLUMN: STORM_TIMES_UNIX_SEC,
    RADAR_STATISTIC_NAME: RADAR_STATISTIC_VALUES
}
RADAR_STATISTIC_TABLE = pandas.DataFrame.from_dict(THIS_DICTIONARY)

# Create table with shape statistic for each storm object.
SHAPE_STATISTIC_NAME = shape_stats.AREA_NAME
SHAPE_STATISTIC_VALUES = numpy.full(NUM_STORM_OBJECTS, 200.)
THIS_DICTIONARY = {
    tracking_utils.STORM_ID_COLUMN: STORM_IDS,
    tracking_utils.TIME_COLUMN: STORM_TIMES_UNIX_SEC,
    SHAPE_STATISTIC_NAME: SHAPE_STATISTIC_VALUES
}
SHAPE_STATISTIC_TABLE = pandas.DataFrame.from_dict(THIS_DICTIONARY)

# Create table with sounding statistic for each storm object.
SOUNDING_STATISTIC_NAME = 'storm_velocity_m_s01_magnitude'
SOUNDING_STATISTIC_VALUES = numpy.full(NUM_STORM_OBJECTS, 10.)
THIS_DICTIONARY = {
    tracking_utils.STORM_ID_COLUMN: STORM_IDS,
    tracking_utils.TIME_COLUMN: STORM_TIMES_UNIX_SEC,
    SOUNDING_STATISTIC_NAME: SOUNDING_STATISTIC_VALUES
}
SOUNDING_STATISTIC_TABLE = pandas.DataFrame.from_dict(THIS_DICTIONARY)

# Create table with wind-speed-based labels for each storm object.
MIN_LEAD_TIME_SEC = 2700
MAX_LEAD_TIME_SEC = 3600
MIN_LINK_DISTANCE_METRES = 1.
MAX_LINK_DISTANCE_METRES = 5000.
WIND_SPEED_PERCENTILE_LEVEL = 100.
WIND_SPEED_CLASS_CUTOFFS_KT = numpy.array([50.])

WIND_SPEED_REGRESSION_COLUMN_NAME = (
    labels.get_column_name_for_regression_label(
        min_lead_time_sec=MIN_LEAD_TIME_SEC,
        max_lead_time_sec=MAX_LEAD_TIME_SEC,
        min_link_distance_metres=MIN_LINK_DISTANCE_METRES,
        max_link_distance_metres=MAX_LINK_DISTANCE_METRES,
        wind_speed_percentile_level=WIND_SPEED_PERCENTILE_LEVEL))

WIND_SPEED_CLASSIFN_COLUMN_NAME = (
    labels.get_column_name_for_classification_label(
        min_lead_time_sec=MIN_LEAD_TIME_SEC,
        max_lead_time_sec=MAX_LEAD_TIME_SEC,
        min_link_distance_metres=MIN_LINK_DISTANCE_METRES,
        max_link_distance_metres=MAX_LINK_DISTANCE_METRES,
        event_type_string=events2storms.WIND_EVENT_TYPE_STRING,
        wind_speed_percentile_level=WIND_SPEED_PERCENTILE_LEVEL,
        wind_speed_class_cutoffs_kt=WIND_SPEED_CLASS_CUTOFFS_KT))

NUM_WIND_OBS_COLUMN_NAME = labels.get_column_name_for_num_wind_obs(
    min_lead_time_sec=MIN_LEAD_TIME_SEC,
    max_lead_time_sec=MAX_LEAD_TIME_SEC,
    min_link_distance_metres=MIN_LINK_DISTANCE_METRES,
    max_link_distance_metres=MAX_LINK_DISTANCE_METRES)

WIND_OBSERVATION_COUNTS = numpy.array(
    [0, 0, 0, 0, 0,
     26, 25, 24,
     3, 100, 30, 2, 50,
     75, 40, 30, 4,
     11, 33, 22,
     45, 25, 5,
     3, 13], dtype=int)
WIND_REGRESSION_LABELS_M_S01 = KT_TO_METRES_PER_SECOND * numpy.array(
    [0., 0., 0., 0., 0.,
     1., 3., 5.,
     10., 12.5, 15., 17.5, 19.9,
     20., 25., 27., 28.,
     30., 35., 39.9,
     40., 45., 49.9,
     50., 60.])

WIND_CLASSIFICATION_LABELS = numpy.full(NUM_STORM_OBJECTS, 0, dtype=int)
WIND_CLASSIFICATION_LABELS[-2:] = 1

THIS_DICTIONARY = {
    tracking_utils.STORM_ID_COLUMN: STORM_IDS,
    tracking_utils.TIME_COLUMN: STORM_TIMES_UNIX_SEC,
    events2storms.STORM_END_TIME_COLUMN: STORM_TIMES_UNIX_SEC,
    WIND_SPEED_REGRESSION_COLUMN_NAME: WIND_REGRESSION_LABELS_M_S01,
    WIND_SPEED_CLASSIFN_COLUMN_NAME: WIND_CLASSIFICATION_LABELS,
    NUM_WIND_OBS_COLUMN_NAME: WIND_OBSERVATION_COUNTS
}
STORM_TO_WINDS_TABLE = pandas.DataFrame.from_dict(THIS_DICTIONARY)

# Add polygons (storm boundaries) to table.
VERTEX_LATITUDES_DEG = numpy.array([53.4, 53.4, 53.6, 53.6, 53.5, 53.5, 53.4])
VERTEX_LONGITUDES_DEG = numpy.array(
    [246.4, 246.6, 246.6, 246.5, 246.5, 246.4, 246.4])

POLYGON_OBJECT_LATLNG = polygons.vertex_arrays_to_polygon_object(
    VERTEX_LONGITUDES_DEG, VERTEX_LATITUDES_DEG)
POLYGON_OBJECT_ARRAY_LATLNG = numpy.full(
    NUM_STORM_OBJECTS, POLYGON_OBJECT_LATLNG, dtype=object)

BUFFER_COLUMN_NAME = tracking_utils.distance_buffer_to_column_name(
    MIN_LINK_DISTANCE_METRES, MAX_LINK_DISTANCE_METRES)

THIS_DICTIONARY = {
    tracking_utils.POLYGON_OBJECT_LATLNG_COLUMN: POLYGON_OBJECT_ARRAY_LATLNG,
    BUFFER_COLUMN_NAME: POLYGON_OBJECT_ARRAY_LATLNG
}
STORM_TO_WINDS_TABLE = STORM_TO_WINDS_TABLE.assign(**THIS_DICTIONARY)

# The following constants are used to test _check_labels_in_table.
LABEL_COLUMN_NAMES = [
    WIND_SPEED_REGRESSION_COLUMN_NAME, WIND_SPEED_CLASSIFN_COLUMN_NAME]

# The following constants are used to test check_feature_table.
FEATURE_COLUMN_NAMES = [
    RADAR_STATISTIC_NAME, SHAPE_STATISTIC_NAME, SOUNDING_STATISTIC_NAME]

# The following constants are used to test join_features_and_label.
THIS_DICTIONARY = {
    tracking_utils.STORM_ID_COLUMN: STORM_IDS,
    tracking_utils.TIME_COLUMN: STORM_TIMES_UNIX_SEC,
    events2storms.STORM_END_TIME_COLUMN: STORM_TIMES_UNIX_SEC,
    WIND_SPEED_REGRESSION_COLUMN_NAME: WIND_REGRESSION_LABELS_M_S01,
    WIND_SPEED_CLASSIFN_COLUMN_NAME: WIND_CLASSIFICATION_LABELS,
    NUM_WIND_OBS_COLUMN_NAME: WIND_OBSERVATION_COUNTS,
    RADAR_STATISTIC_NAME: RADAR_STATISTIC_VALUES,
    SHAPE_STATISTIC_NAME: SHAPE_STATISTIC_VALUES,
    SOUNDING_STATISTIC_NAME: SOUNDING_STATISTIC_VALUES,
    tracking_utils.POLYGON_OBJECT_LATLNG_COLUMN: POLYGON_OBJECT_ARRAY_LATLNG,
    BUFFER_COLUMN_NAME: POLYGON_OBJECT_ARRAY_LATLNG
}

FEATURE_TABLE = pandas.DataFrame.from_dict(THIS_DICTIONARY)

INTEGER_AND_STRING_COLUMNS = [
    tracking_utils.STORM_ID_COLUMN, tracking_utils.TIME_COLUMN,
    events2storms.STORM_END_TIME_COLUMN, WIND_SPEED_CLASSIFN_COLUMN_NAME,
    NUM_WIND_OBS_COLUMN_NAME
]
POLYGON_COLUMNS = [
    tracking_utils.POLYGON_OBJECT_LATLNG_COLUMN, BUFFER_COLUMN_NAME]


def are_feature_tables_equal(feature_table1, feature_table2):
    """Determines whether or not two feature tables are equal.

    :param feature_table1: pandas DataFrame.
    :param feature_table2: pandas DataFrame.
    :return: are_tables_equal_flag: Boolean flag.
    """

    num_rows = len(feature_table1.index)
    if set(list(feature_table1)) != set(list(feature_table2)):
        return False

    for this_column_name in list(feature_table1):
        if this_column_name in INTEGER_AND_STRING_COLUMNS:
            if not numpy.array_equal(feature_table1[this_column_name].values,
                                     feature_table2[this_column_name].values):
                return False

        elif this_column_name in POLYGON_COLUMNS:
            for i in range(num_rows):
                if not feature_table1[this_column_name].values[i].equals(
                        feature_table2[this_column_name].values[i]):
                    return False
        else:
            if not numpy.allclose(feature_table1[this_column_name].values,
                                  feature_table2[this_column_name].values,
                                  atol=TOLERANCE):
                return False

    return True


class FeatureVectorsTests(unittest.TestCase):
    """Each method is a unit test for feature_vectors.py."""

    def test_check_labels_in_table(self):
        """Ensures correct output from _check_labels_in_table."""

        these_label_column_names, these_num_wind_obs_column_names = (
            feature_vectors._check_labels_in_table(FEATURE_TABLE))

        self.assertTrue(
            set(these_label_column_names) == set(LABEL_COLUMN_NAMES))
        self.assertTrue(
            these_num_wind_obs_column_names == [NUM_WIND_OBS_COLUMN_NAME])

    def test_check_feature_table(self):
        """Ensures correct output from check_feature_table."""

        (these_feature_column_names,
         these_label_column_names,
         these_num_wind_obs_column_names) = feature_vectors.check_feature_table(
             FEATURE_TABLE)

        self.assertTrue(
            set(these_feature_column_names) == set(FEATURE_COLUMN_NAMES))
        self.assertTrue(
            set(these_label_column_names) == set(LABEL_COLUMN_NAMES))
        self.assertTrue(
            these_num_wind_obs_column_names == [NUM_WIND_OBS_COLUMN_NAME])

    def test_join_features_and_labels(self):
        """Ensures correct output from join_features_and_labels."""

        this_feature_table = feature_vectors.join_features_and_labels(
            storm_to_events_table=STORM_TO_WINDS_TABLE,
            radar_statistic_table=RADAR_STATISTIC_TABLE,
            shape_statistic_table=SHAPE_STATISTIC_TABLE,
            sounding_statistic_table=SOUNDING_STATISTIC_TABLE)

        self.assertTrue(are_feature_tables_equal(
            this_feature_table, FEATURE_TABLE))


if __name__ == '__main__':
    unittest.main()
