"""Unit tests for feature_vectors_test.py."""

import unittest
import copy
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

# Create table with radar statistic (standard deviation of VIL [vertically
# integrated liquid]) for each storm object.
NUM_STORM_OBJECTS = 25
STORM_IDS = ['storm{0:02d}'.format(s) for s in range(NUM_STORM_OBJECTS)]
STORM_OBJECT_TIMES_UNIX_SEC = numpy.linspace(
    0, 24000, num=NUM_STORM_OBJECTS, dtype=int)

RADAR_STATISTIC_NAME = radar_stats.radar_field_and_statistic_to_column_name(
    radar_field_name=radar_utils.VIL_NAME,
    radar_height_m_asl=250, statistic_name=radar_stats.STANDARD_DEVIATION_NAME)
RADAR_STATISTIC_VALUES = numpy.full(NUM_STORM_OBJECTS, 3.)
RADAR_STATISTIC_DICT = {
    tracking_utils.STORM_ID_COLUMN: STORM_IDS,
    tracking_utils.TIME_COLUMN: STORM_OBJECT_TIMES_UNIX_SEC,
    RADAR_STATISTIC_NAME: RADAR_STATISTIC_VALUES
}
RADAR_STATISTIC_TABLE = pandas.DataFrame.from_dict(RADAR_STATISTIC_DICT)

# Create table with shape statistic (area) for each storm object.
SHAPE_STATISTIC_NAME = shape_stats.AREA_NAME
SHAPE_STATISTIC_VALUES = numpy.full(NUM_STORM_OBJECTS, 200.)
SHAPE_STATISTIC_DICT = {
    tracking_utils.STORM_ID_COLUMN: STORM_IDS,
    tracking_utils.TIME_COLUMN: STORM_OBJECT_TIMES_UNIX_SEC,
    SHAPE_STATISTIC_NAME: SHAPE_STATISTIC_VALUES
}
SHAPE_STATISTIC_TABLE = pandas.DataFrame.from_dict(SHAPE_STATISTIC_DICT)

# Create table with sounding statistic (storm speed) for each storm object.
SOUNDING_STAT_NAME = 'storm_velocity_m_s01_magnitude'
SOUNDING_STAT_VALUES = numpy.full(NUM_STORM_OBJECTS, 10.)
SOUNDING_STAT_DICT = {
    tracking_utils.STORM_ID_COLUMN: STORM_IDS,
    tracking_utils.TIME_COLUMN: STORM_OBJECT_TIMES_UNIX_SEC,
    SOUNDING_STAT_NAME: SOUNDING_STAT_VALUES
}
SOUNDING_STAT_TABLE = pandas.DataFrame.from_dict(SOUNDING_STAT_DICT)

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
    tracking_utils.STORM_ID_COLUMN: STORM_IDS,
    tracking_utils.TIME_COLUMN: STORM_OBJECT_TIMES_UNIX_SEC,
    events2storms.STORM_END_TIME_COLUMN: STORM_END_TIMES_UNIX_SEC,
    labels.NUM_OBSERVATIONS_FOR_LABEL_COLUMN: NUM_OBSERVATIONS_BY_STORM_OBJECT,
    REGRESSION_LABEL_COLUMN_NAME: REGRESSION_LABELS_M_S01,
    CLASSIFICATION_LABEL_COLUMN_NAME: CLASSIFICATION_LABELS
}
STORM_TO_WINDS_TABLE = pandas.DataFrame.from_dict(STORM_TO_WINDS_DICT)

# Add polygons (storm outlines) to table.
VERTEX_LATITUDES_DEG = numpy.array([53.4, 53.4, 53.6, 53.6, 53.5, 53.5, 53.4])
VERTEX_LONGITUDES_DEG = numpy.array(
    [246.4, 246.6, 246.6, 246.5, 246.5, 246.4, 246.4])

POLYGON_OBJECT_LATLNG = polygons.vertex_arrays_to_polygon_object(
    VERTEX_LONGITUDES_DEG, VERTEX_LATITUDES_DEG)
POLYGON_OBJECT_ARRAY_LATLNG = numpy.full(
    NUM_STORM_OBJECTS, POLYGON_OBJECT_LATLNG, dtype=object)

THIS_ARGUMENT_DICT = {
    tracking_utils.POLYGON_OBJECT_LATLNG_COLUMN: POLYGON_OBJECT_ARRAY_LATLNG}
STORM_TO_WINDS_TABLE = STORM_TO_WINDS_TABLE.assign(**THIS_ARGUMENT_DICT)

# The following constants are used to test _select_storms_uniformly_by_category
# and sample_by_uniform_wind_speed.
CUTOFFS_FOR_UNIFORM_SAMPLING_M_S01 = (
    KT_TO_METRES_PER_SECOND * numpy.array([10., 20., 30., 40., 50.]))
CATEGORIES_FOR_UNIFORM_SAMPLING = numpy.array([-1, -1, -1, -1, -1,
                                               0, 0, 0,
                                               1, 1, 1, 1, 1,
                                               2, 2, 2, 2,
                                               3, 3, 3,
                                               4, 4, 4,
                                               5, 5], dtype=int)
LIVE_SELECTED_INDICES_FOR_UNIF_SAMPLING = numpy.array(
    [0, 1, 5, 6, 9, 12, 13, 14, 18, 19, 20, 21, 23, 24])

MIN_OBSERVATIONS_FOR_SAMPLING = 25
NUM_DEAD_STORM_OBJECTS_TO_SELECT = 3

LIVE_SELECTED_INDICES_FOR_MIN_OBS_SAMPLING = numpy.array(
    [5, 6, 9, 10, 12, 13, 14, 15, 18, 20, 21])
LIVE_SELECTED_INDICES_FOR_MIN_OBS_PLUS_SAMPLING = numpy.array(
    [5, 6, 9, 10, 12, 13, 14, 15, 18, 20, 21, 23, 24])
LIVE_SELECTED_INDICES_FOR_MIN_DENSITY_SAMPLING = copy.deepcopy(
    LIVE_SELECTED_INDICES_FOR_MIN_OBS_SAMPLING)
LIVE_SELECTED_INDICES_FOR_MIN_DENSITY_PLUS_SAMPLING = copy.deepcopy(
    LIVE_SELECTED_INDICES_FOR_MIN_OBS_PLUS_SAMPLING)

# The following constants are used to test _get_observation_densities.
POLYGON_OBJECT_XY, THIS_PROJECTION_OBJECT = polygons.project_latlng_to_xy(
    POLYGON_OBJECT_LATLNG)
VERTEX_DICT_XY = polygons.polygon_object_to_vertex_arrays(POLYGON_OBJECT_XY)
BUFFERED_POLYGON_OBJECT_XY = polygons.buffer_simple_polygon(
    VERTEX_DICT_XY[polygons.EXTERIOR_X_COLUMN],
    VERTEX_DICT_XY[polygons.EXTERIOR_Y_COLUMN],
    min_buffer_dist_metres=MIN_DISTANCE_METRES,
    max_buffer_dist_metres=MAX_DISTANCE_METRES)

BUFFER_AREA_METRES2 = BUFFERED_POLYGON_OBJECT_XY.area
OBSERVATION_DENSITY_BY_STORM_OBJECT_M02 = (
    NUM_OBSERVATIONS_BY_STORM_OBJECT / BUFFER_AREA_METRES2)
MIN_OBS_DENSITY_FOR_SAMPLING_M02 = (
    MIN_OBSERVATIONS_FOR_SAMPLING / BUFFER_AREA_METRES2)

BUFFERED_POLYGON_OBJECT_LATLNG = polygons.project_xy_to_latlng(
    BUFFERED_POLYGON_OBJECT_XY, THIS_PROJECTION_OBJECT)
BUFFERED_POLYGON_OBJECT_ARRAY_LATLNG = numpy.full(
    NUM_STORM_OBJECTS, BUFFERED_POLYGON_OBJECT_LATLNG, dtype=object)

BUFFER_COLUMN_NAME = tracking_utils.distance_buffer_to_column_name(
    MIN_DISTANCE_METRES, MAX_DISTANCE_METRES)
THIS_ARGUMENT_DICT = {BUFFER_COLUMN_NAME: BUFFERED_POLYGON_OBJECT_ARRAY_LATLNG}
STORM_TO_WINDS_TABLE = STORM_TO_WINDS_TABLE.assign(**THIS_ARGUMENT_DICT)

# The following constants are used to test
# _indices_from_file_specific_to_overall and
# _indices_from_overall_to_file_specific.
NUM_OBJECTS_BY_FILE = numpy.array([50, 100, 0, 3, 75, 9], dtype=int)
INDICES_BY_FILE = [None] * len(NUM_OBJECTS_BY_FILE)
INDICES_BY_FILE[0] = numpy.array([0, 24, 25, 49], dtype=int)
INDICES_BY_FILE[1] = numpy.array([0, 49, 50, 99], dtype=int)
INDICES_BY_FILE[2] = numpy.array([], dtype=int)
INDICES_BY_FILE[3] = numpy.array([0, 1, 2], dtype=int)
INDICES_BY_FILE[4] = numpy.array([0, 37, 74], dtype=int)
INDICES_BY_FILE[5] = numpy.array([0, 4, 8], dtype=int)

OVERALL_INDICES = numpy.array(
    [0, 24, 25, 49, 50, 99, 100, 149, 150, 151, 152, 153, 190, 227, 228, 232,
     236], dtype=int)

# The following constants are used to test check_feature_table.
FEATURE_COLUMN_NAMES = [
    RADAR_STATISTIC_NAME, SHAPE_STATISTIC_NAME, SOUNDING_STAT_NAME]

# The following constants are used to test
# join_features_and_label_for_storm_objects.
FEATURE_DICT = {
    tracking_utils.STORM_ID_COLUMN: STORM_IDS,
    tracking_utils.TIME_COLUMN: STORM_OBJECT_TIMES_UNIX_SEC,
    events2storms.STORM_END_TIME_COLUMN: STORM_END_TIMES_UNIX_SEC,
    labels.NUM_OBSERVATIONS_FOR_LABEL_COLUMN: NUM_OBSERVATIONS_BY_STORM_OBJECT,
    tracking_utils.POLYGON_OBJECT_LATLNG_COLUMN: POLYGON_OBJECT_ARRAY_LATLNG,
    BUFFER_COLUMN_NAME: BUFFERED_POLYGON_OBJECT_ARRAY_LATLNG,
    REGRESSION_LABEL_COLUMN_NAME: REGRESSION_LABELS_M_S01,
    CLASSIFICATION_LABEL_COLUMN_NAME: CLASSIFICATION_LABELS,
    RADAR_STATISTIC_NAME: RADAR_STATISTIC_VALUES,
    SHAPE_STATISTIC_NAME: SHAPE_STATISTIC_VALUES,
    SOUNDING_STAT_NAME: SOUNDING_STAT_VALUES
}

FEATURE_TABLE = pandas.DataFrame.from_dict(FEATURE_DICT)
INTEGER_AND_STRING_COLUMNS = [
    events2storms.STORM_END_TIME_COLUMN, tracking_utils.STORM_ID_COLUMN,
    tracking_utils.TIME_COLUMN, CLASSIFICATION_LABEL_COLUMN_NAME]
POLYGON_COLUMNS = [
    tracking_utils.POLYGON_OBJECT_LATLNG_COLUMN, BUFFER_COLUMN_NAME]

# The following constants are used to test find_unsampled_file_one_time.
FILE_TIME_UNIX_SEC = 1509936790  # 025310 6 Nov 2017
FILE_SPC_DATE_STRING = '20171105'
TOP_DIRECTORY_NAME = 'feature_vectors'
FEATURE_FILE_NAME_ONE_TIME = (
    'feature_vectors/20171105/features_2017-11-06-025310.p')

# The following constants are used to test find_unsampled_file_time_period.
FILE_START_TIME_UNIX_SEC = 1509883200  # 1200 UTC 5 Nov 2017
FILE_END_TIME_UNIX_SEC = 1509969599  # 115959 UTC 6 Nov 2017
FEATURE_FILE_NAME_TIME_PERIOD = (
    'feature_vectors/features_2017-11-05-120000_2017-11-06-115959.p')


def are_feature_tables_equal(feature_table1, feature_table2):
    """Determines whether or not two feature tables are equal.

    :param feature_table1: pandas DataFrame.
    :param feature_table2: pandas DataFrame.
    :return: are_tables_equal_flag: Boolean flag.  True if tables are equal,
        False if not.
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

    def test_get_observation_densities_buffers_not_in_table(self):
        """Ensures correct output from _get_observation_densities.

        In this case, buffered polygons are not in FEATURE_TABLE, so they must
        be created on the fly.
        """

        this_feature_table = FEATURE_TABLE.drop(
            BUFFER_COLUMN_NAME, axis=1, inplace=False)
        _, these_observation_densities_m02 = (
            feature_vectors._get_observation_densities(this_feature_table))

        self.assertTrue(numpy.allclose(
            these_observation_densities_m02,
            OBSERVATION_DENSITY_BY_STORM_OBJECT_M02, rtol=TOLERANCE))

    def test_get_observation_densities_buffers_in_table(self):
        """Ensures correct output from _get_observation_densities.

        In this case, buffered polygons are already in FEATURE_TABLE, so they
        need not be created on the fly.
        """

        _, these_observation_densities_m02 = (
            feature_vectors._get_observation_densities(FEATURE_TABLE))
        self.assertTrue(numpy.allclose(
            these_observation_densities_m02,
            OBSERVATION_DENSITY_BY_STORM_OBJECT_M02, rtol=TOLERANCE))

    def test_indices_from_file_specific_to_overall(self):
        """Ensures correctness of _indices_from_file_specific_to_overall."""

        these_overall_indices = (
            feature_vectors._indices_from_file_specific_to_overall(
                INDICES_BY_FILE, NUM_OBJECTS_BY_FILE))
        self.assertTrue(numpy.array_equal(
            these_overall_indices, OVERALL_INDICES))

    def test_indices_from_overall_to_file_specific(self):
        """Ensures correctness of _indices_from_overall_to_file_specific."""

        these_indices_by_file = (
            feature_vectors._indices_from_overall_to_file_specific(
                OVERALL_INDICES, NUM_OBJECTS_BY_FILE))

        self.assertTrue(len(these_indices_by_file) == len(INDICES_BY_FILE))

        num_files = len(INDICES_BY_FILE)
        for i in range(num_files):
            self.assertTrue(numpy.array_equal(
                these_indices_by_file[i], INDICES_BY_FILE[i]))

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
                sounding_stat_table=SOUNDING_STAT_TABLE,
                storm_to_winds_table=STORM_TO_WINDS_TABLE,
                label_column_name=CLASSIFICATION_LABEL_COLUMN_NAME))

        self.assertTrue(are_feature_tables_equal(
            this_feature_table, FEATURE_TABLE))

    def test_sample_by_min_observations_return_table(self):
        """Ensures correct output from sample_by_min_observations.

        In this case, return_table = True.
        """

        this_feature_table, _ = feature_vectors.sample_by_min_observations(
            FEATURE_TABLE, min_observations=MIN_OBSERVATIONS_FOR_SAMPLING,
            return_table=True)
        this_feature_table = (
            this_feature_table.iloc[NUM_DEAD_STORM_OBJECTS_TO_SELECT:])

        this_expected_feature_table = FEATURE_TABLE.iloc[
            LIVE_SELECTED_INDICES_FOR_MIN_OBS_SAMPLING]
        self.assertTrue(are_feature_tables_equal(
            this_feature_table, this_expected_feature_table))

    def test_sample_by_min_observations_return_dict(self):
        """Ensures correct output from sample_by_min_observations.

        In this case, return_table = False.
        """

        _, this_metadata_dict = feature_vectors.sample_by_min_observations(
            FEATURE_TABLE, min_observations=MIN_OBSERVATIONS_FOR_SAMPLING,
            return_table=False)

        self.assertTrue(numpy.array_equal(
            this_metadata_dict[feature_vectors.LIVE_SELECTED_INDICES_KEY],
            LIVE_SELECTED_INDICES_FOR_MIN_OBS_SAMPLING))
        self.assertTrue(numpy.array_equal(
            this_metadata_dict[feature_vectors.LIVE_INDICES_KEY],
            LIVE_STORM_OBJECT_INDICES))
        self.assertTrue(numpy.array_equal(
            this_metadata_dict[feature_vectors.DEAD_INDICES_KEY],
            DEAD_STORM_OBJECT_INDICES))

    def test_sample_by_min_observations_plus_return_table(self):
        """Ensures correct output from sample_by_min_observations_plus.

        In this case, return_table = True.
        """

        this_feature_table, _ = feature_vectors.sample_by_min_observations_plus(
            FEATURE_TABLE, min_observations=MIN_OBSERVATIONS_FOR_SAMPLING,
            return_table=True)
        this_feature_table = (
            this_feature_table.iloc[NUM_DEAD_STORM_OBJECTS_TO_SELECT:])

        this_expected_feature_table = FEATURE_TABLE.iloc[
            LIVE_SELECTED_INDICES_FOR_MIN_OBS_PLUS_SAMPLING]
        self.assertTrue(are_feature_tables_equal(
            this_feature_table, this_expected_feature_table))

    def test_sample_by_min_observations_plus_return_dict(self):
        """Ensures correct output from sample_by_min_observations_plus.

        In this case, return_table = False.
        """

        _, this_metadata_dict = feature_vectors.sample_by_min_observations_plus(
            FEATURE_TABLE, min_observations=MIN_OBSERVATIONS_FOR_SAMPLING,
            return_table=False)

        self.assertTrue(numpy.array_equal(
            this_metadata_dict[feature_vectors.LIVE_SELECTED_INDICES_KEY],
            LIVE_SELECTED_INDICES_FOR_MIN_OBS_PLUS_SAMPLING))
        self.assertTrue(numpy.array_equal(
            this_metadata_dict[feature_vectors.LIVE_INDICES_KEY],
            LIVE_STORM_OBJECT_INDICES))
        self.assertTrue(numpy.array_equal(
            this_metadata_dict[feature_vectors.DEAD_INDICES_KEY],
            DEAD_STORM_OBJECT_INDICES))

    def test_sample_by_min_obs_density_return_table(self):
        """Ensures correct output from sample_by_min_obs_density.

        In this case, return_table = True.
        """

        this_feature_table, _ = feature_vectors.sample_by_min_obs_density(
            FEATURE_TABLE,
            min_observation_density_m02=MIN_OBS_DENSITY_FOR_SAMPLING_M02,
            return_table=True)
        this_feature_table = (
            this_feature_table.iloc[NUM_DEAD_STORM_OBJECTS_TO_SELECT:])

        this_expected_feature_table = FEATURE_TABLE.iloc[
            LIVE_SELECTED_INDICES_FOR_MIN_DENSITY_SAMPLING]
        self.assertTrue(are_feature_tables_equal(
            this_feature_table, this_expected_feature_table))

    def test_sample_by_min_obs_density_return_dict(self):
        """Ensures correct output from sample_by_min_obs_density.

        In this case, return_table = False.
        """

        _, this_metadata_dict = feature_vectors.sample_by_min_obs_density(
            FEATURE_TABLE,
            min_observation_density_m02=MIN_OBS_DENSITY_FOR_SAMPLING_M02,
            return_table=False)

        self.assertTrue(numpy.array_equal(
            this_metadata_dict[feature_vectors.LIVE_SELECTED_INDICES_KEY],
            LIVE_SELECTED_INDICES_FOR_MIN_DENSITY_SAMPLING))
        self.assertTrue(numpy.array_equal(
            this_metadata_dict[feature_vectors.LIVE_INDICES_KEY],
            LIVE_STORM_OBJECT_INDICES))
        self.assertTrue(numpy.array_equal(
            this_metadata_dict[feature_vectors.DEAD_INDICES_KEY],
            DEAD_STORM_OBJECT_INDICES))

    def test_sample_by_min_obs_density_plus_return_table(self):
        """Ensures correct output from sample_by_min_obs_density_plus.

        In this case, return_table = True.
        """

        this_feature_table, _ = feature_vectors.sample_by_min_obs_density_plus(
            FEATURE_TABLE,
            min_observation_density_m02=MIN_OBS_DENSITY_FOR_SAMPLING_M02,
            return_table=True)
        this_feature_table = (
            this_feature_table.iloc[NUM_DEAD_STORM_OBJECTS_TO_SELECT:])

        this_expected_feature_table = FEATURE_TABLE.iloc[
            LIVE_SELECTED_INDICES_FOR_MIN_DENSITY_PLUS_SAMPLING]
        self.assertTrue(are_feature_tables_equal(
            this_feature_table, this_expected_feature_table))

    def test_sample_by_min_obs_density_plus_return_dict(self):
        """Ensures correct output from sample_by_min_obs_density_plus.

        In this case, return_table = False.
        """

        _, this_metadata_dict = feature_vectors.sample_by_min_observations_plus(
            FEATURE_TABLE, min_observations=MIN_OBSERVATIONS_FOR_SAMPLING,
            return_table=False)

        self.assertTrue(numpy.array_equal(
            this_metadata_dict[feature_vectors.LIVE_SELECTED_INDICES_KEY],
            LIVE_SELECTED_INDICES_FOR_MIN_DENSITY_PLUS_SAMPLING))
        self.assertTrue(numpy.array_equal(
            this_metadata_dict[feature_vectors.LIVE_INDICES_KEY],
            LIVE_STORM_OBJECT_INDICES))
        self.assertTrue(numpy.array_equal(
            this_metadata_dict[feature_vectors.DEAD_INDICES_KEY],
            DEAD_STORM_OBJECT_INDICES))

    def test_sample_by_uniform_wind_speed_return_table(self):
        """Ensures correct output from sample_by_uniform_wind_speed.

        In this case, return_table = True.
        """

        this_feature_table, _ = feature_vectors.sample_by_uniform_wind_speed(
            FEATURE_TABLE, cutoffs_m_s01=CUTOFFS_FOR_UNIFORM_SAMPLING_M_S01,
            return_table=True)

        this_expected_feature_table = FEATURE_TABLE.iloc[
            LIVE_SELECTED_INDICES_FOR_UNIF_SAMPLING]
        self.assertTrue(are_feature_tables_equal(
            this_feature_table, this_expected_feature_table))

    def test_sample_by_uniform_wind_speed_return_dict(self):
        """Ensures correct output from sample_by_uniform_wind_speed.

        In this case, return_table = False.
        """

        _, this_metadata_dict = feature_vectors.sample_by_uniform_wind_speed(
            FEATURE_TABLE, cutoffs_m_s01=CUTOFFS_FOR_UNIFORM_SAMPLING_M_S01,
            return_table=False)

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
            spc_date_string=FILE_SPC_DATE_STRING,
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
