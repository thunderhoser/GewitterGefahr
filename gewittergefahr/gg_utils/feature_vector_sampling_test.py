"""Unit tests for feature_vector_sampling.py."""

import copy
import unittest
import numpy
import pandas
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import radar_statistics as radar_stats
from gewittergefahr.gg_utils import shape_statistics as shape_stats
from gewittergefahr.gg_utils import feature_vector_sampling as fv_sampling
from gewittergefahr.gg_utils import labels
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import link_events_to_storms as events2storms


TOLERANCE = 1e-6
KT_TO_METRES_PER_SECOND = 1.852 / 3.6

# Create table with radar statistic for each storm object.
NUM_STORM_OBJECTS = 25
STORM_IDS = ['storm{0:02d}'.format(s) for s in range(NUM_STORM_OBJECTS)]

STORM_OBJECT_TIMES_UNIX_SEC = numpy.linspace(
    0, 24000, num=NUM_STORM_OBJECTS, dtype=int)

RADAR_STATISTIC_NAME = radar_stats.radar_field_and_statistic_to_column_name(
    radar_field_name=radar_utils.VIL_NAME,
    radar_height_m_asl=250, statistic_name=radar_stats.STANDARD_DEVIATION_NAME)
RADAR_STATISTIC_VALUES = numpy.full(NUM_STORM_OBJECTS, 3.)
THIS_DICTIONARY = {
    tracking_utils.STORM_ID_COLUMN: STORM_IDS,
    tracking_utils.TIME_COLUMN: STORM_OBJECT_TIMES_UNIX_SEC,
    RADAR_STATISTIC_NAME: RADAR_STATISTIC_VALUES
}
RADAR_STATISTIC_TABLE = pandas.DataFrame.from_dict(THIS_DICTIONARY)

# Create table with shape statistic for each storm object.
SHAPE_STATISTIC_NAME = shape_stats.AREA_NAME
SHAPE_STATISTIC_VALUES = numpy.full(NUM_STORM_OBJECTS, 200.)
THIS_DICTIONARY = {
    tracking_utils.STORM_ID_COLUMN: STORM_IDS,
    tracking_utils.TIME_COLUMN: STORM_OBJECT_TIMES_UNIX_SEC,
    SHAPE_STATISTIC_NAME: SHAPE_STATISTIC_VALUES
}
SHAPE_STATISTIC_TABLE = pandas.DataFrame.from_dict(THIS_DICTIONARY)

# Create table with sounding statistic for each storm object.
SOUNDING_STATISTIC_NAME = 'storm_velocity_m_s01_magnitude'
SOUNDING_STATISTIC_VALUES = numpy.full(NUM_STORM_OBJECTS, 10.)
THIS_DICTIONARY = {
    tracking_utils.STORM_ID_COLUMN: STORM_IDS,
    tracking_utils.TIME_COLUMN: STORM_OBJECT_TIMES_UNIX_SEC,
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

DEAD_STORM_OBJECT_INDICES = numpy.linspace(0, 4, num=5, dtype=int)
LIVE_STORM_OBJECT_INDICES = numpy.linspace(5, 24, num=20, dtype=int)

REMAINING_STORM_LIFETIMES_SEC = numpy.full(NUM_STORM_OBJECTS, 3600, dtype=int)
REMAINING_STORM_LIFETIMES_SEC[DEAD_STORM_OBJECT_INDICES] = 1800
STORM_CELL_END_TIMES_UNIX_SEC = (
    STORM_OBJECT_TIMES_UNIX_SEC + REMAINING_STORM_LIFETIMES_SEC)

THIS_DICTIONARY = {
    tracking_utils.STORM_ID_COLUMN: STORM_IDS,
    tracking_utils.TIME_COLUMN: STORM_OBJECT_TIMES_UNIX_SEC,
    events2storms.STORM_END_TIME_COLUMN: STORM_CELL_END_TIMES_UNIX_SEC,
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

THIS_DICTIONARY = {
    tracking_utils.POLYGON_OBJECT_LATLNG_COLUMN: POLYGON_OBJECT_ARRAY_LATLNG
}
STORM_TO_WINDS_TABLE = STORM_TO_WINDS_TABLE.assign(**THIS_DICTIONARY)

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

# The following constants are used to test _select_storms_uniformly_by_category.
WIND_SPEED_CUTOFFS_FOR_SAMPLING_M_S01 = (
    KT_TO_METRES_PER_SECOND * numpy.array([10., 20., 30., 40., 50.]))
CATEGORIES_FOR_UNIFORM_SAMPLING = numpy.array([-1, -1, -1, -1, -1,
                                               0, 0, 0,
                                               1, 1, 1, 1, 1,
                                               2, 2, 2, 2,
                                               3, 3, 3,
                                               4, 4, 4,
                                               5, 5], dtype=int)
LIVE_SELECTED_IDXS_FOR_UNIFORM_SAMPLING = numpy.array(
    [0, 1, 5, 6, 9, 12, 13, 14, 18, 19, 20, 21, 23, 24])

# The following constants are used to test _select_dead_storms_randomly,
# select_by_min_observations, select_by_min_observations_plus,
# select_by_min_obs_density, and select_by_min_obs_density_plus.
MIN_OBSERVATIONS_FOR_SAMPLING = 25
NUM_DEAD_STORM_OBJECTS_TO_SELECT = 3

LIVE_SELECTED_IDXS_FOR_MIN_OBS = numpy.array(
    [5, 6, 9, 10, 12, 13, 14, 15, 18, 20, 21])
LIVE_SELECTED_IDXS_FOR_MIN_OBS_PLUS = numpy.array(
    [5, 6, 9, 10, 12, 13, 14, 15, 18, 20, 21, 23, 24])
LIVE_SELECTED_IDXS_FOR_MIN_OBS_DENSITY = copy.deepcopy(
    LIVE_SELECTED_IDXS_FOR_MIN_OBS)
LIVE_SELECTED_IDXS_FOR_MIN_OBS_DENSITY_PLUS = copy.deepcopy(
    LIVE_SELECTED_IDXS_FOR_MIN_OBS_PLUS)

# The following constants are used to test _get_wind_observation_densities.
POLYGON_OBJECT_XY, THIS_PROJECTION_OBJECT = polygons.project_latlng_to_xy(
    POLYGON_OBJECT_LATLNG)
VERTEX_DICT_XY = polygons.polygon_object_to_vertex_arrays(POLYGON_OBJECT_XY)
BUFFERED_POLYGON_OBJECT_XY = polygons.buffer_simple_polygon(
    VERTEX_DICT_XY[polygons.EXTERIOR_X_COLUMN],
    VERTEX_DICT_XY[polygons.EXTERIOR_Y_COLUMN],
    min_buffer_dist_metres=MIN_LINK_DISTANCE_METRES,
    max_buffer_dist_metres=MAX_LINK_DISTANCE_METRES)

BUFFER_AREA_METRES2 = BUFFERED_POLYGON_OBJECT_XY.area
WIND_OBSERVATION_DENSITIES_M02 = WIND_OBSERVATION_COUNTS / BUFFER_AREA_METRES2
MIN_OBS_DENSITY_FOR_SAMPLING_M02 = (
    MIN_OBSERVATIONS_FOR_SAMPLING / BUFFER_AREA_METRES2)

BUFFERED_POLYGON_OBJECT_LATLNG = polygons.project_xy_to_latlng(
    BUFFERED_POLYGON_OBJECT_XY, THIS_PROJECTION_OBJECT)
BUFFERED_POLYGON_OBJECT_ARRAY_LATLNG = numpy.full(
    NUM_STORM_OBJECTS, BUFFERED_POLYGON_OBJECT_LATLNG, dtype=object)

BUFFER_COLUMN_NAME = tracking_utils.distance_buffer_to_column_name(
    MIN_LINK_DISTANCE_METRES, MAX_LINK_DISTANCE_METRES)



# The following constants are used to test _check_labels_in_table.
LABEL_COLUMN_NAMES = [
    WIND_SPEED_REGRESSION_COLUMN_NAME, WIND_SPEED_CLASSIFN_COLUMN_NAME]

# The following constants are used to test check_feature_table.
FEATURE_COLUMN_NAMES = [
    RADAR_STATISTIC_NAME, SHAPE_STATISTIC_NAME, SOUNDING_STATISTIC_NAME]

# The following constants are used to test join_features_and_label.
THIS_DICTIONARY = {
    tracking_utils.STORM_ID_COLUMN: STORM_IDS,
    tracking_utils.TIME_COLUMN: STORM_OBJECT_TIMES_UNIX_SEC,
    events2storms.STORM_END_TIME_COLUMN: STORM_CELL_END_TIMES_UNIX_SEC,
    WIND_SPEED_REGRESSION_COLUMN_NAME: WIND_REGRESSION_LABELS_M_S01,
    WIND_SPEED_CLASSIFN_COLUMN_NAME: WIND_CLASSIFICATION_LABELS,
    NUM_WIND_OBS_COLUMN_NAME: WIND_OBSERVATION_COUNTS,
    RADAR_STATISTIC_NAME: RADAR_STATISTIC_VALUES,
    SHAPE_STATISTIC_NAME: SHAPE_STATISTIC_VALUES,
    SOUNDING_STATISTIC_NAME: SOUNDING_STATISTIC_VALUES,
    tracking_utils.POLYGON_OBJECT_LATLNG_COLUMN: POLYGON_OBJECT_ARRAY_LATLNG,
    BUFFER_COLUMN_NAME: BUFFERED_POLYGON_OBJECT_ARRAY_LATLNG
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


class FeatureVectorSamplingTests(unittest.TestCase):
    """Each method is a unit test for feature_vector_sampling.py."""

    def test_find_live_and_dead_storms(self):
        """Ensures correct output from _find_live_and_dead_storms."""

        these_live_indices, these_dead_indices = (
            fv_sampling._find_live_and_dead_storms(
                storm_object_times_unix_sec=STORM_OBJECT_TIMES_UNIX_SEC,
                storm_cell_end_times_unix_sec=STORM_CELL_END_TIMES_UNIX_SEC,
                min_lead_time_sec=MIN_LEAD_TIME_SEC))

        self.assertTrue(numpy.array_equal(
            these_live_indices, LIVE_STORM_OBJECT_INDICES))
        self.assertTrue(numpy.array_equal(
            these_dead_indices, DEAD_STORM_OBJECT_INDICES))

    def test_select_dead_storms_randomly(self):
        """Ensures correct output from _select_dead_storms_randomly."""

        these_selected_indices = fv_sampling._select_dead_storms_randomly(
            live_indices=LIVE_STORM_OBJECT_INDICES,
            dead_indices=DEAD_STORM_OBJECT_INDICES,
            live_selected_indices=LIVE_SELECTED_IDXS_FOR_MIN_OBS)

        # Ensure that total number of selected storm objects matches
        # expectation.
        expected_num_selected = NUM_DEAD_STORM_OBJECTS_TO_SELECT + len(
            LIVE_SELECTED_IDXS_FOR_MIN_OBS)
        self.assertTrue(len(these_selected_indices) == expected_num_selected)

        # Ensure that all previously selected storm objects are still selected.
        for this_index in LIVE_SELECTED_IDXS_FOR_MIN_OBS:
            self.assertTrue(this_index in these_selected_indices)

    def test_select_storms_uniformly_by_category(self):
        """Ensures correct output from _select_storms_uniformly_by_category."""

        these_selected_indices = (
            fv_sampling._select_storms_uniformly_by_category(
                storm_object_categories=CATEGORIES_FOR_UNIFORM_SAMPLING,
                storm_object_priorities=WIND_OBSERVATION_COUNTS))

        self.assertTrue(numpy.array_equal(
            these_selected_indices, LIVE_SELECTED_IDXS_FOR_UNIFORM_SAMPLING))

    def test_get_wind_observation_densities_buffers_not_in_table(self):
        """Ensures correct output from _get_wind_observation_densities.

        In this case, buffered polygons are not in feature table, so they must
        be created on the fly.
        """

        this_feature_table = FEATURE_TABLE.drop(
            BUFFER_COLUMN_NAME, axis=1, inplace=False)

        _, these_observation_densities_m02 = (
            fv_sampling._get_wind_observation_densities(
                feature_table=this_feature_table,
                min_lead_time_sec=MIN_LEAD_TIME_SEC,
                max_lead_time_sec=MAX_LEAD_TIME_SEC,
                min_link_distance_metres=MIN_LINK_DISTANCE_METRES,
                max_link_distance_metres=MAX_LINK_DISTANCE_METRES))

        self.assertTrue(numpy.allclose(
            these_observation_densities_m02, WIND_OBSERVATION_DENSITIES_M02,
            rtol=TOLERANCE))

    def test_get_wind_observation_densities_buffers_in_table(self):
        """Ensures correct output from _get_wind_observation_densities.

        In this case, buffered polygons are already in feature table, so they
        need not  be created on the fly.
        """

        _, these_observation_densities_m02 = (
            fv_sampling._get_wind_observation_densities(
                feature_table=FEATURE_TABLE,
                min_lead_time_sec=MIN_LEAD_TIME_SEC,
                max_lead_time_sec=MAX_LEAD_TIME_SEC,
                min_link_distance_metres=MIN_LINK_DISTANCE_METRES,
                max_link_distance_metres=MAX_LINK_DISTANCE_METRES))

        self.assertTrue(numpy.allclose(
            these_observation_densities_m02, WIND_OBSERVATION_DENSITIES_M02,
            rtol=TOLERANCE))

    def test_indices_from_file_specific_to_overall(self):
        """Ensures correctness of _indices_from_file_specific_to_overall."""

        these_overall_indices = (
            fv_sampling._indices_from_file_specific_to_overall(
                indices_by_file=INDICES_BY_FILE,
                num_objects_by_file=NUM_OBJECTS_BY_FILE))

        self.assertTrue(numpy.array_equal(
            these_overall_indices, OVERALL_INDICES))

    def test_indices_from_overall_to_file_specific(self):
        """Ensures correctness of _indices_from_overall_to_file_specific."""

        these_indices_by_file = (
            fv_sampling._indices_from_overall_to_file_specific(
                overall_indices=OVERALL_INDICES,
                num_objects_by_file=NUM_OBJECTS_BY_FILE))

        self.assertTrue(len(these_indices_by_file) == len(INDICES_BY_FILE))

        num_files = len(INDICES_BY_FILE)
        for i in range(num_files):
            self.assertTrue(numpy.array_equal(
                these_indices_by_file[i], INDICES_BY_FILE[i]))

    def test_sample_by_min_observations_return_table(self):
        """Ensures correct output from sample_by_min_observations.

        In this case, return_table = True.
        """

        this_feature_table, _ = fv_sampling.sample_by_min_observations(
            feature_table=FEATURE_TABLE, min_lead_time_sec=MIN_LEAD_TIME_SEC,
            max_lead_time_sec=MAX_LEAD_TIME_SEC,
            min_link_distance_metres=MIN_LINK_DISTANCE_METRES,
            max_link_distance_metres=MAX_LINK_DISTANCE_METRES,
            min_observation_count=MIN_OBSERVATIONS_FOR_SAMPLING,
            return_table=True)

        this_feature_table = (
            this_feature_table.iloc[NUM_DEAD_STORM_OBJECTS_TO_SELECT:])

        expected_feature_table = FEATURE_TABLE.iloc[
            LIVE_SELECTED_IDXS_FOR_MIN_OBS]
        self.assertTrue(are_feature_tables_equal(
            this_feature_table, expected_feature_table))

    def test_sample_by_min_observations_return_dict(self):
        """Ensures correct output from sample_by_min_observations.

        In this case, return_table = False.
        """

        _, this_metadata_dict = fv_sampling.sample_by_min_observations(
            feature_table=FEATURE_TABLE, min_lead_time_sec=MIN_LEAD_TIME_SEC,
            max_lead_time_sec=MAX_LEAD_TIME_SEC,
            min_link_distance_metres=MIN_LINK_DISTANCE_METRES,
            max_link_distance_metres=MAX_LINK_DISTANCE_METRES,
            min_observation_count=MIN_OBSERVATIONS_FOR_SAMPLING,
            return_table=False)

        self.assertTrue(numpy.array_equal(
            this_metadata_dict[fv_sampling.LIVE_SELECTED_INDICES_KEY],
            LIVE_SELECTED_IDXS_FOR_MIN_OBS))
        self.assertTrue(numpy.array_equal(
            this_metadata_dict[fv_sampling.LIVE_INDICES_KEY],
            LIVE_STORM_OBJECT_INDICES))
        self.assertTrue(numpy.array_equal(
            this_metadata_dict[fv_sampling.DEAD_INDICES_KEY],
            DEAD_STORM_OBJECT_INDICES))

    def test_sample_by_min_observations_plus_return_table(self):
        """Ensures correct output from sample_by_min_observations_plus.

        In this case, return_table = True.
        """

        this_feature_table, _ = fv_sampling.sample_by_min_observations_plus(
            feature_table=FEATURE_TABLE, min_lead_time_sec=MIN_LEAD_TIME_SEC,
            max_lead_time_sec=MAX_LEAD_TIME_SEC,
            min_link_distance_metres=MIN_LINK_DISTANCE_METRES,
            max_link_distance_metres=MAX_LINK_DISTANCE_METRES,
            min_observation_count=MIN_OBSERVATIONS_FOR_SAMPLING,
            wind_speed_percentile_level=WIND_SPEED_PERCENTILE_LEVEL,
            wind_speed_class_cutoffs_kt=WIND_SPEED_CLASS_CUTOFFS_KT,
            return_table=True)

        this_feature_table = (
            this_feature_table.iloc[NUM_DEAD_STORM_OBJECTS_TO_SELECT:])

        expected_feature_table = FEATURE_TABLE.iloc[
            LIVE_SELECTED_IDXS_FOR_MIN_OBS_PLUS]
        self.assertTrue(are_feature_tables_equal(
            this_feature_table, expected_feature_table))

    def test_sample_by_min_observations_plus_return_dict(self):
        """Ensures correct output from sample_by_min_observations_plus.

        In this case, return_table = False.
        """

        _, this_metadata_dict = fv_sampling.sample_by_min_observations_plus(
            feature_table=FEATURE_TABLE, min_lead_time_sec=MIN_LEAD_TIME_SEC,
            max_lead_time_sec=MAX_LEAD_TIME_SEC,
            min_link_distance_metres=MIN_LINK_DISTANCE_METRES,
            max_link_distance_metres=MAX_LINK_DISTANCE_METRES,
            min_observation_count=MIN_OBSERVATIONS_FOR_SAMPLING,
            wind_speed_percentile_level=WIND_SPEED_PERCENTILE_LEVEL,
            wind_speed_class_cutoffs_kt=WIND_SPEED_CLASS_CUTOFFS_KT,
            return_table=False)

        self.assertTrue(numpy.array_equal(
            this_metadata_dict[fv_sampling.LIVE_SELECTED_INDICES_KEY],
            LIVE_SELECTED_IDXS_FOR_MIN_OBS_PLUS))
        self.assertTrue(numpy.array_equal(
            this_metadata_dict[fv_sampling.LIVE_INDICES_KEY],
            LIVE_STORM_OBJECT_INDICES))
        self.assertTrue(numpy.array_equal(
            this_metadata_dict[fv_sampling.DEAD_INDICES_KEY],
            DEAD_STORM_OBJECT_INDICES))

    def test_sample_by_min_obs_density_return_table(self):
        """Ensures correct output from sample_by_min_obs_density.

        In this case, return_table = True.
        """

        this_feature_table, _ = fv_sampling.sample_by_min_obs_density(
            feature_table=FEATURE_TABLE, min_lead_time_sec=MIN_LEAD_TIME_SEC,
            max_lead_time_sec=MAX_LEAD_TIME_SEC,
            min_link_distance_metres=MIN_LINK_DISTANCE_METRES,
            max_link_distance_metres=MAX_LINK_DISTANCE_METRES,
            min_observation_density_m02=MIN_OBS_DENSITY_FOR_SAMPLING_M02,
            return_table=True)

        this_feature_table = (
            this_feature_table.iloc[NUM_DEAD_STORM_OBJECTS_TO_SELECT:])

        expected_feature_table = FEATURE_TABLE.iloc[
            LIVE_SELECTED_IDXS_FOR_MIN_OBS_DENSITY]
        self.assertTrue(are_feature_tables_equal(
            this_feature_table, expected_feature_table))

    def test_sample_by_min_obs_density_return_dict(self):
        """Ensures correct output from sample_by_min_obs_density.

        In this case, return_table = False.
        """

        _, this_metadata_dict = fv_sampling.sample_by_min_obs_density(
            feature_table=FEATURE_TABLE, min_lead_time_sec=MIN_LEAD_TIME_SEC,
            max_lead_time_sec=MAX_LEAD_TIME_SEC,
            min_link_distance_metres=MIN_LINK_DISTANCE_METRES,
            max_link_distance_metres=MAX_LINK_DISTANCE_METRES,
            min_observation_density_m02=MIN_OBS_DENSITY_FOR_SAMPLING_M02,
            return_table=False)

        self.assertTrue(numpy.array_equal(
            this_metadata_dict[fv_sampling.LIVE_SELECTED_INDICES_KEY],
            LIVE_SELECTED_IDXS_FOR_MIN_OBS_DENSITY))
        self.assertTrue(numpy.array_equal(
            this_metadata_dict[fv_sampling.LIVE_INDICES_KEY],
            LIVE_STORM_OBJECT_INDICES))
        self.assertTrue(numpy.array_equal(
            this_metadata_dict[fv_sampling.DEAD_INDICES_KEY],
            DEAD_STORM_OBJECT_INDICES))

    def test_sample_by_min_obs_density_plus_return_table(self):
        """Ensures correct output from sample_by_min_obs_density_plus.

        In this case, return_table = True.
        """

        this_feature_table, _ = fv_sampling.sample_by_min_obs_density_plus(
            feature_table=FEATURE_TABLE, min_lead_time_sec=MIN_LEAD_TIME_SEC,
            max_lead_time_sec=MAX_LEAD_TIME_SEC,
            min_link_distance_metres=MIN_LINK_DISTANCE_METRES,
            max_link_distance_metres=MAX_LINK_DISTANCE_METRES,
            min_observation_density_m02=MIN_OBS_DENSITY_FOR_SAMPLING_M02,
            wind_speed_percentile_level=WIND_SPEED_PERCENTILE_LEVEL,
            wind_speed_class_cutoffs_kt=WIND_SPEED_CLASS_CUTOFFS_KT,
            return_table=True)

        this_feature_table = (
            this_feature_table.iloc[NUM_DEAD_STORM_OBJECTS_TO_SELECT:])

        expected_feature_table = FEATURE_TABLE.iloc[
            LIVE_SELECTED_IDXS_FOR_MIN_OBS_DENSITY_PLUS]
        self.assertTrue(are_feature_tables_equal(
            this_feature_table, expected_feature_table))

    def test_sample_by_min_obs_density_plus_return_dict(self):
        """Ensures correct output from sample_by_min_obs_density_plus.

        In this case, return_table = False.
        """

        _, this_metadata_dict = fv_sampling.sample_by_min_obs_density_plus(
            feature_table=FEATURE_TABLE, min_lead_time_sec=MIN_LEAD_TIME_SEC,
            max_lead_time_sec=MAX_LEAD_TIME_SEC,
            min_link_distance_metres=MIN_LINK_DISTANCE_METRES,
            max_link_distance_metres=MAX_LINK_DISTANCE_METRES,
            min_observation_density_m02=MIN_OBS_DENSITY_FOR_SAMPLING_M02,
            wind_speed_percentile_level=WIND_SPEED_PERCENTILE_LEVEL,
            wind_speed_class_cutoffs_kt=WIND_SPEED_CLASS_CUTOFFS_KT,
            return_table=False)

        self.assertTrue(numpy.array_equal(
            this_metadata_dict[fv_sampling.LIVE_SELECTED_INDICES_KEY],
            LIVE_SELECTED_IDXS_FOR_MIN_OBS_DENSITY_PLUS))
        self.assertTrue(numpy.array_equal(
            this_metadata_dict[fv_sampling.LIVE_INDICES_KEY],
            LIVE_STORM_OBJECT_INDICES))
        self.assertTrue(numpy.array_equal(
            this_metadata_dict[fv_sampling.DEAD_INDICES_KEY],
            DEAD_STORM_OBJECT_INDICES))

    def test_sample_by_uniform_wind_speed_return_table(self):
        """Ensures correct output from sample_by_uniform_wind_speed.

        In this case, return_table = True.
        """

        this_feature_table, _ = fv_sampling.sample_by_uniform_wind_speed(
            feature_table=FEATURE_TABLE, min_lead_time_sec=MIN_LEAD_TIME_SEC,
            max_lead_time_sec=MAX_LEAD_TIME_SEC,
            min_link_distance_metres=MIN_LINK_DISTANCE_METRES,
            max_link_distance_metres=MAX_LINK_DISTANCE_METRES,
            wind_speed_percentile_level=WIND_SPEED_PERCENTILE_LEVEL,
            sampling_cutoffs_m_s01=WIND_SPEED_CUTOFFS_FOR_SAMPLING_M_S01,
            return_table=True)

        expected_feature_table = FEATURE_TABLE.iloc[
            LIVE_SELECTED_IDXS_FOR_UNIFORM_SAMPLING]
        self.assertTrue(are_feature_tables_equal(
            this_feature_table, expected_feature_table))

    def test_sample_by_uniform_wind_speed_return_dict(self):
        """Ensures correct output from sample_by_uniform_wind_speed.

        In this case, return_table = False.
        """

        _, this_metadata_dict = fv_sampling.sample_by_uniform_wind_speed(
            feature_table=FEATURE_TABLE, min_lead_time_sec=MIN_LEAD_TIME_SEC,
            max_lead_time_sec=MAX_LEAD_TIME_SEC,
            min_link_distance_metres=MIN_LINK_DISTANCE_METRES,
            max_link_distance_metres=MAX_LINK_DISTANCE_METRES,
            wind_speed_percentile_level=WIND_SPEED_PERCENTILE_LEVEL,
            sampling_cutoffs_m_s01=WIND_SPEED_CUTOFFS_FOR_SAMPLING_M_S01,
            return_table=False)

        self.assertTrue(numpy.array_equal(
            this_metadata_dict[fv_sampling.WIND_SPEED_CATEGORIES_KEY],
            CATEGORIES_FOR_UNIFORM_SAMPLING))
        self.assertTrue(numpy.array_equal(
            this_metadata_dict[fv_sampling.WIND_OBSERVATION_COUNTS_KEY],
            WIND_OBSERVATION_COUNTS))


if __name__ == '__main__':
    unittest.main()
