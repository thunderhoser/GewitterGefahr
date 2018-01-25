"""Unit tests for gridrad_tracking.py."""

import copy
import unittest
import numpy
import pandas
from gewittergefahr.gg_utils import gridrad_tracking
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import time_conversion

TOLERANCE = 1e-6

# The following constants are used to test _find_local_maxima.
RADAR_MATRIX = numpy.array([
    [0., numpy.nan, 3., 4., numpy.nan, 6.],
    [7., 8., 9., 10., numpy.nan, numpy.nan],
    [13., 14., numpy.nan, numpy.nan, 17., 18.],
    [19., 20., numpy.nan, numpy.nan, numpy.nan, 24.],
    [numpy.nan, numpy.nan, 27., 28., 29., 30.]])

RADAR_METADATA_DICT = {
    radar_utils.NW_GRID_POINT_LAT_COLUMN: 35.,
    radar_utils.NW_GRID_POINT_LNG_COLUMN: 95.,
    radar_utils.LAT_SPACING_COLUMN: 0.01,
    radar_utils.LNG_SPACING_COLUMN: 0.02}

NEIGH_HALF_WIDTH_IN_PIXELS = 1
LOCAL_MAX_ROWS = numpy.array([0, 4], dtype=int)
LOCAL_MAX_COLUMNS = numpy.array([5, 5], dtype=int)
LOCAL_MAX_LATITUDES_DEG = numpy.array([34.96, 35.])
LOCAL_MAX_LONGITUDES_DEG = numpy.array([95.1, 95.1])
LOCAL_MAX_VALUES = numpy.array([30., 6])

LOCAL_MAX_DICT_LATLNG = {
    gridrad_tracking.LATITUDES_KEY: LOCAL_MAX_LATITUDES_DEG,
    gridrad_tracking.LONGITUDES_KEY: LOCAL_MAX_LONGITUDES_DEG,
    gridrad_tracking.MAX_VALUES_KEY: LOCAL_MAX_VALUES
}

# The following constants are used to test _remove_redundant_local_maxima.
SMALL_DISTANCE_BETWEEN_MAXIMA_METRES = 1000.
LARGE_DISTANCE_BETWEEN_MAXIMA_METRES = 10000.
PROJECTION_OBJECT = projections.init_azimuthal_equidistant_projection(
    central_latitude_deg=35., central_longitude_deg=95.)

LOCAL_MAX_X_COORDS_METRES, LOCAL_MAX_Y_COORDS_METRES = (
    projections.project_latlng_to_xy(
        LOCAL_MAX_LATITUDES_DEG, LOCAL_MAX_LONGITUDES_DEG,
        projection_object=PROJECTION_OBJECT, false_easting_metres=0.,
        false_northing_metres=0.))

LOCAL_MAX_DICT_SMALL_DISTANCE = {
    gridrad_tracking.LATITUDES_KEY: LOCAL_MAX_LATITUDES_DEG,
    gridrad_tracking.LONGITUDES_KEY: LOCAL_MAX_LONGITUDES_DEG,
    gridrad_tracking.MAX_VALUES_KEY: LOCAL_MAX_VALUES,
    gridrad_tracking.X_COORDS_KEY: LOCAL_MAX_X_COORDS_METRES,
    gridrad_tracking.Y_COORDS_KEY: LOCAL_MAX_Y_COORDS_METRES
}

LOCAL_MAX_DICT_LARGE_DISTANCE = {
    gridrad_tracking.LATITUDES_KEY: LOCAL_MAX_LATITUDES_DEG[1:],
    gridrad_tracking.LONGITUDES_KEY: LOCAL_MAX_LONGITUDES_DEG[1:],
    gridrad_tracking.MAX_VALUES_KEY: LOCAL_MAX_VALUES[1:],
    gridrad_tracking.X_COORDS_KEY: LOCAL_MAX_X_COORDS_METRES[1:],
    gridrad_tracking.Y_COORDS_KEY: LOCAL_MAX_Y_COORDS_METRES[1:]
}

# The following constants are used to test _link_local_maxima_in_time.
PREVIOUS_TIME_UNIX_SEC = 1516860600  # 0610 UTC 25 Jan 2018
PREVIOUS_LOCAL_MAX_DICT = {
    gridrad_tracking.X_COORDS_KEY: LOCAL_MAX_X_COORDS_METRES,
    gridrad_tracking.Y_COORDS_KEY: LOCAL_MAX_Y_COORDS_METRES,
    gridrad_tracking.VALID_TIME_KEY: PREVIOUS_TIME_UNIX_SEC
}

MAX_LINK_TIME_SECONDS = 300
MAX_LINK_DISTANCE_M_S01 = 10.
MAX_LINK_DISTANCE_METRES = MAX_LINK_TIME_SECONDS * MAX_LINK_DISTANCE_M_S01

CURRENT_TIME_UNIX_SEC = 1516860900  # 0615 UTC 25 Jan 2018
CURRENT_TIME_TOO_LATE_UNIX_SEC = 1516861200

CURRENT_LOCAL_MAX_DICT_BOTH_FAR = {
    gridrad_tracking.X_COORDS_KEY:
        LOCAL_MAX_X_COORDS_METRES + MAX_LINK_DISTANCE_METRES,
    gridrad_tracking.Y_COORDS_KEY:
        LOCAL_MAX_Y_COORDS_METRES - MAX_LINK_DISTANCE_METRES,
    gridrad_tracking.VALID_TIME_KEY: CURRENT_TIME_UNIX_SEC
}
CURRENT_TO_PREV_INDICES_BOTH_FAR = numpy.array([-1, -1], dtype=int)

CURRENT_LOCAL_MAX_DICT_ONE_NEAR = {
    gridrad_tracking.X_COORDS_KEY:
        LOCAL_MAX_X_COORDS_METRES + numpy.array([0., MAX_LINK_DISTANCE_METRES]),
    gridrad_tracking.Y_COORDS_KEY:
        LOCAL_MAX_Y_COORDS_METRES - numpy.array([0., MAX_LINK_DISTANCE_METRES]),
    gridrad_tracking.VALID_TIME_KEY: CURRENT_TIME_UNIX_SEC
}
CURRENT_TO_PREV_INDICES_ONE_NEAR = numpy.array([0, -1], dtype=int)

CURRENT_LOCAL_MAX_DICT_BOTH_NEAR = {
    gridrad_tracking.X_COORDS_KEY: LOCAL_MAX_X_COORDS_METRES,
    gridrad_tracking.Y_COORDS_KEY: LOCAL_MAX_Y_COORDS_METRES,
    gridrad_tracking.VALID_TIME_KEY: CURRENT_TIME_UNIX_SEC
}
CURRENT_TO_PREV_INDICES_BOTH_NEAR = numpy.array([0, 1], dtype=int)

CURRENT_LOCAL_MAX_DICT_TOO_LATE = {
    gridrad_tracking.X_COORDS_KEY: LOCAL_MAX_X_COORDS_METRES,
    gridrad_tracking.Y_COORDS_KEY: LOCAL_MAX_Y_COORDS_METRES,
    gridrad_tracking.VALID_TIME_KEY: CURRENT_TIME_TOO_LATE_UNIX_SEC
}
CURRENT_TO_PREV_INDICES_TOO_LATE = numpy.array([-1, -1], dtype=int)

CURRENT_LOCAL_MAX_DICT_OVERLAP = {
    gridrad_tracking.X_COORDS_KEY: numpy.array(
        [LOCAL_MAX_X_COORDS_METRES[0] + 10., LOCAL_MAX_X_COORDS_METRES[0]]),
    gridrad_tracking.Y_COORDS_KEY: numpy.array(
        [LOCAL_MAX_Y_COORDS_METRES[0] - 10., LOCAL_MAX_Y_COORDS_METRES[0]]),
    gridrad_tracking.VALID_TIME_KEY: CURRENT_TIME_UNIX_SEC
}
CURRENT_TO_PREV_INDICES_OVERLAP = numpy.array([-1, 0], dtype=int)

PREVIOUS_LOCAL_MAX_DICT_EMPTY = {
    gridrad_tracking.X_COORDS_KEY: numpy.array([]),
    gridrad_tracking.Y_COORDS_KEY: numpy.array([]),
    gridrad_tracking.VALID_TIME_KEY: PREVIOUS_TIME_UNIX_SEC
}
CURRENT_LOCAL_MAX_DICT_EMPTY = {
    gridrad_tracking.X_COORDS_KEY: numpy.array([]),
    gridrad_tracking.Y_COORDS_KEY: numpy.array([]),
    gridrad_tracking.VALID_TIME_KEY: CURRENT_TIME_UNIX_SEC
}
CURRENT_TO_PREV_INDICES_NO_LINKS = numpy.array([-1, -1], dtype=int)

# The following constants are used to test _create_storm_id.
STORM_TIME_UNIX_SEC = 1516860900  # 0615 UTC 25 Jan 2018
STORM_SPC_DATE_STRING = '20180124'
PREV_SPC_DATE_STRING = '20180123'
PREV_NUMERIC_ID_USED = 0

STORM_ID_FIRST_IN_DAY = '000000_20180124'
STORM_ID_SECOND_IN_DAY = '000001_20180124'

# The following constants are used to test _local_maxima_to_storm_tracks.
LOCAL_MAX_DICT_TIME0 = {
    gridrad_tracking.LATITUDES_KEY: LOCAL_MAX_LATITUDES_DEG,
    gridrad_tracking.LONGITUDES_KEY: LOCAL_MAX_LONGITUDES_DEG,
    gridrad_tracking.VALID_TIME_KEY: PREVIOUS_TIME_UNIX_SEC,
    gridrad_tracking.CURRENT_TO_PREV_INDICES_KEY:
        CURRENT_TO_PREV_INDICES_NO_LINKS
}

LOCAL_MAX_DICT_TIME1 = {
    gridrad_tracking.LATITUDES_KEY: LOCAL_MAX_LATITUDES_DEG,
    gridrad_tracking.LONGITUDES_KEY: LOCAL_MAX_LONGITUDES_DEG,
    gridrad_tracking.VALID_TIME_KEY: CURRENT_TIME_UNIX_SEC,
    gridrad_tracking.CURRENT_TO_PREV_INDICES_KEY:
        CURRENT_TO_PREV_INDICES_BOTH_NEAR
}

LOCAL_MAX_DICT_BY_TIME = [LOCAL_MAX_DICT_TIME0, LOCAL_MAX_DICT_TIME1]

THESE_STORM_IDS = [
    STORM_ID_FIRST_IN_DAY, STORM_ID_SECOND_IN_DAY, STORM_ID_FIRST_IN_DAY,
    STORM_ID_SECOND_IN_DAY]
THESE_TIMES_UNIX_SEC = numpy.array(
    [PREVIOUS_TIME_UNIX_SEC, PREVIOUS_TIME_UNIX_SEC, CURRENT_TIME_UNIX_SEC,
     CURRENT_TIME_UNIX_SEC])
THESE_SPC_DATES_UNIX_SEC = numpy.full(
    4, time_conversion.time_to_spc_date_unix_sec(PREVIOUS_TIME_UNIX_SEC),
    dtype=int)
THESE_CENTROID_LATITUDES_DEG = numpy.concatenate((
    LOCAL_MAX_LATITUDES_DEG, LOCAL_MAX_LATITUDES_DEG))
THESE_CENTROID_LONGITUDES_DEG = numpy.concatenate((
    LOCAL_MAX_LONGITUDES_DEG, LOCAL_MAX_LONGITUDES_DEG))

STORM_OBJECT_DICT = {
    tracking_utils.STORM_ID_COLUMN: THESE_STORM_IDS,
    tracking_utils.TIME_COLUMN: THESE_TIMES_UNIX_SEC,
    tracking_utils.SPC_DATE_COLUMN: THESE_SPC_DATES_UNIX_SEC,
    tracking_utils.CENTROID_LAT_COLUMN: THESE_CENTROID_LATITUDES_DEG,
    tracking_utils.CENTROID_LNG_COLUMN: THESE_CENTROID_LONGITUDES_DEG
}
STORM_OBJECT_TABLE = pandas.DataFrame.from_dict(STORM_OBJECT_DICT)

# The following constants are used to test _remove_short_tracks.
SMALL_THRESHOLD_DURATION_SEC = 100
LARGE_THRESHOLD_DURATION_SEC = 1000


class GridradTrackingTests(unittest.TestCase):
    """Each method is a unit test for gridrad_tracking.py."""

    def test_find_local_maxima(self):
        """Ensures correct output from _find_local_maxima."""

        this_local_max_dict = gridrad_tracking._find_local_maxima(
            radar_matrix=RADAR_MATRIX, radar_metadata_dict=RADAR_METADATA_DICT,
            neigh_half_width_in_pixels=NEIGH_HALF_WIDTH_IN_PIXELS)

        these_keys = set(list(this_local_max_dict))
        expected_keys = set(list(LOCAL_MAX_DICT_LATLNG))
        self.assertTrue(these_keys == expected_keys)

        for this_key in these_keys:
            self.assertTrue(numpy.allclose(
                this_local_max_dict[this_key], LOCAL_MAX_DICT_LATLNG[this_key],
                atol=TOLERANCE))

    def test_remove_redundant_local_maxima_small_distance(self):
        """Ensures correct output from _remove_redundant_local_maxima.

        In this case, minimum distance between two maxima is small.
        """

        this_local_max_dict = gridrad_tracking._remove_redundant_local_maxima(
            local_max_dict_latlng=LOCAL_MAX_DICT_LATLNG,
            projection_object=PROJECTION_OBJECT,
            min_distance_between_maxima_metres=
            SMALL_DISTANCE_BETWEEN_MAXIMA_METRES)

        these_keys = set(list(this_local_max_dict))
        expected_keys = set(list(LOCAL_MAX_DICT_SMALL_DISTANCE))
        self.assertTrue(these_keys == expected_keys)

        for this_key in these_keys:
            self.assertTrue(numpy.allclose(
                this_local_max_dict[this_key],
                LOCAL_MAX_DICT_SMALL_DISTANCE[this_key], atol=TOLERANCE))

    def test_remove_redundant_local_maxima_large_distance(self):
        """Ensures correct output from _remove_redundant_local_maxima.

        In this case, minimum distance between two maxima is large.
        """

        this_local_max_dict = gridrad_tracking._remove_redundant_local_maxima(
            local_max_dict_latlng=LOCAL_MAX_DICT_LATLNG,
            projection_object=PROJECTION_OBJECT,
            min_distance_between_maxima_metres=
            LARGE_DISTANCE_BETWEEN_MAXIMA_METRES)

        these_keys = set(list(this_local_max_dict))
        expected_keys = set(list(LOCAL_MAX_DICT_LARGE_DISTANCE))
        self.assertTrue(these_keys == expected_keys)

        for this_key in these_keys:
            self.assertTrue(numpy.allclose(
                this_local_max_dict[this_key],
                LOCAL_MAX_DICT_LARGE_DISTANCE[this_key], atol=TOLERANCE))

    def test_link_local_maxima_in_time_both_far(self):
        """Ensures correct output from _link_local_maxima_in_time.

        In this case, both current maxima are too far from previous maxima to be
        linked.
        """

        these_current_to_prev_indices = (
            gridrad_tracking._link_local_maxima_in_time(
                current_local_max_dict=CURRENT_LOCAL_MAX_DICT_BOTH_FAR,
                previous_local_max_dict=PREVIOUS_LOCAL_MAX_DICT,
                max_link_time_seconds=MAX_LINK_TIME_SECONDS,
                max_link_distance_m_s01=MAX_LINK_DISTANCE_M_S01))
        self.assertTrue(numpy.array_equal(
            these_current_to_prev_indices, CURRENT_TO_PREV_INDICES_BOTH_FAR))

    def test_link_local_maxima_in_time_one_near(self):
        """Ensures correct output from _link_local_maxima_in_time.

        In this case, only one current max is close enough to previous maxima to
        be linked.
        """

        these_current_to_prev_indices = (
            gridrad_tracking._link_local_maxima_in_time(
                current_local_max_dict=CURRENT_LOCAL_MAX_DICT_ONE_NEAR,
                previous_local_max_dict=PREVIOUS_LOCAL_MAX_DICT,
                max_link_time_seconds=MAX_LINK_TIME_SECONDS,
                max_link_distance_m_s01=MAX_LINK_DISTANCE_M_S01))
        self.assertTrue(numpy.array_equal(
            these_current_to_prev_indices, CURRENT_TO_PREV_INDICES_ONE_NEAR))

    def test_link_local_maxima_in_time_both_near(self):
        """Ensures correct output from _link_local_maxima_in_time.

        In this case, both current maxima are close enough to previous maxima to
        be linked.
        """

        these_current_to_prev_indices = (
            gridrad_tracking._link_local_maxima_in_time(
                current_local_max_dict=CURRENT_LOCAL_MAX_DICT_BOTH_NEAR,
                previous_local_max_dict=PREVIOUS_LOCAL_MAX_DICT,
                max_link_time_seconds=MAX_LINK_TIME_SECONDS,
                max_link_distance_m_s01=MAX_LINK_DISTANCE_M_S01))
        self.assertTrue(numpy.array_equal(
            these_current_to_prev_indices, CURRENT_TO_PREV_INDICES_BOTH_NEAR))

    def test_link_local_maxima_in_time_too_late(self):
        """Ensures correct output from _link_local_maxima_in_time.

        In this case, current minus previous time is too long for linkage.
        """

        these_current_to_prev_indices = (
            gridrad_tracking._link_local_maxima_in_time(
                current_local_max_dict=CURRENT_LOCAL_MAX_DICT_TOO_LATE,
                previous_local_max_dict=PREVIOUS_LOCAL_MAX_DICT,
                max_link_time_seconds=MAX_LINK_TIME_SECONDS,
                max_link_distance_m_s01=MAX_LINK_DISTANCE_M_S01))
        self.assertTrue(numpy.array_equal(
            these_current_to_prev_indices, CURRENT_TO_PREV_INDICES_TOO_LATE))

    def test_link_local_maxima_in_time_overlap(self):
        """Ensures correct output from _link_local_maxima_in_time.

        In this case, both current maxima are close enough to be linked to the
        same previous max.  But this can't happen, so only one current max is
        linked.
        """

        these_current_to_prev_indices = (
            gridrad_tracking._link_local_maxima_in_time(
                current_local_max_dict=CURRENT_LOCAL_MAX_DICT_OVERLAP,
                previous_local_max_dict=PREVIOUS_LOCAL_MAX_DICT,
                max_link_time_seconds=MAX_LINK_TIME_SECONDS,
                max_link_distance_m_s01=MAX_LINK_DISTANCE_M_S01))
        self.assertTrue(numpy.array_equal(
            these_current_to_prev_indices, CURRENT_TO_PREV_INDICES_OVERLAP))

    def test_link_local_maxima_in_time_no_previous_dict(self):
        """Ensures correct output from _link_local_maxima_in_time.

        In this case, `previous_local_max_dict` is None, meaning that there are
        no previous maxima with which to compare.
        """

        these_current_to_prev_indices = (
            gridrad_tracking._link_local_maxima_in_time(
                current_local_max_dict=CURRENT_LOCAL_MAX_DICT_BOTH_NEAR,
                previous_local_max_dict=None,
                max_link_time_seconds=MAX_LINK_TIME_SECONDS,
                max_link_distance_m_s01=MAX_LINK_DISTANCE_M_S01))
        self.assertTrue(numpy.array_equal(
            these_current_to_prev_indices, CURRENT_TO_PREV_INDICES_NO_LINKS))

    def test_link_local_maxima_in_time_no_previous_maxima(self):
        """Ensures correct output from _link_local_maxima_in_time.

        In this case there are no previous maxima.
        """

        these_current_to_prev_indices = (
            gridrad_tracking._link_local_maxima_in_time(
                current_local_max_dict=CURRENT_LOCAL_MAX_DICT_BOTH_NEAR,
                previous_local_max_dict=PREVIOUS_LOCAL_MAX_DICT_EMPTY,
                max_link_time_seconds=MAX_LINK_TIME_SECONDS,
                max_link_distance_m_s01=MAX_LINK_DISTANCE_M_S01))
        self.assertTrue(numpy.array_equal(
            these_current_to_prev_indices, CURRENT_TO_PREV_INDICES_NO_LINKS))

    def test_link_local_maxima_in_time_no_current_maxima(self):
        """Ensures correct output from _link_local_maxima_in_time.

        In this case there are no previous maxima.
        """

        these_current_to_prev_indices = (
            gridrad_tracking._link_local_maxima_in_time(
                current_local_max_dict=CURRENT_LOCAL_MAX_DICT_EMPTY,
                previous_local_max_dict=PREVIOUS_LOCAL_MAX_DICT,
                max_link_time_seconds=MAX_LINK_TIME_SECONDS,
                max_link_distance_m_s01=MAX_LINK_DISTANCE_M_S01))
        self.assertTrue(numpy.array_equal(
            these_current_to_prev_indices, numpy.array([])))

    def test_create_storm_id_first_in_day(self):
        """Ensures correct output from _create_storm_id.

        In this case, storm to be labeled is the first storm in the SPC date.
        """

        this_storm_id, this_numeric_id, this_spc_date_string = (
            gridrad_tracking._create_storm_id(
                storm_start_time_unix_sec=STORM_TIME_UNIX_SEC,
                prev_numeric_id_used=PREV_NUMERIC_ID_USED,
                prev_spc_date_string=PREV_SPC_DATE_STRING))

        self.assertTrue(this_storm_id == STORM_ID_FIRST_IN_DAY)
        self.assertTrue(this_numeric_id == 0)
        self.assertTrue(this_spc_date_string == STORM_SPC_DATE_STRING)

    def test_create_storm_id_second_in_day(self):
        """Ensures correct output from _create_storm_id.

        In this case, storm to be labeled is the second storm in the SPC date.
        """

        this_storm_id, this_numeric_id, this_spc_date_string = (
            gridrad_tracking._create_storm_id(
                storm_start_time_unix_sec=STORM_TIME_UNIX_SEC,
                prev_numeric_id_used=PREV_NUMERIC_ID_USED,
                prev_spc_date_string=STORM_SPC_DATE_STRING))

        self.assertTrue(this_storm_id == STORM_ID_SECOND_IN_DAY)
        self.assertTrue(this_numeric_id == PREV_NUMERIC_ID_USED + 1)
        self.assertTrue(this_spc_date_string == STORM_SPC_DATE_STRING)

    def test_local_maxima_to_storm_tracks(self):
        """Ensures correct output from _local_maxima_to_storm_tracks."""

        this_storm_object_table = (
            gridrad_tracking._local_maxima_to_storm_tracks(
                LOCAL_MAX_DICT_BY_TIME))
        self.assertTrue(this_storm_object_table.equals(STORM_OBJECT_TABLE))

    def test_remove_short_tracks_short_threshold(self):
        """Ensures correct output from _remove_short_tracks.

        In this case, minimum track duration is short, so all tracks should be
        kept.
        """

        this_storm_object_table = copy.deepcopy(STORM_OBJECT_TABLE)
        this_storm_object_table = gridrad_tracking._remove_short_tracks(
            this_storm_object_table,
            min_duration_seconds=SMALL_THRESHOLD_DURATION_SEC)

        self.assertTrue(this_storm_object_table.equals(STORM_OBJECT_TABLE))

    def test_remove_short_tracks_long_threshold(self):
        """Ensures correct output from _remove_short_tracks.

        In this case, minimum track duration is long, so all tracks should be
        removed.
        """

        this_storm_object_table = copy.deepcopy(STORM_OBJECT_TABLE)
        this_storm_object_table = gridrad_tracking._remove_short_tracks(
            this_storm_object_table,
            min_duration_seconds=LARGE_THRESHOLD_DURATION_SEC)

        self.assertTrue(this_storm_object_table.empty)


if __name__ == '__main__':
    unittest.main()
