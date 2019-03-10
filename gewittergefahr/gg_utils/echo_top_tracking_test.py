"""Unit tests for echo_top_tracking.py."""

import copy
import unittest
import numpy
import pandas
from geopy.distance import vincenty
from gewittergefahr.gg_utils import echo_top_tracking
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import time_conversion

TOLERANCE = 1e-6
RELATIVE_DISTANCE_TOLERANCE = 0.015

# The following constants are used to test _find_local_maxima.
RADAR_MATRIX = numpy.array([
    [0, numpy.nan, 3, 4, numpy.nan, 6],
    [7, 8, 9, 10, numpy.nan, numpy.nan],
    [13, 14, numpy.nan, numpy.nan, 17, 18],
    [19, 20, numpy.nan, numpy.nan, numpy.nan, 24],
    [numpy.nan, numpy.nan, 27, 28, 29, 30]
])

RADAR_METADATA_DICT = {
    radar_utils.NW_GRID_POINT_LAT_COLUMN: 35.,
    radar_utils.NW_GRID_POINT_LNG_COLUMN: 95.,
    radar_utils.LAT_SPACING_COLUMN: 0.01,
    radar_utils.LNG_SPACING_COLUMN: 0.02
}

NEIGH_HALF_WIDTH_PIXELS = 1

LOCAL_MAX_ROWS = numpy.array([0, 4], dtype=int)
LOCAL_MAX_COLUMNS = numpy.array([5, 5], dtype=int)
LOCAL_MAX_LATITUDES_DEG = numpy.array([34.96, 35])
LOCAL_MAX_LONGITUDES_DEG = numpy.array([95.1, 95.1])
LOCAL_MAX_VALUES = numpy.array([30, 6], dtype=float)

LOCAL_MAX_DICT_LATLNG = {
    echo_top_tracking.LATITUDES_KEY: LOCAL_MAX_LATITUDES_DEG,
    echo_top_tracking.LONGITUDES_KEY: LOCAL_MAX_LONGITUDES_DEG,
    echo_top_tracking.MAX_VALUES_KEY: LOCAL_MAX_VALUES
}

# The following constants are used to test _remove_redundant_local_maxima.
SMALL_INTERMAX_DISTANCE_METRES = 1000.
LARGE_INTERMAX_DISTANCE_METRES = 10000.

PROJECTION_OBJECT = projections.init_azimuthal_equidistant_projection(
    central_latitude_deg=35., central_longitude_deg=95.)

LOCAL_MAX_X_COORDS_METRES, LOCAL_MAX_Y_COORDS_METRES = (
    projections.project_latlng_to_xy(
        LOCAL_MAX_LATITUDES_DEG, LOCAL_MAX_LONGITUDES_DEG,
        projection_object=PROJECTION_OBJECT,
        false_easting_metres=0., false_northing_metres=0.)
)

LOCAL_MAX_DICT_SMALL_DISTANCE = {
    echo_top_tracking.LATITUDES_KEY: LOCAL_MAX_LATITUDES_DEG,
    echo_top_tracking.LONGITUDES_KEY: LOCAL_MAX_LONGITUDES_DEG,
    echo_top_tracking.MAX_VALUES_KEY: LOCAL_MAX_VALUES,
    echo_top_tracking.X_COORDS_KEY: LOCAL_MAX_X_COORDS_METRES,
    echo_top_tracking.Y_COORDS_KEY: LOCAL_MAX_Y_COORDS_METRES
}

LOCAL_MAX_DICT_LARGE_DISTANCE = {
    echo_top_tracking.LATITUDES_KEY: LOCAL_MAX_LATITUDES_DEG[:-1],
    echo_top_tracking.LONGITUDES_KEY: LOCAL_MAX_LONGITUDES_DEG[:-1],
    echo_top_tracking.MAX_VALUES_KEY: LOCAL_MAX_VALUES[:-1],
    echo_top_tracking.X_COORDS_KEY: LOCAL_MAX_X_COORDS_METRES[:-1],
    echo_top_tracking.Y_COORDS_KEY: LOCAL_MAX_Y_COORDS_METRES[:-1]
}

# The following constants are used to test _estimate_velocity_by_neigh.
X_COORDS_FOR_NEIGH_METRES = numpy.array([
    0, 1.5, 3, 4.5,
    0, 1.5, 3, 4.5,
    0, 1.5, 3, 4.5
])
Y_COORDS_FOR_NEIGH_METRES = numpy.array([
    0, 0, 0, 0,
    2, 2, 2, 2,
    4, 4, 4, 4
], dtype=float)

X_VELOCITIES_WITH_NAN_M_S01 = numpy.array([
    numpy.nan, 5, 8, 3,
    14, numpy.nan, 1, 5,
    5, 7, numpy.nan, 7
])
Y_VELOCITIES_WITH_NAN_M_S01 = numpy.array([
    numpy.nan, -4, 0, 0,
    3, numpy.nan, 2, 6,
    6, -2, numpy.nan, 3
])

VELOCITY_EFOLD_RADIUS_METRES = 1.

THESE_WEIGHTS = numpy.array([
    numpy.nan, 1.5, 3, numpy.nan,
    2, numpy.nan, numpy.nan, numpy.nan,
    numpy.nan, numpy.nan, numpy.nan, numpy.nan
])

THESE_WEIGHTS = numpy.exp(-1 * THESE_WEIGHTS)
THESE_WEIGHTS[numpy.isnan(THESE_WEIGHTS)] = 0.
THESE_WEIGHTS = THESE_WEIGHTS / numpy.sum(THESE_WEIGHTS)

FIRST_X_VELOCITY_M_S01 = numpy.nansum(
    THESE_WEIGHTS * X_VELOCITIES_WITH_NAN_M_S01)
FIRST_Y_VELOCITY_M_S01 = numpy.nansum(
    THESE_WEIGHTS * Y_VELOCITIES_WITH_NAN_M_S01)

THESE_WEIGHTS = numpy.array([
    numpy.nan, 2, numpy.sqrt(6.25), numpy.nan,
    1.5, numpy.nan, 1.5, 3,
    numpy.sqrt(6.25), 2, numpy.nan, numpy.nan
])

THESE_WEIGHTS = numpy.exp(-1 * THESE_WEIGHTS)
THESE_WEIGHTS[numpy.isnan(THESE_WEIGHTS)] = 0.
THESE_WEIGHTS = THESE_WEIGHTS / numpy.sum(THESE_WEIGHTS)

SECOND_X_VELOCITY_M_S01 = numpy.nansum(
    THESE_WEIGHTS * X_VELOCITIES_WITH_NAN_M_S01)
SECOND_Y_VELOCITY_M_S01 = numpy.nansum(
    THESE_WEIGHTS * Y_VELOCITIES_WITH_NAN_M_S01)

THESE_WEIGHTS = numpy.array([
    numpy.nan, numpy.nan, numpy.nan, numpy.nan,
    numpy.nan, numpy.nan, 2, numpy.sqrt(6.25),
    3, 1.5, numpy.nan, 1.5
])

THESE_WEIGHTS = numpy.exp(-1 * THESE_WEIGHTS)
THESE_WEIGHTS[numpy.isnan(THESE_WEIGHTS)] = 0.
THESE_WEIGHTS = THESE_WEIGHTS / numpy.sum(THESE_WEIGHTS)

THIRD_X_VELOCITY_M_S01 = numpy.nansum(
    THESE_WEIGHTS * X_VELOCITIES_WITH_NAN_M_S01)
THIRD_Y_VELOCITY_M_S01 = numpy.nansum(
    THESE_WEIGHTS * Y_VELOCITIES_WITH_NAN_M_S01)

X_VELOCITIES_NO_NAN_M_S01 = numpy.array([
    FIRST_X_VELOCITY_M_S01, 5, 8, 3,
    14, SECOND_X_VELOCITY_M_S01, 1, 5,
    5, 7, THIRD_X_VELOCITY_M_S01, 7
])
Y_VELOCITIES_NO_NAN_M_S01 = numpy.array([
    FIRST_Y_VELOCITY_M_S01, -4, 0, 0,
    3, SECOND_Y_VELOCITY_M_S01, 2, 6,
    6, -2, THIRD_Y_VELOCITY_M_S01, 3
])

# The following constants are used to test _get_intermediate_velocities.
THESE_X_COORDS_METRES = numpy.array([2, -7, 1, 6, 5, -4], dtype=float)
THESE_Y_COORDS_METRES = numpy.array([4, -1, 5, -1, -3, 9], dtype=float)

FIRST_LMAX_DICT_NO_VELOCITY = {
    echo_top_tracking.VALID_TIME_KEY: 0,
    echo_top_tracking.X_COORDS_KEY: THESE_X_COORDS_METRES + 0.,
    echo_top_tracking.Y_COORDS_KEY: THESE_Y_COORDS_METRES + 0.
}

THESE_X_COORDS_METRES = numpy.array(
    [13, -1, 20, 20, -8, 5, -23, 19], dtype=float)
THESE_Y_COORDS_METRES = numpy.array(
    [-14, 25, -12, 1, -14, 4, -5, 18], dtype=float)
THESE_CURRENT_TO_PREV_INDICES = numpy.array(
    [-1, 1, 2, -1, -1, 4, -1, 5], dtype=int)

SECOND_LMAX_DICT_NO_VELOCITY = {
    echo_top_tracking.VALID_TIME_KEY: 10,
    echo_top_tracking.X_COORDS_KEY: THESE_X_COORDS_METRES + 0.,
    echo_top_tracking.Y_COORDS_KEY: THESE_Y_COORDS_METRES + 0.,
    echo_top_tracking.CURRENT_TO_PREV_INDICES_KEY:
        THESE_CURRENT_TO_PREV_INDICES + 0
}

FIRST_LMAX_DICT_WITH_VELOCITY = copy.deepcopy(FIRST_LMAX_DICT_NO_VELOCITY)
FIRST_LMAX_DICT_WITH_VELOCITY.update({
    echo_top_tracking.X_VELOCITIES_KEY: numpy.full(6, numpy.nan),
    echo_top_tracking.Y_VELOCITIES_KEY: numpy.full(6, numpy.nan)
})

SECOND_LMAX_DICT_WITH_VELOCITY = copy.deepcopy(SECOND_LMAX_DICT_NO_VELOCITY)

THESE_X_VELOCITIES_M_S01 = numpy.array(
    [numpy.nan, 0.6, 1.9, numpy.nan, numpy.nan, 0, numpy.nan, 2.3])
THESE_Y_VELOCITIES_M_S01 = numpy.array(
    [numpy.nan, 2.6, -1.7, numpy.nan, numpy.nan, 0.7, numpy.nan, 0.9])

SECOND_LMAX_DICT_WITH_VELOCITY.update({
    echo_top_tracking.X_VELOCITIES_KEY: THESE_X_VELOCITIES_M_S01,
    echo_top_tracking.Y_VELOCITIES_KEY: THESE_Y_VELOCITIES_M_S01
})

# The following constants are used to test _link_local_maxima_in_time.
FIRST_LOCAL_MAX_DICT_UNLINKED = copy.deepcopy(FIRST_LMAX_DICT_WITH_VELOCITY)
SECOND_LOCAL_MAX_DICT_UNLINKED = copy.deepcopy(SECOND_LMAX_DICT_WITH_VELOCITY)
SECOND_LOCAL_MAX_DICT_UNLINKED.pop(
    echo_top_tracking.CURRENT_TO_PREV_INDICES_KEY)

MAX_LINK_TIME_SECONDS = 100
MAX_VELOCITY_DIFF_M_S01 = 3.
MAX_LINK_DISTANCE_M_S01 = 2.

SECOND_TO_FIRST_INDICES = numpy.array([4, 5, -1, 3, 1, 0, -1, -1], dtype=int)

THESE_X_COORDS_METRES = numpy.array(
    [12, 1, 21, 13, -3, 6, -15, 20], dtype=float)
THESE_Y_COORDS_METRES = numpy.array(
    [-21, 27, -13, 10, -15, -9, -7, 18], dtype=float)

THIRD_LOCAL_MAX_DICT_UNLINKED = {
    echo_top_tracking.VALID_TIME_KEY: 13,
    echo_top_tracking.X_COORDS_KEY: THESE_X_COORDS_METRES + 0.,
    echo_top_tracking.Y_COORDS_KEY: THESE_Y_COORDS_METRES + 0.,
}

THIRD_TO_SECOND_INDICES = numpy.array([-1, 1, 2, 5, 4, -1, -1, 7], dtype=int)

SECOND_LOCAL_MAX_DICT_EMPTY = {
    echo_top_tracking.VALID_TIME_KEY: 10,
    echo_top_tracking.X_COORDS_KEY: numpy.array([]),
    echo_top_tracking.Y_COORDS_KEY: numpy.array([])
}

THIRD_LOCAL_MAX_DICT_EMPTY = {
    echo_top_tracking.VALID_TIME_KEY: 13,
    echo_top_tracking.X_COORDS_KEY: numpy.array([]),
    echo_top_tracking.Y_COORDS_KEY: numpy.array([])
}

# The following constants are used to test _create_storm_id.
STORM_TIME_UNIX_SEC = 1516860900  # 0615 UTC 25 Jan 2018
STORM_SPC_DATE_STRING = '20180124'
PREV_SPC_DATE_STRING = '20180123'
PREV_NUMERIC_ID_USED = 0

STORM_ID_FIRST_IN_DAY = '000000_20180124'
STORM_ID_SECOND_IN_DAY = '000001_20180124'

PREVIOUS_TIME_UNIX_SEC = 1516860600  # 0610 UTC 25 Jan 2018
CURRENT_TIME_UNIX_SEC = 1516860900  # 0615 UTC 25 Jan 2018
CURRENT_TO_PREV_INDICES_NO_LINKS = numpy.array([-1, -1], dtype=int)
CURRENT_TO_PREV_INDICES_BOTH_NEAR = numpy.array([0, 1], dtype=int)

# The following constants are used to test _local_maxima_to_storm_tracks.
FIRST_LOCAL_MAX_DICT_LINKED = {
    echo_top_tracking.LATITUDES_KEY: LOCAL_MAX_LATITUDES_DEG,
    echo_top_tracking.LONGITUDES_KEY: LOCAL_MAX_LONGITUDES_DEG,
    echo_top_tracking.X_COORDS_KEY: LOCAL_MAX_X_COORDS_METRES,
    echo_top_tracking.Y_COORDS_KEY: LOCAL_MAX_Y_COORDS_METRES,
    echo_top_tracking.VALID_TIME_KEY: PREVIOUS_TIME_UNIX_SEC,
    echo_top_tracking.CURRENT_TO_PREV_INDICES_KEY:
        CURRENT_TO_PREV_INDICES_NO_LINKS
}

SECOND_LOCAL_MAX_DICT_LINKED = {
    echo_top_tracking.LATITUDES_KEY: LOCAL_MAX_LATITUDES_DEG,
    echo_top_tracking.LONGITUDES_KEY: LOCAL_MAX_LONGITUDES_DEG,
    echo_top_tracking.X_COORDS_KEY: LOCAL_MAX_X_COORDS_METRES,
    echo_top_tracking.Y_COORDS_KEY: LOCAL_MAX_Y_COORDS_METRES,
    echo_top_tracking.VALID_TIME_KEY: CURRENT_TIME_UNIX_SEC,
    echo_top_tracking.CURRENT_TO_PREV_INDICES_KEY:
        CURRENT_TO_PREV_INDICES_BOTH_NEAR
}

LOCAL_MAX_DICT_BY_TIME = [
    FIRST_LOCAL_MAX_DICT_LINKED, SECOND_LOCAL_MAX_DICT_LINKED
]

THESE_STORM_IDS = [
    STORM_ID_FIRST_IN_DAY, STORM_ID_SECOND_IN_DAY, STORM_ID_FIRST_IN_DAY,
    STORM_ID_SECOND_IN_DAY
]
THESE_TIMES_UNIX_SEC = numpy.array([
    PREVIOUS_TIME_UNIX_SEC, PREVIOUS_TIME_UNIX_SEC, CURRENT_TIME_UNIX_SEC,
    CURRENT_TIME_UNIX_SEC
])
THESE_SPC_DATES_UNIX_SEC = numpy.full(
    4, time_conversion.time_to_spc_date_unix_sec(PREVIOUS_TIME_UNIX_SEC),
    dtype=int)

THESE_CENTROID_LATITUDES_DEG = numpy.concatenate((
    LOCAL_MAX_LATITUDES_DEG, LOCAL_MAX_LATITUDES_DEG
))
THESE_CENTROID_LONGITUDES_DEG = numpy.concatenate((
    LOCAL_MAX_LONGITUDES_DEG, LOCAL_MAX_LONGITUDES_DEG
))
THESE_CENTROID_X_METRES = numpy.concatenate((
    LOCAL_MAX_X_COORDS_METRES, LOCAL_MAX_X_COORDS_METRES
))
THESE_CENTROID_Y_METRES = numpy.concatenate((
    LOCAL_MAX_Y_COORDS_METRES, LOCAL_MAX_Y_COORDS_METRES
))

STORM_OBJECT_DICT = {
    tracking_utils.STORM_ID_COLUMN: THESE_STORM_IDS,
    tracking_utils.TIME_COLUMN: THESE_TIMES_UNIX_SEC,
    tracking_utils.SPC_DATE_COLUMN: THESE_SPC_DATES_UNIX_SEC,
    tracking_utils.CENTROID_LAT_COLUMN: THESE_CENTROID_LATITUDES_DEG,
    tracking_utils.CENTROID_LNG_COLUMN: THESE_CENTROID_LONGITUDES_DEG,
    echo_top_tracking.CENTROID_X_COLUMN: THESE_CENTROID_X_METRES,
    echo_top_tracking.CENTROID_Y_COLUMN: THESE_CENTROID_Y_METRES
}

STORM_OBJECT_TABLE = pandas.DataFrame.from_dict(STORM_OBJECT_DICT)

# The following constants are used to test _remove_short_lived_tracks.
SHORT_MIN_LIFETIME_SEC = 100
LONG_MIN_LIFETIME_SEC = 1000

# The following constants are used to test _get_final_velocities_one_track.
ONE_TRACK_LATITUDES_DEG = numpy.array([40, 40, 41, 41, 40, 40], dtype=float)
ONE_TRACK_LONGITUDES_DEG = numpy.array(
    [265, 266, 266, 267, 267, 266], dtype=float)
ONE_TRACK_TIMES_UNIX_SEC = numpy.array([0, 1, 2, 3, 4, 5], dtype=int)

DEGREES_LAT_TO_METRES = 60. * 1852
DEGREES_TO_RADIANS = numpy.pi / 180

V_VELOCITIES_1POINT_M_S01 = DEGREES_LAT_TO_METRES * numpy.array([
    numpy.nan, 0, 1, 0, -1, 0])

U_VELOCITIES_1POINT_M_S01 = DEGREES_LAT_TO_METRES * numpy.array([
    numpy.nan, numpy.cos(40 * DEGREES_TO_RADIANS), 0,
    numpy.cos(41 * DEGREES_TO_RADIANS), 0, -numpy.cos(40 * DEGREES_TO_RADIANS)
])

V_VELOCITIES_2POINTS_M_S01 = DEGREES_LAT_TO_METRES * numpy.array(
    [numpy.nan, 0, 0.5, 0.5, -0.5, -0.5])

U_VELOCITIES_2POINTS_M_S01 = DEGREES_LAT_TO_METRES * numpy.array([
    numpy.nan,
    numpy.cos(40. * DEGREES_TO_RADIANS),
    0.5 * numpy.cos(40.5 * DEGREES_TO_RADIANS),
    0.5 * numpy.cos(40.5 * DEGREES_TO_RADIANS),
    0.5 * numpy.cos(40.5 * DEGREES_TO_RADIANS),
    -0.5 * numpy.cos(40.5 * DEGREES_TO_RADIANS)
])

# The following constants are used to test _remove_small_polygons.
THIS_LIST_OF_ROW_ARRAYS = [
    numpy.array([0, 0, 0, 0, 1, 1, 2, 2, 2], dtype=int),
    numpy.array([-5, -4, -3], dtype=int),
    numpy.array([0, 1, 1, 2, 3, 5, 8, 13, 6, 6, 6], dtype=int)
]

LOCAL_MAX_DICT_WITH_SMALL = {
    echo_top_tracking.GRID_POINT_ROWS_KEY: THIS_LIST_OF_ROW_ARRAYS,
    echo_top_tracking.LATITUDES_KEY: numpy.array([51.1, 53.5, 60]),
    echo_top_tracking.LONGITUDES_KEY: numpy.array([246, 246.5, 250])
}

MIN_POLYGON_SIZE_PIXELS = 5
THIS_LIST_OF_ROW_ARRAYS = [
    numpy.array([0, 0, 0, 0, 1, 1, 2, 2, 2], dtype=int),
    numpy.array([0, 1, 1, 2, 3, 5, 8, 13, 6, 6, 6], dtype=int)
]

LOCAL_MAX_DICT_WITHOUT_SMALL = {
    echo_top_tracking.GRID_POINT_ROWS_KEY: THIS_LIST_OF_ROW_ARRAYS,
    echo_top_tracking.LATITUDES_KEY: numpy.array([51.1, 60]),
    echo_top_tracking.LONGITUDES_KEY: numpy.array([246, 250])
}

# The following constants are used to test _join_tracks.
LONG_MAX_JOIN_TIME_SEC = 300

THESE_STORM_IDS = ['a', 'b', 'a', 'b']
THESE_TIMES_UNIX_SEC = numpy.array([0, 0, 300, 300], dtype=int)
THESE_LATITUDES_DEG = numpy.array([30, 40, 30, 40], dtype=float)
THESE_LONGITUDES_DEG = numpy.array([290, 300, 290, 300], dtype=float)

THIS_DICT = {
    tracking_utils.STORM_ID_COLUMN: THESE_STORM_IDS,
    tracking_utils.TIME_COLUMN: THESE_TIMES_UNIX_SEC,
    tracking_utils.CENTROID_LAT_COLUMN: THESE_LATITUDES_DEG,
    tracking_utils.CENTROID_LNG_COLUMN: THESE_LONGITUDES_DEG,
    tracking_utils.EAST_VELOCITY_COLUMN: numpy.full(4, 0.),
    tracking_utils.NORTH_VELOCITY_COLUMN: numpy.full(4, 0.)
}

EARLY_STORM_OBJECT_TABLE = pandas.DataFrame.from_dict(THIS_DICT)

THESE_STORM_IDS = ['c', 'd', 'c', 'd']
THESE_TIMES_UNIX_SEC = numpy.array([600, 600, 900, 900], dtype=int)
THESE_LATITUDES_DEG = numpy.array([40, 50, 40, 50], dtype=float)
THESE_LONGITUDES_DEG = numpy.array([300, 250, 300, 250], dtype=float)

THIS_DICT = {
    tracking_utils.STORM_ID_COLUMN: THESE_STORM_IDS,
    tracking_utils.TIME_COLUMN: THESE_TIMES_UNIX_SEC,
    tracking_utils.CENTROID_LAT_COLUMN: THESE_LATITUDES_DEG,
    tracking_utils.CENTROID_LNG_COLUMN: THESE_LONGITUDES_DEG
}

LATE_STORM_OBJECT_TABLE = pandas.DataFrame.from_dict(THIS_DICT)

THIS_DICT = {
    tracking_utils.STORM_ID_COLUMN: ['b', 'd', 'b', 'd'],
    tracking_utils.TIME_COLUMN: THESE_TIMES_UNIX_SEC,
    tracking_utils.CENTROID_LAT_COLUMN: THESE_LATITUDES_DEG,
    tracking_utils.CENTROID_LNG_COLUMN: THESE_LONGITUDES_DEG
}

JOINED_STORM_OBJECT_TABLE = pandas.DataFrame.from_dict(THIS_DICT)

# The following constants are used to test _storm_objects_to_tracks.
THESE_STORM_IDS = ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E', 'E']
THESE_TIMES_UNIX_SEC = numpy.array([6, 7, 3, 4, 0, 1, 0, 1, 4, 5], dtype=int)
THESE_LATITUDES_DEG = numpy.array(
    [53.5, 53.5, 53.5, 53.5, 53.5, 53.5, 53.5, 53.5, 47.6, 47.6])
THESE_LONGITUDES_DEG = numpy.array(
    [113.5, 113.6, 113.2, 113.3, 112.9, 113, 113.2, 113.3, 307.3, 307.3])

THIS_DICT = {
    tracking_utils.STORM_ID_COLUMN: THESE_STORM_IDS,
    tracking_utils.TIME_COLUMN: THESE_TIMES_UNIX_SEC,
    tracking_utils.CENTROID_LAT_COLUMN: THESE_LATITUDES_DEG,
    tracking_utils.CENTROID_LNG_COLUMN: THESE_LONGITUDES_DEG
}

STORM_OBJECT_TABLE_BEFORE_REANALYSIS = pandas.DataFrame.from_dict(THIS_DICT)

THESE_STORM_IDS = ['A', 'B', 'C', 'D', 'E']
THESE_START_TIMES_UNIX_SEC = numpy.array([6, 3, 0, 0, 4], dtype=int)
THESE_END_TIMES_UNIX_SEC = numpy.array([7, 4, 1, 1, 5], dtype=int)
THESE_START_LATITUDES_DEG = numpy.array([53.5, 53.5, 53.5, 53.5, 47.6])
THESE_END_LATITUDES_DEG = numpy.array([53.5, 53.5, 53.5, 53.5, 47.6])
THESE_START_LONGITUDES_DEG = numpy.array([113.5, 113.2, 112.9, 113.2, 307.3])
THESE_END_LONGITUDES_DEG = numpy.array([113.6, 113.3, 113, 113.3, 307.3])

THIS_DICT = {
    tracking_utils.STORM_ID_COLUMN: THESE_STORM_IDS,
    echo_top_tracking.START_TIME_COLUMN: THESE_START_TIMES_UNIX_SEC,
    echo_top_tracking.END_TIME_COLUMN: THESE_END_TIMES_UNIX_SEC,
    echo_top_tracking.START_LATITUDE_COLUMN: THESE_START_LATITUDES_DEG,
    echo_top_tracking.END_LATITUDE_COLUMN: THESE_END_LATITUDES_DEG,
    echo_top_tracking.START_LONGITUDE_COLUMN: THESE_START_LONGITUDES_DEG,
    echo_top_tracking.END_LONGITUDE_COLUMN: THESE_END_LONGITUDES_DEG
}

STORM_TRACK_TABLE_BEFORE_REANALYSIS = pandas.DataFrame.from_dict(THIS_DICT)

STORM_TRACK_TABLE_TAMPERED = copy.deepcopy(STORM_TRACK_TABLE_BEFORE_REANALYSIS)
STORM_TRACK_TABLE_TAMPERED[echo_top_tracking.START_TIME_COLUMN].values[2] = -1
STORM_ID_TAMPERED = 'C'

# The following constants are used to test _get_join_error.
DIST_TOLERANCE_METRES = 1.
JOIN_ERROR_B_TO_A_METRES = 0.
JOIN_ERROR_C_TO_A_METRES = 0.
JOIN_ERROR_D_TO_A_METRES = vincenty((53.5, 113.8), (53.5, 113.5)).meters
JOIN_ERROR_E_TO_A_METRES = vincenty((47.6, 307.3), (53.5, 113.5)).meters

# The following constants are used to test _find_nearby_tracks and
# _reanalyze_tracks.
SHORT_MAX_JOIN_TIME_SEC = 2
MAX_JOIN_ERROR_M_S01 = 1000.

NEARBY_IDS_FOR_A = ['B']
NEARBY_IDS_FOR_B = ['C']
NEARBY_IDS_FOR_C = []
NEARBY_IDS_FOR_D = []
NEARBY_IDS_FOR_E = []

THESE_STORM_IDS = ['A', 'A', 'A', 'A', 'A', 'A', 'D', 'D', 'E', 'E']

THIS_DICT = {
    tracking_utils.STORM_ID_COLUMN: THESE_STORM_IDS,
    tracking_utils.TIME_COLUMN: THESE_TIMES_UNIX_SEC,
    tracking_utils.CENTROID_LAT_COLUMN: THESE_LATITUDES_DEG,
    tracking_utils.CENTROID_LNG_COLUMN: THESE_LONGITUDES_DEG
}

STORM_OBJECT_TABLE_AFTER_REANALYSIS = pandas.DataFrame.from_dict(THIS_DICT)

THESE_STORM_IDS = ['A', 'D', 'E']
THESE_START_TIMES_UNIX_SEC = numpy.array([0, 0, 4], dtype=int)
THESE_END_TIMES_UNIX_SEC = numpy.array([7, 1, 5], dtype=int)
THESE_START_LATITUDES_DEG = numpy.array([53.5, 53.5, 47.6])
THESE_END_LATITUDES_DEG = numpy.array([53.5, 53.5, 47.6])
THESE_START_LONGITUDES_DEG = numpy.array([112.9, 113.2, 307.3])
THESE_END_LONGITUDES_DEG = numpy.array([113.6, 113.3, 307.3])

THIS_DICT = {
    tracking_utils.STORM_ID_COLUMN: THESE_STORM_IDS,
    echo_top_tracking.START_TIME_COLUMN: THESE_START_TIMES_UNIX_SEC,
    echo_top_tracking.END_TIME_COLUMN: THESE_END_TIMES_UNIX_SEC,
    echo_top_tracking.START_LATITUDE_COLUMN: THESE_START_LATITUDES_DEG,
    echo_top_tracking.END_LATITUDE_COLUMN: THESE_END_LATITUDES_DEG,
    echo_top_tracking.START_LONGITUDE_COLUMN: THESE_START_LONGITUDES_DEG,
    echo_top_tracking.END_LONGITUDE_COLUMN: THESE_END_LONGITUDES_DEG
}

STORM_TRACK_TABLE_AFTER_REANALYSIS = pandas.DataFrame.from_dict(THIS_DICT)


def _compare_maxima_with_sans_small_polygons(
        first_local_max_dict, second_local_max_dict):
    """Compares local maxima before and after removing small polygons.

    :param first_local_max_dict: First dictionary.
    :param second_local_max_dict: Second dictionary.
    :return: are_dicts_equal: Boolean flag.
    """

    first_keys = first_local_max_dict.keys()
    second_keys = second_local_max_dict.keys()
    if set(first_keys) != set(second_keys):
        return False

    for this_key in first_keys:
        if this_key == echo_top_tracking.GRID_POINT_ROWS_KEY:
            first_length = len(first_local_max_dict[this_key])
            second_length = len(second_local_max_dict[this_key])
            if first_length != second_length:
                return False

            for i in range(first_length):
                if not numpy.array_equal(first_local_max_dict[this_key][i],
                                         second_local_max_dict[this_key][i]):
                    return False

        else:
            if not numpy.allclose(first_local_max_dict[this_key],
                                  second_local_max_dict[this_key],
                                  atol=TOLERANCE):
                return False

    return True


class EchoTopTrackingTests(unittest.TestCase):
    """Each method is a unit test for echo_top_tracking.py."""

    def test_find_local_maxima(self):
        """Ensures correct output from _find_local_maxima."""

        this_local_max_dict = echo_top_tracking._find_local_maxima(
            radar_matrix=RADAR_MATRIX, radar_metadata_dict=RADAR_METADATA_DICT,
            neigh_half_width_pixels=NEIGH_HALF_WIDTH_PIXELS)

        these_keys = set(list(this_local_max_dict))
        expected_keys = set(list(LOCAL_MAX_DICT_LATLNG))
        self.assertTrue(these_keys == expected_keys)

        for this_key in these_keys:
            self.assertTrue(numpy.allclose(
                this_local_max_dict[this_key], LOCAL_MAX_DICT_LATLNG[this_key],
                atol=TOLERANCE
            ))

    def test_remove_redundant_local_maxima_small_distance(self):
        """Ensures correct output from _remove_redundant_local_maxima.

        In this case, minimum distance between two maxima is small.
        """

        this_local_max_dict = echo_top_tracking._remove_redundant_local_maxima(
            local_max_dict=copy.deepcopy(LOCAL_MAX_DICT_LATLNG),
            projection_object=PROJECTION_OBJECT,
            min_intermax_distance_metres=SMALL_INTERMAX_DISTANCE_METRES)

        these_keys = set(list(this_local_max_dict))
        expected_keys = set(list(LOCAL_MAX_DICT_SMALL_DISTANCE))
        self.assertTrue(these_keys == expected_keys)

        for this_key in these_keys:
            self.assertTrue(numpy.allclose(
                this_local_max_dict[this_key],
                LOCAL_MAX_DICT_SMALL_DISTANCE[this_key], atol=TOLERANCE
            ))

    def test_remove_redundant_local_maxima_large_distance(self):
        """Ensures correct output from _remove_redundant_local_maxima.

        In this case, minimum distance between two maxima is large.
        """

        this_local_max_dict = echo_top_tracking._remove_redundant_local_maxima(
            local_max_dict=copy.deepcopy(LOCAL_MAX_DICT_LATLNG),
            projection_object=PROJECTION_OBJECT,
            min_intermax_distance_metres=LARGE_INTERMAX_DISTANCE_METRES)

        these_keys = set(list(this_local_max_dict))
        expected_keys = set(list(LOCAL_MAX_DICT_LARGE_DISTANCE))
        self.assertTrue(these_keys == expected_keys)

        for this_key in these_keys:
            self.assertTrue(numpy.allclose(
                this_local_max_dict[this_key],
                LOCAL_MAX_DICT_LARGE_DISTANCE[this_key], atol=TOLERANCE
            ))

    def test_estimate_velocity_by_neigh(self):
        """Ensures correct output from _estimate_velocity_by_neigh."""

        these_x_velocities_m_s01, these_y_velocities_m_s01 = (
            echo_top_tracking._estimate_velocity_by_neigh(
                x_coords_metres=X_COORDS_FOR_NEIGH_METRES,
                y_coords_metres=Y_COORDS_FOR_NEIGH_METRES,
                x_velocities_m_s01=X_VELOCITIES_WITH_NAN_M_S01 + 0.,
                y_velocities_m_s01=Y_VELOCITIES_WITH_NAN_M_S01 + 0.,
                e_folding_radius_metres=VELOCITY_EFOLD_RADIUS_METRES)
        )

        self.assertTrue(numpy.allclose(
            these_x_velocities_m_s01, X_VELOCITIES_NO_NAN_M_S01, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            these_y_velocities_m_s01, Y_VELOCITIES_NO_NAN_M_S01, atol=TOLERANCE
        ))

    def test_get_intermediate_velocities_time1(self):
        """Ensures correct output from _get_intermediate_velocities.

        In this case, "current time" = first time.
        """

        this_local_max_dict = echo_top_tracking._get_intermediate_velocities(
            current_local_max_dict=copy.deepcopy(FIRST_LMAX_DICT_NO_VELOCITY),
            previous_local_max_dict=None,
            e_folding_radius_metres=VELOCITY_EFOLD_RADIUS_METRES)

        self.assertTrue(numpy.allclose(
            this_local_max_dict[echo_top_tracking.X_VELOCITIES_KEY],
            FIRST_LMAX_DICT_WITH_VELOCITY[echo_top_tracking.X_VELOCITIES_KEY],
            atol=TOLERANCE, equal_nan=True
        ))

        self.assertTrue(numpy.allclose(
            this_local_max_dict[echo_top_tracking.Y_VELOCITIES_KEY],
            FIRST_LMAX_DICT_WITH_VELOCITY[echo_top_tracking.Y_VELOCITIES_KEY],
            atol=TOLERANCE, equal_nan=True
        ))

    def test_get_intermediate_velocities_time2(self):
        """Ensures correct output from _get_intermediate_velocities.

        In this case, "current time" = second time.
        """

        this_local_max_dict = echo_top_tracking._get_intermediate_velocities(
            current_local_max_dict=copy.deepcopy(SECOND_LMAX_DICT_NO_VELOCITY),
            previous_local_max_dict=copy.deepcopy(FIRST_LMAX_DICT_NO_VELOCITY),
            e_folding_radius_metres=VELOCITY_EFOLD_RADIUS_METRES)

        self.assertTrue(numpy.allclose(
            this_local_max_dict[echo_top_tracking.X_VELOCITIES_KEY],
            SECOND_LMAX_DICT_WITH_VELOCITY[echo_top_tracking.X_VELOCITIES_KEY],
            atol=TOLERANCE, equal_nan=True
        ))

        self.assertTrue(numpy.allclose(
            this_local_max_dict[echo_top_tracking.Y_VELOCITIES_KEY],
            SECOND_LMAX_DICT_WITH_VELOCITY[echo_top_tracking.Y_VELOCITIES_KEY],
            atol=TOLERANCE, equal_nan=True
        ))

    def test_link_local_maxima_in_time_1to2(self):
        """Ensures correct output from _link_local_maxima_in_time.

        In this case, linking maxima from the first and second times.
        """

        these_indices = echo_top_tracking._link_local_maxima_in_time(
            current_local_max_dict=SECOND_LOCAL_MAX_DICT_UNLINKED,
            previous_local_max_dict=FIRST_LOCAL_MAX_DICT_UNLINKED,
            max_link_time_seconds=MAX_LINK_TIME_SECONDS,
            max_velocity_diff_m_s01=MAX_VELOCITY_DIFF_M_S01,
            max_link_distance_m_s01=MAX_LINK_DISTANCE_M_S01)

        self.assertTrue(numpy.array_equal(
            these_indices, SECOND_TO_FIRST_INDICES
        ))

    def test_link_local_maxima_in_time_2to3(self):
        """Ensures correct output from _link_local_maxima_in_time.

        In this case, linking maxima from the second and third times.
        """

        these_indices = echo_top_tracking._link_local_maxima_in_time(
            current_local_max_dict=THIRD_LOCAL_MAX_DICT_UNLINKED,
            previous_local_max_dict=SECOND_LOCAL_MAX_DICT_UNLINKED,
            max_link_time_seconds=MAX_LINK_TIME_SECONDS,
            max_velocity_diff_m_s01=MAX_VELOCITY_DIFF_M_S01,
            max_link_distance_m_s01=MAX_LINK_DISTANCE_M_S01)

        self.assertTrue(numpy.array_equal(
            these_indices, THIRD_TO_SECOND_INDICES
        ))

    def test_link_local_maxima_in_time_too_late(self):
        """Ensures correct output from _link_local_maxima_in_time.

        In this case, current time is too much later than previous time.
        """

        these_indices = echo_top_tracking._link_local_maxima_in_time(
            current_local_max_dict=THIRD_LOCAL_MAX_DICT_UNLINKED,
            previous_local_max_dict=SECOND_LOCAL_MAX_DICT_UNLINKED,
            max_link_time_seconds=1,
            max_velocity_diff_m_s01=MAX_VELOCITY_DIFF_M_S01,
            max_link_distance_m_s01=MAX_LINK_DISTANCE_M_S01)

        expected_indices = numpy.full(len(these_indices), -1, dtype=int)
        self.assertTrue(numpy.array_equal(these_indices, expected_indices))

    def test_link_local_maxima_in_time_no_prev_dict(self):
        """Ensures correct output from _link_local_maxima_in_time.

        In this case there is no dictionary with previous maxima.
        """

        these_indices = echo_top_tracking._link_local_maxima_in_time(
            current_local_max_dict=THIRD_LOCAL_MAX_DICT_UNLINKED,
            previous_local_max_dict=None,
            max_link_time_seconds=MAX_LINK_TIME_SECONDS,
            max_velocity_diff_m_s01=MAX_VELOCITY_DIFF_M_S01,
            max_link_distance_m_s01=MAX_LINK_DISTANCE_M_S01)

        expected_indices = numpy.full(len(these_indices), -1, dtype=int)
        self.assertTrue(numpy.array_equal(these_indices, expected_indices))

    def test_link_local_maxima_in_time_no_prev_maxima(self):
        """Ensures correct output from _link_local_maxima_in_time.

        In this case there are no previous maxima.
        """

        these_indices = echo_top_tracking._link_local_maxima_in_time(
            current_local_max_dict=THIRD_LOCAL_MAX_DICT_UNLINKED,
            previous_local_max_dict=SECOND_LOCAL_MAX_DICT_EMPTY,
            max_link_time_seconds=MAX_LINK_TIME_SECONDS,
            max_velocity_diff_m_s01=MAX_VELOCITY_DIFF_M_S01,
            max_link_distance_m_s01=MAX_LINK_DISTANCE_M_S01)

        expected_indices = numpy.full(len(these_indices), -1, dtype=int)
        self.assertTrue(numpy.array_equal(these_indices, expected_indices))

    def test_link_local_maxima_in_time_no_current_maxima(self):
        """Ensures correct output from _link_local_maxima_in_time.

        In this case there are no current maxima.
        """

        these_indices = echo_top_tracking._link_local_maxima_in_time(
            current_local_max_dict=THIRD_LOCAL_MAX_DICT_EMPTY,
            previous_local_max_dict=SECOND_LOCAL_MAX_DICT_UNLINKED,
            max_link_time_seconds=MAX_LINK_TIME_SECONDS,
            max_velocity_diff_m_s01=MAX_VELOCITY_DIFF_M_S01,
            max_link_distance_m_s01=MAX_LINK_DISTANCE_M_S01)

        expected_indices = numpy.full(len(these_indices), -1, dtype=int)
        self.assertTrue(numpy.array_equal(these_indices, expected_indices))

    def test_create_storm_id_first_in_day(self):
        """Ensures correct output from _create_storm_id.

        In this case, storm to be labeled is the first storm in the SPC date.
        """

        this_storm_id, this_numeric_id, this_spc_date_string = (
            echo_top_tracking._create_storm_id(
                storm_start_time_unix_sec=STORM_TIME_UNIX_SEC,
                prev_numeric_id_used=PREV_NUMERIC_ID_USED,
                prev_spc_date_string=PREV_SPC_DATE_STRING)
        )

        self.assertTrue(this_storm_id == STORM_ID_FIRST_IN_DAY)
        self.assertTrue(this_numeric_id == 0)
        self.assertTrue(this_spc_date_string == STORM_SPC_DATE_STRING)

    def test_create_storm_id_second_in_day(self):
        """Ensures correct output from _create_storm_id.

        In this case, storm to be labeled is the second storm in the SPC date.
        """

        this_storm_id, this_numeric_id, this_spc_date_string = (
            echo_top_tracking._create_storm_id(
                storm_start_time_unix_sec=STORM_TIME_UNIX_SEC,
                prev_numeric_id_used=PREV_NUMERIC_ID_USED,
                prev_spc_date_string=STORM_SPC_DATE_STRING)
        )

        self.assertTrue(this_storm_id == STORM_ID_SECOND_IN_DAY)
        self.assertTrue(this_numeric_id == PREV_NUMERIC_ID_USED + 1)
        self.assertTrue(this_spc_date_string == STORM_SPC_DATE_STRING)

    def test_local_maxima_to_storm_tracks(self):
        """Ensures correct output from _local_maxima_to_storm_tracks."""

        this_storm_object_table = (
            echo_top_tracking._local_maxima_to_storm_tracks(
                LOCAL_MAX_DICT_BY_TIME)
        )

        self.assertTrue(this_storm_object_table.equals(STORM_OBJECT_TABLE))

    def test_remove_short_lived_tracks_short_threshold(self):
        """Ensures correct output from _remove_short_lived_tracks.

        In this case, minimum track duration is short, so all tracks should be
        kept.
        """

        this_storm_object_table = echo_top_tracking._remove_short_lived_tracks(
            storm_object_table=copy.deepcopy(STORM_OBJECT_TABLE),
            min_duration_seconds=SHORT_MIN_LIFETIME_SEC)

        self.assertTrue(this_storm_object_table.equals(STORM_OBJECT_TABLE))

    def test_remove_short_lived_tracks_long_threshold(self):
        """Ensures correct output from _remove_short_lived_tracks.

        In this case, minimum track duration is long, so all tracks should be
        removed.
        """

        this_storm_object_table = echo_top_tracking._remove_short_lived_tracks(
            storm_object_table=copy.deepcopy(STORM_OBJECT_TABLE),
            min_duration_seconds=LONG_MIN_LIFETIME_SEC)

        self.assertTrue(this_storm_object_table.empty)

    def test_get_final_velocities_one_track_1point(self):
        """Ensures correct output from _get_final_velocities_one_track.

        In this case, each velocity is based on the displacement from 1 point
        back in the storm track.
        """

        these_u_velocities_m_s01, these_v_velocities_m_s01 = (
            echo_top_tracking._get_final_velocities_one_track(
                centroid_latitudes_deg=ONE_TRACK_LATITUDES_DEG,
                centroid_longitudes_deg=ONE_TRACK_LONGITUDES_DEG,
                valid_times_unix_sec=ONE_TRACK_TIMES_UNIX_SEC,
                num_points_back=1)
        )

        self.assertTrue(numpy.allclose(
            these_u_velocities_m_s01, U_VELOCITIES_1POINT_M_S01,
            rtol=RELATIVE_DISTANCE_TOLERANCE, equal_nan=True
        ))

        self.assertTrue(numpy.allclose(
            these_v_velocities_m_s01, V_VELOCITIES_1POINT_M_S01,
            rtol=RELATIVE_DISTANCE_TOLERANCE, equal_nan=True
        ))

    def test_get_final_velocities_one_track_2points(self):
        """Ensures correct output from _get_final_velocities_one_track.

        In this case, each velocity is based on the displacement from 2 points
        back in the storm track.
        """

        these_u_velocities_m_s01, these_v_velocities_m_s01 = (
            echo_top_tracking._get_final_velocities_one_track(
                centroid_latitudes_deg=ONE_TRACK_LATITUDES_DEG,
                centroid_longitudes_deg=ONE_TRACK_LONGITUDES_DEG,
                valid_times_unix_sec=ONE_TRACK_TIMES_UNIX_SEC,
                num_points_back=2)
        )

        self.assertTrue(numpy.allclose(
            these_u_velocities_m_s01, U_VELOCITIES_2POINTS_M_S01,
            rtol=RELATIVE_DISTANCE_TOLERANCE, equal_nan=True
        ))

        self.assertTrue(numpy.allclose(
            these_v_velocities_m_s01, V_VELOCITIES_2POINTS_M_S01,
            rtol=RELATIVE_DISTANCE_TOLERANCE, equal_nan=True
        ))

    def test_remove_small_polygons_min0(self):
        """Ensures correct output from _remove_small_polygons.

        In this case polygons with >= 0 grid cells should be kept.
        """

        this_local_max_dict = echo_top_tracking._remove_small_polygons(
            local_max_dict=copy.deepcopy(LOCAL_MAX_DICT_WITH_SMALL),
            min_size_pixels=0)

        self.assertTrue(_compare_maxima_with_sans_small_polygons(
            this_local_max_dict, LOCAL_MAX_DICT_WITH_SMALL
        ))

    def test_remove_small_polygons_min5(self):
        """Ensures correct output from _remove_small_polygons.

        In this case polygons with >= 5 grid cells should be kept.
        """

        this_local_max_dict = echo_top_tracking._remove_small_polygons(
            local_max_dict=copy.deepcopy(LOCAL_MAX_DICT_WITH_SMALL),
            min_size_pixels=MIN_POLYGON_SIZE_PIXELS)

        self.assertTrue(_compare_maxima_with_sans_small_polygons(
            this_local_max_dict, LOCAL_MAX_DICT_WITHOUT_SMALL
        ))

    def test_join_tracks(self):
        """Ensures correct output from _join_tracks."""

        this_storm_object_table = echo_top_tracking._join_tracks(
            early_storm_object_table=EARLY_STORM_OBJECT_TABLE,
            late_storm_object_table=copy.deepcopy(LATE_STORM_OBJECT_TABLE),
            projection_object=PROJECTION_OBJECT,
            max_link_time_seconds=LONG_MAX_JOIN_TIME_SEC,
            max_velocity_diff_m_s01=MAX_VELOCITY_DIFF_M_S01,
            max_link_distance_m_s01=MAX_LINK_DISTANCE_M_S01)

        self.assertTrue(this_storm_object_table.equals(
            JOINED_STORM_OBJECT_TABLE
        ))

    def test_storm_objects_to_tracks_from_scratch(self):
        """Ensures correct output from _storm_objects_to_tracks.

        In this case, the table of storm tracks is created from scratch.
        """

        this_storm_track_table = echo_top_tracking._storm_objects_to_tracks(
            storm_object_table=STORM_OBJECT_TABLE_BEFORE_REANALYSIS)

        self.assertTrue(this_storm_track_table.equals(
            STORM_TRACK_TABLE_BEFORE_REANALYSIS
        ))

    def test_storm_objects_to_tracks_recompute_row(self):
        """Ensures correct output from _storm_objects_to_tracks.

        In this case, only one row of storm_track_table is recomputed.
        """

        this_storm_track_table = echo_top_tracking._storm_objects_to_tracks(
            storm_object_table=STORM_OBJECT_TABLE_BEFORE_REANALYSIS,
            storm_track_table=copy.deepcopy(STORM_TRACK_TABLE_TAMPERED),
            recompute_for_id=STORM_ID_TAMPERED)

        self.assertTrue(this_storm_track_table.equals(
            STORM_TRACK_TABLE_BEFORE_REANALYSIS
        ))

    def test_get_join_error_b_to_a(self):
        """Ensures correct output from _get_join_error.

        In this case, extrapolating track B to start of track A.
        """

        this_join_error_metres = echo_top_tracking._get_join_error(
            storm_track_table=STORM_TRACK_TABLE_BEFORE_REANALYSIS,
            early_track_id='B', late_track_id='A')

        self.assertTrue(numpy.isclose(
            this_join_error_metres, JOIN_ERROR_B_TO_A_METRES,
            atol=DIST_TOLERANCE_METRES
        ))

    def test_get_join_error_c_to_a(self):
        """Ensures correct output from _get_join_error.

        In this case, extrapolating track C to start of track A.
        """

        this_join_error_metres = echo_top_tracking._get_join_error(
            storm_track_table=STORM_TRACK_TABLE_BEFORE_REANALYSIS,
            early_track_id='C', late_track_id='A')

        self.assertTrue(numpy.isclose(
            this_join_error_metres, JOIN_ERROR_C_TO_A_METRES,
            atol=DIST_TOLERANCE_METRES
        ))

    def test_get_join_error_d_to_a(self):
        """Ensures correct output from _get_join_error.

        In this case, extrapolating track D to start of track A.
        """

        this_join_error_metres = echo_top_tracking._get_join_error(
            storm_track_table=STORM_TRACK_TABLE_BEFORE_REANALYSIS,
            early_track_id='D', late_track_id='A')

        self.assertTrue(numpy.isclose(
            this_join_error_metres, JOIN_ERROR_D_TO_A_METRES,
            atol=DIST_TOLERANCE_METRES
        ))

    def test_get_join_error_e_to_a(self):
        """Ensures correct output from _get_join_error.

        In this case, extrapolating track E to start of track A.
        """

        this_join_error_metres = echo_top_tracking._get_join_error(
            storm_track_table=STORM_TRACK_TABLE_BEFORE_REANALYSIS,
            early_track_id='E', late_track_id='A')

        self.assertTrue(numpy.isclose(
            this_join_error_metres, JOIN_ERROR_E_TO_A_METRES,
            atol=DIST_TOLERANCE_METRES
        ))

    def test_find_nearby_tracks_for_a(self):
        """Ensures correct output from _find_nearby_tracks; late track is A."""

        these_nearby_indices = echo_top_tracking._find_nearby_tracks(
            storm_track_table=STORM_TRACK_TABLE_BEFORE_REANALYSIS,
            late_track_id='A', max_time_diff_seconds=SHORT_MAX_JOIN_TIME_SEC,
            max_join_error_m_s01=MAX_JOIN_ERROR_M_S01)

        if these_nearby_indices is None:
            these_nearby_ids = []
        else:
            these_nearby_ids = STORM_TRACK_TABLE_BEFORE_REANALYSIS[
                tracking_utils.STORM_ID_COLUMN
            ].values[these_nearby_indices].tolist()

        self.assertTrue(these_nearby_ids == NEARBY_IDS_FOR_A)

    def test_find_nearby_tracks_for_b(self):
        """Ensures correct output from _find_nearby_tracks; late track is B."""

        these_nearby_indices = echo_top_tracking._find_nearby_tracks(
            storm_track_table=STORM_TRACK_TABLE_BEFORE_REANALYSIS,
            late_track_id='B', max_time_diff_seconds=SHORT_MAX_JOIN_TIME_SEC,
            max_join_error_m_s01=MAX_JOIN_ERROR_M_S01)

        if these_nearby_indices is None:
            these_nearby_ids = []
        else:
            these_nearby_ids = STORM_TRACK_TABLE_BEFORE_REANALYSIS[
                tracking_utils.STORM_ID_COLUMN
            ].values[these_nearby_indices].tolist()

        self.assertTrue(these_nearby_ids == NEARBY_IDS_FOR_B)

    def test_find_nearby_tracks_for_c(self):
        """Ensures correct output from _find_nearby_tracks; late track is C."""

        these_nearby_indices = echo_top_tracking._find_nearby_tracks(
            storm_track_table=STORM_TRACK_TABLE_BEFORE_REANALYSIS,
            late_track_id='C', max_time_diff_seconds=SHORT_MAX_JOIN_TIME_SEC,
            max_join_error_m_s01=MAX_JOIN_ERROR_M_S01)

        if these_nearby_indices is None:
            these_nearby_ids = []
        else:
            these_nearby_ids = STORM_TRACK_TABLE_BEFORE_REANALYSIS[
                tracking_utils.STORM_ID_COLUMN
            ].values[these_nearby_indices].tolist()

        self.assertTrue(these_nearby_ids == NEARBY_IDS_FOR_C)

    def test_find_nearby_tracks_for_d(self):
        """Ensures correct output from _find_nearby_tracks; late track is D."""

        these_nearby_indices = echo_top_tracking._find_nearby_tracks(
            storm_track_table=STORM_TRACK_TABLE_BEFORE_REANALYSIS,
            late_track_id='D', max_time_diff_seconds=SHORT_MAX_JOIN_TIME_SEC,
            max_join_error_m_s01=MAX_JOIN_ERROR_M_S01)

        if these_nearby_indices is None:
            these_nearby_ids = []
        else:
            these_nearby_ids = STORM_TRACK_TABLE_BEFORE_REANALYSIS[
                tracking_utils.STORM_ID_COLUMN
            ].values[these_nearby_indices].tolist()

        self.assertTrue(these_nearby_ids == NEARBY_IDS_FOR_D)

    def test_find_nearby_tracks_for_e(self):
        """Ensures correct output from _find_nearby_tracks; late track is E."""

        these_nearby_indices = echo_top_tracking._find_nearby_tracks(
            storm_track_table=STORM_TRACK_TABLE_BEFORE_REANALYSIS,
            late_track_id='E', max_time_diff_seconds=SHORT_MAX_JOIN_TIME_SEC,
            max_join_error_m_s01=MAX_JOIN_ERROR_M_S01)

        if these_nearby_indices is None:
            these_nearby_ids = []
        else:
            these_nearby_ids = STORM_TRACK_TABLE_BEFORE_REANALYSIS[
                tracking_utils.STORM_ID_COLUMN
            ].values[these_nearby_indices].tolist()

        self.assertTrue(these_nearby_ids == NEARBY_IDS_FOR_E)

    def test_reanalyze_tracks(self):
        """Ensures correct output from _reanalyze_tracks."""

        this_storm_object_table = echo_top_tracking._reanalyze_tracks(
            storm_object_table=copy.deepcopy(
                STORM_OBJECT_TABLE_BEFORE_REANALYSIS),
            max_join_time_sec=SHORT_MAX_JOIN_TIME_SEC,
            max_join_error_m_s01=MAX_JOIN_ERROR_M_S01)

        self.assertTrue(this_storm_object_table.equals(
            STORM_OBJECT_TABLE_AFTER_REANALYSIS
        ))


if __name__ == '__main__':
    unittest.main()
