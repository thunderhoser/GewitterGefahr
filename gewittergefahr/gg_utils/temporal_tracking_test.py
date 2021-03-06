"""Unit tests for temporal_tracking.py."""

import copy
import unittest
from collections import OrderedDict
import numpy
import pandas
from gewittergefahr.gg_utils import temporal_tracking
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils

TOLERANCE = 1e-6
VELOCITY_EFOLD_RADIUS_METRES = 1.

# The following constants are used to test _estimate_velocity_by_neigh.
NEIGH_X_COORDS_METRES = numpy.array([
    0, 1.5, 3, 4.5,
    0, 1.5, 3, 4.5,
    0, 1.5, 3, 4.5
])

NEIGH_Y_COORDS_METRES = numpy.array([
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

X_VELOCITIES_WITHOUT_NAN_M_S01 = numpy.array([
    FIRST_X_VELOCITY_M_S01, 5, 8, 3,
    14, SECOND_X_VELOCITY_M_S01, 1, 5,
    5, 7, THIRD_X_VELOCITY_M_S01, 7
])
Y_VELOCITIES_WITHOUT_NAN_M_S01 = numpy.array([
    FIRST_Y_VELOCITY_M_S01, -4, 0, 0,
    3, SECOND_Y_VELOCITY_M_S01, 2, 6,
    6, -2, THIRD_Y_VELOCITY_M_S01, 3
])

# The following constants are used to test get_intermediate_velocities.
THESE_X_COORDS_METRES = numpy.array([2, -7, 1, 6, 5, -4], dtype=float)
THESE_Y_COORDS_METRES = numpy.array([4, -1, 5, -1, -3, 9], dtype=float)

FIRST_MAX_DICT_NO_VELOCITY = {
    temporal_tracking.VALID_TIME_KEY: 0,
    temporal_tracking.X_COORDS_KEY: THESE_X_COORDS_METRES + 0.,
    temporal_tracking.Y_COORDS_KEY: THESE_Y_COORDS_METRES + 0.,
    temporal_tracking.LONGITUDES_KEY: THESE_X_COORDS_METRES + 0.,
    temporal_tracking.LATITUDES_KEY: THESE_Y_COORDS_METRES + 0.,
}

THESE_X_COORDS_METRES = numpy.array(
    [13, -1, 20, 20, -8, 5, -23, 19], dtype=float)
THESE_Y_COORDS_METRES = numpy.array(
    [-14, 25, -12, 1, -14, 4, -5, 18], dtype=float)

THIS_CURRENT_TO_PREV_MATRIX = numpy.array([
    [0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1]
], dtype=bool)

SECOND_MAX_DICT_NO_VELOCITY = {
    temporal_tracking.VALID_TIME_KEY: 10,
    temporal_tracking.X_COORDS_KEY: THESE_X_COORDS_METRES + 0.,
    temporal_tracking.Y_COORDS_KEY: THESE_Y_COORDS_METRES + 0.,
    temporal_tracking.LONGITUDES_KEY: THESE_X_COORDS_METRES + 0.,
    temporal_tracking.LATITUDES_KEY: THESE_Y_COORDS_METRES + 0.,
    temporal_tracking.CURRENT_TO_PREV_MATRIX_KEY:
        copy.deepcopy(THIS_CURRENT_TO_PREV_MATRIX)
}

FIRST_MAX_DICT_WITH_VELOCITY = copy.deepcopy(FIRST_MAX_DICT_NO_VELOCITY)
FIRST_MAX_DICT_WITH_VELOCITY.update({
    temporal_tracking.X_VELOCITIES_KEY: numpy.full(6, numpy.nan),
    temporal_tracking.Y_VELOCITIES_KEY: numpy.full(6, numpy.nan)
})

SECOND_MAX_DICT_WITH_VELOCITY = copy.deepcopy(SECOND_MAX_DICT_NO_VELOCITY)

THESE_X_VELOCITIES_M_S01 = numpy.array([
    numpy.nan, (-0.3 + 0.6) / 2, 1.9, numpy.nan, -0.9, 0, numpy.nan, 2.3
])
THESE_Y_VELOCITIES_M_S01 = numpy.array([
    numpy.nan, (2.1 + 2.6) / 2, -1.7, numpy.nan, -1.9, 0.7, numpy.nan, 0.9
])

SECOND_MAX_DICT_WITH_VELOCITY.update({
    temporal_tracking.X_VELOCITIES_KEY: THESE_X_VELOCITIES_M_S01,
    temporal_tracking.Y_VELOCITIES_KEY: THESE_Y_VELOCITIES_M_S01
})

# The following constants are used to test _link_local_maxima_by_velocity,
# _link_local_maxima_by_distance, _prune_connections, and
# link_local_maxima_in_time.
MAX_LINK_TIME_SECONDS = 100
MAX_VELOCITY_DIFF_M_S01 = 3.
MAX_LINK_DISTANCE_M_S01 = 2.

FIRST_LOCAL_MAX_DICT_SANS_IDS = copy.deepcopy(FIRST_MAX_DICT_WITH_VELOCITY)
SECOND_LOCAL_MAX_DICT_SANS_IDS = copy.deepcopy(SECOND_MAX_DICT_WITH_VELOCITY)
SECOND_LOCAL_MAX_DICT_SANS_IDS.pop(temporal_tracking.CURRENT_TO_PREV_MATRIX_KEY)

NUM_FIRST_MAXIMA = len(
    FIRST_LOCAL_MAX_DICT_SANS_IDS[temporal_tracking.X_COORDS_KEY]
)
NUM_SECOND_MAXIMA = len(
    SECOND_LOCAL_MAX_DICT_SANS_IDS[temporal_tracking.X_COORDS_KEY]
)

VELOCITY_DIFF_MATRIX_1TO2_M_S01 = numpy.full(
    (NUM_SECOND_MAXIMA, NUM_FIRST_MAXIMA), numpy.inf
)
CURRENT_TO_PREV_MATRIX_VELOCITY_1TO2 = numpy.full(
    (NUM_SECOND_MAXIMA, NUM_FIRST_MAXIMA), False, dtype=bool
)

THIS_X_DISTANCE_MATRIX_METRES = numpy.array([
    [11, 20, 12, 7, 8, 17],
    [3, 6, 2, 7, 6, 3],
    [18, 27, 19, 14, 15, 24],
    [18, 27, 19, 14, 15, 24],
    [10, 1, 9, 14, 13, 4],
    [3, 12, 4, 1, 0, 9],
    [25, 16, 24, 29, 28, 19],
    [17, 26, 18, 13, 14, 23]
], dtype=float)

THIS_Y_DISTANCE_MATRIX_METRES = numpy.array([
    [18, 13, 19, 13, 11, 23],
    [21, 26, 20, 26, 28, 16],
    [16, 11, 17, 11, 9, 21],
    [3, 2, 4, 2, 4, 8],
    [18, 13, 19, 13, 11, 23],
    [0, 5, 1, 5, 7, 5],
    [9, 4, 10, 4, 2, 14],
    [14, 19, 13, 19, 21, 9]
], dtype=float)

DISTANCE_MATRIX_1TO2_M_S01 = numpy.sqrt(
    THIS_X_DISTANCE_MATRIX_METRES ** 2 + THIS_Y_DISTANCE_MATRIX_METRES ** 2
) / 10

CURRENT_TO_PREV_MATRIX_DISTANCE_1TO2 = numpy.array([
    [0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 1, 1, 0],
    [0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0],
    [1, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
], dtype=bool)

CURRENT_TO_PREV_MATRIX_1TO2 = numpy.array([
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [1, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
], dtype=bool)

THESE_X_COORDS_METRES = numpy.array(
    [21, 5, 30, 31, 0, 17, 28], dtype=float
)
THESE_Y_COORDS_METRES = numpy.array(
    [-9, 34, 0, 12, -4, 12, 25], dtype=float
)

THIRD_LOCAL_MAX_DICT_SANS_IDS = {
    temporal_tracking.VALID_TIME_KEY: 15,
    temporal_tracking.X_COORDS_KEY: THESE_X_COORDS_METRES + 0.,
    temporal_tracking.Y_COORDS_KEY: THESE_Y_COORDS_METRES + 0.,
    temporal_tracking.LONGITUDES_KEY: THESE_X_COORDS_METRES + 0.,
    temporal_tracking.LATITUDES_KEY: THESE_Y_COORDS_METRES + 0.
}

# SECOND_EXTRAP_X_COORDS_METRES = numpy.array(
#     [numpy.nan, -0.25, 29.5, numpy.nan, -12.5, 5, numpy.nan, 30.5])
# SECOND_EXTRAP_Y_COORDS_METRES = numpy.array(
#     [numpy.nan, 36.75, -20.5, numpy.nan, -23.5, 7.5, numpy.nan, 22.5])

THIS_X_DISTANCE_MATRIX_METRES = numpy.array([
    [-1, 21.25, 8.5, -1, 33.5, 16, -1, 9.5],
    [-1, 5.25, 24.5, -1, 17.5, 0, -1, 25.5],
    [-1, 30.25, 0.5, -1, 42.5, 25, -1, 0.5],
    [-1, 31.25, 1.5, -1, 43.5, 26, -1, 0.5],
    [-1, 0.25, 29.5, -1, 12.5, 5, -1, 30.5],
    [-1, 17.25, 12.5, -1, 29.5, 12, -1, 13.5],
    [-1, 28.25, 1.5, -1, 40.5, 23, -1, 2.5]
])

THIS_X_DISTANCE_MATRIX_METRES[THIS_X_DISTANCE_MATRIX_METRES < 0] = numpy.inf

THIS_Y_DISTANCE_MATRIX_METRES = numpy.array([
    [-1, 45.75, 11.5, -1, 14.5, 16.5, -1, 31.5],
    [-1, 2.75, 54.5, -1, 57.5, 26.5, -1, 11.5],
    [-1, 36.75, 20.5, -1, 23.5, 7.5, -1, 22.5],
    [-1, 24.75, 32.5, -1, 35.5, 4.5, -1, 10.5],
    [-1, 40.75, 16.5, -1, 19.5, 11.5, -1, 26.5],
    [-1, 24.75, 32.5, -1, 35.5, 4.5, -1, 10.5],
    [-1, 11.75, 45.5, -1, 48.5, 17.5, -1, 2.5]
])

THIS_Y_DISTANCE_MATRIX_METRES[THIS_Y_DISTANCE_MATRIX_METRES < 0] = numpy.inf

VELOCITY_DIFF_MATRIX_2TO3_M_S01 = numpy.sqrt(
    THIS_X_DISTANCE_MATRIX_METRES ** 2 + THIS_Y_DISTANCE_MATRIX_METRES ** 2
) / 5

CURRENT_TO_PREV_MATRIX_VELOCITY_2TO3 = numpy.array([
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1]
], dtype=bool)

THIS_X_DISTANCE_MATRIX_METRES = numpy.array([
    [8, 22, 1, 1, 29, 16, 44, 2],
    [8, 6, 15, 15, 13, 0, 28, 14],
    [17, 31, 10, 10, 38, 25, 53, 11],
    [18, 32, 11, 11, 39, 26, 54, 12],
    [13, 1, 20, 20, 8, 5, 23, 19],
    [4, 18, 3, 3, 25, 12, 40, 2],
    [15, 29, 8, 8, 36, 23, 51, 9]
], dtype=float)

THIS_X_DISTANCE_MATRIX_METRES[THIS_X_DISTANCE_MATRIX_METRES < 0] = numpy.inf

THIS_Y_DISTANCE_MATRIX_METRES = numpy.array([
    [5, 34, 3, 10, 5, 13, 4, 27],
    [48, 9, 46, 33, 48, 30, 39, 16],
    [14, 25, 12, 1, 14, 4, 5, 18],
    [26, 13, 24, 11, 26, 8, 17, 6],
    [10, 29, 8, 5, 10, 8, 1, 22],
    [26, 13, 24, 11, 26, 8, 17, 6],
    [39, 0, 37, 24, 39, 21, 30, 7]
], dtype=float)

THIS_Y_DISTANCE_MATRIX_METRES[THIS_Y_DISTANCE_MATRIX_METRES < 0] = numpy.inf

DISTANCE_MATRIX_2TO3_M_S01 = numpy.sqrt(
    THIS_X_DISTANCE_MATRIX_METRES ** 2 + THIS_Y_DISTANCE_MATRIX_METRES ** 2
) / 5

CURRENT_TO_PREV_MATRIX_DISTANCE_2TO3 = numpy.array([
    [1, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 1]
], dtype=bool)

CURRENT_TO_PREV_MATRIX_2TO3 = numpy.array([
    [1, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1]
], dtype=bool)

# The following constants are used to test _create_partial_storm_id.
PREV_PRIMARY_ID_NUMERIC = 10
NEXT_PRIMARY_ID_NUMERIC = 11
NEXT_PRIMARY_ID_STRING = '11'

PREV_SECONDARY_ID_NUMERIC = 8
NEXT_SECONDARY_ID_NUMERIC = 9
NEXT_SECONDARY_ID_STRING = '9'

# The following constants are used to test _local_maxima_to_tracks_mergers.
FIRST_PRIMARY_ID_STRINGS = ['0', '1', '2', '3', '4', '5']
FIRST_SECONDARY_ID_STRINGS = ['0', '1', '2', '3', '4', '5']

FIRST_LOCAL_MAX_DICT_WITH_IDS = copy.deepcopy(FIRST_LOCAL_MAX_DICT_SANS_IDS)
FIRST_LOCAL_MAX_DICT_WITH_IDS.update({
    temporal_tracking.PRIMARY_IDS_KEY: FIRST_PRIMARY_ID_STRINGS,
    temporal_tracking.SECONDARY_IDS_KEY: FIRST_SECONDARY_ID_STRINGS,
})

PREV_PRIMARY_ID_1TO2_PREMERGE = 5
PREV_SECONDARY_ID_1TO2_PREMERGE = 5

FIRST_NEXT_SECONDARY_IDS_POSTMERGE = [
    ['6'], [], ['6'], [], [], []
]
SECOND_PREV_SECONDARY_IDS_POSTMERGE = [
    [], [], [], [], [], ['0', '2'], [], []
]

SECOND_PRIMARY_ID_STRINGS_POSTMERGE = [
    '', '', '', '', '', '6', '', ''
]
SECOND_SECONDARY_ID_STRINGS_POSTMERGE = [
    '', '', '', '', '', '6', '', ''
]

PREV_PRIMARY_ID_1TO2_POSTMERGE = 6
PREV_SECONDARY_ID_1TO2_POSTMERGE = 6

OLD_TO_NEW_DICT_1TO2 = [
    ('0', '6'),
    ('2', '6')
]
OLD_TO_NEW_DICT_1TO2 = OrderedDict(OLD_TO_NEW_DICT_1TO2)

CURRENT_TO_PREV_MATRIX_1TO2_POSTMERGE = numpy.array([
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
], dtype=bool)

# The following constants are used to test _local_maxima_to_tracks_splits.
FIRST_NEXT_SECONDARY_IDS_POSTSPLIT = [
    ['6'], [], ['6'], ['7', '8'], [], []
]
SECOND_PREV_SECONDARY_IDS_POSTSPLIT = [
    ['3'], [], [], ['3'], [], ['0', '2'], [], []
]

SECOND_PRIMARY_ID_STRINGS_POSTSPLIT = ['3', '', '', '3', '', '6', '', '']
SECOND_SECONDARY_ID_STRINGS_POSTSPLIT = ['7', '', '', '8', '', '6', '', '']

PREV_PRIMARY_ID_1TO2_POSTSPLIT = 6
PREV_SECONDARY_ID_1TO2_POSTSPLIT = 8

CURRENT_TO_PREV_MATRIX_1TO2_POSTSPLIT = numpy.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
], dtype=bool)

# The following constants are used to test _local_maxima_to_tracks_simple.
FIRST_NEXT_SECONDARY_IDS_POST = [
    ['6'], ['1'], ['6'], ['7', '8'], ['4'], ['5']
]
SECOND_PREV_SECONDARY_IDS_POST = [
    ['3'], ['5'], [], ['3'], ['4'], ['0', '2'], ['1'], []
]

SECOND_PRIMARY_ID_STRINGS = ['3', '5', '7', '3', '4', '6', '1', '8']
SECOND_SECONDARY_ID_STRINGS = ['7', '5', '9', '8', '4', '6', '1', '10']

PREV_PRIMARY_ID_1TO2_POST = 8
PREV_SECONDARY_ID_1TO2_POST = 10

SECOND_LOCAL_MAX_DICT_WITH_IDS = copy.deepcopy(SECOND_LOCAL_MAX_DICT_SANS_IDS)
SECOND_LOCAL_MAX_DICT_WITH_IDS.update({
    temporal_tracking.CURRENT_TO_PREV_MATRIX_KEY: CURRENT_TO_PREV_MATRIX_1TO2,
    temporal_tracking.PRIMARY_IDS_KEY: SECOND_PRIMARY_ID_STRINGS,
    temporal_tracking.SECONDARY_IDS_KEY: SECOND_SECONDARY_ID_STRINGS
})

# The following constants are used to test local_maxima_to_storm_tracks.
PRIMARY_ID_STRINGS_1AND2 = FIRST_PRIMARY_ID_STRINGS + SECOND_PRIMARY_ID_STRINGS
THESE_OLD_ID_STRINGS = ['0', '2']
PRIMARY_ID_STRINGS_1AND2 = [
    '6' if s in THESE_OLD_ID_STRINGS else s
    for s in PRIMARY_ID_STRINGS_1AND2
]

SECONDARY_ID_STRINGS_1AND2 = (
    FIRST_SECONDARY_ID_STRINGS + SECOND_SECONDARY_ID_STRINGS
)

# The following constants are used to test _local_maxima_to_tracks_mergers.
PREV_PRIMARY_ID_2TO3_PREMERGE = 8
PREV_SECONDARY_ID_2TO3_PREMERGE = 10

SECOND_NEXT_SECONDARY_IDS_POSTMERGE = [
    ['11'], [], ['11'], [], [], [], [], []
]
THIRD_PREV_SECONDARY_IDS_POSTMERGE = [
    ['7', '9'], [], [], [], [], [], []
]

THIRD_PRIMARY_ID_STRINGS_POSTMERGE = ['9', '', '', '', '', '', '']
THIRD_SECONDARY_ID_STRINGS_POSTMERGE = ['11', '', '', '', '', '', '']

PREV_PRIMARY_ID_2TO3_POSTMERGE = 9
PREV_SECONDARY_ID_2TO3_POSTMERGE = 11

OLD_TO_NEW_DICT_2TO3 = [
    ('3', '9'),
    ('7', '9')
]
OLD_TO_NEW_DICT_2TO3 = OrderedDict(OLD_TO_NEW_DICT_2TO3)

CURRENT_TO_PREV_MATRIX_2TO3_POSTMERGE = numpy.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1]
], dtype=bool)

# The following constants are used to test _local_maxima_to_tracks_splits.
SECOND_NEXT_SECONDARY_IDS_POSTSPLIT = [
    ['11'], [], ['11'], [], [], ['12', '13'], [], ['14', '15']
]
THIRD_PREV_SECONDARY_IDS_POSTSPLIT = [
    ['7', '9'], [], [], ['10'], ['6'], ['6'], ['10']
]

THIRD_PRIMARY_ID_STRINGS_POSTSPLIT = ['9', '', '', '8', '6', '6', '8']
THIRD_SECONDARY_ID_STRINGS_POSTSPLIT = ['11', '', '', '14', '12', '13', '15']

PREV_PRIMARY_ID_2TO3_POSTSPLIT = 9
PREV_SECONDARY_ID_2TO3_POSTSPLIT = 15

CURRENT_TO_PREV_MATRIX_2TO3_POSTSPLIT = numpy.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
], dtype=bool)

# The following constants are used to test _local_maxima_to_tracks_simple.
SECOND_NEXT_SECONDARY_IDS_POST = [
    ['11'], ['5'], ['11'], [], [], ['12', '13'], [], ['14', '15']
]
THIRD_PREV_SECONDARY_IDS_POST = [
    ['7', '9'], ['5'], [], ['10'], ['6'], ['6'], ['10']
]

THIRD_PRIMARY_ID_STRINGS = ['9', '5', '10', '8', '6', '6', '8']
THIRD_SECONDARY_ID_STRINGS = ['11', '5', '16', '14', '12', '13', '15']

PREV_PRIMARY_ID_2TO3_POST = 10
PREV_SECONDARY_ID_2TO3_POST = 16

THIRD_LOCAL_MAX_DICT_WITH_IDS = copy.deepcopy(THIRD_LOCAL_MAX_DICT_SANS_IDS)
THIRD_LOCAL_MAX_DICT_WITH_IDS.update({
    temporal_tracking.CURRENT_TO_PREV_MATRIX_KEY: CURRENT_TO_PREV_MATRIX_2TO3,
    temporal_tracking.PRIMARY_IDS_KEY: THIRD_PRIMARY_ID_STRINGS,
    temporal_tracking.SECONDARY_IDS_KEY: THIRD_SECONDARY_ID_STRINGS
})

# The following constants are used to test partial_to_full_ids and
# full_to_partial_ids.
TOY_PRIMARY_ID_STRINGS = ['Auston', 'Mitch', 'John', 'William']
TOY_SECONDARY_ID_STRINGS = ['Matthews', 'Marner', 'Tavares', 'Nylander']
TOY_FULL_ID_STRINGS = [
    'Auston_Matthews', 'Mitch_Marner', 'John_Tavares', 'William_Nylander'
]

# The following constants are used to test local_maxima_to_storm_tracks.
PRIMARY_ID_STRINGS = PRIMARY_ID_STRINGS_1AND2 + THIRD_PRIMARY_ID_STRINGS
THESE_OLD_ID_STRINGS = ['3', '7']
PRIMARY_ID_STRINGS = [
    '9' if s in THESE_OLD_ID_STRINGS else s
    for s in PRIMARY_ID_STRINGS
]

SECONDARY_ID_STRINGS = (
    FIRST_SECONDARY_ID_STRINGS + SECOND_SECONDARY_ID_STRINGS +
    THIRD_SECONDARY_ID_STRINGS
)

# The following constants are used to test remove_short_lived_storms.
THESE_SHORT_ID_STRINGS = ['10']
THESE_GOOD_FLAGS = numpy.array(
    [s not in THESE_SHORT_ID_STRINGS for s in PRIMARY_ID_STRINGS],
    dtype=bool
)

THESE_GOOD_INDICES = numpy.where(THESE_GOOD_FLAGS)[0]
PRIMARY_ID_STRINGS_GE5SEC = [PRIMARY_ID_STRINGS[k] for k in THESE_GOOD_INDICES]
SECONDARY_ID_STRINGS_GE5SEC = [
    SECONDARY_ID_STRINGS[k] for k in THESE_GOOD_INDICES
]

THESE_SHORT_ID_STRINGS = ['8', '10']
THESE_GOOD_FLAGS = numpy.array(
    [s not in THESE_SHORT_ID_STRINGS for s in PRIMARY_ID_STRINGS],
    dtype=bool
)

THESE_GOOD_INDICES = numpy.where(THESE_GOOD_FLAGS)[0]
PRIMARY_ID_STRINGS_GE10SEC = [PRIMARY_ID_STRINGS[k] for k in THESE_GOOD_INDICES]
SECONDARY_ID_STRINGS_GE10SEC = [
    SECONDARY_ID_STRINGS[k] for k in THESE_GOOD_INDICES
]

# The following constants are used to test check_tracking_periods.
GOOD_START_TIMES_UNIX_SEC = numpy.array([-10, 0, 20, 50, 100], dtype=int)
GOOD_END_TIMES_UNIX_SEC = numpy.array([-5, 10, 35, 80, 200], dtype=int)

FIRST_BAD_START_TIMES_UNIX_SEC = numpy.array([-10, 0, 20, 50, 100], dtype=int)
FIRST_BAD_END_TIMES_UNIX_SEC = numpy.array([-5, 20, 35, 80, 200], dtype=int)

SECOND_BAD_START_TIMES_UNIX_SEC = numpy.array([-10, 0, 5, 50, 100], dtype=int)
SECOND_BAD_END_TIMES_UNIX_SEC = numpy.array([-5, 10, 10, 80, 200], dtype=int)

# The following constants are used to test get_storm_ages.
FIRST_MAX_LINK_TIME_SEC = 10
SECOND_MAX_LINK_TIME_SEC = 5

FIRST_MAIN_START_TIMES_UNIX_SEC = numpy.array([0], dtype=int)
FIRST_MAIN_END_TIMES_UNIX_SEC = numpy.array([15], dtype=int)
SECOND_MAIN_START_TIMES_UNIX_SEC = numpy.array([0, 10], dtype=int)
SECOND_MAIN_END_TIMES_UNIX_SEC = numpy.array([0, 15], dtype=int)

FIRST_STORM_AGES_SEC = numpy.array([
    -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, 0,
    -1, -1, 0, 5, -1, -1, 5
], dtype=int)

FIRST_CELL_START_TIMES_UNIX_SEC = numpy.array([
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 10,
    0, 0, 15, 10, 0, 0, 10
], dtype=int)

FIRST_CELL_END_TIMES_UNIX_SEC = numpy.array([
    15, 10, 15, 15, 10, 15,
    15, 15, 15, 15, 10, 15, 10, 15,
    15, 15, 15, 15, 15, 15, 15
], dtype=int)

FIRST_TRACKING_START_TIMES_UNIX_SEC = numpy.full(
    len(FIRST_STORM_AGES_SEC), 0, dtype=int
)
FIRST_TRACKING_END_TIMES_UNIX_SEC = numpy.full(
    len(FIRST_STORM_AGES_SEC), 15, dtype=int
)

SECOND_STORM_AGES_SEC = numpy.array([
    -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, 0, -1, -1, -1, -1
], dtype=int)

SECOND_CELL_START_TIMES_UNIX_SEC = FIRST_CELL_START_TIMES_UNIX_SEC + 0
SECOND_CELL_END_TIMES_UNIX_SEC = FIRST_CELL_END_TIMES_UNIX_SEC + 0

# TODO(thunderhoser): The next two arrays are hacks, because a storm actually
# shouldn't bridge the gap between two tracking periods.
SECOND_TRACKING_START_TIMES_UNIX_SEC = numpy.array([
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 10,
    0, 0, 10, 10, 0, 0, 10
], dtype=int)

SECOND_TRACKING_END_TIMES_UNIX_SEC = numpy.array([
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 15,
    0, 0, 15, 15, 0, 0, 15
], dtype=int)

# The following constants are used to test get_storm_velocities.
THESE_FIRST_TIME_DIFFS_SEC = numpy.array([
    -1, -1, -1, -1, -1, -1,
    10, 10, -1, 10, 10, 10, 10, -1,
    15, 15, -1, 5, 15, 15, 5
], dtype=float)

THESE_SECOND_TIME_DIFFS_SEC = numpy.array([
    -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, 10, -1, -1,
    5, -1, -1, -1, 15, 15, -1
], dtype=float)

THESE_FIRST_TIME_DIFFS_SEC[THESE_FIRST_TIME_DIFFS_SEC < 0] = numpy.nan
THESE_SECOND_TIME_DIFFS_SEC[THESE_SECOND_TIME_DIFFS_SEC < 0] = numpy.nan

THESE_FIRST_DISPLACEMENTS_METRES = numpy.array([
    0, 0, 0, 0, 0, 0,
    7, 3, 0, 14, -13, 3, -16, 0,
    15, 9, 0, 12, -2, 15, 9
], dtype=float)

THESE_SECOND_DISPLACEMENTS_METRES = numpy.array([
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 4, 0, 0,
    1, 0, 0, 0, -1, 16, 0
], dtype=float)

THESE_FIRST_VELOCITIES_M_S01 = (
    THESE_FIRST_DISPLACEMENTS_METRES / THESE_FIRST_TIME_DIFFS_SEC
)
THESE_SECOND_VELOCITIES_M_S01 = (
    THESE_SECOND_DISPLACEMENTS_METRES / THESE_SECOND_TIME_DIFFS_SEC
)

EAST_VELOCITIES_NO_NEIGH_15SEC_M_S01 = numpy.nanmean(
    numpy.array([THESE_FIRST_VELOCITIES_M_S01, THESE_SECOND_VELOCITIES_M_S01]),
    axis=0
)

THESE_FIRST_DISPLACEMENTS_METRES = numpy.array([
    0, 0, 0, 0, 0, 0,
    -13, 16, 0, 2, -11, 0, -4, 0,
    -8, 25, 0, -6, -8, 8, 7
], dtype=float)

THESE_SECOND_DISPLACEMENTS_METRES = numpy.array([
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, -1, 0, 0,
    3, 0, 0, 0, -9, 7, 0
], dtype=float)

THESE_FIRST_VELOCITIES_M_S01 = (
    THESE_FIRST_DISPLACEMENTS_METRES / THESE_FIRST_TIME_DIFFS_SEC
)
THESE_SECOND_VELOCITIES_M_S01 = (
    THESE_SECOND_DISPLACEMENTS_METRES / THESE_SECOND_TIME_DIFFS_SEC
)

NORTH_VELOCITIES_NO_NEIGH_15SEC_M_S01 = numpy.nanmean(
    numpy.array([THESE_FIRST_VELOCITIES_M_S01, THESE_SECOND_VELOCITIES_M_S01]),
    axis=0
)

THESE_FIRST_TIME_DIFFS_SEC = numpy.array([
    -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1,
    5, 5, -1, 5, 5, 5, 5
], dtype=float)

THESE_SECOND_TIME_DIFFS_SEC = numpy.array([
    -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1,
    5, -1, -1, -1, -1, -1, -1
], dtype=float)

THESE_FIRST_TIME_DIFFS_SEC[THESE_FIRST_TIME_DIFFS_SEC < 0] = numpy.nan
THESE_SECOND_TIME_DIFFS_SEC[THESE_SECOND_TIME_DIFFS_SEC < 0] = numpy.nan

THESE_FIRST_DISPLACEMENTS_METRES = numpy.array([
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    8, 6, 0, 12, -5, 12, 9
], dtype=float)

THESE_SECOND_DISPLACEMENTS_METRES = numpy.array([
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0
], dtype=float)

THESE_FIRST_VELOCITIES_M_S01 = (
    THESE_FIRST_DISPLACEMENTS_METRES / THESE_FIRST_TIME_DIFFS_SEC
)
THESE_SECOND_VELOCITIES_M_S01 = (
    THESE_SECOND_DISPLACEMENTS_METRES / THESE_SECOND_TIME_DIFFS_SEC
)

EAST_VELOCITIES_NO_NEIGH_5SEC_M_S01 = numpy.nanmean(
    numpy.array([THESE_FIRST_VELOCITIES_M_S01, THESE_SECOND_VELOCITIES_M_S01]),
    axis=0
)

THESE_FIRST_DISPLACEMENTS_METRES = numpy.array([
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 9, 0, -6, -8, 8, 7
], dtype=float)

THESE_SECOND_DISPLACEMENTS_METRES = numpy.array([
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    3, 0, 0, 0, 0, 0, 0
], dtype=float)

THESE_FIRST_VELOCITIES_M_S01 = (
    THESE_FIRST_DISPLACEMENTS_METRES / THESE_FIRST_TIME_DIFFS_SEC
)
THESE_SECOND_VELOCITIES_M_S01 = (
    THESE_SECOND_DISPLACEMENTS_METRES / THESE_SECOND_TIME_DIFFS_SEC
)

NORTH_VELOCITIES_NO_NEIGH_5SEC_M_S01 = numpy.nanmean(
    numpy.array([THESE_FIRST_VELOCITIES_M_S01, THESE_SECOND_VELOCITIES_M_S01]),
    axis=0
)

# The following constants are used to test _get_storm_velocities_missing.
THESE_X_COORDS_METRES = numpy.array([
    -1, 0, 1, 0, 0,
    -2, 0, 2, 0, 0,
    -3, 0, 3, 0, 0,
    1000,
    6
], dtype=float)

THESE_Y_COORDS_METRES = numpy.array([
    0, 1, 0, -1, 0,
    0, 2, 0, -2, 0,
    0, 3, 0, -3, 0,
    1000,
    0
], dtype=float)

THESE_TIMES_UNIX_SEC = numpy.array([
    0, 0, 0, 0, 0,
    300, 300, 300, 300, 300,
    600, 600, 600, 600, 600,
    900,
    1200
], dtype=int)

THESE_EAST_VELOCITIES_M_S01 = numpy.array([
    -5, 1, 5, 1, numpy.nan,
    -10, 2, 10, 2, numpy.nan,
    -15, 3, 15, 3, numpy.nan,
    numpy.nan,
    numpy.nan
])

THESE_NORTH_VELOCITIES_M_S01 = numpy.array([
    -1, 2, -1, -2, numpy.nan,
    -2, 4, -2, -4, numpy.nan,
    -3, 6, -3, -6, numpy.nan,
    numpy.nan,
    numpy.nan
])

STORM_OBJECT_TABLE_MISSING_VEL = pandas.DataFrame.from_dict({
    temporal_tracking.CENTROID_X_COLUMN: THESE_X_COORDS_METRES,
    temporal_tracking.CENTROID_Y_COLUMN: THESE_Y_COORDS_METRES,
    tracking_utils.VALID_TIME_COLUMN: THESE_TIMES_UNIX_SEC,
    tracking_utils.EAST_VELOCITY_COLUMN: THESE_EAST_VELOCITIES_M_S01,
    tracking_utils.NORTH_VELOCITY_COLUMN: THESE_NORTH_VELOCITIES_M_S01
})

THESE_EAST_VELOCITIES_M_S01 = numpy.array([
    -5, 1, 5, 1, 0.5,
    -10, 2, 10, 2, 1,
    -15, 3, 15, 3, 1.5,
    1.875,
    15
])

THESE_NORTH_VELOCITIES_M_S01 = numpy.array([
    -1, 2, -1, -2, -0.5,
    -2, 4, -2, -4, -1,
    -3, 6, -3, -6, -1.5,
    -1.125,
    -3
])

STORM_OBJECT_TABLE_FILLED_VEL = pandas.DataFrame.from_dict({
    temporal_tracking.CENTROID_X_COLUMN: THESE_X_COORDS_METRES,
    temporal_tracking.CENTROID_Y_COLUMN: THESE_Y_COORDS_METRES,
    tracking_utils.VALID_TIME_COLUMN: THESE_TIMES_UNIX_SEC,
    tracking_utils.EAST_VELOCITY_COLUMN: THESE_EAST_VELOCITIES_M_S01,
    tracking_utils.NORTH_VELOCITY_COLUMN: THESE_NORTH_VELOCITIES_M_S01
})

# The following constants are used to test finish_segmotion_or_probsevere_ids.
THESE_PRIMARY_ID_STRINGS = [
    'a', 'b', 'c',
    'a', 'b', 'c', 'd',
    'a', 'c', 'd', 'e',
    'a', 'c', 'e',
    'a', 'c', 'e',
    'c', 'e', 'f',
    'c', 'e'
]

STORM_OBJECT_TABLE_UNFINISHED_IDS = pandas.DataFrame.from_dict({
    tracking_utils.PRIMARY_ID_COLUMN: THESE_PRIMARY_ID_STRINGS
})

THESE_SECONDARY_ID_STRINGS = [
    '0', '1', '2',
    '0', '1', '2', '3',
    '0', '2', '3', '4',
    '0', '2', '4',
    '0', '2', '4',
    '2', '4', '5',
    '2', '4'
]

THESE_FULL_ID_STRINGS = [
    'a_0', 'b_1', 'c_2',
    'a_0', 'b_1', 'c_2', 'd_3',
    'a_0', 'c_2', 'd_3', 'e_4',
    'a_0', 'c_2', 'e_4',
    'a_0', 'c_2', 'e_4',
    'c_2', 'e_4', 'f_5',
    'c_2', 'e_4'
]

THESE_FIRST_PREV_SEC_ID_STRINGS = [
    '', '', '',
    '0', '1', '2', '',
    '0', '2', '3', '',
    '0', '2', '4',
    '0', '2', '4',
    '2', '4', '',
    '2', '4'
]

THESE_FIRST_NEXT_SEC_ID_STRINGS = [
    '0', '1', '2',
    '0', '', '2', '3',
    '0', '2', '', '4',
    '0', '2', '4',
    '', '2', '4',
    '2', '4', '',
    '', ''
]

THESE_SECOND_PREV_SEC_ID_STRINGS = [''] * len(THESE_FULL_ID_STRINGS)
THESE_SECOND_NEXT_SEC_ID_STRINGS = [''] * len(THESE_FULL_ID_STRINGS)

STORM_OBJECT_TABLE_FINISHED_IDS = copy.deepcopy(
    STORM_OBJECT_TABLE_UNFINISHED_IDS)

STORM_OBJECT_TABLE_FINISHED_IDS = STORM_OBJECT_TABLE_FINISHED_IDS.assign(**{
    tracking_utils.SECONDARY_ID_COLUMN: THESE_SECONDARY_ID_STRINGS,
    tracking_utils.FULL_ID_COLUMN: THESE_FULL_ID_STRINGS,
    tracking_utils.FIRST_PREV_SECONDARY_ID_COLUMN:
        THESE_FIRST_PREV_SEC_ID_STRINGS,
    tracking_utils.SECOND_PREV_SECONDARY_ID_COLUMN:
        THESE_SECOND_PREV_SEC_ID_STRINGS,
    tracking_utils.FIRST_NEXT_SECONDARY_ID_COLUMN:
        THESE_FIRST_NEXT_SEC_ID_STRINGS,
    tracking_utils.SECOND_NEXT_SECONDARY_ID_COLUMN:
        THESE_SECOND_NEXT_SEC_ID_STRINGS
})


def _add_ids_to_dict(local_max_dict):
    """Adds keys to dictionary with local maxima.

    P = number of local maxima

    :param local_max_dict: Dictionary with at least the following keys.
    local_max_dict['x_coords_metres']: length-P numpy array of x-coordinates.

    :return: local_max_dict: Same as input but with extra keys listed below.
    local_max_dict['primary_storm_ids']: length-P list of primary storm IDs
        (strings).
    local_max_dict['secondary_storm_ids']: length-P list of secondary storm IDs
        (strings).
    """

    num_storm_objects = len(local_max_dict[temporal_tracking.X_COORDS_KEY])

    local_max_dict.update({
        temporal_tracking.PRIMARY_IDS_KEY: [''] * num_storm_objects,
        temporal_tracking.SECONDARY_IDS_KEY: [''] * num_storm_objects
    })

    return local_max_dict


def _add_id_lists_to_dict(local_max_dict):
    """Adds keys to dictionary with local maxima.

    P = number of local maxima

    :param local_max_dict: Dictionary with at least the following keys.
    local_max_dict['x_coords_metres']: length-P numpy array of x-coordinates.

    :return: local_max_dict: Same as input but with extra keys listed below.
        All inner lists will be empty.
    local_max_dict['prev_secondary_ids_listlist']: length-P list, where
        prev_secondary_ids_listlist[i] is a 1-D list with secondary IDs of
        previous maxima to which the [i]th current max is linked.
    local_max_dict['next_secondary_ids_listlist']: length-P list, where
        next_secondary_ids_listlist[i] is a 1-D list with secondary IDs of next
        maxima to which the [i]th current max is linked.
    """

    num_storm_objects = len(local_max_dict[temporal_tracking.X_COORDS_KEY])

    local_max_dict.update({
        temporal_tracking.PREV_SECONDARY_IDS_KEY:
            [['' for _ in range(0)] for _ in range(num_storm_objects)],
        temporal_tracking.NEXT_SECONDARY_IDS_KEY:
            [['' for _ in range(0)] for _ in range(num_storm_objects)]
    })

    return local_max_dict


class TemporalTrackingTests(unittest.TestCase):
    """Each method is a unit test for temporal_tracking.py."""

    def test_estimate_velocity_by_neigh(self):
        """Ensures correct output from _estimate_velocity_by_neigh."""

        these_x_velocities_m_s01, these_y_velocities_m_s01 = (
            temporal_tracking._estimate_velocity_by_neigh(
                x_coords_metres=NEIGH_X_COORDS_METRES,
                y_coords_metres=NEIGH_Y_COORDS_METRES,
                x_velocities_m_s01=X_VELOCITIES_WITH_NAN_M_S01 + 0.,
                y_velocities_m_s01=Y_VELOCITIES_WITH_NAN_M_S01 + 0.,
                e_folding_radius_metres=VELOCITY_EFOLD_RADIUS_METRES)
        )

        self.assertTrue(numpy.allclose(
            these_x_velocities_m_s01, X_VELOCITIES_WITHOUT_NAN_M_S01,
            atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            these_y_velocities_m_s01, Y_VELOCITIES_WITHOUT_NAN_M_S01,
            atol=TOLERANCE
        ))

    def test_get_intermediate_velocities_time1(self):
        """Ensures correct output from get_intermediate_velocities.

        In this case, "current time" = first time.
        """

        this_local_max_dict = temporal_tracking.get_intermediate_velocities(
            current_local_max_dict=copy.deepcopy(FIRST_MAX_DICT_NO_VELOCITY),
            previous_local_max_dict=None, e_folding_radius_metres=0.001)

        self.assertTrue(numpy.allclose(
            this_local_max_dict[temporal_tracking.X_VELOCITIES_KEY],
            FIRST_MAX_DICT_WITH_VELOCITY[temporal_tracking.X_VELOCITIES_KEY],
            atol=TOLERANCE, equal_nan=True
        ))

        self.assertTrue(numpy.allclose(
            this_local_max_dict[temporal_tracking.Y_VELOCITIES_KEY],
            FIRST_MAX_DICT_WITH_VELOCITY[temporal_tracking.Y_VELOCITIES_KEY],
            atol=TOLERANCE, equal_nan=True
        ))

    def test_get_intermediate_velocities_time2(self):
        """Ensures correct output from get_intermediate_velocities.

        In this case, "current time" = second time.
        """

        this_local_max_dict = temporal_tracking.get_intermediate_velocities(
            current_local_max_dict=copy.deepcopy(SECOND_MAX_DICT_NO_VELOCITY),
            previous_local_max_dict=copy.deepcopy(FIRST_MAX_DICT_NO_VELOCITY),
            e_folding_radius_metres=0.001)

        self.assertTrue(numpy.allclose(
            this_local_max_dict[temporal_tracking.X_VELOCITIES_KEY],
            SECOND_MAX_DICT_WITH_VELOCITY[temporal_tracking.X_VELOCITIES_KEY],
            atol=TOLERANCE, equal_nan=True
        ))

        self.assertTrue(numpy.allclose(
            this_local_max_dict[temporal_tracking.Y_VELOCITIES_KEY],
            SECOND_MAX_DICT_WITH_VELOCITY[temporal_tracking.Y_VELOCITIES_KEY],
            atol=TOLERANCE, equal_nan=True
        ))

    def test_link_local_maxima_by_velocity_1to2(self):
        """Ensures correct output from _link_local_maxima_by_velocity.

        In this case, linking maxima from the first and second times.
        """

        this_diff_matrix_m_s01, this_current_to_prev_matrix = (
            temporal_tracking._link_local_maxima_by_velocity(
                current_local_max_dict=SECOND_LOCAL_MAX_DICT_SANS_IDS,
                previous_local_max_dict=FIRST_LOCAL_MAX_DICT_SANS_IDS,
                max_velocity_diff_m_s01=MAX_VELOCITY_DIFF_M_S01)
        )

        self.assertTrue(numpy.allclose(
            this_diff_matrix_m_s01, VELOCITY_DIFF_MATRIX_1TO2_M_S01,
            atol=TOLERANCE
        ))

        self.assertTrue(numpy.array_equal(
            this_current_to_prev_matrix, CURRENT_TO_PREV_MATRIX_VELOCITY_1TO2
        ))

    def test_link_local_maxima_by_velocity_2to3(self):
        """Ensures correct output from _link_local_maxima_by_velocity.

        In this case, linking maxima from the second and third times.
        """

        this_diff_matrix_m_s01, this_current_to_prev_matrix = (
            temporal_tracking._link_local_maxima_by_velocity(
                current_local_max_dict=THIRD_LOCAL_MAX_DICT_SANS_IDS,
                previous_local_max_dict=SECOND_LOCAL_MAX_DICT_SANS_IDS,
                max_velocity_diff_m_s01=MAX_VELOCITY_DIFF_M_S01)
        )

        self.assertTrue(numpy.allclose(
            this_diff_matrix_m_s01, VELOCITY_DIFF_MATRIX_2TO3_M_S01,
            atol=TOLERANCE
        ))

        self.assertTrue(numpy.array_equal(
            this_current_to_prev_matrix, CURRENT_TO_PREV_MATRIX_VELOCITY_2TO3
        ))

    def test_link_local_maxima_by_distance_1to2(self):
        """Ensures correct output from _link_local_maxima_by_distance.

        In this case, linking maxima from the first and second times.
        """

        this_diff_matrix_m_s01, this_current_to_prev_matrix = (
            temporal_tracking._link_local_maxima_by_distance(
                current_local_max_dict=SECOND_LOCAL_MAX_DICT_SANS_IDS,
                previous_local_max_dict=FIRST_LOCAL_MAX_DICT_SANS_IDS,
                max_link_distance_m_s01=MAX_LINK_DISTANCE_M_S01,
                current_to_previous_matrix=copy.deepcopy(
                    CURRENT_TO_PREV_MATRIX_VELOCITY_1TO2)
            )
        )

        self.assertTrue(numpy.allclose(
            this_diff_matrix_m_s01, DISTANCE_MATRIX_1TO2_M_S01, atol=TOLERANCE
        ))

        self.assertTrue(numpy.array_equal(
            this_current_to_prev_matrix, CURRENT_TO_PREV_MATRIX_DISTANCE_1TO2
        ))

    def test_link_local_maxima_by_distance_2to3(self):
        """Ensures correct output from _link_local_maxima_by_distance.

        In this case, linking maxima from the second and third times.
        """

        this_diff_matrix_m_s01, this_current_to_prev_matrix = (
            temporal_tracking._link_local_maxima_by_distance(
                current_local_max_dict=THIRD_LOCAL_MAX_DICT_SANS_IDS,
                previous_local_max_dict=SECOND_LOCAL_MAX_DICT_SANS_IDS,
                max_link_distance_m_s01=MAX_LINK_DISTANCE_M_S01,
                current_to_previous_matrix=copy.deepcopy(
                    CURRENT_TO_PREV_MATRIX_VELOCITY_2TO3)
            )
        )

        self.assertTrue(numpy.allclose(
            this_diff_matrix_m_s01, DISTANCE_MATRIX_2TO3_M_S01, atol=TOLERANCE
        ))

        self.assertTrue(numpy.array_equal(
            this_current_to_prev_matrix, CURRENT_TO_PREV_MATRIX_DISTANCE_2TO3
        ))

    def test_prune_connections_1to2(self):
        """Ensures correct output from _prune_connections.

        In this case, linking maxima from the first and second times.
        """

        this_current_to_prev_matrix = temporal_tracking._prune_connections(
            velocity_diff_matrix_m_s01=VELOCITY_DIFF_MATRIX_1TO2_M_S01,
            distance_matrix_m_s01=DISTANCE_MATRIX_1TO2_M_S01,
            current_to_previous_matrix=copy.deepcopy(
                CURRENT_TO_PREV_MATRIX_DISTANCE_1TO2)
        )

        self.assertTrue(numpy.array_equal(
            this_current_to_prev_matrix, CURRENT_TO_PREV_MATRIX_1TO2
        ))

    def test_prune_connections_2to3(self):
        """Ensures correct output from _prune_connections.

        In this case, linking maxima from the second and third times.
        """

        this_current_to_prev_matrix = temporal_tracking._prune_connections(
            velocity_diff_matrix_m_s01=VELOCITY_DIFF_MATRIX_2TO3_M_S01,
            distance_matrix_m_s01=DISTANCE_MATRIX_2TO3_M_S01,
            current_to_previous_matrix=copy.deepcopy(
                CURRENT_TO_PREV_MATRIX_DISTANCE_2TO3)
        )

        self.assertTrue(numpy.array_equal(
            this_current_to_prev_matrix, CURRENT_TO_PREV_MATRIX_2TO3
        ))

    def test_link_local_maxima_in_time_1to2(self):
        """Ensures correct output from link_local_maxima_in_time.

        In this case, linking maxima from the first and second times.
        """

        this_current_to_prev_matrix = (
            temporal_tracking.link_local_maxima_in_time(
                current_local_max_dict=SECOND_LOCAL_MAX_DICT_SANS_IDS,
                previous_local_max_dict=FIRST_LOCAL_MAX_DICT_SANS_IDS,
                max_link_time_seconds=MAX_LINK_TIME_SECONDS,
                max_velocity_diff_m_s01=MAX_VELOCITY_DIFF_M_S01,
                max_link_distance_m_s01=MAX_LINK_DISTANCE_M_S01)
        )

        self.assertTrue(numpy.array_equal(
            this_current_to_prev_matrix, CURRENT_TO_PREV_MATRIX_1TO2
        ))

    def test_link_local_maxima_in_time_2to3(self):
        """Ensures correct output from link_local_maxima_in_time.

        In this case, linking maxima from the second and third times.
        """

        this_current_to_prev_matrix = (
            temporal_tracking.link_local_maxima_in_time(
                current_local_max_dict=THIRD_LOCAL_MAX_DICT_SANS_IDS,
                previous_local_max_dict=SECOND_LOCAL_MAX_DICT_SANS_IDS,
                max_link_time_seconds=MAX_LINK_TIME_SECONDS,
                max_velocity_diff_m_s01=MAX_VELOCITY_DIFF_M_S01,
                max_link_distance_m_s01=MAX_LINK_DISTANCE_M_S01)
        )

        self.assertTrue(numpy.array_equal(
            this_current_to_prev_matrix, CURRENT_TO_PREV_MATRIX_2TO3
        ))

    def test_create_partial_storm_id_primary(self):
        """Ensures correct output from _create_partial_storm_id.

        In this case, creating primary ID.
        """

        this_primary_id_string, this_primary_id_numeric = (
            temporal_tracking._create_partial_storm_id(
                PREV_PRIMARY_ID_NUMERIC + 0)
        )

        self.assertTrue(this_primary_id_string == NEXT_PRIMARY_ID_STRING)
        self.assertTrue(this_primary_id_numeric == NEXT_PRIMARY_ID_NUMERIC)

    def test_create_partial_storm_id_secondary(self):
        """Ensures correct output from _create_partial_storm_id.

        In this case, creating secondary ID.
        """

        this_secondary_id_string, this_secondary_id_numeric = (
            temporal_tracking._create_partial_storm_id(
                PREV_SECONDARY_ID_NUMERIC + 0)
        )

        self.assertTrue(this_secondary_id_string == NEXT_SECONDARY_ID_STRING)
        self.assertTrue(this_secondary_id_numeric == NEXT_SECONDARY_ID_NUMERIC)

    def test_local_maxima_to_tracks_mergers_1to2(self):
        """Ensures correct output from _local_maxima_to_tracks_mergers.

        In this case, linking maxima from the first and second times.
        """

        this_current_local_max_dict = _add_ids_to_dict(
            copy.deepcopy(SECOND_LOCAL_MAX_DICT_SANS_IDS)
        )
        this_current_local_max_dict = _add_id_lists_to_dict(
            this_current_local_max_dict
        )
        this_prev_local_max_dict = _add_id_lists_to_dict(
            copy.deepcopy(FIRST_LOCAL_MAX_DICT_WITH_IDS)
        )

        this_dict = temporal_tracking._local_maxima_to_tracks_mergers(
            current_local_max_dict=this_current_local_max_dict,
            previous_local_max_dict=this_prev_local_max_dict,
            current_to_previous_matrix=copy.deepcopy(
                CURRENT_TO_PREV_MATRIX_1TO2),
            prev_primary_id_numeric=PREV_PRIMARY_ID_1TO2_PREMERGE,
            prev_secondary_id_numeric=PREV_SECONDARY_ID_1TO2_PREMERGE)

        this_current_to_prev_matrix = this_dict[
            temporal_tracking.CURRENT_TO_PREV_MATRIX_KEY]
        prev_primary_id_numeric = this_dict[
            temporal_tracking.PREVIOUS_PRIMARY_ID_KEY]
        prev_secondary_id_numeric = this_dict[
            temporal_tracking.PREVIOUS_SECONDARY_ID_KEY]
        this_old_to_new_dict = this_dict[
            temporal_tracking.OLD_TO_NEW_PRIMARY_IDS_KEY]

        self.assertTrue(numpy.array_equal(
            this_current_to_prev_matrix, CURRENT_TO_PREV_MATRIX_1TO2_POSTMERGE
        ))
        self.assertTrue(
            prev_primary_id_numeric == PREV_PRIMARY_ID_1TO2_POSTMERGE
        )
        self.assertTrue(
            prev_secondary_id_numeric == PREV_SECONDARY_ID_1TO2_POSTMERGE
        )
        self.assertTrue(this_old_to_new_dict == OLD_TO_NEW_DICT_1TO2)

        this_current_local_max_dict = this_dict[
            temporal_tracking.CURRENT_LOCAL_MAXIMA_KEY]
        this_prev_local_max_dict = this_dict[
            temporal_tracking.PREVIOUS_LOCAL_MAXIMA_KEY]

        these_primary_id_strings = this_current_local_max_dict[
            temporal_tracking.PRIMARY_IDS_KEY]
        these_secondary_id_strings = this_current_local_max_dict[
            temporal_tracking.SECONDARY_IDS_KEY]
        these_prev_secondary_ids_listlist = this_current_local_max_dict[
            temporal_tracking.PREV_SECONDARY_IDS_KEY]
        these_next_secondary_ids_listlist = this_prev_local_max_dict[
            temporal_tracking.NEXT_SECONDARY_IDS_KEY]

        self.assertTrue(
            these_primary_id_strings == SECOND_PRIMARY_ID_STRINGS_POSTMERGE
        )
        self.assertTrue(
            these_secondary_id_strings == SECOND_SECONDARY_ID_STRINGS_POSTMERGE
        )
        self.assertTrue(
            these_prev_secondary_ids_listlist ==
            SECOND_PREV_SECONDARY_IDS_POSTMERGE
        )
        self.assertTrue(
            these_next_secondary_ids_listlist ==
            FIRST_NEXT_SECONDARY_IDS_POSTMERGE
        )

    def test_local_maxima_to_tracks_mergers_2to3(self):
        """Ensures correct output from _local_maxima_to_tracks_mergers.

        In this case, linking maxima from the second and third times.
        """

        this_current_local_max_dict = _add_ids_to_dict(
            copy.deepcopy(THIRD_LOCAL_MAX_DICT_SANS_IDS)
        )
        this_current_local_max_dict = _add_id_lists_to_dict(
            this_current_local_max_dict
        )
        this_prev_local_max_dict = _add_id_lists_to_dict(
            copy.deepcopy(SECOND_LOCAL_MAX_DICT_WITH_IDS)
        )

        this_dict = temporal_tracking._local_maxima_to_tracks_mergers(
            current_local_max_dict=this_current_local_max_dict,
            previous_local_max_dict=this_prev_local_max_dict,
            current_to_previous_matrix=copy.deepcopy(
                CURRENT_TO_PREV_MATRIX_2TO3),
            prev_primary_id_numeric=PREV_PRIMARY_ID_2TO3_PREMERGE,
            prev_secondary_id_numeric=PREV_SECONDARY_ID_2TO3_PREMERGE)

        this_current_to_prev_matrix = this_dict[
            temporal_tracking.CURRENT_TO_PREV_MATRIX_KEY]
        prev_primary_id_numeric = this_dict[
            temporal_tracking.PREVIOUS_PRIMARY_ID_KEY]
        prev_secondary_id_numeric = this_dict[
            temporal_tracking.PREVIOUS_SECONDARY_ID_KEY]
        this_old_to_new_dict = this_dict[
            temporal_tracking.OLD_TO_NEW_PRIMARY_IDS_KEY]

        self.assertTrue(numpy.array_equal(
            this_current_to_prev_matrix, CURRENT_TO_PREV_MATRIX_2TO3_POSTMERGE
        ))
        self.assertTrue(
            prev_primary_id_numeric == PREV_PRIMARY_ID_2TO3_POSTMERGE
        )
        self.assertTrue(
            prev_secondary_id_numeric == PREV_SECONDARY_ID_2TO3_POSTMERGE
        )
        self.assertTrue(this_old_to_new_dict == OLD_TO_NEW_DICT_2TO3)

        this_current_local_max_dict = this_dict[
            temporal_tracking.CURRENT_LOCAL_MAXIMA_KEY]
        this_prev_local_max_dict = this_dict[
            temporal_tracking.PREVIOUS_LOCAL_MAXIMA_KEY]

        these_primary_id_strings = this_current_local_max_dict[
            temporal_tracking.PRIMARY_IDS_KEY]
        these_secondary_id_strings = this_current_local_max_dict[
            temporal_tracking.SECONDARY_IDS_KEY]
        these_prev_secondary_ids_listlist = this_current_local_max_dict[
            temporal_tracking.PREV_SECONDARY_IDS_KEY]
        these_next_secondary_ids_listlist = this_prev_local_max_dict[
            temporal_tracking.NEXT_SECONDARY_IDS_KEY]

        self.assertTrue(
            these_primary_id_strings == THIRD_PRIMARY_ID_STRINGS_POSTMERGE
        )
        self.assertTrue(
            these_secondary_id_strings == THIRD_SECONDARY_ID_STRINGS_POSTMERGE
        )
        self.assertTrue(
            these_prev_secondary_ids_listlist ==
            THIRD_PREV_SECONDARY_IDS_POSTMERGE
        )
        self.assertTrue(
            these_next_secondary_ids_listlist ==
            SECOND_NEXT_SECONDARY_IDS_POSTMERGE
        )

    def test_local_maxima_to_tracks_splits_1to2(self):
        """Ensures correct output from _local_maxima_to_tracks_splits.

        In this case, linking maxima from the first and second times.
        """

        this_current_local_max_dict = _add_id_lists_to_dict(
            copy.deepcopy(SECOND_LOCAL_MAX_DICT_SANS_IDS)
        )

        this_current_local_max_dict.update({
            temporal_tracking.PRIMARY_IDS_KEY:
                copy.deepcopy(SECOND_PRIMARY_ID_STRINGS_POSTMERGE),
            temporal_tracking.SECONDARY_IDS_KEY:
                copy.deepcopy(SECOND_SECONDARY_ID_STRINGS_POSTMERGE),
            temporal_tracking.PREV_SECONDARY_IDS_KEY:
                copy.deepcopy(SECOND_PREV_SECONDARY_IDS_POSTMERGE)
        })

        this_prev_local_max_dict = _add_id_lists_to_dict(
            copy.deepcopy(FIRST_LOCAL_MAX_DICT_WITH_IDS)
        )

        this_prev_local_max_dict.update({
            temporal_tracking.NEXT_SECONDARY_IDS_KEY:
                copy.deepcopy(FIRST_NEXT_SECONDARY_IDS_POSTMERGE)
        })

        this_dict = temporal_tracking._local_maxima_to_tracks_splits(
            current_local_max_dict=this_current_local_max_dict,
            previous_local_max_dict=this_prev_local_max_dict,
            current_to_previous_matrix=copy.deepcopy(
                CURRENT_TO_PREV_MATRIX_1TO2_POSTMERGE),
            prev_secondary_id_numeric=PREV_SECONDARY_ID_1TO2_POSTMERGE)

        this_current_to_prev_matrix = this_dict[
            temporal_tracking.CURRENT_TO_PREV_MATRIX_KEY]
        prev_secondary_id_numeric = this_dict[
            temporal_tracking.PREVIOUS_SECONDARY_ID_KEY]

        self.assertTrue(numpy.array_equal(
            this_current_to_prev_matrix, CURRENT_TO_PREV_MATRIX_1TO2_POSTSPLIT
        ))
        self.assertTrue(
            prev_secondary_id_numeric == PREV_SECONDARY_ID_1TO2_POSTSPLIT
        )

        this_current_local_max_dict = this_dict[
            temporal_tracking.CURRENT_LOCAL_MAXIMA_KEY]
        this_prev_local_max_dict = this_dict[
            temporal_tracking.PREVIOUS_LOCAL_MAXIMA_KEY]

        these_primary_id_strings = this_current_local_max_dict[
            temporal_tracking.PRIMARY_IDS_KEY]
        these_secondary_id_strings = this_current_local_max_dict[
            temporal_tracking.SECONDARY_IDS_KEY]
        these_prev_secondary_ids_listlist = this_current_local_max_dict[
            temporal_tracking.PREV_SECONDARY_IDS_KEY]
        these_next_secondary_ids_listlist = this_prev_local_max_dict[
            temporal_tracking.NEXT_SECONDARY_IDS_KEY]

        self.assertTrue(
            these_primary_id_strings == SECOND_PRIMARY_ID_STRINGS_POSTSPLIT
        )
        self.assertTrue(
            these_secondary_id_strings == SECOND_SECONDARY_ID_STRINGS_POSTSPLIT
        )
        self.assertTrue(
            these_prev_secondary_ids_listlist ==
            SECOND_PREV_SECONDARY_IDS_POSTSPLIT
        )
        self.assertTrue(
            these_next_secondary_ids_listlist ==
            FIRST_NEXT_SECONDARY_IDS_POSTSPLIT
        )

    def test_local_maxima_to_tracks_splits_2to3(self):
        """Ensures correct output from _local_maxima_to_tracks_splits.

        In this case, linking maxima from the second and third times.
        """

        this_current_local_max_dict = _add_id_lists_to_dict(
            copy.deepcopy(THIRD_LOCAL_MAX_DICT_SANS_IDS)
        )

        this_current_local_max_dict.update({
            temporal_tracking.PRIMARY_IDS_KEY:
                copy.deepcopy(THIRD_PRIMARY_ID_STRINGS_POSTMERGE),
            temporal_tracking.SECONDARY_IDS_KEY:
                copy.deepcopy(THIRD_SECONDARY_ID_STRINGS_POSTMERGE),
            temporal_tracking.PREV_SECONDARY_IDS_KEY:
                copy.deepcopy(THIRD_PREV_SECONDARY_IDS_POSTMERGE)
        })

        this_prev_local_max_dict = _add_id_lists_to_dict(
            copy.deepcopy(SECOND_LOCAL_MAX_DICT_WITH_IDS)
        )

        this_prev_local_max_dict.update({
            temporal_tracking.NEXT_SECONDARY_IDS_KEY:
                copy.deepcopy(SECOND_NEXT_SECONDARY_IDS_POSTMERGE)
        })

        this_dict = temporal_tracking._local_maxima_to_tracks_splits(
            current_local_max_dict=this_current_local_max_dict,
            previous_local_max_dict=this_prev_local_max_dict,
            current_to_previous_matrix=copy.deepcopy(
                CURRENT_TO_PREV_MATRIX_2TO3_POSTMERGE),
            prev_secondary_id_numeric=PREV_SECONDARY_ID_2TO3_POSTMERGE)

        this_current_to_prev_matrix = this_dict[
            temporal_tracking.CURRENT_TO_PREV_MATRIX_KEY]
        prev_secondary_id_numeric = this_dict[
            temporal_tracking.PREVIOUS_SECONDARY_ID_KEY]

        self.assertTrue(numpy.array_equal(
            this_current_to_prev_matrix, CURRENT_TO_PREV_MATRIX_2TO3_POSTSPLIT
        ))
        self.assertTrue(
            prev_secondary_id_numeric == PREV_SECONDARY_ID_2TO3_POSTSPLIT
        )

        this_current_local_max_dict = this_dict[
            temporal_tracking.CURRENT_LOCAL_MAXIMA_KEY]
        this_prev_local_max_dict = this_dict[
            temporal_tracking.PREVIOUS_LOCAL_MAXIMA_KEY]

        these_primary_id_strings = this_current_local_max_dict[
            temporal_tracking.PRIMARY_IDS_KEY]
        these_secondary_id_strings = this_current_local_max_dict[
            temporal_tracking.SECONDARY_IDS_KEY]
        these_prev_secondary_ids_listlist = this_current_local_max_dict[
            temporal_tracking.PREV_SECONDARY_IDS_KEY]
        these_next_secondary_ids_listlist = this_prev_local_max_dict[
            temporal_tracking.NEXT_SECONDARY_IDS_KEY]

        self.assertTrue(
            these_primary_id_strings == THIRD_PRIMARY_ID_STRINGS_POSTSPLIT
        )
        self.assertTrue(
            these_secondary_id_strings == THIRD_SECONDARY_ID_STRINGS_POSTSPLIT
        )
        self.assertTrue(
            these_prev_secondary_ids_listlist ==
            THIRD_PREV_SECONDARY_IDS_POSTSPLIT
        )
        self.assertTrue(
            these_next_secondary_ids_listlist ==
            SECOND_NEXT_SECONDARY_IDS_POSTSPLIT
        )

    def test_local_maxima_to_tracks_simple_1to2(self):
        """Ensures correct output from _local_maxima_to_tracks_simple.

        In this case, linking maxima from the first and second times.
        """

        this_current_local_max_dict = _add_id_lists_to_dict(
            copy.deepcopy(SECOND_LOCAL_MAX_DICT_SANS_IDS)
        )

        this_current_local_max_dict.update({
            temporal_tracking.PRIMARY_IDS_KEY:
                copy.deepcopy(SECOND_PRIMARY_ID_STRINGS_POSTSPLIT),
            temporal_tracking.SECONDARY_IDS_KEY:
                copy.deepcopy(SECOND_SECONDARY_ID_STRINGS_POSTSPLIT),
            temporal_tracking.PREV_SECONDARY_IDS_KEY:
                copy.deepcopy(SECOND_PREV_SECONDARY_IDS_POSTSPLIT)
        })

        this_prev_local_max_dict = _add_id_lists_to_dict(
            copy.deepcopy(FIRST_LOCAL_MAX_DICT_WITH_IDS)
        )

        this_prev_local_max_dict.update({
            temporal_tracking.NEXT_SECONDARY_IDS_KEY:
                copy.deepcopy(FIRST_NEXT_SECONDARY_IDS_POSTSPLIT)
        })

        this_dict = temporal_tracking._local_maxima_to_tracks_simple(
            current_local_max_dict=this_current_local_max_dict,
            previous_local_max_dict=this_prev_local_max_dict,
            current_to_previous_matrix=copy.deepcopy(
                CURRENT_TO_PREV_MATRIX_1TO2_POSTSPLIT),
            prev_primary_id_numeric=PREV_PRIMARY_ID_1TO2_POSTSPLIT,
            prev_secondary_id_numeric=PREV_SECONDARY_ID_1TO2_POSTSPLIT)

        prev_primary_id_numeric = this_dict[
            temporal_tracking.PREVIOUS_PRIMARY_ID_KEY]
        prev_secondary_id_numeric = this_dict[
            temporal_tracking.PREVIOUS_SECONDARY_ID_KEY]

        self.assertTrue(
            prev_primary_id_numeric == PREV_PRIMARY_ID_1TO2_POST
        )
        self.assertTrue(
            prev_secondary_id_numeric == PREV_SECONDARY_ID_1TO2_POST
        )

        this_current_local_max_dict = this_dict[
            temporal_tracking.CURRENT_LOCAL_MAXIMA_KEY]
        this_prev_local_max_dict = this_dict[
            temporal_tracking.PREVIOUS_LOCAL_MAXIMA_KEY]

        these_primary_id_strings = this_current_local_max_dict[
            temporal_tracking.PRIMARY_IDS_KEY]
        these_secondary_id_strings = this_current_local_max_dict[
            temporal_tracking.SECONDARY_IDS_KEY]
        these_prev_secondary_ids_listlist = this_current_local_max_dict[
            temporal_tracking.PREV_SECONDARY_IDS_KEY]
        these_next_secondary_ids_listlist = this_prev_local_max_dict[
            temporal_tracking.NEXT_SECONDARY_IDS_KEY]

        self.assertTrue(these_primary_id_strings == SECOND_PRIMARY_ID_STRINGS)
        self.assertTrue(
            these_secondary_id_strings == SECOND_SECONDARY_ID_STRINGS
        )
        self.assertTrue(
            these_prev_secondary_ids_listlist == SECOND_PREV_SECONDARY_IDS_POST
        )
        self.assertTrue(
            these_next_secondary_ids_listlist == FIRST_NEXT_SECONDARY_IDS_POST
        )

    def test_local_maxima_to_tracks_simple_2to3(self):
        """Ensures correct output from _local_maxima_to_tracks_simple.

        In this case, linking maxima from the second and third times.
        """

        this_current_local_max_dict = _add_id_lists_to_dict(
            copy.deepcopy(THIRD_LOCAL_MAX_DICT_SANS_IDS)
        )

        this_current_local_max_dict.update({
            temporal_tracking.PRIMARY_IDS_KEY:
                copy.deepcopy(THIRD_PRIMARY_ID_STRINGS_POSTSPLIT),
            temporal_tracking.SECONDARY_IDS_KEY:
                copy.deepcopy(THIRD_SECONDARY_ID_STRINGS_POSTSPLIT),
            temporal_tracking.PREV_SECONDARY_IDS_KEY:
                copy.deepcopy(THIRD_PREV_SECONDARY_IDS_POSTSPLIT)
        })

        this_prev_local_max_dict = _add_id_lists_to_dict(
            copy.deepcopy(SECOND_LOCAL_MAX_DICT_WITH_IDS)
        )

        this_prev_local_max_dict.update({
            temporal_tracking.NEXT_SECONDARY_IDS_KEY:
                copy.deepcopy(SECOND_NEXT_SECONDARY_IDS_POSTSPLIT)
        })

        this_dict = temporal_tracking._local_maxima_to_tracks_simple(
            current_local_max_dict=this_current_local_max_dict,
            previous_local_max_dict=this_prev_local_max_dict,
            current_to_previous_matrix=copy.deepcopy(
                CURRENT_TO_PREV_MATRIX_2TO3_POSTSPLIT),
            prev_primary_id_numeric=PREV_PRIMARY_ID_2TO3_POSTSPLIT,
            prev_secondary_id_numeric=PREV_SECONDARY_ID_2TO3_POSTSPLIT)

        prev_primary_id_numeric = this_dict[
            temporal_tracking.PREVIOUS_PRIMARY_ID_KEY]
        prev_secondary_id_numeric = this_dict[
            temporal_tracking.PREVIOUS_SECONDARY_ID_KEY]

        self.assertTrue(prev_primary_id_numeric == PREV_PRIMARY_ID_2TO3_POST)
        self.assertTrue(
            prev_secondary_id_numeric == PREV_SECONDARY_ID_2TO3_POST
        )

        this_current_local_max_dict = this_dict[
            temporal_tracking.CURRENT_LOCAL_MAXIMA_KEY]
        this_prev_local_max_dict = this_dict[
            temporal_tracking.PREVIOUS_LOCAL_MAXIMA_KEY]

        these_primary_id_strings = this_current_local_max_dict[
            temporal_tracking.PRIMARY_IDS_KEY]
        these_secondary_id_strings = this_current_local_max_dict[
            temporal_tracking.SECONDARY_IDS_KEY]
        these_prev_secondary_ids_listlist = this_current_local_max_dict[
            temporal_tracking.PREV_SECONDARY_IDS_KEY]
        these_next_secondary_ids_listlist = this_prev_local_max_dict[
            temporal_tracking.NEXT_SECONDARY_IDS_KEY]

        self.assertTrue(these_primary_id_strings == THIRD_PRIMARY_ID_STRINGS)
        self.assertTrue(
            these_secondary_id_strings == THIRD_SECONDARY_ID_STRINGS
        )
        self.assertTrue(
            these_prev_secondary_ids_listlist == THIRD_PREV_SECONDARY_IDS_POST
        )
        self.assertTrue(
            these_next_secondary_ids_listlist == SECOND_NEXT_SECONDARY_IDS_POST
        )

    def test_partial_to_full_ids_good(self):
        """Ensures correct output from partial_to_full_ids.

        In this case, inputs are properly formatted (no errors).
        """

        these_full_id_strings = temporal_tracking.partial_to_full_ids(
            primary_id_strings=TOY_PRIMARY_ID_STRINGS,
            secondary_id_strings=TOY_SECONDARY_ID_STRINGS)

        self.assertTrue(these_full_id_strings == TOY_FULL_ID_STRINGS)

    def test_partial_to_full_ids_bad(self):
        """Ensures correct output from partial_to_full_ids.

        In this case, inputs are *not* properly formatted (there is an
        underscore).
        """

        with self.assertRaises(AssertionError):
            temporal_tracking.partial_to_full_ids(
                primary_id_strings=TOY_PRIMARY_ID_STRINGS + ['foo'],
                secondary_id_strings=TOY_SECONDARY_ID_STRINGS + ['_']
            )

    def test_full_to_partial_ids_good(self):
        """Ensures correct output from full_to_partial_ids.

        In this case, inputs are properly formatted (no errors).
        """

        these_primary_id_strings, these_secondary_id_strings = (
            temporal_tracking.full_to_partial_ids(TOY_FULL_ID_STRINGS)
        )

        self.assertTrue(these_primary_id_strings == TOY_PRIMARY_ID_STRINGS)
        self.assertTrue(these_secondary_id_strings == TOY_SECONDARY_ID_STRINGS)

    def test_full_to_partial_ids_bad(self):
        """Ensures correct output from full_to_partial_ids.

        In this case, inputs are *not* properly formatted (there is an
        underscore).
        """

        with self.assertRaises(AssertionError):
            temporal_tracking.full_to_partial_ids(
                TOY_FULL_ID_STRINGS + ['foo_bar_moo']
            )

    def test_local_maxima_to_storm_tracks_1to2(self):
        """Ensures correct output from local_maxima_to_storm_tracks.

        In this case, linking maxima from the first and second times.
        """

        this_max_dict_by_time = [
            copy.deepcopy(FIRST_LOCAL_MAX_DICT_WITH_IDS),
            copy.deepcopy(SECOND_LOCAL_MAX_DICT_WITH_IDS)
        ]

        this_storm_object_table = (
            temporal_tracking.local_maxima_to_storm_tracks(
                local_max_dict_by_time=this_max_dict_by_time, first_numeric_id=0
            )
        )

        these_primary_id_strings = this_storm_object_table[
            tracking_utils.PRIMARY_ID_COLUMN
        ].values.tolist()

        these_secondary_id_strings = this_storm_object_table[
            tracking_utils.SECONDARY_ID_COLUMN
        ].values.tolist()

        self.assertTrue(these_primary_id_strings == PRIMARY_ID_STRINGS_1AND2)
        self.assertTrue(
            these_secondary_id_strings == SECONDARY_ID_STRINGS_1AND2
        )

    def test_local_maxima_to_storm_tracks_1to3(self):
        """Ensures correct output from local_maxima_to_storm_tracks.

        In this case, linking maxima from all three times.
        """

        this_max_dict_by_time = [
            copy.deepcopy(FIRST_LOCAL_MAX_DICT_WITH_IDS),
            copy.deepcopy(SECOND_LOCAL_MAX_DICT_WITH_IDS),
            copy.deepcopy(THIRD_LOCAL_MAX_DICT_WITH_IDS)
        ]

        this_storm_object_table = (
            temporal_tracking.local_maxima_to_storm_tracks(
                local_max_dict_by_time=this_max_dict_by_time, first_numeric_id=0
            )
        )

        these_primary_id_strings = this_storm_object_table[
            tracking_utils.PRIMARY_ID_COLUMN
        ].values.tolist()

        these_secondary_id_strings = this_storm_object_table[
            tracking_utils.SECONDARY_ID_COLUMN
        ].values.tolist()

        self.assertTrue(these_primary_id_strings == PRIMARY_ID_STRINGS)
        self.assertTrue(these_secondary_id_strings == SECONDARY_ID_STRINGS)

    def test_remove_short_lived_storms_5sec(self):
        """Ensures correct output from remove_short_lived_storms.

        In this case, minimum duration is 5 seconds.
        """

        this_max_dict_by_time = [
            copy.deepcopy(FIRST_LOCAL_MAX_DICT_WITH_IDS),
            copy.deepcopy(SECOND_LOCAL_MAX_DICT_WITH_IDS),
            copy.deepcopy(THIRD_LOCAL_MAX_DICT_WITH_IDS)
        ]

        this_storm_object_table = (
            temporal_tracking.local_maxima_to_storm_tracks(
                local_max_dict_by_time=this_max_dict_by_time, first_numeric_id=0
            )
        )

        this_storm_object_table = temporal_tracking.remove_short_lived_storms(
            storm_object_table=this_storm_object_table, min_duration_seconds=5)

        these_primary_id_strings = this_storm_object_table[
            tracking_utils.PRIMARY_ID_COLUMN
        ].values.tolist()

        these_secondary_id_strings = this_storm_object_table[
            tracking_utils.SECONDARY_ID_COLUMN
        ].values.tolist()

        self.assertTrue(these_primary_id_strings == PRIMARY_ID_STRINGS_GE5SEC)
        self.assertTrue(
            these_secondary_id_strings == SECONDARY_ID_STRINGS_GE5SEC
        )

    def test_remove_short_lived_storms_10sec(self):
        """Ensures correct output from remove_short_lived_storms.

        In this case, minimum duration is 10 seconds.
        """

        this_max_dict_by_time = [
            copy.deepcopy(FIRST_LOCAL_MAX_DICT_WITH_IDS),
            copy.deepcopy(SECOND_LOCAL_MAX_DICT_WITH_IDS),
            copy.deepcopy(THIRD_LOCAL_MAX_DICT_WITH_IDS)
        ]

        this_storm_object_table = (
            temporal_tracking.local_maxima_to_storm_tracks(
                local_max_dict_by_time=this_max_dict_by_time, first_numeric_id=0
            )
        )

        this_storm_object_table = temporal_tracking.remove_short_lived_storms(
            storm_object_table=this_storm_object_table, min_duration_seconds=10)

        these_primary_id_strings = this_storm_object_table[
            tracking_utils.PRIMARY_ID_COLUMN
        ].values.tolist()

        these_secondary_id_strings = this_storm_object_table[
            tracking_utils.SECONDARY_ID_COLUMN
        ].values.tolist()

        self.assertTrue(these_primary_id_strings == PRIMARY_ID_STRINGS_GE10SEC)
        self.assertTrue(
            these_secondary_id_strings == SECONDARY_ID_STRINGS_GE10SEC
        )

    def test_check_tracking_periods_good(self):
        """Ensures correct output from check_tracking_periods.

        In this case all inputs are good.
        """

        random_indices = numpy.random.permutation(
            len(GOOD_START_TIMES_UNIX_SEC))

        these_start_times_unix_sec, these_end_times_unix_sec = (
            temporal_tracking.check_tracking_periods(
                tracking_start_times_unix_sec=
                GOOD_START_TIMES_UNIX_SEC[random_indices] + 0,
                tracking_end_times_unix_sec=
                GOOD_END_TIMES_UNIX_SEC[random_indices] + 0
            )
        )

        self.assertTrue(numpy.array_equal(
            these_start_times_unix_sec, GOOD_START_TIMES_UNIX_SEC
        ))
        self.assertTrue(numpy.array_equal(
            these_end_times_unix_sec, GOOD_END_TIMES_UNIX_SEC
        ))

    def test_check_tracking_periods_bad1(self):
        """Ensures correct output from check_tracking_periods.

        In this case, one start time = previous end time.
        """

        with self.assertRaises(ValueError):
            temporal_tracking.check_tracking_periods(
                tracking_start_times_unix_sec=FIRST_BAD_START_TIMES_UNIX_SEC,
                tracking_end_times_unix_sec=FIRST_BAD_END_TIMES_UNIX_SEC
            )

    def test_check_tracking_periods_bad2(self):
        """Ensures correct output from check_tracking_periods.

        In this case, one start time comes before previous end time.
        """

        with self.assertRaises(ValueError):
            temporal_tracking.check_tracking_periods(
                tracking_start_times_unix_sec=SECOND_BAD_START_TIMES_UNIX_SEC,
                tracking_end_times_unix_sec=SECOND_BAD_END_TIMES_UNIX_SEC
            )

    def test_get_storm_ages_first(self):
        """Ensures correct output from get_storm_ages.

        In this case, using first set of parameters.
        """

        this_max_dict_by_time = [
            copy.deepcopy(FIRST_LOCAL_MAX_DICT_WITH_IDS),
            copy.deepcopy(SECOND_LOCAL_MAX_DICT_WITH_IDS),
            copy.deepcopy(THIRD_LOCAL_MAX_DICT_WITH_IDS)
        ]

        this_storm_object_table = (
            temporal_tracking.local_maxima_to_storm_tracks(
                local_max_dict_by_time=this_max_dict_by_time, first_numeric_id=0
            )
        )

        this_storm_object_table = temporal_tracking.get_storm_ages(
            storm_object_table=this_storm_object_table,
            tracking_start_times_unix_sec=FIRST_MAIN_START_TIMES_UNIX_SEC,
            tracking_end_times_unix_sec=FIRST_MAIN_END_TIMES_UNIX_SEC,
            max_link_time_seconds=FIRST_MAX_LINK_TIME_SEC)

        self.assertTrue(numpy.array_equal(
            this_storm_object_table[tracking_utils.AGE_COLUMN].values,
            FIRST_STORM_AGES_SEC
        ))
        self.assertTrue(numpy.array_equal(
            this_storm_object_table[
                tracking_utils.TRACKING_START_TIME_COLUMN].values,
            FIRST_TRACKING_START_TIMES_UNIX_SEC
        ))
        self.assertTrue(numpy.array_equal(
            this_storm_object_table[
                tracking_utils.TRACKING_END_TIME_COLUMN].values,
            FIRST_TRACKING_END_TIMES_UNIX_SEC
        ))
        self.assertTrue(numpy.array_equal(
            this_storm_object_table[
                tracking_utils.CELL_START_TIME_COLUMN].values,
            FIRST_CELL_START_TIMES_UNIX_SEC
        ))
        self.assertTrue(numpy.array_equal(
            this_storm_object_table[
                tracking_utils.CELL_END_TIME_COLUMN].values,
            FIRST_CELL_END_TIMES_UNIX_SEC
        ))

    def test_get_storm_ages_second(self):
        """Ensures correct output from get_storm_ages.

        In this case, using second set of parameters.
        """

        this_max_dict_by_time = [
            copy.deepcopy(FIRST_LOCAL_MAX_DICT_WITH_IDS),
            copy.deepcopy(SECOND_LOCAL_MAX_DICT_WITH_IDS),
            copy.deepcopy(THIRD_LOCAL_MAX_DICT_WITH_IDS)
        ]

        this_storm_object_table = (
            temporal_tracking.local_maxima_to_storm_tracks(
                local_max_dict_by_time=this_max_dict_by_time, first_numeric_id=0
            )
        )

        this_storm_object_table = temporal_tracking.get_storm_ages(
            storm_object_table=this_storm_object_table,
            tracking_start_times_unix_sec=SECOND_MAIN_START_TIMES_UNIX_SEC,
            tracking_end_times_unix_sec=SECOND_MAIN_END_TIMES_UNIX_SEC,
            max_link_time_seconds=SECOND_MAX_LINK_TIME_SEC)

        self.assertTrue(numpy.array_equal(
            this_storm_object_table[tracking_utils.AGE_COLUMN].values,
            SECOND_STORM_AGES_SEC
        ))
        self.assertTrue(numpy.array_equal(
            this_storm_object_table[
                tracking_utils.TRACKING_START_TIME_COLUMN].values,
            SECOND_TRACKING_START_TIMES_UNIX_SEC
        ))
        self.assertTrue(numpy.array_equal(
            this_storm_object_table[
                tracking_utils.TRACKING_END_TIME_COLUMN].values,
            SECOND_TRACKING_END_TIMES_UNIX_SEC
        ))
        self.assertTrue(numpy.array_equal(
            this_storm_object_table[
                tracking_utils.CELL_START_TIME_COLUMN].values,
            SECOND_CELL_START_TIMES_UNIX_SEC
        ))
        self.assertTrue(numpy.array_equal(
            this_storm_object_table[
                tracking_utils.CELL_END_TIME_COLUMN].values,
            SECOND_CELL_END_TIMES_UNIX_SEC
        ))

    def test_get_storm_velocities_15sec(self):
        """Ensures correct output from get_storm_velocities.

        In this case the time window for backwards differencing is 15 seconds.
        """

        this_max_dict_by_time = [
            copy.deepcopy(FIRST_LOCAL_MAX_DICT_WITH_IDS),
            copy.deepcopy(SECOND_LOCAL_MAX_DICT_WITH_IDS),
            copy.deepcopy(THIRD_LOCAL_MAX_DICT_WITH_IDS)
        ]

        this_storm_object_table = (
            temporal_tracking.local_maxima_to_storm_tracks(
                local_max_dict_by_time=this_max_dict_by_time, first_numeric_id=0
            )
        )

        this_storm_object_table = temporal_tracking.get_storm_velocities(
            storm_object_table=this_storm_object_table,
            min_time_difference_sec=5, max_time_difference_sec=15,
            test_mode=True)

        real_indices = numpy.where(
            numpy.invert(numpy.isnan(EAST_VELOCITIES_NO_NEIGH_15SEC_M_S01))
        )[0]

        these_east_velocities_m_s01 = this_storm_object_table[
            tracking_utils.EAST_VELOCITY_COLUMN].values
        these_north_velocities_m_s01 = this_storm_object_table[
            tracking_utils.NORTH_VELOCITY_COLUMN].values

        self.assertFalse(numpy.any(numpy.isnan(these_east_velocities_m_s01)))
        self.assertFalse(numpy.any(numpy.isnan(these_north_velocities_m_s01)))

        self.assertTrue(numpy.allclose(
            these_east_velocities_m_s01[real_indices],
            EAST_VELOCITIES_NO_NEIGH_15SEC_M_S01[real_indices], atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            these_north_velocities_m_s01[real_indices],
            NORTH_VELOCITIES_NO_NEIGH_15SEC_M_S01[real_indices], atol=TOLERANCE
        ))

    def test_get_storm_velocities_5sec(self):
        """Ensures correct output from get_storm_velocities.

        In this case the time window for backwards differencing is 5 seconds.
        """

        this_max_dict_by_time = [
            copy.deepcopy(FIRST_LOCAL_MAX_DICT_WITH_IDS),
            copy.deepcopy(SECOND_LOCAL_MAX_DICT_WITH_IDS),
            copy.deepcopy(THIRD_LOCAL_MAX_DICT_WITH_IDS)
        ]

        this_storm_object_table = (
            temporal_tracking.local_maxima_to_storm_tracks(
                local_max_dict_by_time=this_max_dict_by_time, first_numeric_id=0
            )
        )

        this_storm_object_table = temporal_tracking.get_storm_velocities(
            storm_object_table=this_storm_object_table,
            min_time_difference_sec=5, max_time_difference_sec=5,
            test_mode=True)

        real_indices = numpy.where(
            numpy.invert(numpy.isnan(EAST_VELOCITIES_NO_NEIGH_5SEC_M_S01))
        )[0]

        these_east_velocities_m_s01 = this_storm_object_table[
            tracking_utils.EAST_VELOCITY_COLUMN].values
        these_north_velocities_m_s01 = this_storm_object_table[
            tracking_utils.NORTH_VELOCITY_COLUMN].values

        self.assertFalse(numpy.any(numpy.isnan(these_east_velocities_m_s01)))
        self.assertFalse(numpy.any(numpy.isnan(these_north_velocities_m_s01)))

        self.assertTrue(numpy.allclose(
            these_east_velocities_m_s01[real_indices],
            EAST_VELOCITIES_NO_NEIGH_5SEC_M_S01[real_indices], atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            these_north_velocities_m_s01[real_indices],
            NORTH_VELOCITIES_NO_NEIGH_5SEC_M_S01[real_indices], atol=TOLERANCE
        ))

    def test_get_storm_velocities_missing(self):
        """Ensures correct output from _get_storm_velocities_missing."""

        this_storm_object_table = (
            temporal_tracking._get_storm_velocities_missing(
                storm_object_table=copy.deepcopy(
                    STORM_OBJECT_TABLE_MISSING_VEL),
                e_folding_radius_metres=VELOCITY_EFOLD_RADIUS_METRES)
        )

        self.assertTrue(numpy.allclose(
            this_storm_object_table[tracking_utils.EAST_VELOCITY_COLUMN].values,
            STORM_OBJECT_TABLE_FILLED_VEL[
                tracking_utils.EAST_VELOCITY_COLUMN].values,
            atol=TOLERANCE
        ))

        self.assertTrue(numpy.allclose(
            this_storm_object_table[tracking_utils.NORTH_VELOCITY_COLUMN].values,
            STORM_OBJECT_TABLE_FILLED_VEL[
                tracking_utils.NORTH_VELOCITY_COLUMN].values,
            atol=TOLERANCE
        ))

    def test_finish_segmotion_or_probsevere_ids(self):
        """Ensures correctness of finish_segmotion_or_probsevere_ids."""

        this_table = temporal_tracking.finish_segmotion_or_probsevere_ids(
            copy.deepcopy(STORM_OBJECT_TABLE_UNFINISHED_IDS)
        )

        expected_columns = list(STORM_OBJECT_TABLE_FINISHED_IDS)
        actual_columns = list(this_table)
        self.assertTrue(set(expected_columns) == set(actual_columns))

        self.assertTrue(this_table[expected_columns].equals(
            STORM_OBJECT_TABLE_FINISHED_IDS[expected_columns]
        ))


if __name__ == '__main__':
    unittest.main()
