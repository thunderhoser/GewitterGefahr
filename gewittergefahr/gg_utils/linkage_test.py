"""Unit tests for linkage.py."""

import copy
import unittest
import numpy
import pandas
from gewittergefahr.gg_io import raw_wind_io
from gewittergefahr.gg_utils import linkage
from gewittergefahr.gg_utils import temporal_tracking
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils

TOLERANCE = 1e-6

# The following constants are used for several unit tests.
THESE_TIMES_UNIX_SEC = numpy.array([
    0, 0, 0,
    1, 1,
    2, 2,
    4, 4,
    5, 5, 5, 5,
    6, 6, 6,
    7, 7,
    10,
    11
], dtype=int)

THESE_PRIMARY_ID_STRINGS = [
    'A', 'B', 'B',
    'A', 'B',
    'A', 'B',
    'A', 'B',
    'A', 'A', 'B', 'B',
    'A', 'B', 'B',
    'A', 'A',
    'A',
    'A'
]

THESE_SECONDARY_ID_STRINGS = [
    'A1', 'B1', 'B2',
    'A1', 'B2',
    'A1', 'B3',
    'A1', 'B4',
    'A2', 'A3', 'B4', 'B5',
    'A2', 'B4', 'B5',
    'A2', 'A3',
    'A4',
    'A4'
]

THESE_LATITUDES_DEG = numpy.array([
    50, 60.5, 59.5,
    50, 59.5,
    50, 60,
    50, 60.5,
    50.5, 49.5, 60.5, 59.5,
    50.5, 60.5, 59.5,
    50.5, 49.5,
    50,
    50
])

THESE_LONGITUDES_DEG = numpy.array([
    240, 270, 270,
    240.5, 271,
    241, 272,
    242, 274,
    242.5, 242.5, 275, 275,
    243, 276, 276,
    243.5, 243.5,
    245,
    245.5
])

# THESE_START_TIMES_UNIX_SEC = numpy.full(len(THESE_TIMES_UNIX_SEC), 0, dtype=int)
# THESE_END_TIMES_UNIX_SEC = numpy.array(
#     [11 if p == 'A' else 6 for p in THESE_PRIMARY_ID_STRINGS], dtype=int
# )

THESE_START_TIMES_UNIX_SEC = numpy.array([
    0, 0, 0,
    0, 0,
    0, 2,
    0, 4,
    5, 5, 4, 5,
    5, 4, 5,
    5, 5,
    10,
    10
], dtype=int)

THESE_END_TIMES_UNIX_SEC = numpy.array([
    4, 0, 1,
    4, 1,
    4, 2,
    4, 6,
    7, 7, 6, 6,
    7, 6, 6,
    7, 7,
    11,
    11
], dtype=int)

THESE_FIRST_PREV_SEC_ID_STRINGS = [
    '', '', '',
    'A1', 'B2',
    'A1', 'B1',
    'A1', 'B3',
    'A1', 'A1', 'B4', 'B3',
    'A2', 'B4', 'B5',
    'A2', 'A3',
    'A2',
    'A4'
]

THESE_SECOND_PREV_SEC_ID_STRINGS = [
    '', '', '',
    '', '',
    '', 'B2',
    '', '',
    '', '', '', '',
    '', '', '',
    '', '',
    'A3',
    ''
]

THESE_FIRST_NEXT_SEC_ID_STRINGS = [
    'A1', 'B3', 'B2',
    'A1', 'B3',
    'A1', 'B4',
    'A2', 'B4',
    'A2', 'A3', 'B4', 'B5',
    'A2', '', '',
    'A4', 'A4',
    'A4',
    ''
]

THESE_SECOND_NEXT_SEC_ID_STRINGS = [
    '', '', '',
    '', '',
    '', 'B5',
    'A3', '',
    '', '', '', '',
    '', '', '',
    '', '',
    '',
    ''
]

MAIN_STORM_OBJECT_TABLE = pandas.DataFrame.from_dict({
    tracking_utils.VALID_TIME_COLUMN: THESE_TIMES_UNIX_SEC,
    tracking_utils.PRIMARY_ID_COLUMN: THESE_PRIMARY_ID_STRINGS,
    tracking_utils.SECONDARY_ID_COLUMN: THESE_SECONDARY_ID_STRINGS,
    tracking_utils.CENTROID_LATITUDE_COLUMN: THESE_LATITUDES_DEG,
    tracking_utils.CENTROID_LONGITUDE_COLUMN: THESE_LONGITUDES_DEG,
    linkage.STORM_CENTROID_X_COLUMN: THESE_LONGITUDES_DEG,
    linkage.STORM_CENTROID_Y_COLUMN: THESE_LATITUDES_DEG,
    tracking_utils.CELL_START_TIME_COLUMN: THESE_START_TIMES_UNIX_SEC,
    tracking_utils.CELL_END_TIME_COLUMN: THESE_END_TIMES_UNIX_SEC,
    tracking_utils.FIRST_PREV_SECONDARY_ID_COLUMN:
        THESE_FIRST_PREV_SEC_ID_STRINGS,
    tracking_utils.SECOND_PREV_SECONDARY_ID_COLUMN:
        THESE_SECOND_PREV_SEC_ID_STRINGS,
    tracking_utils.FIRST_NEXT_SECONDARY_ID_COLUMN:
        THESE_FIRST_NEXT_SEC_ID_STRINGS,
    tracking_utils.SECOND_NEXT_SECONDARY_ID_COLUMN:
        THESE_SECOND_NEXT_SEC_ID_STRINGS
})

THIS_NESTED_ARRAY = MAIN_STORM_OBJECT_TABLE[[
    tracking_utils.VALID_TIME_COLUMN, tracking_utils.VALID_TIME_COLUMN
]].values.tolist()

MAIN_STORM_OBJECT_TABLE = MAIN_STORM_OBJECT_TABLE.assign(**{
    linkage.STORM_VERTICES_X_COLUMN: THIS_NESTED_ARRAY,
    linkage.STORM_VERTICES_Y_COLUMN: THIS_NESTED_ARRAY
})

THIS_NUM_STORM_OBJECTS = len(MAIN_STORM_OBJECT_TABLE.index)
THESE_X_VERTICES_RELATIVE = numpy.array([-0.25, 0.25, 0.25, -0.25, -0.25])
THESE_Y_VERTICES_RELATIVE = numpy.array([-0.25, -0.25, 0.25, 0.25, -0.25])

for k in range(THIS_NUM_STORM_OBJECTS):
    MAIN_STORM_OBJECT_TABLE[linkage.STORM_VERTICES_X_COLUMN].values[k] = (
        MAIN_STORM_OBJECT_TABLE[linkage.STORM_CENTROID_X_COLUMN].values[k] +
        THESE_X_VERTICES_RELATIVE
    )

    MAIN_STORM_OBJECT_TABLE[linkage.STORM_VERTICES_Y_COLUMN].values[k] = (
        MAIN_STORM_OBJECT_TABLE[linkage.STORM_CENTROID_Y_COLUMN].values[k] +
        THESE_Y_VERTICES_RELATIVE
    )

# The following constants are used to test _filter_storms_by_time.
EARLY_START_TIME_UNIX_SEC = 4
LATE_START_TIME_UNIX_SEC = 6
EARLY_END_TIME_UNIX_SEC = 6
LATE_END_TIME_UNIX_SEC = 7

EARLY_START_BAD_INDICES = numpy.array(
    [9, 10, 12, 13, 15, 16, 17, 18, 19], dtype=int
)
LATE_START_BAD_INDICES = numpy.array([18, 19], dtype=int)
EARLY_END_BAD_INDICES = numpy.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=int)
LATE_END_BAD_INDICES = numpy.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 14, 15], dtype=int
)

THESE_BAD_INDICES = numpy.unique(numpy.concatenate((
    EARLY_START_BAD_INDICES, EARLY_END_BAD_INDICES
)))

STORM_OBJECT_TABLE_EARLY_START_EARLY_END = MAIN_STORM_OBJECT_TABLE.drop(
    MAIN_STORM_OBJECT_TABLE.index[THESE_BAD_INDICES], axis=0, inplace=False
)

THESE_BAD_INDICES = numpy.unique(numpy.concatenate((
    EARLY_START_BAD_INDICES, LATE_END_BAD_INDICES
)))

STORM_OBJECT_TABLE_EARLY_START_LATE_END = MAIN_STORM_OBJECT_TABLE.drop(
    MAIN_STORM_OBJECT_TABLE.index[THESE_BAD_INDICES], axis=0, inplace=False
)

THESE_BAD_INDICES = numpy.unique(numpy.concatenate((
    LATE_START_BAD_INDICES, EARLY_END_BAD_INDICES
)))

STORM_OBJECT_TABLE_LATE_START_EARLY_END = MAIN_STORM_OBJECT_TABLE.drop(
    MAIN_STORM_OBJECT_TABLE.index[THESE_BAD_INDICES], axis=0, inplace=False
)

THESE_BAD_INDICES = numpy.unique(numpy.concatenate((
    LATE_START_BAD_INDICES, LATE_END_BAD_INDICES
)))

STORM_OBJECT_TABLE_LATE_START_LATE_END = MAIN_STORM_OBJECT_TABLE.drop(
    MAIN_STORM_OBJECT_TABLE.index[THESE_BAD_INDICES], axis=0, inplace=False
)

# The following constants are used to test _interp_one_storm_in_time.
STORM_OBJECT_TABLE_1CELL = MAIN_STORM_OBJECT_TABLE.loc[
    MAIN_STORM_OBJECT_TABLE[tracking_utils.SECONDARY_ID_COLUMN] == 'A1'
]

INTERP_TIME_1CELL_UNIX_SEC = 3
THESE_X_VERTICES = 241.5 + THESE_X_VERTICES_RELATIVE
THESE_Y_VERTICES = 50. + THESE_Y_VERTICES_RELATIVE

VERTEX_TABLE_1OBJECT_INTERP = pandas.DataFrame.from_dict({
    tracking_utils.SECONDARY_ID_COLUMN: ['A1'] * 5,
    linkage.STORM_VERTEX_X_COLUMN: THESE_X_VERTICES,
    linkage.STORM_VERTEX_Y_COLUMN: THESE_Y_VERTICES
})

EXTRAP_TIME_1CELL_UNIX_SEC = 5
THESE_X_VERTICES = 242.5 + THESE_X_VERTICES_RELATIVE
THESE_Y_VERTICES = 50. + THESE_Y_VERTICES_RELATIVE

VERTEX_TABLE_1OBJECT_EXTRAP = pandas.DataFrame.from_dict({
    tracking_utils.SECONDARY_ID_COLUMN: ['A1'] * 5,
    linkage.STORM_VERTEX_X_COLUMN: THESE_X_VERTICES,
    linkage.STORM_VERTEX_Y_COLUMN: THESE_Y_VERTICES
})

# The following constants are used to test _get_bounding_box_for_storms.
BOUNDING_BOX_PADDING_METRES = 1000.
BOUNDING_BOX_X_LIMITS_METRES = numpy.array([-760.25, 1276.25])
BOUNDING_BOX_Y_LIMITS_METRES = numpy.array([-950.75, 1060.75])

# The following constants are used to test _filter_events_by_bounding_box.
THESE_EVENT_X_METRES = numpy.array(
    [-1000, -500, 0, 500, 1000, -1000, -500, 500, 1000], dtype=float
)
THESE_EVENT_Y_METRES = numpy.array(
    [-1000, -500, 0, 500, 1000, 1000, 500, -500, -1000], dtype=float
)

EVENT_TABLE_FULL_DOMAIN = pandas.DataFrame.from_dict({
    linkage.EVENT_X_COLUMN: THESE_EVENT_X_METRES,
    linkage.EVENT_Y_COLUMN: THESE_EVENT_Y_METRES
})

THESE_BAD_INDICES = numpy.array([0, 5, 8], dtype=int)
EVENT_TABLE_IN_BOUNDING_BOX = EVENT_TABLE_FULL_DOMAIN.drop(
    EVENT_TABLE_FULL_DOMAIN.index[THESE_BAD_INDICES], axis=0, inplace=False
)

# The following constants are used to test _interp_storms_in_time.
INTERP_TIME_UNIX_SEC = 3

THESE_CENTROID_ID_STRINGS = [
    'A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'B4', 'B5'
]
THESE_CENTROID_X_COORDS = numpy.array(
    [241.5, 241.5, 241.5, 241.5, 273, 273, 273, 273, 273]
)
THESE_CENTROID_Y_COORDS = numpy.array(
    [50, 49.5, 50.5, 50, 59.75, 60.5, 60.25, 60.25, 59.833333]
)

THESE_VERTEX_ID_STRINGS = []
THESE_VERTEX_X_COORDS = numpy.array([])
THESE_VERTEX_Y_COORDS = numpy.array([])

for k in range(len(THESE_CENTROID_ID_STRINGS)):
    THESE_VERTEX_ID_STRINGS += [THESE_CENTROID_ID_STRINGS[k]] * 5

    THESE_VERTEX_X_COORDS = numpy.concatenate((
        THESE_VERTEX_X_COORDS,
        THESE_CENTROID_X_COORDS[k] + THESE_X_VERTICES_RELATIVE
    ))

    THESE_VERTEX_Y_COORDS = numpy.concatenate((
        THESE_VERTEX_Y_COORDS,
        THESE_CENTROID_Y_COORDS[k] + THESE_Y_VERTICES_RELATIVE
    ))

INTERP_VERTEX_TABLE = pandas.DataFrame.from_dict({
    tracking_utils.SECONDARY_ID_COLUMN: THESE_VERTEX_ID_STRINGS,
    linkage.STORM_VERTEX_X_COLUMN: THESE_VERTEX_X_COORDS,
    linkage.STORM_VERTEX_Y_COLUMN: THESE_VERTEX_Y_COORDS
})

# The following constants are used to test _find_nearest_storms_one_time.
MAX_LINK_DISTANCE_METRES = 10000.

EVENT_X_COORDS_1TIME_METRES = numpy.array(
    [49000, 49000, 49000, 49000, -46500, -36500, -31500, 0], dtype=float
)
EVENT_Y_COORDS_1TIME_METRES = numpy.array(
    [55000, 50000, 45000, 0, -43625, -43625, -43625, -43625], dtype=float
)

NEAREST_PRIMARY_ID_STRINGS_1TIME = [
    'bar', 'bar', 'bar', None, 'foo', 'foo', 'foo', None
]
LINK_DISTANCES_1TIME_METRES = numpy.array(
    [0, 0, 5000, numpy.nan, 0, 0, 5000, numpy.nan]
)

# The following constants are used to test _find_nearest_storms.
INTERP_TIME_RESOLUTION_SEC = 10

THESE_X_METRES = numpy.array([
    49000, 49000, 49000, 49000, -46500, -36500, -31500, 0,
    49000, 49000, 49000, 49000, -46000, -36000, -31000, 0
], dtype=float)

THESE_Y_METRES = numpy.array([
    55000, 50000, 45000, 0, -43625, -43625, -43625, -43625,
    55000, 50000, 45000, 0, -43500, -43500, -43500, -43500
], dtype=float)

THESE_TIMES_UNIX_SEC = numpy.array([
    600, 600, 600, 600, 600, 600, 600, 600,
    700, 700, 700, 700, 700, 700, 700, 700
], dtype=int)

THIS_DICT = {
    linkage.EVENT_X_COLUMN: THESE_X_METRES,
    linkage.EVENT_Y_COLUMN: THESE_Y_METRES,
    linkage.EVENT_LONGITUDE_COLUMN: THESE_X_METRES,
    linkage.EVENT_LATITUDE_COLUMN: THESE_Y_METRES,
    linkage.EVENT_TIME_COLUMN: THESE_TIMES_UNIX_SEC
}
EVENT_TABLE_2TIMES = pandas.DataFrame.from_dict(THIS_DICT)

THESE_PRIMARY_ID_STRINGS = [
    'bar', 'bar', 'bar', None, 'foo', 'foo', 'foo', None,
    None, None, None, None, 'foo', 'foo', 'foo', None
]
THESE_LINK_DISTANCES_METRES = numpy.array([
    0, 0, 5000, numpy.nan, 0, 0, 5000, numpy.nan,
    numpy.nan, numpy.nan, numpy.nan, numpy.nan, 0, 0, 5000, numpy.nan
])

THIS_DICT = {
    linkage.NEAREST_SECONDARY_ID_COLUMN: THESE_PRIMARY_ID_STRINGS,
    linkage.LINKAGE_DISTANCE_COLUMN: THESE_LINK_DISTANCES_METRES
}
EVENT_TO_STORM_TABLE_SIMPLE = EVENT_TABLE_2TIMES.assign(**THIS_DICT)

# The following constants are used to test _reverse_wind_linkages.
THESE_STATION_IDS = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'
]
THESE_LATITUDES_DEG = numpy.array([
    1, 2, 3, 4, 5, 6, 7, 8,
    1, 2, 3, 4, 5, 6, 7, 8
], dtype=float)

THESE_LONGITUDES_DEG = THESE_LATITUDES_DEG + 0.
THESE_U_WINDS_M_S01 = THESE_LATITUDES_DEG + 0.
THESE_V_WINDS_M_S01 = THESE_LATITUDES_DEG + 0.

THIS_DICT = {
    raw_wind_io.STATION_ID_COLUMN: THESE_STATION_IDS,
    linkage.EVENT_LATITUDE_COLUMN: THESE_LATITUDES_DEG,
    linkage.EVENT_LONGITUDE_COLUMN: THESE_LONGITUDES_DEG,
    raw_wind_io.U_WIND_COLUMN: THESE_U_WINDS_M_S01,
    raw_wind_io.V_WIND_COLUMN: THESE_V_WINDS_M_S01
}

WIND_TO_STORM_TABLE = EVENT_TO_STORM_TABLE_SIMPLE.assign(**THIS_DICT)

# STORM_TO_WINDS_TABLE = copy.deepcopy(STORM_OBJECT_TABLE_2CELLS)
# THIS_NESTED_ARRAY = STORM_TO_WINDS_TABLE[[
#     tracking_utils.VALID_TIME_COLUMN, tracking_utils.VALID_TIME_COLUMN
# ]].values.tolist()
#
# THIS_DICT = {
#     linkage.WIND_STATION_IDS_COLUMN: THIS_NESTED_ARRAY,
#     linkage.EVENT_LATITUDES_COLUMN: THIS_NESTED_ARRAY,
#     linkage.EVENT_LONGITUDES_COLUMN: THIS_NESTED_ARRAY,
#     linkage.U_WINDS_COLUMN: THIS_NESTED_ARRAY,
#     linkage.V_WINDS_COLUMN: THIS_NESTED_ARRAY,
#     linkage.LINKAGE_DISTANCES_COLUMN: THIS_NESTED_ARRAY,
#     linkage.RELATIVE_EVENT_TIMES_COLUMN: THIS_NESTED_ARRAY
# }
#
# STORM_TO_WINDS_TABLE = STORM_TO_WINDS_TABLE.assign(**THIS_DICT)
#
# THESE_STATION_IDS = ['e', 'f', 'g', 'e', 'f', 'g']
# THESE_WIND_LATITUDES_DEG = numpy.array([5, 6, 7, 5, 6, 7], dtype=float)
# THESE_WIND_LONGITUDES_DEG = numpy.array([5, 6, 7, 5, 6, 7], dtype=float)
# THESE_U_WINDS_M_S01 = numpy.array([5, 6, 7, 5, 6, 7], dtype=float)
# THESE_V_WINDS_M_S01 = numpy.array([5, 6, 7, 5, 6, 7], dtype=float)
# THESE_LINK_DISTANCES_METRES = numpy.array([0, 0, 5000, 0, 0, 5000], dtype=float)
#
# FOO_ROWS = numpy.array([0, 2, 4], dtype=int)
#
# for this_row in FOO_ROWS:
#     STORM_TO_WINDS_TABLE[linkage.WIND_STATION_IDS_COLUMN].values[this_row] = (
#         THESE_STATION_IDS
#     )
#     STORM_TO_WINDS_TABLE[linkage.EVENT_LATITUDES_COLUMN].values[this_row] = (
#         THESE_WIND_LATITUDES_DEG
#     )
#     STORM_TO_WINDS_TABLE[linkage.EVENT_LONGITUDES_COLUMN].values[this_row] = (
#         THESE_WIND_LONGITUDES_DEG
#     )
#     STORM_TO_WINDS_TABLE[linkage.U_WINDS_COLUMN].values[this_row] = (
#         THESE_U_WINDS_M_S01
#     )
#     STORM_TO_WINDS_TABLE[linkage.V_WINDS_COLUMN].values[this_row] = (
#         THESE_V_WINDS_M_S01
#     )
#     STORM_TO_WINDS_TABLE[linkage.LINKAGE_DISTANCES_COLUMN].values[this_row] = (
#         THESE_LINK_DISTANCES_METRES
#     )
#
# STORM_TO_WINDS_TABLE[linkage.RELATIVE_EVENT_TIMES_COLUMN].values[0] = (
#     numpy.array([600, 600, 600, 700, 700, 700], dtype=int)
# )
# STORM_TO_WINDS_TABLE[linkage.RELATIVE_EVENT_TIMES_COLUMN].values[2] = (
#     numpy.array([300, 300, 300, 400, 400, 400], dtype=int)
# )
# STORM_TO_WINDS_TABLE[linkage.RELATIVE_EVENT_TIMES_COLUMN].values[4] = (
#     numpy.array([-100, -100, -100, 0, 0, 0], dtype=int)
# )
#
# THESE_STATION_IDS = ['a', 'b', 'c']
# THESE_WIND_LATITUDES_DEG = numpy.array([1, 2, 3], dtype=float)
# THESE_WIND_LONGITUDES_DEG = numpy.array([1, 2, 3], dtype=float)
# THESE_U_WINDS_M_S01 = numpy.array([1, 2, 3], dtype=float)
# THESE_V_WINDS_M_S01 = numpy.array([1, 2, 3], dtype=float)
# THESE_LINK_DISTANCES_METRES = numpy.array([0, 0, 5000], dtype=float)
#
# BAR_ROWS = numpy.array([1, 3, 5], dtype=int)
#
# for this_row in BAR_ROWS:
#     STORM_TO_WINDS_TABLE[linkage.WIND_STATION_IDS_COLUMN].values[this_row] = (
#         THESE_STATION_IDS
#     )
#     STORM_TO_WINDS_TABLE[linkage.EVENT_LATITUDES_COLUMN].values[this_row] = (
#         THESE_WIND_LATITUDES_DEG
#     )
#     STORM_TO_WINDS_TABLE[linkage.EVENT_LONGITUDES_COLUMN].values[this_row] = (
#         THESE_WIND_LONGITUDES_DEG
#     )
#     STORM_TO_WINDS_TABLE[linkage.U_WINDS_COLUMN].values[this_row] = (
#         THESE_U_WINDS_M_S01
#     )
#     STORM_TO_WINDS_TABLE[linkage.V_WINDS_COLUMN].values[this_row] = (
#         THESE_V_WINDS_M_S01
#     )
#     STORM_TO_WINDS_TABLE[linkage.LINKAGE_DISTANCES_COLUMN].values[this_row] = (
#         THESE_LINK_DISTANCES_METRES
#     )
#
# STORM_TO_WINDS_TABLE[linkage.RELATIVE_EVENT_TIMES_COLUMN].values[1] = (
#     numpy.array([600, 600, 600], dtype=int)
# )
# STORM_TO_WINDS_TABLE[linkage.RELATIVE_EVENT_TIMES_COLUMN].values[3] = (
#     numpy.array([300, 300, 300], dtype=int)
# )
# STORM_TO_WINDS_TABLE[linkage.RELATIVE_EVENT_TIMES_COLUMN].values[5] = (
#     numpy.array([0, 0, 0], dtype=int)
# )

# The following constants are used to test _remove_storms_near_start_of_period.
THESE_PRIMARY_ID_STRINGS = [
    'A', 'A', 'A', 'A', 'A', 'A',
    'B', 'B',
    'C', 'C', 'C',
    'D', 'D', 'D', 'D',
    'E', 'E', 'E', 'E', 'E',
    'H',
    'G',
    'F'
]

THESE_VALID_TIMES_UNIX_SEC = numpy.array([
    0, 300, 600, 900, 1200, 1500,
    0, 300,
    0, 300, 600,
    600, 900, 1200, 1500,
    2400, 2700, 3000, 3300, 3600,
    2700,
    3000,
    2400
], dtype=int)

THESE_START_TIMES_UNIX_SEC = numpy.array([
    0, 0, 0, 0, 0, 0,
    0, 0,
    0, 0, 0,
    0, 0, 0, 0,
    2400, 2400, 2400, 2400, 2400,
    2400,
    2400,
    2400
], dtype=int)

MIN_TIME_SINCE_START_SEC = 590

STORM_OBJECT_TABLE_WITH_PERIOD_START = pandas.DataFrame.from_dict({
    tracking_utils.PRIMARY_ID_COLUMN: THESE_PRIMARY_ID_STRINGS,
    tracking_utils.VALID_TIME_COLUMN: THESE_VALID_TIMES_UNIX_SEC,
    tracking_utils.TRACKING_START_TIME_COLUMN: THESE_START_TIMES_UNIX_SEC
})

THESE_PRIMARY_ID_STRINGS = [
    'A', 'A', 'A', 'A',
    'C',
    'D', 'D', 'D', 'D',
    'E', 'E', 'E',
    'G',
]

THESE_VALID_TIMES_UNIX_SEC = numpy.array([
    600, 900, 1200, 1500,
    600,
    600, 900, 1200, 1500,
    3000, 3300, 3600,
    3000
], dtype=int)

THESE_START_TIMES_UNIX_SEC = numpy.array([
    0, 0, 0, 0,
    0,
    0, 0, 0, 0,
    2400, 2400, 2400,
    2400
], dtype=int)

STORM_OBJECT_TABLE_SANS_PERIOD_START = pandas.DataFrame.from_dict({
    tracking_utils.PRIMARY_ID_COLUMN: THESE_PRIMARY_ID_STRINGS,
    tracking_utils.VALID_TIME_COLUMN: THESE_VALID_TIMES_UNIX_SEC,
    tracking_utils.TRACKING_START_TIME_COLUMN: THESE_START_TIMES_UNIX_SEC
})

# The following constants are used to test _share_linkages_between_periods.
THESE_EARLY_PRIMARY_ID_STRINGS = ['A', 'C', 'A', 'B', 'C']
THESE_EARLY_SEC_ID_STRINGS = ['1', '2', '3', '4', '2']

THESE_EARLY_FULL_ID_STRINGS = temporal_tracking.partial_to_full_ids(
    primary_id_strings=THESE_EARLY_PRIMARY_ID_STRINGS,
    secondary_id_strings=THESE_EARLY_SEC_ID_STRINGS)

THESE_EARLY_TIMES_UNIX_SEC = numpy.array([0, 0, 1, 1, 1], dtype=int)

THESE_EVENT_LATITUDES_DEG = [
    numpy.array([53, 53]), numpy.array([55, 55]),
    numpy.array([53, 53]), numpy.array([54, 54]), numpy.array([55, 55])
]
THESE_EVENT_LONGITUDES_DEG = [
    numpy.array([246, 247]), numpy.array([246, 247]),
    numpy.array([246, 247]), numpy.array([246, 247]), numpy.array([246, 247])
]
THESE_LINK_DIST_METRES = [
    numpy.array([1000, 2000]), numpy.array([0, 0]),
    numpy.array([1000, 2000]), numpy.array([5000, 10000]), numpy.array([0, 0])
]
THESE_RELATIVE_TIMES_UNIX_SEC = [
    numpy.array([1, 2]), numpy.array([5, 6]),
    numpy.array([0, 1]), numpy.array([2, 3]), numpy.array([4, 5])
]
THESE_FUJITA_RATINGS = [
    ['F0', 'F1'], ['EF4', 'EF5'],
    ['F0', 'F1'], ['EF2', 'EF3'], ['EF4', 'EF5']
]

for k in range(len(THESE_EARLY_PRIMARY_ID_STRINGS)):
    THESE_EVENT_LATITUDES_DEG[k] = THESE_EVENT_LATITUDES_DEG[k].astype(float)
    THESE_EVENT_LONGITUDES_DEG[k] = THESE_EVENT_LONGITUDES_DEG[k].astype(float)
    THESE_LINK_DIST_METRES[k] = THESE_LINK_DIST_METRES[k].astype(float)
    THESE_RELATIVE_TIMES_UNIX_SEC[k] = THESE_RELATIVE_TIMES_UNIX_SEC[k].astype(
        int)

THIS_DICT = {
    tracking_utils.PRIMARY_ID_COLUMN: THESE_EARLY_PRIMARY_ID_STRINGS,
    tracking_utils.SECONDARY_ID_COLUMN: THESE_EARLY_SEC_ID_STRINGS,
    tracking_utils.FULL_ID_COLUMN: THESE_EARLY_FULL_ID_STRINGS,
    tracking_utils.VALID_TIME_COLUMN: THESE_EARLY_TIMES_UNIX_SEC,
    linkage.EVENT_LATITUDES_COLUMN: THESE_EVENT_LATITUDES_DEG,
    linkage.EVENT_LONGITUDES_COLUMN: THESE_EVENT_LONGITUDES_DEG,
    linkage.LINKAGE_DISTANCES_COLUMN: THESE_LINK_DIST_METRES,
    linkage.RELATIVE_EVENT_TIMES_COLUMN: THESE_RELATIVE_TIMES_UNIX_SEC,
    linkage.FUJITA_RATINGS_COLUMN: THESE_FUJITA_RATINGS
}

EARLY_STORM_TO_TORNADOES_TABLE_SANS_SHARING = pandas.DataFrame.from_dict(
    THIS_DICT)

THESE_LATE_PRIMARY_ID_STRINGS = ['B', 'C', 'D', 'C', 'D']
THESE_LATE_SEC_ID_STRINGS = ['4', '2', '5', '2', '6']
THESE_LATE_FULL_ID_STRINGS = temporal_tracking.partial_to_full_ids(
    primary_id_strings=THESE_LATE_PRIMARY_ID_STRINGS,
    secondary_id_strings=THESE_LATE_SEC_ID_STRINGS)

THESE_LATE_TIMES_UNIX_SEC = numpy.array([2, 2, 2, 3, 3], dtype=int)

THESE_EVENT_LATITUDES_DEG = [
    numpy.array([53.5, 53.5]), numpy.array([54.5, 54.5]), numpy.array([70, 70]),
    numpy.array([54.5, 54.5]), numpy.array([70, 70])
]
THESE_EVENT_LONGITUDES_DEG = [
    numpy.array([246, 247]), numpy.array([246, 247]), numpy.array([246, 247]),
    numpy.array([246, 247]), numpy.array([246, 247])
]
THESE_LINK_DIST_METRES = [
    numpy.array([333, 666]), numpy.array([0, 1]), numpy.array([2, 3]),
    numpy.array([0, 1]), numpy.array([2, 3])
]
THESE_RELATIVE_TIMES_UNIX_SEC = [
    numpy.array([0, 2]), numpy.array([2, 4]), numpy.array([4, 6]),
    numpy.array([1, 3]), numpy.array([3, 5])
]
THESE_FUJITA_RATINGS = [
    ['f0', 'f1'], ['ef2', 'ef3'], ['ef4', 'ef5'],
    ['ef2', 'ef3'], ['ef4', 'ef5']
]

for k in range(len(THESE_LATE_PRIMARY_ID_STRINGS)):
    THESE_EVENT_LATITUDES_DEG[k] = THESE_EVENT_LATITUDES_DEG[k].astype(float)
    THESE_EVENT_LONGITUDES_DEG[k] = THESE_EVENT_LONGITUDES_DEG[k].astype(float)
    THESE_LINK_DIST_METRES[k] = THESE_LINK_DIST_METRES[k].astype(float)
    THESE_RELATIVE_TIMES_UNIX_SEC[k] = THESE_RELATIVE_TIMES_UNIX_SEC[k].astype(
        int)

THIS_DICT = {
    tracking_utils.PRIMARY_ID_COLUMN: THESE_LATE_PRIMARY_ID_STRINGS,
    tracking_utils.SECONDARY_ID_COLUMN: THESE_LATE_SEC_ID_STRINGS,
    tracking_utils.FULL_ID_COLUMN: THESE_LATE_FULL_ID_STRINGS,
    tracking_utils.VALID_TIME_COLUMN: THESE_LATE_TIMES_UNIX_SEC,
    linkage.EVENT_LATITUDES_COLUMN: THESE_EVENT_LATITUDES_DEG,
    linkage.EVENT_LONGITUDES_COLUMN: THESE_EVENT_LONGITUDES_DEG,
    linkage.LINKAGE_DISTANCES_COLUMN: THESE_LINK_DIST_METRES,
    linkage.RELATIVE_EVENT_TIMES_COLUMN: THESE_RELATIVE_TIMES_UNIX_SEC,
    linkage.FUJITA_RATINGS_COLUMN: THESE_FUJITA_RATINGS
}

LATE_STORM_TO_TORNADOES_TABLE_SANS_SHARING = pandas.DataFrame.from_dict(
    THIS_DICT)

THESE_EVENT_LATITUDES_DEG = [
    numpy.array([53, 53]), numpy.array([54.5, 54.5, 55, 55]),
    numpy.array([53, 53]), numpy.array([53.5, 53.5, 54, 54]),
    numpy.array([54.5, 54.5, 55, 55])
]
THESE_EVENT_LONGITUDES_DEG = [
    numpy.array([246, 247]), numpy.array([246, 247, 246, 247]),
    numpy.array([246, 247]), numpy.array([246, 247, 246, 247]),
    numpy.array([246, 247, 246, 247])
]
THESE_LINK_DIST_METRES = [
    numpy.array([1000, 2000]), numpy.array([0, 1, 0, 0]),
    numpy.array([1000, 2000]), numpy.array([333, 666, 5000, 10000]),
    numpy.array([0, 1, 0, 0])
]
THESE_RELATIVE_TIMES_UNIX_SEC = [
    numpy.array([1, 2]), numpy.array([4, 6, 5, 6]),
    numpy.array([0, 1]), numpy.array([1, 3, 2, 3]),
    numpy.array([3, 5, 4, 5])
]
THESE_FUJITA_RATINGS = [
    ['F0', 'F1'], ['ef2', 'ef3', 'EF4', 'EF5'],
    ['F0', 'F1'], ['f0', 'f1', 'EF2', 'EF3'],
    ['ef2', 'ef3', 'EF4', 'EF5']
]

for k in range(len(THESE_EARLY_PRIMARY_ID_STRINGS)):
    THESE_EVENT_LATITUDES_DEG[k] = THESE_EVENT_LATITUDES_DEG[k].astype(float)
    THESE_EVENT_LONGITUDES_DEG[k] = THESE_EVENT_LONGITUDES_DEG[k].astype(float)
    THESE_LINK_DIST_METRES[k] = THESE_LINK_DIST_METRES[k].astype(float)
    THESE_RELATIVE_TIMES_UNIX_SEC[k] = THESE_RELATIVE_TIMES_UNIX_SEC[k].astype(
        int)

THIS_DICT = {
    tracking_utils.PRIMARY_ID_COLUMN: THESE_EARLY_PRIMARY_ID_STRINGS,
    tracking_utils.SECONDARY_ID_COLUMN: THESE_EARLY_SEC_ID_STRINGS,
    tracking_utils.FULL_ID_COLUMN: THESE_EARLY_FULL_ID_STRINGS,
    tracking_utils.VALID_TIME_COLUMN: THESE_EARLY_TIMES_UNIX_SEC,
    linkage.EVENT_LATITUDES_COLUMN: THESE_EVENT_LATITUDES_DEG,
    linkage.EVENT_LONGITUDES_COLUMN: THESE_EVENT_LONGITUDES_DEG,
    linkage.LINKAGE_DISTANCES_COLUMN: THESE_LINK_DIST_METRES,
    linkage.RELATIVE_EVENT_TIMES_COLUMN: THESE_RELATIVE_TIMES_UNIX_SEC,
    linkage.FUJITA_RATINGS_COLUMN: THESE_FUJITA_RATINGS
}

EARLY_STORM_TO_TORNADOES_TABLE_WITH_SHARING = pandas.DataFrame.from_dict(
    THIS_DICT)

THESE_EVENT_LATITUDES_DEG = [
    numpy.array([53.5, 53.5, 54, 54]), numpy.array([54.5, 54.5, 55, 55]),
    numpy.array([70, 70]), numpy.array([54.5, 54.5, 55, 55]),
    numpy.array([70, 70])
]
THESE_EVENT_LONGITUDES_DEG = [
    numpy.array([246, 247, 246, 247]), numpy.array([246, 247, 246, 247]),
    numpy.array([246, 247]), numpy.array([246, 247, 246, 247]),
    numpy.array([246, 247])
]
THESE_LINK_DIST_METRES = [
    numpy.array([333, 666, 5000, 10000]), numpy.array([0, 1, 0, 0]),
    numpy.array([2, 3]), numpy.array([0, 1, 0, 0]),
    numpy.array([2, 3])
]
THESE_RELATIVE_TIMES_UNIX_SEC = [
    numpy.array([0, 2, 1, 2]), numpy.array([2, 4, 3, 4]),
    numpy.array([4, 6]), numpy.array([1, 3, 2, 3]),
    numpy.array([3, 5])
]
THESE_FUJITA_RATINGS = [
    ['f0', 'f1', 'EF2', 'EF3'], ['ef2', 'ef3', 'EF4', 'EF5'],
    ['ef4', 'ef5'], ['ef2', 'ef3', 'EF4', 'EF5'],
    ['ef4', 'ef5']
]

for k in range(len(THESE_LATE_PRIMARY_ID_STRINGS)):
    THESE_EVENT_LATITUDES_DEG[k] = THESE_EVENT_LATITUDES_DEG[k].astype(float)
    THESE_EVENT_LONGITUDES_DEG[k] = THESE_EVENT_LONGITUDES_DEG[k].astype(float)
    THESE_LINK_DIST_METRES[k] = THESE_LINK_DIST_METRES[k].astype(float)
    THESE_RELATIVE_TIMES_UNIX_SEC[k] = THESE_RELATIVE_TIMES_UNIX_SEC[k].astype(
        int)

THIS_DICT = {
    tracking_utils.PRIMARY_ID_COLUMN: THESE_LATE_PRIMARY_ID_STRINGS,
    tracking_utils.SECONDARY_ID_COLUMN: THESE_LATE_SEC_ID_STRINGS,
    tracking_utils.FULL_ID_COLUMN: THESE_LATE_FULL_ID_STRINGS,
    tracking_utils.VALID_TIME_COLUMN: THESE_LATE_TIMES_UNIX_SEC,
    linkage.EVENT_LATITUDES_COLUMN: THESE_EVENT_LATITUDES_DEG,
    linkage.EVENT_LONGITUDES_COLUMN: THESE_EVENT_LONGITUDES_DEG,
    linkage.LINKAGE_DISTANCES_COLUMN: THESE_LINK_DIST_METRES,
    linkage.RELATIVE_EVENT_TIMES_COLUMN: THESE_RELATIVE_TIMES_UNIX_SEC,
    linkage.FUJITA_RATINGS_COLUMN: THESE_FUJITA_RATINGS
}

LATE_STORM_TO_TORNADOES_TABLE_WITH_SHARING = pandas.DataFrame.from_dict(
    THIS_DICT)

STRING_COLUMNS = [
    tracking_utils.PRIMARY_ID_COLUMN, tracking_utils.SECONDARY_ID_COLUMN,
    tracking_utils.FULL_ID_COLUMN
]
NON_FLOAT_ARRAY_COLUMNS = [
    linkage.RELATIVE_EVENT_TIMES_COLUMN, linkage.FUJITA_RATINGS_COLUMN
]
FLOAT_ARRAY_COLUMNS = [
    linkage.EVENT_LATITUDES_COLUMN, linkage.EVENT_LONGITUDES_COLUMN,
    linkage.LINKAGE_DISTANCES_COLUMN
]

# The following constants are used to test find_linkage_file.
TOP_DIRECTORY_NAME = 'linkage'
FILE_TIME_UNIX_SEC = 1517523991  # 222631 1 Feb 2018
FILE_SPC_DATE_STRING = '20180201'

LINKAGE_FILE_NAME_WIND_ONE_TIME = (
    'linkage/2018/20180201/storm_to_winds_2018-02-01-222631.p'
)
LINKAGE_FILE_NAME_WIND_ONE_DATE = 'linkage/2018/storm_to_winds_20180201.p'
LINKAGE_FILE_NAME_TORNADO_ONE_TIME = (
    'linkage/2018/20180201/storm_to_tornadoes_2018-02-01-222631.p'
)
LINKAGE_FILE_NAME_TORNADO_ONE_DATE = (
    'linkage/2018/storm_to_tornadoes_20180201.p'
)


def _compare_vertex_tables(first_table, second_table):
    """Compares two tables with interpolated storm vertices.

    Such tables may be produced by either `_interp_storms_in_time` or
    `_interp_one_storm_in_time`.

    :param first_table: First table.
    :param second_table: Second table.
    :return: are_tables_equal: Boolean flag.
    """

    first_columns = list(first_table)
    second_columns = list(second_table)
    if set(first_columns) != set(second_columns):
        return False

    first_num_vertices = len(first_table.index)
    second_num_vertices = len(second_table.index)
    if first_num_vertices != second_num_vertices:
        return False

    for this_column in first_columns:
        if this_column == tracking_utils.SECONDARY_ID_COLUMN:
            if not numpy.array_equal(first_table[this_column].values,
                                     second_table[this_column].values):
                return False

        else:
            if not numpy.allclose(first_table[this_column].values,
                                  second_table[this_column].values,
                                  atol=TOLERANCE):
                return False

    return True


def _compare_storm_to_events_tables(first_table, second_table):
    """Compares two tables (pandas DataFrames) with storm-to-event linkages.

    :param first_table: First table.
    :param second_table: Second table.
    :return: are_tables_equal: Boolean flag.
    """

    first_columns = list(first_table)
    second_columns = list(second_table)
    if set(first_columns) != set(second_columns):
        return False

    if len(first_table.index) != len(second_table.index):
        return False

    string_columns = [
        tracking_utils.PRIMARY_ID_COLUMN, tracking_utils.SECONDARY_ID_COLUMN,
        tracking_utils.FULL_ID_COLUMN
    ]
    exact_array_columns = [
        linkage.RELATIVE_EVENT_TIMES_COLUMN, linkage.WIND_STATION_IDS_COLUMN,
        linkage.FUJITA_RATINGS_COLUMN
    ]

    num_rows = len(first_table.index)

    for i in range(num_rows):
        for this_column in first_columns:
            if this_column in string_columns:
                if (first_table[this_column].values[i] !=
                        second_table[this_column].values[i]):
                    return False

            elif this_column in exact_array_columns:
                if not numpy.array_equal(first_table[this_column].values[i],
                                         second_table[this_column].values[i]):
                    return False

            else:
                if not numpy.allclose(first_table[this_column].values[i],
                                      second_table[this_column].values[i],
                                      atol=TOLERANCE):
                    return False

    return True


class LinkageTests(unittest.TestCase):
    """Each method is a unit test for linkage.py."""

    def test_filter_storms_early_start_early_end(self):
        """Ensures correct output from _filter_storms_by_time.

        In this case, storms should be dropped if they start after early start
        time or end before early end time.
        """

        this_storm_object_table = linkage._filter_storms_by_time(
            storm_object_table=MAIN_STORM_OBJECT_TABLE,
            max_start_time_unix_sec=EARLY_START_TIME_UNIX_SEC,
            min_end_time_unix_sec=EARLY_END_TIME_UNIX_SEC)

        self.assertTrue(this_storm_object_table.equals(
            STORM_OBJECT_TABLE_EARLY_START_EARLY_END
        ))

    def test_filter_storms_early_start_late_end(self):
        """Ensures correct output from _filter_storms_by_time.

        In this case, storms should be dropped if they start after early start
        time or end before late end time.
        """

        this_storm_object_table = linkage._filter_storms_by_time(
            storm_object_table=MAIN_STORM_OBJECT_TABLE,
            max_start_time_unix_sec=EARLY_START_TIME_UNIX_SEC,
            min_end_time_unix_sec=LATE_END_TIME_UNIX_SEC)

        self.assertTrue(this_storm_object_table.equals(
            STORM_OBJECT_TABLE_EARLY_START_LATE_END
        ))

    def test_filter_storms_late_start_early_end(self):
        """Ensures correct output from _filter_storms_by_time.

        In this case, storms should be dropped if they start after late start
        time or end before early end time.
        """

        this_storm_object_table = linkage._filter_storms_by_time(
            storm_object_table=MAIN_STORM_OBJECT_TABLE,
            max_start_time_unix_sec=LATE_START_TIME_UNIX_SEC,
            min_end_time_unix_sec=EARLY_END_TIME_UNIX_SEC)

        self.assertTrue(this_storm_object_table.equals(
            STORM_OBJECT_TABLE_LATE_START_EARLY_END
        ))

    def test_filter_storms_late_start_late_end(self):
        """Ensures correct output from _filter_storms_by_time.

        In this case, storms should be dropped if they start after late start
        time or end before late end time.
        """

        this_storm_object_table = linkage._filter_storms_by_time(
            storm_object_table=MAIN_STORM_OBJECT_TABLE,
            max_start_time_unix_sec=LATE_START_TIME_UNIX_SEC,
            min_end_time_unix_sec=LATE_END_TIME_UNIX_SEC)

        self.assertTrue(this_storm_object_table.equals(
            STORM_OBJECT_TABLE_LATE_START_LATE_END
        ))

    def test_interp_one_storm_in_time_interp(self):
        """Ensures correct output from _interp_one_storm_in_time.

        In this case the method is doing strict interpolation, not
        extrapolation.
        """

        this_vertex_table = linkage._interp_one_storm_in_time(
            storm_object_table_1cell=STORM_OBJECT_TABLE_1CELL,
            secondary_id_string=STORM_OBJECT_TABLE_1CELL[
                tracking_utils.SECONDARY_ID_COLUMN].values[0],
            target_time_unix_sec=INTERP_TIME_1CELL_UNIX_SEC)

        self.assertTrue(_compare_vertex_tables(
            this_vertex_table, VERTEX_TABLE_1OBJECT_INTERP
        ))

    def test_interp_one_storm_in_time_extrap(self):
        """Ensures correct output from _interp_one_storm_in_time.

        In this case the method is doing extrapolation, not interpolation.
        """

        this_vertex_table = linkage._interp_one_storm_in_time(
            storm_object_table_1cell=STORM_OBJECT_TABLE_1CELL,
            secondary_id_string=STORM_OBJECT_TABLE_1CELL[
                tracking_utils.SECONDARY_ID_COLUMN].values[0],
            target_time_unix_sec=EXTRAP_TIME_1CELL_UNIX_SEC)

        self.assertTrue(_compare_vertex_tables(
            this_vertex_table, VERTEX_TABLE_1OBJECT_EXTRAP
        ))

    def test_get_bounding_box_for_storms(self):
        """Ensures correct output from _get_bounding_box_for_storms."""

        these_x_limits_metres, these_y_limits_metres = (
            linkage._get_bounding_box_for_storms(
                storm_object_table=MAIN_STORM_OBJECT_TABLE,
                padding_metres=BOUNDING_BOX_PADDING_METRES)
        )

        self.assertTrue(numpy.allclose(
            these_x_limits_metres, BOUNDING_BOX_X_LIMITS_METRES, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            these_y_limits_metres, BOUNDING_BOX_Y_LIMITS_METRES, atol=TOLERANCE
        ))

    def test_filter_events_by_bounding_box(self):
        """Ensures correct output from _filter_events_by_bounding_box."""

        this_event_table = linkage._filter_events_by_bounding_box(
            event_table=EVENT_TABLE_FULL_DOMAIN,
            x_limits_metres=BOUNDING_BOX_X_LIMITS_METRES,
            y_limits_metres=BOUNDING_BOX_Y_LIMITS_METRES)

        self.assertTrue(this_event_table.equals(EVENT_TABLE_IN_BOUNDING_BOX))

    def test_interp_storms_in_time(self):
        """Ensures correct output from _interp_storms_in_time."""

        this_vertex_table = linkage._interp_storms_in_time(
            storm_object_table=MAIN_STORM_OBJECT_TABLE,
            target_time_unix_sec=INTERP_TIME_UNIX_SEC,
            max_time_before_start_sec=10, max_time_after_end_sec=10)

        print(INTERP_VERTEX_TABLE)
        print('\n\n\n')
        print(this_vertex_table)

        self.assertTrue(_compare_vertex_tables(
            this_vertex_table, INTERP_VERTEX_TABLE
        ))

    # def test_find_nearest_storms_one_time(self):
    #     """Ensures correct output from _find_nearest_storms_one_time."""
    #
    #     these_nearest_id_strings, these_link_distances_metres = (
    #         linkage._find_nearest_storms_one_time(
    #             interp_vertex_table=INTERP_VERTEX_TABLE_2OBJECTS,
    #             event_x_coords_metres=EVENT_X_COORDS_1TIME_METRES,
    #             event_y_coords_metres=EVENT_Y_COORDS_1TIME_METRES,
    #             max_link_distance_metres=MAX_LINK_DISTANCE_METRES)
    #     )
    #
    #     self.assertTrue(
    #         these_nearest_id_strings == NEAREST_PRIMARY_ID_STRINGS_1TIME
    #     )
    #     self.assertTrue(numpy.allclose(
    #         these_link_distances_metres, LINK_DISTANCES_1TIME_METRES,
    #         equal_nan=True, atol=TOLERANCE
    #     ))
    #
    # def test_find_nearest_storms(self):
    #     """Ensures correct output from _find_nearest_storms."""
    #
    #     this_wind_to_storm_table = linkage._find_nearest_storms(
    #         storm_object_table=STORM_OBJECT_TABLE_2CELLS,
    #         event_table=EVENT_TABLE_2TIMES,
    #         max_time_before_storm_start_sec=MAX_TIME_BEFORE_STORM_START_SEC,
    #         max_time_after_storm_end_sec=MAX_TIME_AFTER_STORM_END_SEC,
    #         max_link_distance_metres=MAX_LINK_DISTANCE_METRES,
    #         interp_time_resolution_sec=INTERP_TIME_RESOLUTION_SEC)
    #
    #     self.assertTrue(this_wind_to_storm_table.equals(
    #         EVENT_TO_STORM_TABLE_SIMPLE
    #     ))
    #
    # def test_reverse_wind_linkages(self):
    #     """Ensures correct output from _reverse_wind_linkages."""
    #
    #     this_storm_to_winds_table = linkage._reverse_wind_linkages(
    #         storm_object_table=STORM_OBJECT_TABLE_2CELLS,
    #         wind_to_storm_table=WIND_TO_STORM_TABLE)
    #
    #     self.assertTrue(_compare_storm_to_events_tables(
    #         this_storm_to_winds_table, STORM_TO_WINDS_TABLE
    #     ))

    def test_remove_storms_near_start_of_period(self):
        """Ensures correct output from _remove_storms_near_start_of_period."""

        this_storm_object_table = linkage._remove_storms_near_start_of_period(
            storm_object_table=copy.deepcopy(
                STORM_OBJECT_TABLE_WITH_PERIOD_START),
            min_time_elapsed_sec=MIN_TIME_SINCE_START_SEC)

        expected_columns = list(STORM_OBJECT_TABLE_SANS_PERIOD_START)
        actual_columns = list(this_storm_object_table)
        self.assertTrue(set(expected_columns) == set(actual_columns))

        this_storm_object_table.reset_index(inplace=True)

        self.assertTrue(this_storm_object_table[actual_columns].equals(
            STORM_OBJECT_TABLE_SANS_PERIOD_START[actual_columns]
        ))

    def test_share_linkages_between_periods(self):
        """Ensures correct output from _share_linkages_between_periods."""

        this_early_table, this_late_table = (
            linkage._share_linkages_between_periods(
                early_storm_to_events_table=copy.deepcopy(
                    EARLY_STORM_TO_TORNADOES_TABLE_SANS_SHARING),
                late_storm_to_events_table=copy.deepcopy(
                    LATE_STORM_TO_TORNADOES_TABLE_SANS_SHARING)
            )
        )

        self.assertTrue(_compare_storm_to_events_tables(
            this_early_table, EARLY_STORM_TO_TORNADOES_TABLE_WITH_SHARING
        ))
        self.assertTrue(_compare_storm_to_events_tables(
            this_late_table, LATE_STORM_TO_TORNADOES_TABLE_WITH_SHARING
        ))

    def test_find_linkage_file_wind_one_time(self):
        """Ensures correct output from find_linkage_file.

        In this case, file contains wind linkages for one time.
        """

        this_file_name = linkage.find_linkage_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            event_type_string=linkage.WIND_EVENT_STRING,
            raise_error_if_missing=False, unix_time_sec=FILE_TIME_UNIX_SEC,
            spc_date_string=FILE_SPC_DATE_STRING)

        self.assertTrue(this_file_name == LINKAGE_FILE_NAME_WIND_ONE_TIME)

    def test_find_linkage_file_wind_one_spc_date(self):
        """Ensures correct output from find_linkage_file.

        In this case, file contains wind linkages for one SPC date.
        """

        this_file_name = linkage.find_linkage_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            event_type_string=linkage.WIND_EVENT_STRING,
            raise_error_if_missing=False, unix_time_sec=None,
            spc_date_string=FILE_SPC_DATE_STRING)

        self.assertTrue(this_file_name == LINKAGE_FILE_NAME_WIND_ONE_DATE)

    def test_find_linkage_file_tornado_one_time(self):
        """Ensures correct output from find_linkage_file.

        In this case, file contains tornado linkages for one time.
        """

        this_file_name = linkage.find_linkage_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            event_type_string=linkage.TORNADO_EVENT_STRING,
            raise_error_if_missing=False, unix_time_sec=FILE_TIME_UNIX_SEC,
            spc_date_string=FILE_SPC_DATE_STRING)

        self.assertTrue(this_file_name == LINKAGE_FILE_NAME_TORNADO_ONE_TIME)

    def test_find_linkage_file_tornado_one_spc_date(self):
        """Ensures correct output from find_linkage_file.

        In this case, file contains tornado linkages for one SPC date.
        """

        this_file_name = linkage.find_linkage_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            event_type_string=linkage.TORNADO_EVENT_STRING,
            raise_error_if_missing=False, unix_time_sec=None,
            spc_date_string=FILE_SPC_DATE_STRING)

        self.assertTrue(this_file_name == LINKAGE_FILE_NAME_TORNADO_ONE_DATE)


if __name__ == '__main__':
    unittest.main()
