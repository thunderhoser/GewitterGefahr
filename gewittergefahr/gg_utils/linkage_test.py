"""Unit tests for linkage.py."""

import copy
import unittest
import numpy
import pandas
from gewittergefahr.gg_io import tornado_io
from gewittergefahr.gg_io import raw_wind_io
from gewittergefahr.gg_utils import linkage
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils

TOLERANCE = 1e-6
LARGE_INTEGER = int(1e10)

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
MAX_TOY_EXTRAP_TIME_SEC = 10

THESE_CENTROID_ID_STRINGS = [
    'A1', 'A2', 'A3', 'A4', 'A4', 'B1', 'B2', 'B3', 'B4', 'B5'
]
THESE_CENTROID_X_COORDS = numpy.array(
    [241.5, 241.5, 241.5, 241.5, 241.5, 273, 273, 273, 273, 273]
)
THESE_CENTROID_Y_COORDS = numpy.array(
    [50, 49.5, 50.5, 50, 50, 59.75, 60.5, 60.25, 60.25, 59.833333]
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
MAX_LINK_DISTANCE_METRES = 12.
MAX_WIND_EXTRAP_TIME_SEC = 10

WIND_X_COORDS_1TIME_METRES = numpy.array(
    [231.5, 241.4, 231.5, 241.5, 251.5, 251.5, 251.5, 241.5]
)
WIND_Y_COORDS_1TIME_METRES = numpy.array(
    [40, 50, 60, 60, 60, 50, 40, 40], dtype=float
)

NEAREST_SEC_ID_STRINGS_1TIME = [None, 'A1', None, 'A3', None, 'A1', None, 'A2']

THIS_SHORT_DISTANCE_METRES = numpy.sqrt(9.25 ** 2 + 0.25 ** 2)
THIS_LONG_DISTANCE_METRES = numpy.sqrt(9.75 ** 2 + 0.25 ** 2)

LINK_DISTANCES_1TIME_METRES = numpy.array([
    numpy.nan, 0, numpy.nan, THIS_SHORT_DISTANCE_METRES,
    numpy.nan, THIS_LONG_DISTANCE_METRES, numpy.nan, THIS_SHORT_DISTANCE_METRES
])

# The following constants are used to test _find_nearest_storms for wind.
INTERP_TIME_INTERVAL_SEC = 1

THESE_X_METRES = numpy.array([
    231.5, 241.4, 231.5, 241.5, 251.5, 251.5, 251.5, 241.5,
    267, 267, 267, 277, 287, 287, 287, 277
])

THESE_Y_METRES = numpy.array([
    40, 50, 60, 60, 60, 50, 40, 40,
    50, 60, 70, 70, 70, 60, 50, 50
], dtype=float)

THESE_TIMES_UNIX_SEC = numpy.array([
    3, 3, 3, 3, 3, 3, 3, 3,
    7, 7, 7, 7, 7, 7, 7, 7
], dtype=float)

WIND_TABLE = pandas.DataFrame.from_dict({
    linkage.EVENT_X_COLUMN: THESE_X_METRES,
    linkage.EVENT_Y_COLUMN: THESE_Y_METRES,
    linkage.EVENT_LONGITUDE_COLUMN: THESE_X_METRES,
    linkage.EVENT_LATITUDE_COLUMN: THESE_Y_METRES,
    linkage.EVENT_TIME_COLUMN: THESE_TIMES_UNIX_SEC
})

THIS_TINY_DISTANCE_METRES = numpy.sqrt(7.25 ** 2 + 0.25 ** 2)

THESE_SECONDARY_ID_STRINGS = [
    None, 'A1', None, 'A3', None, 'A1', None, 'A2',
    None, 'B4', None, 'B2', None, 'B4', None, 'B3'
]

THESE_TIMES_UNIX_SEC = numpy.array([
    -1, 3, -1, 3, -1, 3, -1, 3,
    -1, 7, -1, 7, -1, 7, -1, 7
], dtype=int)

THESE_DISTANCES_METRES = numpy.array([
    numpy.nan, 0, numpy.nan, THIS_SHORT_DISTANCE_METRES,
    numpy.nan, THIS_LONG_DISTANCE_METRES, numpy.nan, THIS_SHORT_DISTANCE_METRES,
    numpy.nan, THIS_LONG_DISTANCE_METRES, numpy.nan, THIS_TINY_DISTANCE_METRES,
    numpy.nan, THIS_LONG_DISTANCE_METRES, numpy.nan, THIS_TINY_DISTANCE_METRES
])

WIND_TO_STORM_TABLE = WIND_TABLE.assign(**{
    linkage.NEAREST_SECONDARY_ID_COLUMN: THESE_SECONDARY_ID_STRINGS,
    linkage.NEAREST_TIME_COLUMN: THESE_TIMES_UNIX_SEC,
    linkage.LINKAGE_DISTANCE_COLUMN: THESE_DISTANCES_METRES
})

# The following constants are used to test _interp_tornadoes_along_tracks.
THESE_START_TIMES_UNIX_SEC = numpy.array([1, 5, 5, 5, 60], dtype=int)
THESE_END_TIMES_UNIX_SEC = numpy.array([11, 10, 12, 14, 60], dtype=int)

THESE_START_LATITUDES_DEG = numpy.array([59.5, 61, 51, 49, 89])
THESE_END_LATITUDES_DEG = numpy.array([59.5, 66, 58, 58, 89])

THESE_START_LONGITUDES_DEG = numpy.array([271, 275, 242.5, 242.5, 300])
THESE_END_LONGITUDES_DEG = numpy.array([281, 275, 242.5, 242.5, 300])
THESE_FUJITA_STRINGS = ['EF1', 'EF2', 'EF3', 'EF4', 'EF5']

TORNADO_TABLE_BEFORE_INTERP = pandas.DataFrame.from_dict({
    tornado_io.START_TIME_COLUMN: THESE_START_TIMES_UNIX_SEC,
    tornado_io.END_TIME_COLUMN: THESE_END_TIMES_UNIX_SEC,
    tornado_io.START_LAT_COLUMN: THESE_START_LATITUDES_DEG,
    tornado_io.END_LAT_COLUMN: THESE_END_LATITUDES_DEG,
    tornado_io.START_LNG_COLUMN: THESE_START_LONGITUDES_DEG,
    tornado_io.END_LNG_COLUMN: THESE_END_LONGITUDES_DEG,
    tornado_io.FUJITA_RATING_COLUMN: THESE_FUJITA_STRINGS
})

TORNADO_INTERP_TIME_INTERVAL_SEC = 1

THESE_TIMES_UNIX_SEC = numpy.array([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    5, 6, 7, 8, 9, 10,
    5, 6, 7, 8, 9, 10, 11, 12,
    5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
    60
], dtype=int)

THESE_LATITUDES_DEG = numpy.array([
    59.5, 59.5, 59.5, 59.5, 59.5, 59.5, 59.5, 59.5, 59.5, 59.5, 59.5,
    61, 62, 63, 64, 65, 66,
    51, 52, 53, 54, 55, 56, 57, 58,
    49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    89
])

THESE_LONGITUDES_DEG = numpy.array([
    271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281,
    275, 275, 275, 275, 275, 275,
    242.5, 242.5, 242.5, 242.5, 242.5, 242.5, 242.5, 242.5,
    242.5, 242.5, 242.5, 242.5, 242.5, 242.5, 242.5, 242.5, 242.5, 242.5,
    300
])

THESE_UNIQUE_ID_STRINGS = [
    tornado_io.create_tornado_id(
        start_time_unix_sec=t, start_latitude_deg=y, start_longitude_deg=x
    ) for t, x, y in
    zip(THESE_START_TIMES_UNIX_SEC, THESE_START_LONGITUDES_DEG,
        THESE_START_LATITUDES_DEG)
]

THESE_ID_STRINGS = (
    [THESE_UNIQUE_ID_STRINGS[0]] * 11 + [THESE_UNIQUE_ID_STRINGS[1]] * 6 +
    [THESE_UNIQUE_ID_STRINGS[2]] * 8 + [THESE_UNIQUE_ID_STRINGS[3]] * 10 +
    [THESE_UNIQUE_ID_STRINGS[4]] * 1
)

THESE_FUJITA_STRINGS = (
    [THESE_FUJITA_STRINGS[0]] * 11 + [THESE_FUJITA_STRINGS[1]] * 6 +
    [THESE_FUJITA_STRINGS[2]] * 8 + [THESE_FUJITA_STRINGS[3]] * 10 +
    [THESE_FUJITA_STRINGS[4]] * 1
)

TORNADO_TABLE_AFTER_INTERP = pandas.DataFrame.from_dict({
    linkage.EVENT_TIME_COLUMN: THESE_TIMES_UNIX_SEC,
    linkage.EVENT_LATITUDE_COLUMN: THESE_LATITUDES_DEG,
    linkage.EVENT_LONGITUDE_COLUMN: THESE_LONGITUDES_DEG,
    linkage.EVENT_Y_COLUMN: THESE_LATITUDES_DEG,
    linkage.EVENT_X_COLUMN: THESE_LONGITUDES_DEG,
    tornado_io.TORNADO_ID_COLUMN: THESE_ID_STRINGS,
    tornado_io.FUJITA_RATING_COLUMN: THESE_FUJITA_STRINGS
})

# The following constants are used to test _find_nearest_storms for tornado
# occurrence.
MAX_TORNADO_EXTRAP_TIME_SEC = 2

THESE_SECONDARY_ID_STRINGS = [
    'B2', 'B3', 'B5', 'B5', 'B5', 'B5', 'B5', 'B5', None, None, None,
    'B4', 'B4', 'B4', 'B4', None, None,
    'A2', 'A2', 'A2', 'A2', 'A4', 'A4', 'A4', 'A4',
    'A3', 'A3', 'A3', 'A3', 'A4', 'A4', 'A4', 'A4', 'A4', None,
    None
]

# THESE_TIMES_UNIX_SEC = numpy.array([
#     1, 1, 3, 3, 3, 3, 7, 8, -1, -1, -1,
#     5, 5, 7, 8, -1, -1,
#     5, 5, 5, 5, 5, 5, 5, 12,
#     5, 5, 5, 5, 5, 5, 5, 12, 13, -1,
#     -1
# ], dtype=int)

THESE_TIMES_UNIX_SEC = numpy.array([
    1, 2, 3, 4, 5, 6, 7, 8, -1, -1, -1,
    5, 6, 7, 8, -1, -1,
    5, 6, 7, 8, 9, 10, 11, 12,
    5, 6, 7, 8, 9, 10, 11, 12, 13, -1,
    -1
], dtype=int)

THESE_DISTANCES_METRES = numpy.array([
    0, 0, 0, 0, 0, 0, 0, 0, numpy.nan, numpy.nan, numpy.nan,
    0.5, 0.5, 0.5, 0.5, numpy.nan, numpy.nan,
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, numpy.nan,
    numpy.nan
])

TORNADO_TO_STORM_TABLE = TORNADO_TABLE_AFTER_INTERP.assign(**{
    linkage.NEAREST_SECONDARY_ID_COLUMN: THESE_SECONDARY_ID_STRINGS,
    linkage.NEAREST_TIME_COLUMN: THESE_TIMES_UNIX_SEC,
    linkage.LINKAGE_DISTANCE_COLUMN: THESE_DISTANCES_METRES
})

# The following constants are used to test _reverse_tornado_linkages.
STORM_TO_TORNADOES_TABLE = copy.deepcopy(MAIN_STORM_OBJECT_TABLE)

THIS_NESTED_ARRAY = STORM_TO_TORNADOES_TABLE[[
    tracking_utils.VALID_TIME_COLUMN, tracking_utils.VALID_TIME_COLUMN
]].values.tolist()

STORM_TO_TORNADOES_TABLE = STORM_TO_TORNADOES_TABLE.assign(**{
    linkage.EVENT_LATITUDES_COLUMN: THIS_NESTED_ARRAY,
    linkage.EVENT_LONGITUDES_COLUMN: THIS_NESTED_ARRAY,
    linkage.FUJITA_RATINGS_COLUMN: THIS_NESTED_ARRAY,
    linkage.TORNADO_IDS_COLUMN: THIS_NESTED_ARRAY,
    linkage.LINKAGE_DISTANCES_COLUMN: THIS_NESTED_ARRAY,
    linkage.RELATIVE_EVENT_TIMES_COLUMN: THIS_NESTED_ARRAY,
    linkage.MAIN_OBJECT_FLAGS_COLUMN: THIS_NESTED_ARRAY
})

for k in range(len(STORM_TO_TORNADOES_TABLE.index)):
    STORM_TO_TORNADOES_TABLE[linkage.EVENT_LATITUDES_COLUMN].values[k] = (
        numpy.array([])
    )
    STORM_TO_TORNADOES_TABLE[linkage.EVENT_LONGITUDES_COLUMN].values[k] = (
        numpy.array([])
    )
    STORM_TO_TORNADOES_TABLE[linkage.FUJITA_RATINGS_COLUMN].values[k] = []
    STORM_TO_TORNADOES_TABLE[linkage.TORNADO_IDS_COLUMN].values[k] = []
    STORM_TO_TORNADOES_TABLE[linkage.LINKAGE_DISTANCES_COLUMN].values[k] = (
        numpy.array([])
    )
    STORM_TO_TORNADOES_TABLE[linkage.RELATIVE_EVENT_TIMES_COLUMN].values[k] = (
        numpy.array([], dtype=int)
    )
    STORM_TO_TORNADOES_TABLE[linkage.MAIN_OBJECT_FLAGS_COLUMN].values[k] = (
        numpy.array([], dtype=bool)
    )

# THESE_TIMES_UNIX_SEC = numpy.array([
#     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,  # ends at row 10
#     5, 6, 7, 8, 9, 10,                  # ends at row 16
#     5, 6, 7, 8, 9, 10, 11, 12,          # ends at row 24
#     5, 6, 7, 8, 9, 10, 11, 12, 13, 14,  # ends at row 34
#     60                                  # ends at row 35
# ], dtype=int)

STORM_ROW_TO_TORNADO_ROWS = {
    0: [17, 18, 19, 20, 25, 26, 27, 28],
    1: [1],
    2: [0, 1],
    3: [17, 18, 19, 20, 25, 26, 27, 28],
    4: [0, 1],
    5: [17, 18, 19, 20, 25, 26, 27, 28],
    6: [1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14],
    7: [17, 18, 19, 20, 25, 26, 27, 28],
    8: [11, 12, 13, 14],
    9: [17, 18, 19, 20, 21, 22, 23, 24, 29, 30, 31, 32, 33],
    10: [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
    11: [11, 12, 13, 14],
    12: [4, 5, 6, 7],
    13: [18, 19, 20, 21, 22, 23, 24, 29, 30, 31, 32, 33],
    14: [12, 13, 14],
    15: [5, 6, 7],
    16: [19, 20, 21, 22, 23, 24, 29, 30, 31, 32, 33],
    17: [21, 22, 23, 24, 27, 28, 29, 30, 31, 32, 33],
    18: [22, 23, 24, 30, 31, 32, 33],
    19: [23, 24, 31, 32, 33]
}

STORM_ROW_TO_MAIN_OBJECT_FLAGS = {
    0: [0, 0, 0, 0, 0, 0, 0, 0],
    1: [0],
    2: [0, 0],
    3: [0, 0, 0, 0, 0, 0, 0, 0],
    4: [1, 0],
    5: [0, 0, 0, 0, 0, 0, 0, 0],
    6: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    7: [0, 0, 0, 0, 0, 0, 0, 0],
    8: [0, 0, 0, 0],
    9: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    10: [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    11: [1, 0, 0, 0],
    12: [1, 0, 0, 0],
    13: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    14: [1, 1, 1],
    15: [1, 1, 1],
    16: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    17: [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    18: [1, 0, 0, 1, 0, 0, 0],
    19: [1, 1, 1, 1, 1]
}

for this_storm_row in STORM_ROW_TO_TORNADO_ROWS:
    these_tornado_rows = numpy.array(
        STORM_ROW_TO_TORNADO_ROWS[this_storm_row], dtype=int
    )

    STORM_TO_TORNADOES_TABLE[linkage.EVENT_LATITUDES_COLUMN].values[
        this_storm_row
    ] = TORNADO_TO_STORM_TABLE[linkage.EVENT_LATITUDE_COLUMN].values[
        these_tornado_rows]

    STORM_TO_TORNADOES_TABLE[linkage.EVENT_LONGITUDES_COLUMN].values[
        this_storm_row
    ] = TORNADO_TO_STORM_TABLE[linkage.EVENT_LONGITUDE_COLUMN].values[
        these_tornado_rows]

    STORM_TO_TORNADOES_TABLE[linkage.LINKAGE_DISTANCES_COLUMN].values[
        this_storm_row
    ] = TORNADO_TO_STORM_TABLE[linkage.LINKAGE_DISTANCE_COLUMN].values[
        these_tornado_rows]

    STORM_TO_TORNADOES_TABLE[linkage.FUJITA_RATINGS_COLUMN].values[
        this_storm_row
    ] = TORNADO_TO_STORM_TABLE[tornado_io.FUJITA_RATING_COLUMN].values[
        these_tornado_rows
    ].tolist()

    STORM_TO_TORNADOES_TABLE[linkage.TORNADO_IDS_COLUMN].values[
        this_storm_row
    ] = TORNADO_TO_STORM_TABLE[tornado_io.TORNADO_ID_COLUMN].values[
        these_tornado_rows
    ].tolist()

    these_relative_times_sec = (
        TORNADO_TO_STORM_TABLE[linkage.EVENT_TIME_COLUMN].values[
            these_tornado_rows] -
        STORM_TO_TORNADOES_TABLE[tracking_utils.VALID_TIME_COLUMN].values[
            this_storm_row]
    )

    STORM_TO_TORNADOES_TABLE[linkage.RELATIVE_EVENT_TIMES_COLUMN].values[
        this_storm_row
    ] = these_relative_times_sec

    these_main_object_flags = numpy.array(
        STORM_ROW_TO_MAIN_OBJECT_FLAGS[this_storm_row], dtype=bool
    )

    STORM_TO_TORNADOES_TABLE[linkage.MAIN_OBJECT_FLAGS_COLUMN].values[
        this_storm_row
    ] = these_main_object_flags

# The following constants are used to test _reverse_wind_linkages.
THESE_STATION_ID_STRINGS = [
    'i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii',
    'ix', 'x', 'xi', 'xii', 'xiii', 'xiv', 'xv', 'xvi'
]

THESE_U_WINDS_M_S01 = THESE_X_METRES + 0.
THESE_V_WINDS_M_S01 = THESE_Y_METRES + 0.

WIND_TO_STORM_TABLE = WIND_TO_STORM_TABLE.assign(**{
    raw_wind_io.STATION_ID_COLUMN: THESE_STATION_ID_STRINGS,
    raw_wind_io.U_WIND_COLUMN: THESE_U_WINDS_M_S01,
    raw_wind_io.V_WIND_COLUMN: THESE_V_WINDS_M_S01
})

STORM_TO_WINDS_TABLE = copy.deepcopy(MAIN_STORM_OBJECT_TABLE)

THIS_NESTED_ARRAY = STORM_TO_WINDS_TABLE[[
    tracking_utils.VALID_TIME_COLUMN, tracking_utils.VALID_TIME_COLUMN
]].values.tolist()

STORM_TO_WINDS_TABLE = STORM_TO_WINDS_TABLE.assign(**{
    linkage.WIND_STATION_IDS_COLUMN: THIS_NESTED_ARRAY,
    linkage.EVENT_LATITUDES_COLUMN: THIS_NESTED_ARRAY,
    linkage.EVENT_LONGITUDES_COLUMN: THIS_NESTED_ARRAY,
    linkage.U_WINDS_COLUMN: THIS_NESTED_ARRAY,
    linkage.V_WINDS_COLUMN: THIS_NESTED_ARRAY,
    linkage.LINKAGE_DISTANCES_COLUMN: THIS_NESTED_ARRAY,
    linkage.RELATIVE_EVENT_TIMES_COLUMN: THIS_NESTED_ARRAY,
    linkage.MAIN_OBJECT_FLAGS_COLUMN: THIS_NESTED_ARRAY
})

for k in range(len(STORM_TO_WINDS_TABLE.index)):
    STORM_TO_WINDS_TABLE[linkage.WIND_STATION_IDS_COLUMN].values[k] = []
    STORM_TO_WINDS_TABLE[linkage.EVENT_LATITUDES_COLUMN].values[k] = (
        numpy.array([])
    )
    STORM_TO_WINDS_TABLE[linkage.EVENT_LONGITUDES_COLUMN].values[k] = (
        numpy.array([])
    )
    STORM_TO_WINDS_TABLE[linkage.U_WINDS_COLUMN].values[k] = numpy.array([])
    STORM_TO_WINDS_TABLE[linkage.V_WINDS_COLUMN].values[k] = numpy.array([])
    STORM_TO_WINDS_TABLE[linkage.LINKAGE_DISTANCES_COLUMN].values[k] = (
        numpy.array([])
    )
    STORM_TO_WINDS_TABLE[linkage.RELATIVE_EVENT_TIMES_COLUMN].values[k] = (
        numpy.array([], dtype=int)
    )
    STORM_TO_WINDS_TABLE[linkage.MAIN_OBJECT_FLAGS_COLUMN].values[k] = (
        numpy.array([], dtype=bool)
    )

STORM_ROW_TO_STATION_ID_STRINGS = {
    0: ['ii', 'iv', 'vi', 'viii'],
    1: ['xvi'],
    2: ['xii', 'xvi'],
    3: ['ii', 'iv', 'vi', 'viii'],
    4: ['xii', 'xvi'],
    5: ['ii', 'iv', 'vi', 'viii'],
    6: ['x', 'xiv', 'xvi'],
    8: ['x', 'xiv'],
    11: ['x', 'xiv'],
    14: ['x', 'xiv']
}

STORM_ROW_TO_MAIN_OBJECT_FLAGS = {
    0: [0, 0, 0, 0],
    1: [0],
    2: [0, 0],
    3: [0, 0, 0, 0],
    4: [1, 0],
    5: [1, 0, 1, 0],
    6: [0, 0, 1],
    8: [0, 0],
    11: [0, 0],
    14: [1, 1]
}

for this_storm_row in STORM_ROW_TO_STATION_ID_STRINGS:
    these_station_id_strings = STORM_ROW_TO_STATION_ID_STRINGS[this_storm_row]
    these_main_object_flags = numpy.array(
        STORM_ROW_TO_MAIN_OBJECT_FLAGS[this_storm_row], dtype=bool
    )

    these_event_rows = numpy.array([
        WIND_TO_STORM_TABLE[
            raw_wind_io.STATION_ID_COLUMN].values.tolist().index(s)
        for s in these_station_id_strings
    ], dtype=int)

    STORM_TO_WINDS_TABLE[linkage.WIND_STATION_IDS_COLUMN].values[
        this_storm_row
    ] = these_station_id_strings

    STORM_TO_WINDS_TABLE[linkage.EVENT_LATITUDES_COLUMN].values[
        this_storm_row
    ] = WIND_TO_STORM_TABLE[linkage.EVENT_LATITUDE_COLUMN].values[
        these_event_rows]

    STORM_TO_WINDS_TABLE[linkage.EVENT_LONGITUDES_COLUMN].values[
        this_storm_row
    ] = WIND_TO_STORM_TABLE[linkage.EVENT_LONGITUDE_COLUMN].values[
        these_event_rows]

    STORM_TO_WINDS_TABLE[linkage.U_WINDS_COLUMN].values[this_storm_row] = (
        WIND_TO_STORM_TABLE[raw_wind_io.U_WIND_COLUMN].values[these_event_rows]
    )

    STORM_TO_WINDS_TABLE[linkage.V_WINDS_COLUMN].values[this_storm_row] = (
        WIND_TO_STORM_TABLE[raw_wind_io.V_WIND_COLUMN].values[these_event_rows]
    )

    STORM_TO_WINDS_TABLE[linkage.LINKAGE_DISTANCES_COLUMN].values[
        this_storm_row
    ] = WIND_TO_STORM_TABLE[linkage.LINKAGE_DISTANCE_COLUMN].values[
        these_event_rows]

    these_relative_times_sec = (
        WIND_TO_STORM_TABLE[linkage.EVENT_TIME_COLUMN].values[
            these_event_rows] -
        STORM_TO_WINDS_TABLE[tracking_utils.VALID_TIME_COLUMN].values[
            this_storm_row]
    )

    STORM_TO_WINDS_TABLE[linkage.RELATIVE_EVENT_TIMES_COLUMN].values[
        this_storm_row
    ] = these_relative_times_sec

    STORM_TO_WINDS_TABLE[linkage.MAIN_OBJECT_FLAGS_COLUMN].values[
        this_storm_row
    ] = these_main_object_flags

ROW_TO_EARLY_KEEP_FLAGS = {
    0: [1, 1, 1, 1],
    1: [1],
    2: [1, 1],
    3: [1, 1, 1, 1],
    4: [1, 1],
    5: [1, 1, 1, 1],
    6: [0, 0, 1],
    8: [0, 0],
    11: [0, 0],
    14: [0, 0]
}

ROW_TO_LATE_KEEP_FLAGS = {
    0: [0, 0, 0, 0],
    1: [0],
    2: [0, 0],
    3: [0, 0, 0, 0],
    4: [0, 0],
    5: [0, 0, 0, 0],
    6: [1, 1, 0],
    8: [1, 1],
    11: [1, 1],
    14: [1, 1]
}

# The following constants are used to test _share_linkages_between_periods.
EARLY_STORM_TO_WINDS_TABLE_PRELIM = copy.deepcopy(STORM_TO_WINDS_TABLE)
LATE_STORM_TO_WINDS_TABLE_PRELIM = copy.deepcopy(STORM_TO_WINDS_TABLE)

LINKAGE_ARRAY_COLUMNS = [
    linkage.EVENT_LATITUDES_COLUMN, linkage.EVENT_LONGITUDES_COLUMN,
    linkage.U_WINDS_COLUMN, linkage.V_WINDS_COLUMN,
    linkage.LINKAGE_DISTANCES_COLUMN, linkage.RELATIVE_EVENT_TIMES_COLUMN,
    linkage.MAIN_OBJECT_FLAGS_COLUMN
]

LINKAGE_LIST_COLUMNS = [linkage.WIND_STATION_IDS_COLUMN]

for this_storm_row in ROW_TO_EARLY_KEEP_FLAGS:
    these_indices = numpy.where(numpy.array(
        ROW_TO_EARLY_KEEP_FLAGS[this_storm_row], dtype=bool
    ))[0]

    for this_column in LINKAGE_ARRAY_COLUMNS:
        EARLY_STORM_TO_WINDS_TABLE_PRELIM[this_column].values[
            this_storm_row
        ] = (
            EARLY_STORM_TO_WINDS_TABLE_PRELIM[
                this_column].values[this_storm_row][these_indices]
        )

    for this_column in LINKAGE_LIST_COLUMNS:
        EARLY_STORM_TO_WINDS_TABLE_PRELIM[this_column].values[
            this_storm_row
        ] = [
            EARLY_STORM_TO_WINDS_TABLE_PRELIM[
                this_column].values[this_storm_row][m]
            for m in these_indices
        ]

for this_storm_row in ROW_TO_LATE_KEEP_FLAGS:
    these_indices = numpy.where(numpy.array(
        ROW_TO_LATE_KEEP_FLAGS[this_storm_row], dtype=bool
    ))[0]

    for this_column in LINKAGE_ARRAY_COLUMNS:
        LATE_STORM_TO_WINDS_TABLE_PRELIM[this_column].values[this_storm_row] = (
            LATE_STORM_TO_WINDS_TABLE_PRELIM[
                this_column].values[this_storm_row][these_indices]
        )

    for this_column in LINKAGE_LIST_COLUMNS:
        LATE_STORM_TO_WINDS_TABLE_PRELIM[this_column].values[this_storm_row] = [
            LATE_STORM_TO_WINDS_TABLE_PRELIM[
                this_column].values[this_storm_row][m]
            for m in these_indices
        ]

EARLY_STORM_TO_WINDS_TABLE_PRELIM = EARLY_STORM_TO_WINDS_TABLE_PRELIM.loc[
    EARLY_STORM_TO_WINDS_TABLE_PRELIM[tracking_utils.VALID_TIME_COLUMN] <= 5
]

LATE_STORM_TO_WINDS_TABLE_PRELIM = LATE_STORM_TO_WINDS_TABLE_PRELIM.loc[
    LATE_STORM_TO_WINDS_TABLE_PRELIM[tracking_utils.VALID_TIME_COLUMN] > 5
]

EARLY_STORM_TO_WINDS_TABLE = STORM_TO_WINDS_TABLE.loc[
    STORM_TO_WINDS_TABLE[tracking_utils.VALID_TIME_COLUMN] <= 5
]

LATE_STORM_TO_WINDS_TABLE = STORM_TO_WINDS_TABLE.loc[
    STORM_TO_WINDS_TABLE[tracking_utils.VALID_TIME_COLUMN] > 5
]

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
        linkage.RELATIVE_EVENT_TIMES_COLUMN, linkage.MAIN_OBJECT_FLAGS_COLUMN,
        linkage.WIND_STATION_IDS_COLUMN, linkage.FUJITA_RATINGS_COLUMN,
        linkage.TORNADO_IDS_COLUMN
    ]
    float_array_columns = [
        linkage.EVENT_LATITUDES_COLUMN, linkage.EVENT_LONGITUDES_COLUMN,
        linkage.U_WINDS_COLUMN, linkage.V_WINDS_COLUMN,
        linkage.LINKAGE_DISTANCES_COLUMN
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

            elif this_column in float_array_columns:
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
            max_time_before_start_sec=MAX_TOY_EXTRAP_TIME_SEC,
            max_time_after_end_sec=MAX_TOY_EXTRAP_TIME_SEC)

        self.assertTrue(_compare_vertex_tables(
            this_vertex_table, INTERP_VERTEX_TABLE
        ))

    def test_find_nearest_storms_one_time(self):
        """Ensures correct output from _find_nearest_storms_one_time."""

        these_nearest_id_strings, these_link_distances_metres = (
            linkage._find_nearest_storms_one_time(
                interp_vertex_table=INTERP_VERTEX_TABLE,
                event_x_coords_metres=WIND_X_COORDS_1TIME_METRES,
                event_y_coords_metres=WIND_Y_COORDS_1TIME_METRES,
                max_link_distance_metres=MAX_LINK_DISTANCE_METRES)
        )

        self.assertTrue(
            these_nearest_id_strings == NEAREST_SEC_ID_STRINGS_1TIME
        )
        self.assertTrue(numpy.allclose(
            these_link_distances_metres, LINK_DISTANCES_1TIME_METRES,
            equal_nan=True, atol=TOLERANCE
        ))

    def test_find_nearest_storms(self):
        """Ensures correct output from _find_nearest_storms."""

        this_wind_to_storm_table = linkage._find_nearest_storms(
            storm_object_table=MAIN_STORM_OBJECT_TABLE,
            event_table=WIND_TABLE,
            max_time_before_storm_start_sec=MAX_WIND_EXTRAP_TIME_SEC,
            max_time_after_storm_end_sec=MAX_WIND_EXTRAP_TIME_SEC,
            max_link_distance_metres=MAX_LINK_DISTANCE_METRES,
            interp_time_interval_sec=INTERP_TIME_INTERVAL_SEC,
            event_type_string=linkage.WIND_EVENT_STRING)

        self.assertTrue(
            this_wind_to_storm_table[
                linkage.NEAREST_SECONDARY_ID_COLUMN].values.tolist() ==
            WIND_TO_STORM_TABLE[
                linkage.NEAREST_SECONDARY_ID_COLUMN].values.tolist()
        )

        self.assertTrue(numpy.array_equal(
            this_wind_to_storm_table[linkage.NEAREST_TIME_COLUMN].values,
            WIND_TO_STORM_TABLE[linkage.NEAREST_TIME_COLUMN].values
        ))

        self.assertTrue(numpy.allclose(
            this_wind_to_storm_table[linkage.LINKAGE_DISTANCE_COLUMN].values,
            WIND_TO_STORM_TABLE[linkage.LINKAGE_DISTANCE_COLUMN].values,
            equal_nan=True, atol=TOLERANCE
        ))

    def test_find_nearest_storms_tornado(self):
        """Ensures correct output from _find_nearest_storms."""

        this_tornado_to_storm_table = linkage._find_nearest_storms(
            storm_object_table=MAIN_STORM_OBJECT_TABLE,
            event_table=TORNADO_TABLE_AFTER_INTERP,
            max_time_before_storm_start_sec=MAX_TORNADO_EXTRAP_TIME_SEC,
            max_time_after_storm_end_sec=MAX_TORNADO_EXTRAP_TIME_SEC,
            max_link_distance_metres=MAX_LINK_DISTANCE_METRES,
            interp_time_interval_sec=INTERP_TIME_INTERVAL_SEC,
            event_type_string=linkage.TORNADO_EVENT_STRING)

        self.assertTrue(
            this_tornado_to_storm_table[
                linkage.NEAREST_SECONDARY_ID_COLUMN].values.tolist() ==
            TORNADO_TO_STORM_TABLE[
                linkage.NEAREST_SECONDARY_ID_COLUMN].values.tolist()
        )

        self.assertTrue(numpy.array_equal(
            this_tornado_to_storm_table[linkage.NEAREST_TIME_COLUMN].values,
            TORNADO_TO_STORM_TABLE[linkage.NEAREST_TIME_COLUMN].values
        ))

        # TODO(thunderhoser): Finish this unit test.

        # self.assertTrue(numpy.allclose(
        #     this_tornado_to_storm_table[linkage.LINKAGE_DISTANCE_COLUMN].values,
        #     TORNADO_TO_STORM_TABLE[linkage.LINKAGE_DISTANCE_COLUMN].values,
        #     equal_nan=True, atol=TOLERANCE
        # ))

    def test_reverse_wind_linkages(self):
        """Ensures correct output from _reverse_wind_linkages."""

        this_storm_to_winds_table = linkage._reverse_wind_linkages(
            storm_object_table=MAIN_STORM_OBJECT_TABLE,
            wind_to_storm_table=WIND_TO_STORM_TABLE)

        self.assertTrue(_compare_storm_to_events_tables(
            this_storm_to_winds_table, STORM_TO_WINDS_TABLE
        ))

    def test_reverse_tornado_linkages(self):
        """Ensures correct output from _reverse_tornado_linkages."""

        this_storm_to_tornadoes_table = linkage._reverse_tornado_linkages(
            storm_object_table=MAIN_STORM_OBJECT_TABLE,
            tornado_to_storm_table=TORNADO_TO_STORM_TABLE)

        # for i in range(len(this_storm_to_tornadoes_table)):
        #     these_tornado_id_strings = this_storm_to_tornadoes_table[
        #         linkage.TORNADO_IDS_COLUMN].values[i]
        #
        #     these_tornado_times_unix_sec = (
        #         this_storm_to_tornadoes_table[
        #             tracking_utils.VALID_TIME_COLUMN].values[i] +
        #         this_storm_to_tornadoes_table[
        #             linkage.RELATIVE_EVENT_TIMES_COLUMN].values[i]
        #     )
        #
        #     print(i)
        #     this_num_tornadoes = len(these_tornado_id_strings)
        #
        #     for j in range(this_num_tornadoes):
        #         print('{0:s} ... time = {1:d}'.format(
        #             these_tornado_id_strings[j], these_tornado_times_unix_sec[j]
        #         ))
        #
        #     print('\n')

        self.assertTrue(_compare_storm_to_events_tables(
            this_storm_to_tornadoes_table, STORM_TO_TORNADOES_TABLE
        ))

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
                    EARLY_STORM_TO_WINDS_TABLE_PRELIM),
                late_storm_to_events_table=copy.deepcopy(
                    LATE_STORM_TO_WINDS_TABLE_PRELIM)
            )
        )

        for i in range(len(this_early_table.index)):
            these_station_id_strings = (
                this_early_table[linkage.WIND_STATION_IDS_COLUMN].values[i]
            )

            if len(these_station_id_strings) == 0:
                continue

            these_sort_indices = numpy.argsort(
                numpy.array(these_station_id_strings)
            )

            for this_column in LINKAGE_ARRAY_COLUMNS:
                this_early_table[this_column].values[i] = (
                    this_early_table[this_column].values[i][these_sort_indices]
                )

            for this_column in LINKAGE_LIST_COLUMNS:
                this_early_table[this_column].values[i] = [
                    this_early_table[this_column].values[i][m]
                    for m in these_sort_indices
                ]

        self.assertTrue(_compare_storm_to_events_tables(
            this_early_table, EARLY_STORM_TO_WINDS_TABLE
        ))
        self.assertTrue(_compare_storm_to_events_tables(
            this_late_table, LATE_STORM_TO_WINDS_TABLE
        ))

    def test_interp_tornadoes_along_tracks(self):
        """Ensures correct output from _interp_tornadoes_along_tracks."""

        this_tornado_table = linkage._interp_tornadoes_along_tracks(
            tornado_table=copy.deepcopy(TORNADO_TABLE_BEFORE_INTERP),
            interp_time_interval_sec=TORNADO_INTERP_TIME_INTERVAL_SEC)

        actual_num_events = len(this_tornado_table.index)
        expected_num_events = len(TORNADO_TABLE_AFTER_INTERP.index)
        self.assertTrue(actual_num_events == expected_num_events)

        for this_column in list(this_tornado_table):
            if this_column in [linkage.EVENT_TIME_COLUMN,
                               tornado_io.TORNADO_ID_COLUMN,
                               tornado_io.FUJITA_RATING_COLUMN]:
                self.assertTrue(numpy.array_equal(
                    this_tornado_table[this_column].values,
                    TORNADO_TABLE_AFTER_INTERP[this_column].values
                ))
            else:
                self.assertTrue(numpy.allclose(
                    this_tornado_table[this_column].values,
                    TORNADO_TABLE_AFTER_INTERP[this_column].values,
                    atol=TOLERANCE
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
