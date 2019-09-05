"""Unit tests for linkage.py."""

import copy
import unittest
import numpy
import pandas
from gewittergefahr.gg_io import raw_wind_io
from gewittergefahr.gg_utils import linkage
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils

X_VERTICES_RELATIVE = numpy.array([-0.25, 0.25, 0.25, -0.25, -0.25])
Y_VERTICES_RELATIVE = numpy.array([-0.25, -0.25, 0.25, 0.25, -0.25])


def create_storm_objects():
    """Creates storm objects for use in unit tests.

    :return: storm_object_table: pandas DataFrame with many of the columns
        documented in `storm_tracking_io.write_file`.
    """

    valid_times_unix_sec = numpy.array([
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

    primary_id_strings = [
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

    secondary_id_strings = [
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

    centroid_latitudes_deg = numpy.array([
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

    centroid_longitudes_deg = numpy.array([
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

    start_times_unix_sec = numpy.array([
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

    end_times_unix_sec = numpy.array([
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

    first_prev_secondary_id_strings = [
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

    second_prev_secondary_id_strings = [
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

    first_next_secondary_id_strings = [
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

    second_next_secondary_id_strings = [
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

    storm_object_table = pandas.DataFrame.from_dict({
        tracking_utils.VALID_TIME_COLUMN: valid_times_unix_sec,
        tracking_utils.PRIMARY_ID_COLUMN: primary_id_strings,
        tracking_utils.SECONDARY_ID_COLUMN: secondary_id_strings,
        tracking_utils.CENTROID_LATITUDE_COLUMN: centroid_latitudes_deg,
        tracking_utils.CENTROID_LONGITUDE_COLUMN: centroid_longitudes_deg,
        linkage.STORM_CENTROID_X_COLUMN: centroid_longitudes_deg,
        linkage.STORM_CENTROID_Y_COLUMN: centroid_latitudes_deg,
        tracking_utils.CELL_START_TIME_COLUMN: start_times_unix_sec,
        tracking_utils.CELL_END_TIME_COLUMN: end_times_unix_sec,
        tracking_utils.FIRST_PREV_SECONDARY_ID_COLUMN:
            first_prev_secondary_id_strings,
        tracking_utils.SECOND_PREV_SECONDARY_ID_COLUMN:
            second_prev_secondary_id_strings,
        tracking_utils.FIRST_NEXT_SECONDARY_ID_COLUMN:
            first_next_secondary_id_strings,
        tracking_utils.SECOND_NEXT_SECONDARY_ID_COLUMN:
            second_next_secondary_id_strings
    })

    nested_array = storm_object_table[[
        tracking_utils.VALID_TIME_COLUMN, tracking_utils.VALID_TIME_COLUMN
    ]].values.tolist()

    storm_object_table = storm_object_table.assign(**{
        linkage.STORM_VERTICES_X_COLUMN: nested_array,
        linkage.STORM_VERTICES_Y_COLUMN: nested_array
    })

    num_storm_objects = len(storm_object_table.index)

    for j in range(num_storm_objects):
        storm_object_table[linkage.STORM_VERTICES_X_COLUMN].values[j] = (
            storm_object_table[linkage.STORM_CENTROID_X_COLUMN].values[j] +
            X_VERTICES_RELATIVE
        )

        storm_object_table[linkage.STORM_VERTICES_Y_COLUMN].values[j] = (
            storm_object_table[linkage.STORM_CENTROID_Y_COLUMN].values[j] +
            Y_VERTICES_RELATIVE
        )

    return storm_object_table


TOLERANCE = 1e-6
STORM_OBJECT_TABLE = create_storm_objects()

# The following constants are used to test _filter_storms_by_time.
FIRST_START_TIME_UNIX_SEC = 4
FIRST_END_TIME_UNIX_SEC = 6
THESE_INDICES = numpy.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 16, 17, 18, 19], dtype=int
)

STORM_OBJECT_TABLE_FIRST_FILTER = STORM_OBJECT_TABLE.drop(
    STORM_OBJECT_TABLE.index[THESE_INDICES], axis=0, inplace=False
)

SECOND_START_TIME_UNIX_SEC = 4
SECOND_END_TIME_UNIX_SEC = 7
THESE_INDICES = numpy.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    dtype=int
)

STORM_OBJECT_TABLE_SECOND_FILTER = STORM_OBJECT_TABLE.drop(
    STORM_OBJECT_TABLE.index[THESE_INDICES], axis=0, inplace=False
)

THIRD_START_TIME_UNIX_SEC = 6
THIRD_END_TIME_UNIX_SEC = 6
THESE_INDICES = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 18, 19], dtype=int)

STORM_OBJECT_TABLE_THIRD_FILTER = STORM_OBJECT_TABLE.drop(
    STORM_OBJECT_TABLE.index[THESE_INDICES], axis=0, inplace=False
)

FOURTH_START_TIME_UNIX_SEC = 6
FOURTH_END_TIME_UNIX_SEC = 7
THESE_INDICES = numpy.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 14, 15, 18, 19], dtype=int
)

STORM_OBJECT_TABLE_FOURTH_FILTER = STORM_OBJECT_TABLE.drop(
    STORM_OBJECT_TABLE.index[THESE_INDICES], axis=0, inplace=False
)

# The following constants are used to test _interp_one_storm_in_time.
STORM_OBJECT_TABLE_1CELL = STORM_OBJECT_TABLE.loc[
    STORM_OBJECT_TABLE[tracking_utils.SECONDARY_ID_COLUMN] == 'A1'
]

INTERP_TIME_1CELL_UNIX_SEC = 3
THESE_X_VERTICES = 241.5 + X_VERTICES_RELATIVE
THESE_Y_VERTICES = 50. + Y_VERTICES_RELATIVE

INTERP_VERTEX_TABLE_1OBJECT = pandas.DataFrame.from_dict({
    tracking_utils.SECONDARY_ID_COLUMN: ['A1'] * 5,
    linkage.STORM_VERTEX_X_COLUMN: THESE_X_VERTICES,
    linkage.STORM_VERTEX_Y_COLUMN: THESE_Y_VERTICES
})

EXTRAP_TIME_1CELL_UNIX_SEC = 5
THESE_X_VERTICES = 242.5 + X_VERTICES_RELATIVE
THESE_Y_VERTICES = 50. + Y_VERTICES_RELATIVE

EXTRAP_VERTEX_TABLE_1OBJECT = pandas.DataFrame.from_dict({
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
MAX_EXTRAPOLATION_TIME_SEC = 10

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
        THESE_CENTROID_X_COORDS[k] + X_VERTICES_RELATIVE
    ))

    THESE_VERTEX_Y_COORDS = numpy.concatenate((
        THESE_VERTEX_Y_COORDS,
        THESE_CENTROID_Y_COORDS[k] + Y_VERTICES_RELATIVE
    ))

INTERP_VERTEX_TABLE = pandas.DataFrame.from_dict({
    tracking_utils.SECONDARY_ID_COLUMN: THESE_VERTEX_ID_STRINGS,
    linkage.STORM_VERTEX_X_COLUMN: THESE_VERTEX_X_COORDS,
    linkage.STORM_VERTEX_Y_COLUMN: THESE_VERTEX_Y_COORDS
})

# The following constants are used to test _find_nearest_storms_one_time.
MAX_LINK_DISTANCE_METRES = 12.

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

# The following constants are used to test _find_nearest_storms.
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

STORM_TO_WINDS_TABLE = copy.deepcopy(STORM_OBJECT_TABLE)

THIS_NESTED_ARRAY = STORM_TO_WINDS_TABLE[[
    tracking_utils.VALID_TIME_COLUMN, tracking_utils.VALID_TIME_COLUMN
]].values.tolist()

THESE_COLUMNS = [
    linkage.EVENT_LATITUDES_COLUMN, linkage.EVENT_LONGITUDES_COLUMN,
    linkage.WIND_STATION_IDS_COLUMN, linkage.U_WINDS_COLUMN,
    linkage.V_WINDS_COLUMN, linkage.LINKAGE_DISTANCES_COLUMN,
    linkage.RELATIVE_EVENT_TIMES_COLUMN, linkage.MAIN_OBJECT_FLAGS_COLUMN
]

NUM_STORM_OBJECTS = len(STORM_TO_WINDS_TABLE.index)

for this_column in THESE_COLUMNS:
    STORM_TO_WINDS_TABLE = STORM_TO_WINDS_TABLE.assign(
        **{this_column: THIS_NESTED_ARRAY}
    )

    for k in range(NUM_STORM_OBJECTS):
        if this_column == linkage.WIND_STATION_IDS_COLUMN:
            STORM_TO_WINDS_TABLE[this_column].values[k] = []
        elif this_column == linkage.RELATIVE_EVENT_TIMES_COLUMN:
            STORM_TO_WINDS_TABLE[this_column].values[k] = numpy.array(
                [], dtype=int
            )
        elif this_column == linkage.MAIN_OBJECT_FLAGS_COLUMN:
            STORM_TO_WINDS_TABLE[this_column].values[k] = numpy.array(
                [], dtype=bool
            )
        else:
            STORM_TO_WINDS_TABLE[this_column].values[k] = numpy.array([])

STORM_ROW_TO_STATION_ID_STRINGS = {
    0: ['ii', 'iv', 'vi', 'viii'],
    2: ['xii'],
    3: ['ii', 'iv', 'vi', 'viii'],
    4: ['xii'],
    5: ['ii', 'iv', 'vi', 'viii'],
    6: ['x', 'xiv', 'xvi'],
    8: ['x', 'xiv'],
    11: ['x', 'xiv'],
    14: ['x', 'xiv']
}

STORM_ROW_TO_MAIN_OBJECT_FLAGS = {
    0: [0, 0, 0, 0],
    2: [0],
    3: [0, 0, 0, 0],
    4: [1],
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

    these_wind_rows = numpy.array([
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
        these_wind_rows]

    STORM_TO_WINDS_TABLE[linkage.EVENT_LONGITUDES_COLUMN].values[
        this_storm_row
    ] = WIND_TO_STORM_TABLE[linkage.EVENT_LONGITUDE_COLUMN].values[
        these_wind_rows]

    STORM_TO_WINDS_TABLE[linkage.U_WINDS_COLUMN].values[this_storm_row] = (
        WIND_TO_STORM_TABLE[raw_wind_io.U_WIND_COLUMN].values[these_wind_rows]
    )

    STORM_TO_WINDS_TABLE[linkage.V_WINDS_COLUMN].values[this_storm_row] = (
        WIND_TO_STORM_TABLE[raw_wind_io.V_WIND_COLUMN].values[these_wind_rows]
    )

    STORM_TO_WINDS_TABLE[linkage.LINKAGE_DISTANCES_COLUMN].values[
        this_storm_row
    ] = WIND_TO_STORM_TABLE[linkage.LINKAGE_DISTANCE_COLUMN].values[
        these_wind_rows]

    these_relative_times_sec = (
        WIND_TO_STORM_TABLE[linkage.EVENT_TIME_COLUMN].values[
            these_wind_rows] -
        STORM_TO_WINDS_TABLE[tracking_utils.VALID_TIME_COLUMN].values[
            this_storm_row]
    )

    STORM_TO_WINDS_TABLE[linkage.RELATIVE_EVENT_TIMES_COLUMN].values[
        this_storm_row
    ] = these_relative_times_sec

    STORM_TO_WINDS_TABLE[linkage.MAIN_OBJECT_FLAGS_COLUMN].values[
        this_storm_row
    ] = these_main_object_flags

# The following constants are used to test _share_linkages_with_predecessors.
EARLY_STORM_TO_WINDS_TABLE_PRELIM = copy.deepcopy(STORM_TO_WINDS_TABLE)
LATE_STORM_TO_WINDS_TABLE_PRELIM = copy.deepcopy(STORM_TO_WINDS_TABLE)

ROW_TO_EARLY_KEEP_FLAGS = {
    0: [1, 1, 1, 1],
    2: [1],
    3: [1, 1, 1, 1],
    4: [1],
    5: [1, 1, 1, 1],
    6: [0, 0, 1],
    8: [0, 0],
    11: [0, 0],
    14: [0, 0]
}

ROW_TO_LATE_KEEP_FLAGS = {
    0: [0, 0, 0, 0],
    2: [0],
    3: [0, 0, 0, 0],
    4: [0],
    5: [0, 0, 0, 0],
    6: [1, 1, 0],
    8: [1, 1],
    11: [1, 1],
    14: [1, 1]
}

for this_storm_row in ROW_TO_EARLY_KEEP_FLAGS:
    these_indices = numpy.where(numpy.array(
        ROW_TO_EARLY_KEEP_FLAGS[this_storm_row], dtype=bool
    ))[0]

    for this_column in THESE_COLUMNS:
        this_array = EARLY_STORM_TO_WINDS_TABLE_PRELIM[this_column].values[
            this_storm_row]

        if isinstance(this_array, numpy.ndarray):
            EARLY_STORM_TO_WINDS_TABLE_PRELIM[this_column].values[
                this_storm_row
            ] = this_array[these_indices]
        else:
            EARLY_STORM_TO_WINDS_TABLE_PRELIM[this_column].values[
                this_storm_row
            ] = [this_array[k] for k in these_indices]

for this_storm_row in ROW_TO_LATE_KEEP_FLAGS:
    these_indices = numpy.where(numpy.array(
        ROW_TO_LATE_KEEP_FLAGS[this_storm_row], dtype=bool
    ))[0]

    for this_column in THESE_COLUMNS:
        this_array = LATE_STORM_TO_WINDS_TABLE_PRELIM[this_column].values[
            this_storm_row]

        if isinstance(this_array, numpy.ndarray):
            LATE_STORM_TO_WINDS_TABLE_PRELIM[this_column].values[
                this_storm_row
            ] = this_array[these_indices]
        else:
            LATE_STORM_TO_WINDS_TABLE_PRELIM[this_column].values[
                this_storm_row
            ] = [this_array[k] for k in these_indices]

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


def _sort_wind_obs_for_each_storm(storm_to_winds_table):
    """Sorts wind observations (by ID, then time) for each storm.

    :param storm_to_winds_table: pandas DataFrame created by
        `linkage._reverse_wind_linkages`.
    :return: storm_to_winds_table: Same but with wind observations sorted for
        each storm.
    """

    columns_to_sort = [
        linkage.EVENT_LATITUDES_COLUMN, linkage.EVENT_LONGITUDES_COLUMN,
        linkage.WIND_STATION_IDS_COLUMN, linkage.U_WINDS_COLUMN,
        linkage.V_WINDS_COLUMN, linkage.LINKAGE_DISTANCES_COLUMN,
        linkage.RELATIVE_EVENT_TIMES_COLUMN, linkage.MAIN_OBJECT_FLAGS_COLUMN
    ]

    num_storm_objects = len(storm_to_winds_table.index)

    for i in range(num_storm_objects):
        these_id_strings = storm_to_winds_table[
            linkage.WIND_STATION_IDS_COLUMN].values[i]
        these_relative_times_sec = numpy.round(
            storm_to_winds_table[linkage.RELATIVE_EVENT_TIMES_COLUMN].values[i]
        ).astype(int)

        these_strings = numpy.array([
            '{0:s}_{1:d}'.format(s, t) for s, t in
            zip(these_id_strings, these_relative_times_sec)
        ])

        these_sort_indices = numpy.argsort(these_strings)

        for this_column in columns_to_sort:
            this_array = storm_to_winds_table[this_column].values[i]

            if isinstance(this_array, numpy.ndarray):
                storm_to_winds_table[this_column].values[i] = (
                    this_array[these_sort_indices]
                )
            else:
                storm_to_winds_table[this_column].values[i] = [
                    this_array[k] for k in these_sort_indices
                ]

    return storm_to_winds_table


def compare_storm_to_events_tables(first_table, second_table):
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
            if this_column == linkage.LINKAGE_DISTANCES_COLUMN:
                continue

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

    def test_filter_storms_first(self):
        """Ensures correct output from _filter_storms_by_time.

        In this case, using first set of time constraints.
        """

        this_storm_object_table = linkage._filter_storms_by_time(
            storm_object_table=STORM_OBJECT_TABLE,
            max_start_time_unix_sec=FIRST_START_TIME_UNIX_SEC,
            min_end_time_unix_sec=FIRST_END_TIME_UNIX_SEC)

        self.assertTrue(this_storm_object_table.equals(
            STORM_OBJECT_TABLE_FIRST_FILTER
        ))

    def test_filter_storms_second(self):
        """Ensures correct output from _filter_storms_by_time.

        In this case, using second set of time constraints.
        """

        this_storm_object_table = linkage._filter_storms_by_time(
            storm_object_table=STORM_OBJECT_TABLE,
            max_start_time_unix_sec=SECOND_START_TIME_UNIX_SEC,
            min_end_time_unix_sec=SECOND_END_TIME_UNIX_SEC)

        self.assertTrue(this_storm_object_table.equals(
            STORM_OBJECT_TABLE_SECOND_FILTER
        ))

    def test_filter_storms_third(self):
        """Ensures correct output from _filter_storms_by_time.

        In this case, using third set of time constraints.
        """

        this_storm_object_table = linkage._filter_storms_by_time(
            storm_object_table=STORM_OBJECT_TABLE,
            max_start_time_unix_sec=THIRD_START_TIME_UNIX_SEC,
            min_end_time_unix_sec=THIRD_END_TIME_UNIX_SEC)

        self.assertTrue(this_storm_object_table.equals(
            STORM_OBJECT_TABLE_THIRD_FILTER
        ))

    def test_filter_storms_fourth(self):
        """Ensures correct output from _filter_storms_by_time.

        In this case, using fourth set of time constraints.
        """

        this_storm_object_table = linkage._filter_storms_by_time(
            storm_object_table=STORM_OBJECT_TABLE,
            max_start_time_unix_sec=FOURTH_START_TIME_UNIX_SEC,
            min_end_time_unix_sec=FOURTH_END_TIME_UNIX_SEC)

        self.assertTrue(this_storm_object_table.equals(
            STORM_OBJECT_TABLE_FOURTH_FILTER
        ))

    def test_interp_one_storm_in_time(self):
        """Ensures correct output from _interp_one_storm_in_time.

        In this case the method is actually interpolating, not extrapolating.
        """

        this_vertex_table = linkage._interp_one_storm_in_time(
            storm_object_table_1cell=STORM_OBJECT_TABLE_1CELL,
            secondary_id_string=STORM_OBJECT_TABLE_1CELL[
                tracking_utils.SECONDARY_ID_COLUMN].values[0],
            target_time_unix_sec=INTERP_TIME_1CELL_UNIX_SEC)

        self.assertTrue(_compare_vertex_tables(
            this_vertex_table, INTERP_VERTEX_TABLE_1OBJECT
        ))

    def test_extrap_one_storm_in_time(self):
        """Ensures correct output from _interp_one_storm_in_time.

        In this case the method is extrapolating, not interpolating.
        """

        this_vertex_table = linkage._interp_one_storm_in_time(
            storm_object_table_1cell=STORM_OBJECT_TABLE_1CELL,
            secondary_id_string=STORM_OBJECT_TABLE_1CELL[
                tracking_utils.SECONDARY_ID_COLUMN].values[0],
            target_time_unix_sec=EXTRAP_TIME_1CELL_UNIX_SEC)

        self.assertTrue(_compare_vertex_tables(
            this_vertex_table, EXTRAP_VERTEX_TABLE_1OBJECT
        ))

    def test_get_bounding_box_for_storms(self):
        """Ensures correct output from _get_bounding_box_for_storms."""

        these_x_limits_metres, these_y_limits_metres = (
            linkage._get_bounding_box_for_storms(
                storm_object_table=STORM_OBJECT_TABLE,
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
            storm_object_table=STORM_OBJECT_TABLE,
            target_time_unix_sec=INTERP_TIME_UNIX_SEC,
            max_time_before_start_sec=MAX_EXTRAPOLATION_TIME_SEC,
            max_time_after_end_sec=MAX_EXTRAPOLATION_TIME_SEC)

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
            storm_object_table=STORM_OBJECT_TABLE,
            event_table=WIND_TABLE,
            max_time_before_storm_start_sec=MAX_EXTRAPOLATION_TIME_SEC,
            max_time_after_storm_end_sec=MAX_EXTRAPOLATION_TIME_SEC,
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

    def test_reverse_wind_linkages(self):
        """Ensures correct output from _reverse_wind_linkages."""

        this_storm_to_winds_table = linkage._reverse_wind_linkages(
            storm_object_table=STORM_OBJECT_TABLE,
            wind_to_storm_table=WIND_TO_STORM_TABLE)

        self.assertTrue(compare_storm_to_events_tables(
            this_storm_to_winds_table, STORM_TO_WINDS_TABLE
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

    def test_share_linkages_with_predecessors(self):
        """Ensures correct output from _share_linkages_with_predecessors."""

        this_actual_early_table, this_actual_late_table = (
            linkage._share_linkages_with_predecessors(
                early_storm_to_events_table=copy.deepcopy(
                    EARLY_STORM_TO_WINDS_TABLE_PRELIM),
                late_storm_to_events_table=copy.deepcopy(
                    LATE_STORM_TO_WINDS_TABLE_PRELIM)
            )
        )

        this_actual_early_table = _sort_wind_obs_for_each_storm(
            this_actual_early_table)
        this_expected_early_table = _sort_wind_obs_for_each_storm(
            copy.deepcopy(EARLY_STORM_TO_WINDS_TABLE)
        )

        self.assertTrue(compare_storm_to_events_tables(
            this_actual_early_table, this_expected_early_table
        ))

        this_actual_late_table = _sort_wind_obs_for_each_storm(
            this_actual_late_table)
        this_expected_late_table = _sort_wind_obs_for_each_storm(
            copy.deepcopy(LATE_STORM_TO_WINDS_TABLE)
        )

        self.assertTrue(compare_storm_to_events_tables(
            this_actual_late_table, this_expected_late_table
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
