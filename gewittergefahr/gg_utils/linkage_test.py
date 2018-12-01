"""Unit tests for linkage.py."""

import copy
import unittest
import numpy
import pandas
from gewittergefahr.gg_io import raw_wind_io
from gewittergefahr.gg_utils import linkage
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils

TOLERANCE = 1e-6

# The following constants are used for several unit tests.
THESE_STORM_IDS = [
    'a', 'b', 'c',
    'a', 'b', 'c', 'd',
    'a', 'c', 'd', 'e', 'f',
    'a', 'c', 'e', 'f',
    'a', 'e', 'f', 'g',
    'a', 'g'
]

THESE_TIMES_UNIX_SEC = numpy.array([
    0, 0, 0,
    300, 300, 300, 300,
    600, 600, 600, 600, 600,
    900, 900, 900, 900,
    1200, 1200, 1200, 1200,
    1500, 1500
], dtype=int)

THESE_START_TIMES_UNIX_SEC = numpy.array([
    0, 0, 0,
    0, 0, 0, 300,
    0, 0, 300, 600, 600,
    0, 0, 600, 600,
    0, 600, 600, 1200,
    0, 1200
], dtype=int)

THESE_END_TIMES_UNIX_SEC = numpy.array([
    1500, 300, 900,
    1500, 300, 900, 600,
    1500, 900, 600, 1200, 1200,
    1500, 900, 1200, 1200,
    1500, 1200, 1200, 1500,
    1500, 1500
], dtype=int)

THIS_DICT = {
    tracking_utils.STORM_ID_COLUMN: THESE_STORM_IDS,
    tracking_utils.TIME_COLUMN: THESE_TIMES_UNIX_SEC,
    tracking_utils.CELL_START_TIME_COLUMN: THESE_START_TIMES_UNIX_SEC,
    tracking_utils.CELL_END_TIME_COLUMN: THESE_END_TIMES_UNIX_SEC
}
MAIN_STORM_OBJECT_TABLE = pandas.DataFrame.from_dict(THIS_DICT)

# The following constants are used to test _filter_storms_by_time.
EARLY_START_TIME_UNIX_SEC = 300
LATE_START_TIME_UNIX_SEC = 600
EARLY_END_TIME_UNIX_SEC = 900
LATE_END_TIME_UNIX_SEC = 1200

THESE_BAD_INDICES = numpy.array(
    [1, 4, 6, 9, 10, 11, 14, 15, 17, 18, 19, 21], dtype=int
)
STORM_OBJECT_TABLE_EARLY_START_EARLY_END = MAIN_STORM_OBJECT_TABLE.drop(
    MAIN_STORM_OBJECT_TABLE.index[THESE_BAD_INDICES], axis=0, inplace=False)

THESE_BAD_INDICES = numpy.array(
    [1, 2, 4, 5, 6, 8, 9, 13, 10, 11, 14, 15, 17, 18, 19, 21], dtype=int
)
STORM_OBJECT_TABLE_EARLY_START_LATE_END = MAIN_STORM_OBJECT_TABLE.drop(
    MAIN_STORM_OBJECT_TABLE.index[THESE_BAD_INDICES], axis=0, inplace=False)

THESE_BAD_INDICES = numpy.array([1, 4, 6, 9, 19, 21], dtype=int)
STORM_OBJECT_TABLE_LATE_START_EARLY_END = MAIN_STORM_OBJECT_TABLE.drop(
    MAIN_STORM_OBJECT_TABLE.index[THESE_BAD_INDICES], axis=0, inplace=False)

THESE_BAD_INDICES = numpy.array([1, 2, 4, 5, 6, 8, 9, 13, 19, 21], dtype=int)
STORM_OBJECT_TABLE_LATE_START_LATE_END = MAIN_STORM_OBJECT_TABLE.drop(
    MAIN_STORM_OBJECT_TABLE.index[THESE_BAD_INDICES], axis=0, inplace=False)

# The following constants are used to test _interp_one_storm_in_time.
STORM_ID_1CELL = 'foo'
THESE_TIMES_UNIX_SEC = numpy.array([0, 300, 600], dtype=int)
THESE_CENTROID_X_METRES = numpy.array([5000, 10000, 12000], dtype=float)
THESE_CENTROID_Y_METRES = numpy.array([5000, 6000, 9000], dtype=float)

THIS_DICT = {
    tracking_utils.TIME_COLUMN: THESE_TIMES_UNIX_SEC,
    linkage.STORM_CENTROID_X_COLUMN: THESE_CENTROID_X_METRES,
    linkage.STORM_CENTROID_Y_COLUMN: THESE_CENTROID_Y_METRES
}
STORM_OBJECT_TABLE_1CELL = pandas.DataFrame.from_dict(THIS_DICT)

THIS_NESTED_ARRAY = STORM_OBJECT_TABLE_1CELL[[
    tracking_utils.TIME_COLUMN, tracking_utils.TIME_COLUMN
]].values.tolist()

THIS_DICT = {
    linkage.STORM_VERTICES_X_COLUMN: THIS_NESTED_ARRAY,
    linkage.STORM_VERTICES_Y_COLUMN: THIS_NESTED_ARRAY
}
STORM_OBJECT_TABLE_1CELL = STORM_OBJECT_TABLE_1CELL.assign(**THIS_DICT)

STORM_OBJECT_TABLE_1CELL[linkage.STORM_VERTICES_X_COLUMN].values[0] = (
    numpy.array([0, 10000, 10000, 0, 0], dtype=float)
)
STORM_OBJECT_TABLE_1CELL[linkage.STORM_VERTICES_Y_COLUMN].values[0] = (
    numpy.array([0, 0, 10000, 10000, 0], dtype=float)
)
STORM_OBJECT_TABLE_1CELL[linkage.STORM_VERTICES_X_COLUMN].values[1] = (
    numpy.array([5000, 15000, 15000, 5000, 5000], dtype=float)
)
STORM_OBJECT_TABLE_1CELL[linkage.STORM_VERTICES_Y_COLUMN].values[1] = (
    numpy.array([-4000, -4000, 16000, 16000, -4000], dtype=float)
)
STORM_OBJECT_TABLE_1CELL[linkage.STORM_VERTICES_X_COLUMN].values[2] = (
    numpy.array([2000, 22000, 22000, 2000, 2000], dtype=float)
)
STORM_OBJECT_TABLE_1CELL[linkage.STORM_VERTICES_Y_COLUMN].values[2] = (
    numpy.array([4000, 4000, 14000, 14000, 4000], dtype=float)
)

INTERP_TIME_1CELL_UNIX_SEC = 375
THESE_STORM_IDS = ['foo'] * 5
THESE_VERTEX_X_METRES = numpy.array(
    [5500, 15500, 15500, 5500, 5500], dtype=float)
THESE_VERTEX_Y_METRES = numpy.array(
    [-3250, -3250, 16750, 16750, -3250], dtype=float)

THIS_DICT = {
    tracking_utils.STORM_ID_COLUMN: THESE_STORM_IDS,
    linkage.STORM_VERTEX_X_COLUMN: THESE_VERTEX_X_METRES,
    linkage.STORM_VERTEX_Y_COLUMN: THESE_VERTEX_Y_METRES
}
VERTEX_TABLE_1OBJECT_INTERP = pandas.DataFrame.from_dict(THIS_DICT)

EXTRAP_TIME_1CELL_UNIX_SEC = 750
THESE_VERTEX_X_METRES = numpy.array(
    [3000, 23000, 23000, 3000, 3000], dtype=float)
THESE_VERTEX_Y_METRES = numpy.array(
    [5500, 5500, 15500, 15500, 5500], dtype=float)

THIS_DICT = {
    tracking_utils.STORM_ID_COLUMN: THESE_STORM_IDS,
    linkage.STORM_VERTEX_X_COLUMN: THESE_VERTEX_X_METRES,
    linkage.STORM_VERTEX_Y_COLUMN: THESE_VERTEX_Y_METRES
}
VERTEX_TABLE_1OBJECT_EXTRAP = pandas.DataFrame.from_dict(THIS_DICT)

# The following constants are used to test _get_bounding_box_for_storms.
BOUNDING_BOX_PADDING_METRES = 1000.
BOUNDING_BOX_X_METRES = numpy.array([-1000, 23000], dtype=float)
BOUNDING_BOX_Y_METRES = numpy.array([-5000, 17000], dtype=float)

# The following constants are used to test _filter_events_by_bounding_box.
THESE_EVENT_X_METRES = numpy.array(
    [59000, 59000, 59000, 59000, 59000, -46500, -36500, -31500, 0, 1e6]
)
THESE_EVENT_Y_METRES = numpy.array(
    [55000, 50000, 45000, 0, -1e6, -49000, -49000, -49000, -49000, -49000.]

)
EVENT_X_LIMITS_METRES = numpy.array([-59000, 72000], dtype=float)
EVENT_Y_LIMITS_METRES = numpy.array([-62000, 65000], dtype=float)

THIS_DICT = {
    linkage.EVENT_X_COLUMN: THESE_EVENT_X_METRES,
    linkage.EVENT_Y_COLUMN: THESE_EVENT_Y_METRES
}
EVENT_TABLE_FULL_DOMAIN = pandas.DataFrame.from_dict(THIS_DICT)

BAD_INDICES = numpy.array([4, 9], dtype=int)
EVENT_TABLE_IN_BOUNDING_BOX = EVENT_TABLE_FULL_DOMAIN.drop(
    EVENT_TABLE_FULL_DOMAIN.index[BAD_INDICES], axis=0, inplace=False)

# The following constants are used to test _interp_storms_in_time.
THESE_STORM_IDS = [
    'foo', 'bar',
    'foo', 'bar',
    'foo', 'bar'
]

THESE_TIMES_UNIX_SEC = numpy.array([
    0, 0,
    300, 300,
    700, 600
], dtype=int)

THESE_CENTROIDS_X_METRES = numpy.array([
    -50000, 50000,
    -48000, 55000,
    -46000, 59000
], dtype=float)

THESE_CENTROIDS_Y_METRES = numpy.array([
    -50000, 50000,
    -49000, 52000,
    -48500, 55000
], dtype=float)

THESE_START_TIMES_UNIX_SEC = numpy.array([
    0, 0,
    0, 0,
    0, 0
], dtype=int)

THESE_END_TIMES_UNIX_SEC = numpy.array([
    700, 600,
    700, 600,
    700, 600
], dtype=int)

THIS_DICT = {
    tracking_utils.TIME_COLUMN: THESE_TIMES_UNIX_SEC,
    tracking_utils.STORM_ID_COLUMN: THESE_STORM_IDS,
    linkage.STORM_CENTROID_X_COLUMN: THESE_CENTROIDS_X_METRES,
    linkage.STORM_CENTROID_Y_COLUMN: THESE_CENTROIDS_Y_METRES,
    tracking_utils.CELL_START_TIME_COLUMN: THESE_START_TIMES_UNIX_SEC,
    tracking_utils.CELL_END_TIME_COLUMN: THESE_END_TIMES_UNIX_SEC
}
STORM_OBJECT_TABLE_2CELLS = pandas.DataFrame.from_dict(THIS_DICT)

THIS_NESTED_ARRAY = STORM_OBJECT_TABLE_2CELLS[[
    tracking_utils.TIME_COLUMN, tracking_utils.TIME_COLUMN
]].values.tolist()

THIS_DICT = {
    linkage.STORM_VERTICES_X_COLUMN: THIS_NESTED_ARRAY,
    linkage.STORM_VERTICES_Y_COLUMN: THIS_NESTED_ARRAY
}
STORM_OBJECT_TABLE_2CELLS = STORM_OBJECT_TABLE_2CELLS.assign(**THIS_DICT)

STORM_OBJECT_TABLE_2CELLS[linkage.STORM_VERTICES_X_COLUMN].values[0] = (
    numpy.array([-55000, -45000, -45000, -55000, -55000], dtype=float)
)
STORM_OBJECT_TABLE_2CELLS[linkage.STORM_VERTICES_Y_COLUMN].values[0] = (
    numpy.array([-55000, -55000, -45000, -45000, -55000], dtype=float)
)
STORM_OBJECT_TABLE_2CELLS[linkage.STORM_VERTICES_X_COLUMN].values[1] = (
    numpy.array([45000, 55000, 55000, 45000, 45000], dtype=float)
)
STORM_OBJECT_TABLE_2CELLS[linkage.STORM_VERTICES_Y_COLUMN].values[1] = (
    numpy.array([45000, 45000, 55000, 55000, 45000], dtype=float)
)
STORM_OBJECT_TABLE_2CELLS[linkage.STORM_VERTICES_X_COLUMN].values[2] = (
    numpy.array([-53000, -43000, -43000, -53000, -53000], dtype=float)
)
STORM_OBJECT_TABLE_2CELLS[linkage.STORM_VERTICES_Y_COLUMN].values[2] = (
    numpy.array([-59000, -59000, -39000, -39000, -59000], dtype=float)
)
STORM_OBJECT_TABLE_2CELLS[linkage.STORM_VERTICES_X_COLUMN].values[3] = (
    numpy.array([50000, 60000, 60000, 50000, 50000], dtype=float)
)
STORM_OBJECT_TABLE_2CELLS[linkage.STORM_VERTICES_Y_COLUMN].values[3] = (
    numpy.array([42000, 42000, 62000, 62000, 42000], dtype=float)
)
STORM_OBJECT_TABLE_2CELLS[linkage.STORM_VERTICES_X_COLUMN].values[4] = (
    numpy.array([-56000, -36000, -36000, -56000, -56000], dtype=float)
)
STORM_OBJECT_TABLE_2CELLS[linkage.STORM_VERTICES_Y_COLUMN].values[4] = (
    numpy.array([-53500, -53500, -43500, -43500, -53500], dtype=float)
)
STORM_OBJECT_TABLE_2CELLS[linkage.STORM_VERTICES_X_COLUMN].values[5] = (
    numpy.array([49000, 69000, 69000, 49000, 49000], dtype=float)
)
STORM_OBJECT_TABLE_2CELLS[linkage.STORM_VERTICES_Y_COLUMN].values[5] = (
    numpy.array([50000, 50000, 60000, 60000, 50000], dtype=float)
)

INTERP_TIME_2CELLS_UNIX_SEC = 600
MAX_TIME_BEFORE_STORM_START_SEC = 0
MAX_TIME_AFTER_STORM_END_SEC = 0

THESE_VERTEX_X_METRES = numpy.array(
    [49000, 69000, 69000, 49000, 49000, -56500, -36500, -36500, -56500, -56500],
    dtype=float)
THESE_VERTEX_Y_METRES = numpy.array(
    [50000, 50000, 60000, 60000, 50000, -53625, -53625, -43625, -43625, -53625],
    dtype=float)
THESE_STORM_IDS = [
    'bar', 'bar', 'bar', 'bar', 'bar',
    'foo', 'foo', 'foo', 'foo', 'foo'
]

THIS_DICT = {
    tracking_utils.STORM_ID_COLUMN: THESE_STORM_IDS,
    linkage.STORM_VERTEX_X_COLUMN: THESE_VERTEX_X_METRES,
    linkage.STORM_VERTEX_Y_COLUMN: THESE_VERTEX_Y_METRES
}
INTERP_VERTEX_TABLE_2OBJECTS = pandas.DataFrame.from_dict(THIS_DICT)

# The following constants are used to test _find_nearest_storms_one_time.
MAX_LINK_DISTANCE_METRES = 10000.
EVENT_X_COORDS_1TIME_METRES = numpy.array(
    [49000, 49000, 49000, 49000, -46500, -36500, -31500, 0], dtype=float)
EVENT_Y_COORDS_1TIME_METRES = numpy.array(
    [55000, 50000, 45000, 0, -43625, -43625, -43625, -43625], dtype=float)

NEAREST_STORM_IDS_1TIME = [
    'bar', 'bar', 'bar', None, 'foo', 'foo', 'foo', None
]
LINKAGE_DISTANCES_1TIME_METRES = numpy.array(
    [0, 0, 5000, numpy.nan, 0, 0, 5000, numpy.nan])

# The following constants are used to test _find_nearest_storms.
INTERP_TIME_RESOLUTION_SEC = 10

THESE_X_METRES = numpy.array(
    [49000, 49000, 49000, 49000, -46500, -36500, -31500, 0,
     49000, 49000, 49000, 49000, -46000, -36000, -31000, 0], dtype=float)
THESE_Y_METRES = numpy.array(
    [55000, 50000, 45000, 0, -43625, -43625, -43625, -43625,
     55000, 50000, 45000, 0, -43500, -43500, -43500, -43500], dtype=float)
THESE_TIMES_UNIX_SEC = numpy.array(
    [600, 600, 600, 600, 600, 600, 600, 600,
     700, 700, 700, 700, 700, 700, 700, 700], dtype=int)

THIS_DICT = {
    linkage.EVENT_X_COLUMN: THESE_X_METRES,
    linkage.EVENT_Y_COLUMN: THESE_Y_METRES,
    linkage.EVENT_TIME_COLUMN: THESE_TIMES_UNIX_SEC
}
EVENT_TABLE_2TIMES = pandas.DataFrame.from_dict(THIS_DICT)

THESE_STORM_IDS = [
    'bar', 'bar', 'bar', None, 'foo', 'foo', 'foo', None,
    None, None, None, None, 'foo', 'foo', 'foo', None]
THESE_LINKAGE_DISTANCES_METRES = numpy.array(
    [0, 0, 5000, numpy.nan, 0, 0, 5000, numpy.nan,
     numpy.nan, numpy.nan, numpy.nan, numpy.nan, 0, 0, 5000, numpy.nan])

THIS_DICT = {
    linkage.NEAREST_STORM_ID_COLUMN: THESE_STORM_IDS,
    linkage.LINKAGE_DISTANCE_COLUMN: THESE_LINKAGE_DISTANCES_METRES
}
EVENT_TO_STORM_TABLE_SIMPLE = EVENT_TABLE_2TIMES.assign(**THIS_DICT)

# The following constants are used to test _reverse_wind_linkages.
THESE_STATION_IDS = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'
]
THESE_LATITUDES_DEG = numpy.array(
    [1, 2, 3, 4, 5, 6, 7, 8,
     1, 2, 3, 4, 5, 6, 7, 8], dtype=float)

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

STORM_TO_WINDS_TABLE = copy.deepcopy(STORM_OBJECT_TABLE_2CELLS)
THIS_NESTED_ARRAY = STORM_TO_WINDS_TABLE[[
    tracking_utils.TIME_COLUMN, tracking_utils.TIME_COLUMN
]].values.tolist()

THIS_DICT = {
    linkage.WIND_STATION_IDS_COLUMN: THIS_NESTED_ARRAY,
    linkage.EVENT_LATITUDES_COLUMN: THIS_NESTED_ARRAY,
    linkage.EVENT_LONGITUDES_COLUMN: THIS_NESTED_ARRAY,
    linkage.U_WINDS_COLUMN: THIS_NESTED_ARRAY,
    linkage.V_WINDS_COLUMN: THIS_NESTED_ARRAY,
    linkage.LINKAGE_DISTANCES_COLUMN: THIS_NESTED_ARRAY,
    linkage.RELATIVE_EVENT_TIMES_COLUMN: THIS_NESTED_ARRAY
}
STORM_TO_WINDS_TABLE = STORM_TO_WINDS_TABLE.assign(**THIS_DICT)

THESE_STATION_IDS = ['e', 'f', 'g', 'e', 'f', 'g']
THESE_WIND_LATITUDES_DEG = numpy.array([5, 6, 7, 5, 6, 7], dtype=float)
THESE_WIND_LONGITUDES_DEG = numpy.array([5, 6, 7, 5, 6, 7], dtype=float)
THESE_U_WINDS_M_S01 = numpy.array([5, 6, 7, 5, 6, 7], dtype=float)
THESE_V_WINDS_M_S01 = numpy.array([5, 6, 7, 5, 6, 7], dtype=float)
THESE_LINK_DISTANCES_METRES = numpy.array([0, 0, 5000, 0, 0, 5000], dtype=float)

FOO_ROWS = numpy.array([0, 2, 4], dtype=int)

for this_row in FOO_ROWS:
    STORM_TO_WINDS_TABLE[
        linkage.WIND_STATION_IDS_COLUMN
    ].values[this_row] = THESE_STATION_IDS

    STORM_TO_WINDS_TABLE[
        linkage.EVENT_LATITUDES_COLUMN
    ].values[this_row] = THESE_WIND_LATITUDES_DEG

    STORM_TO_WINDS_TABLE[
        linkage.EVENT_LONGITUDES_COLUMN
    ].values[this_row] = THESE_WIND_LONGITUDES_DEG

    STORM_TO_WINDS_TABLE[
        linkage.U_WINDS_COLUMN
    ].values[this_row] = THESE_U_WINDS_M_S01

    STORM_TO_WINDS_TABLE[
        linkage.V_WINDS_COLUMN
    ].values[this_row] = THESE_V_WINDS_M_S01

    STORM_TO_WINDS_TABLE[
        linkage.LINKAGE_DISTANCES_COLUMN
    ].values[this_row] = THESE_LINK_DISTANCES_METRES

STORM_TO_WINDS_TABLE[linkage.RELATIVE_EVENT_TIMES_COLUMN].values[0] = (
    numpy.array([600, 600, 600, 700, 700, 700], dtype=int)
)
STORM_TO_WINDS_TABLE[linkage.RELATIVE_EVENT_TIMES_COLUMN].values[2] = (
    numpy.array([300, 300, 300, 400, 400, 400], dtype=int)
)
STORM_TO_WINDS_TABLE[linkage.RELATIVE_EVENT_TIMES_COLUMN].values[4] = (
    numpy.array([-100, -100, -100, 0, 0, 0], dtype=int)
)

THESE_STATION_IDS = ['a', 'b', 'c']
THESE_WIND_LATITUDES_DEG = numpy.array([1, 2, 3], dtype=float)
THESE_WIND_LONGITUDES_DEG = numpy.array([1, 2, 3], dtype=float)
THESE_U_WINDS_M_S01 = numpy.array([1, 2, 3], dtype=float)
THESE_V_WINDS_M_S01 = numpy.array([1, 2, 3], dtype=float)
THESE_LINK_DISTANCES_METRES = numpy.array([0, 0, 5000], dtype=float)

BAR_ROWS = numpy.array([1, 3, 5], dtype=int)

for this_row in BAR_ROWS:
    STORM_TO_WINDS_TABLE[
        linkage.WIND_STATION_IDS_COLUMN
    ].values[this_row] = THESE_STATION_IDS

    STORM_TO_WINDS_TABLE[
        linkage.EVENT_LATITUDES_COLUMN
    ].values[this_row] = THESE_WIND_LATITUDES_DEG

    STORM_TO_WINDS_TABLE[
        linkage.EVENT_LONGITUDES_COLUMN
    ].values[this_row] = THESE_WIND_LONGITUDES_DEG

    STORM_TO_WINDS_TABLE[
        linkage.U_WINDS_COLUMN
    ].values[this_row] = THESE_U_WINDS_M_S01

    STORM_TO_WINDS_TABLE[
        linkage.V_WINDS_COLUMN
    ].values[this_row] = THESE_V_WINDS_M_S01

    STORM_TO_WINDS_TABLE[
        linkage.LINKAGE_DISTANCES_COLUMN
    ].values[this_row] = THESE_LINK_DISTANCES_METRES

STORM_TO_WINDS_TABLE[linkage.RELATIVE_EVENT_TIMES_COLUMN].values[1] = (
    numpy.array([600, 600, 600], dtype=int)
)
STORM_TO_WINDS_TABLE[linkage.RELATIVE_EVENT_TIMES_COLUMN].values[3] = (
    numpy.array([300, 300, 300], dtype=int)
)
STORM_TO_WINDS_TABLE[linkage.RELATIVE_EVENT_TIMES_COLUMN].values[5] = (
    numpy.array([0, 0, 0], dtype=int)
)

# The following constants are used to test _share_linkages_between_periods.
THESE_EARLY_STORM_IDS = ['A', 'C', 'A', 'B', 'C']
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

for k in range(len(THESE_EARLY_STORM_IDS)):
    THESE_EVENT_LATITUDES_DEG[k] = THESE_EVENT_LATITUDES_DEG[k].astype(float)
    THESE_EVENT_LONGITUDES_DEG[k] = THESE_EVENT_LONGITUDES_DEG[k].astype(float)
    THESE_LINK_DIST_METRES[k] = THESE_LINK_DIST_METRES[k].astype(float)
    THESE_RELATIVE_TIMES_UNIX_SEC[k] = THESE_RELATIVE_TIMES_UNIX_SEC[k].astype(
        int)

THIS_DICT = {
    tracking_utils.STORM_ID_COLUMN: THESE_EARLY_STORM_IDS,
    tracking_utils.TIME_COLUMN: THESE_EARLY_TIMES_UNIX_SEC,
    linkage.EVENT_LATITUDES_COLUMN: THESE_EVENT_LATITUDES_DEG,
    linkage.EVENT_LONGITUDES_COLUMN: THESE_EVENT_LONGITUDES_DEG,
    linkage.LINKAGE_DISTANCES_COLUMN: THESE_LINK_DIST_METRES,
    linkage.RELATIVE_EVENT_TIMES_COLUMN: THESE_RELATIVE_TIMES_UNIX_SEC,
    linkage.FUJITA_RATINGS_COLUMN: THESE_FUJITA_RATINGS
}

EARLY_STORM_TO_TORNADOES_TABLE_SANS_SHARING = pandas.DataFrame.from_dict(
    THIS_DICT)

THESE_LATE_STORM_IDS = ['B', 'C', 'D', 'C', 'D']
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

for k in range(len(THESE_LATE_STORM_IDS)):
    THESE_EVENT_LATITUDES_DEG[k] = THESE_EVENT_LATITUDES_DEG[k].astype(float)
    THESE_EVENT_LONGITUDES_DEG[k] = THESE_EVENT_LONGITUDES_DEG[k].astype(float)
    THESE_LINK_DIST_METRES[k] = THESE_LINK_DIST_METRES[k].astype(float)
    THESE_RELATIVE_TIMES_UNIX_SEC[k] = THESE_RELATIVE_TIMES_UNIX_SEC[k].astype(
        int)

THIS_DICT = {
    tracking_utils.STORM_ID_COLUMN: THESE_LATE_STORM_IDS,
    tracking_utils.TIME_COLUMN: THESE_LATE_TIMES_UNIX_SEC,
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

for k in range(len(THESE_EARLY_STORM_IDS)):
    THESE_EVENT_LATITUDES_DEG[k] = THESE_EVENT_LATITUDES_DEG[k].astype(float)
    THESE_EVENT_LONGITUDES_DEG[k] = THESE_EVENT_LONGITUDES_DEG[k].astype(float)
    THESE_LINK_DIST_METRES[k] = THESE_LINK_DIST_METRES[k].astype(float)
    THESE_RELATIVE_TIMES_UNIX_SEC[k] = THESE_RELATIVE_TIMES_UNIX_SEC[k].astype(
        int)

THIS_DICT = {
    tracking_utils.STORM_ID_COLUMN: THESE_EARLY_STORM_IDS,
    tracking_utils.TIME_COLUMN: THESE_EARLY_TIMES_UNIX_SEC,
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

for k in range(len(THESE_LATE_STORM_IDS)):
    THESE_EVENT_LATITUDES_DEG[k] = THESE_EVENT_LATITUDES_DEG[k].astype(float)
    THESE_EVENT_LONGITUDES_DEG[k] = THESE_EVENT_LONGITUDES_DEG[k].astype(float)
    THESE_LINK_DIST_METRES[k] = THESE_LINK_DIST_METRES[k].astype(float)
    THESE_RELATIVE_TIMES_UNIX_SEC[k] = THESE_RELATIVE_TIMES_UNIX_SEC[k].astype(
        int)

THIS_DICT = {
    tracking_utils.STORM_ID_COLUMN: THESE_LATE_STORM_IDS,
    tracking_utils.TIME_COLUMN: THESE_LATE_TIMES_UNIX_SEC,
    linkage.EVENT_LATITUDES_COLUMN: THESE_EVENT_LATITUDES_DEG,
    linkage.EVENT_LONGITUDES_COLUMN: THESE_EVENT_LONGITUDES_DEG,
    linkage.LINKAGE_DISTANCES_COLUMN: THESE_LINK_DIST_METRES,
    linkage.RELATIVE_EVENT_TIMES_COLUMN: THESE_RELATIVE_TIMES_UNIX_SEC,
    linkage.FUJITA_RATINGS_COLUMN: THESE_FUJITA_RATINGS
}
LATE_STORM_TO_TORNADOES_TABLE_WITH_SHARING = pandas.DataFrame.from_dict(
    THIS_DICT)

STRING_COLUMNS = [tracking_utils.STORM_ID_COLUMN]
NON_FLOAT_ARRAY_COLUMNS = [
    linkage.RELATIVE_EVENT_TIMES_COLUMN,
    linkage.FUJITA_RATINGS_COLUMN]
FLOAT_ARRAY_COLUMNS = [
    linkage.EVENT_LATITUDES_COLUMN, linkage.EVENT_LONGITUDES_COLUMN,
    linkage.LINKAGE_DISTANCES_COLUMN]

# The following constants are used to test find_linkage_file.
TOP_DIRECTORY_NAME = 'linkage'
FILE_TIME_UNIX_SEC = 1517523991  # 222631 1 Feb 2018
FILE_SPC_DATE_STRING = '20180201'

LINKAGE_FILE_NAME_WIND_ONE_TIME = (
    'linkage/2018/20180201/storm_to_winds_2018-02-01-222631.p')
LINKAGE_FILE_NAME_WIND_ONE_DATE = 'linkage/2018/storm_to_winds_20180201.p'
LINKAGE_FILE_NAME_TORNADO_ONE_TIME = (
    'linkage/2018/20180201/storm_to_tornadoes_2018-02-01-222631.p')
LINKAGE_FILE_NAME_TORNADO_ONE_DATE = (
    'linkage/2018/storm_to_tornadoes_20180201.p')


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

    string_columns = [tracking_utils.STORM_ID_COLUMN]
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

    def test_filter_storms_by_time_early_start_early_end(self):
        """Ensures correct output from _filter_storms_by_time.

        In this case, storm cells will be dropped if they start after 300
        seconds or end before 900 seconds.
        """

        this_storm_object_table = linkage._filter_storms_by_time(
            storm_object_table=MAIN_STORM_OBJECT_TABLE,
            max_start_time_unix_sec=EARLY_START_TIME_UNIX_SEC,
            min_end_time_unix_sec=EARLY_END_TIME_UNIX_SEC)

        self.assertTrue(this_storm_object_table.equals(
            STORM_OBJECT_TABLE_EARLY_START_EARLY_END))

    def test_filter_storms_by_time_early_start_late_end(self):
        """Ensures correct output from _filter_storms_by_time.

        In this case, storm cells will be dropped if they start after 300
        seconds or end before 1200 seconds.
        """

        this_storm_object_table = linkage._filter_storms_by_time(
            storm_object_table=MAIN_STORM_OBJECT_TABLE,
            max_start_time_unix_sec=EARLY_START_TIME_UNIX_SEC,
            min_end_time_unix_sec=LATE_END_TIME_UNIX_SEC)

        self.assertTrue(this_storm_object_table.equals(
            STORM_OBJECT_TABLE_EARLY_START_LATE_END))

    def test_filter_storms_by_time_late_start_early_end(self):
        """Ensures correct output from _filter_storms_by_time.

        In this case, storm cells will be dropped if they start after 600
        seconds or end before 900 seconds.
        """

        this_storm_object_table = linkage._filter_storms_by_time(
            storm_object_table=MAIN_STORM_OBJECT_TABLE,
            max_start_time_unix_sec=LATE_START_TIME_UNIX_SEC,
            min_end_time_unix_sec=EARLY_END_TIME_UNIX_SEC)

        self.assertTrue(this_storm_object_table.equals(
            STORM_OBJECT_TABLE_LATE_START_EARLY_END))

    def test_filter_storms_by_time_late_start_late_end(self):
        """Ensures correct output from _filter_storms_by_time.

        In this case, storm cells will be dropped if they start after 600
        seconds or end before 1200 seconds.
        """

        this_storm_object_table = linkage._filter_storms_by_time(
            storm_object_table=MAIN_STORM_OBJECT_TABLE,
            max_start_time_unix_sec=LATE_START_TIME_UNIX_SEC,
            min_end_time_unix_sec=LATE_END_TIME_UNIX_SEC)

        self.assertTrue(this_storm_object_table.equals(
            STORM_OBJECT_TABLE_LATE_START_LATE_END))

    def test_interp_one_storm_in_time_interp(self):
        """Ensures correct output from _interp_one_storm_in_time.

        In this case the method is doing strict interpolation, not
        extrapolation.
        """

        this_vertex_table = linkage._interp_one_storm_in_time(
            storm_object_table_1cell=STORM_OBJECT_TABLE_1CELL,
            storm_id=STORM_ID_1CELL,
            target_time_unix_sec=INTERP_TIME_1CELL_UNIX_SEC)

        self.assertTrue(this_vertex_table.equals(VERTEX_TABLE_1OBJECT_INTERP))

    def test_interp_one_storm_in_time_extrap(self):
        """Ensures correct output from _interp_one_storm_in_time.

        In this case the method is doing extrapolation, not interpolation.
        """

        this_vertex_table = linkage._interp_one_storm_in_time(
            storm_object_table_1cell=STORM_OBJECT_TABLE_1CELL,
            storm_id=STORM_ID_1CELL,
            target_time_unix_sec=EXTRAP_TIME_1CELL_UNIX_SEC)

        self.assertTrue(this_vertex_table.equals(VERTEX_TABLE_1OBJECT_EXTRAP))

    def test_get_bounding_box_for_storms(self):
        """Ensures correct output from _get_bounding_box_for_storms."""

        these_x_limits_metres, these_y_limits_metres = (
            linkage._get_bounding_box_for_storms(
                storm_object_table=STORM_OBJECT_TABLE_1CELL,
                padding_metres=BOUNDING_BOX_PADDING_METRES)
        )

        self.assertTrue(numpy.allclose(
            these_x_limits_metres, BOUNDING_BOX_X_METRES, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_y_limits_metres, BOUNDING_BOX_Y_METRES, atol=TOLERANCE))

    def test_filter_events_by_bounding_box(self):
        """Ensures correct output from _filter_events_by_bounding_box."""

        this_event_table = linkage._filter_events_by_bounding_box(
            event_table=EVENT_TABLE_FULL_DOMAIN,
            x_limits_metres=EVENT_X_LIMITS_METRES,
            y_limits_metres=EVENT_Y_LIMITS_METRES)

        self.assertTrue(this_event_table.equals(EVENT_TABLE_IN_BOUNDING_BOX))

    def test_interp_storms_in_time(self):
        """Ensures correct output from _interp_storms_in_time."""

        this_vertex_table = linkage._interp_storms_in_time(
            storm_object_table=STORM_OBJECT_TABLE_2CELLS,
            target_time_unix_sec=INTERP_TIME_2CELLS_UNIX_SEC,
            max_time_before_start_sec=MAX_TIME_BEFORE_STORM_START_SEC,
            max_time_after_end_sec=MAX_TIME_AFTER_STORM_END_SEC)

        self.assertTrue(this_vertex_table.equals(INTERP_VERTEX_TABLE_2OBJECTS))

    def test_find_nearest_storms_one_time(self):
        """Ensures correct output from _find_nearest_storms_one_time."""

        these_nearest_storm_ids, these_link_distances_metres = (
            linkage._find_nearest_storms_one_time(
                interp_vertex_table=INTERP_VERTEX_TABLE_2OBJECTS,
                event_x_coords_metres=EVENT_X_COORDS_1TIME_METRES,
                event_y_coords_metres=EVENT_Y_COORDS_1TIME_METRES,
                max_link_distance_metres=MAX_LINK_DISTANCE_METRES)
        )

        self.assertTrue(these_nearest_storm_ids == NEAREST_STORM_IDS_1TIME)
        self.assertTrue(numpy.allclose(
            these_link_distances_metres, LINKAGE_DISTANCES_1TIME_METRES,
            equal_nan=True, atol=TOLERANCE))

    def test_find_nearest_storms(self):
        """Ensures correct output from _find_nearest_storms."""

        this_wind_to_storm_table = linkage._find_nearest_storms(
            storm_object_table=STORM_OBJECT_TABLE_2CELLS,
            event_table=EVENT_TABLE_2TIMES,
            max_time_before_storm_start_sec=MAX_TIME_BEFORE_STORM_START_SEC,
            max_time_after_storm_end_sec=MAX_TIME_AFTER_STORM_END_SEC,
            max_link_distance_metres=MAX_LINK_DISTANCE_METRES,
            interp_time_resolution_sec=INTERP_TIME_RESOLUTION_SEC)

        self.assertTrue(this_wind_to_storm_table.equals(
            EVENT_TO_STORM_TABLE_SIMPLE))

    def test_reverse_wind_linkages(self):
        """Ensures correct output from _reverse_wind_linkages."""

        this_storm_to_winds_table = linkage._reverse_wind_linkages(
            storm_object_table=STORM_OBJECT_TABLE_2CELLS,
            wind_to_storm_table=WIND_TO_STORM_TABLE)

        self.assertTrue(_compare_storm_to_events_tables(
            this_storm_to_winds_table, STORM_TO_WINDS_TABLE))

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
