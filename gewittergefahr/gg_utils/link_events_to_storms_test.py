"""Unit tests for link_events_to_storms.py."""

import copy
import unittest
import numpy
import pandas
from gewittergefahr.gg_io import raw_wind_io
from gewittergefahr.gg_utils import link_events_to_storms as events2storms
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils

TOLERANCE = 1e-6

# The following constants are used to test _find_start_end_times_of_storms.
THESE_STORM_IDS = ['a', 'b', 'c',
                   'a', 'b', 'c', 'd',
                   'a', 'c', 'd', 'e', 'f',
                   'a', 'c', 'e', 'f',
                   'a', 'e', 'f', 'g',
                   'a', 'g']
THESE_TIMES_UNIX_SEC = numpy.array([0, 0, 0,
                                    300, 300, 300, 300,
                                    600, 600, 600, 600, 600,
                                    900, 900, 900, 900,
                                    1200, 1200, 1200, 1200,
                                    1500, 1500])
THESE_START_TIMES_UNIX_SEC = numpy.array([0, 0, 0,
                                          0, 0, 0, 300,
                                          0, 0, 300, 600, 600,
                                          0, 0, 600, 600,
                                          0, 600, 600, 1200,
                                          0, 1200])
THESE_END_TIMES_UNIX_SEC = numpy.array([1500, 300, 900,
                                        1500, 300, 900, 600,
                                        1500, 900, 600, 1200, 1200,
                                        1500, 900, 1200, 1200,
                                        1500, 1200, 1200, 1500,
                                        1500, 1500])

THIS_DICTIONARY = {
    tracking_utils.STORM_ID_COLUMN: THESE_STORM_IDS,
    tracking_utils.TIME_COLUMN: THESE_TIMES_UNIX_SEC,
    tracking_utils.CELL_START_TIME_COLUMN: THESE_START_TIMES_UNIX_SEC,
    tracking_utils.CELL_END_TIME_COLUMN: THESE_END_TIMES_UNIX_SEC
}
MAIN_STORM_OBJECT_TABLE = pandas.DataFrame.from_dict(THIS_DICTIONARY)

# The following constants are used to test _filter_storms_by_time.
EARLY_START_TIME_UNIX_SEC = 300
LATE_START_TIME_UNIX_SEC = 600
EARLY_END_TIME_UNIX_SEC = 900
LATE_END_TIME_UNIX_SEC = 1200

THESE_INVALID_ROWS = numpy.array(
    [1, 4, 6, 9, 10, 11, 14, 15, 17, 18, 19, 21], dtype=int)
STORM_OBJECT_TABLE_EARLY_START_EARLY_END = MAIN_STORM_OBJECT_TABLE.drop(
    MAIN_STORM_OBJECT_TABLE.index[THESE_INVALID_ROWS], axis=0, inplace=False)

THESE_INVALID_ROWS = numpy.array(
    [1, 2, 4, 5, 6, 8, 9, 13, 10, 11, 14, 15, 17, 18, 19, 21], dtype=int)
STORM_OBJECT_TABLE_EARLY_START_LATE_END = MAIN_STORM_OBJECT_TABLE.drop(
    MAIN_STORM_OBJECT_TABLE.index[THESE_INVALID_ROWS], axis=0, inplace=False)

THESE_INVALID_ROWS = numpy.array([1, 4, 6, 9, 19, 21], dtype=int)
STORM_OBJECT_TABLE_LATE_START_EARLY_END = MAIN_STORM_OBJECT_TABLE.drop(
    MAIN_STORM_OBJECT_TABLE.index[THESE_INVALID_ROWS], axis=0, inplace=False)

THESE_INVALID_ROWS = numpy.array([1, 2, 4, 5, 6, 8, 9, 13, 19, 21], dtype=int)
STORM_OBJECT_TABLE_LATE_START_LATE_END = MAIN_STORM_OBJECT_TABLE.drop(
    MAIN_STORM_OBJECT_TABLE.index[THESE_INVALID_ROWS], axis=0, inplace=False)

# The following constants are used to test _interp_one_storm_in_time.
STORM_ID_FOR_INTERP = 'foo'
THESE_TIMES_UNIX_SEC = numpy.array([0, 300, 600])
THESE_CENTROID_X_METRES = numpy.array([5000., 10000., 12000.])
THESE_CENTROID_Y_METRES = numpy.array([5000., 6000., 9000.])

THIS_DICTIONARY = {
    tracking_utils.TIME_COLUMN: THESE_TIMES_UNIX_SEC,
    events2storms.STORM_CENTROID_X_COLUMN: THESE_CENTROID_X_METRES,
    events2storms.STORM_CENTROID_Y_COLUMN: THESE_CENTROID_Y_METRES
}
STORM_OBJECT_TABLE_1CELL = pandas.DataFrame.from_dict(THIS_DICTIONARY)

THIS_NESTED_ARRAY = STORM_OBJECT_TABLE_1CELL[[
    tracking_utils.TIME_COLUMN, tracking_utils.TIME_COLUMN]].values.tolist()
THIS_DICTIONARY = {
    events2storms.STORM_VERTICES_X_COLUMN: THIS_NESTED_ARRAY,
    events2storms.STORM_VERTICES_Y_COLUMN: THIS_NESTED_ARRAY
}
STORM_OBJECT_TABLE_1CELL = STORM_OBJECT_TABLE_1CELL.assign(**THIS_DICTIONARY)

STORM_OBJECT_TABLE_1CELL[events2storms.STORM_VERTICES_X_COLUMN].values[
    0] = numpy.array([0., 10000., 10000., 0., 0.])
STORM_OBJECT_TABLE_1CELL[events2storms.STORM_VERTICES_Y_COLUMN].values[
    0] = numpy.array([0., 0., 10000., 10000., 0.])

STORM_OBJECT_TABLE_1CELL[events2storms.STORM_VERTICES_X_COLUMN].values[
    1] = numpy.array([5000., 15000., 15000., 5000., 5000.])
STORM_OBJECT_TABLE_1CELL[events2storms.STORM_VERTICES_Y_COLUMN].values[
    1] = numpy.array([-4000., -4000., 16000., 16000., -4000.])

STORM_OBJECT_TABLE_1CELL[events2storms.STORM_VERTICES_X_COLUMN].values[
    2] = numpy.array([2000., 22000., 22000., 2000., 2000.])
STORM_OBJECT_TABLE_1CELL[events2storms.STORM_VERTICES_Y_COLUMN].values[
    2] = numpy.array([4000., 4000., 14000., 14000., 4000.])

INTERP_TIME_1CELL_UNIX_SEC = 375
EXTRAP_TIME_1CELL_UNIX_SEC = 750

THESE_VERTEX_X_METRES = numpy.array([5500., 15500., 15500., 5500., 5500.])
THESE_VERTEX_Y_METRES = numpy.array([-3250., -3250., 16750., 16750., -3250.])
THIS_STORM_ID_LIST = ['foo'] * 5
THIS_DICTIONARY = {
    tracking_utils.STORM_ID_COLUMN: THIS_STORM_ID_LIST,
    events2storms.STORM_VERTEX_X_COLUMN: THESE_VERTEX_X_METRES,
    events2storms.STORM_VERTEX_Y_COLUMN: THESE_VERTEX_Y_METRES
}
VERTEX_TABLE_1OBJECT_NO_EXTRAP = pandas.DataFrame.from_dict(THIS_DICTIONARY)

THESE_VERTEX_X_METRES = numpy.array([3000., 23000., 23000., 3000., 3000.])
THESE_VERTEX_Y_METRES = numpy.array([5500., 5500., 15500., 15500., 5500.])
THIS_DICTIONARY = {
    tracking_utils.STORM_ID_COLUMN: THIS_STORM_ID_LIST,
    events2storms.STORM_VERTEX_X_COLUMN: THESE_VERTEX_X_METRES,
    events2storms.STORM_VERTEX_Y_COLUMN: THESE_VERTEX_Y_METRES
}
VERTEX_TABLE_1OBJECT_EXTRAP = pandas.DataFrame.from_dict(THIS_DICTIONARY)

# The following constants are used to test _get_bounding_box_of_storms.
BOUNDING_BOX_PADDING_METRES = 1000.
BOUNDING_BOX_X_METRES = numpy.array([-1000., 23000.])
BOUNDING_BOX_Y_METRES = numpy.array([-5000., 17000.])

# The following constants are used to test _interp_storms_in_time.
THIS_STORM_ID_LIST = ['foo', 'bar',
                      'foo', 'bar',
                      'foo', 'bar']
THESE_TIMES_UNIX_SEC = numpy.array([0, 0,
                                    300, 300,
                                    700, 600])
THESE_CENTROIDS_X_METRES = numpy.array([-50000., 50000.,
                                        -48000, 55000.,
                                        -46000., 59000.])
THESE_CENTROIDS_Y_METRES = numpy.array([-50000., 50000.,
                                        -49000, 52000.,
                                        -48500., 55000.])
THESE_START_TIMES_UNIX_SEC = numpy.array([0, 0,
                                          0, 0,
                                          0, 0])
THESE_END_TIMES_UNIX_SEC = numpy.array([700, 600,
                                        700, 600,
                                        700, 600])

THIS_DICTIONARY = {
    tracking_utils.TIME_COLUMN: THESE_TIMES_UNIX_SEC,
    tracking_utils.STORM_ID_COLUMN: THIS_STORM_ID_LIST,
    events2storms.STORM_CENTROID_X_COLUMN: THESE_CENTROIDS_X_METRES,
    events2storms.STORM_CENTROID_Y_COLUMN: THESE_CENTROIDS_Y_METRES,
    tracking_utils.CELL_START_TIME_COLUMN: THESE_START_TIMES_UNIX_SEC,
    tracking_utils.CELL_END_TIME_COLUMN: THESE_END_TIMES_UNIX_SEC
}
STORM_OBJECT_TABLE_2CELLS = pandas.DataFrame.from_dict(THIS_DICTIONARY)

THIS_NESTED_ARRAY = STORM_OBJECT_TABLE_2CELLS[[
    tracking_utils.TIME_COLUMN, tracking_utils.TIME_COLUMN]].values.tolist()
THIS_DICTIONARY = {
    events2storms.STORM_VERTICES_X_COLUMN: THIS_NESTED_ARRAY,
    events2storms.STORM_VERTICES_Y_COLUMN: THIS_NESTED_ARRAY
}
STORM_OBJECT_TABLE_2CELLS = STORM_OBJECT_TABLE_2CELLS.assign(**THIS_DICTIONARY)

STORM_OBJECT_TABLE_2CELLS[events2storms.STORM_VERTICES_X_COLUMN].values[
    0] = numpy.array([-55000., -45000., -45000., -55000., -55000.])
STORM_OBJECT_TABLE_2CELLS[events2storms.STORM_VERTICES_Y_COLUMN].values[
    0] = numpy.array([-55000., -55000., -45000., -45000., -55000.])

STORM_OBJECT_TABLE_2CELLS[events2storms.STORM_VERTICES_X_COLUMN].values[
    1] = numpy.array([45000., 55000., 55000., 45000., 45000.])
STORM_OBJECT_TABLE_2CELLS[events2storms.STORM_VERTICES_Y_COLUMN].values[
    1] = numpy.array([45000., 45000., 55000., 55000., 45000.])

STORM_OBJECT_TABLE_2CELLS[events2storms.STORM_VERTICES_X_COLUMN].values[
    2] = numpy.array([-53000., -43000., -43000., -53000., -53000.])
STORM_OBJECT_TABLE_2CELLS[events2storms.STORM_VERTICES_Y_COLUMN].values[
    2] = numpy.array([-59000., -59000., -39000., -39000., -59000.])

STORM_OBJECT_TABLE_2CELLS[events2storms.STORM_VERTICES_X_COLUMN].values[
    3] = numpy.array([50000., 60000., 60000., 50000., 50000.])
STORM_OBJECT_TABLE_2CELLS[events2storms.STORM_VERTICES_Y_COLUMN].values[
    3] = numpy.array([42000., 42000., 62000., 62000., 42000.])

STORM_OBJECT_TABLE_2CELLS[events2storms.STORM_VERTICES_X_COLUMN].values[
    4] = numpy.array([-56000., -36000., -36000., -56000., -56000.])
STORM_OBJECT_TABLE_2CELLS[events2storms.STORM_VERTICES_Y_COLUMN].values[
    4] = numpy.array([-53500., -53500., -43500., -43500., -53500.])

STORM_OBJECT_TABLE_2CELLS[events2storms.STORM_VERTICES_X_COLUMN].values[
    5] = numpy.array([49000., 69000., 69000., 49000., 49000.])
STORM_OBJECT_TABLE_2CELLS[events2storms.STORM_VERTICES_Y_COLUMN].values[
    5] = numpy.array([50000., 50000., 60000., 60000., 50000.])

INTERP_TIME_2CELLS_UNIX_SEC = 600
MAX_TIME_BEFORE_STORM_START_SEC = 0
MAX_TIME_AFTER_STORM_END_SEC = 0

THESE_VERTEX_X_METRES = numpy.array(
    [49000., 69000., 69000., 49000., 49000.,
     -56500., -36500., -36500., -56500., -56500.])
THESE_VERTEX_Y_METRES = numpy.array(
    [50000., 50000., 60000., 60000., 50000.,
     -53625., -53625., -43625., -43625., -53625.])
THIS_STORM_ID_LIST = ['bar', 'bar', 'bar', 'bar', 'bar',
                      'foo', 'foo', 'foo', 'foo', 'foo']

THIS_DICTIONARY = {
    tracking_utils.STORM_ID_COLUMN: THIS_STORM_ID_LIST,
    events2storms.STORM_VERTEX_X_COLUMN: THESE_VERTEX_X_METRES,
    events2storms.STORM_VERTEX_Y_COLUMN: THESE_VERTEX_Y_METRES
}
INTERP_VERTEX_TABLE_2OBJECTS = pandas.DataFrame.from_dict(THIS_DICTIONARY)

# The following constants are used to test _filter_events_by_location.
THESE_EVENT_X_METRES = numpy.array(
    [59000., 59000., 59000., 59000., 59000.,
     -46500., -36500., -31500., 0., 1e6])
THESE_EVENT_Y_METRES = numpy.array(
    [55000., 50000., 45000., 0., -1e6,
     -49000., -49000., -49000., -49000., -49000.])
EVENT_X_LIMITS_METRES = numpy.array([-59000., 72000.])
EVENT_Y_LIMITS_METRES = numpy.array([-62000., 65000.])

THIS_DICTIONARY = {
    events2storms.EVENT_X_COLUMN: THESE_EVENT_X_METRES,
    events2storms.EVENT_Y_COLUMN: THESE_EVENT_Y_METRES
}
EVENT_TABLE_FULL_DOMAIN = pandas.DataFrame.from_dict(THIS_DICTIONARY)

INVALID_ROWS = numpy.array([4, 9], dtype=int)
EVENT_TABLE_BOUNDING_BOX = EVENT_TABLE_FULL_DOMAIN.drop(
    EVENT_TABLE_FULL_DOMAIN.index[INVALID_ROWS], axis=0, inplace=False)

# The following constants are used to test _find_nearest_storms_at_one_time.
MAX_LINK_DISTANCE_METRES = 10000.
EVENT_X_COORDS_1TIME_METRES = numpy.array([49000., 49000., 49000., 49000.,
                                           -46500., -36500., -31500., 0.])
EVENT_Y_COORDS_1TIME_METRES = numpy.array([55000., 50000., 45000., 0.,
                                           -43625., -43625., -43625., -43625.])

NEAREST_STORM_IDS_1TIME = ['bar', 'bar', 'bar', None,
                           'foo', 'foo', 'foo', None]
LINKAGE_DISTANCES_1TIME_METRES = numpy.array([0., 0., 5000., numpy.nan,
                                              0., 0., 5000., numpy.nan])

# The following constants are used to test _find_nearest_storms.
THESE_X_METRES = numpy.array([49000., 49000., 49000., 49000.,
                              -46500., -36500., -31500., 0.,
                              49000., 49000., 49000., 49000.,
                              -46000., -36000., -31000., 0., ])
THESE_Y_METRES = numpy.array([55000., 50000., 45000., 0.,
                              -43625., -43625., -43625., -43625.,
                              55000., 50000., 45000., 0.,
                              -43500., -43500., -43500., -43500.])
THESE_TIMES_UNIX_SEC = numpy.array([600, 600, 600, 600,
                                    600, 600, 600, 600,
                                    700, 700, 700, 700,
                                    700, 700, 700, 700])

THIS_DICTIONARY = {
    events2storms.EVENT_X_COLUMN: THESE_X_METRES,
    events2storms.EVENT_Y_COLUMN: THESE_Y_METRES,
    events2storms.EVENT_TIME_COLUMN: THESE_TIMES_UNIX_SEC
}
EVENT_TABLE_2TIMES = pandas.DataFrame.from_dict(THIS_DICTIONARY)

THESE_STORM_IDS = ['bar', 'bar', 'bar', None,
                   'foo', 'foo', 'foo', None,
                   None, None, None, None,
                   'foo', 'foo', 'foo', None]
THESE_LINKAGE_DISTANCES_METRES = numpy.array(
    [0., 0., 5000., numpy.nan,
     0., 0., 5000., numpy.nan,
     numpy.nan, numpy.nan, numpy.nan, numpy.nan,
     0., 0., 5000., numpy.nan])

THIS_DICTIONARY = {
    events2storms.NEAREST_STORM_ID_COLUMN: THESE_STORM_IDS,
    events2storms.LINKAGE_DISTANCE_COLUMN: THESE_LINKAGE_DISTANCES_METRES
}
EVENT_TO_STORM_TABLE_SIMPLE = EVENT_TABLE_2TIMES.assign(**THIS_DICTIONARY)

# The following constants are used to test _create_storm_to_winds_table.
THESE_STATION_IDS = ['a', 'b', 'c', 'd',
                     'e', 'f', 'g', 'h',
                     'a', 'b', 'c', 'd',
                     'e', 'f', 'g', 'h']
THESE_LATITUDES_DEG = numpy.array([1., 2., 3., 4.,
                                   5., 6., 7., 8.,
                                   1., 2., 3., 4.,
                                   5., 6., 7., 8.])
THESE_LONGITUDES_DEG = copy.deepcopy(THESE_LATITUDES_DEG)
THESE_U_WINDS_M_S01 = copy.deepcopy(THESE_LATITUDES_DEG)
THESE_V_WINDS_M_S01 = copy.deepcopy(THESE_LATITUDES_DEG)

THIS_DICTIONARY = {
    raw_wind_io.STATION_ID_COLUMN: THESE_STATION_IDS,
    events2storms.EVENT_LATITUDE_COLUMN: THESE_LATITUDES_DEG,
    events2storms.EVENT_LONGITUDE_COLUMN: THESE_LONGITUDES_DEG,
    raw_wind_io.U_WIND_COLUMN: THESE_U_WINDS_M_S01,
    raw_wind_io.V_WIND_COLUMN: THESE_V_WINDS_M_S01
}
WIND_TO_STORM_TABLE = EVENT_TO_STORM_TABLE_SIMPLE.assign(**THIS_DICTIONARY)

STORM_TO_WINDS_TABLE = copy.deepcopy(STORM_OBJECT_TABLE_2CELLS)
THIS_NESTED_ARRAY = STORM_TO_WINDS_TABLE[[
    tracking_utils.TIME_COLUMN, tracking_utils.TIME_COLUMN]].values.tolist()
THIS_ARGUMENT_DICT = {
    events2storms.WIND_STATION_IDS_COLUMN: THIS_NESTED_ARRAY,
    events2storms.EVENT_LATITUDES_COLUMN: THIS_NESTED_ARRAY,
    events2storms.EVENT_LONGITUDES_COLUMN: THIS_NESTED_ARRAY,
    events2storms.U_WINDS_COLUMN: THIS_NESTED_ARRAY,
    events2storms.V_WINDS_COLUMN: THIS_NESTED_ARRAY,
    events2storms.LINKAGE_DISTANCES_COLUMN: THIS_NESTED_ARRAY,
    events2storms.RELATIVE_EVENT_TIMES_COLUMN: THIS_NESTED_ARRAY
}
STORM_TO_WINDS_TABLE = STORM_TO_WINDS_TABLE.assign(**THIS_ARGUMENT_DICT)

THESE_STATION_IDS = ['e', 'f', 'g', 'e', 'f', 'g']
THESE_WIND_LATITUDES_DEG = numpy.array([5., 6., 7., 5., 6., 7.])
THESE_WIND_LONGITUDES_DEG = numpy.array([5., 6., 7., 5., 6., 7.])
THESE_U_WINDS_M_S01 = numpy.array([5., 6., 7., 5., 6., 7.])
THESE_V_WINDS_M_S01 = numpy.array([5., 6., 7., 5., 6., 7.])
THESE_LINK_DISTANCES_METRES = numpy.array([0., 0., 5000., 0., 0., 5000.])

FOO_ROWS = numpy.array([0, 2, 4], dtype=int)
for this_row in FOO_ROWS:
    STORM_TO_WINDS_TABLE[events2storms.WIND_STATION_IDS_COLUMN].values[
        this_row] = THESE_STATION_IDS
    STORM_TO_WINDS_TABLE[events2storms.EVENT_LATITUDES_COLUMN].values[
        this_row] = THESE_WIND_LATITUDES_DEG
    STORM_TO_WINDS_TABLE[events2storms.EVENT_LONGITUDES_COLUMN].values[
        this_row] = THESE_WIND_LONGITUDES_DEG
    STORM_TO_WINDS_TABLE[events2storms.U_WINDS_COLUMN].values[
        this_row] = THESE_U_WINDS_M_S01
    STORM_TO_WINDS_TABLE[events2storms.V_WINDS_COLUMN].values[
        this_row] = THESE_V_WINDS_M_S01
    STORM_TO_WINDS_TABLE[events2storms.LINKAGE_DISTANCES_COLUMN].values[
        this_row] = THESE_LINK_DISTANCES_METRES

STORM_TO_WINDS_TABLE[events2storms.RELATIVE_EVENT_TIMES_COLUMN].values[
    0] = numpy.array([600, 600, 600, 700, 700, 700])
STORM_TO_WINDS_TABLE[events2storms.RELATIVE_EVENT_TIMES_COLUMN].values[
    2] = numpy.array([300, 300, 300, 400, 400, 400])
STORM_TO_WINDS_TABLE[events2storms.RELATIVE_EVENT_TIMES_COLUMN].values[
    4] = numpy.array([-100, -100, -100, 0, 0, 0])

THESE_STATION_IDS = ['a', 'b', 'c']
THESE_WIND_LATITUDES_DEG = numpy.array([1., 2., 3.])
THESE_WIND_LONGITUDES_DEG = numpy.array([1., 2., 3.])
THESE_U_WINDS_M_S01 = numpy.array([1., 2., 3.])
THESE_V_WINDS_M_S01 = numpy.array([1., 2., 3.])
THESE_LINK_DISTANCES_METRES = numpy.array([0., 0., 5000.])

BAR_ROWS = numpy.array([1, 3, 5], dtype=int)
for this_row in BAR_ROWS:
    STORM_TO_WINDS_TABLE[events2storms.WIND_STATION_IDS_COLUMN].values[
        this_row] = THESE_STATION_IDS
    STORM_TO_WINDS_TABLE[events2storms.EVENT_LATITUDES_COLUMN].values[
        this_row] = THESE_WIND_LATITUDES_DEG
    STORM_TO_WINDS_TABLE[events2storms.EVENT_LONGITUDES_COLUMN].values[
        this_row] = THESE_WIND_LONGITUDES_DEG
    STORM_TO_WINDS_TABLE[events2storms.U_WINDS_COLUMN].values[
        this_row] = THESE_U_WINDS_M_S01
    STORM_TO_WINDS_TABLE[events2storms.V_WINDS_COLUMN].values[
        this_row] = THESE_V_WINDS_M_S01
    STORM_TO_WINDS_TABLE[events2storms.LINKAGE_DISTANCES_COLUMN].values[
        this_row] = THESE_LINK_DISTANCES_METRES

STORM_TO_WINDS_TABLE[events2storms.RELATIVE_EVENT_TIMES_COLUMN].values[
    1] = numpy.array([600, 600, 600])
STORM_TO_WINDS_TABLE[events2storms.RELATIVE_EVENT_TIMES_COLUMN].values[
    3] = numpy.array([300, 300, 300])
STORM_TO_WINDS_TABLE[events2storms.RELATIVE_EVENT_TIMES_COLUMN].values[
    5] = numpy.array([0, 0, 0])

# The following constants are used to test _share_linkages_between_periods.
THESE_EARLY_STORM_IDS = ['A', 'C', 'A', 'B', 'C']
THESE_EARLY_TIMES_UNIX_SEC = numpy.array([0, 0, 1, 1, 1], dtype=int)
THESE_EVENT_LATITUDES_DEG = [
    numpy.array([53., 53.]), numpy.array([55., 55.]),
    numpy.array([53., 53.]), numpy.array([54., 54.]), numpy.array([55., 55.])]
THESE_EVENT_LONGITUDES_DEG = [
    numpy.array([246., 247.]), numpy.array([246., 247.]),
    numpy.array([246., 247.]), numpy.array([246., 247.]),
    numpy.array([246., 247.])]
THESE_LINK_DIST_METRES = [
    numpy.array([1000., 2000.]), numpy.array([0., 0.]),
    numpy.array([1000., 2000.]), numpy.array([5000., 10000.]),
    numpy.array([0., 0.])]
THESE_RELATIVE_TIMES_UNIX_SEC = [
    numpy.array([1, 2], dtype=int), numpy.array([5, 6], dtype=int),
    numpy.array([0, 1], dtype=int), numpy.array([2, 3], dtype=int),
    numpy.array([4, 5], dtype=int)]
THESE_FUJITA_RATINGS = [
    ['F0', 'F1'], ['EF4', 'EF5'],
    ['F0', 'F1'], ['EF2', 'EF3'], ['EF4', 'EF5']]

THIS_DICT = {
    tracking_utils.STORM_ID_COLUMN: THESE_EARLY_STORM_IDS,
    tracking_utils.TIME_COLUMN: THESE_EARLY_TIMES_UNIX_SEC,
    events2storms.EVENT_LATITUDES_COLUMN: THESE_EVENT_LATITUDES_DEG,
    events2storms.EVENT_LONGITUDES_COLUMN: THESE_EVENT_LONGITUDES_DEG,
    events2storms.LINKAGE_DISTANCES_COLUMN: THESE_LINK_DIST_METRES,
    events2storms.RELATIVE_EVENT_TIMES_COLUMN: THESE_RELATIVE_TIMES_UNIX_SEC,
    events2storms.FUJITA_RATINGS_COLUMN: THESE_FUJITA_RATINGS
}
EARLY_STORM_TO_TORNADOES_TABLE_SANS_SHARING = pandas.DataFrame.from_dict(
    THIS_DICT)

THESE_LATE_STORM_IDS = ['B', 'C', 'D', 'C', 'D']
THESE_LATE_TIMES_UNIX_SEC = numpy.array([2, 2, 2, 3, 3], dtype=int)
THESE_EVENT_LATITUDES_DEG = [
    numpy.array([53.5, 53.5]), numpy.array([54.5, 54.5]),
    numpy.array([70., 70.]), numpy.array([54.5, 54.5]), numpy.array([70., 70.])]
THESE_EVENT_LONGITUDES_DEG = [
    numpy.array([246., 247.]), numpy.array([246., 247.]),
    numpy.array([246., 247.]), numpy.array([246., 247.]),
    numpy.array([246., 247.])]
THESE_LINK_DIST_METRES = [
    numpy.array([333., 666.]), numpy.array([0., 1.]), numpy.array([2., 3.]),
    numpy.array([0., 1.]), numpy.array([2., 3.])]
THESE_RELATIVE_TIMES_UNIX_SEC = [
    numpy.array([0, 2], dtype=int), numpy.array([2, 4], dtype=int),
    numpy.array([4, 6], dtype=int), numpy.array([1, 3], dtype=int),
    numpy.array([3, 5], dtype=int)]
THESE_FUJITA_RATINGS = [
    ['f0', 'f1'], ['ef2', 'ef3'], ['ef4', 'ef5'],
    ['ef2', 'ef3'], ['ef4', 'ef5']]

THIS_DICT = {
    tracking_utils.STORM_ID_COLUMN: THESE_LATE_STORM_IDS,
    tracking_utils.TIME_COLUMN: THESE_LATE_TIMES_UNIX_SEC,
    events2storms.EVENT_LATITUDES_COLUMN: THESE_EVENT_LATITUDES_DEG,
    events2storms.EVENT_LONGITUDES_COLUMN: THESE_EVENT_LONGITUDES_DEG,
    events2storms.LINKAGE_DISTANCES_COLUMN: THESE_LINK_DIST_METRES,
    events2storms.RELATIVE_EVENT_TIMES_COLUMN: THESE_RELATIVE_TIMES_UNIX_SEC,
    events2storms.FUJITA_RATINGS_COLUMN: THESE_FUJITA_RATINGS
}
LATE_STORM_TO_TORNADOES_TABLE_SANS_SHARING = pandas.DataFrame.from_dict(
    THIS_DICT)

THESE_EVENT_LATITUDES_DEG = [
    numpy.array([53., 53.]), numpy.array([54.5, 54.5, 55., 55.]),
    numpy.array([53., 53.]), numpy.array([53.5, 53.5, 54., 54.]),
    numpy.array([54.5, 54.5, 55., 55.])]
THESE_EVENT_LONGITUDES_DEG = [
    numpy.array([246., 247.]), numpy.array([246., 247., 246., 247.]),
    numpy.array([246., 247.]), numpy.array([246., 247., 246., 247.]),
    numpy.array([246., 247., 246., 247.])]
THESE_LINK_DIST_METRES = [
    numpy.array([1000., 2000.]), numpy.array([0., 1., 0., 0.]),
    numpy.array([1000., 2000.]), numpy.array([333., 666., 5000., 10000.]),
    numpy.array([0., 1., 0., 0.])]
THESE_RELATIVE_TIMES_UNIX_SEC = [
    numpy.array([1, 2], dtype=int), numpy.array([4, 6, 5, 6], dtype=int),
    numpy.array([0, 1], dtype=int), numpy.array([1, 3, 2, 3], dtype=int),
    numpy.array([3, 5, 4, 5], dtype=int)]
THESE_FUJITA_RATINGS = [
    ['F0', 'F1'], ['ef2', 'ef3', 'EF4', 'EF5'],
    ['F0', 'F1'], ['f0', 'f1', 'EF2', 'EF3'], ['ef2', 'ef3', 'EF4', 'EF5']]

THIS_DICT = {
    tracking_utils.STORM_ID_COLUMN: THESE_EARLY_STORM_IDS,
    tracking_utils.TIME_COLUMN: THESE_EARLY_TIMES_UNIX_SEC,
    events2storms.EVENT_LATITUDES_COLUMN: THESE_EVENT_LATITUDES_DEG,
    events2storms.EVENT_LONGITUDES_COLUMN: THESE_EVENT_LONGITUDES_DEG,
    events2storms.LINKAGE_DISTANCES_COLUMN: THESE_LINK_DIST_METRES,
    events2storms.RELATIVE_EVENT_TIMES_COLUMN: THESE_RELATIVE_TIMES_UNIX_SEC,
    events2storms.FUJITA_RATINGS_COLUMN: THESE_FUJITA_RATINGS
}
EARLY_STORM_TO_TORNADOES_TABLE_WITH_SHARING = pandas.DataFrame.from_dict(
    THIS_DICT)

THESE_EVENT_LATITUDES_DEG = [
    numpy.array([53.5, 53.5, 54., 54.]), numpy.array([54.5, 54.5, 55., 55.]),
    numpy.array([70., 70.]), numpy.array([54.5, 54.5, 55., 55.]),
    numpy.array([70., 70.])]
THESE_EVENT_LONGITUDES_DEG = [
    numpy.array([246., 247., 246., 247.]),
    numpy.array([246., 247., 246., 247.]), numpy.array([246., 247.]),
    numpy.array([246., 247., 246., 247.]), numpy.array([246., 247.])]
THESE_LINK_DIST_METRES = [
    numpy.array([333., 666., 5000., 10000.]), numpy.array([0., 1., 0., 0.]),
    numpy.array([2., 3.]), numpy.array([0., 1., 0., 0.]), numpy.array([2., 3.])]
THESE_RELATIVE_TIMES_UNIX_SEC = [
    numpy.array([0, 2, 1, 2], dtype=int), numpy.array([2, 4, 3, 4], dtype=int),
    numpy.array([4, 6], dtype=int), numpy.array([1, 3, 2, 3], dtype=int),
    numpy.array([3, 5], dtype=int)]
THESE_FUJITA_RATINGS = [
    ['f0', 'f1', 'EF2', 'EF3'], ['ef2', 'ef3', 'EF4', 'EF5'], ['ef4', 'ef5'],
    ['ef2', 'ef3', 'EF4', 'EF5'], ['ef4', 'ef5']]

THIS_DICT = {
    tracking_utils.STORM_ID_COLUMN: THESE_LATE_STORM_IDS,
    tracking_utils.TIME_COLUMN: THESE_LATE_TIMES_UNIX_SEC,
    events2storms.EVENT_LATITUDES_COLUMN: THESE_EVENT_LATITUDES_DEG,
    events2storms.EVENT_LONGITUDES_COLUMN: THESE_EVENT_LONGITUDES_DEG,
    events2storms.LINKAGE_DISTANCES_COLUMN: THESE_LINK_DIST_METRES,
    events2storms.RELATIVE_EVENT_TIMES_COLUMN: THESE_RELATIVE_TIMES_UNIX_SEC,
    events2storms.FUJITA_RATINGS_COLUMN: THESE_FUJITA_RATINGS
}
LATE_STORM_TO_TORNADOES_TABLE_WITH_SHARING = pandas.DataFrame.from_dict(
    THIS_DICT)

STRING_COLUMNS = [tracking_utils.STORM_ID_COLUMN]
NON_FLOAT_ARRAY_COLUMNS = [
    events2storms.RELATIVE_EVENT_TIMES_COLUMN,
    events2storms.FUJITA_RATINGS_COLUMN]
FLOAT_ARRAY_COLUMNS = [
    events2storms.EVENT_LATITUDES_COLUMN, events2storms.EVENT_LONGITUDES_COLUMN,
    events2storms.LINKAGE_DISTANCES_COLUMN]

# The following constants are used to test find_storm_to_events_file.
TOP_DIRECTORY_NAME = 'storm_to_events'
FILE_TIME_UNIX_SEC = 1517523991  # 222631 1 Feb 2018
FILE_SPC_DATE_STRING = '20180201'

STORM_TO_WINDS_ONE_TIME_FILE_NAME = (
    'storm_to_events/2018/20180201/storm_to_winds_2018-02-01-222631.p')
STORM_TO_WINDS_SPC_DATE_FILE_NAME = (
    'storm_to_events/2018/storm_to_winds_20180201.p')
STORM_TO_TORNADOES_ONE_TIME_FILE_NAME = (
    'storm_to_events/2018/20180201/storm_to_tornadoes_2018-02-01-222631.p')
STORM_TO_TORNADOES_SPC_DATE_FILE_NAME = (
    'storm_to_events/2018/storm_to_tornadoes_20180201.p')


def _compare_tables(expected_table, actual_table):
    """Determines whether or not two pandas DataFrames are equal.

    :param expected_table: expected pandas DataFrame.
    :param actual_table: actual pandas DataFrame.
    :return: tables_equal_flag: Boolean flag.
    """

    expected_num_rows = len(expected_table.index)
    actual_num_rows = len(actual_table.index)
    if expected_num_rows != actual_num_rows:
        return False

    expected_column_names = list(expected_table)
    actual_column_names = list(actual_table)
    if set(expected_column_names) != set(actual_column_names):
        return False

    for i in range(expected_num_rows):
        for this_column_name in expected_column_names:
            if this_column_name in STRING_COLUMNS:
                are_entries_equal = (
                    expected_table[this_column_name].values[i] ==
                    actual_table[this_column_name].values[i])

            elif this_column_name in NON_FLOAT_ARRAY_COLUMNS:
                are_entries_equal = numpy.array_equal(
                    expected_table[this_column_name].values[i],
                    actual_table[this_column_name].values[i])

            elif this_column_name in FLOAT_ARRAY_COLUMNS:
                are_entries_equal = numpy.allclose(
                    expected_table[this_column_name].values[i],
                    actual_table[this_column_name].values[i], atol=TOLERANCE)

            else:
                are_entries_equal = numpy.isclose(
                    expected_table[this_column_name].values[i],
                    actual_table[this_column_name].values[i], atol=TOLERANCE)

            if not are_entries_equal:
                return False

    return True


class LinkEventsToStormsTests(unittest.TestCase):
    """Each method is a unit test for link_events_to_storms.py."""

    def test_get_bounding_box_of_storms(self):
        """Ensures correct output from _get_bounding_box_of_storms."""

        these_x_limits_metres, these_y_limits_metres = (
            events2storms._get_bounding_box_of_storms(
                STORM_OBJECT_TABLE_1CELL,
                padding_metres=BOUNDING_BOX_PADDING_METRES))

        self.assertTrue(numpy.allclose(
            these_x_limits_metres, BOUNDING_BOX_X_METRES, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_y_limits_metres, BOUNDING_BOX_Y_METRES, atol=TOLERANCE))

    def test_filter_events_by_location(self):
        """Ensures correct output from _filter_events_by_location."""

        this_filtered_event_table = events2storms._filter_events_by_location(
            event_table=EVENT_TABLE_FULL_DOMAIN,
            x_limits_metres=EVENT_X_LIMITS_METRES,
            y_limits_metres=EVENT_Y_LIMITS_METRES)

        self.assertTrue(
            this_filtered_event_table.equals(EVENT_TABLE_BOUNDING_BOX))

    def test_filter_storms_by_time_early_start_early_end(self):
        """Ensures correct output from _filter_storms_by_time.

        In this case, storm cells will be dropped if they start after 300
        seconds or end before 900 seconds.
        """

        this_storm_object_table = events2storms._filter_storms_by_time(
            MAIN_STORM_OBJECT_TABLE,
            max_start_time_unix_sec=EARLY_START_TIME_UNIX_SEC,
            min_end_time_unix_sec=EARLY_END_TIME_UNIX_SEC)
        self.assertTrue(this_storm_object_table.equals(
            STORM_OBJECT_TABLE_EARLY_START_EARLY_END))

    def test_filter_storms_by_time_early_start_late_end(self):
        """Ensures correct output from _filter_storms_by_time.

        In this case, storm cells will be dropped if they start after 300
        seconds or end before 1200 seconds.
        """

        this_storm_object_table = events2storms._filter_storms_by_time(
            MAIN_STORM_OBJECT_TABLE,
            max_start_time_unix_sec=EARLY_START_TIME_UNIX_SEC,
            min_end_time_unix_sec=LATE_END_TIME_UNIX_SEC)
        self.assertTrue(this_storm_object_table.equals(
            STORM_OBJECT_TABLE_EARLY_START_LATE_END))

    def test_filter_storms_by_time_late_start_early_end(self):
        """Ensures correct output from _filter_storms_by_time.

        In this case, storm cells will be dropped if they start after 600
        seconds or end before 900 seconds.
        """

        this_storm_object_table = events2storms._filter_storms_by_time(
            MAIN_STORM_OBJECT_TABLE,
            max_start_time_unix_sec=LATE_START_TIME_UNIX_SEC,
            min_end_time_unix_sec=EARLY_END_TIME_UNIX_SEC)
        self.assertTrue(this_storm_object_table.equals(
            STORM_OBJECT_TABLE_LATE_START_EARLY_END))

    def test_filter_storms_by_time_late_start_late_end(self):
        """Ensures correct output from _filter_storms_by_time.

        In this case, storm cells will be dropped if they start after 600
        seconds or end before 1200 seconds.
        """

        this_storm_object_table = events2storms._filter_storms_by_time(
            MAIN_STORM_OBJECT_TABLE,
            max_start_time_unix_sec=LATE_START_TIME_UNIX_SEC,
            min_end_time_unix_sec=LATE_END_TIME_UNIX_SEC)
        self.assertTrue(this_storm_object_table.equals(
            STORM_OBJECT_TABLE_LATE_START_LATE_END))

    def test_interp_one_storm_in_time_no_extrap(self):
        """Ensures correct output from _interp_one_storm_in_time.

        In this case the method is *not* extrapolating.
        """

        this_vertex_table = events2storms._interp_one_storm_in_time(
            STORM_OBJECT_TABLE_1CELL, storm_id=STORM_ID_FOR_INTERP,
            target_time_unix_sec=INTERP_TIME_1CELL_UNIX_SEC)
        self.assertTrue(this_vertex_table.equals(
            VERTEX_TABLE_1OBJECT_NO_EXTRAP))

    def test_interp_one_storm_in_time_extrap(self):
        """Ensures correct output from _interp_one_storm_in_time.

        In this case the method is extrapolating.
        """

        this_vertex_table = events2storms._interp_one_storm_in_time(
            STORM_OBJECT_TABLE_1CELL, storm_id=STORM_ID_FOR_INTERP,
            target_time_unix_sec=EXTRAP_TIME_1CELL_UNIX_SEC)
        self.assertTrue(this_vertex_table.equals(
            VERTEX_TABLE_1OBJECT_EXTRAP))

    def test_interp_storms_in_time(self):
        """Ensures correct output from _interp_storms_in_time."""

        this_vertex_table = events2storms._interp_storms_in_time(
            STORM_OBJECT_TABLE_2CELLS,
            target_time_unix_sec=INTERP_TIME_2CELLS_UNIX_SEC,
            max_time_before_start_sec=MAX_TIME_BEFORE_STORM_START_SEC,
            max_time_after_end_sec=MAX_TIME_AFTER_STORM_END_SEC)
        self.assertTrue(this_vertex_table.equals(INTERP_VERTEX_TABLE_2OBJECTS))

    def test_find_nearest_storms_at_one_time(self):
        """Ensures correct output from _find_nearest_storms_at_one_time."""

        these_nearest_storm_ids, these_linkage_distances_metres = (
            events2storms._find_nearest_storms_at_one_time(
                INTERP_VERTEX_TABLE_2OBJECTS,
                event_x_coords_metres=EVENT_X_COORDS_1TIME_METRES,
                event_y_coords_metres=EVENT_Y_COORDS_1TIME_METRES,
                max_link_distance_metres=MAX_LINK_DISTANCE_METRES))

        self.assertTrue(these_nearest_storm_ids == NEAREST_STORM_IDS_1TIME)
        self.assertTrue(numpy.allclose(
            these_linkage_distances_metres, LINKAGE_DISTANCES_1TIME_METRES,
            equal_nan=True, atol=TOLERANCE))

    def test_find_nearest_storms(self):
        """Ensures correct output from _find_nearest_storms."""

        this_wind_to_storm_table = events2storms._find_nearest_storms(
            storm_object_table=STORM_OBJECT_TABLE_2CELLS,
            event_table=EVENT_TABLE_2TIMES,
            max_time_before_storm_start_sec=MAX_TIME_BEFORE_STORM_START_SEC,
            max_time_after_storm_end_sec=MAX_TIME_AFTER_STORM_END_SEC,
            max_link_distance_metres=MAX_LINK_DISTANCE_METRES,
            interp_time_resolution_sec=
            events2storms.DEFAULT_INTERP_TIME_RESOLUTION_FOR_WIND_SEC)

        self.assertTrue(this_wind_to_storm_table.equals(
            EVENT_TO_STORM_TABLE_SIMPLE))

    def test_create_storm_to_winds_table(self):
        """Ensures correct output from _create_storm_to_winds_table."""

        this_storm_to_winds_table = (
            events2storms._create_storm_to_winds_table(
                storm_object_table=STORM_OBJECT_TABLE_2CELLS,
                wind_to_storm_table=WIND_TO_STORM_TABLE))

        self.assertTrue(set(list(this_storm_to_winds_table)) ==
                        set(list(STORM_TO_WINDS_TABLE)))
        self.assertTrue(len(this_storm_to_winds_table.index) ==
                        len(STORM_TO_WINDS_TABLE.index))

        num_rows = len(this_storm_to_winds_table.index)
        string_columns = [tracking_utils.STORM_ID_COLUMN,
                          events2storms.WIND_STATION_IDS_COLUMN]

        for i in range(num_rows):
            for this_column_name in list(STORM_TO_WINDS_TABLE):
                if this_column_name in string_columns:
                    self.assertTrue(
                        this_storm_to_winds_table[this_column_name].values[i] ==
                        STORM_TO_WINDS_TABLE[this_column_name].values[i])
                else:
                    self.assertTrue(numpy.allclose(
                        this_storm_to_winds_table[this_column_name].values[i],
                        STORM_TO_WINDS_TABLE[this_column_name].values[i],
                        atol=TOLERANCE))

    def test_share_linkages_between_periods(self):
        """Ensures correct output from _share_linkages_between_periods."""

        this_early_table = copy.deepcopy(
            EARLY_STORM_TO_TORNADOES_TABLE_SANS_SHARING)
        this_late_table = copy.deepcopy(
            LATE_STORM_TO_TORNADOES_TABLE_SANS_SHARING)

        this_early_table, this_late_table = (
            events2storms._share_linkages_between_periods(
                early_storm_to_events_table=this_early_table,
                late_storm_to_events_table=this_late_table,
                event_type_string=events2storms.TORNADO_EVENT_TYPE_STRING))

        self.assertTrue(_compare_tables(
            this_early_table, EARLY_STORM_TO_TORNADOES_TABLE_WITH_SHARING))
        self.assertTrue(_compare_tables(
            this_late_table, LATE_STORM_TO_TORNADOES_TABLE_WITH_SHARING))

    def test_find_storm_to_events_file_one_time_wind(self):
        """Ensures correct output from find_storm_to_events_file.

        In this case, event type is damaging wind and file contains data for one
        time step.
        """

        this_file_name = events2storms.find_storm_to_events_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            event_type_string=events2storms.WIND_EVENT_TYPE_STRING,
            raise_error_if_missing=False, unix_time_sec=FILE_TIME_UNIX_SEC)
        self.assertTrue(this_file_name == STORM_TO_WINDS_ONE_TIME_FILE_NAME)

    def test_find_storm_to_events_file_one_time_tornado(self):
        """Ensures correct output from find_storm_to_events_file.

        In this case, event type is tornado and file contains data for one time
        step.
        """

        this_file_name = events2storms.find_storm_to_events_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            event_type_string=events2storms.TORNADO_EVENT_TYPE_STRING,
            raise_error_if_missing=False, unix_time_sec=FILE_TIME_UNIX_SEC)
        self.assertTrue(this_file_name == STORM_TO_TORNADOES_ONE_TIME_FILE_NAME)

    def test_find_storm_to_events_file_spc_date_wind(self):
        """Ensures correct output from find_storm_to_events_file.

        In this case, event type is damaging wind and file contains data for one
        SPC date.
        """

        this_file_name = events2storms.find_storm_to_events_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            event_type_string=events2storms.WIND_EVENT_TYPE_STRING,
            raise_error_if_missing=False, spc_date_string=FILE_SPC_DATE_STRING)
        self.assertTrue(this_file_name == STORM_TO_WINDS_SPC_DATE_FILE_NAME)

    def test_find_storm_to_events_file_spc_date_tornado(self):
        """Ensures correct output from find_storm_to_events_file.

        In this case, event type is tornado and file contains data for one SPC
        date.
        """

        this_file_name = events2storms.find_storm_to_events_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            event_type_string=events2storms.TORNADO_EVENT_TYPE_STRING,
            raise_error_if_missing=False, spc_date_string=FILE_SPC_DATE_STRING)
        self.assertTrue(this_file_name == STORM_TO_TORNADOES_SPC_DATE_FILE_NAME)


if __name__ == '__main__':
    unittest.main()
